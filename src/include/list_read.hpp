#pragma once

// stats_duck list-argument reader, used by the SQL functions that take a
// LIST(DOUBLE) or LIST(LIST(DOUBLE)) argument. lin_hyp uses it for beta, cov
// and L. emmeans (#21) will use it the same way.
//
// A nested list is not always laid out back to back in row order. A
// list_transform, a join, or a dictionary-encoded vector can hand back rows
// out of order or scattered through the underlying vector. So every read
// goes through UnifiedVectorFormat and its selection vector. It never
// assumes contiguous layout.
//
// Converting a vector to UnifiedVectorFormat costs time proportional to the
// vector's size. Each reader below does that conversion once, in its
// constructor, for the whole chunk. A caller then calls Read() once per row.
// This keeps the total conversion cost proportional to the chunk size, not
// to the chunk size times the number of rows in it.
//
// This header is DuckDB-coupled on purpose. It touches Vector and
// UnifiedVectorFormat directly. It belongs with the function layer, not the
// DuckDB-free kernel headers (linalg.hpp, lin_hyp.hpp, and so on).

#include "linalg.hpp"

#include "duckdb/common/types/vector.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace duckdb {

// Reads one run of a flat DOUBLE list (`entry`, inside `fmt`) into `out`. A
// NULL element reads as NaN, so a kernel's own non-finite check is what
// catches a missing value. That replaces a second, separate validity check
// for the same thing. ListVectorReader and ListMatrixReader below share
// this function. Both read a flat run of doubles the same way. They just
// read from different levels of nesting.
inline void ReadDoubles(const UnifiedVectorFormat &fmt, const list_entry_t &entry,
                        std::vector<double> &out) {
	const auto data = UnifiedVectorFormat::GetData<double>(fmt);
	out.assign(entry.length, 0.0);
	for (idx_t j = 0; j < entry.length; j++) {
		const auto idx = fmt.sel->get_index(entry.offset + j);
		out[j] = fmt.validity.RowIsValid(idx) ? data[idx] : std::numeric_limits<double>::quiet_NaN();
	}
}

// Reads a LIST(DOUBLE) argument, one chunk row at a time. Used for lin_hyp's
// beta and cov.
struct ListVectorReader {
	UnifiedVectorFormat list_fmt;
	const list_entry_t *entries = nullptr;
	UnifiedVectorFormat child_fmt;

	ListVectorReader(Vector &v, idx_t count) {
		v.ToUnifiedFormat(count, list_fmt);
		entries = UnifiedVectorFormat::GetData<list_entry_t>(list_fmt);
		ListVector::GetEntry(v).ToUnifiedFormat(ListVector::GetListSize(v), child_fmt);
	}

	// False when row's own list argument is NULL. Read() is not meaningful
	// for such a row. The caller decides what a NULL argument means for it.
	bool RowValid(idx_t row) const {
		return list_fmt.validity.RowIsValid(list_fmt.sel->get_index(row));
	}

	// Reads row's doubles into out, resized to row's own length.
	void Read(idx_t row, std::vector<double> &out) const {
		ReadDoubles(child_fmt, entries[list_fmt.sel->get_index(row)], out);
	}
};

// Reads a LIST(LIST(DOUBLE)) argument, one chunk row at a time. Each row of
// the chunk holds its own matrix, stored as a list of same-width sub-lists,
// one sub-list per matrix row. Used for lin_hyp's L.
struct ListMatrixReader {
	UnifiedVectorFormat outer_fmt;
	const list_entry_t *outer_entries = nullptr;
	UnifiedVectorFormat row_fmt;
	const list_entry_t *row_entries = nullptr;
	UnifiedVectorFormat val_fmt;

	ListMatrixReader(Vector &v, idx_t count) {
		v.ToUnifiedFormat(count, outer_fmt);
		outer_entries = UnifiedVectorFormat::GetData<list_entry_t>(outer_fmt);

		auto &rows_vec = ListVector::GetEntry(v);
		rows_vec.ToUnifiedFormat(ListVector::GetListSize(v), row_fmt);
		row_entries = UnifiedVectorFormat::GetData<list_entry_t>(row_fmt);

		ListVector::GetEntry(rows_vec).ToUnifiedFormat(ListVector::GetListSize(rows_vec), val_fmt);
	}

	// False when row's own list argument is NULL. Rows() and Read() are not
	// meaningful for such a row. The caller decides what a NULL argument
	// means for it.
	bool RowValid(idx_t row) const {
		return outer_fmt.validity.RowIsValid(outer_fmt.sel->get_index(row));
	}

	// The number of sub-lists (matrix rows) row holds.
	idx_t Rows(idx_t row) const {
		return outer_entries[outer_fmt.sel->get_index(row)].length;
	}

	// Reads row's nested list into out, a Rows(row) x expected_width matrix.
	// Returns false, with error set, when a sub-list is NULL. Also returns
	// false when a sub-list's length is not expected_width. A rectangular
	// matrix cannot hold either shape. A NULL scalar inside a sub-list is
	// not one of these failures: ReadDoubles reads it as NaN, the same way
	// ListVectorReader::Read does.
	//
	// error names the failing sub-list as a 1-based index. For a width
	// mismatch it also names both widths. The wording matches what lin_hyp
	// has always reported: beta is the argument every row's width is
	// compared against, for every caller this header has today. A caller
	// that needs different wording can ignore error. It already has
	// Rows(row) and expected_width, enough to write its own message.
	bool Read(idx_t row, std::size_t expected_width, statsduck::linalg::Mat &out,
	          std::string &error) const {
		const auto outer_entry = outer_entries[outer_fmt.sel->get_index(row)];
		const idx_t q = outer_entry.length;
		out = statsduck::linalg::Mat(q, expected_width);
		std::vector<double> buf;
		for (idx_t a = 0; a < q; a++) {
			const auto r_idx = row_fmt.sel->get_index(outer_entry.offset + a);
			if (!row_fmt.validity.RowIsValid(r_idx)) {
				error = "the hypothesis matrix has a NULL row";
				return false;
			}
			const auto r_entry = row_entries[r_idx];
			if (r_entry.length != expected_width) {
				error = "every hypothesis row must have the same width as beta (" +
				        std::to_string(expected_width) + "); row " + std::to_string(a + 1) +
				        " has " + std::to_string(r_entry.length);
				return false;
			}
			ReadDoubles(val_fmt, r_entry, buf);
			for (std::size_t b = 0; b < expected_width; b++) {
				out(a, b) = buf[b];
			}
		}
		return true;
	}
};

} // namespace duckdb
