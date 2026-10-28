#include "lin_hyp_function.hpp"
#include "register_documented.hpp"

#include "lin_hyp.hpp"   // statsduck::lin_hyp — the DuckDB-free kernel
#include "list_read.hpp" // ListVectorReader, ListMatrixReader

#include "duckdb_compat.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <vector>

namespace duckdb {

namespace compat = statsduck::compat;

namespace {

// One row of the per-constraint table, mirroring lm_fit's coefficients element.
static LogicalType LinHypRowType() {
	child_list_t<LogicalType> f;
	f.emplace_back("estimate", LogicalType::DOUBLE);
	f.emplace_back("std_error", LogicalType::DOUBLE);
	f.emplace_back("t_statistic", LogicalType::DOUBLE);
	f.emplace_back("p_value", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(f));
}

// Field order MUST match the writes in LinHypExec (indices below).
static LogicalType LinHypResultType() {
	child_list_t<LogicalType> c;
	c.emplace_back("estimates", LogicalType::LIST(LinHypRowType())); // 0
	c.emplace_back("ss", LogicalType::DOUBLE);                       // 1
	c.emplace_back("df_num", LogicalType::BIGINT);                   // 2
	c.emplace_back("df_den", LogicalType::DOUBLE);                   // 3
	c.emplace_back("f_statistic", LogicalType::DOUBLE);              // 4
	c.emplace_back("f_p_value", LogicalType::DOUBLE);                // 5
	c.emplace_back("chisq", LogicalType::DOUBLE);                    // 6
	c.emplace_back("chisq_p_value", LogicalType::DOUBLE);            // 7
	return LogicalType::STRUCT(std::move(c));
}

// Write a DOUBLE child, mapping NaN to SQL NULL (the SetD idiom from lm_fit).
static inline void SetD(Vector &v, idx_t row, double x) {
	if (std::isnan(x)) {
		FlatVector::SetNull(v, row, true);
	} else {
		compat::WriteData<double>(v)[row] = x;
	}
}

static void LinHypExec(DataChunk &args, ExpressionState &, Vector &result) {
	const idx_t count = args.size();

	// L is LIST(LIST(DOUBLE)). ListMatrixReader does its three-level unwrap:
	// one entry per row of the chunk, a list per constraint, and that
	// list's own doubles. beta and cov are LIST(DOUBLE). ListVectorReader
	// does their two-level unwrap. Each reader converts its argument's
	// format once here, not once per row.
	ListMatrixReader l_reader(args.data[0], count);
	ListVectorReader beta_reader(args.data[1], count);
	ListVectorReader cov_reader(args.data[2], count);

	UnifiedVectorFormat s2_fmt, df_fmt;
	args.data[3].ToUnifiedFormat(count, s2_fmt);
	args.data[4].ToUnifiedFormat(count, df_fmt);
	const auto s2_data = UnifiedVectorFormat::GetData<double>(s2_fmt);
	const auto df_data = UnifiedVectorFormat::GetData<double>(df_fmt);

	// Pass 1 — evaluate every row and tally the output list length.
	std::vector<statsduck::LinHypResult> fits(count);
	std::vector<bool> emit(count, false);
	idx_t total_rows = 0;
	std::vector<double> beta_buf, cov_buf;

	for (idx_t i = 0; i < count; i++) {
		const auto s2_idx = s2_fmt.sel->get_index(i);
		const auto df_idx = df_fmt.sel->get_index(i);
		// This check is what produces the NULL result for a NULL argument, and
		// it must not be removed. DuckDB does not mask runtime NULLs for a
		// scalar function: the default null handling on the declaration only
		// lets the binder fold a NULL that is constant at bind time. A NULL
		// arriving in a column still reaches the executor, so every argument
		// is tested per row here.
		if (!l_reader.RowValid(i) || !beta_reader.RowValid(i) || !cov_reader.RowValid(i) ||
		    !s2_fmt.validity.RowIsValid(s2_idx) || !df_fmt.validity.RowIsValid(df_idx)) {
			continue; // a NULL argument gives a NULL result
		}

		beta_reader.Read(i, beta_buf);
		cov_reader.Read(i, cov_buf);
		const std::size_t k = beta_buf.size();
		if (k == 0) {
			throw InvalidInputException("lin_hyp: beta is empty");
		}
		if (cov_buf.size() != k * k) {
			throw InvalidInputException(
			    "lin_hyp: the covariance must hold k² = %llu elements for a beta of length %llu, got %llu",
			    static_cast<unsigned long long>(k * k), static_cast<unsigned long long>(k),
			    static_cast<unsigned long long>(cov_buf.size()));
		}

		statsduck::linalg::Mat L;
		idx_t bad_row = 0;
		std::size_t bad_width = 0;
		switch (l_reader.Read(i, k, L, bad_row, bad_width)) {
		case ListMatrixReader::Status::kOk:
			break;
		case ListMatrixReader::Status::kNoRows:
			throw InvalidInputException("lin_hyp: the hypothesis matrix has no rows");
		case ListMatrixReader::Status::kNullRow:
			throw InvalidInputException("lin_hyp: the hypothesis matrix has a NULL row");
		case ListMatrixReader::Status::kWidthMismatch:
			throw InvalidInputException(
			    "lin_hyp: every hypothesis row must have the same width as beta (%llu); row %llu "
			    "has %llu",
			    static_cast<unsigned long long>(k), static_cast<unsigned long long>(bad_row + 1),
			    static_cast<unsigned long long>(bad_width));
		}

		statsduck::linalg::Mat V(k, k);
		V.data = cov_buf; // row-major k×k, exactly Mat's layout

		auto r = statsduck::lin_hyp(L, beta_buf, V, s2_data[s2_idx], df_data[df_idx]);
		if (!r.ok) {
			throw InvalidInputException(r.error);
		}
		if (!r.estimable) {
			continue; // not estimable → a NULL result for this row
		}
		total_rows += r.estimate.size();
		fits[i] = std::move(r);
		emit[i] = true;
	}

	auto &children = StructVector::GetEntries(result);

	// estimates: LIST<STRUCT>. Reserve first, then fetch the child pointers —
	// Reserve can reallocate, and fetching first is the classic bug. See
	// notes/engineering/2026-06-lm-fit-robust-se-aggregate.md §2.
	Vector &rows_list = compat::Deref(children[0]);
	auto row_entries_out = compat::WriteData<list_entry_t>(rows_list);
	ListVector::Reserve(rows_list, total_rows);
	ListVector::SetListSize(rows_list, total_rows);
	Vector &rows_child = compat::ListChild(rows_list);
	auto &rf = StructVector::GetEntries(rows_child);

	auto df_num_data = compat::WriteData<int64_t>(compat::Deref(children[2]));

	idx_t out = 0;
	for (idx_t i = 0; i < count; i++) {
		if (!emit[i]) {
			FlatVector::SetNull(result, i, true);
			row_entries_out[i] = list_entry_t(out, 0);
			continue;
		}
		const auto &r = fits[i];
		const idx_t q = r.estimate.size();
		row_entries_out[i] = list_entry_t(out, q);
		for (idx_t a = 0; a < q; a++) {
			const idx_t pos = out + a;
			SetD(compat::Deref(rf[0]), pos, r.estimate[a]);
			SetD(compat::Deref(rf[1]), pos, r.std_error[a]);
			SetD(compat::Deref(rf[2]), pos, r.t_statistic[a]);
			SetD(compat::Deref(rf[3]), pos, r.p_value[a]);
		}
		out += q;

		SetD(compat::Deref(children[1]), i, r.ss);
		df_num_data[i] = static_cast<int64_t>(r.df_num);
		SetD(compat::Deref(children[3]), i, r.df_den);
		SetD(compat::Deref(children[4]), i, r.f_statistic);
		SetD(compat::Deref(children[5]), i, r.f_p_value);
		SetD(compat::Deref(children[6]), i, r.chisq);
		SetD(compat::Deref(children[7]), i, r.chisq_p_value);
	}
}

} // namespace

void RegisterLinHyp(ExtensionLoader &loader) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	ScalarFunction fn("lin_hyp",
	                  {LogicalType::LIST(list_double), list_double, list_double,
	                   LogicalType::DOUBLE, LogicalType::DOUBLE},
	                  LinHypResultType(), LinHypExec);
	compat::SetFallible(fn);
	statsduck::RegisterDocumented(
	    loader, std::move(fn),
	    "General linear hypothesis L·β = 0 against a fit: per-constraint estimates with "
	    "standard errors, plus the joint sum of squares, F and chi-square.",
	    {"L", "beta", "cov", "sigma2", "df"},
	    "SELECT lin_hyp([[0, 1, -1]], [10.0, 2.0, -1.0], "
	    "[4.3,-0.14,-0.52, -0.14,0.017,0.0023, -0.52,0.0023,0.101], 7.46, 9.0)");
}

} // namespace duckdb
