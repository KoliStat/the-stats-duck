#include "lin_hyp_function.hpp"
#include "register_documented.hpp"

#include "lin_hyp.hpp" // statsduck::lin_hyp — the DuckDB-free kernel

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace duckdb {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

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
		FlatVector::GetData<double>(v)[row] = x;
	}
}

// Read a run of a flat DOUBLE list into `out`, reading NULL as NaN so it
// reaches the kernel's own non-finite check instead of needing separate
// validity handling.
static void ReadDoubles(const UnifiedVectorFormat &fmt, const list_entry_t &entry,
                        std::vector<double> &out) {
	const auto data = UnifiedVectorFormat::GetData<double>(fmt);
	out.assign(entry.length, 0.0);
	for (idx_t j = 0; j < entry.length; j++) {
		const auto idx = fmt.sel->get_index(entry.offset + j);
		out[j] = fmt.validity.RowIsValid(idx) ? data[idx] : kNaN;
	}
}

static void LinHypExec(DataChunk &args, ExpressionState &, Vector &result) {
	const idx_t count = args.size();

	// L is LIST(LIST(DOUBLE)), so it unpacks in three levels: the outer list
	// gives one entry per row of the chunk, its child is a list per constraint,
	// and that child holds the doubles.
	auto &l_vec = args.data[0];
	UnifiedVectorFormat l_fmt;
	l_vec.ToUnifiedFormat(count, l_fmt);
	const auto l_entries = UnifiedVectorFormat::GetData<list_entry_t>(l_fmt);

	auto &l_rows_vec = ListVector::GetEntry(l_vec);
	UnifiedVectorFormat l_rows_fmt;
	l_rows_vec.ToUnifiedFormat(ListVector::GetListSize(l_vec), l_rows_fmt);
	const auto l_row_entries = UnifiedVectorFormat::GetData<list_entry_t>(l_rows_fmt);

	auto &l_vals_vec = ListVector::GetEntry(l_rows_vec);
	UnifiedVectorFormat l_vals_fmt;
	l_vals_vec.ToUnifiedFormat(ListVector::GetListSize(l_rows_vec), l_vals_fmt);

	auto &beta_vec = args.data[1];
	UnifiedVectorFormat beta_fmt;
	beta_vec.ToUnifiedFormat(count, beta_fmt);
	const auto beta_entries = UnifiedVectorFormat::GetData<list_entry_t>(beta_fmt);
	UnifiedVectorFormat beta_child_fmt;
	ListVector::GetEntry(beta_vec).ToUnifiedFormat(ListVector::GetListSize(beta_vec),
	                                               beta_child_fmt);

	auto &cov_vec = args.data[2];
	UnifiedVectorFormat cov_fmt;
	cov_vec.ToUnifiedFormat(count, cov_fmt);
	const auto cov_entries = UnifiedVectorFormat::GetData<list_entry_t>(cov_fmt);
	UnifiedVectorFormat cov_child_fmt;
	ListVector::GetEntry(cov_vec).ToUnifiedFormat(ListVector::GetListSize(cov_vec), cov_child_fmt);

	UnifiedVectorFormat s2_fmt, df_fmt;
	args.data[3].ToUnifiedFormat(count, s2_fmt);
	args.data[4].ToUnifiedFormat(count, df_fmt);
	const auto s2_data = UnifiedVectorFormat::GetData<double>(s2_fmt);
	const auto df_data = UnifiedVectorFormat::GetData<double>(df_fmt);

	// Pass 1 — evaluate every row and tally the output list length.
	std::vector<statsduck::LinHypResult> fits(count);
	std::vector<bool> emit(count, false);
	idx_t total_rows = 0;
	std::vector<double> beta_buf, cov_buf, l_buf;

	for (idx_t i = 0; i < count; i++) {
		const auto l_idx = l_fmt.sel->get_index(i);
		const auto beta_idx = beta_fmt.sel->get_index(i);
		const auto cov_idx = cov_fmt.sel->get_index(i);
		const auto s2_idx = s2_fmt.sel->get_index(i);
		const auto df_idx = df_fmt.sel->get_index(i);
		if (!l_fmt.validity.RowIsValid(l_idx) || !beta_fmt.validity.RowIsValid(beta_idx) ||
		    !cov_fmt.validity.RowIsValid(cov_idx) || !s2_fmt.validity.RowIsValid(s2_idx) ||
		    !df_fmt.validity.RowIsValid(df_idx)) {
			continue; // a NULL argument gives a NULL result
		}

		ReadDoubles(beta_child_fmt, beta_entries[beta_idx], beta_buf);
		ReadDoubles(cov_child_fmt, cov_entries[cov_idx], cov_buf);
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

		const auto l_entry = l_entries[l_idx];
		const std::size_t q = l_entry.length;
		if (q == 0) {
			throw InvalidInputException("lin_hyp: the hypothesis matrix has no rows");
		}
		statsduck::linalg::Mat L(q, k);
		for (idx_t a = 0; a < q; a++) {
			const auto row_idx = l_rows_fmt.sel->get_index(l_entry.offset + a);
			if (!l_rows_fmt.validity.RowIsValid(row_idx)) {
				throw InvalidInputException("lin_hyp: the hypothesis matrix has a NULL row");
			}
			const auto row_entry = l_row_entries[row_idx];
			if (row_entry.length != k) {
				throw InvalidInputException(
				    "lin_hyp: every hypothesis row must have the same width as beta (%llu); row %llu "
				    "has %llu",
				    static_cast<unsigned long long>(k), static_cast<unsigned long long>(a + 1),
				    static_cast<unsigned long long>(row_entry.length));
			}
			ReadDoubles(l_vals_fmt, row_entry, l_buf);
			for (std::size_t b = 0; b < k; b++) {
				L(a, b) = l_buf[b];
			}
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
	Vector &rows_list = *children[0];
	auto row_entries_out = FlatVector::GetData<list_entry_t>(rows_list);
	ListVector::Reserve(rows_list, total_rows);
	ListVector::SetListSize(rows_list, total_rows);
	Vector &rows_child = ListVector::GetEntry(rows_list);
	auto &rf = StructVector::GetEntries(rows_child);

	auto df_num_data = FlatVector::GetData<int64_t>(*children[2]);

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
			SetD(*rf[0], pos, r.estimate[a]);
			SetD(*rf[1], pos, r.std_error[a]);
			SetD(*rf[2], pos, r.t_statistic[a]);
			SetD(*rf[3], pos, r.p_value[a]);
		}
		out += q;

		SetD(*children[1], i, r.ss);
		df_num_data[i] = static_cast<int64_t>(r.df_num);
		SetD(*children[3], i, r.df_den);
		SetD(*children[4], i, r.f_statistic);
		SetD(*children[5], i, r.f_p_value);
		SetD(*children[6], i, r.chisq);
		SetD(*children[7], i, r.chisq_p_value);
	}
}

} // namespace

void RegisterLinHyp(ExtensionLoader &loader) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	ScalarFunction fn("lin_hyp",
	                  {LogicalType::LIST(list_double), list_double, list_double,
	                   LogicalType::DOUBLE, LogicalType::DOUBLE},
	                  LinHypResultType(), LinHypExec);
	statsduck::RegisterDocumented(
	    loader, std::move(fn),
	    "General linear hypothesis L·β = 0 against a fit: per-constraint estimates with "
	    "standard errors, plus the joint sum of squares, F and chi-square.",
	    {"L", "beta", "cov", "sigma2", "df"},
	    "SELECT lin_hyp([[0, 1, -1]], [10.0, 2.0, -1.0], "
	    "[4.3,-0.14,-0.52, -0.14,0.017,0.0023, -0.52,0.0023,0.101], 7.46, 9.0)");
}

} // namespace duckdb
