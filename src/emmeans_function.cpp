#include "emmeans_function.hpp"
#include "register_documented.hpp"

#include "emmeans.hpp"   // statsduck::emmeans — the DuckDB-free kernel
#include "list_read.hpp" // ListVectorReader, ListMatrixReader

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace duckdb {

namespace {

// The defaults of the optional arguments, which the four- and five-argument
// overloads take instead of reading a vector.
constexpr double kDefaultAlpha = 0.05;
constexpr statsduck::EmmAdjust kDefaultAdjust = statsduck::EmmAdjust::TUKEY;

// One marginal mean. It carries no index field because the list position is
// the index. Field order follows the spec.
static LogicalType EmmMeanRowType() {
	child_list_t<LogicalType> f;
	f.emplace_back("estimate", LogicalType::DOUBLE);
	f.emplace_back("std_error", LogicalType::DOUBLE);
	f.emplace_back("t_statistic", LogicalType::DOUBLE);
	f.emplace_back("p_value", LogicalType::DOUBLE);
	f.emplace_back("lower_cl", LogicalType::DOUBLE);
	f.emplace_back("upper_cl", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(f));
}

// One pairwise difference, mean i minus mean j. i and j are 1-based here, so
// they index the means list directly. Field order follows the spec.
static LogicalType EmmDiffRowType() {
	child_list_t<LogicalType> f;
	f.emplace_back("i", LogicalType::BIGINT);
	f.emplace_back("j", LogicalType::BIGINT);
	f.emplace_back("estimate", LogicalType::DOUBLE);
	f.emplace_back("std_error", LogicalType::DOUBLE);
	f.emplace_back("t_statistic", LogicalType::DOUBLE);
	f.emplace_back("p_value", LogicalType::DOUBLE);
	f.emplace_back("p_adjusted", LogicalType::DOUBLE);
	f.emplace_back("lower_cl", LogicalType::DOUBLE);
	f.emplace_back("upper_cl", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(f));
}

// Field order MUST match the writes in Finalize — EmmeansExec below, which is
// this function's Finalize (indices in the comments).
static LogicalType EmmeansResultType() {
	child_list_t<LogicalType> c;
	c.emplace_back("means", LogicalType::LIST(EmmMeanRowType())); // 0
	c.emplace_back("diffs", LogicalType::LIST(EmmDiffRowType())); // 1
	c.emplace_back("df_den", LogicalType::DOUBLE);                // 2
	c.emplace_back("alpha", LogicalType::DOUBLE);                 // 3
	c.emplace_back("adjust", LogicalType::VARCHAR);               // 4
	c.emplace_back("k_compare", LogicalType::BIGINT);             // 5
	return LogicalType::STRUCT(std::move(c));
}

// Write a DOUBLE child, mapping NaN to SQL NULL (the SetD idiom from lm_fit).
// ±Inf passes through unchanged.
static inline void SetD(Vector &v, idx_t row, double x) {
	if (std::isnan(x)) {
		FlatVector::SetNull(v, row, true);
	} else {
		FlatVector::GetData<double>(v)[row] = x;
	}
}

static void EmmeansExec(DataChunk &args, ExpressionState &, Vector &result) {
	const idx_t count = args.size();
	const idx_t n_args = args.ColumnCount();

	// L is LIST(LIST(DOUBLE)). ListMatrixReader does its three-level unwrap:
	// one entry per row of the chunk, a list per grid row, and that list's own
	// doubles. beta and cov are LIST(DOUBLE). ListVectorReader does their
	// two-level unwrap. Each reader converts its argument's format once here,
	// not once per row.
	ListMatrixReader l_reader(args.data[0], count);
	ListVectorReader beta_reader(args.data[1], count);
	ListVectorReader cov_reader(args.data[2], count);

	UnifiedVectorFormat df_fmt, alpha_fmt, adjust_fmt;
	args.data[3].ToUnifiedFormat(count, df_fmt);
	const auto df_data = UnifiedVectorFormat::GetData<double>(df_fmt);

	// The optional arguments: a null pointer means the overload does not have
	// that argument, so the row takes the default.
	const double *alpha_data = nullptr;
	const string_t *adjust_data = nullptr;
	if (n_args >= 5) {
		args.data[4].ToUnifiedFormat(count, alpha_fmt);
		alpha_data = UnifiedVectorFormat::GetData<double>(alpha_fmt);
	}
	if (n_args >= 6) {
		args.data[5].ToUnifiedFormat(count, adjust_fmt);
		adjust_data = UnifiedVectorFormat::GetData<string_t>(adjust_fmt);
	}

	// Pass 1 — evaluate every row and tally both output list lengths.
	std::vector<statsduck::EmmeansResult> fits(count);
	std::vector<bool> emit(count, false);
	idx_t total_means = 0, total_diffs = 0;
	std::vector<double> beta_buf, cov_buf;

	for (idx_t row = 0; row < count; row++) {
		const auto df_idx = df_fmt.sel->get_index(row);
		const idx_t alpha_idx = alpha_data ? alpha_fmt.sel->get_index(row) : 0;
		const idx_t adjust_idx = adjust_data ? adjust_fmt.sel->get_index(row) : 0;
		// This check is what produces the NULL result for a NULL argument, and
		// it must not be removed. DuckDB does not mask runtime NULLs for a
		// scalar function: the default null handling on the declaration only
		// lets the binder fold a NULL that is constant at bind time. A NULL
		// arriving in a column still reaches the executor, so every argument
		// is tested per row here. The NULL tests come before the shape checks,
		// so a NULL argument gives a NULL result rather than an error.
		if (!l_reader.RowValid(row) || !beta_reader.RowValid(row) || !cov_reader.RowValid(row) ||
		    !df_fmt.validity.RowIsValid(df_idx) ||
		    (alpha_data && !alpha_fmt.validity.RowIsValid(alpha_idx)) ||
		    (adjust_data && !adjust_fmt.validity.RowIsValid(adjust_idx))) {
			continue; // a NULL argument gives a NULL result
		}

		beta_reader.Read(row, beta_buf);
		cov_reader.Read(row, cov_buf);
		const std::size_t k = beta_buf.size();
		// These three messages answer the same malformed input lin_hyp answers,
		// in the same words, because two neighbouring functions must not differ
		// there. See docs/fitter_conventions.md.
		if (k == 0) {
			throw InvalidInputException("emmeans: beta is empty");
		}
		if (cov_buf.size() != k * k) {
			throw InvalidInputException(
			    "emmeans: the covariance must hold k² = %llu elements for a beta of length %llu, got %llu",
			    static_cast<unsigned long long>(k * k), static_cast<unsigned long long>(k),
			    static_cast<unsigned long long>(cov_buf.size()));
		}

		statsduck::linalg::Mat L;
		idx_t bad_row = 0;
		std::size_t bad_width = 0;
		switch (l_reader.Read(row, k, L, bad_row, bad_width)) {
		case ListMatrixReader::Status::kOk:
			break;
		case ListMatrixReader::Status::kNoRows:
			throw InvalidInputException("emmeans: the grid has no rows");
		case ListMatrixReader::Status::kNullRow:
			throw InvalidInputException("emmeans: the grid has a NULL row");
		case ListMatrixReader::Status::kWidthMismatch:
			throw InvalidInputException(
			    "emmeans: every grid row must have the same width as beta (%llu); row %llu has %llu",
			    static_cast<unsigned long long>(k), static_cast<unsigned long long>(bad_row + 1),
			    static_cast<unsigned long long>(bad_width));
		}

		const double alpha = alpha_data ? alpha_data[alpha_idx] : kDefaultAlpha;
		statsduck::EmmAdjust adjust = kDefaultAdjust;
		if (adjust_data) {
			const auto &name_s = adjust_data[adjust_idx];
			const std::string name(name_s.GetData(), name_s.GetSize());
			if (!statsduck::parse_emm_adjust(name, adjust)) {
				throw InvalidInputException("emmeans: unknown adjust method '%s' — supported: "
				                            "'tukey', 'bonferroni', 'holm', 'hochberg', "
				                            "'BH' (or 'fdr'), 'BY', 'none'",
				                            name);
			}
		}

		statsduck::linalg::Mat V(k, k);
		V.data = cov_buf; // row-major k×k, exactly Mat's layout

		auto r = statsduck::emmeans(L, beta_buf, V, df_data[df_idx], alpha, adjust);
		if (!r.ok) {
			throw InvalidInputException(r.error);
		}
		// Unlike lin_hyp there is no top-level estimable flag to test here.
		// Estimability is per element, so an evaluated grid is always reported:
		// one unusable row must not discard the others.
		total_means += r.means.size();
		total_diffs += r.diffs.size();
		fits[row] = std::move(r);
		emit[row] = true;
	}

	auto &children = StructVector::GetEntries(result);

	// Two LIST children instead of lin_hyp's one, each with its own length and
	// its own offset. Reserve before fetching the child vector — Reserve can
	// reallocate, and fetching first is the classic bug. See
	// notes/engineering/2026-06-lm-fit-robust-se-aggregate.md §2.
	Vector &means_list = *children[0];
	ListVector::Reserve(means_list, total_means);
	ListVector::SetListSize(means_list, total_means);
	auto means_entries = FlatVector::GetData<list_entry_t>(means_list);
	auto &mean_f = StructVector::GetEntries(ListVector::GetEntry(means_list));

	Vector &diffs_list = *children[1];
	ListVector::Reserve(diffs_list, total_diffs);
	ListVector::SetListSize(diffs_list, total_diffs);
	auto diffs_entries = FlatVector::GetData<list_entry_t>(diffs_list);
	auto &diff_f = StructVector::GetEntries(ListVector::GetEntry(diffs_list));

	auto i_data = FlatVector::GetData<int64_t>(*diff_f[0]);
	auto j_data = FlatVector::GetData<int64_t>(*diff_f[1]);
	auto k_compare_data = FlatVector::GetData<int64_t>(*children[5]);

	// Both offsets walk per row. Each row's list_entry_t is set from the same
	// counter the writes use, so the two agree only while the counter advances
	// by that row's own length. Advancing by less overlaps the next row and
	// reports one row's numbers under another's. The two lists hold different
	// numbers of elements, so advancing one by the other's count does exactly
	// that on every row where the two differ.
	idx_t means_out = 0, diffs_out = 0;
	for (idx_t row = 0; row < count; row++) {
		if (!emit[row]) {
			FlatVector::SetNull(result, row, true);
			means_entries[row] = list_entry_t(means_out, 0);
			diffs_entries[row] = list_entry_t(diffs_out, 0);
			// The VARCHAR child is set NULL as well, unlike the numeric ones.
			// An unwritten string_t holds whatever its buffer held, and a copy
			// of this vector would read that as a pointer. An unwritten double
			// is only a number nobody reads.
			FlatVector::SetNull(*children[4], row, true);
			continue;
		}
		const auto &r = fits[row];

		const idx_t n_means = r.means.size();
		means_entries[row] = list_entry_t(means_out, n_means);
		for (idx_t a = 0; a < n_means; a++) {
			const idx_t pos = means_out + a;
			const auto &m = r.means[a];
			// No branch on m.estimable: a non-estimable mean carries NaN in
			// every numeric field, and SetD renders each NaN as SQL NULL.
			SetD(*mean_f[0], pos, m.estimate);
			SetD(*mean_f[1], pos, m.std_error);
			SetD(*mean_f[2], pos, m.t_statistic);
			SetD(*mean_f[3], pos, m.p_value);
			SetD(*mean_f[4], pos, m.lower_cl);
			SetD(*mean_f[5], pos, m.upper_cl);
		}
		means_out += n_means;

		const idx_t n_diffs = r.diffs.size();
		diffs_entries[row] = list_entry_t(diffs_out, n_diffs);
		for (idx_t a = 0; a < n_diffs; a++) {
			const idx_t pos = diffs_out + a;
			const auto &d = r.diffs[a];
			// The kernel's i and j are 0-based; SQL lists are 1-based, so they
			// index means directly only after the + 1. They are written even
			// for a non-estimable pair, which is what keeps them valid
			// indices: only the numeric fields go NULL.
			i_data[pos] = static_cast<int64_t>(d.i) + 1;
			j_data[pos] = static_cast<int64_t>(d.j) + 1;
			SetD(*diff_f[2], pos, d.estimate);
			SetD(*diff_f[3], pos, d.std_error);
			SetD(*diff_f[4], pos, d.t_statistic);
			SetD(*diff_f[5], pos, d.p_value);
			SetD(*diff_f[6], pos, d.p_adjusted);
			SetD(*diff_f[7], pos, d.lower_cl);
			SetD(*diff_f[8], pos, d.upper_cl);
		}
		diffs_out += n_diffs;

		SetD(*children[2], row, r.df_den);
		SetD(*children[3], row, r.alpha);
		FlatVector::GetData<string_t>(*children[4])[row] =
		    StringVector::AddString(*children[4], statsduck::emm_adjust_name(r.adjust));
		k_compare_data[row] = static_cast<int64_t>(r.k_compare);
	}
}

} // namespace

void RegisterEmmeans(ExtensionLoader &loader) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	const auto grid = LogicalType::LIST(list_double);
	const auto ret = EmmeansResultType();
	const auto dbl = LogicalType::DOUBLE;

	// Four, five and six arguments, so alpha and adjust are optional. The grid
	// is declared LIST(LIST(DOUBLE)) and cast into, because DuckDB infers an
	// integer grid as INTEGER[][] and [[1.0, 2.0]] as DECIMAL(2,1)[][].
	ScalarFunctionSet set("emmeans");
	set.AddFunction(ScalarFunction({grid, list_double, list_double, dbl}, ret, EmmeansExec));
	set.AddFunction(ScalarFunction({grid, list_double, list_double, dbl, dbl}, ret, EmmeansExec));
	set.AddFunction(
	    ScalarFunction({grid, list_double, list_double, dbl, dbl, LogicalType::VARCHAR}, ret, EmmeansExec));

	statsduck::RegisterDocumented(
	    loader, std::move(set),
	    "Estimated marginal means of a fit: each grid row's mean L·β with its standard error, "
	    "t, p and confidence limits, every pairwise difference, and a multiplicity adjustment "
	    "over that family ('tukey', 'bonferroni', 'holm', 'hochberg', 'BH' or 'fdr', 'BY', "
	    "'none'). alpha defaults to 0.05 and adjust to 'tukey'. Under 'tukey' a df that is not "
	    "an integer and is below 4 makes the adjustment accurate to about 1e-5 rather than "
	    "1e-12. Nothing else about the function is affected by that limit.",
	    {"L", "beta", "cov", "df", "alpha", "adjust"},
	    "SELECT emmeans([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [2.0, 6.0, 6.0], "
	    "[0.8889,0,0, 0,0.8889,0, 0,0,0.8889], 6.0)");
}

} // namespace duckdb
