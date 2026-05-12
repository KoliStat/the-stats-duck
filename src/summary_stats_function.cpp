#include "summary_stats_function.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace duckdb {

// summary_stats(column) — exact descriptive statistics.
//
// Approach: we buffer the non-null values per aggregate state in a
// heap-allocated std::vector, then compute all statistics (moments + order
// stats) in Finalize. This gives exact quantiles without a sketch data
// structure, at the cost of memory proportional to the number of rows per
// group. For typical descriptive-stats workflows (a few million rows, a
// handful of groups) this is fine; billion-row global aggregations would
// want a t-digest, which is a future optimization.
//
// Quantiles use the R type 7 / Excel INC definition:
//   position = 1 + q * (n - 1)  (1-based)
// with linear interpolation between adjacent order statistics.

namespace {

// ── Bind data ──────────────────────────────────────────────────────────────

struct SummaryStatsBindData : public FunctionData {
	// When true, skewness/kurtosis use the bias-corrected sample formulas
	// (matching SAS PROC MEANS, scipy.stats with bias=False, Excel SKEW/KURT).
	// When false, use the population formulas m3/m2^1.5 and m4/m2² - 3
	// (matching R's default, scipy with bias=True, and the canonical
	// Jarque-Bera definition).
	bool bias_correction = true;
	// Quantile algorithm (Hyndman & Fan 1996 type number):
	//   7 — R / Excel INC default: pos = 1 + q*(n-1)  (our v0.2 behaviour)
	//   5 — SAS PROC UNIVARIATE default: pos = q*n + 0.5
	// Other types are valid in R but not yet supported here.
	int32_t quantile_type = 7;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<SummaryStatsBindData>();
		copy->bias_correction = bias_correction;
		copy->quantile_type = quantile_type;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<SummaryStatsBindData>();
		return bias_correction == other.bias_correction && quantile_type == other.quantile_type;
	}
};

static unique_ptr<FunctionData> SummaryStatsBindNoArg(ClientContext &, AggregateFunction &,
                                                      vector<unique_ptr<Expression>> &) {
	return make_uniq<SummaryStatsBindData>();
}

static unique_ptr<FunctionData> SummaryStatsBindBias(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("summary_stats: bias_correction must be a constant boolean");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	auto bd = make_uniq<SummaryStatsBindData>();
	bd->bias_correction = val.GetValue<bool>();
	Function::EraseArgument(function, arguments, 1);
	return std::move(bd);
}

static unique_ptr<FunctionData> SummaryStatsBindBiasQuantile(ClientContext &context, AggregateFunction &function,
                                                              vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[1]->IsFoldable() || !arguments[2]->IsFoldable()) {
		throw BinderException("summary_stats: bias_correction and quantile_type must be constants");
	}
	Value bias_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	Value qtype_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
	auto bd = make_uniq<SummaryStatsBindData>();
	bd->bias_correction = bias_val.GetValue<bool>();
	bd->quantile_type = qtype_val.GetValue<int32_t>();
	if (bd->quantile_type != 5 && bd->quantile_type != 7) {
		throw BinderException("summary_stats: quantile_type must be 5 (SAS) or 7 (R/Excel, default); got %d",
		                      bd->quantile_type);
	}
	Function::EraseArgument(function, arguments, 2);
	Function::EraseArgument(function, arguments, 1);
	return std::move(bd);
}

// ── Result struct type ─────────────────────────────────────────────────────

static LogicalType SummaryStatsResultType() {
	child_list_t<LogicalType> children;
	children.emplace_back("n", LogicalType::BIGINT);
	children.emplace_back("n_missing", LogicalType::BIGINT);
	children.emplace_back("mean", LogicalType::DOUBLE);
	children.emplace_back("sd", LogicalType::DOUBLE);
	children.emplace_back("variance", LogicalType::DOUBLE);
	children.emplace_back("min", LogicalType::DOUBLE);
	children.emplace_back("q1", LogicalType::DOUBLE);
	children.emplace_back("median", LogicalType::DOUBLE);
	children.emplace_back("q3", LogicalType::DOUBLE);
	children.emplace_back("max", LogicalType::DOUBLE);
	children.emplace_back("iqr", LogicalType::DOUBLE);
	children.emplace_back("skewness", LogicalType::DOUBLE);
	children.emplace_back("kurtosis", LogicalType::DOUBLE);
	children.emplace_back("mode", LogicalType::DOUBLE);
	children.emplace_back("mode_frequency", LogicalType::BIGINT);
	children.emplace_back("is_multimodal", LogicalType::BOOLEAN);
	return LogicalType::STRUCT(std::move(children));
}

// ── Quantile helper (Hyndman & Fan type 5 or 7) ─────────────────────────────

static double Quantile(const std::vector<double> &sorted, double q, int32_t qtype) {
	// Precondition: `sorted` is non-empty and already sorted ascending.
	idx_t n = sorted.size();
	if (n == 1) {
		return sorted[0];
	}
	double n_d = static_cast<double>(n);
	// 1-based position. Type 7 anchors at the endpoints (q=0 → pos 1, q=1 →
	// pos n). Type 5 centres each datum at (k - 0.5)/n, so pos = q*n + 0.5.
	double pos = (qtype == 5) ? q * n_d + 0.5 : 1.0 + q * (n_d - 1.0);
	if (pos <= 1.0) {
		return sorted[0];
	}
	if (pos >= n_d) {
		return sorted[n - 1];
	}
	double floor_pos = std::floor(pos);
	double frac = pos - floor_pos;
	idx_t lo = static_cast<idx_t>(floor_pos) - 1; // 0-based
	return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
}

// ── Aggregate state ────────────────────────────────────────────────────────

struct SummaryStatsState {
	std::vector<double> *values;
	int64_t n_missing;
};

static void SummaryStatsInit(const AggregateFunction &, data_ptr_t state_p) {
	auto &state = *reinterpret_cast<SummaryStatsState *>(state_p);
	state.values = nullptr;
	state.n_missing = 0;
}

static void SummaryStatsUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	auto &input = inputs[0];
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	auto states = FlatVector::GetData<SummaryStatsState *>(state_vector);
	auto values = UnifiedVectorFormat::GetData<double>(idata);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		auto idx = idata.sel->get_index(i);
		if (!idata.validity.RowIsValid(idx)) {
			state.n_missing++;
			continue;
		}
		double v = values[idx];
		if (std::isnan(v)) {
			// NaN inputs are treated as missing (same as R's na.rm behavior).
			state.n_missing++;
			continue;
		}
		if (!state.values) {
			state.values = new std::vector<double>();
		}
		state.values->push_back(v);
	}
}

static void SummaryStatsCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src = FlatVector::GetData<SummaryStatsState *>(source);
	auto tgt = FlatVector::GetData<SummaryStatsState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto &s = *src[i];
		auto &t = *tgt[i];
		t.n_missing += s.n_missing;
		if (s.values && !s.values->empty()) {
			if (!t.values) {
				t.values = new std::vector<double>();
			}
			t.values->insert(t.values->end(), s.values->begin(), s.values->end());
		}
	}
}

static void SummaryStatsDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	auto states = FlatVector::GetData<SummaryStatsState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		delete states[i]->values;
		states[i]->values = nullptr;
	}
}

static void SummaryStatsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                 idx_t count, idx_t offset) {
	auto states = FlatVector::GetData<SummaryStatsState *>(state_vector);
	auto &children = StructVector::GetEntries(result);

	bool bias_correction = true;
	int32_t quantile_type = 7;
	if (aggr_input_data.bind_data) {
		auto &bd = aggr_input_data.bind_data->Cast<SummaryStatsBindData>();
		bias_correction = bd.bias_correction;
		quantile_type = bd.quantile_type;
	}

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		auto idx = i + offset;

		if (!state.values || state.values->empty()) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		auto &values = *state.values;
		idx_t n = values.size();
		double dn = static_cast<double>(n);

		// Moments ─────────────────────────────────────────────────────
		double mean = 0.0;
		for (double v : values) {
			mean += v;
		}
		mean /= dn;

		double m2 = 0.0; // central moment order 2
		double m3 = 0.0; // central moment order 3
		double m4 = 0.0; // central moment order 4
		for (double v : values) {
			double d = v - mean;
			double d2 = d * d;
			m2 += d2;
			m3 += d2 * d;
			m4 += d2 * d2;
		}

		double variance = 0.0;
		double sd = 0.0;
		if (n >= 2) {
			variance = m2 / (dn - 1.0); // unbiased (Bessel-corrected)
			sd = std::sqrt(variance);
		}

		// Skewness: g1 is the population formula (m3/n) / (m2/n)^1.5; the
		// bias_correction branch then applies the Fisher-Pearson adjustment.
		// NaN for n < 3 or zero variance regardless of formula.
		double skewness = std::numeric_limits<double>::quiet_NaN();
		if (n >= 3 && m2 > 0.0) {
			double m2n = m2 / dn;
			double m3n = m3 / dn;
			double g1 = m3n / std::pow(m2n, 1.5);
			skewness = bias_correction ? std::sqrt(dn * (dn - 1.0)) / (dn - 2.0) * g1 : g1;
		}

		// Excess kurtosis: g2 is the population m4/m2² - 3; the bias_correction
		// branch applies R's e1071 / Excel KURT adjustment. NaN for n < 4 or
		// zero variance.
		double kurtosis = std::numeric_limits<double>::quiet_NaN();
		if (n >= 4 && m2 > 0.0) {
			double m2n = m2 / dn;
			double m4n = m4 / dn;
			double g2 = m4n / (m2n * m2n) - 3.0;
			kurtosis = bias_correction
			               ? ((dn - 1.0) / ((dn - 2.0) * (dn - 3.0))) * ((dn + 1.0) * g2 + 6.0)
			               : g2;
		}

		// Order stats ─────────────────────────────────────────────────
		std::sort(values.begin(), values.end());
		double vmin = values.front();
		double vmax = values.back();
		double q1 = Quantile(values, 0.25, quantile_type);
		double median = Quantile(values, 0.50, quantile_type);
		double q3 = Quantile(values, 0.75, quantile_type);
		double iqr = q3 - q1;

		// Mode ─────────────────────────────────────────────────────────
		// Single scan over the sorted values tracks run lengths. Each
		// completed run is fed to `consider_run`, which keeps the longest
		// seen so far and counts ties at the maximum length. After the loop
		// the trailing run is fed once. For all-distinct input (n>=2 and
		// every run is length 1) we report mode = NaN / frequency = 0,
		// matching SAS PROC UNIVARIATE's "Mode ." convention.
		int64_t mode_count = 0;
		double mode_value = values[0];
		int64_t n_modes_at_max = 0;
		int64_t run_len = 1;
		auto consider_run = [&](double run_val) {
			if (run_len > mode_count) {
				mode_count = run_len;
				mode_value = run_val;
				n_modes_at_max = 1;
			} else if (run_len == mode_count) {
				n_modes_at_max++;
			}
		};
		for (idx_t k = 1; k < n; k++) {
			if (values[k] == values[k - 1]) {
				run_len++;
			} else {
				consider_run(values[k - 1]);
				run_len = 1;
			}
		}
		consider_run(values[n - 1]);

		bool all_distinct = n > 1 && mode_count == 1;
		double mode_out = all_distinct ? std::numeric_limits<double>::quiet_NaN() : mode_value;
		int64_t mode_freq_out = all_distinct ? 0 : mode_count;
		bool is_multimodal = !all_distinct && n_modes_at_max > 1;

		// Write out ──────────────────────────────────────────────────
		FlatVector::GetData<int64_t>(*children[0])[idx] = static_cast<int64_t>(n);
		FlatVector::GetData<int64_t>(*children[1])[idx] = state.n_missing;
		FlatVector::GetData<double>(*children[2])[idx] = mean;
		FlatVector::GetData<double>(*children[3])[idx] = sd;
		FlatVector::GetData<double>(*children[4])[idx] = variance;
		FlatVector::GetData<double>(*children[5])[idx] = vmin;
		FlatVector::GetData<double>(*children[6])[idx] = q1;
		FlatVector::GetData<double>(*children[7])[idx] = median;
		FlatVector::GetData<double>(*children[8])[idx] = q3;
		FlatVector::GetData<double>(*children[9])[idx] = vmax;
		FlatVector::GetData<double>(*children[10])[idx] = iqr;
		FlatVector::GetData<double>(*children[11])[idx] = skewness;
		FlatVector::GetData<double>(*children[12])[idx] = kurtosis;
		FlatVector::GetData<double>(*children[13])[idx] = mode_out;
		FlatVector::GetData<int64_t>(*children[14])[idx] = mode_freq_out;
		FlatVector::GetData<bool>(*children[15])[idx] = is_multimodal;
	}
}

} // namespace

static AggregateFunction MakeSummaryStats(vector<LogicalType> args, bind_aggregate_function_t bind_fn) {
	return AggregateFunction("summary_stats", std::move(args), SummaryStatsResultType(),
	                         AggregateFunction::StateSize<SummaryStatsState>, SummaryStatsInit, SummaryStatsUpdate,
	                         SummaryStatsCombine, SummaryStatsFinalize, FunctionNullHandling::SPECIAL_HANDLING,
	                         nullptr, bind_fn, SummaryStatsDestroy);
}

void RegisterSummaryStats(ExtensionLoader &loader) {
	AggregateFunctionSet set("summary_stats");
	set.AddFunction(MakeSummaryStats({LogicalType::DOUBLE}, SummaryStatsBindNoArg));
	set.AddFunction(MakeSummaryStats({LogicalType::DOUBLE, LogicalType::BOOLEAN}, SummaryStatsBindBias));
	set.AddFunction(MakeSummaryStats({LogicalType::DOUBLE, LogicalType::BOOLEAN, LogicalType::INTEGER},
	                                  SummaryStatsBindBiasQuantile));
	loader.RegisterFunction(set);
}

} // namespace duckdb
