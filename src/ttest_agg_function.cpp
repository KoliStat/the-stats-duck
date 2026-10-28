#include "ttest_agg_function.hpp"
#include "register_documented.hpp"
#include "distributions.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb_compat.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression.hpp"

#include <cmath>
#include <limits>

namespace duckdb {

namespace compat = statsduck::compat;

// ─── Bind Data ──────────────────────────────────────────────────────────────────

struct TTestAggBindData : public FunctionData {
	double mu = 0.0;
	double alpha = 0.05;
	string alternative = "two-sided";
	bool equal_var = false;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<TTestAggBindData>();
		copy->mu = mu;
		copy->alpha = alpha;
		copy->alternative = alternative;
		copy->equal_var = equal_var;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<TTestAggBindData>();
		return mu == other.mu && alpha == other.alpha && alternative == other.alternative &&
		       equal_var == other.equal_var;
	}
};

// ─── Result STRUCT type ─────────────────────────────────────────────────────────

static LogicalType TTestResultType() {
	child_list_t<LogicalType> children;
	children.emplace_back("test_type", LogicalType::VARCHAR);
	children.emplace_back("t_statistic", LogicalType::DOUBLE);
	children.emplace_back("df", LogicalType::DOUBLE);
	children.emplace_back("p_value", LogicalType::DOUBLE);
	children.emplace_back("alternative", LogicalType::VARCHAR);
	children.emplace_back("mean_diff", LogicalType::DOUBLE);
	children.emplace_back("ci_lower", LogicalType::DOUBLE);
	children.emplace_back("ci_upper", LogicalType::DOUBLE);
	children.emplace_back("cohens_d", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(children));
}

// ─── P-value and CI helpers ─────────────────────────────────────────────────────

static double AggComputePValue(double t_stat, double df, const string &alternative) {
	if (alternative == "two-sided") {
		return 2.0 * (1.0 - stats_duck::StudentTCDF(std::abs(t_stat), df));
	} else if (alternative == "less") {
		return stats_duck::StudentTCDF(t_stat, df);
	} else {
		return 1.0 - stats_duck::StudentTCDF(t_stat, df);
	}
}

static void AggComputeCI(double mean_diff, double se, double df, double alpha, const string &alternative,
                          double &ci_lower, double &ci_upper) {
	if (alternative == "two-sided") {
		double t_crit = stats_duck::StudentTQuantile(1.0 - alpha / 2.0, df);
		ci_lower = mean_diff - t_crit * se;
		ci_upper = mean_diff + t_crit * se;
	} else if (alternative == "less") {
		double t_crit = stats_duck::StudentTQuantile(1.0 - alpha, df);
		ci_lower = -std::numeric_limits<double>::infinity();
		ci_upper = mean_diff + t_crit * se;
	} else {
		double t_crit = stats_duck::StudentTQuantile(1.0 - alpha, df);
		ci_lower = mean_diff - t_crit * se;
		ci_upper = std::numeric_limits<double>::infinity();
	}
}

// ─── Welford state helpers ──────────────────────────────────────────────────────

struct WelfordState {
	int64_t count;
	double mean;
	double m2;
};

static void WelfordInit(WelfordState &s) {
	s.count = 0;
	s.mean = 0.0;
	s.m2 = 0.0;
}

static void WelfordUpdate(WelfordState &s, double val) {
	s.count++;
	double delta = val - s.mean;
	s.mean += delta / static_cast<double>(s.count);
	double delta2 = val - s.mean;
	s.m2 += delta * delta2;
}

static void WelfordCombine(WelfordState &target, const WelfordState &source) {
	if (source.count == 0) {
		return;
	}
	if (target.count == 0) {
		target = source;
		return;
	}
	int64_t combined = target.count + source.count;
	double delta = source.mean - target.mean;
	target.mean +=
	    delta * static_cast<double>(source.count) / static_cast<double>(combined);
	target.m2 += source.m2 + delta * delta * static_cast<double>(target.count) *
	                              static_cast<double>(source.count) /
	                              static_cast<double>(combined);
	target.count = combined;
}

// ─── Finalize helper: write one STRUCT result ───────────────────────────────────

static void WriteStructResult(compat::StructChildren &children, idx_t idx, const string &test_type,
                               double t_stat, double df, double p_value, const string &alternative,
                               double mean_diff, double ci_lower, double ci_upper, double cohens_d) {
	compat::WriteData<string_t>(compat::Deref(children[0]))[idx] = StringVector::AddString(compat::Deref(children[0]), test_type);
	compat::WriteData<double>(compat::Deref(children[1]))[idx] = t_stat;
	compat::WriteData<double>(compat::Deref(children[2]))[idx] = df;
	compat::WriteData<double>(compat::Deref(children[3]))[idx] = p_value;
	compat::WriteData<string_t>(compat::Deref(children[4]))[idx] = StringVector::AddString(compat::Deref(children[4]), alternative);
	compat::WriteData<double>(compat::Deref(children[5]))[idx] = mean_diff;
	compat::WriteData<double>(compat::Deref(children[6]))[idx] = ci_lower;
	compat::WriteData<double>(compat::Deref(children[7]))[idx] = ci_upper;
	compat::WriteData<double>(compat::Deref(children[8]))[idx] = cohens_d;
}

// ─── Bind helpers for extracting constant parameters ────────────────────────────

static void ValidateAlternative(const string &alt) {
	if (alt != "two-sided" && alt != "less" && alt != "greater") {
		throw InvalidInputException("alternative must be 'two-sided', 'less', or 'greater'");
	}
}

static void ValidateAlpha(double alpha) {
	if (alpha <= 0.0 || alpha >= 1.0) {
		throw InvalidInputException("alpha must be between 0 and 1");
	}
}

// Extract a constant double argument, erase it from arguments, return value
static double ExtractConstantDouble(ClientContext &context, compat::BindFunction &function,
                                     vector<unique_ptr<Expression>> &arguments, idx_t arg_idx) {
	if (!arguments[arg_idx]->IsFoldable()) {
		throw BinderException("Parameter must be a constant value");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[arg_idx]);
	double result = val.GetValue<double>();
	compat::EraseArgument(function, arguments, arg_idx);
	return result;
}

static string ExtractConstantString(ClientContext &context, compat::BindFunction &function,
                                     vector<unique_ptr<Expression>> &arguments, idx_t arg_idx) {
	if (!arguments[arg_idx]->IsFoldable()) {
		throw BinderException("Parameter must be a constant value");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[arg_idx]);
	string result = val.GetValue<string>();
	compat::EraseArgument(function, arguments, arg_idx);
	return result;
}

static bool ExtractConstantBool(ClientContext &context, compat::BindFunction &function,
                                 vector<unique_ptr<Expression>> &arguments, idx_t arg_idx) {
	if (!arguments[arg_idx]->IsFoldable()) {
		throw BinderException("Parameter must be a constant value");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[arg_idx]);
	bool result = val.GetValue<bool>();
	compat::EraseArgument(function, arguments, arg_idx);
	return result;
}

// =============================================================================
// ttest_1samp aggregate
// =============================================================================

struct TTest1SampState {
	WelfordState ws;
};

STATS_DUCK_AGGREGATE_INIT(TTest1SampInit) {
	auto &state = *reinterpret_cast<TTest1SampState *>(state_p);
	WelfordInit(state.ws);
}

static void TTest1SampUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	auto &input = inputs[0];
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	auto states = FlatVector::GetData<TTest1SampState *>(state_vector);
	auto values = UnifiedVectorFormat::GetData<double>(idata);

	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (!idata.validity.RowIsValid(idx)) {
			continue;
		}
		// NaN is treated as missing, like NULL.
		if (std::isnan(values[idx])) {
			continue;
		}
		WelfordUpdate(states[i]->ws, values[idx]);
	}
}

static void TTest1SampCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src = FlatVector::GetData<TTest1SampState *>(source);
	auto tgt = FlatVector::GetData<TTest1SampState *>(target);
	for (idx_t i = 0; i < count; i++) {
		WelfordCombine(tgt[i]->ws, src[i]->ws);
	}
}

static void TTest1SampFinalize(Vector &state_vector, compat::FinalizeInput &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
	auto states = FlatVector::GetData<TTest1SampState *>(state_vector);
	auto &children = StructVector::GetEntries(result);

	double mu = 0.0, alpha = 0.05;
	string alternative = "two-sided";
	if (aggr_input_data.bind_data) {
		auto &bd = aggr_input_data.bind_data->Cast<TTestAggBindData>();
		mu = bd.mu;
		alpha = bd.alpha;
		alternative = bd.alternative;
	}

	for (idx_t i = 0; i < count; i++) {
		auto &ws = states[i]->ws;
		auto idx = i + offset;

		if (ws.count < 2) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		double n = static_cast<double>(ws.count);
		double var = ws.m2 / (n - 1.0);
		double sd = std::sqrt(var);
		double se = sd / std::sqrt(n);
		double df = n - 1.0;
		double mean_diff = ws.mean - mu;
		double t_stat = mean_diff / se;
		double p_value = AggComputePValue(t_stat, df, alternative);
		double cohens_d = mean_diff / sd;
		double ci_lower, ci_upper;
		AggComputeCI(mean_diff, se, df, alpha, alternative, ci_lower, ci_upper);

		WriteStructResult(children, idx, "One-sample", t_stat, df, p_value, alternative, mean_diff, ci_lower,
		                  ci_upper, cohens_d);
	}
}

// Bind: ttest_1samp(col) — defaults
STATS_DUCK_AGGREGATE_BIND(TTest1SampBind1) {
	return make_uniq<TTestAggBindData>();
}

// Bind: ttest_1samp(col, mu)
STATS_DUCK_AGGREGATE_BIND(TTest1SampBind2) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->mu = ExtractConstantDouble(context, function, arguments, 1);
	return std::move(bd);
}

// Bind: ttest_1samp(col, mu, alpha)
STATS_DUCK_AGGREGATE_BIND(TTest1SampBind3) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->alpha = ExtractConstantDouble(context, function, arguments, 2);
	ValidateAlpha(bd->alpha);
	bd->mu = ExtractConstantDouble(context, function, arguments, 1);
	return std::move(bd);
}

// Bind: ttest_1samp(col, mu, alpha, alternative)
STATS_DUCK_AGGREGATE_BIND(TTest1SampBind4) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->alternative = ExtractConstantString(context, function, arguments, 3);
	ValidateAlternative(bd->alternative);
	bd->alpha = ExtractConstantDouble(context, function, arguments, 2);
	ValidateAlpha(bd->alpha);
	bd->mu = ExtractConstantDouble(context, function, arguments, 1);
	return std::move(bd);
}

static AggregateFunction MakeTTest1SampAgg(vector<LogicalType> args, bind_aggregate_function_t bind_fn) {
	return AggregateFunction("ttest_1samp", std::move(args), TTestResultType(), AggregateFunction::StateSize<TTest1SampState>,
	                         TTest1SampInit, TTest1SampUpdate, TTest1SampCombine, TTest1SampFinalize,
	                         FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, bind_fn);
}

void RegisterTTest1SampAgg(ExtensionLoader &loader) {
	AggregateFunctionSet set("ttest_1samp");
	set.AddFunction(MakeTTest1SampAgg({LogicalType::DOUBLE}, TTest1SampBind1));
	set.AddFunction(MakeTTest1SampAgg({LogicalType::DOUBLE, LogicalType::DOUBLE}, TTest1SampBind2));
	set.AddFunction(
	    MakeTTest1SampAgg({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, TTest1SampBind3));
	set.AddFunction(MakeTTest1SampAgg(
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR}, TTest1SampBind4));
	statsduck::RegisterDocumented(loader, std::move(set),
	                              "One-sample t-test of the column mean against a hypothesized value "
	                              "(default mu = 0).",
	                              {"column", "mu", "alpha", "alternative"},
	                              "SELECT ttest_1samp(reaction_ms, 250.0) FROM trials");
}

// =============================================================================
// ttest_paired aggregate
// =============================================================================

// Reuses TTest1SampState (accumulates differences)

static void TTestPairedUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat idata1, idata2;
	inputs[0].ToUnifiedFormat(count, idata1);
	inputs[1].ToUnifiedFormat(count, idata2);

	auto states = FlatVector::GetData<TTest1SampState *>(state_vector);
	auto v1 = UnifiedVectorFormat::GetData<double>(idata1);
	auto v2 = UnifiedVectorFormat::GetData<double>(idata2);

	for (idx_t i = 0; i < count; i++) {
		auto idx1 = idata1.sel->get_index(i);
		auto idx2 = idata2.sel->get_index(i);
		if (!idata1.validity.RowIsValid(idx1) || !idata2.validity.RowIsValid(idx2)) {
			continue;
		}
		// A NaN on either side drops the pair, like a NULL.
		if (std::isnan(v1[idx1]) || std::isnan(v2[idx2])) {
			continue;
		}
		WelfordUpdate(states[i]->ws, v1[idx1] - v2[idx2]);
	}
}

static void TTestPairedFinalize(Vector &state_vector, compat::FinalizeInput &aggr_input_data, Vector &result,
                                 idx_t count, idx_t offset) {
	auto states = FlatVector::GetData<TTest1SampState *>(state_vector);
	auto &children = StructVector::GetEntries(result);

	double alpha = 0.05;
	string alternative = "two-sided";
	if (aggr_input_data.bind_data) {
		auto &bd = aggr_input_data.bind_data->Cast<TTestAggBindData>();
		alpha = bd.alpha;
		alternative = bd.alternative;
	}

	for (idx_t i = 0; i < count; i++) {
		auto &ws = states[i]->ws;
		auto idx = i + offset;

		if (ws.count < 2) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		double n = static_cast<double>(ws.count);
		double var = ws.m2 / (n - 1.0);
		double sd = std::sqrt(var);
		double se = sd / std::sqrt(n);
		double df = n - 1.0;
		double mean_diff = ws.mean;
		double t_stat = mean_diff / se;
		double p_value = AggComputePValue(t_stat, df, alternative);
		double cohens_d = mean_diff / sd;
		double ci_lower, ci_upper;
		AggComputeCI(mean_diff, se, df, alpha, alternative, ci_lower, ci_upper);

		WriteStructResult(children, idx, "Paired", t_stat, df, p_value, alternative, mean_diff, ci_lower,
		                  ci_upper, cohens_d);
	}
}

// Bind: ttest_paired(col1, col2) — defaults
STATS_DUCK_AGGREGATE_BIND(TTestPairedBind2) {
	return make_uniq<TTestAggBindData>();
}

// Bind: ttest_paired(col1, col2, alpha)
STATS_DUCK_AGGREGATE_BIND(TTestPairedBind3) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->alpha = ExtractConstantDouble(context, function, arguments, 2);
	ValidateAlpha(bd->alpha);
	return std::move(bd);
}

// Bind: ttest_paired(col1, col2, alpha, alternative)
STATS_DUCK_AGGREGATE_BIND(TTestPairedBind4) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->alternative = ExtractConstantString(context, function, arguments, 3);
	ValidateAlternative(bd->alternative);
	bd->alpha = ExtractConstantDouble(context, function, arguments, 2);
	ValidateAlpha(bd->alpha);
	return std::move(bd);
}

static AggregateFunction MakeTTestPairedAgg(vector<LogicalType> args, bind_aggregate_function_t bind_fn) {
	return AggregateFunction("ttest_paired", std::move(args), TTestResultType(),
	                         AggregateFunction::StateSize<TTest1SampState>, TTest1SampInit, TTestPairedUpdate,
	                         TTest1SampCombine, TTestPairedFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                         nullptr, bind_fn);
}

void RegisterTTestPairedAgg(ExtensionLoader &loader) {
	AggregateFunctionSet set("ttest_paired");
	set.AddFunction(MakeTTestPairedAgg({LogicalType::DOUBLE, LogicalType::DOUBLE}, TTestPairedBind2));
	set.AddFunction(
	    MakeTTestPairedAgg({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, TTestPairedBind3));
	set.AddFunction(MakeTTestPairedAgg(
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR}, TTestPairedBind4));
	statsduck::RegisterDocumented(loader, std::move(set),
	                              "Paired t-test on per-row differences.",
	                              {"column1", "column2", "alpha", "alternative"},
	                              "SELECT ttest_paired(before, after) FROM measurements");
}

// =============================================================================
// ttest_2samp aggregate
// =============================================================================

struct TTest2SampState {
	WelfordState ws_a;
	WelfordState ws_b;
};

STATS_DUCK_AGGREGATE_INIT(TTest2SampInit) {
	auto &state = *reinterpret_cast<TTest2SampState *>(state_p);
	WelfordInit(state.ws_a);
	WelfordInit(state.ws_b);
}

static void TTest2SampUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat idata1, idata2;
	inputs[0].ToUnifiedFormat(count, idata1);
	inputs[1].ToUnifiedFormat(count, idata2);

	auto states = FlatVector::GetData<TTest2SampState *>(state_vector);
	auto v1 = UnifiedVectorFormat::GetData<double>(idata1);
	auto v2 = UnifiedVectorFormat::GetData<double>(idata2);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];

		// Each column drops its own NULL and NaN values.
		auto idx1 = idata1.sel->get_index(i);
		if (idata1.validity.RowIsValid(idx1) && !std::isnan(v1[idx1])) {
			WelfordUpdate(state.ws_a, v1[idx1]);
		}

		auto idx2 = idata2.sel->get_index(i);
		if (idata2.validity.RowIsValid(idx2) && !std::isnan(v2[idx2])) {
			WelfordUpdate(state.ws_b, v2[idx2]);
		}
	}
}

static void TTest2SampCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src = FlatVector::GetData<TTest2SampState *>(source);
	auto tgt = FlatVector::GetData<TTest2SampState *>(target);
	for (idx_t i = 0; i < count; i++) {
		WelfordCombine(tgt[i]->ws_a, src[i]->ws_a);
		WelfordCombine(tgt[i]->ws_b, src[i]->ws_b);
	}
}

static void TTest2SampFinalize(Vector &state_vector, compat::FinalizeInput &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
	auto states = FlatVector::GetData<TTest2SampState *>(state_vector);
	auto &children = StructVector::GetEntries(result);

	double alpha = 0.05;
	string alternative = "two-sided";
	bool equal_var = false;
	if (aggr_input_data.bind_data) {
		auto &bd = aggr_input_data.bind_data->Cast<TTestAggBindData>();
		alpha = bd.alpha;
		alternative = bd.alternative;
		equal_var = bd.equal_var;
	}

	for (idx_t i = 0; i < count; i++) {
		auto &a = states[i]->ws_a;
		auto &b = states[i]->ws_b;
		auto idx = i + offset;

		if (a.count < 2 || b.count < 2) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		double n1 = static_cast<double>(a.count);
		double n2 = static_cast<double>(b.count);
		double var1 = a.m2 / (n1 - 1.0);
		double var2 = b.m2 / (n2 - 1.0);
		double mean_diff = a.mean - b.mean;

		double se, df;
		string test_type;

		if (equal_var) {
			double sp2 = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
			se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));
			df = n1 + n2 - 2.0;
			test_type = "Student Two-sample";
		} else {
			double v1n = var1 / n1;
			double v2n = var2 / n2;
			se = std::sqrt(v1n + v2n);
			double num = (v1n + v2n) * (v1n + v2n);
			double denom = (v1n * v1n) / (n1 - 1.0) + (v2n * v2n) / (n2 - 1.0);
			df = num / denom;
			test_type = "Welch Two-sample";
		}

		double t_stat = mean_diff / se;
		double p_value = AggComputePValue(t_stat, df, alternative);
		double sp = std::sqrt(((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0));
		double cohens_d = mean_diff / sp;
		double ci_lower, ci_upper;
		AggComputeCI(mean_diff, se, df, alpha, alternative, ci_lower, ci_upper);

		WriteStructResult(children, idx, test_type, t_stat, df, p_value, alternative, mean_diff, ci_lower,
		                  ci_upper, cohens_d);
	}
}

// Bind: ttest_2samp(col1, col2) — defaults (Welch)
STATS_DUCK_AGGREGATE_BIND(TTest2SampBind2) {
	return make_uniq<TTestAggBindData>();
}

// Bind: ttest_2samp(col1, col2, equal_var)
STATS_DUCK_AGGREGATE_BIND(TTest2SampBind3) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->equal_var = ExtractConstantBool(context, function, arguments, 2);
	return std::move(bd);
}

// Bind: ttest_2samp(col1, col2, equal_var, alpha)
STATS_DUCK_AGGREGATE_BIND(TTest2SampBind4) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->alpha = ExtractConstantDouble(context, function, arguments, 3);
	ValidateAlpha(bd->alpha);
	bd->equal_var = ExtractConstantBool(context, function, arguments, 2);
	return std::move(bd);
}

// Bind: ttest_2samp(col1, col2, equal_var, alpha, alternative)
STATS_DUCK_AGGREGATE_BIND(TTest2SampBind5) {
	auto bd = make_uniq<TTestAggBindData>();
	bd->alternative = ExtractConstantString(context, function, arguments, 4);
	ValidateAlternative(bd->alternative);
	bd->alpha = ExtractConstantDouble(context, function, arguments, 3);
	ValidateAlpha(bd->alpha);
	bd->equal_var = ExtractConstantBool(context, function, arguments, 2);
	return std::move(bd);
}

static AggregateFunction MakeTTest2SampAgg(vector<LogicalType> args, bind_aggregate_function_t bind_fn) {
	return AggregateFunction("ttest_2samp", std::move(args), TTestResultType(),
	                         AggregateFunction::StateSize<TTest2SampState>, TTest2SampInit, TTest2SampUpdate,
	                         TTest2SampCombine, TTest2SampFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                         nullptr, bind_fn);
}

void RegisterTTest2SampAgg(ExtensionLoader &loader) {
	AggregateFunctionSet set("ttest_2samp");
	set.AddFunction(MakeTTest2SampAgg({LogicalType::DOUBLE, LogicalType::DOUBLE}, TTest2SampBind2));
	set.AddFunction(
	    MakeTTest2SampAgg({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN}, TTest2SampBind3));
	set.AddFunction(MakeTTest2SampAgg({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN,
	                                   LogicalType::DOUBLE},
	                                  TTest2SampBind4));
	set.AddFunction(MakeTTest2SampAgg({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN,
	                                   LogicalType::DOUBLE, LogicalType::VARCHAR},
	                                  TTest2SampBind5));
	statsduck::RegisterDocumented(loader, std::move(set),
	                              "Two-sample t-test — Welch's by default, Student's pooled with "
	                              "equal_var := true.",
	                              {"column1", "column2", "equal_var", "alpha", "alternative"},
	                              "SELECT ttest_2samp(score_a, score_b) FROM results");
}

} // namespace duckdb
