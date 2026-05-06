#include "rank_correlation_function.hpp"
#include "distributions.hpp"
#include "stats_validation.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace duckdb {

namespace {

// ── Bind data ──────────────────────────────────────────────────────────────

struct SpearmanBindData : public FunctionData {
	double alpha = 0.05;
	std::string alternative = "two-sided";

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<SpearmanBindData>();
		copy->alpha = alpha;
		copy->alternative = alternative;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<SpearmanBindData>();
		return alpha == other.alpha && alternative == other.alternative;
	}
};

struct KendallBindData : public FunctionData {
	std::string alternative = "two-sided";

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<KendallBindData>();
		copy->alternative = alternative;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		return alternative == other_p.Cast<KendallBindData>().alternative;
	}
};

static double ExtractConstantDouble(ClientContext &context, AggregateFunction &function,
                                    vector<unique_ptr<Expression>> &arguments, idx_t arg_idx,
                                    const char *fn_name) {
	if (!arguments[arg_idx]->IsFoldable()) {
		throw BinderException("%s: parameter must be a constant value", fn_name);
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[arg_idx]);
	double result = val.GetValue<double>();
	Function::EraseArgument(function, arguments, arg_idx);
	return result;
}

static std::string ExtractConstantString(ClientContext &context, AggregateFunction &function,
                                         vector<unique_ptr<Expression>> &arguments, idx_t arg_idx,
                                         const char *fn_name) {
	if (!arguments[arg_idx]->IsFoldable()) {
		throw BinderException("%s: parameter must be a constant value", fn_name);
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[arg_idx]);
	std::string result = val.GetValue<std::string>();
	Function::EraseArgument(function, arguments, arg_idx);
	return result;
}

// Spearman binders: (x,y), (x,y,alpha), (x,y,alpha,alternative)

static unique_ptr<FunctionData> SpearmanBind2(ClientContext &, AggregateFunction &,
                                              vector<unique_ptr<Expression>> &) {
	return make_uniq<SpearmanBindData>();
}

static unique_ptr<FunctionData> SpearmanBind3(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
	auto bd = make_uniq<SpearmanBindData>();
	bd->alpha = ExtractConstantDouble(context, function, arguments, 2, "spearman_test");
	sdv::ValidateAlpha(bd->alpha, "spearman_test");
	return std::move(bd);
}

static unique_ptr<FunctionData> SpearmanBind4(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
	auto bd = make_uniq<SpearmanBindData>();
	bd->alternative = ExtractConstantString(context, function, arguments, 3, "spearman_test");
	sdv::ValidateAlternative(bd->alternative, "spearman_test");
	bd->alpha = ExtractConstantDouble(context, function, arguments, 2, "spearman_test");
	sdv::ValidateAlpha(bd->alpha, "spearman_test");
	return std::move(bd);
}

// Kendall binders: (x,y), (x,y,alternative)

static unique_ptr<FunctionData> KendallBind2(ClientContext &, AggregateFunction &,
                                             vector<unique_ptr<Expression>> &) {
	return make_uniq<KendallBindData>();
}

static unique_ptr<FunctionData> KendallBind3(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	auto bd = make_uniq<KendallBindData>();
	bd->alternative = ExtractConstantString(context, function, arguments, 2, "kendall_test");
	sdv::ValidateAlternative(bd->alternative, "kendall_test");
	return std::move(bd);
}

// ── Shared aggregate state (paired buffers) ────────────────────────────────

struct CorrelationState {
	std::vector<double> *x;
	std::vector<double> *y;
};

static void CorrelationInit(const AggregateFunction &, data_ptr_t state_p) {
	auto &state = *reinterpret_cast<CorrelationState *>(state_p);
	state.x = nullptr;
	state.y = nullptr;
}

static void CorrelationUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector,
                              idx_t count) {
	UnifiedVectorFormat xdata, ydata;
	inputs[0].ToUnifiedFormat(count, xdata);
	inputs[1].ToUnifiedFormat(count, ydata);

	auto states = FlatVector::GetData<CorrelationState *>(state_vector);
	auto xv = UnifiedVectorFormat::GetData<double>(xdata);
	auto yv = UnifiedVectorFormat::GetData<double>(ydata);

	for (idx_t i = 0; i < count; i++) {
		auto x_idx = xdata.sel->get_index(i);
		auto y_idx = ydata.sel->get_index(i);
		// Pairwise complete: drop any pair where either side is NULL or NaN.
		if (!xdata.validity.RowIsValid(x_idx) || !ydata.validity.RowIsValid(y_idx)) {
			continue;
		}
		double x = xv[x_idx];
		double y = yv[y_idx];
		if (std::isnan(x) || std::isnan(y)) {
			continue;
		}
		auto &state = *states[i];
		if (!state.x) {
			state.x = new std::vector<double>();
			state.y = new std::vector<double>();
		}
		state.x->push_back(x);
		state.y->push_back(y);
	}
}

static void CorrelationCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src = FlatVector::GetData<CorrelationState *>(source);
	auto tgt = FlatVector::GetData<CorrelationState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto &s = *src[i];
		auto &t = *tgt[i];
		if (!s.x || s.x->empty()) {
			continue;
		}
		if (!t.x) {
			t.x = new std::vector<double>();
			t.y = new std::vector<double>();
		}
		t.x->insert(t.x->end(), s.x->begin(), s.x->end());
		t.y->insert(t.y->end(), s.y->begin(), s.y->end());
	}
}

static void CorrelationDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	auto states = FlatVector::GetData<CorrelationState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		delete states[i]->x;
		delete states[i]->y;
	}
}

// ── Spearman ───────────────────────────────────────────────────────────────

static LogicalType SpearmanResultType() {
	child_list_t<LogicalType> children;
	children.emplace_back("test_type", LogicalType::VARCHAR);
	children.emplace_back("rho", LogicalType::DOUBLE);
	children.emplace_back("t_statistic", LogicalType::DOUBLE);
	children.emplace_back("df", LogicalType::DOUBLE);
	children.emplace_back("p_value", LogicalType::DOUBLE);
	children.emplace_back("alternative", LogicalType::VARCHAR);
	children.emplace_back("ci_lower", LogicalType::DOUBLE);
	children.emplace_back("ci_upper", LogicalType::DOUBLE);
	children.emplace_back("n", LogicalType::BIGINT);
	return LogicalType::STRUCT(std::move(children));
}

// Replace `values` in-place with their fractional (mid-)ranks. Tied values
// share the average of their positions: e.g. [10, 20, 20, 30] → [1, 2.5, 2.5, 4].
static void AssignMidranks(std::vector<double> &values) {
	idx_t n = values.size();
	std::vector<idx_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(),
	          [&](idx_t a, idx_t b) { return values[a] < values[b]; });

	std::vector<double> ranks(n);
	idx_t i = 0;
	while (i < n) {
		idx_t j = i;
		while (j < n && values[order[j]] == values[order[i]]) {
			j++;
		}
		double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
		for (idx_t k = i; k < j; k++) {
			ranks[order[k]] = avg_rank;
		}
		i = j;
	}
	values = std::move(ranks);
}

static double SpearmanPValue(double t_stat, double df, const std::string &alternative) {
	if (alternative == "two-sided") {
		return 2.0 * (1.0 - stats_duck::StudentTCDF(std::abs(t_stat), df));
	}
	if (alternative == "less") {
		return stats_duck::StudentTCDF(t_stat, df);
	}
	return 1.0 - stats_duck::StudentTCDF(t_stat, df);
}

static void SpearmanFinalize(Vector &state_vector, AggregateInputData &aggr_input_data,
                             Vector &result, idx_t count, idx_t offset) {
	auto states = FlatVector::GetData<CorrelationState *>(state_vector);
	auto &children = StructVector::GetEntries(result);

	double alpha = 0.05;
	std::string alternative = "two-sided";
	if (aggr_input_data.bind_data) {
		auto &bd = aggr_input_data.bind_data->Cast<SpearmanBindData>();
		alpha = bd.alpha;
		alternative = bd.alternative;
	}

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		auto idx = i + offset;

		if (!state.x || state.x->size() < 3) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		// Rank x and y independently (midranks for ties).
		std::vector<double> rx = *state.x;
		std::vector<double> ry = *state.y;
		AssignMidranks(rx);
		AssignMidranks(ry);

		// Pearson on the rank arrays.
		idx_t n = rx.size();
		double n_d = static_cast<double>(n);
		double mean_rx = std::accumulate(rx.begin(), rx.end(), 0.0) / n_d;
		double mean_ry = std::accumulate(ry.begin(), ry.end(), 0.0) / n_d;

		double cxx = 0.0, cyy = 0.0, cxy = 0.0;
		for (idx_t k = 0; k < n; k++) {
			double dx = rx[k] - mean_rx;
			double dy = ry[k] - mean_ry;
			cxx += dx * dx;
			cyy += dy * dy;
			cxy += dx * dy;
		}

		// All-tied input on either side → no rank variance, rho undefined.
		if (cxx <= 0.0 || cyy <= 0.0) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		double rho = cxy / std::sqrt(cxx * cyy);
		if (rho > 1.0) {
			rho = 1.0;
		} else if (rho < -1.0) {
			rho = -1.0;
		}

		double df = n_d - 2.0;
		double denom = 1.0 - rho * rho;
		double t_stat;
		if (denom <= 0.0) {
			t_stat = std::copysign(std::numeric_limits<double>::infinity(), rho);
		} else {
			t_stat = rho * std::sqrt(df / denom);
		}

		// StudentTCDF handles ±inf correctly, so no short-circuit on infinite t_stat:
		// for rho=+1 with alternative='less', p must be 1.0, not 0.0.
		double p_value = SpearmanPValue(t_stat, df, alternative);

		// Fisher-z CI on rho — two-sided regardless of `alternative`, matching cor.test.
		double ci_lower = std::numeric_limits<double>::quiet_NaN();
		double ci_upper = std::numeric_limits<double>::quiet_NaN();
		if (n >= 4 && std::abs(rho) < 1.0) {
			double z = std::atanh(rho);
			double se = 1.0 / std::sqrt(n_d - 3.0);
			double z_crit = stats_duck::NormalQuantile(1.0 - alpha / 2.0);
			ci_lower = std::tanh(z - z_crit * se);
			ci_upper = std::tanh(z + z_crit * se);
		}

		FlatVector::GetData<string_t>(*children[0])[idx] =
		    StringVector::AddString(*children[0], "Spearman Correlation");
		FlatVector::GetData<double>(*children[1])[idx] = rho;
		FlatVector::GetData<double>(*children[2])[idx] = t_stat;
		FlatVector::GetData<double>(*children[3])[idx] = df;
		FlatVector::GetData<double>(*children[4])[idx] = p_value;
		FlatVector::GetData<string_t>(*children[5])[idx] =
		    StringVector::AddString(*children[5], alternative);
		FlatVector::GetData<double>(*children[6])[idx] = ci_lower;
		FlatVector::GetData<double>(*children[7])[idx] = ci_upper;
		FlatVector::GetData<int64_t>(*children[8])[idx] = static_cast<int64_t>(n);
	}
}

static AggregateFunction MakeSpearman(vector<LogicalType> args, bind_aggregate_function_t bind_fn) {
	return AggregateFunction("spearman_test", std::move(args), SpearmanResultType(),
	                         AggregateFunction::StateSize<CorrelationState>, CorrelationInit,
	                         CorrelationUpdate, CorrelationCombine, SpearmanFinalize,
	                         FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, bind_fn,
	                         CorrelationDestroy);
}

// ── Kendall tau-b ──────────────────────────────────────────────────────────

static LogicalType KendallResultType() {
	child_list_t<LogicalType> children;
	children.emplace_back("test_type", LogicalType::VARCHAR);
	children.emplace_back("tau", LogicalType::DOUBLE);
	children.emplace_back("z_statistic", LogicalType::DOUBLE);
	children.emplace_back("p_value", LogicalType::DOUBLE);
	children.emplace_back("alternative", LogicalType::VARCHAR);
	children.emplace_back("n", LogicalType::BIGINT);
	return LogicalType::STRUCT(std::move(children));
}

// Tied-group moments needed by Knight's variance formula. Computed by
// scanning the sorted vector and summing over each run of equal values.
struct TieMoments {
	double sum_t_tm1 = 0.0;      // Σ t(t-1)
	double sum_t_tm1_2tp5 = 0.0; // Σ t(t-1)(2t+5)
	double sum_t_tm1_tm2 = 0.0;  // Σ t(t-1)(t-2)
};

static TieMoments TieMomentsOf(const std::vector<double> &values) {
	TieMoments m;
	std::vector<double> sorted = values;
	std::sort(sorted.begin(), sorted.end());
	idx_t n = sorted.size();
	idx_t i = 0;
	while (i < n) {
		idx_t j = i;
		while (j < n && sorted[j] == sorted[i]) {
			j++;
		}
		double t = static_cast<double>(j - i);
		if (t > 1.0) {
			m.sum_t_tm1 += t * (t - 1.0);
			m.sum_t_tm1_2tp5 += t * (t - 1.0) * (2.0 * t + 5.0);
			m.sum_t_tm1_tm2 += t * (t - 1.0) * (t - 2.0);
		}
		i = j;
	}
	return m;
}

static double KendallPValue(double z, const std::string &alternative) {
	if (alternative == "two-sided") {
		return 2.0 * (1.0 - stats_duck::NormalCDF(std::abs(z)));
	}
	if (alternative == "less") {
		return stats_duck::NormalCDF(z);
	}
	return 1.0 - stats_duck::NormalCDF(z);
}

static void KendallFinalize(Vector &state_vector, AggregateInputData &aggr_input_data,
                            Vector &result, idx_t count, idx_t offset) {
	auto states = FlatVector::GetData<CorrelationState *>(state_vector);
	auto &children = StructVector::GetEntries(result);

	std::string alternative = "two-sided";
	if (aggr_input_data.bind_data) {
		alternative = aggr_input_data.bind_data->Cast<KendallBindData>().alternative;
	}

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		auto idx = i + offset;

		if (!state.x || state.x->size() < 3) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}

		const auto &x = *state.x;
		const auto &y = *state.y;
		idx_t n = x.size();
		double n_d = static_cast<double>(n);

		// O(n²) pair counting. P = concordant, Q = discordant, T = tied on x only,
		// U = tied on y only. Pairs tied on both contribute to neither denominator.
		double P = 0.0, Q = 0.0, T = 0.0, U = 0.0;
		for (idx_t a = 0; a + 1 < n; a++) {
			for (idx_t b = a + 1; b < n; b++) {
				double dx = x[a] - x[b];
				double dy = y[a] - y[b];
				if (dx == 0.0 && dy == 0.0) {
					continue;
				}
				if (dx == 0.0) {
					T += 1.0;
				} else if (dy == 0.0) {
					U += 1.0;
				} else if ((dx > 0.0) == (dy > 0.0)) {
					P += 1.0;
				} else {
					Q += 1.0;
				}
			}
		}

		double denom = std::sqrt((P + Q + T) * (P + Q + U));
		if (denom <= 0.0) {
			FlatVector::SetNull(result, idx, true);
			continue;
		}
		double tau = (P - Q) / denom;
		if (tau > 1.0) {
			tau = 1.0;
		} else if (tau < -1.0) {
			tau = -1.0;
		}

		// Knight's tie-corrected variance for S = P - Q.
		auto mx = TieMomentsOf(x);
		auto my = TieMomentsOf(y);
		double v0 = n_d * (n_d - 1.0) * (2.0 * n_d + 5.0);
		double vt = mx.sum_t_tm1_2tp5;
		double vu = my.sum_t_tm1_2tp5;
		double v1 = (n_d > 1.0) ? (mx.sum_t_tm1 * my.sum_t_tm1) / (2.0 * n_d * (n_d - 1.0)) : 0.0;
		double v2 = (n_d > 2.0) ? (mx.sum_t_tm1_tm2 * my.sum_t_tm1_tm2) /
		                              (9.0 * n_d * (n_d - 1.0) * (n_d - 2.0))
		                        : 0.0;
		double var_S = (v0 - vt - vu) / 18.0 + v1 + v2;

		double z = (var_S > 0.0) ? (P - Q) / std::sqrt(var_S) : 0.0;
		double p_value = KendallPValue(z, alternative);

		FlatVector::GetData<string_t>(*children[0])[idx] =
		    StringVector::AddString(*children[0], "Kendall tau-b");
		FlatVector::GetData<double>(*children[1])[idx] = tau;
		FlatVector::GetData<double>(*children[2])[idx] = z;
		FlatVector::GetData<double>(*children[3])[idx] = p_value;
		FlatVector::GetData<string_t>(*children[4])[idx] =
		    StringVector::AddString(*children[4], alternative);
		FlatVector::GetData<int64_t>(*children[5])[idx] = static_cast<int64_t>(n);
	}
}

static AggregateFunction MakeKendall(vector<LogicalType> args, bind_aggregate_function_t bind_fn) {
	return AggregateFunction("kendall_test", std::move(args), KendallResultType(),
	                         AggregateFunction::StateSize<CorrelationState>, CorrelationInit,
	                         CorrelationUpdate, CorrelationCombine, KendallFinalize,
	                         FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, bind_fn,
	                         CorrelationDestroy);
}

} // namespace

void RegisterSpearmanTest(ExtensionLoader &loader) {
	AggregateFunctionSet set("spearman_test");
	set.AddFunction(MakeSpearman({LogicalType::DOUBLE, LogicalType::DOUBLE}, SpearmanBind2));
	set.AddFunction(
	    MakeSpearman({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, SpearmanBind3));
	set.AddFunction(MakeSpearman(
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
	    SpearmanBind4));
	loader.RegisterFunction(set);
}

void RegisterKendallTest(ExtensionLoader &loader) {
	AggregateFunctionSet set("kendall_test");
	set.AddFunction(MakeKendall({LogicalType::DOUBLE, LogicalType::DOUBLE}, KendallBind2));
	set.AddFunction(
	    MakeKendall({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR}, KendallBind3));
	loader.RegisterFunction(set);
}

} // namespace duckdb
