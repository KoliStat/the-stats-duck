#include "lm_fit_function.hpp"

#include "lm_core.hpp" // statsduck::fit_lm / Vcov / LmResult / parse_vcov / vcov_name

#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Result STRUCT type.
//===--------------------------------------------------------------------===//

// One row per coefficient — the element type of the `coefficients` list.
static LogicalType LmFitCoefType() {
	child_list_t<LogicalType> f;
	f.emplace_back("term", LogicalType::VARCHAR);
	f.emplace_back("estimate", LogicalType::DOUBLE);
	f.emplace_back("std_error", LogicalType::DOUBLE);
	f.emplace_back("t_statistic", LogicalType::DOUBLE);
	f.emplace_back("p_value", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(f));
}

// Field order MUST match the writes in Finalize (indices below).
static LogicalType LmFitResultType() {
	child_list_t<LogicalType> c;
	c.emplace_back("coefficients", LogicalType::LIST(LmFitCoefType())); // 0
	c.emplace_back("n", LogicalType::BIGINT);                           // 1
	c.emplace_back("k", LogicalType::BIGINT);                           // 2
	c.emplace_back("df_residual", LogicalType::BIGINT);                 // 3
	c.emplace_back("r_squared", LogicalType::DOUBLE);                   // 4
	c.emplace_back("adj_r_squared", LogicalType::DOUBLE);               // 5
	c.emplace_back("sigma", LogicalType::DOUBLE);                       // 6
	c.emplace_back("f_statistic", LogicalType::DOUBLE);                 // 7
	c.emplace_back("f_p_value", LogicalType::DOUBLE);                   // 8
	c.emplace_back("has_intercept", LogicalType::BOOLEAN);              // 9
	c.emplace_back("vcov_type", LogicalType::VARCHAR);                  // 10
	return LogicalType::STRUCT(std::move(c));
}

//===--------------------------------------------------------------------===//
// Bind data — carries the constant vcov + intercept flag through to Finalize.
//===--------------------------------------------------------------------===//

struct LmFitBindData : public FunctionData {
	statsduck::Vcov vcov = statsduck::Vcov::kConst;
	bool intercept = true;

	unique_ptr<FunctionData> Copy() const override {
		auto c = make_uniq<LmFitBindData>();
		c->vcov = vcov;
		c->intercept = intercept;
		return std::move(c);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<LmFitBindData>();
		return vcov == o.vcov && intercept == o.intercept;
	}
};

static Value EvalConst(ClientContext &context, Expression &expr, const char *what) {
	if (!expr.IsFoldable()) {
		throw BinderException("lm_fit: '%s' must be a constant value", what);
	}
	Value v = ExpressionExecutor::EvaluateScalar(context, expr);
	if (v.IsNull()) {
		throw BinderException("lm_fit: '%s' must not be NULL", what);
	}
	return v;
}

static statsduck::Vcov ParseVcovArg(ClientContext &context, AggregateFunction &function,
                                    vector<unique_ptr<Expression>> &arguments, idx_t idx) {
	Value v = EvalConst(context, *arguments[idx], "vcov");
	statsduck::Vcov out;
	if (!statsduck::parse_vcov(v.GetValue<string>(), out)) {
		throw BinderException("lm_fit: unknown vcov '%s' — use 'const', 'HC0', 'HC1', 'HC2', or 'HC3'",
		                      v.GetValue<string>());
	}
	return out;
}

// lm_fit(y, x) — defaults: classical SEs, intercept on.
static unique_ptr<FunctionData> LmFitBind2(ClientContext &, AggregateFunction &,
                                           vector<unique_ptr<Expression>> &) {
	return make_uniq<LmFitBindData>();
}

// lm_fit(y, x, vcov)
static unique_ptr<FunctionData> LmFitBind3(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto bd = make_uniq<LmFitBindData>();
	bd->vcov = ParseVcovArg(context, function, arguments, 2);
	Function::EraseArgument(function, arguments, 2);
	return std::move(bd);
}

// lm_fit(y, x, vcov, add_intercept)
static unique_ptr<FunctionData> LmFitBind4(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto bd = make_uniq<LmFitBindData>();
	// Evaluate both constants first, then erase high→low so indices stay valid.
	bd->intercept = EvalConst(context, *arguments[3], "add_intercept").GetValue<bool>();
	bd->vcov = ParseVcovArg(context, function, arguments, 2);
	Function::EraseArgument(function, arguments, 3);
	Function::EraseArgument(function, arguments, 2);
	return std::move(bd);
}

//===--------------------------------------------------------------------===//
// State — buffers the design rows (HC2/HC3 leverages and the sandwich meat need
// per-row x, so sufficient statistics alone won't do). Heap-owned, lazy alloc.
//===--------------------------------------------------------------------===//

struct LmFitAccumulator {
	std::size_t p = 0;       // predictor width (list length)
	bool width_set = false;
	bool ragged = false;     // a row's list length disagreed → poison the group
	std::vector<double> y;   // n
	std::vector<double> x;   // n*p, row-major (predictors only; intercept added at fit)
};

struct LmFitState {
	LmFitAccumulator *acc;
};

static void LmFitInit(const AggregateFunction &, data_ptr_t state_p) {
	reinterpret_cast<LmFitState *>(state_p)->acc = nullptr;
}

static void LmFitUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	auto &y_in = inputs[0];
	auto &x_in = inputs[1]; // LIST(DOUBLE)

	UnifiedVectorFormat y_fmt, x_fmt;
	y_in.ToUnifiedFormat(count, y_fmt);
	x_in.ToUnifiedFormat(count, x_fmt);
	auto y_vals = UnifiedVectorFormat::GetData<double>(y_fmt);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(x_fmt);

	auto &child = ListVector::GetEntry(x_in);
	const idx_t child_size = ListVector::GetListSize(x_in);
	UnifiedVectorFormat child_fmt;
	child.ToUnifiedFormat(child_size, child_fmt);
	auto child_vals = UnifiedVectorFormat::GetData<double>(child_fmt);

	auto states = FlatVector::GetData<LmFitState *>(state_vector);

	for (idx_t i = 0; i < count; i++) {
		const auto yidx = y_fmt.sel->get_index(i);
		const auto xidx = x_fmt.sel->get_index(i);
		// Complete-case: skip NULL response or NULL design row.
		if (!y_fmt.validity.RowIsValid(yidx) || !x_fmt.validity.RowIsValid(xidx)) {
			continue;
		}
		const double yv = y_vals[yidx];
		if (!std::isfinite(yv)) {
			continue;
		}

		auto &st = *states[i];
		if (!st.acc) {
			st.acc = new LmFitAccumulator();
		}
		auto &acc = *st.acc;
		if (acc.ragged) {
			continue;
		}
		const auto ent = list_entries[xidx];
		const std::size_t len = ent.length;
		if (!acc.width_set) {
			acc.p = len;
			acc.width_set = true;
		} else if (len != acc.p) {
			acc.ragged = true; // inconsistent predictor count → group result becomes NULL
			continue;
		}

		// Gather predictors; roll back the partial append if any element is
		// NULL / non-finite so y and x stay row-aligned.
		const std::size_t base = acc.x.size();
		bool bad = false;
		for (std::size_t j = 0; j < len; j++) {
			const auto cidx = child_fmt.sel->get_index(ent.offset + j);
			if (!child_fmt.validity.RowIsValid(cidx)) {
				bad = true;
				break;
			}
			const double cv = child_vals[cidx];
			if (!std::isfinite(cv)) {
				bad = true;
				break;
			}
			acc.x.push_back(cv);
		}
		if (bad) {
			acc.x.resize(base);
			continue;
		}
		acc.y.push_back(yv);
	}
}

static void LmFitCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src = FlatVector::GetData<LmFitState *>(source);
	auto tgt = FlatVector::GetData<LmFitState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto *sa = src[i]->acc;
		if (!sa) {
			continue;
		}
		if (!tgt[i]->acc) {
			tgt[i]->acc = new LmFitAccumulator();
		}
		auto &ta = *tgt[i]->acc;
		if (sa->ragged) {
			ta.ragged = true;
		}
		if (ta.ragged) {
			continue;
		}
		if (sa->width_set) {
			if (!ta.width_set) {
				ta.p = sa->p;
				ta.width_set = true;
			} else if (ta.p != sa->p) {
				ta.ragged = true;
				continue;
			}
		}
		ta.y.insert(ta.y.end(), sa->y.begin(), sa->y.end());
		ta.x.insert(ta.x.end(), sa->x.begin(), sa->x.end());
	}
}

static void LmFitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	auto states = FlatVector::GetData<LmFitState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		delete states[i]->acc;
		states[i]->acc = nullptr;
	}
}

//===--------------------------------------------------------------------===//
// Finalize.
//===--------------------------------------------------------------------===//

// Write a DOUBLE struct-field, mapping NaN → SQL NULL (matches lm/lm_summary).
static inline void SetD(Vector &v, idx_t row, double x) {
	if (std::isnan(x)) {
		FlatVector::SetNull(v, row, true);
	} else {
		FlatVector::GetData<double>(v)[row] = x;
	}
}

static void LmFitFinalize(Vector &state_vector, AggregateInputData &input_data, Vector &result, idx_t count,
                          idx_t offset) {
	auto states = FlatVector::GetData<LmFitState *>(state_vector);

	statsduck::LmOptions opt;
	if (input_data.bind_data) {
		auto &bd = input_data.bind_data->Cast<LmFitBindData>();
		opt.vcov = bd.vcov;
		opt.intercept = bd.intercept;
	}

	// Pass 1 — fit each group; tally total coefficients for the list child.
	std::vector<statsduck::LmResult> fits(count);
	std::vector<bool> ok(count, false);
	idx_t total_coefs = 0;
	for (idx_t i = 0; i < count; i++) {
		auto *acc = states[i]->acc;
		if (!acc || acc->ragged || !acc->width_set || acc->y.empty()) {
			continue;
		}
		const std::size_t n = acc->y.size();
		const std::size_t p = acc->p;
		if (acc->x.size() != n * p) {
			continue; // defensive: row/width desync — emit NULL for this group
		}
		statsduck::linalg::Mat Xp(n, p);
		Xp.data = acc->x; // row-major n*p, exactly Mat's layout
		auto r = statsduck::fit_lm(acc->y, Xp, opt);
		if (r.ok) {
			total_coefs += r.k;
			fits[i] = std::move(r);
			ok[i] = true;
		}
	}

	auto &children = StructVector::GetEntries(result);

	// coefficients: LIST<STRUCT>. Grow the list child, then fetch field pointers.
	Vector &coef_list = *children[0];
	auto coef_entries = FlatVector::GetData<list_entry_t>(coef_list);
	const idx_t anchor = ListVector::GetListSize(coef_list);
	ListVector::Reserve(coef_list, anchor + total_coefs);
	Vector &coef_child = ListVector::GetEntry(coef_list); // STRUCT vector
	auto &cf = StructVector::GetEntries(coef_child);
	auto term_d = FlatVector::GetData<string_t>(*cf[0]);
	auto est_d = FlatVector::GetData<double>(*cf[1]);
	auto se_d = FlatVector::GetData<double>(*cf[2]);

	auto n_d = FlatVector::GetData<int64_t>(*children[1]);
	auto k_d = FlatVector::GetData<int64_t>(*children[2]);
	auto dfr_d = FlatVector::GetData<int64_t>(*children[3]);
	auto hint_d = FlatVector::GetData<bool>(*children[9]);

	idx_t out = anchor;
	for (idx_t i = 0; i < count; i++) {
		const idx_t idx = i + offset;
		if (!ok[i]) {
			FlatVector::SetNull(result, idx, true);
			coef_entries[idx] = list_entry_t(out, 0);
			continue;
		}
		auto &r = fits[i];
		coef_entries[idx] = list_entry_t(out, r.k);
		for (std::size_t j = 0; j < r.k; j++) {
			const idx_t pos = out + j;
			term_d[pos] = StringVector::AddString(*cf[0], r.terms[j]);
			est_d[pos] = r.beta[j];
			se_d[pos] = r.std_error[j];
			SetD(*cf[3], pos, r.t_statistic[j]);
			SetD(*cf[4], pos, r.p_value[j]);
		}
		out += r.k;

		n_d[idx] = static_cast<int64_t>(r.n);
		k_d[idx] = static_cast<int64_t>(r.k);
		dfr_d[idx] = static_cast<int64_t>(r.df_residual);
		SetD(*children[4], idx, r.r_squared);
		SetD(*children[5], idx, r.adj_r_squared);
		SetD(*children[6], idx, r.sigma);
		SetD(*children[7], idx, r.f_statistic);
		SetD(*children[8], idx, r.f_p_value);
		hint_d[idx] = r.has_intercept;
		FlatVector::GetData<string_t>(*children[10])[idx] =
		    StringVector::AddString(*children[10], statsduck::vcov_name(r.vcov));
	}
	ListVector::SetListSize(coef_list, out);
}

} // namespace

void RegisterLmFit(ExtensionLoader &loader) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	const auto ret = LmFitResultType();
	auto make = [&](vector<LogicalType> args, bind_aggregate_function_t bind) {
		return AggregateFunction("lm_fit", std::move(args), ret, AggregateFunction::StateSize<LmFitState>,
		                         LmFitInit, LmFitUpdate, LmFitCombine, LmFitFinalize,
		                         FunctionNullHandling::SPECIAL_HANDLING, nullptr, bind, LmFitDestroy);
	};

	AggregateFunctionSet set("lm_fit");
	set.AddFunction(make({LogicalType::DOUBLE, list_double}, LmFitBind2));
	set.AddFunction(make({LogicalType::DOUBLE, list_double, LogicalType::VARCHAR}, LmFitBind3));
	set.AddFunction(
	    make({LogicalType::DOUBLE, list_double, LogicalType::VARCHAR, LogicalType::BOOLEAN}, LmFitBind4));
	loader.RegisterFunction(set);
}

} // namespace duckdb
