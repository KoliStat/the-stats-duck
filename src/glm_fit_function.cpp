#include "glm_fit_function.hpp"
#include "register_documented.hpp"

#include "cluster_ids.hpp" // statsduck::densify_clusters
#include "glm_core.hpp"    // statsduck::fit_glm / Family / Link / the parsers
#include "lm_core.hpp"     // statsduck::Vcov / parse_vcov / vcov_name

#include "duckdb/common/string_util.hpp"
#include "duckdb_compat.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace duckdb {

namespace compat = statsduck::compat;

namespace {

//===--------------------------------------------------------------------===//
// Result STRUCT type.
//===--------------------------------------------------------------------===//

// One row per coefficient — the element type of the `coefficients` list. The
// statistic is z, not t: glm_fit uses a normal reference for every family,
// gaussian and gamma included. src/include/glm_core.hpp records why.
static LogicalType GlmFitCoefType() {
	child_list_t<LogicalType> f;
	f.emplace_back("term", LogicalType::VARCHAR);
	f.emplace_back("estimate", LogicalType::DOUBLE);
	f.emplace_back("std_error", LogicalType::DOUBLE);
	f.emplace_back("z_statistic", LogicalType::DOUBLE);
	f.emplace_back("p_value", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(f));
}

// Field order MUST match the writes in Finalize (indices below).
static LogicalType GlmFitResultType() {
	child_list_t<LogicalType> c;
	c.emplace_back("coefficients", LogicalType::LIST(GlmFitCoefType())); // 0
	c.emplace_back("n", LogicalType::BIGINT);                            // 1
	c.emplace_back("k", LogicalType::BIGINT);                            // 2
	c.emplace_back("df_residual", LogicalType::BIGINT);                  // 3
	c.emplace_back("deviance", LogicalType::DOUBLE);                     // 4
	c.emplace_back("null_deviance", LogicalType::DOUBLE);                // 5
	c.emplace_back("dispersion", LogicalType::DOUBLE);                   // 6
	c.emplace_back("loglik", LogicalType::DOUBLE);                       // 7
	c.emplace_back("aic", LogicalType::DOUBLE);                          // 8
	c.emplace_back("iterations", LogicalType::BIGINT);                   // 9
	c.emplace_back("converged", LogicalType::BOOLEAN);                   // 10
	c.emplace_back("separation", LogicalType::BOOLEAN);                  // 11
	c.emplace_back("family", LogicalType::VARCHAR);                      // 12
	c.emplace_back("link", LogicalType::VARCHAR);                        // 13
	c.emplace_back("has_intercept", LogicalType::BOOLEAN);               // 14
	c.emplace_back("vcov_type", LogicalType::VARCHAR);                   // 15
	c.emplace_back("n_clusters", LogicalType::BIGINT);                   // 16 (NULL unless CR*)
	c.emplace_back("rank", LogicalType::BIGINT);                         // 17
	c.emplace_back("cov", LogicalType::LIST(LogicalType::DOUBLE));       // 18 (row-major k×k)
	return LogicalType::STRUCT(std::move(c));
}

//===--------------------------------------------------------------------===//
// The options STRUCT.
//
// lm_fit puts its optional inputs in positional overloads. glm_fit cannot: it
// has six of them, and `weights` and `offset` are per-row DOUBLEs that no
// position can tell apart. So they go in one trailing STRUCT.
//
// The fields are read BY NAME, never by position. A DuckDB struct type is
// ordered, so {'link': 'probit', 'vcov': 'HC1'} and the same two fields
// reversed are two different types carrying the same meaning, and both have to
// work. An unrecognized name is a bind error that lists the valid names: a
// silently ignored {'weight': n} would otherwise give an unweighted fit that
// looks correct.
//===--------------------------------------------------------------------===//

// `link`, `vcov` and `intercept` are constants, folded at bind time into the
// bind data. `weights`, `offset` and `cluster` are per-row columns that stay in
// the plan and are read in Update.
enum class OptField { kLink, kVcov, kIntercept, kWeights, kOffset, kCluster };

struct OptionName {
	const char *name;
	OptField field;
};

// Also the order the error message lists, so a caller reads the constants
// first and the per-row columns second.
static constexpr OptionName kOptionNames[] = {
    {"link", OptField::kLink},       {"vcov", OptField::kVcov},
    {"intercept", OptField::kIntercept}, {"weights", OptField::kWeights},
    {"offset", OptField::kOffset},   {"cluster", OptField::kCluster},
};

static string OptionNameList() {
	string out;
	for (auto &o : kOptionNames) {
		if (!out.empty()) {
			out += ", ";
		}
		out += "'";
		out += o.name;
		out += "'";
	}
	return out;
}

//===--------------------------------------------------------------------===//
// Bind data — carries the folded constants and the layout of the per-row
// options through to Update and Finalize.
//===--------------------------------------------------------------------===//

struct GlmFitBindData : public FunctionData {
	statsduck::GlmOptions opts;
	// Child positions of the per-row options inside the runtime options
	// struct, or -1 when the caller did not give that option. The bind
	// rewrites the argument's type to a struct of exactly the per-row fields
	// it saw, in this file's own order, so these indices are this file's to
	// choose rather than the caller's.
	int weights_child = -1;
	int offset_child = -1;
	int cluster_child = -1;
	// Whether an options struct survived the bind as a runtime argument. It
	// does not when the caller gave only constants, which are already folded.
	bool has_options_arg = false;

	bool HasWeights() const {
		return weights_child >= 0;
	}
	bool HasOffset() const {
		return offset_child >= 0;
	}
	bool HasCluster() const {
		return cluster_child >= 0;
	}

	// Copied by assignment rather than field by field as lm_fit's is. There
	// are ten fields to carry here — six in GlmOptions and four of this
	// struct's own — against lm_fit's three, and assignment cannot forget one.
	unique_ptr<FunctionData> Copy() const override {
		auto c = make_uniq<GlmFitBindData>();
		*c = *this;
		return std::move(c);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<GlmFitBindData>();
		return opts.family == o.opts.family && opts.link == o.opts.link && opts.vcov == o.opts.vcov &&
		       opts.intercept == o.opts.intercept && opts.max_iter == o.opts.max_iter && opts.tol == o.opts.tol &&
		       weights_child == o.weights_child && offset_child == o.offset_child &&
		       cluster_child == o.cluster_child && has_options_arg == o.has_options_arg;
	}
};

static Value EvalConst(ClientContext &context, Expression &expr, const char *what) {
	if (!expr.IsFoldable()) {
		throw BinderException("glm_fit: '%s' must be a constant value", what);
	}
	Value v = ExpressionExecutor::EvaluateScalar(context, expr);
	if (v.IsNull()) {
		throw BinderException("glm_fit: '%s' must not be NULL", what);
	}
	return v;
}

static statsduck::Family ParseFamilyArg(ClientContext &context, Expression &expr) {
	const string name = EvalConst(context, expr, "family").GetValue<string>();
	statsduck::Family out;
	if (!statsduck::parse_family(name, out)) {
		throw BinderException("glm_fit: unknown family '%s' — use 'gaussian', 'binomial', "
		                      "'poisson' or 'gamma'",
		                      name);
	}
	return out;
}

// The expression that produced struct field `idx`, when there is one to reach.
//
// A struct literal binds to a struct_pack call whose children are its fields
// in order, so one constant field of an otherwise per-row struct can still be
// folded. Only struct_pack is accepted. Another struct-returning function can
// have the same number of children without those children being the fields:
// struct_concat({'link': 'logit'}, {'weights': w}) has two children, and both
// are structs rather than field values. Reading child 0 as `link` there gave
// "unknown link '{'link': logit}'" before this guard.
//
// A struct from anywhere else — a column of struct type, or any other
// function — has no child expression to reach, and its constant-only fields
// are then rejected as non-constant.
static optional_ptr<Expression> StructFieldExpr(Expression &expr, idx_t idx) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return nullptr;
	}
	auto &fn = expr.Cast<BoundFunctionExpression>();
	if (compat::FunctionName(fn) != "struct_pack") {
		return nullptr;
	}
	if (compat::Children(fn).size() != StructType::GetChildCount(compat::ReturnType(expr)) ||
	    idx >= compat::Children(fn).size()) {
		return nullptr;
	}
	return compat::Children(fn)[idx].get();
}

// CR0/CR1 need a cluster key, which only the options struct can carry.
static void RejectClusterlessCR(statsduck::Vcov vcov, bool has_cluster) {
	if ((vcov == statsduck::Vcov::kCR0 || vcov == statsduck::Vcov::kCR1) && !has_cluster) {
		throw BinderException("glm_fit: CR0/CR1 requires a cluster field — "
		                      "glm_fit(y, x, family, {'vcov': 'CR1', 'cluster': key})");
	}
}

// A cluster key only makes sense with a cluster-robust vcov.
static void RequireClusterVcov(statsduck::Vcov vcov, bool has_cluster) {
	if (has_cluster && vcov != statsduck::Vcov::kCR0 && vcov != statsduck::Vcov::kCR1) {
		throw BinderException("glm_fit: a cluster field was given but vcov is not "
		                      "cluster-robust; use 'CR0' or 'CR1'");
	}
}

// Read the trailing options struct at `idx`. Returns true when the struct
// carries at least one per-row field and must stay as a runtime argument.
static bool ReadOptions(ClientContext &context, compat::BindFunction &function,
                        vector<unique_ptr<Expression>> &arguments, idx_t idx, GlmFitBindData &bd,
                        bool &link_given) {
	auto &opts_expr = *arguments[idx];
	const LogicalType &opts_type = compat::ReturnType(opts_expr);
	if (opts_type.id() != LogicalTypeId::STRUCT && !compat::IsTuple(opts_type)) {
		throw BinderException("glm_fit: the options argument must be a STRUCT, for example "
		                      "{'link': 'probit', 'vcov': 'HC1'}");
	}
	auto &fields = StructType::GetChildTypes(opts_type);
	if (!fields.empty() && StructType::IsUnnamed(opts_type)) {
		throw BinderException("glm_fit: the options STRUCT needs named fields — valid names are %s",
		                      OptionNameList());
	}

	// A struct whose every field is constant folds in one call, and reading
	// the fields off that Value avoids looking inside the expression at all.
	// A struct that also carries a column does not fold, and each constant
	// field is then folded on its own through StructFieldExpr above.
	const bool whole_folds = opts_expr.IsFoldable();
	Value whole;
	if (whole_folds) {
		whole = ExpressionExecutor::EvaluateScalar(context, opts_expr);
	}

	auto const_value = [&](idx_t field_idx, const char *what) -> Value {
		if (whole_folds) {
			if (whole.IsNull()) {
				throw BinderException("glm_fit: '%s' must not be NULL", what);
			}
			Value v = StructValue::GetChildren(whole)[field_idx];
			if (v.IsNull()) {
				throw BinderException("glm_fit: '%s' must not be NULL", what);
			}
			return v;
		}
		auto child = StructFieldExpr(opts_expr, field_idx);
		if (!child) {
			throw BinderException("glm_fit: '%s' must be a constant value — write the options "
			                      "as a struct literal, as in {'link': 'probit'}",
			                      what);
		}
		return EvalConst(context, *child, what);
	};

	constexpr idx_t kOptionCount = sizeof(kOptionNames) / sizeof(kOptionNames[0]);
	bool seen[kOptionCount] = {};
	// The per-row fields in this file's own order, which fixes the child
	// indices the runtime reads.
	child_list_t<LogicalType> per_row;
	bool want_weights = false, want_offset = false, want_cluster = false;

	for (idx_t f = 0; f < fields.size(); f++) {
		// child_list_t pairs are Identifier-keyed on 2.0, string-keyed on 1.x
		const string name = compat::Str(fields[f].first);
		idx_t match = kOptionCount; // sentinel: no match
		for (idx_t o = 0; o < kOptionCount; o++) {
			if (StringUtil::CIEquals(name, kOptionNames[o].name)) {
				match = o;
				break;
			}
		}
		if (match == kOptionCount) {
			throw BinderException("glm_fit: unknown option '%s' — valid options are %s", name,
			                      OptionNameList());
		}
		// Defensive only. struct_pack's own collision set is case-insensitive,
		// so DuckDB rejects {'link': 'logit', 'LINK': 'probit'} with
		// "Duplicate struct entry name" before this bind runs. Nothing known
		// reaches this branch.
		if (seen[match]) {
			throw BinderException("glm_fit: option '%s' was given twice", kOptionNames[match].name);
		}
		seen[match] = true;

		switch (kOptionNames[match].field) {
		case OptField::kLink: {
			const string v = const_value(f, "link").GetValue<string>();
			if (!statsduck::parse_link(v, bd.opts.link)) {
				throw BinderException("glm_fit: unknown link '%s' — use 'identity', 'logit', "
				                      "'probit', 'cloglog', 'log', 'inverse' or 'sqrt'",
				                      v);
			}
			link_given = true;
			break;
		}
		case OptField::kVcov: {
			const string v = const_value(f, "vcov").GetValue<string>();
			if (!statsduck::parse_vcov(v, bd.opts.vcov)) {
				throw BinderException("glm_fit: unknown vcov '%s' — use 'const', 'HC0'–'HC3', "
				                      "or 'CR0'/'CR1' (clustered)",
				                      v);
			}
			break;
		}
		case OptField::kIntercept: {
			// The type is checked before the read. GetValue<bool>() on
			// anything else escapes as DuckDB's own cast error, which named
			// neither this function nor this option: {'intercept': 'false'}
			// gave "Could not convert string 'false' to INT8", INT8 being
			// DuckDB's storage name for bool. Writing 'false' for a boolean is
			// an ordinary mistake, and misuse has to raise a BinderException.
			const Value v = const_value(f, "intercept");
			if (v.type().id() != LogicalTypeId::BOOLEAN) {
				throw BinderException("glm_fit: 'intercept' must be a BOOLEAN constant, "
				                      "true or false, not %s",
				                      v.type().ToString());
			}
			bd.opts.intercept = BooleanValue::Get(v);
			break;
		}
		case OptField::kWeights:
			want_weights = true;
			break;
		case OptField::kOffset:
			want_offset = true;
			break;
		case OptField::kCluster:
			want_cluster = true;
			break;
		}
	}

	if (want_weights) {
		bd.weights_child = static_cast<int>(per_row.size());
		per_row.emplace_back("weights", LogicalType::DOUBLE);
	}
	if (want_offset) {
		bd.offset_child = static_cast<int>(per_row.size());
		per_row.emplace_back("offset", LogicalType::DOUBLE);
	}
	if (want_cluster) {
		bd.cluster_child = static_cast<int>(per_row.size());
		per_row.emplace_back("cluster", LogicalType::VARCHAR);
	}

	if (per_row.empty()) {
		return false;
	}
	// Rewrite the declared argument type from the "any struct" placeholder to
	// exactly the per-row fields, in this file's order and with this file's
	// types. FunctionBinder::CastToFunctionArguments then inserts a struct
	// cast, and DuckDB's struct-to-struct cast matches members BY NAME: it
	// reorders them, converts each field (so an INTEGER weight or a non-text
	// cluster key needs no cast from the caller), and drops the constant
	// fields this bind has already folded. Leaving the placeholder in place
	// would make that an unresolved cast target.
	compat::BindArguments(function)[idx] = LogicalType::STRUCT(std::move(per_row));
	return true;
}

// glm_fit(y, x) — gaussian with its canonical identity link.
STATS_DUCK_AGGREGATE_BIND(GlmFitBind2) {
	auto bd = make_uniq<GlmFitBindData>();
	bd->opts.link = statsduck::canonical_link(bd->opts.family);
	return std::move(bd);
}

// glm_fit(y, x, family)
STATS_DUCK_AGGREGATE_BIND(GlmFitBind3) {
	auto bd = make_uniq<GlmFitBindData>();
	bd->opts.family = ParseFamilyArg(context, *arguments[2]);
	bd->opts.link = statsduck::canonical_link(bd->opts.family);
	compat::EraseArgument(function, arguments, 2);
	return std::move(bd);
}

// glm_fit(y, x, family, options)
STATS_DUCK_AGGREGATE_BIND(GlmFitBind4) {
	auto bd = make_uniq<GlmFitBindData>();
	bd->opts.family = ParseFamilyArg(context, *arguments[2]);
	bool link_given = false;
	bd->has_options_arg = ReadOptions(context, function, arguments, 3, *bd, link_given);
	if (!link_given) {
		bd->opts.link = statsduck::canonical_link(bd->opts.family);
	} else if (!statsduck::link_allowed(bd->opts.family, bd->opts.link)) {
		throw BinderException("glm_fit: the %s link is not allowed for family %s",
		                      statsduck::link_name(bd->opts.link),
		                      statsduck::family_name(bd->opts.family));
	}
	RejectClusterlessCR(bd->opts.vcov, bd->HasCluster());
	RequireClusterVcov(bd->opts.vcov, bd->HasCluster());
	// Erase high → low so the lower index stays valid. The options struct, if
	// it survived, becomes runtime input 2 once the family is gone.
	if (!bd->has_options_arg) {
		compat::EraseArgument(function, arguments, 3);
	}
	compat::EraseArgument(function, arguments, 2);
	return std::move(bd);
}

//===--------------------------------------------------------------------===//
// State — buffers the design rows. docs/fitter_conventions.md is explicit that
// leverage-based and score-based estimators do not reduce to streaming
// moments, so the state holds the raw rows and the maths happens in Finalize.
// Heap-owned, lazy alloc, exactly as lm_fit does it.
//===--------------------------------------------------------------------===//

struct GlmFitAccumulator {
	std::size_t p = 0;   // predictor width (list length)
	bool width_set = false;
	bool ragged = false; // a row's list length disagreed → poison the group
	std::vector<double> y;
	std::vector<double> x;            // n*p, row-major (predictors only)
	std::vector<double> weights;      // n prior weights (empty unless given)
	std::vector<double> offset;       // n offsets (empty unless given)
	std::vector<std::string> cluster; // n cluster keys (CR* only; else empty)
};

struct GlmFitState {
	GlmFitAccumulator *acc;
};

STATS_DUCK_AGGREGATE_INIT(GlmFitInit) {
	reinterpret_cast<GlmFitState *>(state_p)->acc = nullptr;
}

static void GlmFitUpdate(Vector inputs[], AggregateInputData &aggr_input, idx_t, Vector &state_vector,
                         idx_t count) {
	auto &y_in = inputs[0];
	auto &x_in = inputs[1]; // LIST(DOUBLE)

	const GlmFitBindData *bd =
	    aggr_input.bind_data ? &aggr_input.bind_data->Cast<GlmFitBindData>() : nullptr;
	const bool has_options = bd && bd->has_options_arg;

	UnifiedVectorFormat y_fmt, x_fmt;
	y_in.ToUnifiedFormat(count, y_fmt);
	x_in.ToUnifiedFormat(count, x_fmt);
	auto y_vals = UnifiedVectorFormat::GetData<double>(y_fmt);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(x_fmt);

	auto &child = compat::ListChild(x_in);
	const idx_t child_size = ListVector::GetListSize(x_in);
	UnifiedVectorFormat child_fmt;
	child.ToUnifiedFormat(child_size, child_fmt);
	auto child_vals = UnifiedVectorFormat::GetData<double>(child_fmt);

	// The per-row options, read out of the options struct by the child index
	// the bind recorded. The struct's own validity is read too: a NULL struct
	// supplies no value for any field it carries.
	//
	// Each child index is composed through the struct's own selection vector,
	// child_sel[parent_sel[i]], which is the idiom Vector::RecursiveToUnified-
	// Format uses. That is correct for a flat struct, a constant struct, and a
	// dictionary over a flat struct, because StructVector::GetEntries recurses
	// through a dictionary to its child. The one shape it would read out of
	// order is a dictionary whose struct child is itself not flat, so the two
	// selections would need a third composition. No query is known to produce
	// that shape, and lm_fit reads its list child the same way.
	UnifiedVectorFormat opt_fmt, w_fmt, off_fmt, cl_fmt;
	const double *w_vals = nullptr;
	const double *off_vals = nullptr;
	const string_t *cl_vals = nullptr;
	const bool has_weights = has_options && bd->HasWeights();
	const bool has_offset = has_options && bd->HasOffset();
	const bool has_cluster = has_options && bd->HasCluster();
	if (has_options) {
		inputs[2].ToUnifiedFormat(count, opt_fmt);
		auto &opt_children = StructVector::GetEntries(inputs[2]);
		if (has_weights) {
			compat::Deref(opt_children[NumericCast<idx_t>(bd->weights_child)]).ToUnifiedFormat(count, w_fmt);
			w_vals = UnifiedVectorFormat::GetData<double>(w_fmt);
		}
		if (has_offset) {
			compat::Deref(opt_children[NumericCast<idx_t>(bd->offset_child)]).ToUnifiedFormat(count, off_fmt);
			off_vals = UnifiedVectorFormat::GetData<double>(off_fmt);
		}
		if (has_cluster) {
			compat::Deref(opt_children[NumericCast<idx_t>(bd->cluster_child)]).ToUnifiedFormat(count, cl_fmt);
			cl_vals = UnifiedVectorFormat::GetData<string_t>(cl_fmt);
		}
	}

	auto states = FlatVector::GetData<GlmFitState *>(state_vector);

	for (idx_t i = 0; i < count; i++) {
		const auto yidx = y_fmt.sel->get_index(i);
		const auto xidx = x_fmt.sel->get_index(i);
		// Complete-case: skip a NULL response or a NULL design row.
		if (!y_fmt.validity.RowIsValid(yidx) || !x_fmt.validity.RowIsValid(xidx)) {
			continue;
		}
		const double yv = y_vals[yidx];
		if (!std::isfinite(yv)) {
			continue;
		}

		// The per-row options next, before anything is appended, so a bad one
		// costs no roll-back.
		double wv = 1.0;
		double offv = 0.0;
		idx_t clidx = 0;
		if (has_options) {
			const auto oidx = opt_fmt.sel->get_index(i);
			const bool opt_null = !opt_fmt.validity.RowIsValid(oidx);
			if (has_weights) {
				const auto widx = w_fmt.sel->get_index(oidx);
				if (opt_null || !w_fmt.validity.RowIsValid(widx)) {
					continue;
				}
				wv = w_vals[widx];
				// A weight of exactly 0 influences no estimate, so the row is
				// dropped like a NULL and is absent from n. Counting it would
				// inflate df_residual and deflate the dispersion. A NEGATIVE
				// weight is bad data, not missing data: it is kept so the
				// kernel fails the group. A NaN or infinite weight is missing
				// data at this layer, as a NaN or infinite response or offset
				// is; the kernel's own finiteness check serves callers that do
				// not come through here.
				if (!std::isfinite(wv) || wv == 0.0) {
					continue;
				}
			}
			if (has_offset) {
				const auto oftidx = off_fmt.sel->get_index(oidx);
				if (opt_null || !off_fmt.validity.RowIsValid(oftidx)) {
					continue;
				}
				offv = off_vals[oftidx];
				if (!std::isfinite(offv)) {
					continue;
				}
			}
			if (has_cluster) {
				clidx = cl_fmt.sel->get_index(oidx);
				if (opt_null || !cl_fmt.validity.RowIsValid(clidx)) {
					continue;
				}
			}
		}

		auto &st = *states[i];
		if (!st.acc) {
			st.acc = new GlmFitAccumulator();
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
			acc.ragged = true; // inconsistent predictor count → NULL group
			continue;
		}

		// Gather the predictors; roll back the partial append if any element
		// is NULL or non-finite so y and x stay row-aligned.
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
		if (has_weights) {
			acc.weights.push_back(wv);
		}
		if (has_offset) {
			acc.offset.push_back(offv);
		}
		if (has_cluster) {
			acc.cluster.push_back(cl_vals[clidx].GetString());
		}
	}
}

static void GlmFitCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src = FlatVector::GetData<GlmFitState *>(source);
	auto tgt = FlatVector::GetData<GlmFitState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto *sa = src[i]->acc;
		if (!sa) {
			continue;
		}
		if (!tgt[i]->acc) {
			tgt[i]->acc = new GlmFitAccumulator();
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
		ta.weights.insert(ta.weights.end(), sa->weights.begin(), sa->weights.end());
		ta.offset.insert(ta.offset.end(), sa->offset.begin(), sa->offset.end());
		ta.cluster.insert(ta.cluster.end(), sa->cluster.begin(), sa->cluster.end());
	}
}

static void GlmFitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	auto states = FlatVector::GetData<GlmFitState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		delete states[i]->acc;
		states[i]->acc = nullptr;
	}
}

//===--------------------------------------------------------------------===//
// Finalize.
//===--------------------------------------------------------------------===//

// Write a DOUBLE struct-field, mapping NaN → SQL NULL (matches lm_fit).
// ±Inf passes through unchanged, so a gaussian loglik of +inf at zero deviance
// stays representable.
static inline void SetD(Vector &v, idx_t row, double x) {
	if (std::isnan(x)) {
		FlatVector::SetNull(v, row, true);
	} else {
		compat::WriteData<double>(v)[row] = x;
	}
}

static void GlmFitFinalize(Vector &state_vector, compat::FinalizeInput &input_data, Vector &result, idx_t count,
                           idx_t offset) {
	auto states = FlatVector::GetData<GlmFitState *>(state_vector);

	statsduck::GlmOptions opt;
	bool has_weights = false, has_offset = false, has_cluster = false;
	if (input_data.bind_data) {
		auto &bd = input_data.bind_data->Cast<GlmFitBindData>();
		opt = bd.opts;
		has_weights = bd.has_options_arg && bd.HasWeights();
		has_offset = bd.has_options_arg && bd.HasOffset();
		has_cluster = bd.has_options_arg && bd.HasCluster();
	}

	// Pass 1 — fit each group; tally the totals for the two list children.
	std::vector<statsduck::GlmResult> fits(count);
	std::vector<bool> ok(count, false);
	idx_t total_coefs = 0;
	idx_t total_cov = 0;
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
		if ((has_weights && acc->weights.size() != n) || (has_offset && acc->offset.size() != n) ||
		    (has_cluster && acc->cluster.size() != n)) {
			continue; // defensive: option/row desync — emit NULL for this group
		}
		statsduck::linalg::Mat Xp(n, p);
		Xp.data = acc->x; // row-major n*p, exactly Mat's layout
		// densify_clusters sorts rather than hashing, which keeps the libc++
		// std::__hash_memory symbol out of the wasm build. ids has to outlive
		// the fit call, so it is declared here.
		std::vector<int> ids;
		if (has_cluster) {
			ids = statsduck::densify_clusters(acc->cluster);
		}
		statsduck::GlmResult r =
		    statsduck::fit_glm(acc->y, Xp, opt, has_weights ? &acc->weights : nullptr,
		                       has_offset ? &acc->offset : nullptr, has_cluster ? &ids : nullptr);
		if (r.ok) {
			total_coefs += r.k;
			total_cov += r.k * r.k;
			fits[i] = std::move(r);
			ok[i] = true;
		}
	}

	auto &children = StructVector::GetEntries(result);

	// coefficients: LIST<STRUCT>. Grow the list child, then fetch field pointers.
	Vector &coef_list = compat::Deref(children[0]);
	auto coef_entries = compat::WriteData<list_entry_t>(coef_list);
	const idx_t anchor = ListVector::GetListSize(coef_list);
	ListVector::Reserve(coef_list, anchor + total_coefs);
	Vector &coef_child = compat::ListChild(coef_list); // STRUCT vector
	auto &cf = StructVector::GetEntries(coef_child);
	auto term_d = compat::WriteData<string_t>(compat::Deref(cf[0]));

	// cov: a second, independently managed LIST child — one flat row-major k×k
	// block per group. Same rule: Reserve first, then fetch the child pointer.
	Vector &cov_list = compat::Deref(children[18]);
	auto cov_entries = compat::WriteData<list_entry_t>(cov_list);
	const idx_t cov_anchor = ListVector::GetListSize(cov_list);
	ListVector::Reserve(cov_list, cov_anchor + total_cov);
	Vector &cov_child = compat::ListChild(cov_list);
	auto cov_d = compat::WriteData<double>(cov_child);

	auto n_d = compat::WriteData<int64_t>(compat::Deref(children[1]));
	auto k_d = compat::WriteData<int64_t>(compat::Deref(children[2]));
	auto dfr_d = compat::WriteData<int64_t>(compat::Deref(children[3]));
	auto iter_d = compat::WriteData<int64_t>(compat::Deref(children[9]));
	auto conv_d = compat::WriteData<bool>(compat::Deref(children[10]));
	auto sep_d = compat::WriteData<bool>(compat::Deref(children[11]));
	auto hint_d = compat::WriteData<bool>(compat::Deref(children[14]));
	auto ncl_d = compat::WriteData<int64_t>(compat::Deref(children[16]));
	auto rank_d = compat::WriteData<int64_t>(compat::Deref(children[17]));

	// The three echoed names come from the bind rather than from the fit, so
	// they are known for every group, a failed one included.
	const char *family_str = statsduck::family_name(opt.family);
	const char *link_str = statsduck::link_name(opt.link);
	const char *vcov_str = statsduck::vcov_name(opt.vcov);

	idx_t out = anchor;
	idx_t cov_out = cov_anchor;
	for (idx_t i = 0; i < count; i++) {
		const idx_t idx = i + offset;
		// Write the three VARCHAR children for every row, including a row
		// about to be set NULL. FlatVector::SetNull on a STRUCT also nulls
		// its children, so a failed group still reads NULL here — the write
		// exists so the string_t slot is never uninitialized memory.
		compat::WriteData<string_t>(compat::Deref(children[12]))[idx] =
		    StringVector::AddString(compat::Deref(children[12]), family_str);
		compat::WriteData<string_t>(compat::Deref(children[13]))[idx] =
		    StringVector::AddString(compat::Deref(children[13]), link_str);
		compat::WriteData<string_t>(compat::Deref(children[15]))[idx] =
		    StringVector::AddString(compat::Deref(children[15]), vcov_str);

		if (!ok[i]) {
			FlatVector::SetNull(result, idx, true);
			coef_entries[idx] = list_entry_t(out, 0);
			cov_entries[idx] = list_entry_t(cov_out, 0);
			continue;
		}
		auto &r = fits[i];
		coef_entries[idx] = list_entry_t(out, r.k);
		for (std::size_t j = 0; j < r.k; j++) {
			const idx_t pos = out + j;
			term_d[pos] = StringVector::AddString(compat::Deref(cf[0]), r.terms[j]);
			// Every numeric field goes through SetD: an aliased term (a column
			// dropped as dependent) carries NaN, which SetD renders as NULL.
			SetD(compat::Deref(cf[1]), pos, r.beta[j]);
			SetD(compat::Deref(cf[2]), pos, r.std_error[j]);
			SetD(compat::Deref(cf[3]), pos, r.z_statistic[j]);
			SetD(compat::Deref(cf[4]), pos, r.p_value[j]);
		}
		out += r.k;

		const idx_t kk = r.k * r.k;
		cov_entries[idx] = list_entry_t(cov_out, kk);
		for (idx_t j = 0; j < kk; j++) {
			const double v = r.cov[j];
			if (std::isnan(v)) {
				FlatVector::SetNull(cov_child, cov_out + j, true); // an aliased term
			} else {
				cov_d[cov_out + j] = v;
			}
		}
		cov_out += kk;

		n_d[idx] = static_cast<int64_t>(r.n);
		k_d[idx] = static_cast<int64_t>(r.k);
		dfr_d[idx] = static_cast<int64_t>(r.df_residual);
		SetD(compat::Deref(children[4]), idx, r.deviance);
		SetD(compat::Deref(children[5]), idx, r.null_deviance);
		SetD(compat::Deref(children[6]), idx, r.dispersion);
		SetD(compat::Deref(children[7]), idx, r.loglik);
		SetD(compat::Deref(children[8]), idx, r.aic);
		iter_d[idx] = static_cast<int64_t>(r.iterations);
		conv_d[idx] = r.converged;
		sep_d[idx] = r.separation;
		// GlmResult carries no has_intercept field, so this comes from the
		// bind. The kernel prepends a constant column exactly when
		// opts.intercept is set, so the two agree by construction.
		hint_d[idx] = opt.intercept;
		if (r.n_clusters > 0) {
			ncl_d[idx] = static_cast<int64_t>(r.n_clusters);
		} else {
			FlatVector::SetNull(compat::Deref(children[16]), idx, true); // NULL for classical/HC
		}
		rank_d[idx] = static_cast<int64_t>(r.rank);
	}
	ListVector::SetListSize(coef_list, out);
	ListVector::SetListSize(cov_list, cov_out);
}

} // namespace

void RegisterGlmFit(ExtensionLoader &loader) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	const auto ret = GlmFitResultType();
	auto make = [&](vector<LogicalType> args, bind_aggregate_function_t bind) {
		return AggregateFunction("glm_fit", std::move(args), ret, AggregateFunction::StateSize<GlmFitState>,
		                         GlmFitInit, GlmFitUpdate, GlmFitCombine, GlmFitFinalize,
		                         FunctionNullHandling::SPECIAL_HANDLING, nullptr, bind, GlmFitDestroy);
	};

	AggregateFunctionSet set("glm_fit");
	set.AddFunction(make({LogicalType::DOUBLE, list_double}, GlmFitBind2));
	set.AddFunction(make({LogicalType::DOUBLE, list_double, LogicalType::VARCHAR}, GlmFitBind3));
	// The options argument is declared as the unresolved "any struct" type, the
	// way struct_extract declares its own first argument. The bind replaces it
	// with a concrete struct of the per-row fields the caller actually gave, or
	// erases it when every field was a constant.
	set.AddFunction(make({LogicalType::DOUBLE, list_double, LogicalType::VARCHAR, LogicalTypeId::STRUCT},
	                     GlmFitBind4));
	statsduck::RegisterDocumented(
	    loader, std::move(set),
	    "Generalized linear model aggregate — one model per group, fitted by iteratively reweighted "
	    "least squares over the gaussian, binomial, poisson and gamma families. The binomial "
	    "response is a proportion and the `weights` option is the trial count, so 7 of 10 is y = 0.7 "
	    "with weight 10. Options go in one trailing STRUCT read by field name: 'link', 'vcov' "
	    "('const', 'HC0'–'HC3', 'CR0'/'CR1'), 'intercept', 'weights', 'offset' and 'cluster'. "
	    "z_statistic uses a normal reference for every family, including gaussian and gamma where R "
	    "uses a t.",
	    {"y", "x", "family", "options"},
	    "SELECT glm_fit(y, [x1, x2], 'binomial', {'vcov': 'HC1'}) FROM data");
}

} // namespace duckdb
