//===----------------------------------------------------------------------===//
// duckdb_compat.hpp — one binding layer across the DuckDB 1.x and 2.x lines.
//
// DuckDB v2.0 reorganized the vector helpers, split their accessors into
// const and mutable halves, redesigned the aggregate bind callback and the
// aggregate Init and Finalize inputs, changed what the parser extension's
// parse callback receives, and keyed the catalog and column APIs by
// Identifier. Very little was removed — Function::EraseArgument and the
// ExecuteWithNulls executors are the exceptions — but a great deal moved.
//
// This header feature-detects the new layout and exposes accessors, type
// aliases and declaration macros that compile on both lines, so the
// *_function.cpp files do not each carry a #if. Feature detection rather
// than version macros: the same approach src/ggsql.cpp already uses for the
// parser extension. DuckDB does not expose a usable version macro at
// preprocessor time, and __has_include on a header that exists on one line
// only is exact.
//
// Tier 1 — the vector layer, each verified against the v2.0.0 headers:
//
//   - ListVector and StructVector moved from duckdb/common/types/vector.hpp
//     to duckdb/common/vector/{list,struct}_vector.hpp. Their members are
//     unchanged, so including the new headers is the whole fix.
//   - FlatVector::GetData<T> returns const T*. Writes go through the new
//     FlatVector::GetDataMutable<T>. Read sites need no change: on the 1.x
//     line GetData<T> returns T*, which converts to const T* silently.
//   - ListVector::GetEntry is deprecated in favour of GetChild (const) and
//     GetChildMutable. Still present, but it warns.
//   - FlatVector::Validity gained a ValidityMutable counterpart.
//   - StructVector::GetEntries hands back vector<Vector> instead of
//     vector<unique_ptr<Vector>>, so `*children[i]` no longer compiles.
//
// FlatVector::SetNull and FlatVector::GetData's read form are unchanged, which
// is why 56 and 205 call sites respectively are untouched by this port.
//
// Tier 2 — the signature and API-shape changes — is documented at each shim.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/common/vector_operations/ternary_executor.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#if __has_include("duckdb/common/vector/list_vector.hpp")
// DuckDB v2.0 and later.
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#define STATS_DUCK_DUCKDB_VECTOR_V2 1
// The same test stands in for "DuckDB 2.0 or later" everywhere below: the
// vector headers moved in the release that changed the bind callbacks, the
// aggregate Init and Finalize inputs, the parse callback and the catalog API.
#define STATS_DUCK_DUCKDB_V2 1
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/common/identifier.hpp"
#include "duckdb/common/optional.hpp"
#include "duckdb/parser/literal.hpp"
#else
// DuckDB 1.x: the helpers live in the types header, which duckdb.hpp pulls in.
#include "duckdb/common/types/vector.hpp"
#endif

namespace statsduck {
namespace compat {

//===--------------------------------------------------------------------===//
// Tier 1: the vector layer.
//===--------------------------------------------------------------------===//

//! Writeable payload pointer for a flat vector. Use this wherever the result
//! is assigned to; plain FlatVector::GetData<T> still serves reads on both
//! lines.
template <class T>
inline T *WriteData(duckdb::Vector &vector) {
#ifdef STATS_DUCK_DUCKDB_VECTOR_V2
	return duckdb::FlatVector::GetDataMutable<T>(vector);
#else
	return duckdb::FlatVector::GetData<T>(vector);
#endif
}

//! The child vector of a LIST, for writing. A mutable reference also serves
//! reads on both lines, so this is the one call sites use.
inline duckdb::Vector &ListChild(duckdb::Vector &vector) {
#ifdef STATS_DUCK_DUCKDB_VECTOR_V2
	return duckdb::ListVector::GetChildMutable(vector);
#else
	return duckdb::ListVector::GetEntry(vector);
#endif
}

//! The child vector of a LIST, for reading, where only a const reference is
//! held.
inline const duckdb::Vector &ListChildConst(const duckdb::Vector &vector) {
#ifdef STATS_DUCK_DUCKDB_VECTOR_V2
	return duckdb::ListVector::GetChild(vector);
#else
	return duckdb::ListVector::GetEntry(vector);
#endif
}

//! The validity mask of a flat vector, for writing.
inline duckdb::ValidityMask &WriteValidity(duckdb::Vector &vector) {
#ifdef STATS_DUCK_DUCKDB_VECTOR_V2
	return duckdb::FlatVector::ValidityMutable(vector);
#else
	return duckdb::FlatVector::Validity(vector);
#endif
}

//! A STRUCT's child vectors. 1.x hands back vector<unique_ptr<Vector>>, 2.0
//! vector<Vector>. Deref takes either; overloads rather than an #if, so each
//! line uses the pair that applies and never instantiates the other.
//! StructChildren names the container itself, for a function that receives
//! the children as a parameter instead of binding them from GetEntries.
#ifdef STATS_DUCK_DUCKDB_VECTOR_V2
using StructChildren = duckdb::vector<duckdb::Vector>;
#else
using StructChildren = duckdb::vector<duckdb::unique_ptr<duckdb::Vector>>;
#endif
inline duckdb::Vector &Deref(duckdb::unique_ptr<duckdb::Vector> &p) {
	return *p;
}
inline const duckdb::Vector &Deref(const duckdb::unique_ptr<duckdb::Vector> &p) {
	return *p;
}
inline duckdb::Vector &Deref(duckdb::Vector &v) {
	return v;
}
inline const duckdb::Vector &Deref(const duckdb::Vector &v) {
	return v;
}

//===--------------------------------------------------------------------===//
// Tier 2: the signature and API-shape changes.
//
// v2.0 redesigned the aggregate bind callback and the aggregate Init and
// Finalize inputs. A signature cannot be bridged by an accessor, so these are
// macros that expand to the right declaration on each line and keep the body
// identical. Where a body needs a value whose type differs by line, a small
// accessor hides the difference.
//===--------------------------------------------------------------------===//

//! Aggregate bind. On 1.x the callback takes (context, function, arguments).
//! On 2.0 it takes one BindAggregateFunctionInput; the macro declares a thin
//! adapter under the real name that unpacks it, and the body that follows the
//! macro becomes NAMEImpl with the three 1.x-style parameters. Bodies refer to
//! `context`, `function` and `arguments` by those names on both lines.
//!
//! `function` is an AggregateFunction& on 1.x and a BoundAggregateFunction&
//! on 2.0. The only members a bind here touches are reached through
//! EraseArgument and BindArguments below, which take either.
#ifdef STATS_DUCK_DUCKDB_V2
#define STATS_DUCK_AGGREGATE_BIND(NAME)                                                                            \
	static duckdb::unique_ptr<duckdb::FunctionData> NAME##Impl(                                                    \
	    duckdb::ClientContext &context, duckdb::BoundAggregateFunction &function,                                   \
	    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);                                        \
	static duckdb::unique_ptr<duckdb::FunctionData> NAME(duckdb::BindAggregateFunctionInput &input) {              \
		return NAME##Impl(input.GetClientContext(), input.GetBoundFunction(), input.GetArguments());                \
	}                                                                                                              \
	static duckdb::unique_ptr<duckdb::FunctionData> NAME##Impl(                                                    \
	    duckdb::ClientContext &context, duckdb::BoundAggregateFunction &function,                                   \
	    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments)
#else
#define STATS_DUCK_AGGREGATE_BIND(NAME)                                                                            \
	static duckdb::unique_ptr<duckdb::FunctionData> NAME(                                                          \
	    duckdb::ClientContext &context, duckdb::AggregateFunction &function,                                        \
	    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments)
#endif

//! Aggregate Init. 1.x initializes one state; 2.0 initializes a batch and
//! passes an AggregateStateInput no body here reads. The macro keeps the
//! one-state body as NAMEOne and loops over the batch on 2.0.
#ifdef STATS_DUCK_DUCKDB_V2
#define STATS_DUCK_AGGREGATE_INIT(NAME)                                                                            \
	static void NAME##One(duckdb::data_ptr_t state_p);                                                             \
	static void NAME(duckdb::AggregateStateInput &, duckdb::data_ptr_t *states, duckdb::idx_t count) {            \
		for (duckdb::idx_t i = 0; i < count; i++) {                                                                \
			NAME##One(states[i]);                                                                                  \
		}                                                                                                          \
	}                                                                                                              \
	static void NAME##One(duckdb::data_ptr_t state_p)
#else
#define STATS_DUCK_AGGREGATE_INIT(NAME) static void NAME(const duckdb::AggregateFunction &, duckdb::data_ptr_t state_p)
#endif

//! Aggregate Finalize's second parameter. 2.0's AggregateFinalizeInputData
//! derives from AggregateInputData, and bind_data — the only member a Finalize
//! here reads — lives on the base, so the bodies are untouched.
#ifdef STATS_DUCK_DUCKDB_V2
using FinalizeInput = duckdb::AggregateFinalizeInputData;
#else
using FinalizeInput = duckdb::AggregateInputData;
#endif

//! The type of a bind's `function` parameter, for a helper that takes it.
#ifdef STATS_DUCK_DUCKDB_V2
using BindFunction = duckdb::BoundAggregateFunction;
#else
using BindFunction = duckdb::AggregateFunction;
#endif

//! Drop a bind-time argument from the expression list and the signature.
//! 1.x had Function::EraseArgument. 2.0 removed it along with the
//! original_arguments bookkeeping it maintained — the serializer now writes an
//! empty vector in that slot — so there the two erase_at calls are the whole
//! operation. Callers erase from the highest index down.
#ifdef STATS_DUCK_DUCKDB_V2
inline void EraseArgument(duckdb::BoundSimpleFunction &bound_function,
                          duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments,
                          duckdb::idx_t argument_index) {
	D_ASSERT(arguments.size() == bound_function.GetArguments().size());
	D_ASSERT(argument_index < arguments.size());
	arguments.erase_at(argument_index);
	bound_function.GetArguments().erase_at(argument_index);
}
inline duckdb::vector<duckdb::LogicalType> &BindArguments(duckdb::BoundSimpleFunction &bound_function) {
	return bound_function.GetArguments();
}
#else
inline void EraseArgument(duckdb::SimpleFunction &bound_function,
                          duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments,
                          duckdb::idx_t argument_index) {
	duckdb::Function::EraseArgument(bound_function, arguments, argument_index);
}
inline duckdb::vector<duckdb::LogicalType> &BindArguments(duckdb::SimpleFunction &bound_function) {
	return bound_function.arguments;
}
#endif

//! Names and identifiers. 2.0 keys catalog lookups, column lookups, bind
//! result names and named-parameter maps by Identifier, which is explicit from
//! a string, so every string that reaches one of those goes through Ident.
//! Both lines return by value: a 1.x reference to a temporary built from a
//! const char* would dangle.
#ifdef STATS_DUCK_DUCKDB_V2
using BindNames = duckdb::vector<duckdb::Identifier>;
inline duckdb::Identifier Ident(const duckdb::string &s) {
	return duckdb::Identifier(s);
}
inline duckdb::Identifier Ident(const char *s) {
	return duckdb::Identifier(s);
}
inline duckdb::string Str(const duckdb::Identifier &i) {
	return i.GetIdentifierName();
}
#else
using BindNames = duckdb::vector<duckdb::string>;
inline duckdb::string Ident(const duckdb::string &s) {
	return s;
}
inline duckdb::string Ident(const char *s) {
	return duckdb::string(s);
}
inline duckdb::string Str(const duckdb::string &s) {
	return s;
}
#endif
//! The key a table function's named-parameter map is searched by.
inline auto NamedKey(const duckdb::string &s) -> decltype(Ident(s)) {
	return Ident(s);
}

//! Register a keyword argument on a table function. 1.x filled a name-to-type
//! map; 2.0 adds a keyword-only parameter to the signature.
inline void AddNamedParameter(duckdb::TableFunction &fn, const char *name, duckdb::LogicalType type) {
#ifdef STATS_DUCK_DUCKDB_V2
	// Without a default a keyword-only parameter is required on 2.0, and a
	// 1.x named parameter never was. The default is a NULL of the parameter's
	// type; the binder places it into the bind's map for an omitted
	// parameter, so a bind reads one as a NULL value, not a missing key.
	duckdb::Value null_default(type);
	fn.GetSignature().AddKeywordOnly(name, std::move(type), std::move(null_default));
#else
	fn.named_parameters[name] = std::move(type);
#endif
}

//! Append a positional argument to a table function built up by hand.
inline void AddArgument(duckdb::TableFunction &fn, duckdb::LogicalType type) {
#ifdef STATS_DUCK_DUCKDB_V2
	fn.GetSignature().AddParameter(std::move(type));
#else
	fn.arguments.push_back(std::move(type));
#endif
}

//! The scalar executors with a null-producing lambda. 1.x had ExecuteWithNulls,
//! whose lambda took (args..., ValidityMask &, idx_t) and cleared a bit to
//! produce NULL. 2.0 dropped it for a protocol where the lambda returns
//! optional<RESULT>, empty meaning NULL. The call sites keep the 1.x lambda
//! shape; on 2.0 each is run against a one-bit mask on the stack and the bit
//! decides the optional. ValidityMask(validity_t *, idx_t) wraps the buffer
//! without allocating, so the per-row cost is two stores and a test.
//!
//! Ternary on 2.0 takes no count — the vectors carry their size — so that
//! parameter is accepted and ignored there.
template <class IN, class RES, class FUNC>
inline void UnaryWithNulls(duckdb::Vector &input, duckdb::Vector &result, duckdb::idx_t count, FUNC fun) {
#ifdef STATS_DUCK_DUCKDB_V2
	duckdb::UnaryExecutor::Execute<IN, RES>(input, result, count, [&](IN a) -> std::optional<RES> {
		duckdb::validity_t bits = ~duckdb::validity_t(0);
		duckdb::ValidityMask scratch(&bits, 1);
		RES v = fun(a, scratch, 0);
		if (!scratch.RowIsValid(0)) {
			return std::nullopt;
		}
		return v;
	});
#else
	duckdb::UnaryExecutor::ExecuteWithNulls<IN, RES>(input, result, count, fun);
#endif
}
template <class L, class R, class RES, class FUNC>
inline void BinaryWithNulls(duckdb::Vector &left, duckdb::Vector &right, duckdb::Vector &result, duckdb::idx_t count,
                            FUNC fun) {
#ifdef STATS_DUCK_DUCKDB_V2
	duckdb::BinaryExecutor::Execute<L, R, RES>(left, right, result, count, [&](L a, R b) -> std::optional<RES> {
		duckdb::validity_t bits = ~duckdb::validity_t(0);
		duckdb::ValidityMask scratch(&bits, 1);
		RES v = fun(a, b, scratch, 0);
		if (!scratch.RowIsValid(0)) {
			return std::nullopt;
		}
		return v;
	});
#else
	duckdb::BinaryExecutor::ExecuteWithNulls<L, R, RES>(left, right, result, count, fun);
#endif
}
template <class A, class B, class C, class RES, class FUNC>
inline void TernaryWithNulls(duckdb::Vector &a, duckdb::Vector &b, duckdb::Vector &c, duckdb::Vector &result,
                             duckdb::idx_t count, FUNC fun) {
#ifdef STATS_DUCK_DUCKDB_V2
	(void)count;
	duckdb::TernaryExecutor::Execute<A, B, C, RES>(a, b, c, result, [&](A x, B y, C z) -> std::optional<RES> {
		duckdb::validity_t bits = ~duckdb::validity_t(0);
		duckdb::ValidityMask scratch(&bits, 1);
		RES v = fun(x, y, z, scratch, 0);
		if (!scratch.RowIsValid(0)) {
			return std::nullopt;
		}
		return v;
	});
#else
	duckdb::TernaryExecutor::ExecuteWithNulls<A, B, C, RES>(a, b, c, result, count, fun);
#endif
}

//! dtime_t's payload. 1.x named it micros; 2.0's dtime_base_t names it value.
inline int64_t TimeMicros(duckdb::dtime_t t) {
#ifdef STATS_DUCK_DUCKDB_V2
	return t.value;
#else
	return t.micros;
#endif
}

//! The function_info a scalar function carries. Public on 1.x, behind
//! accessors on 2.0, where the bound expression's function is private too.
inline void SetExtraFunctionInfo(duckdb::ScalarFunction &fn, duckdb::shared_ptr<duckdb::ScalarFunctionInfo> info) {
#ifdef STATS_DUCK_DUCKDB_V2
	fn.SetExtraFunctionInfo(std::move(info));
#else
	fn.function_info = std::move(info);
#endif
}
inline duckdb::ScalarFunctionInfo &ExtraFunctionInfo(const duckdb::BoundFunctionExpression &expr) {
#ifdef STATS_DUCK_DUCKDB_V2
	return expr.Function().GetExtraFunctionInfo();
#else
	return *expr.function.function_info;
#endif
}
//! The info of the first overload in a set, or null when it carries none.
inline duckdb::ScalarFunctionInfo *FirstExtraFunctionInfo(duckdb::ScalarFunctionSet &set) {
#ifdef STATS_DUCK_DUCKDB_V2
	auto &fn = set.GetFunctionByOffset(0);
	return fn->HasExtraFunctionInfo() ? &fn->GetExtraFunctionInfo() : nullptr;
#else
	return set.GetFunctionReferenceByOffset(0).function_info.get();
#endif
}
//! The parts of a bound expression that 2.0 put behind accessors: the return
//! type (protected on Expression), the children (private on
//! BoundFunctionExpression) and the function's name.
inline const duckdb::LogicalType &ReturnType(const duckdb::Expression &expr) {
#ifdef STATS_DUCK_DUCKDB_V2
	return expr.GetReturnType();
#else
	return expr.return_type;
#endif
}
inline const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &
Children(const duckdb::BoundFunctionExpression &expr) {
#ifdef STATS_DUCK_DUCKDB_V2
	return expr.GetChildren();
#else
	return expr.children;
#endif
}
//! Comparable to a string literal on both lines: a string on 1.x, an
//! Identifier on 2.0, which has operator== against const char *.
#ifdef STATS_DUCK_DUCKDB_V2
inline const duckdb::Identifier &FunctionName(const duckdb::BoundFunctionExpression &expr) {
	return expr.Function().GetName();
}
#else
inline const duckdb::string &FunctionName(const duckdb::BoundFunctionExpression &expr) {
	return expr.function.name;
}
#endif

//! A string constant in a parsed expression tree. 1.x built a ConstantExpression
//! from a Value; 2.0 deleted that constructor and takes a Literal.
inline duckdb::unique_ptr<duckdb::ConstantExpression> ConstantString(const duckdb::string &s) {
#ifdef STATS_DUCK_DUCKDB_V2
	return duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Literal::String(s));
#else
	return duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(s));
#endif
}

//! Mark a scalar function as one that can throw while executing. 2.0
//! enforces the flag: an execution-time error from a function without it
//! is rethrown as an internal error. 1.x keeps the same flag as a public
//! field with the same default, so the mark states the same fact there.
inline void SetFallible(duckdb::ScalarFunction &fn) {
#ifdef STATS_DUCK_DUCKDB_V2
	fn.SetFallible();
#else
	fn.errors = duckdb::FunctionErrors::CAN_THROW_RUNTIME_ERROR;
#endif
}

inline void SetFallible(duckdb::ScalarFunctionSet &set) {
#ifdef STATS_DUCK_DUCKDB_V2
	set.SetFallible();
#else
	for (auto &fn : set.functions) {
		fn.errors = duckdb::FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	}
#endif
}

//! Whether a type is 2.0's TUPLE, the unnamed struct that row(...) builds
//! there. 1.x has no such id: row(...) builds a STRUCT with empty names.
inline bool IsTuple(const duckdb::LogicalType &type) {
#ifdef STATS_DUCK_DUCKDB_V2
	return type.id() == duckdb::LogicalTypeId::TUPLE;
#else
	(void)type;
	return false;
#endif
}

//! Mark a scalar function volatile. A public field on 1.x, a setter on 2.0.
inline void SetVolatile(duckdb::ScalarFunction &fn) {
#ifdef STATS_DUCK_DUCKDB_V2
	fn.SetStability(duckdb::FunctionStability::VOLATILE);
#else
	fn.stability = duckdb::FunctionStability::VOLATILE;
#endif
}

//! The function behind a bound function expression; both lines expose `name`.
#ifdef STATS_DUCK_DUCKDB_V2
inline const duckdb::BoundScalarFunction &BoundFunction(const duckdb::BoundFunctionExpression &expr) {
	return expr.Function();
}
#else
inline const duckdb::ScalarFunction &BoundFunction(const duckdb::BoundFunctionExpression &expr) {
	return expr.function;
}
#endif

} // namespace compat
} // namespace statsduck
