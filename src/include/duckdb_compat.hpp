//===----------------------------------------------------------------------===//
// duckdb_compat.hpp — one binding layer across the DuckDB 1.x and 2.x lines.
//
// DuckDB v2.0 reorganized the vector helpers and split their accessors into
// const and mutable halves. None of it was removed except Function::
// EraseArgument. This header feature-detects the new layout and exposes
// accessors that compile on both lines, so the *_function.cpp files do not
// each carry a #if.
//
// Feature detection, not version macros: the same approach src/ggsql.cpp
// already uses for the parser extension. DuckDB does not expose a usable
// version macro at preprocessor time, and __has_include on a header that only
// exists in one line is exact.
//
// What changed in v2.0, each verified against the v2.0.0 headers:
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
//   - Function::EraseArgument is gone. In 1.x it erased an argument from the
//     expression list and the signature, saving the pre-erasure signature in
//     SimpleFunction::original_arguments so a serialized plan could restore
//     it. v2.0 dropped that field and writes an empty vector for the
//     corresponding serialization property (function_serialization.hpp:87),
//     so on that line erasing is the two erase_at calls and nothing else.
//
// FlatVector::SetNull and FlatVector::GetData's read form are unchanged, which
// is why 56 and 205 call sites respectively are untouched by this port.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"

#if __has_include("duckdb/common/vector/list_vector.hpp")
// DuckDB v2.0 and later.
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#define STATS_DUCK_DUCKDB_VECTOR_V2 1
#else
// DuckDB 1.x: the helpers live in the types header, which duckdb.hpp pulls in.
#include "duckdb/common/types/vector.hpp"
#endif

namespace statsduck {
namespace compat {

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

//! The child vector of a LIST, for writing.
inline duckdb::Vector &ListChild(duckdb::Vector &vector) {
#ifdef STATS_DUCK_DUCKDB_VECTOR_V2
	return duckdb::ListVector::GetChildMutable(vector);
#else
	return duckdb::ListVector::GetEntry(vector);
#endif
}

//! The child vector of a LIST, for reading.
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

// NOT YET SHIMMED: Function::EraseArgument.
//
// v2.0 removed it, and the obvious replacement — erasing from the expression
// list and from bound_function.arguments — does not compile, because v2.0 also
// moved `arguments` off SimpleFunction onto a new BoundSimpleFunction and gave
// SimpleFunction a FunctionSignature instead.
//
// That is not an isolated rename. v2.0 redesigned the whole bind callback:
//
//   1.x   bind(ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
//   2.0   bind(BindAggregateFunctionInput &)
//
// where the input carries the context, a BoundAggregateFunction, the arguments
// and the argument names behind GetClientContext() / GetBoundFunction() /
// GetArguments(). A signature change cannot be bridged by an accessor, so the
// 53 bind callbacks in src/ need their declarations rewritten before an
// EraseArgument shim has a bound-function type to take. Both land together.

} // namespace compat
} // namespace statsduck
