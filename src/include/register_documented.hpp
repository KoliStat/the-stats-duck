#pragma once

//===----------------------------------------------------------------------===//
// RegisterDocumented — register a function together with the
// FunctionDescription metadata that duckdb_functions() (and therefore the
// community-extensions docs page's "Added Functions" table) renders.
//
// One FunctionDescription with EMPTY parameter_types applies to every
// overload of the function (see duckdb_functions.cpp's
// GetFunctionDescriptionIndex), so each function needs exactly one call here
// regardless of its overload matrix. parameter_names should follow the
// longest overload. test/sql/function_docs.test enforces coverage.
//===----------------------------------------------------------------------===//

#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

namespace statsduck {

namespace docs_detail {

inline duckdb::FunctionDescription MakeDescription(duckdb::string description,
                                                   duckdb::vector<duckdb::string> parameter_names,
                                                   const char *example) {
	duckdb::FunctionDescription d;
	d.description = std::move(description);
	d.parameter_names = std::move(parameter_names);
	if (example && example[0]) {
		d.examples.push_back(example);
	}
	return d;
}

template <class INFO, class FN>
inline void RegisterWith(duckdb::ExtensionLoader &loader, FN fn, duckdb::string description,
                         duckdb::vector<duckdb::string> parameter_names, const char *example) {
	INFO info(std::move(fn));
	info.descriptions.push_back(MakeDescription(description, std::move(parameter_names), example));
	loader.RegisterFunction(std::move(info));
}

} // namespace docs_detail

// example may be nullptr/"" for internal support functions (they still get a
// description so the docs table never shows a bare NULL row).

inline void RegisterDocumented(duckdb::ExtensionLoader &loader, duckdb::ScalarFunction fn, duckdb::string description,
                               duckdb::vector<duckdb::string> parameter_names, const char *example) {
	docs_detail::RegisterWith<duckdb::CreateScalarFunctionInfo>(loader, std::move(fn), description,
	                                                            std::move(parameter_names), example);
}

inline void RegisterDocumented(duckdb::ExtensionLoader &loader, duckdb::ScalarFunctionSet fn, duckdb::string description,
                               duckdb::vector<duckdb::string> parameter_names, const char *example) {
	docs_detail::RegisterWith<duckdb::CreateScalarFunctionInfo>(loader, std::move(fn), description,
	                                                            std::move(parameter_names), example);
}

inline void RegisterDocumented(duckdb::ExtensionLoader &loader, duckdb::AggregateFunction fn, duckdb::string description,
                               duckdb::vector<duckdb::string> parameter_names, const char *example) {
	docs_detail::RegisterWith<duckdb::CreateAggregateFunctionInfo>(loader, std::move(fn), description,
	                                                               std::move(parameter_names), example);
}

inline void RegisterDocumented(duckdb::ExtensionLoader &loader, duckdb::AggregateFunctionSet fn,
                               duckdb::string description, duckdb::vector<duckdb::string> parameter_names,
                               const char *example) {
	docs_detail::RegisterWith<duckdb::CreateAggregateFunctionInfo>(loader, std::move(fn), description,
	                                                               std::move(parameter_names), example);
}

inline void RegisterDocumented(duckdb::ExtensionLoader &loader, duckdb::TableFunction fn, duckdb::string description,
                               duckdb::vector<duckdb::string> parameter_names, const char *example) {
	docs_detail::RegisterWith<duckdb::CreateTableFunctionInfo>(loader, std::move(fn), description,
	                                                           std::move(parameter_names), example);
}

inline void RegisterDocumented(duckdb::ExtensionLoader &loader, duckdb::TableFunctionSet fn, duckdb::string description,
                               duckdb::vector<duckdb::string> parameter_names, const char *example) {
	docs_detail::RegisterWith<duckdb::CreateTableFunctionInfo>(loader, std::move(fn), description,
	                                                           std::move(parameter_names), example);
}

} // namespace statsduck
