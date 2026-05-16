#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers table_one(data, variables [, by]) — a table function that
//! produces a Table-1-style descriptives summary in long format. Auto-
//! classifies each variable as numeric or categorical from its catalog
//! type. Output schema:
//!   (variable VARCHAR, level VARCHAR, statistic VARCHAR,
//!    group VARCHAR, display VARCHAR)
//! `level` is NULL for numeric variables; `group` is either "Overall"
//! or the corresponding `by`-column value when `by` is set.
//!
//! v0.4 MVP: no force_categorical / force_numerical overrides, no
//! between-group p-value column, "Overall" is always included. Those
//! all land in v0.5.
void RegisterTableOne(ExtensionLoader &loader);

} // namespace duckdb
