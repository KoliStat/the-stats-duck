#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers meta(table_name) — a table function returning one row per column
//! with metadata + light per-column stats. Output schema:
//!   (column_name VARCHAR, column_type VARCHAR, kind VARCHAR,
//!    n_rows BIGINT, n_missing BIGINT, n_distinct BIGINT,
//!    min DOUBLE, p25 DOUBLE, median DOUBLE, p75 DOUBLE, max DOUBLE,
//!    mean DOUBLE, stddev DOUBLE,
//!    top VARCHAR, top_freq BIGINT)
//!
//! `kind` is one of 'numeric' / 'categorical' / 'temporal' / 'boolean' / 'other'.
//! Numeric stats (min, p25, median, p75, max, mean, stddev) are NULL for
//! non-numeric columns; `top` / `top_freq` (the mode and its count) are NULL
//! for non-categorical / non-boolean columns. `n_rows` is the total row count
//! in the source table and is repeated on every row so any single row is
//! self-contained.
//!
//! Dataset-level summaries fall out via aggregation, e.g.
//!   SELECT count(*) FILTER (WHERE kind='numeric') FROM meta('penguins');
void RegisterMeta(ExtensionLoader &loader);

} // namespace duckdb
