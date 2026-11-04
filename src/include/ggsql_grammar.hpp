#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"
#include "ggsql_marks.hpp"

namespace duckdb {
namespace ggsql {

struct DrawLayer {
	string mark;
	// Optional STAT modifier. Empty or "identity" means no transform. "smooth"
	// injects a Vega-Lite loess transform on (x, y). "summary" rewrites the
	// layer's data SQL to aggregate by x (and color / facet if mapped) with
	// AVG(y). Unknown names are rejected at parse time.
	string stat;
};

struct ScaleSpec {
	string aesthetic;             // channel to scale (x, y, color, ...)
	string sub_object = "scale";  // channel sub-object: "scale" (TO/ZERO/DOMAIN) or "axis" (LABEL)
	string property;              // sub-object property: "scheme", "domain", "zero", "title", ...
	string value_json;            // value formatted as JSON ('"viridis"', '[0,100]', 'false', '"My label"', ...)
};

struct TypeOverride {
	string aesthetic; // channel name (color, x, ...)
	string type;      // quantitative | ordinal | nominal | temporal
};

struct VisualizeStatement {
	vector<AestheticMapping> aesthetics;
	string from_table;
	vector<DrawLayer> layers;
	vector<ScaleSpec> scales;
	vector<TypeOverride> type_overrides;
	string facet_layout; // "" (grid), "rows", or "cols" — only meaningful when a facet aesthetic is present
	string title;        // "" if not set; raw text (caller JSON-escapes for output)
	string subtitle;     // "" if not set; SUBTITLE alone (no TITLE) is rejected at parse time
	// Optional leading `WITH [RECURSIVE] <cte> AS (...) [, ...]` block. Captured
	// verbatim as a single SQL string and prepended to each layer's projected
	// SQL so the FROM clause and aesthetic expressions can reference CTEs.
	// Empty string when no WITH clause was provided.
	string with_clause;
};

// Strip surrounding single quotes from a SQL string literal and collapse '' → '.
// Tokens without surrounding single quotes are returned unchanged (so bare
// identifiers like `viridis` remain usable for operators that accept either).
string UnquoteSqlString(const string &raw);

// Escape a raw string for embedding inside a JSON string literal.
string JsonEscape(const string &s);

struct ParseResult {
	bool success = false;
	VisualizeStatement stmt;
	string error;
};

ParseResult ParseGgsql(const string &query);

// The statement keyword, in both spellings that upstream ggsql accepts (#84).
// Defined once here; the parser extension and the DuckDB 2.0 token adapter in
// ggsql.cpp use it as well.
bool IsVisualizeKeyword(const string &token);

// Whether a statement is ours: the keyword in either spelling, after an
// optional leading WITH block. Decided on the grammar's own tokens, so the
// claim and ParseGgsql agree about comments, string literals and parentheses
// (#86). A statement that is not ours stays with DuckDB's parser and keeps
// DuckDB's own error message.
bool IsVisualizeStatement(const string &query);

} // namespace ggsql
} // namespace duckdb
