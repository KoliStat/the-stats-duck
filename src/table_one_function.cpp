#include "table_one_function.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace duckdb {

// table_one — Table-1 style descriptives summary, MVP shape (v0.4).
//
// At bind time we look up the source table in the catalog, classify each
// declared variable as numeric or categorical from its column type, and
// stash a (mostly schema-only) BindData. At init-global time we open an
// internal Connection and run one summary query per (variable, group)
// pair, formatting results into pre-built output rows. The execute step
// just streams those rows out STANDARD_VECTOR_SIZE at a time.
//
// Long-format output makes the schema static (DuckDB table functions
// don't love dynamic columns) and lets clients pivot to wide for display.

namespace {

// ── BindData / state ───────────────────────────────────────────────────────

struct TableOneRow {
	std::string variable;
	std::string level; // empty → emitted as NULL for numeric rows
	std::string statistic;
	std::string stratum; // "Overall" or a `by`-column value
	std::string display;
};

struct TableOneBindData : public TableFunctionData {
	// Parsed inputs
	std::string data_table;            // unqualified table name (catalog resolution happens here)
	std::vector<std::string> variables;
	std::string by_column;             // empty → no grouping; "Overall" only
	// Classified per variable (filled at bind time)
	std::vector<bool> is_numeric;
};

struct TableOneGlobalState : public GlobalTableFunctionState {
	std::vector<TableOneRow> rows;
	idx_t cursor = 0;
};

// ── Helpers ────────────────────────────────────────────────────────────────

//! Identifier quoting: wrap an identifier in double quotes and double up any
//! existing double quotes inside. Used for both table and column names so
//! catalog lookups with special characters (or names that shadow keywords)
//! survive the round-trip through generated SQL.
static std::string QuoteIdent(const std::string &raw) {
	std::string out;
	out.reserve(raw.size() + 2);
	out.push_back('"');
	for (char c : raw) {
		if (c == '"') {
			out.push_back('"');
		}
		out.push_back(c);
	}
	out.push_back('"');
	return out;
}

//! Single-quote a value to embed inside a SQL literal. Doubles up any
//! single quotes already present.
static std::string QuoteString(const std::string &raw) {
	std::string out;
	out.reserve(raw.size() + 2);
	out.push_back('\'');
	for (char c : raw) {
		if (c == '\'') {
			out.push_back('\'');
		}
		out.push_back(c);
	}
	out.push_back('\'');
	return out;
}

//! Classify a DuckDB LogicalType as numeric for table_one's purposes.
//! INTEGER/FLOAT family is numeric; everything else (VARCHAR, BOOLEAN,
//! ENUM, BLOB, dates, etc.) gets the categorical treatment.
static bool IsNumericKind(LogicalTypeId tid) {
	switch (tid) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return true;
	default:
		return false;
	}
}

static std::string FormatInt(int64_t v) {
	return std::to_string(v);
}

static std::string FormatDouble1(double v) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.1f", v);
	return buf;
}

static std::string FormatPercent(double n, double total) {
	if (total <= 0.0) {
		return "0 (0.0%)";
	}
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.0f (%.1f%%)", n, 100.0 * n / total);
	return buf;
}

// ── Bind ───────────────────────────────────────────────────────────────────

static unique_ptr<FunctionData> TableOneBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types,
                                             vector<string> &names) {
	auto bd = make_uniq<TableOneBindData>();

	// Positional arg 0: data table name (VARCHAR).
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("table_one: 'data' (source table name) is required");
	}
	bd->data_table = input.inputs[0].GetValue<string>();

	// Named param: variables LIST<VARCHAR> (required).
	auto it_vars = input.named_parameters.find("variables");
	if (it_vars == input.named_parameters.end() || it_vars->second.IsNull()) {
		throw BinderException("table_one: 'variables' list is required");
	}
	auto &vars_val = it_vars->second;
	auto &vars_children = ListValue::GetChildren(vars_val);
	for (auto &v : vars_children) {
		bd->variables.push_back(v.GetValue<string>());
	}
	if (bd->variables.empty()) {
		throw BinderException("table_one: 'variables' must not be empty");
	}

	// Named param: by VARCHAR (optional).
	auto it_by = input.named_parameters.find("by");
	if (it_by != input.named_parameters.end() && !it_by->second.IsNull()) {
		bd->by_column = it_by->second.GetValue<string>();
	}

	// Catalog lookup — get column types so we can classify each variable.
	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);
	auto &schema = DEFAULT_SCHEMA;
	auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, bd->data_table,
	                              OnEntryNotFound::THROW_EXCEPTION);
	auto &tbl = entry->Cast<TableCatalogEntry>();
	auto &cols = tbl.GetColumns();

	for (auto &v : bd->variables) {
		if (!cols.ColumnExists(v)) {
			throw BinderException("table_one: column '%s' not found in table '%s'",
			                      v, bd->data_table);
		}
		auto &col = cols.GetColumn(v);
		bd->is_numeric.push_back(IsNumericKind(col.GetType().id()));
	}
	if (!bd->by_column.empty() && !cols.ColumnExists(bd->by_column)) {
		throw BinderException("table_one: 'by' column '%s' not found in table '%s'",
		                      bd->by_column, bd->data_table);
	}

	// Output schema (static — see header comment).
	names = {"variable", "level", "statistic", "stratum", "display"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::VARCHAR};
	return std::move(bd);
}

// ── Per-variable row generation (runs via an internal Connection) ──────────

//! For a numeric variable, fetch summary stats per group and append the
//! formatted rows. If `group_value` is empty, the group label is "Overall"
//! and the WHERE clause is omitted; otherwise we restrict to rows where
//! the by-column equals `group_value`.
static void EmitNumericRows(Connection &conn, const std::string &table,
                            const std::string &var, const std::string &by_col,
                            const std::string &group_value,
                            const std::string &stratum_label,
                            std::vector<TableOneRow> &out) {
	std::stringstream sql;
	sql << "SELECT (s).n, (s).n_missing, (s).mean, (s).sd, "
	       "(s).median, (s).q1, (s).q3, (s).min, (s).max "
	    << "FROM (SELECT summary_stats(" << QuoteIdent(var) << "::DOUBLE) AS s "
	    << "FROM " << QuoteIdent(table);
	if (!group_value.empty()) {
		sql << " WHERE " << QuoteIdent(by_col) << " = " << QuoteString(group_value);
	}
	sql << ")";

	auto result = conn.Query(sql.str());
	if (result->HasError()) {
		throw InvalidInputException("table_one: failed to summarise '%s' (%s)", var,
		                            result->GetError());
	}
	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return; // empty group — skip silently
	}

	auto n = chunk->GetValue(0, 0).GetValue<int64_t>();
	auto n_missing = chunk->GetValue(1, 0).GetValue<int64_t>();
	bool stats_null = chunk->GetValue(2, 0).IsNull();

	out.push_back({var, "", "n", stratum_label, FormatInt(n)});
	out.push_back({var, "", "missing", stratum_label, FormatInt(n_missing)});
	if (stats_null) {
		// Not enough non-NULL data for moments / quantiles.
		out.push_back({var, "", "mean (sd)", stratum_label, "."});
		out.push_back({var, "", "median [q1, q3]", stratum_label, "."});
		out.push_back({var, "", "min, max", stratum_label, "."});
		return;
	}

	double mean = chunk->GetValue(2, 0).GetValue<double>();
	double sd = chunk->GetValue(3, 0).GetValue<double>();
	double median = chunk->GetValue(4, 0).GetValue<double>();
	double q1 = chunk->GetValue(5, 0).GetValue<double>();
	double q3 = chunk->GetValue(6, 0).GetValue<double>();
	double vmin = chunk->GetValue(7, 0).GetValue<double>();
	double vmax = chunk->GetValue(8, 0).GetValue<double>();

	out.push_back({var, "", "mean (sd)", stratum_label,
	               FormatDouble1(mean) + " (" + FormatDouble1(sd) + ")"});
	out.push_back({var, "", "median [q1, q3]", stratum_label,
	               FormatDouble1(median) + " [" + FormatDouble1(q1) + ", " +
	                   FormatDouble1(q3) + "]"});
	out.push_back({var, "", "min, max", stratum_label,
	               FormatDouble1(vmin) + ", " + FormatDouble1(vmax)});
}

//! For a categorical variable, emit one row per level (sorted ascending) with
//! `n (%)` formatted display, plus a trailing "missing" row.
static void EmitCategoricalRows(Connection &conn, const std::string &table,
                                const std::string &var, const std::string &by_col,
                                const std::string &group_value,
                                const std::string &stratum_label,
                                std::vector<TableOneRow> &out) {
	std::stringstream sql;
	sql << "SELECT " << QuoteIdent(var) << "::VARCHAR AS lvl, COUNT(*) AS n "
	    << "FROM " << QuoteIdent(table) << " WHERE " << QuoteIdent(var) << " IS NOT NULL";
	if (!group_value.empty()) {
		sql << " AND " << QuoteIdent(by_col) << " = " << QuoteString(group_value);
	}
	sql << " GROUP BY 1 ORDER BY 1";

	// Trailing missing count (separate query — cleaner than COALESCE wrangling).
	std::stringstream missing_sql;
	missing_sql << "SELECT COUNT(*) FROM " << QuoteIdent(table) << " WHERE "
	            << QuoteIdent(var) << " IS NULL";
	if (!group_value.empty()) {
		missing_sql << " AND " << QuoteIdent(by_col) << " = " << QuoteString(group_value);
	}

	auto result = conn.Query(sql.str());
	if (result->HasError()) {
		throw InvalidInputException("table_one: failed to tabulate '%s' (%s)", var,
		                            result->GetError());
	}

	// First pass: collect all rows, summing for the denominator.
	struct Cell {
		std::string level;
		int64_t n;
	};
	std::vector<Cell> cells;
	int64_t total = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t i = 0; i < chunk->size(); i++) {
			std::string lvl = chunk->GetValue(0, i).ToString();
			int64_t n = chunk->GetValue(1, i).GetValue<int64_t>();
			cells.push_back({lvl, n});
			total += n;
		}
	}

	for (auto &c : cells) {
		out.push_back({var, c.level, "n (%)", stratum_label,
		               FormatPercent(static_cast<double>(c.n), static_cast<double>(total))});
	}

	auto miss_result = conn.Query(missing_sql.str());
	if (!miss_result->HasError()) {
		auto miss_chunk = miss_result->Fetch();
		if (miss_chunk && miss_chunk->size() > 0) {
			int64_t miss = miss_chunk->GetValue(0, 0).GetValue<int64_t>();
			if (miss > 0) {
				out.push_back({var, "", "missing", stratum_label, FormatInt(miss)});
			}
		}
	}
}

// ── InitGlobal: pre-compute every row ──────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> TableOneInitGlobal(ClientContext &context,
                                                                TableFunctionInitInput &input) {
	auto &bd = input.bind_data->Cast<TableOneBindData>();
	auto state = make_uniq<TableOneGlobalState>();

	Connection conn(*context.db);

	// If `by` is set, discover its distinct values up front so we can iterate
	// groups in a stable order. NULLs in `by` are excluded from the breakdown.
	std::vector<std::string> by_values;
	if (!bd.by_column.empty()) {
		std::stringstream sql;
		sql << "SELECT DISTINCT " << QuoteIdent(bd.by_column) << "::VARCHAR AS v "
		    << "FROM " << QuoteIdent(bd.data_table) << " WHERE "
		    << QuoteIdent(bd.by_column) << " IS NOT NULL ORDER BY 1";
		auto result = conn.Query(sql.str());
		if (result->HasError()) {
			throw InvalidInputException("table_one: failed to enumerate `by` values (%s)",
			                            result->GetError());
		}
		while (auto chunk = result->Fetch()) {
			for (idx_t i = 0; i < chunk->size(); i++) {
				by_values.push_back(chunk->GetValue(0, i).ToString());
			}
		}
	}

	// Group iteration order: Overall first, then each by-value alphabetically.
	auto emit_var = [&](idx_t var_idx) {
		const std::string &var = bd.variables[var_idx];
		bool numeric = bd.is_numeric[var_idx];

		// Overall
		if (numeric) {
			EmitNumericRows(conn, bd.data_table, var, bd.by_column, "", "Overall", state->rows);
		} else {
			EmitCategoricalRows(conn, bd.data_table, var, bd.by_column, "", "Overall", state->rows);
		}
		// Per-group breakdown.
		for (auto &g : by_values) {
			if (numeric) {
				EmitNumericRows(conn, bd.data_table, var, bd.by_column, g, g, state->rows);
			} else {
				EmitCategoricalRows(conn, bd.data_table, var, bd.by_column, g, g, state->rows);
			}
		}
	};

	for (idx_t i = 0; i < bd.variables.size(); i++) {
		emit_var(i);
	}

	return std::move(state);
}

// ── Execute: stream pre-computed rows ──────────────────────────────────────

static void TableOneExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<TableOneGlobalState>();
	idx_t emitted = 0;
	while (state.cursor < state.rows.size() && emitted < STANDARD_VECTOR_SIZE) {
		auto &row = state.rows[state.cursor++];
		output.SetValue(0, emitted, Value(row.variable));
		// level: empty string → NULL (numeric variables have no level)
		if (row.level.empty()) {
			FlatVector::SetNull(output.data[1], emitted, true);
		} else {
			output.SetValue(1, emitted, Value(row.level));
		}
		output.SetValue(2, emitted, Value(row.statistic));
		output.SetValue(3, emitted, Value(row.stratum));
		output.SetValue(4, emitted, Value(row.display));
		emitted++;
	}
	output.SetCardinality(emitted);
}

} // namespace

void RegisterTableOne(ExtensionLoader &loader) {
	TableFunction fn("table_one", {LogicalType::VARCHAR}, TableOneExecute, TableOneBind,
	                  TableOneInitGlobal);
	fn.named_parameters["variables"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["by"] = LogicalType::VARCHAR;
	loader.RegisterFunction(fn);
}

} // namespace duckdb
