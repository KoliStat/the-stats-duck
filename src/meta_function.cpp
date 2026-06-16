#include "meta_function.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace duckdb {

// meta — per-column dataset profile.
//
// At bind time we resolve the source table in the catalog and capture each
// declared column's name, type-as-text, and a semantic kind (numeric /
// categorical / temporal / boolean / other). At init-global time we open an
// internal Connection and run three batches:
//   1. one combined query with count(*), count(col_i), count(DISTINCT col_i)
//      across all columns — the cheap metadata that applies to every kind;
//   2. one combined query computing min / p25 / median / p75 / max / avg /
//      stddev_samp for the numeric columns only (cast to DOUBLE);
//   3. one mode query per categorical / boolean column, giving (top, top_freq)
//      with ties broken by the smaller value.
// Execute then streams the pre-built rows STANDARD_VECTOR_SIZE at a time.

namespace {

// ── Helpers ────────────────────────────────────────────────────────────────

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

enum class ColKind {
	NUMERIC,
	CATEGORICAL,
	TEMPORAL,
	BOOLEAN,
	OTHER,
};

static const char *KindToString(ColKind k) {
	switch (k) {
	case ColKind::NUMERIC:
		return "numeric";
	case ColKind::CATEGORICAL:
		return "categorical";
	case ColKind::TEMPORAL:
		return "temporal";
	case ColKind::BOOLEAN:
		return "boolean";
	case ColKind::OTHER:
		return "other";
	}
	return "other";
}

//! Same numeric classifier used by corr_matrix / lm / table_one — anything
//! that casts to DOUBLE cleanly. We split out boolean and temporal from
//! "other" since both have natural per-column profiling (mode and min/max-as-
//! text respectively); the meta output currently fills mode for boolean but
//! leaves temporal min/max NULL (typed numeric column would lose meaning).
static ColKind ClassifyKind(LogicalTypeId tid) {
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
		return ColKind::NUMERIC;
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::ENUM:
		return ColKind::CATEGORICAL;
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return ColKind::TEMPORAL;
	case LogicalTypeId::BOOLEAN:
		return ColKind::BOOLEAN;
	default:
		return ColKind::OTHER;
	}
}

// ── BindData / state ───────────────────────────────────────────────────────

struct MetaRow {
	std::string column_name;
	std::string column_type;
	ColKind kind = ColKind::OTHER;
	int64_t n_rows = 0;
	int64_t n_missing = 0;
	int64_t n_distinct = 0;
	double min = std::nan("");
	double p25 = std::nan("");
	double median = std::nan("");
	double p75 = std::nan("");
	double max = std::nan("");
	double mean = std::nan("");
	double stddev = std::nan("");
	std::string top;
	bool has_top = false;
	int64_t top_freq = 0;
};

struct MetaBindData : public TableFunctionData {
	std::string data_table;
	std::vector<std::string> column_names;
	std::vector<std::string> column_types;
	std::vector<ColKind> kinds;
};

struct MetaGlobalState : public GlobalTableFunctionState {
	std::vector<MetaRow> rows;
	idx_t cursor = 0;
};

// ── Bind ───────────────────────────────────────────────────────────────────

static unique_ptr<FunctionData> MetaBind(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types,
                                         vector<string> &names) {
	auto bd = make_uniq<MetaBindData>();

	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("meta: 'data' (source table name) is required");
	}
	bd->data_table = input.inputs[0].GetValue<string>();

	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);
	auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA,
	                              bd->data_table, OnEntryNotFound::THROW_EXCEPTION);
	auto &tbl = entry->Cast<TableCatalogEntry>();
	auto &cols = tbl.GetColumns();
	for (auto &col : cols.Logical()) {
		bd->column_names.push_back(col.GetName());
		bd->column_types.push_back(col.GetType().ToString());
		bd->kinds.push_back(ClassifyKind(col.GetType().id()));
	}

	names = {"column_name", "column_type", "kind",   "n_rows", "n_missing",
	         "n_distinct",  "min",         "p25",    "median", "p75",
	         "max",         "mean",        "stddev", "top",    "top_freq"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::BIGINT,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::VARCHAR, LogicalType::BIGINT};
	return std::move(bd);
}

// ── Init / fetch ──────────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> MetaInitGlobal(ClientContext &context,
                                                           TableFunctionInitInput &input) {
	auto &bd = input.bind_data->Cast<MetaBindData>();
	auto state = make_uniq<MetaGlobalState>();

	const idx_t K = bd.column_names.size();
	state->rows.reserve(K);
	for (idx_t i = 0; i < K; i++) {
		MetaRow row;
		row.column_name = bd.column_names[i];
		row.column_type = bd.column_types[i];
		row.kind = bd.kinds[i];
		state->rows.push_back(std::move(row));
	}

	if (K == 0) {
		return std::move(state);
	}

	Connection conn(*context.db);
	auto quoted_table = QuoteIdent(bd.data_table);

	// 1. count(*) + per-column (non-null, distinct) — applies to every kind.
	{
		std::stringstream sql;
		sql << "SELECT count(*)";
		for (idx_t i = 0; i < K; i++) {
			auto qname = QuoteIdent(bd.column_names[i]);
			sql << ", count(" << qname << ")";
			sql << ", count(DISTINCT " << qname << ")";
		}
		sql << " FROM " << quoted_table;
		auto result = conn.Query(sql.str());
		if (result->HasError()) {
			throw InvalidInputException("meta: failed to compute counts on '%s' (%s)",
			                            bd.data_table, result->GetError());
		}
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			throw InvalidInputException("meta: empty result from counts query");
		}
		auto n_rows = chunk->GetValue(0, 0).GetValue<int64_t>();
		for (idx_t i = 0; i < K; i++) {
			state->rows[i].n_rows = n_rows;
			auto non_null = chunk->GetValue(1 + 2 * i, 0).GetValue<int64_t>();
			state->rows[i].n_missing = n_rows - non_null;
			state->rows[i].n_distinct = chunk->GetValue(2 + 2 * i, 0).GetValue<int64_t>();
		}
	}

	// 2. Numeric distribution — min, p25, median, p75, max, mean, stddev_samp.
	//    Single combined query keyed by numeric-column index.
	std::vector<idx_t> numeric_idxs;
	for (idx_t i = 0; i < K; i++) {
		if (bd.kinds[i] == ColKind::NUMERIC) {
			numeric_idxs.push_back(i);
		}
	}
	if (!numeric_idxs.empty()) {
		std::stringstream sql;
		sql << "SELECT ";
		for (size_t j = 0; j < numeric_idxs.size(); j++) {
			idx_t i = numeric_idxs[j];
			auto qcast = QuoteIdent(bd.column_names[i]) + "::DOUBLE";
			if (j > 0) {
				sql << ", ";
			}
			sql << "min(" << qcast << "), "
			    << "quantile_cont(" << qcast << ", 0.25), "
			    << "quantile_cont(" << qcast << ", 0.5), "
			    << "quantile_cont(" << qcast << ", 0.75), "
			    << "max(" << qcast << "), "
			    << "avg(" << qcast << "), "
			    << "stddev_samp(" << qcast << ")";
		}
		sql << " FROM " << quoted_table;
		auto result = conn.Query(sql.str());
		if (result->HasError()) {
			throw InvalidInputException("meta: failed to compute numeric stats on '%s' (%s)",
			                            bd.data_table, result->GetError());
		}
		auto chunk = result->Fetch();
		if (chunk && chunk->size() > 0) {
			for (size_t j = 0; j < numeric_idxs.size(); j++) {
				idx_t i = numeric_idxs[j];
				idx_t base = 7 * j;
				auto get_dbl = [&](idx_t col_in_chunk) -> double {
					auto v = chunk->GetValue(col_in_chunk, 0);
					if (v.IsNull()) {
						return std::nan("");
					}
					return v.GetValue<double>();
				};
				state->rows[i].min = get_dbl(base + 0);
				state->rows[i].p25 = get_dbl(base + 1);
				state->rows[i].median = get_dbl(base + 2);
				state->rows[i].p75 = get_dbl(base + 3);
				state->rows[i].max = get_dbl(base + 4);
				state->rows[i].mean = get_dbl(base + 5);
				state->rows[i].stddev = get_dbl(base + 6);
			}
		}
	}

	// 3. Mode (top, top_freq) for categorical and boolean columns. One query
	//    per column — could be folded into a single GROUPING SETS pass, but
	//    that adds complexity for little gain when categorical columns are
	//    a small share of the typical schema.
	for (idx_t i = 0; i < K; i++) {
		if (bd.kinds[i] != ColKind::CATEGORICAL && bd.kinds[i] != ColKind::BOOLEAN) {
			continue;
		}
		auto qname = QuoteIdent(bd.column_names[i]);
		std::stringstream sql;
		sql << "SELECT " << qname << "::VARCHAR AS v, count(*) AS c "
		    << "FROM " << quoted_table << " WHERE " << qname << " IS NOT NULL"
		    << " GROUP BY 1 ORDER BY 2 DESC, 1 ASC LIMIT 1";
		auto result = conn.Query(sql.str());
		if (result->HasError()) {
			// Don't fail the whole call on a single mode query failure; just
			// leave top/top_freq NULL for this column.
			continue;
		}
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			continue;
		}
		if (chunk->GetValue(0, 0).IsNull()) {
			continue;
		}
		state->rows[i].top = chunk->GetValue(0, 0).GetValue<string>();
		state->rows[i].has_top = true;
		state->rows[i].top_freq = chunk->GetValue(1, 0).GetValue<int64_t>();
	}

	return std::move(state);
}

// ── Execute ───────────────────────────────────────────────────────────────

static inline void SetDoubleOrNull(DataChunk &output, idx_t col, idx_t row, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(output.data[col], row, true);
	} else {
		output.SetValue(col, row, Value::DOUBLE(v));
	}
}

static void MetaExecute(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<MetaGlobalState>();
	idx_t emitted = 0;
	while (state.cursor < state.rows.size() && emitted < STANDARD_VECTOR_SIZE) {
		const auto &r = state.rows[state.cursor++];
		output.SetValue(0, emitted, Value(r.column_name));
		output.SetValue(1, emitted, Value(r.column_type));
		output.SetValue(2, emitted, Value(KindToString(r.kind)));
		output.SetValue(3, emitted, Value::BIGINT(r.n_rows));
		output.SetValue(4, emitted, Value::BIGINT(r.n_missing));
		output.SetValue(5, emitted, Value::BIGINT(r.n_distinct));
		SetDoubleOrNull(output, 6, emitted, r.min);
		SetDoubleOrNull(output, 7, emitted, r.p25);
		SetDoubleOrNull(output, 8, emitted, r.median);
		SetDoubleOrNull(output, 9, emitted, r.p75);
		SetDoubleOrNull(output, 10, emitted, r.max);
		SetDoubleOrNull(output, 11, emitted, r.mean);
		SetDoubleOrNull(output, 12, emitted, r.stddev);
		if (r.has_top) {
			output.SetValue(13, emitted, Value(r.top));
			output.SetValue(14, emitted, Value::BIGINT(r.top_freq));
		} else {
			FlatVector::SetNull(output.data[13], emitted, true);
			FlatVector::SetNull(output.data[14], emitted, true);
		}
		emitted++;
	}
	output.SetCardinality(emitted);
}

} // namespace

void RegisterMeta(ExtensionLoader &loader) {
	TableFunction fn("meta", {LogicalType::VARCHAR}, MetaExecute, MetaBind, MetaInitGlobal);
	loader.RegisterFunction(fn);
}

} // namespace duckdb
