#include "lm_function.hpp"

#include "distributions.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Helpers shared by both lm and lm_summary.
//===--------------------------------------------------------------------===//

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

//! Same numeric kind check used by corr_matrix / table_one — everything we can
//! cast to DOUBLE without losing the intent of the column.
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

//===--------------------------------------------------------------------===//
// Linear-algebra primitives (small dense — p is rarely above a few dozen).
// Row-major storage: M[i,j] = M[i*ncols + j].
//===--------------------------------------------------------------------===//

//! In-place Cholesky decomposition of a symmetric positive-definite matrix A
//! (size n × n). On success returns true and overwrites the lower triangle
//! with L such that A = L·L'. On failure (A not positive-definite, e.g. due
//! to perfectly collinear predictors) returns false.
static bool CholeskyDecompose(std::vector<double> &A, idx_t n) {
	for (idx_t j = 0; j < n; j++) {
		double diag = A[j * n + j];
		for (idx_t k = 0; k < j; k++) {
			diag -= A[j * n + k] * A[j * n + k];
		}
		if (diag <= 0.0 || !std::isfinite(diag)) {
			return false;
		}
		A[j * n + j] = std::sqrt(diag);
		for (idx_t i = j + 1; i < n; i++) {
			double s = A[i * n + j];
			for (idx_t k = 0; k < j; k++) {
				s -= A[i * n + k] * A[j * n + k];
			}
			A[i * n + j] = s / A[j * n + j];
		}
	}
	return true;
}

//! Solve LL'·x = b given L (lower triangle of A in row-major form). Forward
//! then backward substitution.
static void CholeskySolve(const std::vector<double> &L, const std::vector<double> &b,
                          std::vector<double> &x, idx_t n) {
	std::vector<double> z(n, 0.0);
	// Forward: L·z = b
	for (idx_t i = 0; i < n; i++) {
		double s = b[i];
		for (idx_t j = 0; j < i; j++) {
			s -= L[i * n + j] * z[j];
		}
		z[i] = s / L[i * n + i];
	}
	// Backward: L'·x = z
	x.assign(n, 0.0);
	for (idx_t ii = 0; ii < n; ii++) {
		idx_t i = n - 1 - ii;
		double s = z[i];
		for (idx_t j = i + 1; j < n; j++) {
			s -= L[j * n + i] * x[j];
		}
		x[i] = s / L[i * n + i];
	}
}

//===--------------------------------------------------------------------===//
// Fit results.
//===--------------------------------------------------------------------===//

struct LmFit {
	idx_t n;                          // number of complete-case rows
	idx_t p;                          // number of predictors (excluding intercept)
	bool has_intercept;               // intercept term present in the design
	std::vector<std::string> terms;   // length k — "(Intercept)" iff has_intercept, then each x
	std::vector<double> beta;         // length k
	std::vector<double> std_error;    // length k
	std::vector<double> t_statistic;  // length k
	std::vector<double> p_value;      // length k
	double r_squared;
	double adj_r_squared;
	double f_statistic;
	double f_p_value;
	double sigma;                     // residual standard error
	idx_t df_residual;                // n - p - 1
	idx_t df_model;                   // p
	bool ok;                          // false → fit failed (insufficient data / singular)
	std::string error;                // populated when ok = false
};

//===--------------------------------------------------------------------===//
// Materialize y + X by streaming the source table once via an internal
// Connection. Filters all rows with NULL in y or any x (complete-case).
//===--------------------------------------------------------------------===//

struct YXBuffers {
	std::vector<double> y;
	std::vector<std::vector<double>> x; // x[j] is column j of X (size n)
};

static YXBuffers MaterializeYX(Connection &conn, const std::string &table, const std::string &y_col,
                               const std::vector<std::string> &x_cols) {
	std::stringstream sql;
	sql << "SELECT " << QuoteIdent(y_col) << "::DOUBLE";
	for (auto &xc : x_cols) {
		sql << ", " << QuoteIdent(xc) << "::DOUBLE";
	}
	sql << " FROM " << QuoteIdent(table);
	sql << " WHERE " << QuoteIdent(y_col) << " IS NOT NULL";
	for (auto &xc : x_cols) {
		sql << " AND " << QuoteIdent(xc) << " IS NOT NULL";
	}
	auto result = conn.Query(sql.str());
	if (result->HasError()) {
		throw InvalidInputException("lm: failed to materialize data — %s", result->GetError());
	}
	YXBuffers buf;
	buf.x.assign(x_cols.size(), {});
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto y_vec = FlatVector::GetData<double>(chunk->data[0]);
		for (idx_t i = 0; i < chunk->size(); i++) {
			buf.y.push_back(y_vec[i]);
		}
		for (idx_t j = 0; j < x_cols.size(); j++) {
			auto x_vec = FlatVector::GetData<double>(chunk->data[1 + j]);
			for (idx_t i = 0; i < chunk->size(); i++) {
				buf.x[j].push_back(x_vec[i]);
			}
		}
	}
	return buf;
}

//===--------------------------------------------------------------------===//
// FitOls — OLS fit via Cholesky on X'X. Populates every field of LmFit.
// When has_intercept is false, the design has no constant column. R²/F use
// the "uncentered" formulation (TSS = Σ y², not Σ(y - ȳ)²) to match R's
// summary.lm output for no-intercept models — interpret with care.
//===--------------------------------------------------------------------===//

static LmFit FitOls(Connection &conn, const std::string &table, const std::string &y_col,
                    const std::vector<std::string> &x_cols, bool has_intercept) {
	LmFit fit;
	fit.ok = false;
	fit.p = x_cols.size();
	fit.has_intercept = has_intercept;
	fit.terms.reserve(fit.p + (has_intercept ? 1 : 0));
	if (has_intercept) {
		fit.terms.emplace_back("(Intercept)");
	}
	for (auto &xc : x_cols) {
		fit.terms.push_back(xc);
	}

	auto buf = MaterializeYX(conn, table, y_col, x_cols);
	fit.n = buf.y.size();
	idx_t k = fit.terms.size(); // number of parameters

	if (fit.n <= k) {
		fit.error = StringUtil::Format(
		    "lm: need n > k (k=%llu params), got n=%llu — not enough complete-case rows",
		    static_cast<unsigned long long>(k), static_cast<unsigned long long>(fit.n));
		return fit;
	}

	// Column accessor: column j of X for row r. j=0 is the intercept (1)
	// when has_intercept is true; otherwise it's the first predictor.
	auto Xcol = [&](idx_t r, idx_t j) -> double {
		if (has_intercept) {
			if (j == 0) {
				return 1.0;
			}
			return buf.x[j - 1][r];
		}
		return buf.x[j][r];
	};

	// Build A = X'X (upper triangle only) and b = X'y. Row-major.
	std::vector<double> A(k * k, 0.0);
	std::vector<double> b(k, 0.0);
	for (idx_t row = 0; row < fit.n; row++) {
		double y_v = buf.y[row];
		for (idx_t i = 0; i < k; i++) {
			double xi = Xcol(row, i);
			b[i] += xi * y_v;
			for (idx_t j = i; j < k; j++) {
				double xj = Xcol(row, j);
				A[i * k + j] += xi * xj;
			}
		}
	}
	// Symmetrize A (only upper triangle was populated above).
	for (idx_t i = 0; i < k; i++) {
		for (idx_t j = 0; j < i; j++) {
			A[i * k + j] = A[j * k + i];
		}
	}

	// Cholesky decompose A. A copy is kept around so we can recover the
	// variance-covariance matrix below — the decomposition writes over A.
	std::vector<double> L = A;
	if (!CholeskyDecompose(L, k)) {
		fit.error =
		    "lm: design matrix is singular (X'X not positive-definite); "
		    "check for perfectly collinear predictors or a constant column";
		return fit;
	}

	// Solve LL'·β = b
	CholeskySolve(L, b, fit.beta, k);

	// Compute residuals e = y - X·β and RSS, plus a few sums we'll reuse for
	// the model summary.
	double rss = 0.0;
	double y_sum = 0.0;
	double y_sq_sum = 0.0;
	for (idx_t row = 0; row < fit.n; row++) {
		double yhat = 0.0;
		for (idx_t j = 0; j < k; j++) {
			yhat += fit.beta[j] * Xcol(row, j);
		}
		double e = buf.y[row] - yhat;
		rss += e * e;
		y_sum += buf.y[row];
		y_sq_sum += buf.y[row] * buf.y[row];
	}
	double y_mean = y_sum / static_cast<double>(fit.n);
	double tss_centered = 0.0;
	for (idx_t row = 0; row < fit.n; row++) {
		double dy = buf.y[row] - y_mean;
		tss_centered += dy * dy;
	}
	// "Uncentered" TSS used by R when there's no intercept.
	double tss = has_intercept ? tss_centered : y_sq_sum;

	fit.df_model = fit.p;
	fit.df_residual = fit.n - k;
	double sigma2 = rss / static_cast<double>(fit.df_residual);
	fit.sigma = std::sqrt(sigma2);

	// Compute the diagonal of A⁻¹ by solving A·v_i = e_i for each i, taking
	// v_i[i]. Same Cholesky factor; O(k) solves of O(k²) each.
	fit.std_error.assign(k, 0.0);
	fit.t_statistic.assign(k, 0.0);
	fit.p_value.assign(k, 0.0);
	std::vector<double> ei(k, 0.0);
	std::vector<double> vi(k, 0.0);
	for (idx_t i = 0; i < k; i++) {
		std::fill(ei.begin(), ei.end(), 0.0);
		ei[i] = 1.0;
		CholeskySolve(L, ei, vi, k);
		double var_ii = sigma2 * vi[i];
		fit.std_error[i] = var_ii > 0.0 ? std::sqrt(var_ii) : 0.0;
		if (fit.std_error[i] > 0.0) {
			fit.t_statistic[i] = fit.beta[i] / fit.std_error[i];
			double abs_t = std::fabs(fit.t_statistic[i]);
			// Two-sided t-test p-value: 2 · P(T > |t|) on df_residual.
			double tail = 1.0 - stats_duck::StudentTCDF(abs_t, static_cast<double>(fit.df_residual));
			fit.p_value[i] = 2.0 * tail;
		} else {
			fit.t_statistic[i] = std::numeric_limits<double>::quiet_NaN();
			fit.p_value[i] = std::numeric_limits<double>::quiet_NaN();
		}
	}

	// Model-level statistics. R² is centered when an intercept is present
	// (TSS = Σ(y - ȳ)²); R reports the *uncentered* version when there's no
	// intercept (TSS = Σ y²), which we mirror — that's what `tss` holds above.
	// For adj-R², the divisor n - p uses the full parameter count k matching
	// R: with intercept k = p + 1 and divisor n - p - 1 = df_residual; without
	// intercept k = p and divisor n - p = df_residual.
	if (tss > 0.0) {
		fit.r_squared = 1.0 - rss / tss;
		double n_d = static_cast<double>(fit.n);
		double r_n = has_intercept ? n_d - 1.0 : n_d;
		fit.adj_r_squared =
		    1.0 - (1.0 - fit.r_squared) * r_n / static_cast<double>(fit.df_residual);
	} else {
		fit.r_squared = std::numeric_limits<double>::quiet_NaN();
		fit.adj_r_squared = std::numeric_limits<double>::quiet_NaN();
	}
	// F-statistic for the joint hypothesis β_1 = ... = β_p = 0.
	if (fit.df_model > 0 && fit.df_residual > 0 && rss > 0.0 && tss > 0.0) {
		double ess = tss - rss;
		fit.f_statistic = (ess / static_cast<double>(fit.df_model)) /
		                  (rss / static_cast<double>(fit.df_residual));
		fit.f_p_value = 1.0 - stats_duck::FCDF(fit.f_statistic,
		                                       static_cast<double>(fit.df_model),
		                                       static_cast<double>(fit.df_residual));
	} else {
		fit.f_statistic = std::numeric_limits<double>::quiet_NaN();
		fit.f_p_value = std::numeric_limits<double>::quiet_NaN();
	}

	fit.ok = true;
	return fit;
}

//===--------------------------------------------------------------------===//
// Formula DSL — a small subset of R's formula syntax.
//
// Grammar (v0.6):
//   formula    := IDENT  '~'  rhs
//   rhs        := term  (('+' | '-')  term)*
//   term       := IDENT          (predictor column)
//                | '0'            (alias for '- 1' — drop intercept)
//                | '1'            (no-op when '+' / drops intercept when '-')
//
// Identifiers are bare `[A-Za-z_][A-Za-z0-9_.]*` or `"..."` (SQL-quoted, with
// '""' escape). Whitespace between tokens is ignored. The intercept is
// included by default and can be removed with `- 1` or `+ 0` (matches R's
// convention; both spellings appear in the wild).
//
// Not supported in v0.6: interactions (`x1:x2`), wildcards (`*`, `^`,
// `.`), inline expressions (`I(x^2)`, `log(x)`), or transformations. Quote
// computed columns into a CTE if you need them today.
//===--------------------------------------------------------------------===//

struct FormulaParse {
	bool ok = false;
	std::string error;
	std::string y;
	std::vector<std::string> x;
	bool intercept = true;
};

namespace fml {

enum TokKind { IDENT, TILDE, PLUS, MINUS, ZERO, ONE, END_TOK };

struct Tok {
	TokKind kind;
	std::string text;
	idx_t pos;
};

static bool IsIdentStart(char c) {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}
static bool IsIdentCont(char c) {
	return IsIdentStart(c) || (c >= '0' && c <= '9') || c == '.';
}

static std::vector<Tok> Tokenize(const std::string &s, std::string &error) {
	std::vector<Tok> tokens;
	idx_t i = 0;
	while (i < s.size()) {
		char c = s[i];
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
			i++;
			continue;
		}
		idx_t start = i;
		if (c == '~') { tokens.push_back({TILDE, "", start}); i++; continue; }
		if (c == '+') { tokens.push_back({PLUS, "", start}); i++; continue; }
		if (c == '-') { tokens.push_back({MINUS, "", start}); i++; continue; }
		if (c >= '0' && c <= '9') {
			std::string num;
			while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
				num += s[i];
				i++;
			}
			if (num == "0") {
				tokens.push_back({ZERO, "", start});
			} else if (num == "1") {
				tokens.push_back({ONE, "", start});
			} else {
				error = StringUtil::Format(
				    "formula: unexpected number '%s' at position %llu — only '0' / '1' are valid",
				    num, static_cast<unsigned long long>(start));
				return tokens;
			}
			continue;
		}
		if (c == '"') {
			std::string id;
			i++;
			while (i < s.size()) {
				if (s[i] == '"') {
					if (i + 1 < s.size() && s[i + 1] == '"') {
						id += '"';
						i += 2;
						continue;
					}
					break;
				}
				id += s[i];
				i++;
			}
			if (i >= s.size()) {
				error = "formula: unterminated quoted identifier";
				return tokens;
			}
			i++; // closing "
			tokens.push_back({IDENT, id, start});
			continue;
		}
		if (IsIdentStart(c)) {
			std::string id;
			while (i < s.size() && IsIdentCont(s[i])) {
				id += s[i];
				i++;
			}
			tokens.push_back({IDENT, id, start});
			continue;
		}
		error = StringUtil::Format("formula: unexpected character '%c' at position %llu",
		                            c, static_cast<unsigned long long>(start));
		return tokens;
	}
	tokens.push_back({END_TOK, "", i});
	return tokens;
}

} // namespace fml

static FormulaParse ParseFormula(const std::string &formula) {
	FormulaParse out;
	std::string err;
	auto tokens = fml::Tokenize(formula, err);
	if (!err.empty()) {
		out.error = err;
		return out;
	}
	if (tokens.empty() || tokens[0].kind != fml::IDENT) {
		out.error = "formula: expected response column name on LHS of '~'";
		return out;
	}
	out.y = tokens[0].text;
	if (tokens.size() < 2 || tokens[1].kind != fml::TILDE) {
		out.error = "formula: expected '~' after response column";
		return out;
	}
	idx_t i = 2;
	if (i >= tokens.size() || tokens[i].kind == fml::END_TOK) {
		out.error = "formula: expected at least one term after '~'";
		return out;
	}
	bool first = true;
	while (i < tokens.size() && tokens[i].kind != fml::END_TOK) {
		bool subtract = false;
		if (!first) {
			if (tokens[i].kind == fml::PLUS) {
				subtract = false;
				i++;
			} else if (tokens[i].kind == fml::MINUS) {
				subtract = true;
				i++;
			} else {
				out.error = "formula: expected '+' or '-' between terms";
				return out;
			}
			if (i >= tokens.size() || tokens[i].kind == fml::END_TOK) {
				out.error = "formula: expected term after operator";
				return out;
			}
		}
		first = false;
		auto &t = tokens[i];
		if (t.kind == fml::ZERO) {
			// `+ 0` drops the intercept; `- 0` is a no-op.
			if (!subtract) {
				out.intercept = false;
			}
			i++;
			continue;
		}
		if (t.kind == fml::ONE) {
			// `+ 1` is the default (intercept on); `- 1` drops the intercept.
			if (subtract) {
				out.intercept = false;
			}
			i++;
			continue;
		}
		if (t.kind == fml::IDENT) {
			if (subtract) {
				out.error =
				    "formula: subtracting a predictor (other than '- 1' / '- 0' to drop the "
				    "intercept) is not supported — wrap into a CTE if you need to omit a column";
				return out;
			}
			for (auto &x : out.x) {
				if (x == t.text) {
					out.error = "formula: duplicate predictor '" + t.text + "'";
					return out;
				}
			}
			out.x.push_back(t.text);
			i++;
			continue;
		}
		out.error = "formula: unexpected token in RHS";
		return out;
	}
	if (out.x.empty()) {
		out.error =
		    "formula: at least one predictor is required (intercept-only models are not supported)";
		return out;
	}
	out.ok = true;
	return out;
}

//===--------------------------------------------------------------------===//
// Shared bind data — both lm and lm_summary need the same trio (table, y, x)
// plus a flag for whether the design has an intercept column.
//===--------------------------------------------------------------------===//

struct LmBindData : public TableFunctionData {
	std::string data_table;
	std::string y_col;
	std::vector<std::string> x_cols;
	bool has_intercept = true;
	bool is_summary;
};

static unique_ptr<FunctionData> LmBindCommon(ClientContext &context, TableFunctionBindInput &input,
                                              bool is_summary) {
	auto bd = make_uniq<LmBindData>();
	bd->is_summary = is_summary;
	const char *fname = is_summary ? "lm_summary" : "lm";

	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("%s: 'data' (source table name) is required", fname);
	}
	bd->data_table = input.inputs[0].GetValue<string>();

	// Two ways to specify the model:
	//   formula := 'y ~ x1 + x2'             (R-style DSL, ergonomic)
	//   y := 'y_col', x := ['x1', 'x2']      (explicit lists, easy to generate)
	// They are mutually exclusive — passing both is a bind error.
	auto it_formula = input.named_parameters.find("formula");
	auto it_y = input.named_parameters.find("y");
	auto it_x = input.named_parameters.find("x");
	bool has_formula = it_formula != input.named_parameters.end() && !it_formula->second.IsNull();
	bool has_y = it_y != input.named_parameters.end() && !it_y->second.IsNull();
	bool has_x = it_x != input.named_parameters.end() && !it_x->second.IsNull();

	if (has_formula && (has_y || has_x)) {
		throw BinderException(
		    "%s: cannot specify 'formula' together with 'y' or 'x' — choose one form", fname);
	}

	if (has_formula) {
		auto parsed = ParseFormula(it_formula->second.GetValue<string>());
		if (!parsed.ok) {
			throw BinderException("%s: %s", fname, parsed.error);
		}
		bd->y_col = std::move(parsed.y);
		bd->x_cols = std::move(parsed.x);
		bd->has_intercept = parsed.intercept;
	} else {
		if (!has_y) {
			throw BinderException("%s: either 'formula' or ('y' + 'x') is required", fname);
		}
		bd->y_col = it_y->second.GetValue<string>();
		if (!has_x) {
			throw BinderException("%s: 'x' (predictor list) is required when not using formula",
			                      fname);
		}
		auto &x_children = ListValue::GetChildren(it_x->second);
		for (auto &v : x_children) {
			if (v.IsNull()) {
				throw BinderException("%s: 'x' list entries must not be NULL", fname);
			}
			bd->x_cols.push_back(v.GetValue<string>());
		}
		if (bd->x_cols.empty()) {
			throw BinderException("%s: 'x' must contain at least one predictor column", fname);
		}
		bd->has_intercept = true;
	}

	// Catalog lookup — y and every x must exist and be numeric.
	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);
	auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA,
	                              bd->data_table, OnEntryNotFound::THROW_EXCEPTION);
	auto &tbl = entry->Cast<TableCatalogEntry>();
	auto &cols = tbl.GetColumns();

	auto check_col = [&](const std::string &c, const char *role) {
		if (!cols.ColumnExists(c)) {
			throw BinderException("%s: %s column '%s' not found in table '%s'", fname, role, c,
			                      bd->data_table);
		}
		auto &col = cols.GetColumn(c);
		if (!IsNumericKind(col.GetType().id())) {
			throw BinderException(
			    "%s: %s column '%s' is not numeric (type %s); OLS requires numeric columns",
			    fname, role, c, col.GetType().ToString());
		}
	};
	check_col(bd->y_col, "response");
	for (auto &xc : bd->x_cols) {
		check_col(xc, "predictor");
	}
	return std::move(bd);
}

static unique_ptr<FunctionData> LmBind(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	auto bd = LmBindCommon(context, input, /*is_summary=*/false);
	names = {"term", "estimate", "std_error", "t_statistic", "p_value"};
	return_types = {LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE};
	return bd;
}

static unique_ptr<FunctionData> LmSummaryBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<string> &names) {
	auto bd = LmBindCommon(context, input, /*is_summary=*/true);
	names = {"r_squared",   "adj_r_squared", "f_statistic", "f_p_value",
	         "df_model",    "df_residual",   "sigma",       "n"};
	return_types = {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::DOUBLE, LogicalType::BIGINT};
	return bd;
}

//===--------------------------------------------------------------------===//
// Global state — holds the fit result and a cursor over emitted rows.
//===--------------------------------------------------------------------===//

struct LmGlobalState : public GlobalTableFunctionState {
	LmFit fit;
	idx_t cursor = 0;
};

static unique_ptr<GlobalTableFunctionState> LmInitGlobal(ClientContext &context,
                                                         TableFunctionInitInput &input) {
	auto &bd = input.bind_data->Cast<LmBindData>();
	auto state = make_uniq<LmGlobalState>();
	Connection conn(*context.db);
	state->fit = FitOls(conn, bd.data_table, bd.y_col, bd.x_cols, bd.has_intercept);
	if (!state->fit.ok) {
		throw InvalidInputException(state->fit.error);
	}
	return std::move(state);
}

//===--------------------------------------------------------------------===//
// Execute — stream pre-computed rows.
//===--------------------------------------------------------------------===//

static inline void SetDoubleOrNull(DataChunk &output, idx_t col, idx_t row, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(output.data[col], row, true);
	} else {
		output.SetValue(col, row, Value::DOUBLE(v));
	}
}

static void LmExecute(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<LmGlobalState>();
	idx_t emitted = 0;
	while (state.cursor < state.fit.terms.size() && emitted < STANDARD_VECTOR_SIZE) {
		idx_t i = state.cursor++;
		output.SetValue(0, emitted, Value(state.fit.terms[i]));
		SetDoubleOrNull(output, 1, emitted, state.fit.beta[i]);
		SetDoubleOrNull(output, 2, emitted, state.fit.std_error[i]);
		SetDoubleOrNull(output, 3, emitted, state.fit.t_statistic[i]);
		SetDoubleOrNull(output, 4, emitted, state.fit.p_value[i]);
		emitted++;
	}
	output.SetCardinality(emitted);
}

static void LmSummaryExecute(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<LmGlobalState>();
	if (state.cursor > 0) {
		output.SetCardinality(0);
		return;
	}
	state.cursor = 1;
	SetDoubleOrNull(output, 0, 0, state.fit.r_squared);
	SetDoubleOrNull(output, 1, 0, state.fit.adj_r_squared);
	SetDoubleOrNull(output, 2, 0, state.fit.f_statistic);
	SetDoubleOrNull(output, 3, 0, state.fit.f_p_value);
	output.SetValue(4, 0, Value::BIGINT(static_cast<int64_t>(state.fit.df_model)));
	output.SetValue(5, 0, Value::BIGINT(static_cast<int64_t>(state.fit.df_residual)));
	SetDoubleOrNull(output, 6, 0, state.fit.sigma);
	output.SetValue(7, 0, Value::BIGINT(static_cast<int64_t>(state.fit.n)));
	output.SetCardinality(1);
}

} // namespace

void RegisterLm(ExtensionLoader &loader) {
	{
		TableFunction fn("lm", {LogicalType::VARCHAR}, LmExecute, LmBind, LmInitGlobal);
		fn.named_parameters["y"] = LogicalType::VARCHAR;
		fn.named_parameters["x"] = LogicalType::LIST(LogicalType::VARCHAR);
		fn.named_parameters["formula"] = LogicalType::VARCHAR;
		loader.RegisterFunction(fn);
	}
	{
		TableFunction fn("lm_summary", {LogicalType::VARCHAR}, LmSummaryExecute, LmSummaryBind,
		                 LmInitGlobal);
		fn.named_parameters["y"] = LogicalType::VARCHAR;
		fn.named_parameters["x"] = LogicalType::LIST(LogicalType::VARCHAR);
		fn.named_parameters["formula"] = LogicalType::VARCHAR;
		loader.RegisterFunction(fn);
	}
}

} // namespace duckdb
