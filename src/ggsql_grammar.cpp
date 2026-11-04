#include "ggsql_grammar.hpp"

#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace ggsql {

namespace {

struct TokenizeResult {
	vector<string> tokens;
	string error;
};

// Tokenizer: at depth 0, splits on whitespace and emits commas as their own
// token. Inside parens or quotes (single or double) all characters are
// accumulated into the current token so that `coalesce(a, b)`, `"my col"`,
// `'don''t'`, and `(SELECT max(x) FROM t)` all become single tokens. SQL
// doubled-quote escapes (`''` and `""`) are handled. Trailing semicolons are
// dropped.
TokenizeResult Tokenize(const string &input) {
	TokenizeResult out;
	auto &tokens = out.tokens;
	string current;
	int paren_depth = 0;
	char quote_char = 0;

	auto flush = [&]() {
		if (!current.empty()) {
			tokens.push_back(std::move(current));
			current.clear();
		}
	};

	for (size_t i = 0; i < input.size(); i++) {
		char c = input[i];

		if (quote_char != 0) {
			current += c;
			if (c == quote_char) {
				if (i + 1 < input.size() && input[i + 1] == quote_char) {
					current += input[++i]; // SQL '' / "" escape
				} else {
					quote_char = 0;
				}
			}
			continue;
		}

		if (c == '\'' || c == '"') {
			current += c;
			quote_char = c;
			continue;
		}

		if (paren_depth > 0) {
			if (c == '(') {
				paren_depth++;
			} else if (c == ')') {
				paren_depth--;
			}
			current += c;
			continue;
		}

		// Outside parens, outside quotes.
		// SQL comments act as delimiters and are skipped. They can't occur inside
		// a quote (handled above); inside parens they ride along in the expression
		// token for DuckDB to strip. `--` runs to end of line; `/* */` to its close.
		if (c == '-' && i + 1 < input.size() && input[i + 1] == '-') {
			flush();
			i += 2;
			while (i < input.size() && input[i] != '\n') {
				i++;
			}
			continue; // the for-loop's ++ steps past the newline (or end)
		}
		if (c == '/' && i + 1 < input.size() && input[i + 1] == '*') {
			flush();
			i += 2;
			while (i + 1 < input.size() && !(input[i] == '*' && input[i + 1] == '/')) {
				i++;
			}
			i = (i + 1 < input.size()) ? i + 1 : input.size(); // land on the closing '/'
			continue;
		}
		if (c == '(') {
			paren_depth++;
			current += c;
		} else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
			flush();
		} else if (c == ',') {
			flush();
			tokens.emplace_back(",");
		} else if (c == ';') {
			flush();
		} else {
			current += c;
		}
	}
	flush();

	if (quote_char != 0) {
		out.error = string("Unterminated string literal or quoted identifier (missing closing ") +
		            quote_char + ")";
	} else if (paren_depth != 0) {
		out.error = "Unbalanced parentheses in input";
	}
	return out;
}

bool IEqual(const string &tok, const char *kw) {
	return StringUtil::CIEquals(tok, kw);
}

// Upstream ggsql's names for our channels and marks (#88, Tier 1 of #58). An
// alias is resolved where it is read, so the statement, the layer SQL and the
// spec only ever carry the canonical name. The `smooth` mark is not here: it
// stands for `line STAT smooth`, which the DRAW clause handles itself.
struct Alias {
	const char *written;
	const char *canonical;
};

const Alias kChannelAliases[] = {{"colour", "color"}, {"label", "text"}, {"xmin", "x"},  {"xmax", "x2"},
                                 {"ymin", "y"},       {"ymax", "y2"},    {"xend", "x2"}, {"yend", "y2"}};

const Alias kMarkAliases[] = {{"range", "errorbar"}, {"ribbon", "errorband"}, {"tile", "heatmap"}};

template <size_t N>
string Canonical(const string &name, const Alias (&aliases)[N]) {
	for (const auto &alias : aliases) {
		if (IEqual(name, alias.written)) {
			return alias.canonical;
		}
	}
	return name;
}

string CanonicalChannel(const string &name) {
	return Canonical(name, kChannelAliases);
}

string CanonicalMark(const string &name) {
	return Canonical(name, kMarkAliases);
}

} // namespace

string UnquoteSqlString(const string &raw) {
	if (raw.size() < 2 || raw.front() != '\'' || raw.back() != '\'') {
		return raw;
	}
	string out;
	out.reserve(raw.size() - 2);
	for (size_t i = 1; i + 1 < raw.size(); i++) {
		// Collapse SQL '' escape to a single quote.
		if (raw[i] == '\'' && i + 2 < raw.size() && raw[i + 1] == '\'') {
			out += '\'';
			i++;
		} else {
			out += raw[i];
		}
	}
	return out;
}

string JsonEscape(const string &s) {
	string out;
	out.reserve(s.size());
	for (unsigned char c : s) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\b':
			out += "\\b";
			break;
		case '\f':
			out += "\\f";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (c < 0x20) {
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				out += buf;
			} else {
				out += static_cast<char>(c);
			}
		}
	}
	return out;
}

// The statement keyword, in both spellings that upstream ggsql accepts (#84).
bool IsVisualizeKeyword(const string &token) {
	return IEqual(token, "VISUALIZE") || IEqual(token, "VISUALISE");
}

namespace {

// `VISUALIZE(` written without a space. The tokenizer glues a depth-0 `(` to
// the word before it, so the keyword and the parenthesis arrive as one token.
// The claim treats the token as ours and the parser names the problem (#86).
bool IsKeywordGluedToParen(const string &token) {
	return token.size() > 9 && token[9] == '(' && IsVisualizeKeyword(token.substr(0, 9));
}

// The spelling the user wrote, upper-cased, from a keyword token or a glued one.
string KeywordSpelling(const string &token) {
	return StringUtil::Upper(token.substr(0, 9));
}

// Walks a leading WITH block by its structure and returns the index of the
// token after it. The block is `WITH [RECURSIVE] <name> ... AS ... (<body>)`,
// repeated after each `,`. Anything may stand between the name and AS (a
// column list, USING KEY) and between AS and the body (MATERIALIZED, NOT
// MATERIALIZED). The body is the first token after AS that holds a `(`; on
// DuckDB 2.0 the token adapter can glue a word to the parenthesis. The name
// position takes any token, so a CTE called `visualize` is a CTE and not the
// statement keyword. Returns tokens.size() when the block runs to the end of
// the input or does not have this shape. tokens[0] must be WITH.
size_t WithBlockEnd(const vector<string> &tokens) {
	const size_t none = tokens.size();
	size_t i = 1;
	if (i < tokens.size() && IEqual(tokens[i], "RECURSIVE")) {
		i++;
	}
	while (true) {
		if (i >= tokens.size()) {
			return none;
		}
		i++; // the CTE name, whatever it is
		while (i < tokens.size() && !IEqual(tokens[i], "AS")) {
			i++;
		}
		if (i >= tokens.size()) {
			return none;
		}
		i++; // AS
		while (i < tokens.size() && tokens[i].find('(') == string::npos) {
			i++;
		}
		if (i >= tokens.size()) {
			return none;
		}
		i++; // the parenthesised body, one token
		if (i < tokens.size() && tokens[i] == ",") {
			i++;
			continue;
		}
		return i;
	}
}

// Where the statement keyword is, when the token stream is ours.
struct Claim {
	bool ours = false;
	size_t keyword = 0;          // index of the keyword token, or of the glued `VISUALIZE(` token
	bool malformed_with = false; // ours, but the WITH block before the keyword is not <name> AS (...)
};

// Decides ownership on the tokens. The first token is the keyword, or it is
// WITH and the keyword follows the CTE list. A WITH statement whose CTE list
// is followed by anything else (SELECT, a misspelt SELECT, ...) is DuckDB's,
// whatever words appear later at depth 0: a column alias `visualise` or a
// comment does not make a statement ours. When the WITH block does not have
// the <name> AS (...) shape but a keyword token follows it anyway, the
// statement is ours with an error that names the WITH clause (#86).
Claim ClaimTokens(const vector<string> &tokens) {
	Claim claim;
	if (tokens.empty()) {
		return claim;
	}
	auto is_keyword = [&](size_t k) { return IsVisualizeKeyword(tokens[k]) || IsKeywordGluedToParen(tokens[k]); };
	if (is_keyword(0)) {
		claim.ours = true;
		return claim;
	}
	if (!IEqual(tokens[0], "WITH")) {
		return claim;
	}
	size_t end = WithBlockEnd(tokens);
	if (end < tokens.size()) {
		if (is_keyword(end)) {
			claim.ours = true;
			claim.keyword = end;
		}
		return claim;
	}
	for (size_t k = 1; k < tokens.size(); k++) {
		if (is_keyword(k)) {
			claim.ours = true;
			claim.keyword = k;
			claim.malformed_with = true;
			return claim;
		}
	}
	return claim;
}

} // namespace

// A tokenizer error does not decide ownership by itself: the tokens read so
// far do. A direct statement with an unterminated quote later on is ours, and
// ParseGgsql reports the quote. A WITH statement with the quote inside a CTE
// body never shows the keyword as a token, so it stays with DuckDB, whose own
// message names the quote.
bool IsVisualizeStatement(const string &query) {
	return ClaimTokens(Tokenize(query).tokens).ours;
}

ParseResult ParseGgsql(const string &query) {
	ParseResult result;
	auto tokenized = Tokenize(query);
	if (!tokenized.error.empty()) {
		result.error = tokenized.error;
		return result;
	}
	auto &tokens = tokenized.tokens;
	size_t i = 0;

	auto at_end = [&]() { return i >= tokens.size(); };
	auto peek = [&]() -> const string & {
		static const string empty;
		return at_end() ? empty : tokens[i];
	};
	// The error for a missing keyword: what was expected and what was found.
	auto fail_expected = [&](const char *shown) {
		result.error = string("Expected '") + shown + "', got " + (at_end() ? "end of input" : "'" + tokens[i] + "'");
	};
	auto consume = [&](const char *kw) -> bool {
		if (at_end() || !IEqual(tokens[i], kw)) {
			fail_expected(kw);
			return false;
		}
		i++;
		return true;
	};

	// The same claim the parser extension made before calling us. A leading
	// WITH block ends where the claim found the keyword. Its tokens, joined
	// with single spaces, are the SQL prepended to every layer; SQL is
	// whitespace-tolerant, so `WITH t AS (…) , u AS (…)` parses the same as
	// `WITH t AS (…), u AS (…)`.
	auto claim = ClaimTokens(tokens);
	if (!claim.ours) {
		if (!at_end() && IEqual(peek(), "WITH")) {
			result.error = "WITH clause is not followed by VISUALIZE or VISUALISE";
		} else {
			fail_expected("VISUALIZE");
		}
		return result;
	}
	if (claim.malformed_with) {
		result.error = "Malformed WITH clause before " + KeywordSpelling(tokens[claim.keyword]) +
		               ": expected <name> AS (<query>), with commas between CTEs";
		return result;
	}
	if (claim.keyword > 0) {
		string with_text;
		for (size_t k = 0; k < claim.keyword; k++) {
			if (!with_text.empty()) {
				with_text += " ";
			}
			with_text += tokens[k];
		}
		result.stmt.with_clause = std::move(with_text);
		i = claim.keyword;
	}

	// The statement keyword, in either spelling. The spelling the user wrote
	// is kept for the error messages below.
	if (IsKeywordGluedToParen(tokens[i])) {
		result.error = KeywordSpelling(tokens[i]) + " must be followed by a space, not by '('";
		return result;
	}
	const string keyword = KeywordSpelling(tokens[i]);
	i++;

	// mapping_list := <expr> AS <ident> (',' <expr> AS <ident>)*
	//
	// <expr> may span multiple tokens (e.g. `bill_len * 2`, `log(bill_dep)`).
	// Boundary scan: each mapping consumes tokens until the next top-level `,`,
	// `FROM`, or `DRAW`. The last token in that range is the aesthetic name; the
	// token immediately preceding it must be `AS`; everything earlier is the
	// expression text (concatenated with single spaces — whitespace inside
	// parens is preserved by DuckDB's downstream parser).
	//
	// Limitations (deferred to a paren-aware tokenizer in a later phase):
	//   - top-level commas inside expressions break the scan (`coalesce(a, b)`)
	//   - `FROM` / `DRAW` keywords inside subqueries break the scan
	//   - quoted identifiers `"my col"` are not supported
	while (true) {
		// Scan forward to the next mapping boundary at depth 0.
		size_t boundary = i;
		while (boundary < tokens.size()) {
			const string &t = tokens[boundary];
			if (t == "," || IEqual(t, "FROM") || IEqual(t, "DRAW")) {
				break;
			}
			boundary++;
		}

		if (boundary == i) {
			result.error = "Expected expression after " + keyword + " (or ',')";
			return result;
		}

		// `… AS` with nothing after the AS.
		if (IEqual(tokens[boundary - 1], "AS")) {
			result.error = "Expected aesthetic name after 'AS'";
			return result;
		}

		// Need at least 3 tokens: <something> AS <aesth>. With 2 we either have
		// no AS at all, or a leading AS with empty expression.
		if (boundary - i < 3) {
			if (boundary - i == 2 && IEqual(tokens[i], "AS")) {
				result.error = "Empty expression before 'AS'";
			} else {
				result.error = "Expected 'AS' before aesthetic '" + tokens[boundary - 1] + "'";
			}
			return result;
		}

		size_t aesth_idx = boundary - 1;
		size_t as_idx = boundary - 2;
		if (!IEqual(tokens[as_idx], "AS")) {
			result.error = "Expected 'AS' before aesthetic '" + tokens[aesth_idx] + "'";
			return result;
		}

		string expr;
		for (size_t k = i; k < as_idx; k++) {
			if (!expr.empty()) {
				expr += " ";
			}
			expr += tokens[k];
		}

		// Aesthetic may carry a `:type` suffix (e.g. `color:nominal`) — split it.
		string aesth_token = tokens[aesth_idx];
		size_t colon = aesth_token.find(':');
		string type_override;
		if (colon != string::npos) {
			type_override = aesth_token.substr(colon + 1);
			aesth_token = aesth_token.substr(0, colon);
			if (aesth_token.empty()) {
				result.error = "Empty aesthetic name before ':'";
				return result;
			}
			if (type_override != "quantitative" && type_override != "ordinal" &&
			    type_override != "nominal" && type_override != "temporal") {
				result.error =
				    "Invalid aesthetic type '" + type_override +
				    "' (expected quantitative, ordinal, nominal, or temporal)";
				return result;
			}
		}

		// Upstream's channel names become ours here, so the type override and
		// the layer SQL see the canonical name.
		aesth_token = CanonicalChannel(aesth_token);

		AestheticMapping mapping;
		mapping.expression = std::move(expr);
		mapping.aesthetic = aesth_token;
		result.stmt.aesthetics.push_back(std::move(mapping));
		if (!type_override.empty()) {
			TypeOverride ov;
			ov.aesthetic = aesth_token;
			ov.type = std::move(type_override);
			result.stmt.type_overrides.push_back(std::move(ov));
		}
		i = boundary;

		if (!at_end() && peek() == ",") {
			i++;
			continue;
		}
		break;
	}

	if (!consume("FROM")) {
		return result;
	}
	if (at_end()) {
		result.error = "Expected table name after 'FROM'";
		return result;
	}
	result.stmt.from_table = tokens[i++];

	while (!at_end() && !IEqual(peek(), "FACET") && !IEqual(peek(), "SCALE") &&
	       !IEqual(peek(), "TITLE") && !IEqual(peek(), "SUBTITLE")) {
		if (!consume("DRAW")) {
			return result;
		}
		if (at_end()) {
			result.error = "Expected mark name after 'DRAW'";
			return result;
		}
		DrawLayer layer;
		const string &written = tokens[i++];
		if (IEqual(written, "smooth")) {
			// Upstream's smooth layer is our loess line (#88).
			layer.mark = "line";
			layer.stat = "smooth";
		} else {
			layer.mark = CanonicalMark(written);
		}
		// Optional `STAT <name>` modifier — applies a per-layer transform.
		// Recognized: identity (no-op), smooth (loess on y ~ x), summary
		// (aggregate AVG(y) by x). Unknown stat names are a parse error.
		if (!at_end() && IEqual(peek(), "STAT")) {
			if (!layer.stat.empty()) {
				result.error = "DRAW smooth already applies STAT smooth and does not take a STAT clause";
				return result;
			}
			i++;
			if (at_end()) {
				result.error = "Expected stat name after 'STAT'";
				return result;
			}
			string stat_name = StringUtil::Lower(tokens[i++]);
			if (stat_name != "identity" && stat_name != "smooth" && stat_name != "summary") {
				result.error = "Unknown stat '" + stat_name +
				               "' (expected identity, smooth, or summary)";
				return result;
			}
			layer.stat = std::move(stat_name);
		}
		result.stmt.layers.push_back(std::move(layer));
	}

	if (result.stmt.layers.empty()) {
		result.error = "At least one 'DRAW <mark>' clause is required";
		return result;
	}

	// Optional `FACET BY <row_expr> [, <col_expr>] [ROWS|COLS]`. The 1D form
	// stores a synthetic aesthetic named "facet" and an optional facet_layout
	// ("rows" / "cols" / "" grid). The 2D form (two comma-separated expressions)
	// stores both `facet` (row) and `facet2` (column) and produces a row × column
	// grid; ROWS/COLS is rejected in the 2D form since the layout is fixed.
	// Compile() detects these aesthetics to wrap the spec in a Vega-Lite facet
	// operator and BuildProjectedSql carries them through as projected columns.
	if (!at_end() && IEqual(peek(), "FACET")) {
		i++; // consume FACET (case-insensitive — already verified)
		if (!consume("BY")) {
			return result;
		}
		auto is_facet_terminator = [&]() {
			return at_end() || IEqual(peek(), "SCALE") || IEqual(peek(), "ROWS") ||
			       IEqual(peek(), "COLS") || IEqual(peek(), "TITLE") || IEqual(peek(), "SUBTITLE");
		};
		if (is_facet_terminator() || peek() == ",") {
			result.error = "Expected expression after 'FACET BY'";
			return result;
		}
		string row_expr;
		while (!is_facet_terminator() && peek() != ",") {
			if (!row_expr.empty()) {
				row_expr += " ";
			}
			row_expr += tokens[i++];
		}
		AestheticMapping facet;
		facet.expression = std::move(row_expr);
		facet.aesthetic = "facet";
		result.stmt.aesthetics.push_back(std::move(facet));

		bool two_dim = false;
		if (!at_end() && peek() == ",") {
			i++; // consume ","
			if (is_facet_terminator() || peek() == ",") {
				result.error = "Expected column expression after 'FACET BY <row>,'";
				return result;
			}
			string col_expr;
			while (!is_facet_terminator() && peek() != ",") {
				if (!col_expr.empty()) {
					col_expr += " ";
				}
				col_expr += tokens[i++];
			}
			if (!at_end() && peek() == ",") {
				result.error = "FACET BY accepts at most two expressions (row, column)";
				return result;
			}
			AestheticMapping facet2;
			facet2.expression = std::move(col_expr);
			facet2.aesthetic = "facet2";
			result.stmt.aesthetics.push_back(std::move(facet2));
			two_dim = true;
		}

		// Optional layout for 1D facet: ROWS (vertical), COLS (horizontal), or
		// nothing (grid — Vega-Lite default). The 2D form has a fixed row ×
		// column layout, so ROWS/COLS would be redundant or contradictory.
		if (!at_end() && (IEqual(peek(), "ROWS") || IEqual(peek(), "COLS"))) {
			if (two_dim) {
				result.error =
				    "FACET BY <row>, <col> does not accept ROWS/COLS (always row × column)";
				return result;
			}
			result.stmt.facet_layout = StringUtil::Lower(tokens[i]);
			i++;
		}
	}

	// Optional `SCALE <aesthetic> <op> <args...> [<op> <args...>]...` clauses.
	// Zero or more SCALE clauses; each targets one channel and carries one or
	// more options that merge into a single scale/axis block. Supported ops:
	//   TO <scheme>             — color scheme name
	//   ZERO <true|false>       — include zero in quantitative scale
	//   DOMAIN <num1> <num2>    — explicit numeric domain
	//   LABEL '<text>'          — axis title
	while (!at_end() && IEqual(peek(), "SCALE")) {
		i++;
		if (at_end()) {
			result.error = "Expected aesthetic name after 'SCALE'";
			return result;
		}
		string aesthetic = CanonicalChannel(tokens[i++]);
		// Consume one or more options for this channel until the next clause
		// (SCALE / FACET / TITLE / SUBTITLE / end).
		bool any_op = false;
		while (!at_end() && !IEqual(peek(), "SCALE") && !IEqual(peek(), "FACET") &&
		       !IEqual(peek(), "TITLE") && !IEqual(peek(), "SUBTITLE")) {
			ScaleSpec scale;
			scale.aesthetic = aesthetic;
			string op = tokens[i++];
			if (IEqual(op, "TO")) {
				if (at_end()) {
					result.error = "Expected scheme name after 'SCALE <aesthetic> TO'";
					return result;
				}
				scale.property = "scheme";
				scale.value_json = "\"" + tokens[i++] + "\"";
			} else if (IEqual(op, "ZERO")) {
				if (at_end()) {
					result.error = "Expected boolean (true/false) after 'SCALE <aesthetic> ZERO'";
					return result;
				}
				string v = tokens[i++];
				if (IEqual(v, "true")) {
					scale.value_json = "true";
				} else if (IEqual(v, "false")) {
					scale.value_json = "false";
				} else {
					result.error = "Expected 'true' or 'false' after 'SCALE <aesthetic> ZERO', got '" +
					               v + "'";
					return result;
				}
				scale.property = "zero";
			} else if (IEqual(op, "DOMAIN")) {
				if (at_end()) {
					result.error = "Expected two numeric bounds after 'SCALE <aesthetic> DOMAIN'";
					return result;
				}
				string lo = tokens[i++];
				if (at_end()) {
					result.error = "Expected upper bound after 'SCALE <aesthetic> DOMAIN <lo>'";
					return result;
				}
				string hi = tokens[i++];
				scale.property = "domain";
				scale.value_json = "[" + lo + "," + hi + "]";
			} else if (IEqual(op, "LABEL")) {
				if (at_end()) {
					result.error = "Expected axis label after 'SCALE <aesthetic> LABEL'";
					return result;
				}
				scale.sub_object = "axis";
				scale.property = "title";
				scale.value_json = "\"" + JsonEscape(UnquoteSqlString(tokens[i++])) + "\"";
			} else {
				result.error = "Unknown SCALE operator '" + op +
				               "' (expected TO / ZERO / DOMAIN / LABEL)";
				return result;
			}
			result.stmt.scales.push_back(std::move(scale));
			any_op = true;
		}
		if (!any_op) {
			result.error = "Expected SCALE operator (TO / ZERO / DOMAIN / LABEL) after aesthetic";
			return result;
		}
	}

	// Optional `TITLE '<text>' [SUBTITLE '<text>']`. SUBTITLE without TITLE is
	// rejected — keeps the surface area honest (Vega-Lite ignores a subtitle
	// without a title anyway, so silently accepting it would mask typos).
	if (!at_end() && IEqual(peek(), "TITLE")) {
		i++;
		if (at_end()) {
			result.error = "Expected title text after 'TITLE'";
			return result;
		}
		result.stmt.title = UnquoteSqlString(tokens[i++]);
		if (!at_end() && IEqual(peek(), "SUBTITLE")) {
			i++;
			if (at_end()) {
				result.error = "Expected subtitle text after 'SUBTITLE'";
				return result;
			}
			result.stmt.subtitle = UnquoteSqlString(tokens[i++]);
		}
	} else if (!at_end() && IEqual(peek(), "SUBTITLE")) {
		result.error = "SUBTITLE requires a preceding TITLE clause";
		return result;
	}

	if (!at_end()) {
		result.error = "Unexpected token '" + tokens[i] + "' after " + keyword + " clause";
		return result;
	}

	result.success = true;
	return result;
}

} // namespace ggsql
} // namespace duckdb
