# `VISUALIZE` — grammar-of-graphics charts in SQL

`VISUALIZE` is a SQL dialect (the **ggsql** parser extension shipped in
`stats_duck`) that turns a query into a chart. It does **not** draw anything
itself — it returns a complete [Vega-Lite v5](https://vega.github.io/vega-lite/)
JSON spec plus the SQL needed to feed it. Your client runs the SQL and hands the
rows to `vega-embed`.

```
VISUALIZE <expr> AS <aesthetic> [, …] FROM <table> DRAW <mark> [clauses…]
```

Every `VISUALIZE` returns **one row, two columns**:

| column | type | what it is |
|--------|------|------------|
| `spec` | `VARCHAR` | a full Vega-Lite v5 JSON spec |
| `layer_sqls` | `MAP(VARCHAR, VARCHAR)` | one SQL string per named layer (`layer_0`, `layer_1`, …) |

The client renders it by running each layer's SQL and passing the results to
vega-embed through its `datasets` API (the spec references them by name).

> A statement that doesn't start with `VISUALIZE` (or `WITH … VISUALIZE`) is
> handled by DuckDB's normal SQL parser, untouched.

---

## Setup

Every example below runs against this table:

```sql
CREATE TABLE penguins AS SELECT * FROM (VALUES
  ('Adelie',    'male',   2007, 39.1, 18.7, 181, 3750),
  ('Adelie',    'female', 2007, 39.5, 17.4, 186, 3800),
  ('Adelie',    'female', 2008, 40.3, 18.0, 195, 3250),
  ('Adelie',    'male',   2009, 41.1, 19.0, 192, 4050),
  ('Chinstrap', 'female', 2007, 46.5, 17.9, 192, 3500),
  ('Chinstrap', 'male',   2008, 50.0, 19.5, 196, 3900),
  ('Chinstrap', 'female', 2009, 45.4, 18.7, 188, 3525),
  ('Gentoo',    'male',   2007, 46.1, 13.2, 211, 4500),
  ('Gentoo',    'female', 2008, 46.5, 14.8, 217, 5200),
  ('Gentoo',    'male',   2009, 49.0, 16.3, 220, 5950),
  ('Gentoo',    'female', 2009, 48.7, 14.1, 210, 4875)
) AS t(species, sex, year, bill_len, bill_dep, flipper_len, body_mass);
```

---

## 1. Your first chart

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point;
```

A scatter plot. You map columns (or expressions) to **aesthetics** with `AS`, name
the data with `FROM`, and pick a **mark** with `DRAW`.

---

## 2. Aesthetics

Aesthetics are the visual channels a column drives:

```sql
VISUALIZE bill_len AS x, bill_dep AS y, species AS color FROM penguins DRAW point;
```

The full channel set: `x`, `y`, `color`, `fill`, `stroke`, `shape`, `size`,
`opacity`, `tooltip`, `text`, `x2`, `y2`. Unknown channels are silently dropped.

### Type overrides

ggsql infers each channel's Vega-Lite type (quantitative / nominal / temporal).
Force it by appending `:type` — handy when an integer should be treated as a
category:

```sql
VISUALIZE bill_len AS x, bill_dep AS y, year AS color:ordinal FROM penguins DRAW point;
```

Valid types: `:quantitative`, `:ordinal`, `:nominal`, `:temporal`.

---

## 3. Marks

`DRAW <mark>` chooses the geometry. The 15 built-in marks and the aesthetics they
require:

| mark | requires | use |
|------|----------|-----|
| `point` | x, y | scatter |
| `line` | x, y | trend / series (sorted by x) |
| `area` | x, y | filled line |
| `bar` | x, y | categorical magnitudes |
| `histogram` | x | distribution of one column (auto-binned) |
| `tick` | — | 1-D strip of marks |
| `rule` | — | reference line (map only `x` **or** only `y`) |
| `text` | x, y, text | data labels |
| `boxplot` | x, y | distribution summary per category |
| `violin` | x, y | density per category (vertical violins) |
| `density` | x | smooth KDE curve |
| `errorbar` | x, y, y2 | interval whiskers (y…y2) |
| `errorband` | x, y, y2 | shaded interval band |
| `heatmap` | x, y, color | matrix of a value over two categories |
| `regression` | x, y | server-side linear fit |

### Points, lines, areas

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW line;
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW area;
```

`line` and `area` wrap their layer SQL in `ORDER BY x` so the path is monotone.

### Bars and histograms

```sql
VISUALIZE species AS x, body_mass AS y FROM penguins DRAW bar;
VISUALIZE body_mass AS x FROM penguins DRAW histogram;
```

`histogram` needs only `x`; Vega-Lite bins it.

### Distributions: boxplot, violin, density

```sql
VISUALIZE species AS x, body_mass AS y FROM penguins DRAW boxplot;
VISUALIZE species AS x, body_mass AS y FROM penguins DRAW violin;
VISUALIZE body_mass AS x FROM penguins DRAW density;
```

`violin` lays out one vertical violin per category of `x` using Vega-Lite's
`column` facet (it composes with `FACET BY … ROWS` but conflicts with
`FACET BY … COLS`). `density` runs a KDE on `x`, grouped by `color` if mapped:

```sql
VISUALIZE bill_len AS x, species AS color FROM penguins DRAW density;
```

### Intervals: errorbar, errorband, rule, tick

`errorbar` / `errorband` draw an interval from `y` to `y2`. Build the bounds with
a CTE (see §9):

```sql
WITH s AS (
  SELECT species, min(bill_len) AS lo, max(bill_len) AS hi FROM penguins GROUP BY species
)
VISUALIZE species AS x, lo AS y, hi AS y2 FROM s DRAW errorbar;
```

`rule` is a reference line — map only `x` for a vertical rule, only `y` for a
horizontal one; `tick` is a 1-D strip:

```sql
VISUALIZE 18 AS y FROM penguins DRAW rule;
VISUALIZE bill_len AS x FROM penguins DRAW tick;
```

### Text labels

```sql
VISUALIZE bill_len AS x, bill_dep AS y, species AS text FROM penguins DRAW text;
```

### Heatmap

A `rect` mark with categorical `x`/`y` and quantitative `color` — contingency
tables, correlation matrices:

```sql
VISUALIZE species AS x, sex AS y, body_mass AS color FROM penguins DRAW heatmap;
```

### Regression

A server-side linear fit of `y ~ x` (grouped by `color` if mapped). Overlay it on
a scatter:

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point DRAW regression;
```

---

## 4. Layers

Repeat `DRAW` to stack marks. Each becomes its own Vega-Lite layer with its own
`layer_sqls` entry:

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point DRAW line;
```

---

## 5. Statistical transforms — `STAT`

Append `STAT` to a layer to transform it before drawing:

| `STAT` | effect |
|--------|--------|
| `identity` | no-op (the default) |
| `smooth` | LOESS fit (Vega-Lite `loess`), grouped by `color` |
| `summary` | rewrites the layer SQL to `AVG(y) GROUP BY x [, color, facets]` — collapse to per-group means |

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point DRAW line STAT smooth;
VISUALIZE bill_len AS x, bill_dep AS y, species AS color FROM penguins DRAW line STAT summary;
```

`smooth` is the idiomatic scatter-with-trend overlay (`DRAW point DRAW line STAT smooth`).

---

## 6. Facets — small multiples

Split into panels by a column. `ROWS` / `COLS` pick the direction (default
`COLS`-like single channel):

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point FACET BY species ROWS;
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point FACET BY species COLS;
```

A **2-D facet grid** — rows × columns — by giving two expressions:

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point FACET BY species, sex;
```

---

## 7. Scales & axes — `SCALE`

`SCALE <channel> <option…>` tunes a channel's scale or axis. Options:

| option | effect |
|--------|--------|
| `TO <scheme>` | color scheme (`viridis`, `accent`, …) |
| `ZERO true\|false` | include / exclude zero on a quantitative scale |
| `DOMAIN <lo> <hi>` | fixed numeric domain |
| `LABEL '<text>'` | axis title |

**Stack multiple options** on one `SCALE`, or repeat `SCALE <channel>` — they
merge into one scale/axis block either way:

```sql
VISUALIZE bill_len AS x, bill_dep AS y, species AS color
FROM penguins
DRAW point
SCALE color TO viridis
SCALE x LABEL 'Bill length (mm)' ZERO false
SCALE y ZERO false;
```

`SCALE` on an unmapped channel is a silent no-op.

---

## 8. Titles

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins
DRAW point
TITLE 'Bill dimensions' SUBTITLE 'Source: Palmer Station LTER';
```

Single-quote escapes work the SQL way (`'Bill''s mom'` → `Bill's mom`).

---

## 9. CTEs — `WITH`

A leading `WITH` is supported and scoped to each layer's SQL, so it composes with
wrapping marks (`line`, `area`, `errorband`, `regression`, …):

```sql
WITH heavy AS (SELECT * FROM penguins WHERE body_mass > 4000)
VISUALIZE bill_len AS x, bill_dep AS y, species AS color FROM heavy DRAW point;
```

---

## 10. SQL expressions in mappings

Any aesthetic can be an expression — it rides through to the layer SQL verbatim
(parens, function calls, and subqueries are tokenized correctly):

```sql
VISUALIZE bill_len * 2 AS x, log(body_mass) AS y FROM penguins DRAW point;
VISUALIZE bill_len AS x, (SELECT avg(bill_dep) FROM penguins) AS y FROM penguins DRAW rule;
```

---

## 11. Comments

SQL comments may appear anywhere in a `VISUALIZE` statement — `--` to end of
line, and `/* … */`. They're skipped like whitespace (and never mistaken for
comments inside a string literal):

```sql
VISUALIZE
  bill_len AS x        -- horizontal axis
  , bill_dep AS y      /* vertical axis */
FROM penguins
DRAW point;
```

---

## 12. Custom marks

Each mark is a scalar function named `ggsql_mark_v1_<name>`, discovered through
DuckDB's catalog. Other extensions can register their own marks the same way —
no change to `stats_duck` needed. Introspect a mark's contract:

```sql
SELECT ggsql_mark_v1_violin();   -- {"name":"violin","required_aesthetics":["x","y"]}
```

List everything available:

```sql
SELECT function_name FROM duckdb_functions()
WHERE function_name LIKE 'ggsql_mark_v1_%' ORDER BY 1;
```

---

## Grammar reference

```
[WITH [RECURSIVE] <cte> AS (...) [, <cte> AS (...)]*]
VISUALIZE <expr> AS <aesthetic> [: <type>] (, <expr> AS <aesthetic> ...)
FROM <table>
DRAW <mark> [STAT <identity|smooth|summary>] (DRAW <mark> [STAT ...])*
[FACET BY <expr> [ROWS | COLS] | FACET BY <row_expr>, <col_expr>]
[SCALE <channel> {TO <scheme> | ZERO true|false | DOMAIN <lo> <hi> | LABEL '<text>'}+]*
[TITLE '<text>' [SUBTITLE '<text>']]
```

- **Channels:** `x`, `y`, `color`, `fill`, `stroke`, `shape`, `size`, `opacity`,
  `tooltip`, `text`, `x2`, `y2`.
- **Types:** `:quantitative`, `:ordinal`, `:nominal`, `:temporal`.
- **Marks:** `point`, `line`, `bar`, `histogram`, `text`, `area`, `rule`, `tick`,
  `errorbar`, `errorband`, `boxplot`, `violin`, `heatmap`, `density`, `regression`.
- **Comments:** `--` and `/* … */` anywhere outside string literals.
