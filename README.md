# The Stats Duck

A statistical computing toolkit for DuckDB.

The Stats Duck adds statistical analysis to SQL: descriptive statistics,
hypothesis tests, grammar-of-graphics visualization, and direct readers/writers
for SAS, SPSS, and Stata files. Functions are implemented as streaming
aggregates and scalar primitives, so they scale from local notebooks to
billion-row warehouses and also run inside DuckDB-WASM in the browser.

The Stats Duck is built by [KoliStat](https://kolistat.com/) (see the
[product page](https://kolistat.com/products/the-stats-duck)). It is the
statistical engine of [Bedevere Wise](https://bedeverewise.app/), KoliStat's
browser-based statistical workbench.

> The extension installs and loads in DuckDB under the technical name
> `stats_duck` (matching the binary, the SQL function namespace, and the
> `INSTALL` keyword). "The Stats Duck" is the project / brand name; `stats_duck`
> is what you type at the SQL prompt.

The name is a play on [The Fat Duck](https://thefatduck.co.uk/), Heston
Blumenthal's restaurant in Bray, Berkshire. That restaurant has been the
inspiration for the project name from the start.

## Scope

The Stats Duck is meant to cover the everyday work of a general-purpose
statistician without leaving SQL. The current release covers four areas:

- **Hypothesis tests** — parametric and non-parametric, with effect sizes and
  confidence intervals returned alongside the test statistic.
- **Visualizations (VISUALIZE)** — `VISUALIZE … FROM <table> DRAW <mark>`, a
  Posit-published Grammar-of-Graphics SQL dialect compiled to Vega-Lite v5.
  No server-side rendering: the extension emits a spec + per-layer SQL, and
  the client (browser, notebook, …) runs the SQL and feeds the rows to vega.
- **Statistical file I/O** — readers and writers for SAS, SPSS, and Stata
  files, integrated with DuckDB's virtual file system so they work with local
  paths, `httpfs://`, `s3://`, and registered WASM file buffers.
- **Streaming aggregates** — every test is a single-pass aggregate over the
  data, so it works with `GROUP BY`, window frames, and DuckDB's
  parallel execution.

Future releases will add Spearman/Kendall correlations, regression with full
diagnostics, multiple-testing corrections, and more distribution families.

Further documentation:

- [`docs/visualize.md`](docs/visualize.md) — a worked example of every
  `VISUALIZE` mark and clause.
- [`docs/kernel_api.md`](docs/kernel_api.md) — the DuckDB-free C++ kernel
  and how a sibling repo consumes it.
- [`docs/fitter_conventions.md`](docs/fitter_conventions.md) — conventions
  for model-fitting functions: inputs, the return STRUCT, validation.

## Functions

### Hypothesis tests (aggregate)

| Function                                                              | Description                                  |
| --------------------------------------------------------------------- | -------------------------------------------- |
| `ttest_1samp(column, [mu], [alpha], [alternative])`                   | One-sample t-test                            |
| `ttest_2samp(column1, column2, [equal_var], [alpha], [alternative])`  | Two-sample t-test (Welch's or Student's)     |
| `ttest_paired(column1, column2, [alpha], [alternative])`              | Paired t-test                                |
| `mann_whitney_u(column1, column2, [alternative], [continuity])`       | Mann-Whitney U test (Wilcoxon rank-sum)      |
| `wilcoxon_signed_rank(column1, column2, [alternative], [continuity])` | Wilcoxon signed-rank test                    |
| `pearson_test(x, y, [alpha], [alternative])`                          | Pearson correlation with significance        |
| `spearman_test(x, y, [alpha], [alternative])`                         | Spearman rank correlation                    |
| `kendall_test(x, y, [alternative])`                                   | Kendall's tau-b rank correlation             |
| `anova_oneway(value, group)`                                          | One-way ANOVA                                |
| `chisq_independence(row, col, [continuity])`                          | Chi-square test of independence              |
| `chisq_goodness_of_fit(category)`                                     | Chi-square goodness-of-fit (uniform)         |
| `jarque_bera(column)`                                                 | Jarque-Bera normality test                   |
| `shapiro_wilk(column)`                                                | Shapiro-Wilk normality test (Royston AS R94) |
| `anderson_darling(column)`                                            | Anderson-Darling normality test              |
| `ks_test_1samp(column)`                                               | Kolmogorov-Smirnov one-sample (vs fitted normal) |
| `ks_test_2samp(column1, column2)`                                     | Kolmogorov-Smirnov two-sample                |
| `sign_test_1samp(column, [mu], [alternative])`                        | Sign test on the median                      |
| `sign_test_paired(column1, column2, [alternative])`                   | Paired sign test                             |

All tests return a `STRUCT` with the test statistic, degrees of freedom,
p-value, and relevant effect sizes / confidence intervals.

Every test skips NULL inputs, and treats NaN the same way: a NaN value
counts as missing and never enters the test. For two-column tests a NaN
on either side drops the pair (paired tests) or that single value
(two-sample tests), exactly as a NULL would.

#### Common parameters

| Parameter     | Type      | Default       | Description                                                      |
| ------------- | --------- | ------------- | ---------------------------------------------------------------- |
| `mu`          | `DOUBLE`  | `0.0`         | Hypothesized population mean (one-sample t-test only)            |
| `equal_var`   | `BOOLEAN` | `false`       | Assume equal variances — Student's pooled test (two-sample only) |
| `alpha`       | `DOUBLE`  | `0.05`        | Significance level for confidence intervals (t-tests only)       |
| `alternative` | `VARCHAR` | `'two-sided'` | `'two-sided'`, `'less'`, or `'greater'`                          |
| `continuity`  | `BOOLEAN` | `false`       | Apply continuity correction (mann_whitney_u, chisq_independence) |

#### Result struct fields

**t-test:** `test_type`, `t_statistic`, `df`, `p_value`, `alternative`, `mean_diff`, `ci_lower`, `ci_upper`, `cohens_d`

**Mann-Whitney:** `test_type`, `u_statistic`, `z_statistic`, `p_value`, `alternative`, `rank_biserial`

**Wilcoxon:** `test_type`, `w_statistic`, `z_statistic`, `p_value`, `alternative`, `effect_size_r`

**Pearson:** `test_type`, `r`, `t_statistic`, `df`, `p_value`, `alternative`, `ci_lower`, `ci_upper`, `n`

**Spearman:** `test_type`, `rho`, `t_statistic`, `df`, `p_value`, `alternative`, `ci_lower`, `ci_upper`, `n`

**Kendall:** `test_type`, `tau`, `z_statistic`, `p_value`, `alternative`, `n`

**ANOVA:** `test_type`, `f_statistic`, `df_between`, `df_within`, `p_value`, `ss_between`, `ss_within`, `eta_squared`, `n_groups`, `n`

**Chi-square:** `test_type`, `chi_square`, `df`, `p_value`, `n`, `n_rows`/`n_cols` or `n_categories`

**Jarque-Bera:** `test_type`, `jb_statistic`, `skewness`, `excess_kurtosis`, `df`, `p_value`, `n`

**Anderson-Darling:** `test_type`, `a_squared`, `a_squared_adjusted`, `p_value`, `n`

**Shapiro-Wilk:** `test_type`, `w_statistic`, `p_value`, `n`

**KS one-sample:** `test_type`, `d_statistic`, `p_value`, `n`

**KS two-sample:** `test_type`, `d_statistic`, `p_value`, `n_x`, `n_y`

**Sign test:** `test_type`, `m_statistic`, `n_pos`, `n_neg`, `n_zero`, `p_value`, `alternative`, `n`

### Descriptive statistics (aggregate)

| Function                                                    | Description                                                                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `summary_stats(column, [bias_correction], [quantile_type])` | n, n_missing, mean, sd, variance, min, q1, median, q3, max, iqr, skewness, kurtosis, mode, mode_frequency, is_multimodal |

`bias_correction` (BOOLEAN, default `true`) toggles the skewness/kurtosis
formulas. With `true` the output matches SAS PROC MEANS, scipy with
`bias=False`, and Excel SKEW/KURT. With `false` the population formulas
`m3/m2^1.5` and `m4/m2² - 3` are used, matching R's default. Mean, SD,
variance, quantiles, and IQR are unaffected.

`mode` is the smallest modal value (when more than one value shares the
maximum frequency, the smallest of them is returned and `is_multimodal`
is set to `true`). For all-distinct input — every value appears exactly
once — `mode` is `NaN` and `mode_frequency` is `0`, matching SAS PROC
UNIVARIATE's "Mode ." output.

`quantile_type` (INTEGER, default `7`) picks the Hyndman & Fan (1996)
quantile algorithm used for `q1`, `median`, and `q3`. Supported values:

- `7` — R / Excel INC default. `position = 1 + q * (n - 1)`.
- `5` — SAS PROC UNIVARIATE default. `position = q * n + 0.5`.

For `x = [1, 2, 3, 4]`: type 7 gives Q1=1.75 / Q3=3.25 (matching R); type 5
gives Q1=1.5 / Q3=3.5 (matching SAS PROC MEANS).

### Distribution functions (scalar)

| Function                 | Description             |
| ------------------------ | ----------------------- |
| `dnorm(x, [mean], [sd])` | Normal PDF              |
| `pnorm(x, [mean], [sd])` | Normal CDF              |
| `qnorm(p, [mean], [sd])` | Normal quantile         |
| `dt(x, df)`              | Student's t PDF         |
| `pt(x, df)`              | Student's t CDF         |
| `qt(p, df)`              | Student's t quantile    |
| `dchisq(x, df)`          | Chi-square PDF          |
| `pchisq(x, df)`          | Chi-square CDF          |
| `qchisq(p, df)`          | Chi-square quantile     |
| `df(x, df1, df2)`        | F distribution PDF      |
| `pf(x, df1, df2)`        | F distribution CDF      |
| `qf(p, df1, df2)`        | F distribution quantile |
| `dgamma(x, shape, [rate])` | Gamma PDF (rate=1 default) |
| `pgamma(x, shape, [rate])` | Gamma CDF             |
| `qgamma(p, shape, [rate])` | Gamma quantile        |
| `dbeta(x, alpha, beta)`  | Beta PDF on [0, 1]      |
| `pbeta(x, alpha, beta)`  | Beta CDF                |
| `qbeta(p, alpha, beta)`  | Beta quantile           |
| `dexp(x, [rate])`        | Exponential PDF (rate=1 default) |
| `pexp(x, [rate])`        | Exponential CDF         |
| `qexp(p, [rate])`        | Exponential quantile (closed form) |
| `dweibull(x, shape, [scale])` | Weibull PDF (scale=1 default) |
| `pweibull(x, shape, [scale])` | Weibull CDF |
| `qweibull(p, shape, [scale])` | Weibull quantile (closed form) |
| `dlnorm(x, [meanlog], [sdlog])` | Log-normal PDF (meanlog=0, sdlog=1 defaults) |
| `plnorm(x, [meanlog], [sdlog])` | Log-normal CDF |
| `qlnorm(p, [meanlog], [sdlog])` | Log-normal quantile |
| `dpois(k, lambda)`       | Poisson PMF (discrete) |
| `ppois(q, lambda)`       | Poisson CDF |
| `qpois(p, lambda)`       | Poisson quantile (integer search) |
| `dnbinom(k, size, prob)` | Negative binomial PMF (count of failures before `size` successes; `prob` per-trial) |
| `pnbinom(q, size, prob)` | Negative binomial CDF — closed form via regularized incomplete beta |
| `qnbinom(p, size, prob)` | Negative binomial quantile (integer search) |
| `dhyper(x, m, n, k)`     | Hypergeometric PMF (`m` successes / `n` failures / `k` draws without replacement) |
| `phyper(q, m, n, k)`     | Hypergeometric CDF (direct PMF sum; tractable for typical population sizes) |
| `qhyper(p, m, n, k)`     | Hypergeometric quantile |
| `ptukey(q, k, df)`       | Studentized range CDF for `k` means on `df` degrees of freedom — the distribution behind `emmeans`'s Tukey adjustment |
| `qtukey(p, k, df)`       | Studentized range quantile — the Tukey HSD critical value |
| `rnorm([mean], [sd])`    | Random sample from a normal — **volatile, per-row** |
| `rt(df)` / `rchisq(df)` / `rf(df1, df2)` | Student-t / χ² / F samples |
| `rgamma(shape, [rate])` / `rbeta(alpha, beta)` / `rexp([rate])` | Gamma / Beta / Exponential samples |
| `rweibull(shape, [scale])` / `rlnorm([meanlog], [sdlog])` / `rpois(lambda)` | Weibull / Log-normal / Poisson samples |
| `rnbinom(size, prob)` / `rhyper(m, n, k)` | Negative binomial / Hypergeometric samples |
| `poibin_cdf(probs LIST<DOUBLE>, k BIGINT)` | Poisson Binomial CDF — `P(X ≤ k)` for `X = Σᵢ Bᵢ`, `Bᵢ ∼ Bernoulli(pᵢ)` |
| `bin_edges(x [, method])` *(aggregate)* | Auto bin-edge vector for `x` — `sturges` (default), `fd`, `scott`, `sqrt`, `rice`, `auto` |
| `bin_label(x, edges)` | Label for the bin containing `x` given an edge vector (typically from `bin_edges`) |
| `bootstrap(x, statistic, n_iters [, seed])` *(aggregate)* | With-replacement resampling — emits `LIST<DOUBLE>` of length `n_iters`. `statistic` ∈ `{mean, median, sum, stddev, variance, min, max}` |

### Dataset profile (table function)

| Function       | Description                                              |
| -------------- | -------------------------------------------------------- |
| `meta(data)`   | One row per column with kind classification + light stats |

```sql
SELECT * FROM meta('penguins');
```

Output columns (fixed schema):
`column_name`, `column_type`, `kind`, `n_rows`, `n_missing`, `n_distinct`,
`min`, `p25`, `median`, `p75`, `max`, `mean`, `stddev`, `top`, `top_freq`.

- `kind` is a semantic classification —
  `numeric` / `categorical` / `temporal` / `boolean` / `other` — derived
  from the catalog type. It controls which distribution columns are
  populated: numeric kinds fill `min` / `p25` / `median` / `p75` / `max` /
  `mean` / `stddev` (cast to DOUBLE; `quantile_cont` type 7, sample
  stddev), and `categorical` / `boolean` fill `top` (the mode, ties broken
  by smaller value) and `top_freq` (its count).
- `n_rows` is the source table's row count, repeated on every row so any
  single row is self-contained.
- Dataset-level summaries fall out via aggregation:

  ```sql
  SELECT count(*) FILTER (WHERE kind = 'numeric')     AS n_numeric,
         count(*) FILTER (WHERE kind = 'categorical') AS n_categorical,
         sum(n_missing)                               AS total_missing,
         count(*) FILTER (WHERE n_missing > 0)        AS cols_with_nulls
  FROM meta('penguins');
  ```

- Overlaps DuckDB's built-in `SUMMARIZE` but is a table function
  (joinable, filterable, composable in CTEs), adds the kind classifier,
  and reports a mode for categorical / boolean columns. Per-column
  detail (skewness, kurtosis, custom quantile types, bias-corrected
  variants) lives in `summary_stats(column)`.

### Table 1 summary (table function)

| Function                                          | Description                                                |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `table_one(data, variables [, by])`               | Long-format descriptives table for mixed variable types    |
| `corr_matrix(data, variables [, method])`         | Long-format pairwise correlation matrix (`pearson` / `spearman` / `kendall`) |
| `lm(data, y, x)` / `lm_summary(data, y, x)`       | OLS regression — `lm` returns per-term coefficients, `lm_summary` returns model R² / F / σ |
| `lm_fit(y, x [, vcov [, cluster] [, add_intercept]])` | OLS regression **aggregate** (one model per `GROUP BY`) with classical, HC0–HC3, or CR0/CR1 cluster-robust standard errors |
| `glm_fit(y, x [, family [, options]])`            | Generalized linear model **aggregate** (one model per `GROUP BY`), fit by iteratively reweighted least squares — gaussian, binomial, poisson, or gamma, with classical or robust covariance |
| `lin_hyp(L, beta, cov, sigma2, df)`               | General linear hypothesis L·β = 0 against a fit — **scalar**; per-constraint estimates with standard errors, plus the joint sum of squares, F and chi-square |
| `emmeans(L, beta, cov, df [, alpha [, adjust]])`  | Estimated marginal means for a fit — **scalar**; each mean with its standard error and confidence limits, every pairwise difference, and a multiplicity adjustment including Tukey |

```sql
SELECT * FROM table_one(
    'patients',
    variables := ['age', 'sex', 'bmi'],
    by := ['arm']         -- optional; pass multiple columns for cross-stratification
);
```

Output columns (long format, fixed schema):
`variable`, `level`, `statistic`, `stratum`, `display`, `p_value`, `effect_size`

- Each numeric variable yields rows for `n`, `missing`, `mean (sd)`,
  `median [q1, q3]`, `min, max` — `level` is NULL.
- Each categorical variable yields one row per level with `n (%)` plus a
  trailing `Missing` level row that is always emitted (filter with
  `WHERE level <> 'Missing'` if you don't want it). All level percentages
  share the stratum-total denominator so they sum to 100%.
- `stratum` is `'Overall'` when `by` is unset / empty; otherwise the Cartesian
  product of distinct value tuples across the listed by-columns, labelled
  by joining values with `' / '` in declared order (e.g. `'Adelie / female'`
  for `by := ['species', 'sex']`). Rows where any by-column is NULL are
  excluded from the stratum breakdown.
- `p_value` is the between-group test result, repeated on every row of the
  same variable so a PIVOT can grab it with `FIRST(p_value)`. NULL when
  `by` is unset or has only one stratum. Numeric variables use one-way
  ANOVA (`anova_oneway`); categorical variables use chi-square independence
  (`chisq_independence`). NULL when the underlying test is infeasible (zero
  variance, too few samples).
- `effect_size` is the matching magnitude — **η² (eta-squared)** for
  numeric variables (from ANOVA's `ss_between / ss_total`), **Cramér's V**
  for categorical (`√(χ² / (n · (min(rows, cols) - 1)))`). Both are in
  [0, 1] and larger means stronger association, so a single uniform
  column name works across kinds. Same repetition and NULL handling as
  `p_value`.
- Variable types are auto-classified from the catalog: integer / floating-
  point types are numeric, everything else (VARCHAR, BOOLEAN, ENUM,
  date/time) is categorical. Override per-variable with
  `force_categorical := ['stage']` (integer column that's really a
  category) or `force_numerical := ['height']` (VARCHAR column holding
  numeric strings). Entries must appear in `variables`, and the two lists
  must not overlap.

Pivot to wide for display:

```sql
PIVOT table_one('patients', variables := ['age', 'sex'], by := ['arm'])
    ON stratum USING first(display)
    GROUP BY variable, level, statistic;
```

### Linear regression (table function)

```sql
SELECT * FROM lm('mtcars', y := 'mpg', x := ['wt', 'hp']);
--   term        estimate  std_error  t_statistic  p_value
--   (Intercept) 37.2273   1.5988     23.285       2.57e-20
--   wt          -3.8778   0.6327     -6.129       1.12e-06
--   hp          -0.0318   0.0090     -3.519       1.45e-03

SELECT * FROM lm_summary('mtcars', y := 'mpg', x := ['wt', 'hp']);
--   r_squared  adj_r_squared  f_statistic  f_p_value  df_model  df_residual  sigma  n
--   0.8268     0.8148         69.21        9.11e-12   2         29           2.593  32
```

A `formula` named parameter accepts an R-style spec as an alternative to the
explicit `y` / `x` form:

```sql
SELECT * FROM lm('mtcars', formula := 'mpg ~ wt + hp');
SELECT * FROM lm('mtcars', formula := 'mpg ~ wt + hp - 1');  -- no intercept
SELECT * FROM lm('weird_names', formula := '"My Y" ~ "x.with.dots"');
```

The formula grammar supports additive predictors, `- 1` or `+ 0` to drop the
intercept, bare and `"..."`-quoted identifiers, and free whitespace.
Interactions (`x1:x2`), wildcards (`*`, `^`, `.`) and inline expressions
(`I(x^2)`, `log(x)`) are not supported in v0.6 — wrap into a CTE if you need
transformed columns. `formula` and `y` / `x` are mutually exclusive.

Both functions run on the same OLS kernel as `lm_fit`, so the three cannot
disagree. Rows with NULL in `y` or any `x` are dropped (complete-case). Term
order follows the user-supplied predictor order, after the intercept. Calling
`lm` and `lm_summary` with the same arguments fits the model twice — use a CTE
if you need both shapes from a single fit. Collinear predictors do not fail:
the later column of a dependent set is dropped and reported with a NULL
estimate, standard error, t and p, so `df_residual` is `n − rank` and
`df_model` counts the slopes actually estimated. Insufficient rows
(`n ≤ rank`) is still an error. When the intercept is removed, R²/adj-R² use
the uncentered TSS = Σ y² to match R's `summary.lm` — interpret with care.

### Linear regression aggregate with robust SEs (`lm_fit`)

`lm_fit` is the **aggregate** companion to `lm`: it fits OLS over the rows of a
group, so a single `GROUP BY` returns one regression per key. Where `lm` takes
column *names* (and labels its terms), the aggregate takes the design-matrix row
as a `LIST(DOUBLE)` of predictor *values* — so coefficients come back **by
position**. The intercept is prepended automatically.

```sql
-- example data: y ~ x1 over 8 rows in a table `points(y, x1)`
SELECT (u).term, (u).estimate, (u).std_error
FROM (SELECT unnest((lm_fit(y, [x1])).coefficients) AS u FROM points);
--   term         estimate  std_error
--   (Intercept)  0.0357    0.1404
--   x1           1.9976    0.0278
```

The headline is **heteroskedasticity-consistent standard errors**. A third,
constant argument selects the covariance estimator — `'const'` (default,
classical), `'HC0'` (Eicker–Huber–White), `'HC1'` (the Stata `,robust`
default, `× n/(n−k)`), `'HC2'`, or `'HC3'` (the recommended small-sample
default). Only the `std_error` / `t_statistic` / `p_value` change; the point
estimates do not:

```sql
SELECT (u).term, (u).std_error
FROM (SELECT unnest((lm_fit(y, [x1], 'HC3')).coefficients) AS u FROM points);
--   term         std_error
--   (Intercept)  0.1313
--   x1           0.0282
```

The killer query is a regression *per group*, computed where the data lives:

```sql
-- a separate fit for every cylinder count, robust SEs, in one pass
SELECT cyl, lm_fit(mpg, [wt, hp], 'HC1') AS model
FROM mtcars GROUP BY cyl;
```

**Cluster-robust standard errors** (`'CR0'`, or `'CR1'` — the Stata
`vce(cluster)` / statsmodels `cov_type='cluster'` default) account for
within-cluster correlation. Unlike `vcov`, the cluster key is a real **per-row
column** (not a constant); pass it as `VARCHAR` and cast a non-text key with
`::VARCHAR`:

```sql
-- SEs clustered by firm; 'CR1' applies the [G/(G−1)]·[(N−1)/(N−k)] correction
SELECT (u).term, (u).estimate, (u).std_error
FROM (SELECT unnest((lm_fit(ret, [mktrf, smb], 'CR1', firm_id::VARCHAR)).coefficients) AS u
      FROM panel);
```

Cluster-robust inference uses a `t(G − 1)` reference (G = number of clusters,
surfaced as `n_clusters`), so it differs from the `t(n − rank)` used by
classical / HC. `'cluster'` is accepted as an alias for `'CR1'`.

`lm_fit` returns a single `STRUCT`:

| Field                                                       | Type                | Notes |
| ----------------------------------------------------------- | ------------------- | ----- |
| `coefficients`                                              | `LIST<STRUCT>`      | one element per term: `term`, `estimate`, `std_error`, `t_statistic`, `p_value` |
| `n`, `k`, `df_residual`                                     | `BIGINT`            | rows used, design columns (incl. intercept), `n − rank` |
| `r_squared`, `adj_r_squared`, `sigma`                       | `DOUBLE`            | classical model fit |
| `f_statistic`, `f_p_value`                                  | `DOUBLE`            | classical overall-significance F (not robustified) |
| `has_intercept`                                             | `BOOLEAN`           | |
| `vcov_type`                                                 | `VARCHAR`           | the estimator actually used |
| `n_clusters`                                                | `BIGINT`            | number of clusters G (CR0/CR1 only; NULL otherwise) |
| `rank`                                                      | `BIGINT`            | parameters actually estimated; below `k` when dependent columns were dropped |
| `loglik`                                                    | `DOUBLE`            | Gaussian log-likelihood at the fit (R `logLik`, statsmodels `.llf`); `+inf` for an exact fit |
| `cov`                                                       | `LIST<DOUBLE>`      | full k×k coefficient covariance for the chosen `vcov`, row-major in coefficient order; SQL lists are 1-based, so `cov[i*k + j + 1]` is `cov(βᵢ, βⱼ)` and the diagonal is `std_error²`; NULL wherever an aliased term is involved |

A trailing constant `add_intercept := false` (positionally
`lm_fit(y, x, 'const', false)`, or `lm_fit(y, x, 'CR1', cluster, false)` when
clustered) drops the constant term — note aggregates take **positional**
constants, not the `name := value` form `lm` uses. Rows with a NULL `y`, any NULL
list element, or (when clustered) a NULL cluster key are dropped (complete-case).
A group with too few rows (`n ≤ rank`), no independent column at all, or — for
CR0/CR1 — fewer than two clusters yields a **NULL** result for that group
rather than aborting the query. `t`/`p` use the t(n−rank) distribution for
classical/HC and t(G−1) for CR0/CR1 (matching Stata / statsmodels `use_t`).

**Collinear predictors fit rather than fail.** A dependent column is dropped
the way R drops it: selection runs left to right, so the earlier column of a
dependent set is kept and the later one becomes an aliased term. The aliased
term keeps its name in `coefficients` but its `estimate`, `std_error`,
`t_statistic` and `p_value` are NULL, as are the `cov` entries in its row and
column — R's "not defined because of singularities". `rank` reports how many
parameters were estimated, and every degrees-of-freedom formula uses it
(σ², adjusted R², the HC1 and CR1 finite-sample factors, the t reference, the
F numerator), so a full-rank fit is unaffected:

```sql
-- x3 is x1 + x2, so it is dropped: rank 3 of k 4, df_residual = n − 3
SELECT (u).term, (u).estimate
FROM (SELECT unnest((lm_fit(y, [x1, x2, x1 + x2])).coefficients) AS u FROM t);
--   (Intercept)  10.1206
--   x1            2.0579
--   x2           -0.9787
--   x3            NULL     -- aliased
``` All numerics run on the
shared header-only linear-algebra kernel (`(X'X)⁻¹`, the robust sandwich),
validated against statsmodels — see `test/cpp/test_lm_fit.cpp`. The bias-reduced
CR2/CR3 cluster estimators are a planned follow-up.

### Generalized linear models (`glm_fit`)

```sql
glm_fit(y, x [, family [, options]])
```

`glm_fit` is the iterative sibling of `lm_fit`: an **aggregate**, one model
per `GROUP BY` group, fit by iteratively reweighted least squares (IRLS).
`family` defaults to `'gaussian'`, which makes `glm_fit(y, [x1, x2])` the
same fit as `lm_fit(y, [x1, x2])` — concretely, not just in name:

```sql
SELECT printf('%.10f', abs((g).coefficients[2].estimate - (l).coefficients[2].estimate))
FROM (SELECT glm_fit(y, [x1, x2]) AS g, lm_fit(y, [x1, x2]) AS l FROM ds5);
--   0.0000000000
```

| Family     | Canonical link | Also supported             | Dispersion |
| ---------- | --------------- | --------------------------- | ---------- |
| `gaussian` | `identity`      | `log`, `inverse`            | estimated  |
| `binomial` | `logit`         | `probit`, `cloglog`, `log`  | fixed at 1 |
| `poisson`  | `log`           | `identity`, `sqrt`          | fixed at 1 |
| `gamma`    | `inverse`       | `log`, `identity`           | estimated  |

**Options go in one trailing STRUCT, read by field name**, because
`glm_fit` has more optional inputs than a position could ever
disambiguate — in particular `weights` and `offset`, which are both
per-row `DOUBLE` columns. Field order does not matter:

```sql
-- these two calls give the same fit
glm_fit(y, [x1, x2], 'binomial', {'link': 'probit', 'vcov': 'HC1'})
glm_fit(y, [x1, x2], 'binomial', {'vcov': 'HC1', 'link': 'probit'})
```

| Field       | Type      | Per-row | Default                    | Notes |
| ----------- | --------- | ------- | --------------------------- | ----- |
| `link`      | `VARCHAR` | no      | the family's canonical link | |
| `vcov`      | `VARCHAR` | no      | `'const'`                   | exactly what `lm_fit` accepts: `const`, `HC0`–`HC3`, `CR0`/`CR1` (`'cluster'` an alias for `CR1`) |
| `intercept` | `BOOLEAN` | no      | `true`                      | prepends a constant column |
| `weights`   | `DOUBLE`  | yes     | 1                            | prior weights; binomial trial counts |
| `offset`    | `DOUBLE`  | yes     | 0                            | added to the linear predictor with a fixed coefficient of 1 |
| `cluster`   | `VARCHAR` | yes     | none                         | required by `CR0`/`CR1`, rejected otherwise |

An unrecognized field name is a bind error listing those six names.
Omitting the struct entirely is the same as passing an empty one.

A worked example per family, starting with gaussian:

```sql
SELECT (u).term, (u).estimate, (u).std_error
FROM (SELECT unnest((glm_fit(y, [x1, x2])).coefficients) AS u FROM ds5);
--   term         estimate  std_error
--   (Intercept)  10.1206   2.0741
--   x1            2.0579   0.1303
--   x2           -0.9787   0.3181
```

The binomial response is a **proportion**, not a count, and `weights` is
the trial count: 7 successes out of 10 trials is `y = 0.7`,
`weights = 10`. Plain 0/1 data is the case where every weight is 1. A
count of successes passed as `y` fails the `[0, 1]` domain check instead
of silently being read as a proportion. The ten rows below are the same
data as forty individual 0/1 rows grouped into cells, and the two ways of
writing it give the same coefficients and the same standard errors:

```sql
-- 10 (x1, x2) cells, 4 trials each
CREATE TABLE ds2 AS SELECT * FROM (VALUES
  (0.25, 1.0, 0.0, 4.0), (0.5, 1.0, 1.0, 4.0), (0.25, 2.0, 0.0, 4.0),
  (0.5, 2.0, 1.0, 4.0), (0.5, 3.0, 0.0, 4.0), (0.5, 3.0, 1.0, 4.0),
  (0.5, 4.0, 0.0, 4.0), (0.75, 4.0, 1.0, 4.0), (0.5, 5.0, 0.0, 4.0),
  (0.75, 5.0, 1.0, 4.0)
) AS t(y, x1, x2, n_trials);

SELECT (u).term, (u).estimate, (u).std_error
FROM (SELECT unnest((glm_fit(y, [x1, x2], 'binomial', {'weights': n_trials})).coefficients) AS u FROM ds2);
--   term         estimate  std_error
--   (Intercept)  -1.3913   0.8801
--   x1            0.3218   0.2385
--   x2            0.8520   0.6640
```

Poisson with an offset for exposure (`ln()`, not DuckDB's base-10 `log()`):

```sql
CREATE TABLE ds3 AS SELECT * FROM (VALUES
  (16.0, 0.0, 10.0), (29.0, 0.2, 20.0), (15.0, 0.4, 15.0), (10.0, 0.6, 5.0),
  (59.0, 0.8, 30.0), (39.0, 1.0, 25.0), (30.0, 1.2, 10.0), (89.0, 1.4, 40.0),
  (25.0, 1.6, 12.0), (60.0, 1.8, 18.0), (69.0, 2.0, 22.0), (21.0, 2.2, 8.0),
  (52.0, 2.4, 14.0), (53.0, 2.6, 16.0), (91.0, 2.8, 19.0)
) AS t(y, x1, exposure);

SELECT (u).term, (u).estimate, (u).std_error
FROM (SELECT unnest((glm_fit(y, [x1], 'poisson', {'offset': ln(exposure)})).coefficients) AS u FROM ds3);
--   term         estimate  std_error
--   (Intercept)  0.2311    0.0917
--   x1           0.4398    0.0490
```

Gamma: its own canonical link is `inverse`, so `log` has to be asked for:

```sql
CREATE TABLE ds4 AS SELECT * FROM (VALUES
  (1.2169, 0.0), (0.5808, 0.14285714285714285), (1.4591, 0.2857142857142857),
  (2.8649, 0.42857142857142855), (2.3548, 0.5714285714285714),
  (1.4105, 0.7142857142857142), (2.4365, 0.8571428571428571), (1.2456, 1.0),
  (0.9774, 1.1428571428571428), (1.6468, 1.2857142857142856),
  (2.7775, 1.4285714285714284), (1.301, 1.5714285714285714),
  (3.1178, 1.7142857142857142), (3.5281, 1.857142857142857), (3.2777, 2.0)
) AS t(y, x1);

SELECT (u).term, (u).estimate, (u).std_error
FROM (SELECT unnest((glm_fit(y, [x1], 'gamma', {'link': 'log'})).coefficients) AS u FROM ds4);
--   term         estimate  std_error
--   (Intercept)  0.2496    0.2057
--   x1           0.4161    0.1751
```

`glm_fit` returns a single `STRUCT`:

| Field                       | Type            | Notes |
| ---------------------------- | --------------- | ----- |
| `coefficients`               | `LIST<STRUCT>`  | one element per term: `term`, `estimate`, `std_error`, `z_statistic`, `p_value` |
| `n`, `k`, `df_residual`      | `BIGINT`        | rows used, design columns (intercept included), `n − rank` |
| `deviance`, `null_deviance`  | `DOUBLE`        | the fitted model's deviance, and the intercept-only model's. `null_deviance` is NULL when the intercept-only fit fails on its own, which a successful main fit does not rule out |
| `dispersion`                 | `DOUBLE`        | fixed at 1 for binomial and poisson; the Pearson estimate for gaussian and gamma |
| `loglik`, `aic`              | `DOUBLE`        | |
| `iterations`                 | `BIGINT`        | outer IRLS passes used; never the step-halvings inside one pass |
| `converged`                  | `BOOLEAN`       | |
| `separation`                 | `BOOLEAN`       | binomial only; always `false` for the other three families — see below |
| `family`, `link`             | `VARCHAR`       | echoed canonically, not as the caller spelled them |
| `has_intercept`              | `BOOLEAN`       | |
| `vcov_type`                  | `VARCHAR`       | the estimator actually used |
| `n_clusters`                 | `BIGINT`        | NULL unless `CR0`/`CR1` |
| `rank`                       | `BIGINT`        | parameters actually estimated |
| `cov`                        | `LIST<DOUBLE>`  | row-major k², in `coefficients` order |

A few things worth stating plainly:

- **`z_statistic` uses a normal reference for all four families, never a t
  one.** R and statsmodels switch to a t reference for gaussian and gamma,
  where the dispersion is estimated, so their p-values there are very
  slightly larger than this function's — ours are very slightly smaller.
  A caller who wants the t reference has `df_residual` and can compute it.
- **`converged = false` returns the best iterate, not NULL.** A fit that
  used every allowed iteration, or hit a step it could not recover from,
  still reports its coefficients, its standard errors, and every other
  field in full. `iterations` counts outer IRLS passes; it never counts
  the step-halvings inside one pass.
- **`deviance` and `null_deviance` stand in for `r_squared` and
  `f_statistic`.** Neither of those generalizes past OLS. A reader who
  wants a pseudo-R² can compute `1 - deviance / null_deviance` directly.
  Guard that expression against a NULL `null_deviance`: it is computed by
  actually fitting the intercept-only model, that fit can fail on its own,
  and the main fit still succeeds and reports everything else. The #22
  whole-branch review measured the frequency over random designs: 0 of
  79,965 canonical-link fits with unit weights and no offset, and 868 of
  63,121 fits with a random offset — 1.4%, mostly gamma with `inverse`.

**The separation flag has a blind spot in both directions.** `separation`
is computed only for binomial fits — always `false` for the other three
families. It is `true` when the fit did **not** converge and at least one
fitted value has landed within `1e-10` of 0 or 1. Perfect separation makes
a logistic coefficient diverge: every step still lowers the deviance by
pushing the coefficient further out, so the loop never converges, and a
caller looking only at `estimate` and `std_error` sees one huge number
beside another with nothing to say why. `separation = true` is evidence of
that. But the flag can also fire on severe near-separation, which it
cannot tell apart from the perfect case, and it needs a non-converged fit
to fire at all. A response with no variation at all — every `y` the same,
which a `GROUP BY` over a rare outcome produces routinely — typically
*converges* well before the iteration limit, because IRLS tests the
deviance rather than the coefficients, and the deviance flattens near the
boundary long before the coefficients do. Measured on an all-zero binomial
response: the fit converges at iteration 23 with an intercept standard
error around 40311, and `separation` stays `false` throughout. So
`separation = true` is evidence of trouble, but `separation = false` is
**not** evidence of a sound fit. A caller who wants that assurance has to
look at the magnitude of `std_error` too.

**A fitted value outside the family's support is a failed fit, not a
computed one.** Poisson's `identity` link and binomial's `log` link can
both reach an impossible fitted value that their canonical-link siblings
(`log` and `logit`) cannot — a negative expected count, or a probability
above 1. When that happens partway through fitting, the group comes back
NULL instead of a value computed at an impossible mean, which is correct,
because any number computed there would be wrong:

```sql
SELECT (a).n IS NULL, (b).n IS NULL
FROM (SELECT glm_fit(y, [x1], 'poisson', {'link': 'identity'}) AS a,
             glm_fit(y, [x1], 'poisson')                       AS b
      FROM (VALUES
        (2.0, -1.387), (0.0, 4.348), (1.0, 2.565), (0.0, 0.52),
        (5.0, -0.029), (0.0, 2.755), (3.0, 2.156), (5.0, -4.418)) t(y, x1));
--   true  false
```

This is a real behavioral difference from the canonical links, not a bug:
the same data fits cleanly under `log`, and comes back NULL under
`identity`, because only the second one can reach a negative fitted count.

**The robust covariance estimators use score residuals, not response
residuals**, so the formula differs from `lm_fit`'s in general. At
gaussian with an identity link the two formulas reduce to the same number
exactly, because the link's derivative and the family's variance function
are both 1 there — which is why the `lm_fit` equivalence shown earlier
holds for every covariance estimator, not only the classical one. The
tests assert that reduction bit for bit:

```sql
SELECT printf('%.10f', abs((g).coefficients[2].std_error - (l).coefficients[2].std_error))
FROM (SELECT glm_fit(y, [x1, x2], 'gaussian', {'vcov': 'HC3'}) AS g,
             lm_fit(y, [x1, x2], 'HC3')                        AS l
      FROM ds5);
--   0.0000000000
```

**`lin_hyp` works against a `glm_fit` output unchanged.** It takes only
coefficients and a covariance, so it does not care what produced them:

```sql
CREATE TABLE ds_bin AS SELECT * FROM (VALUES
  (0.0, 1.0), (0.0, 2.0), (1.0, 3.0), (0.0, 4.0), (1.0, 5.0), (0.0, 6.0),
  (1.0, 7.0), (1.0, 8.0), (0.0, 9.0), (1.0, 10.0), (1.0, 11.0), (1.0, 12.0)
) AS t(y, x1);

SELECT (h).df_num, printf('%.4f', (h).chisq)
FROM (SELECT lin_hyp([[0, 1]],
                     list_transform(g.coefficients, c -> c.estimate),
                     g.cov, g.dispersion, g.df_residual) AS h
      FROM (SELECT glm_fit(y, [x1], 'binomial') AS g FROM ds_bin));
--   df_num  chisq
--   1       2.5741
```

`sigma2` takes the fit's own dispersion, not a constant: `g.dispersion` is
`1.0` here because binomial's dispersion is fixed at 1. For a gaussian or
gamma fit, where dispersion is the Pearson estimate, a literal `1.0` would
be silently wrong — pass that fit's own `dispersion` the same way.
`df_residual` is the denominator df. `emmeans` works against a GLM the same
way, but reports its means on a different scale than a reader might
expect — see the next section.

### General linear hypothesis (`lin_hyp`)

`lin_hyp` tests **L·β = 0** against a fit you already have. One row of `L` per
constraint, each as wide as `beta`; several rows means a joint test. It is
stateless — it takes the coefficients and their covariance, not the data — so
it works with any fit, and it is what SAS's CONTRAST and ESTIMATE compute.

```sql
-- Is x1's coefficient zero? (a Type III test of one term)
SELECT (h).df_num, (h).f_statistic, (h).f_p_value
FROM (SELECT lin_hyp([[0, 1, 0]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM ds1));
--   df_num  f_statistic  f_p_value
--   1       249.4064     7.21e-08

-- A contrast: is x1 − x2 zero? The estimate comes back with its standard error.
SELECT (h).estimates[1].estimate, (h).estimates[1].std_error
FROM (SELECT lin_hyp([[0, 1, -1]], …) AS h FROM …);
--   3.0366  0.3370
```

| Field | Type | Notes |
| ----- | ---- | ----- |
| `estimates` | `LIST<STRUCT>` | one element per row of `L`: `estimate`, `std_error`, `t_statistic`, `p_value` |
| `ss` | `DOUBLE` | the hypothesis sum of squares, `chisq · sigma2` |
| `df_num` | `BIGINT` | the rank of `L·cov·Lᵀ`, so redundant constraints count once |
| `df_den` | `DOUBLE` | the `df` argument, echoed |
| `f_statistic`, `f_p_value` | `DOUBLE` | referenced to F(`df_num`, `df_den`) |
| `chisq`, `chisq_p_value` | `DOUBLE` | the Wald statistic, referenced to χ²(`df_num`) |

Pass `cov` exactly as `lm_fit` returns it: flat, row-major, k² elements. `df`
is a DOUBLE so a fractional denominator df fits; `lm_fit`'s `df_residual` casts
in on its own.

A p-value smaller than about 1e-16 reads as exactly **0**, because the p-values
are computed as `1 − CDF` and the far tail is lost in that subtraction. The
joint example above is one: its `chisq_p_value` comes back 0 where statsmodels
reports about 1e-58. The same applies to `lm_fit`.

**`ss` is a sum of squares only when `cov` is the classical σ²(XᵀX)⁻¹.** With an
HC or CR covariance the F and the chi-square are a Wald test — the same thing
statsmodels' `wald_test` reports — and `ss` is not a partition of variance.

The whole result is **NULL** when the hypothesis cannot be evaluated: when it
puts weight on a coefficient the fit could not estimate (a rank-deficient fit's
aliased terms, SAS's "non-estimable"), when it carries no information at all, or
when the supplied `cov` fails one of the two covariance checks below. A
hypothesis that merely avoids those aliased terms is fine. A dimension mismatch
— an `L` row that is not as wide as `beta`, or a `cov` that is not k² long —
raises instead.

Those checks matter because `cov` may be hand-built rather than `lm_fit`'s own
output. A negative variance among the constraints, or a Wald statistic that is
not `>= 0`, gives NULL for that row rather than a plausible-looking number.
Neither is a positive-definiteness test: an indefinite `cov` that passes both is
evaluated normally, because catching every such matrix needs an
eigendecomposition. An asymmetric `cov` is read as its symmetric part, which is
the only part a quadratic form uses.

Type I and II sums of squares are not computable this way: they need nested
refits and the term-to-column mapping. Only Type III reduces to a hypothesis on
the full model, which is what this function tests.

### Estimated marginal means (`emmeans`)

`emmeans` reports the marginal means of a grid you supply against a fit you
already have, with their standard errors and confidence limits, every
pairwise difference between them, and a multiplicity adjustment for that
family. Like `lin_hyp`, it takes coefficients and a covariance, not data, so
it works with any fit. It is what SAS prints as LSMEANS with the DIFF and CL
options, and what R's `emmeans` package reports for a reference grid followed
by `pairs()`.

```sql
emmeans(L, beta, cov, df [, alpha [, adjust]])
```

`L` is one row per mean, nested, each as wide as `beta`. `cov` stays flat
row-major instead, because `lm_fit` produces it that way. `alpha` defaults
to `0.05` and `adjust` to `'tukey'`; the other choices are `'bonferroni'`,
`'holm'`, `'hochberg'`, `'BH'` (or `'fdr'`), `'BY'`, and `'none'`. There is no
`sigma2` argument: `lin_hyp` takes one only to scale its hypothesis sum of
squares, and `emmeans` has no sum of squares to scale.

```sql
-- ds1: three cells of three observations, no intercept, so each coefficient
-- is that cell's own mean.
CREATE TABLE ds1 AS SELECT * FROM (VALUES
    (1.0, 1, 0, 0), (2.0, 1, 0, 0), (3.0, 1, 0, 0),
    (4.0, 0, 1, 0), (6.0, 0, 1, 0), (8.0, 0, 1, 0),
    (5.0, 0, 0, 1), (5.0, 0, 0, 1), (8.0, 0, 0, 1)
) AS t(y, g1, g2, g3);

SELECT printf('%.4f', (u).estimate)  AS estimate,
       printf('%.4f', (u).std_error) AS std_error,
       printf('%.4f', (u).lower_cl)  AS lower_cl,
       printf('%.4f', (u).upper_cl)  AS upper_cl
FROM (SELECT unnest((emmeans([[1,0,0],[0,1,0],[0,0,1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.df_residual)).means) AS u
      FROM (SELECT lm_fit(y, [g1, g2, g3], 'const', false) AS m FROM ds1));
--   estimate  std_error  lower_cl  upper_cl
--   2.0000    0.9428     -0.3070   4.3070
--   6.0000    0.9428     3.6930    8.3070
--   6.0000    0.9428     3.6930    8.3070

SELECT (u).i, (u).j,
       printf('%.4f', (u).estimate)   AS estimate,
       printf('%.4f', (u).p_value)    AS p_value,
       printf('%.4f', (u).p_adjusted) AS p_adjusted
FROM (SELECT unnest((emmeans([[1,0,0],[0,1,0],[0,0,1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.df_residual)).diffs) AS u
      FROM (SELECT lm_fit(y, [g1, g2, g3], 'const', false) AS m FROM ds1));
--   i  j  estimate  p_value  p_adjusted
--   1  2  -4.0000   0.0240   0.0544
--   1  3  -4.0000   0.0240   0.0544
--   2  3  0.0000    1.0000   1.0000
```

| Field | Type | Notes |
| ----- | ---- | ----- |
| `means` | `LIST<STRUCT>` | one element per row of `L`, in `L`'s order: `estimate`, `std_error`, `t_statistic`, `p_value`, `lower_cl`, `upper_cl` |
| `diffs` | `LIST<STRUCT>` | one element per pair, `i < j` row-major — (1,2), (1,3), …, (2,3), … : `i`, `j`, `estimate`, `std_error`, `t_statistic`, `p_value`, `p_adjusted`, `lower_cl`, `upper_cl` |
| `df_den` | `DOUBLE` | the `df` argument, echoed |
| `alpha` | `DOUBLE` | the `alpha` argument, echoed |
| `adjust` | `VARCHAR` | the canonical method name, echoed — `'fdr'` echoes as `'BH'` |
| `k_compare` | `BIGINT` | the number of estimable means |

`i` and `j` in `diffs` are **1-based**, so they index `means` directly.
`means[(u).i]` is the first mean in a difference.

**Against a GLM, the means come back on the link scale, not the response
scale.** `emmeans` receives only `beta` and `cov`; it knows nothing about
the link that produced them. A logistic fit's marginal means are log-odds,
not probabilities, and a Poisson fit's are log rates, not counts.
Cell-means coding makes this concrete: with no intercept, each coefficient
of a logistic fit is already that cell's mean on the link scale.

```sql
-- three cells, binomial, no intercept: cell 1 is 1 success of 4
CREATE TABLE ds_cells AS SELECT * FROM (VALUES
  (1.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
  (1.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0),
  (1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)
) AS t(y, g1, g2, g3);

SELECT printf('%.4f', (e).means[1].estimate),
       printf('%.4f', (e).means[2].estimate),
       printf('%.4f', (e).means[3].estimate)
FROM (SELECT emmeans([[1,0,0],[0,1,0],[0,0,1]],
                     list_transform(g.coefficients, c -> c.estimate),
                     g.cov, g.df_residual) AS e
      FROM (SELECT glm_fit(y, [g1, g2, g3], 'binomial', {'intercept': false}) AS g FROM ds_cells));
--   -1.0986  0.0000  1.0986
```

Cell 1's own success rate is 0.25, and `ln(0.25 / 0.75)` is `-1.0986` — the
log-odds, not the `0.25` a reader expecting the response scale might look
for. Converting back is on the caller: `1 / (1 + exp(-estimate))` inverts
a logit mean, and the inverse differs by link. `lin_hyp`'s own hypothesis
test needs no such conversion — whether `L·beta` is exactly zero holds or
fails directly on the coefficient scale. But a nonzero point estimate from
`lin_hyp`'s own `estimates` field carries the same link-scale caveat as a
mean here, for the same reason: it is a linear combination of `beta`, not
of the response.

A mean can fail to be estimable three ways.

First, it can put weight on a coefficient whose own diagonal entry in
`cov` is not finite. A rank-deficient fit leaves exactly that behind, but
a hand-built `cov` can do the same with no such fit involved; the code
cannot tell the two apart.

Second, a non-finite value can reach its estimate or its variance from
somewhere else entirely. A NULL inside the matching row of `L` costs only
that one mean. A NULL anywhere in `beta`, or in an **off-diagonal** entry
of `cov`, costs every mean at once instead, because both feed every
row's calculation, and the diagonal check above does not see an
off-diagonal NULL at all.

Third, its own variance can be negative by more than rounding should
allow.

When any of these happens, every numeric field of that element of
`means` is NULL, and the same holds for every element of `diffs` that
touches it. A pair can also fail on its own, the same way, even when
both of its means are fine: its own variance can be negative by more than
rounding should allow. `means` and `diffs` keep their length and their
positions regardless, so `i` and `j` stay valid indices.

That is the data-shaped kind of failure, and it is always a NULL, never an
error. Misuse is different, and raises instead: a ragged `L`, a row of `L`
that is not as wide as `beta`, an `L` with no rows, a NULL row of `L`, a
`cov` that is not k² long, or an empty `beta`. Those six raise in close to
the same words `lin_hyp` uses for the same six shapes, because two
neighbouring functions must not answer the same malformed input
differently. A `df` that is not positive and finite, an `alpha` outside
`(0, 1)`, or an unrecognized `adjust` name raise too. `lin_hyp` has
neither an `alpha` nor an `adjust` argument to compare against, and its own
`df` check does not reject an infinite value the way this one does.

Five things worth knowing before reading `adjust` as a knob with no cost:

- **A mean's own confidence limits are never adjusted for multiplicity**,
  whatever `adjust` says. `lower_cl` and `upper_cl` in `means` are always
  `estimate ± t(1 − alpha/2, df) · std_error`. SAS does the same: ADJUST
  governs the comparisons, not the means.
- **Only `'tukey'` and `'bonferroni'` move a difference's confidence limits.**
  `'holm'`, `'hochberg'`, `'BH'` (`'fdr'`), and `'BY'` adjust `p_adjusted`
  only. `lower_cl` and `upper_cl` in `diffs` stay at the unadjusted
  `t(1 − alpha/2, df)` multiplier under those four, because none of them has
  a matching simultaneous interval. R's `emmeans` warns of the same thing.
- **`'tukey'` is the Tukey-Kramer approximation whenever the means being
  compared do not share a standard error.** That covers any unbalanced
  design. It is exact only for k means with a common variance and equal
  group sizes. With unequal standard errors, the same studentized-range
  multiplier is applied to each pair's own standard error instead. SAS and
  R's `emmeans` both do this too. At exactly two means, `p_adjusted` and
  `p_value` come from two different code paths that happen to agree
  mathematically, not the same one, so they can differ by about 1e-13 in
  practice. `p_adjusted` can land very slightly below `p_value` rather
  than at or above it; that is the quadrature's own accuracy, not a real
  crossing.
- **`k_compare` is the number of estimable means.** For `'tukey'` it is
  exactly the family size the studentized range uses, so
  `1 - ptukey(sqrt(2) * abs(t_statistic), k_compare, df_den)` reproduces a
  pair's `p_adjusted`. `'bonferroni'`'s own family size is the count of
  estimable pairs that have a p-value instead. That count usually, but not
  always, equals `k_compare`; `docs/kernel_api.md` has the one case where it
  differs.
- **The adjustment assumes the rows of `L` are means.** The studentized range
  describes the spread of a set of means, so `'tukey'`'s family-wise
  guarantee is meaningful because the family is every pairwise difference
  among the rows of `L`. A caller who passes rows that are themselves
  contrasts still gets an arithmetically valid answer, but the family-wise
  interpretation no longer holds. The function has no way to detect this,
  so nothing in the result flags it.

**Named arguments do not work on `emmeans`.** DuckDB's binder matches a
scalar-function call positionally, regardless of any `name :=` label.
`emmeans(L, beta, cov, df, alpha := 0.01)` happens to work, but only because
`alpha` is genuinely the fifth argument. `emmeans(L, beta, cov, df, adjust :=
'bonferroni')` fails instead, with a cast error: the five-argument overload
reads that fifth position as `alpha`, a `DOUBLE`, and gets the string
`'bonferroni'`. Call `emmeans` positionally.

**The default `'tukey'` adjustment evaluates a double integral for every
row**, through `qtukey`'s quantile search, and is far slower than the
other methods. Measured over 1000 rows of a five-mean grid, `'tukey'` took
about 28.5 s against `'bonferroni'`'s 0.158 s — a factor of about 180. A
caller running `emmeans` once per group over many groups should expect
that, or choose a cheaper `adjust` method.

### Multiple-testing correction (scalar)

| Function                                | Description                                                                |
| --------------------------------------- | -------------------------------------------------------------------------- |
| `adjust_p(pvals, method)`               | Apply a multiple-testing correction to a list of p-values                  |

`adjust_p` takes a `LIST<DOUBLE>` of raw p-values and a method name, and
returns adjusted p-values in input order. Methods (case-sensitive, matching
R's `p.adjust`):

- `'bonferroni'` — `min(1, n · p_i)`.
- `'holm'` — Holm step-down (1979).
- `'hochberg'` — Hochberg step-up (1988).
- `'BH'` (alias `'fdr'`) — Benjamini-Hochberg FDR (1995).
- `'BY'` — Benjamini-Yekutieli FDR (2001) for arbitrary dependence.
- `'none'` — pass-through, returns the input unchanged.

NULLs in the input list are passed through to the output at the same
position and are excluded from `n`.

### Data import (table function)

| Function                                | Description                   |
| --------------------------------------- | ----------------------------- |
| `read_stat(path, [format], [encoding])` | Read SAS / SPSS / Stata files |

### Data export (COPY function)

| Statement                         | Description                       |
| --------------------------------- | --------------------------------- |
| `COPY <table> TO 'file.xpt'`      | Write SAS Transport (XPT v5)      |
| `COPY <table> TO 'file.sas7bdat'` | Write SAS7BDAT (see caveat below) |
| `COPY <table> TO 'file.sav'`      | Write SPSS SAV                    |

> **SAS7BDAT caveat.** ReadStat's SAS7BDAT writer is reverse-engineered: files
> round-trip through ReadStat-family readers (this extension's `read_stat()`,
> pyreadstat, haven, R) but are **not opened by real SAS / SAS Universal
> Viewer / SAS OnDemand**. Use XPT for SAS-native readability.

### Visualizations (VISUALIZE parser extension)

A Grammar-of-Graphics SQL dialect: `VISUALIZE` returns a single row with two
columns — `spec` (a complete Vega-Lite v5 JSON spec) and `layer_sqls` (a
`MAP(VARCHAR, VARCHAR)` of named SQL strings, one per layer). The client
runs each layer's SQL and feeds the rows to vega-embed via the `datasets` API.

**Tutorial:** [`docs/visualize.md`](docs/visualize.md) walks through a worked
example of every mark and clause.

**See also:** [posit-dev/ggsql-duckdb](https://github.com/posit-dev/ggsql-duckdb)
— the dedicated grammar-of-graphics DuckDB extension from the ggplot2 team that
inspired this syntax. `stats_duck`'s
`VISUALIZE` is **not** a reimplementation of ggsql and doesn't track its syntax — it's
a deliberately minimal, WebAssembly-friendly built-in supporting only a fixed set of
marks and clauses, for plotting stats output inline. For the full grammar of graphics,
use ggsql.

```
[WITH [RECURSIVE] <cte> AS (...) [, <cte> AS (...)]*]
VISUALIZE <expr> AS <aesthetic> [: <type>] (, <expr> AS <aesthetic> ...)
FROM <table>
DRAW <mark> [STAT <identity|smooth|summary>] (DRAW <mark> [STAT ...])*
[FACET BY <expr> [ROWS | COLS] | FACET BY <row_expr>, <col_expr>]
[SCALE <channel> {TO <scheme> | ZERO true|false | DOMAIN <lo> <hi> | LABEL '<text>'}+]*
[TITLE '<text>' [SUBTITLE '<text>']]
```

Multiple `SCALE` options may be **stacked** on one channel
(`SCALE x LABEL 'Bill Depth' ZERO false`) or split across repeated `SCALE x`
clauses — they merge into one scale/axis block either way. SQL comments
(`-- …` to end of line, and `/* … */`) may appear anywhere in a `VISUALIZE`
statement; they're skipped like whitespace (but never treated as comments
inside a string literal).

A leading `WITH` clause is supported; CTEs are scoped to each layer's
projected SQL so they compose with wrapping marks (`line`, `bar`, `area`,
`errorband`, `regression`) without extra work. `WITH … SELECT …` statements
without a top-level `VISUALIZE` keyword fall through to DuckDB's normal SQL
parser unchanged.

**Marks:** `point`, `line`, `bar`, `histogram`, `text`, `area`, `rule`, `tick`,
`errorbar`, `errorband`, `boxplot`, `violin`, `heatmap`, `density`, `regression`. Custom
marks register as `visualize_mark_v1_<name>` scalar functions and are discovered
via DuckDB's catalog, so other extensions can ship their own marks without
modifying stats_duck.

`heatmap` is a `rect` mark with ordinal x/y and quantitative color (correlation
matrices, contingency tables). `density` runs Vega-Lite's KDE on the `x`
aesthetic, grouped by `color` if mapped (one curve per category). `violin`
renders one horizontal density per category of `x`, laid out via vega-lite's
`column` facet (composes with `FACET BY ... ROWS` but conflicts with
`FACET BY ... COLS`). `regression` fits a linear `y ~ x` model server-side
via Vega-Lite's regression transform, also grouped by `color`. Use
`DRAW point DRAW regression` for a scatter-with-fit overlay.

**Aesthetic channels:** `x`, `y`, `color`, `fill`, `stroke`, `shape`, `size`,
`opacity`, `tooltip`, `text`, `x2`, `y2`. Unknown channels are silently dropped.

**Type overrides:** append `:quantitative`, `:ordinal`, `:nominal`, or
`:temporal` to an aesthetic to force its Vega-Lite type
(e.g. `year AS color:ordinal`).

**Axis labels:** `SCALE x LABEL 'Bill length (mm)'` injects an `axis.title` into
the channel; pairs with `TO` / `ZERO` / `DOMAIN` on the same channel.

**Titles:** `TITLE 'Plot title' [SUBTITLE 'Plot subtitle']` appears once per
spec, after `SCALE` clauses. Always emitted as a Vega-Lite `TitleParams` object
so a subtitle can be added without reshaping consumer code.

## SAS compatibility

stats_duck defaults follow modern statistical conventions (scipy /
R with `correct=FALSE` / `var.equal=FALSE`). The one exception is
`summary_stats` — its skewness/kurtosis formulas are the Fisher-Pearson
bias-corrected ones, which is what SAS PROC MEANS, pandas, and Excel
report. To reproduce SAS PROC output exactly, use the toggles below.

| SAS procedure / statistic                      | stats_duck call                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| PROC MEANS — mean, SD, skewness, kurtosis      | `summary_stats(x)` _(default already matches SAS)_                  |
| PROC TTEST — Pooled (equal variances)          | `ttest_2samp(x, y, true)`                                           |
| PROC TTEST — Satterthwaite (default)           | `ttest_2samp(x, y)` _(default Welch's matches Satterthwaite)_       |
| PROC NPAR1WAY — Wilcoxon two-sample Z          | `mann_whitney_u(x, y, 'two-sided', true)` _(continuity correction)_ |
| PROC FREQ — Continuity Adj. χ² (2x2 only)      | `chisq_independence(row, col, true)` _(Yates' correction)_          |
| PROC FREQ — Chi-Square (no adjustment)         | `chisq_independence(row, col)` _(default)_                          |
| PROC CORR — Pearson / Spearman / Kendall       | `pearson_test(x, y)` / `spearman_test(x, y)` / `kendall_test(x, y)` |
| PROC GLM — one-way ANOVA F-test                | `anova_oneway(value, group)`                                        |
| PROC UNIVARIATE — Signed Rank (sign-rank test) | `wilcoxon_signed_rank(x, 0)` _(against a 0 column or constant)_     |
| PROC UNIVARIATE — Sign test (M statistic)      | `sign_test_1samp(x, [mu_0])`                                        |
| PROC UNIVARIATE — quantiles (Type 5)           | `summary_stats(x, true, 5)`                                         |

Defaults preserve modern conventions so users on the modern side of the
fence get sensible numbers without touching the API; SAS users add the
appropriate flag during migration and validation.

## Examples

### Hypothesis tests

#### One-sample t-test

```sql
-- Test whether the mean of v3 differs from zero
SELECT ttest_1samp(v3) FROM measurements;

-- Test against a specific mean
SELECT ttest_1samp(v3, 5.0) FROM measurements;

-- One-sided test with 99% confidence interval
SELECT ttest_1samp(v3, 0.0, 0.01, 'greater') FROM measurements;
```

#### Two-sample t-test

```sql
-- Welch's t-test (default, does not assume equal variances)
SELECT ttest_2samp(group_a, group_b) FROM experiment;

-- Student's t-test (assumes equal variances)
SELECT ttest_2samp(group_a, group_b, true) FROM experiment;
```

#### Paired t-test

```sql
-- Compare before/after measurements
SELECT ttest_paired(before, after) FROM patients;
```

#### Mann-Whitney U test

Non-parametric alternative to the two-sample t-test:

```sql
-- Compare two independent samples
SELECT mann_whitney_u(group_a, group_b) FROM experiment;

-- One-sided test
SELECT mann_whitney_u(group_a, group_b, 'less') FROM experiment;
```

#### Wilcoxon signed-rank test

Non-parametric alternative to the paired t-test:

```sql
-- Compare paired measurements
SELECT wilcoxon_signed_rank(before, after) FROM patients;
```

### Working with results

The result of any hypothesis test is a `STRUCT` — access individual fields with
dot notation:

```sql
SELECT (ttest_1samp(v3)).t_statistic,
       (ttest_1samp(v3)).p_value
FROM measurements;
```

Or unpack all fields:

```sql
SELECT (r).*
FROM (SELECT ttest_1samp(v3) AS r FROM measurements);
```

#### Group-by

Run a test per group with a plain `GROUP BY`:

```sql
SELECT id3,
       (ttest_1samp(v3)).t_statistic,
       (ttest_1samp(v3)).p_value
FROM measurements
GROUP BY id3;
```

#### Inline data

Use `VALUES` for quick experiments with literal data:

```sql
SELECT (r).*
FROM (SELECT ttest_1samp(v) AS r FROM (VALUES (2.0), (4.0), (6.0), (8.0), (10.0)) AS t(v));
```

### Data import

#### Reading statistical file formats

Read SAS (`.sas7bdat`, `.xpt`), SPSS (`.sav`, `.zsav`, `.por`), and Stata
(`.dta`) files:

```sql
-- Auto-detect format from file extension
SELECT * FROM read_stat('data.sas7bdat');

-- Explicit format
SELECT * FROM read_stat('data.dat', format := 'dta');

-- Replacement scan: query files directly
SELECT * FROM 'survey.sav';
```

Date, datetime, and time columns are automatically detected from format
metadata and converted to DuckDB's native temporal types.

`read_stat` is wired through DuckDB's virtual file system, so the same call
works against:

- local paths: `read_stat('/data/survey.sav')`
- remote files (with the `httpfs` extension loaded): `read_stat('https://…/survey.sav')`
- object stores: `read_stat('s3://bucket/survey.sav')`
- in-browser DuckDB-WASM file buffers registered via `registerFileBuffer()`

#### Writing statistical file formats

The same VFS-backed writers, exposed as DuckDB `COPY` functions:

```sql
-- SAS Transport (XPT v5) — universally readable, FDA / pharma standard
COPY survey TO 'survey.xpt';

-- SAS7BDAT — round-trips through stats_duck and pyreadstat (NOT real SAS, see caveat)
COPY survey TO 'survey.sas7bdat' (COMPRESSION 'rows');

-- SPSS SAV
COPY survey TO 'survey.sav';

-- COPY <subquery> form works too
COPY (SELECT * FROM huge_table WHERE region = 'EU') TO 'eu.xpt';
```

NULL handling differs by storage format: numeric NULLs round-trip as
SAS/SPSS system-missing, but VARCHAR NULLs collapse to empty strings (these
formats have no NULL/empty distinction for character columns).

### Visualizations (VISUALIZE)

```sql
-- Set up — penguin morphology, classic ggplot2 demo data
CREATE TABLE penguins AS SELECT * FROM (VALUES
    (39.1, 18.7, 'Adelie',    2018), (39.5, 17.4, 'Adelie',    2019),
    (44.0, 18.0, 'Gentoo',    2020), (45.2, 14.5, 'Chinstrap', 2021)
) AS t(bill_len, bill_dep, species, year);
```

#### Scatter

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point;
```

#### Scatter colored by species, with a viridis scale

```sql
VISUALIZE bill_len AS x, bill_dep AS y, species AS color
FROM penguins
DRAW point
SCALE color TO viridis;
```

#### Multi-layer (scatter + line overlay)

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point DRAW line;
```

#### Faceted plot — one panel per species, vertically stacked

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins
DRAW point FACET BY species ROWS;
```

#### 2D facet grid — species rows × sex columns

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins
DRAW point FACET BY species, sex;
```

#### Scatter with LOESS overlay via STAT smooth

```sql
VISUALIZE bill_len AS x, bill_dep AS y FROM penguins
DRAW point
DRAW line STAT smooth;
```

#### SQL expressions in mappings

```sql
VISUALIZE bill_len * 2 AS x, log(bill_dep) AS y FROM penguins DRAW point;
```

#### Type-annotated aesthetic (year is INTEGER but should be ordinal here)

```sql
VISUALIZE bill_len AS x, bill_dep AS y, year AS color:ordinal
FROM penguins
DRAW point
SCALE color TO viridis;
```

The output of any `VISUALIZE` query is a single row with `(spec, layer_sqls)`.
The client side (a browser app, a notebook renderer, …) is responsible for
running each layer's SQL and feeding the rows into vega-embed via its
`datasets` API. See [Bedevere Wise](https://bedeverewise.app/) for a reference
DuckDB-WASM consumer.

## Building

```bash
git submodule update --init --recursive
make release
```

### Building with MinGW

DuckDB extensions are tagged at build time with a _platform string_ that the
host process uses to validate ABI compatibility before loading. The default
`make release` on Windows uses MSVC and stamps `windows_amd64`. A consumer
DuckDB built with mingw-w64 (e.g. via zig's bundled clang + mingw-w64
sysroot, or the rtools/MSYS2 toolchains) will refuse to load that binary
with:

```
Failed to load 'stats_duck.duckdb_extension'. The file was built for the
platform 'windows_amd64', but we can only load extensions built for
platform 'windows_amd64_mingw'.
```

For those hosts, build with:

```bash
make mingw_release
```

This drives a CMake build with `-G "MinGW Makefiles"`, sets `CC=gcc`,
`CXX=g++`, and stamps `DUCKDB_EXPLICIT_PLATFORM=windows_amd64_mingw`. The
binary lands at:

```
build/mingw_release/extension/stats_duck/stats_duck.duckdb_extension
build/mingw_release/repository/<duckdb_version>/windows_amd64_mingw/stats_duck.duckdb_extension
```

The MSVC `windows_amd64` binary at `build/release/...` is left alone — both
flavors can coexist on disk and ship side by side. CI already produces both
on every release; this target is for local-only verification or for
distributing to consumers that bundle their own DuckDB and need an exact
ABI match.

**Toolchain requirements.** Anything ABI-compatible with `x86_64-w64-mingw32`
works. Tested with TDM-GCC 10.3 and MSYS2's `mingw-w64-x86_64-toolchain`. On
TDM-GCC the Makefile passes `-D_WIN32_WINNT=0x0A00` because TDM-GCC's default
predates the Vista-era APIs DuckDB uses; MSYS2 already defaults to that
value, where the flag is harmless.

**vcpkg is intentionally not used** for the MinGW build — its Windows
default triplet is MSVC and would not match the toolchain. zlib (the only
external dep, used for SPSS `.zsav` read/write) becomes optional and
gracefully degrades if absent (see `CMakeLists.txt:97`); install zlib via
your mingw package manager if you need `.zsav` support.

### Building with zig as the C++ toolchain (libc++ hosts)

DuckDB's platform string `windows_amd64_mingw` does not distinguish which C++
runtime the binary is linked against. Two `windows_amd64_mingw` extensions can
both pass the platform check on load and then segfault during function
registration if their `std::map` / `std::shared_ptr` / etc. layouts differ —
which they do across **libstdc++** (GNU's STL, what mingw-w64 GCC uses) and
**libc++** (LLVM's STL, what zig's bundled clang uses with `link_libcpp`).

Some consumers embed a DuckDB built with zig and `link_libcpp = true`, so
their DuckDB links against libc++. The `make mingw_release` target above
produces a libstdc++-linked binary and is _not_ compatible — it'll segfault
inside such a host at function registration. Use the zig variant instead:

```bash
make zig_mingw_release
```

This drives the same CMake configuration as `mingw_release` but routes
`CC` / `CXX` through `scripts/zig-shims/zig-cc.cmd` / `zig-cxx.cmd`, which
forward to `zig cc -target x86_64-windows-gnu` and
`zig c++ -target x86_64-windows-gnu`. Output:

```
build/zig_mingw_release/extension/stats_duck/stats_duck.duckdb_extension
build/zig_mingw_release/repository/<duckdb_version>/windows_amd64_mingw/stats_duck.duckdb_extension
```

Same platform stamp (`windows_amd64_mingw`), different C++ runtime. On a
correctly-built artifact, `objdump -p stats_duck.duckdb_extension | grep DLL`
shows only Windows system DLLs (`KERNEL32.dll`, `api-ms-win-crt-*`) — no
`libstdc++-6.dll` or `libgcc_s_seh-1.dll`. If any GNU runtime DLL appears,
the build slipped back to GCC.

**Requirements.** zig 0.16+ on `PATH`. Tested with the `zig.zig`
WinGet-installed copy.

## Testing

```bash
make release && make test
```

Or on Windows where the Makefile test runner may not work:

```bash
./build/release/test/Release/unittest.exe "test/sql/*"
```

## KoliStat

The Stats Duck is developed by [KoliStat](https://kolistat.com/):

- [The Stats Duck product page](https://kolistat.com/products/the-stats-duck)
- [Bedevere Wise](https://bedeverewise.app/) — KoliStat's browser-based
  statistical workbench, where this extension runs on DuckDB-WASM
  ([product page](https://kolistat.com/products/bedevere))

## License

The Stats Duck is released under the [Apache License 2.0](LICENSE). If you use
it in academic work, see [`CITATION.cff`](CITATION.cff) for citation metadata.
