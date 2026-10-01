# Fitter conventions: the model-fitting API standard

Every SQL-facing model-fitting function in `stats_duck` follows the
conventions on this page. Today that is `lm_fit`. Next come `glm_fit`
(#22), `lin_hyp` (#20), `emmeans` (#21), and the fitters in downstream
sibling repos. The conventions exist for two reasons. A new fitter can be
designed by following them step by step. And a user who has learned one
fitter can predict how the others behave.

[`kernel_api.md`](kernel_api.md) covers the DuckDB-free C++ kernel
underneath. This page covers the SQL surface and the numeric and
validation rules that apply across fitters.

The functions that predate this standard (the hypothesis-test aggregates,
`summary_stats`, the `lm` table functions) deviate in places. The appendix
at the end records each deviation. New fitters follow this page without
exception.

## Input conventions

- **One row is one observation.** Fitters are aggregates, so `GROUP BY`
  gives one fit per group. Post-fit helpers such as `lin_hyp` and `emmeans`
  are scalar functions over a fit's outputs instead.
- **The design row is a `LIST(DOUBLE)` of predictor values.** An aggregate
  sees values, not column names, so coefficients are positional. Terms are
  labelled `x1` … `xk` in list order, with `(Intercept)` first when
  `add_intercept` is true (the default). Callers pass predictors without a
  constant column.
- **Per-row key columns (cluster, subject, strata) are `VARCHAR`
  arguments.** DuckDB does not cast integers to `VARCHAR` in this position
  on its own, so callers write `id::VARCHAR` (`test/sql/lm_fit_agg.test`
  does). Keys are opaque labels. A fitter must not depend on their order
  or density.
- **Optional behavior arrives as trailing arguments or new overloads.** An
  existing position never changes meaning. Before adding an overload, check
  the full positional matrix for ambiguity. WLS (Increment B, #19) is
  deferred for exactly this reason: a weight column and a cluster column
  must not land in the same position.
- **Missing data: listwise drop.** A row leaves the fit when the response,
  any element of the design row, or the key column is NULL, NaN, or
  infinite (the complete-case loop in `src/lm_fit_function.cpp`). A design
  row whose length differs from the first one seen marks the whole group
  as failed. A group where too few rows survive is a failed fit, which is
  a NULL result, not an error.

## The return STRUCT

- **One STRUCT per fit.** Never parallel columns. Never several return
  variants for the same fitter.
- **Field order is part of the contract, and it is append-only.** Finalize
  writes the children by index. The type function carries the comment
  `// Field order MUST match the writes in Finalize` with an index comment
  per field (`LmFitResultType` in `src/lm_fit_function.cpp`). A released
  field never moves or disappears. New fields are appended.
- **NaN becomes SQL NULL at the vector layer** (the `SetD` helper in
  `src/lm_fit_function.cpp`). ±Inf passes through unchanged, so
  `loglik = +inf` at zero RSS stays representable.
- **Failure is a value.**
  - A *data-shaped* failure (too few rows, a degenerate design, fewer than
    two clusters) makes the group's STRUCT NULL. It never raises an
    exception.
  - *Misuse* (an unknown `vcov` name, a non-constant option flag) is a
    bind-time error. The query fails before it runs.
  - *Iterative* fitters (`glm_fit` and later) also expose
    `converged BOOLEAN` and `iterations BIGINT`. They return the best
    iterate with `converged = false` instead of NULL. This mirrors
    `optimize::Result` in the kernel, which reports `converged = false`
    with its best point rather than failing.
- **Covariance is a flat row-major `LIST<DOUBLE>`** of length k², with
  `cov[i*k + j] = cov(βᵢ, βⱼ)`, in `coefficients` order. Aliased positions
  are NaN, and so NULL, once rank-deficient fits land (#36).
- **Degrees of freedom follow the estimated rank.** A fitter reports
  `rank`, and every df formula uses it: σ², adjusted R², the HC and CR
  small-sample factors, and the t reference. `df_residual = n − rank` is a
  property of the fit. Cluster-robust inference uses a `t(G−1)` reference
  without changing `df_residual`. statsmodels does the same without saying
  so; this page says so.
- **p-values and confidence intervals come from the in-repo distribution
  kernels** (`src/include/distributions.hpp`). There is never a second CDF
  implementation.
- **Statistic naming.** A t-referenced fitter names the column
  `t_statistic`. A z-referenced fitter names it `z_statistic`. A fitter
  STRUCT never has a bare `statistic` field.

### Worked example: `lm_fit`, the template

Fields 0–11 are shipped. Fields 12–14 are the approved design of lm_fit
Increment A (#37) and land with it.

| # | Field | Type | Notes |
|---|-------|------|-------|
| 0 | `coefficients` | `LIST<STRUCT(term VARCHAR, estimate, std_error, t_statistic, p_value DOUBLE)>` | positional terms, `(Intercept)` first |
| 1 | `n` | `BIGINT` | rows used, after the listwise drop |
| 2 | `k` | `BIGINT` | design columns, intercept included |
| 3 | `df_residual` | `BIGINT` | `n − k` today; `n − rank` once rank-deficient fits land (#36) |
| 4 | `r_squared` | `DOUBLE` | |
| 5 | `adj_r_squared` | `DOUBLE` | |
| 6 | `sigma` | `DOUBLE` | residual standard error |
| 7 | `f_statistic` | `DOUBLE` | classical, not robustified |
| 8 | `f_p_value` | `DOUBLE` | |
| 9 | `has_intercept` | `BOOLEAN` | |
| 10 | `vcov_type` | `VARCHAR` | the canonical estimator name, echoed back |
| 11 | `n_clusters` | `BIGINT` | NULL unless CR0/CR1 |
| 12 | `rank` | `BIGINT` | *lands with Increment A (#37)* |
| 13 | `loglik` | `DOUBLE` | *lands with Increment A (#37)* |
| 14 | `cov` | `LIST<DOUBLE>` | row-major k²; *lands with Increment A (#37)* |

## Numeric and portability discipline

- **Deterministic by construction.** Identical inputs give bit-identical
  results, on every platform. Randomness exists only in the sampling
  functions. The `r*` scalars draw from a per-thread generator seeded from
  `std::random_device` and are registered `VOLATILE`
  (`src/random_sampling_function.cpp`). `bootstrap` takes an optional seed
  argument for reproducible runs. A fitter never draws random numbers, and
  it never sums in an order that depends on a hash table.
- **The wasm string-hash pitfall.** Never put a default-hash
  `std::string`-keyed `std::unordered_map` or `unordered_set` in extension
  code. It pulls in a libc++ symbol (`std::__hash_memory`) that duckdb-wasm
  does not export, and the extension then fails at load time in the
  browser. Use `PortableStringHash` (`src/include/portable_string_hash.hpp`),
  as `anova_oneway`, `chisq_*`, and the ReadStat type tables do. Or avoid
  hashing: the `lm_fit` cluster path groups keys by sorting
  (`DensifyClusters` in `src/lm_fit_function.cpp`).
  `scripts/check-wasm-string-hash.sh` enforces the rule in CI.
- **Buffer rows when the estimator needs them.** Leverage-based (HC2, HC3)
  and score-based (CR*, REML-style) estimators do not reduce to a fixed set
  of streaming moments. The aggregate state buffers the raw rows and does
  the math in Finalize. Do not reshape the math to force a streaming form.
- **The math lives behind the kernel boundary.** Fitting numerics go in a
  DuckDB-free core translation unit (`src/lm_core.cpp` is the pattern), as
  described in [`kernel_api.md`](kernel_api.md). The `*_function.cpp` file
  only binds arguments, buffers rows, and writes vectors.

## Validation discipline

- **The oracle runs offline; the goldens are committed.** Reference values
  come from statsmodels through a generator script
  (`test/cpp/gen_lm_fit_fixtures.py` is the pattern) run in a throwaway
  virtualenv. The oracle is never a build or CI dependency. The generator
  states its own conventions, and any deliberate difference from the
  oracle is written down next to the affected golden.
- **A tolerance band carries a written reason.** A loose band is a finding
  to explain, not a default to accept. `test/cpp/test_lm_fit.cpp` names
  its bands in one place (1e-6 for coefficients, standard errors, σ and
  R²; 1e-3 for the F statistic; 1e-2 relative for p-values) but does not
  yet say why the F and p bands are looser. New fitters must say why.
- **Two test layers, both required.** Standalone C++ goldens
  (`scripts/run-cpp-tests.sh`, no DuckDB) cover the kernel and core math.
  sqllogictest files (`test/sql/*.test`) cover the SQL surface, with float
  output through `printf('%.4f', …)` so results are deterministic text.
- **New-fitter checklist.** Generator script and fixtures header → failing
  C++ goldens, including the failure-as-value cases → core implementation
  → SQL tests → README function-table row → cross-links here and in
  `kernel_api.md` → CHANGELOG entry.

## Appendix: audit of the existing surface (2026-10)

The standard above binds new fitters. The functions below predate it. A
deviation is grandfathered when fixing it would rename or retype a
released field, or change a documented result. A deviation that would
surprise a consumer writing generic code across functions has a follow-up
issue in the last column.

Audited on 2026-10-01 from the sources in `src/`. Readers, writers,
`VISUALIZE`, `table_one`, `meta`, `corr_matrix`, and the list- or
scalar-returning helpers (`bootstrap`, `bin_edges`, `adjust_p`,
`poibin_cdf`) are out of scope: they return rows or plain values, not a
result STRUCT.

| Function | Result | NaN → NULL | Missing data | Data-shaped failure | Deviations and follow-ups |
|---|---|---|---|---|---|
| `lm_fit` | STRUCT (the template) | yes, `SetD` | listwise drop: NULL, NaN, Inf; a ragged design row fails the group | NULL group; misuse is a bind error | none; it defines the standard |
| `lm`, `lm_summary` | table-function rows | NaN yes, ±Inf no (`SetDoubleOrNull`) | complete-case `WHERE … IS NOT NULL` filter | raises `InvalidInputException` for `n ≤ k` or a singular `X'X` | failure is an error, not a value; the re-host on `lm_core` (#45) decides the replacement |
| `ttest_1samp`, `ttest_paired` | STRUCT | no | NULL rows skipped (paired: either side); NaN inputs not filtered | `n < 2` → NULL; zero variance → ±Inf or NaN fields | NaN inputs #52; NaN outputs #53 |
| `ttest_2samp` | STRUCT | no | the two samples accumulate independently; NaN not filtered | either `n < 2` → NULL; Welch `df` is NaN when both variances are 0 | #52, #53 |
| `mann_whitney_u` | STRUCT | no | samples independent; NaN not filtered and reaches `std::sort` | either `n < 2` → NULL; all ties → `z = 0`, `p = 1` | #52; the all-ties value is grandfathered |
| `wilcoxon_signed_rank` | STRUCT | no | pair dropped when either side is NULL; zero differences dropped; NaN not filtered | fewer than 2 non-zero differences → NULL; tied ranks → `z = 0` | #52; the tie value is grandfathered |
| `sign_test_1samp`, `sign_test_paired` | STRUCT | not needed (`p` is always finite) | NULL and NaN skipped; paired drops the pair | no signed difference (`n_pos + n_neg < 1`) → NULL | none |
| `pearson_test`, `spearman_test` | STRUCT | no | pair dropped when either side is NULL or NaN | `n < 3` or zero variance → NULL; at `n = 3` the CI fields are NaN; `t_statistic` is ±Inf at `\|r\| = 1` | CI NaN #53; ±Inf passes through under the standard |
| `kendall_test` | STRUCT | no | as Pearson | `n < 3` or a zero denominator → NULL; degenerate variance → `z = 0`, `p = 1` | grandfathered |
| `anova_oneway` | STRUCT | no | row dropped when the value or the group is NULL, or the value is NaN | fewer than 2 groups, or `n − k ≤ 0` → NULL; zero within-group variance → `f = +Inf`, `p = 0` | sums in hash-table order #54 |
| `chisq_independence`, `chisq_goodness_of_fit` | STRUCT | no NaN path (VARCHAR inputs) | row dropped when a label is NULL | empty table, `1 × N`, or fewer than 2 categories → NULL | sums in hash-table order #54 |
| `jarque_bera` | STRUCT | not needed | NULL and NaN skipped | `n < 4` or zero variance → NULL | none |
| `shapiro_wilk` | STRUCT | clamps `w` and `p` into [0, 1] | NULL and NaN skipped | `n < 3` or `n > 5000` → NULL; all-equal input → `w = 1`, `p = 1` | all-equal input differs from Anderson–Darling and KS, which return NULL; grandfathered |
| `anderson_darling` | STRUCT | clamps `p` | NULL and NaN skipped | `n < 8` or zero variance → NULL | none |
| `ks_test_1samp` | STRUCT | clamps `p` | NULL and NaN skipped | `n < 3` or zero variance → NULL | none |
| `ks_test_2samp` | STRUCT | clamps `p` | the two samples accumulate independently | an empty sample → NULL | none |
| `summary_stats` | STRUCT | no, by design | NULL and NaN are counted in `n_missing` | empty input → NULL; `n = 1` → `sd = 0` | NaN for `mode`, `skewness`, and `kurtosis` is the documented SAS "Mode ." convention; grandfathered |

Observations across the surface:

- No aggregate raises an exception for a data-shaped failure. Every one
  returns a NULL STRUCT, which is what the standard asks. Only the `lm`
  table functions raise.
- Statistic naming already follows the standard: `t_statistic`,
  `z_statistic`, `u_statistic`, `w_statistic`, `m_statistic`,
  `f_statistic`, `chi_square`. No function has a bare `statistic` field.
- Two-sample tests accumulate their two columns independently
  (`ttest_2samp`, `mann_whitney_u`, `ks_test_2samp`). That is correct for
  two independent samples. The listwise rule applies to one observation
  per row.
- Bind-time errors use `BinderException` in most files and
  `InvalidInputException` in `src/ttest_agg_function.cpp`. A caller sees
  an error either way. New fitters use `BinderException`.
- `src/ttest_function.cpp` holds table-function t-tests that are compiled
  but never registered (#55).
