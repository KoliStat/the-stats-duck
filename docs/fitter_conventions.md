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
