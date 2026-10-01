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
