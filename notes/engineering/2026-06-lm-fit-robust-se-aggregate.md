# `lm_fit`: why a robust-SE regression aggregate can't stream

*2026-06 — Epic 1.1 (`lm_fit`, the OLS aggregate with HC0–HC3 standard errors).*

Three things about this function were non-obvious enough to write down: it
can't be a streaming aggregate, it's the first nested-list output in the
codebase, and aggregates can't take the named parameters that `lm` uses.

## 1. Robust SEs force buffering the raw rows

The natural shape for a regression aggregate is to stream *sufficient
statistics*: accumulate `X'X` (k×k) and `X'y` (k) in the state, solve at
Finalize. That is what gives you β and the *classical* covariance
`σ̂²(X'X)⁻¹` in O(k²) memory regardless of n. The existing `ttest`/`anova`
aggregates all work this way (Welford moments).

It does not extend to the robust covariances, and that is the whole point of
this function:

- **HC0/HC1** need the "meat" `Σᵢ êᵢ² xᵢxᵢᵀ`. Since `êᵢ = yᵢ − xᵢᵀβ` and β isn't
  known until the end, expanding `êᵢ²xᵢxᵢᵀ` pulls in **fourth-order** moments of
  x (a k²×k² tensor). Technically streamable, but ugly.
- **HC2/HC3** need the per-row leverage `hᵢᵢ = xᵢᵀ(X'X)⁻¹xᵢ` inside a *nonlinear*
  weight (`1/(1−hᵢᵢ)`, `1/(1−hᵢᵢ)²`). That cannot be reduced to any fixed set of
  moments — you need `xᵢ` for every row, *after* you've inverted `X'X`.
- **Cluster-robust** (the planned follow-up) needs the per-cluster score sums
  `Σ_{i∈g} xᵢêᵢ` — again two-pass, again per-row.

So the state **buffers the rows** (`y` and a row-major `x`), exactly like
DuckDB's own `quantile`/`list` aggregates, and the fit + sandwich happen in
Finalize on the full data. Memory is O(n·k) per group — the same thing the `lm`
table function already does via `MaterializeYX`, and bounded per-group in a
`GROUP BY`. Don't try to be clever with moments here; you'd buy back a little
memory and lose HC2/HC3 entirely.

Consequence: the state owns a heap buffer, so unlike `ttest` it needs a
`destructor` callback (the `anova` pattern), and `Combine` concatenates the
buffers.

## 2. First `LIST<STRUCT>` output from an aggregate

`coefficients` is a `LIST<STRUCT(term, estimate, std_error, t_statistic,
p_value)>` — one tidy row per term, so `unnest(.., recursive := true)` gives a
coefficient table. No existing aggregate here returns a list-of-structs; the
nearest prior art is `bootstrap` (a flat `LIST<DOUBLE>`). The construction in
Finalize is two levels of the same idiom:

1. `StructVector::GetEntries(result)` → the top STRUCT's field vectors; child 0
   is the `coefficients` LIST vector.
2. `ListVector::Reserve(list, anchor + total_coefs)`, **then** `GetEntry(list)` →
   the child is itself a STRUCT vector; `StructVector::GetEntries(child)` gives
   the five per-coefficient field vectors.
3. Fetch the field data pointers **after** `Reserve` (it can reallocate), set
   each row's `list_entry_t{offset, k}`, write fields at `offset+j`, then
   `ListVector::SetListSize(list, total)`.

NaN → SQL NULL per field via `FlatVector::SetNull` on the child field vector (a
zero-variance coefficient has a NULL `t`/`p`), matching `lm`'s `SetDoubleOrNull`.
A degenerate group (n ≤ k, or singular `X'X`) sets the whole struct NULL and an
empty `{offset, 0}` list entry — never throws, so one bad group doesn't abort a
`GROUP BY`.

## 3. Aggregates don't get `:=` named parameters

`lm('t', y := 'mpg', x := ['wt'])` works because **table functions** have a
`named_parameters` map. Aggregate functions don't — their only knob is
positional, foldable constant arguments extracted at bind and erased from the
signature (`Function::EraseArgument`, the `ttest`/`bootstrap` pattern). Hence
`lm_fit(y, x, 'HC1', false)` is positional, and `vcov`/`add_intercept` are
overload-distinguished constants, not keywords.

This is also *why* `lm_fit`'s coefficients are by position while `lm`'s are
named: the aggregate receives a `LIST(DOUBLE)` of predictor **values** per row
and never sees the column names, so it can only label terms `(Intercept)`, `x1`,
`x2`, … The table function receives the names as strings and labels accordingly.

## Validation note

statsmodels (`OLS(...).fit(cov_type='HC*', use_t=True)`) is the oracle — matching
it validates the HC formulas, not just our re-derivation of them. The primary
fixture (`test/cpp/gen_lm_fit_fixtures.py`, dataset DS1) carries a single
high-leverage point, which makes the four HC estimators diverge sharply (x1's SE
runs 0.126 → 0.145 → 0.234 → 0.475 across HC0→HC3). A bug in any one leverage
weighting fails loudly there instead of hiding behind near-identical numbers.

The math lives in a DuckDB-free `lm_core` (built on `linalg.hpp`) so it is unit-
tested directly (`test/cpp/test_lm_fit.cpp`) and can later back `lm`/`lm_summary`
too — the aggregate is a thin vector-plumbing wrapper over it.
