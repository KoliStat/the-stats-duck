# `lin_hyp` — general linear hypothesis (#20) design

**Status:** approved in conversation 2026-10-14; awaiting spec review before an
implementation plan is written.

**Goal:** a stateless SQL function that tests a general linear hypothesis
`Lβ = 0` against an already-computed fit, returning the per-row estimates with
their standard errors and the joint test. It serves three callers: SAS-style
CONTRAST and ESTIMATE, Type III sums of squares (the caller builds `L` to select
a term's columns), and `emmeans` (#21), which needs exactly this sandwich.

**Out of scope, deliberately:** see [Non-goals](#non-goals). Type I and II sums
of squares are *not* part of this issue.

## Why stateless

Type III for a term is the Wald test of that term's coefficients in the full
model, so it follows from `β` and `cov(β)` alone. Type I is sequential and
Type II conditions on the terms that do not contain a given term; both need
nested refits or the design matrix, which a function handed only `β` and `cov`
cannot reconstruct. Splitting them keeps this function honest about what it can
compute, and matches the issue's own "stateless; kolilang builds L".

## SQL surface

```sql
lin_hyp(L, beta, cov, sigma2, df) -> STRUCT
```

| Argument | Type | Meaning |
|---|---|---|
| `L` | `LIST(LIST(DOUBLE))` | the hypothesis matrix, q rows each k wide; q rows means a joint test of q constraints |
| `beta` | `LIST(DOUBLE)` | the coefficients, length k |
| `cov` | `LIST(DOUBLE)` | `cov(β)`, flat row-major, length k², exactly as `lm_fit` returns it |
| `sigma2` | `DOUBLE` | residual variance, used only to turn the Wald statistic into a sum of squares |
| `df` | `DOUBLE` | denominator degrees of freedom for the t and F references |

`df` is `DOUBLE`, not `BIGINT`, so a fractional Satterthwaite df fits later
without a new overload; `lm_fit`'s `BIGINT df_residual` passes in by implicit
cast. **This differs from the shape shown during the design conversation**,
where `df_den` was sketched as `BIGINT`; echoing a `DOUBLE` input as `BIGINT`
would lose information, so both are `DOUBLE`.

The worked call, which is the example the function will register:

```sql
-- Type III test on x2, from an lm_fit result
SELECT lin_hyp(
         [[0, 0, 1]],                                     -- L: 1 × k
         list_transform(m.coefficients, c -> c.estimate), -- beta
         m.cov,                                           -- flat row-major k²
         m.sigma * m.sigma,                               -- sigma²
         m.df_residual)
FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM t);
```

### Return STRUCT

Field order is part of the contract and append-only, per
`docs/fitter_conventions.md`.

| # | Field | Type | Notes |
|---|-------|------|-------|
| 0 | `estimates` | `LIST<STRUCT(estimate, std_error, t_statistic, p_value DOUBLE)>` | one element per row of `L`, in `L`'s order |
| 1 | `ss` | `DOUBLE` | hypothesis sum of squares, `chisq · sigma2` |
| 2 | `df_num` | `BIGINT` | rank of `L·V·Lᵀ`; equals q when `L` has full row rank |
| 3 | `df_den` | `DOUBLE` | the `df` argument, echoed |
| 4 | `f_statistic` | `DOUBLE` | `chisq / df_num`, referenced to F(`df_num`, `df_den`) |
| 5 | `f_p_value` | `DOUBLE` | |
| 6 | `chisq` | `DOUBLE` | the Wald statistic, referenced to χ²(`df_num`) |
| 7 | `chisq_p_value` | `DOUBLE` | |

Conventions this satisfies: one STRUCT per call; append-only field order; NaN
mapped to SQL NULL at the vector layer; a t-referenced statistic named
`t_statistic` rather than a bare `statistic`; p-values from the in-repo
distribution kernels.

## Numerics

With θ = `L·β` and C = `L·V·Lᵀ` (q × q, the covariance of θ):

- per row i: `std_error` = √C_ii, `t_statistic` = θ_i / √C_ii,
  `p_value` = 2·(1 − T_cdf(|t|, `df`));
- `chisq` = W = θᵀ C⁻ θ;
- `df_num` = rank(C);
- `f_statistic` = W / `df_num`, `f_p_value` from F(`df_num`, `df`);
- `chisq_p_value` from χ²(`df_num`);
- `ss` = W · `sigma2`.

**C is equilibrated, then inverted with `pinv`, always, on one code path.**
`linalg::pinv` returns the pseudo-inverse together with the numerical rank, so a
single call yields both C⁻ and `df_num`, and redundant constraints need no
special case. For a full-rank C the pseudo-inverse is the inverse, so nothing is
lost. `inv_spd` is deliberately unused here: a second path would have to be
tested for no gain at these sizes.

The equilibration is not optional, and this spec got it wrong first: the
sentence above originally said "inverted with `pinv`, always", and said nothing
about conditioning. `linalg::pinv` decides the numerical rank by comparing each
singular value against `1e-12 · σ_max`, a tolerance relative to the *largest*
singular value. C's diagonal holds the constraints' variances, so two
constraints whose variances are more than about 1e12 apart — standard errors
more than about 1e6 apart — make a genuinely full-rank C look rank-deficient. A
real direction is then dropped, and both `df_num` and the F statistic come back
wrong with nothing to tell the caller. A change of units alone reaches that: on
DS1 the joint test of both slopes gives `df_num` 2 and F 132.5324744, and the
same test on `y ~ x1·1e3 + x2·1e-3` gave `df_num` 1 and F 9.4662513. So, before
the `pinv` call, with D = diag(1/√C_ii):

- C is symmetrized — only the symmetric part has a meaning in a quadratic form,
  and the matrix product leaves a ~1e-17 asymmetry behind anyway;
- C is scaled to unit diagonal, giving a correlation matrix D·C·D, and
  C⁻ = D·(D·C·D)⁻·D, which is exact when C has full rank;
- θ is scaled to match, so W = (Dθ)ᵀ(D·C·D)⁻(Dθ).

Both the rank and W are then invariant to a per-row rescaling of `L`. Where
C_ii ≤ 0 the scale factor is left at 1, so a zero row of `L` still behaves: that
row and column of C are zero, and `pinv` drops the direction as it should. The
per-row `std_error` keeps reading the **unscaled** C_ii, because √C_ii is the
reported standard error of that constraint.

**`emmeans` (#21) reaches for the same `pinv` on the same kind of matrix and
needs the same step.** The postmortem is
`notes/engineering/2026-10-lin-hyp-rank-tolerance-and-scale.md`.

This matches the oracle, verified rather than assumed (probe run 2026-10-14 on
DS1): statsmodels' `wald_test` given the redundant `L` = `[[0,1,-1],[0,-1,1]]`
reports `df_num` 1, not 2, and an F of 81.19250894 — the same F as the single
contrast `[[0,1,-1]]`, which is what rank 1 means. A design that used q as the
numerator df would disagree with statsmodels there.

Where `std_error` is zero the t statistic and p-value are NaN, following
`lm_core`'s existing rule, and therefore NULL in SQL.

### `ss` under a robust covariance

With an HC or CR covariance, `f_statistic` and `chisq` are a Wald test — the
same quantity statsmodels' `wald_test` reports — and `ss` is **not** a classical
partition of variance. It is still computed as W · `sigma2`, and the
documentation says plainly that it is a sum of squares only when `cov` is the
classical σ²(XᵀX)⁻¹. The alternative, a nullable `sigma2` that suppresses `ss`,
would need `FunctionNullHandling::SPECIAL_HANDLING` and hand-validation of NULLs
inside every list argument; that is not worth its weight until a caller asks.

## Failure behavior

Three outcomes, distinguished because SQL must treat them differently.

**Not estimable → NULL STRUCT.** If the hypothesis cannot be evaluated, the
whole STRUCT is NULL. This is SAS's "non-estimable", and per the fitter standard
a data-shaped failure is a value, never an exception. Five cases reach it: an
`L` that puts non-zero weight on a coefficient the fit could not estimate, a
non-finite entry surviving in θ or C for any other reason, a fully zero `L`
(rank(C) = 0, a hypothesis carrying no information), a strictly negative C_ii,
and a negative Wald statistic W.

The first case needs care, and it is the one that matters in practice now that
#36 has landed. A rank-deficient fit carries NaN in the rows and columns of its
aliased coefficients. A hypothesis that puts **no** weight on an aliased
coefficient is still perfectly estimable — testing x1 in a fit where x3 was
dropped is an ordinary test — but IEEE arithmetic makes `0 · NaN = NaN`, so a
naive `L·V·Lᵀ` would poison every such product and report the whole
rank-deficient fit untestable. So the aliased set is identified first, as
`{j : V(j,j) is not finite}`, and:

- if any `L(i,j) ≠ 0` for an aliased j, the hypothesis is not estimable;
- otherwise those rows and columns of `V`, and those entries of `β`, are zeroed
  in a working copy before the product. They carry zero weight by construction,
  so zeroing changes no result and keeps the arithmetic finite.

θ and C are then checked for non-finite entries anyway, which catches a NaN
arriving from anywhere else, such as an off-diagonal NaN with finite diagonals.

**A `cov` caught not being a covariance is also not estimable.** This spec
invites a hand-built covariance and the kernel is a public API a sibling
repository can consume, so the input is not always `lm_fit`'s positive
semi-definite output. Two guards, in the same spirit as the non-finite rule
above, neither of them a full positive-definiteness test:

- a strictly negative C_ii is a negative variance, so the hypothesis cannot be
  evaluated. C_ii = 0 stays estimable: a zero row of `L` legitimately produces
  one;
- a negative W is the other proof. An indefinite C can pass the diagonal check
  and still give a negative quadratic form. `cov` = [[−1,0],[0,1]] with `L` = I
  and `beta` = [1,2] used to report `df_num` 2, `chisq` 3, F 1.5 and p 0.274 —
  nothing in that output looks wrong, and it came from a negative variance.

Neither raises. By the taxonomy here "this is not a covariance" is data-shaped,
and raising would abort a whole `GROUP BY` for one bad group. The two guards
are sufficient conditions, not a complete test: an indefinite C whose diagonal
is non-negative and whose quadratic form still comes out positive is evaluated
normally. Catching every such matrix needs an eigendecomposition, which this
spec does not ask for, so a consumer needing the stronger guarantee should
check its own input. An asymmetric
`cov` is not a third case: the symmetrization described under Numerics means the
symmetric part is what is tested, deterministically, rather than whichever half
the matrix product happened to favour.

**NULL list elements are read as NaN**, so they reach the same check and produce
a NULL result. One rule covers both, with no separate validity handling.

**Misuse → runtime error.** A ragged `L`, an `L` row that is not k wide, a `cov`
whose length is not k², a non-positive `df`, or a non-finite `sigma2` raises
`InvalidInputException` naming the expected and actual sizes. List lengths are
only knowable per row at execution time, so this cannot be a bind-time error;
that is the one place this function departs from the standard's
"misuse is a bind-time error", and the reason is recorded here.

A top-level NULL argument yields a NULL result, but **not** through DuckDB's
default null handling: that only lets the binder fold a NULL which is constant
at bind time. A NULL arriving in a column still reaches the executor, so the
wrapper tests every argument's validity per row and skips the row. The function
does not declare `SPECIAL_HANDLING`, and that per-row check is what produces the
NULL; removing it would read whatever the invalid slot happens to hold.

## Structure

| File | Role |
|---|---|
| `src/include/lin_hyp.hpp` | DuckDB-free API: `LinHypResult`, `lin_hyp(L, beta, V, sigma2, df)` |
| `src/lin_hyp.cpp` | the numerics, on `linalg` and `distributions` only |
| `src/include/lin_hyp_function.hpp` | `RegisterLinHyp` declaration, house pattern |
| `src/lin_hyp_function.cpp` | the SQL wrapper: unpack lists, call the kernel, write vectors |
| `test/cpp/test_lin_hyp.cpp` | standalone goldens, no DuckDB |
| `test/cpp/gen_lin_hyp_fixtures.py` | the offline statsmodels generator |
| `test/sql/lin_hyp.test` | the SQL surface |

The kernel depends on `linalg.hpp` and `distributions.hpp` and **not** on
`lm_core.hpp`: the whole point is that it works with any fit, including a future
`glm_fit` or a hand-built covariance. It goes in its own translation unit rather
than into `lm_core` because it has one clear purpose and no shared state with
fitting.

`LinHypResult` reports its own outcome rather than throwing:

```cpp
struct LinHypResult {
    bool ok = false;         // false → misuse; `error` says what
    std::string error;
    bool estimable = false;  // ok but not estimable → the SQL layer emits NULL
    std::vector<double> estimate, std_error, t_statistic, p_value; // length q
    double ss = 0.0;
    std::size_t df_num = 0;
    double df_den = 0.0;
    double f_statistic = 0.0, f_p_value = 0.0;
    double chisq = 0.0, chisq_p_value = 0.0;
};
```

Build wiring: `src/lin_hyp.cpp` and `src/lin_hyp_function.cpp` join the source
list in `CMakeLists.txt`; `RegisterLinHyp(loader)` joins
`src/stats_duck_extension.cpp`; `scripts/run-cpp-tests.sh` gains a `test_lin_hyp`
target compiling `test_lin_hyp.cpp` + `lin_hyp.cpp` + `linalg.cpp`.

## Validation

statsmodels is the oracle and has exactly these three calls:
`OLSResults.t_test(R)` for the per-row table,
`OLSResults.wald_test(R, use_f=True)` for the joint F and
`OLSResults.wald_test(R, use_f=False)` for the same W read against chi-square,
which is where `chisq_p_value`'s golden comes from. The generator runs offline
in a throwaway virtualenv and is never a build or CI dependency.

Fixtures, all on DS1 (`test/cpp/test_lm_fit.cpp`), whose fit is already validated
so only the hypothesis layer is under test:

| Case | `L` | What it pins | Oracle value (DS1) |
|---|---|---|---|
| single coefficient | `[[0,1,0]]` | the Type III test of one term; q = 1 | F = 249.4063533; per-row estimate 2.057940123 and standard error 0.1303103701, which **are DS1's existing `beta[1]` and `se[1]` goldens** |
| contrast | `[[0,1,-1]]` | ESTIMATE-style difference of two coefficients | F = 81.19250894 |
| joint | `[[0,1,0],[0,0,1]]` | q = 2; spanning all slopes must reproduce the model F | F = 132.5324744, which **is DS1's existing `f_statistic` golden** |
| redundant | `[[0,1,-1],[0,-1,1]]` | `df_num` = 1 from the rank of C, not 2 | F = 81.19250894, equal to the single contrast |
| robust | `[[0,1,0]]` with the HC1 covariance | the sandwich carries through unchanged | F = 201.5619805 |
| weak | `[[0,1,2]]` | `chisq_p_value` against the chi-square oracle. Every case above has a chi-square tail below 1e-16, where `1 − CDF` returns 0, so the golden needs a hypothesis the data does not reject | `chisq` = 0.02350072355, `chisq_p_value` = 0.8781621288 |

Two of these tie back to numbers already in `test/cpp/test_lm_fit.cpp`, which is
deliberate: if the hypothesis layer is right, a single-coefficient test must
reproduce that coefficient's own estimate and standard error, and a test
spanning every slope must reproduce the model F. Those two cases need no new
constants at all.

Generator self-checks, asserted on every run so a statsmodels upgrade that moved
a convention fails loudly: `chisq` = `f_statistic` · `df_num`; the chi-square
call reports the same W the F call scales; for q = 1 the joint F equals t²; for
the classical covariance `ss` = `chisq` · σ²; and rescaling a predictor by 1e3
or 1e6 moves neither `df_num` nor the joint F, which makes the invariance an
oracle fact rather than our own claim.

C++ tests add the structural cases a numeric oracle cannot express: a `cov`
carrying NaN in an aliased row is not estimable; an off-diagonal NaN with finite
diagonals is not estimable; a ragged `L` and a mis-sized `cov` are errors; a
zero `L` is not estimable; a negative variance, an indefinite `cov` and an
asymmetric `cov` behave as the failure section says; and the joint test is
unchanged when the predictors are rescaled 1e3 and 1e-3.

SQL tests cover the worked call above against a real `lm_fit` result, the
aliased-coefficient NULL (using a collinear design, which fits since #36 — this
doubles as an integration test of the two features), one error case, every field
of the result STRUCT read at least once, and the rescaled joint test through
`lm_fit` — which is the path the scale bug was found on.

## Documentation

README gains a `lin_hyp` subsection after the `lm_fit` one with the worked
example and the `ss` caveat; `docs/kernel_api.md` gains a `lin_hyp.hpp` section,
since this is a kernel API a sibling repo can consume; the CHANGELOG gains an
Added entry. The function registers its description, parameter names and a
runnable example through `RegisterDocumented`, which
`test/sql/function_docs.test` enforces.

`docs/fitter_conventions.md` already says post-fit helpers are scalar functions
consuming a fit's outputs, so it needs no change; `lin_hyp` simply complies.

## Non-goals

- **Type I and II sums of squares.** They need nested refits and the term-to-column
  mapping, so they belong to a data-seeing function. A separate issue, filed when
  this lands.
- **A non-zero right-hand side (`Lβ = c`).** Testing β₁ = 1 rather than 0 is a
  real use and costs three lines (θ = Lβ − c), but it is not in the issue's
  signature and nothing needs it yet. It arrives as a trailing argument when
  something does, which reinterprets no existing position.
- **A fit-struct convenience overload** (`lin_hyp(m, L)`). Considered during the
  design conversation and declined: it would couple the function to `lm_fit`'s
  STRUCT, which defeats the point of a stateless helper usable with `glm_fit` or
  a hand-built covariance. Revisit once #21 and kolilang show the real call sites.
- **`emmeans` (#21).** A separate issue that consumes this one.
