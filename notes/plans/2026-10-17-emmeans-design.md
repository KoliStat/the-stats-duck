# emmeans design (#21)

Date: 2026-10-17. Status: the binding specification for #21.

`emmeans` reports estimated marginal means for a fit you already have, with
their standard errors and confidence limits, plus every pairwise difference
between them and a multiplicity adjustment for that family. It is what SAS
prints as LSMEANS with the DIFF and CL options, and what R's `emmeans`
package reports for a reference grid followed by `pairs()`.

This document is the authority for the implementation plan. Where the plan
and this document disagree, this document is right, and the plan should be
corrected rather than reinterpreted.

## Scope

In scope: the marginal means of q supplied reference-grid rows, all
q(q−1)/2 pairwise differences, t-based inference against a caller-supplied
denominator degrees of freedom, confidence limits, and six multiplicity
adjustments including Tukey.

Out of scope, deliberately:

- **Building the reference grid.** The caller supplies `L`. Constructing a
  grid from a model formula needs a term model and the factor levels, which
  a function taking only β and cov(β) cannot reconstruct. The same reasoning
  keeps Type I and II sums of squares out of `lin_hyp` (#65).
- **Satterthwaite or Kenward-Roger degrees of freedom.** One `df` applies to
  every comparison. Per-contrast df belongs with the mixed-model fitters in
  the downstream repository.
- **Weighted marginal means.** Averaging over a factor with weights other
  than equal ones is a property of the grid, so it belongs in whatever
  builds `L`.
- **A joint test across the grid.** The test that all means are equal is a
  hypothesis on the full model, so `lin_hyp` already computes it. Note the
  consequence in Numerics below: `emmeans` inverts nothing.

## The SQL surface

```
emmeans(L, beta, cov, df [, alpha [, adjust]]) -> STRUCT
```

A scalar function. `docs/fitter_conventions.md` already states that post-fit
helpers are scalar functions over a fit's outputs, while fitters themselves
are aggregates.

| Argument | Type | Notes |
| --- | --- | --- |
| `L` | `LIST(LIST(DOUBLE))` | q rows, each k wide. One row per marginal mean. |
| `beta` | `LIST(DOUBLE)` | length k, in the fit's coefficient order. |
| `cov` | `LIST(DOUBLE)` | flat row-major k², exactly as `lm_fit` returns it. |
| `df` | `DOUBLE` | denominator degrees of freedom, > 0. `lm_fit`'s `df_residual` casts in. |
| `alpha` | `DOUBLE` | confidence level complement, default 0.05, in (0, 1). |
| `adjust` | `VARCHAR` | default `tukey`. One of `tukey`, `bonferroni`, `holm`, `hochberg`, `BH` (or `fdr`), `BY`, `none`. |

There is no `sigma2` argument. `lin_hyp` takes one only to scale its
hypothesis sum of squares, and `emmeans` has no sum of squares, so the
position does not exist. The two signatures therefore differ by more than
their last argument, which is worth stating because they otherwise look like
siblings.

`L` stays a nested list rather than a flat row-major one. The nesting costs
16 bytes per row and the function is called once per fit, so at realistic
sizes — a handful of means against a few dozen coefficients — the overhead
is a few hundred bytes against arithmetic that dominates it. Readability
decides it instead: `[[0,1,0],[0,0,1]]` reads as a matrix, while a flat list
has to be chunked by eye. The rule that follows, and which belongs in
`docs/fitter_conventions.md`: **a matrix a machine produces goes flat, a
matrix a human writes goes nested.** `cov` is flat because `lm_fit` produces
it. Break the rule for a matrix argument evaluated once per data row, or one
large enough for offsets to matter. Neither applies here.

DuckDB infers `[[1.0, 2.0]]` as `DECIMAL(2,1)[][]` and an integer grid as
`INTEGER[][]`, so the binding accepts and casts rather than demanding
DOUBLE, as `lin_hyp` already does.

### The result STRUCT

Field order is part of the contract and is append-only, per
`docs/fitter_conventions.md`. The type function carries the comment
`// Field order MUST match the writes in Finalize` with an index comment per
field.

| # | Field | Type |
| --- | --- | --- |
| 0 | `means` | `LIST<STRUCT(estimate, std_error, t_statistic, p_value, lower_cl, upper_cl DOUBLE)>` |
| 1 | `diffs` | `LIST<STRUCT(i BIGINT, j BIGINT, estimate, std_error, t_statistic, p_value, p_adjusted, lower_cl, upper_cl DOUBLE)>` |
| 2 | `df_den` | `DOUBLE` |
| 3 | `alpha` | `DOUBLE` |
| 4 | `adjust` | `VARCHAR` |
| 5 | `k_compare` | `BIGINT` |

`means` follows `L`'s row order. It carries no index field because the list
position is the index.

`diffs` enumerates pairs with i < j, row-major: (1,2), (1,3), … (1,q),
(2,3), … and so on. `i` and `j` are 1-based, so they index `means` directly
in SQL, where lists are 1-based.

`df_den`, `alpha` and `adjust` are echoed so a result is self-describing,
following `lm_fit`'s `vcov_type`. `adjust` echoes the canonical name, so
`fdr` comes back as `BH`.

`k_compare` is the number of estimable means. It is the family size the
adjustment used, so a reader can reproduce the adjusted p-values. It is not
the number of pairs.

### What each field means

A mean is θᵢ = ℓᵢ·β, where ℓᵢ is row i of `L`.

- `estimate` is θᵢ, and `std_error` is √C_ii where C = L·cov·Lᵀ.
- `t_statistic` is θᵢ divided by `std_error`, and `p_value` is the two-sided
  probability against t(`df`). This tests the hypothesis θᵢ = 0, which is
  what SAS prints for an LSMEAN. Where `std_error` is 0 — a zero row of `L`
  produces one — `t_statistic` and `p_value` are NaN, and so NULL, while the
  estimate and the limits stay valid.
- `lower_cl` and `upper_cl` are θᵢ ± t(1 − α/2, df) · `std_error`. **They
  are never adjusted**, whatever `adjust` says. SAS does the same: ADJUST
  governs the comparisons, not the means.

A difference is θᵢ − θⱼ with variance C_ii + C_jj − 2·C_ij.

- `p_value` is the unadjusted two-sided probability against t(`df`), and it
  is always present, so no adjustment hides the raw number.
- `p_adjusted` is `p_value` after the chosen method.
- `lower_cl` and `upper_cl` are adjusted when the method has a matching
  simultaneous interval, and unadjusted otherwise. See the table below.

## Adjustment methods

Tukey and Bonferroni control the family-wise error rate with a matching
simultaneous interval, so they adjust both the p-value and the limits. Holm
and Hochberg control the family-wise rate as step-down procedures with no
corresponding interval. Benjamini-Hochberg and Benjamini-Yekutieli control
the false discovery rate, which is a different guarantee and also has no
interval here. R's `emmeans` behaves the same way and warns that the limits
are unadjusted for those methods.

| `adjust` | `p_adjusted` | Limits |
| --- | --- | --- |
| `tukey` | 1 − ptukey(√2·abs(t), k_compare, df) | multiplier qtukey(1 − α, k_compare, df) / √2 |
| `bonferroni` | min(1, m · p) | multiplier t(1 − α/(2m), df) |
| `holm`, `hochberg` | step-down on the estimable p-values | unadjusted t(1 − α/2, df) |
| `BH` (`fdr`), `BY` | step-up on the estimable p-values | unadjusted t(1 − α/2, df) |
| `none` | equals `p_value` | unadjusted t(1 − α/2, df) |

`m` is the number of estimable pairs **that have a finite p-value**, not
q(q−1)/2 and not quite the count of estimable pairs either. The distinction
only bites when a grid repeats a level: such a pair has a zero variance, so
it is estimable and keeps its estimate, but its t and p are NaN. A
comparison with no p-value is not a member of a multiplicity family, so it
is excluded from `m`.

That exclusion is also forced rather than chosen. The shared adjustment
helper sorts the p-values to rank them, and a NaN makes every comparison
against it false, which breaks the transitivity a strict weak ordering
requires — so passing one in is undefined behaviour, not merely untidy. The
helper's own Bonferroni count is the size of its valid set, so keeping a
wider `m` for the limits would report `min(1, 3p)` beside a
`t(1 − α/6)` interval, which is self-inconsistent.

One consequence to document rather than fix: `m` is therefore not always
recoverable from the result. A non-estimable pair is a NULL element of
`diffs`, so counting the non-NULL elements gives `m` in every ordinary case;
it overcounts only for the degenerate zero-variance pair above. The
documentation says so rather than the result carrying a seventh field for
a case a caller reaches by asking for the same mean twice.

The adjustment runs over that set only. Including the excluded slots would
change the ranks that the step-down and step-up procedures depend on.
`AdjustInPlace` in `src/adjust_p_function.cpp` already takes a `valid` mask
and writes NaN into the invalid slots, which is exactly this behaviour, and
is why it is worth extracting rather than reimplementing.

With fewer than two estimable means there are no comparisons. `diffs` is an
empty list when q = 1, and a list of NULL elements when pairs exist but
cannot be evaluated. Adjustment is then a no-op.

### Two honest limits on Tukey

**It is Tukey-Kramer whenever the standard errors differ.** The studentized
range is exact for k means with a common variance and equal group sizes.
With unequal standard errors the same multiplier is applied to each pair's
own standard error, which is the Tukey-Kramer approximation and is
conservative. SAS and R's `emmeans` both do this. The documentation must say
so rather than implying exactness.

**It assumes the rows of `L` are means.** The studentized range describes
the spread of a set of means, so the adjustment is meaningful because the
family is all pairwise differences among the q rows. A caller who passes
rows that are themselves contrasts still gets arithmetic, but the
family-wise interpretation no longer holds. The documentation says this; the
code does not attempt to detect it, because it cannot.

## Semantics: three outcomes

The kernel follows `lin_hyp`'s three-outcome contract, which
`docs/fitter_conventions.md` calls failure-is-a-value.

**Misuse raises.** The query fails before it runs, or at the row, with a
`BinderException` for a constant-foldable problem:

- a row of `L` whose width is not k, including a ragged `L`;
- an `L` with zero rows, since a grid of no means is a request for nothing;
- a NULL **row** of `L`. A NULL row carries no width, so it cannot be read
  as k NaNs without inventing k: it is a malformed matrix rather than a
  missing value;
- `cov` whose length is not k²;
- a `beta` of length 0, since there is nothing to estimate;
- `df` not greater than 0, or not finite;
- `alpha` outside (0, 1), or not finite;
- an unrecognized `adjust` name.

Those first three match `lin_hyp` exactly, which raises on the same three
shapes with the same reasoning. `test/sql/lin_hyp.test` pins all of them.
Two neighbouring functions answering the same malformed input differently is
the outcome `docs/fitter_conventions.md` exists to prevent.

**A data-shaped failure is a NULL.** It never raises:

- a NULL in any argument position makes the whole STRUCT NULL, which is
  DuckDB's ordinary scalar behaviour;
- a NULL element **inside** a row of `L`, or inside `beta` or `cov`, reads
  as NaN. The non-finite checks then make the affected means not estimable,
  so those elements of `means` are NULL and every pair touching them is
  NULL, while the rest of the grid is reported normally. This is `lin_hyp`'s
  rule given the per-row granularity `emmeans` has: one unusable grid row
  must not discard the others. A NULL on `cov`'s diagonal is indistinguishable
  from the NaN a rank-deficient fit puts there, and that is correct — both
  mean the coefficient was not estimated;
- a mean that cannot be evaluated is a NULL element of `means`;
- a pair involving such a mean is a NULL element of `diffs`.

The lists keep their length and their positions when elements go NULL, so
`i` and `j` stay valid indices into `means`.

**A mean is not estimable when** its row of `L` puts non-zero weight on an
aliased coefficient, or `C_ii` is negative past the tolerance defined under
Numerics, or θᵢ or `C_ii` is not finite after the aliased zeroing.

A rank-deficient fit carries NaN in the rows and columns of its dropped
coefficients (#36). Those rows and columns are zeroed in a working copy
before the products, because IEEE makes 0 · NaN = NaN, so a grid row that
avoids the aliased coefficients is evaluated normally. This is `lin_hyp`'s
rule, and the shared helper below is where it will live for both.

## Numerics

**One sandwich produces everything.** θ = L·β through `linalg::matvec`, and
C = L·cov·Lᵀ through `linalg::sandwich`. Every reported number is then an
entry of θ or C: a mean's variance is C_ii, and a difference's variance is
C_ii + C_jj − 2·C_ij.

**Nothing is inverted.** There is no `pinv`, no equilibration and no rank
tolerance in this function, because no joint test is formed.

This corrects a prediction in
`notes/plans/2026-10-14-lin-hyp-design.md`, which says that any kernel
inverting a covariance with `pinv` needs the equilibration step and names
`emmeans` as the next such consumer. On this design `emmeans` is not one.
The claim about `pinv` itself stands, and the bug it describes was real: a
change of units alone moved `df_num` and the F. It would apply here only if
`emmeans` later gained a joint test across the grid.

**The symmetric part is what is used.** Where `cov` is asymmetric, C is too,
so C_ij is taken as (C_ij + C_ji)/2. This matches `lin_hyp`, which
symmetrizes for the same reason: only the symmetric part has a meaning in a
quadratic form, and a deterministic choice beats whichever half the matrix
product happened to favour.

**Negative variances need a relative tolerance, and the scale differs by
case.** Both a mean's variance and a difference's variance are computed
sums, so both can land slightly below zero from rounding when the true value
is zero. A difference is the worse case: C_ii + C_jj − 2·C_ij cancels three
terms of similar size, so it is reachable whenever two marginal means nearly
coincide, not only for degenerate inputs. One rule with two scales, because
a tolerance needs something to be relative *to*:

- **a mean.** Its variance is C_ii, and the scale is the largest diagonal
  entry of C, written s = max over j of C_jj. A C_ii that is negative but
  within `tol · s` counts as zero. Below that the mean is not estimable.
  When s is itself not positive there is no scale to work with, so any
  negative C_ii makes the mean not estimable;
- **a difference.** Its variance is C_ii + C_jj − 2·C_ij, and the scale is
  C_ii + C_jj, which is the sum of the two parts that cannot cancel. A
  variance that is negative but within `tol · (C_ii + C_jj)` counts as zero.
  Below that the pair is not estimable;
- **a variance counted as zero keeps its estimate.** `std_error` is 0, with
  a NaN `t_statistic` and a NaN `p_value`, matching how a zero row of `L`
  behaves in `lin_hyp`. Only the cases past the tolerance become NULL;
- `tol` is 1e-12, four orders of magnitude above the double epsilon. That is
  enough to absorb cancellation among a few terms of similar size, while
  still rejecting an indefinite `cov`, whose negative quadratic forms are
  not that small. `cov = [[-1, 0], [0, 1]]` must still be rejected.

#66 proposes the same treatment for `lin_hyp`, so that the two agree.

**Determinism.** Fixed loop orders, no hashing, no random numbers, and fixed
node counts in the quadrature below, so the algorithm itself contributes no
platform dependence of its own, which is what `docs/fitter_conventions.md`
actually requires. That is not bit-identical results on every platform.
`std::erfc`, `std::exp`, `std::pow` and `std::cos` are not bit-identical
across libm implementations, and the wasm build uses a different one from
the desktop build. The quadrature can still differ in the last few bits
between platforms for that reason, not from anything this design chooses.

## The studentized range distribution

`distributions.hpp` gains `StudentizedRangeCDF(q, k, df)` and
`StudentizedRangeQuantile(p, k, df)`, exposed as the SQL scalars `ptukey`
and `qtukey`. The names match the existing R-style family — `pnorm`, `pt`,
`pchisq`, `qnorm` — and exposing them has a second purpose: the new numerics
get their own tests and their own SQL coverage, independent of `emmeans`.

The distribution of Q = (max − min) / S for k independent standard normals
and an independent S with ν degrees of freedom:

```
P(Q <= q; k, nu) = integral, s from 0 to inf, of  f_nu(s) * W(q*s; k) ds
W(w; k)          = k * integral over z of  phi(z) * [Phi(z) - Phi(z-w)]^(k-1) dz
```

where `f_nu` is the density of s = sqrt(chi-squared_nu / nu).

Implementation:

- both integrals by Gauss-Legendre with a fixed node count over a truncated,
  panelled range. The spec does not fix the node count, because the right
  number is whatever meets the tolerance band below with margin. The
  implementer chooses it by refining until the band is met on the whole test
  grid, then states the chosen value and that evidence in the header. A node
  count tuned to one case and asserted for the rest is the failure mode to
  avoid here;
- the inner integrand decays like phi(z), so truncation at ±8 is far past
  sufficient for the target accuracy;
- the outer integrand concentrates near s = 1 with spread about 1/sqrt(2ν),
  so the range is set from ν and panelled;
- above a ν cutoff the outer integral is replaced by its limit, W(q; k),
  which is the normal-range probability. R switches at ν around 25000, which
  is the right order of magnitude; the exact value is chosen so the
  discontinuity at the switch is below the CDF's own tolerance, and the
  implementation states the value, the measured jump at it, and the reason;
- `StudentizedRangeQuantile` is bracketed bisection on the CDF with a fixed
  iteration count, so the result does not depend on an adaptive stopping
  rule that could differ between platforms.

**A licence constraint.** The obvious reference implementation is R's
`ptukey`, which is GPL. This repository is Apache 2.0, so that code must not
be ported or adapted, in whole or in part. The implementation comes from the
published mathematics, and the header cites its references the way
`distributions.hpp` already cites Numerical Recipes and Acklam. The oracle
is `scipy.stats.studentized_range`, which is BSD licensed and is used
offline to generate goldens, never as a build or CI dependency.

Note for the implementer: `distributions.hpp` is in namespace `stats_duck`
with an underscore, while the kernel is `statsduck` without one, so calls
across that boundary need qualifying. `src/lin_hyp.cpp` already does this.

## Shared extractions

Three pieces are currently private to the `lin_hyp` and `adjust_p` files and
are needed again. Each moves to a shared home, and the existing consumer
moves onto the shared copy, so there is one implementation of each
convention. `lin_hyp`'s 113 existing C++ checks and 58 SQL assertions cover
those refactors, and `adjust_p` has its own SQL tests.

**`src/include/estimability.hpp`** — the aliased-coefficient rule, as
`aliased_columns(V)`, `zero_aliased(V, aliased)` and
`row_touches_aliased(row, k, aliased)`. DuckDB-free, namespace `statsduck`.
This is the convention #36 introduced, and it now has two consumers.

**`src/include/p_adjust.hpp`** — the `AdjustMethod` enum, `AdjustInPlace`,
and a `parse_adjust(name, out) -> bool` in the shape of
`lm_core::parse_vcov`, which returns false rather than throwing so the
caller chooses the exception. `idx_t` becomes `std::size_t` at the boundary.
`src/adjust_p_function.cpp` keeps only its binding and its own error
message.

**`src/include/list_read.hpp`** — reading a `LIST(LIST(DOUBLE))` into a
`linalg::Mat`, and a `LIST(DOUBLE)` into a vector, lifted from
`src/lin_hyp_function.cpp`. This one is DuckDB-coupled, so it belongs with
the function layer and not with the kernel. It exists because the nested
read walks two levels of offsets, and a nested list arriving from
`list_transform`, a join, or a dictionary-encoded vector can have rows that
are out of order or not contiguous. The walk must not assume layout. One
tested implementation is better than two.

## Validation

The oracle runs offline in a throwaway virtualenv and the goldens are
committed, per `docs/fitter_conventions.md`. Two generators:
`test/cpp/gen_tukey_fixtures.py` and `test/cpp/gen_emmeans_fixtures.py`.

### The studentized range

Oracle: `scipy.stats.studentized_range`, over k in {2, 3, 4, 5, 6, 10, 20}
and ν in {1, 2, 5, 10, 30, 60, 120, 1000}, in both directions — the CDF at
given q, and the quantile at p in {0.90, 0.95, 0.99}.

Four identities carry more weight than the committed constants, because they
check the new code against code this repository has already validated:

1. **k = 2 is the t distribution exactly:**
   ptukey(q, 2, ν) = 2·pt(q/√2, ν) − 1. This anchors the whole double
   integral against `StudentTCDF`.
2. **ν to infinity is the normal range,** and at k = 2 that is
   2·Phi(q/√2) − 1, against `NormalCDF`.
3. **Round trip:** qtukey(ptukey(q, k, ν), k, ν) = q.
4. **Monotone** increasing in q and decreasing in k.

A failure in identity 1 or 2 means the quadrature is wrong, not that a
constant drifted, which is the distinction that makes them worth having.

### The emmeans linear algebra

Oracle: statsmodels' `t_test`, the same oracle
`test/cpp/gen_lin_hyp_fixtures.py` already uses.

- each mean's estimate, standard error, t and p from `t_test(L_row)`;
- each pair's estimate and standard error from `t_test(L_i − L_j)`.

The second one matters most. The square root of C_ii + C_jj − 2·C_ij is the
single most likely thing to be implemented wrongly, and the oracle reaches
the same number by an entirely different route.

### Identities on the fixture itself

These need no oracle and no new constants:

- for a one-way cell-means design, each marginal mean equals that level's
  sample mean, and a pairwise difference equals the difference of sample
  means with the pooled standard error. A transposed sandwich or a wrong `L`
  convention fails here;
- with exactly two means, Tukey's `p_adjusted` equals the unadjusted
  `p_value` to within the studentized range's own accuracy, because there
  is one comparison and the studentized range at k = 2 reduces to
  √2·abs(t). The two numbers still come from different code paths
  (`StudentTCDF` for `p_value`, `StudentizedRangeCDF` for `p_adjusted`), so
  they agree to the quadrature's accuracy rather than bit for bit.
  Measured on the shipped function the gap is about 1e-13.
  `p_adjusted` can land very slightly below `p_value` rather than at or
  above it; that is the gap, not a meaningful ordering. This ties the
  distribution work and the linear algebra together in one assertion,
  loosely rather than exactly;
- `bonferroni` with one estimable pair leaves `p_value` unchanged;
- `none` gives `p_adjusted` equal to `p_value` for every pair.

### Tolerance bands and their reasons

`docs/fitter_conventions.md` requires a written reason for every band.

| Quantity | Band | Reason |
| --- | --- | --- |
| estimates, standard errors | 1e-10 relative | one matrix product and a square root, no iteration |
| `t_statistic` | 1e-10 relative | a ratio of two quantities already at that band |
| unadjusted `p_value` | 1e-8 absolute | inherits `StudentTCDF`, whose band is established |
| `ptukey` | 1e-8 absolute | fixed-node Gauss-Legendre on a smooth, exponentially decaying integrand; the error is set by the truncation point and the node count, both fixed, and scipy's own implementation targets about 1e-10 |
| `qtukey` | 1e-6 relative | inherits the CDF error divided by the density at the root, and that density is small in the upper tail, so the same absolute error becomes a larger relative one |
| `p_adjusted` under Tukey | 1e-7 absolute | follows `ptukey` |

### The SQL layer

`test/sql/emmeans.test` and `test/sql/tukey.test`, with floats printed
through `printf('%.4f', …)` so the expected output is deterministic text.

Beyond a worked three-level factor example:

- a non-estimable mean, and the NULL propagation into every pair that
  touches it, with the surrounding pairs still valid;
- q = 1, giving an empty `diffs` list rather than NULL;
- every `adjust` name reachable, including that `tukey` and `bonferroni`
  move the limits while `holm` and `BH` do not;
- `alpha` moving the limits;
- `k_compare` dropping when a mean is not estimable;
- each misuse case raising: a ragged `L`, a row width that is not k, a `cov`
  that is not k² long, a `df` of 0, an `alpha` of 1, an unknown `adjust`;
- `ptukey` against `pt` at k = 2 in SQL, so the identity is checked at both
  layers.

**A multi-row chunk test is required, not optional.** Several fits in one
query, with a non-estimable row in the middle, so that the rows emit
different numbers of list elements. Every other query in these files returns
a single row, so an offset-walk bug in the output lists would pass the
entire suite otherwise. This is the gap that nearly escaped in #20.

### The metadata net

`test/sql/function_docs.test` holds three `IN (...)` lists and a count
assertion, currently 96. Adding `emmeans`, `ptukey` and `qtukey` makes it
99, and all three names must appear in all three lists. In #20 this was
missed and then reported as unaffected, so the plan gives it its own
numbered step rather than a line inside a documentation step.

## Sequencing

This issue carries more than one deliverable, so the order matters and the
implementation plan should follow it.

1. **The three extractions first.** Each is a refactor of code that works
   and has tests. Landing them before anything consumes them means a
   regression shows up attached to the refactor that caused it, rather than
   mixed into new work where it reads as a bug in the new function.
2. **The studentized range second, as its own unit.** It is independently
   verifiable — against scipy, and against the `pt` and `NormalCDF`
   identities — and it ships its own SQL functions. Its tests must pass
   before `emmeans` is started, because every Tukey number in `emmeans`
   depends on it and debugging a wrong adjusted p-value is much harder when
   either layer could be the cause.
3. **The `emmeans` kernel third**, then its SQL surface, then the
   documentation and the metadata net.

The practical consequence: if the studentized range turns out to need more
work than expected, steps 1 and 2 are still a complete, reviewable
contribution on their own, and `emmeans` with the other five adjustments
could ship separately. That is the fallback, not the plan.

## Files

New:

- `src/include/estimability.hpp`
- `src/include/p_adjust.hpp`
- `src/include/list_read.hpp`
- `src/include/emmeans.hpp`, `src/emmeans.cpp`
- `src/include/emmeans_function.hpp`, `src/emmeans_function.cpp`
- `test/cpp/gen_tukey_fixtures.py`, `test/cpp/test_tukey.cpp`
- `test/cpp/gen_emmeans_fixtures.py`, `test/cpp/test_emmeans.cpp`
- `test/sql/emmeans.test`, `test/sql/tukey.test`

Modified:

- `src/include/distributions.hpp` — the studentized range
- `src/distribution_functions.cpp` — register `ptukey` and `qtukey`
- `src/lin_hyp.cpp` — onto `estimability.hpp`
- `src/lin_hyp_function.cpp` — onto `list_read.hpp`
- `src/adjust_p_function.cpp` — onto `p_adjust.hpp`
- `src/stats_duck_extension.cpp` — register `emmeans`
- `CMakeLists.txt` — `EXTENSION_SOURCES` lists sources explicitly
- `scripts/run-cpp-tests.sh` — two new suites
- `test/sql/function_docs.test` — the net above
- `README.md`, `docs/kernel_api.md`, `docs/fitter_conventions.md` (the
  matrix-argument rule), `CHANGELOG.md`

## Decisions already taken

- `adjust` defaults to `tukey`, following R's `emmeans`, where pairwise
  comparisons are Tukey-adjusted by default. SAS defaults to no adjustment.
  The unadjusted `p_value` is always present either way.
- `L` stays `LIST(LIST(DOUBLE))`, with the shared reader.
- `k_compare` is the count of estimable means, and it is echoed.
- Tukey is computed, not only documented, which is why the studentized range
  is part of this issue rather than a follow-up.
