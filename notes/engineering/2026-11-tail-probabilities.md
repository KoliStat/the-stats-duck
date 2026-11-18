# A p-value is an upper tail, not 1 - CDF

Date: 2026-11-05. Issue #81. Branch `fix/81-tail-pvalues`.

## What happened

`pnorm(-8.5)` returned 0. The true value is 9.5e-18. `pnorm(-8)` was 1.8 %
low, and `pnorm(-5)` had lost five digits. Every test and fitter in the
extension reported a `p_value` of exactly 0 for a strong effect: a one-way
ANOVA on the Palmer penguins (F = 342 on 2 and 330 df, p = 3.7e-81), a
logistic coefficient with z = -9.65 (p = 4.8e-22), the joint `lin_hyp` test in
`test/sql/lin_hyp.test` (W = 265 on 2 df, p = 2.8e-58). That last one had been
pinned as `0.0000` with a comment saying so, and the README documented the
limit instead of fixing it.

## Why

A double has about 16 significant digits, and the doubles next to 1.0 are
1.1e-16 apart. Two expressions in the code went through 1.0 on their way to a
small number.

- The normal CDF was `0.5 * (1 + erf(z / sqrt 2))`. For a negative `z` the
  `erf` term is close to -1, so the sum is a small difference of two numbers
  near 1. Below z = -8 it has lost digits, and at z = -8.5 `erf` rounds to
  exactly -1 and the sum is 0.
- Every upper-tail p-value was `1 - CDF`. The incomplete beta and gamma
  routines in `distributions.hpp` were accurate all along, in the sense that
  `BetaIncomplete` at a small argument returns a small number with full
  relative precision. The CDF then returned `1 - that`, which rounds to 1.0
  once the tail is below 1.1e-16, and the caller's `1 - CDF` returned 0.

So the loss was in two subtractions, not in the special functions.

## The fix

| reference | CDF | upper tail now | from |
| --- | --- | --- | --- |
| normal | `0.5 * erfc(-z / sqrt 2)` | `0.5 * erfc(z / sqrt 2)` | `std::erfc` |
| Student's t, t > 0 | `1 - 0.5 * I_x(df/2, 1/2)` | `0.5 * I_x(df/2, 1/2)`, x = df / (df + t^2) | the continued fraction, directly |
| chi-square | `P(df/2, x/2)` | `Q(df/2, x/2)` | `GammaQ`, the upper incomplete gamma |
| F | `1 - I_z(df2/2, df1/2)` | `I_z(df2/2, df1/2)`, z = df2 / (df2 + df1 x) | the continued fraction, directly |

The four functions are `NormalUpperTail`, `StudentTUpperTail`,
`ChiSquareUpperTail` and `FUpperTail` in `src/include/distributions.hpp`, next
to their CDFs. A two-sided p-value is `2 * XUpperTail(|statistic|)`. The CDFs
themselves keep their form: a CDF that is 1.0 from z = 8.3 on is the correct
double for that value. Only the tail needs its own function.

`BetaIncomplete` switches to `1 - I_{1-x}(b, a)` when x is above the mean of
the distribution. That is the right move for convergence, and it only happens
when the result is not small, so the upper-tail functions never lose digits to
it. `GammaQ` uses the continued fraction when x is above a + 1 and
`1 - GammaPSeries` below, where Q is at least about 0.3.

The `1 - CDF` sites were in twelve source files and served anova, chi-square
independence and goodness of fit, Pearson, Spearman, Kendall, Mann-Whitney,
Wilcoxon, Jarque-Bera, Shapiro-Wilk, the three t-test aggregates and the
t-test table functions, `lm_fit`, `glm_fit`, `lin_hyp` and `emmeans`.
`nonparametric_function.cpp` carried its own copy of the normal CDF with the
same `erf` form; it is gone and the file includes the shared header.

## What stays as it was

- The Tukey-adjusted p-value in `emmeans` is still `1 - StudentizedRangeCDF`.
  That CDF is a Gauss-Legendre quadrature with an absolute accuracy of about
  1e-12 (see `2026-10-studentized-range-quadrature.md`). A tail below that is
  below the method's resolution in either form, so an upper-tail function
  would promise digits the integral does not have. The line carries a comment.
- The quantile functions bisect on the CDF. A quantile for a `p` within 1e-16
  of 1 is therefore as coarse as the CDF there. Nothing in the extension asks
  for one.
- A p-value below about 1e-308 underflows to 0. `pnorm(-37)` is 5.7e-300 and
  is tested; `pnorm(-38)` is subnormal and is not, because libm
  implementations differ on where `erfc` gives up.

## How the tests pin it

`test/sql/tail_pvalues.test` drives each function to a p-value far below
1e-16 with a small deterministic table built from `range()`, and reports
`ROUND(p_value / oracle, 6)` next to the statistic printed to four decimals.
The oracle is scipy's survival function at the same statistic
(`scipy.stats.t.sf`, `f.sf`, `chi2.sf`, `norm.sf`, and statsmodels for the two
fitters). A passing block prints `1.0`; a failing one prints the ratio, which
is `0.0` when the tail is lost, so the failure reads as "tail lost" or
"different statistic" without a debugger.

The ratio is the point. A `printf('%.4e')` of the p-value itself would pin the
last digits of `erfc` and of the continued fractions, and those differ in the
last places between libm implementations, including the wasm build. A ratio
rounded to six places does not see that, and still catches a lost tail, a wrong
tail (upper for lower), a wrong df, or a wrong statistic.

Every one of the eighteen ratios was `1.0` on the first green run. That also
says the statistics themselves, Kendall's tie-free z and Royston's W included,
agree with scipy to at least six places on these inputs.

## Rule

Never write `1 - XCDF(...)` for a probability that can be small. Call the
`UpperTail` function, and for a two-sided test call it on the absolute
statistic and double it. When a new distribution gets a CDF that any p-value
will use, give it an upper tail at the same time, computed from the
complementary special function, not from the CDF.
