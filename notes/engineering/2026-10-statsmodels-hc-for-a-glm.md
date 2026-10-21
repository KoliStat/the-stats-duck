# statsmodels serves HC0's numbers under all four HC names, for a GLM

*October 2026, found during the #22 whole-branch review.*

`GLM(...).fit(cov_type="HC2")` in statsmodels 0.15.0 returns HC0's standard
errors and labels them HC2. So do `HC1` and `HC3`. An OLS fit on the same
library does distinguish all four. Nothing warns.

Two comments in `test/cpp/test_glm_fit.cpp` used to say that this question
"could not be checked", and one of them told a later task to extend
`test/cpp/gen_glm_fit_fixtures.py` if it wanted HC fixtures in the generated
header. A task that followed that instruction would have committed HC0's
numbers three more times, as HC1, HC2 and HC3 goldens, and then reported the
implementation validated against them. The goldens would have agreed with
each other, so no test would have failed.

## The mechanism

`statsmodels/base/covtype.py`, in `get_robustcov_results`, dispatches an HC
request by attribute name:

```python
res.cov_params_default = getattr(self, "cov_" + cov_type.upper(), None)
if res.cov_params_default is None:
    # results classes that do not have cov_HCx attribute
    res.cov_params_default = sw.cov_white_simple(self, use_correction=False)
```

`cov_HC0` through `cov_HC3` are defined on `RegressionResults`, which
`OLSResults` inherits from, in
`statsmodels/regression/linear_model.py`. `GLMResults`, in
`statsmodels/genmod/generalized_linear_model.py`, defines none of them. So
every one of the four names misses the `getattr` and falls through to the
same `cov_white_simple` call. statsmodels' own docstring for that function
says it "produces the same result as cov_hc0, and does not include any small
sample correction".

The label is set before the dispatch, from
`descriptions[cov_type.upper()]`, so the result still reports
`cov_type == "HC2"` and prints "Standard Errors are heteroscedasticity
robust (HC2)". That is the part that makes it silent.

The module-level functions `sandwich_covariance.cov_hc0` through `cov_hc3`
are not a way around it. They read `results.resid`, which a `GLMResults`
does not have, and raise `AttributeError` on a GLM result. They are written
around OLS response residuals.

## Measured, statsmodels 0.15.0

DS1, the 40-row binomial logit from `test/cpp/glm_fit_fixtures.hpp`,
intercept standard error:

| `cov_type` | statsmodels GLM | R 4.6.1 |
| --- | --- | --- |
| HC0 | 8.69158508520134210e-01 | 8.69158507121863155e-01 |
| HC1 | 8.69158508520134210e-01 | 9.03707983511240376e-01 |
| HC2 | 8.69158508520134210e-01 | 9.07611749680332869e-01 |
| HC3 | 8.69158508520134210e-01 | 9.47914435957977197e-01 |

The statsmodels column is one number repeated. It agrees with R's HC0 to
1.6e-09 on this entry, which is the previous-iterate working-weight
difference recorded at `test_binomial_hc_against_r` and not a formula
difference.

The control: the same loop over `OLS` instead of `GLM`, on DS5's data,
returns four different intercept standard errors — 2.004498125392621,
2.314595064571059, 2.432967716906331 and 3.280238651232119. So the four
names do work elsewhere in the library, and the repetition above is
specific to the GLM path.

To reproduce either, fit the model four times and compare:

```python
for ct in ("HC0", "HC1", "HC2", "HC3"):
    print(ct, GLM(y, X, family=sm.families.Binomial()).fit(cov_type=ct).bse)
```

## What to use instead

**R is the only available oracle for a GLM's HC2 and HC3.** R computes them
from its own machinery: `fit$weights` for the working weights and
`hatvalues()` for the leverage of the weighted design. The exact program is
in `test/cpp/test_glm_fit.cpp`, above `kDs1RHcStdError`, and the reviewer
re-ran it and reproduced all 48 values bit for bit.

statsmodels is still an oracle for HC0, since HC0 is what it returns. HC1
is HC0 scaled by `n/(n − rank)`, so it follows from that covariance by
hand. HC2 and HC3 do not: they need the hat diagonal of the weighted
design, and that is the part this path does not give.

Building HC2 and HC3 by hand out of statsmodels pieces —
`get_hat_matrix_diag()` does exist on `GLMResults` — is not an oracle. It
is a second copy of the formula under test, written by whoever is testing
it, and it would agree with a wrong implementation for whatever reason the
implementation was wrong.

## The general shape

A library offered four estimator names, accepted all four, labelled its
output with the one it was asked for, and returned one estimator's numbers
for all of them. It did this for one model class and not for another.
Nothing in the return value says so. The only way to notice is to ask for
two of them and compare. That costs two lines, and it is worth spending on
any library that offers a menu of estimators.

Related: `docs/fitter_conventions.md`'s rule that a claim about evidence has
to name its evidence. "statsmodels has no HC fixture" and "statsmodels
cannot produce a GLM HC2" are different facts, and the first one reads as a
generator gap that somebody could go and close.
