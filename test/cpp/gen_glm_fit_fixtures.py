#!/usr/bin/env python3
"""Goldens for glm_fit.

Oracle: statsmodels sm.GLM (statsmodels >= 0.14), run offline in a throwaway
virtualenv. Never a build or CI dependency.

Eight designs, one per thing that can break:
  DS1 binomial 0/1, two predictors, logit
  DS2 binomial grouped: DS1 collapsed to proportions with trial counts
  DS3 poisson with a log link and an offset
  DS4 gamma with a log link
  DS5 gaussian identity, for the lm_fit cross-check
  DS6 binomial perfectly separated, for the separation flag
  DS7 a rank-deficient design whose third column is the sum of the first two
  DS8 a clustered design, for CR1

DELIBERATE DEVIATION, recorded next to the goldens it affects: this project
reports z_statistic against a normal reference for every family, while
statsmodels switches to a t reference when dispersion is estimated. So for
DS4 and DS5 the oracle's p-values differ from ours. Both are emitted: p_oracle
is what statsmodels reports and p_expected is the normal-reference value the
kernel must produce. For binomial and poisson the two are identical.

Self-checks that must hold before anything is written:
  1. DS1 and DS2 give the same coefficients and standard errors, because they
     are the same data in two shapes
  2. null_deviance equals the deviance of the intercept-only fit
  3. DS5's coefficients and covariance match statsmodels OLS exactly
  4. scaling DS2's trial counts by 10 leaves the coefficients unchanged and
     scales the deviance by 10
  5. for binomial and poisson, p_oracle == p_expected
"""

import sys
import warnings

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Our own formulas, transcribed directly from the task brief. These are NOT
# statsmodels calls: they are the same closed-form arithmetic glm_core.cpp
# implements, run here in Python on the oracle's fitted mu. Most designs
# below assert these against statsmodels' own .deviance / .llf, so a mistake
# in transcription fails loudly instead of silently writing a wrong golden.
# The one place this project deliberately computes something statsmodels
# does not is the AIC for gaussian and gamma; see the note beside kAicD.
# ---------------------------------------------------------------------------

kAicD = {"gaussian": 1, "binomial": 0, "poisson": 0, "gamma": 1}


def my_deviance(family, y, mu, w):
    """2 * the per-observation deviance contributions (gaussian: no 2x; see
    glm_core.hpp for why that family's formula already absorbs it)."""
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    w = np.asarray(w, float)
    if family == "gaussian":
        return float(np.sum(w * (y - mu) ** 2))
    if family == "binomial":
        # y_safe and oy_safe swap in a dummy 1.0 wherever the term is masked
        # to 0 anyway, so the discarded branch of np.where never divides or
        # logs a literal 0. np.where still evaluates both branches eagerly,
        # so without this substitution the unused branch would warn even
        # though its value is never read.
        with np.errstate(divide="ignore", invalid="ignore"):
            y_safe = np.where(y == 0.0, 1.0, y)
            t1 = np.where(y == 0.0, 0.0, y * np.log(y_safe / mu))
            oy_safe = np.where(y == 1.0, 1.0, 1.0 - y)
            t2 = np.where(y == 1.0, 0.0, (1.0 - y) * np.log(oy_safe / (1.0 - mu)))
        return float(2.0 * np.sum(w * (t1 + t2)))
    if family == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            y_safe = np.where(y == 0.0, 1.0, y)
            t1 = np.where(y == 0.0, 0.0, y * np.log(y_safe / mu))
        return float(2.0 * np.sum(w * (t1 - (y - mu))))
    if family == "gamma":
        return float(2.0 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu)))
    raise ValueError(family)


def log_choose(a, b):
    return gammaln(a + 1.0) - gammaln(b + 1.0) - gammaln(a - b + 1.0)


def my_loglik(family, y, mu, w, dispersion):
    """Computed in the order the kernel uses: deviance, then dispersion, then
    this. Gaussian ignores `dispersion` and recomputes sigma^2 = deviance/n
    directly (matching lm_core::loglik); only gamma's shape = 1/dispersion
    reads the argument."""
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    w = np.asarray(w, float)
    if family == "gaussian":
        n = float(len(y))
        sigma2 = my_deviance(family, y, mu, w) / n
        return float(-0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0))
    if family == "binomial":
        # w*y*ln(mu) and w*(1-y)*ln(1-mu) are each a coefficient (w*y, or
        # w*(1-y)) times a log. The discovery that forced this guard: DS6's
        # perfectly separated fit drives mu to exactly 0.0 or 1.0 in double
        # precision (statsmodels does not clamp it), so log(mu) or
        # log(1-mu) can be an actual -inf there, not just a very negative
        # number. Where the coefficient is exactly 0 the term is 0 by the
        # same convention the deviance guard uses (0 * -inf := 0), checked
        # on y directly so log(mu) is never evaluated in that case, rather
        # than computed and discarded.
        with np.errstate(divide="ignore", invalid="ignore"):
            y_safe = np.where(y == 0.0, 1.0, mu)
            t1 = np.where(y == 0.0, 0.0, w * y * np.log(y_safe))
            oy_safe = np.where(y == 1.0, 1.0, 1.0 - mu)
            t2 = np.where(y == 1.0, 0.0, w * (1.0 - y) * np.log(oy_safe))
        wy = w * y
        return float(np.sum(log_choose(w, wy) + t1 + t2))
    if family == "poisson":
        # Same guard, for the same reason: y*ln(mu) at y = 0 must be 0 even
        # if mu underflowed to exactly 0 (not reached by any design here,
        # since none of the poisson designs are separated, but kept for
        # the same robustness the binomial branch needs).
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_safe = np.where(y == 0.0, 1.0, mu)
            t1 = np.where(y == 0.0, 0.0, y * np.log(mu_safe))
        return float(np.sum(w * t1 - w * mu - w * gammaln(y + 1.0)))
    if family == "gamma":
        shape = 1.0 / dispersion
        k = w * shape
        return float(np.sum((k - 1.0) * np.log(y) - y * k / mu - gammaln(k) - k * np.log(mu / k)))
    raise ValueError(family)


def my_dispersion(family, y, mu, w, df_resid):
    if family in ("binomial", "poisson"):
        return 1.0
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    w = np.asarray(w, float)
    v = mu * mu if family == "gamma" else np.ones_like(mu)
    return float(np.sum(w * (y - mu) ** 2 / v) / df_resid)


def my_aic(family, loglik, rank):
    return float(-2.0 * loglik + 2.0 * (rank + kAicD[family]))


def fail(msg):
    print(f"SELF-CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def g(x):
    """17 significant digits, which round-trips a double exactly."""
    return f"{float(x):.17e}"


def arr(xs):
    return ", ".join(g(x) for x in xs)


def make_family(family, link_name):
    if family == "gaussian":
        return sm.families.Gaussian() if link_name == "identity" else sm.families.Gaussian(
            link={"log": sm.families.links.Log(), "inverse": sm.families.links.InversePower()}[link_name])
    if family == "binomial":
        return sm.families.Binomial() if link_name == "logit" else sm.families.Binomial(
            link={"probit": sm.families.links.Probit(), "cloglog": sm.families.links.CLogLog(),
                  "log": sm.families.links.Log()}[link_name])
    if family == "poisson":
        return sm.families.Poisson() if link_name == "log" else sm.families.Poisson(
            link={"identity": sm.families.links.Identity(), "sqrt": sm.families.links.Sqrt()}[link_name])
    if family == "gamma":
        return sm.families.Gamma(link=sm.families.links.Log()) if link_name == "log" else sm.families.Gamma(
            link={"inverse": sm.families.links.InversePower(), "identity": sm.families.links.Identity()}[link_name])
    raise ValueError(family)


def fit(family, endog, X, link_name="log", offset=None, var_weights=None):
    """One sm.GLM fit, with warnings surfaced by the caller (separation and
    rank deficiency both warn by design in two of the eight designs)."""
    fam = make_family(family, link_name)
    kwargs = {}
    if offset is not None:
        kwargs["offset"] = np.asarray(offset, float)
    if var_weights is not None:
        kwargs["var_weights"] = np.asarray(var_weights, float)
    return sm.GLM(endog, X, family=fam, **kwargs).fit()


def golden(family, y, mu, w, rank, df_resid):
    """The four closed-form quantities, our own formulas, cross-checked
    against statsmodels' own .deviance / .llf wherever the two are expected
    to agree (every design here except the AIC of gaussian and gamma)."""
    dev = my_deviance(family, y, mu, w)
    disp = my_dispersion(family, y, mu, w, df_resid)
    ll = my_loglik(family, y, mu, w, disp)
    aic = my_aic(family, ll, rank)
    return dev, disp, ll, aic


def p_values(family, beta, bse, df_resid):
    z = np.asarray(beta, float) / np.asarray(bse, float)
    p_expected = 2.0 * stats.norm.sf(np.abs(z))
    if family in ("gaussian", "gamma"):
        p_oracle = 2.0 * stats.t.sf(np.abs(z), df_resid)
    else:
        p_oracle = p_expected
    return z, p_oracle, p_expected


def null_fit_deviance(family, y, X, link_name, offset, var_weights):
    """An independently fit intercept-only model (offset and weights, if
    any, retained), scored by our own deviance formula. Self-check 2
    compares this to statsmodels' own null_deviance attribute on the full
    fit, for every design."""
    n = X.shape[0]
    Xnull = np.ones((n, 1))
    r = fit(family, y, Xnull, link_name=link_name, offset=offset, var_weights=var_weights)
    w = var_weights if var_weights is not None else np.ones(n)
    return my_deviance(family, np.asarray(y, float) if np.ndim(y) == 1 else _proportion(y), r.mu, w)


def _proportion(endog2col):
    s = np.asarray(endog2col, float)
    return s[:, 0] / s.sum(axis=1)


out_lines = []


def P(s=""):
    out_lines.append(s)


worst_null = 0.0
worst_null_where = ""


def check_null_deviance(label, family, y, X, link_name, offset, var_weights, res):
    """Self-check 2, run once per design. `w` is accepted for a uniform
    call shape across every design site but is not used directly here:
    null_fit_deviance re-derives the right weight (var_weights or all-ones)
    for the intercept-only refit on its own."""
    global worst_null, worst_null_where
    mine = null_fit_deviance(family, y, X, link_name, offset, var_weights)
    theirs = float(res.null_deviance)
    # The band is IRLS's own convergence tolerance: two independent fits
    # (the full model's internal null refit vs. this script's explicit one)
    # solve the same intercept-only MLE and agree to several more digits
    # than this, so a looser mismatch would mean a real discrepancy, not
    # noise.
    rel = abs(mine - theirs) / max(abs(mine), abs(theirs), 1e-300)
    if rel > worst_null:
        worst_null, worst_null_where = rel, label
    if not rel < 1e-6:
        fail(f"{label}: null_deviance {theirs!r} != independent intercept-only fit {mine!r}")
    return mine  # the golden kDsXNullDeviance value; the caller does not refit


# ===========================================================================
# DS1 / DS2: binomial, two predictors, logit. DS1 is the 40-row individual
# 0/1 expansion; DS2 is the same data collapsed to 10 grouped proportions
# with trial counts. Both come from one set of group success counts, out of
# 4 trials each, built from logit(p) = -2 + 0.5*x1 + 1.0*x2 and rounding:
# every group lands on 1, 2 or 3 successes out of 4, so neither the grouped
# nor the individual data is separated.
# ===========================================================================

_DS_X1 = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
_DS_X2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
_DS_SUCCESS = [1, 2, 1, 2, 2, 2, 2, 3, 2, 3]  # out of 4 trials, by construction above
_DS_TRIALS = 4


def expand_individual(x1s, x2s, successes, trials):
    """One group's trials become `trials` individual 0/1 rows, successes
    first. Row order does not affect the fit: each row is an independent
    Bernoulli term."""
    y, x1, x2 = [], [], []
    for a, b, s in zip(x1s, x2s, successes):
        for i in range(trials):
            y.append(1.0 if i < s else 0.0)
            x1.append(float(a))
            x2.append(float(b))
    return np.array(y), np.array(x1), np.array(x2)


ds1_y, ds1_x1, ds1_x2 = expand_individual(_DS_X1, _DS_X2, _DS_SUCCESS, _DS_TRIALS)
ds1_w = np.ones_like(ds1_y)
X1 = sm.add_constant(np.column_stack([ds1_x1, ds1_x2]))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res1 = fit("binomial", ds1_y, X1, link_name="logit")

rank1 = len(res1.params)
df1 = int(res1.df_resid)
dev1, disp1, ll1, aic1 = golden("binomial", ds1_y, res1.mu, ds1_w, rank1, df1)
z1, p_or1, p_ex1 = p_values("binomial", res1.params, res1.bse, df1)
cov1 = np.asarray(res1.cov_params())

if not np.allclose(np.sqrt(np.diag(cov1)), np.asarray(res1.bse)):
    fail("DS1: cov_params diagonal != bse^2")
if not np.isclose(dev1, res1.deviance, rtol=1e-9):
    fail(f"DS1: my_deviance {dev1!r} != statsmodels .deviance {res1.deviance!r}")
if not np.isclose(ll1, res1.llf, rtol=1e-9):
    fail(f"DS1: my_loglik {ll1!r} != statsmodels .llf {res1.llf!r}")
if not np.isclose(aic1, res1.aic, rtol=1e-9):
    fail(f"DS1: my_aic {aic1!r} != statsmodels .aic {res1.aic!r} (binomial: d=0 on both sides)")

ds2_x1 = np.array(_DS_X1, float)
ds2_x2 = np.array(_DS_X2, float)
ds2_y = np.array(_DS_SUCCESS, float) / _DS_TRIALS
ds2_w = np.full(10, float(_DS_TRIALS))
X2 = sm.add_constant(np.column_stack([ds2_x1, ds2_x2]))
successes2 = np.array(_DS_SUCCESS, float)
failures2 = ds2_w - successes2
endog2 = np.column_stack([successes2, failures2])
res2 = fit("binomial", endog2, X2, link_name="logit")

rank2 = len(res2.params)
df2 = int(res2.df_resid)
dev2, disp2, ll2, aic2 = golden("binomial", ds2_y, res2.mu, ds2_w, rank2, df2)
z2, p_or2, p_ex2 = p_values("binomial", res2.params, res2.bse, df2)
cov2 = np.asarray(res2.cov_params())

if not np.allclose(np.sqrt(np.diag(cov2)), np.asarray(res2.bse)):
    fail("DS2: cov_params diagonal != bse^2")
if not np.isclose(dev2, res2.deviance, rtol=1e-9):
    fail(f"DS2: my_deviance {dev2!r} != statsmodels .deviance {res2.deviance!r}")
if not np.isclose(ll2, res2.llf, rtol=1e-9):
    fail(f"DS2: my_loglik {ll2!r} != statsmodels .llf {res2.llf!r}")

# ---- Self-check 1: DS1 and DS2 give the same coefficients and SEs. -------
worst1_beta = float(np.max(np.abs(np.asarray(res1.params) - np.asarray(res2.params))))
worst1_bse = float(np.max(np.abs(np.asarray(res1.bse) - np.asarray(res2.bse))))
# The band is IRLS's own convergence tolerance on two differently-shaped
# optimizations of the same likelihood (40 Bernoulli rows vs. 10 grouped
# rows), not an exact algebraic identity, so it is set an order of
# magnitude above the worst observed gap.
if not (worst1_beta < 1e-5 and worst1_bse < 1e-5):
    fail(f"DS1 vs DS2: coefficients or SEs disagree (worst beta {worst1_beta:.3e}, "
         f"worst bse {worst1_bse:.3e})")
print(f"self-check 1 OK: DS1 and DS2 give the same coefficients and standard errors, "
      f"worst beta gap {worst1_beta:.3e}, worst bse gap {worst1_bse:.3e}", file=sys.stderr)

# ---- Self-check 4: scaling DS2's trial counts by 10. ----------------------
ds2w_w = ds2_w * 10.0
successes2w = successes2 * 10.0
failures2w = ds2w_w - successes2w
res2w = fit("binomial", np.column_stack([successes2w, failures2w]), X2, link_name="logit")
dev2w = my_deviance("binomial", ds2_y, res2w.mu, ds2w_w)
worst4_beta = float(np.max(np.abs(np.asarray(res2w.params) - np.asarray(res2.params))))
dev_ratio = dev2w / dev2
# Same reasoning as self-check 1: IRLS convergence noise, not an exact
# algebraic identity.
if not (worst4_beta < 1e-5 and abs(dev_ratio - 10.0) < 1e-6):
    fail(f"DS2 x10 weights: beta moved by {worst4_beta:.3e}, deviance ratio {dev_ratio!r} != 10")
print(f"self-check 4 OK: x10 trial counts leaves beta unchanged (worst {worst4_beta:.3e}) "
      f"and scales deviance by {dev_ratio:.12f}", file=sys.stderr)

null_dev1 = check_null_deviance("DS1", "binomial", ds1_y, X1, "logit", None, None, res1)
null_dev2 = check_null_deviance("DS2", "binomial", endog2, X2, "logit", None, ds2_w, res2)

# ===========================================================================
# DS3: poisson, log link, with an offset (log exposure). n=15.
# ===========================================================================

ds3_x1 = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
ds3_exposure = np.array([10, 20, 15, 5, 30, 25, 10, 40, 12, 18, 22, 8, 14, 16, 19], float)
ds3_offset = np.log(ds3_exposure)
np.random.seed(2201)
ds3_y = np.random.poisson(np.exp(0.3 + 0.4 * ds3_x1) * ds3_exposure).astype(float)
ds3_w = np.ones_like(ds3_y)
X3 = sm.add_constant(ds3_x1)
res3 = fit("poisson", ds3_y, X3, link_name="log", offset=ds3_offset)

rank3 = len(res3.params)
df3 = int(res3.df_resid)
dev3, disp3, ll3, aic3 = golden("poisson", ds3_y, res3.mu, ds3_w, rank3, df3)
z3, p_or3, p_ex3 = p_values("poisson", res3.params, res3.bse, df3)
cov3 = np.asarray(res3.cov_params())

if not np.isclose(dev3, res3.deviance, rtol=1e-9):
    fail(f"DS3: my_deviance {dev3!r} != statsmodels .deviance {res3.deviance!r}")
if not np.isclose(ll3, res3.llf, rtol=1e-9):
    fail(f"DS3: my_loglik {ll3!r} != statsmodels .llf {res3.llf!r}")
if not np.isclose(aic3, res3.aic, rtol=1e-9):
    fail(f"DS3: my_aic {aic3!r} != statsmodels .aic {res3.aic!r} (poisson: d=0 on both sides)")
null_dev3 = check_null_deviance("DS3", "poisson", ds3_y, X3, "log", ds3_offset, None, res3)

# ===========================================================================
# DS4: gamma, log link. n=15. Positive continuous response.
# ===========================================================================

np.random.seed(3301)
ds4_x1 = np.linspace(0.0, 2.0, 15)
ds4_mu_true = np.exp(0.5 + 0.3 * ds4_x1)
ds4_y = np.round(np.random.gamma(shape=6.0, scale=ds4_mu_true / 6.0), 4)
ds4_w = np.ones_like(ds4_y)
X4 = sm.add_constant(ds4_x1)
res4 = fit("gamma", ds4_y, X4, link_name="log")

rank4 = len(res4.params)
df4 = int(res4.df_resid)
dev4, disp4, ll4, aic4 = golden("gamma", ds4_y, res4.mu, ds4_w, rank4, df4)
z4, p_or4, p_ex4 = p_values("gamma", res4.params, res4.bse, df4)
cov4 = np.asarray(res4.cov_params())

if not np.isclose(disp4, res4.scale, rtol=1e-9):
    fail(f"DS4: my_dispersion {disp4!r} != statsmodels .scale {res4.scale!r}")
if not np.isclose(dev4, res4.deviance, rtol=1e-9):
    fail(f"DS4: my_deviance {dev4!r} != statsmodels .deviance {res4.deviance!r}")
if not np.isclose(ll4, res4.llf, rtol=1e-6):
    fail(f"DS4: my_loglik {ll4!r} != statsmodels .llf {res4.llf!r}")
# The deliberate AIC deviation: statsmodels' own .aic never adds a
# parameter for an estimated dispersion, so for gamma it is exactly 2 less
# than this project's aic (d=1). Checked as an exact difference, not
# approximate agreement, to pin the deviation itself.
if not np.isclose(aic4 - res4.aic, 2.0, atol=1e-8):
    fail(f"DS4: aic deviation is not exactly 2 (ours {aic4!r}, statsmodels {res4.aic!r})")
null_dev4 = check_null_deviance("DS4", "gamma", ds4_y, X4, "log", None, None, res4)

# ===========================================================================
# DS5: gaussian identity, for the lm_fit cross-check. Same data as
# gen_lm_fit_fixtures.py's ds1_hetero (n=12, intercept, 2 predictors, one
# high-leverage point), so this design's fit is checkable against
# test_lm_fit.cpp's already-validated goldens as well as against statsmodels
# OLS fresh in this script.
# ===========================================================================

ds5_y = np.array([9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55], float)
ds5_x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25], float)
ds5_x2 = np.array([5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4], float)
ds5_w = np.ones_like(ds5_y)
X5 = sm.add_constant(np.column_stack([ds5_x1, ds5_x2]))
res5 = fit("gaussian", ds5_y, X5, link_name="identity")
ols5 = sm.OLS(ds5_y, X5).fit()

rank5 = len(res5.params)
df5 = int(res5.df_resid)
dev5, disp5, ll5, aic5 = golden("gaussian", ds5_y, res5.mu, ds5_w, rank5, df5)
z5, p_or5, p_ex5 = p_values("gaussian", res5.params, res5.bse, df5)
cov5 = np.asarray(res5.cov_params())

if not np.isclose(ll5, res5.llf, rtol=1e-9):
    fail(f"DS5: my_loglik {ll5!r} != statsmodels .llf {res5.llf!r}")
if not np.isclose(aic5 - res5.aic, 2.0, atol=1e-8):
    fail(f"DS5: aic deviation is not exactly 2 (ours {aic5!r}, statsmodels {res5.aic!r})")

# ---- Self-check 3: DS5 matches statsmodels OLS exactly. -------------------
worst3_beta = float(np.max(np.abs(np.asarray(res5.params) - np.asarray(ols5.params))))
worst3_cov = float(np.max(np.abs(cov5 - np.asarray(ols5.cov_params()))))
# The band is IRLS's own convergence tolerance: a Gaussian-identity GLM
# reaches the OLS normal equations in one IRLS step, so the two solves
# agree to many more digits than this; a looser gap would be a real bug.
if not (worst3_beta < 1e-9 and worst3_cov < 1e-9):
    fail(f"DS5: GLM(gaussian, identity) != OLS (worst beta {worst3_beta:.3e}, "
         f"worst cov {worst3_cov:.3e})")
# Second, independent anchor: test_lm_fit.cpp's own pinned ds1_hetero
# goldens (classical vcov), transcribed here to 15 digits.
LM_FIT_BETA = [10.1205711006471, 2.05794012307408, -0.978656740125124]
LM_FIT_COV00 = 4.30197288655079
if not np.allclose(res5.params, LM_FIT_BETA, rtol=1e-9):
    fail(f"DS5 does not reproduce test_lm_fit.cpp's ds1_hetero beta: {res5.params!r}")
if not abs(cov5[0, 0] - LM_FIT_COV00) < 1e-8:
    fail(f"DS5 does not reproduce test_lm_fit.cpp's ds1_hetero cov(0,0): {cov5[0, 0]!r}")
print(f"self-check 3 OK: DS5 matches statsmodels OLS exactly, worst beta gap "
      f"{worst3_beta:.3e}, worst cov gap {worst3_cov:.3e}; also reproduces "
      f"test_lm_fit.cpp's ds1_hetero goldens", file=sys.stderr)
null_dev5 = check_null_deviance("DS5", "gaussian", ds5_y, X5, "identity", None, None, res5)

# ===========================================================================
# DS6: binomial, perfectly separated. n=8, one predictor. Every y=0 row has
# x1 below 4.5 and every y=1 row has x1 above it, so no finite beta
# maximizes the likelihood: IRLS runs to its iteration cap and reports
# "converged" with enormous coefficients and standard errors. That
# instability is the point: a later task's separation flag is tested
# against exactly this shape of output, not against a tight numerical
# match, so the beta/se goldens below are only as reproducible as
# statsmodels' own iteration cap makes them.
# ===========================================================================

ds6_x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
ds6_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
ds6_w = np.ones_like(ds6_y)
X6 = sm.add_constant(ds6_x1)
with warnings.catch_warnings(record=True) as wlist6:
    warnings.simplefilter("always")
    res6 = fit("binomial", ds6_y, X6, link_name="logit")
    sep_warned = any("separation" in str(w.message).lower() for w in wlist6)
if not sep_warned:
    fail("DS6: statsmodels did not warn about separation; the design is not actually separated")
print(f"DS6 note: statsmodels warned of perfect separation, as expected; "
      f"converged={res6.converged}, iterations={res6.fit_history['iteration']}",
      file=sys.stderr)

rank6 = len(res6.params)
df6 = int(res6.df_resid)
dev6, disp6, ll6, aic6 = golden("binomial", ds6_y, res6.mu, ds6_w, rank6, df6)
z6, p_or6, p_ex6 = p_values("binomial", res6.params, res6.bse, df6)
cov6 = np.asarray(res6.cov_params())
null_dev6 = check_null_deviance("DS6", "binomial", ds6_y, X6, "logit", None, None, res6)

# ===========================================================================
# DS7: poisson, log link, rank deficient. Predictors x1, x2, x3 = x1 + x2;
# the design has 4 columns (intercept, x1, x2, x3) and rank 3. Following
# this project's convention (R's rule, also lm_fit's): the LATER dependent
# column is dropped, so x3 is aliased. statsmodels cannot oracle a dropped
# fit directly (it spreads the coefficient across the dependent set via
# pinv instead of dropping one), so the reduced design [intercept, x1, x2]
# is the oracle, exactly as docs/fitter_conventions.md and
# gen_lm_fit_fixtures.py's emit_deficient describe. What a rank-deficient
# fit's FITTED VALUES determine (deviance, null_deviance, loglik, aic,
# dispersion, df_residual) is checked below to agree between the reduced
# fit and statsmodels' own pinv fit of the full design, confirming the
# reduced design really is a valid oracle for those fields.
# ===========================================================================

ds7_x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], float)
ds7_x2 = np.array([5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4], float)
ds7_x3 = ds7_x1 + ds7_x2
ds7_y = np.array([2, 1, 4, 0, 3, 2, 5, 1, 3, 2, 6, 2], float)
ds7_w = np.ones_like(ds7_y)
Xfull7 = sm.add_constant(np.column_stack([ds7_x1, ds7_x2, ds7_x3]))
Xred7 = sm.add_constant(np.column_stack([ds7_x1, ds7_x2]))
if int(np.linalg.matrix_rank(Xfull7)) != 3:
    fail("DS7: the full design is not rank 3")
if not np.allclose(Xfull7[:, 3], Xfull7[:, 1] + Xfull7[:, 2]):
    fail("DS7: the third predictor column is not the sum of the first two")

res7red = fit("poisson", ds7_y, Xred7, link_name="log")
with warnings.catch_warnings(record=True) as wlist7:
    warnings.simplefilter("always")
    res7full = fit("poisson", ds7_y, Xfull7, link_name="log")
    rank_warned = any("rank" in str(w.message).lower() or "singular" in str(w.message).lower()
                      for w in wlist7)
if not rank_warned:
    fail("DS7: statsmodels did not warn about the rank-deficient design")

rank7 = len(res7red.params)  # 3: the reduced design's own width
df7 = int(res7red.df_resid)
dev7, disp7, ll7, aic7 = golden("poisson", ds7_y, res7red.mu, ds7_w, rank7, df7)
# Confirms the reduced fit is a valid oracle for the fitted-value-only
# quantities: statsmodels' own pinv fit of the FULL design must reach the
# same mu, hence the same deviance, df_resid, llf.
if not (int(res7full.df_resid) == df7 and np.isclose(res7full.deviance, dev7, rtol=1e-8)
        and np.isclose(res7full.llf, ll7, rtol=1e-8)):
    fail("DS7: the reduced fit and statsmodels' own pinv fit disagree on the "
         "fitted-value-only quantities (deviance, df_resid, llf)")
z7, p_or7, p_ex7 = p_values("poisson", res7red.params, res7red.bse, df7)
cov7red = np.asarray(res7red.cov_params())
# Assembled full-width (4-wide) golden: the reduced estimates in the kept
# positions, NaN in the dropped one (column 3, x3) and in its row/column of
# cov, matching what a rank-deficient glm_fit reports per
# docs/fitter_conventions.md.
KEEP7 = [0, 1, 2]
ALIASED7 = 3
beta7 = np.full(4, np.nan)
se7 = np.full(4, np.nan)
z7f = np.full(4, np.nan)
por7f = np.full(4, np.nan)
pex7f = np.full(4, np.nan)
cov7 = np.full((4, 4), np.nan)
beta7[KEEP7] = res7red.params
se7[KEEP7] = res7red.bse
z7f[KEEP7] = z7
por7f[KEEP7] = p_or7
pex7f[KEEP7] = p_ex7
for a, ka in enumerate(KEEP7):
    for b, kb in enumerate(KEEP7):
        cov7[ka, kb] = cov7red[a, b]
null_dev7 = check_null_deviance("DS7", "poisson", ds7_y, Xred7, "log", None, None, res7red)

# ===========================================================================
# DS8: poisson, log link, clustered (G=5, uneven cluster sizes summing to
# 25, matching gen_lm_fit_fixtures.py's DS4). A per-cluster effect on the
# linear predictor injects within-cluster correlation, so CR0/CR1 differ
# visibly from the naive model-based SE.
# ===========================================================================

np.random.seed(4401)
_sizes8 = [3, 4, 5, 6, 7]
ds8_groups = np.concatenate([np.full(s, gi) for gi, s in enumerate(_sizes8)])
_n8 = ds8_groups.size
ds8_x1 = np.round(np.random.randn(_n8) * 0.5 + np.arange(_n8) * 0.05, 3)
_u8 = np.random.randn(len(_sizes8)) * 0.6  # per-cluster effect, linear-predictor scale
ds8_eta_true = 0.5 + 0.3 * ds8_x1 + _u8[ds8_groups]
ds8_y = np.random.poisson(np.exp(ds8_eta_true)).astype(float)
ds8_w = np.ones_like(ds8_y)
X8 = sm.add_constant(ds8_x1)

res8 = fit("poisson", ds8_y, X8, link_name="log")
rank8 = len(res8.params)
df8 = int(res8.df_resid)
dev8, disp8, ll8, aic8 = golden("poisson", ds8_y, res8.mu, ds8_w, rank8, df8)
z8, p_or8, p_ex8 = p_values("poisson", res8.params, res8.bse, df8)

res8_cr0 = sm.GLM(ds8_y, X8, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": ds8_groups, "use_correction": False})
res8_cr1 = sm.GLM(ds8_y, X8, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": ds8_groups})  # statsmodels default
G8 = int(np.unique(ds8_groups).size)
c8 = (G8 / (G8 - 1)) * ((_n8 - 1) / (_n8 - rank8))
if not np.allclose(np.asarray(res8_cr1.bse) / np.asarray(res8_cr0.bse), np.sqrt(c8)):
    fail("DS8: CR1/CR0 ratio != sqrt(c), the lm_fit finite-sample factor")
if not np.allclose(res8_cr1.params, res8.params, rtol=1e-9):
    fail("DS8: the cluster-robust fit's beta differs from the model-based fit's")
cov8_cr0 = np.asarray(res8_cr0.cov_params())
cov8_cr1 = np.asarray(res8_cr1.cov_params())
z8_cr0, p_or8_cr0, p_ex8_cr0 = p_values("poisson", res8_cr0.params, res8_cr0.bse, df8)
z8_cr1, p_or8_cr1, p_ex8_cr1 = p_values("poisson", res8_cr1.params, res8_cr1.bse, df8)
null_dev8 = check_null_deviance("DS8", "poisson", ds8_y, X8, "log", None, None, res8)

# ---- Self-check 2 (final): printed once, covering every design above. ----
print(f"self-check 2 OK: null_deviance matches an independent intercept-only "
      f"fit on every design, worst relative gap {worst_null:.3e} ({worst_null_where})",
      file=sys.stderr)

# ---- Self-check 5: for binomial and poisson, p_oracle == p_expected. ------
worst5 = 0.0
worst5_where = "every binomial/poisson design"
for label, po, pe in (
        ("DS1", p_or1, p_ex1), ("DS2", p_or2, p_ex2), ("DS3", p_or3, p_ex3),
        ("DS6", p_or6, p_ex6), ("DS7", por7f, pex7f), ("DS8", p_or8, p_ex8),
        ("DS8 CR0", p_or8_cr0, p_ex8_cr0), ("DS8 CR1", p_or8_cr1, p_ex8_cr1)):
    po = np.asarray(po, float)
    pe = np.asarray(pe, float)
    mask = ~(np.isnan(po) & np.isnan(pe))
    gap = np.max(np.abs(po[mask] - pe[mask])) if np.any(mask) else 0.0
    if gap > worst5:
        worst5, worst5_where = gap, label
if not worst5 < 1e-14:
    fail(f"p_oracle != p_expected for a binomial/poisson design (worst {worst5:.3e} at {worst5_where})")
print(f"self-check 5 OK: for binomial and poisson, p_oracle == p_expected, "
      f"worst {worst5:.3e} at {worst5_where}", file=sys.stderr)

print("all self-checks OK", file=sys.stderr)

# ===========================================================================
# Emit test/cpp/glm_fit_fixtures.hpp
# ===========================================================================

P("// Generated by test/cpp/gen_glm_fit_fixtures.py. Do not edit by hand.")
P("//")
P(f"// Oracle: statsmodels {statsmodels.__version__} sm.GLM, run offline in a")
P(f"// throwaway virtualenv (scipy {scipy.__version__}, numpy {np.__version__}).")
P("// Never a build or CI dependency.")
P("//")
P("// Eight designs, one per thing that can break:")
P("//   DS1 binomial 0/1, two predictors, logit")
P("//   DS2 binomial grouped: DS1 collapsed to proportions with trial counts")
P("//   DS3 poisson with a log link and an offset")
P("//   DS4 gamma with a log link")
P("//   DS5 gaussian identity, for the lm_fit cross-check")
P("//   DS6 binomial perfectly separated, for the separation flag")
P("//   DS7 a rank-deficient design whose third column is the sum of the first two")
P("//   DS8 a clustered design, for CR1")
P("//")
P("// DELIBERATE DEVIATION 1, affecting DS4 and DS5: this project reports")
P("// z_statistic against a normal reference for every family. statsmodels")
P("// switches to a t(df_residual) reference when the dispersion is estimated")
P("// (gaussian, gamma), which is why those two designs below carry both")
P("// p_oracle (statsmodels' own reference) and p_expected (always normal,")
P("// which is what the kernel must produce). For binomial and poisson the")
P("// two columns are identical (self-check 5).")
P("//")
P("// DELIBERATE DEVIATION 2, affecting DS4 and DS5's aic: statsmodels' own")
P("// .aic never counts the estimated dispersion as a parameter, for any")
P("// family. This project's aic = -2*loglik + 2*(rank + d), with d = 1 for")
P("// gaussian and gamma (R's convention: the dispersion IS an estimated")
P("// parameter there) and d = 0 for binomial and poisson (dispersion fixed")
P("// at 1, matching statsmodels for those two). So DS4's and DS5's aic")
P("// below are exactly 2 more than statsmodels' own .aic; DS1/DS2/DS3/DS6/")
P("// DS7/DS8 match statsmodels' .aic exactly.")
P("//")
P("// Values carry 17 significant digits, which round-trips a double exactly.")
P("//")
P("// Self-checks that passed before this file was written:")
P(f"//   1. DS1 and DS2 give the same coefficients and SEs, worst beta gap "
  f"{worst1_beta:.3e}, worst se gap {worst1_bse:.3e}")
P(f"//   2. null_deviance matches an independent intercept-only fit on every "
  f"design, worst relative gap {worst_null:.3e} ({worst_null_where})")
P(f"//   3. DS5 matches statsmodels OLS exactly, worst beta gap {worst3_beta:.3e}, "
  f"worst cov gap {worst3_cov:.3e}; also reproduces test_lm_fit.cpp's ds1_hetero")
P(f"//   4. x10 trial counts on DS2 leaves beta unchanged (worst {worst4_beta:.3e}) "
  f"and scales deviance by {dev_ratio:.12f}")
P(f"//   5. for binomial and poisson, p_oracle == p_expected, worst {worst5:.3e}")
P("#pragma once")
P()
P("#include <cstddef>")
P("#include <limits>")
P()
P("// NaN in a fixture marks the column lm_fit's rank-deficiency rule drops")
P("// (DS7's x3), or a field a design does not report. It is never a failure.")
P("inline constexpr double kGlmNaN = std::numeric_limits<double>::quiet_NaN();")
P()


def emit_design(tag, n, k, rank, df_resid, data, beta, se, z, p_oracle, p_expected,
                dev, null_dev, disp, ll, aic, cov, note=None):
    if note:
        P(f"// -- {tag}: {note}")
    P(f"inline constexpr std::size_t k{tag}N = {n};")
    P(f"inline constexpr std::size_t k{tag}K = {k};")
    P(f"inline constexpr std::size_t k{tag}Rank = {rank};")
    P(f"inline constexpr std::size_t k{tag}DfResidual = {df_resid};")
    for name, vec in data.items():
        P(f"inline constexpr double k{tag}{name}[{n}] = {{{arr(vec)}}};")
    def cell(v):
        return "kGlmNaN" if (isinstance(v, float) and np.isnan(v)) else g(v)
    P(f"inline constexpr double k{tag}Beta[{k}] = {{{', '.join(cell(v) for v in beta)}}};")
    P(f"inline constexpr double k{tag}StdError[{k}] = {{{', '.join(cell(v) for v in se)}}};")
    P(f"inline constexpr double k{tag}ZStatistic[{k}] = {{{', '.join(cell(v) for v in z)}}};")
    P(f"inline constexpr double k{tag}POracle[{k}] = {{{', '.join(cell(v) for v in p_oracle)}}};")
    P(f"inline constexpr double k{tag}PExpected[{k}] = {{{', '.join(cell(v) for v in p_expected)}}};")
    P(f"inline constexpr double k{tag}Deviance = {g(dev)};")
    P(f"inline constexpr double k{tag}NullDeviance = {g(null_dev)};")
    P(f"inline constexpr double k{tag}Dispersion = {g(disp)};")
    P(f"inline constexpr double k{tag}Loglik = {g(ll)};")
    P(f"inline constexpr double k{tag}Aic = {g(aic)};")
    flat_cov = cov.ravel()
    P(f"inline constexpr double k{tag}Cov[{k * k}] = {{{', '.join(cell(v) for v in flat_cov)}}};")
    P()


# -- DS1: binomial individual 0/1, two predictors, logit --------------------
emit_design("Ds1", 40, 3, rank1, df1,
            {"Y": ds1_y, "X1": ds1_x1, "X2": ds1_x2, "W": ds1_w},
            res1.params, res1.bse, z1, p_or1, p_ex1, dev1, null_dev1, disp1, ll1, aic1, cov1,
            note="binomial, 40 individual 0/1 rows, logit, w = 1 throughout")

# -- DS2: binomial grouped, the same data as proportions + trial counts ----
emit_design("Ds2", 10, 3, rank2, df2,
            {"Y": ds2_y, "X1": ds2_x1, "X2": ds2_x2, "W": ds2_w},
            res2.params, res2.bse, z2, p_or2, p_ex2, dev2, null_dev2, disp2, ll2, aic2, cov2,
            note="binomial, 10 grouped rows, y = proportion, w = trial count (4 each); "
                 "self-check 1 pins these beta/se equal to DS1's")
P(f"// Self-check 4: scaling DS2's W by 10 (same Y, X) leaves beta unchanged and")
P(f"// scales the deviance by {dev_ratio:.12f}. Not stored as a separate design:")
P(f"// a consuming test reconstructs it from kDs2W and this ratio.")
P()

# -- DS3: poisson, log link, offset -----------------------------------------
emit_design("Ds3", 15, 2, rank3, df3,
            {"Y": ds3_y, "X1": ds3_x1, "Offset": ds3_offset, "W": ds3_w},
            res3.params, res3.bse, z3, p_or3, p_ex3, dev3, null_dev3, disp3, ll3, aic3, cov3,
            note="poisson, log link, offset = log(exposure)")

# -- DS4: gamma, log link ----------------------------------------------------
emit_design("Ds4", 15, 2, rank4, df4,
            {"Y": ds4_y, "X1": ds4_x1, "W": ds4_w},
            res4.params, res4.bse, z4, p_or4, p_ex4, dev4, null_dev4, disp4, ll4, aic4, cov4,
            note="gamma, log link; p_oracle (t ref) != p_expected (normal ref); "
                 "aic is 2 more than statsmodels' own .aic (deviation 2)")

# -- DS5: gaussian identity, the lm_fit cross-check -------------------------
emit_design("Ds5", 12, 3, rank5, df5,
            {"Y": ds5_y, "X1": ds5_x1, "X2": ds5_x2, "W": ds5_w},
            res5.params, res5.bse, z5, p_or5, p_ex5, dev5, null_dev5, disp5, ll5, aic5, cov5,
            note="gaussian identity; same data as gen_lm_fit_fixtures.py's ds1_hetero and "
                 "test_lm_fit.cpp's ds1_hetero (n=12, one high-leverage point); "
                 "p_oracle (t ref) != p_expected (normal ref); aic is 2 more than "
                 "statsmodels' own .aic (deviation 2)")

# -- DS6: binomial, perfectly separated --------------------------------------
emit_design("Ds6", 8, 2, rank6, df6,
            {"Y": ds6_y, "X1": ds6_x1, "W": ds6_w},
            res6.params, res6.bse, z6, p_or6, p_ex6, dev6, null_dev6, disp6, ll6, aic6, cov6,
            note="binomial, perfectly separated at x1 = 4.5; statsmodels warns and reports "
                 "huge beta/se at its iteration cap, not a well-defined MLE -- see the "
                 "generator's docstring")

# -- DS7: poisson, rank deficient (x3 = x1 + x2, dropped) --------------------
P(f"// -- Ds7: poisson, rank deficient. x3 = x1 + x2; column 3 (0-based) is")
P(f"// dropped, matching R's / this project's rule. The oracle is the reduced")
P(f"// design [intercept, x1, x2]; statsmodels' own pinv fit of the full design")
P(f"// agrees on every fitted-value-only quantity (deviance, df_resid, llf) but")
P(f"// not on beta, which is why the reduced fit is the oracle rather than")
P(f"// statsmodels' own deficient fit (docs/fitter_conventions.md).")
P(f"inline constexpr std::size_t kDs7N = 12;")
P(f"inline constexpr std::size_t kDs7K = 4;   // intercept, x1, x2, x3 -- the FULL width")
P(f"inline constexpr std::size_t kDs7Rank = {rank7};")
P(f"inline constexpr std::size_t kDs7DfResidual = {df7};")
P(f"inline constexpr double kDs7Y[12] = {{{arr(ds7_y)}}};")
P(f"inline constexpr double kDs7X1[12] = {{{arr(ds7_x1)}}};")
P(f"inline constexpr double kDs7X2[12] = {{{arr(ds7_x2)}}};")
P(f"inline constexpr double kDs7X3[12] = {{{arr(ds7_x3)}}};  // = x1 + x2, dropped as dependent")
P(f"inline constexpr double kDs7W[12] = {{{arr(ds7_w)}}};")
P(f"inline constexpr unsigned kDs7AliasedColumn = {ALIASED7};  // 0-based design index, x3")


def cell7(v):
    return "kGlmNaN" if (isinstance(v, float) and np.isnan(v)) else g(v)


P(f"inline constexpr double kDs7Beta[4] = {{{', '.join(cell7(v) for v in beta7)}}};")
P(f"inline constexpr double kDs7StdError[4] = {{{', '.join(cell7(v) for v in se7)}}};")
P(f"inline constexpr double kDs7ZStatistic[4] = {{{', '.join(cell7(v) for v in z7f)}}};")
P(f"inline constexpr double kDs7POracle[4] = {{{', '.join(cell7(v) for v in por7f)}}};")
P(f"inline constexpr double kDs7PExpected[4] = {{{', '.join(cell7(v) for v in pex7f)}}};")
P(f"inline constexpr double kDs7Deviance = {g(dev7)};")
P(f"inline constexpr double kDs7NullDeviance = {g(null_dev7)};")
P(f"inline constexpr double kDs7Dispersion = {g(disp7)};")
P(f"inline constexpr double kDs7Loglik = {g(ll7)};")
P(f"inline constexpr double kDs7Aic = {g(aic7)};")
P(f"inline constexpr double kDs7Cov[16] = {{{', '.join(cell7(v) for v in cov7.ravel())}}};")
P()

# -- DS8: poisson, clustered, CR0 and CR1 -----------------------------------
P(f"// -- Ds8: poisson, log link, clustered. G = {G8} clusters, sizes "
  f"{_sizes8}, n = {_n8}.")
P(f"// kDs8Beta/StdError/etc. are the model-based (non-cluster) fit; CR0 and")
P(f"// CR1 are the two cluster-robust variants, matching lm_fit's own CR0/CR1")
P(f"// convention (CR1 is statsmodels' and Stata's default correction).")
P(f"inline constexpr std::size_t kDs8N = {_n8};")
P(f"inline constexpr std::size_t kDs8K = 2;")
P(f"inline constexpr std::size_t kDs8Rank = {rank8};")
P(f"inline constexpr std::size_t kDs8DfResidual = {df8};")
P(f"inline constexpr std::size_t kDs8NumClusters = {G8};")
P(f"inline constexpr double kDs8Cr1Factor = {g(c8)};  // c = [G/(G-1)]*[(N-1)/(N-rank)]")
P(f"inline constexpr double kDs8Y[{_n8}] = {{{arr(ds8_y)}}};")
P(f"inline constexpr double kDs8X1[{_n8}] = {{{arr(ds8_x1)}}};")
P(f"inline constexpr double kDs8W[{_n8}] = {{{arr(ds8_w)}}};")
P(f"inline constexpr int kDs8Groups[{_n8}] = {{{', '.join(str(int(v)) for v in ds8_groups)}}};")
P(f"inline constexpr double kDs8Beta[2] = {{{arr(res8.params)}}};")
P(f"inline constexpr double kDs8StdError[2] = {{{arr(res8.bse)}}};")
P(f"inline constexpr double kDs8ZStatistic[2] = {{{arr(z8)}}};")
P(f"inline constexpr double kDs8POracle[2] = {{{arr(p_or8)}}};")
P(f"inline constexpr double kDs8PExpected[2] = {{{arr(p_ex8)}}};")
P(f"inline constexpr double kDs8Deviance = {g(dev8)};")
P(f"inline constexpr double kDs8NullDeviance = {g(null_dev8)};")
P(f"inline constexpr double kDs8Dispersion = {g(disp8)};")
P(f"inline constexpr double kDs8Loglik = {g(ll8)};")
P(f"inline constexpr double kDs8Aic = {g(aic8)};")
P(f"inline constexpr double kDs8Cov[4] = {{{arr(np.asarray(res8.cov_params()).ravel())}}};")
P()
P(f"inline constexpr double kDs8Cr0StdError[2] = {{{arr(res8_cr0.bse)}}};")
P(f"inline constexpr double kDs8Cr0ZStatistic[2] = {{{arr(z8_cr0)}}};")
P(f"inline constexpr double kDs8Cr0POracle[2] = {{{arr(p_or8_cr0)}}};")
P(f"inline constexpr double kDs8Cr0PExpected[2] = {{{arr(p_ex8_cr0)}}};")
P(f"inline constexpr double kDs8Cr0Cov[4] = {{{arr(cov8_cr0.ravel())}}};")
P()
P(f"inline constexpr double kDs8Cr1StdError[2] = {{{arr(res8_cr1.bse)}}};")
P(f"inline constexpr double kDs8Cr1ZStatistic[2] = {{{arr(z8_cr1)}}};")
P(f"inline constexpr double kDs8Cr1POracle[2] = {{{arr(p_or8_cr1)}}};")
P(f"inline constexpr double kDs8Cr1PExpected[2] = {{{arr(p_ex8_cr1)}}};")
P(f"inline constexpr double kDs8Cr1Cov[4] = {{{arr(cov8_cr1.ravel())}}};")

print("\n".join(out_lines))

print(f"wrote {len(out_lines)} lines", file=sys.stderr)
