#!/usr/bin/env python3
# Golden-value generator for the lin_hyp general-linear-hypothesis kernel (#20).
#
# statsmodels has exactly the three calls this function implements:
#   OLSResults.t_test(L)                  -> per-row estimate / se / t / p
#   OLSResults.wald_test(L, use_f=True)   -> the joint F, its numerator df, p
#   OLSResults.wald_test(L, use_f=False)  -> the same W referenced to chi-square
# so matching it validates our formulas rather than our re-derivation of them.
# The chi-square form matters on its own: taking chisq_p_value from our own F
# p-value would only re-check our arithmetic, not the reference distribution.
#
# This is an OFFLINE generator. Its output is hand-copied into the C++ test as
# golden constants; statsmodels is never a build or CI dependency. Regenerate
# with a throwaway venv that has numpy, scipy and statsmodels:
#     python -m venv venv && venv/Scripts/pip install numpy scipy statsmodels
#     venv/Scripts/python test/cpp/gen_lin_hyp_fixtures.py
#
# Every case runs on DS1 from test/cpp/test_lm_fit.cpp, whose fit is already
# validated, so only the hypothesis layer is under test here. Two of the cases
# need no new constants at all: a single-coefficient test must reproduce that
# coefficient's own estimate and standard error, and a test spanning every
# slope must reproduce the model F. Those identities are asserted below.

import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats

DS1_Y = [9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55]
DS1_X1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25]
DS1_X2 = [5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4]


def g(x):
    return f"{float(x):.15g}"


def joint(res, L):
    """The joint Wald test. scalar=True keeps the statistic 0-dimensional."""
    with warnings.catch_warnings():  # a redundant L warns by design
        warnings.simplefilter("ignore")
        return res.wald_test(np.atleast_2d(np.asarray(L, float)),
                             use_f=True, scalar=True)


def joint_chisq(res, L):
    """The same W referenced to chi-square(df_num), not F. use_f=False."""
    with warnings.catch_warnings():  # a redundant L warns by design
        warnings.simplefilter("ignore")
        return res.wald_test(np.atleast_2d(np.asarray(L, float)),
                             use_f=False, scalar=True)


def rows(res, L):
    """The per-row table, as four flat arrays."""
    t = res.t_test(np.atleast_2d(np.asarray(L, float)))
    return (np.ravel(t.effect), np.ravel(t.sd),
            np.ravel(t.tvalue), np.ravel(t.pvalue))


def emit(res, name, L, per_row=True):
    L = np.atleast_2d(np.asarray(L, float))
    w = joint(res, L)
    c = joint_chisq(res, L)
    print(f"// --- {name}: L = {L.tolist()} ---")
    print(f"// df_num        = {int(w.df_num)}")
    print(f"// df_den        = {g(w.df_denom)}")
    print(f"// f_statistic   = {g(w.statistic)}")
    print(f"// f_p_value     = {g(w.pvalue)}")
    print(f"// chisq         = {g(c.statistic)}")
    print(f"// chisq_p_value = {g(c.pvalue)}")
    if per_row:
        est, se, tv, pv = rows(res, L)
        print(f"// estimate      = {[g(v) for v in est]}")
        print(f"// std_error     = {[g(v) for v in se]}")
        print(f"// t_statistic   = {[g(v) for v in tv]}")
        print(f"// p_value       = {[g(v) for v in pv]}")
    print()
    # The chi-square call must report the same W the F call scales, or the two
    # statsmodels conventions have drifted apart and the goldens are not
    # comparable.
    assert np.isclose(float(c.statistic),
                      float(w.statistic) * int(w.df_num)), \
        f"{name}: chi-square W != F * df_num"
    return w


y = np.asarray(DS1_Y, float)
X = np.column_stack([DS1_X1, DS1_X2]).astype(float)
Xd = sm.add_constant(X, prepend=True)
fit = sm.OLS(y, Xd).fit(use_t=True)
hc1 = sm.OLS(y, Xd).fit(cov_type="HC1", use_t=True)

print(f"// DS1: n = {len(y)}, k = {Xd.shape[1]}, df_residual = {int(fit.df_resid)}")
print(f"// sigma2 = {g(fit.scale)}")
print()

w_single = emit(fit, "single coefficient (Type III on x1)", [[0, 1, 0]])
emit(fit, "contrast x1 - x2", [[0, 1, -1]])
w_joint = emit(fit, "joint test of both slopes", [[0, 1, 0], [0, 0, 1]])
w_redundant = emit(fit, "redundant constraints", [[0, 1, -1], [0, -1, 1]],
                   per_row=False)
emit(hc1, "single coefficient, HC1 covariance", [[0, 1, 0]])

# A hypothesis DS1 does NOT reject. The five cases above all have a huge W, so
# their chi-square tail is below 1e-16 and our `1 - CDF` returns exactly 0
# (distributions.hpp, shared with lm_core, deliberately unchanged). A
# chisq_p_value golden therefore needs a hypothesis the data is consistent
# with: beta[1] + 2*beta[2] is about 0.1, so this one is weak.
emit(fit, "weak single: beta1 + 2*beta2 = 0", [[0, 1, 2]])

# ---- Identities asserted on every run, so a statsmodels change fails loudly.
# For q = 1 the joint F is the squared t.
est, se, tv, pv = rows(fit, [[0, 1, 0]])
assert np.isclose(tv[0] ** 2, float(w_single.statistic)), "q=1: F != t^2"
# A single-coefficient test reproduces that coefficient's own estimate and se.
assert np.isclose(est[0], fit.params[1]) and np.isclose(se[0], fit.bse[1]), \
    "single-coefficient test does not reproduce beta[1] / se[1]"
# Spanning every slope reproduces the model F.
assert np.isclose(float(w_joint.statistic), fit.fvalue), \
    "joint test over all slopes != model F"
# A redundant L is rank 1, so it equals the single contrast it duplicates.
assert int(w_redundant.df_num) == 1, "redundant L did not reduce to rank 1"
assert np.isclose(float(w_redundant.statistic),
                  float(joint(fit, [[0, 1, -1]]).statistic)), \
    "redundant L != the contrast it duplicates"
# ss = chisq * sigma2 under the classical covariance.
chisq_joint = float(w_joint.statistic) * int(w_joint.df_num)
print(f"// joint ss = chisq * sigma2 = {g(chisq_joint * fit.scale)}")

# ---- Scale invariance of the joint test, which the oracle also has.
# Rescaling a predictor rescales its coefficient and that row and column of
# the covariance by the inverse factor, so the test of "both slopes are zero"
# is the same hypothesis and must give the same df_num and the same F. Our own
# implementation got this wrong before #20's review: pinv decides the rank with
# a tolerance relative to C's largest singular value, so widely different
# constraint variances read as redundant. See
# notes/engineering/2026-10-lin-hyp-rank-tolerance-and-scale.md.
print("// --- scale invariance of the joint test ---")
for s1, s2 in ((1.0, 1.0), (1e3, 1e-3), (1e6, 1.0)):
    Xs = sm.add_constant(np.column_stack([np.asarray(DS1_X1, float) * s1,
                                          np.asarray(DS1_X2, float) * s2]),
                         prepend=True)
    ws = joint(sm.OLS(y, Xs).fit(use_t=True), [[0, 1, 0], [0, 0, 1]])
    print(f"// x1*{s1:g}, x2*{s2:g}: df_num = {int(ws.df_num)}, "
          f"f_statistic = {g(ws.statistic)}")
    assert int(ws.df_num) == int(w_joint.df_num), "rescaling moved df_num"
    assert np.isclose(float(ws.statistic), float(w_joint.statistic)), \
        "rescaling moved the joint F"
print()

# ---- A q = 2 chi-square tail, on a hand-built covariance.
# Every full-rank two-row hypothesis on DS1's two slopes is the same hypothesis
# as [[0,1,0],[0,0,1]], so it always reproduces the model F — there is no weak
# q = 2 case to be had on DS1. The chi-square reference for df_num = 2 is
# therefore pinned on a hand-built C instead: C = I and theta = (1, 1) give
# W = 2 exactly, so the golden is the chi-square tail itself. scipy, not
# statsmodels, because there is no fit here.
print("// --- q = 2 chi-square tail, C = I and theta = (1, 1), so W = 2 ---")
print(f"// chisq_p_value, df_num 2 = {g(stats.chi2.sf(2.0, 2))}")
print(f"//   the same W on df_den 9 = {g(stats.chi2.sf(2.0, 9))}  (must NOT match)")
print()
print("// all identities asserted OK", file=__import__("sys").stderr)
