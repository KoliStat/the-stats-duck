#!/usr/bin/env python3
# Golden-value generator for the lin_hyp general-linear-hypothesis kernel (#20).
#
# statsmodels has exactly the two calls this function implements:
#   OLSResults.t_test(L)                 -> per-row estimate / se / t / p
#   OLSResults.wald_test(L, use_f=True)  -> the joint F, its numerator df, p
# so matching it validates our formulas rather than our re-derivation of them.
#
# This is an OFFLINE generator. Its output is hand-copied into the C++ test as
# golden constants; statsmodels is never a build or CI dependency. Regenerate
# with a throwaway venv that has numpy + statsmodels:
#     python -m venv venv && venv/Scripts/pip install numpy statsmodels
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


def rows(res, L):
    """The per-row table, as four flat arrays."""
    t = res.t_test(np.atleast_2d(np.asarray(L, float)))
    return (np.ravel(t.effect), np.ravel(t.sd),
            np.ravel(t.tvalue), np.ravel(t.pvalue))


def emit(res, name, L, per_row=True):
    L = np.atleast_2d(np.asarray(L, float))
    w = joint(res, L)
    print(f"// --- {name}: L = {L.tolist()} ---")
    print(f"// df_num      = {int(w.df_num)}")
    print(f"// df_den      = {g(w.df_denom)}")
    print(f"// f_statistic = {g(w.statistic)}")
    print(f"// f_p_value   = {g(w.pvalue)}")
    print(f"// chisq       = {g(float(w.statistic) * int(w.df_num))}")
    if per_row:
        est, se, tv, pv = rows(res, L)
        print(f"// estimate    = {[g(v) for v in est]}")
        print(f"// std_error   = {[g(v) for v in se]}")
        print(f"// t_statistic = {[g(v) for v in tv]}")
        print(f"// p_value     = {[g(v) for v in pv]}")
    print()
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
print("// all identities asserted OK", file=__import__("sys").stderr)
