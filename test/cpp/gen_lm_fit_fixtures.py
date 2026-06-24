#!/usr/bin/env python3
# Golden-value generator for the lm_core regression kernel (Epic 1.1).
#
# Emits reference values (OLS coefficients, classical + HC0/HC1/HC2/HC3 robust
# standard errors, t / p, R^2, adj-R^2, F, sigma) for the small deterministic
# datasets baked into test/cpp/test_lm_fit.cpp. statsmodels is the oracle: its
# cov_type='HC*' implementations are the standard reference the rest of the
# ecosystem is validated against, so matching them validates OUR formulas (not
# just our re-derivation of them).
#
# This is an OFFLINE generator — its output is hand-copied into the C++ test as
# golden constants. It is NOT part of any build (statsmodels is not a build dep).
# Regenerate with a venv that has numpy + statsmodels:
#     python -m venv venv && venv/Scripts/pip install numpy statsmodels
#     venv/Scripts/python test/cpp/gen_lm_fit_fixtures.py
#
# Datasets are chosen so HC0/HC1/HC2/HC3 differ visibly (DS1 has a high-leverage
# point), so a bug in any single weighting is caught.

import numpy as np
import statsmodels.api as sm

VCOVS = ["nonrobust", "HC0", "HC1", "HC2", "HC3"]


def g(x):
    return f"{x:.15g}"


def emit(name, y, Xcols, has_intercept):
    """Xcols: list of predictor columns (WITHOUT the intercept)."""
    y = np.asarray(y, float)
    X = np.column_stack(Xcols).astype(float)
    if has_intercept:
        Xd = sm.add_constant(X, prepend=True)  # intercept first, matches lm_core
        terms = ["(Intercept)"] + [f"x{j+1}" for j in range(X.shape[1])]
    else:
        Xd = X
        terms = [f"x{j+1}" for j in range(X.shape[1])]

    base = sm.OLS(y, Xd).fit(use_t=True)  # classical; source of beta/R2/F/sigma
    n, k = Xd.shape

    print(f"// ===== dataset '{name}' : n={n}, k={k}, "
          f"intercept={'true' if has_intercept else 'false'} =====")
    print(f"// terms: {terms}")
    print(f"// y  = {[g(v) for v in y]}")
    for j in range(X.shape[1]):
        print(f"// x{j+1} = {[g(v) for v in X[:, j]]}")
    print(f"// beta          = {[g(v) for v in base.params]}")
    print(f"// se_nonrobust  = {[g(v) for v in base.bse]}")
    print(f"// r_squared     = {g(base.rsquared)}")
    print(f"// adj_r_squared = {g(base.rsquared_adj)}")
    print(f"// f_statistic   = {g(base.fvalue)}")
    print(f"// f_p_value     = {g(base.f_pvalue)}")
    print(f"// sigma         = {g(np.sqrt(base.scale))}")
    print(f"// df_resid      = {int(base.df_resid)}")
    for cov in VCOVS:
        r = sm.OLS(y, Xd).fit(cov_type=cov, use_t=True) if cov != "nonrobust" \
            else base
        tag = cov
        print(f"// [{tag:9s}] se = {[g(v) for v in r.bse]}")
        print(f"// [{tag:9s}] t  = {[g(v) for v in r.tvalues]}")
        print(f"// [{tag:9s}] p  = {[g(v) for v in r.pvalues]}")
    print()


# ---- DS1: intercept, 2 predictors, n=12, one high-leverage point (x1=25) ----
emit(
    "ds1_hetero",
    y=[9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55],
    Xcols=[
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25],   # x1 — last point high leverage
        [5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4],       # x2
    ],
    has_intercept=True,
)

# ---- DS2: NO intercept, 2 predictors, n=10 (uncentered R^2/F path) ----
emit(
    "ds2_noint",
    y=[3, 5, 11, 10, 19, 18, 27, 26, 36, 34],
    Xcols=[
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, 1, 4, 3, 6, 5, 8, 7, 10, 9],
    ],
    has_intercept=False,
)

# ---- DS3: intercept, 1 predictor, n=8 (simplest sanity, ~y=2x) ----
emit(
    "ds3_simple",
    y=[2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1],
    Xcols=[[1, 2, 3, 4, 5, 6, 7, 8]],
    has_intercept=True,
)
