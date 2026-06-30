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
# point), so a bug in any single weighting is caught. DS4 adds within-cluster
# error correlation so the cluster-robust SEs (CR0/CR1) differ sharply from HC.

import numpy as np
import statsmodels.api as sm
from scipy import stats

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


def emit_clustered(name, y, Xcols, groups, has_intercept=True):
    """Cluster-robust goldens: CR0 (raw sandwich) and CR1 (Stata/statsmodels
    default). Conventions are pinned to statsmodels and self-checked here:
      - CR1 == statsmodels default (use_correction=True); CR0 == use_correction
        =False (raw bread*M*bread).
      - CR1 cov == CR0 cov * c,  c = [G/(G-1)]*[(N-1)/(N-k)].
      - p-values use a t(G-1) reference (G = #clusters); the .df_resid attribute
        stays n-k. lm_core mirrors this: reported df_residual = n-k, the p-value
        df is G-1."""
    y = np.asarray(y, float)
    X = np.column_stack(Xcols).astype(float)
    groups = np.asarray(groups, int)
    Xd = sm.add_constant(X, prepend=True) if has_intercept else X
    terms = (["(Intercept)"] if has_intercept else []) + \
        [f"x{j+1}" for j in range(X.shape[1])]
    n, k = Xd.shape
    G = int(np.unique(groups).size)

    cr0 = sm.OLS(y, Xd).fit(cov_type="cluster",
                            cov_kwds={"groups": groups, "use_correction": False},
                            use_t=True)
    cr1 = sm.OLS(y, Xd).fit(cov_type="cluster",
                            cov_kwds={"groups": groups}, use_t=True)  # default

    c = (G / (G - 1)) * ((n - 1) / (n - k))
    assert np.allclose(np.asarray(cr1.bse) / np.asarray(cr0.bse), np.sqrt(c)), \
        "CR1/CR0 ratio != sqrt(c) — finite-sample factor mismatch"
    for r in (cr0, cr1):
        tv = np.asarray(r.tvalues)
        assert np.allclose(2 * stats.t.sf(np.abs(tv), G - 1),
                           np.asarray(r.pvalues)), "cluster p-values df != G-1"

    print(f"// ===== clustered dataset '{name}' : n={n}, k={k}, G={G}, "
          f"intercept={'true' if has_intercept else 'false'} =====")
    print(f"// terms: {terms}")
    print(f"// y      = {[g(v) for v in y]}")
    for j in range(X.shape[1]):
        print(f"// x{j+1}     = {[g(v) for v in X[:, j]]}")
    print(f"// groups = {[int(v) for v in groups]}")
    print(f"// beta            = {[g(v) for v in cr1.params]}")
    print(f"// n_clusters (G)  = {G}")
    print(f"// df_residual n-k = {int(cr1.df_resid)}   // reported unchanged")
    print(f"// df_infer   G-1  = {G - 1}   // t reference for cluster p-values")
    print(f"// CR1 factor c    = {g(c)}")
    for tag, r in (("CR0", cr0), ("CR1", cr1)):
        print(f"// [{tag}] se = {[g(v) for v in r.bse]}")
        print(f"// [{tag}] t  = {[g(v) for v in r.tvalues]}")
        print(f"// [{tag}] p  = {[g(v) for v in r.pvalues]}")
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

# ---- DS4: clustered, intercept, 2 predictors, G=5 uneven clusters, n=25 ----
# A per-cluster random effect injects within-cluster error correlation, so the
# cluster-robust SEs are much larger than HC would give (ignoring the grouping).
# Data rounded to 3 decimals so the C++ test can carry exact literals.
np.random.seed(12345)
_sizes = [3, 4, 5, 6, 7]
_groups = np.concatenate([np.full(s, gi) for gi, s in enumerate(_sizes)])
_n = _groups.size
_x1 = np.round(np.random.randn(_n) * 2 + np.arange(_n) * 0.1, 3)
_x2 = np.round(np.random.randn(_n), 3)
_u = np.random.randn(len(_sizes))                 # per-cluster effect
_eps = _u[_groups] * 1.5 + np.random.randn(_n) * 0.5
_y = np.round(1.0 + 2.0 * _x1 - 1.0 * _x2 + _eps, 3)
emit_clustered("ds4_cluster", _y, [_x1, _x2], _groups, has_intercept=True)
# Same data, NO intercept — exercises the 5-arg lm_fit(y,x,vcov,cluster,false).
emit_clustered("ds4_cluster_noint", _y, [_x1, _x2], _groups, has_intercept=False)
