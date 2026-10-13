#!/usr/bin/env python3
# Golden-value generator for the lm_core regression kernel (Epic 1.1).
#
# Emits reference values (OLS coefficients, classical + HC0/HC1/HC2/HC3 robust
# standard errors, t / p, R^2, adj-R^2, F, sigma) for the small deterministic
# datasets stored in test/cpp/test_lm_fit.cpp. statsmodels is the oracle: its
# cov_type='HC*' implementations are the standard reference that other
# statistics packages are validated against, so matching them validates our
# formulas (not just our re-derivation of them).
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

import warnings

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
    print(f"// rank          = {int(base.df_model) + (1 if has_intercept else 0)}")
    print(f"// loglik        = {g(base.llf)}")
    # The Gaussian MLE log-likelihood lm_core implements; statsmodels' .llf is
    # the same quantity. Checked here so a divergence in either definition
    # fails the generator, not the C++ test.
    rss = float(base.ssr)
    assert np.isclose(base.llf, -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)), \
        "statsmodels .llf != -n/2 (ln 2pi + ln(RSS/n) + 1)"
    for cov in VCOVS:
        r = sm.OLS(y, Xd).fit(cov_type=cov, use_t=True) if cov != "nonrobust" \
            else base
        tag = cov
        print(f"// [{tag:9s}] se = {[g(v) for v in r.bse]}")
        print(f"// [{tag:9s}] t  = {[g(v) for v in r.tvalues]}")
        print(f"// [{tag:9s}] p  = {[g(v) for v in r.pvalues]}")
        # Full coefficient covariance, row-major k*k, aligned to `terms`. Its
        # diagonal must be bse^2 (self-check of the layout we copy into C++).
        C = np.asarray(r.cov_params())
        assert np.allclose(np.sqrt(np.diag(C)), np.asarray(r.bse)), \
            "cov_params diagonal != bse^2"
        print(f"// [{tag:9s}] cov (row-major) = {[g(v) for v in C.ravel()]}")
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
    print(f"// rank            = {k}")
    print(f"// loglik          = {g(cr1.llf)}   // same for CR0: vcov does not change beta")
    for tag, r in (("CR0", cr0), ("CR1", cr1)):
        print(f"// [{tag}] se = {[g(v) for v in r.bse]}")
        print(f"// [{tag}] t  = {[g(v) for v in r.tvalues]}")
        print(f"// [{tag}] p  = {[g(v) for v in r.pvalues]}")
        C = np.asarray(r.cov_params())
        assert np.allclose(np.sqrt(np.diag(C)), np.asarray(r.bse)), \
            "cov_params diagonal != bse^2"
        print(f"// [{tag}] cov (row-major) = {[g(v) for v in C.ravel()]}")
    print()


def emit_deficient(name, y, full_cols, kept_pred, label, groups=None,
                   has_intercept=True):
    """Rank-deficient design goldens by REDUCED-MODEL EQUIVALENCE.

    R-style fitting DROPS a dependent column, so a deficient fit is
    definitionally the fit of the design with that column removed. statsmodels
    cannot oracle the dropped fit directly: it solves with pinv, whose
    minimum-norm solution keeps every column and spreads the coefficient across
    the dependent set. So the oracle is the reduced design, and this function
    records both sides, asserting what the two conventions share and what they
    do not.

    full_cols   every predictor column, in the order fit_lm receives them.
    kept_pred   indices into full_cols that survive selection (the rest alias).
    groups      cluster labels; when given, CR1 goldens are emitted too.
    """
    y = np.asarray(y, float)
    Xf = np.column_stack(full_cols).astype(float)
    Xr = Xf[:, list(kept_pred)]
    Rf = sm.add_constant(Xf, prepend=True) if has_intercept else Xf
    Rr = sm.add_constant(Xr, prepend=True) if has_intercept else Xr
    n, k_red = Rr.shape
    k_full = Rf.shape[1]
    kept_design = ([0] + [i + 1 for i in kept_pred]) if has_intercept         else list(kept_pred)

    assert int(np.linalg.matrix_rank(Rr)) == k_red, "the reduced design is not full rank"
    assert int(np.linalg.matrix_rank(Rf)) == k_red, "the dropped columns are not dependent"

    red = sm.OLS(y, Rr).fit(use_t=True)
    with warnings.catch_warnings():  # the deficient fit warns by design
        warnings.simplefilter("ignore")
        defi = sm.OLS(y, Rf).fit(use_t=True)  # pinv / minimum-norm, NOT our convention

    # The two conventions pick different bases for the same column space, so
    # everything determined by the FITTED VALUES agrees. These are exactly the
    # quantities the C++ test asserts on the dropped fit.
    assert int(defi.df_resid) == n - k_red, "statsmodels df_resid != n - rank"
    for attr in ("ssr", "llf", "rsquared", "rsquared_adj", "fvalue"):
        assert np.isclose(getattr(red, attr), getattr(defi, attr)), f"{attr} differs"
    # ... and the coefficients do not agree, which is why the reduced fit is the
    # oracle rather than statsmodels' own deficient fit.
    assert not np.allclose(np.asarray(defi.params)[kept_design],
                           np.asarray(red.params)),         "pinv coefficients unexpectedly equal the dropped ones"

    print(f"// ===== rank-deficient dataset '{name}' : n={n}, k={k_full}, "
          f"rank={k_red} =====")
    print(f"// {label}")
    print(f"// kept design columns = {kept_design}   (the rest are aliased -> NaN)")
    print(f"// df_residual     = {n - k_red}   // n - rank, NOT n - k")
    print(f"// beta  (reduced) = {[g(v) for v in red.params]}")
    print(f"// se    (reduced) = {[g(v) for v in red.bse]}")
    print(f"// sigma           = {g(np.sqrt(red.scale))}")
    print(f"// r_squared       = {g(red.rsquared)}")
    print(f"// adj_r_squared   = {g(red.rsquared_adj)}")
    print(f"// f_statistic     = {g(red.fvalue)}")
    print(f"// loglik          = {g(red.llf)}")
    print(f"// cov   (reduced, row-major) = "
          f"{[g(v) for v in np.asarray(red.cov_params()).ravel()]}")
    if groups is None:
        hc1 = sm.OLS(y, Rr).fit(cov_type="HC1", use_t=True)
        print(f"// [HC1] se  (reduced) = {[g(v) for v in hc1.bse]}")
        print(f"// [HC1] cov (reduced, row-major) = "
              f"{[g(v) for v in np.asarray(hc1.cov_params()).ravel()]}")
    else:
        grp = np.asarray(groups, int)
        cr1 = sm.OLS(y, Rr).fit(cov_type="cluster",
                                cov_kwds={"groups": grp}, use_t=True)
        G = int(np.unique(grp).size)
        print(f"// [CR1] G = {G}, factor c = "
              f"{g((G / (G - 1)) * ((n - 1) / (n - k_red)))}   // (N-1)/(N-rank)")
        print(f"// [CR1] se  (reduced) = {[g(v) for v in cr1.bse]}")
        print(f"// [CR1] cov (reduced, row-major) = "
              f"{[g(v) for v in np.asarray(cr1.cov_params()).ravel()]}")
    print(f"// statsmodels' OWN deficient fit (pinv) for contrast:")
    print(f"//   params = {[g(v) for v in defi.params]}   // spread, not aliased")
    print(f"//   rank = {int(np.linalg.matrix_rank(Rf))}, "
          f"df_resid = {int(defi.df_resid)}, ssr matches the reduced fit")
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

# ---- DS5: rank-deficient variants of DS1 (reduced-model equivalence) ----
# Every reduced design below is DS1's [1, x1, x2], so DS1's goldens above are
# the oracle and the C++ test reuses them directly instead of carrying copies.
_ds1_y = [9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55]
_ds1_x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25]
_ds1_x2 = [5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4]
_ds1_x3 = [a + b for a, b in zip(_ds1_x1, _ds1_x2)]
_ds1_2x1 = [2.0 * v for v in _ds1_x1]

# Trailing alias: x3 = x1 + x2, so the LAST column is dropped.
emit_deficient("ds5_sum", _ds1_y, [_ds1_x1, _ds1_x2, _ds1_x3], [0, 1],
               "x3 = x1 + x2 (dropped); HC1 goldens exercise the n/(n-rank) factor")

# Middle alias: the duplicate sits BETWEEN the kept columns, so the kept design
# index set is non-contiguous and a scatter that assumes a prefix fails here.
emit_deficient("ds5_middle", _ds1_y, [_ds1_x1, _ds1_2x1, _ds1_x2], [0, 2],
               "2*x1 sits between the kept columns and is dropped")

# Clustered deficient fit: DS4's design with x2 duplicated. CR1's finite-sample
# factor uses (N-1)/(N-rank), so a k-based factor gives visibly wrong SEs.
emit_deficient("ds5_cluster", _y, [_x1, _x2, _x2], [0, 1],
               "x2 duplicated (dropped); CR1 over DS4's 5 clusters",
               groups=_groups)
