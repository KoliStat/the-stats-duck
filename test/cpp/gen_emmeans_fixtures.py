#!/usr/bin/env python3
"""Goldens for emmeans.

Oracle: statsmodels OLS .t_test, the same oracle gen_lin_hyp_fixtures.py
uses, plus scipy.stats.studentized_range for the Tukey p-values and the
Tukey confidence limits.

Design DS1: three groups, cell-means coded (no intercept), so L = I(3)
and each marginal mean is that group's sample mean.
    g1 = [1, 2, 3]      mean 2
    g2 = [4, 6, 8]      mean 6
    g3 = [5, 5, 8]      mean 6
Within-group SS = 2 + 8 + 6 = 16 on 6 df, so s^2 = 8/3 and every mean has
standard error sqrt(s^2/3) = 0.9428090416, every difference
sqrt(2*s^2/3) = 1.3333333333. Those are checkable without an oracle and
the test asserts them directly.

Design DS2 is rank deficient, and it is also the only design here whose
covariance has off-diagonal entries, so it is the one that exercises the
cross term in the pairwise variance. Its base data is the 12-row
regression that test_lm_fit.cpp and test_lin_hyp.cpp both pin and both
call DS1; here it is DS2, because DS1 is the three-group design above.
The design is [1, x1, x2, x1 + x2]: the last column is the sum of two
others, so the fit is rank 3.

This repository drops the LATER column of a dependent set, the way R's
dqrdc2 does, and reports that coefficient and its row and column of the
covariance as NaN. statsmodels spreads the coefficient across the
dependent set instead, so it cannot oracle a rank-deficient fit directly
(docs/fitter_conventions.md). So DS2 is oracled by fitting the REDUCED
design [1, x1, x2] and assembling the full-width beta and cov here, with
NaN in the dropped position. That is what lm_fit actually produces.

Each mean comes from t_test(L_row). Each pair comes from
t_test(L_i - L_j), which reaches the pairwise standard error by a route
that does not use C_ii + C_jj - 2*C_ij, so it independently checks the
shortcut the kernel takes.

Self-checks, which must hold before anything is written:
  1. mean[g] equals the sample mean of group g
  2. se[g] equals sqrt(s^2/n_g)
  3. the pairwise se from t_test equals sqrt(C_ii + C_jj - 2*C_ij)
  4. at two means, the Tukey p equals the unadjusted p

This is an OFFLINE generator. It writes test/cpp/emmeans_fixtures.hpp on
stdout and reports the self-checks on stderr. statsmodels and scipy are
never a build or CI dependency. Regenerate with a throwaway venv:
    python -m venv venv && venv/Scripts/pip install statsmodels scipy
    venv/Scripts/python test/cpp/gen_emmeans_fixtures.py > test/cpp/emmeans_fixtures.hpp
"""

import sys
import warnings

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats

SR = stats.studentized_range

ALPHA = 0.05

# ---- DS1: three groups, cell-means coded.
DS1_GROUPS = ([1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [5.0, 5.0, 8.0])

# ---- DS2: the 12-row regression test_lm_fit.cpp pins, plus a dependent
# fourth column. Same numbers as gen_lin_hyp_fixtures.py's DS1.
DS2_Y = [9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55]
DS2_X1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25]
DS2_X2 = [5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4]

# The DS2 grid. Rows 0 and 1 are predictions at two covariate settings and
# put no weight on the dependent column. Row 2 puts weight on it, so the
# kernel must report that row alone as not estimable.
DS2_L = [
    [1.0, 1.0, 5.0, 0.0],
    [1.0, 10.0, 3.0, 0.0],
    [1.0, 1.0, 5.0, 1.0],
]
DS2_ALIASED = 3  # the column lm_fit drops


def g(x):
    """17 significant digits, which round-trips a double exactly."""
    return f"{float(x):.17e}"


def fail(msg):
    print(f"SELF-CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def row_test(res, row):
    """statsmodels' per-row table for one linear combination of beta."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = res.t_test(np.atleast_2d(np.asarray(row, float)))
    lo, hi = np.ravel(t.conf_int(alpha=ALPHA))
    return dict(estimate=float(np.ravel(t.effect)[0]),
                std_error=float(np.ravel(t.sd)[0]),
                t_statistic=float(np.ravel(t.tvalue)[0]),
                p_value=float(np.ravel(t.pvalue)[0]),
                lower_cl=float(lo), upper_cl=float(hi))


def pairs(n):
    """i < j, row-major: the kernel's own enumeration order."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# --------------------------------- DS1 --------------------------------------
y1 = np.concatenate([np.asarray(gp, float) for gp in DS1_GROUPS])
X1 = np.zeros((len(y1), 3))
start = 0
for col, gp in enumerate(DS1_GROUPS):
    X1[start:start + len(gp), col] = 1.0
    start += len(gp)

fit1 = sm.OLS(y1, X1).fit(use_t=True)
beta1 = np.asarray(fit1.params, float)
cov1 = np.asarray(fit1.cov_params(), float)
df1 = float(fit1.df_resid)
L1 = np.eye(3)
C1 = L1 @ cov1 @ L1.T

ds1_means = [row_test(fit1, L1[i]) for i in range(3)]
ds1_pairs = pairs(3)
ds1_diffs = [row_test(fit1, L1[i] - L1[j]) for (i, j) in ds1_pairs]

# ---- Self-check 1: each marginal mean is that group's sample mean.
worst_mean = 0.0
for gi, gp in enumerate(DS1_GROUPS):
    want = float(np.mean(gp))
    worst_mean = max(worst_mean, abs(ds1_means[gi]["estimate"] - want))
# The band is the arithmetic's own noise: both sides are exact small integers
# divided by 3, and the fit reaches them through a QR solve.
if not worst_mean < 1e-12:
    fail(f"a marginal mean is not the group mean (worst {worst_mean:.3e})")
print(f"self-check 1 OK: each mean is the group mean, worst {worst_mean:.3e}",
      file=sys.stderr)

# ---- Self-check 2: each standard error is sqrt(s^2 / n_g).
s2 = float(fit1.scale)
if not abs(s2 - 8.0 / 3.0) < 1e-12:
    fail(f"s^2 is {s2!r}, not 8/3")
worst_se = 0.0
for gi, gp in enumerate(DS1_GROUPS):
    want = np.sqrt(s2 / len(gp))
    worst_se = max(worst_se, abs(ds1_means[gi]["std_error"] - want))
if not worst_se < 1e-12:
    fail(f"a standard error is not sqrt(s^2/n_g) (worst {worst_se:.3e})")
print(f"self-check 2 OK: se[g] = sqrt(s^2/n_g), worst {worst_se:.3e}; "
      f"s^2 = 8/3, mean se = {np.sqrt(s2 / 3.0):.10f}, pair se = "
      f"{np.sqrt(2.0 * s2 / 3.0):.10f}", file=sys.stderr)

# --------------------------------- DS2 --------------------------------------
y2 = np.asarray(DS2_Y, float)
X2_red = sm.add_constant(np.column_stack([DS2_X1, DS2_X2]).astype(float),
                         prepend=True)
# The full design, for the record: its last column is the sum of two others,
# so independent_columns keeps 0, 1, 2 and drops 3.
X2_full = np.column_stack([X2_red,
                           np.asarray(DS2_X1, float) + np.asarray(DS2_X2, float)])
if np.linalg.matrix_rank(X2_full) != 3:
    fail("DS2's full design is not rank 3")
if not np.allclose(X2_full[:, 3], X2_full[:, 1] + X2_full[:, 2]):
    fail("DS2's last column is not the sum of columns 1 and 2")

fit2 = sm.OLS(y2, X2_red).fit(use_t=True)
beta2_red = np.asarray(fit2.params, float)
cov2_red = np.asarray(fit2.cov_params(), float)
df2 = float(fit2.df_resid)

# Assemble the full-width fit lm_fit would report: the reduced estimates in
# the kept positions, NaN in the dropped one and in its row and column of cov.
beta2 = np.full(4, np.nan)
cov2 = np.full((4, 4), np.nan)
keep = [c for c in range(4) if c != DS2_ALIASED]
beta2[keep] = beta2_red
for a, ka in enumerate(keep):
    for b, kb in enumerate(keep):
        cov2[ka, kb] = cov2_red[a, b]

# A cross-check against a fit this repository already validates: the reduced
# DS2 fit is test_lin_hyp.cpp's kBeta / kCov, transcribed there to 15 digits.
LIN_HYP_BETA = [10.1205711006471, 2.05794012307408, -0.978656740125124]
LIN_HYP_COV00 = 4.30197288655079
if not np.allclose(beta2_red, LIN_HYP_BETA, rtol=1e-12, atol=0.0):
    fail(f"DS2's reduced fit does not reproduce test_lin_hyp.cpp's kBeta: "
         f"{beta2_red!r}")
if not abs(cov2_red[0, 0] - LIN_HYP_COV00) < 1e-11:
    fail(f"DS2's reduced covariance does not reproduce kCov(0,0): "
         f"{cov2_red[0, 0]!r}")

L2 = np.asarray(DS2_L, float)
# Only rows 0 and 1 are estimable, and dropping their (zero) weight on the
# aliased column turns them into rows of the reduced design.
if np.any(L2[:2, DS2_ALIASED] != 0.0):
    fail("DS2 grid rows 0 and 1 must put no weight on the aliased column")
if L2[2, DS2_ALIASED] == 0.0:
    fail("DS2 grid row 2 must put weight on the aliased column")
L2_red = L2[:2][:, keep]
C2 = L2_red @ cov2_red @ L2_red.T

ds2_means = [row_test(fit2, L2_red[i]) for i in range(2)]
ds2_diff01 = row_test(fit2, L2_red[0] - L2_red[1])

# ---- Self-check 3: the shortcut the kernel takes is the right formula.
# t_test(L_i - L_j) forms the standard error from the reduced design, which
# never builds C, so agreement is independent confirmation that
# sqrt(C_ii + C_jj - 2*C_ij) is the pairwise standard error. DS1's C is
# diagonal, so the cross term is zero there; DS2's is not, which is why DS2
# is in this file at all.
worst_sw = 0.0
worst_sw_where = ""
for label, C, pair_rows, pair_list in (
        ("DS1", C1, ds1_diffs, ds1_pairs),
        ("DS2", C2, [ds2_diff01], [(0, 1)])):
    off = max(abs(C[i, j]) for (i, j) in pair_list)
    for (i, j), row in zip(pair_list, pair_rows):
        want = np.sqrt(C[i, i] + C[j, j] - 2.0 * C[i, j])
        rel = abs(row["std_error"] - want) / want
        if rel > worst_sw:
            worst_sw, worst_sw_where = rel, f"{label} ({i},{j})"
    print(f"  {label}: largest |C_ij| over the reported pairs = {off:.6g}",
          file=sys.stderr)
# The band is the oracle's own noise. Both sides are a few arithmetic
# operations on the same covariance, so they agree to the last few bits.
if not worst_sw < 1e-13:
    fail(f"pairwise se != sqrt(C_ii + C_jj - 2*C_ij) "
         f"(worst relative {worst_sw:.3e} at {worst_sw_where})")
print(f"self-check 3 OK: pairwise se = sqrt(C_ii + C_jj - 2*C_ij), "
      f"worst relative {worst_sw:.3e}", file=sys.stderr)

# ------------- the adjustments, on DS1's three pairs ------------------------
m1 = len(ds1_pairs)          # three estimable pairs
k1 = 3                       # three estimable means, the Tukey family size

# Tukey: the tail of the studentized range at sqrt(2)*|t|, and the limit
# multiplier q(1 - alpha; k, df) / sqrt(2).
tukey_crit = float(SR.ppf(1.0 - ALPHA, k1, df1)) / np.sqrt(2.0)
for row in ds1_diffs:
    row["p_tukey"] = float(SR.sf(np.sqrt(2.0) * abs(row["t_statistic"]), k1, df1))
    row["tukey_lower"] = row["estimate"] - tukey_crit * row["std_error"]
    row["tukey_upper"] = row["estimate"] + tukey_crit * row["std_error"]

# Bonferroni: min(1, m*p), with limits from the t quantile at 1 - alpha/(2m).
bonf_crit = float(stats.t.ppf(1.0 - ALPHA / (2.0 * m1), df1))
for row in ds1_diffs:
    row["p_bonferroni"] = min(1.0, m1 * row["p_value"])
    row["bonf_lower"] = row["estimate"] - bonf_crit * row["std_error"]
    row["bonf_upper"] = row["estimate"] + bonf_crit * row["std_error"]

# Holm, the step-down form R's p.adjust computes. Here to check the wiring
# into statsduck::adjust_p, which test_p_adjust.cpp already tests on its own.
order = sorted(range(m1), key=lambda d: ds1_diffs[d]["p_value"])
cummax = 0.0
for rank0, d in enumerate(order):
    q = (m1 - rank0) * ds1_diffs[d]["p_value"]
    cummax = max(cummax, q)
    ds1_diffs[d]["p_holm"] = min(1.0, cummax)

# ---- Self-check 4: at two means the Tukey p is the unadjusted p.
# Q = sqrt(2)*|T| when k = 2, so the studentized range tail at sqrt(2)*|t|
# is exactly the two-sided t p-value. A Tukey implementation that has the
# sqrt(2) in the wrong place fails here and nowhere else.
worst_k2 = 0.0
for row in ds1_diffs:
    want = row["p_value"]
    got = float(SR.sf(np.sqrt(2.0) * abs(row["t_statistic"]), 2.0, df1))
    worst_k2 = max(worst_k2, abs(got - want))
# The band is the oracle's own quadrature, which the tukey fixtures measure
# at about 2e-13 against the same identity.
if not worst_k2 < 1e-11:
    fail(f"at k = 2 the Tukey p is not the unadjusted p (worst {worst_k2:.3e})")
print(f"self-check 4 OK: at k = 2 the Tukey p is the unadjusted p, "
      f"worst {worst_k2:.3e}", file=sys.stderr)

# --------------------------------- emit -------------------------------------
MEAN_FIELDS = ("estimate", "std_error", "t_statistic", "p_value",
               "lower_cl", "upper_cl")
DIFF_FIELDS = ("estimate", "std_error", "t_statistic", "p_value",
               "lower_cl", "upper_cl",
               "p_tukey", "tukey_lower", "tukey_upper",
               "p_bonferroni", "bonf_lower", "bonf_upper", "p_holm")


def emit_mean_row(row):
    return "    {" + ", ".join(g(row[f]) for f in MEAN_FIELDS) + "},"


def cell(row, field):
    """One emitted field, with NaN written as the header's own constant."""
    if field not in row:
        return "kEmmNaN"
    v = float(row[field])
    return "kEmmNaN" if np.isnan(v) else g(v)


def emit_diff_row(i, j, row):
    body = ", ".join(cell(row, f) for f in DIFF_FIELDS)
    return f"{{{i}, {j}, {body}}}"


P = print
P("// Generated by test/cpp/gen_emmeans_fixtures.py. Do not edit by hand.")
P("//")
P(f"// Oracle: statsmodels {statsmodels.__version__} OLSResults.t_test for every")
P("// estimate, standard error, t, p and confidence limit, and scipy")
P(f"// {scipy.__version__} stats.studentized_range for the Tukey p-values and")
P("// the Tukey limits. Both run offline; neither is a build or CI dependency.")
P("//")
P("// DS1: three groups, cell-means coded, so L = I(3) and each marginal mean")
P("// is that group's sample mean. s^2 = 8/3 on 6 df, so every mean has")
P("// standard error sqrt(8/9) and every difference sqrt(16/9) = 4/3. The test")
P("// asserts those by hand as well as against these goldens.")
P("//")
P("// DS2: rank deficient. The design is [1, x1, x2, x1 + x2] on the 12-row")
P("// regression test_lm_fit.cpp pins, so column 3 is the sum of columns 1 and")
P("// 2 and lm_fit drops it. beta and cov below are the full-width fit lm_fit")
P("// reports: the reduced estimates in the kept positions and NaN in the")
P("// dropped one, and in its row and column of cov. statsmodels spreads a")
P("// dropped coefficient across the dependent set instead of dropping it, so")
P("// the reduced design was oracled and the full width assembled from it")
P("// (docs/fitter_conventions.md). DS2's covariance is the only one here with")
P("// off-diagonal entries, so it is the case that exercises the cross term in")
P("// the pairwise variance.")
P("//")
P(f"// Confidence limits are at alpha = {ALPHA:g}.")
P("//")
P("// Values carry 17 significant digits, which round-trips a double exactly.")
P("//")
P("// Self-checks that passed before this file was written:")
P(f"//   each mean is its group's sample mean, worst {worst_mean:.3e} absolute")
P(f"//   se[g] = sqrt(s^2/n_g), worst {worst_se:.3e} absolute")
P(f"//   pairwise se = sqrt(C_ii + C_jj - 2*C_ij), worst {worst_sw:.3e} relative")
P(f"//   at k = 2 the Tukey p is the unadjusted p, worst {worst_k2:.3e} absolute")
P("#pragma once")
P()
P("#include <limits>")
P()
P("// NaN in a fixture is the aliased marker lm_fit writes, or a field this")
P("// case does not report. It is never a failure.")
P("inline constexpr double kEmmNaN = std::numeric_limits<double>::quiet_NaN();")
P()
P("struct EmmMeanCase {")
P("\tdouble estimate;")
P("\tdouble std_error;")
P("\tdouble t_statistic;")
P("\tdouble p_value;")
P("\tdouble lower_cl;")
P("\tdouble upper_cl;")
P("};")
P()
P("// One pair, with the three adjustments the test checks against an oracle.")
P("// lower_cl and upper_cl are the UNADJUSTED limits, estimate +- t(1-alpha/2,")
P("// df)*se, which is what holm, hochberg, BH, BY and none all report.")
P("// p_tukey and the tukey limits come from the studentized range; p_bonferroni")
P("// and the bonf limits from min(1, m*p) and t(1 - alpha/(2m)); p_holm from the")
P("// step-down form, which pins the wiring into statsduck::adjust_p.")
P("struct EmmDiffCase {")
P("\tunsigned i;")
P("\tunsigned j;")
P("\tdouble estimate;")
P("\tdouble std_error;")
P("\tdouble t_statistic;")
P("\tdouble p_value;")
P("\tdouble lower_cl;")
P("\tdouble upper_cl;")
P("\tdouble p_tukey;")
P("\tdouble tukey_lower;")
P("\tdouble tukey_upper;")
P("\tdouble p_bonferroni;")
P("\tdouble bonf_lower;")
P("\tdouble bonf_upper;")
P("\tdouble p_holm;")
P("};")
P()
P(f"inline constexpr double kEmmAlpha = {g(ALPHA)};")
P()
P("// -- DS1: three groups, cell-means coded ----------------------------------")
P(f"inline constexpr double kDs1Df = {g(df1)};")
P(f"inline constexpr double kDs1Sigma2 = {g(s2)}; // 8/3")
P("inline constexpr double kDs1Beta[3] = {" + ", ".join(g(b) for b in beta1) + "};")
P("inline constexpr double kDs1Cov[9] = {")
for i in range(3):
    P("    " + ", ".join(g(cov1[i, j]) for j in range(3)) + ("," if i < 2 else ""))
P("};")
P()
P("// L = I(3), in L's order.")
P("inline constexpr EmmMeanCase kDs1Means[3] = {")
for row in ds1_means:
    P(emit_mean_row(row))
P("};")
P()
P("// i < j, row-major, 0-based -- the kernel's own order.")
P("inline constexpr EmmDiffCase kDs1Diffs[3] = {")
for (i, j), row in zip(ds1_pairs, ds1_diffs):
    P("    " + emit_diff_row(i, j, row) + ",")
P("};")
P()
P("// The family sizes and limit multipliers the adjustments use on DS1.")
P(f"inline constexpr unsigned kDs1NumPairs = {m1};")
P(f"inline constexpr unsigned kDs1KCompare = {k1};")
P(f"inline constexpr double kDs1TukeyCrit = {g(tukey_crit)}; "
  "// qtukey(1-alpha, k, df)/sqrt(2)")
P(f"inline constexpr double kDs1BonfCrit = {g(bonf_crit)}; "
  "// t(1-alpha/(2m), df)")
P()
P("// -- DS2: rank deficient, cov with off-diagonal entries -------------------")
P(f"inline constexpr double kDs2Df = {g(df2)};")
P(f"inline constexpr unsigned kDs2AliasedColumn = {DS2_ALIASED};")
P("inline constexpr double kDs2Beta[4] = {" +
  ", ".join("kEmmNaN" if np.isnan(b) else g(b) for b in beta2) + "};")
P("inline constexpr double kDs2Cov[16] = {")
for i in range(4):
    cells = ", ".join("kEmmNaN" if np.isnan(cov2[i, j]) else g(cov2[i, j])
                      for j in range(4))
    P("    " + cells + ("," if i < 3 else ""))
P("};")
P()
P("// The grid: rows 0 and 1 avoid the dropped column, row 2 puts weight on it.")
P("inline constexpr double kDs2L[12] = {")
for i in range(3):
    P("    " + ", ".join(g(L2[i, j]) for j in range(4)) + ("," if i < 2 else ""))
P("};")
P()
P("// Only the two estimable rows have goldens; row 2 has no value to report.")
P("inline constexpr EmmMeanCase kDs2Means[2] = {")
for row in ds2_means:
    P(emit_mean_row(row))
P("};")
P()
P("// The one pair between two estimable rows. The DS2 test runs with")
P("// EmmAdjust::NONE, so the adjusted fields are not reported here and the")
P("// test asserts p_adjusted == p_value instead.")
P("inline constexpr EmmDiffCase kDs2Diff01 = " + emit_diff_row(0, 1, ds2_diff01) + ";")

print(f"all self-checks OK: DS1 {len(ds1_means)} means / {len(ds1_diffs)} pairs, "
      f"DS2 {len(ds2_means)} means / 1 pair", file=sys.stderr)
