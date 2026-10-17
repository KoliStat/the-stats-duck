# `lin_hyp` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `lin_hyp(L, beta, cov, sigma2, df)`, a stateless SQL function that tests the general linear hypothesis `Lβ = 0` against an already-computed fit and returns the per-row estimates with their standard errors alongside the joint test.

**Architecture:** A DuckDB-free kernel translation unit (`src/lin_hyp.hpp` / `src/lin_hyp.cpp`) holds the numerics and is unit-tested standalone with statsmodels goldens. A thin SQL wrapper (`src/lin_hyp_function.cpp`) unpacks the list arguments, calls the kernel once per row, and writes the result STRUCT. The kernel depends on `linalg` and `distributions` only — never on `lm_core` — because the function must work with any fit.

**Tech Stack:** C++17; the in-repo `linalg` kernel (`matvec`, `sandwich`, `pinv`) and `distributions` (`StudentTCDF`, `FCDF`, `ChiSquareCDF`); DuckDB's vector layer for the SQL wrapper; MSVC for the extension build and g++ for the standalone tests; statsmodels as an offline oracle.

**Spec:** `notes/plans/2026-10-14-lin-hyp-design.md`

## Global Constraints

- Branch `feat/20-lin-hyp`, already created off `v0.9`. Stage files **explicitly** — never `git add -A`, because the `duckdb` submodule is expected-dirty (the applied fmt patch).
- `docs/` is served by GitHub Pages from `main`; only finished public docs go there. Internal plans live in `notes/plans/`.
- statsmodels is the sole oracle and never a build or CI dependency: generators run offline in a throwaway virtualenv and their output is pasted in as constants.
- Never put a default-hash `std::string`-keyed `std::unordered_map` or `unordered_set` in extension code — it breaks the wasm build. `lin_hyp` needs no maps at all; `scripts/check-wasm-string-hash.sh` enforces this in CI.
- The return STRUCT's field order is part of the contract and append-only, per `docs/fitter_conventions.md`. A t-referenced statistic is named `t_statistic`, never a bare `statistic`. NaN maps to SQL NULL at the vector layer.
- `df` is `DOUBLE`, not `BIGINT`, and `df_den` echoes it as `DOUBLE`.
- Plain English in comments, commit messages and docs: short linear sentences, no buzzwords (`~/.claude/CLAUDE.md`).
- Absolute paths for build and test commands; the shell's working directory drifts in this repo.
- Commit message trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Review Focus

Five input classes the spec implies that no task's happy-path tests would otherwise exercise, most likely to bite first. Each one's test is added to the task that owns the code.

1. **More constraints than coefficients (q > k).** A generator emitting one row per level of a factor easily exceeds k. C is then singular by construction, so `df_num` must be the rank and the F must still be finite. Test in Task 2.
2. **A zero row inside an otherwise valid `L`.** Partial degeneracy, not a fully zero matrix: rank drops by one and the surviving constraints must still be tested. Test in Task 2.
3. **A zero variance on `cov`'s diagonal.** `lm_core` writes `std_error = 0` when a coefficient's variance is non-positive, so `std_error` is 0 and `t_statistic` and `p_value` must be NULL rather than ±inf. Test in Task 2.
4. **An empty `L` (q = 0).** An empty SQL list is easy to produce by accident from `list_filter`; it is misuse, not a NULL result, and must raise. Test in Task 2 (kernel) and Task 3 (SQL message).
5. **A NULL element inside `beta`, `cov` or an `L` row.** NULL is a SQL concept the kernel never sees, so only the wrapper can get this wrong; it must read NULL as NaN and return a NULL STRUCT rather than a garbage number. Test in Task 3.

---

### Task 1: The statsmodels oracle

**Files:**
- Create: `test/cpp/gen_lin_hyp_fixtures.py`

**Interfaces:**
- Consumes: nothing in this repo; DS1's data, copied from `test/cpp/test_lm_fit.cpp`.
- Produces: the golden constants Task 2 pastes into `test/cpp/test_lin_hyp.cpp`. No C++ symbol.

- [ ] **Step 1: Write the generator**

Create `test/cpp/gen_lin_hyp_fixtures.py`:

```python
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
```

- [ ] **Step 2: Run it and confirm the identities pass**

```bash
d="C:/Users/massi/AppData/Local/Temp/claude/C--Users-massi-source-repos-kolistat-the-stats-duck/996dafbd-cf73-4ee8-9e10-56b09afbffc4/scratchpad"
python -m venv "$d/venv"
"$d/venv/Scripts/python" -m pip install --quiet numpy scipy statsmodels
"$d/venv/Scripts/python" test/cpp/gen_lin_hyp_fixtures.py
```

Expected: no assertion error, `all identities asserted OK` on stderr, and output whose numbers match the constants Task 2 uses — in particular `f_statistic = 249.406353276441` for the single test, `81.1925089443785` for the contrast and for the redundant case, `132.532474368046` for the joint test, and `201.561980521578` for HC1.

- [ ] **Step 3: Commit**

```bash
git add test/cpp/gen_lin_hyp_fixtures.py
git commit -m "Add the statsmodels oracle for lin_hyp (#20)

Emits per-row t_test and joint wald_test goldens for five hypotheses on
DS1, and asserts four identities on every run: for q = 1 the joint F is
the squared t, a single-coefficient test reproduces that coefficient's
own estimate and standard error, a test spanning every slope reproduces
the model F, and a redundant hypothesis reduces to the contrast it
duplicates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The kernel

**Files:**
- Create: `src/include/lin_hyp.hpp`
- Create: `src/lin_hyp.cpp`
- Create: `test/cpp/test_lin_hyp.cpp`
- Modify: `scripts/run-cpp-tests.sh` (add the `test_lin_hyp` target after `test_lm_fit`)

**Interfaces:**
- Consumes: `statsduck::linalg::{Mat, matvec, sandwich, pinv, Pinv}` from `src/include/linalg.hpp`; `stats_duck::{StudentTCDF, FCDF, ChiSquareCDF}` from `src/include/distributions.hpp`. Note the two namespaces differ by an underscore, as in `lm_core.cpp`.
- Produces: `statsduck::LinHypResult` and `statsduck::lin_hyp(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V, double sigma2, double df)`. Task 3 calls exactly this.

- [ ] **Step 1: Write the header**

Create `src/include/lin_hyp.hpp`:

```cpp
#pragma once

// stats_duck general linear hypothesis — the test of L·β = 0 against an
// already-computed fit.
//
// DuckDB-free and Eigen-free at the boundary, exactly like linalg.hpp and
// lm_core.hpp, so it is directly unit-testable and shareable downstream. It
// deliberately does NOT depend on lm_core: the inputs are a coefficient
// vector and its covariance, whatever produced them, so the same function
// serves lm_fit today and glm_fit later.
//
// Scope: the Wald test of q linear constraints, with the per-row estimates and
// standard errors that SAS's ESTIMATE prints. Type I and II sums of squares are
// NOT here — they need nested refits and the term-to-column mapping, which this
// function cannot reconstruct from β and cov(β).

#include "linalg.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace statsduck {

// Three outcomes, which callers must distinguish:
//   ok = false                  → misuse; `error` says what. Callers raise.
//   ok = true, estimable = false → the hypothesis cannot be evaluated (it puts
//                                  weight on a coefficient the fit could not
//                                  estimate, or it carries no information).
//                                  Callers report SQL NULL, per the
//                                  failure-is-a-value rule in
//                                  docs/fitter_conventions.md.
//   ok = true, estimable = true  → the full result below.
struct LinHypResult {
	bool ok = false;
	std::string error;
	bool estimable = false;

	// All length q, one per row of L, in L's order.
	std::vector<double> estimate;    // L·β
	std::vector<double> std_error;   // √diag(L·V·Lᵀ); 0 where the variance is not positive
	std::vector<double> t_statistic; // estimate / std_error; NaN where std_error is 0
	std::vector<double> p_value;     // two-sided, t(df); NaN where t is NaN

	double ss = 0.0;        // chisq · sigma2 — a sum of squares only when V is σ²(XᵀX)⁻¹
	std::size_t df_num = 0; // rank of L·V·Lᵀ, which is q when L has full row rank
	double df_den = 0.0;    // the df argument, echoed
	double f_statistic = 0.0;
	double f_p_value = 0.0;
	double chisq = 0.0;     // the Wald statistic θᵀ(L·V·Lᵀ)⁻θ
	double chisq_p_value = 0.0;
};

// Test L·β = 0.
//   L       q × k, one row per constraint; q rows means a joint test.
//   beta    length k.
//   V       k × k covariance of beta, for whichever estimator produced it.
//   sigma2  residual variance, used only to scale the Wald statistic into `ss`.
//   df      denominator degrees of freedom for the t and F references; > 0.
//
// A rank-deficient fit carries NaN in the rows and columns of its aliased
// coefficients. A hypothesis that puts no weight there is still estimable, so
// those rows and columns are zeroed in a working copy before the product rather
// than being allowed to poison it (IEEE makes 0 · NaN = NaN).
LinHypResult lin_hyp(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V,
                     double sigma2, double df);

} // namespace statsduck
```

- [ ] **Step 2: Write a stub so the tests link**

Create `src/lin_hyp.cpp` with only this, so Step 4 fails on behavior rather than on linking:

```cpp
#include "lin_hyp.hpp"

namespace statsduck {

LinHypResult lin_hyp(const linalg::Mat &, const std::vector<double> &, const linalg::Mat &, double,
                     double) {
	return LinHypResult(); // stub: ok = false, replaced in Step 5
}

} // namespace statsduck
```

- [ ] **Step 3: Write the failing tests**

Create `test/cpp/test_lin_hyp.cpp`. Goldens come from `test/cpp/gen_lin_hyp_fixtures.py` (Task 1); the DS1 values that also appear in `test/cpp/test_lm_fit.cpp` are called out where they do.

```cpp
// Direct C++ unit tests for the statsduck::lin_hyp general linear hypothesis (#20).
//
// Golden values come from statsmodels via test/cpp/gen_lin_hyp_fixtures.py:
// OLSResults.t_test for the per-row table and OLSResults.wald_test(use_f=True)
// for the joint test. Every case runs on DS1, the dataset test_lm_fit.cpp
// already validates, so only the hypothesis layer is under test.
//
// Two cases carry no new constants on purpose. A single-coefficient test must
// reproduce that coefficient's own estimate and standard error (DS1's beta[1]
// and se[1]), and a test spanning both slopes must reproduce the model F
// (DS1's f_statistic). If the hypothesis layer is right those must agree.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "lin_hyp.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using statsduck::lin_hyp;
using statsduck::LinHypResult;
using statsduck::linalg::Mat;

static int g_checks = 0;
static int g_fail = 0;

static void check(bool cond, const char *expr, int line) {
	++g_checks;
	if (!cond) {
		++g_fail;
		std::printf("  FAIL (line %d): %s\n", line, expr);
	}
}
static void check_close(double a, double b, double tol, const char *label, int line) {
	++g_checks;
	const double d = std::fabs(a - b);
	if (!(d <= tol)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : |%.12g - %.12g| = %.3g > %.3g\n", line, label, a, b, d, tol);
	}
}
static void check_rel(double a, double b, double rtol, const char *label, int line) {
	++g_checks;
	const double denom = std::fabs(b) > 1e-300 ? std::fabs(b) : 1e-300;
	const double rd = std::fabs(a - b) / denom;
	if (!(rd <= rtol)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : rel|%.12g vs %.12g| = %.3g > %.3g\n", line, label, a, b, rd,
		            rtol);
	}
}
#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_CLOSE(a, b, tol) check_close((a), (b), (tol), #a " ~= " #b, __LINE__)

// Same bands as test_lm_fit.cpp, for the same reasons.
static const double TOL = 1e-6;    // estimates, standard errors
static const double TOL_F = 1e-3;  // F statistics
static const double RTOL_P = 1e-2; // p-values, relative

// Build a row-major Mat from a flat value list.
static Mat mat(std::size_t r, std::size_t c, std::vector<double> vals) {
	Mat m(r, c);
	m.data = std::move(vals);
	return m;
}

// ───────────────────────── DS1's fitted model ────────────────────────────────
// Classical OLS on DS1. beta and cov are the values test_lm_fit.cpp pins, so
// this file takes the fit as given and tests only the hypothesis layer.
static const std::vector<double> kBeta = {10.1205711006471, 2.05794012307408, -0.978656740125124};
static const Mat kCov = mat(3, 3, {4.30197288655079, -0.140244518800206, -0.523287599804694,
                                   -0.140244518800206, 0.0169807925681187, 0.00229470169839442,
                                   -0.523287599804694, 0.00229470169839442, 0.101177222385041});
static const double kSigma2 = 7.46416743950907; // DS1's sigma², = 2.73206285423836²
static const double kDf = 9.0;                  // DS1's df_residual

// HC1 covariance for the same fit, from test_lm_fit.cpp's ds1.hc1.cov golden.
static const Mat kCovHc1 = mat(3, 3, {5.35735031293671, -0.191061988599497, -0.649665858329423,
                                      -0.191061988599497, 0.0210114900597774, 0.0105710141727397,
                                      -0.649665858329423, 0.0105710141727397, 0.105966155376544});

// ───────────────── A single coefficient: Type III on x1 ──────────────────────
static void test_single_coefficient() {
	std::printf("single_coefficient (q=1, Type III on x1)\n");
	auto r = lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, kCov, kSigma2, kDf);
	CHECK(r.ok);
	CHECK(r.estimable);
	CHECK(r.df_num == 1);
	CHECK_CLOSE(r.df_den, 9.0, 0.0);
	CHECK(r.estimate.size() == 1 && r.std_error.size() == 1);

	// No new constants: a one-coefficient test reproduces that coefficient.
	CHECK_CLOSE(r.estimate[0], kBeta[1], TOL);
	CHECK_CLOSE(r.std_error[0], std::sqrt(kCov(1, 1)), TOL);

	CHECK_CLOSE(r.estimate[0], 2.05794012307408, TOL);
	CHECK_CLOSE(r.std_error[0], 0.130310370148038, TOL);
	CHECK_CLOSE(r.t_statistic[0], 15.792604385485, TOL);
	check_rel(r.p_value[0], 7.20757277116357e-08, RTOL_P, "single.p", __LINE__);
	CHECK_CLOSE(r.f_statistic, 249.406353276441, TOL_F);
	check_rel(r.f_p_value, 7.20757277116357e-08, RTOL_P, "single.f_p", __LINE__);

	// For q = 1 the Wald statistic is the squared t, and F equals it.
	CHECK_CLOSE(r.chisq, r.t_statistic[0] * r.t_statistic[0], TOL_F);
	CHECK_CLOSE(r.f_statistic, r.chisq, TOL_F);
	// ss = chisq · sigma² under the classical covariance.
	CHECK_CLOSE(r.ss, r.chisq * kSigma2, 1e-3);
}

// ───────────────── A contrast: x1 − x2, SAS's ESTIMATE ───────────────────────
static void test_contrast() {
	std::printf("contrast (x1 - x2)\n");
	auto r = lin_hyp(mat(1, 3, {0, 1, -1}), kBeta, kCov, kSigma2, kDf);
	CHECK(r.ok && r.estimable);
	CHECK(r.df_num == 1);
	// The estimate is the difference of the two coefficients.
	CHECK_CLOSE(r.estimate[0], kBeta[1] - kBeta[2], TOL);
	CHECK_CLOSE(r.estimate[0], 3.03659686319921, TOL);
	CHECK_CLOSE(r.std_error[0], 0.336999423673648, TOL);
	CHECK_CLOSE(r.t_statistic[0], 9.01068859435163, TOL);
	CHECK_CLOSE(r.f_statistic, 81.1925089443785, TOL_F);
	check_rel(r.f_p_value, 8.45554186853592e-06, RTOL_P, "contrast.f_p", __LINE__);
}

// ───────────────── A joint test: both slopes at once ─────────────────────────
static void test_joint() {
	std::printf("joint (both slopes, q=2)\n");
	auto r = lin_hyp(mat(2, 3, {0, 1, 0, 0, 0, 1}), kBeta, kCov, kSigma2, kDf);
	CHECK(r.ok && r.estimable);
	CHECK(r.df_num == 2);
	CHECK(r.estimate.size() == 2);
	// Each row still reports its own coefficient.
	CHECK_CLOSE(r.estimate[0], 2.05794012307408, TOL);
	CHECK_CLOSE(r.estimate[1], -0.978656740125124, TOL);
	CHECK_CLOSE(r.std_error[1], 0.318083671987483, TOL);
	// No new constant: spanning every slope must reproduce DS1's model F.
	CHECK_CLOSE(r.f_statistic, 132.532474368046, TOL_F);
	check_rel(r.f_p_value, 2.10741740745106e-07, RTOL_P, "joint.f_p", __LINE__);
	CHECK_CLOSE(r.chisq, r.f_statistic * 2.0, TOL_F);
}

// ───────────────── Redundant constraints: rank, not row count ───────────────
static void test_redundant() {
	std::printf("redundant (rank 1 of 2 rows)\n");
	auto r = lin_hyp(mat(2, 3, {0, 1, -1, 0, -1, 1}), kBeta, kCov, kSigma2, kDf);
	CHECK(r.ok && r.estimable);
	// The second row is the negation of the first, so the rank is 1.
	CHECK(r.df_num == 1);
	// And the test is therefore the contrast it duplicates.
	CHECK_CLOSE(r.f_statistic, 81.1925089443785, TOL_F);
}

// ───────────────── The robust covariance carries through ────────────────────
static void test_hc1() {
	std::printf("hc1 (robust covariance)\n");
	auto r = lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, kCovHc1, kSigma2, kDf);
	CHECK(r.ok && r.estimable);
	CHECK_CLOSE(r.std_error[0], 0.144953406513188, TOL); // DS1's HC1 se[1]
	CHECK_CLOSE(r.f_statistic, 201.561980521578, TOL_F);
}

// ───────────────── Aliased coefficients, after #36 ──────────────────────────
static void test_aliased() {
	std::printf("aliased (rank-deficient fit)\n");
	const double nan = std::numeric_limits<double>::quiet_NaN();
	// A rank-deficient fit: column 2 was dropped, so its row and column of the
	// covariance are NaN and so is its coefficient.
	const std::vector<double> beta = {10.0, 2.0, nan};
	const Mat cov = mat(3, 3, {4.0, -0.1, nan, -0.1, 0.02, nan, nan, nan, nan});

	// A hypothesis on the aliased coefficient is not estimable.
	{
		auto r = lin_hyp(mat(1, 3, {0, 0, 1}), beta, cov, 1.0, 9.0);
		CHECK(r.ok);
		CHECK(!r.estimable);
	}
	// A hypothesis that avoids it IS estimable: 0 · NaN must not poison it.
	{
		auto r = lin_hyp(mat(1, 3, {0, 1, 0}), beta, cov, 1.0, 9.0);
		CHECK(r.ok);
		CHECK(r.estimable);
		CHECK(r.df_num == 1);
		CHECK_CLOSE(r.estimate[0], 2.0, TOL);
		CHECK_CLOSE(r.std_error[0], std::sqrt(0.02), TOL);
	}
	// So is a contrast among the surviving coefficients.
	{
		auto r = lin_hyp(mat(1, 3, {1, 1, 0}), beta, cov, 1.0, 9.0);
		CHECK(r.ok && r.estimable);
		CHECK_CLOSE(r.estimate[0], 12.0, TOL);
	}
}

// ───────────────── Review Focus cases ───────────────────────────────────────
static void test_review_focus() {
	std::printf("review_focus (q>k, zero row, zero variance, empty L)\n");

	// 1. More constraints than coefficients: C is singular by construction, so
	//    df_num is the rank and the F is still finite.
	{
		auto r = lin_hyp(mat(4, 3, {0, 1, 0, 0, 0, 1, 0, 1, -1, 0, 1, 1}), kBeta, kCov, kSigma2, kDf);
		CHECK(r.ok && r.estimable);
		CHECK(r.df_num == 2); // the four rows span a 2-dimensional space
		CHECK(std::isfinite(r.f_statistic));
		CHECK(r.estimate.size() == 4); // every row still reports its own estimate
		CHECK_CLOSE(r.f_statistic, 132.532474368046, TOL_F); // the same span as the joint test
	}

	// 2. A zero row inside an otherwise valid L: the rank drops by one and the
	//    surviving constraint is still tested.
	{
		auto r = lin_hyp(mat(2, 3, {0, 1, 0, 0, 0, 0}), kBeta, kCov, kSigma2, kDf);
		CHECK(r.ok && r.estimable);
		CHECK(r.df_num == 1);
		CHECK_CLOSE(r.f_statistic, 249.406353276441, TOL_F); // as the single test
		// The zero row has zero variance, so its t and p are NaN.
		CHECK_CLOSE(r.estimate[1], 0.0, TOL);
		CHECK_CLOSE(r.std_error[1], 0.0, TOL);
		CHECK(std::isnan(r.t_statistic[1]) && std::isnan(r.p_value[1]));
	}

	// 3. A zero variance on the diagonal: std_error is 0, and t and p are NaN
	//    rather than ±inf, following lm_core's rule.
	{
		const Mat cov0 = mat(2, 2, {0.0, 0.0, 0.0, 4.0});
		auto r = lin_hyp(mat(1, 2, {1, 0}), {5.0, 1.0}, cov0, 1.0, 9.0);
		CHECK(r.ok);
		CHECK(!r.estimable); // rank(C) = 0: the hypothesis carries no information
	}
	{
		// The same zero variance alongside a usable constraint.
		const Mat cov0 = mat(2, 2, {0.0, 0.0, 0.0, 4.0});
		auto r = lin_hyp(mat(2, 2, {1, 0, 0, 1}), {5.0, 1.0}, cov0, 1.0, 9.0);
		CHECK(r.ok && r.estimable);
		CHECK(r.df_num == 1);
		CHECK_CLOSE(r.std_error[0], 0.0, TOL);
		CHECK(std::isnan(r.t_statistic[0]));
		CHECK_CLOSE(r.std_error[1], 2.0, TOL);
	}

	// 4. An empty L is misuse, not a NULL result.
	{
		auto r = lin_hyp(Mat(0, 3), kBeta, kCov, kSigma2, kDf);
		CHECK(!r.ok);
		CHECK(!r.error.empty());
	}
}

// ───────────────── Misuse paths ─────────────────────────────────────────────
static void test_errors() {
	std::printf("errors (dimensions, df, sigma2)\n");
	// L's width must match beta.
	CHECK(!lin_hyp(mat(1, 2, {0, 1}), kBeta, kCov, kSigma2, kDf).ok);
	// cov must be k × k.
	CHECK(!lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, mat(2, 2, {1, 0, 0, 1}), kSigma2, kDf).ok);
	// df must be positive.
	CHECK(!lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, kCov, kSigma2, 0.0).ok);
	CHECK(!lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, kCov, kSigma2, -1.0).ok);
	// sigma2 must be finite and non-negative.
	CHECK(!lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, kCov, -1.0, kDf).ok);
	CHECK(!lin_hyp(mat(1, 3, {0, 1, 0}), kBeta, kCov,
	               std::numeric_limits<double>::quiet_NaN(), kDf)
	           .ok);
	// An empty beta.
	CHECK(!lin_hyp(mat(1, 0, {}), {}, Mat(0, 0), kSigma2, kDf).ok);
}

int main() {
	std::printf("== test_lin_hyp ==\n");
	test_single_coefficient();
	test_contrast();
	test_joint();
	test_redundant();
	test_hc1();
	test_aliased();
	test_review_focus();
	test_errors();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL LIN_HYP TESTS PASSED\n");
		return 0;
	}
	return 1;
}
```

Then add the target to `scripts/run-cpp-tests.sh`, immediately after the `test_lm_fit` block:

```bash
build_and_run test_lin_hyp \
  "$ROOT/test/cpp/test_lin_hyp.cpp" \
  "$ROOT/src/lin_hyp.cpp" \
  "$ROOT/src/linalg.cpp"
```

and extend the comment at the top of that file, which lists the suites, with:

```bash
#   lin_hyp general linear hypothesis (#20): test_lin_hyp.cpp + lin_hyp.cpp + linalg.cpp
```

- [ ] **Step 4: Run the tests and watch them fail**

Run: `bash scripts/run-cpp-tests.sh`

Expected: `test_linalg`, `test_lm_fit` and `test_optimize` still pass; `test_lin_hyp` compiles and then fails on behavior, reporting roughly 60 failures beginning with `FAIL (line ...): r.ok` — the stub returns `ok = false`. It must **not** crash: every test reads `r.estimate[0]` only after a size check or an `estimable` check, so a stub result cannot run off the end.

If it does crash instead of reporting, fix the test's bounds guard before going further — a red run that aborts hides which assertions are wrong.

- [ ] **Step 5: Implement the kernel**

Replace the whole of `src/lin_hyp.cpp`:

```cpp
#include "lin_hyp.hpp"

#include "distributions.hpp" // stats_duck::StudentTCDF / FCDF / ChiSquareCDF

#include <cmath>
#include <limits>
#include <string>

namespace statsduck {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

LinHypResult Fail(const std::string &msg) {
	LinHypResult r;
	r.ok = false;
	r.error = msg;
	return r;
}

} // namespace

LinHypResult lin_hyp(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V,
                     double sigma2, double df) {
	const std::size_t q = L.rows;
	const std::size_t k = beta.size();

	if (q == 0) {
		return Fail("lin_hyp: the hypothesis matrix has no rows");
	}
	if (k == 0) {
		return Fail("lin_hyp: beta is empty");
	}
	if (L.cols != k) {
		return Fail("lin_hyp: the hypothesis matrix has " + std::to_string(L.cols) +
		            " columns but beta has " + std::to_string(k) + " elements");
	}
	if (V.rows != k || V.cols != k) {
		return Fail("lin_hyp: the covariance must be " + std::to_string(k) + "×" +
		            std::to_string(k) + ", got " + std::to_string(V.rows) + "×" +
		            std::to_string(V.cols));
	}
	if (!(df > 0.0)) {
		return Fail("lin_hyp: df must be positive");
	}
	if (!std::isfinite(sigma2) || sigma2 < 0.0) {
		return Fail("lin_hyp: sigma2 must be finite and not negative");
	}

	LinHypResult r;
	r.ok = true;
	r.df_den = df;

	// A rank-deficient fit carries NaN in the row and column of each aliased
	// coefficient. A hypothesis that puts no weight on one is still estimable,
	// but IEEE makes 0 · NaN = NaN, so the aliased entries are zeroed in a
	// working copy instead of being multiplied by zero. They carry zero weight
	// by construction, so zeroing changes no surviving result.
	std::vector<bool> aliased(k, false);
	for (std::size_t j = 0; j < k; j++) {
		aliased[j] = !std::isfinite(V(j, j));
	}
	for (std::size_t i = 0; i < q; i++) {
		for (std::size_t j = 0; j < k; j++) {
			if (aliased[j] && L(i, j) != 0.0) {
				return r; // estimable stays false — SAS's "non-est"
			}
		}
	}
	std::vector<double> beta_w = beta;
	linalg::Mat V_w = V;
	for (std::size_t a = 0; a < k; a++) {
		if (!aliased[a]) {
			continue;
		}
		beta_w[a] = 0.0;
		for (std::size_t b = 0; b < k; b++) {
			V_w(a, b) = 0.0;
			V_w(b, a) = 0.0;
		}
	}

	// θ = L·β and C = L·V·Lᵀ: the estimated combinations and their covariance.
	const std::vector<double> theta = linalg::matvec(L, beta_w);
	const linalg::Mat C = linalg::sandwich(L, V_w);
	if (theta.size() != q || C.rows != q || C.cols != q) {
		return Fail("lin_hyp: failed to form L·β or L·V·Lᵀ");
	}
	// Anything non-finite left here came from somewhere other than an aliased
	// column, so the hypothesis still cannot be evaluated.
	for (const double v : theta) {
		if (!std::isfinite(v)) {
			return r;
		}
	}
	for (const double v : C.data) {
		if (!std::isfinite(v)) {
			return r;
		}
	}

	// pinv returns the pseudo-inverse together with the numerical rank, so one
	// call covers both the full-rank and the redundant case and df_num is the
	// rank rather than the row count. statsmodels' wald_test agrees: given a
	// redundant hypothesis it reports the rank too.
	const linalg::Pinv pi = linalg::pinv(C);
	const std::size_t rank = pi.rank > 0 ? static_cast<std::size_t>(pi.rank) : 0;
	if (rank == 0) {
		return r; // a hypothesis with no information, such as a zero L
	}
	r.estimable = true;
	r.df_num = rank;

	// The Wald statistic W = θᵀ C⁻ θ.
	const std::vector<double> c_inv_theta = linalg::matvec(pi.value, theta);
	double w = 0.0;
	for (std::size_t i = 0; i < q; i++) {
		w += theta[i] * c_inv_theta[i];
	}
	const double rank_d = static_cast<double>(rank);
	r.chisq = w;
	r.chisq_p_value = 1.0 - stats_duck::ChiSquareCDF(w, rank_d);
	r.f_statistic = w / rank_d;
	r.f_p_value = 1.0 - stats_duck::FCDF(r.f_statistic, rank_d, df);
	// Only a sum of squares when V is the classical σ²(XᵀX)⁻¹; see the README.
	r.ss = w * sigma2;

	r.estimate = theta;
	r.std_error.assign(q, 0.0);
	r.t_statistic.assign(q, kNaN);
	r.p_value.assign(q, kNaN);
	for (std::size_t i = 0; i < q; i++) {
		const double var = C(i, i);
		const double se = var > 0.0 ? std::sqrt(var) : 0.0;
		r.std_error[i] = se;
		if (se > 0.0) {
			const double t = theta[i] / se;
			r.t_statistic[i] = t;
			r.p_value[i] = 2.0 * (1.0 - stats_duck::StudentTCDF(std::fabs(t), df));
		}
	}
	return r;
}

} // namespace statsduck
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `bash scripts/run-cpp-tests.sh`

Expected: all four suites pass, ending in `All C++ unit tests passed.`, with `test_lin_hyp` reporting `0 failures`.

If the `q > k` case reports `df_num == 3` rather than 2, the four rows of that `L` were not as expected: rows 1 and 2 span the two slopes, and rows 3 and 4 (`x1 − x2` and `x1 + x2`) lie in that same span, so the rank is 2. Print `pi.rank` and check `L` before changing the tolerance.

- [ ] **Step 7: Commit**

```bash
git add src/include/lin_hyp.hpp src/lin_hyp.cpp test/cpp/test_lin_hyp.cpp scripts/run-cpp-tests.sh
git commit -m "Add the lin_hyp kernel: general linear hypothesis (#20)

Tests L*beta = 0 for q constraints at once, returning the per-row
estimates with their standard errors, t and p alongside the joint Wald
test as an F and a chi-square. DuckDB-free and independent of lm_core,
so the same function serves lm_fit now and glm_fit later.

The numerator df is the rank of L*V*L', not the row count, so redundant
constraints reduce the way statsmodels' wald_test reduces them. One pinv
call yields both the pseudo-inverse and that rank.

A hypothesis that puts no weight on a rank-deficient fit's aliased
coefficients stays estimable: the aliased rows and columns are zeroed in
a working copy first, because IEEE makes 0 * NaN = NaN and a naive
product would report every such hypothesis untestable.

Goldens come from statsmodels. Two cases carry no new constants: a
single-coefficient test reproduces DS1's beta[1] and se[1], and a test
spanning both slopes reproduces DS1's model F.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The SQL function

**Files:**
- Create: `src/include/lin_hyp_function.hpp`
- Create: `src/lin_hyp_function.cpp`
- Create: `test/sql/lin_hyp.test`
- Modify: `CMakeLists.txt` (add two sources to `EXTENSION_SOURCES`, after `src/linalg.cpp`)
- Modify: `src/stats_duck_extension.cpp` (include the header, call `RegisterLinHyp`)

**Interfaces:**
- Consumes: `statsduck::lin_hyp` and `statsduck::LinHypResult` from Task 2; `statsduck::RegisterDocumented` from `src/include/register_documented.hpp`.
- Produces: `duckdb::RegisterLinHyp(ExpressionState &)` — precisely `void RegisterLinHyp(ExtensionLoader &loader)` — and the SQL function `lin_hyp(LIST(LIST(DOUBLE)), LIST(DOUBLE), LIST(DOUBLE), DOUBLE, DOUBLE) -> STRUCT`.

- [ ] **Step 1: Write the header**

Create `src/include/lin_hyp_function.hpp`:

```cpp
#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers lin_hyp(L LIST<LIST<DOUBLE>>, beta LIST<DOUBLE>,
//! cov LIST<DOUBLE>, sigma2 DOUBLE, df DOUBLE) → STRUCT.
//!
//! Tests the general linear hypothesis L·β = 0 against an already-computed
//! fit. `L` holds one row per constraint, each as wide as `beta`; `cov` is the
//! flat row-major k×k covariance of `beta`, exactly as lm_fit returns it.
//! Returns the per-row estimates with their standard errors, t and p, plus the
//! joint test as a sum of squares, an F and a chi-square. A hypothesis that
//! cannot be evaluated — one touching a coefficient the fit could not estimate,
//! or one carrying no information — yields NULL. A dimension mismatch raises.
void RegisterLinHyp(ExtensionLoader &loader);

} // namespace duckdb
```

- [ ] **Step 2: Write the failing SQL test**

Create `test/sql/lin_hyp.test`. The goldens are the same statsmodels values Task 2 uses, printed to four decimals.

```
# name: test/sql/lin_hyp.test
# description: lin_hyp — the general linear hypothesis L·β = 0 against a fit
# group: [stats_duck]

require stats_duck

# DS1, the dataset test/cpp/test_lm_fit.cpp and test/sql/lm_fit_agg.test use.
statement ok
CREATE TABLE ds1 AS SELECT * FROM (VALUES
    (9.0, 1.0, 5.0), (8.0, 2.0, 3.0), (7.0, 3.0, 8.0), (14.0, 4.0, 2.0),
    (12.0, 5.0, 7.0), (20.0, 6.0, 4.0), (15.0, 7.0, 9.0), (28.0, 8.0, 1.0),
    (22.0, 9.0, 6.0), (31.0, 10.0, 3.0), (29.0, 11.0, 8.0), (55.0, 25.0, 4.0)
) AS t(y, x1, x2);

# The documented call: a Type III test on x1, straight off an lm_fit result.
# The estimate and standard error are x1's own, which is what a one-row
# hypothesis on a single coefficient means.
query TTTTTT
SELECT (h).df_num, printf('%.1f', (h).df_den),
       printf('%.4f', (h).estimates[1].estimate),
       printf('%.4f', (h).estimates[1].std_error),
       printf('%.4f', (h).estimates[1].t_statistic),
       printf('%.4f', (h).f_statistic)
FROM (SELECT lin_hyp([[0, 1, 0]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM ds1))
----
1	9.0	2.0579	0.1303	15.7926	249.4064

# A contrast, SAS's ESTIMATE: x1 − x2 with its own standard error.
query TTT
SELECT printf('%.4f', (h).estimates[1].estimate),
       printf('%.4f', (h).estimates[1].std_error),
       printf('%.4f', (h).f_statistic)
FROM (SELECT lin_hyp([[0, 1, -1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM ds1))
----
3.0366	0.3370	81.1925

# A joint test of both slopes reproduces the model's own F, and every row still
# reports its own estimate.
query TTTTT
SELECT (h).df_num, len((h).estimates),
       printf('%.4f', (h).f_statistic),
       printf('%.4f', m.f_statistic),
       printf('%.4f', (h).chisq)
FROM (SELECT lin_hyp([[0, 1, 0], [0, 0, 1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h, m
      FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM ds1))
----
2	2	132.5325	132.5325	265.0649

# Redundant constraints reduce to their rank, so df_num is 1 and the test is
# the contrast the two rows duplicate.
query TT
SELECT (h).df_num, printf('%.4f', (h).f_statistic)
FROM (SELECT lin_hyp([[0, 1, -1], [0, -1, 1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM ds1))
----
1	81.1925

# A robust covariance carries through: the HC1 standard error of x1.
query TT
SELECT printf('%.4f', (h).estimates[1].std_error), printf('%.4f', (h).f_statistic)
FROM (SELECT lin_hyp([[0, 1, 0]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2], 'HC1') AS m FROM ds1))
----
0.1450	201.5620

# ── Rank-deficient fits (#36) ────────────────────────────────────────────────
# x3 = x1 + x2 is dropped, so its coefficient and its row and column of cov are
# NULL. A hypothesis ON the aliased term is not estimable, so the whole result
# is NULL.
query T
SELECT lin_hyp([[0, 0, 0, 1]],
               list_transform(m.coefficients, c -> c.estimate),
               m.cov, m.sigma * m.sigma, m.df_residual)
FROM (SELECT lm_fit(y, [x1, x2, x1 + x2]) AS m FROM ds1)
----
NULL

# A hypothesis that avoids the aliased term is still estimable, and gives the
# same answer as the full-rank fit — 0 · NULL must not poison the product.
query TT
SELECT (h).df_num, printf('%.4f', (h).estimates[1].estimate)
FROM (SELECT lin_hyp([[0, 1, 0, 0]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2, x1 + x2]) AS m FROM ds1))
----
1	2.0579

# ── Review Focus: a NULL element inside a list argument ──────────────────────
# NULL is read as NaN, so the hypothesis is not estimable and the result is
# NULL rather than a garbage number.
query TTT
SELECT lin_hyp([[0, 1, 0]], [1.0, NULL, 3.0], [1.0,0,0, 0,1.0,0, 0,0,1.0], 1.0, 9.0) IS NULL,
       lin_hyp([[0, 1, 0]], [1.0, 2.0, 3.0], [1.0,0,0, 0,NULL,0, 0,0,1.0], 1.0, 9.0) IS NULL,
       lin_hyp([[0, NULL, 0]], [1.0, 2.0, 3.0], [1.0,0,0, 0,1.0,0, 0,0,1.0], 1.0, 9.0) IS NULL
----
true	true	true

# A top-level NULL argument gives a NULL result through default null handling.
query T
SELECT lin_hyp(NULL, [1.0, 2.0], [1.0,0,0,1.0], 1.0, 9.0) IS NULL
----
true

# ── Misuse raises ───────────────────────────────────────────────────────────
# An L row that is not as wide as beta.
statement error
SELECT lin_hyp([[0, 1]], [1.0, 2.0, 3.0], [1.0,0,0, 0,1.0,0, 0,0,1.0], 1.0, 9.0)
----
columns but beta has

# A cov whose length is not k².
statement error
SELECT lin_hyp([[0, 1, 0]], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 1.0], 1.0, 9.0)
----
covariance

# A ragged L.
statement error
SELECT lin_hyp([[0, 1, 0], [0, 1]], [1.0, 2.0, 3.0], [1.0,0,0, 0,1.0,0, 0,0,1.0], 1.0, 9.0)
----
same width

# An empty L.
statement error
SELECT lin_hyp([], [1.0, 2.0], [1.0,0,0,1.0], 1.0, 9.0)
----
no rows

# A non-positive df.
statement error
SELECT lin_hyp([[0, 1]], [1.0, 2.0], [1.0,0,0,1.0], 1.0, 0.0)
----
df must be positive
```

- [ ] **Step 3: Run it and watch it fail**

Run:

```bash
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/test/Release/unittest.exe" "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/lin_hyp.test"
```

Expected: failure on the first query with `Catalog Error: Scalar Function with name lin_hyp does not exist!`, because nothing is registered yet.

- [ ] **Step 4: Write the wrapper**

Create `src/lin_hyp_function.cpp`:

```cpp
#include "lin_hyp_function.hpp"
#include "register_documented.hpp"

#include "lin_hyp.hpp" // statsduck::lin_hyp — the DuckDB-free kernel

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace duckdb {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// One row of the per-constraint table, mirroring lm_fit's coefficients element.
static LogicalType LinHypRowType() {
	child_list_t<LogicalType> f;
	f.emplace_back("estimate", LogicalType::DOUBLE);
	f.emplace_back("std_error", LogicalType::DOUBLE);
	f.emplace_back("t_statistic", LogicalType::DOUBLE);
	f.emplace_back("p_value", LogicalType::DOUBLE);
	return LogicalType::STRUCT(std::move(f));
}

// Field order MUST match the writes in LinHypExec (indices below).
static LogicalType LinHypResultType() {
	child_list_t<LogicalType> c;
	c.emplace_back("estimates", LogicalType::LIST(LinHypRowType())); // 0
	c.emplace_back("ss", LogicalType::DOUBLE);                       // 1
	c.emplace_back("df_num", LogicalType::BIGINT);                   // 2
	c.emplace_back("df_den", LogicalType::DOUBLE);                   // 3
	c.emplace_back("f_statistic", LogicalType::DOUBLE);              // 4
	c.emplace_back("f_p_value", LogicalType::DOUBLE);                // 5
	c.emplace_back("chisq", LogicalType::DOUBLE);                    // 6
	c.emplace_back("chisq_p_value", LogicalType::DOUBLE);            // 7
	return LogicalType::STRUCT(std::move(c));
}

// Write a DOUBLE child, mapping NaN to SQL NULL (the SetD idiom from lm_fit).
static inline void SetD(Vector &v, idx_t row, double x) {
	if (std::isnan(x)) {
		FlatVector::SetNull(v, row, true);
	} else {
		FlatVector::GetData<double>(v)[row] = x;
	}
}

// Read a run of a flat DOUBLE list into `out`, reading NULL as NaN so it
// reaches the kernel's own non-finite check instead of needing separate
// validity handling.
static void ReadDoubles(const UnifiedVectorFormat &fmt, const list_entry_t &entry,
                        std::vector<double> &out) {
	const auto data = UnifiedVectorFormat::GetData<double>(fmt);
	out.assign(entry.length, 0.0);
	for (idx_t j = 0; j < entry.length; j++) {
		const auto idx = fmt.sel->get_index(entry.offset + j);
		out[j] = fmt.validity.RowIsValid(idx) ? data[idx] : kNaN;
	}
}

static void LinHypExec(DataChunk &args, ExpressionState &, Vector &result) {
	const idx_t count = args.size();

	// L is LIST(LIST(DOUBLE)), so it unpacks in three levels: the outer list
	// gives one entry per row of the chunk, its child is a list per constraint,
	// and that child holds the doubles.
	auto &l_vec = args.data[0];
	UnifiedVectorFormat l_fmt;
	l_vec.ToUnifiedFormat(count, l_fmt);
	const auto l_entries = UnifiedVectorFormat::GetData<list_entry_t>(l_fmt);

	auto &l_rows_vec = ListVector::GetEntry(l_vec);
	UnifiedVectorFormat l_rows_fmt;
	l_rows_vec.ToUnifiedFormat(ListVector::GetListSize(l_vec), l_rows_fmt);
	const auto l_row_entries = UnifiedVectorFormat::GetData<list_entry_t>(l_rows_fmt);

	auto &l_vals_vec = ListVector::GetEntry(l_rows_vec);
	UnifiedVectorFormat l_vals_fmt;
	l_vals_vec.ToUnifiedFormat(ListVector::GetListSize(l_rows_vec), l_vals_fmt);

	auto &beta_vec = args.data[1];
	UnifiedVectorFormat beta_fmt;
	beta_vec.ToUnifiedFormat(count, beta_fmt);
	const auto beta_entries = UnifiedVectorFormat::GetData<list_entry_t>(beta_fmt);
	UnifiedVectorFormat beta_child_fmt;
	ListVector::GetEntry(beta_vec).ToUnifiedFormat(ListVector::GetListSize(beta_vec),
	                                               beta_child_fmt);

	auto &cov_vec = args.data[2];
	UnifiedVectorFormat cov_fmt;
	cov_vec.ToUnifiedFormat(count, cov_fmt);
	const auto cov_entries = UnifiedVectorFormat::GetData<list_entry_t>(cov_fmt);
	UnifiedVectorFormat cov_child_fmt;
	ListVector::GetEntry(cov_vec).ToUnifiedFormat(ListVector::GetListSize(cov_vec), cov_child_fmt);

	UnifiedVectorFormat s2_fmt, df_fmt;
	args.data[3].ToUnifiedFormat(count, s2_fmt);
	args.data[4].ToUnifiedFormat(count, df_fmt);
	const auto s2_data = UnifiedVectorFormat::GetData<double>(s2_fmt);
	const auto df_data = UnifiedVectorFormat::GetData<double>(df_fmt);

	// Pass 1 — evaluate every row and tally the output list length.
	std::vector<statsduck::LinHypResult> fits(count);
	std::vector<bool> emit(count, false);
	idx_t total_rows = 0;
	std::vector<double> beta_buf, cov_buf, l_buf;

	for (idx_t i = 0; i < count; i++) {
		const auto l_idx = l_fmt.sel->get_index(i);
		const auto beta_idx = beta_fmt.sel->get_index(i);
		const auto cov_idx = cov_fmt.sel->get_index(i);
		const auto s2_idx = s2_fmt.sel->get_index(i);
		const auto df_idx = df_fmt.sel->get_index(i);
		if (!l_fmt.validity.RowIsValid(l_idx) || !beta_fmt.validity.RowIsValid(beta_idx) ||
		    !cov_fmt.validity.RowIsValid(cov_idx) || !s2_fmt.validity.RowIsValid(s2_idx) ||
		    !df_fmt.validity.RowIsValid(df_idx)) {
			continue; // a NULL argument gives a NULL result
		}

		ReadDoubles(beta_child_fmt, beta_entries[beta_idx], beta_buf);
		ReadDoubles(cov_child_fmt, cov_entries[cov_idx], cov_buf);
		const std::size_t k = beta_buf.size();
		if (k == 0) {
			throw InvalidInputException("lin_hyp: beta is empty");
		}
		if (cov_buf.size() != k * k) {
			throw InvalidInputException(
			    "lin_hyp: the covariance must hold k² = %llu elements for a beta of length %llu, got %llu",
			    static_cast<unsigned long long>(k * k), static_cast<unsigned long long>(k),
			    static_cast<unsigned long long>(cov_buf.size()));
		}

		const auto l_entry = l_entries[l_idx];
		const std::size_t q = l_entry.length;
		if (q == 0) {
			throw InvalidInputException("lin_hyp: the hypothesis matrix has no rows");
		}
		statsduck::linalg::Mat L(q, k);
		for (idx_t a = 0; a < q; a++) {
			const auto row_idx = l_rows_fmt.sel->get_index(l_entry.offset + a);
			if (!l_rows_fmt.validity.RowIsValid(row_idx)) {
				throw InvalidInputException("lin_hyp: the hypothesis matrix has a NULL row");
			}
			const auto row_entry = l_row_entries[row_idx];
			if (row_entry.length != k) {
				throw InvalidInputException(
				    "lin_hyp: every hypothesis row must have the same width as beta (%llu); row %llu "
				    "has %llu",
				    static_cast<unsigned long long>(k), static_cast<unsigned long long>(a + 1),
				    static_cast<unsigned long long>(row_entry.length));
			}
			ReadDoubles(l_vals_fmt, row_entry, l_buf);
			for (std::size_t b = 0; b < k; b++) {
				L(a, b) = l_buf[b];
			}
		}

		statsduck::linalg::Mat V(k, k);
		V.data = cov_buf; // row-major k×k, exactly Mat's layout

		auto r = statsduck::lin_hyp(L, beta_buf, V, s2_data[s2_idx], df_data[df_idx]);
		if (!r.ok) {
			throw InvalidInputException(r.error);
		}
		if (!r.estimable) {
			continue; // not estimable → a NULL result for this row
		}
		total_rows += r.estimate.size();
		fits[i] = std::move(r);
		emit[i] = true;
	}

	auto &children = StructVector::GetEntries(result);

	// estimates: LIST<STRUCT>. Reserve first, then fetch the child pointers —
	// Reserve can reallocate, and fetching first is the classic bug. See
	// notes/engineering/2026-06-lm-fit-robust-se-aggregate.md §2.
	Vector &rows_list = *children[0];
	auto row_entries_out = FlatVector::GetData<list_entry_t>(rows_list);
	ListVector::Reserve(rows_list, total_rows);
	ListVector::SetListSize(rows_list, total_rows);
	Vector &rows_child = ListVector::GetEntry(rows_list);
	auto &rf = StructVector::GetEntries(rows_child);

	auto df_num_data = FlatVector::GetData<int64_t>(*children[2]);

	idx_t out = 0;
	for (idx_t i = 0; i < count; i++) {
		if (!emit[i]) {
			FlatVector::SetNull(result, i, true);
			row_entries_out[i] = list_entry_t(out, 0);
			continue;
		}
		const auto &r = fits[i];
		const idx_t q = r.estimate.size();
		row_entries_out[i] = list_entry_t(out, q);
		for (idx_t a = 0; a < q; a++) {
			const idx_t pos = out + a;
			SetD(*rf[0], pos, r.estimate[a]);
			SetD(*rf[1], pos, r.std_error[a]);
			SetD(*rf[2], pos, r.t_statistic[a]);
			SetD(*rf[3], pos, r.p_value[a]);
		}
		out += q;

		SetD(*children[1], i, r.ss);
		df_num_data[i] = static_cast<int64_t>(r.df_num);
		SetD(*children[3], i, r.df_den);
		SetD(*children[4], i, r.f_statistic);
		SetD(*children[5], i, r.f_p_value);
		SetD(*children[6], i, r.chisq);
		SetD(*children[7], i, r.chisq_p_value);
	}
}

} // namespace

void RegisterLinHyp(ExtensionLoader &loader) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	ScalarFunction fn("lin_hyp",
	                  {LogicalType::LIST(list_double), list_double, list_double,
	                   LogicalType::DOUBLE, LogicalType::DOUBLE},
	                  LinHypResultType(), LinHypExec);
	statsduck::RegisterDocumented(
	    loader, std::move(fn),
	    "General linear hypothesis L·β = 0 against a fit: per-constraint estimates with "
	    "standard errors, plus the joint sum of squares, F and chi-square.",
	    {"L", "beta", "cov", "sigma2", "df"},
	    "SELECT lin_hyp([[0, 1, -1]], [10.0, 2.0, -1.0], "
	    "[4.3,-0.14,-0.52, -0.14,0.017,0.0023, -0.52,0.0023,0.101], 7.46, 9.0)");
}

} // namespace duckdb
```

- [ ] **Step 5: Wire it into the build and the extension**

In `CMakeLists.txt`, inside `set(EXTENSION_SOURCES`, immediately after the `src/linalg.cpp` line, add:

```cmake
    src/lin_hyp.cpp
    src/lin_hyp_function.cpp
```

In `src/stats_duck_extension.cpp`, add the include alongside the other function headers:

```cpp
#include "lin_hyp_function.hpp"
```

and, in `LoadInternal`, after the OLS registrations (the `RegisterLmFit(loader);` line), add:

```cpp
	// General linear hypothesis — contrasts, ESTIMATE, Type III sums of squares
	RegisterLinHyp(loader);
```

- [ ] **Step 6: Build**

Run:

```bash
cmake --build "C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release" --config Release --target unittest
```

Expected: exit 0. If MSVC reports `InvalidInputException` as undeclared, add `#include "duckdb/common/exception.hpp"` to `src/lin_hyp_function.cpp`.

- [ ] **Step 7: Run the SQL test and watch it pass**

Run:

```bash
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/test/Release/unittest.exe" "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/lin_hyp.test"
```

Expected: `All tests passed`.

Two failures are likely here and both are the test's fault rather than the code's. If the `statement error` cases report `Query unexpectedly succeeded`, the expected substring does not appear in the real message — run the query by hand, read the message, and correct the substring. If `df_den` prints as `9` rather than `9.0`, change the `printf` format rather than the field's type; `df_den` is a DOUBLE on purpose.

- [ ] **Step 8: Run the whole suite**

Run:

```bash
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/test/Release/unittest.exe" "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"
```

Expected: every file passes, with the assertion count up by the new file's worth. `test/sql/function_docs.test` must still pass: it fails if a registered function has no description, which `RegisterDocumented` supplies in Step 4.

- [ ] **Step 9: Commit**

```bash
git add src/include/lin_hyp_function.hpp src/lin_hyp_function.cpp test/sql/lin_hyp.test CMakeLists.txt src/stats_duck_extension.cpp
git commit -m "Expose lin_hyp as a SQL scalar function (#20)

lin_hyp(L, beta, cov, sigma2, df) returns one STRUCT per row: the
per-constraint estimates with their standard errors, t and p, and the
joint test as a sum of squares, an F and a chi-square.

This is the extension's first STRUCT-returning scalar function, and L is
its first nested-list argument, so the wrapper unpacks three levels of
list and builds the result's LIST<STRUCT> child with the reserve-then-
fetch idiom the lm_fit engineering note describes.

A hypothesis that cannot be evaluated gives NULL, including one that
touches a rank-deficient fit's aliased coefficient; a dimension mismatch
raises, because list lengths are only knowable per row at execution time.
A NULL inside any list argument is read as NaN and so yields NULL rather
than a garbage number.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Documentation and close-out

**Files:**
- Modify: `README.md` (a new subsection after the `lm_fit` one)
- Modify: `docs/kernel_api.md` (a `lin_hyp.hpp` section after the `optimize.hpp` one)
- Modify: `CHANGELOG.md` (under `## [Unreleased]` → `### Added`)

**Interfaces:**
- Consumes: the shipped function from Task 3.
- Produces: no code. The public documentation of the function.

- [ ] **Step 1: Add the README subsection**

In `README.md`, immediately before the `### Multiple-testing correction (scalar)` heading, insert:

````markdown
### General linear hypothesis (`lin_hyp`)

`lin_hyp` tests **L·β = 0** against a fit you already have. One row of `L` per
constraint, each as wide as `beta`; several rows means a joint test. It is
stateless — it takes the coefficients and their covariance, not the data — so
it works with any fit, and it is what SAS's CONTRAST and ESTIMATE compute.

```sql
-- Is x1's coefficient zero? (a Type III test of one term)
SELECT (h).df_num, (h).f_statistic, (h).f_p_value
FROM (SELECT lin_hyp([[0, 1, 0]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.sigma * m.sigma, m.df_residual) AS h
      FROM (SELECT lm_fit(y, [x1, x2]) AS m FROM ds1));
--   df_num  f_statistic  f_p_value
--   1       249.4064     7.21e-08

-- A contrast: is x1 − x2 zero? The estimate comes back with its standard error.
SELECT (h).estimates[1].estimate, (h).estimates[1].std_error
FROM (SELECT lin_hyp([[0, 1, -1]], …) AS h FROM …);
--   3.0366  0.3370
```

| Field | Type | Notes |
| ----- | ---- | ----- |
| `estimates` | `LIST<STRUCT>` | one element per row of `L`: `estimate`, `std_error`, `t_statistic`, `p_value` |
| `ss` | `DOUBLE` | the hypothesis sum of squares, `chisq · sigma2` |
| `df_num` | `BIGINT` | the rank of `L·cov·Lᵀ`, so redundant constraints count once |
| `df_den` | `DOUBLE` | the `df` argument, echoed |
| `f_statistic`, `f_p_value` | `DOUBLE` | referenced to F(`df_num`, `df_den`) |
| `chisq`, `chisq_p_value` | `DOUBLE` | the Wald statistic, referenced to χ²(`df_num`) |

Pass `cov` exactly as `lm_fit` returns it: flat, row-major, k² elements. `df`
is a DOUBLE so a fractional denominator df fits; `lm_fit`'s `df_residual` casts
in on its own.

**`ss` is a sum of squares only when `cov` is the classical σ²(XᵀX)⁻¹.** With an
HC or CR covariance the F and the chi-square are a Wald test — the same thing
statsmodels' `wald_test` reports — and `ss` is not a partition of variance.

The whole result is **NULL** when the hypothesis cannot be evaluated: when it
puts weight on a coefficient the fit could not estimate (a rank-deficient fit's
aliased terms, SAS's "non-estimable"), or when it carries no information at all.
A hypothesis that merely avoids those aliased terms is fine. A dimension
mismatch — an `L` row that is not as wide as `beta`, or a `cov` that is not k²
long — raises instead.

Type I and II sums of squares are not computable this way: they need nested
refits and the term-to-column mapping. Only Type III reduces to a hypothesis on
the full model, which is what this function tests.
````

- [ ] **Step 2: Add the `kernel_api.md` section**

In `docs/kernel_api.md`, immediately before the `## Testing & validation` heading, insert:

````markdown
## `lin_hyp.hpp` — general linear hypothesis

Tests `L·β = 0` for q constraints at once, given a fit's coefficients and their
covariance. Header-only API over `linalg`; it does **not** depend on
`lm_core.hpp`, so the same function serves any fit.

```cpp
LinHypResult lin_hyp(const linalg::Mat &L,          // q × k, one row per constraint
                     const std::vector<double> &beta,
                     const linalg::Mat &V,          // k × k covariance of beta
                     double sigma2, double df);
```

| Symbol | Kind | Notes |
| --- | --- | --- |
| `LinHypResult` | struct | `ok`/`error`; `estimable`; per-row `estimate`/`std_error`/`t_statistic`/`p_value`; joint `ss`, `df_num`, `df_den`, `f_statistic`, `f_p_value`, `chisq`, `chisq_p_value` |

Conventions worth knowing:

- **Three outcomes, not two.** `ok = false` is misuse (a dimension mismatch, a
  non-positive `df`). `ok = true` with `estimable = false` means the hypothesis
  cannot be evaluated, and callers should report NULL. Only then is the rest of
  the struct filled in.
- **`df_num` is the rank of `L·V·Lᵀ`**, not `L`'s row count, so redundant
  constraints count once. One `pinv` call yields both the pseudo-inverse and
  that rank.
- **Aliased coefficients do not spread.** A rank-deficient fit carries NaN in
  the rows and columns of its dropped coefficients. A hypothesis putting
  non-zero weight there is not estimable; one that avoids them is evaluated
  normally, with those rows and columns zeroed in a working copy first, because
  IEEE makes `0 · NaN = NaN`.
- **`ss` = `chisq` · `sigma2`** is a sum of squares only under the classical
  covariance; with a robust `V` the F and chi-square are a Wald test.
````

- [ ] **Step 3: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, as the first bullet:

```markdown
- **`lin_hyp(L, beta, cov, sigma2, df)` — the general linear hypothesis
  L·β = 0** (#20). One row of `L` per constraint, several rows for a joint
  test. Returns the per-constraint estimates with their standard errors, t and
  p, plus the joint test as a sum of squares, an F and a chi-square. It is
  stateless — coefficients and covariance in, no data — so it works with any
  fit, and it computes what SAS's CONTRAST and ESTIMATE compute, as well as
  Type III sums of squares when the caller builds `L` to select a term's
  columns. `df_num` is the rank of `L·cov·Lᵀ`, so redundant constraints count
  once, matching statsmodels' `wald_test`. A hypothesis that touches a
  rank-deficient fit's aliased coefficient is reported NULL, SAS's
  "non-estimable", while one that avoids it is evaluated normally. Validated
  against statsmodels' `t_test` and `wald_test`; the numerics live in the
  DuckDB-free `src/include/lin_hyp.hpp`. Type I and II sums of squares need
  nested refits and are not part of this function.
```

- [ ] **Step 4: Verify the docs describe the shipped function**

Run:

```bash
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/test/Release/unittest.exe" "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/function_docs.test"
```

Expected: `All tests passed` — the registration carries a description, parameter names and a runnable example.

Then check that the README's two numbers are the ones the function prints, by running the first example from `test/sql/lin_hyp.test`: `df_num` 1, `f_statistic` 249.4064. If they differ, the README is wrong, not the code.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/kernel_api.md CHANGELOG.md
git commit -m "docs: document lin_hyp (#20)

README gains a subsection with the worked contrast and Type III calls,
the field table, and the two caveats that matter: ss is a sum of squares
only under the classical covariance, and a hypothesis touching an
aliased coefficient is NULL. kernel_api.md documents the DuckDB-free
API, including that df_num is a rank and that aliased coefficients are
zeroed rather than multiplied by zero.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Open the pull request**

```bash
git push -u origin feat/20-lin-hyp
```

Open a PR against `v0.9` whose body states: what the function computes; that the per-row table and the joint test are both validated against statsmodels; that two of the five fixtures carry no new constants because they must reproduce DS1's own `beta[1]`, `se[1]` and model F; the three deliberate behaviors (rank as `df_num`, NULL for a non-estimable hypothesis, raise on a dimension mismatch); and the non-goals from the spec, naming Type I/II sums of squares as a follow-up issue to file.

- [ ] **Step 7: File the Type I/II follow-up**

```bash
gh issue create --repo KoliStat/the-stats-duck --label kernel --label feat \
  --title "Type I and II sums of squares (the data-seeing half of #20)" \
  --body "Split out of #20, which shipped the stateless lin_hyp.

Type III for a term is the Wald test of that term's coefficients in the full model, so lin_hyp computes it from beta and cov alone. Type I and II cannot work that way:

- **Type I** is sequential: SS(term | the terms before it), which needs nested fits in the model's term order.
- **Type II** conditions each term on the terms that do not contain it, which needs the term-to-column mapping and refits.

So this needs a function that re-sees the data — an aggregate or a table function — plus a way to declare which design columns belong to which term. lm_fit's coefficients are positional (x1, x2, …) because an aggregate never sees column names, so the term grouping has to come from the caller or from the lm table-function path.

Depends on nothing further; lin_hyp (#20) is the inner evaluator for the Type III column of the resulting table."
```

---

## Self-Review

**Spec coverage.** Every section of `notes/plans/2026-10-14-lin-hyp-design.md` maps to a task: the SQL surface and return STRUCT to Task 3 Step 4; the numerics, including `pinv`-with-rank and the aliased-coefficient handling, to Task 2 Step 5; the three failure outcomes to Task 2 Steps 3 and 5 and Task 3 Steps 2 and 4; the file structure to Tasks 2 and 3; the validation plan, all five fixtures and the generator self-checks, to Tasks 1 and 2; the documentation section to Task 4; the non-goals to Task 4 Step 7, which files the Type I/II follow-up.

**Placeholders.** None: every step carries the code or the exact command and its expected output.

**Type consistency.** `statsduck::lin_hyp(const linalg::Mat &, const std::vector<double> &, const linalg::Mat &, double, double)` is declared in Task 2 Step 1, stubbed in Step 2 with the same signature, implemented in Step 5, and called in Task 3 Step 4. `LinHypResult`'s field names used in Task 3 (`estimate`, `std_error`, `t_statistic`, `p_value`, `ss`, `df_num`, `df_den`, `f_statistic`, `f_p_value`, `chisq`, `chisq_p_value`, `ok`, `error`, `estimable`) all appear in the Task 2 header. `RegisterLinHyp(ExtensionLoader &)` is declared in Task 3 Step 1 and called in Step 5. The SQL field names in the tests and README match `LinHypResultType`'s order.

**Review Focus coverage.** q > k, a zero row, a zero variance and an empty `L` are tested in Task 2 Step 3's `test_review_focus` and `test_errors`; a NULL element inside a list argument is tested in Task 3 Step 2, which is the only layer that can get it wrong.
