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

// Index a result vector safely. A stub or a not-yet-estimable result returns
// empty vectors, and the first (red) run of a new test must report that
// rather than abort the binary.
static double at(const std::vector<double> &v, std::size_t i) {
	return i < v.size() ? v[i] : std::numeric_limits<double>::quiet_NaN();
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
	CHECK_CLOSE(at(r.estimate, 0), kBeta[1], TOL);
	CHECK_CLOSE(at(r.std_error, 0), std::sqrt(kCov(1, 1)), TOL);

	CHECK_CLOSE(at(r.estimate, 0), 2.05794012307408, TOL);
	CHECK_CLOSE(at(r.std_error, 0), 0.130310370148038, TOL);
	CHECK_CLOSE(at(r.t_statistic, 0), 15.792604385485, TOL);
	check_rel(at(r.p_value, 0), 7.20757277116357e-08, RTOL_P, "single.p", __LINE__);
	CHECK_CLOSE(r.f_statistic, 249.406353276441, TOL_F);
	check_rel(r.f_p_value, 7.20757277116357e-08, RTOL_P, "single.f_p", __LINE__);

	// For q = 1 the Wald statistic is the squared t, and F equals it.
	CHECK_CLOSE(r.chisq, at(r.t_statistic, 0) * at(r.t_statistic, 0), TOL_F);
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
	CHECK_CLOSE(at(r.estimate, 0), kBeta[1] - kBeta[2], TOL);
	CHECK_CLOSE(at(r.estimate, 0), 3.03659686319921, TOL);
	CHECK_CLOSE(at(r.std_error, 0), 0.336999423673648, TOL);
	CHECK_CLOSE(at(r.t_statistic, 0), 9.01068859435163, TOL);
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
	CHECK_CLOSE(at(r.estimate, 0), 2.05794012307408, TOL);
	CHECK_CLOSE(at(r.estimate, 1), -0.978656740125124, TOL);
	CHECK_CLOSE(at(r.std_error, 1), 0.318083671987483, TOL);
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
	CHECK_CLOSE(at(r.std_error, 0), 0.144953406513188, TOL); // DS1's HC1 se[1]
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
		CHECK_CLOSE(at(r.estimate, 0), 2.0, TOL);
		CHECK_CLOSE(at(r.std_error, 0), std::sqrt(0.02), TOL);
	}
	// So is a contrast among the surviving coefficients.
	{
		auto r = lin_hyp(mat(1, 3, {1, 1, 0}), beta, cov, 1.0, 9.0);
		CHECK(r.ok && r.estimable);
		CHECK_CLOSE(at(r.estimate, 0), 12.0, TOL);
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
		CHECK_CLOSE(at(r.estimate, 1), 0.0, TOL);
		CHECK_CLOSE(at(r.std_error, 1), 0.0, TOL);
		CHECK(std::isnan(at(r.t_statistic, 1)) && std::isnan(at(r.p_value, 1)));
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
		CHECK_CLOSE(at(r.std_error, 0), 0.0, TOL);
		CHECK(std::isnan(at(r.t_statistic, 0)));
		CHECK_CLOSE(at(r.std_error, 1), 2.0, TOL);
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
