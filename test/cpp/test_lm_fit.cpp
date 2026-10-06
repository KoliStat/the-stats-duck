// Direct C++ unit tests for the statsduck::fit_lm regression core (Epic 1.1).
//
// Tests the DuckDB-free OLS + HC0/HC1/HC2/HC3 + CR0/CR1 implementation in
// src/include/lm_core.hpp against golden values produced offline by statsmodels
// 0.14.x (see test/cpp/gen_lm_fit_fixtures.py). statsmodels is the oracle:
// matching its cov_type='HC*'/'cluster' output validates OUR formulas, not just
// a re-derivation of them. Same golden-value + tolerance-band discipline as
// test_linalg.cpp / the r*-distribution SQL tests. Builds standalone (no DuckDB)
// via scripts/run-cpp-tests.sh.
//
// DS1 carries a high-leverage point (x1 = 25), so HC0→HC3 SEs diverge sharply
// (x1's SE runs 0.126 → 0.145 → 0.234 → 0.475) — a bug in any single leverage
// weighting fails loudly here. DS4 carries a per-cluster random effect, so the
// cluster-robust SEs (CR0/CR1) are far larger than HC ignoring the grouping.
//
// Exit 0 = all pass.

#include "lm_core.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using statsduck::fit_lm;
using statsduck::LmOptions;
using statsduck::LmResult;
using statsduck::Vcov;
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
// Relative closeness — used for p-values, whose extreme tails are library
// dependent at the ~1e-8 level; 1% relative still catches a wrong df / wrong
// reference distribution (which would be off by large factors).
static void check_rel(double a, double b, double rtol, const char *label, int line) {
	++g_checks;
	const double denom = std::fabs(b) > 1e-300 ? std::fabs(b) : 1e-300;
	const double rd = std::fabs(a - b) / denom;
	if (!(rd <= rtol)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : rel|%.12g vs %.12g| = %.3g > %.3g\n", line, label, a, b, rd, rtol);
	}
}
#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_CLOSE(a, b, tol) check_close((a), (b), (tol), #a " ~= " #b, __LINE__)

// Tolerances.
static const double TOL = 1e-6;     // β, SE, σ, R², adj-R², loglik
static const double TOL_F = 1e-3;   // F-statistic
static const double RTOL_P = 1e-2;  // p-values (relative)
// Covariance entries are products of two SE-scale quantities, so the 1e-6
// agreement on SEs propagates to about 2·SE·1e-6 on a variance (4e-6 for
// DS1's intercept variance of 4.3). 1e-5 leaves headroom and still separates
// the estimators, whose entries differ by 1e-2 or more.
static const double TOL_COV = 1e-5;

// Build an (n × p) predictor matrix from p column vectors.
static Mat predictors(const std::vector<std::vector<double>> &cols) {
	const std::size_t p = cols.size();
	const std::size_t n = p ? cols[0].size() : 0;
	Mat m(n, p);
	for (std::size_t j = 0; j < p; j++) {
		for (std::size_t r = 0; r < n; r++) {
			m(r, j) = cols[j][r];
		}
	}
	return m;
}

static void check_vec(const std::vector<double> &got, const std::vector<double> &want, double tol,
                      const char *label, int line) {
	if (got.size() != want.size()) {
		++g_fail;
		++g_checks;
		std::printf("  FAIL (line %d): %s : size %zu != %zu\n", line, label, got.size(), want.size());
		return;
	}
	for (std::size_t i = 0; i < want.size(); i++) {
		check_close(got[i], want[i], tol, label, line);
	}
}
static void check_vec_rel(const std::vector<double> &got, const std::vector<double> &want, double rtol,
                          const char *label, int line) {
	if (got.size() != want.size()) {
		++g_fail;
		++g_checks;
		std::printf("  FAIL (line %d): %s : size %zu != %zu\n", line, label, got.size(), want.size());
		return;
	}
	for (std::size_t i = 0; i < want.size(); i++) {
		check_rel(got[i], want[i], rtol, label, line);
	}
}

// Index a result vector safely. A failed fit returns empty vectors, and the
// first (red) run of a new test must report that rather than abort the binary.
static double at(const std::vector<double> &v, std::size_t i) {
	return i < v.size() ? v[i] : std::numeric_limits<double>::quiet_NaN();
}

// Guard the scatter helpers: a FAILED fit reports k = 0 and empty vectors, so
// indexing by a kept position would run off the end and abort the binary
// instead of reporting the failure.
static bool kept_in_range(const std::vector<std::size_t> &kept, std::size_t k) {
	for (const std::size_t j : kept) {
		if (j >= k) {
			return false;
		}
	}
	return true;
}

// A scattered k-vector from a rank-deficient fit: the kept positions carry the
// reduced fit's values, every aliased position is NaN.
static void check_scatter(const std::vector<double> &got, const std::vector<std::size_t> &kept,
                          const std::vector<double> &want, std::size_t k, double tol,
                          const char *label, int line) {
	++g_checks;
	if (got.size() != k || kept.size() != want.size() || !kept_in_range(kept, k)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : size %zu (want %zu), kept %zu (want %zu)\n", line, label,
		            got.size(), k, kept.size(), want.size());
		return;
	}
	std::vector<bool> is_kept(k, false);
	for (std::size_t i = 0; i < kept.size(); i++) {
		is_kept[kept[i]] = true;
		check_close(got[kept[i]], want[i], tol, label, line);
	}
	for (std::size_t j = 0; j < k; j++) {
		if (is_kept[j]) {
			continue;
		}
		++g_checks;
		if (!std::isnan(got[j])) {
			++g_fail;
			std::printf("  FAIL (line %d): %s : position %zu is aliased, expected NaN, got %.12g\n", line,
			            label, j, got[j]);
		}
	}
}

// The same for the k×k covariance: the kept rows and columns carry the reduced
// matrix, and every entry touching an aliased term is NaN.
static void check_cov_scatter(const std::vector<double> &got, const std::vector<std::size_t> &kept,
                              const std::vector<double> &want, std::size_t k, double tol,
                              const char *label, int line) {
	++g_checks;
	const std::size_t rank = kept.size();
	if (got.size() != k * k || want.size() != rank * rank || !kept_in_range(kept, k)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : cov size %zu (want %zu), reduced %zu (want %zu)\n", line,
		            label, got.size(), k * k, want.size(), rank * rank);
		return;
	}
	std::vector<bool> is_kept(k, false);
	for (std::size_t i = 0; i < rank; i++) {
		is_kept[kept[i]] = true;
	}
	for (std::size_t i = 0; i < rank; i++) {
		for (std::size_t j = 0; j < rank; j++) {
			check_close(got[kept[i] * k + kept[j]], want[i * rank + j], tol, label, line);
		}
	}
	for (std::size_t a = 0; a < k; a++) {
		for (std::size_t b = 0; b < k; b++) {
			if (is_kept[a] && is_kept[b]) {
				continue;
			}
			++g_checks;
			if (!std::isnan(got[a * k + b])) {
				++g_fail;
				std::printf("  FAIL (line %d): %s : cov(%zu,%zu) touches an aliased term, expected "
				            "NaN, got %.12g\n",
				            line, label, a, b, got[a * k + b]);
			}
		}
	}
}

// The defining property of R-style column dropping: a rank-deficient fit IS the
// fit of its reduced design, with the dropped coefficients reported as NaN.
// `kept` maps each reduced coefficient to its position in the deficient fit.
// Comparing the two fits directly, instead of against copied constants, keeps
// the goldens in one place (the reduced fit's own test) so they cannot drift.
// gen_lm_fit_fixtures.py asserts the same equivalence against statsmodels.
static void check_equivalent(const LmResult &def, const LmResult &red,
                             const std::vector<std::size_t> &kept, const char *label, int line) {
	check(def.ok && red.ok, label, line);
	if (!def.ok || !red.ok) {
		return; // nothing to compare; the check above already recorded the failure
	}
	check(def.rank == red.k, label, line);
	check(def.n == red.n, label, line);
	check(def.df_residual == red.df_residual, label, line);
	check(def.n_clusters == red.n_clusters, label, line);
	check_scatter(def.beta, kept, red.beta, def.k, TOL, label, line);
	check_scatter(def.std_error, kept, red.std_error, def.k, TOL, label, line);
	check_scatter(def.t_statistic, kept, red.t_statistic, def.k, TOL, label, line);
	check_scatter(def.p_value, kept, red.p_value, def.k, TOL, label, line);
	check_cov_scatter(def.cov, kept, red.cov, def.k, TOL_COV, label, line);
	check_close(def.sigma, red.sigma, TOL, label, line);
	check_close(def.r_squared, red.r_squared, TOL, label, line);
	check_close(def.adj_r_squared, red.adj_r_squared, TOL, label, line);
	check_close(def.f_statistic, red.f_statistic, TOL_F, label, line);
	check_close(def.loglik, red.loglik, TOL, label, line);
}

static LmResult fit(const std::vector<double> &y, const Mat &X, Vcov v, bool intercept) {
	LmOptions opt;
	opt.vcov = v;
	opt.intercept = intercept;
	return fit_lm(y, X, opt);
}

static LmResult fit_cl(const std::vector<double> &y, const Mat &X, Vcov v, bool intercept,
                       const std::vector<int> &clusters) {
	LmOptions opt;
	opt.vcov = v;
	opt.intercept = intercept;
	return fit_lm(y, X, opt, &clusters);
}

// ───────────────────────── DS1: intercept, 2 predictors, n=12 ────────────────
// One high-leverage point (x1 = 25). Golden source: gen_lm_fit_fixtures.py.
// DS1's data at file scope: the rank-deficient tests below build deficient
// designs whose kept columns are exactly these, so this fit is their oracle.
static const std::vector<double> kDs1Y = {9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55};
static const std::vector<double> kDs1X1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25};
static const std::vector<double> kDs1X2 = {5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4};

static void test_ds1_hetero() {
	std::printf("ds1_hetero (intercept, 2 predictors, n=12, high-leverage)\n");
	const std::vector<double> &y = kDs1Y;
	const Mat X = predictors({kDs1X1, kDs1X2});

	const std::vector<double> beta = {10.1205711006471, 2.05794012307408, -0.978656740125124};

	// Classical fit — also carries the model-level summary.
	{
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(r.ok);
		CHECK(r.n == 12);
		CHECK(r.k == 3);
		CHECK(r.df_residual == 9);
		CHECK(r.has_intercept);
		CHECK(r.terms.size() == 3 && r.terms[0] == "(Intercept)" && r.terms[1] == "x1" &&
		      r.terms[2] == "x2");
		check_vec(r.beta, beta, TOL, "ds1.const.beta", __LINE__);
		check_vec(r.std_error, {2.07411978596965, 0.130310370148038, 0.318083671987483}, TOL,
		          "ds1.const.se", __LINE__);
		CHECK_CLOSE(r.r_squared, 0.967161068784953, TOL);
		CHECK_CLOSE(r.adj_r_squared, 0.959863528514942, TOL);
		CHECK_CLOSE(r.f_statistic, 132.532474368047, TOL_F);
		check_rel(r.f_p_value, 2.10741740745104e-07, RTOL_P, "ds1.f_p", __LINE__);
		CHECK_CLOSE(r.sigma, 2.73206285423836, TOL);
		// Increment A fields (#34, #35): rank (== k until rank-deficient fits
		// land), the Gaussian log-likelihood, and the full k×k covariance
		// (row-major, aligned to terms). The diagonal is SE² by construction,
		// so that check is tight; the goldens below carry the oracle values.
		CHECK(r.rank == 3);
		CHECK(r.cov.size() == 9);
		for (std::size_t j = 0; j < 3; j++) {
			CHECK_CLOSE(r.cov[j * 3 + j], r.std_error[j] * r.std_error[j], 1e-12);
		}
		CHECK_CLOSE(r.cov[0 * 3 + 1], r.cov[1 * 3 + 0], 1e-12);
		CHECK_CLOSE(r.loglik, -27.3618533411821, TOL);
		check_vec(r.cov,
		          {4.30197288655079, -0.140244518800206, -0.523287599804694, -0.140244518800206,
		           0.0169807925681187, 0.00229470169839442, -0.523287599804694, 0.00229470169839442,
		           0.101177222385041},
		          TOL_COV, "ds1.const.cov", __LINE__);
	}
	// HC0 — White. β is invariant across vcov; SE/t/p change.
	{
		auto r = fit(y, X, Vcov::kHC0, true);
		CHECK(r.ok);
		check_vec(r.beta, beta, TOL, "ds1.hc0.beta", __LINE__);
		check_vec(r.std_error, {2.00449812539262, 0.125533332405513, 0.281912427062746}, TOL,
		          "ds1.hc0.se", __LINE__);
		check_vec(r.t_statistic, {5.04893018977747, 16.3935751854836, -3.47149201729692}, TOL,
		          "ds1.hc0.t", __LINE__);
		check_vec_rel(r.p_value, {0.000691348081107212, 5.20320775759817e-08, 0.00703220770281606},
		              RTOL_P, "ds1.hc0.p", __LINE__);
	}
	// HC1 = HC0 × n/(n−k).
	{
		auto r = fit(y, X, Vcov::kHC1, true);
		check_vec(r.std_error, {2.31459506457106, 0.144953406513188, 0.325524431305154}, TOL,
		          "ds1.hc1.se", __LINE__);
		// The robust covariance is what lin_hyp / emmeans consume, so it is
		// validated against statsmodels cov_params(), not only the classical one.
		check_vec(r.cov,
		          {5.35735031293671, -0.191061988599497, -0.649665858329423, -0.191061988599497,
		           0.0210114900597774, 0.0105710141727397, -0.649665858329423, 0.0105710141727397,
		           0.105966155376544},
		          TOL_COV, "ds1.hc1.cov", __LINE__);
	}
	// HC2 — leverage weight 1/(1−h).
	{
		auto r = fit(y, X, Vcov::kHC2, true);
		check_vec(r.std_error, {2.43296771690633, 0.23428267668393, 0.326062128664413}, TOL,
		          "ds1.hc2.se", __LINE__);
	}
	// HC3 — leverage weight 1/(1−h)². Largest, driven by the high-leverage row.
	{
		auto r = fit(y, X, Vcov::kHC3, true);
		check_vec(r.std_error, {3.28023865123212, 0.474879784067986, 0.383856780518927}, TOL,
		          "ds1.hc3.se", __LINE__);
		check_vec(r.t_statistic, {3.08531548362972, 4.33360229708044, -2.5495361546098}, TOL,
		          "ds1.hc3.t", __LINE__);
		check_vec_rel(r.p_value, {0.0130276414567639, 0.00189509686010231, 0.0312184942489756},
		              RTOL_P, "ds1.hc3.p", __LINE__);
		CHECK(std::string(statsduck::vcov_name(r.vcov)) == "HC3");
	}
}

// ───────────────────────── DS2: NO intercept, 2 predictors, n=10 ─────────────
// Exercises the uncentered R²/F path (TSS = Σy²).
static void test_ds2_noint() {
	std::printf("ds2_noint (no intercept, 2 predictors, n=10)\n");
	const std::vector<double> y = {3, 5, 11, 10, 19, 18, 27, 26, 36, 34};
	const Mat X = predictors({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	                          {2, 1, 4, 3, 6, 5, 8, 7, 10, 9}});
	const std::vector<double> beta = {1.46666666666667, 2.06666666666667};

	{
		auto r = fit(y, X, Vcov::kConst, false);
		CHECK(r.ok);
		CHECK(r.k == 2);
		CHECK(r.df_residual == 8);
		CHECK(!r.has_intercept);
		CHECK(r.terms.size() == 2 && r.terms[0] == "x1" && r.terms[1] == "x2");
		check_vec(r.beta, beta, TOL, "ds2.const.beta", __LINE__);
		check_vec(r.std_error, {0.511565583679384, 0.511565583679384}, TOL, "ds2.const.se", __LINE__);
		CHECK_CLOSE(r.r_squared, 0.995663956639566, TOL);
		CHECK_CLOSE(r.adj_r_squared, 0.994579945799458, TOL);
		CHECK_CLOSE(r.f_statistic, 918.5, TOL_F);
		CHECK_CLOSE(r.sigma, 1.61245154965971, TOL);
		CHECK(r.rank == 2);
		CHECK(r.cov.size() == 4);
		CHECK_CLOSE(r.loglik, -17.8512248006129, TOL); // no-intercept path
		check_vec(r.cov, {0.261699346405229, -0.258300653594771, -0.258300653594771, 0.261699346405229},
		          TOL_COV, "ds2.const.cov", __LINE__);
	}
	{
		auto r = fit(y, X, Vcov::kHC0, false);
		check_vec(r.std_error, {0.445961373869256, 0.468229479484472}, TOL, "ds2.hc0.se", __LINE__);
	}
	{
		auto r = fit(y, X, Vcov::kHC3, false);
		check_vec(r.std_error, {0.549203493672471, 0.588771131434858}, TOL, "ds2.hc3.se", __LINE__);
	}
}

// ───────────────────────── DS3: intercept, 1 predictor, n=8 ──────────────────
static void test_ds3_simple() {
	std::printf("ds3_simple (intercept, 1 predictor, n=8)\n");
	const std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1};
	const Mat X = predictors({{1, 2, 3, 4, 5, 6, 7, 8}});

	{
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(r.ok);
		check_vec(r.beta, {0.0357142857142889, 1.99761904761905}, TOL, "ds3.const.beta", __LINE__);
		check_vec(r.std_error, {0.140385362081028, 0.0278004442668841}, TOL, "ds3.const.se", __LINE__);
		CHECK_CLOSE(r.r_squared, 0.998839286601139, TOL);
		CHECK_CLOSE(r.sigma, 0.180167470594215, TOL);
		CHECK(r.rank == 2);
		CHECK_CLOSE(r.loglik, 3.51016777175683, TOL); // positive: σ < 1 here
	}
	{
		auto r = fit(y, X, Vcov::kHC1, true);
		check_vec(r.std_error, {0.111185996792465, 0.0229806595898712}, TOL, "ds3.hc1.se", __LINE__);
	}
}

// ───────────────────────── DS4: clustered, intercept, 2 predictors, n=25 ─────
// G=5 uneven clusters with a per-cluster random effect. Golden source:
// gen_lm_fit_fixtures.py (statsmodels cov_type='cluster'). CR0 = raw sandwich;
// CR1 = CR0 × [G/(G−1)]·[(N−1)/(N−k)] (statsmodels default). p-values: t(G−1).
// DS4's data at file scope, for the clustered rank-deficient case below.
static const std::vector<double> kDs4Y = {-0.297, 2.143,  -1.086, 0.126,  8.36,  9.089, 3.768,
                                          5.903,  7.854,  9.056,  5.972,  -0.748, 4.735, 6.761,
                                          9.777,  10.746, -1.561, 6.323,  14.583, 6.845, 4.295,
                                          8.41,   20.68,  2.813,  4.244};
static const std::vector<double> kDs4X1 = {-0.409, 1.058, -0.839, -0.811, 4.332,  3.287, 0.786,
                                           1.263,  2.338, 3.393,  3.014,  -1.492, 1.75,  1.758,
                                           4.106,  3.273, -2.403, 0.956,  5.138,  1.023, 0.921,
                                           3.054,  8.698, 0.258,  1.246};
static const std::vector<double> kDs4X2 = {0.124,  0.303,  0.524, 0.001,  1.344,  -0.714, -0.831,
                                           -2.37,  -1.861, -0.861, 0.56,  -1.266, 0.12,   -1.064,
                                           0.333,  -2.359, -0.2,  -1.542, -0.971, -1.307, 0.286,
                                           0.378,  -0.754, 0.331, 1.35};
static const std::vector<int> kDs4G = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                                       3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4};

static void test_ds4_cluster() {
	std::printf("ds4_cluster (intercept, 2 predictors, n=25, G=5 clusters)\n");
	const std::vector<double> &y = kDs4Y;
	const Mat X = predictors({kDs4X1, kDs4X2});
	const std::vector<int> &g = kDs4G;
	const std::vector<double> beta = {1.77805604849209, 2.04352854122001, -1.04867236368235};

	// Unclustered classical fit on the same data → n_clusters == 0.
	{
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(r.ok);
		CHECK(r.n_clusters == 0);
	}
	// CR0 — raw cluster sandwich (no finite-sample correction).
	{
		auto r = fit_cl(y, X, Vcov::kCR0, true, g);
		CHECK(r.ok);
		CHECK(r.n == 25);
		CHECK(r.k == 3);
		CHECK(r.df_residual == 22); // reported df stays n−k …
		CHECK(r.n_clusters == 5);   // … while inference uses t(G−1)
		check_vec(r.beta, beta, TOL, "ds4.cr0.beta", __LINE__);
		check_vec(r.std_error, {0.416769971097275, 0.0715707663240662, 0.165112976402852}, TOL,
		          "ds4.cr0.se", __LINE__);
		check_vec(r.t_statistic, {4.2662767756775, 28.5525591827128, -6.35124135321587}, TOL,
		          "ds4.cr0.t", __LINE__);
		check_vec_rel(r.p_value, {0.0129885429969767, 8.95422460575341e-06, 0.00314880404976347},
		              RTOL_P, "ds4.cr0.p", __LINE__);
		CHECK(std::string(statsduck::vcov_name(r.vcov)) == "CR0");
		check_vec(r.cov,
		          {0.173697208808423, -0.0092148985536128, 0.0244365529635018, -0.00921489855361279,
		           0.00512237459221409, 0.00278582959788755, 0.0244365529635018, 0.00278582959788755,
		           0.0272622949766088},
		          TOL_COV, "ds4.cr0.cov", __LINE__);
		CHECK_CLOSE(r.loglik, -32.9761104667665, TOL); // vcov does not change β or RSS
	}
	// CR1 — Stata / statsmodels default: CR0 × [G/(G−1)]·[(N−1)/(N−k)].
	{
		auto r = fit_cl(y, X, Vcov::kCR1, true, g);
		CHECK(r.ok);
		CHECK(r.n_clusters == 5);
		CHECK(r.df_residual == 22);
		check_vec(r.beta, beta, TOL, "ds4.cr1.beta", __LINE__);
		check_vec(r.std_error, {0.486682473686185, 0.083576649024175, 0.19281041669548}, TOL,
		          "ds4.cr1.se", __LINE__);
		check_vec(r.t_statistic, {3.65342116190234, 24.4509508945364, -5.43887815635292}, TOL,
		          "ds4.cr1.t", __LINE__);
		check_vec_rel(r.p_value, {0.021705842695979, 1.66012518838537e-05, 0.00554715018524578},
		              RTOL_P, "ds4.cr1.p", __LINE__);
		CHECK(std::string(statsduck::vcov_name(r.vcov)) == "CR1");
		// The cluster-robust covariance is the one lin_hyp / emmeans need.
		CHECK(r.rank == 3);
		CHECK(r.cov.size() == 9);
		for (std::size_t j = 0; j < 3; j++) {
			CHECK_CLOSE(r.cov[j * 3 + j], r.std_error[j] * r.std_error[j], 1e-12);
		}
		check_vec(r.cov,
		          {0.236859830193304, -0.0125657707549265, 0.033322572222957, -0.0125657707549265,
		           0.00698505626211012, 0.00379885854257393, 0.033322572222957, 0.00379885854257393,
		           0.0371758567862847},
		          TOL_COV, "ds4.cr1.cov", __LINE__);
	}
}

// ───────────── DS5: rank-deficient fits — R-style column dropping ───────────
// A dependent column is DROPPED rather than spread across the dependent set, so
// the fit equals the fit of the reduced design and the aliased coefficient is
// NaN (R's "not defined because of singularities"). Each case here reduces to a
// design fitted elsewhere in this file, so check_equivalent compares the two
// fits directly and no new goldens are needed. The ds5_* sections of
// gen_lm_fit_fixtures.py assert the same equivalence against statsmodels, and
// record why statsmodels cannot oracle the dropped fit itself: it solves with
// pinv, whose minimum-norm solution keeps every column and spreads the
// coefficient (for the middle-alias case below it reports 0.4116 and 0.8232
// across x1 and 2·x1 where dropping gives 2.0579 and NaN).
static void test_ds5_rank_deficient() {
	std::printf("ds5_rank_deficient (dropped columns, aliased NaN, df on rank)\n");

	const std::size_t n1 = kDs1X1.size();
	std::vector<double> x1_plus_x2(n1), two_x1(n1);
	for (std::size_t r = 0; r < n1; r++) {
		x1_plus_x2[r] = kDs1X1[r] + kDs1X2[r];
		two_x1[r] = 2.0 * kDs1X1[r];
	}
	const Mat ds1_reduced = predictors({kDs1X1, kDs1X2});

	// Trailing alias: [x1, x2, x1+x2] keeps design columns {0, 1, 2}.
	{
		auto red = fit(kDs1Y, ds1_reduced, Vcov::kConst, true);
		auto def = fit(kDs1Y, predictors({kDs1X1, kDs1X2, x1_plus_x2}), Vcov::kConst, true);
		CHECK(def.ok);
		CHECK(def.k == 4);
		CHECK(def.rank == 3);
		CHECK(def.df_residual == 9); // n − rank, not n − k
		// Every term keeps its name, so unnest still shows the aliased one.
		CHECK(def.terms.size() == 4 && def.terms[3] == "x3");
		check_equivalent(def, red, {0, 1, 2}, "ds5.sum.const", __LINE__);
		CHECK_CLOSE(def.loglik, -27.3618533411821, TOL); // absolute pin, DS1's value
	}

	// Middle alias: [x1, 2·x1, x2] keeps {0, 1, 3}. The kept set is not a
	// prefix, so a scatter that assumed one would fail here.
	{
		auto red = fit(kDs1Y, ds1_reduced, Vcov::kConst, true);
		auto def = fit(kDs1Y, predictors({kDs1X1, two_x1, kDs1X2}), Vcov::kConst, true);
		CHECK(def.rank == 3 && def.k == 4);
		CHECK(def.terms.size() == 4 && def.terms[2] == "x2"); // aliased, by its own name
		check_equivalent(def, red, {0, 1, 3}, "ds5.middle.const", __LINE__);
	}

	// HC1 on the deficient design. The finite-sample factor is n/(n−rank)
	// = 12/9; a k-based 12/8 would inflate every standard error by 6%.
	{
		auto red = fit(kDs1Y, ds1_reduced, Vcov::kHC1, true);
		auto def = fit(kDs1Y, predictors({kDs1X1, two_x1, kDs1X2}), Vcov::kHC1, true);
		check_equivalent(def, red, {0, 1, 3}, "ds5.middle.hc1", __LINE__);
		CHECK_CLOSE(at(def.std_error, 1), 0.144953406513188, TOL); // absolute pin, DS1 HC1
	}

	// HC3 too: its leverage weights are computed in the reduced space.
	//
	// Note the column order. In [x1, x1+x2, x2] it is x2 that is dependent —
	// x2 = (x1+x2) − x1 — so selection keeps [1, x1, x1+x2], a different basis
	// for the same column space. The fitted values still match DS1's, but the
	// coefficients do not (that basis reports 3.0366 for x1, which is
	// 2.0579 + 0.9787). Keeping the dependent column last is what makes DS1's
	// coefficients the ones that survive.
	{
		auto red = fit(kDs1Y, ds1_reduced, Vcov::kHC3, true);
		auto def = fit(kDs1Y, predictors({kDs1X1, kDs1X2, x1_plus_x2}), Vcov::kHC3, true);
		check_equivalent(def, red, {0, 1, 2}, "ds5.hc3", __LINE__);
	}

	// Clustered deficient fit: DS4's design with x2 duplicated. CR1's factor is
	// [G/(G−1)]·[(N−1)/(N−rank)] = (5/4)·(24/22); a k-based (24/21) would miss.
	{
		auto red = fit_cl(kDs4Y, predictors({kDs4X1, kDs4X2}), Vcov::kCR1, true, kDs4G);
		auto def = fit_cl(kDs4Y, predictors({kDs4X1, kDs4X2, kDs4X2}), Vcov::kCR1, true, kDs4G);
		CHECK(def.k == 4 && def.rank == 3);
		CHECK(def.df_residual == 22);
		CHECK(def.n_clusters == 5); // inference still references t(G−1)
		check_equivalent(def, red, {0, 1, 2}, "ds5.cluster.cr1", __LINE__);
		CHECK_CLOSE(at(def.std_error, 1), 0.083576649024175, TOL); // absolute pin, DS4 CR1
	}

	// A constant predictor beside the intercept: the predictor aliases, and the
	// fit is DS3's. The intercept itself is always kept — it comes first.
	{
		const std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1};
		const std::vector<double> x1 = {1, 2, 3, 4, 5, 6, 7, 8};
		const std::vector<double> five(8, 5.0);
		auto red = fit(y, predictors({x1}), Vcov::kConst, true);
		auto def = fit(y, predictors({x1, five}), Vcov::kConst, true);
		CHECK(def.k == 3 && def.rank == 2);
		check_equivalent(def, red, {0, 1}, "ds5.constant_pred", __LINE__);
	}

	// No intercept requested, and the first predictor is a duplicate of the
	// second: dropping keeps the earlier column, so x2 aliases.
	{
		auto red = fit(kDs1Y, predictors({kDs1X1}), Vcov::kConst, false);
		auto def = fit(kDs1Y, predictors({kDs1X1, kDs1X1}), Vcov::kConst, false);
		CHECK(def.k == 2 && def.rank == 1);
		CHECK(!def.has_intercept);
		check_equivalent(def, red, {0}, "ds5.noint", __LINE__);
	}

	// A design with no independent column at all is not estimable.
	{
		const std::vector<double> y = {1, 2, 3, 4};
		auto r = fit(y, predictors({{0, 0, 0, 0}}), Vcov::kConst, false);
		CHECK(!r.ok);
		CHECK(!r.error.empty());
	}

	// rank == n leaves no residual degrees of freedom, exactly as n == k did
	// before: [1, x1, 2·x1, x2] over 3 rows has rank 3.
	{
		const std::vector<double> y = {1, 2, 3};
		auto r = fit(y, predictors({{1, 2, 4}, {2, 4, 8}, {5, 1, 3}}), Vcov::kConst, true);
		CHECK(!r.ok);
	}
}

// Cluster-robust requires a per-row cluster id and ≥ 2 clusters.
static void test_cluster_errors() {
	std::printf("cluster error paths\n");
	const std::vector<double> y = {1, 2, 3, 4, 5, 6, 7, 8};
	const Mat X = predictors({{1, 2, 1, 2, 1, 2, 1, 2}}); // k=2, n=8

	// CR* with no cluster ids → fail.
	{
		LmOptions opt;
		opt.vcov = Vcov::kCR1;
		auto r = fit_lm(y, X, opt, nullptr);
		CHECK(!r.ok);
		CHECK(!r.error.empty());
	}
	// CR* with a wrong-length cluster vector → fail.
	{
		LmOptions opt;
		opt.vcov = Vcov::kCR0;
		std::vector<int> g = {0, 0, 1}; // length 3 != n
		auto r = fit_lm(y, X, opt, &g);
		CHECK(!r.ok);
	}
	// A single cluster (G=1) → fail (cluster-robust needs ≥ 2).
	{
		LmOptions opt;
		opt.vcov = Vcov::kCR1;
		std::vector<int> g(8, 0);
		auto r = fit_lm(y, X, opt, &g);
		CHECK(!r.ok);
	}
}

// ───────────────────────── Error paths & helpers ────────────────────────────
static void test_errors() {
	std::printf("error paths\n");
	// n ≤ k → not enough rows.
	{
		const std::vector<double> y = {1, 2};
		const Mat X = predictors({{1, 2}}); // k = 2 (intercept + x1), n = 2
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(!r.ok);
		CHECK(!r.error.empty());
	}
	// Perfectly collinear predictors (x2 = 2·x1) used to fail here. Since the
	// rank-deficient path landed they fit, with the later column aliased —
	// see test_ds5_rank_deficient for the full behavior.
	{
		const std::vector<double> y = {1, 3, 2, 5, 4, 7};
		const Mat X = predictors({{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}});
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(r.ok);
		CHECK(r.k == 3 && r.rank == 2);
		CHECK(std::isnan(at(r.beta, 2)) && std::isnan(at(r.std_error, 2)));
	}
	// Row-count mismatch.
	{
		const std::vector<double> y = {1, 2, 3};
		const Mat X = predictors({{1, 2, 3, 4}});
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(!r.ok);
	}
}

static void test_vcov_parse() {
	std::printf("parse_vcov / vcov_name\n");
	Vcov v;
	CHECK(statsduck::parse_vcov("const", v) && v == Vcov::kConst);
	CHECK(statsduck::parse_vcov("OLS", v) && v == Vcov::kConst);
	CHECK(statsduck::parse_vcov("nonrobust", v) && v == Vcov::kConst);
	CHECK(statsduck::parse_vcov("hc0", v) && v == Vcov::kHC0);
	CHECK(statsduck::parse_vcov("HC1", v) && v == Vcov::kHC1);
	CHECK(statsduck::parse_vcov("Hc2", v) && v == Vcov::kHC2);
	CHECK(statsduck::parse_vcov("HC3", v) && v == Vcov::kHC3);
	CHECK(statsduck::parse_vcov("cr0", v) && v == Vcov::kCR0);
	CHECK(statsduck::parse_vcov("CR1", v) && v == Vcov::kCR1);
	CHECK(statsduck::parse_vcov("cluster", v) && v == Vcov::kCR1); // alias → CR1
	CHECK(!statsduck::parse_vcov("HC4", v));
	CHECK(!statsduck::parse_vcov("CR2", v)); // not shipped yet
	CHECK(!statsduck::parse_vcov("bogus", v));
	CHECK(std::string(statsduck::vcov_name(Vcov::kConst)) == "const");
	CHECK(std::string(statsduck::vcov_name(Vcov::kHC0)) == "HC0");
	CHECK(std::string(statsduck::vcov_name(Vcov::kHC3)) == "HC3");
	CHECK(std::string(statsduck::vcov_name(Vcov::kCR0)) == "CR0");
	CHECK(std::string(statsduck::vcov_name(Vcov::kCR1)) == "CR1");
}

int main() {
	std::printf("== test_lm_fit ==\n");
	test_ds1_hetero();
	test_ds2_noint();
	test_ds3_simple();
	test_ds4_cluster();
	test_ds5_rank_deficient();
	test_cluster_errors();
	test_errors();
	test_vcov_parse();
	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	return g_fail == 0 ? 0 : 1;
}
