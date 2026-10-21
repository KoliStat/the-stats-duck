// Direct C++ unit tests for statsduck::fit_glm, the IRLS loop and its
// classical and robust covariance (#22).
//
// The first two tests compare a gaussian identity-link fit against fit_lm
// on the same data, first with the classical covariance and then with
// each of the six robust estimators. Gaussian with an identity link IS
// ordinary least squares, so the two fitters must agree exactly. Those
// tests come first because they catch the most for the least: fit_lm is
// already validated against statsmodels, so this ties the new fitter to a
// checked one instead of to a fresh set of constants. Both bands are 0.0,
// which passes only on exact equality, and both measure 0.
//
// The oracle comparisons come next. Against the statsmodels goldens in
// test/cpp/glm_fit_fixtures.hpp: DS1 binomial, DS3 poisson with an
// offset, DS4 gamma, DS5 gaussian, DS7 rank deficient and DS8 clustered,
// then DS8's CR0 and CR1 goldens for the cluster-robust sandwich. Against
// R 4.6.1: DS1's HC0 through HC3, derived from R's own fit$weights and
// hatvalues(); a WEIGHTED gaussian fit's loglik and aic, which no
// statsmodels fixture here covers because every gaussian design in the
// generated header has unit weights; and gamma with its canonical inverse
// link beside poisson with a sqrt link, the two links nothing else in this
// file fits. Every band is MEASURED rather than assumed, and the
// measurement is recorded beside it and reprinted on every run.
//
// TWO ORACLES, BECAUSE NEITHER COVERS EVERYTHING. gen_glm_fit_fixtures.py
// generates statsmodels' cov_type='cluster' for DS8 and nothing else
// robust, so the statsmodels goldens carry no HC values at all. R supplies
// those. STATSMODELS CANNOT SUPPLY A GLM'S HC2 OR HC3 AT ALL. In 0.15.0,
// cov_HC0 through cov_HC3 are defined only on OLSResults, so a GLM's
// cov_type request falls through to one undifferentiated routine and
// returns HC0's numbers under all four names, labelled with the name that
// was asked for. R is the only available oracle for a GLM's HC2 and HC3.
// Extending the generator would therefore commit HC0's numbers three more
// times, as HC1, HC2 and HC3 goldens. The mechanism, the measurements and
// the OLS control are in
// notes/engineering/2026-10-statsmodels-hc-for-a-glm.md. The R values live
// in this file, with the exact R program beside them.
//
// Then the identities that need no oracle at all. HC2 and HC3 against
// HC0 on a design whose weighted hat diagonals are all equal, which is
// the only test that tells the WORKING weight from the prior weight in
// the leverage. HC1 and CR1 against HC0 and CR0 on a rank-deficient
// design, which is the only test that tells n - rank from n - k. A
// weighted sandwich against fit_lm on the sqrt(w)-scaled design. HC0
// against CR0 with one cluster per row. An aliased column under every
// estimator. The cluster-id convention against fit_lm's. null_deviance
// against an independent intercept-only fit. Grouped binomial against
// ungrouped. A scaled trial count. And a deficient fit against the
// reduced design it must reproduce bit for bit. Then the three mechanisms
// that have no golden: the support test on the fitted mean, step halving,
// and the probit and cloglog links, which are checked against an
// independent Nelder-Mead minimization of the same deviance. Last, the
// input classes that have no fit to report: no residual degrees of
// freedom, a zero-weight row, and a design with no columns at all.
//
// DS2 is not fitted here. Its numbers are used only as the grouped half of
// the grouping identity. DS6 IS fitted. test_separation fits it, but
// never checks it against its goldens. The fixture header marks every
// one of them unsafe to assert tightly. The reason is written at DS6's
// own definition. What test_separation asserts instead is the
// separation flag, the termination state, and a near-zero deviance at a
// loose, order-of-magnitude band.
//
// A NOTE ON THE HISTORY. Commit ef8b283's subject reads "oracle the robust
// covariance on a binomial and a clustered fit", and the binomial half of
// that claim was wrong when it landed. That commit oracled only DS8's
// clustered CR0 and CR1; DS1's only robust assertions were that beta does
// not move and that the standard error does. The binomial HC oracle is
// the R comparison in this file, added afterwards. The subject stands in
// the history, so this paragraph is the correction.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "glm_core.hpp"

#include "glm_fit_fixtures.hpp"
#include "optimize.hpp" // statsduck::optimize::nelder_mead, for the probit and cloglog oracle

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

using statsduck::linalg::Mat;

static int g_checks = 0;
static int g_fail = 0;

static void check(bool cond, const char *msg) {
	++g_checks;
	if (!cond) {
		++g_fail;
		std::printf("  FAIL: %s\n", msg);
	}
}

// True when a and b agree to within rtol, relative to the larger of the
// two magnitudes. a == b (including both equal to 0) always passes, which
// lets a caller ask for an exact match by passing rtol = 0.0. Copied from
// test_glm_core.cpp so the two suites read the same way.
static bool close_rel(double a, double b, double rtol) {
	const double diff = std::fabs(a - b);
	if (diff == 0.0) {
		return true;
	}
	const double scale = std::max(std::fabs(a), std::fabs(b));
	return diff / scale <= rtol;
}

// Relative deviation between two values, against the larger magnitude.
// Exactly equal values, both zeros included, give 0.
static double rel_dev(double a, double b) {
	const double diff = std::fabs(a - b);
	if (diff == 0.0) {
		return 0.0;
	}
	const double scale = std::max(std::fabs(a), std::fabs(b));
	return scale > 0.0 ? diff / scale : diff;
}

// Index a result vector safely. A failed fit returns empty vectors, and
// the first (red) run of a new test must report that rather than abort the
// binary.
static double at(const std::vector<double> &v, std::size_t i) {
	return i < v.size() ? v[i] : std::numeric_limits<double>::quiet_NaN();
}

// The generated fixtures are raw arrays. fit_glm takes std::vector, so
// each response, weight and offset column is copied once here.
template <std::size_t N>
static std::vector<double> vec(const double (&a)[N]) {
	return std::vector<double>(a, a + N);
}

// An n x p predictor matrix from p column pointers. The intercept is NOT
// one of them: fit_glm prepends the constant itself, like fit_lm.
static Mat design(std::size_t n, const std::vector<const double *> &cols) {
	Mat m(n, cols.size());
	for (std::size_t j = 0; j < cols.size(); j++) {
		for (std::size_t r = 0; r < n; r++) {
			m(r, j) = cols[j][r];
		}
	}
	return m;
}

static const std::vector<double> ds1_y = vec(kDs1Y);
static const std::vector<double> ds2_y = vec(kDs2Y);
static const std::vector<double> ds2_w = vec(kDs2W);
static const std::vector<double> ds3_y = vec(kDs3Y);
static const std::vector<double> ds3_offset = vec(kDs3Offset);
static const std::vector<double> ds4_y = vec(kDs4Y);
static const std::vector<double> ds5_y = vec(kDs5Y);
static const std::vector<double> ds6_y = vec(kDs6Y);
static const std::vector<double> ds7_y = vec(kDs7Y);
static const std::vector<double> ds8_y = vec(kDs8Y);
// DS8's cluster labels. The fixture stores them as a raw int array, and
// fit_glm takes a std::vector<int>, so they are copied once here.
static const std::vector<int> ds8_clusters(kDs8Groups, kDs8Groups + kDs8N);

static Mat ds1_design() {
	return design(kDs1N, {kDs1X1, kDs1X2});
}
static Mat ds2_design() {
	return design(kDs2N, {kDs2X1, kDs2X2});
}
static Mat ds3_design() {
	return design(kDs3N, {kDs3X1});
}
static Mat ds4_design() {
	return design(kDs4N, {kDs4X1});
}
static Mat ds5_design() {
	return design(kDs5N, {kDs5X1, kDs5X2});
}
static Mat ds6_design() {
	return design(kDs6N, {kDs6X1});
}
static Mat ds7_design() {
	return design(kDs7N, {kDs7X1, kDs7X2, kDs7X3});
}
static Mat ds8_design() {
	return design(kDs8N, {kDs8X1});
}

// ── The strongest check needs no oracle ──────────────────────────────────
static void test_gaussian_equals_lm_fit() {
	std::printf("gaussian identity equals fit_lm on DS5\n");

	// Gaussian with an identity link IS ordinary least squares. The two
	// fitters must agree on every coefficient, every standard error, the
	// full covariance matrix, df_residual, and dispersion against sigma
	// squared. If the weighted solve or the covariance assembly is wrong,
	// this is where it shows.
	statsduck::GlmOptions go;
	go.family = statsduck::Family::kGaussian;
	go.link = statsduck::Link::kIdentity;
	const statsduck::GlmResult g = statsduck::fit_glm(ds5_y, ds5_design(), go);
	statsduck::LmOptions lo;
	const statsduck::LmResult o = statsduck::fit_lm(ds5_y, ds5_design(), lo);

	check(g.ok && o.ok, "both fitters succeed on DS5");
	check(g.k == o.k && g.rank == o.rank, "same k and rank");
	check(g.df_residual == o.df_residual, "same residual df");
	// The band is 1e-12 rather than something looser because gaussian with
	// an identity link converges in one step: the working weights are all
	// 1 and the working response is y, so IRLS IS the normal-equations
	// solve. Any larger discrepancy means the two paths differ
	// structurally, not numerically.
	double worst = 0.0;
	for (std::size_t j = 0; j < o.k; j++) {
		worst = std::max(worst, rel_dev(at(g.beta, j), at(o.beta, j)));
		worst = std::max(worst, rel_dev(at(g.std_error, j), at(o.std_error, j)));
		check(close_rel(at(g.beta, j), at(o.beta, j), 1e-12), "same coefficient");
		check(close_rel(at(g.std_error, j), at(o.std_error, j), 1e-12), "same standard error");
	}
	for (std::size_t e = 0; e < o.cov.size(); e++) {
		worst = std::max(worst, rel_dev(at(g.cov, e), at(o.cov, e)));
		check(close_rel(at(g.cov, e), at(o.cov, e), 1e-12), "same covariance entry");
	}
	worst = std::max(worst, rel_dev(g.dispersion, o.sigma * o.sigma));
	check(close_rel(g.dispersion, o.sigma * o.sigma, 1e-12), "dispersion is sigma squared");
	// The log-likelihood at a band of 0.0, which passes only on exact
	// equality. glm_core's gaussian branch computes it through lm_core's
	// own sequence of log calls so the two can never differ by a rounding
	// bit (commit 0245a1d), and the Σ ln(w)/2 prior-weight term added
	// afterwards is exactly 0.0 at DS5's unit weights. Nothing asserted
	// that equality until this line; the comment in glm_core.cpp was the
	// only record of it.
	check(close_rel(g.loglik, o.loglik, 0.0), "the same log-likelihood as fit_lm, bit for bit");
	std::printf("  worst relative gap against fit_lm: %.3g (%d iterations)\n", worst, g.iterations);
	check(g.converged, "an identity-link gaussian fit converges");
	check(g.iterations <= 3, "and does so in very few iterations, since it is linear");
	check(!g.separation, "separation is false for a gaussian fit");
}

// ── Every robust estimator reduces to fit_lm's at gaussian identity ──────
static void test_robust_equals_lm_fit() {
	std::printf("every robust estimator equals fit_lm's at gaussian identity\n");

	// This is the test for the whole sandwich, and it needs no oracle.
	//
	// At gaussian with an identity link link_deriv is 1 and variance is 1,
	// so the score u_i = x_i*w_i*(y_i - mu_i) reduces to the response
	// residual times the row of the design, which is exactly what
	// lm_core's meat is built from. The bread is XtWX with every working
	// weight 1, which is XtX. So each robust estimator here must agree
	// with lm_core's. lm_core's are already validated against statsmodels.
	// test_lm_fit.cpp's ds1_hetero carries their absolute goldens, and DS5
	// IS that design. The fixture header's self-check 3 records that it
	// reproduces ds1_hetero.
	//
	// The band is 0.0, which close_rel passes only on exact equality. That
	// is not optimism: both paths are written to accumulate the same terms
	// in the same order, and multiplying by a weight of exactly 1 and
	// dividing by a variance of exactly 1 are exact operations in IEEE
	// double. Measured 0 on every entry of all six estimators. A band of
	// 1e-12 would hide a real structural difference behind rounding, so
	// the exact one is the useful assertion here.
	const statsduck::Vcov hc[] = {statsduck::Vcov::kHC0, statsduck::Vcov::kHC1,
	                              statsduck::Vcov::kHC2, statsduck::Vcov::kHC3};
	for (const statsduck::Vcov v : hc) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kGaussian;
		go.link = statsduck::Link::kIdentity;
		go.vcov = v;
		statsduck::LmOptions lo;
		lo.vcov = v;
		const statsduck::GlmResult g = statsduck::fit_glm(ds5_y, ds5_design(), go);
		const statsduck::LmResult o = statsduck::fit_lm(ds5_y, ds5_design(), lo);
		check(g.ok && o.ok, "both fitters succeed");
		double worst = 0.0;
		for (std::size_t e = 0; e < o.cov.size(); e++) {
			worst = std::max(worst, rel_dev(at(g.cov, e), at(o.cov, e)));
			check(close_rel(at(g.cov, e), at(o.cov, e), 0.0),
			      "the glm sandwich matches lm_core's at gaussian identity");
		}
		for (std::size_t j = 0; j < o.k; j++) {
			worst = std::max(worst, rel_dev(at(g.std_error, j), at(o.std_error, j)));
			check(close_rel(at(g.std_error, j), at(o.std_error, j), 0.0),
			      "and so does every standard error");
		}
		// n_clusters stays 0 under an HC estimator, so the SQL layer
		// writes NULL there.
		check(g.n_clusters == 0, "an HC fit reports no cluster count");
		std::printf("  %-5s DS5: worst gap against fit_lm %.3g\n", statsduck::vcov_name(v), worst);
	}

	// The same for the cluster-robust pair, which also has to count the
	// clusters. DS8 is a poisson design, fitted here as gaussian identity
	// on purpose: the point is the sandwich algebra, not the family.
	for (const statsduck::Vcov v : {statsduck::Vcov::kCR0, statsduck::Vcov::kCR1}) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kGaussian;
		go.link = statsduck::Link::kIdentity;
		go.vcov = v;
		statsduck::LmOptions lo;
		lo.vcov = v;
		const statsduck::GlmResult g =
		    statsduck::fit_glm(ds8_y, ds8_design(), go, nullptr, nullptr, &ds8_clusters);
		const statsduck::LmResult o = statsduck::fit_lm(ds8_y, ds8_design(), lo, &ds8_clusters);
		check(g.ok && o.ok, "both cluster fits succeed");
		check(g.n_clusters == o.n_clusters, "same cluster count");
		check(g.n_clusters == kDs8NumClusters, "which is the fixture's G");
		check(g.df_residual == o.df_residual, "df_residual stays n - rank under a cluster fit");
		double worst = 0.0;
		for (std::size_t e = 0; e < o.cov.size(); e++) {
			worst = std::max(worst, rel_dev(at(g.cov, e), at(o.cov, e)));
			check(close_rel(at(g.cov, e), at(o.cov, e), 0.0), "the cluster sandwich matches");
		}
		for (std::size_t j = 0; j < o.k; j++) {
			worst = std::max(worst, rel_dev(at(g.std_error, j), at(o.std_error, j)));
			check(close_rel(at(g.std_error, j), at(o.std_error, j), 0.0),
			      "and so does every standard error");
		}
		std::printf("  %-5s DS8: worst gap against fit_lm %.3g, G = %zu\n",
		            statsduck::vcov_name(v), worst, g.n_clusters);
	}

	// Changing the estimator never moves the point estimates. The band is
	// 0.0 because the loop never sees opts.vcov at all: the two fits run
	// the identical arithmetic up to the covariance assembly.
	statsduck::GlmOptions c1, c2;
	c1.family = c2.family = statsduck::Family::kBinomial;
	c1.link = c2.link = statsduck::Link::kLogit;
	c2.vcov = statsduck::Vcov::kHC1;
	const statsduck::GlmResult p1 = statsduck::fit_glm(ds1_y, ds1_design(), c1);
	const statsduck::GlmResult p2 = statsduck::fit_glm(ds1_y, ds1_design(), c2);
	check(p1.ok && p2.ok, "both DS1 fits succeed");
	check(p1.deviance == p2.deviance, "and reach the same deviance");
	for (std::size_t j = 0; j < p1.k; j++) {
		check(close_rel(at(p2.beta, j), at(p1.beta, j), 0.0),
		      "beta is bit-identical across covariance estimators");
	}
	// And the estimator does change the standard errors, so the test above
	// is not passing because vcov was ignored. The assertion is on x1,
	// index 1, whose classical standard error is 0.238466 against an HC1
	// 0.246672. That is a 3.33% separation, so the 1e-9 band is nowhere
	// near it. The intercept, printed below, separates by 2.68%.
	check(!close_rel(at(p2.std_error, 1), at(p1.std_error, 1), 1e-9),
	      "but the standard errors do differ");
	std::printf("  DS1 binomial intercept: classical SE %.6g, HC1 SE %.6g; x1: %.6g, %.6g\n",
	            at(p1.std_error, 0), at(p2.std_error, 0), at(p1.std_error, 1),
	            at(p2.std_error, 1));
}

// ── The oracle comparisons ───────────────────────────────────────────────

// ── The bands. The coefficient band is MEASURED, not assumed ─────────────
//
// Both this kernel and statsmodels stop when the deviance change falls
// below about 1e-8. Near a quadratic minimum a deviance change of eps
// corresponds to a coefficient change of order sqrt(eps), so the stopping
// rule on its own guarantees only about 1e-4. IRLS converges
// quadratically, so the last step overshoots far past the criterion and
// the real agreement is much tighter. The band below comes from running
// the loop over DS1, DS3, DS4, DS5, DS7 and DS8 and printing the worst
// relative deviation on every reported quantity. The suite still prints
// those numbers on every run, on the TOL_MEASURED line, so a change in
// them is visible rather than silent.
//
// Measured worst cases, 2026-10-21:
//   beta 2.56e-07 (DS4)   std_error 1.95e-06 (DS7)   z 1.95e-06 (DS7)
//   cov  6.00e-06 (DS1)   deviance and null_deviance 9.98e-15 (DS4)
//   dispersion, loglik, aic 2.94e-08 (DS4)   p-value 8.88e-07 absolute (DS7)
//
// DS8 joined the table when the robust covariance landed, and it moved
// none of those worsts: beta 6.86e-15, std_error and z 4.26e-08, cov
// 8.51e-08, deviance 6.46e-16, scale 1.34e-16, p 9.87e-09 absolute. Its
// standard-error gap is the same previous-iterate mechanism described
// below, one order smaller than DS1's.
//
// The two worst cases have different causes, and both were checked
// rather than assumed:
//
//   DS4's 2.56e-07 on beta is the stopping rule, exactly as described
//   above. Refitting the same data with tol = 1e-12 moves this kernel's
//   beta and closes the gap to 6.5e-09, and with no tolerance at all to
//   3.9e-10. So the oracle sits essentially at the MLE and the kernel
//   stops a step short of it, which is what tol = 1e-8 buys.
//
//   The standard-error gaps on DS7 (1.95e-06) and DS1 (4.78e-07) do NOT
//   shrink with a tighter tolerance. Both are identical at tol = 1e-8,
//   1e-12 and 0, while beta on those two fits agrees to 2.7e-12 and
//   3.2e-15. They are a property of the oracle's covariance, not of this
//   loop: statsmodels evaluates (XtWX) inverse at the working weights of
//   the PREVIOUS iterate, while this kernel recomputes them at the final
//   fitted values. Three measurements say the same thing. The fixture
//   header's own self-check 1 shows the effect inside statsmodels: its
//   DS1 and DS2 standard errors disagree by 4.2e-07 on data that is
//   identical in two shapes, while its coefficients there agree to
//   2.2e-12. This kernel's grouped and ungrouped standard errors on that
//   same pair agree to 4.3e-12, which the identity test below pins. And
//   this kernel's deficient DS7 fit reproduces its own reduced-design
//   fit exactly, to the last bit, on beta, every standard error and
//   every covariance entry.
static const double TOL_COEF = 2e-5;
// A covariance entry is a product of two standard-error-scale quantities,
// so a given relative error on a standard error doubles here. An
// off-diagonal entry can also be far smaller than the variances on either
// side of it, which turns the same absolute perturbation into a larger
// relative one: DS1's cov(x1, x2) is 1.44e-02 beside an intercept
// variance of 7.75e-01. Measured 6.00e-06, so this leaves eight times
// the measured worst case.
static const double TOL_COV = 5e-5;
// The deviance and null_deviance are at a minimum in beta, so a
// first-order coefficient difference moves them only at second order:
// DS4's beta differs by 2.56e-07 and its deviance by 9.98e-15. They get
// the tight band the design spec asks for.
static const double TOL_DEV = 1e-10;
// The dispersion is a Pearson sum rather than a minimized one, so it
// moves at FIRST order in the fitted values, and loglik and aic follow it
// for gamma. They inherit the coefficient band above. Measured 2.94e-08.
static const double TOL_SCALE = TOL_COEF;
// P-values are compared ABSOLUTELY. A far-tail p-value cannot be compared
// relatively: NormalCDF(8.97) rounds to exactly 1.0 in double precision,
// so this kernel reports p = 0 where DS3's oracle carries 2.9e-19, and
// the absolute gap of 2.9e-19 is the honest measure of that. The band is
// driven by the z agreement instead: |dp/dz| is at most 2*NormalPDF(0) =
// 0.8, so a z agreeing to 1.95e-06 relative at DS7's |z| = 1.25 moves p
// by about 2e-06. Measured 8.88e-07. The band still separates a normal
// reference from a t one, which differ by 2e-02 on DS4.
static const double TOL_P = 1e-5;

struct Fixture {
	const char *name;
	statsduck::Family family;
	statsduck::Link link;
	const std::vector<double> *y;
	Mat (*build_design)();
	const std::vector<double> *weights; // nullptr when every weight is 1
	const std::vector<double> *offset;  // nullptr when there is none
	std::size_t k, rank, df_residual;
	const double *beta;
	const double *std_error;
	const double *z_statistic;
	const double *p_expected;
	const double *cov;
	double deviance, null_deviance, dispersion, loglik, aic;
};

static const Fixture kFixtures[] = {
    {"DS1 binomial logit", statsduck::Family::kBinomial, statsduck::Link::kLogit, &ds1_y,
     ds1_design, nullptr, nullptr, kDs1K, kDs1Rank, kDs1DfResidual, kDs1Beta, kDs1StdError,
     kDs1ZStatistic, kDs1PExpected, kDs1Cov, kDs1Deviance, kDs1NullDeviance, kDs1Dispersion,
     kDs1Loglik, kDs1Aic},
    {"DS3 poisson log with an offset", statsduck::Family::kPoisson, statsduck::Link::kLog, &ds3_y,
     ds3_design, nullptr, &ds3_offset, kDs3K, kDs3Rank, kDs3DfResidual, kDs3Beta, kDs3StdError,
     kDs3ZStatistic, kDs3PExpected, kDs3Cov, kDs3Deviance, kDs3NullDeviance, kDs3Dispersion,
     kDs3Loglik, kDs3Aic},
    {"DS4 gamma log", statsduck::Family::kGamma, statsduck::Link::kLog, &ds4_y, ds4_design, nullptr,
     nullptr, kDs4K, kDs4Rank, kDs4DfResidual, kDs4Beta, kDs4StdError, kDs4ZStatistic,
     kDs4PExpected, kDs4Cov, kDs4Deviance, kDs4NullDeviance, kDs4Dispersion, kDs4Loglik, kDs4Aic},
    {"DS5 gaussian identity", statsduck::Family::kGaussian, statsduck::Link::kIdentity, &ds5_y,
     ds5_design, nullptr, nullptr, kDs5K, kDs5Rank, kDs5DfResidual, kDs5Beta, kDs5StdError,
     kDs5ZStatistic, kDs5PExpected, kDs5Cov, kDs5Deviance, kDs5NullDeviance, kDs5Dispersion,
     kDs5Loglik, kDs5Aic},
    // DS7 is rank deficient: its x3 is x1 + x2, so x3 is the aliased
    // column and its goldens are NaN there. The oracle is the reduced
    // design, which is this project's convention rather than
    // statsmodels' minimum-norm spread. It is in this table and not only
    // in the structural test below because that is what checks the
    // scatter back into k-space against real numbers.
    {"DS7 poisson rank deficient", statsduck::Family::kPoisson, statsduck::Link::kLog, &ds7_y,
     ds7_design, nullptr, nullptr, kDs7K, kDs7Rank, kDs7DfResidual, kDs7Beta, kDs7StdError,
     kDs7ZStatistic, kDs7PExpected, kDs7Cov, kDs7Deviance, kDs7NullDeviance, kDs7Dispersion,
     kDs7Loglik, kDs7Aic},
    // DS8's MODEL-BASED fit. The clustered estimators on the same design
    // are oracled separately below. This row is here so that a
    // disagreement there can be attributed to the sandwich rather than to
    // the fit underneath it: the goldens for both come from the same
    // statsmodels fit of the same data.
    {"DS8 poisson clustered, model-based", statsduck::Family::kPoisson, statsduck::Link::kLog,
     &ds8_y, ds8_design, nullptr, nullptr, kDs8K, kDs8Rank, kDs8DfResidual, kDs8Beta, kDs8StdError,
     kDs8ZStatistic, kDs8PExpected, kDs8Cov, kDs8Deviance, kDs8NullDeviance, kDs8Dispersion,
     kDs8Loglik, kDs8Aic},
};

static void test_oracles() {
	std::printf("statsmodels goldens: DS1, DS3, DS4, DS5, DS7, DS8\n");

	double worst_beta = 0.0, worst_se = 0.0, worst_z = 0.0, worst_cov = 0.0;
	double worst_dev = 0.0, worst_scale = 0.0, worst_p = 0.0;
	for (const Fixture &f : kFixtures) {
		statsduck::GlmOptions go;
		go.family = f.family;
		go.link = f.link;
		const statsduck::GlmResult g =
		    statsduck::fit_glm(*f.y, f.build_design(), go, f.weights, f.offset);
		check(g.ok, "the oracle fixture fits");
		check(g.converged, "and converges");
		check(g.k == f.k, "same k as the oracle");
		check(g.rank == f.rank, "same rank");
		check(g.df_residual == f.df_residual, "same residual df");
		check(g.terms.size() == f.k, "one term name per column");

		double fb = 0.0, fs = 0.0, fz = 0.0, fc = 0.0, fp = 0.0;
		for (std::size_t j = 0; j < f.k; j++) {
			// A NaN golden is the aliased column of a rank-deficient
			// design. It has to stay NaN, and there is nothing to compare.
			if (std::isnan(f.beta[j])) {
				check(std::isnan(at(g.beta, j)), "an aliased coefficient is NaN");
				check(std::isnan(at(g.std_error, j)), "so is its standard error");
				check(std::isnan(at(g.z_statistic, j)), "so is its z statistic");
				check(std::isnan(at(g.p_value, j)), "so is its p-value");
				continue;
			}
			fb = std::max(fb, rel_dev(at(g.beta, j), f.beta[j]));
			fs = std::max(fs, rel_dev(at(g.std_error, j), f.std_error[j]));
			fz = std::max(fz, rel_dev(at(g.z_statistic, j), f.z_statistic[j]));
			fp = std::max(fp, std::fabs(at(g.p_value, j) - f.p_expected[j]));
			check(close_rel(at(g.beta, j), f.beta[j], TOL_COEF), "coefficient matches the oracle");
			check(close_rel(at(g.std_error, j), f.std_error[j], TOL_COEF),
			      "standard error matches the oracle");
			check(close_rel(at(g.z_statistic, j), f.z_statistic[j], TOL_COEF),
			      "z statistic matches the oracle");
			// p_expected, never p_oracle. The two differ for DS4 and DS5
			// because this project uses a normal reference for every
			// family while statsmodels switches to a t reference when the
			// dispersion is estimated.
			check(std::fabs(at(g.p_value, j) - f.p_expected[j]) <= TOL_P,
			      "p-value matches the normal-referenced expectation");
		}
		for (std::size_t e = 0; e < f.k * f.k; e++) {
			if (std::isnan(f.cov[e])) {
				check(std::isnan(at(g.cov, e)), "an aliased covariance entry is NaN");
				continue;
			}
			fc = std::max(fc, rel_dev(at(g.cov, e), f.cov[e]));
			check(close_rel(at(g.cov, e), f.cov[e], TOL_COV), "covariance entry matches the oracle");
		}
		check(close_rel(g.deviance, f.deviance, TOL_DEV), "deviance matches the oracle");
		check(close_rel(g.null_deviance, f.null_deviance, TOL_DEV),
		      "null_deviance matches the oracle");
		check(close_rel(g.dispersion, f.dispersion, TOL_SCALE), "dispersion matches the oracle");
		check(close_rel(g.loglik, f.loglik, TOL_SCALE), "loglik matches the oracle");
		check(close_rel(g.aic, f.aic, TOL_SCALE), "aic matches the oracle");

		const double fd = std::max(rel_dev(g.deviance, f.deviance),
		                           rel_dev(g.null_deviance, f.null_deviance));
		const double fsc = std::max(rel_dev(g.dispersion, f.dispersion),
		                            std::max(rel_dev(g.loglik, f.loglik), rel_dev(g.aic, f.aic)));
		std::printf("  %-32s beta %.3g se %.3g z %.3g cov %.3g dev %.3g scale %.3g p %.3g (%d it)\n",
		            f.name, fb, fs, fz, fc, fd, fsc, fp, g.iterations);
		worst_beta = std::max(worst_beta, fb);
		worst_se = std::max(worst_se, fs);
		worst_z = std::max(worst_z, fz);
		worst_cov = std::max(worst_cov, fc);
		worst_dev = std::max(worst_dev, fd);
		worst_scale = std::max(worst_scale, fsc);
		worst_p = std::max(worst_p, fp);
	}
	std::printf("  TOL_MEASURED worst: beta %.3g  se %.3g  z %.3g  cov %.3g  dev %.3g  "
	            "scale %.3g  p(abs) %.3g\n",
	            worst_beta, worst_se, worst_z, worst_cov, worst_dev, worst_scale, worst_p);
}

// ── The robust oracle: a family where link_deriv and variance are not 1 ──
//
// The gaussian equivalence above proves the sandwich algebra. It cannot
// prove the score formula, because at gaussian identity link_deriv and
// variance are both exactly 1 and drop out. DS8 is poisson with a log
// link, so link_deriv(eta) is e^eta and variance(mu) is mu, and neither
// is 1 on any row. Its CR0 and CR1 goldens are what pin the score
// formula.
//
// WHAT IS NOT HERE, and why. The fixture header carries no HC goldens
// for any design: gen_glm_fit_fixtures.py generates statsmodels'
// cov_type='cluster' for DS8 and nothing else robust. So an HC estimator
// on a non-gaussian family has no golden to compare against. Rather than
// leave that path with only the gaussian check, the HC meat is tied to
// the CR meat here by an exact identity: with one cluster per row, CR0's
// per-cluster score sum is a single score, so CR0 IS HC0. That makes
// DS8's oracled CR0 cover the HC accumulation too, on the same
// non-gaussian family.
//
// The bands below are MEASURED, and they come out far tighter than the
// model-based bands above. Measured 2026-10-21 on DS8:
//   CR0  se 1.70e-15  cov 3.97e-15  z 7.97e-15  p 3.44e-15 absolute
//   CR1  se 1.94e-15  cov 3.92e-15  z 8.13e-15  p 3.44e-15 absolute
// against the same fit's model-based cov at 8.51e-08. That difference is
// informative rather than odd. Task 4 measured that the oracle's
// model-based covariance is evaluated at the previous iterate's working
// weights. These numbers say that stale bread does not reach the oracle's
// cluster path. What is left is the coefficient agreement, 6.86e-15,
// carried through a smooth function of beta.
//
// So the band is 1e-12, which is 250 times the worst measured gap. That
// is deliberately loose for a measurement at rounding level. The point of
// the band is to catch a structural error, and the smallest structural
// error here is far larger. Both were measured, by building the kernel
// with the error in it:
//   dropping link_deriv from the score moves DS8's CR0 covariance 0.79
//     relative, against a golden it otherwise matches at 4.0e-15
//   dropping CR1's finite-sample factor moves it 0.23 relative, which is
//     the factor's own size
// 1e-12 still leaves room for another platform's exp and log to land a
// few bits elsewhere.
//
// The p-value is compared ABSOLUTELY against the same constant. Both of
// DS8's robust p-values are near 0.3, so 1e-12 absolute is about 3e-12
// relative there, and the measured gap is 3.44e-15.
static const double TOL_ROBUST = 1e-12;

struct RobustFixture {
	const char *name;
	statsduck::Vcov vcov;
	const double *std_error;
	const double *z_statistic;
	const double *p_expected;
	const double *cov;
};

static const RobustFixture kRobustFixtures[] = {
    {"DS8 CR0", statsduck::Vcov::kCR0, kDs8Cr0StdError, kDs8Cr0ZStatistic, kDs8Cr0PExpected,
     kDs8Cr0Cov},
    {"DS8 CR1", statsduck::Vcov::kCR1, kDs8Cr1StdError, kDs8Cr1ZStatistic, kDs8Cr1PExpected,
     kDs8Cr1Cov},
};

static void test_robust_oracles() {
	std::printf("statsmodels robust goldens: DS8 poisson, CR0 and CR1\n");

	for (const RobustFixture &f : kRobustFixtures) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kPoisson;
		go.link = statsduck::Link::kLog;
		go.vcov = f.vcov;
		const statsduck::GlmResult g =
		    statsduck::fit_glm(ds8_y, ds8_design(), go, nullptr, nullptr, &ds8_clusters);
		check(g.ok && g.converged, "the clustered fixture fits and converges");
		check(g.n_clusters == kDs8NumClusters, "and reports the fixture's cluster count");
		// df_residual is a property of the fit, so it does not move with
		// the estimator even though the cluster-robust reference does.
		check(g.df_residual == kDs8DfResidual, "df_residual is still n - rank");

		double fs = 0.0, fz = 0.0, fc = 0.0, fp = 0.0, fb = 0.0;
		for (std::size_t j = 0; j < kDs8K; j++) {
			// beta against the MODEL-BASED golden. The estimator must not
			// move the point estimate, and the oracle's own self-check
			// pins its clustered beta equal to its model-based beta. The
			// band here is TOL_COEF rather than TOL_ROBUST, because this
			// comparison inherits the stopping-rule gap the table above
			// measures rather than the sandwich's own.
			fb = std::max(fb, rel_dev(at(g.beta, j), kDs8Beta[j]));
			fs = std::max(fs, rel_dev(at(g.std_error, j), f.std_error[j]));
			fz = std::max(fz, rel_dev(at(g.z_statistic, j), f.z_statistic[j]));
			fp = std::max(fp, std::fabs(at(g.p_value, j) - f.p_expected[j]));
			check(close_rel(at(g.beta, j), kDs8Beta[j], TOL_COEF),
			      "the clustered fit's coefficient is the model-based one");
			check(close_rel(at(g.std_error, j), f.std_error[j], TOL_ROBUST),
			      "robust standard error matches the oracle");
			check(close_rel(at(g.z_statistic, j), f.z_statistic[j], TOL_ROBUST),
			      "robust z statistic matches the oracle");
			// p_expected, never p_oracle. For poisson the two are the same
			// column, since statsmodels uses a normal reference there too.
			check(std::fabs(at(g.p_value, j) - f.p_expected[j]) <= TOL_ROBUST,
			      "robust p-value matches the normal-referenced expectation");
		}
		for (std::size_t e = 0; e < kDs8K * kDs8K; e++) {
			fc = std::max(fc, rel_dev(at(g.cov, e), f.cov[e]));
			check(close_rel(at(g.cov, e), f.cov[e], TOL_ROBUST),
			      "robust covariance entry matches the oracle");
		}
		std::printf("  %-8s beta %.3g se %.3g z %.3g cov %.3g p(abs) %.3g (%d it)\n", f.name, fb,
		            fs, fz, fc, fp, g.iterations);
	}

	// CR1 is CR0 times [G/(G-1)]*[(N-1)/(N-rank)], and the fixture carries
	// that factor as its own constant. Checking the RATIO separates a wrong
	// factor from a wrong meat: both estimators would have to be wrong by
	// the same multiple to pass this and the two oracle rows above.
	//
	// The band is 1e-14, measured 1.70e-16. The ratio is one division of
	// two numbers that differ by an exact constant factor, so rounding is
	// all that can separate them, and 1e-14 is 59 times that.
	statsduck::GlmOptions c0, c1;
	c0.family = c1.family = statsduck::Family::kPoisson;
	c0.link = c1.link = statsduck::Link::kLog;
	c0.vcov = statsduck::Vcov::kCR0;
	c1.vcov = statsduck::Vcov::kCR1;
	const statsduck::GlmResult g0 =
	    statsduck::fit_glm(ds8_y, ds8_design(), c0, nullptr, nullptr, &ds8_clusters);
	const statsduck::GlmResult g1 =
	    statsduck::fit_glm(ds8_y, ds8_design(), c1, nullptr, nullptr, &ds8_clusters);
	double worst_ratio = 0.0;
	for (std::size_t e = 0; e < kDs8K * kDs8K; e++) {
		const double ratio = at(g1.cov, e) / at(g0.cov, e);
		worst_ratio = std::max(worst_ratio, rel_dev(ratio, kDs8Cr1Factor));
		check(close_rel(ratio, kDs8Cr1Factor, 1e-14),
		      "CR1 is CR0 times the fixture's finite-sample factor");
	}
	std::printf("  CR1/CR0 against the fixture's factor %.12g: worst %.3g\n", kDs8Cr1Factor,
	            worst_ratio);

	// HC0 IS CR0 with one cluster per row. A cluster of one row has a
	// score sum equal to that row's own score. So the two meats are the
	// same sum of outer products. The CR0 above is oracled, so this gives
	// the HC accumulation an oracled check on a non-gaussian family.
	//
	// The band is 1e-13, measured 2.17e-16. The two paths compute the same
	// sum in a different order. HC0 squares the score first and then
	// scales the outer product. CR0 scales the design row first and then
	// multiplies two scaled entries. Only rounding separates them, so the
	// band is 460 times the measurement.
	std::vector<int> one_per_row(kDs8N);
	for (std::size_t i = 0; i < kDs8N; i++) {
		one_per_row[i] = static_cast<int>(i);
	}
	statsduck::GlmOptions h0, s0;
	h0.family = s0.family = statsduck::Family::kPoisson;
	h0.link = s0.link = statsduck::Link::kLog;
	h0.vcov = statsduck::Vcov::kHC0;
	s0.vcov = statsduck::Vcov::kCR0;
	const statsduck::GlmResult hc0 = statsduck::fit_glm(ds8_y, ds8_design(), h0);
	const statsduck::GlmResult cr0_singletons =
	    statsduck::fit_glm(ds8_y, ds8_design(), s0, nullptr, nullptr, &one_per_row);
	check(hc0.ok && cr0_singletons.ok, "both the HC0 and the singleton-cluster fit succeed");
	check(hc0.n_clusters == 0 && cr0_singletons.n_clusters == kDs8N,
	      "and only the clustered one counts clusters");
	double worst_hc = 0.0;
	for (std::size_t e = 0; e < kDs8K * kDs8K; e++) {
		worst_hc = std::max(worst_hc, rel_dev(at(hc0.cov, e), at(cr0_singletons.cov, e)));
		check(close_rel(at(hc0.cov, e), at(cr0_singletons.cov, e), 1e-13),
		      "HC0 equals CR0 with one cluster per row");
	}
	std::printf("  HC0 against singleton-cluster CR0 on DS8 poisson: worst %.3g\n", worst_hc);
}

// ── DS1 binomial, HC0 through HC3, against R 4.6.1 ──────────────────────
//
// THE HC ORACLE. statsmodels gives this project no HC golden for any
// design: gen_glm_fit_fixtures.py generates cov_type='cluster' for DS8 and
// nothing else robust. It also cannot give one for HC2 or HC3 on a GLM. In
// 0.15.0, cov_HC0 through cov_HC3 exist only on OLSResults
// (statsmodels/regression/linear_model.py), GLMResults defines none of
// them, and statsmodels/base/covtype.py dispatches with
// getattr(self, "cov_" + cov_type.upper(), None) and falls back to one
// cov_white_simple call. So GLM(...).fit(cov_type=X) returns identical bse
// for X = HC0, HC1, HC2 and HC3, labelled with whichever name was asked
// for. Measured on this very design: all four come back as
// 8.69158508520134210e-01 on the intercept, against R's four distinct
// values below. R is the only available oracle for a GLM's HC2 and HC3,
// which is why R is the oracle here. The mechanism and the OLS control
// are in notes/engineering/2026-10-statsmodels-hc-for-a-glm.md.
//
// The values below come from R's OWN machinery, not from a second copy of
// this kernel's algebra. fit$weights are R's working weights and
// hatvalues() is R's leverage, taken from the QR decomposition of
// sqrt(w)X inside lm.influence. Exact program, reproducible:
//
//   fit <- glm(y ~ x1 + x2, family = binomial)
//   X <- model.matrix(fit); n <- nrow(X); k <- ncol(X)
//   w <- fit$weights
//   u <- w * residuals(fit, type = "working")   # (y-mu)*(dmu/deta)/V
//   h <- hatvalues(fit)
//   Ai <- solve(t(X) %*% (w * X))
//   sw <- function(s) Ai %*% (t(X) %*% (s * X)) %*% Ai
//   hc0 <- sw(u^2); hc1 <- hc0 * n/(n-k)
//   hc2 <- sw(u^2/(1-h)); hc3 <- sw(u^2/(1-h)^2)
//
// R's own sum(hatvalues(fit)) is exactly 3, which is the rank. That is the
// trace identity on the weighted design, and it confirms h here is the
// WEIGHTED leverage rather than the unweighted one.
//
// MEASURED worst relative gap, this kernel against those values:
//   standard error  HC0 2.56e-09  HC1 2.56e-09  HC2 1.32e-08  HC3 2.82e-08
//   covariance      HC0 1.61e-08  HC1 1.61e-08  HC2 5.02e-07  HC3 1.12e-06
//
// THE BAND IS WIDE FOR A KNOWN REASON, not because the formulas differ. R
// evaluates its working weights and its leverage at the mu of the iterate
// BEFORE the last one, which is the same mechanism Task 4 measured in
// statsmodels. Recomputing the weights in R at fitted(fit), and taking the
// leverage from R's own hat(qr(sqrt(w) * X)), closes the gap to 4e-15 on
// every one of the twelve standard errors. So the residual difference is
// entirely the stale iterate and not the algebra.
//
// A covariance entry is looser than its standard errors for the reason
// Task 4 recorded at TOL_COV: an off-diagonal entry can be far smaller
// than the variances on either side of it, which turns the same absolute
// perturbation into a larger relative one.
static const double TOL_R_SE = 1e-7;  // 3.5 times the measured worst
static const double TOL_R_COV = 5e-6; // 4.5 times the measured worst

// Row order matches kRHcFixtures below: HC0, HC1, HC2, HC3.
static const double kDs1RHcStdError[4][3] = {
    {8.69158507121863155e-01, 2.37241461900041500e-01, 6.62577562961442212e-01},
    {9.03707983511240376e-01, 2.46671926216198295e-01, 6.88915345632949649e-01},
    {9.07611749680332869e-01, 2.48781719672927865e-01, 6.88591538762508559e-01},
    {9.47914435957977197e-01, 2.60915171941325130e-01, 7.15774073368389385e-01}};
static const double kDs1RHcCov[4][9] = {
    {7.55436510502305869e-01, -1.73805930609675224e-01, -2.49236894727575126e-01,
     -1.73805930609675363e-01, 5.62835112444688493e-02, 9.91079375253763224e-03,
     -2.49236894727574987e-01, 9.91079375253763398e-03, 4.39009026939923941e-01},
    {8.16688119461952300e-01, -1.87898303361811042e-01, -2.69445291597378511e-01,
     -1.87898303361811181e-01, 6.08470391832095711e-02, 1.07143716243650079e-02,
     -2.69445291597378345e-01, 1.07143716243650097e-02, 4.74604353448566441e-01},
    {8.23759088157795305e-01, -1.90663463384605353e-01, -2.66997741153744139e-01,
     -1.90663463384605186e-01, 6.18923440434192632e-02, 9.97286250869467220e-03,
     -2.66997741153744250e-01, 9.97286250869474158e-03, 4.74158307255319378e-01},
    {8.98541777897530092e-01, -2.09205611728346313e-01, -2.86018847338184778e-01,
     -2.09205611728346563e-01, 6.80767269491712596e-02, 9.95086176166554669e-03,
     -2.86018847338184667e-01, 9.95086176166542005e-03, 5.12332524106376419e-01}};

static void test_binomial_hc_against_r() {
	std::printf("R 4.6.1 goldens: DS1 binomial logit, HC0 through HC3\n");

	const statsduck::Vcov estimators[4] = {statsduck::Vcov::kHC0, statsduck::Vcov::kHC1,
	                                       statsduck::Vcov::kHC2, statsduck::Vcov::kHC3};
	double worst_se = 0.0, worst_cov = 0.0;
	for (int i = 0; i < 4; i++) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kBinomial;
		go.link = statsduck::Link::kLogit;
		go.vcov = estimators[i];
		const statsduck::GlmResult g = statsduck::fit_glm(ds1_y, ds1_design(), go);
		check(g.ok && g.converged, "the binomial HC fit converges");
		double fs = 0.0, fc = 0.0;
		for (std::size_t j = 0; j < kDs1K; j++) {
			fs = std::max(fs, rel_dev(at(g.std_error, j), kDs1RHcStdError[i][j]));
			check(close_rel(at(g.std_error, j), kDs1RHcStdError[i][j], TOL_R_SE),
			      "the standard error matches R");
		}
		for (std::size_t e = 0; e < kDs1K * kDs1K; e++) {
			fc = std::max(fc, rel_dev(at(g.cov, e), kDs1RHcCov[i][e]));
			check(close_rel(at(g.cov, e), kDs1RHcCov[i][e], TOL_R_COV),
			      "and so does the covariance entry");
		}
		std::printf("  %-5s se %.3g cov %.3g\n", statsduck::vcov_name(estimators[i]), fs, fc);
		worst_se = std::max(worst_se, fs);
		worst_cov = std::max(worst_cov, fc);
	}
	std::printf("  R_MEASURED worst: se %.3g  cov %.3g\n", worst_se, worst_cov);
}

// ── A WEIGHTED gaussian fit's loglik and aic, against R 4.6.1 ───────────
//
// THE GAP THIS CLOSES. Every gaussian fixture in this suite carries unit
// weights, and the gaussian loglik's prior-weight term Σ ln(w)/2 is
// exactly 0 there. So the term was simply absent from glm_core.cpp and
// nothing failed. The #22 whole-branch review measured a 12-row w = 3 fit
// through SQL and found loglik off by Σ ln(w)/2 and aic off by Σ ln(w),
// while beta, the standard errors, the deviance and the dispersion all
// matched R to 12 digits. This test is the fixture that was missing.
//
// DS5's y, x1 and x2 with prior weights 0.5, 1.0, 1.5, ... 6.0. The
// weights are deliberately NOT constant and NOT all above 1: half of them
// have a negative log, so the term's sign is exercised in both directions
// and a constant-weight shortcut would not reproduce it.
//
// Exact program, reproducible:
//
//   y  <- c(9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55)
//   x1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25)
//   x2 <- c(5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4)
//   w  <- seq(0.5, 6, by = 0.5)
//   fit <- glm(y ~ x1 + x2, family = gaussian, weights = w)
//   coef(fit); sqrt(diag(vcov(fit))); fit$deviance
//   summary(fit)$dispersion; logLik(fit); AIC(fit)
//
// R's AIC for a gaussian glm is −2·logLik + 2·(rank + 1), the same d = 1
// convention this project uses, so the aic below is R's number directly.
// The deviation of 2 recorded at the fixture header is against
// statsmodels' .aic, not against R's AIC.
//
// FOR THE RECORD, measured by suppressing the term and refitting this
// same data: loglik −3.47270392197173408e+01 and aic
// 7.74540784394346815e+01. That is Σ ln(w)/2 = 5.83472416447127173 lower
// on the loglik and Σ ln(w) higher on the aic, which is the whole error.
static const double kRWgBeta[3] = {1.16926234165278515e+01, 1.96279043475914361e+00,
                                   -1.00771938143776585e+00};
static const double kRWgStdError[3] = {2.31498611912133123e+00, 1.21091449186467831e-01,
                                       3.22633123616013207e-01};
static const double kRWgDeviance = 2.29262916812073911e+02;
static const double kRWgNullDeviance = 7.59658974358974319e+03;
static const double kRWgDispersion = 2.54736574235637683e+01;
static const double kRWgLoglik = -2.88923150552460584e+01;
static const double kRWgAic = 6.57846301104921167e+01;

static void test_weighted_gaussian_loglik_against_r() {
	std::printf("R 4.6.1 goldens: a WEIGHTED gaussian fit's loglik and aic\n");

	std::vector<double> w(kDs5N);
	for (std::size_t i = 0; i < kDs5N; i++) {
		w[i] = 0.5 * static_cast<double>(i + 1); // 0.5, 1.0, ... 6.0
	}
	statsduck::GlmOptions go;
	go.family = statsduck::Family::kGaussian;
	go.link = statsduck::canonical_link(go.family);
	const statsduck::GlmResult g = statsduck::fit_glm(ds5_y, ds5_design(), go, &w);
	check(g.ok && g.converged, "the weighted gaussian fit converges");
	check(g.rank == 3 && g.df_residual == 9, "rank 3 on 12 rows, as R reports");

	// The band is 1e-12 on every quantity. A gaussian identity fit is one
	// weighted solve in both this kernel and R, so there is no iteration
	// difference to absorb: MEASURED worsts are 1.74e-15 on the fit and
	// 4.32e-16 on the loglik and aic. The band is 1e-12 rather than
	// tighter so that a future change in accumulation order is a
	// reviewable difference rather than a failure here, which is the same
	// reason test_gaussian_equals_lm_fit gives for its own 1e-12. Both
	// numbers reprint on every run.
	double worst_fit = 0.0, worst_ll = 0.0;
	for (std::size_t j = 0; j < 3; j++) {
		worst_fit = std::max(worst_fit, rel_dev(at(g.beta, j), kRWgBeta[j]));
		worst_fit = std::max(worst_fit, rel_dev(at(g.std_error, j), kRWgStdError[j]));
		check(close_rel(at(g.beta, j), kRWgBeta[j], 1e-12), "the coefficient matches R");
		check(close_rel(at(g.std_error, j), kRWgStdError[j], 1e-12),
		      "and so does the standard error");
	}
	worst_fit = std::max(worst_fit, rel_dev(g.deviance, kRWgDeviance));
	worst_fit = std::max(worst_fit, rel_dev(g.null_deviance, kRWgNullDeviance));
	worst_fit = std::max(worst_fit, rel_dev(g.dispersion, kRWgDispersion));
	check(close_rel(g.deviance, kRWgDeviance, 1e-12), "the deviance matches R");
	check(close_rel(g.null_deviance, kRWgNullDeviance, 1e-12), "and the null deviance");
	check(close_rel(g.dispersion, kRWgDispersion, 1e-12), "and the dispersion");

	// The two this test exists for.
	worst_ll = std::max(rel_dev(g.loglik, kRWgLoglik), rel_dev(g.aic, kRWgAic));
	check(close_rel(g.loglik, kRWgLoglik, 1e-12), "the WEIGHTED loglik matches R's logLik");
	check(close_rel(g.aic, kRWgAic, 1e-12), "and the aic matches R's AIC");
	std::printf("  R_MEASURED worst: fit %.3g  loglik and aic %.3g\n", worst_fit, worst_ll);
}

// ── Gamma's canonical inverse link, and poisson with a sqrt link ────────
//
// THE GAP THIS CLOSES. Five of the seven links were FITTED in this suite:
// identity, log, logit, probit and cloglog. Inverse and sqrt were not.
// Inverse is gamma's canonical link, so `glm_fit(y, x, 'gamma')` with no
// options ran a path no test reached past test_glm_core's scalar round
// trips, and the SQL test's only gamma/inverse assertion was that the
// echoed link name reads `inverse`. Both are pinned here.
//
// The link is taken from canonical_link rather than written out, so this
// also covers the dispatch the SQL layer uses when no link is given.
//
// Exact program, reproducible. ds4_y, ds4_x1, ds3_y and ds3_x1 are the
// fixture header's own kDs4Y, kDs4X1, kDs3Y and kDs3X1. DS3's offset is
// left out: it is log(exposure), which belongs to a log link, not a sqrt
// one.
//
//   gi <- glm(ds4_y ~ ds4_x1, family = Gamma(link = "inverse"))
//   coef(gi); sqrt(diag(vcov(gi))); gi$deviance; summary(gi)$dispersion
//   ps <- glm(ds3_y ~ ds3_x1, family = poisson(link = "sqrt"))
//   coef(ps); sqrt(diag(vcov(ps))); ps$deviance
//
// R converges in 5 iterations on the gamma fit and 4 on the poisson one,
// and reports converged = TRUE for both.
static const double kRGiBeta[2] = {7.47491429760872572e-01, -2.15607528609750299e-01};
static const double kRGiStdError[2] = {1.28482737851597334e-01, 8.89993177989926582e-02};
static const double kRGiDeviance = 2.45752408342686168e+00;
static const double kRGiDispersion = 1.75336550725355450e-01;
static const double kRPsBeta[2] = {4.60327965048774068e+00, 1.36695326105509052e+00};
static const double kRPsStdError[2] = {2.45798020062543698e-01, 1.49403576166799196e-01};
static const double kRPsDeviance = 1.42809802502182748e+02;

static void test_canonical_gamma_and_sqrt_links_against_r() {
	std::printf("R 4.6.1 goldens: gamma/inverse (canonical) and poisson/sqrt\n");

	statsduck::GlmOptions gi;
	gi.family = statsduck::Family::kGamma;
	gi.link = statsduck::canonical_link(gi.family);
	check(gi.link == statsduck::Link::kInverse, "gamma's canonical link is inverse");
	const statsduck::GlmResult g = statsduck::fit_glm(ds4_y, ds4_design(), gi);
	check(g.ok && g.converged, "the gamma/inverse fit converges");

	// The bands are the TOL_COEF family above, reused because both gaps
	// have the cause that block describes: a non-linear fit stops a step
	// short of the MLE, and this kernel and R stop at different points.
	// MEASURED worsts, reprinted on every run: gamma/inverse 1.47e-07,
	// which is the same order as DS4's 2.56e-07 against statsmodels on
	// the log link, and poisson/sqrt 5.79e-16.
	double worst_g = 0.0;
	for (std::size_t j = 0; j < kDs4K; j++) {
		worst_g = std::max(worst_g, rel_dev(at(g.beta, j), kRGiBeta[j]));
		worst_g = std::max(worst_g, rel_dev(at(g.std_error, j), kRGiStdError[j]));
		check(close_rel(at(g.beta, j), kRGiBeta[j], TOL_COEF), "the gamma coefficient matches R");
		check(close_rel(at(g.std_error, j), kRGiStdError[j], TOL_COEF),
		      "and so does its standard error");
	}
	check(close_rel(g.deviance, kRGiDeviance, TOL_DEV), "the gamma/inverse deviance matches R");
	check(close_rel(g.dispersion, kRGiDispersion, TOL_SCALE), "and its dispersion");

	statsduck::GlmOptions ps;
	ps.family = statsduck::Family::kPoisson;
	ps.link = statsduck::Link::kSqrt;
	const statsduck::GlmResult p = statsduck::fit_glm(ds3_y, ds3_design(), ps);
	check(p.ok && p.converged, "the poisson/sqrt fit converges");
	double worst_p = 0.0;
	for (std::size_t j = 0; j < kDs3K; j++) {
		worst_p = std::max(worst_p, rel_dev(at(p.beta, j), kRPsBeta[j]));
		worst_p = std::max(worst_p, rel_dev(at(p.std_error, j), kRPsStdError[j]));
		check(close_rel(at(p.beta, j), kRPsBeta[j], TOL_COEF), "the poisson coefficient matches R");
		check(close_rel(at(p.std_error, j), kRPsStdError[j], TOL_COEF),
		      "and so does its standard error");
	}
	check(close_rel(p.deviance, kRPsDeviance, TOL_DEV), "the poisson/sqrt deviance matches R");
	std::printf("  R_MEASURED worst: gamma/inverse %.3g  poisson/sqrt %.3g\n", worst_g, worst_p);
}

// ── The hat diagonal uses the WORKING weight, not the prior weight ──────
static void test_weighted_hat_diagonal() {
	std::printf("the HC2 and HC3 hat diagonal uses the working weight\n");

	// This test exists because of a hole a reviewer found by mutation.
	// Replacing `h *= final_w[r]` with `h *= w[r]`, the prior weight
	// instead of the working weight, passed the whole suite. It moves
	// DS1's HC2 intercept standard error from 0.9076 to 1.1317, a 24.7%
	// error, and HC3's from 0.9479 to 1.4885, a 57% error.
	//
	// Neither of the two tests that look weighted can see it.
	// test_weighted_sandwich is gaussian identity, where the working
	// weight EQUALS the prior weight by construction, which is the whole
	// point of the identity it uses. So it pins that A weight appears in
	// the leverage, not that the WORKING weight does. The gaussian
	// equivalence has every weight at 1, where the two are both 1.
	//
	// THE IDENTITY. For a full-rank weighted projection the hat diagonals
	// sum to the rank:
	//   sum_i h_i = trace(W^(1/2) X A^-1 Xt W^(1/2)) = trace(A^-1 A) = rank
	// with A = Xt W X. h is internal and unreported, so the sum cannot be
	// asserted directly through the public interface. This design makes
	// the identity observable instead, by making every h_i equal.
	//
	// An intercept plus one 0/1 dummy, with EQUAL group sizes, is
	// saturated by group: the fit reproduces each group's mean, so eta is
	// constant within a group and the working weight is too. The weighted
	// hat diagonal of a saturated group design is 1/n_g for every row,
	// whatever the weights are, and with equal groups that is rank/n for
	// every row. Then the trace identity is the statement that every h_i
	// is rank/n, and HC2 and HC3 become exactly
	//   HC2 = HC0 / (1 - rank/n)        HC3 = HC0 / (1 - rank/n)^2
	// entrywise. Confirmed independently in R 4.6.1: for the data below
	// hatvalues() returns 0.2 on every row to 1.1e-16 and sums to 2.
	//
	// WHY IT SEPARATES. The fit is poisson with a log link, so the working
	// weight is mu: 2 in the first group and 6 in the second, never the
	// prior weight of 1. Under the mutation h becomes x' A^-1 x with A
	// still built from the working weights, which is h/mu_g, so 0.1 and
	// 0.0333 rather than 0.2. The identity then fails on both estimators.
	// MEASURED under the mutation: HC2 misses the identity by 0.133
	// relative and HC3 by 0.2475, against the 1e-14 bands below.
	const std::vector<double> y = {1, 2, 3, 2, 2, 5, 7, 6, 8, 4};
	const std::vector<double> dummy = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
	Mat X(10, 1);
	for (std::size_t i = 0; i < 10; i++) {
		X(i, 0) = dummy[i];
	}
	statsduck::GlmOptions base;
	base.family = statsduck::Family::kPoisson;
	base.link = statsduck::Link::kLog;

	statsduck::GlmOptions o0 = base, o2 = base, o3 = base;
	o0.vcov = statsduck::Vcov::kHC0;
	o2.vcov = statsduck::Vcov::kHC2;
	o3.vcov = statsduck::Vcov::kHC3;
	const statsduck::GlmResult g0 = statsduck::fit_glm(y, X, o0);
	const statsduck::GlmResult g2 = statsduck::fit_glm(y, X, o2);
	const statsduck::GlmResult g3 = statsduck::fit_glm(y, X, o3);
	check(g0.ok && g2.ok && g3.ok, "the three saturated group fits succeed");
	check(g0.rank == 2 && g0.n == 10, "with rank 2 over ten rows");

	// The band is 1e-14. Measured 1.39e-16 on HC2 and 4.44e-16 on HC3.
	// The identity is exact, so only rounding separates the two sides, and
	// 1e-14 is 22 times the worst of that. It is 12 orders inside the
	// 0.133 the mutation moves.
	const double h_bar = static_cast<double>(g0.rank) / static_cast<double>(g0.n);
	const double one_minus = 1.0 - h_bar;
	double w2 = 0.0, w3 = 0.0;
	for (std::size_t e = 0; e < g0.cov.size(); e++) {
		const double e2 = at(g0.cov, e) / one_minus;
		const double e3 = at(g0.cov, e) / (one_minus * one_minus);
		w2 = std::max(w2, rel_dev(at(g2.cov, e), e2));
		w3 = std::max(w3, rel_dev(at(g3.cov, e), e3));
		check(close_rel(at(g2.cov, e), e2, 1e-14), "HC2 is HC0 over (1 - rank/n)");
		check(close_rel(at(g3.cov, e), e3, 1e-14), "HC3 is HC0 over (1 - rank/n) squared");
	}
	// The fitted means are what make the working weight differ from the
	// prior weight. Without this the test could pass on a design where
	// they happen to coincide, which would make it vacuous again.
	const double mu_a = statsduck::link_inv(statsduck::Link::kLog, at(g0.beta, 0));
	const double mu_b =
	    statsduck::link_inv(statsduck::Link::kLog, at(g0.beta, 0) + at(g0.beta, 1));
	check(close_rel(mu_a, 2.0, 1e-9) && close_rel(mu_b, 6.0, 1e-9),
	      "the two group means are 2 and 6, so the working weights are 2 and 6");
	check(!close_rel(mu_a, 1.0, 0.1) && !close_rel(mu_b, 1.0, 0.1),
	      "and neither equals the prior weight of 1, so the two weights are told apart");
	std::printf("  rank/n %.17g, group means %.12g and %.12g; HC2 %.3g HC3 %.3g\n", h_bar, mu_a,
	            mu_b, w2, w3);
}

// ── The finite-sample factors divide by rank, never by k ────────────────
static void test_finite_sample_factors_use_rank() {
	std::printf("HC1 and CR1 divide by n - rank, not n - k\n");

	// Another hole found by mutation. Changing n - rank to n - k in both
	// factors passed the whole suite. It is not vacuous: DS7 is rank
	// deficient, with k = 4 and rank = 3, so the two denominators are 8
	// and 9 and the standard errors move 6.1%.
	//
	// The only rank-deficient robust fit in the suite asserted NaN
	// placement and finiteness, never a value. Two checks close it, and
	// both are exact rather than banded.
	//
	// FIRST, the ratio. HC1 is HC0 times n/(n - rank) and CR1 is CR0 times
	// [G/(G-1)]*[(N-1)/(N-rank)]. Dividing the two covariances entrywise
	// recovers the factor with nothing else in it, and the expected value
	// is computed here from the kernel's own reported n, rank and
	// n_clusters. On DS7 that is 12/9 = 1.3333 for HC1 against 12/8 = 1.5
	// under the mutation, and 1.8333 against 2.0625 for CR1. Measured 0 on
	// HC1 and 1.21e-16 on CR1, against 0.111 for the k form.
	//
	// The band is 1e-14. Only rounding separates the ratio from the exact
	// factor, and 1e-14 is 80 times the worst measurement. It is 13 orders
	// inside the 0.111 the mutation moves.
	std::vector<double> y(kDs7Y, kDs7Y + kDs7N);
	const std::vector<int> cl = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
	statsduck::GlmOptions base;
	base.family = statsduck::Family::kPoisson;
	base.link = statsduck::Link::kLog;

	statsduck::GlmOptions h0 = base, h1 = base, c0 = base, c1 = base;
	h0.vcov = statsduck::Vcov::kHC0;
	h1.vcov = statsduck::Vcov::kHC1;
	c0.vcov = statsduck::Vcov::kCR0;
	c1.vcov = statsduck::Vcov::kCR1;
	const statsduck::GlmResult ghc0 = statsduck::fit_glm(y, ds7_design(), h0);
	const statsduck::GlmResult ghc1 = statsduck::fit_glm(y, ds7_design(), h1);
	const statsduck::GlmResult gcr0 =
	    statsduck::fit_glm(y, ds7_design(), c0, nullptr, nullptr, &cl);
	const statsduck::GlmResult gcr1 =
	    statsduck::fit_glm(y, ds7_design(), c1, nullptr, nullptr, &cl);
	check(ghc0.ok && ghc1.ok && gcr0.ok && gcr1.ok, "all four DS7 robust fits succeed");
	check(ghc1.k == 4 && ghc1.rank == 3, "on a design whose k and rank differ");
	check(gcr1.n_clusters == 3, "with three clusters");

	const double nd = static_cast<double>(ghc1.n);
	const double rank_d = static_cast<double>(ghc1.rank);
	const double gd = static_cast<double>(gcr1.n_clusters);
	const double hc1_factor = nd / (nd - rank_d);
	const double cr1_factor = (gd / (gd - 1.0)) * ((nd - 1.0) / (nd - rank_d));
	double worst_hc = 0.0, worst_cr = 0.0;
	for (std::size_t e = 0; e < ghc0.cov.size(); e++) {
		if (std::isnan(at(ghc0.cov, e))) {
			continue; // the aliased row and column
		}
		const double rh = at(ghc1.cov, e) / at(ghc0.cov, e);
		const double rc = at(gcr1.cov, e) / at(gcr0.cov, e);
		worst_hc = std::max(worst_hc, rel_dev(rh, hc1_factor));
		worst_cr = std::max(worst_cr, rel_dev(rc, cr1_factor));
		check(close_rel(rh, hc1_factor, 1e-14), "HC1 over HC0 is n/(n - rank)");
		check(close_rel(rc, cr1_factor, 1e-14), "CR1 over CR0 carries the rank form too");
	}

	// SECOND, the deficient fit against the reduced one. DS7's x3 is
	// x1 + x2, so dropping it leaves a design holding the same values as
	// one built from x1 and x2 alone. The reduced design has k == rank, so
	// the two denominators coincide there and the mutation cannot move it.
	// It moves only the deficient side, which is what makes this a
	// separation rather than a restatement. test_identities already pins
	// this identity for the classical estimator; the robust ones need it
	// for the factor.
	//
	// The band is 0.0, which close_rel passes only on exact equality, and
	// it measures 0 on both. Every working weight, every score and every
	// factor is built from the same numbers in the same order on the two
	// sides.
	for (const statsduck::Vcov v : {statsduck::Vcov::kHC1, statsduck::Vcov::kCR1}) {
		statsduck::GlmOptions go = base;
		go.vcov = v;
		const bool clustered = (v == statsduck::Vcov::kCR1);
		const std::vector<int> *ids = clustered ? &cl : nullptr;
		const statsduck::GlmResult def =
		    statsduck::fit_glm(y, ds7_design(), go, nullptr, nullptr, ids);
		const statsduck::GlmResult red = statsduck::fit_glm(
		    y, design(kDs7N, {kDs7X1, kDs7X2}), go, nullptr, nullptr, ids);
		check(def.ok && red.ok, "the deficient and the reduced robust fit both succeed");
		check(def.k == 4 && def.rank == 3 && red.k == 3 && red.rank == 3,
		      "and only one of them is deficient");
		for (std::size_t a = 0; a < 3; a++) {
			check(close_rel(at(def.std_error, a), at(red.std_error, a), 0.0),
			      "the deficient robust standard error is bitwise the reduced one");
			for (std::size_t b = 0; b < 3; b++) {
				check(close_rel(at(def.cov, a * def.k + b), at(red.cov, a * red.k + b), 0.0),
				      "and so is every kept covariance entry");
			}
		}
	}
	std::printf("  DS7 n %g rank %g k 4: HC1 factor %.12g (worst %.3g), CR1 factor %.12g "
	            "(worst %.3g)\n",
	            nd, rank_d, hc1_factor, worst_hc, cr1_factor, worst_cr);
}

// ── The weighted sandwich, which no gaussian unit-weight test reaches ────
static void test_weighted_sandwich() {
	std::printf("a weighted sandwich equals fit_lm's on the sqrt(w)-scaled design\n");

	// This test exists because the gaussian equivalence above cannot see
	// the prior weight at all. There every weight is 1. The w factor in
	// the score and the w factor in the hat diagonal are then both
	// multiplications by 1, so a build that omitted either would still
	// pass. MEASURED: removing `h *= final_w[r]` from the hat diagonal
	// fails 12 checks, all of them in this test, and nothing else in the
	// suite. It moves this test's HC2 standard error by 24.8% and its HC3
	// standard error by 46.8%.
	//
	// The identity. Write W for the diagonal prior weights. For gaussian
	// with an identity link the working weight IS the prior weight. So
	// weighted least squares on (y, X) is ordinary least squares on
	// (W^(1/2) y, W^(1/2) X). Under that transform every piece of the
	// sandwich maps across exactly:
	//   beta       (Xt W X)^-1 Xt W y is the OLS estimate on the scaled data
	//   bread      Xt W X is the scaled design's Xt X
	//   score      x_i w_i (y_i - mu_i) is the scaled row times its own
	//              scaled residual, since each side contributes one w^(1/2)
	//   leverage   w_i x_i' (Xt W X)^-1 x_i is the scaled design's own
	//              hat diagonal, which is where the w factor shows up
	//   dispersion sum w (y - mu)^2 / (n - rank) is the scaled RSS over the
	//              same df, so it equals fit_lm's sigma squared
	// fit_lm has no weights argument, so the scaled design is how it is
	// asked the same question. Its intercept column has to be scaled too,
	// which is why intercept is false below and the scaled constant is
	// passed as the first predictor column.
	//
	// The weights are perfect squares so that every sqrt is exact in
	// binary, and x and y are small integers. That removes the scaling
	// itself as a source of difference and leaves only the solve.
	//
	// SCOPE NOTE: this is the first weighted non-binomial fit in the
	// suite. The fixture header records that the w-scaled LOG-LIKELIHOOD
	// is unverified against statsmodels for gaussian, poisson and gamma.
	// Nothing here asserts a log-likelihood, an AIC or a deviance. The
	// quantities compared are beta, the dispersion, the standard errors
	// and the covariance, and the identity above derives each of them
	// independently of that formula.
	const std::vector<double> y = {3, 5, 4, 9, 7, 6, 12, 10};
	const std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8};
	const std::vector<double> w = {1, 4, 9, 16, 1, 4, 9, 16};
	const std::vector<int> cl = {0, 0, 1, 1, 2, 2, 3, 3};
	const std::size_t n = y.size();
	Mat X(n, 1);
	for (std::size_t i = 0; i < n; i++) {
		X(i, 0) = x[i];
	}
	std::vector<double> y_scaled(n, 0.0);
	Mat X_scaled(n, 2);
	for (std::size_t i = 0; i < n; i++) {
		const double s = std::sqrt(w[i]);
		y_scaled[i] = s * y[i];
		X_scaled(i, 0) = s;        // the scaled intercept column
		X_scaled(i, 1) = s * x[i]; // the scaled predictor
	}

	// The band is 1e-12. Measured worst over all seven estimators: beta 0,
	// dispersion 1.13e-16, standard error 1.14e-15, covariance 2.48e-15,
	// the last two on HC2. The two fitters reach the same numbers by
	// different arithmetic. This kernel multiplies by w once and fit_lm
	// multiplies by sqrt(w) twice. So the agreement is rounding, and the
	// band is 400 times it. A structural difference is far larger, as the
	// 24.8% measurement above shows.
	const statsduck::Vcov all[] = {statsduck::Vcov::kConst, statsduck::Vcov::kHC0,
	                               statsduck::Vcov::kHC1,   statsduck::Vcov::kHC2,
	                               statsduck::Vcov::kHC3,   statsduck::Vcov::kCR0,
	                               statsduck::Vcov::kCR1};
	for (const statsduck::Vcov v : all) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kGaussian;
		go.link = statsduck::Link::kIdentity;
		go.vcov = v;
		statsduck::LmOptions lo;
		lo.vcov = v;
		lo.intercept = false; // the scaled design carries its own constant
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, go, &w, nullptr, &cl);
		const statsduck::LmResult o = statsduck::fit_lm(y_scaled, X_scaled, lo, &cl);
		check(g.ok && o.ok, "both the weighted and the scaled fit succeed");
		check(g.k == o.k && g.rank == o.rank && g.df_residual == o.df_residual,
		      "with the same k, rank and residual df");
		double worst = 0.0;
		for (std::size_t j = 0; j < o.k; j++) {
			worst = std::max(worst, rel_dev(at(g.beta, j), at(o.beta, j)));
			worst = std::max(worst, rel_dev(at(g.std_error, j), at(o.std_error, j)));
			check(close_rel(at(g.beta, j), at(o.beta, j), 1e-12), "the same coefficient");
			check(close_rel(at(g.std_error, j), at(o.std_error, j), 1e-12),
			      "the same weighted standard error");
		}
		for (std::size_t e = 0; e < o.cov.size(); e++) {
			worst = std::max(worst, rel_dev(at(g.cov, e), at(o.cov, e)));
			check(close_rel(at(g.cov, e), at(o.cov, e), 1e-12), "the same covariance entry");
		}
		worst = std::max(worst, rel_dev(g.dispersion, o.sigma * o.sigma));
		check(close_rel(g.dispersion, o.sigma * o.sigma, 1e-12),
		      "and the dispersion is the scaled fit's sigma squared");
		std::printf("  %-5s weighted: worst gap %.3g, se0 %.12g\n", statsduck::vcov_name(v), worst,
		            at(g.std_error, 0));
	}
}

// ── An aliased column stays NaN under every estimator ────────────────────
static void test_aliased_under_every_estimator() {
	std::printf("an aliased column is NaN under every estimator\n");

	// DS7's x3 is x1 + x2, so x3 is the aliased column. The scatter back
	// into k-space writes only the kept rows and columns of the reduced
	// covariance, so the aliased row and column must stay NaN whichever
	// estimator filled that covariance. A sandwich that wrote into the
	// wrong index, or that returned a rank x rank matrix the scatter
	// misread, would show up here as a finite number where NaN belongs.
	//
	// Three clusters over twelve rows gives DS7 a cluster-robust fit as
	// well, so the CR path is covered by the same rule. The ids are dense
	// 0, 1 and 2, which is the kernel's documented contract.
	const std::vector<int> cl = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
	const statsduck::Vcov all[] = {statsduck::Vcov::kConst, statsduck::Vcov::kHC0,
	                               statsduck::Vcov::kHC1,   statsduck::Vcov::kHC2,
	                               statsduck::Vcov::kHC3,   statsduck::Vcov::kCR0,
	                               statsduck::Vcov::kCR1};
	for (const statsduck::Vcov v : all) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kPoisson;
		go.link = statsduck::Link::kLog;
		go.vcov = v;
		const statsduck::GlmResult g =
		    statsduck::fit_glm(ds7_y, ds7_design(), go, nullptr, nullptr, &cl);
		check(g.ok, "the deficient fit succeeds");
		check(g.k == kDs7K && g.rank == kDs7Rank, "with one column dropped");
		const std::size_t ja = kDs7AliasedColumn;
		check(std::isnan(at(g.beta, ja)), "the aliased coefficient is NaN");
		check(std::isnan(at(g.std_error, ja)), "so is its standard error");
		check(std::isnan(at(g.z_statistic, ja)), "so is its z statistic");
		check(std::isnan(at(g.p_value, ja)), "so is its p-value");
		for (std::size_t b = 0; b < g.k; b++) {
			check(std::isnan(at(g.cov, ja * g.k + b)), "its whole covariance row is NaN");
			check(std::isnan(at(g.cov, b * g.k + ja)), "and its whole covariance column");
		}
		// Every other entry is finite, so the NaN marks the aliased term
		// rather than a covariance that failed to be computed.
		for (std::size_t a = 0; a < g.k; a++) {
			for (std::size_t b = 0; b < g.k; b++) {
				if (a == ja || b == ja) {
					continue;
				}
				check(std::isfinite(at(g.cov, a * g.k + b)), "and every kept entry is finite");
			}
		}
	}
}

// ── The cluster-id convention is lm_core's ───────────────────────────────
static void test_cluster_id_convention() {
	std::printf("the cluster count follows lm_core's max-id convention\n");

	// DECISION, recorded here because it sets CR1's [G/(G-1)] factor:
	// glm_fit derives G from the LARGEST id plus one, exactly as fit_lm
	// does, rather than from the number of distinct ids. So {0, 2, 4, 6, 8}
	// gives G = 9 even though only five labels appear.
	//
	// Both headers require DENSE 0-based labels, and densify_clusters is
	// the supported producer. So on every supported path the largest id
	// plus one IS the distinct count. Keeping the rule identical to
	// fit_lm's has two benefits. It makes the gaussian equivalence test
	// above a real check rather than one restricted to dense inputs. And
	// it keeps the two fitters from reporting different CR1 numbers for
	// the same ids, which would be a worse defect than a documented
	// precondition. Tightening the rule belongs in one change across both
	// fitters.
	//
	// This test asserts that AGREEMENT. It does not assert that the
	// number is the right count. It fits the same sparse ids through both
	// kernels and requires the same G and the same covariance. An empty
	// cluster contributes a zero score sum, so only the factor and
	// n_clusters move, and they move together in both fitters.
	//
	// The band is 0.0, which close_rel passes only on exact equality,
	// because this is the same gaussian identity reduction as the test
	// above and it measures 0 there too.
	std::vector<int> sparse(kDs8N, 0);
	for (std::size_t i = 0; i < kDs8N; i++) {
		sparse[i] = 2 * ds8_clusters[i]; // 0, 2, 4, 6, 8 — dense ids doubled
	}
	for (const statsduck::Vcov v : {statsduck::Vcov::kCR0, statsduck::Vcov::kCR1}) {
		statsduck::GlmOptions go;
		go.family = statsduck::Family::kGaussian;
		go.link = statsduck::Link::kIdentity;
		go.vcov = v;
		statsduck::LmOptions lo;
		lo.vcov = v;
		const statsduck::GlmResult g =
		    statsduck::fit_glm(ds8_y, ds8_design(), go, nullptr, nullptr, &sparse);
		const statsduck::LmResult o = statsduck::fit_lm(ds8_y, ds8_design(), lo, &sparse);
		check(g.ok && o.ok, "both fitters accept a sparse id vector");
		check(g.n_clusters == o.n_clusters, "and agree on the cluster count");
		check(g.n_clusters == 9, "which is the largest id plus one, not the distinct count");
		for (std::size_t e = 0; e < o.cov.size(); e++) {
			check(close_rel(at(g.cov, e), at(o.cov, e), 0.0),
			      "and on every covariance entry under that convention");
		}
	}

	// A negative id and a single cluster are still failures, under glm as
	// under lm. These are the two conditions the convention does screen.
	statsduck::GlmOptions go;
	go.family = statsduck::Family::kGaussian;
	go.link = statsduck::Link::kIdentity;
	go.vcov = statsduck::Vcov::kCR1;
	std::vector<int> negative(kDs8N, 0);
	negative[3] = -1;
	const statsduck::GlmResult neg =
	    statsduck::fit_glm(ds8_y, ds8_design(), go, nullptr, nullptr, &negative);
	check(!neg.ok && !neg.error.empty(), "a negative cluster id is a failed fit");
	const std::vector<int> one(kDs8N, 0);
	const statsduck::GlmResult single =
	    statsduck::fit_glm(ds8_y, ds8_design(), go, nullptr, nullptr, &one);
	check(!single.ok && !single.error.empty(), "and so is a single cluster");
	const statsduck::GlmResult missing = statsduck::fit_glm(ds8_y, ds8_design(), go);
	check(!missing.ok && !missing.error.empty(), "and so is a cluster estimator with no ids");
	std::printf("  sparse ids give G = 9 in both fitters; negative, single and missing ids fail\n");
}

// ── The identities that need no oracle ───────────────────────────────────
static void test_identities() {
	std::printf("oracle-free identities: the null model, grouping, scaling, rank deficiency\n");

	statsduck::GlmOptions nb;
	nb.family = statsduck::Family::kBinomial;
	nb.link = statsduck::Link::kLogit;

	// null_deviance is the intercept-only model's deviance, computed two
	// ways: inside the full fit, and by fitting that model here. A single
	// golden number would not catch a wrong null model.
	//
	// Measured 0: the two are bitwise equal, because inside the full fit
	// null_deviance comes from the same recursive call on the same inputs
	// that this line makes directly. The band is 1e-10 rather than 0 to
	// leave room for a future implementation that reaches the null model
	// another way, as R does with a closed-form weighted mean. A change
	// of that kind should still agree this closely, and should be
	// reviewed rather than silently fail here.
	const statsduck::GlmResult full = statsduck::fit_glm(ds1_y, ds1_design(), nb);
	const statsduck::GlmResult nullfit = statsduck::fit_glm(ds1_y, Mat(ds1_y.size(), 0), nb);
	check(nullfit.ok, "the intercept-only fit succeeds");
	check(close_rel(full.null_deviance, nullfit.deviance, 1e-10),
	      "null_deviance equals the intercept-only fit's deviance");
	check(nullfit.k == 1 && nullfit.rank == 1, "the null model has exactly one parameter");

	// Grouped and ungrouped binomial are the same data in two shapes, so
	// the coefficients and the standard errors must not move. This is the
	// test for the whole weights path. The band is 1e-8 and the measured
	// agreement is far inside it: 2.7e-11 on beta and 4.3e-12 on the
	// standard errors, printed below.
	const statsduck::GlmResult grouped = statsduck::fit_glm(ds2_y, ds2_design(), nb, &ds2_w);
	check(grouped.ok, "the grouped fit succeeds");
	double wb = 0.0, ws = 0.0;
	for (std::size_t j = 0; j < full.k; j++) {
		wb = std::max(wb, rel_dev(at(grouped.beta, j), at(full.beta, j)));
		ws = std::max(ws, rel_dev(at(grouped.std_error, j), at(full.std_error, j)));
		check(close_rel(at(grouped.beta, j), at(full.beta, j), 1e-8),
		      "grouping does not change the coefficients");
		check(close_rel(at(grouped.std_error, j), at(full.std_error, j), 1e-8),
		      "nor the standard errors");
	}
	std::printf("  grouped against ungrouped: worst beta %.3g, worst se %.3g\n", wb, ws);

	// Scaling every trial count scales the deviance and leaves beta alone.
	//
	// Two bands, both measured. The coefficients agree to 5.9e-11, not
	// exactly, because scaling the weights scales the working weights and
	// so the whole solve: the two fits follow different floating-point
	// paths to the same optimum and stop at slightly different iterates.
	// 1e-8 is 170 times that. The deviance agrees to 2.2e-15, which is
	// rounding, because the 10x factor comes straight out of the
	// per-row sum; it shares the 1e-8 band for symmetry rather than
	// needing it.
	std::vector<double> w10 = ds2_w;
	for (double &v : w10) {
		v *= 10.0;
	}
	const statsduck::GlmResult scaled = statsduck::fit_glm(ds2_y, ds2_design(), nb, &w10);
	check(scaled.ok, "the scaled fit succeeds");
	for (std::size_t j = 0; j < grouped.k; j++) {
		check(close_rel(at(scaled.beta, j), at(grouped.beta, j), 1e-8),
		      "scaling the trial counts leaves the coefficients unchanged");
	}
	check(close_rel(scaled.deviance, 10.0 * grouped.deviance, 1e-8),
	      "and scales the deviance by the same factor");

	// A rank-deficient design: the later column of a dependent set is
	// aliased. DS7 is a poisson fixture, so it is fitted as one. A
	// binomial fit of counts above 1 is a domain failure, not a fit.
	statsduck::GlmOptions pg;
	pg.family = statsduck::Family::kPoisson;
	pg.link = statsduck::Link::kLog;
	const statsduck::GlmResult rd = statsduck::fit_glm(ds7_y, ds7_design(), pg);
	check(rd.ok, "a rank-deficient fit is not a failure");
	check(rd.rank < rd.k, "the rank is below k");
	check(rd.terms.size() == rd.k, "every column is still named");
	check(std::isnan(at(rd.beta, rd.k - 1)), "the dependent column is aliased");
	check(std::isfinite(at(rd.beta, 1)), "the kept columns are estimated");
	check(rd.df_residual == rd.n - rd.rank, "df_residual counts the rank, not k");

	// The sharpest statement about aliasing, and it needs no oracle: the
	// deficient fit must BE the reduced fit. Dropping x3 leaves X_kept
	// holding the same values as a design built from x1 and x2 alone, so
	// the starting values, every working weight, every solve and every
	// reported quantity are identical bit for bit. The band is 0.0, which
	// close_rel passes only on exact equality. That is far stronger than
	// the 2e-5 oracle band DS7's goldens carry above, and it is what
	// would catch a scatter that mixed up a column index.
	const statsduck::GlmResult reduced =
	    statsduck::fit_glm(ds7_y, design(kDs7N, {kDs7X1, kDs7X2}), pg);
	check(reduced.ok && reduced.k == 3 && reduced.rank == 3, "the reduced design fits too");
	check(rd.iterations == reduced.iterations, "in the same number of passes");
	for (std::size_t j = 0; j < 3; j++) {
		check(close_rel(at(rd.beta, j), at(reduced.beta, j), 0.0),
		      "the deficient fit's kept coefficients are bitwise the reduced fit's");
		check(close_rel(at(rd.std_error, j), at(reduced.std_error, j), 0.0),
		      "and so are its standard errors");
		for (std::size_t b = 0; b < 3; b++) {
			check(close_rel(at(rd.cov, j * rd.k + b), at(reduced.cov, j * reduced.k + b), 0.0),
			      "and every covariance entry");
		}
	}
	check(close_rel(rd.deviance, reduced.deviance, 0.0), "and the deviance");
	check(close_rel(rd.dispersion, reduced.dispersion, 0.0), "and the dispersion");
	check(close_rel(rd.loglik, reduced.loglik, 0.0), "and the log-likelihood");
	check(rd.df_residual == reduced.df_residual, "and the residual degrees of freedom");
}

// ── The binomial separation heuristic ────────────────────────────────────
static void test_separation() {
	std::printf("the binomial separation heuristic\n");

	// DS6 is perfectly separated: a threshold on x1 predicts y exactly, so
	// the likelihood has no interior maximum and the coefficients diverge.
	// Without a flag the caller sees an enormous estimate beside an
	// enormous standard error and nothing saying why. That is how
	// separation reaches a published table.
	statsduck::GlmOptions sg;
	sg.family = statsduck::Family::kBinomial;
	sg.link = statsduck::Link::kLogit;
	const statsduck::GlmResult s = statsduck::fit_glm(ds6_y, ds6_design(), sg);
	check(s.ok, "a separated fit still returns a result, not a failure");
	check(!s.converged, "it does not converge");
	check(s.separation, "and the separation flag is set");
	// This assertion depends on max_iter staying 25. Task 4 measured the
	// relative deviance change at pass 25 as 2.44e-08. The tolerance is
	// 1e-08. That is a margin of only 2.4 times. At max_iter = 28 the same
	// fit reports converged = true, with beta[1] = 42.33. So a future
	// change to max_iter should break this assertion. This comment
	// explains why, so the break is not a mystery.
	check(s.iterations == sg.max_iter, "it used every iteration");
	check(std::isfinite(at(s.beta, 1)), "the reported coefficient is finite, if large");
	// The deviance is near zero. DS6 is built to fit almost perfectly.
	// The band here is loose and absolute. It is never the project's
	// usual 1e-10 relative check. The fixture header marks every one of
	// DS6's goldens unsafe at that tight tolerance. Statsmodels' own
	// unclamped computation gives 3.34e-09. This kernel's clamped
	// link_inv gives about 7.12e-09. That is a 36% relative gap between
	// two numbers that both read as zero. 1e-6 sits two orders of
	// magnitude above either one, so it catches a real defect without
	// reopening the clamped-vs-unclamped gap this comment just described.
	check(s.deviance < 1e-6, "and the deviance is near zero");
	std::printf("  DS6: converged=%d iterations=%d separation=%d deviance=%.3g beta1=%.6g\n",
	            s.converged, s.iterations, s.separation, s.deviance, at(s.beta, 1));

	// The flag is binomial-only. A poisson fit that fails to converge must
	// not claim separation, because the heuristic has no meaning there.
	statsduck::GlmOptions pg;
	pg.family = statsduck::Family::kPoisson;
	pg.link = statsduck::Link::kLog;
	const statsduck::GlmResult p =
	    statsduck::fit_glm(ds3_y, ds3_design(), pg, nullptr, &ds3_offset);
	check(p.ok && p.converged, "the poisson fixture converges");
	check(!p.separation, "and never reports separation");

	// A well-behaved binomial fit does not report separation either.
	const statsduck::GlmResult ok1 = statsduck::fit_glm(ds1_y, ds1_design(), sg);
	check(ok1.converged && !ok1.separation, "a clean logistic fit is not separated");
}

// ── No residual degrees of freedom (Review Focus item 3) ─────────────────
static void test_no_residual_df() {
	std::printf("a fit with no residual df fails, before any division\n");

	// The dispersion divides by n - rank, so one row with one predictor
	// plus an intercept divides by zero and would put Inf or NaN across
	// the whole result. The decision has to be made before any division.
	Mat tiny(1, 1);
	tiny(0, 0) = 2.0;
	statsduck::GlmOptions tg;
	tg.family = statsduck::Family::kGaussian;
	const statsduck::GlmResult t1 = statsduck::fit_glm({5.0}, tiny, tg);
	check(!t1.ok, "one row with an intercept and one predictor is a failed fit");
	check(!t1.error.empty(), "and says why");

	// Two rows and two parameters is exactly saturated: n == rank, so
	// still no df.
	Mat two(2, 1);
	two(0, 0) = 1.0;
	two(1, 0) = 2.0;
	const statsduck::GlmResult t2 = statsduck::fit_glm({3.0, 7.0}, two, tg);
	check(!t2.ok, "a saturated gaussian fit has no residual df and fails");

	// Three rows and two parameters is the smallest fit that works.
	Mat three(3, 1);
	three(0, 0) = 1.0;
	three(1, 0) = 2.0;
	three(2, 0) = 4.0;
	const statsduck::GlmResult t3 = statsduck::fit_glm({3.0, 7.0, 11.0}, three, tg);
	check(t3.ok && t3.df_residual == 1, "three rows and two parameters leaves one df");
	check(std::isfinite(t3.dispersion), "and a finite dispersion");

	// Binomial and poisson have a fixed dispersion, so they do not divide
	// at all. They still need df_residual to be meaningful, so the rule is
	// uniform.
	statsduck::GlmOptions bg;
	bg.family = statsduck::Family::kBinomial;
	bg.link = statsduck::Link::kLogit;
	const statsduck::GlmResult b1 = statsduck::fit_glm({1.0}, tiny, bg);
	check(!b1.ok, "the n <= rank rule applies to binomial too");
}

// ── A response with no variation (Review Focus item 4) ───────────────────
static void test_no_variation_response() {
	std::printf("a response with no variation in y\n");

	// Every row says the same thing. For binomial that is separation with
	// no information at all: there is not even a threshold to find, only
	// a constant answer. For poisson, an all-zero count drives mu toward
	// 0 and the log link toward minus infinity. Either outcome is
	// acceptable: a failed fit, or a finite result with converged and
	// separation set honestly. Neither outcome allows a NaN in a field
	// that reads as an estimate.
	statsduck::GlmOptions sg;
	sg.family = statsduck::Family::kBinomial;
	sg.link = statsduck::Link::kLogit;
	statsduck::GlmOptions pg;
	pg.family = statsduck::Family::kPoisson;
	pg.link = statsduck::Link::kLog;

	const std::vector<double> all_zero(ds1_y.size(), 0.0);
	const statsduck::GlmResult bz = statsduck::fit_glm(all_zero, ds1_design(), sg);
	if (bz.ok) {
		for (std::size_t j = 0; j < bz.k; j++) {
			const double b = at(bz.beta, j);
			check(!std::isnan(b), "an all-zero binomial response gives no NaN coefficient");
		}
		check(std::isfinite(bz.deviance), "and a finite deviance");
		check(!bz.converged || bz.separation || std::isfinite(at(bz.std_error, 0)),
		      "either it did not converge, or it is flagged, or the errors are finite");
	}
	// MEASURED: ok, and converged at iteration 23, inside the 25-iteration
	// cap. beta is about (-24.57, 3e-14, 4e-14). The deviance is about
	// 8.00e-09. The loop's own stopping rule calls this converged. The
	// relative deviance change drops under tol. Beta is still large at
	// that point, but finite. The three-condition heuristic needs a
	// non-converged fit. This fit does not meet that condition. The flag
	// does not fire. Yet an all-zero response carries no more information
	// than DS6's separating one. What marks the fit instead is the
	// standard error: about 40311 on the intercept, against a coefficient
	// of -24.57. That is four orders of magnitude apart. This is a real
	// limitation of the heuristic, not a defect in this test. The same
	// limitation is written down beside the flag, in GlmResult::separation.

	const std::vector<double> all_one(ds1_y.size(), 1.0);
	const statsduck::GlmResult bo = statsduck::fit_glm(all_one, ds1_design(), sg);
	check(bo.ok || !bo.error.empty(), "an all-one binomial response either fits or says why");
	// MEASURED: the mirror image of all_zero. It converges at iteration
	// 23. beta is about (24.57, 7e-15, 9e-14). The deviance and the
	// standard errors are the same as all_zero's.
	std::printf("  all-zero/all-one binomial: converged at %d it, beta0 %.6g / %.6g, se0 %.6g\n",
	            bz.iterations, at(bz.beta, 0), at(bo.beta, 0), at(bz.std_error, 0));

	// All-zero poisson counts. mu -> 0 and eta -> -inf under a log link.
	const statsduck::GlmResult pz = statsduck::fit_glm(all_zero, ds1_design(), pg);
	if (pz.ok) {
		check(std::isfinite(pz.deviance), "an all-zero poisson deviance is finite");
		check(!std::isnan(at(pz.beta, 0)), "and the intercept is not NaN");
	}
	// MEASURED: ok, and converged at iteration 22. beta is about (-24.62,
	// -4e-13, -3e-13). The deviance is about 8.00e-09. The intercept's
	// standard error is about 198777. separation stays false here no
	// matter what. The family is poisson, not binomial, so the flag
	// cannot apply.
	std::printf("  all-zero poisson: converged at %d it, beta0 %.6g, se0 %.6g\n", pz.iterations,
	            at(pz.beta, 0), at(pz.std_error, 0));

	// A constant gaussian response is a legitimate fit with zero deviance.
	const std::vector<double> all_five(ds5_y.size(), 5.0);
	statsduck::GlmOptions gg;
	gg.family = statsduck::Family::kGaussian;
	const statsduck::GlmResult gc = statsduck::fit_glm(all_five, ds5_design(), gg);
	check(gc.ok, "a constant gaussian response fits");
	// The band is absolute, not close_rel's relative one. close_rel's
	// ratio against a target of exactly 0.0 is diff / diff. That ratio is
	// 1.0 for any nonzero value. So close_rel only ever passes here on
	// bit-exact equality. The true residual is a rounding artifact, not
	// an exact zero. beta recovers the intercept to within about 1.4e-15
	// relative of 5. That leaves a per-row residual near 5e-15. Squared
	// and summed over the rows, the deviance lands near 1e-28. Measured
	// 8.44e-29. 1e-20 is nine orders of magnitude looser. It is still far
	// below anything a real fitting defect would leave.
	check(std::fabs(gc.deviance) <= 1e-20, "with a deviance at rounding level, not an estimate");
	check(close_rel(at(gc.beta, 0), 5.0, 1e-10), "an intercept of 5");
	check(!gc.separation, "and no separation claim");
	std::printf("  constant-5 gaussian: converged at %d it, deviance %.3g, beta0 %.17g\n",
	            gc.iterations, gc.deviance, at(gc.beta, 0));
}

// ── The fitted mean never leaves the family's support ────────────────────

// Smallest and largest fitted mean, recomputed from the reported
// coefficients of a single-predictor fit with an intercept.
static void mu_range(const statsduck::GlmResult &g, const std::vector<double> &x,
                     statsduck::Link l, double &lo, double &hi) {
	lo = std::numeric_limits<double>::quiet_NaN();
	hi = lo;
	if (g.beta.size() < 2) {
		return; // a failed fit reports nothing to recompute from
	}
	for (std::size_t i = 0; i < x.size(); i++) {
		const double mu = statsduck::link_inv(l, g.beta[0] + g.beta[1] * x[i]);
		lo = (i == 0) ? mu : std::min(lo, mu);
		hi = (i == 0) ? mu : std::max(hi, mu);
	}
}

static void test_fitted_mean_stays_in_support() {
	std::printf("the fitted mean never leaves the family's support\n");

	// link_inv's clamp does not cover this. It clamps a probability link
	// into [kMuEps, 1 - kMuEps], but the identity link is never clamped
	// and the log link is clamped at the floor only, and both of those
	// are allowed for a family whose support they can leave. The loop
	// therefore tests the support of every proposed fitted mean and
	// halves the step when it fails, which is what R's validmu does.
	//
	// Both inputs below are real failures found by a random search over
	// the input space, not constructed cases. Without the support test
	// each one returns ok = true and converged = true with a fitted mean
	// outside its own support, and the reported deviance, dispersion,
	// log-likelihood, AIC and standard errors are all computed there. The
	// mechanism is specific: the binomial and poisson deviance skip the
	// y*ln(y/mu) term at y = 0, so a negative mu at a zero-count row
	// contributes a finite negative amount instead of NaN and the step is
	// accepted.

	// A poisson fit with an identity link. Without the support test this
	// converges in 5 passes to beta = (2.638, -0.784), whose smallest
	// fitted mean is -0.771: a negative expected count.
	{
		const std::vector<double> y = {2, 0, 1, 0, 5, 0, 3, 5};
		const std::vector<double> x = {-1.387, 4.348, 2.565, 0.520, -0.029, 2.755, 2.156, -4.418};
		Mat X(8, 1);
		for (std::size_t i = 0; i < 8; i++) {
			X(i, 0) = x[i];
		}
		statsduck::GlmOptions po;
		po.family = statsduck::Family::kPoisson;
		po.link = statsduck::Link::kIdentity;
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, po);
		double lo = 0.0, hi = 0.0;
		mu_range(g, x, po.link, lo, hi);
		// Either outcome is acceptable: a failed fit, or a fit whose every
		// fitted mean is positive. A converged fit with a negative
		// expected count is not. This one fails, because the violation
		// happens on the first pass, which has no previous coefficient
		// vector to halve toward.
		check(!g.ok || lo > 0.0, "a poisson fit never reports a negative expected count");
		check(!g.ok || std::isfinite(g.deviance), "and its deviance is finite");
		check(!g.ok == !g.error.empty(), "a failed fit says why and a successful one does not");
	}

	// A binomial fit with a log link. Without the support test this
	// converges in 7 passes to a largest fitted probability of 1.445.
	{
		const std::vector<double> y = {0, 0, 1, 0, 1, 1, 0, 1};
		const std::vector<double> x = {2.5763, 0.2360,  -1.4722, -2.3383,
		                               -3.5882, 2.0167, 4.6552,  -4.8203};
		Mat X(8, 1);
		for (std::size_t i = 0; i < 8; i++) {
			X(i, 0) = x[i];
		}
		statsduck::GlmOptions bo;
		bo.family = statsduck::Family::kBinomial;
		bo.link = statsduck::Link::kLog;
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, bo);
		double lo = 0.0, hi = 0.0;
		mu_range(g, x, bo.link, lo, hi);
		check(!g.ok || hi < 1.0, "a binomial fit never reports a probability above 1");
		check(!g.ok || lo > 0.0, "nor one below 0");
	}
}

// ── Step halving, the mechanism that keeps a bad step out ────────────────

// The IRLS fixed point, measured in standard errors.
//
// At the maximum likelihood estimate the weighted score is zero for every
// estimated coefficient:
//   score_j = sum_i w_i (y_i - mu_i) * link_deriv(eta_i) / variance(mu_i) * x_ij
// Exactly zero is not reachable in floating point, so the question is how
// big the remaining score is on a scale that means something. The Newton
// step it implies has j-th component of order std_error_j^2 * score_j, so
// |score_j| * std_error_j is that step expressed in standard errors of
// the estimate: dimensionless, and comparable across fits.
//
// Precondition: an intercept, exactly one predictor column, full rank,
// and a family whose dispersion is fixed at 1. The dispersion scales the
// score and the standard error by different powers, so the measure is
// only calibrated for binomial and poisson.
static double worst_score_in_se(const statsduck::GlmResult &g, statsduck::Family f,
                                statsduck::Link l, const std::vector<double> &y,
                                const std::vector<double> &x) {
	if (g.beta.size() < 2) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	double worst = 0.0;
	for (std::size_t j = 0; j < 2; j++) {
		double s = 0.0;
		for (std::size_t i = 0; i < y.size(); i++) {
			const double eta = g.beta[0] + g.beta[1] * x[i];
			const double mu = statsduck::link_inv(l, eta);
			const double d = statsduck::link_deriv(l, eta);
			s += (y[i] - mu) * d / statsduck::variance(f, mu) * (j == 0 ? 1.0 : x[i]);
		}
		worst = std::max(worst, std::fabs(s) * at(g.std_error, j));
	}
	return worst;
}

// How far from the exact stationary point a converged fit is allowed to
// sit, in standard errors of the estimate (see worst_score_in_se).
//
// This band has a theoretical bound AND a measurement, and they agree.
// The bound: the loop stops when the deviance change falls below
// tol*(|dev| + 0.1), which is 1.7e-07 on the poisson fit below. Near the
// optimum a coefficient error of c standard errors costs about c^2 in
// deviance, so c is at most sqrt(1.7e-07) = 4.1e-04, and the score in
// these units is about 2c, so about 1e-03. The measurements: 5.0e-05 on
// the poisson identity halving fit, 5.8e-08 on the exhausted-halvings
// fit, 1.9e-07 on probit and 2.2e-05 on cloglog. The band is 20 times
// the worst of those and matches the bound, so it fails only when a fit
// stops somewhere that is not a stationary point at all. That is what it
// is for: removing the step halving puts the first of those fits 47
// standard errors from zero.
static const double TOL_SCORE = 1e-3;

static void test_step_halving() {
	std::printf("step halving, and the exhausted-halvings branch\n");

	// Nothing else in this suite reaches the halving block: the committed
	// fixtures take 77 IRLS passes between them and halve zero times. So
	// without these inputs the mechanism this task reasons about most
	// would have no test at all.
	//
	// Every input below was found by comparing two builds of the kernel,
	// one with the halving and one without, over 40,000 seeded fits
	// across ten family and link pairs. 327 of them differ structurally.
	// The three kept here are the ones whose assertions SEPARATE the two
	// builds, which a test that merely reaches the block would not: the
	// measured no-halving outcome is written beside each.
	//
	// The kernel reports no halving count, so the assertions are on the
	// outcome. They avoid the exact pass count on purpose, because that
	// is the one quantity a last-bit difference in a platform's exp or
	// log could move by one; converged and the deviance are far from any
	// boundary here.
	const std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8};
	Mat X(8, 1);
	for (std::size_t i = 0; i < 8; i++) {
		X(i, 0) = x[i];
	}

	// A poisson fit with an identity link. WITH halving: converges in 7
	// passes to a deviance of 16.95. WITHOUT: the second pass leaves the
	// support, the loop stops there, and it reports converged = false
	// with the first pass's deviance of 93.83, five and a half times
	// larger.
	{
		const std::vector<double> y = {1, 4, 2, 2, 0, 1, 8, 9};
		statsduck::GlmOptions pi;
		pi.family = statsduck::Family::kPoisson;
		pi.link = statsduck::Link::kIdentity;
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, pi);
		check(g.ok && g.converged, "a fit that needs step halving still converges");
		check(g.deviance < 20.0, "to the deviance of 16.95 that halving reaches, not the 93.83 "
		                         "the first pass alone leaves");
		double lo = 0.0, hi = 0.0;
		mu_range(g, x, pi.link, lo, hi);
		check(lo > 0.0, "with every fitted mean inside the support");
		// The real question is not whether it finished but whether it
		// finished at the maximum likelihood estimate.
		const double score = worst_score_in_se(g, pi.family, pi.link, y, x);
		check(score < TOL_SCORE, "and at the IRLS fixed point, so halving did not move the answer");
		std::printf("  halving, poisson identity: dev %.12g, score %.3g se, smallest mu %.4g\n",
		            g.deviance, score, lo);
	}

	// The same shape in a second family, where the step halving rescues a
	// gamma fit with an identity link. WITH halving: converges to a
	// deviance of 5.639. WITHOUT: stops at pass 2, converged = false,
	// deviance 57.98. No score check here: worst_score_in_se is only
	// calibrated for a family whose dispersion is fixed at 1.
	{
		const std::vector<double> y = {3.6, 0.3, 2.3, 0.4, 0.7, 2.9, 3.8, 1.6};
		statsduck::GlmOptions gi;
		gi.family = statsduck::Family::kGamma;
		gi.link = statsduck::Link::kIdentity;
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, gi);
		check(g.ok && g.converged, "a gamma identity-link fit converges through its halvings");
		check(g.deviance < 10.0, "to 5.639 rather than the 57.98 of the step before them");
		double lo = 0.0, hi = 0.0;
		mu_range(g, x, gi.link, lo, hi);
		check(lo > 0.0, "and keeps every fitted mean positive");
		std::printf("  halving, gamma identity: dev %.12g, smallest mu %.4g\n", g.deviance, lo);
	}

	// 25 halvings in one pass, which exhausts them. The step could not be
	// made to lower the deviance, so the loop tests the convergence
	// criterion on it and keeps the iterate it had already accepted. This
	// is the branch the header's `converged` contract describes, and the
	// converged assertion below is what separates it from a build that
	// reports false there instead.
	//
	// This input was harder to find than it looks, and the first candidate
	// was wrong. Of 671 exhaustions across the 40,000 seeded fits, 623 are
	// inside the recursive intercept-only fit that produces
	// null_deviance, whose converged flag nobody reads. Only 48 happen in
	// a fit that is reported, and only 8 of those are inside tol and so
	// set converged through this branch. This is one of the 8: the
	// exhaustion lands on pass 5 with a relative deviance change of
	// 1.2e-16, a last-bit increase at a fit that has already converged,
	// which is exactly the case the branch exists for.
	{
		const std::vector<double> y = {5, 3, 8, 8, 0, 6, 4, 5};
		statsduck::GlmOptions po;
		po.family = statsduck::Family::kPoisson;
		po.link = statsduck::Link::kLog;
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, po);
		check(g.ok, "a fit that exhausts its halvings still reports a result");
		check(g.converged, "and reports converged, because the step it could not improve was "
		                   "already inside tol");
		const double score = worst_score_in_se(g, po.family, po.link, y, x);
		check(score < TOL_SCORE, "which the score confirms: it is at the fixed point");
		std::printf("  exhausted halvings: dev %.12g, score %.3g se\n", g.deviance, score);
	}
}

// ── The probit and cloglog links, which nothing else fits ────────────────
// The IRLS answer against an independent Nelder-Mead minimization of the
// same deviance. The deviance is flat at its minimum, so it agrees far
// more tightly than the coefficients do: measured 6.6e-16 on probit and
// 1.0e-11 on cloglog, against coefficients at 5.8e-08 and 1.6e-05. The
// coefficient band is 12 times the worst of those, and the gap is the
// stopping rule again, not a disagreement about where the optimum is:
// cloglog's 1.6e-05 is the same 2.2e-05 standard errors TOL_SCORE
// measured, converted by its own standard-error-to-estimate ratio of
// 0.84. Two different optima would differ by far more than either band.
static const double TOL_NM_DEV = 1e-9;
static const double TOL_NM_COEF = 2e-4;

static void test_probit_and_cloglog() {
	std::printf("probit and cloglog fits, with no oracle\n");

	// No other test in this suite fits a probit or a cloglog model, so
	// the loop's use of link_eta, link_inv and link_deriv for those two
	// links is otherwise never executed. There is no golden for them, so
	// these are identities instead.
	const std::vector<double> y = {1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1};
	const std::vector<double> x = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
	Mat X(16, 1);
	for (std::size_t i = 0; i < 16; i++) {
		X(i, 0) = x[i];
	}
	double ybar = 0.0;
	for (const double v : y) {
		ybar += v;
	}
	ybar /= static_cast<double>(y.size());

	for (const statsduck::Link l : {statsduck::Link::kProbit, statsduck::Link::kCLogLog}) {
		statsduck::GlmOptions bo;
		bo.family = statsduck::Family::kBinomial;
		bo.link = l;
		const statsduck::GlmResult g = statsduck::fit_glm(y, X, bo);
		check(g.ok && g.converged, "the fit converges");
		check(g.deviance < g.null_deviance, "and explains something, so its deviance is below the "
		                                    "null model's");
		// The score equations hold at the answer. What this catches is a
		// loop that stops short of its own fixed point, or one whose
		// working response or working weight is mis-assembled: those
		// change which equation the loop solves, and this measures the
		// true one. It does NOT catch a wrong link_deriv, because a
		// constant factor there divides out of the fixed point and the
		// check uses the same function the loop does. test_glm_core's
		// central difference covers the derivative itself.
		const double score = worst_score_in_se(g, bo.family, l, y, x);
		check(score < TOL_SCORE, "the reported coefficients satisfy the score equations");

		// An independent oracle for these two links: minimize the same
		// deviance with Nelder-Mead, from a starting point that is not
		// the IRLS answer. The optimizer never calls variance or
		// link_deriv, so it checks the loop against the objective itself
		// rather than against the loop's own algebra. deviance() is the
		// trusted part here: test_glm_core pins it against hand-computed
		// values for all four families.
		const std::vector<double> w(y.size(), 1.0);
		auto dev_at = [&](const std::vector<double> &b) {
			std::vector<double> mu(y.size());
			for (std::size_t i = 0; i < y.size(); i++) {
				mu[i] = statsduck::link_inv(l, b[0] + b[1] * x[i]);
			}
			return statsduck::deviance(bo.family, y, mu, w);
		};
		statsduck::optimize::Options oo;
		oo.max_eval = 20000;
		oo.xtol = 1e-12;
		oo.ftol = 1e-14;
		const statsduck::optimize::Result nm =
		    statsduck::optimize::nelder_mead(dev_at, {0.5, -0.5}, oo);
		check(nm.converged, "the independent minimization converges too");
		// The deviance is flat at its minimum, so it agrees far more
		// tightly than the coefficients do. Both bands are measured and
		// printed below.
		check(close_rel(g.deviance, nm.fx, TOL_NM_DEV), "to the same deviance");
		check(close_rel(at(g.beta, 0), nm.x[0], TOL_NM_COEF), "and the same intercept");
		check(close_rel(at(g.beta, 1), nm.x[1], TOL_NM_COEF), "and the same slope");

		// An intercept-only fit of ANY link must reproduce the mean of y.
		// The score equation for a lone intercept is
		// sum_i (y_i - mu)*d/V(mu) = 0 with mu constant, so the link's
		// own factor divides out and mu = ybar whatever the link is.
		const statsduck::GlmResult only = statsduck::fit_glm(y, Mat(y.size(), 0), bo);
		check(only.ok && only.converged, "so does an intercept-only fit");
		const double mu0 = statsduck::link_inv(l, at(only.beta, 0));
		check(close_rel(mu0, ybar, 1e-9), "whose fitted mean is the mean of y");
		check(close_rel(g.null_deviance, only.deviance, 1e-12),
		      "and whose deviance is the full fit's null_deviance");
		std::printf("  %-7s: dev %.12g, score %.3g se, intercept-only mu - ybar %.3g; against "
		            "nelder-mead: dev %.3g, beta %.3g and %.3g, %d evals\n",
		            statsduck::link_name(l), g.deviance, score, mu0 - ybar,
		            rel_dev(g.deviance, nm.fx), rel_dev(at(g.beta, 0), nm.x[0]),
		            rel_dev(at(g.beta, 1), nm.x[1]), nm.n_eval);
	}
}

// ── The aliased set comes from the positive-weight rows ──────────────────
static void test_zero_weight_rows_do_not_set_the_rank() {
	std::printf("a zero-weight row changes neither the rank nor the support test\n");

	// x1 is zero on every row of positive weight and non-zero only on the
	// row whose weight is 0. Over all five rows the intercept and x1 are
	// independent, so a rank computed from the full design would be 2 and
	// the fit would try to estimate x1 from a row that carries no weight.
	// Xt W X is then singular: its x1 row and column are all zeros, the
	// Cholesky fails, and the inverse behind the standard errors fails
	// with it, so the whole fit comes back as a failure.
	//
	// Restricted to the positive-weight rows, x1 is a zero column. It is
	// dropped, the fit runs on the intercept alone, and x1 is reported
	// aliased. That is the behaviour this checks, and it is why the set
	// is computed from the positive-weight rows.
	const std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 99.0};
	const std::vector<double> w = {1.0, 1.0, 1.0, 1.0, 0.0};
	Mat x(5, 1);
	x(4, 0) = 1.0; // every other entry is 0
	statsduck::GlmOptions go;
	go.family = statsduck::Family::kGaussian;
	const statsduck::GlmResult g = statsduck::fit_glm(y, x, go, &w);
	check(g.ok, "the fit succeeds");
	check(g.k == 2 && g.rank == 1, "x1 is dropped, so the rank is 1 of 2");
	check(std::isnan(at(g.beta, 1)), "and x1 is reported aliased");
	// The intercept is the weighted mean of the four rows that count. The
	// zero-weight row carries y = 99 and would move it to 21.8.
	//
	// Measured 0: the intercept comes back as exactly 2.5. A gaussian
	// identity fit is a one-step weighted solve, the four surviving
	// responses are 1, 2, 3 and 4, and their mean is representable
	// exactly, so no rounding enters. The band is 1e-12 rather than 0 so
	// that a future change in the accumulation order is a reviewable
	// difference rather than a failure here.
	check(close_rel(at(g.beta, 0), 2.5, 1e-12), "the zero-weight row does not move the estimate");
	// n counts every row given to the kernel, including the zero-weight
	// one, so df_residual is 5 - 1. The SQL layer drops such rows at
	// accumulation instead, which is where that decision belongs.
	check(g.n == 5 && g.df_residual == 4, "n counts the row the kernel was given");

	// The non-gaussian sibling, which is the case that matters: the
	// support test on the fitted mean is a no-op for gaussian, where any
	// finite mean is in the support, so the test above does not exercise
	// it at all.
	//
	// Four ordinary rows at weight 1 lie exactly on mu = 10 - 2x, so the
	// poisson identity fit of them is beta = (10, -2). A fifth row
	// carries weight 0 and an extreme covariate, x = 100000, which puts
	// its OWN fitted mean at about -200000: far outside the poisson
	// support, and entirely irrelevant to the fit, because the row
	// influences no estimate.
	//
	// A support test that ignored the prior weight would fail this whole
	// fit on that one row. The two controls below separate the mechanism
	// from the arithmetic: dropping the row, and giving it an ordinary
	// covariate, must both give the same answer.
	const std::vector<double> py = {8.0, 6.0, 4.0, 2.0, 0.0};
	const std::vector<double> pw = {1.0, 1.0, 1.0, 1.0, 0.0};
	Mat px(5, 1);
	px(0, 0) = 1.0;
	px(1, 0) = 2.0;
	px(2, 0) = 3.0;
	px(3, 0) = 4.0;
	px(4, 0) = 100000.0; // the extreme covariate, on the weightless row
	statsduck::GlmOptions po;
	po.family = statsduck::Family::kPoisson;
	po.link = statsduck::Link::kIdentity;
	const statsduck::GlmResult p = statsduck::fit_glm(py, px, po, &pw);
	check(p.ok, "a zero-weight row with an extreme covariate does not fail a poisson fit");
	check(p.converged, "which still converges");
	check(close_rel(at(p.beta, 0), 10.0, 1e-9), "to the intercept its positive-weight rows imply");
	check(close_rel(at(p.beta, 1), -2.0, 1e-9), "and their slope");

	// Control 1: the same four rows with the fifth dropped entirely.
	Mat px4(4, 1);
	for (std::size_t i = 0; i < 4; i++) {
		px4(i, 0) = static_cast<double>(i + 1);
	}
	const statsduck::GlmResult p4 =
	    statsduck::fit_glm({8.0, 6.0, 4.0, 2.0}, px4, po);
	check(p4.ok && close_rel(at(p4.beta, 1), at(p.beta, 1), 1e-9),
	      "dropping the row entirely gives the same slope");

	// Control 2: the same five rows with an ordinary covariate on the
	// weightless one. The band is 1e-9 on all four of these because a
	// poisson identity fit of exactly-on-the-line data converges to the
	// line, and the measured agreement is 1e-15 or better; 1e-9 leaves
	// room for the iteration to stop one step earlier on another
	// platform without ever admitting a different line.
	Mat px_ord = px;
	px_ord(4, 0) = 5.0;
	const statsduck::GlmResult p5 = statsduck::fit_glm(py, px_ord, po, &pw);
	check(p5.ok && close_rel(at(p5.beta, 1), at(p.beta, 1), 1e-9),
	      "and so does an ordinary covariate on it");
	std::printf("  poisson identity, weightless x = 1e5: beta (%.12g, %.12g); row dropped "
	            "(%.12g, %.12g); ordinary covariate (%.12g, %.12g)\n",
	            at(p.beta, 0), at(p.beta, 1), at(p4.beta, 0), at(p4.beta, 1), at(p5.beta, 0),
	            at(p5.beta, 1));
}

// ── A design with no columns at all ──────────────────────────────────────
static void test_zero_column_design() {
	std::printf("a zero-column design is a fit, with or without an intercept\n");

	// A zero-column design is a legitimate input, not misuse. With an
	// intercept the prepended constant is the only column, which is how
	// null_deviance is computed, and the identity test above covers that
	// case. Without an intercept there are no parameters at all, so mu is
	// the link inverse of the offset alone and there is nothing to
	// iterate.
	statsduck::GlmOptions po;
	po.family = statsduck::Family::kPoisson;
	po.link = statsduck::Link::kLog;
	po.intercept = false;
	const std::vector<double> off(kDs3N, 1.0);
	const statsduck::GlmResult g =
	    statsduck::fit_glm(ds3_y, Mat(kDs3N, 0), po, nullptr, &off);
	check(g.ok, "an offset-only fit succeeds");
	check(g.k == 0 && g.rank == 0, "with no parameters");
	check(g.df_residual == kDs3N, "so every row is a residual degree of freedom");
	check(g.iterations == 0 && g.converged, "and nothing to iterate");
	check(g.terms.empty() && g.beta.empty() && g.cov.empty(), "no terms and no covariance");

	// mu is exactly link_inv(offset), so the deviance is the closed form
	// at that one fitted value. e^1 for every row here.
	//
	// Both bands are measured 0: with no parameters the loop does not
	// iterate at all, so the kernel evaluates the same deviance call on
	// the same fitted values this line does, and its own null model is
	// this same model. The band is 1e-12 rather than 0 for the same
	// reason as above, to keep a future ordering change reviewable.
	const std::vector<double> mu(kDs3N, statsduck::link_inv(statsduck::Link::kLog, 1.0));
	const std::vector<double> w(kDs3N, 1.0);
	check(close_rel(g.deviance, statsduck::deviance(statsduck::Family::kPoisson, ds3_y, mu, w),
	                1e-12),
	      "and the deviance is the one at link_inv(offset)");
	check(close_rel(g.null_deviance, g.deviance, 1e-12),
	      "its own null model is the same model, so null_deviance matches");
}

int main() {
	std::printf("== test_glm_fit ==\n");
	test_gaussian_equals_lm_fit();
	test_robust_equals_lm_fit();
	test_oracles();
	test_robust_oracles();
	test_binomial_hc_against_r();
	test_weighted_gaussian_loglik_against_r();
	test_canonical_gamma_and_sqrt_links_against_r();
	test_weighted_hat_diagonal();
	test_finite_sample_factors_use_rank();
	test_weighted_sandwich();
	test_aliased_under_every_estimator();
	test_cluster_id_convention();
	test_identities();
	test_fitted_mean_stays_in_support();
	test_step_halving();
	test_probit_and_cloglog();
	test_separation();
	test_no_residual_df();
	test_no_variation_response();
	test_zero_weight_rows_do_not_set_the_rank();
	test_zero_column_design();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL GLM_FIT TESTS PASSED\n");
		return 0;
	}
	return 1;
}
