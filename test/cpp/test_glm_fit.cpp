// Direct C++ unit tests for statsduck::fit_glm, the IRLS loop and its
// classical covariance (#22).
//
// The first test compares a gaussian identity-link fit against fit_lm on
// the same data. Gaussian with an identity link IS ordinary least squares,
// so the two fitters must agree exactly. That test comes first because it
// catches the most for the least: fit_lm is already validated against
// statsmodels, so this ties the new fitter to a checked one instead of to
// a fresh set of constants.
//
// The oracle comparisons come next, against the statsmodels goldens in
// test/cpp/glm_fit_fixtures.hpp: DS1 binomial, DS3 poisson with an
// offset, DS4 gamma, DS5 gaussian and DS7 rank deficient. Their band is
// MEASURED rather than assumed, and the measurement is recorded beside
// it and reprinted on every run.
//
// Then the identities that need no oracle at all: null_deviance against
// an independent intercept-only fit, grouped binomial against ungrouped,
// a scaled trial count, and the aliasing of a dependent column. Last, the
// two input classes that have no fit to report: no residual degrees of
// freedom, and a design with no columns at all.
//
// DS2, DS6 and DS8 are not here. DS2's numbers are used, as the grouped
// half of the grouping identity. DS6 is the separated design, whose
// goldens the fixture header marks as unsafe to assert tightly, and it
// belongs to the task that adds the separation flag. DS8 is clustered and
// belongs to the task that adds the robust covariance.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "glm_core.hpp"

#include "glm_fit_fixtures.hpp"

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
static const std::vector<double> ds7_y = vec(kDs7Y);

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
static Mat ds7_design() {
	return design(kDs7N, {kDs7X1, kDs7X2, kDs7X3});
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
	std::printf("  worst relative gap against fit_lm: %.3g (%d iterations)\n", worst, g.iterations);
	check(g.converged, "an identity-link gaussian fit converges");
	check(g.iterations <= 3, "and does so in very few iterations, since it is linear");
	check(!g.separation, "separation is false for a gaussian fit");
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
// the loop over DS1, DS3, DS4, DS5 and DS7 and printing the worst
// relative deviation on every reported quantity. The suite still prints
// those numbers on every run, on the TOL_MEASURED line, so a change in
// them is visible rather than silent.
//
// Measured worst cases, 2026-10-21:
//   beta 2.56e-07 (DS4)   std_error 1.95e-06 (DS7)   z 1.95e-06 (DS7)
//   cov  6.00e-06 (DS1)   deviance and null_deviance 9.98e-15 (DS4)
//   dispersion, loglik, aic 2.94e-08 (DS4)   p-value 8.88e-07 absolute (DS7)
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
};

static void test_oracles() {
	std::printf("statsmodels goldens: DS1, DS3, DS4, DS5, DS7\n");

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

// ── The identities that need no oracle ───────────────────────────────────
static void test_identities() {
	std::printf("oracle-free identities: the null model, grouping, scaling, rank deficiency\n");

	statsduck::GlmOptions nb;
	nb.family = statsduck::Family::kBinomial;
	nb.link = statsduck::Link::kLogit;

	// null_deviance is the intercept-only model's deviance, computed two
	// ways: inside the full fit, and by fitting that model here. A single
	// golden number would not catch a wrong null model.
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
	test_oracles();
	test_identities();
	test_no_residual_df();
	test_zero_column_design();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL GLM_FIT TESTS PASSED\n");
		return 0;
	}
	return 1;
}
