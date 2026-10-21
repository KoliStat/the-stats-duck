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
// a scaled trial count, and a deficient fit against the reduced design it
// must reproduce bit for bit. Then the three mechanisms that have no
// golden: the support test on the fitted mean, step halving, and the
// probit and cloglog links, which are checked against an independent
// Nelder-Mead minimization of the same deviance. Last, the input classes
// that have no fit to report: no residual degrees of freedom, a
// zero-weight row, and a design with no columns at all.
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
	test_oracles();
	test_identities();
	test_fitted_mean_stays_in_support();
	test_step_halving();
	test_probit_and_cloglog();
	test_no_residual_df();
	test_zero_weight_rows_do_not_set_the_rank();
	test_zero_column_design();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL GLM_FIT TESTS PASSED\n");
		return 0;
	}
	return 1;
}
