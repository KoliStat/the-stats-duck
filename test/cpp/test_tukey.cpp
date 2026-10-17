// Direct C++ unit tests for the studentized range distribution (#21):
// stats_duck::StudentizedRangeCDF and stats_duck::StudentizedRangeQuantile,
// the distribution behind Tukey's multiplicity adjustment.
//
// The identity tests come first and the fixture tests after, because a
// failure in an identity says the quadrature is wrong while a fixture failure
// only says a number moved. Three of the four identities check the new code
// against code this repository has already validated: StudentTCDF for k = 2,
// NormalCDF for the large-nu limit, and the CDF itself for the round trip.
//
// The fixtures in tukey_fixtures.hpp come from scipy.stats.studentized_range
// (BSD), generated offline by test/cpp/gen_tukey_fixtures.py. R's ptukey is
// GPL and was not used.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "distributions.hpp"
#include "tukey_fixtures.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

static int g_checks = 0;
static int g_fail = 0;

static void check(bool cond, const char *msg) {
	++g_checks;
	if (!cond) {
		++g_fail;
		std::printf("  FAIL: %s\n", msg);
	}
}

// True when a and b are within tol of each other.
static bool close(double a, double b, double tol) {
	return std::fabs(a - b) <= tol;
}

// ── Identities ──────────────────────────────────────────────────────────────

static void test_identities() {
	std::printf("identities (t at k = 2, the normal range, the round trip, monotonicity)\n");

	// Identity 1: at k = 2 the studentized range is the t distribution exactly.
	// Q = sqrt(2)*|T|, so P(Q <= q) = 2*F_t(q/sqrt(2)) - 1. This anchors the
	// whole double integral against StudentTCDF, which is already validated.
	double worst_t = 0.0;
	const double nus[] = {1.0, 2.0, 5.0, 10.0, 30.0, 120.0, 1000.0};
	for (const double nu : nus) {
		for (const double q : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}) {
			const double want = 2.0 * stats_duck::StudentTCDF(q / std::sqrt(2.0), nu) - 1.0;
			const double got = stats_duck::StudentizedRangeCDF(q, 2.0, nu);
			worst_t = std::fmax(worst_t, std::fabs(got - want));
			check(close(got, want, 1e-8), "k = 2 reduces to the t distribution");
		}
	}
	std::printf("  worst |ptukey(q,2,nu) - (2*pt(q/sqrt(2),nu) - 1)| = %.3e\n", worst_t);

	// Identity 2: as nu grows the outer integral collapses to the normal range,
	// and at k = 2 that is 2*Phi(q/sqrt(2)) - 1.
	//
	// nu = 1e9 is below the degrees of freedom at which the implementation
	// replaces the outer integral by that limit, so this runs the quadrature
	// and checks that it converges to the limit on its own. nu = 1e12 is above
	// it, so the same check also pins the replacement branch, which is the
	// inner integral alone, against NormalCDF.
	double worst_z = 0.0;
	for (const double nu : {1e9, 1e12}) {
		for (const double q : {1.0, 2.0, 4.0, 6.0}) {
			const double want = 2.0 * stats_duck::NormalCDF(q / std::sqrt(2.0)) - 1.0;
			const double got = stats_duck::StudentizedRangeCDF(q, 2.0, nu);
			worst_z = std::fmax(worst_z, std::fabs(got - want));
			check(close(got, want, 1e-7), "large nu reduces to the normal range");
		}
	}
	std::printf("  worst |ptukey(q,2,nu) - (2*pnorm(q/sqrt(2)) - 1)| at nu 1e9, 1e12 = %.3e\n",
	            worst_z);

	// Identity 3: the quantile inverts the CDF.
	double worst_rt = 0.0;
	for (const double k : {2.0, 3.0, 5.0, 10.0}) {
		for (const double nu : {5.0, 20.0, 60.0}) {
			for (const double p : {0.5, 0.9, 0.95, 0.99}) {
				const double q = stats_duck::StudentizedRangeQuantile(p, k, nu);
				const double back = stats_duck::StudentizedRangeCDF(q, k, nu);
				worst_rt = std::fmax(worst_rt, std::fabs(back - p));
				check(close(back, p, 1e-8), "quantile inverts the CDF");
			}
		}
	}
	std::printf("  worst |ptukey(qtukey(p,k,nu),k,nu) - p| = %.3e\n", worst_rt);

	// Identity 4: monotone increasing in q, decreasing in k.
	for (const double nu : {5.0, 30.0}) {
		double prev = -1.0;
		for (const double q : {0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0}) {
			const double cur = stats_duck::StudentizedRangeCDF(q, 4.0, nu);
			check(cur > prev, "CDF increases in q");
			prev = cur;
		}
		double last = 2.0;
		for (const double k : {2.0, 3.0, 4.0, 6.0, 10.0, 20.0}) {
			const double cur = stats_duck::StudentizedRangeCDF(3.0, k, nu);
			check(cur < last, "CDF decreases in k");
			last = cur;
		}
	}
}

// ── Fixtures ────────────────────────────────────────────────────────────────

static void test_fixtures() {
	std::printf("fixtures against scipy.stats.studentized_range, %d cases each way\n",
	            static_cast<int>(sizeof(kTukeyCdf) / sizeof(kTukeyCdf[0])));

	// The CDF band is 1e-8 absolute: this is fixed-node Gauss-Legendre on a
	// smooth, exponentially decaying integrand; the error is set by the
	// truncation point and the node count, both fixed, and scipy's own
	// implementation targets about 1e-10.
	double worst_cdf = 0.0;
	double worst_cdf_k = 0.0, worst_cdf_nu = 0.0, worst_cdf_q = 0.0;
	for (const TukeyCdfCase &c : kTukeyCdf) {
		const double got = stats_duck::StudentizedRangeCDF(c.q, c.k, c.nu);
		const double err = std::fabs(got - c.cdf);
		if (err > worst_cdf) {
			worst_cdf = err;
			worst_cdf_k = c.k;
			worst_cdf_nu = c.nu;
			worst_cdf_q = c.q;
		}
		check(close(got, c.cdf, 1e-8), "CDF matches the oracle");
	}
	std::printf("  worst CDF error %.3e absolute at k = %g, nu = %g, q = %.6g\n", worst_cdf,
	            worst_cdf_k, worst_cdf_nu, worst_cdf_q);

	// The quantile band is 1e-6 relative: it inherits the CDF error divided by
	// the density at the root, and that density is small in the upper tail, so
	// the same absolute error becomes a larger relative one. Relative, not
	// absolute, because q runs from 1.1 to 298 across this grid.
	double worst_q = 0.0;
	double worst_q_k = 0.0, worst_q_nu = 0.0, worst_q_p = 0.0;
	for (const TukeyQuantileCase &c : kTukeyQuantile) {
		const double got = stats_duck::StudentizedRangeQuantile(c.p, c.k, c.nu);
		const double err = std::fabs(got - c.q) / c.q;
		if (err > worst_q) {
			worst_q = err;
			worst_q_k = c.k;
			worst_q_nu = c.nu;
			worst_q_p = c.p;
		}
		check(err <= 1e-6, "quantile matches the oracle");
	}
	std::printf("  worst quantile error %.3e relative at k = %g, nu = %g, p = %g\n", worst_q,
	            worst_q_k, worst_q_nu, worst_q_p);
}

// ── A df that is not an integer ─────────────────────────────────────────────

static void test_non_integer_df() {
	std::printf("a df that is not an integer\n");

	// The committed grid is all integer df, and the k = 2 identity holds at any
	// df, so it is the check for the rest. At 4 and above it holds to the same
	// band as everywhere else.
	double worst_high = 0.0;
	for (const double nu : {4.5, 6.5, 12.7, 45.3, 1999.5}) {
		for (const double q : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}) {
			const double want = 2.0 * stats_duck::StudentTCDF(q / std::sqrt(2.0), nu) - 1.0;
			const double got = stats_duck::StudentizedRangeCDF(q, 2.0, nu);
			worst_high = std::fmax(worst_high, std::fabs(got - want));
			check(close(got, want, 1e-8), "non-integer df of 4 or more reduces to the t distribution");
		}
	}
	std::printf("  worst at df >= 4 (not an integer): %.3e\n", worst_high);

	// Below 4 it does not, and that is a known limitation of this quadrature,
	// recorded in the comment on StudentizedRangeCDF: the density carries
	// s^(df-1), whose branch point at s = 0 sits against the lower end of the
	// integration range unless df is an integer. The deviation is printed
	// rather than asserted against a band, because a band here would be a
	// record of the defect and not a target. What is asserted below holds
	// whatever the quadrature error is.
	for (const double nu : {0.5, 1.5, 2.5, 3.5}) {
		double worst = 0.0;
		for (const double q : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}) {
			const double want = 2.0 * stats_duck::StudentTCDF(q / std::sqrt(2.0), nu) - 1.0;
			worst = std::fmax(worst, std::fabs(stats_duck::StudentizedRangeCDF(q, 2.0, nu) - want));
		}
		std::printf("  df = %4g: worst deviation from the t identity %.3e\n", nu, worst);
	}

	// A probability, increasing in q, invertible: these hold at every df and
	// are what stops a silent regression from turning the small-df case from
	// inaccurate into nonsense.
	for (const double nu : {0.5, 1.5, 2.5, 3.5, 4.5, 12.7}) {
		for (const double k : {2.0, 5.0, 20.0}) {
			double prev = -1.0;
			for (const double q : {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0}) {
				const double p = stats_duck::StudentizedRangeCDF(q, k, nu);
				check(p >= 0.0 && p <= 1.0, "a df that is not an integer still gives a probability");
				check(p > prev, "a df that is not an integer is still increasing in q");
				prev = p;
			}
			for (const double p : {0.5, 0.95}) {
				const double q = stats_duck::StudentizedRangeQuantile(p, k, nu);
				check(close(stats_duck::StudentizedRangeCDF(q, k, nu), p, 1e-8),
				      "the quantile still inverts the CDF at a df that is not an integer");
			}
		}
	}
}

// ── Domain and edge values ──────────────────────────────────────────────────

static void test_domain() {
	std::printf("domain (q <= 0, k < 2, df <= 0, NaN, saturation, p at the ends)\n");

	const double nan = std::numeric_limits<double>::quiet_NaN();
	check(stats_duck::StudentizedRangeCDF(0.0, 3.0, 10.0) == 0.0, "q = 0 gives 0");
	check(stats_duck::StudentizedRangeCDF(-1.0, 3.0, 10.0) == 0.0, "negative q gives 0");
	check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, 1.0, 10.0)), "k < 2 is NaN");
	check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, 3.0, 0.0)), "non-positive df is NaN");
	check(std::isnan(stats_duck::StudentizedRangeCDF(nan, 3.0, 10.0)), "NaN q is NaN");
	check(stats_duck::StudentizedRangeCDF(1e300, 3.0, 10.0) == 1.0, "huge q saturates at 1");

	check(stats_duck::StudentizedRangeQuantile(0.0, 3.0, 10.0) == 0.0, "p = 0 is the support floor");
	check(std::isinf(stats_duck::StudentizedRangeQuantile(1.0, 3.0, 10.0)), "p = 1 is +inf");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(-0.1, 3.0, 10.0)), "p below 0 is NaN");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(1.1, 3.0, 10.0)), "p above 1 is NaN");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, 1.0, 10.0)), "k < 2 is NaN");

	// The same arguments the CDF rejects, rejected by the quantile too, plus
	// NaN in each slot. Each of these must return in bounded time: a hang here
	// is a failure, not a slow test.
	check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, 3.0, 0.0)),
	      "quantile: non-positive df is NaN");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(nan, 3.0, 10.0)), "quantile: NaN p is NaN");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, nan, 10.0)), "quantile: NaN k is NaN");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, 3.0, nan)), "quantile: NaN df is NaN");
	check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, nan, 10.0)), "CDF: NaN k is NaN");
	check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, 3.0, nan)), "CDF: NaN df is NaN");
	check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, 3.0, -5.0)), "CDF: negative df is NaN");

	// An infinite df is the normal range: s is then 1 with probability 1.
	const double inf = std::numeric_limits<double>::infinity();
	check(stats_duck::StudentizedRangeCDF(3.0, 3.0, inf) == stats_duck::NormalRangeCDF(3.0, 3.0),
	      "an infinite df is the normal range exactly");

	// An argument the CDF cannot compute must not come back from the quantile as
	// a number. Both of these make the CDF NaN, and without the NaN check in
	// the quantile its bracket would collapse and it would return about 0.
	check(std::isnan(stats_duck::StudentizedRangeCDF(3.0, inf, 10.0)), "CDF: an infinite k is NaN");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, inf, 10.0)),
	      "quantile: an infinite k is NaN, not a number near zero");
	check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, 3.0, 1e-300)),
	      "quantile: a df the quadrature underflows on is NaN, not a number near zero");
}

int main() {
	std::printf("== test_tukey ==\n");
	test_identities();
	test_fixtures();
	test_non_integer_df();
	test_domain();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL TUKEY TESTS PASSED\n");
		return 0;
	}
	return 1;
}
