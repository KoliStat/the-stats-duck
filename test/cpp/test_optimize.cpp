// Direct C++ unit tests for the statsduck::optimize kernel.
//
// Tests the header-only, DuckDB-free AND Eigen-free Nelder-Mead minimizer in
// src/include/optimize.hpp on classic optimization test functions with known
// minima — same standalone harness as test_linalg / test_lm_fit (exit 0 = all
// pass, built by scripts/run-cpp-tests.sh).

#include "optimize.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace O = statsduck::optimize;

static int g_checks = 0;
static int g_fail = 0;

static void check(bool cond, const char *expr, int line) {
	++g_checks;
	if (!cond) {
		++g_fail;
		std::printf("  FAIL (line %d): %s\n", line, expr);
	}
}
static void check_close(double a, double b, double tol, const char *expr, int line) {
	++g_checks;
	const double d = std::fabs(a - b);
	if (!(d <= tol)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : |%.12g - %.12g| = %.3g > %.3g\n", line, expr, a, b, d, tol);
	}
}
#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_CLOSE(a, b, tol) check_close((a), (b), (tol), #a " ~= " #b, __LINE__)

static void test_quadratic_1d() {
	std::printf("quadratic_1d\n");
	// f(x) = 2(x-3)^2 + 7 : exact minimum x=3, f=7. n=1 (two-point simplex) is
	// the REML spike's actual use case.
	auto r = O::nelder_mead([](const std::vector<double> &x) {
		return 2.0 * (x[0] - 3.0) * (x[0] - 3.0) + 7.0;
	}, {0.0});
	CHECK(r.converged);
	CHECK_CLOSE(r.x[0], 3.0, 1e-5);
	CHECK_CLOSE(r.fx, 7.0, 1e-9);
	CHECK(r.n_eval > 0 && r.n_iter > 0);
}

static void test_sphere_5d() {
	std::printf("sphere_5d\n");
	auto r = O::nelder_mead([](const std::vector<double> &x) {
		double s = 0;
		for (double v : x) s += v * v;
		return s;
	}, {1.0, 2.0, 3.0, 4.0, 5.0});
	CHECK(r.converged);
	for (int j = 0; j < 5; ++j) CHECK_CLOSE(r.x[j], 0.0, 1e-4);
	CHECK_CLOSE(r.fx, 0.0, 1e-8);
}

static void test_rosenbrock_2d() {
	std::printf("rosenbrock_2d\n");
	// Classic start (-1.2, 1); minimum (1, 1), f=0.
	auto rosen = [](const std::vector<double> &x) {
		const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
		return a * a + 100.0 * b * b;
	};
	auto r = O::nelder_mead(rosen, {-1.2, 1.0});
	CHECK(r.converged);
	CHECK_CLOSE(r.x[0], 1.0, 1e-4);
	CHECK_CLOSE(r.x[1], 1.0, 1e-4);
	CHECK_CLOSE(r.fx, 0.0, 1e-8);
}

static void test_beale_2d() {
	std::printf("beale_2d\n");
	// Beale: minimum (3, 0.5), f=0. Start (1, 1).
	auto beale = [](const std::vector<double> &x) {
		const double t1 = 1.5 - x[0] + x[0] * x[1];
		const double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
		const double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
		return t1 * t1 + t2 * t2 + t3 * t3;
	};
	auto r = O::nelder_mead(beale, {1.0, 1.0});
	CHECK(r.converged);
	CHECK_CLOSE(r.x[0], 3.0, 1e-3);
	CHECK_CLOSE(r.x[1], 0.5, 1e-3);
	CHECK_CLOSE(r.fx, 0.0, 1e-8);
}

static void test_nan_region() {
	std::printf("nan_region\n");
	// f is NaN for x<1 (invalid region), minimum at x=1.1 NEAR the wall.
	// Descending from x0=3 the simplex probes below 1; NaN must act as +inf
	// (repel) rather than poison the ordering. This is the REML-readiness test:
	// likelihood surfaces return NaN off their valid domain.
	int nan_hits = 0;
	auto f = [&](const std::vector<double> &x) {
		if (x[0] < 1.0) {
			++nan_hits;
			return std::numeric_limits<double>::quiet_NaN();
		}
		return (x[0] - 1.1) * (x[0] - 1.1);
	};
	auto r = O::nelder_mead(f, {3.0});
	CHECK(r.converged);
	CHECK_CLOSE(r.x[0], 1.1, 1e-5);
	CHECK_CLOSE(r.fx, 0.0, 1e-9);
	CHECK(nan_hits > 0); // the wall was actually probed
}

static void test_max_eval_exhaustion() {
	std::printf("max_eval_exhaustion\n");
	O::Options opts;
	opts.max_eval = 10; // nowhere near enough for Rosenbrock
	auto r = O::nelder_mead([](const std::vector<double> &x) {
		const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
		return a * a + 100.0 * b * b;
	}, {-1.2, 1.0}, opts);
	CHECK(!r.converged);
	CHECK(r.n_eval <= 10 + 3); // may finish the in-flight iteration step
	CHECK(r.x.size() == 2);    // still returns the best point found
}

static void test_degenerate_inputs() {
	std::printf("degenerate_inputs\n");
	auto f = [](const std::vector<double> &x) { return x.empty() ? 0.0 : x[0]; };
	// Empty x0: converged=false, x echoed, fx=NaN, zero evaluations.
	auto r0 = O::nelder_mead(f, {});
	CHECK(!r0.converged);
	CHECK(r0.x.empty());
	CHECK(std::isnan(r0.fx));
	CHECK(r0.n_eval == 0);
	// max_eval <= 0: same contract.
	O::Options bad;
	bad.max_eval = 0;
	auto r1 = O::nelder_mead(f, {1.0}, bad);
	CHECK(!r1.converged);
	CHECK(r1.x.size() == 1 && r1.x[0] == 1.0);
	CHECK(std::isnan(r1.fx));
	CHECK(r1.n_eval == 0);
}

static void test_determinism() {
	std::printf("determinism\n");
	auto beale = [](const std::vector<double> &x) {
		const double t1 = 1.5 - x[0] + x[0] * x[1];
		const double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
		const double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
		return t1 * t1 + t2 * t2 + t3 * t3;
	};
	auto a = O::nelder_mead(beale, {1.0, 1.0});
	auto b = O::nelder_mead(beale, {1.0, 1.0});
	CHECK(a.x[0] == b.x[0]); // bit-identical, not merely close
	CHECK(a.x[1] == b.x[1]);
	CHECK(a.fx == b.fx);
	CHECK(a.n_eval == b.n_eval);
	CHECK(a.n_iter == b.n_iter);
}

int main() {
	test_quadratic_1d();
	test_sphere_5d();
	test_rosenbrock_2d();
	test_beale_2d();
	test_nan_region();
	test_max_eval_exhaustion();
	test_degenerate_inputs();
	test_determinism();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL OPTIMIZE TESTS PASSED\n");
		return 0;
	}
	return 1;
}
