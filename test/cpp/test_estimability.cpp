// Direct C++ unit tests for statsduck::estimability — the aliased-coefficient
// convention #36 introduced, shared by lin_hyp today and emmeans next (#21).
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "estimability.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace linalg = statsduck::linalg;

static int g_checks = 0;
static int g_fail = 0;

static void check(bool cond, const char *msg) {
	++g_checks;
	if (!cond) {
		++g_fail;
		std::printf("  FAIL: %s\n", msg);
	}
}

// Safe accessor for a double vector. A failing call can leave the vector
// empty, and indexing an empty vector directly is undefined behaviour: it
// can crash the binary before any later check gets to report, which turns a
// clear assertion failure into a confusing one. Out of range reads back as
// NaN, which no real covariance entry equals.
static double at(const std::vector<double> &v, std::size_t i) {
	return i < v.size() ? v[i] : std::numeric_limits<double>::quiet_NaN();
}

// Same reasoning for a bool vector. There is no NaN for bool, so out of range
// reads back as false; every check below also checks size() first, so a
// falsely-false read cannot pass a check it should fail.
static bool atb(const std::vector<bool> &v, std::size_t i) {
	return i < v.size() ? static_cast<bool>(v[i]) : false;
}

static void test_aliased_columns_and_zeroing() {
	std::printf("aliased_columns / row_touches_aliased / zero_aliased\n");

	// A 3x3 covariance whose middle coefficient is aliased.
	linalg::Mat V(3, 3);
	for (std::size_t i = 0; i < 3; i++)
		for (std::size_t j = 0; j < 3; j++)
			V(i, j) = 1.0 + static_cast<double>(i * 3 + j);
	V(1, 1) = std::numeric_limits<double>::quiet_NaN();

	const std::vector<bool> a = statsduck::aliased_columns(V);
	check(a.size() == 3, "aliased_columns returns one flag per column");
	check(!atb(a, 0) && atb(a, 1) && !atb(a, 2), "only the NaN diagonal is aliased");

	// A row with no weight on the aliased column is still estimable.
	const double safe[3] = {1.0, 0.0, -1.0};
	const double unsafe[3] = {1.0, 1e-300, 0.0};
	check(!statsduck::row_touches_aliased(safe, 3, a), "zero weight does not touch");
	check(statsduck::row_touches_aliased(unsafe, 3, a), "any non-zero weight touches");

	// Zeroing clears the aliased row and column and leaves the rest alone.
	const linalg::Mat Vz = statsduck::zero_aliased(V, a);
	check(Vz(1, 1) == 0.0 && Vz(0, 1) == 0.0 && Vz(1, 2) == 0.0, "aliased row and column zeroed");
	check(Vz(0, 0) == V(0, 0) && Vz(2, 2) == V(2, 2), "other entries untouched");
	check(std::isfinite(Vz(1, 1)), "no NaN survives in the working copy");

	const std::vector<double> b = {5.0, 7.0, 9.0};
	const std::vector<double> bz = statsduck::zero_aliased(b, a);
	check(bz.size() == 3 && at(bz, 0) == 5.0 && at(bz, 1) == 0.0 && at(bz, 2) == 9.0,
	      "beta aliased entry zeroed");

	// A fully finite covariance aliases nothing.
	linalg::Mat F(2, 2);
	F(0, 0) = 1.0;
	F(0, 1) = 0.0;
	F(1, 0) = 0.0;
	F(1, 1) = 1.0;
	const std::vector<bool> none = statsduck::aliased_columns(F);
	check(!atb(none, 0) && !atb(none, 1), "a finite covariance aliases nothing");
}

int main() {
	std::printf("== test_estimability ==\n");
	test_aliased_columns_and_zeroing();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL ESTIMABILITY TESTS PASSED\n");
		return 0;
	}
	return 1;
}
