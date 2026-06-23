// Direct C++ unit tests for the statsduck::linalg kernel (Epic 0.1).
//
// Tests the Eigen-FREE public API in src/include/linalg.hpp against offline
// golden values (small hand-verifiable systems + known matrix identities) — the
// same golden-value + tolerance-band discipline used by the r*/bootstrap SQL
// tests. Builds standalone (no DuckDB) via scripts/run-linalg-tests.sh.
//
// Exit 0 = all pass.

#include "linalg.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace L = statsduck::linalg;
using L::Mat;

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

static const double TOL = 1e-9;

// Build a row-major Mat from a flat value list.
static Mat mat(std::size_t r, std::size_t c, std::vector<double> vals) {
	Mat m(r, c);
	m.data = std::move(vals);
	return m;
}

static void test_cholesky_solve() {
	std::printf("cholesky_solve\n");
	// SPD A=[[4,2],[2,3]], x_true=[1,1] -> b = A x = [6,5].
	const Mat A = mat(2, 2, {4, 2, 2, 3});
	auto s = L::cholesky_solve(A, {6, 5});
	CHECK(s.ok);
	CHECK(s.rank == 2);
	CHECK_CLOSE(s.x[0], 1.0, TOL);
	CHECK_CLOSE(s.x[1], 1.0, TOL);

	// Non-SPD (indefinite) -> ok=false.
	auto bad = L::cholesky_solve(mat(2, 2, {1, 2, 2, 1}), {1, 1});
	CHECK(!bad.ok);

	// Dimension mismatch -> ok=false.
	CHECK(!L::cholesky_solve(A, {1, 2, 3}).ok);
}

static void test_qr_solve() {
	std::printf("qr_solve\n");
	// Overdetermined exact fit: y = 1 + 1*x at x=0,1,2 -> y=1,2,3.
	const Mat A = mat(3, 2, {1, 0, 1, 1, 1, 2});
	auto s = L::qr_solve(A, {1, 2, 3});
	CHECK(s.ok);
	CHECK_CLOSE(s.x[0], 1.0, TOL);
	CHECK_CLOSE(s.x[1], 1.0, TOL);

	// Square full-rank system = exact solve.
	auto sq = L::qr_solve(mat(2, 2, {4, 2, 2, 3}), {6, 5});
	CHECK_CLOSE(sq.x[0], 1.0, TOL);
	CHECK_CLOSE(sq.x[1], 1.0, TOL);
}

static void test_svd_solve() {
	std::printf("svd_solve\n");
	// Full rank -> same as QR.
	auto full = L::svd_solve(mat(3, 2, {1, 0, 1, 1, 1, 2}), {1, 2, 3});
	CHECK(full.ok);
	CHECK(full.rank == 2);
	CHECK_CLOSE(full.x[0], 1.0, TOL);
	CHECK_CLOSE(full.x[1], 1.0, TOL);

	// Rank-deficient A=[[1,1],[1,1]], b=[2,2]: minimum-norm solution is [1,1].
	auto def = L::svd_solve(mat(2, 2, {1, 1, 1, 1}), {2, 2});
	CHECK(def.ok);
	CHECK(def.rank == 1);
	CHECK_CLOSE(def.x[0], 1.0, TOL);
	CHECK_CLOSE(def.x[1], 1.0, TOL);
}

static void test_rank() {
	std::printf("rank\n");
	CHECK(L::rank(mat(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1})) == 3); // identity
	CHECK(L::rank(mat(2, 2, {1, 2, 2, 4})) == 1);                // rank-1
	CHECK(L::rank(mat(2, 2, {0, 0, 0, 0})) == 0);                // zero
}

static void test_pinv() {
	std::printf("pinv\n");
	// Invertible A=[[4,2],[2,3]] -> pinv = inv = (1/8)[[3,-2],[-2,4]].
	auto p = L::pinv(mat(2, 2, {4, 2, 2, 3}));
	CHECK(p.rank == 2);
	CHECK(p.value.rows == 2 && p.value.cols == 2);
	CHECK_CLOSE(p.value(0, 0), 0.375, TOL);
	CHECK_CLOSE(p.value(0, 1), -0.25, TOL);
	CHECK_CLOSE(p.value(1, 0), -0.25, TOL);
	CHECK_CLOSE(p.value(1, 1), 0.5, TOL);

	// Non-square 3x2 with orthonormal columns -> pinv 2x3 = [[1,0,0],[0,1,0]].
	auto q = L::pinv(mat(3, 2, {1, 0, 0, 1, 0, 0}));
	CHECK(q.rank == 2);
	CHECK(q.value.rows == 2 && q.value.cols == 3);
	CHECK_CLOSE(q.value(0, 0), 1.0, TOL);
	CHECK_CLOSE(q.value(0, 1), 0.0, TOL);
	CHECK_CLOSE(q.value(0, 2), 0.0, TOL);
	CHECK_CLOSE(q.value(1, 0), 0.0, TOL);
	CHECK_CLOSE(q.value(1, 1), 1.0, TOL);
	CHECK_CLOSE(q.value(1, 2), 0.0, TOL);

	// Rank-deficient pinv: rank reported correctly.
	auto r = L::pinv(mat(2, 2, {1, 1, 1, 1}));
	CHECK(r.rank == 1);
}

static void test_inv_spd() {
	std::printf("inv_spd\n");
	bool ok = false;
	Mat inv = L::inv_spd(mat(2, 2, {4, 2, 2, 3}), &ok);
	CHECK(ok);
	CHECK_CLOSE(inv(0, 0), 0.375, TOL);
	CHECK_CLOSE(inv(0, 1), -0.25, TOL);
	CHECK_CLOSE(inv(1, 0), -0.25, TOL);
	CHECK_CLOSE(inv(1, 1), 0.5, TOL);

	// Non-SPD -> ok=false.
	bool ok2 = true;
	L::inv_spd(mat(2, 2, {1, 2, 2, 1}), &ok2);
	CHECK(!ok2);
}

static void test_sandwich() {
	std::printf("sandwich\n");
	// L=[[1,0]] (1x2), A=[[4,2],[2,3]] -> L A L^T = [[4]].
	Mat s = L::sandwich(mat(1, 2, {1, 0}), mat(2, 2, {4, 2, 2, 3}));
	CHECK(s.rows == 1 && s.cols == 1);
	CHECK_CLOSE(s(0, 0), 4.0, TOL);

	// L = I2 -> sandwich = A.
	Mat id = L::sandwich(mat(2, 2, {1, 0, 0, 1}), mat(2, 2, {4, 2, 2, 3}));
	CHECK_CLOSE(id(0, 0), 4.0, TOL);
	CHECK_CLOSE(id(0, 1), 2.0, TOL);
	CHECK_CLOSE(id(1, 0), 2.0, TOL);
	CHECK_CLOSE(id(1, 1), 3.0, TOL);
}

static void test_matvec() {
	std::printf("matvec\n");
	// A=[[1,2],[3,4]], x=[1,1] -> [3,7].
	auto y = L::matvec(mat(2, 2, {1, 2, 3, 4}), {1, 1});
	CHECK(y.size() == 2);
	CHECK_CLOSE(y[0], 3.0, TOL);
	CHECK_CLOSE(y[1], 7.0, TOL);

	// Dimension mismatch -> empty.
	CHECK(L::matvec(mat(2, 2, {1, 2, 3, 4}), {1, 2, 3}).empty());
}

int main() {
	test_cholesky_solve();
	test_qr_solve();
	test_svd_solve();
	test_rank();
	test_pinv();
	test_inv_spd();
	test_sandwich();
	test_matvec();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL LINALG TESTS PASSED\n");
		return 0;
	}
	return 1;
}
