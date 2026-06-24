// Direct C++ unit tests for the statsduck::fit_lm regression core (Epic 1.1).
//
// Tests the DuckDB-free OLS + HC0/HC1/HC2/HC3 implementation in
// src/include/lm_core.hpp against golden values produced offline by statsmodels
// 0.14.x (see test/cpp/gen_lm_fit_fixtures.py). statsmodels is the oracle:
// matching its cov_type='HC*' output validates OUR formulas, not just a
// re-derivation of them. Same golden-value + tolerance-band discipline as
// test_linalg.cpp / the r*-distribution SQL tests. Builds standalone (no DuckDB)
// via scripts/run-linalg-tests.sh.
//
// DS1 carries a high-leverage point (x1 = 25), so HC0→HC3 SEs diverge sharply
// (x1's SE runs 0.126 → 0.145 → 0.234 → 0.475) — a bug in any single leverage
// weighting fails loudly here.
//
// Exit 0 = all pass.

#include "lm_core.hpp"

#include <cmath>
#include <cstdio>
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
static const double TOL = 1e-6;     // β, SE, σ, R², adj-R²
static const double TOL_F = 1e-3;   // F-statistic
static const double RTOL_P = 1e-2;  // p-values (relative)

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

static LmResult fit(const std::vector<double> &y, const Mat &X, Vcov v, bool intercept) {
	LmOptions opt;
	opt.vcov = v;
	opt.intercept = intercept;
	return fit_lm(y, X, opt);
}

// ───────────────────────── DS1: intercept, 2 predictors, n=12 ────────────────
// One high-leverage point (x1 = 25). Golden source: gen_lm_fit_fixtures.py.
static void test_ds1_hetero() {
	std::printf("ds1_hetero (intercept, 2 predictors, n=12, high-leverage)\n");
	const std::vector<double> y = {9, 8, 7, 14, 12, 20, 15, 28, 22, 31, 29, 55};
	const Mat X = predictors({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25},
	                          {5, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8, 4}});

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
	}
	{
		auto r = fit(y, X, Vcov::kHC1, true);
		check_vec(r.std_error, {0.111185996792465, 0.0229806595898712}, TOL, "ds3.hc1.se", __LINE__);
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
	// Perfectly collinear predictors (x2 = 2·x1) with intercept → singular XᵀX.
	{
		const std::vector<double> y = {1, 3, 2, 5, 4, 7};
		const Mat X = predictors({{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}});
		auto r = fit(y, X, Vcov::kConst, true);
		CHECK(!r.ok);
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
	CHECK(!statsduck::parse_vcov("HC4", v));
	CHECK(!statsduck::parse_vcov("bogus", v));
	CHECK(std::string(statsduck::vcov_name(Vcov::kConst)) == "const");
	CHECK(std::string(statsduck::vcov_name(Vcov::kHC0)) == "HC0");
	CHECK(std::string(statsduck::vcov_name(Vcov::kHC3)) == "HC3");
}

int main() {
	std::printf("== test_lm_fit ==\n");
	test_ds1_hetero();
	test_ds2_noint();
	test_ds3_simple();
	test_errors();
	test_vcov_parse();
	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	return g_fail == 0 ? 0 : 1;
}
