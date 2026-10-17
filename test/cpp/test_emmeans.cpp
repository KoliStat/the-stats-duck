// Direct C++ unit tests for the statsduck::emmeans kernel (#21).
//
// Golden values come from statsmodels and scipy via
// test/cpp/gen_emmeans_fixtures.py, which writes test/cpp/emmeans_fixtures.hpp.
// Each marginal mean is oracled by OLSResults.t_test(L_row) and each pair by
// t_test(L_i - L_j). That second route never forms C = L·V·Lᵀ, so it is
// independent confirmation that sqrt(C_ii + C_jj - 2·C_ij) is the pairwise
// standard error — the one shortcut this kernel takes.
//
// DS1 is three groups with cell-means coding, so every mean is a group mean
// and every standard error is sqrt(s²/n). Those values are asserted by hand
// as well as against the goldens, so a wrong fixture cannot hide a wrong
// kernel. DS2 is rank deficient and its covariance has off-diagonal entries,
// so it covers both the aliased-column rule and the cross term.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "emmeans.hpp"

#include "emmeans_fixtures.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using statsduck::EmmAdjust;
using statsduck::EmmDiff;
using statsduck::EmmeansResult;
using statsduck::EmmMean;
using statsduck::emmeans;
using statsduck::linalg::Mat;

static const double kNaN = std::numeric_limits<double>::quiet_NaN();

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
		std::printf("  FAIL (line %d): %s : |%.17g - %.17g| = %.3g > %.3g\n", line, label, a, b, d, tol);
	}
}
// A p-value's band is relative with an absolute floor. See the band block
// below for why a tail needs both halves.
static void check_p(double a, double b, double rtol, double atol, const char *label, int line) {
	++g_checks;
	const double d = std::fabs(a - b);
	const double allowed = atol + rtol * std::fabs(b);
	if (!(d <= allowed)) {
		++g_fail;
		std::printf("  FAIL (line %d): %s : |%.17g - %.17g| = %.3g > %.3g\n", line, label, a, b, d,
		            allowed);
	}
}
#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_MSG(cond, why) check((cond), (why), __LINE__)
#define CHECK_CLOSE(a, b, tol) check_close((a), (b), (tol), #a " ~= " #b, __LINE__)

// A label carrying the loop index, so a failure inside a fixture loop says
// which case failed and not only which line.
static const char *lbl(const char *what, std::size_t idx) {
	static char buf[160];
	std::snprintf(buf, sizeof(buf), "%s[%d]", what, static_cast<int>(idx));
	return buf;
}

// ─────────────────────────── bounds-safe reads ───────────────────────────────
// A failing call returns empty vectors. Indexing one turns a clear assertion
// failure into a crash that hides which assertion was wrong, which is the
// defect #33, #36 and #20 each hit. These return an all-NaN element instead,
// so the run reports the assertion that found the problem and keeps going.
static EmmMean mean_at(const EmmeansResult &r, std::size_t i) {
	if (i < r.means.size()) {
		return r.means[i];
	}
	EmmMean m;
	m.estimate = m.std_error = m.t_statistic = m.p_value = m.lower_cl = m.upper_cl = kNaN;
	return m;
}
static EmmDiff diff_at(const EmmeansResult &r, std::size_t i) {
	if (i < r.diffs.size()) {
		return r.diffs[i];
	}
	EmmDiff d;
	// A sentinel index, so an i/j assertion fails instead of matching 0 by luck.
	d.i = d.j = static_cast<std::size_t>(-1);
	d.estimate = d.std_error = d.t_statistic = d.p_value = d.p_adjusted = kNaN;
	d.lower_cl = d.upper_cl = kNaN;
	return d;
}

// ───────────────────────────── tolerance bands ───────────────────────────────
// Every golden carries 17 significant digits, so no band here is a
// transcription limit. Each is the measured distance between two independent
// implementations of the same quantity, with headroom stated.
//
// Estimates and standard errors: both sides are a handful of arithmetic
// operations on the same covariance, differing only in the order of a dot
// product. Measured worst: exact on DS1, 3.6e-15 absolute on DS2. 1e-9 leaves
// six orders of headroom for a compiler's choice of fused multiply-add.
static const double TOL = 1e-9;
// t statistics inherit both the estimate's and the standard error's error.
// Measured worst: 1.8e-15 absolute.
static const double TOL_T = 1e-8;
// Confidence limits add a quantile inversion, and that is the loosest step in
// this file. distributions.hpp inverts its own CDF by bisection; scipy inverts
// a different implementation of a different CDF. Measured worst: 1.0e-14
// absolute on a t quantile and 1.9e-11 on a studentized range quantile, which
// agrees with the ~1e-11 relative accuracy test_tukey.cpp measures for qtukey.
// 1e-8 is 500 times the worse of the two.
static const double TOL_CL = 1e-8;
// p-values get a relative band WITH an absolute floor. Both sides compute a
// tail as 2·(1 − CDF), and that subtraction throws away the leading digits: at
// DS2's smallest p, 7.7e-10, one ULP of the intermediate is already 1.2e-7 of
// the answer, so a relative band alone would reject a correct tail. The floor
// is that rounding — measured worst 9.5e-17 absolute, so 1e-15 is about ten
// times it — and the relative part covers every p of order 1e-3 and up, where
// the measured worst is 6.8e-14 relative.
static const double RTOL_P = 1e-9;
static const double ATOL_P = 1e-15;
// The studentized range tail is documented in distributions.hpp as accurate to
// about 1e-12 for every integer df, and DS1's df is 6. Measured worst: 8.9e-13
// absolute, 1.6e-11 relative. The floor is ten times the documented accuracy.
static const double RTOL_TUKEY_P = 1e-8;
static const double ATOL_TUKEY_P = 1e-11;

// Build a row-major Mat from a flat value list.
static Mat mat(std::size_t r, std::size_t c, std::vector<double> vals) {
	Mat m(r, c);
	m.data = std::move(vals);
	return m;
}

// ───────────────────────── the fitted models ─────────────────────────────────
// DS1: three groups, cell-means coded, so L = I(3) selects the three cell
// means and beta is (2, 6, 6) with cov = (s²/3)·I.
static const std::vector<double> kBetaDs1(kDs1Beta, kDs1Beta + 3);
static const Mat kCovDs1 = mat(3, 3, std::vector<double>(kDs1Cov, kDs1Cov + 9));

// DS2: the rank-deficient fit, full width, with NaN in the dropped column and
// in its row and column of cov — what lm_fit reports.
static const std::vector<double> kBetaDs2(kDs2Beta, kDs2Beta + 4);
static const Mat kCovDs2 = mat(4, 4, std::vector<double>(kDs2Cov, kDs2Cov + 16));
static const Mat kLDs2 = mat(3, 4, std::vector<double>(kDs2L, kDs2L + 12));

// L = I(3): the three cell means of DS1, in order.
static Mat ds1_grid() {
	Mat L(3, 3);
	for (std::size_t i = 0; i < 3; i++) {
		for (std::size_t j = 0; j < 3; j++) {
			L(i, j) = (i == j) ? 1.0 : 0.0;
		}
	}
	return L;
}

// ──────────────────── DS1, the values computable by hand ─────────────────────
static void test_ds1_hand_values() {
	std::printf("ds1_hand_values (three cell means, s2 = 8/3 on 6 df)\n");
	const Mat L = ds1_grid();
	const auto r = emmeans(L, kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(r.ok, "DS1 fits");
	CHECK_MSG(r.means.size() == 3, "three means");
	CHECK_MSG(r.diffs.size() == 3, "three pairs from three means");
	CHECK_MSG(r.k_compare == 3, "all three means are estimable");
	CHECK_MSG(r.df_den == 6.0, "df is echoed exactly");
	CHECK_MSG(r.alpha == 0.05, "alpha is echoed exactly");
	CHECK_MSG(r.adjust == EmmAdjust::TUKEY, "the method is echoed");
	CHECK_MSG(std::string(statsduck::emm_adjust_name(r.adjust)) == "tukey",
	          "and its canonical name is tukey");

	// Hand-computable: the means are the group means, the SEs come from s² = 8/3.
	CHECK_CLOSE(mean_at(r, 0).estimate, 2.0, 1e-10);
	CHECK_CLOSE(mean_at(r, 1).estimate, 6.0, 1e-10);
	CHECK_CLOSE(mean_at(r, 2).estimate, 6.0, 1e-10);
	CHECK_CLOSE(mean_at(r, 0).std_error, 0.9428090416, 1e-9);
	CHECK_CLOSE(diff_at(r, 0).std_error, 1.3333333333, 1e-9);
	CHECK_CLOSE(diff_at(r, 0).estimate, -4.0, 1e-10);
	CHECK_MSG(diff_at(r, 0).i == 0 && diff_at(r, 0).j == 1, "pairs are 0-based in the kernel");
	CHECK_MSG(diff_at(r, 1).i == 0 && diff_at(r, 1).j == 2, "and enumerated row-major");
	CHECK_MSG(diff_at(r, 2).i == 1 && diff_at(r, 2).j == 2, "third pair is 2-3");

	// Groups 2 and 3 have the same mean, so their difference is exactly zero
	// and its t is exactly zero. No oracle needed for either.
	CHECK_CLOSE(diff_at(r, 2).estimate, 0.0, 1e-13);
	CHECK_CLOSE(diff_at(r, 2).t_statistic, 0.0, 1e-13);
}

// ─────────────────── DS1 against the statsmodels oracle ──────────────────────
static void test_ds1_against_oracle() {
	std::printf("ds1_against_oracle (tukey)\n");
	const Mat L = ds1_grid();
	const auto r = emmeans(L, kBetaDs1, kCovDs1, kDs1Df, kEmmAlpha, EmmAdjust::TUKEY);
	CHECK(r.ok);
	CHECK(r.means.size() == 3 && r.diffs.size() == 3);

	for (std::size_t g = 0; g < 3; g++) {
		const EmmMean m = mean_at(r, g);
		const EmmMeanCase &w = kDs1Means[g];
		check(m.estimable, lbl("ds1 mean estimable", g), __LINE__);
		check_close(m.estimate, w.estimate, TOL, lbl("ds1 mean estimate", g), __LINE__);
		check_close(m.std_error, w.std_error, TOL, lbl("ds1 mean std_error", g), __LINE__);
		check_close(m.t_statistic, w.t_statistic, TOL_T, lbl("ds1 mean t", g), __LINE__);
		check_p(m.p_value, w.p_value, RTOL_P, ATOL_P, lbl("ds1 mean p", g), __LINE__);
		// A mean's limits are never adjusted, whatever the pairwise method is.
		check_close(m.lower_cl, w.lower_cl, TOL_CL, lbl("ds1 mean lower_cl", g), __LINE__);
		check_close(m.upper_cl, w.upper_cl, TOL_CL, lbl("ds1 mean upper_cl", g), __LINE__);
	}

	for (std::size_t d = 0; d < 3; d++) {
		const EmmDiff p = diff_at(r, d);
		const EmmDiffCase &w = kDs1Diffs[d];
		check(p.i == w.i && p.j == w.j, lbl("ds1 pair indices", d), __LINE__);
		check(p.estimable, lbl("ds1 pair estimable", d), __LINE__);
		check_close(p.estimate, w.estimate, TOL, lbl("ds1 pair estimate", d), __LINE__);
		check_close(p.std_error, w.std_error, TOL, lbl("ds1 pair std_error", d), __LINE__);
		check_close(p.t_statistic, w.t_statistic, TOL_T, lbl("ds1 pair t", d), __LINE__);
		check_p(p.p_value, w.p_value, RTOL_P, ATOL_P, lbl("ds1 pair p", d), __LINE__);
		// Tukey: the studentized range tail at sqrt(2)·|t| over k = 3 means.
		check_p(p.p_adjusted, w.p_tukey, RTOL_TUKEY_P, ATOL_TUKEY_P, lbl("ds1 pair p_tukey", d),
		       __LINE__);
		// And the Tukey limits, estimate ± q(1-alpha; k, df)/sqrt(2) · se.
		check_close(p.lower_cl, w.tukey_lower, TOL_CL, lbl("ds1 pair tukey lower", d), __LINE__);
		check_close(p.upper_cl, w.tukey_upper, TOL_CL, lbl("ds1 pair tukey upper", d), __LINE__);
	}
}

// ───────────────── DS1 under bonferroni and under holm ───────────────────────
static void test_ds1_other_adjustments() {
	std::printf("ds1_other_adjustments (bonferroni, holm)\n");
	const Mat L = ds1_grid();

	// Bonferroni adjusts the p-values through statsduck::adjust_p and widens
	// the limits with t(1 - alpha/(2m), df), m being the three estimable pairs.
	const auto rb = emmeans(L, kBetaDs1, kCovDs1, kDs1Df, kEmmAlpha, EmmAdjust::BONFERRONI);
	CHECK(rb.ok);
	CHECK_MSG(std::string(statsduck::emm_adjust_name(rb.adjust)) == "bonferroni",
	          "bonferroni echoes its own name");
	for (std::size_t d = 0; d < 3; d++) {
		const EmmDiff p = diff_at(rb, d);
		const EmmDiffCase &w = kDs1Diffs[d];
		check_p(p.p_adjusted, w.p_bonferroni, RTOL_P, ATOL_P, lbl("ds1 pair p_bonferroni", d),
		        __LINE__);
		check_close(p.lower_cl, w.bonf_lower, TOL_CL, lbl("ds1 pair bonf lower", d), __LINE__);
		check_close(p.upper_cl, w.bonf_upper, TOL_CL, lbl("ds1 pair bonf upper", d), __LINE__);
		// The raw p is reported unchanged alongside the adjusted one.
		check_p(p.p_value, w.p_value, RTOL_P, ATOL_P, lbl("ds1 pair raw p under bonferroni", d),
		        __LINE__);
	}

	// Holm is a step-down method, so it needs the ranks of the whole family.
	// Its limits stay at the unadjusted t quantile, which is what the mean
	// limits use, so the goldens for those are kDs1Means' own limits.
	const auto rh = emmeans(L, kBetaDs1, kCovDs1, kDs1Df, kEmmAlpha, EmmAdjust::HOLM);
	CHECK(rh.ok);
	CHECK_MSG(std::string(statsduck::emm_adjust_name(rh.adjust)) == "holm",
	          "holm echoes its own name");
	for (std::size_t d = 0; d < 3; d++) {
		const EmmDiff p = diff_at(rh, d);
		const EmmDiffCase &w = kDs1Diffs[d];
		check_p(p.p_adjusted, w.p_holm, RTOL_P, ATOL_P, lbl("ds1 pair p_holm", d), __LINE__);
		// estimate ± t(1 - alpha/2, df)·se: no multiplicity in the interval.
		check_close(p.lower_cl, w.lower_cl, TOL_CL, lbl("ds1 pair holm lower", d), __LINE__);
		check_close(p.upper_cl, w.upper_cl, TOL_CL, lbl("ds1 pair holm upper", d), __LINE__);
		// Correcting for multiplicity can only widen an interval, so holm's,
		// which does not, sits strictly inside bonferroni's.
		check(p.lower_cl > w.bonf_lower && p.upper_cl < w.bonf_upper,
		      lbl("holm limits are narrower than bonferroni's", d), __LINE__);
	}
}

// ────────────────────────── the misuse cases ─────────────────────────────────
static void test_misuse() {
	std::printf("misuse (every ok = false path)\n");
	const Mat L = ds1_grid();

	// An empty grid is misuse, matching lin_hyp's "no rows". A grid of no means
	// is a request for nothing, and two neighbouring functions must not answer
	// the same malformed input differently.
	const Mat L0(0, 3);
	const auto r0 = emmeans(L0, kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(!r0.ok, "an empty L is misuse");
	CHECK_MSG(r0.error.find("no rows") != std::string::npos, "and the message says so");

	// A row width that is not k.
	const auto r1 = emmeans(mat(1, 2, {1.0, 0.0}), kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(!r1.ok, "a grid whose rows are not k wide is misuse");
	CHECK_MSG(r1.error.find("emmeans") != std::string::npos, "the message names the function");
	CHECK_MSG(r1.error.find('2') != std::string::npos && r1.error.find('3') != std::string::npos,
	          "and reports both actual sizes");

	// A covariance that is not k × k.
	const auto r2 = emmeans(L, kBetaDs1, mat(2, 2, {1.0, 0.0, 0.0, 1.0}), 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(!r2.ok, "a covariance that is not k by k is misuse");
	CHECK_MSG(r2.error.find("covariance") != std::string::npos, "and the message says which input");

	// An empty beta. L's rows are zero wide too, so the width check passes and
	// this is the one that must fire.
	const auto r3 = emmeans(Mat(1, 0), std::vector<double>(), Mat(0, 0), 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(!r3.ok, "an empty beta is misuse");
	CHECK_MSG(r3.error.find("beta") != std::string::npos, "and the message names beta");

	// df must be positive and finite.
	for (const double bad_df : {0.0, -1.0, std::numeric_limits<double>::infinity(), kNaN}) {
		const auto rd = emmeans(L, kBetaDs1, kCovDs1, bad_df, 0.05, EmmAdjust::TUKEY);
		CHECK_MSG(!rd.ok, "a df that is not positive and finite is misuse");
		CHECK_MSG(rd.error.find("df") != std::string::npos, "and the message names df");
	}

	// alpha must be strictly inside (0, 1).
	for (const double bad_alpha : {0.0, 1.0, -0.1, 1.5, kNaN}) {
		const auto ra = emmeans(L, kBetaDs1, kCovDs1, 6.0, bad_alpha, EmmAdjust::TUKEY);
		CHECK_MSG(!ra.ok, "an alpha outside (0, 1) is misuse");
		CHECK_MSG(ra.error.find("alpha") != std::string::npos, "and the message names alpha");
	}

	// A failing call returns empty vectors, not a half-filled result.
	CHECK_MSG(r0.means.empty() && r0.diffs.empty(), "a failing call reports nothing");
}

// ───────────────────── parse_emm_adjust / emm_adjust_name ────────────────────
static void test_adjust_names() {
	std::printf("adjust_names (parse and echo)\n");
	struct NameCase {
		const char *in;
		EmmAdjust want;
		const char *canonical;
	};
	// fdr is an alias for BH, so it echoes as BH. Case sensitive, matching
	// statsduck::parse_adjust, which is the same SQL surface.
	static const NameCase kCases[] = {
	    {"tukey", EmmAdjust::TUKEY, "tukey"},
	    {"bonferroni", EmmAdjust::BONFERRONI, "bonferroni"},
	    {"holm", EmmAdjust::HOLM, "holm"},
	    {"hochberg", EmmAdjust::HOCHBERG, "hochberg"},
	    {"BH", EmmAdjust::BH, "BH"},
	    {"fdr", EmmAdjust::BH, "BH"},
	    {"BY", EmmAdjust::BY, "BY"},
	    {"none", EmmAdjust::NONE, "none"},
	};
	for (const NameCase &c : kCases) {
		EmmAdjust got = EmmAdjust::NONE;
		check(statsduck::parse_emm_adjust(c.in, got), lbl(c.in, 0), __LINE__);
		check(got == c.want, lbl("parsed value", 0), __LINE__);
		check(std::string(statsduck::emm_adjust_name(c.want)) == c.canonical, lbl(c.in, 1), __LINE__);
	}
	for (const char *bad : {"Tukey", "TUKEY", "bh", "by", "sidak", "", "fdr "}) {
		EmmAdjust got = EmmAdjust::TUKEY;
		check(!statsduck::parse_emm_adjust(bad, got), lbl(bad, 2), __LINE__);
		check(got == EmmAdjust::TUKEY, lbl("a rejected name leaves out untouched", 0), __LINE__);
	}
}

// ───────────────── DS2: a rank-deficient fit, per-row estimability ───────────
static void test_ds2_aliased_column() {
	std::printf("ds2_aliased_column (rank deficient, per-row estimability)\n");
	// DS2 is rank deficient: its last column is the sum of two others, so
	// lm_fit drops it and cov carries NaN in that row and column. A grid row
	// avoiding the aliased column is evaluated; one touching it is not.
	const auto r2 = emmeans(kLDs2, kBetaDs2, kCovDs2, kDs2Df, kEmmAlpha, EmmAdjust::NONE);
	CHECK_MSG(r2.ok, "a rank-deficient fit is not misuse");
	CHECK_MSG(mean_at(r2, 0).estimable, "a row avoiding the aliased column is estimable");
	CHECK_MSG(mean_at(r2, 1).estimable, "and so is the second such row");
	CHECK_MSG(!mean_at(r2, 2).estimable, "a row touching it is not");
	CHECK_MSG(r2.k_compare == 2, "k_compare counts only the estimable means");
	CHECK_MSG(r2.diffs.size() == 3, "every pair is still present");
	CHECK_MSG(!diff_at(r2, 1).estimable, "a pair touching the aliased mean is not estimable");
	CHECK_MSG(!diff_at(r2, 2).estimable, "and so is the other such pair");
	CHECK_MSG(diff_at(r2, 0).estimable, "a pair between two good means still is");
	CHECK_MSG(std::isfinite(mean_at(r2, 0).estimate), "no NaN leaks from the aliased column");
	CHECK_MSG(std::isfinite(mean_at(r2, 0).std_error), "nor into its standard error");
	// Positions stay stable even though pair 1 and pair 2 carry no values, so
	// the indices still name the rows they came from.
	CHECK_MSG(diff_at(r2, 1).i == 0 && diff_at(r2, 1).j == 2, "a dropped pair keeps its indices");

	// The two estimable rows against the oracle. This is the only fixture here
	// whose covariance has off-diagonal entries, so the pair below is the one
	// case where the cross term in C_ii + C_jj - 2·C_ij actually contributes.
	for (std::size_t g = 0; g < 2; g++) {
		const EmmMean m = mean_at(r2, g);
		const EmmMeanCase &w = kDs2Means[g];
		check_close(m.estimate, w.estimate, TOL, lbl("ds2 mean estimate", g), __LINE__);
		check_close(m.std_error, w.std_error, TOL, lbl("ds2 mean std_error", g), __LINE__);
		check_close(m.t_statistic, w.t_statistic, TOL_T, lbl("ds2 mean t", g), __LINE__);
		check_p(m.p_value, w.p_value, RTOL_P, ATOL_P, lbl("ds2 mean p", g), __LINE__);
		check_close(m.lower_cl, w.lower_cl, TOL_CL, lbl("ds2 mean lower_cl", g), __LINE__);
		check_close(m.upper_cl, w.upper_cl, TOL_CL, lbl("ds2 mean upper_cl", g), __LINE__);
	}
	const EmmDiff p = diff_at(r2, 0);
	CHECK_CLOSE(p.estimate, kDs2Diff01.estimate, TOL);
	CHECK_CLOSE(p.std_error, kDs2Diff01.std_error, TOL);
	CHECK_CLOSE(p.t_statistic, kDs2Diff01.t_statistic, TOL_T);
	check_p(p.p_value, kDs2Diff01.p_value, RTOL_P, ATOL_P, "ds2 pair p", __LINE__);
	CHECK_CLOSE(p.lower_cl, kDs2Diff01.lower_cl, TOL_CL);
	CHECK_CLOSE(p.upper_cl, kDs2Diff01.upper_cl, TOL_CL);
	CHECK_MSG(p.p_adjusted == p.p_value, "under none the adjusted p is the raw p");

	// The pairwise standard error must not be reachable from the two means'
	// standard errors alone. Here it is not: with a cross term of 0.382 the
	// independent-sum formula is off by 18%, so a kernel that forgot C_ij
	// would fail this.
	const double naive = std::sqrt(mean_at(r2, 0).std_error * mean_at(r2, 0).std_error +
	                               mean_at(r2, 1).std_error * mean_at(r2, 1).std_error);
	CHECK_MSG(std::fabs(naive - p.std_error) > 0.1, "the cross term moves the pairwise SE");
}

// ──────────────── identical grid rows, and the easiest adjustments ───────────
static void test_duplicated_rows() {
	std::printf("duplicated_rows (zero variance by cancellation)\n");
	// Two identical grid rows. C_ii = C_jj = C_ij, so the pairwise variance is
	// exactly zero by cancellation. A user who repeats a level reaches this, so
	// it must not be mistaken for a negative variance.
	Mat Ld(2, 3);
	Ld(0, 0) = 1.0;
	Ld(0, 1) = 0.0;
	Ld(0, 2) = 0.0;
	Ld(1, 0) = 1.0;
	Ld(1, 1) = 0.0;
	Ld(1, 2) = 0.0;
	const auto rd = emmeans(Ld, kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::NONE);
	CHECK_MSG(rd.ok && rd.diffs.size() == 1, "one pair from two rows");
	CHECK_MSG(diff_at(rd, 0).estimable, "an identical pair is estimable, not NULL");
	CHECK_MSG(diff_at(rd, 0).estimate == 0.0, "the difference is exactly zero");
	CHECK_MSG(diff_at(rd, 0).std_error == 0.0, "and so is its standard error");
	CHECK_MSG(std::isnan(diff_at(rd, 0).t_statistic), "t is NaN at a zero standard error");
	CHECK_MSG(std::isnan(diff_at(rd, 0).p_value), "and so is p");
	CHECK_MSG(std::isnan(diff_at(rd, 0).p_adjusted), "an unrankable pair has no adjusted p either");
	CHECK_MSG(mean_at(rd, 0).estimate == mean_at(rd, 1).estimate,
	          "identical rows give identical means");
	// The interval of a point estimate with no variance is that point.
	CHECK_MSG(diff_at(rd, 0).lower_cl == 0.0 && diff_at(rd, 0).upper_cl == 0.0,
	          "a zero standard error gives a zero-width interval");
	CHECK_MSG(rd.k_compare == 2, "both identical rows count as estimable means");

	// With one comparison, bonferroni multiplies by one and changes nothing.
	Mat L_two_rows(2, 3);
	L_two_rows(0, 0) = 1.0;
	L_two_rows(1, 1) = 1.0;
	const auto rb1 = emmeans(L_two_rows, kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::BONFERRONI);
	CHECK_MSG(rb1.diffs.size() == 1, "two means give one pair");
	CHECK_MSG(diff_at(rb1, 0).p_adjusted == diff_at(rb1, 0).p_value,
	          "bonferroni over one pair leaves p alone");

	// Under none, every adjusted p equals its raw p exactly, for every pair.
	const auto rn = emmeans(ds1_grid(), kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::NONE);
	CHECK(rn.diffs.size() == 3);
	for (std::size_t d = 0; d < rn.diffs.size(); d++) {
		check(diff_at(rn, d).p_adjusted == diff_at(rn, d).p_value, lbl("none leaves every p alone", d),
		      __LINE__);
	}
	CHECK_MSG(std::string(statsduck::emm_adjust_name(rn.adjust)) == "none",
	          "the echoed name is canonical");
}

// ──────────────── the tolerance decision does not depend on scale ───────────
static void test_scale_invariance() {
	std::printf("scale_invariance (cov x 1e12)\n");
	const Mat L = ds1_grid();
	const auto r = emmeans(L, kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::TUKEY);

	// Scaling cov by a constant scales every variance by it, so the estimates
	// do not move, the standard errors scale by its square root, and the t
	// statistics, p-values and k_compare do not move at all. In #20 a units
	// change silently moved df_num and the F, so this is pinned rather than
	// assumed.
	Mat Vbig = kCovDs1;
	for (double &v : Vbig.data) {
		v *= 1e12;
	}
	const auto rb = emmeans(L, kBetaDs1, Vbig, 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(rb.ok && rb.k_compare == 3, "k_compare is unaffected by scale");
	CHECK_MSG(rb.diffs.size() == 3, "and so is the number of pairs");
	for (std::size_t i = 0; i < 3; i++) {
		check_close(mean_at(rb, i).estimate, mean_at(r, i).estimate, 1e-10,
		            lbl("estimates do not depend on cov's scale", i), __LINE__);
		// 1e-4 absolute on a standard error of 9.4e5 is 1e-10 relative, which
		// is a few ULP of the product and the square root.
		check_close(mean_at(rb, i).std_error, mean_at(r, i).std_error * 1e6, 1e-4,
		            lbl("standard errors scale by the square root", i), __LINE__);
		// The t statistic shrinks by exactly the same factor, so the p-value
		// it feeds moves too; what must not move is the estimability decision.
		check(mean_at(rb, i).estimable, lbl("every mean is still estimable", i), __LINE__);
	}
	// 1e-6 absolute on a t of 3 is 3e-7 relative, which covers the 1e6 the
	// comparison multiplies the smaller side by.
	CHECK_CLOSE(diff_at(rb, 0).t_statistic * 1e6, diff_at(r, 0).t_statistic, 1e-6);
	for (std::size_t d = 0; d < 3; d++) {
		check(diff_at(rb, d).estimable, lbl("every pair is still estimable", d), __LINE__);
		check_close(diff_at(rb, d).estimate, diff_at(r, d).estimate, 1e-10,
		            lbl("pair estimates do not depend on cov's scale", d), __LINE__);
	}

	// The other direction. A tiny covariance must not read as a negative one,
	// which is the failure mode an absolute tolerance would have.
	Mat Vtiny = kCovDs1;
	for (double &v : Vtiny.data) {
		v *= 1e-12;
	}
	const auto rt = emmeans(L, kBetaDs1, Vtiny, 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(rt.ok && rt.k_compare == 3, "a tiny covariance is still a covariance");
	for (std::size_t d = 0; d < 3; d++) {
		check(diff_at(rt, d).estimable, lbl("and its pairs are still estimable", d), __LINE__);
	}
	CHECK_CLOSE(diff_at(rt, 0).t_statistic / 1e6, diff_at(r, 0).t_statistic, 1e-6);
}

// ──────────── fewer than two estimable means: no ptukey call at all ─────────
static void test_one_estimable_mean() {
	std::printf("one_estimable_mean (tukey below k = 2)\n");
	// Tukey with fewer than two estimable means. ptukey is undefined for k < 2
	// — it returns NaN — so it must not be called at all.
	Mat L_one_row(1, 3);
	L_one_row(0, 0) = 1.0;
	L_one_row(0, 1) = 0.0;
	L_one_row(0, 2) = 0.0;
	const auto r1 = emmeans(L_one_row, kBetaDs1, kCovDs1, 6.0, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(r1.ok, "a single grid row is not misuse");
	CHECK_MSG(r1.means.size() == 1 && r1.diffs.empty(), "one mean, no pairs");
	CHECK_MSG(r1.k_compare == 1, "k_compare reports what it found");
	CHECK_MSG(std::isfinite(mean_at(r1, 0).estimate), "the mean is reported normally");
	CHECK_MSG(std::isfinite(mean_at(r1, 0).lower_cl), "and its limits are finite");
	CHECK_MSG(std::isfinite(mean_at(r1, 0).upper_cl), "both of them");
	// A one-row grid is not a special case numerically: it is DS1's first cell
	// mean, so the DS1 goldens still apply.
	CHECK_CLOSE(mean_at(r1, 0).estimate, kDs1Means[0].estimate, TOL);
	CHECK_CLOSE(mean_at(r1, 0).std_error, kDs1Means[0].std_error, TOL);
	CHECK_CLOSE(mean_at(r1, 0).lower_cl, kDs1Means[0].lower_cl, TOL_CL);

	// The harder shape: one estimable mean WITH a pair present. A grid of DS2's
	// first row and its aliased row gives k_compare = 1 and one pair, so the
	// Tukey branch is reached with something to write into. Nothing may come
	// back NaN on the estimable mean, which is what a ptukey call at k = 1
	// would produce.
	Mat Lmix(2, 4);
	for (std::size_t j = 0; j < 4; j++) {
		Lmix(0, j) = kDs2L[j];     // avoids the dropped column
		Lmix(1, j) = kDs2L[2 * 4 + j]; // touches it
	}
	const auto rm = emmeans(Lmix, kBetaDs2, kCovDs2, kDs2Df, 0.05, EmmAdjust::TUKEY);
	CHECK_MSG(rm.ok, "a grid with one usable row is not misuse");
	CHECK_MSG(rm.k_compare == 1, "only one mean was estimable");
	CHECK_MSG(rm.diffs.size() == 1, "the pair is still reported");
	CHECK_MSG(!diff_at(rm, 0).estimable, "but it is not estimable");
	CHECK_MSG(mean_at(rm, 0).estimable, "the usable row is");
	CHECK_MSG(std::isfinite(mean_at(rm, 0).estimate) && std::isfinite(mean_at(rm, 0).std_error) &&
	              std::isfinite(mean_at(rm, 0).t_statistic) &&
	              std::isfinite(mean_at(rm, 0).p_value) && std::isfinite(mean_at(rm, 0).lower_cl) &&
	              std::isfinite(mean_at(rm, 0).upper_cl),
	          "and no NaN reaches any field it reports");
	CHECK_CLOSE(mean_at(rm, 0).estimate, kDs2Means[0].estimate, TOL);
}

// ──────────────────────── extreme df and alpha ──────────────────────────────
static void test_extreme_df_and_alpha() {
	std::printf("extreme_df_and_alpha (df = 0.5, alpha = 1e-10)\n");
	const Mat L = ds1_grid();
	// A fractional, tiny df with an extreme alpha drives StudentTQuantile deep
	// into its tail. The result may be enormous, but it must be reached in
	// bounded time and must be either finite or an infinity, never a NaN that
	// reads as a valid limit.
	const auto re = emmeans(L, kBetaDs1, kCovDs1, 0.5, 1e-10, EmmAdjust::NONE);
	CHECK_MSG(re.ok, "a fractional df is accepted");
	CHECK_MSG(re.df_den == 0.5 && re.alpha == 1e-10, "df and alpha are echoed exactly");
	const double lo = mean_at(re, 0).lower_cl, hi = mean_at(re, 0).upper_cl;
	CHECK_MSG(!std::isnan(lo) && !std::isnan(hi), "limits are not NaN");
	CHECK_MSG(lo <= mean_at(re, 0).estimate && mean_at(re, 0).estimate <= hi,
	          "the estimate lies inside its own interval");
	// The same must hold for a pair, whose limits take a different route.
	const double plo = diff_at(re, 0).lower_cl, phi = diff_at(re, 0).upper_cl;
	CHECK_MSG(!std::isnan(plo) && !std::isnan(phi), "a pair's limits are not NaN either");
	CHECK_MSG(plo <= diff_at(re, 0).estimate && diff_at(re, 0).estimate <= phi,
	          "and the difference lies inside its own interval");
	// At this df the interval is enormous, which is the honest answer, not a
	// failure: half a degree of freedom carries almost no information.
	CHECK_MSG(hi - lo > 1e9, "and the interval is enormous, as it should be");

	// Tukey at the same extremes, which goes through the studentized range
	// quantile instead. It is documented as degrading below df = 4 at a
	// non-integer df, so the value is not pinned — only that it is a number.
	const auto rt = emmeans(L, kBetaDs1, kCovDs1, 0.5, 1e-10, EmmAdjust::TUKEY);
	CHECK_MSG(rt.ok, "tukey accepts the same extremes");
	for (std::size_t d = 0; d < rt.diffs.size(); d++) {
		check(!std::isnan(diff_at(rt, d).lower_cl) && !std::isnan(diff_at(rt, d).upper_cl),
		      lbl("tukey limits are not NaN at df = 0.5", d), __LINE__);
		check(diff_at(rt, d).lower_cl <= diff_at(rt, d).upper_cl,
		      lbl("and they are the right way round", d), __LINE__);
	}

	// df at the boundary is misuse, not a very wide interval.
	const auto rz = emmeans(L, kBetaDs1, kCovDs1, 0.0, 0.05, EmmAdjust::NONE);
	CHECK_MSG(!rz.ok, "df = 0 is misuse");
	CHECK_MSG(rz.error.find("df") != std::string::npos, "and the message names df");

	// So is alpha at either boundary: an alpha of 0 asks for an interval of
	// infinite width and an alpha of 1 for one of zero width, and neither is a
	// confidence level.
	for (const double edge : {0.0, 1.0}) {
		const auto ra = emmeans(L, kBetaDs1, kCovDs1, 6.0, edge, EmmAdjust::NONE);
		CHECK_MSG(!ra.ok, "alpha at a boundary is misuse");
	}
}

int main() {
	std::printf("=== test_emmeans ===\n");
	test_ds1_hand_values();
	test_ds1_against_oracle();
	test_ds1_other_adjustments();
	test_ds2_aliased_column();
	test_duplicated_rows();
	test_scale_invariance();
	test_one_estimable_mean();
	test_extreme_df_and_alpha();
	test_misuse();
	test_adjust_names();
	std::printf("%s: %d checks, %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_checks, g_fail);
	return g_fail == 0 ? 0 : 1;
}
