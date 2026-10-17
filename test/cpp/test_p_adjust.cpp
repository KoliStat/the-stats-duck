// Direct C++ unit tests for statsduck::adjust_p and the method parsing it
// shares with emmeans (#21).
//
// The adjustment values are the ones test/sql/adjust_p.test already pins
// against R 4.5.3's p.adjust(), so this extraction is held to released
// behaviour rather than to a fresh opinion.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "p_adjust.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

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

// Safe accessor for a double vector. A failing call can leave the vector
// empty, and indexing an empty vector directly is undefined behaviour: it
// can crash the binary before any later check gets to report, which turns a
// clear assertion failure into a confusing one. Out of range reads back as
// NaN, which no real adjusted p-value equals.
static double at(const std::vector<double> &v, std::size_t i) {
	return i < v.size() ? v[i] : std::numeric_limits<double>::quiet_NaN();
}

static void test_adjust_p_against_released_values() {
	std::printf("adjust_p (bonferroni, holm, hochberg, BH, BY, the valid mask)\n");

	const std::vector<double> p = {0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205};
	const std::vector<bool> all_valid(8, true);
	std::vector<double> out;

	statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::BONFERRONI, out);
	const double bonf[8] = {0.008, 0.064, 0.312, 0.328, 0.336, 0.48, 0.592, 1.0};
	for (std::size_t i = 0; i < 8; i++)
		check(close(at(out, i), bonf[i], 1e-12), "bonferroni matches the released value");

	statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::HOLM, out);
	const double holm[8] = {0.008, 0.056, 0.234, 0.234, 0.234, 0.234, 0.234, 0.234};
	for (std::size_t i = 0; i < 8; i++)
		check(close(at(out, i), holm[i], 1e-4), "holm matches the released value");

	// hochberg, BH and BY below get a 1e-4 band, not bonferroni's 1e-12: like
	// holm above, test/sql/adjust_p.test rounds them to four places before
	// pinning them, so a tighter band would fail on the oracle's own rounding
	// rather than on a real disagreement.
	statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::HOCHBERG, out);
	const double hochberg[8] = {0.008, 0.056, 0.148, 0.148, 0.148, 0.148, 0.148, 0.205};
	for (std::size_t i = 0; i < 8; i++)
		check(close(at(out, i), hochberg[i], 1e-4), "hochberg matches the released value");

	statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::BENJAMINI_HOCHBERG, out);
	const double bh[8] = {0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0846, 0.205};
	for (std::size_t i = 0; i < 8; i++)
		check(close(at(out, i), bh[i], 1e-4), "BH matches the released value");

	// The alias claim end to end: "fdr" and "BH" must parse to methods whose
	// adjust_p output is identical, not merely to the same enum value (the
	// other test function only checks the latter).
	statsduck::AdjustMethod m_bh = statsduck::AdjustMethod::NONE;
	statsduck::AdjustMethod m_fdr = statsduck::AdjustMethod::NONE;
	check(statsduck::parse_adjust("BH", m_bh) && statsduck::parse_adjust("fdr", m_fdr),
	      "both BH and fdr parse");
	std::vector<double> out_bh, out_fdr;
	statsduck::adjust_p(p, all_valid, m_bh, out_bh);
	statsduck::adjust_p(p, all_valid, m_fdr, out_fdr);
	bool identical = out_bh.size() == 8 && out_fdr.size() == 8;
	for (std::size_t i = 0; i < 8; i++)
		identical = identical && at(out_bh, i) == at(out_fdr, i);
	check(identical, "fdr and BH give identical adjust_p output, not just the same enum");

	statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::BENJAMINI_YEKUTIELI, out);
	const double by[8] = {0.0217, 0.087, 0.1826, 0.1826, 0.1826, 0.2174, 0.2299, 0.5572};
	for (std::size_t i = 0; i < 8; i++)
		check(close(at(out, i), by[i], 1e-4), "BY matches the released value");

	// The valid mask: an invalid slot comes back NaN and does not change the ranks.
	std::vector<bool> mask(8, true);
	mask[2] = false;
	statsduck::adjust_p(p, mask, statsduck::AdjustMethod::BONFERRONI, out);
	check(std::isnan(at(out, 2)), "an invalid slot is NaN");
	check(close(at(out, 0), 0.007, 1e-12), "the multiplier counts only valid entries");
}

static void test_parse_adjust_and_adjust_name() {
	std::printf("parse_adjust / adjust_name (aliases, case sensitivity)\n");

	// Parsing, including the alias and the canonical echo.
	statsduck::AdjustMethod m;
	check(statsduck::parse_adjust("fdr", m) && m == statsduck::AdjustMethod::BENJAMINI_HOCHBERG,
	      "fdr parses as Benjamini-Hochberg");
	check(std::string(statsduck::adjust_name(m)) == "BH", "fdr echoes as BH");
	check(statsduck::parse_adjust("none", m) && m == statsduck::AdjustMethod::NONE, "none parses");
	check(!statsduck::parse_adjust("tukey", m), "tukey is not a p-list method");
	check(!statsduck::parse_adjust("Bonferroni", m), "parsing is case sensitive, as adjust_p already is");
}

int main() {
	std::printf("== test_p_adjust ==\n");
	test_adjust_p_against_released_values();
	test_parse_adjust_and_adjust_name();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL P_ADJUST TESTS PASSED\n");
		return 0;
	}
	return 1;
}
