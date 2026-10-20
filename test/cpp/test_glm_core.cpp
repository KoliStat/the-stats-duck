// Direct C++ unit tests for statsduck::Family, statsduck::Link, their
// parsers, and the four scalar functions the IRLS loop will use (#22):
// link_eta, link_inv, link_deriv, and variance.
//
// The round-trip and central-difference checks come first. A failure in
// one of them means a link's algebra is wrong. A failure in a closed-form
// constant only means a number moved. See src/include/glm_core.hpp for the
// algebra table and for why link_inv clamps its result.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "glm_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

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
// two magnitudes. a == b (including both equal to 0) always passes. That
// lets a closed-form check below ask for an exact match by passing
// rtol = 0.0.
static bool close_rel(double a, double b, double rtol) {
	const double diff = std::fabs(a - b);
	if (diff == 0.0) {
		return true;
	}
	const double scale = std::max(std::fabs(a), std::fabs(b));
	return diff / scale <= rtol;
}

// All seven links, shared by the two tests below.
static const statsduck::Link kLinks[] = {
    statsduck::Link::kIdentity, statsduck::Link::kLogit, statsduck::Link::kProbit,
    statsduck::Link::kCLogLog, statsduck::Link::kLog, statsduck::Link::kInverse,
    statsduck::Link::kSqrt};

static void test_round_trip() {
	std::printf("link_inv inverts link_eta, inside the clamped domain\n");

	// Round trip. mu is sampled inside every link's domain: the three
	// probability links need mu in (0, 1), and log, inverse and sqrt need mu > 0.
	for (const statsduck::Link l : kLinks) {
		const bool prob = (l == statsduck::Link::kLogit || l == statsduck::Link::kProbit ||
		                   l == statsduck::Link::kCLogLog);
		const double mus_prob[] = {0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999};
		const double mus_pos[] = {1e-6, 0.01, 0.5, 1.0, 3.0, 100.0, 1e6};
		const double *mus = prob ? mus_prob : mus_pos;
		for (std::size_t i = 0; i < 7; i++) {
			const double mu = mus[i];
			const double back = statsduck::link_inv(l, statsduck::link_eta(l, mu));
			check(close_rel(back, mu, 1e-10), "link_inv inverts link_eta");
		}
	}
}

static void test_derivative_matches_central_difference() {
	std::printf("link_deriv matches a central difference of link_inv\n");

	// The derivative against a central difference of link_inv. The band is 1e-6
	// relative because a central difference is second-order accurate: with
	// h = 1e-5 the truncation error is about h^2 = 1e-10 relative, and the
	// subtraction loses about 1e-11, so 1e-6 is loose enough to never be the
	// thing that fails while still catching a wrong factor or sign.
	for (const statsduck::Link l : kLinks) {
		for (const double eta : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
			if (l == statsduck::Link::kInverse && std::fabs(eta) < 1e-9) continue; // mu = 1/eta
			const double h = 1e-5;
			const double fd = (statsduck::link_inv(l, eta + h) - statsduck::link_inv(l, eta - h)) / (2 * h);
			check(close_rel(statsduck::link_deriv(l, eta), fd, 1e-6),
			      "link_deriv matches a central difference of link_inv");
		}
	}
}

static void test_closed_forms_and_variance() {
	std::printf("known closed forms and the variance functions\n");

	// Known closed forms, so a self-consistent but wrong pair cannot pass.
	check(close_rel(statsduck::link_eta(statsduck::Link::kLogit, 0.5), 0.0, 0.0), "logit(0.5) = 0");
	check(close_rel(statsduck::link_inv(statsduck::Link::kLogit, 0.0), 0.5, 1e-15), "expit(0) = 0.5");
	check(close_rel(statsduck::link_deriv(statsduck::Link::kLogit, 0.0), 0.25, 1e-15),
	      "d expit/d eta at 0 is 1/4");
	check(close_rel(statsduck::link_inv(statsduck::Link::kProbit, 0.0), 0.5, 1e-12), "probit inverse at 0");
	check(close_rel(statsduck::link_deriv(statsduck::Link::kIdentity, 7.0), 1.0, 0.0),
	      "the identity link's derivative is 1");

	// Variance functions.
	check(close_rel(statsduck::variance(statsduck::Family::kGaussian, 3.0), 1.0, 0.0), "gaussian V = 1");
	check(close_rel(statsduck::variance(statsduck::Family::kPoisson, 3.0), 3.0, 0.0), "poisson V = mu");
	check(close_rel(statsduck::variance(statsduck::Family::kBinomial, 0.25), 0.1875, 1e-15),
	      "binomial V = mu(1-mu)");
	check(close_rel(statsduck::variance(statsduck::Family::kGamma, 3.0), 9.0, 0.0), "gamma V = mu^2");
}

static void test_parsing_and_link_tables() {
	std::printf("parsing, canonical links, and what each family allows\n");

	// Parsing, the canonical links, and what each family allows.
	statsduck::Family f;
	statsduck::Link l;
	check(statsduck::parse_family("binomial", f) && f == statsduck::Family::kBinomial, "binomial parses");
	check(statsduck::parse_family("GAUSSIAN", f) && f == statsduck::Family::kGaussian,
	      "family parsing is case insensitive, as parse_vcov is");
	check(!statsduck::parse_family("negbin", f), "negative binomial is not in this increment");
	check(!statsduck::parse_family("", f), "an empty family name is unknown");
	check(statsduck::parse_link("cloglog", l) && l == statsduck::Link::kCLogLog, "cloglog parses");
	check(!statsduck::parse_link("logaritmic", l), "a misspelled link is unknown");

	check(statsduck::canonical_link(statsduck::Family::kGaussian) == statsduck::Link::kIdentity, "gaussian canonical");
	check(statsduck::canonical_link(statsduck::Family::kBinomial) == statsduck::Link::kLogit, "binomial canonical");
	check(statsduck::canonical_link(statsduck::Family::kPoisson) == statsduck::Link::kLog, "poisson canonical");
	check(statsduck::canonical_link(statsduck::Family::kGamma) == statsduck::Link::kInverse, "gamma canonical");

	check(statsduck::link_allowed(statsduck::Family::kBinomial, statsduck::Link::kProbit), "binomial allows probit");
	check(!statsduck::link_allowed(statsduck::Family::kPoisson, statsduck::Link::kLogit),
	      "a probability link on poisson is not allowed");
	check(!statsduck::link_allowed(statsduck::Family::kBinomial, statsduck::Link::kInverse),
	      "binomial does not allow inverse");
	// Every family allows its own canonical link, which is the invariant that
	// makes the family-only call form work.
	for (const statsduck::Family fam : {statsduck::Family::kGaussian, statsduck::Family::kBinomial,
	                                    statsduck::Family::kPoisson, statsduck::Family::kGamma}) {
		check(statsduck::link_allowed(fam, statsduck::canonical_link(fam)),
		      "the canonical link is always allowed");
	}

	// Canonical echo: a name parses to the enum and the enum echoes that name.
	for (const char *n : {"gaussian", "binomial", "poisson", "gamma"}) {
		check(statsduck::parse_family(n, f) && std::string(statsduck::family_name(f)) == n,
		      "family_name echoes the canonical spelling");
	}
	for (const char *n : {"identity", "logit", "probit", "cloglog", "log", "inverse", "sqrt"}) {
		check(statsduck::parse_link(n, l) && std::string(statsduck::link_name(l)) == n,
		      "link_name echoes the canonical spelling");
	}
}

int main() {
	std::printf("== test_glm_core ==\n");
	test_round_trip();
	test_derivative_matches_central_difference();
	test_closed_forms_and_variance();
	test_parsing_and_link_tables();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL GLM_CORE TESTS PASSED\n");
		return 0;
	}
	return 1;
}
