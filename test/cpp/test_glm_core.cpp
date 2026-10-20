// Direct C++ unit tests for statsduck::Family, statsduck::Link, their
// parsers, the four scalar functions the IRLS loop will use (#22):
// link_eta, link_inv, link_deriv, and variance; and the four closed-form
// quantities a fit reports (#22 continued): deviance, glm_loglik, glm_aic,
// and pearson_dispersion.
//
// The round-trip and central-difference checks come first. A failure in
// one of them means a link's algebra is wrong. A failure in a closed-form
// constant only means a number moved. See src/include/glm_core.hpp for the
// algebra table and for why link_inv clamps its result.
//
// The zero-and-one binomial tests matter most of the four closed-form
// functions: a plain 0/1 logistic fit is the common case, not an edge
// case, and an unguarded deviance returns NaN for it.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "glm_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

static void test_deviance() {
	std::printf("deviance: the four families, hand-computed\n");

	// Gaussian deviance is the weighted residual sum of squares.
	const std::vector<double> y = {1.0, 2.0, 4.0};
	const std::vector<double> mu = {1.5, 2.0, 3.0};
	const std::vector<double> w = {1.0, 1.0, 1.0};
	check(close_rel(statsduck::deviance(statsduck::Family::kGaussian, y, mu, w), 1.25, 1e-15),
	      "gaussian deviance is 0.25 + 0 + 1");

	// Poisson: 2 Σ w [y ln(y/mu) − (y − mu)]. At y = mu every term is 0.
	const std::vector<double> ymu = {2.0, 3.0};
	check(close_rel(statsduck::deviance(statsduck::Family::kPoisson, ymu, ymu, {1.0, 1.0}), 0.0, 1e-15),
	      "a perfect poisson fit has zero deviance");

	// Weights scale the deviance linearly.
	const double d1 = statsduck::deviance(statsduck::Family::kGaussian, y, mu, {1.0, 1.0, 1.0});
	const double d2 = statsduck::deviance(statsduck::Family::kGaussian, y, mu, {3.0, 3.0, 3.0});
	check(close_rel(d2, 3.0 * d1, 1e-15), "the deviance is linear in the weights");

	// Binomial at y = 0.5, mu = 0.5 is zero; a mismatch is positive.
	check(close_rel(statsduck::deviance(statsduck::Family::kBinomial, {0.5}, {0.5}, {1.0}), 0.0, 1e-15),
	      "a perfect binomial fit has zero deviance");
	check(statsduck::deviance(statsduck::Family::kBinomial, {0.5}, {0.9}, {1.0}) > 0.0,
	      "a binomial mismatch has positive deviance");
}

static void test_loglik_aic_dispersion() {
	std::printf("glm_loglik, glm_aic and pearson_dispersion\n");

	// Poisson, a perfect fit: the y*ln(mu) - mu - lgamma(y+1) term, hand
	// summed. lgamma(3) = ln(2), lgamma(4) = ln(6).
	const std::vector<double> y = {2.0, 3.0};
	const std::vector<double> w = {1.0, 1.0};
	const double ll_pois =
	    (2.0 * std::log(2.0) - 2.0 - std::log(2.0)) + (3.0 * std::log(3.0) - 3.0 - std::log(6.0));
	check(close_rel(statsduck::glm_loglik(statsduck::Family::kPoisson, y, y, w, 1.0), ll_pois, 1e-12),
	      "poisson log-likelihood matches the hand-summed value");

	// Binomial, w = 1: LogChoose(1, y) is 0 for y in {0, 1}, so the
	// log-likelihood is just ln(mu) or ln(1 - mu).
	const std::vector<double> y1 = {1.0};
	const std::vector<double> mu1 = {0.7};
	check(close_rel(statsduck::glm_loglik(statsduck::Family::kBinomial, y1, mu1, {1.0}, 1.0),
	                std::log(0.7), 1e-12),
	      "binomial log-likelihood at y = 1, w = 1 is ln(mu)");

	// pearson_dispersion is exactly 1.0 for binomial and poisson, regardless of
	// the data or of df_residual, so the caller never special-cases them.
	check(statsduck::pearson_dispersion(statsduck::Family::kBinomial, {0.3}, {0.9}, {1.0}, 1) == 1.0,
	      "binomial dispersion is exactly 1.0");
	check(statsduck::pearson_dispersion(statsduck::Family::kPoisson, {9.0}, {1.0}, {1.0}, 1) == 1.0,
	      "poisson dispersion is exactly 1.0 even far from the fit");

	// Gaussian dispersion: Σ w(y-mu)^2 / variance(mu) / df_residual. variance
	// is 1 for gaussian, so this is just the weighted RSS over df_residual.
	const std::vector<double> yg = {1.0, 2.0, 4.0};
	const std::vector<double> mug = {1.5, 2.0, 3.0};
	const std::vector<double> wg = {1.0, 1.0, 1.0};
	check(close_rel(statsduck::pearson_dispersion(statsduck::Family::kGaussian, yg, mug, wg, 2), 0.625, 1e-15),
	      "gaussian dispersion is the weighted RSS (1.25) over df_residual (2)");

	// Gamma dispersion: Σ w(y-mu)^2/mu^2 / df_residual.
	const std::vector<double> yga = {4.0};
	const std::vector<double> muga = {5.0};
	check(close_rel(statsduck::pearson_dispersion(statsduck::Family::kGamma, yga, muga, {1.0}, 1), 0.04, 1e-15),
	      "gamma dispersion is (y-mu)^2/mu^2 over df_residual, here 1/25");

	// glm_aic: -2*loglik + 2*(rank + d). d = 0 for binomial and poisson, 1 for
	// gaussian and gamma, so the same loglik and rank give a different aic by
	// family.
	check(close_rel(statsduck::glm_aic(statsduck::Family::kPoisson, -10.0, 3), 26.0, 1e-15),
	      "poisson aic: -2*(-10) + 2*3 = 26, d = 0");
	check(close_rel(statsduck::glm_aic(statsduck::Family::kBinomial, -10.0, 3), 26.0, 1e-15),
	      "binomial aic: d = 0, same as poisson for the same loglik and rank");
	check(close_rel(statsduck::glm_aic(statsduck::Family::kGaussian, -10.0, 3), 28.0, 1e-15),
	      "gaussian aic: -2*(-10) + 2*(3 + 1) = 28, d = 1");
	check(close_rel(statsduck::glm_aic(statsduck::Family::kGamma, -10.0, 3), 28.0, 1e-15),
	      "gamma aic: d = 1, same as gaussian for the same loglik and rank");

	// The gaussian log-likelihood is +inf at zero deviance (a perfect fit),
	// matching lm_core::loglik; the resulting aic is -inf. Both pass through
	// as values rather than becoming NULL (docs/fitter_conventions.md).
	const std::vector<double> yperf = {1.0, 2.0, 3.0};
	const double ll_perfect =
	    statsduck::glm_loglik(statsduck::Family::kGaussian, yperf, yperf, {1.0, 1.0, 1.0}, 1.0);
	check(std::isinf(ll_perfect) && ll_perfect > 0.0, "a perfect gaussian fit has loglik = +inf");
	check(statsduck::glm_aic(statsduck::Family::kGaussian, ll_perfect, 3) ==
	          -std::numeric_limits<double>::infinity(),
	      "and aic = -inf, passed through rather than becoming NULL");

	// The computation order ruling: pearson_dispersion must run before
	// glm_loglik for gamma, whose shape = 1/dispersion reads the argument.
	// Passing 1.0 instead of the real dispersion gives a different, wrong
	// log-likelihood, which this checks directly.
	const std::vector<double> ygam = {4.0, 6.0};
	const std::vector<double> mugam = {5.0, 5.0};
	const std::vector<double> wgam = {1.0, 1.0};
	const double disp_gam =
	    statsduck::pearson_dispersion(statsduck::Family::kGamma, ygam, mugam, wgam, 1);
	const double ll_right_order =
	    statsduck::glm_loglik(statsduck::Family::kGamma, ygam, mugam, wgam, disp_gam);
	const double ll_wrong_order =
	    statsduck::glm_loglik(statsduck::Family::kGamma, ygam, mugam, wgam, 1.0);
	check(disp_gam != 1.0, "this gamma fit's dispersion is not 1.0, so the two orders can differ");
	check(std::fabs(ll_right_order - ll_wrong_order) > 1e-6,
	      "gamma log-likelihood genuinely depends on calling pearson_dispersion first");
}

int main() {
	std::printf("== test_glm_core ==\n");
	test_round_trip();
	test_derivative_matches_central_difference();
	test_closed_forms_and_variance();
	test_parsing_and_link_tables();
	test_deviance();
	test_loglik_aic_dispersion();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL GLM_CORE TESTS PASSED\n");
		return 0;
	}
	return 1;
}
