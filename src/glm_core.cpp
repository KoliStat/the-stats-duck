// stats_duck GLM families, links and closed-form statistics (#22).
//
// This file holds the pure layer: the family and link enums, their
// parsers, the four scalar functions over one observation, and the four
// closed-form quantities over a whole fit's y, mu and w vectors. None of
// it fits anything. Every function here takes the fitted values as an
// argument instead of computing them, which is what makes the whole file
// testable without a loop.
//
// The fitting machinery lives in src/glm_fit_core.cpp: the starting
// values, the support tests, the normal-matrix assembly, the IRLS loop
// and the coefficient covariance. Both files declare into
// src/include/glm_core.hpp, which is not split.

#include "glm_core.hpp"

#include "distributions.hpp" // stats_duck::NormalCDF / NormalQuantile / NormalPDF

#include <algorithm>
#include <cmath>
#include <limits>

namespace statsduck {

namespace {

// Lower-case a copy (ASCII). Used for case-insensitive family/link
// parsing, the same way lm_core.cpp's own ToLower is used for parse_vcov.
std::string ToLower(const std::string &s) {
	std::string out = s;
	std::transform(out.begin(), out.end(), out.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return out;
}

// NormalQuantile (distributions.hpp) is Acklam's rational approximation,
// accurate to about 1e-9. That is not tight enough for this file's own
// round-trip test, which holds every link to 1e-10 relative: measured
// worst case 4.9e-9 at mu = 0.001 without this refinement.
//
// One Newton correction, using the exact NormalCDF and NormalPDF, is a
// standard way to sharpen this specific approximation. Acklam's own paper
// describes a refinement step along these lines, there using Halley's
// method; Newton's is simpler and, starting from an already-good estimate,
// is more than enough to clear the 1e-10 band. This still computes the
// same mathematical quantity, NormalQuantile(mu); it is not a second
// implementation of the normal distribution, since both the value being
// corrected and the correction itself come from the one CDF this file
// already depends on.
//
// Skipped when NormalQuantile itself is already infinite, at mu = 0 or
// mu = 1: that is the exact, unrefinable answer, and NormalPDF there is 0,
// so the correction below would divide 0 by 0 instead of leaving infinity
// alone.
double ProbitQuantileRefined(double mu) {
	const double z0 = stats_duck::NormalQuantile(mu);
	if (!std::isfinite(z0)) {
		return z0;
	}
	return z0 - (stats_duck::NormalCDF(z0) - mu) / stats_duck::NormalPDF(z0);
}

} // namespace

bool parse_family(const std::string &name, Family &out) {
	const std::string n = ToLower(name);
	if (n == "gaussian") {
		out = Family::kGaussian;
	} else if (n == "binomial") {
		out = Family::kBinomial;
	} else if (n == "poisson") {
		out = Family::kPoisson;
	} else if (n == "gamma") {
		out = Family::kGamma;
	} else {
		return false;
	}
	return true;
}

bool parse_link(const std::string &name, Link &out) {
	const std::string n = ToLower(name);
	if (n == "identity") {
		out = Link::kIdentity;
	} else if (n == "logit") {
		out = Link::kLogit;
	} else if (n == "probit") {
		out = Link::kProbit;
	} else if (n == "cloglog") {
		out = Link::kCLogLog;
	} else if (n == "log") {
		out = Link::kLog;
	} else if (n == "inverse") {
		out = Link::kInverse;
	} else if (n == "sqrt") {
		out = Link::kSqrt;
	} else {
		return false;
	}
	return true;
}

const char *family_name(Family f) {
	switch (f) {
	case Family::kGaussian:
		return "gaussian";
	case Family::kBinomial:
		return "binomial";
	case Family::kPoisson:
		return "poisson";
	case Family::kGamma:
		return "gamma";
	}
	return "gaussian"; // unreachable
}

const char *link_name(Link l) {
	switch (l) {
	case Link::kIdentity:
		return "identity";
	case Link::kLogit:
		return "logit";
	case Link::kProbit:
		return "probit";
	case Link::kCLogLog:
		return "cloglog";
	case Link::kLog:
		return "log";
	case Link::kInverse:
		return "inverse";
	case Link::kSqrt:
		return "sqrt";
	}
	return "identity"; // unreachable
}

Link canonical_link(Family f) {
	switch (f) {
	case Family::kGaussian:
		return Link::kIdentity;
	case Family::kBinomial:
		return Link::kLogit;
	case Family::kPoisson:
		return Link::kLog;
	case Family::kGamma:
		return Link::kInverse;
	}
	return Link::kIdentity; // unreachable
}

bool link_allowed(Family f, Link l) {
	switch (f) {
	case Family::kGaussian:
		return l == Link::kIdentity || l == Link::kLog || l == Link::kInverse;
	case Family::kBinomial:
		return l == Link::kLogit || l == Link::kProbit || l == Link::kCLogLog || l == Link::kLog;
	case Family::kPoisson:
		return l == Link::kLog || l == Link::kIdentity || l == Link::kSqrt;
	case Family::kGamma:
		return l == Link::kInverse || l == Link::kLog || l == Link::kIdentity;
	}
	return false; // unreachable
}

double link_inv(Link l, double eta) {
	switch (l) {
	case Link::kIdentity:
		return eta;
	case Link::kLogit: {
		const double mu = 1.0 / (1.0 + std::exp(-eta));
		return std::min(1.0 - kMuEps, std::max(kMuEps, mu));
	}
	case Link::kProbit: {
		const double mu = stats_duck::NormalCDF(eta);
		return std::min(1.0 - kMuEps, std::max(kMuEps, mu));
	}
	case Link::kCLogLog: {
		const double mu = 1.0 - std::exp(-std::exp(eta));
		return std::min(1.0 - kMuEps, std::max(kMuEps, mu));
	}
	case Link::kLog:
		return std::max(kMuEps, std::exp(eta));
	case Link::kInverse: {
		// 1/eta can be negative, when eta is. A plain floor at kMuEps would
		// force a negative mu up to a tiny positive one instead of leaving it
		// alone — the wrong fix, not a smaller one. Floor it away from 0 on
		// whichever side it already falls on instead: a non-negative mu is
		// floored up to kMuEps, a negative one is capped down to -kMuEps.
		const double mu = 1.0 / eta;
		return mu >= 0.0 ? std::max(kMuEps, mu) : std::min(-kMuEps, mu);
	}
	case Link::kSqrt:
		return std::max(kMuEps, eta * eta);
	}
	return std::numeric_limits<double>::quiet_NaN(); // unreachable
}

double link_eta(Link l, double mu) {
	switch (l) {
	case Link::kIdentity:
		return mu;
	case Link::kLogit:
		return std::log(mu / (1.0 - mu));
	case Link::kProbit:
		return ProbitQuantileRefined(mu);
	case Link::kCLogLog:
		return std::log(-std::log(1.0 - mu));
	case Link::kLog:
		return std::log(mu);
	case Link::kInverse:
		return 1.0 / mu;
	case Link::kSqrt:
		return std::sqrt(mu);
	}
	return std::numeric_limits<double>::quiet_NaN(); // unreachable
}

double link_deriv(Link l, double eta) {
	switch (l) {
	case Link::kIdentity:
		return 1.0;
	case Link::kLogit: {
		// mu comes from the already-clamped link_inv. See the header: this
		// keeps the derivative away from 0 and away from infinity for every
		// eta, with no extra guard needed here.
		const double mu = link_inv(Link::kLogit, eta);
		return mu * (1.0 - mu);
	}
	case Link::kProbit:
		return stats_duck::NormalPDF(eta);
	case Link::kCLogLog:
		// Settles to 0 at both extremes; see the header for why this stays
		// correct even though the inner e^eta overflows first for a large eta.
		return std::exp(eta - std::exp(eta));
	case Link::kLog:
		// Can overflow to infinity for an eta above about 709.78. See the
		// header: the IRLS loop guards against that, not this function.
		return std::exp(eta);
	case Link::kInverse:
		return -1.0 / (eta * eta);
	case Link::kSqrt:
		return 2.0 * eta;
	}
	return std::numeric_limits<double>::quiet_NaN(); // unreachable
}

double variance(Family f, double mu) {
	switch (f) {
	case Family::kGaussian:
		return 1.0;
	case Family::kBinomial:
		return mu * (1.0 - mu);
	case Family::kPoisson:
		return mu;
	case Family::kGamma:
		return mu * mu;
	}
	return std::numeric_limits<double>::quiet_NaN(); // unreachable
}

double deviance(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                const std::vector<double> &w) {
	const std::size_t n = y.size();
	double total = 0.0;
	switch (f) {
	case Family::kGaussian:
		for (std::size_t i = 0; i < n; i++) {
			const double d = y[i] - mu[i];
			total += w[i] * d * d;
		}
		return total;
	case Family::kBinomial:
		for (std::size_t i = 0; i < n; i++) {
			// y·ln(y/mu) is 0·ln(0) at y = 0. The symmetric term,
			// (1-y)·ln((1-y)/(1-mu)), is 0·ln(0) at y = 1. Both are NaN
			// in IEEE unless defined as 0 here, by the usual x·ln(x)
			// convention at x = 0. Every plain 0/1 logistic fit hits
			// this. It is the common case, not an edge case.
			double term = 0.0;
			if (y[i] != 0.0) {
				term += y[i] * std::log(y[i] / mu[i]);
			}
			if (y[i] != 1.0) {
				term += (1.0 - y[i]) * std::log((1.0 - y[i]) / (1.0 - mu[i]));
			}
			total += w[i] * term;
		}
		return 2.0 * total;
	case Family::kPoisson:
		for (std::size_t i = 0; i < n; i++) {
			// Same guard, same reason: y·ln(y/mu) is 0·ln(0) at y = 0,
			// which all-zero count data hits.
			double term = -(y[i] - mu[i]);
			if (y[i] != 0.0) {
				term += y[i] * std::log(y[i] / mu[i]);
			}
			total += w[i] * term;
		}
		return 2.0 * total;
	case Family::kGamma:
		for (std::size_t i = 0; i < n; i++) {
			total += w[i] * (-std::log(y[i] / mu[i]) + (y[i] - mu[i]) / mu[i]);
		}
		return 2.0 * total;
	}
	return std::numeric_limits<double>::quiet_NaN(); // unreachable
}

double glm_loglik(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                   const std::vector<double> &w, double dispersion) {
	const std::size_t n = y.size();
	switch (f) {
	case Family::kGaussian: {
		// Ignores `dispersion`. Sigma^2 is recomputed from deviance/n
		// directly below, matching lm_core::loglik. See the header for
		// why that is a different quantity from the passed-in argument.
		const double nd = static_cast<double>(n);
		const double sigma2 = deviance(f, y, mu, w) / nd;
		const double two_pi = 2.0 * 3.14159265358979323846;
		// Two separate std::log calls, summed, not one log of a product.
		// Mathematically the same value, but this is the exact sequence
		// lm_core.cpp uses, kept identical so the two never drift apart
		// by a rounding bit on the same input.
		return -0.5 * nd * (std::log(two_pi) + std::log(sigma2) + 1.0);
	}
	case Family::kBinomial: {
		double total = 0.0;
		for (std::size_t i = 0; i < n; i++) {
			// w*y*ln(mu) and w*(1-y)*ln(1-mu) each skip their own log
			// entirely when their own coefficient (y, or 1-y) is exactly
			// 0. This is the same 0*ln(.) := 0 convention the deviance
			// above uses. This guards an UNCLAMPED mu of exactly 0.0 or
			// 1.0, not the clamped boundary: at mu = kMuEps, ln(mu) is
			// about -23, an ordinary finite number, so 0*ln(kMuEps) is
			// already 0 with no guard needed. link_inv always clamps mu,
			// so the real risk is a caller that bypasses link_inv and
			// passes a raw mu straight in. Guarding on y keeps a true
			// ln(0) from ever being multiplied against a zero coefficient
			// to produce NaN instead of 0.
			if (y[i] != 0.0) {
				total += w[i] * y[i] * std::log(mu[i]);
			}
			if (y[i] != 1.0) {
				total += w[i] * (1.0 - y[i]) * std::log(1.0 - mu[i]);
			}
			total += stats_duck::detail::LogChoose(w[i], w[i] * y[i]);
		}
		return total;
	}
	case Family::kPoisson: {
		double total = 0.0;
		for (std::size_t i = 0; i < n; i++) {
			// Same guard as binomial, on the same term shape: y*ln(mu) is
			// skipped at y = 0 rather than computed and discarded.
			double term = -mu[i] - std::lgamma(y[i] + 1.0);
			if (y[i] != 0.0) {
				term += y[i] * std::log(mu[i]);
			}
			total += w[i] * term;
		}
		return total;
	}
	case Family::kGamma: {
		double total = 0.0;
		const double shape = 1.0 / dispersion;
		for (std::size_t i = 0; i < n; i++) {
			const double k = w[i] * shape; // the effective shape for this row
			total += (k - 1.0) * std::log(y[i]) - y[i] * k / mu[i] - std::lgamma(k) -
			         k * std::log(mu[i] / k);
		}
		return total;
	}
	}
	return std::numeric_limits<double>::quiet_NaN(); // unreachable
}

double glm_aic(Family f, double loglik, std::size_t rank) {
	const double d = (f == Family::kGaussian || f == Family::kGamma) ? 1.0 : 0.0;
	return -2.0 * loglik + 2.0 * (static_cast<double>(rank) + d);
}

double pearson_dispersion(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                          const std::vector<double> &w, std::size_t df_residual) {
	if (f == Family::kBinomial || f == Family::kPoisson) {
		return 1.0;
	}
	double total = 0.0;
	for (std::size_t i = 0; i < y.size(); i++) {
		const double d = y[i] - mu[i];
		total += w[i] * d * d / variance(f, mu[i]);
	}
	return total / static_cast<double>(df_residual);
}

} // namespace statsduck
