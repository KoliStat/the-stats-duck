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

} // namespace statsduck
