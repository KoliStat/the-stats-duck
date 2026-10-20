#pragma once

// stats_duck GLM families and links (#22).
//
// This file holds the pure scalar functions the IRLS loop will use. IRLS
// stands for iteratively reweighted least squares. It is the algorithm
// glm_fit uses to fit a generalized linear model. This increment has no
// loop yet. It only has the family and link enums, their parsers, and four
// scalar functions: link_eta, link_inv, link_deriv, and variance.
//
// DuckDB-free and Eigen-free at the boundary, like linalg.hpp and
// lm_core.hpp. This file includes lm_core.hpp for one reason: the Vcov
// enum. A later task adds the glm_fit result struct, and that struct
// reuses Vcov so a GLM's standard errors are named the same way an OLS
// fit's are. This file never calls lm_core's fitting code. It does not
// call fit_lm, parse_vcov, or vcov_name. (lm_core.hpp also brings in
// linalg::Mat, through its own include of linalg.hpp. Nothing in this
// increment uses Mat either; a later task's IRLS loop will.)
//
// glm_core.hpp will grow. The IRLS loop and the glm_fit result type come
// in a later task and will live here too.

#include "lm_core.hpp" // statsduck::Vcov — the enum only, never fit_lm/parse_vcov

#include <string>

namespace statsduck {

// The exponential-family distribution assumed for the response.
// Four families in this increment. Negative binomial is not one of them.
enum class Family { kGaussian, kBinomial, kPoisson, kGamma };

// The link function g. It relates the linear predictor eta to the mean mu.
//   eta = g(mu)
//   mu  = g inverse(eta)
enum class Link { kIdentity, kLogit, kProbit, kCLogLog, kLog, kInverse, kSqrt };

// Parse a family name. Case-insensitive, matching parse_vcov.
// Accepts "gaussian", "binomial", "poisson", "gamma".
// Returns false and leaves `out` untouched for anything else, including
// "negbin": negative binomial is not in this increment.
// The caller picks the exception type and message, the same contract
// parse_vcov uses.
bool parse_family(const std::string &name, Family &out);

// Parse a link name. Case-insensitive, matching parse_vcov.
// Accepts "identity", "logit", "probit", "cloglog", "log", "inverse",
// "sqrt". Returns false and leaves `out` untouched for anything else.
bool parse_link(const std::string &name, Link &out);

// The canonical name for a family or a link. glm_fit echoes this back.
// parse_family(family_name(f), g) sets g equal to f. The same holds for
// parse_link and link_name.
const char *family_name(Family f);
const char *link_name(Link l);

// The family's default link: the link used when a caller gives a family
// but no link.
//   gaussian -> identity   binomial -> logit
//   poisson  -> log        gamma    -> inverse
Link canonical_link(Family f);

// Whether link l makes sense for family f.
//
// This follows R's family() link list, limited to the seven links above.
// R also allows "cauchit" for binomial; this file has no cauchit link, so
// it is left out.
//   gaussian   identity, log, inverse
//   binomial   logit, probit, cloglog, log
//   poisson    log, identity, sqrt
//   gamma      inverse, log, identity
//
// Two rules a test checks directly: a probability link (logit, probit,
// cloglog) is allowed only for binomial. inverse is not allowed for
// binomial; logit is binomial's own canonical link instead.
//
// Every family allows its own canonical link. A caller can give just a
// family and skip the link only because this holds for all four.
bool link_allowed(Family f, Link l);

// The floor mu is clamped to. For the three probability links, 1 minus
// this is the ceiling. See link_inv below for where the clamp applies.
//
// Without a clamp, an extreme eta can push mu all the way to the boundary
// of its domain. For example, the logit inverse underflows to exactly 0
// around eta = -745. A later task computes the deviance from mu, and the
// deviance takes log(mu) or divides by mu. At mu = 0 that returns infinity
// for a fit that should simply report a very small probability.
//
// 1e-10 is small enough to leave a converged fit unchanged: a real mu this
// close to the boundary already carries almost no information. It is also
// far enough from 0 that mu * (1 - mu) — used below, in the binomial
// variance and in the logit derivative — never underflows to exactly 0
// either.
constexpr double kMuEps = 1e-10;

// g inverse: the fitted mean for a given linear predictor value.
//
// Clamped per kMuEps above. The three probability links (logit, probit,
// cloglog) clamp to [kMuEps, 1 - kMuEps]. log, inverse, and sqrt clamp to
// at least kMuEps. identity is never clamped: a Gaussian mean can be any
// real number, so there is no boundary to clamp against.
//
// inverse is the one case where "at least kMuEps" needs care: 1/eta is
// negative for a negative eta, and a plain floor at kMuEps would force
// that negative value up into a tiny positive one, which is not a clamp,
// it is a different, wrong answer. inverse instead floors away from 0 on
// whichever side the raw value already falls on: at least kMuEps when it
// is non-negative, at most -kMuEps when it is negative. log and sqrt do
// not need this: e^eta and eta^2 are never negative to begin with.
//
// link_inv(link_eta(mu)) round-trips to mu only when mu is inside this
// clamped range. Outside it, the clamp is the point, so a round trip is
// not expected there.
double link_inv(Link l, double eta);

// g: the linear predictor for a given mean.
//
// mu must already sit inside the link's domain: the open interval (0, 1)
// for the three probability links, mu > 0 for log, inverse, and sqrt, and
// any real number for identity.
//
// link_eta does not clamp mu. At the exact boundary (mu = 0 or 1 for a
// probability link, mu = 0 for log, inverse, or sqrt) it returns the
// mathematically correct positive or negative infinity, not a clamped
// finite number. distributions.hpp's NormalQuantile already works this
// way at p = 0 and p = 1, so this matches it.
//
// This is a deliberate choice, not an oversight. The next task's IRLS loop
// calls link_eta only on a starting mu, and R builds its own starting
// values to stay off the boundary — for example (y + 0.5) / 2 for
// binomial data. Keeping mu off the boundary is that caller's
// responsibility. link_eta does not take it on here.
//
// probit is not a direct call to NormalQuantile. NormalQuantile is
// accurate to about 1e-9 (Acklam's rational approximation; see
// distributions.hpp), which is not tight enough for this file's own
// round-trip test: it holds every link to 1e-10 relative, and the
// measured round-trip error through a direct NormalQuantile call reached
// 4.9e-9 at mu = 0.001. link_eta's probit case instead sharpens
// NormalQuantile's result with one Newton correction computed from the
// exact NormalCDF and NormalPDF, which is still the same mathematical
// quantity and not a second approximation of the normal distribution.
// See ProbitQuantileRefined in glm_core.cpp for the formula, and for why
// it is skipped at the exact boundary, where NormalQuantile already
// returns the exact, unrefinable infinity.
double link_eta(Link l, double mu);

// dmu/deta at a given linear predictor value. The IRLS loop uses this for
// its working weights.
//
// This function adds no guard beyond what the formula and link_inv
// already give.
//
// logit's derivative is computed from the already-clamped link_inv (see
// the case below), so it stays inside [kMuEps * (1 - kMuEps), 0.25] for
// every eta. cloglog's e^(eta - e^eta) settles to 0 at both extremes by
// its own shape: its inner e^eta can overflow to infinity first, but
// finite minus infinity is still negative infinity, and e^(-infinity) is
// still 0, so the final answer comes out right anyway. probit's
// NormalPDF(eta) only ever decays toward 0. None of these three can
// return infinity.
//
// log's e^eta can return infinity, for an eta above about 709.78. This
// file does not guard that case. link_inv for the same link is clamped at
// the floor only, not at the ceiling, so link_inv(kLog, eta) can already
// return infinity at that same eta — that is this file's own choice, not
// a gap. Since the fitted mean can already be infinite there, the loop
// that calls this function already needs to check for a non-finite mu, or
// a non-finite deviance, and recover by halving its step, the way R's
// glm.fit does. That check is the one place this needs handling. A second,
// partial guard here would only duplicate it.
double link_deriv(Link l, double eta);

// V(mu): the variance function of the family, up to the dispersion
// parameter phi. The working variance is phi * variance(f, mu); phi is
// not covered here.
//   gaussian   1            binomial   mu * (1 - mu)
//   poisson    mu           gamma      mu^2
double variance(Family f, double mu);

} // namespace statsduck
