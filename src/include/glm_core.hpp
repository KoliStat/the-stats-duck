#pragma once

// stats_duck GLM families, links, and the IRLS fitting loop (#22).
//
// IRLS stands for iteratively reweighted least squares. It is the
// algorithm fit_glm uses to fit a generalized linear model.
//
// The file is in two parts. The first part is the pure scalar and
// closed-form layer, all of it testable without a fitting loop: the
// family and link enums and their parsers; four scalar functions over a
// single observation, link_eta, link_inv, link_deriv and variance; and
// four closed-form quantities over a whole fit's y, mu and w vectors,
// deviance, glm_loglik, glm_aic and pearson_dispersion. Each of those
// takes the fitted values as a plain argument instead of computing them.
// The second part is fit_glm: the loop that produces the fitted values,
// its options, and its result.
//
// The two parts have one header and two sources. src/glm_core.cpp
// implements the first part and src/glm_fit_core.cpp the second. The
// header is deliberately not split: a caller wants one include for a
// GLM, and the two halves are one contract.
//
// DuckDB-free and Eigen-free at the boundary, like linalg.hpp and
// lm_core.hpp. This file includes lm_core.hpp for one reason: the Vcov
// enum, which GlmResult reuses so a GLM's standard errors are named the
// same way an OLS fit's are. This file never calls lm_core's fitting
// code. It does not call fit_lm, parse_vcov, or vcov_name. (lm_core.hpp
// also brings in linalg::Mat, through its own include of linalg.hpp.
// fit_glm takes its design matrix as a Mat.)

#include "lm_core.hpp" // statsduck::Vcov — the enum only, never fit_lm/parse_vcov

#include <cstddef>
#include <string>
#include <vector>

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

// Four closed-form quantities a fit reports: deviance, pearson_dispersion,
// glm_loglik, glm_aic. Each is computable from the fitted values alone.
// None needs a fitting loop to test, and this increment has no loop yet.
//
// A fit computes them in this order: deviance, then pearson_dispersion,
// then glm_loglik, then glm_aic. glm_loglik takes pearson_dispersion's
// result as its `dispersion` argument. glm_aic takes glm_loglik's result
// as its `loglik` argument.
//
// The order does not matter for binomial and poisson. pearson_dispersion
// always returns 1.0 there, no matter when it runs. The order does matter
// for gamma: glm_loglik's shape parameter is 1/dispersion, so passing a
// guessed 1.0 instead of the real dispersion gives a wrong log-likelihood.
// For gaussian, glm_loglik does not read `dispersion` at all. It
// recomputes its own sigma^2 from deviance/n directly. That is a
// different quantity from pearson_dispersion's deviance/df_residual for
// gaussian. The `dispersion` parameter still sits on every family's
// signature. A fitting loop calls pearson_dispersion once and passes that
// one value to every family, uniformly, rather than special-casing
// gaussian.
//
// The binomial response y is a PROPORTION in [0, 1], not a 0/1 count. The
// weight w is the trial count. So 7 successes out of 10 trials is
// y = 0.7, w = 10. A plain 0/1 logistic fit is the special case w = 1.

// Sigma of the per-observation deviance contributions, already doubled.
// A caller never multiplies by 2 again. The four families:
//   gaussian  Σ w·(y − mu)²
//   binomial  2 Σ w·[ y·ln(y/mu) + (1 − y)·ln((1 − y)/(1 − mu)) ]
//   poisson   2 Σ w·[ y·ln(y/mu) − (y − mu) ]
//   gamma     2 Σ w·[ −ln(y/mu) + (y − mu)/mu ]
// Gaussian carries no extra 2x factor. Its own algebra already cancels it,
// unlike the other three families.
//
// A binomial or Poisson y of exactly 0 makes a y·ln(y/mu) term compute
// 0·ln(0). A binomial y of exactly 1 makes the symmetric term do the
// same. Both are NaN in IEEE unless defined as 0 by convention. This is
// not an edge case. Every plain 0/1 logistic fit has y exactly 0 and
// exactly 1. An all-zero Poisson count hits the same 0·ln(0). Both terms
// are defined as 0 at that point, so the common case returns a value
// instead of NaN.
double deviance(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                const std::vector<double> &w);

// The log-likelihood at the fitted mu. Call pearson_dispersion first. See
// the dispersion argument's contract above this block.
//   gaussian  −n/2·(ln 2π σ² + 1),  σ² = deviance/n, n = y.size()
//             (matches lm_core::loglik; +inf at zero deviance)
//   binomial  Σ [ LogChoose(w, w·y) + w·y·ln mu + w·(1 − y)·ln(1 − mu) ]
//   poisson   Σ w·[ y·ln mu − mu − lgamma(y + 1) ]
//   gamma     shape = 1/dispersion, through std::lgamma
//
// LogChoose is distributions.hpp's (stats_duck::detail::LogChoose,
// stats_duck WITH an underscore). This file adds no new special function.
//
// Guarded against an UNCLAMPED mu of exactly 0.0 or exactly 1.0. This is
// not the clamped boundary: at mu = kMuEps, ln(mu) is about −23, an
// ordinary finite number, so 0·ln(kMuEps) is already 0 with no guard
// needed. link_inv always clamps, so a mu that came from it never reaches
// a true 0.0 or 1.0. The risk is a caller that bypasses link_inv and
// passes a raw mu straight in. The terms w·y·ln(mu) and w·(1 − y)·ln(1 − mu)
// use the same 0·ln(·) := 0 convention the deviance above does. Each is
// checked on y, so the log is never evaluated when its own coefficient is
// 0. Found while building this task's fixture generator: statsmodels does
// not clamp mu, and a perfectly separated binomial fit there drives mu to
// exactly 0.0 or exactly 1.0 in double precision. Feeding that raw mu
// through an unguarded version of this formula gave 0·(−inf), which is
// NaN rather than the 0 the limit requires.
double glm_loglik(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                   const std::vector<double> &w, double dispersion);

// Akaike information criterion: −2·loglik + 2·(rank + d). d is 1 for
// gaussian and gamma, whose dispersion is an estimated parameter. That is
// R's convention. d is 0 for binomial and poisson, whose dispersion is
// fixed at 1. statsmodels' own .aic never adds the extra parameter, for
// any family. So this project's aic is exactly 2 more than statsmodels'
// for gaussian and gamma, and equal to statsmodels' for binomial and
// poisson.
//
// The Gaussian log-likelihood is +inf at zero deviance, so its aic is
// −inf. Both pass through as values rather than becoming NULL. ±Inf is
// representable under docs/fitter_conventions.md.
double glm_aic(Family f, double loglik, std::size_t rank);

// The dispersion (scale) parameter, by Pearson's chi-square:
//   Σ w·(y − mu)² / variance(f, mu) / df_residual
// for gaussian and gamma. Exactly 1.0 for binomial and poisson, regardless
// of the data. A caller never has to know which families estimate a
// dispersion and which do not.
double pearson_dispersion(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                           const std::vector<double> &w, std::size_t df_residual);

// ─── The fit: options, result, and the IRLS loop ────────────────────────

// Options for one fit.
//
// `link` has no per-family default here. A caller holding a family and no
// link asks canonical_link for one and stores it in this field. fit_glm
// rejects a link the family does not allow, through link_allowed above.
//
// `vcov` names the covariance estimator. It reuses lm_core's enum, so a
// GLM's standard errors are named the way an OLS fit's are. All seven are
// computed. kConst is dispersion * (Xt W X) inverse. The other six are
// the score-residual sandwich, which the fit_glm comment below sets out
// in full.
//
// `max_iter` and `tol` are R's glm.control defaults: at most 25 outer
// iterations, and a relative deviance change below 1e-8.
struct GlmOptions {
	Family family = Family::kGaussian;
	Link link = Link::kIdentity; // set from canonical_link unless given
	Vcov vcov = Vcov::kConst;
	bool intercept = true; // prepend a column of 1s to the design matrix
	int max_iter = 25;
	double tol = 1e-8;
};

// The result of one fit. There are three outcomes, and a caller has to
// tell them apart:
//
//   ok = true,  converged = true    The fit converged.
//   ok = true,  converged = false   The loop used every iteration, or a
//                                   step failed and could not be
//                                   recovered. The best iterate is still
//                                   reported in full. This is a value,
//                                   not a NULL group.
//   ok = false, error populated     A data-shaped failure. The SQL layer
//                                   reports the group as NULL and never
//                                   raises. See
//                                   docs/fitter_conventions.md.
//
// A step that 25 halvings could not improve is NOT by itself
// converged = false. The loop tests the convergence criterion on that
// step before giving up, and reports converged = true when it holds. The
// reasoning is that the IRLS direction is a descent direction unless the
// score is already zero, so a step that cannot lower the deviance at all
// means the score is at rounding level, which is convergence. A random
// search over 180,000 fits found 14 that converged through this branch,
// and a pattern search from each reported beta could not improve any of
// them.
//
// terms, beta, std_error, z_statistic and p_value are all length k, the
// full design width, and cov is k*k. A column dropped as dependent is
// NaN in beta, std_error, z_statistic and p_value, and in its own row
// and column of cov, while `terms` still names it. That is R's
// convention and #36's: the earlier column of a dependent set is kept,
// the later one is aliased.
struct GlmResult {
	bool ok = false;
	std::string error;

	std::size_t n = 0;           // observations used
	std::size_t k = 0;           // parameters (predictors + intercept)
	std::size_t rank = 0;        // parameters actually estimated
	std::size_t n_clusters = 0;  // #clusters G (vcov CR* only); 0 when unclustered
	std::size_t df_residual = 0; // n - rank

	std::vector<std::string> terms; // length k: "(Intercept)", "x1", "x2", ...
	std::vector<double> beta;
	std::vector<double> std_error;
	// beta / std_error, against a NORMAL reference for all four families,
	// with p_value = 2 * (1 - NormalCDF(|z|)). R and statsmodels switch to
	// a t reference for gaussian and gamma, whose dispersion is estimated.
	// This project does not. The deviation is deliberate and it is written
	// down in the fixtures, which carry both the oracle's t-based p-value
	// and the normal-based one this kernel must produce.
	std::vector<double> z_statistic;
	std::vector<double> p_value;
	// k*k row-major coefficient covariance, in `terms` order:
	// cov[i*k + j] = cov(beta_i, beta_j). The diagonal is std_error
	// squared. Rows and columns of an aliased term are NaN.
	std::vector<double> cov;

	double deviance = 0.0;
	// The intercept-only model's deviance, from actually fitting that
	// model with the same family, link, weights and offset. NaN when that
	// fit itself fails.
	double null_deviance = 0.0;
	double dispersion = 1.0; // exactly 1 for binomial and poisson
	double loglik = 0.0;
	double aic = 0.0;
	// Outer IRLS passes used, never the step halvings inside them. A fit
	// that converges on its third pass reports 3 whether or not steps were
	// halved inside those passes. A fit with no parameters to estimate
	// reports 0: there is nothing to iterate.
	int iterations = 0;
	bool converged = false;
	// Always false in this increment. The next increment adds the binomial
	// separation heuristic that sets it.
	bool separation = false;
};

// Fit y (length n) on the predictor matrix X_pred (n x p, row-major,
// WITHOUT an intercept column — opts.intercept controls whether a
// constant is prepended), by iteratively reweighted least squares.
//
// A ZERO-COLUMN X_pred is a legitimate input, not misuse. With
// opts.intercept true the prepended constant is then the only column,
// which is how null_deviance is computed. With opts.intercept false as
// well there are no parameters at all, and mu is the link inverse of the
// offset alone. That case reports rank 0, iterations 0 and converged
// true, because there is nothing to iterate.
//
// weights: optional, length n, the PRIOR weights. For binomial these are
// the trial counts and y is the proportion of successes, so 7 of 10 is
// y = 0.7 and w = 10. Defaults to 1 for every row.
//
// A row of weight exactly 0 influences nothing: not the aliased set, not
// the solve, not the deviance, and not the support test on the fitted
// mean. Its contribution is 0 either way. That invariant holds for a
// caller reaching this kernel directly, which docs/kernel_api.md invites
// a sibling repository to do, and not only through the aggregate.
//
// The one thing such a row does affect is `n`: the kernel counts every
// row it is given, so a zero-weight row raises n and therefore
// df_residual. The SQL layer drops those rows at accumulation, like a
// NULL, so they do not reach here through that path.
//
// offset: optional, length n, added to the linear predictor and not
// estimated. A poisson model of a rate uses log(exposure) here.
//
// cluster_ids: optional, length n, DENSE 0-based cluster labels (0..G-1),
// as densify_clusters (cluster_ids.hpp) produces. Required when opts.vcov
// is kCR0 or kCR1, and ignored otherwise. Fewer than two clusters leaves
// the cluster-robust covariance undefined, so that is a failed fit.
//
// DENSE IS A PRECONDITION, NOT A PREFERENCE, and violating it is not
// reported. G is taken as the LARGEST id plus one, which is fit_lm's
// convention and is kept identical to it on purpose. So ids {0, 2, 4}
// give G = 5 from three labels. Three things then change silently: CR1's
// [G/(G-1)] factor shrinks, the reported n_clusters is the inflated G,
// and the per-cluster score buffer allocates G*rank doubles from the
// largest id. CR0 is unaffected, because an empty cluster contributes a
// zero score sum. Pass ids through densify_clusters and none of this
// arises.
//
// The loop:
//   mu    = R's glm.fit starting values: (w*y + 0.5)/(w + 1) binomial,
//           y + 0.1 poisson, y gaussian and gamma
//   eta   = link_eta(mu)
//   repeat up to opts.max_iter times:
//     z   = eta - offset + (y - mu) / link_deriv(eta)
//     W   = w * link_deriv(eta)^2 / variance(mu)
//     solve (Xt W X) beta = Xt W z over the kept columns
//     eta = X beta + offset,  mu = link_inv(eta)
//     halve the step, at most 25 times, while it is bad: the new
//     coefficients or the new deviance are not finite, a fitted mean on
//     a POSITIVE-WEIGHT row has left the family's support, or the
//     deviance increased
//     stop when |dev - dev_prev| / (|dev| + 0.1) < opts.tol
//
// The support test is R's validmu, and it is not redundant with
// link_inv's clamp. link_inv clamps a probability link away from 0 and 1,
// but the identity link is never clamped and the log link is clamped at
// the floor only. Poisson allows an identity link and binomial allows a
// log link, so without this test a fit can converge to a negative
// expected count or to a fitted probability above 1, and every quantity
// computed from those fitted values is then wrong. See MuInSupport in
// glm_core.cpp for the mechanism and for the two inputs that found it.
//
// The test looks only at rows of positive prior weight, which is the same
// rule the normal matrix and the aliased set follow. A zero-weight row
// influences no estimate, but a fitted mean is still computed for it from
// its own covariates, so testing it would let one weightless row with an
// extreme covariate fail a fit that is sound over every row that counts.
//
// The solve is linalg::cholesky_solve on Xt W X, falling back to
// linalg::svd_solve when the Cholesky reports the matrix is not positive
// definite. That mirrors fit_lm.
//
// Step halving needs a previous coefficient vector to halve toward, and
// the first pass has none: the starting point is a mean vector, not a
// coefficient vector. So the first pass takes its step as it stands, and
// a first step that is not finite, or that leaves the family's support,
// is a failed fit. R's glm.fit has the same restriction, and stops there
// with "no valid set of coefficients has been found".
//
// THE FIRST PASS ALMOST ALWAYS INCREASES THE DEVIANCE, for every family,
// and that is normal rather than a problem. The starting values are at or
// near the saturated model, where the deviance is at its own minimum:
// gaussian and gamma start at mu = y, so their starting deviance is
// exactly 0; poisson starts at mu = y + 0.1, which is 0.0052 on DS3
// against a fitted 17.1; binomial starts at (w*y + 0.5)/(w + 1), which is
// 23.0 on DS1 against a fitted 51.9. So a rule that halved on an
// increased deviance from the first pass onward would halve away every
// first step and the loop would never start. Measured: applying the
// halving test to the first pass collapses every fit in this project's
// fixtures to all-zero coefficients, the gaussian identity one-step solve
// included. Do not reinstate it for any family.
//
// THE RANK-DEFICIENT COLUMN SET IS COMPUTED ONCE AND HELD FIXED for the
// whole loop. This deviates from R, which re-pivots every iteration.
// Holding it fixed is justified rather than merely convenient: for
// strictly positive weights rank(W^1/2 X) = rank(X), because W^1/2 is
// invertible, so the set cannot change between iterations. Fixing it also
// makes the fit deterministic where re-pivoting could oscillate. The set
// comes from linalg::independent_columns on X restricted to the rows
// whose PRIOR weight is positive, because a zero-weight row contributes
// nothing to the solve and can change the rank.
//
// THE COVARIANCE IS EVALUATED AT THE REPORTED ESTIMATE: W is recomputed
// from the final eta and mu. The classical estimator is then
// dispersion * (Xt W X) inverse.
//
// This deviates from statsmodels, which evaluates its covariance at the
// working weights of the PREVIOUS iterate. The deviation is deliberate
// and it was measured. Standard-error gaps against statsmodels do not
// shrink as tol tightens: they are identical at 1e-8, at 1e-12 and at 0,
// while the coefficients close to 2.7e-12. Patching this kernel to use
// the last solve's weights collapsed DS1's gap from 4.78e-07 to
// 1.14e-15, which reproduces the mechanism to machine precision.
// Recomputing at the final fitted values is the correct quantity, and it
// is also the self-consistent one: this kernel's grouped and ungrouped
// binomial standard errors agree to 4.29e-12 on data that is the same in
// two shapes, where statsmodels' own two shapes disagree by 4.2e-07. So
// expect an oracle band near 1e-5 on a standard error and near 1e-4 on a
// robust one, and do not read it as an error in this kernel.
//
// THE ROBUST ESTIMATORS USE SCORE RESIDUALS, not response residuals.
// That is the one real difference from lm_core. The score contribution
// of observation i is the vector x_i * u_i with the scalar
//   u_i = w_i * (y_i - mu_i) * link_deriv(eta_i) / variance(mu_i)
// and A = Xt W X is the bread, so:
//   HC0  A inverse * sum_i u_i^2 x_i x_i' * A inverse
//   HC1  HC0 times n / (n - rank)
//   HC2  divide each score by (1 - h_i)^(1/2)
//   HC3  divide each score by (1 - h_i)
//   CR0  sum the scores within a cluster before the outer product
//   CR1  CR0 times [G/(G-1)]*[(N-1)/(N-rank)]
// h_i = w_i * x_i' A inverse x_i is the hat diagonal of the WEIGHTED
// design, with w_i the working weight. None of the six multiplies by the
// dispersion: the meat already carries the scale of the data.
//
// At gaussian with an identity link link_deriv is 1 and variance is 1,
// so u_i is w_i * (y_i - mu_i) and all six reduce to lm_core's
// estimators. That reduction is exact, bit for bit, and test_glm_fit.cpp
// asserts it at a band of 0.
//
// n_clusters is G under CR0 and CR1 and 0 otherwise, so the SQL layer
// writes NULL there. df_residual stays n - rank under every estimator,
// because it is a property of the fit rather than of the covariance.
// z_statistic keeps its normal reference under every estimator too,
// cluster-robust included: this kernel reports z for all four families
// and so has no t reference to switch to. A consumer that wants the
// G - 1 reference reads n_clusters.
//
// Reported quantities are computed in this order, and the order is part
// of the contract: the fitted values, then deviance, then
// pearson_dispersion, then glm_loglik, then glm_aic. glm_loglik takes
// pearson_dispersion's value, never a placeholder 1.0. For binomial and
// poisson pearson_dispersion is always exactly 1.0, so the order does not
// matter there. For gaussian and gamma it does.
//
// Data-shaped failures, each ok = false with a populated `error`:
//   - y empty, or y.size() different from X_pred.rows;
//   - a supplied weights or offset whose length is not y.size();
//   - opts.max_iter below 1, which would report a result with no fit in it;
//   - a link the family does not allow (link_allowed);
//   - a non-finite y; a binomial y outside [0, 1]; a negative poisson y;
//     a gamma y not strictly positive; a negative or non-finite weight; a
//     non-finite offset;
//   - every row of weight 0, so nothing informs the fit;
//   - a design with no independent column at all;
//   - n <= rank, which leaves no residual degrees of freedom. The
//     dispersion divides by n - rank, so this has to be decided before
//     any division rather than after it;
//   - kCR0 or kCR1 with no cluster ids, with ids that are not dense
//     non-negative labels, or with fewer than two clusters;
//   - a first step whose deviance or coefficients are not finite, or
//     whose fitted mean leaves the family's support, which the paragraph
//     on step halving above explains, or a first weighted solve that
//     fails outright, which means the design is numerically singular at
//     the starting values. Those conditions are failures only on the
//     FIRST pass. On a later pass the loop stops and reports the iterate
//     it had already accepted, with converged false;
//   - a final Xt W X that cannot be inverted, which leaves the standard
//     errors undefined even though the coefficients converged.
// A rank-deficient design is NOT one of them: it is a fit, with the
// dependent columns aliased.
GlmResult fit_glm(const std::vector<double> &y, const linalg::Mat &X_pred, const GlmOptions &opts,
                  const std::vector<double> *weights = nullptr,
                  const std::vector<double> *offset = nullptr,
                  const std::vector<int> *cluster_ids = nullptr);

} // namespace statsduck
