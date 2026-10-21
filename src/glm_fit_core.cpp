// stats_duck GLM fitting machinery: the IRLS loop and its covariance (#22).
//
// This file is the second half of glm_core.cpp, split out when the robust
// covariance landed. src/glm_core.cpp holds the pure layer: the family and
// link enums, their parsers, the four scalar link functions, the variance
// function, and the closed-form deviance, log-likelihood, AIC and
// dispersion. This file holds everything that fits: the starting values,
// the support tests, the normal-matrix assembly, the loop itself, and the
// coefficient covariance.
//
// Both halves declare into src/include/glm_core.hpp. The header is not
// split, because a caller wants one include for a GLM and the two halves
// are one contract.
//
// DuckDB-free and Eigen-free at the boundary, like glm_core.cpp.

#include "glm_core.hpp"

#include "distributions.hpp" // stats_duck::NormalCDF, for the coefficient p-values

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace statsduck {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// At most 25 step halvings inside one outer iteration, matching R's inner
// loop in glm.fit.
constexpr int kMaxHalvings = 25;

GlmResult GlmFail(const std::string &msg) {
	GlmResult r;
	r.ok = false;
	r.error = msg;
	return r;
}

bool AllFinite(const std::vector<double> &v) {
	for (const double x : v) {
		if (!std::isfinite(x)) {
			return false;
		}
	}
	return true;
}

// R's glm.fit starting values. They are not an arbitrary choice: they are
// what makes the awkward cases converge. The binomial one in particular
// keeps the starting mu off 0 and off 1, which is where link_eta would
// return an infinity.
double StartMu(Family f, double y, double w) {
	switch (f) {
	case Family::kBinomial:
		return (w * y + 0.5) / (w + 1.0);
	case Family::kPoisson:
		return y + 0.1;
	case Family::kGaussian:
	case Family::kGamma:
		return y;
	}
	return y; // unreachable
}

// Whether one FITTED MEAN sits inside the family's support. This is R's
// validmu, and the loop tests it on every proposed step.
//
// It is not covered by link_inv's clamp, and the gap produced wrong
// answers before this existed. link_inv clamps a probability link into
// [kMuEps, 1 - kMuEps], but the identity link is never clamped and the
// log link is clamped at the floor only. Both of those are allowed for a
// family whose support they can leave: poisson with an identity link
// reaches a negative fitted mean, and binomial with a log link reaches a
// fitted probability above 1. Neither is caught by a non-finite
// deviance. The binomial and poisson deviance skip the y*ln(y/mu) term
// at y = 0, so a negative mu at a zero-count row contributes a finite
// negative amount instead of NaN, the step is accepted, and the fit
// converges to a mean outside its own support. Everything computed from
// those fitted values is then wrong: the deviance, the dispersion, the
// log-likelihood, the AIC and the standard errors.
//
// Gamma is already protected by its own deviance, which takes ln(mu) and
// returns NaN for a negative mean. The test below still covers it, so
// the rule is one rule for all four families rather than a different one
// per family.
bool MuInSupport(Family f, double mu) {
	if (!std::isfinite(mu)) {
		return false;
	}
	switch (f) {
	case Family::kGaussian:
		return true; // a gaussian mean can be any real number
	case Family::kBinomial:
		return mu > 0.0 && mu < 1.0;
	case Family::kPoisson:
	case Family::kGamma:
		return mu > 0.0;
	}
	return false; // unreachable
}

// The same test over a whole fitted-value vector, SKIPPING the rows whose
// prior weight is not positive.
//
// The weight filter is not a refinement, it is required. A zero-weight
// row influences no estimate, and this file's contract says its
// contribution is 0 either way. But evaluate() computes a fitted mean for
// every row it is given, so a weightless row's own covariates produce
// their own eta and mu. Without the filter, one weightless row carrying
// an extreme covariate fails a fit that is perfectly good over every row
// that counts. The finite-deviance test does not catch that either: the
// poisson deviance skips the y*ln(y/mu) term at y = 0, so a weightless
// zero-count row contributes 0 however extreme its own mu is. The filter
// mirrors the one in normal_matrix, for the same reason.
bool AllMuInSupport(Family f, const std::vector<double> &mu, const std::vector<double> &w) {
	for (std::size_t i = 0; i < mu.size(); i++) {
		if (!(w[i] > 0.0)) {
			continue;
		}
		if (!MuInSupport(f, mu[i])) {
			return false;
		}
	}
	return true;
}

// Whether one response value sits inside the family's domain. A
// non-finite y fails for every family: it would otherwise spread NaN
// through the whole result, and the poisson test below (y >= 0) is false
// for NaN but would not catch a +infinity.
bool ResponseInDomain(Family f, double y) {
	if (!std::isfinite(y)) {
		return false;
	}
	switch (f) {
	case Family::kGaussian:
		return true;
	case Family::kBinomial:
		return y >= 0.0 && y <= 1.0;
	case Family::kPoisson:
		return y >= 0.0;
	case Family::kGamma:
		return y > 0.0;
	}
	return false; // unreachable
}

// One fit. `with_null_deviance` is what stops the recursion: the
// intercept-only fit this function runs to get null_deviance is itself a
// fit, and it must not go on to fit a null model of its own.
GlmResult FitGlm(const std::vector<double> &y, const linalg::Mat &X_pred, const GlmOptions &opts,
                 const std::vector<double> *weights, const std::vector<double> *offset,
                 const std::vector<int> *cluster_ids, bool with_null_deviance) {
	const std::size_t n = y.size();
	const std::size_t p = X_pred.cols; // predictors (excludes the intercept)

	// ── Validation. Every failure here is data-shaped: ok = false with a
	// message, which the SQL layer reports as a NULL group.
	if (n == 0) {
		return GlmFail("glm_fit: no observations");
	}
	if (X_pred.rows != n) {
		return GlmFail("glm_fit: design matrix row count does not match the response length");
	}
	if (weights && weights->size() != n) {
		return GlmFail("glm_fit: a weight is required for every observation");
	}
	if (offset && offset->size() != n) {
		return GlmFail("glm_fit: an offset is required for every observation");
	}
	if (!link_allowed(opts.family, opts.link)) {
		return GlmFail(std::string("glm_fit: the ") + link_name(opts.link) +
		               " link is not allowed for the " + family_name(opts.family) + " family");
	}
	if (opts.max_iter < 1) {
		// Without this the loop would run zero times and the result would
		// carry no fit at all.
		return GlmFail("glm_fit: max_iter must be at least 1");
	}

	const std::vector<double> w = weights ? *weights : std::vector<double>(n, 1.0);
	std::size_t n_positive = 0; // rows of positive prior weight
	for (std::size_t i = 0; i < n; i++) {
		if (!ResponseInDomain(opts.family, y[i])) {
			return GlmFail(std::string("glm_fit: a response value lies outside the ") +
			               family_name(opts.family) + " family's domain");
		}
		if (!std::isfinite(w[i]) || w[i] < 0.0) {
			return GlmFail("glm_fit: weights must be finite and non-negative");
		}
		if (w[i] > 0.0) {
			n_positive++;
		}
		if (offset && !std::isfinite((*offset)[i])) {
			return GlmFail("glm_fit: offsets must be finite");
		}
	}
	if (n_positive == 0) {
		return GlmFail("glm_fit: every observation has weight 0, so nothing informs the fit");
	}

	// Cluster-robust setup, identical to fit_lm's: the ids are required,
	// must cover every row, must be dense 0-based labels, and must resolve
	// to at least two clusters.
	const bool clustered = (opts.vcov == Vcov::kCR0 || opts.vcov == Vcov::kCR1);
	std::size_t G = 0;
	if (clustered) {
		if (!cluster_ids || cluster_ids->size() != n) {
			return GlmFail("glm_fit: cluster-robust SE requires a cluster id for every row");
		}
		int max_id = -1;
		for (const int id : *cluster_ids) {
			if (id < 0) {
				return GlmFail("glm_fit: cluster ids must be non-negative dense labels");
			}
			if (id > max_id) {
				max_id = id;
			}
		}
		G = static_cast<std::size_t>(max_id) + 1;
		if (G < 2) {
			return GlmFail("glm_fit: cluster-robust SE needs at least 2 clusters");
		}
	}

	// ── The design. X_pred arrives without a constant column, so the
	// intercept is prepended here, exactly as fit_lm does. A zero-column
	// X_pred is legitimate: with an intercept it is the null model, and
	// without one there are no parameters at all.
	const std::size_t k = p + (opts.intercept ? 1 : 0);
	linalg::Mat X_all(n, k);
	for (std::size_t r = 0; r < n; r++) {
		std::size_t c = 0;
		if (opts.intercept) {
			X_all(r, 0) = 1.0;
			c = 1;
		}
		for (std::size_t j = 0; j < p; j++) {
			X_all(r, c + j) = X_pred(r, j);
		}
	}

	// The aliased set, computed ONCE from the rows of positive prior
	// weight and held fixed for the whole loop. The header says why that
	// is sound: for strictly positive weights rank(W^1/2 X) = rank(X),
	// because W^1/2 is invertible, so the set cannot change between
	// iterations. A zero-weight row is excluded because it contributes
	// nothing to the solve and can change the rank. The convention is R's
	// dqrdc2: the earlier column of a dependent set is kept, so the later
	// one is the aliased term.
	const bool all_positive = (n_positive == n);
	linalg::Mat X_positive; // only materialized when some weight is 0
	if (!all_positive) {
		X_positive = linalg::Mat(n_positive, k);
		std::size_t out = 0;
		for (std::size_t r = 0; r < n; r++) {
			if (w[r] <= 0.0) {
				continue;
			}
			for (std::size_t j = 0; j < k; j++) {
				X_positive(out, j) = X_all(r, j);
			}
			out++;
		}
	}
	const linalg::ColumnSelection sel =
	    linalg::independent_columns(all_positive ? X_all : X_positive);
	const std::size_t rank = static_cast<std::size_t>(sel.rank);
	if (rank == 0 && k > 0) {
		return GlmFail("glm_fit: the design has no independent columns");
	}
	// The dispersion divides by n - rank, so a fit with no residual
	// degrees of freedom has to be refused before any division rather
	// than after it, where it would spread Inf or NaN across the result.
	if (n <= rank) {
		return GlmFail("glm_fit: need n > rank (rank " + std::to_string(rank) + " of " +
		               std::to_string(k) + " columns), got n = " + std::to_string(n) +
		               ". There are not enough complete-case rows");
	}

	// The loop runs on the independent columns only. The dropped columns
	// lie in the span of the kept ones, so the fitted values, and every
	// quantity computed from them, equal the reduced model's.
	const bool deficient = rank < k;
	linalg::Mat X_kept;
	if (deficient) {
		X_kept = linalg::Mat(n, rank);
		for (std::size_t r = 0; r < n; r++) {
			for (std::size_t j = 0; j < rank; j++) {
				X_kept(r, j) = X_all(r, static_cast<std::size_t>(sel.keep[j]));
			}
		}
	}
	const linalg::Mat &X = deficient ? X_kept : X_all;

	// eta = X beta + offset, mu = link_inv(eta), and the deviance there.
	// The accumulation starts from the offset (0 when there is none) and
	// adds the same terms in the same order fit_lm's own fitted-value loop
	// does, which is what lets a gaussian identity-link fit reproduce it
	// bit for bit.
	auto evaluate = [&](const std::vector<double> &b, std::vector<double> &eta_out,
	                    std::vector<double> &mu_out) {
		for (std::size_t r = 0; r < n; r++) {
			double e = offset ? (*offset)[r] : 0.0;
			for (std::size_t j = 0; j < rank; j++) {
				e += b[j] * X(r, j);
			}
			eta_out[r] = e;
			mu_out[r] = link_inv(opts.link, e);
		}
		return deviance(opts.family, y, mu_out, w);
	};

	// Xt W X over the kept columns, and Xt W z alongside it when `z` is
	// given. Only the upper triangle is accumulated and then mirrored, the
	// same way fit_lm builds its normal matrix.
	//
	// A row whose working weight is not strictly positive is skipped. For
	// a weight of exactly 0 that changes no sum, since the row's
	// contribution is 0 either way. The test is written as `not greater
	// than 0` rather than `equal to 0` so that it also screens a negative
	// or NaN working weight. A negative one would make Xt W X indefinite,
	// the Cholesky would fail, and svd_solve would return the minimum-norm
	// solution of an indefinite system, which is not the IRLS step. The
	// support test on the fitted mean should make that unreachable: it
	// keeps variance(mu) positive for every family, and the prior weights
	// are already validated non-negative. This is the second line of
	// defence, not the first.
	auto normal_matrix = [&](const std::vector<double> &wk, const std::vector<double> *z,
	                         linalg::Mat &XtWX, std::vector<double> &XtWz) {
		XtWX = linalg::Mat(rank, rank);
		XtWz.assign(rank, 0.0);
		for (std::size_t r = 0; r < n; r++) {
			const double wr = wk[r];
			if (!(wr > 0.0)) {
				continue;
			}
			for (std::size_t i = 0; i < rank; i++) {
				const double xi = X(r, i);
				if (z) {
					XtWz[i] += wr * xi * (*z)[r];
				}
				for (std::size_t j = i; j < rank; j++) {
					XtWX(i, j) += wr * xi * X(r, j);
				}
			}
		}
		for (std::size_t i = 0; i < rank; i++) {
			for (std::size_t j = 0; j < i; j++) {
				XtWX(i, j) = XtWX(j, i);
			}
		}
	};

	// ── The loop. mu starts at R's glm.fit values and eta at link_eta(mu),
	// so the first working response and the first working weights come
	// from a mean vector rather than from a coefficient vector.
	std::vector<double> beta(rank, 0.0);
	std::vector<double> eta(n, 0.0), mu(n, 0.0);
	for (std::size_t i = 0; i < n; i++) {
		mu[i] = StartMu(opts.family, y[i], w[i]);
		eta[i] = link_eta(opts.link, mu[i]);
	}
	double dev = deviance(opts.family, y, mu, w);
	int iterations = 0;
	bool converged = false;

	if (rank == 0) {
		// No parameters, so there is nothing to iterate: eta is the offset
		// alone. This is the k = 0 null model of a fit without an
		// intercept.
		dev = evaluate(beta, eta, mu);
		converged = true;
	} else {
		std::vector<double> z(n, 0.0), work_w(n, 0.0);
		std::vector<double> eta_new(n, 0.0), mu_new(n, 0.0);
		linalg::Mat XtWX;
		std::vector<double> XtWz;
		for (int iter = 1; iter <= opts.max_iter; iter++) {
			iterations = iter;

			// The working response and the working weights at the current
			// (eta, mu).
			for (std::size_t r = 0; r < n; r++) {
				const double d = link_deriv(opts.link, eta[r]);
				const double off = offset ? (*offset)[r] : 0.0;
				z[r] = eta[r] - off + (y[r] - mu[r]) / d;
				work_w[r] = w[r] * d * d / variance(opts.family, mu[r]);
			}

			normal_matrix(work_w, &z, XtWX, XtWz);
			linalg::Solution sol = linalg::cholesky_solve(XtWX, XtWz);
			if (!sol.ok) {
				// Collinearity was already removed by the selection above,
				// so a Cholesky failure here is numerical. Fall back to the
				// SVD, as fit_lm does.
				sol = linalg::svd_solve(XtWX, XtWz);
			}
			if (!sol.ok || sol.x.size() != rank) {
				if (iter == 1) {
					return GlmFail("glm_fit: the first weighted solve failed (the design is "
					               "numerically singular at the starting values)");
				}
				break; // keep the iterate already accepted
			}

			std::vector<double> beta_new = sol.x;
			double dev_new = AllFinite(beta_new) ? evaluate(beta_new, eta_new, mu_new) : kNaN;

			// A step is bad on any of four counts: its coefficients are
			// not finite, its deviance is not finite, a fitted mean left
			// the family's support, or the deviance increased. The
			// support test is the one that cannot be dropped: see
			// MuInSupport above for the wrong answers it prevents.
			// `mu_new` is only current when beta_new is finite, which is
			// why the two finiteness tests come first: || short-circuits,
			// so the support test never reads a stale mu_new.
			auto step_is_bad = [&]() {
				return !AllFinite(beta_new) || !std::isfinite(dev_new) ||
				       !AllMuInSupport(opts.family, mu_new, w) || dev_new > dev;
			};

			// Halve the step while it is bad. The first pass has no
			// coefficient vector to halve toward, so it takes its step as
			// it stands; see the header.
			int halvings = 0;
			if (iter > 1) {
				while (halvings < kMaxHalvings && step_is_bad()) {
					for (std::size_t j = 0; j < rank; j++) {
						beta_new[j] = 0.5 * (beta_new[j] + beta[j]);
					}
					dev_new = AllFinite(beta_new) ? evaluate(beta_new, eta_new, mu_new) : kNaN;
					halvings++;
				}
			}

			if (!AllFinite(beta_new) || !std::isfinite(dev_new) ||
			    !AllMuInSupport(opts.family, mu_new, w)) {
				if (iter == 1) {
					return GlmFail("glm_fit: the first step produced a non-finite deviance or a "
					               "fitted mean outside the family's support (the starting values "
					               "or the offset saturate the link)");
				}
				break; // keep the iterate already accepted
			}

			const double rel = std::fabs(dev_new - dev) / (std::fabs(dev) + 0.1);
			if (dev_new > dev && halvings == kMaxHalvings) {
				// The step could not be made to lower the deviance. At a
				// converged fit that is a rounding-level increase, so the
				// criterion is tested before giving up. Either way the
				// already-accepted iterate is kept, since it has the lower
				// deviance of the two.
				if (rel < opts.tol) {
					converged = true;
				}
				break;
			}

			beta = beta_new;
			eta = eta_new;
			mu = mu_new;
			dev = dev_new;
			if (rel < opts.tol) {
				converged = true;
				break;
			}
		}
	}

	// ── The result. The order below is the contract: the fitted values are
	// already in `mu`, then deviance, then pearson_dispersion, then
	// glm_loglik, then glm_aic. glm_loglik reads the dispersion, so it can
	// only run after pearson_dispersion.
	GlmResult res;
	res.ok = true;
	res.n = n;
	res.k = k;
	res.rank = rank;
	res.n_clusters = clustered ? G : 0;
	res.df_residual = n - rank;
	res.iterations = iterations;
	res.converged = converged;
	res.deviance = dev;
	res.dispersion = pearson_dispersion(opts.family, y, mu, w, res.df_residual);
	res.loglik = glm_loglik(opts.family, y, mu, w, res.dispersion);
	res.aic = glm_aic(opts.family, res.loglik, rank);

	res.terms.reserve(k);
	if (opts.intercept) {
		res.terms.emplace_back("(Intercept)");
	}
	for (std::size_t j = 0; j < p; j++) {
		res.terms.emplace_back("x" + std::to_string(j + 1));
	}

	// The coefficient covariance, evaluated at the reported estimate: the
	// working weights are recomputed from the final eta and mu. The
	// classical estimator is dispersion * (Xt W X) inverse.
	res.beta.assign(k, kNaN);
	res.std_error.assign(k, kNaN);
	res.z_statistic.assign(k, kNaN);
	res.p_value.assign(k, kNaN);
	res.cov.assign(k * k, kNaN);
	if (rank > 0) {
		std::vector<double> final_w(n, 0.0);
		for (std::size_t r = 0; r < n; r++) {
			const double d = link_deriv(opts.link, eta[r]);
			final_w[r] = w[r] * d * d / variance(opts.family, mu[r]);
		}
		linalg::Mat XtWX;
		std::vector<double> unused;
		normal_matrix(final_w, nullptr, XtWX, unused);
		bool inv_ok = false;
		const linalg::Mat XtWXinv = linalg::inv_spd(XtWX, &inv_ok);
		if (!inv_ok) {
			return GlmFail("glm_fit: failed to invert XᵀWX (singular reduced design at the "
			               "fitted values)");
		}

		// Scatter the reduced fit back into k-space: a kept column carries
		// its estimates, an aliased one stays NaN, which the SQL layer
		// renders NULL. For a full-rank design sel.keep[a] == a.
		for (std::size_t a = 0; a < rank; a++) {
			const std::size_t ja = static_cast<std::size_t>(sel.keep[a]);
			res.beta[ja] = beta[a];
			const double var = res.dispersion * XtWXinv(a, a);
			const double se = var > 0.0 ? std::sqrt(var) : 0.0;
			res.std_error[ja] = se;
			if (se > 0.0) {
				const double zj = beta[a] / se;
				res.z_statistic[ja] = zj;
				// A normal reference for all four families. See GlmResult.
				res.p_value[ja] = 2.0 * (1.0 - stats_duck::NormalCDF(std::fabs(zj)));
			}
			for (std::size_t b = 0; b < rank; b++) {
				res.cov[ja * k + static_cast<std::size_t>(sel.keep[b])] =
				    res.dispersion * XtWXinv(a, b);
			}
		}
	}

	// null_deviance comes from actually fitting the intercept-only model,
	// with the same family, link, weights and offset. A closed form would
	// not catch a wrong null model. The covariance estimator is forced to
	// the classical one because only the deviance is read from that fit,
	// and a cluster-robust estimator would demand cluster ids it does not
	// need.
	if (with_null_deviance) {
		GlmOptions null_opts = opts;
		null_opts.vcov = Vcov::kConst;
		const GlmResult null_fit =
		    FitGlm(y, linalg::Mat(n, 0), null_opts, weights, offset, nullptr, false);
		res.null_deviance = null_fit.ok ? null_fit.deviance : kNaN;
	}

	return res;
}

} // namespace

GlmResult fit_glm(const std::vector<double> &y, const linalg::Mat &X_pred, const GlmOptions &opts,
                  const std::vector<double> *weights, const std::vector<double> *offset,
                  const std::vector<int> *cluster_ids) {
	return FitGlm(y, X_pred, opts, weights, offset, cluster_ids, true);
}

} // namespace statsduck
