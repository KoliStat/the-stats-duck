#pragma once

// stats_duck derivative-free optimization kernel.
//
// Like linalg.hpp this boundary is DELIBERATELY DuckDB-agnostic — and, unlike
// linalg, Eigen-free too, so it is header-only: no extra TU, no build wiring.
// Shared downstream (the-stats-duck-bio) via the same submodule include path.
//
// Scope: classic Nelder-Mead simplex minimization. Chosen for the REML /
// profile-likelihood surfaces of the mixed-model roadmap (E4): few parameters
// (1-5), no cheap gradients, positivity handled by log-reparameterization in
// the caller. Deterministic by construction — no RNG, no restarts.

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace statsduck {
namespace optimize {

struct Options {
	double xtol = 1e-8;      // max coordinate spread of simplex vs best vertex
	double ftol = 1e-10;     // max |f_i - f_best| spread across simplex
	int max_eval = 2000;     // hard budget on objective evaluations
	double init_step = 0.05; // relative bump building the initial simplex
};

// Failure-as-value (no exceptions), same convention as linalg::Solution.ok:
// converged=false on degenerate input (empty x0, max_eval<=0) or budget
// exhaustion; x then still holds the best point seen (or echoes x0).
struct Result {
	std::vector<double> x;
	double fx = std::numeric_limits<double>::quiet_NaN();
	int n_eval = 0;
	int n_iter = 0;
	bool converged = false;
};

// Minimize f over R^n from start x0. Classic Nelder-Mead coefficients:
// reflect 1, expand 2, contract 1/2, shrink 1/2. The initial simplex bumps
// each coordinate by init_step*|x0_i| (fixed 0.00025 when x0_i == 0 — the
// fminsearch convention). f returning NaN/±Inf is treated as +Inf so invalid
// regions (e.g. off-domain likelihood evaluations) repel the simplex instead
// of poisoning the vertex ordering.
inline Result nelder_mead(const std::function<double(const std::vector<double> &)> &f,
                          std::vector<double> x0, const Options &opts = {}) {
	Result res;
	res.x = std::move(x0);
	const std::size_t n = res.x.size();
	if (n == 0 || opts.max_eval <= 0) {
		return res; // converged=false, fx=NaN, n_eval=0
	}
	const double kInf = std::numeric_limits<double>::infinity();
	int n_eval = 0;
	auto eval = [&](const std::vector<double> &p) {
		++n_eval;
		const double v = f(p);
		return std::isfinite(v) ? v : kInf;
	};

	// Initial simplex: x0 plus one per-coordinate bump.
	std::vector<std::vector<double>> v(n + 1, res.x);
	std::vector<double> fv(n + 1);
	for (std::size_t i = 0; i < n; ++i) {
		const double xi = v[i + 1][i];
		v[i + 1][i] = (xi != 0.0) ? xi + opts.init_step * std::fabs(xi) : 0.00025;
	}
	for (std::size_t i = 0; i <= n; ++i) {
		fv[i] = eval(v[i]);
	}

	// Stable insertion sort keeps vertex ordering deterministic under ties.
	auto order = [&] {
		for (std::size_t i = 1; i <= n; ++i) {
			double fi = fv[i];
			std::vector<double> vi = v[i];
			std::size_t j = i;
			while (j > 0 && fv[j - 1] > fi) {
				fv[j] = fv[j - 1];
				v[j] = v[j - 1];
				--j;
			}
			fv[j] = fi;
			v[j] = std::move(vi);
		}
	};
	order();

	bool converged = false;
	int n_iter = 0;
	while (n_eval < opts.max_eval) {
		// Converged when the simplex is flat in f AND small in x. Guard on a
		// finite best value: an all-infinite simplex yields NaN spreads whose
		// comparisons are all false — without the guard, max() folding NaN to 0
		// would fake convergence.
		if (std::isfinite(fv[0])) {
			double fspread = 0.0, xspread = 0.0;
			for (std::size_t i = 1; i <= n; ++i) {
				fspread = std::max(fspread, std::fabs(fv[i] - fv[0]));
				for (std::size_t j = 0; j < n; ++j) {
					xspread = std::max(xspread, std::fabs(v[i][j] - v[0][j]));
				}
			}
			if (fspread <= opts.ftol && xspread <= opts.xtol) {
				converged = true;
				break;
			}
		}
		++n_iter;

		// Centroid of all vertices but the worst.
		std::vector<double> c(n, 0.0);
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < n; ++j) {
				c[j] += v[i][j] / static_cast<double>(n);
			}
		}
		// Point at parameter t along the worst->centroid axis: c + t*(worst - c).
		auto point = [&](double t) {
			std::vector<double> p(n);
			for (std::size_t j = 0; j < n; ++j) {
				p[j] = c[j] + t * (v[n][j] - c[j]);
			}
			return p;
		};

		auto xr = point(-1.0);
		const double fr = eval(xr); // reflection
		if (fr < fv[0]) {
			auto xe = point(-2.0);
			const double fe = eval(xe); // expansion
			if (fe < fr) {
				v[n] = std::move(xe);
				fv[n] = fe;
			} else {
				v[n] = std::move(xr);
				fv[n] = fr;
			}
		} else if (fr < fv[n - 1]) {
			v[n] = std::move(xr); // plain reflection accepted
			fv[n] = fr;
		} else {
			bool shrink = false;
			if (fr < fv[n]) { // outside contraction
				auto xc = point(-0.5);
				const double fc = eval(xc);
				if (fc <= fr) {
					v[n] = std::move(xc);
					fv[n] = fc;
				} else {
					shrink = true;
				}
			} else { // inside contraction
				auto xc = point(0.5);
				const double fc = eval(xc);
				if (fc < fv[n]) {
					v[n] = std::move(xc);
					fv[n] = fc;
				} else {
					shrink = true;
				}
			}
			if (shrink) { // shrink all vertices toward the best
				for (std::size_t i = 1; i <= n; ++i) {
					for (std::size_t j = 0; j < n; ++j) {
						v[i][j] = v[0][j] + 0.5 * (v[i][j] - v[0][j]);
					}
					fv[i] = eval(v[i]);
				}
			}
		}
		order();
	}

	res.x = v[0];
	res.fx = fv[0];
	res.n_eval = n_eval;
	res.n_iter = n_iter;
	res.converged = converged;
	return res;
}

} // namespace optimize
} // namespace statsduck
