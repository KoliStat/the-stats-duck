#include "lm_core.hpp"

#include "distributions.hpp" // stats_duck::StudentTCDF / FCDF — header-only, DuckDB-free

#include <algorithm>
#include <cmath>
#include <limits>

namespace statsduck {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Lower-case a copy (ASCII) — for case-insensitive vcov parsing.
std::string ToLower(const std::string &s) {
	std::string out = s;
	std::transform(out.begin(), out.end(), out.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return out;
}

LmResult Fail(const std::string &msg) {
	LmResult r;
	r.ok = false;
	r.error = msg;
	return r;
}

} // namespace

bool parse_vcov(const std::string &name, Vcov &out) {
	const std::string n = ToLower(name);
	if (n == "const" || n == "none" || n == "nonrobust" || n == "ols" || n == "iid") {
		out = Vcov::kConst;
	} else if (n == "hc0") {
		out = Vcov::kHC0;
	} else if (n == "hc1") {
		out = Vcov::kHC1;
	} else if (n == "hc2") {
		out = Vcov::kHC2;
	} else if (n == "hc3") {
		out = Vcov::kHC3;
	} else if (n == "cr0") {
		out = Vcov::kCR0;
	} else if (n == "cr1" || n == "cluster") {
		out = Vcov::kCR1;
	} else {
		return false;
	}
	return true;
}

const char *vcov_name(Vcov v) {
	switch (v) {
	case Vcov::kConst:
		return "const";
	case Vcov::kHC0:
		return "HC0";
	case Vcov::kHC1:
		return "HC1";
	case Vcov::kHC2:
		return "HC2";
	case Vcov::kHC3:
		return "HC3";
	case Vcov::kCR0:
		return "CR0";
	case Vcov::kCR1:
		return "CR1";
	}
	return "const";
}

LmResult fit_lm(const std::vector<double> &y, const linalg::Mat &X_pred, const LmOptions &opts,
                const std::vector<int> *cluster_ids) {
	const std::size_t n = y.size();
	const std::size_t p = X_pred.cols; // predictors (excludes intercept)

	if (X_pred.rows != n) {
		return Fail("lm_fit: design matrix row count does not match the response length");
	}
	const std::size_t k = p + (opts.intercept ? 1 : 0);
	if (k == 0) {
		return Fail("lm_fit: model has no terms (need at least one predictor or an intercept)");
	}
	if (n <= k) {
		return Fail("lm_fit: need n > k (" + std::to_string(k) + " params), got n = " +
		            std::to_string(n) + " — not enough complete-case rows");
	}

	// Cluster-robust setup: the ids are required, must cover every row, be dense
	// 0-based labels, and resolve to at least two clusters (G−1 ≥ 1 df).
	const bool clustered = (opts.vcov == Vcov::kCR0 || opts.vcov == Vcov::kCR1);
	std::size_t G = 0;
	if (clustered) {
		if (!cluster_ids || cluster_ids->size() != n) {
			return Fail("lm_fit: cluster-robust SE requires a cluster id for every row");
		}
		int max_id = -1;
		for (const int id : *cluster_ids) {
			if (id < 0) {
				return Fail("lm_fit: cluster ids must be non-negative dense labels");
			}
			if (id > max_id) {
				max_id = id;
			}
		}
		G = static_cast<std::size_t>(max_id) + 1;
		if (G < 2) {
			return Fail("lm_fit: cluster-robust SE needs at least 2 clusters");
		}
	}

	// Materialize the full design matrix X (n × k), intercept first when present.
	linalg::Mat X(n, k);
	for (std::size_t r = 0; r < n; r++) {
		std::size_t c = 0;
		if (opts.intercept) {
			X(r, 0) = 1.0;
			c = 1;
		}
		for (std::size_t j = 0; j < p; j++) {
			X(r, c + j) = X_pred(r, j);
		}
	}

	// Normal equations: XᵀX (k×k, SPD when full rank) and Xᵀy (k).
	linalg::Mat XtX(k, k);
	std::vector<double> Xty(k, 0.0);
	for (std::size_t r = 0; r < n; r++) {
		for (std::size_t i = 0; i < k; i++) {
			const double xi = X(r, i);
			Xty[i] += xi * y[r];
			for (std::size_t j = i; j < k; j++) {
				XtX(i, j) += xi * X(r, j);
			}
		}
	}
	for (std::size_t i = 0; i < k; i++) { // symmetrize (only upper triangle filled)
		for (std::size_t j = 0; j < i; j++) {
			XtX(i, j) = XtX(j, i);
		}
	}

	// β via Cholesky on the SPD normal matrix. Not-SPD ⇒ collinear / singular.
	auto sol = linalg::cholesky_solve(XtX, Xty);
	if (!sol.ok) {
		return Fail("lm_fit: design matrix is singular (XᵀX not positive-definite); "
		            "check for perfectly collinear predictors or a constant column");
	}
	const std::vector<double> &beta = sol.x;

	// (XᵀX)⁻¹ — the "bread" of the sandwich and the classical covariance core.
	bool inv_ok = false;
	linalg::Mat XtXinv = linalg::inv_spd(XtX, &inv_ok);
	if (!inv_ok) {
		return Fail("lm_fit: failed to invert XᵀX (singular design)");
	}

	// Residuals, RSS, and the sums reused by the model summary.
	std::vector<double> resid(n, 0.0);
	double rss = 0.0, y_sum = 0.0, y_sq_sum = 0.0;
	for (std::size_t r = 0; r < n; r++) {
		double yhat = 0.0;
		for (std::size_t j = 0; j < k; j++) {
			yhat += beta[j] * X(r, j);
		}
		const double e = y[r] - yhat;
		resid[r] = e;
		rss += e * e;
		y_sum += y[r];
		y_sq_sum += y[r] * y[r];
	}

	const double df_resid = static_cast<double>(n - k);
	const double sigma2 = rss / df_resid;

	// ── Coefficient covariance V (k×k) per the chosen estimator ──────────────
	linalg::Mat V(k, k);
	if (opts.vcov == Vcov::kConst) {
		for (std::size_t i = 0; i < k; i++) {
			for (std::size_t j = 0; j < k; j++) {
				V(i, j) = sigma2 * XtXinv(i, j);
			}
		}
	} else if (clustered) {
		// Cluster-robust (Liang-Zeger) sandwich: V = (XᵀX)⁻¹ · M · (XᵀX)⁻¹ with
		// the "meat" M = Σ_g s_g s_gᵀ, s_g = Σ_{i∈g} xᵢêᵢ the cluster-g score sum.
		// One pass fills the per-cluster score sums; CR1 then applies the
		// Stata/statsmodels finite-sample factor [G/(G−1)]·[(N−1)/(N−k)].
		std::vector<double> score(G * k, 0.0); // G×k row-major: per-cluster s_g
		for (std::size_t r = 0; r < n; r++) {
			double *sg = &score[static_cast<std::size_t>((*cluster_ids)[r]) * k];
			const double e = resid[r];
			for (std::size_t j = 0; j < k; j++) {
				sg[j] += e * X(r, j);
			}
		}
		linalg::Mat meat(k, k);
		for (std::size_t gi = 0; gi < G; gi++) {
			const double *sg = &score[gi * k];
			for (std::size_t i = 0; i < k; i++) {
				const double si = sg[i];
				for (std::size_t j = 0; j < k; j++) {
					meat(i, j) += si * sg[j];
				}
			}
		}
		V = linalg::sandwich(XtXinv, meat); // (XᵀX)⁻¹ · M · (XᵀX)⁻¹
		if (opts.vcov == Vcov::kCR1) {
			const double Gd = static_cast<double>(G);
			const double nd = static_cast<double>(n);
			const double kd = static_cast<double>(k);
			const double c = (Gd / (Gd - 1.0)) * ((nd - 1.0) / (nd - kd));
			for (auto &v : V.data) {
				v *= c;
			}
		}
	} else {
		// Heteroskedasticity-consistent sandwich: V = (XᵀX)⁻¹ · M · (XᵀX)⁻¹,
		// M = Σᵢ wᵢ xᵢxᵢᵀ with wᵢ the squared residual, leverage-adjusted for
		// HC2/HC3. Leverage hᵢᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ.
		linalg::Mat meat(k, k);
		std::vector<double> xi(k, 0.0);
		for (std::size_t r = 0; r < n; r++) {
			for (std::size_t j = 0; j < k; j++) {
				xi[j] = X(r, j);
			}
			double w = resid[r] * resid[r];
			if (opts.vcov == Vcov::kHC2 || opts.vcov == Vcov::kHC3) {
				const std::vector<double> hx = linalg::matvec(XtXinv, xi); // (XᵀX)⁻¹ xᵢ
				double h = 0.0;
				for (std::size_t j = 0; j < k; j++) {
					h += xi[j] * hx[j];
				}
				double denom = 1.0 - h;
				// Guard the degenerate h→1 (exact-fit / saturated row).
				if (denom < 1e-12) {
					denom = 1e-12;
				}
				w /= (opts.vcov == Vcov::kHC3) ? (denom * denom) : denom;
			}
			for (std::size_t i = 0; i < k; i++) {
				const double wi = w * xi[i];
				for (std::size_t j = 0; j < k; j++) {
					meat(i, j) += wi * xi[j];
				}
			}
		}
		V = linalg::sandwich(XtXinv, meat); // (XᵀX)⁻¹ · M · (XᵀX)⁻¹
		if (opts.vcov == Vcov::kHC1) {
			const double c = static_cast<double>(n) / df_resid;
			for (auto &v : V.data) {
				v *= c;
			}
		}
	}

	// ── Assemble the result ──────────────────────────────────────────────────
	LmResult res;
	res.ok = true;
	res.n = n;
	res.k = k;
	res.df_residual = n - k;
	res.n_clusters = clustered ? G : 0;
	res.has_intercept = opts.intercept;
	res.vcov = opts.vcov;
	res.sigma = std::sqrt(sigma2);

	res.terms.reserve(k);
	if (opts.intercept) {
		res.terms.emplace_back("(Intercept)");
	}
	for (std::size_t j = 0; j < p; j++) {
		res.terms.emplace_back("x" + std::to_string(j + 1));
	}

	res.beta = beta;
	res.std_error.assign(k, 0.0);
	res.t_statistic.assign(k, 0.0);
	res.p_value.assign(k, 0.0);
	// Cluster-robust inference uses a t(G−1) reference (G = #clusters); classical
	// and HC use t(n−k). df_residual itself is always reported as n−k.
	const double df_infer = clustered ? static_cast<double>(G - 1) : df_resid;
	for (std::size_t j = 0; j < k; j++) {
		const double var = V(j, j);
		const double se = var > 0.0 ? std::sqrt(var) : 0.0;
		res.std_error[j] = se;
		if (se > 0.0) {
			const double t = beta[j] / se;
			res.t_statistic[j] = t;
			res.p_value[j] = 2.0 * (1.0 - stats_duck::StudentTCDF(std::fabs(t), df_infer));
		} else {
			res.t_statistic[j] = kNaN;
			res.p_value[j] = kNaN;
		}
	}

	// ── Model-level statistics (classical decomposition) ─────────────────────
	// R² is centered when an intercept is present (TSS = Σ(y−ȳ)²); R reports the
	// uncentered version (TSS = Σy²) without an intercept, which we mirror.
	const double y_mean = y_sum / static_cast<double>(n);
	double tss_centered = 0.0;
	for (std::size_t r = 0; r < n; r++) {
		const double dy = y[r] - y_mean;
		tss_centered += dy * dy;
	}
	const double tss = opts.intercept ? tss_centered : y_sq_sum;
	const double df_model = static_cast<double>(p); // predictors excluding intercept

	if (tss > 0.0) {
		res.r_squared = 1.0 - rss / tss;
		const double n_d = static_cast<double>(n);
		const double r_n = opts.intercept ? n_d - 1.0 : n_d;
		res.adj_r_squared = 1.0 - (1.0 - res.r_squared) * r_n / df_resid;
	} else {
		res.r_squared = kNaN;
		res.adj_r_squared = kNaN;
	}

	if (df_model > 0.0 && df_resid > 0.0 && rss > 0.0 && tss > 0.0) {
		const double ess = tss - rss;
		res.f_statistic = (ess / df_model) / (rss / df_resid);
		res.f_p_value = 1.0 - stats_duck::FCDF(res.f_statistic, df_model, df_resid);
	} else {
		res.f_statistic = kNaN;
		res.f_p_value = kNaN;
	}

	return res;
}

} // namespace statsduck
