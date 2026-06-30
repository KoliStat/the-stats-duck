#pragma once

// stats_duck linear-model fit core (Epic 1.1).
//
// DuckDB-free, Eigen-free: speaks only std::vector<double> and the row-major
// linalg::Mat, exactly like the linalg kernel it is built on. That keeps it
// directly unit-testable (test/cpp/test_lm_fit.cpp) and shareable downstream
// (the-stats-duck-bio) the same way linalg.hpp is — and lets the DuckDB-facing
// lm_fit aggregate (and, later, the lm/lm_summary table functions) be thin
// wrappers over a single, validated implementation rather than each carrying
// their own Cholesky.
//
// Scope: ordinary least squares with classical, heteroskedasticity-consistent
// (HC0/HC1/HC2/HC3), and cluster-robust (CR0/CR1) covariance for the coefficient
// standard errors. The cluster-robust path takes a per-row cluster id and is the
// reason the row-oriented fit buffers the full design (per-cluster score sums
// don't reduce to streaming sufficient statistics).

#include "linalg.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace statsduck {

// Covariance estimator for the coefficient standard errors.
//   kConst — classical OLS:        σ̂² (XᵀX)⁻¹
//   kHC0   — White / Eicker-Huber: (XᵀX)⁻¹ (Σ êᵢ² xᵢxᵢᵀ) (XᵀX)⁻¹
//   kHC1   — HC0 × n/(n−k)         (Stata's `, robust` default)
//   kHC2   — leverage-adjusted:    weight êᵢ²/(1−hᵢᵢ)
//   kHC3   — jackknife approx:     weight êᵢ²/(1−hᵢᵢ)²  (small-sample default)
// hᵢᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ is the i-th hat-matrix diagonal (leverage).
//   kCR0   — cluster-robust (Liang-Zeger): (XᵀX)⁻¹ (Σ_g s_g s_gᵀ) (XᵀX)⁻¹,
//            s_g = Σ_{i∈g} xᵢêᵢ the cluster-g score sum.
//   kCR1   — CR0 × [G/(G−1)]·[(N−1)/(N−k)]  (Stata `vce(cluster)` / statsmodels
//            `cov_type='cluster'` default). G = number of clusters.
// CR0/CR1 require a per-row cluster id (see fit_lm) and use a t(G−1) reference
// distribution for the coefficient p-values (df_residual itself stays n−k).
enum class Vcov { kConst, kHC0, kHC1, kHC2, kHC3, kCR0, kCR1 };

struct LmOptions {
	Vcov vcov = Vcov::kConst;
	bool intercept = true; // prepend a column of 1s to the design matrix
};

struct LmResult {
	bool ok = false;
	std::string error;

	std::size_t n = 0;           // observations used
	std::size_t k = 0;           // parameters (predictors + intercept)
	std::size_t df_residual = 0; // n − k
	std::size_t n_clusters = 0;  // #clusters G (vcov CR* only); 0 when unclustered
	bool has_intercept = true;
	Vcov vcov = Vcov::kConst;

	std::vector<std::string> terms;      // length k: "(Intercept)", "x1", "x2", …
	std::vector<double> beta;            // length k
	std::vector<double> std_error;       // length k (per `vcov`)
	std::vector<double> t_statistic;     // length k: βⱼ / SEⱼ
	std::vector<double> p_value;         // length k: two-sided, t(df_residual)

	double r_squared = 0.0;
	double adj_r_squared = 0.0;
	double f_statistic = 0.0; // classical overall-significance F (not robustified)
	double f_p_value = 0.0;
	double sigma = 0.0;       // residual standard error √(RSS/df_residual)
};

// Fit y (length n) on the predictor matrix X_pred (n × p, row-major, WITHOUT an
// intercept column — opts.intercept controls whether a constant is prepended).
// Returns ok=false with a populated `error` on too-few rows (n ≤ k) or a
// singular / collinear design (XᵀX not positive-definite). All statistics are
// computed on the linalg kernel.
//
// cluster_ids: optional, length n, DENSE 0-based cluster labels (0..G−1).
// Required iff opts.vcov is kCR0/kCR1 (else ok=false); ignored otherwise. With
// fewer than two clusters the cluster-robust covariance is undefined (ok=false).
LmResult fit_lm(const std::vector<double> &y, const linalg::Mat &X_pred, const LmOptions &opts,
                const std::vector<int> *cluster_ids = nullptr);

// Parse a covariance-type string (case-insensitive). Accepts:
//   "const" | "none" | "nonrobust" | "ols" | "iid"  → kConst
//   "hc0" | "hc1" | "hc2" | "hc3"                    → kHC*
//   "cr0" | "cr1" | "cluster"                        → kCR* ("cluster" → kCR1)
// Returns false (leaving `out` untouched) on an unrecognized name.
bool parse_vcov(const std::string &name, Vcov &out);

// Canonical name for a Vcov ("const", "HC0", …) — echoed back in LmResult.
const char *vcov_name(Vcov v);

} // namespace statsduck
