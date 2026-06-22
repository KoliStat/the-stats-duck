#pragma once

// stats_duck dense linear-algebra kernel.
//
// This API boundary is DELIBERATELY DuckDB-agnostic and Eigen-free — it speaks
// only `std::vector<double>` and the row-major `Mat` below. That is what lets
// this header be shared, via git submodule, with downstream extensions (e.g.
// the-stats-duck-bio) without pulling in DuckDB or Eigen. The implementation
// (linalg.cpp) uses Eigen; that dependency stays confined to that one
// translation unit.
//
// Scope (Epic 0.1): Householder QR, Cholesky, SVD; tolerance-based rank;
// Moore–Penrose pseudo-inverse; symmetric/SPD solve + inverse; the covariance
// sandwich L·A·Lᵀ. Pure numerics — no statistics, no model assembly (that lives
// in lm_fit / glm_fit / …).

#include <cstddef>
#include <vector>

namespace statsduck {
namespace linalg {

// Row-major dense matrix.
struct Mat {
	std::size_t rows = 0;
	std::size_t cols = 0;
	std::vector<double> data; // size == rows*cols, row-major

	Mat() = default;
	Mat(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

	double &operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
	double operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
};

// Solution of a linear / least-squares system, with the numerical rank used
// (so callers can report and act on rank deficiency).
struct Solution {
	std::vector<double> x; // length == A.cols
	int rank = 0;
	bool ok = false; // false on dimension mismatch, or non-SPD where required
};

// Solve a symmetric positive-definite system A x = b via Cholesky. ok = false if
// A is not SPD — the caller should fall back to svd_solve.
Solution cholesky_solve(const Mat &A, const std::vector<double> &b);

// Least-squares solve of A x = b via Householder QR. Assumes full column rank;
// for possibly rank-deficient A use svd_solve.
Solution qr_solve(const Mat &A, const std::vector<double> &b);

// Rank-revealing least-squares (minimum-norm solution) via SVD. `tol` is
// relative to the largest singular value; Solution.rank is the numerical rank.
Solution svd_solve(const Mat &A, const std::vector<double> &b, double tol = 1e-12);

// Numerical rank via SVD: count of singular values > tol * sigma_max.
int rank(const Mat &A, double tol = 1e-12);

// Moore–Penrose pseudo-inverse. A is m×n; value is n×m. Also returns the rank.
struct Pinv {
	Mat value; // n×m
	int rank = 0;
};
Pinv pinv(const Mat &A, double tol = 1e-12);

// Inverse of a symmetric positive-definite matrix — e.g. (XᵀWX)⁻¹ for
// model-based covariance. If `ok` is non-null it is set false when A is not SPD.
Mat inv_spd(const Mat &A, bool *ok = nullptr);

// Covariance sandwich L · A · Lᵀ  (L: k×n, A: n×n → k×k): cov(θ) for θ = L·β
// given cov(β) = A. Used by lin_hyp / emmeans.
Mat sandwich(const Mat &L, const Mat &A);

// Matrix–vector product A·x (A: m×n, x length n → length m).
std::vector<double> matvec(const Mat &A, const std::vector<double> &x);

} // namespace linalg
} // namespace statsduck
