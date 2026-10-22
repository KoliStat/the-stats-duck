#pragma once

// stats_duck aliased-coefficient convention — the rule #36 introduced for
// lin_hyp, shared here with emmeans (#21).
//
// A rank-deficient fit drops a dependent column and reports its coefficient,
// and the matching row and column of its covariance, as NaN (lm_core.hpp).
// A caller that never puts weight on that coefficient can still get a real
// answer. But the NaN cannot simply be multiplied away: IEEE makes
// 0 * NaN = NaN, not 0. So the aliased entries are zeroed in a working copy
// first, and any row or combination that DOES put weight on one is reported
// not estimable instead of evaluated against a poisoned NaN.
//
// DuckDB-free and Eigen-free, like linalg.hpp: only std::vector<double> and
// the row-major linalg::Mat, so a DuckDB-free kernel (emmeans) can include
// it directly.

#include "linalg.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace statsduck {

// One flag per column of V (expected k x k, a coefficient covariance):
// column j is aliased when V's own diagonal entry V(j, j) is not finite.
inline std::vector<bool> aliased_columns(const linalg::Mat &V) {
	std::vector<bool> aliased(V.cols, false);
	for (std::size_t j = 0; j < V.cols; j++) {
		aliased[j] = !std::isfinite(V(j, j));
	}
	return aliased;
}

// True when `row` (length k) puts any non-zero weight on an aliased column.
// Uses != 0.0 rather than a tolerance: a weight is either present or not,
// and the caller already chose it. Precondition: aliased has at least k
// entries — aliased_columns always returns exactly V.cols of them, and V is
// the k x k covariance the row's weights are tested against.
inline bool row_touches_aliased(const double *row, std::size_t k, const std::vector<bool> &aliased) {
	for (std::size_t j = 0; j < k; j++) {
		if (aliased[j] && row[j] != 0.0) {
			return true;
		}
	}
	return false;
}

// A copy of V with every aliased row and column set to 0. Safe to use in a
// quadratic form afterwards: the only coefficients that reach this point are
// ones row_touches_aliased already confirmed carry zero weight, so zeroing
// changes no surviving result.
inline linalg::Mat zero_aliased(const linalg::Mat &V, const std::vector<bool> &aliased) {
	linalg::Mat Vz = V;
	const std::size_t k = V.cols;
	for (std::size_t a = 0; a < k && a < aliased.size(); a++) {
		if (!aliased[a]) {
			continue;
		}
		for (std::size_t b = 0; b < k; b++) {
			Vz(a, b) = 0.0;
			Vz(b, a) = 0.0;
		}
	}
	return Vz;
}

// A copy of beta with every aliased entry set to 0, for the same reason.
inline std::vector<double> zero_aliased(const std::vector<double> &beta, const std::vector<bool> &aliased) {
	std::vector<double> bz = beta;
	for (std::size_t a = 0; a < bz.size() && a < aliased.size(); a++) {
		if (aliased[a]) {
			bz[a] = 0.0;
		}
	}
	return bz;
}

} // namespace statsduck
