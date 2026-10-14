#include "lin_hyp.hpp"

#include "distributions.hpp" // stats_duck::StudentTCDF / FCDF / ChiSquareCDF

#include <cmath>
#include <limits>
#include <string>

namespace statsduck {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

LinHypResult Fail(const std::string &msg) {
	LinHypResult r;
	r.ok = false;
	r.error = msg;
	return r;
}

} // namespace

LinHypResult lin_hyp(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V,
                     double sigma2, double df) {
	const std::size_t q = L.rows;
	const std::size_t k = beta.size();

	if (q == 0) {
		return Fail("lin_hyp: the hypothesis matrix has no rows");
	}
	if (k == 0) {
		return Fail("lin_hyp: beta is empty");
	}
	if (L.cols != k) {
		return Fail("lin_hyp: the hypothesis matrix has " + std::to_string(L.cols) +
		            " columns but beta has " + std::to_string(k) + " elements");
	}
	if (V.rows != k || V.cols != k) {
		return Fail("lin_hyp: the covariance must be " + std::to_string(k) + "×" +
		            std::to_string(k) + ", got " + std::to_string(V.rows) + "×" +
		            std::to_string(V.cols));
	}
	if (!(df > 0.0)) {
		return Fail("lin_hyp: df must be positive");
	}
	if (!std::isfinite(sigma2) || sigma2 < 0.0) {
		return Fail("lin_hyp: sigma2 must be finite and not negative");
	}

	LinHypResult r;
	r.ok = true;
	r.df_den = df;

	// A rank-deficient fit carries NaN in the row and column of each aliased
	// coefficient. A hypothesis that puts no weight on one is still estimable,
	// but IEEE makes 0 · NaN = NaN, so the aliased entries are zeroed in a
	// working copy instead of being multiplied by zero. They carry zero weight
	// by construction, so zeroing changes no surviving result.
	std::vector<bool> aliased(k, false);
	for (std::size_t j = 0; j < k; j++) {
		aliased[j] = !std::isfinite(V(j, j));
	}
	for (std::size_t i = 0; i < q; i++) {
		for (std::size_t j = 0; j < k; j++) {
			if (aliased[j] && L(i, j) != 0.0) {
				return r; // estimable stays false — SAS's "non-est"
			}
		}
	}
	std::vector<double> beta_w = beta;
	linalg::Mat V_w = V;
	for (std::size_t a = 0; a < k; a++) {
		if (!aliased[a]) {
			continue;
		}
		beta_w[a] = 0.0;
		for (std::size_t b = 0; b < k; b++) {
			V_w(a, b) = 0.0;
			V_w(b, a) = 0.0;
		}
	}

	// θ = L·β and C = L·V·Lᵀ: the estimated combinations and their covariance.
	const std::vector<double> theta = linalg::matvec(L, beta_w);
	const linalg::Mat C = linalg::sandwich(L, V_w);
	if (theta.size() != q || C.rows != q || C.cols != q) {
		return Fail("lin_hyp: failed to form L·β or L·V·Lᵀ");
	}
	// Anything non-finite left here came from somewhere other than an aliased
	// column, so the hypothesis still cannot be evaluated.
	for (const double v : theta) {
		if (!std::isfinite(v)) {
			return r;
		}
	}
	for (const double v : C.data) {
		if (!std::isfinite(v)) {
			return r;
		}
	}

	// pinv returns the pseudo-inverse together with the numerical rank, so one
	// call covers both the full-rank and the redundant case and df_num is the
	// rank rather than the row count. statsmodels' wald_test agrees: given a
	// redundant hypothesis it reports the rank too.
	const linalg::Pinv pi = linalg::pinv(C);
	const std::size_t rank = pi.rank > 0 ? static_cast<std::size_t>(pi.rank) : 0;
	if (rank == 0) {
		return r; // a hypothesis with no information, such as a zero L
	}
	r.estimable = true;
	r.df_num = rank;

	// The Wald statistic W = θᵀ C⁻ θ.
	const std::vector<double> c_inv_theta = linalg::matvec(pi.value, theta);
	double w = 0.0;
	for (std::size_t i = 0; i < q; i++) {
		w += theta[i] * c_inv_theta[i];
	}
	const double rank_d = static_cast<double>(rank);
	r.chisq = w;
	r.chisq_p_value = 1.0 - stats_duck::ChiSquareCDF(w, rank_d);
	r.f_statistic = w / rank_d;
	r.f_p_value = 1.0 - stats_duck::FCDF(r.f_statistic, rank_d, df);
	// Only a sum of squares when V is the classical σ²(XᵀX)⁻¹; see the README.
	r.ss = w * sigma2;

	r.estimate = theta;
	r.std_error.assign(q, 0.0);
	r.t_statistic.assign(q, kNaN);
	r.p_value.assign(q, kNaN);
	for (std::size_t i = 0; i < q; i++) {
		const double var = C(i, i);
		const double se = var > 0.0 ? std::sqrt(var) : 0.0;
		r.std_error[i] = se;
		if (se > 0.0) {
			const double t = theta[i] / se;
			r.t_statistic[i] = t;
			r.p_value[i] = 2.0 * (1.0 - stats_duck::StudentTCDF(std::fabs(t), df));
		}
	}
	return r;
}

} // namespace statsduck
