#include "lin_hyp.hpp"

#include "distributions.hpp" // stats_duck::StudentTCDF / FCDF / ChiSquareCDF
#include "estimability.hpp"

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
	// The sign test alone lets +Inf through, and the t and F CDFs then compute
	// Inf/Inf = NaN: every p-value would come back NaN with ok still true.
	// emmeans makes the same check.
	if (!(df > 0.0) || !std::isfinite(df)) {
		return Fail("lin_hyp: df must be positive and finite");
	}
	if (!std::isfinite(sigma2) || sigma2 < 0.0) {
		return Fail("lin_hyp: sigma2 must be finite and not negative");
	}

	LinHypResult r;
	r.ok = true;
	r.df_den = df;

	// A rank-deficient fit carries NaN in the row and column of each aliased
	// coefficient. See estimability.hpp (#36): zeroing a working copy, not
	// multiplying by zero, is what makes a hypothesis that avoids those
	// coefficients safe to evaluate.
	const std::vector<bool> aliased = aliased_columns(V);
	for (std::size_t i = 0; i < q; i++) {
		if (row_touches_aliased(&L.data[i * k], k, aliased)) {
			return r; // estimable stays false — SAS's "non-est"
		}
	}
	const std::vector<double> beta_w = zero_aliased(beta, aliased);
	const linalg::Mat V_w = zero_aliased(V, aliased);

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

	// A caller may hand us something that is not a covariance: the kernel is a
	// public API and the covariance may be hand-built. A strictly negative
	// variance on C's diagonal is proof that it is not one. That is a
	// data-shaped failure, so the hypothesis is reported not estimable rather
	// than raising, which would abort a whole GROUP BY for one bad group. A
	// zero diagonal entry stays estimable: a zero row of L produces one.
	for (std::size_t i = 0; i < q; i++) {
		if (C(i, i) < 0.0) {
			return r;
		}
	}

	// Scale C to unit diagonal before pinv. pinv's rank tolerance is relative
	// to the largest singular value, and C's diagonal holds the constraints'
	// variances, so without this a constraint whose variance is about 1e12
	// smaller than another's is read as redundant and dropped from the test.
	// A change of units alone reaches that, and it moves both df_num and the F
	// statistic with no signal to the caller. See
	// notes/engineering/2026-10-lin-hyp-rank-tolerance-and-scale.md.
	//
	// Symmetrize in the same pass. The quadratic form only has a meaning for
	// the symmetric part, and the matrix product leaves a ~1e-17 asymmetry
	// behind even for a symmetric V.
	//
	// D = diag(1/√C_ii) makes D·C·D a correlation matrix, so
	// C⁻ = D·(D·C·D)⁻·D and W is unchanged when C has full rank.
	std::vector<double> d(q, 1.0);
	for (std::size_t i = 0; i < q; i++) {
		if (C(i, i) > 0.0) {
			d[i] = 1.0 / std::sqrt(C(i, i));
		}
	}
	linalg::Mat Cs(q, q);
	for (std::size_t i = 0; i < q; i++) {
		for (std::size_t j = 0; j < q; j++) {
			Cs(i, j) = 0.5 * (C(i, j) + C(j, i)) * d[i] * d[j];
		}
	}
	std::vector<double> theta_s(q);
	for (std::size_t i = 0; i < q; i++) {
		theta_s[i] = theta[i] * d[i];
	}

	// pinv returns the pseudo-inverse together with the numerical rank, so one
	// call covers both the full-rank and the redundant case and df_num is the
	// rank rather than the row count. statsmodels' wald_test agrees: given a
	// redundant hypothesis it reports the rank too.
	const linalg::Pinv pi = linalg::pinv(Cs);
	const std::size_t rank = pi.rank > 0 ? static_cast<std::size_t>(pi.rank) : 0;
	if (rank == 0) {
		return r; // a hypothesis with no information, such as a zero L
	}

	// The Wald statistic W = θᵀ C⁻ θ, computed on the scaled matrix.
	const std::vector<double> c_inv_theta = linalg::matvec(pi.value, theta_s);
	double w = 0.0;
	for (std::size_t i = 0; i < q; i++) {
		w += theta_s[i] * c_inv_theta[i];
	}
	// A negative W is the other proof that the supplied covariance is not one:
	// an indefinite C can pass the diagonal check and still give a negative
	// quadratic form. Not estimable, for the same reason as above.
	if (!(w >= 0.0)) {
		return r;
	}
	r.estimable = true;
	r.df_num = rank;
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
