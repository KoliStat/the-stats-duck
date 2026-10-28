#include "emmeans.hpp"

#include "distributions.hpp" // stats_duck::StudentT* and StudentizedRange*
#include "estimability.hpp"
#include "p_adjust.hpp"

#include <cmath>
#include <limits>
#include <string>

namespace statsduck {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Relative tolerance on a negative variance. C = L·V·Lᵀ is a difference of
// products, so a variance that is mathematically zero can come back at about
// 1e-16 times the size of the terms that cancelled. Treating that as proof
// that V is not a covariance would make two identical grid rows unusable, and
// repeating a level is something a user does on purpose. The scale differs by
// case, because the terms that cancel differ: max_j C_jj for a mean and
// C_ii + C_jj for a difference. See emmeans.hpp.
const double kNegVarTol = 1e-12;

EmmeansResult Fail(const std::string &msg) {
	EmmeansResult r;
	r.ok = false;
	r.error = msg;
	return r;
}

// Every numeric field NaN, so a caller that ignores `estimable` fails loudly
// rather than reading a zero as an estimate.
void MarkNotEstimable(EmmMean &m) {
	m.estimable = false;
	m.estimate = kNaN;
	m.std_error = kNaN;
	m.t_statistic = kNaN;
	m.p_value = kNaN;
	m.lower_cl = kNaN;
	m.upper_cl = kNaN;
}

void MarkNotEstimable(EmmDiff &d) {
	d.estimable = false;
	d.estimate = kNaN;
	d.std_error = kNaN;
	d.t_statistic = kNaN;
	d.p_value = kNaN;
	d.p_adjusted = kNaN;
	d.lower_cl = kNaN;
	d.upper_cl = kNaN;
}

// Half the width of a confidence interval. A zero standard error gives a
// zero-width interval at every confidence level, and saying that outright
// rather than multiplying is what keeps an INFINITE critical value from
// turning a point estimate's limits into NaN. An alpha below about 2.2e-16
// reaches that: 1 − alpha/2 then rounds to exactly 1, and StudentTQuantile
// returns +infinity there by design. IEEE makes infinity · 0 = NaN.
double HalfWidth(double crit, double std_error) {
	return std_error > 0.0 ? crit * std_error : 0.0;
}

// The six methods statsduck::adjust_p can compute, mapped one to one. TUKEY is
// not one of them — the studentized range needs the t statistic and the family
// size, not the p-value — and the caller never reaches here with it.
AdjustMethod ToAdjustMethod(EmmAdjust m) {
	switch (m) {
	case EmmAdjust::BONFERRONI:
		return AdjustMethod::BONFERRONI;
	case EmmAdjust::HOLM:
		return AdjustMethod::HOLM;
	case EmmAdjust::HOCHBERG:
		return AdjustMethod::HOCHBERG;
	case EmmAdjust::BH:
		return AdjustMethod::BENJAMINI_HOCHBERG;
	case EmmAdjust::BY:
		return AdjustMethod::BENJAMINI_YEKUTIELI;
	case EmmAdjust::NONE:
		return AdjustMethod::NONE;
	case EmmAdjust::TUKEY:
		break;
	}
	return AdjustMethod::NONE;
}

} // namespace

bool parse_emm_adjust(const std::string &name, EmmAdjust &out) {
	if (name == "tukey") {
		out = EmmAdjust::TUKEY;
	} else if (name == "bonferroni") {
		out = EmmAdjust::BONFERRONI;
	} else if (name == "holm") {
		out = EmmAdjust::HOLM;
	} else if (name == "hochberg") {
		out = EmmAdjust::HOCHBERG;
	} else if (name == "BH" || name == "fdr") {
		out = EmmAdjust::BH;
	} else if (name == "BY") {
		out = EmmAdjust::BY;
	} else if (name == "none") {
		out = EmmAdjust::NONE;
	} else {
		return false;
	}
	return true;
}

const char *emm_adjust_name(EmmAdjust m) {
	switch (m) {
	case EmmAdjust::TUKEY:
		return "tukey";
	case EmmAdjust::BONFERRONI:
		return "bonferroni";
	case EmmAdjust::HOLM:
		return "holm";
	case EmmAdjust::HOCHBERG:
		return "hochberg";
	case EmmAdjust::BH:
		return "BH";
	case EmmAdjust::BY:
		return "BY";
	case EmmAdjust::NONE:
		return "none";
	}
	return "tukey";
}

EmmeansResult emmeans(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V,
                      double df, double alpha, EmmAdjust adjust) {
	const std::size_t n_means = L.rows; // `q` elsewhere in this plan, but q is
	                                    // the studentized range's quantile
	const std::size_t k = beta.size();

	if (n_means == 0) {
		return Fail("emmeans: the grid has no rows");
	}
	if (L.cols != k) {
		return Fail("emmeans: the grid has " + std::to_string(L.cols) + " columns but beta has " +
		            std::to_string(k) + " elements");
	}
	if (V.rows != k || V.cols != k) {
		return Fail("emmeans: the covariance must be " + std::to_string(k) + "×" + std::to_string(k) +
		            ", got " + std::to_string(V.rows) + "×" + std::to_string(V.cols));
	}
	if (k == 0) {
		return Fail("emmeans: beta is empty");
	}
	if (!(df > 0.0) || !std::isfinite(df)) {
		return Fail("emmeans: df must be positive and finite");
	}
	if (!(alpha > 0.0) || !(alpha < 1.0)) {
		return Fail("emmeans: alpha must be inside (0, 1)");
	}

	EmmeansResult r;
	r.ok = true;
	r.df_den = df;
	r.alpha = alpha;
	r.adjust = adjust;

	// A rank-deficient fit carries NaN in the row and column of each aliased
	// coefficient. See estimability.hpp (#36): zeroing a working copy, not
	// multiplying by zero, is what makes a grid row that avoids those
	// coefficients safe to evaluate.
	const std::vector<bool> aliased = aliased_columns(V);
	const std::vector<double> beta_w = zero_aliased(beta, aliased);
	const linalg::Mat V_w = zero_aliased(V, aliased);

	// θ = L·β and C = L·V·Lᵀ: the marginal means and their covariance. Formed
	// once; every number below is an entry of one or the other.
	const std::vector<double> theta = linalg::matvec(L, beta_w);
	const linalg::Mat C = linalg::sandwich(L, V_w);
	if (theta.size() != n_means || C.rows != n_means || C.cols != n_means) {
		return Fail("emmeans: failed to form L·β or L·V·Lᵀ");
	}

	// The tolerance scale for a mean's variance: the largest variance in the
	// grid. Only finite positive entries can set it, and when none does the
	// tolerance is zero, so any negative variance is then taken at face value.
	double max_diag = 0.0;
	for (std::size_t i = 0; i < n_means; i++) {
		const double v = C(i, i);
		if (std::isfinite(v) && v > max_diag) {
			max_diag = v;
		}
	}
	const double tol_mean = kNegVarTol * max_diag;

	const double t_crit = stats_duck::StudentTQuantile(1.0 - 0.5 * alpha, df);

	r.means.assign(n_means, EmmMean());
	for (std::size_t i = 0; i < n_means; i++) {
		EmmMean &m = r.means[i];
		const double var = C(i, i);
		// Three ways a row cannot be evaluated: it puts weight on a coefficient
		// the fit dropped; something non-finite reached θ or C from somewhere
		// other than an aliased column; or the variance is negative past the
		// tolerance, which proves V is not a covariance.
		if (row_touches_aliased(&L.data[i * k], k, aliased) || !std::isfinite(theta[i]) ||
		    !std::isfinite(var) || var < -tol_mean) {
			MarkNotEstimable(m);
			continue;
		}
		m.estimable = true;
		r.k_compare++;
		m.estimate = theta[i];
		m.std_error = var > 0.0 ? std::sqrt(var) : 0.0;
		if (m.std_error > 0.0) {
			m.t_statistic = m.estimate / m.std_error;
			m.p_value = 2.0 * (1.0 - stats_duck::StudentTCDF(std::fabs(m.t_statistic), df));
		} else {
			// A zero standard error gives no t and no p, but the estimate and
			// the limits are still the right answer: the interval is a point.
			m.t_statistic = kNaN;
			m.p_value = kNaN;
		}
		// A mean's limits are never adjusted for multiplicity, whatever the
		// pairwise method is.
		const double half = HalfWidth(t_crit, m.std_error);
		m.lower_cl = m.estimate - half;
		m.upper_cl = m.estimate + half;
	}

	// Pairs: i < j, row-major. EVERY pair is emitted, including one that
	// touches a non-estimable mean, so positions stay stable and a caller can
	// index the list — dropping a pair would shift every later one.
	r.diffs.reserve(n_means * (n_means - 1) / 2);
	for (std::size_t i = 0; i < n_means; i++) {
		for (std::size_t j = i + 1; j < n_means; j++) {
			EmmDiff d;
			d.i = i;
			d.j = j;
			if (!r.means[i].estimable || !r.means[j].estimable) {
				MarkNotEstimable(d);
				r.diffs.push_back(d);
				continue;
			}
			// Only the symmetric part of C has a meaning in a variance, and the
			// matrix product leaves about 1e-17 of asymmetry behind even when V
			// itself is symmetric.
			const double c_ij = 0.5 * (C(i, j) + C(j, i));
			const double var = C(i, i) + C(j, j) - 2.0 * c_ij;
			// The scale is this pair's own two variances, which are the terms
			// that cancel when the two grid rows are identical.
			const double scale = C(i, i) + C(j, j);
			const double tol_pair = scale > 0.0 ? kNegVarTol * scale : 0.0;
			if (!std::isfinite(var) || var < -tol_pair) {
				MarkNotEstimable(d);
				r.diffs.push_back(d);
				continue;
			}
			d.estimable = true;
			d.estimate = theta[i] - theta[j];
			d.std_error = var > 0.0 ? std::sqrt(var) : 0.0;
			if (d.std_error > 0.0) {
				d.t_statistic = d.estimate / d.std_error;
				d.p_value = 2.0 * (1.0 - stats_duck::StudentTCDF(std::fabs(d.t_statistic), df));
			} else {
				// Two identical grid rows reach this. The difference is exactly
				// zero with no variance, so there is no t and no p, but the
				// estimate and the interval are still the right answer.
				d.t_statistic = kNaN;
				d.p_value = kNaN;
			}
			d.p_adjusted = kNaN; // both filled in below, once the family is known
			d.lower_cl = kNaN;
			d.upper_cl = kNaN;
			r.diffs.push_back(d);
		}
	}

	// The family is the estimable pairs that have a p-value to adjust. A pair
	// whose standard error is zero has none, and the step-down and step-up
	// procedures work on the ranks of the p-values, so including it would
	// corrupt every other pair's rank. Its own adjusted p stays NaN, which is
	// what its raw p already is.
	std::vector<double> p_raw(r.diffs.size(), kNaN);
	std::vector<bool> valid(r.diffs.size(), false);
	std::size_t m = 0;
	for (std::size_t d = 0; d < r.diffs.size(); d++) {
		p_raw[d] = r.diffs[d].p_value;
		valid[d] = r.diffs[d].estimable && std::isfinite(p_raw[d]);
		if (valid[d]) {
			m++;
		}
	}

	// pair_crit is the multiplier on a difference's standard error. It differs
	// by method; a mean's own limits never do.
	double pair_crit = kNaN;
	if (adjust == EmmAdjust::TUKEY) {
		// The studentized range is undefined below two means, so neither
		// function is called there. Defence in depth, not the load-bearing
		// protection: what actually keeps a NaN out of the output is the
		// structural invariant that an estimable pair needs two estimable
		// means, so k_compare >= 2 holds whenever any pair is written and the
		// loop below reads pair_crit. Removing this guard leaves the output
		// bit-identical, which is why no test can catch its absence.
		if (r.k_compare >= 2) {
			const double k_d = static_cast<double>(r.k_compare);
			pair_crit = stats_duck::StudentizedRangeQuantile(1.0 - alpha, k_d, df) / std::sqrt(2.0);
			for (EmmDiff &d : r.diffs) {
				if (d.estimable) {
					const double q = std::sqrt(2.0) * std::fabs(d.t_statistic);
					d.p_adjusted = 1.0 - stats_duck::StudentizedRangeCDF(q, k_d, df);
				}
			}
		}
	} else {
		// The other six go through the shared helper, which also writes NaN
		// into the slots the mask excluded. NONE is one of them: adjust_p
		// copies the p-value through, so p_adjusted equals p_value.
		std::vector<double> p_out;
		adjust_p(p_raw, valid, ToAdjustMethod(adjust), p_out);
		for (std::size_t d = 0; d < r.diffs.size(); d++) {
			if (r.diffs[d].estimable) {
				r.diffs[d].p_adjusted = p_out[d];
			}
		}
		if (adjust == EmmAdjust::BONFERRONI) {
			// Bonferroni is the only one of the six with an interval: the level
			// is split across the family. An empty family is no multiplicity at
			// all, so the divisor is at least one, which is the unadjusted
			// quantile.
			const double m_d = static_cast<double>(m > 0 ? m : 1);
			pair_crit = stats_duck::StudentTQuantile(1.0 - alpha / (2.0 * m_d), df);
		} else {
			// A step-down or step-up procedure gives no interval, so the limits
			// stay unadjusted.
			pair_crit = t_crit;
		}
	}
	for (EmmDiff &d : r.diffs) {
		if (d.estimable) {
			const double half = HalfWidth(pair_crit, d.std_error);
			d.lower_cl = d.estimate - half;
			d.upper_cl = d.estimate + half;
		}
	}

	return r;
}

} // namespace statsduck
