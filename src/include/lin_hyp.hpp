#pragma once

// stats_duck general linear hypothesis — the test of L·β = 0 against an
// already-computed fit.
//
// DuckDB-free and Eigen-free at the boundary, exactly like linalg.hpp and
// lm_core.hpp, so it is directly unit-testable and shareable downstream. It
// deliberately does NOT depend on lm_core: the inputs are a coefficient
// vector and its covariance, whatever produced them, so the same function
// serves lm_fit today and glm_fit later.
//
// Scope: the Wald test of q linear constraints, with the per-row estimates and
// standard errors that SAS's ESTIMATE prints. Type I and II sums of squares are
// NOT here — they need nested refits and the term-to-column mapping, which this
// function cannot reconstruct from β and cov(β).

#include "linalg.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace statsduck {

// Three outcomes, which callers must distinguish:
//   ok = false                  → misuse; `error` says what. Callers raise.
//   ok = true, estimable = false → the hypothesis cannot be evaluated (it puts
//                                  weight on a coefficient the fit could not
//                                  estimate, or it carries no information).
//                                  Callers report SQL NULL, per the
//                                  failure-is-a-value rule in
//                                  docs/fitter_conventions.md.
//   ok = true, estimable = true  → the full result below.
struct LinHypResult {
	bool ok = false;
	std::string error;
	bool estimable = false;

	// All length q, one per row of L, in L's order.
	std::vector<double> estimate;    // L·β
	std::vector<double> std_error;   // √diag(L·V·Lᵀ); 0 where the variance is not positive
	std::vector<double> t_statistic; // estimate / std_error; NaN where std_error is 0
	std::vector<double> p_value;     // two-sided, t(df); NaN where t is NaN

	double ss = 0.0;        // chisq · sigma2 — a sum of squares only when V is σ²(XᵀX)⁻¹
	std::size_t df_num = 0; // rank of L·V·Lᵀ, which is q when L has full row rank
	double df_den = 0.0;    // the df argument, echoed
	double f_statistic = 0.0;
	double f_p_value = 0.0;
	double chisq = 0.0;     // the Wald statistic θᵀ(L·V·Lᵀ)⁻θ
	double chisq_p_value = 0.0;
};

// Test L·β = 0.
//   L       q × k, one row per constraint; q rows means a joint test.
//   beta    length k.
//   V       k × k covariance of beta, for whichever estimator produced it.
//   sigma2  residual variance, used only to scale the Wald statistic into `ss`.
//   df      denominator degrees of freedom for the t and F references; > 0.
//
// A rank-deficient fit carries NaN in the rows and columns of its aliased
// coefficients. A hypothesis that puts no weight there is still estimable, so
// those rows and columns are zeroed in a working copy before the product rather
// than being allowed to poison it (IEEE makes 0 · NaN = NaN).
LinHypResult lin_hyp(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V,
                     double sigma2, double df);

} // namespace statsduck
