#pragma once

// stats_duck estimated marginal means — the means a grid of linear
// combinations picks out of an already-computed fit, their standard errors and
// confidence limits, every pairwise difference between them, and a
// multiplicity adjustment for that family.
//
// DuckDB-free and Eigen-free at the boundary, exactly like linalg.hpp and
// lin_hyp.hpp, so it is directly unit-testable and shareable downstream. It
// deliberately does NOT depend on lm_core: the inputs are a coefficient vector
// and its covariance, whatever produced them, so the same function serves
// lm_fit today and glm_fit later.
//
// Scope: θ = L·β and C = L·V·Lᵀ, formed once, and then every reported number
// is an entry of θ or of C. Three things this is NOT:
//
//   1. It does not build the grid. L arrives already assembled — which levels,
//      which covariate values and which weights are the caller's decisions,
//      and they need the data and the term-to-column mapping, neither of which
//      is reconstructible from β and cov(β).
//   2. It does not compute Satterthwaite or Kenward-Roger degrees of freedom.
//      `df` is supplied and echoed. Those methods need the variance-component
//      structure of a mixed fit, not a single covariance matrix.
//   3. It inverts nothing. There is no pseudo-inverse here, no equilibration
//      and no rank tolerance, because there is no quadratic form to invert
//      through — unlike lin_hyp, which needs all three. The only tolerance is
//      the one that decides whether a slightly negative variance is a rounding
//      artefact; see below.
//
// `i` and `j` in EmmDiff are 0-BASED, indexing L's rows directly. The SQL
// layer adds one before reporting, because SQL users count levels from 1. That
// is the obvious place for an off-by-one, so it is stated here rather than
// left for a reader to infer.

#include "linalg.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace statsduck {

// The multiplicity adjustments emmeans offers. Six of these map one to one
// onto statsduck::AdjustMethod, which is everything computable from a list of
// p-values. TUKEY is the seventh and cannot be: the studentized range needs
// the t statistic and the family size, not the p-value, which is why
// p_adjust.hpp's parse_adjust deliberately rejects the name "tukey".
enum class EmmAdjust {
	TUKEY,
	BONFERRONI,
	HOLM,
	HOCHBERG,
	BH,
	BY,
	NONE,
};

// Parse a method name. Case-sensitive, matching statsduck::parse_adjust, since
// both are the same SQL surface. Accepts exactly "tukey", "bonferroni",
// "holm", "hochberg", "BH", "fdr" (an alias for BH), "BY" and "none". Returns
// false, leaving `out` untouched, on anything else.
bool parse_emm_adjust(const std::string &name, EmmAdjust &out);

// Canonical name for a method, echoed back by the SQL layer: "tukey",
// "bonferroni", "holm", "hochberg", "BH", "BY" or "none". "fdr" comes back as
// "BH", since they are the same method.
const char *emm_adjust_name(EmmAdjust m);

// One marginal mean: one row of L.
//
// estimable = false means this row cannot be evaluated — it puts weight on a
// coefficient the fit could not estimate, or its variance is negative past the
// tolerance. Callers report SQL NULL, per the failure-is-a-value rule in
// docs/fitter_conventions.md. Every numeric field of a non-estimable element is
// NaN: the flag is the contract, and the NaN is there so a caller that forgets
// to check it fails loudly instead of reading a zero as an estimate.
struct EmmMean {
	bool estimable = false;
	double estimate = 0.0,   // θᵢ
	    std_error = 0.0,     // √C_ii, 0 where the variance is not positive
	    t_statistic = 0.0,   // estimate / std_error; NaN at a zero std_error
	    p_value = 0.0,       // two-sided, t(df); NaN where t is NaN
	    lower_cl = 0.0,      // estimate ∓ t(1 − alpha/2, df) · std_error,
	    upper_cl = 0.0;      //   never adjusted for multiplicity
};

// One pairwise difference, mean i minus mean j.
//
// i and j are 0-based row indices into L, and i < j always. Every pair is
// reported, including pairs that touch a non-estimable mean, so a caller can
// index the list by position: dropping them would shift every later pair.
struct EmmDiff {
	std::size_t i = 0, j = 0; // 0-based here; the SQL layer adds 1
	bool estimable = false;
	double estimate = 0.0,  // θᵢ − θⱼ
	    std_error = 0.0,    // √(C_ii + C_jj − 2·C_ij)
	    t_statistic = 0.0,  // estimate / std_error; NaN at a zero std_error
	    p_value = 0.0,      // two-sided, t(df); NaN where t is NaN
	    p_adjusted = 0.0,   // p_value under `adjust`; NaN where p_value is NaN
	    lower_cl = 0.0,     // estimate ∓ the method's multiplier · std_error
	    upper_cl = 0.0;
};

// Two levels of outcome, which callers must keep apart:
//
//   ok = false  → misuse; `error` says what, naming the function and the
//                 actual sizes. `means` and `diffs` are empty. Callers raise.
//   ok = true   → the grid was evaluated. Estimability is then PER ELEMENT,
//                 on each EmmMean and each EmmDiff. There is no top-level
//                 estimable flag, unlike LinHypResult, because one unusable
//                 grid row must not discard the others.
struct EmmeansResult {
	bool ok = false;
	std::string error;
	std::vector<EmmMean> means; // one per row of L, in L's order
	std::vector<EmmDiff> diffs; // i < j, row-major
	double df_den = 0.0,        // the df argument, echoed
	    alpha = 0.0;            // the alpha argument, echoed
	EmmAdjust adjust = EmmAdjust::TUKEY; // the method, echoed
	std::size_t k_compare = 0;  // estimable means: the family size Tukey uses
};

// Estimated marginal means for the grid L.
//   L       q × k, one row per marginal mean. q = 0 is misuse, not an empty
//           answer, matching lin_hyp: a grid of no means is a request for
//           nothing, and two neighbouring functions must not answer the same
//           malformed input differently. A ZERO ROW inside L is fine — it is
//           the degenerate contrast that estimates zero with zero variance.
//   beta    length k.
//   V       k × k covariance of beta, for whichever estimator produced it.
//   df      denominator degrees of freedom for every t and studentized range
//           reference; must be positive and finite.
//   alpha   two-sided level for the confidence limits; must be in (0, 1).
//   adjust  the multiplicity adjustment for the pairwise family.
//
// Validation runs in this order, and the first failure is the one reported: L
// has no rows, a row width that is not k, a covariance that is not k × k, a
// beta of length 0, a df that is not positive and finite, an alpha outside
// (0, 1).
//
// A rank-deficient fit carries NaN in the rows and columns of its aliased
// coefficients. Those rows and columns are zeroed in a working copy before the
// products rather than being allowed to poison them (IEEE makes 0 · NaN =
// NaN), and the test for which grid rows may use the result is applied PER
// ROW: a row that puts non-zero weight on an aliased coefficient is not
// estimable, and the rest of the grid is evaluated normally. lin_hyp treats
// any such row as making the whole hypothesis non-estimable, which is right
// there — a joint test is one answer — and wrong here. See estimability.hpp.
//
// V need not come from lm_fit, so it is not assumed to be a covariance.
// C = L·V·Lᵀ is symmetrized where it is read off the diagonal, because only
// the symmetric part has a meaning. A variance that comes out negative is
// either a rounding artefact or proof that V is not a covariance, and the two
// are told apart by a RELATIVE tolerance of 1e-12 whose scale differs by case:
// max_j C_jj for a mean, and C_ii + C_jj for a difference. A negative variance
// inside the tolerance counts as zero, so the element keeps its estimate with
// a zero standard error and a NaN t and p; past the tolerance the element is
// not estimable. When no positive scale exists the tolerance is zero, so any
// negative value is not estimable. A variance of exactly zero stays estimable:
// two identical grid rows produce one, and so does a zero row of L.
//
// The adjustment, with m the pairs that have a p-value to adjust and
// k_compare the estimable means:
//   TUKEY       p_adjusted = 1 − ptukey(√2·|t|, k_compare, df), and limits at
//               qtukey(1 − alpha, k_compare, df)/√2 · std_error. Neither
//               function is called at all when k_compare < 2, where the
//               studentized range is undefined.
//   BONFERRONI  p_adjusted through statsduck::adjust_p, limits at
//               t(1 − alpha/(2m), df).
//   HOLM, HOCHBERG, BH, BY
//               p_adjusted through statsduck::adjust_p; the limits stay at
//               t(1 − alpha/2, df), because a step-down or step-up procedure
//               gives no interval.
//   NONE        p_adjusted = p_value, limits at t(1 − alpha/2, df).
// A mean's limits are never adjusted under any method.
EmmeansResult emmeans(const linalg::Mat &L, const std::vector<double> &beta, const linalg::Mat &V,
                      double df, double alpha, EmmAdjust adjust);

} // namespace statsduck
