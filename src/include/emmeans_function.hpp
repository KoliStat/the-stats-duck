#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers emmeans(L LIST<LIST<DOUBLE>>, beta LIST<DOUBLE>,
//! cov LIST<DOUBLE>, df DOUBLE [, alpha DOUBLE [, adjust VARCHAR]]) → STRUCT.
//!
//! Estimated marginal means for an already-computed fit. `L` holds one row per
//! marginal mean, each as wide as `beta`; `cov` is the flat row-major k×k
//! covariance of `beta`, exactly as lm_fit returns it. Returns the means with
//! their standard errors, t, p and confidence limits, every pairwise
//! difference between them, and the multiplicity adjustment `adjust` names.
//! `alpha` defaults to 0.05 and `adjust` to 'tukey'.
//!
//! Estimability is per element: a mean that touches a coefficient the fit
//! could not estimate is NULL, every pair touching it is NULL, and the rest of
//! the grid is reported normally. A dimension mismatch, a non-positive df, an
//! alpha outside (0, 1) or an unknown adjustment name raises.
void RegisterEmmeans(ExtensionLoader &loader);

} // namespace duckdb
