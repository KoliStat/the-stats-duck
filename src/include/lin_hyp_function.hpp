#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers lin_hyp(L LIST<LIST<DOUBLE>>, beta LIST<DOUBLE>,
//! cov LIST<DOUBLE>, sigma2 DOUBLE, df DOUBLE) → STRUCT.
//!
//! Tests the general linear hypothesis L·β = 0 against an already-computed
//! fit. `L` holds one row per constraint, each as wide as `beta`; `cov` is the
//! flat row-major k×k covariance of `beta`, exactly as lm_fit returns it.
//! Returns the per-row estimates with their standard errors, t and p, plus the
//! joint test as a sum of squares, an F and a chi-square. A hypothesis that
//! cannot be evaluated — one touching a coefficient the fit could not estimate,
//! or one carrying no information — yields NULL. A dimension mismatch raises.
void RegisterLinHyp(ExtensionLoader &loader);

} // namespace duckdb
