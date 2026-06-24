#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// lm_fit(y, x [, vcov [, add_intercept]]) — OLS regression as an aggregate.
//
// A streaming-friendly companion to the lm()/lm_summary() table functions: it
// fits an ordinary-least-squares model per group, so a single GROUP BY query
// returns one regression per key. Unlike lm() (which takes column *names*), the
// aggregate consumes the design-matrix row as a LIST(DOUBLE) of predictor
// *values*, so coefficients come back by position.
//
//   y             DOUBLE        — response
//   x             DOUBLE[]      — predictor row (intercept auto-prepended)
//   vcov          VARCHAR const — 'const' (default) | 'HC0' | 'HC1' | 'HC2' | 'HC3'
//   add_intercept BOOLEAN const — prepend a constant term (default true)
//
// Returns a STRUCT:
//   coefficients  LIST<STRUCT(term, estimate, std_error, t_statistic, p_value)>
//   n, k, df_residual            BIGINT
//   r_squared, adj_r_squared,
//   sigma, f_statistic, f_p_value DOUBLE
//   has_intercept                BOOLEAN
//   vcov_type                    VARCHAR
//
// The std_error / t_statistic / p_value reflect the chosen `vcov`; HC* are the
// heteroskedasticity-consistent (Eicker–Huber–White) sandwich estimators. All
// numerics run on the shared statsduck linalg + lm_core layers. A group with too
// few rows (n ≤ k) or a singular/collinear design yields a NULL result for that
// group rather than aborting the query.
void RegisterLmFit(ExtensionLoader &loader);

} // namespace duckdb
