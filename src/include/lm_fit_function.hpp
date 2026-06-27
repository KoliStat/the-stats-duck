#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// lm_fit(y, x [, vcov [, cluster] [, add_intercept]]) — OLS regression aggregate.
//
// A streaming-friendly companion to the lm()/lm_summary() table functions: it
// fits an ordinary-least-squares model per group, so a single GROUP BY query
// returns one regression per key. Unlike lm() (which takes column *names*), the
// aggregate consumes the design-matrix row as a LIST(DOUBLE) of predictor
// *values*, so coefficients come back by position.
//
//   y             DOUBLE        — response
//   x             DOUBLE[]      — predictor row (intercept auto-prepended)
//   vcov          VARCHAR const — 'const' (default) | 'HC0'–'HC3' | 'CR0' | 'CR1'
//   cluster       VARCHAR       — per-row cluster key; REQUIRED iff vcov is CR*
//                                 (cast a non-text key with ::VARCHAR). Real
//                                 per-row data, not a constant.
//   add_intercept BOOLEAN const — prepend a constant term (default true)
//
// Overloads: (y,x), (y,x,vcov), (y,x,vcov,add_intercept), (y,x,vcov,cluster),
// (y,x,vcov,cluster,add_intercept). Passing CR0/CR1 without a cluster column, or
// a cluster column with a non-CR vcov, is a bind error.
//
// Returns a STRUCT:
//   coefficients  LIST<STRUCT(term, estimate, std_error, t_statistic, p_value)>
//   n, k, df_residual            BIGINT
//   r_squared, adj_r_squared,
//   sigma, f_statistic, f_p_value DOUBLE
//   has_intercept                BOOLEAN
//   vcov_type                    VARCHAR
//   n_clusters                   BIGINT  — #clusters G (CR* only; else NULL)
//
// The std_error / t_statistic / p_value reflect the chosen `vcov`: HC* are the
// heteroskedasticity-consistent (Eicker–Huber–White) sandwich estimators; CR0/CR1
// are the cluster-robust (Liang–Zeger) sandwich, whose p-values use a t(G−1)
// reference (df_residual is still reported as n−k). All numerics run on the shared
// statsduck linalg + lm_core layers. A group with too few rows (n ≤ k), a
// singular/collinear design, or (for CR*) fewer than two clusters yields a NULL
// result for that group rather than aborting the query.
void RegisterLmFit(ExtensionLoader &loader);

} // namespace duckdb
