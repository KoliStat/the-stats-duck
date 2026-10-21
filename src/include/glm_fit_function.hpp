#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// glm_fit(y, x [, family [, options]]) — generalized linear model aggregate.
//
// The iterative sibling of lm_fit. It fits one model per group by iteratively
// reweighted least squares, over four families, and reports the coefficient
// covariance so lin_hyp and emmeans work against the fit unchanged.
//
//   y        DOUBLE    — response. For binomial this is a PROPORTION in
//                        [0, 1], not a 0/1 count, and `weights` is the number
//                        of trials: 7 successes of 10 is y = 0.7, weight 10.
//                        A plain 0/1 logistic fit is the case weight = 1.
//   x        DOUBLE[]  — predictor row. The intercept is prepended unless the
//                        `intercept` option turns it off.
//   family   VARCHAR   — 'gaussian' (default), 'binomial', 'poisson', 'gamma'.
//                        A constant, folded at bind time.
//   options  STRUCT    — the optional inputs, read BY FIELD NAME:
//     link      VARCHAR const  — 'identity', 'logit', 'probit', 'cloglog',
//                                'log', 'inverse', 'sqrt'. Defaults to the
//                                family's canonical link.
//     vcov      VARCHAR const  — 'const' (default), 'HC0'–'HC3', or
//                                'CR0'/'CR1'. 'cluster' is an alias for CR1.
//     intercept BOOLEAN const  — prepend a constant term (default true)
//     weights   DOUBLE         — per-row prior weights (binomial: trial counts)
//     offset    DOUBLE         — per-row addition to the linear predictor, not
//                                estimated. A poisson rate model uses
//                                ln(exposure) here.
//     cluster   VARCHAR        — per-row cluster key; REQUIRED when vcov is
//                                CR0 or CR1 and rejected otherwise.
//
// lm_fit puts its optional inputs in positional overloads. glm_fit cannot: it
// has six of them, and `weights` and `offset` are per-row DOUBLEs that no
// position can tell apart. So they go in one trailing STRUCT. Field order
// there does not matter, and an unrecognized field name is a bind error that
// lists the valid names. Omitting the struct is the same as passing an empty
// one. Overloads: (y, x), (y, x, family), (y, x, family, options).
//
// Returns a STRUCT:
//   coefficients  LIST<STRUCT(term, estimate, std_error, z_statistic, p_value)>
//   n, k, df_residual                             BIGINT
//   deviance, null_deviance, dispersion,
//   loglik, aic                                   DOUBLE
//   iterations                                    BIGINT
//   converged, separation                         BOOLEAN
//   family, link                                  VARCHAR
//   has_intercept                                 BOOLEAN
//   vcov_type                                     VARCHAR
//   n_clusters                                    BIGINT — #clusters G (CR*
//                                                          only; else NULL)
//   rank                                          BIGINT
//   cov                                           LIST<DOUBLE> — row-major k²
//
// The statistic is z, not t: the reference is normal for every family,
// including gaussian and gamma where R and statsmodels switch to a t. The
// deviation is deliberate and src/include/glm_core.hpp records it.
//
// A row leaves the fit when its response, any design element, its weight, its
// offset or its cluster key is NULL, NaN or infinite, and also when its prior
// weight is exactly 0: such a row influences no estimate, so counting it would
// inflate df_residual and deflate the dispersion.
//
// Failure is a value, per docs/fitter_conventions.md. A data-shaped failure —
// too few rows, a response outside the family's support, a fitted mean that
// leaves that support, fewer than two clusters — makes that group's STRUCT
// NULL and never aborts the query. A fit that used every iteration is not a
// failure: it reports its best iterate with converged = false. Misuse — an
// unknown family, link or vcov, an unrecognized option name, a non-constant
// value in a constant-only option — is a bind error.
void RegisterGlmFit(ExtensionLoader &loader);

} // namespace duckdb
