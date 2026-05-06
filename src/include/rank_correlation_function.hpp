#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers spearman_test(x, y, [alpha], [alternative]) — Spearman rank
//! correlation. Significance via Student's t with df = n-2 (the Pearson-on-
//! ranks asymptotic, what R's cor.test returns). Fisher-z confidence interval
//! on rho. Returns STRUCT(test_type, rho, t_statistic, df, p_value,
//! alternative, ci_lower, ci_upper, n).
void RegisterSpearmanTest(ExtensionLoader &loader);

//! Registers kendall_test(x, y, [alternative]) — Kendall's tau-b with Knight's
//! tie-corrected variance and a normal-approximation z-statistic. Always uses
//! the normal approximation (R's cor.test uses an exact test for small-n
//! tie-free input by default; we don't, hence small-n p-values may differ).
//! Returns STRUCT(test_type, tau, z_statistic, p_value, alternative, n).
void RegisterKendallTest(ExtensionLoader &loader);

} // namespace duckdb
