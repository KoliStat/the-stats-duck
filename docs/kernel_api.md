# Kernel API — the DuckDB-free numerics core

`stats_duck`'s statistical models are built on a small C++ core that is
**deliberately DuckDB-free and Eigen-free at its API boundary**: every signature
speaks only `std::vector<double>` and a row-major `Mat`. That boundary is what
lets the core be reused by a downstream extension (e.g. the planned
`the-stats-duck-bio`) **without** pulling in DuckDB — a sibling repo adds
`stats_duck` as a git submodule, puts `src/include` on its include path, and
calls the kernel directly.

This page is the orientation map. **The headers are the source of truth** — each
function's preconditions, tolerances, and result semantics live in the
doc-comments of [`src/include/linalg.hpp`](../src/include/linalg.hpp),
[`src/include/lm_core.hpp`](../src/include/lm_core.hpp), and
[`src/include/optimize.hpp`](../src/include/optimize.hpp).

## Consuming the core from a sibling repo

The API boundary is header-only, but the implementation is not: `linalg.cpp`
uses Eigen (confined to that one translation unit) and `lm_core.cpp` builds on
it. So a consumer:

1. adds `src/include` to its include path → `#include "linalg.hpp"`,
   `#include "lm_core.hpp"`;
2. compiles `src/linalg.cpp` and `src/lm_core.cpp` into its own build;
3. makes the Eigen headers (`third_party/eigen`) available **when compiling
   `linalg.cpp`** — that is the only Eigen dependency, and it never leaks past
   that TU.

`optimize.hpp` is simpler still: it is **fully header-only** (Eigen-free even in
implementation) — include it and go, nothing extra to compile.

No DuckDB, no `idx_t`, no ABI coupling to the extension. Everything is in
namespace `statsduck` (linear algebra in `statsduck::linalg`, optimization in
`statsduck::optimize`).

## `linalg.hpp` — dense linear-algebra kernel

Pure numerics: solves, rank, pseudo-inverse, the covariance sandwich. Row-major
`Mat { rows, cols, data }` with `operator()(i, j)`.

| Function | Signature | Notes |
| --- | --- | --- |
| `cholesky_solve` | `(const Mat& A, const vector<double>& b) -> Solution` | SPD `A`; `ok = false` if not positive-definite |
| `qr_solve` | `(const Mat& A, const vector<double>& b) -> Solution` | least squares, assumes full column rank |
| `svd_solve` | `(const Mat& A, const vector<double>& b, double tol = 1e-12) -> Solution` | rank-revealing minimum-norm solution |
| `rank` | `(const Mat& A, double tol = 1e-12) -> int` | numerical rank (singular values > `tol·σ_max`) |
| `pinv` | `(const Mat& A, double tol = 1e-12) -> Pinv` | Moore–Penrose pseudo-inverse + rank |
| `inv_spd` | `(const Mat& A, bool* ok = nullptr) -> Mat` | inverse of an SPD matrix (e.g. `(XᵀX)⁻¹`) |
| `sandwich` | `(const Mat& L, const Mat& A) -> Mat` | `L · A · Lᵀ` — the covariance sandwich |
| `matvec` | `(const Mat& A, const vector<double>& x) -> vector<double>` | matrix–vector product |

`Solution { vector<double> x; int rank; bool ok; }` carries the numerical rank so
callers can detect and report rank deficiency.

## `lm_core.hpp` — ordinary least squares + robust covariance

Fits OLS and assembles the full inferential summary on the linalg kernel.

```cpp
LmResult fit_lm(const std::vector<double>& y,      // response, length n
                const linalg::Mat& X_pred,         // n × p predictors (NO intercept column)
                const LmOptions& opts,             // { Vcov vcov; bool intercept; }
                const std::vector<int>* cluster_ids = nullptr); // CR* only: dense 0-based, length n
```

| Symbol | Kind | Notes |
| --- | --- | --- |
| `Vcov` | enum | `kConst`, `kHC0`–`kHC3` (heteroskedasticity-consistent), `kCR0`/`kCR1` (cluster-robust) |
| `LmOptions` | struct | `vcov` (default `kConst`), `intercept` (default `true`, prepends a constant) |
| `LmResult` | struct | `ok`/`error`; `n`, `k`, `df_residual`, `n_clusters`; per-term `terms`/`beta`/`std_error`/`t_statistic`/`p_value`; model `r_squared`, `adj_r_squared`, `f_statistic`, `f_p_value`, `sigma` |
| `parse_vcov` | `(const string&, Vcov&) -> bool` | case-insensitive: `const`/`none`/`ols`/…, `hc0`–`hc3`, `cr0`/`cr1`/`cluster` (→ `kCR1`) |
| `vcov_name` | `(Vcov) -> const char*` | canonical label echoed back in `LmResult.vcov` |

Conventions worth knowing before consuming `LmResult`:

- **Intercept** is prepended by `fit_lm` when `opts.intercept` — pass `X_pred`
  *without* a constant column. `terms` are labelled `(Intercept)`, `x1`, `x2`, …
- **Covariance** drives only `std_error`/`t_statistic`/`p_value`; `beta` is
  invariant across `vcov`. HC* use the sandwich with leverage-weighted meat;
  CR0/CR1 use per-cluster score sums (CR1 with the
  `[G/(G−1)]·[(N−1)/(N−k)]` finite-sample factor).
- **Inference df.** Classical/HC use `t(n−k)`; cluster-robust uses `t(G−1)`,
  where `G = n_clusters`. `df_residual` is always reported as `n−k` regardless —
  it is a property of the fit, not the SE estimator.
- **Failure is a value, not an exception.** `ok = false` with a populated
  `error` on `n ≤ k`, a singular/collinear design, or (for CR*) a missing
  cluster vector or fewer than two clusters. Callers branch on `ok`.

## `optimize.hpp` — derivative-free minimization

Classic Nelder–Mead simplex minimization, built for profile-likelihood /
REML-style surfaces: few parameters (1–5), no cheap gradients, positivity
handled by log-reparameterization in the caller. Fully header-only.

```cpp
optimize::Result r = optimize::nelder_mead(
    [](const std::vector<double>& x) { return f(x); },  // objective
    x0,                                                  // start point
    opts);                                               // optional Options
```

| Symbol | Kind | Notes |
| --- | --- | --- |
| `Options` | struct | `xtol` (1e-8), `ftol` (1e-10), `max_eval` (2000), `init_step` (0.05) |
| `Result` | struct | `x`, `fx`, `n_eval`, `n_iter`, `converged` |
| `nelder_mead` | `(function<double(const vector<double>&)>, vector<double> x0, const Options& = {}) -> Result` | minimizes from `x0` |

Conventions:

- **NaN/±Inf objective values act as +Inf** — invalid regions of a likelihood
  surface repel the simplex instead of poisoning the vertex ordering.
- **Convergence requires both tolerances jointly**: the simplex must be flat in
  `f` (`ftol`) *and* small in `x` (`xtol`); otherwise the run ends at
  `max_eval` with `converged = false` and `x` holding the best point seen.
- **Deterministic by construction** — no RNG, no restarts; identical inputs
  give bit-identical results. `n = 1` (two-point simplex) is supported.
- **Failure is a value**: degenerate input (empty `x0`, `max_eval ≤ 0`) returns
  `converged = false` with `fx = NaN`, never throws.

## Testing & validation

All three layers are unit-tested directly, standalone (no DuckDB), against
golden values — `test/cpp/test_linalg.cpp`, `test/cpp/test_lm_fit.cpp`
(statsmodels oracle), and `test/cpp/test_optimize.cpp` (classic optimization
test functions). Build and run them with `scripts/run-cpp-tests.sh` (or
`make test_cpp`).
