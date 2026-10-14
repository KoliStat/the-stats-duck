# Kernel API — the DuckDB-free numerics core

`stats_duck`'s statistical models are built on a small C++ core that is
**deliberately DuckDB-free and Eigen-free at its API boundary**: every signature
speaks only `std::vector<double>` and a row-major `Mat`. That boundary is what
lets the core be reused by a downstream extension (e.g. the planned
`the-stats-duck-bio`) **without** pulling in DuckDB — a sibling repo adds
`stats_duck` as a git submodule, puts `src/include` on its include path, and
calls the kernel directly.

The SQL-facing counterpart of this page is
[`fitter_conventions.md`](fitter_conventions.md): the conventions every
model-fitting function's SQL surface follows (inputs, the return-STRUCT
standard, validation discipline).

This page is the orientation map. **The headers are the source of truth** — each
function's preconditions, tolerances, and result semantics live in the
doc-comments of [`src/include/linalg.hpp`](../src/include/linalg.hpp),
[`src/include/lm_core.hpp`](../src/include/lm_core.hpp),
[`src/include/lin_hyp.hpp`](../src/include/lin_hyp.hpp), and
[`src/include/optimize.hpp`](../src/include/optimize.hpp).

## Consuming the core from a sibling repo

The API boundary is header-only, but the implementation is not: `linalg.cpp`
uses Eigen (confined to that one translation unit), and `lm_core.cpp` and
`lin_hyp.cpp` build on it. So a consumer:

1. adds `src/include` to its include path → `#include "linalg.hpp"`,
   `#include "lm_core.hpp"`, `#include "lin_hyp.hpp"`;
2. compiles `src/linalg.cpp`, `src/lm_core.cpp`, and `src/lin_hyp.cpp` into its own build;
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
| `independent_columns` | `(const Mat& A, double tol = 1e-7) -> ColumnSelection` | greedy left-to-right independent column subset (modified Gram–Schmidt, two passes, no Eigen); the earlier column of a dependent set wins, as in R's `dqrdc2` |
| `pinv` | `(const Mat& A, double tol = 1e-12) -> Pinv` | Moore–Penrose pseudo-inverse + rank |
| `inv_spd` | `(const Mat& A, bool* ok = nullptr) -> Mat` | inverse of an SPD matrix (e.g. `(XᵀX)⁻¹`) |
| `sandwich` | `(const Mat& L, const Mat& A) -> Mat` | `L · A · Lᵀ` — the covariance sandwich |
| `matvec` | `(const Mat& A, const vector<double>& x) -> vector<double>` | matrix–vector product |

`Solution { vector<double> x; int rank; bool ok; }` carries the numerical rank so
callers can detect and report rank deficiency.
`ColumnSelection { vector<int> keep; int rank; }` lists the independent columns in
ascending order; the columns it leaves out are the aliased terms a rank-deficient
fit reports as NULL (#36). A column is kept when its residual norm after
orthogonalization exceeds `tol` times its own norm; zero and NaN columns are dropped.

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
| `LmResult` | struct | `ok`/`error`; `n`, `k`, `df_residual`, `rank`, `n_clusters`; per-term `terms`/`beta`/`std_error`/`t_statistic`/`p_value`; model `r_squared`, `adj_r_squared`, `f_statistic`, `f_p_value`, `sigma`, `loglik`; `cov` (k×k row-major) |
| `parse_vcov` | `(const string&, Vcov&) -> bool` | case-insensitive: `const`/`none`/`ols`/…, `hc0`–`hc3`, `cr0`/`cr1`/`cluster` (→ `kCR1`) |
| `vcov_name` | `(Vcov) -> const char*` | canonical label echoed back in `LmResult.vcov` |

Conventions worth knowing before consuming `LmResult`:

- **Intercept** is prepended by `fit_lm` when `opts.intercept` — pass `X_pred`
  *without* a constant column. `terms` are labelled `(Intercept)`, `x1`, `x2`, …
- **Covariance** drives only `std_error`/`t_statistic`/`p_value`; `beta` is
  invariant across `vcov`. HC* use the sandwich with leverage-weighted meat;
  CR0/CR1 use per-cluster score sums (CR1 with the
  `[G/(G−1)]·[(N−1)/(N−k)]` finite-sample factor).
- **Inference df.** Classical/HC use `t(n−rank)`; cluster-robust uses
  `t(G−1)`, where `G = n_clusters`. `df_residual` is always reported as
  `n−rank` regardless — it is a property of the fit, not the SE estimator.
- **Rank deficiency is not an error.** A dependent column is dropped the way
  R's `dqrdc2` drops it (the earlier column of a dependent set wins), the fit
  runs on the kept columns, and the dropped coefficient is NaN in `beta`,
  `std_error`, `t_statistic`, `p_value` and the matching row and column of
  `cov`. `terms` still names it. Every df formula counts estimated parameters,
  so `rank`: σ², adjusted R², the HC1 and CR1 finite-sample factors, the t
  reference, and the F numerator. A full-rank fit has `rank == k`, so those
  formulas are unchanged there.
- **Failure is a value, not an exception.** `ok = false` with a populated
  `error` on `n ≤ k`, a singular/collinear design, or (for CR*) a missing
  cluster vector or fewer than two clusters. Callers branch on `ok`.
- **The full covariance is exposed.** `cov` is the k×k matrix for the chosen
  `vcov`, row-major (`cov[i*k + j]`), aligned to `terms`; its diagonal is
  `std_error²`.
  `loglik` is the Gaussian MLE value `−n/2·(ln 2π + ln(RSS/n) + 1)`, which is
  `+inf` at RSS = 0.

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

## `lin_hyp.hpp` — general linear hypothesis

Tests `L·β = 0` for q constraints at once, given a fit's coefficients and their
covariance. Header-only API over `linalg`; it does **not** depend on
`lm_core.hpp`, so the same function serves any fit.

```cpp
LinHypResult lin_hyp(const linalg::Mat &L,          // q × k, one row per constraint
                     const std::vector<double> &beta,
                     const linalg::Mat &V,          // k × k covariance of beta
                     double sigma2, double df);
```

| Symbol | Kind | Notes |
| --- | --- | --- |
| `LinHypResult` | struct | `ok`/`error`; `estimable`; per-row `estimate`/`std_error`/`t_statistic`/`p_value`; joint `ss`, `df_num`, `df_den`, `f_statistic`, `f_p_value`, `chisq`, `chisq_p_value` |

Conventions worth knowing:

- **Three outcomes, not two.** `ok = false` is misuse (a dimension mismatch, a
  non-positive `df`). `ok = true` with `estimable = false` means the hypothesis
  cannot be evaluated, and callers should report NULL. The rest of the struct —
  `estimate`, `ss`, `df_num`, `chisq` and the others — is filled in only when
  both `ok` and `estimable` are true.
- **`df_num` is the rank of `L·V·Lᵀ`**, not `L`'s row count, so redundant
  constraints count once. One `pinv` call yields both the pseudo-inverse and
  that rank.
- **C is scaled to unit diagonal before `pinv`, and symmetrized.** `pinv`
  decides the numerical rank against a tolerance relative to the largest
  singular value, and C's diagonal holds the constraints' variances, so
  without the scaling two constraints whose variances are more than about
  1e12 apart read as redundant — a change of units is enough to reach that.
  With `D = diag(1/√C_ii)`, `C⁻ = D·(D·C·D)⁻·D`, which leaves the Wald
  statistic unchanged when C has full rank. Any kernel that inverts a
  covariance with `pinv` needs the same step.
- **A supplied covariance that is not one gives `estimable = false`.** A
  strictly negative `C_ii`, or a negative Wald statistic, means the input is
  not a covariance. That is a data-shaped failure, so it is a NULL result
  rather than an error. `C_ii = 0` stays estimable: a zero row of `L`
  produces one.
- **Aliased coefficients do not spread.** A rank-deficient fit carries NaN in
  the rows and columns of its dropped coefficients. A hypothesis putting
  non-zero weight there is not estimable; one that avoids them is evaluated
  normally, with those rows and columns zeroed in a working copy first, because
  IEEE makes `0 · NaN = NaN`.
- **`ss` = `chisq` · `sigma2`** is a sum of squares only under the classical
  covariance; with a robust `V` the F and chi-square are a Wald test.

## Testing & validation

All four layers are unit-tested directly, standalone (no DuckDB), against
golden values — `test/cpp/test_linalg.cpp`, `test/cpp/test_lm_fit.cpp`
(statsmodels oracle), `test/cpp/test_lin_hyp.cpp` (statsmodels oracle), and
`test/cpp/test_optimize.cpp` (classic optimization test functions). Build and
run them with `scripts/run-cpp-tests.sh` (or `make test_cpp`).
