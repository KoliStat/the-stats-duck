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
[`src/include/glm_core.hpp`](../src/include/glm_core.hpp),
[`src/include/lin_hyp.hpp`](../src/include/lin_hyp.hpp),
[`src/include/optimize.hpp`](../src/include/optimize.hpp),
[`src/include/distributions.hpp`](../src/include/distributions.hpp), and
[`src/include/emmeans.hpp`](../src/include/emmeans.hpp).

## Consuming the core from a sibling repo

The API boundary is header-only, but the implementation is not: `linalg.cpp`
uses Eigen (confined to that one translation unit), and `lm_core.cpp`,
`glm_core.cpp`, `glm_fit_core.cpp`, `lin_hyp.cpp` and `emmeans.cpp` build on
it. So a consumer:

1. adds `src/include` to its include path → `#include "linalg.hpp"`,
   `#include "lm_core.hpp"`, `#include "glm_core.hpp"`,
   `#include "lin_hyp.hpp"`, `#include "emmeans.hpp"`;
2. compiles `src/linalg.cpp`, `src/lm_core.cpp`, `src/glm_core.cpp`,
   `src/glm_fit_core.cpp`, `src/lin_hyp.cpp`, and `src/emmeans.cpp` into its
   own build;
3. makes the Eigen headers (`third_party/eigen`) available **when compiling
   `linalg.cpp`** — that is the only Eigen dependency, and it never leaks past
   that TU.

`glm_core.hpp` is one header over two sources, `glm_core.cpp` and
`glm_fit_core.cpp`: the family/link scalar functions and the closed-form
statistics in the first, the IRLS loop itself in the second. The header
stays whole because a caller wants one include for a GLM, and the two
halves are one contract. It includes `lm_core.hpp` for one symbol only, the
`Vcov` enum, so a GLM's standard errors are named the way an OLS fit's are;
it never calls `fit_lm`, `parse_vcov`, or `vcov_name`.

`emmeans.cpp` additionally pulls in `distributions.hpp`, `estimability.hpp`
and `p_adjust.hpp`, all three header-only, so nothing beyond `emmeans.cpp`
itself needs adding to a build that already compiles step 2 above.

`optimize.hpp` is simpler still: it is **fully header-only** (Eigen-free even in
implementation) — include it and go, nothing extra to compile.

`cluster_ids.hpp` is header-only too, and smaller again: one function,
`densify_clusters`, that turns arbitrary `VARCHAR` cluster keys into dense
0-based ids by sorting rather than hashing — the sort is what keeps the
result independent of hash-table order, and what keeps a `std::string`-keyed
hash map out of the wasm build (see the portability discipline in
[`fitter_conventions.md`](fitter_conventions.md)). It sits above the kernel
boundary: the SQL-facing `lm_fit_function.cpp` and `glm_fit_function.cpp`
both call it to turn a caller's cluster column into the dense ids
`fit_lm`/`fit_glm` require for CR0/CR1, so it needs adding only by a
consumer that wants the same convention for its own cluster keys.

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

## `glm_core.hpp` — generalized linear models by IRLS

Fits a GLM — gaussian, binomial, poisson, or gamma — by iteratively
reweighted least squares (IRLS), on the same `linalg` kernel `lm_core`
uses.

```cpp
GlmResult fit_glm(const std::vector<double> &y,       // response, length n
                  const linalg::Mat &X_pred,          // n × p predictors (NO intercept column)
                  const GlmOptions &opts,              // family, link, vcov, intercept, max_iter, tol
                  const std::vector<double> *weights = nullptr,  // prior weights, length n
                  const std::vector<double> *offset = nullptr,   // length n, not estimated
                  const std::vector<int> *cluster_ids = nullptr); // CR* only: dense 0-based, length n
```

| Symbol | Kind | Notes |
| --- | --- | --- |
| `Family` | enum | `kGaussian`, `kBinomial`, `kPoisson`, `kGamma` |
| `Link` | enum | `kIdentity`, `kLogit`, `kProbit`, `kCLogLog`, `kLog`, `kInverse`, `kSqrt` |
| `GlmOptions` | struct | `family` (default `kGaussian`), `link` (default `kIdentity`, overwritten from `canonical_link` unless the caller gives one), `vcov` (default `kConst`), `intercept` (default `true`), `max_iter` (25), `tol` (1e-8) |
| `GlmResult` | struct | `ok`/`error`; `n`, `k`, `rank`, `n_clusters`, `df_residual`; per-term `terms`/`beta`/`std_error`/`z_statistic`/`p_value`; `cov` (k×k row-major); `deviance`, `null_deviance`, `dispersion`, `loglik`, `aic`; `iterations`, `converged`, `separation` |
| `parse_family`, `parse_link` | `(const string&, Family/Link&) -> bool` | case-insensitive, in the shape of `parse_vcov`; return `false` instead of throwing, so the caller picks the exception |
| `family_name`, `link_name` | `(Family/Link) -> const char*` | canonical labels, echoed back by `glm_fit` |
| `canonical_link` | `(Family) -> Link` | the link used when a caller gives a family and no link |
| `link_allowed` | `(Family, Link) -> bool` | whether a link makes sense for a family — a `logit` link on `poisson` is a bind error, not a fit |
| `link_eta`, `link_inv`, `link_deriv`, `variance` | scalar functions over one observation | `g`, `g⁻¹`, `dmu/deta`, `V(mu)` — pure, so each is tested without a fitting loop: a round trip `link_inv(link_eta(mu)) == mu` per link, and `link_deriv` against a central difference |
| `deviance`, `glm_loglik`, `glm_aic`, `pearson_dispersion` | closed-form functions over a whole fit's `y`, `mu`, `w` | computable from the fitted values alone; no loop needed to test these either |

Conventions worth knowing before consuming `GlmResult`:

- **The binomial response is a proportion, not a count.** `y` must lie in
  `[0, 1]`; `weights` is the trial count, so 7 of 10 is `y = 0.7`,
  `weights = 10`. A plain 0/1 logistic fit is the case `weights = 1`
  everywhere.
- **The IRLS loop.** Starting values follow R's `glm.fit`:
  `(w·y + 0.5)/(w + 1)` for binomial, `y + 0.1` for poisson, `y` for
  gaussian and gamma. Then, up to `opts.max_iter` times:

  ```
  z   = eta − offset + (y − mu) / link_deriv(eta)      working response
  W   = w_prior · link_deriv(eta)² / variance(mu)      working weights
  solve (Xᵀ W X)·beta = Xᵀ W z over the kept columns
  eta = X·beta + offset
  mu  = link_inv(eta)
  ```

  stopping when `|dev − dev_prev| / (|dev| + 0.1) < opts.tol`. A step whose
  coefficients or deviance are not finite, whose fitted mean has left the
  family's support on a positive-weight row, or whose deviance increased,
  is halved, at most 25 times — except on the first pass, which has no
  previous coefficient vector to halve toward and almost always raises the
  deviance anyway, because the starting values sit at or near the
  saturated model, where the deviance is already near its minimum. Halving
  the first pass collapses every fixture in this project to all-zero
  coefficients, the gaussian identity one-step solve included, so it is
  not done. `iterations` counts outer passes, never the halvings inside
  one.
- **The rank-deficient column set is computed once and held fixed for the
  whole loop.** This deviates from R, which re-pivots every iteration.
  Holding it fixed is safe, not just convenient: for strictly positive
  weights, `rank(W^½X) = rank(X)`, because `W^½` is invertible, so the set
  cannot change between iterations. Fixing it also keeps the fit
  deterministic, where re-pivoting could oscillate. The set comes from
  `X_pred` restricted to the rows of positive prior weight, because a
  zero-weight row contributes nothing to the solve and can change the
  rank.
- **The dispersion rule.** Fixed at 1 for binomial and poisson. For
  gaussian and gamma it is the Pearson statistic,
  `Σ w·(y − mu)² / variance(mu) / (n − rank)`, which reduces to the usual
  σ² for gaussian. The classical covariance is `dispersion · (XᵀWX)⁻¹`.
- **The robust and cluster-robust estimators use score residuals, not
  response residuals — the one real difference from `lm_core`.** The score
  contribution of observation i is
  `x_i · w_i · (y_i − mu_i) · link_deriv(eta_i) / variance(mu_i)`, and
  `HC0` through `HC3` and `CR0`/`CR1` sandwich or sum it the same way
  `lm_core` sandwiches or sums its own response residuals. At gaussian
  with an identity link, `link_deriv` is 1 and `variance` is 1, so the
  score reduces to `lm_core`'s `x_i · w_i · (y_i − mu_i)`, and all six
  estimators match `lm_core`'s bit for bit — `test_glm_fit.cpp` asserts
  that reduction at a band of 0.
- **`link_inv`'s clamp.** `mu` is clamped to `[kMuEps, 1 − kMuEps]` for the
  three probability links. `log`, `inverse`, and `sqrt` clamp to at least
  `kMuEps`: `inverse` floors away from 0 on whichever side the raw value
  already falls, since `1/eta` can be negative, where `log` and `sqrt`
  never need that, because `e^eta` and `eta²` can never go negative in the
  first place. `identity` is never clamped, because a gaussian mean has no
  boundary to clamp against. `kMuEps` is `1e-10`. Without the clamp, an
  extreme `eta` can push `mu` to the exact boundary — the logit inverse
  underflows to exactly 0 around `eta ≈ −709.78` — and the deviance's
  `log(mu)` then returns infinity for a fit that should report a very
  small probability instead. The threshold is where `e^-eta` overflows,
  because `link_inv` computes `1/(1 + e^-eta)`; the algebraically equal
  `e^eta/(1 + e^eta)` would hold out to about `−745.13` instead.
- **A separate support test catches what the clamp cannot.** `link_inv`
  clamps the probability and strictly-positive links, but `identity` is
  never clamped, and `identity` is allowed for poisson while `log` is
  allowed for binomial. Without an explicit validity test on `mu` at every
  step, a poisson fit with an identity link can converge on a negative
  fitted count, and a binomial fit with a log link can converge on a
  probability above 1 — the deviance's `y·ln(y/mu)` term is skipped at
  `y = 0` and returns a finite number instead of NaN, so nothing else
  stops it. The test looks only at rows of positive prior weight, the same
  rule the normal matrix and the aliased set follow.
- **Failure is a value, not an exception, in the shape of `fit_lm`.**
  `ok = false` with a populated `error` covers the data-shaped failures:
  too few rows, a domain violation (`y` outside a family's support, a
  negative or non-finite weight), fewer than two clusters under CR0/CR1,
  or a design with no independent column. `converged = false` is a
  different outcome from `ok = false`: the fit still succeeded, and
  reports its best iterate in full, the convention
  `docs/fitter_conventions.md` asks of every iterative fitter.

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
- **Two covariance checks, not a positive-definiteness test.** A strictly
  negative `C_ii`, or a Wald statistic that is not `>= 0`, proves the input is
  not a covariance, and either one gives `estimable = false`. That is a
  data-shaped failure, so it is a NULL result rather than an error. The pair
  is not exhaustive: an indefinite `C` that passes both is evaluated normally,
  because catching every such matrix needs an eigendecomposition. `C_ii = 0`
  stays estimable: a zero row of `L` produces one.
- **Aliased coefficients do not spread.** A rank-deficient fit carries NaN in
  the rows and columns of its dropped coefficients. A hypothesis putting
  non-zero weight there is not estimable; one that avoids them is evaluated
  normally, with those rows and columns zeroed in a working copy first, because
  IEEE makes `0 · NaN = NaN`.
- **`ss` = `chisq` · `sigma2`** is a sum of squares only under the classical
  covariance; with a robust `V` the F and chi-square are a Wald test.

## `distributions.hpp` — probability distributions

Header-only PDF/CDF/quantile functions, in namespace `stats_duck` (with the
underscore). `linalg` and the fitters above are `statsduck`, without one, so
a call across that boundary needs qualifying, as `lin_hyp.cpp` already does.
This is where every fitter's p-values and confidence intervals come from,
per `docs/fitter_conventions.md`: there is never a second CDF implementation.
Most of the header backs the `d*`/`p*`/`q*` SQL scalars in the README's
distribution-functions table one for one. The two functions below are the
newest, and worth a closer look here, because `emmeans`'s `tukey` adjustment
depends on them and they carry their own accuracy limit.

```cpp
double StudentizedRangeCDF(double q, double k, double df);
double StudentizedRangeQuantile(double p, double k, double df);
```

`Q = (max − min) / S` for `k` independent standard normals and an
independent `S` with `df` degrees of freedom. This is the distribution
behind Tukey's HSD, and the one behind `emmeans`'s Tukey adjustment.
Exposed as the SQL scalars `ptukey` and `qtukey`.

- **`k = 2` is the t distribution exactly:**
  `ptukey(q, 2, df) = 2 · pt(q / sqrt(2), df) − 1`. This is the fastest way to
  convince yourself the two-integral quadrature is right, because it reduces
  the whole thing to a distribution this header already validates elsewhere.
- **Both return NaN instead of throwing when `k < 2`, `k` is infinite,
  `df <= 0`, or any argument is NaN**, unlike the rest of this header,
  which throws `std::invalid_argument`. An infinite `k` is rejected on
  purpose rather than left to run: without the check, the inner integral's
  `[Phi(z) - Phi(z-w)]^(k-1)` factor underflows to zero at every node for a
  large enough `k`, and `k` (infinity) times that zero is `NaN` under IEEE
  arithmetic, not `0`. That happens to be the right answer here, but only
  by accident, so `NormalRangeCDF` checks `std::isinf(k)` explicitly rather
  than relying on it. They are called per row from `emmeans`
  and from the `ptukey`/`qtukey` SQL scalars, where a NaN travels with the
  row and an exception would abort the query. `StudentizedRangeQuantile`
  also returns NaN for a `p` outside `[0, 1]`; `StudentizedRangeCDF` places
  no such restriction on `q` — `q <= 0` is a valid input, and the answer is
  `0.0`. An infinite `df` is not out of domain either: it is the
  normal-range limit the outer integral converges to, so
  `StudentizedRangeCDF(q, k, INFINITY)` is a value, not a NaN.
- **A `df` that is not an integer and is below 4 is accurate to about 1e-5
  rather than 1e-12.** Every integer `df`, and every real `df` at or above 4,
  measures at about 1e-12 against `scipy.stats.studentized_range`. Below
  `df = 4` the error grows as `df` falls: roughly 6e-10 at `df = 3.5`, 1e-7 at
  2.5, 2e-5 at 1.5, and 1e-2 at 0.5, all worst at `k = 2`. The reason is that
  the density of `S` carries a factor `s^(df−1)` that is a polynomial only
  for an integer `df`. For any other `df` it has a branch point at `s = 0`,
  and a fixed-node quadrature converges on a branch point only slowly.
  `ptukey` and `qtukey`'s own registered SQL descriptions carry this
  sentence too, but a reader of this page may never see those, so it is
  stated here as well. `notes/engineering/2026-10-studentized-range-quadrature.md`
  has the full measurements and what was tried instead.
- **Fixed node counts, chosen by measurement, not adaptive ones.** Both
  integrals are panelled Gauss-Legendre: 48 nodes on 2 panels inner, 28 on 2
  outer. The quantile is a fixed 100-step bisection. Both counts were chosen
  by testing a range of node and panel counts against the whole scipy
  fixture grid, and stopping where the error stopped improving. Identical
  inputs give the same answer on every platform, short of whatever the
  platform's own `erf`/`exp`/`pow` returns in its last bit. The quadrature
  itself adds no platform dependence of its own.

## `emmeans.hpp` — estimated marginal means

Reports a grid of marginal means θ = L·β against a fit's coefficients and
covariance, their standard errors and confidence limits, every pairwise
difference, and a multiplicity adjustment for that family. Header-only API
over `linalg`; like `lin_hyp.hpp`, it does **not** depend on `lm_core.hpp`,
so the same function serves any fit.

```cpp
EmmeansResult emmeans(const linalg::Mat &L,          // q × k, one row per mean
                      const std::vector<double> &beta,
                      const linalg::Mat &V,          // k × k covariance of beta
                      double df, double alpha, EmmAdjust adjust);
```

| Symbol | Kind | Notes |
| --- | --- | --- |
| `EmmAdjust` | enum | `TUKEY`, `BONFERRONI`, `HOLM`, `HOCHBERG`, `BH`, `BY`, `NONE` |
| `EmmMean` | struct | `estimable`; `estimate`, `std_error`, `t_statistic`, `p_value`, `lower_cl`, `upper_cl` |
| `EmmDiff` | struct | `i`, `j` (0-based row indices into `L`, `i < j`); `estimable`; `estimate`, `std_error`, `t_statistic`, `p_value`, `p_adjusted`, `lower_cl`, `upper_cl` |
| `EmmeansResult` | struct | `ok`/`error`; `means`; `diffs`; `df_den`, `alpha`, `adjust` (echoed); `k_compare` |
| `parse_emm_adjust` | `(const string&, EmmAdjust&) -> bool` | case-sensitive; `tukey`, `bonferroni`, `holm`, `hochberg`, `BH`, `fdr` (alias for `BH`), `BY`, `none` |
| `emm_adjust_name` | `(EmmAdjust) -> const char*` | canonical label echoed back; `fdr` returns `"BH"` |

Conventions worth knowing:

- **`ok = false` is misuse; callers raise.** `error` names the problem, and
  `means`/`diffs` come back empty. The checks run in this order, and the
  first failure is the one reported: `L` with no rows; a row width that is
  not k; a `V` that is not k × k; a `beta` of length 0; and a `df` that is
  not positive and finite. Four of those five
  raise in close to the same words `lin_hyp` uses for the same shapes on
  `L`, `beta` and `V`. `lin_hyp`'s own `df` check only rejects a
  non-positive value, not an infinite one, so this one is stricter. An
  `alpha` outside `(0, 1)` is the sixth kernel-level check, and has no
  `lin_hyp` counterpart to compare against, since `lin_hyp` takes no
  `alpha`. The SQL layer adds one misuse case of its own, ahead of all six:
  an unrecognized `adjust` name, rejected before `emmeans` is ever called.
- **Nothing is inverted.** There is no `pinv`, no equilibration and no rank
  tolerance here, because `emmeans` forms no quadratic form to invert
  through. `lin_hyp` needs all three for its joint test; `emmeans` has no
  joint test to form. θ and C = L·V·Lᵀ are each formed once, through
  `linalg::matvec` and `linalg::sandwich`, and every reported number is an
  entry of one or the other.
- **Estimability is per element, not per result.** Unlike `LinHypResult`,
  there is no top-level `estimable` flag. A rank-deficient fit's aliased rows
  and columns are zeroed in a working copy (the rule `estimability.hpp`
  shares with `lin_hyp`), and the test for which grid rows may use the result
  runs per row, so one unusable row of `L` does not discard the rest of the
  grid.
- **A mean's estimate or variance can go non-finite independent of
  aliasing.** `aliased_columns` only inspects `V`'s own diagonal, so a NaN
  anywhere else still reaches `theta[i]` or `C(i,i)`: `linalg::matvec` and
  `linalg::sandwich` are dense products that multiply through every term
  unconditionally, and `0 · NaN = NaN` under IEEE, not `0`. A NaN confined
  to one row of `L` only poisons that row's own `theta[i]`, because each
  row's dot product is independent. A NaN anywhere in `beta`, or in an
  **off-diagonal** entry of `V`, reaches every row's dot product instead.
  That typically poisons every mean at once and drops `k_compare` to `0`.
  A NaN on `V`'s own diagonal is caught by the aliasing check above and
  stays local to the rows that use that column. The same NaN one cell
  over, off the diagonal, is not caught at all.
- **The negative-variance tolerance is relative, and its scale differs by
  case.** Both a mean's variance (`C_ii`) and a difference's variance
  (`C_ii + C_jj − 2·C_ij`) are sums that can land slightly below zero from
  rounding when the true value is zero. The tolerance is `1e-12` relative to
  `max_j C_jj` for a mean, and relative to `C_ii + C_jj` for a difference.
  A tolerance needs a scale to be relative to, and that scale is whatever
  the cancelling terms are, which differs between the two cases. A variance
  inside the tolerance keeps its estimate, with a zero `std_error` and a
  NaN `t_statistic`/`p_value`. Past the tolerance, the element is not
  estimable.
- **`tukey` needs two estimable means at minimum.** Below `k_compare = 2` the
  studentized range is undefined, so neither `StudentizedRangeCDF` nor
  `StudentizedRangeQuantile` is called at all. At exactly two, `p_adjusted`
  comes from `StudentizedRangeCDF` and `p_value` from `StudentTCDF`. Those
  are two different code paths that agree mathematically at k = 2, not
  the same computation, so they can differ by about 1e-13 in practice.
  `p_adjusted` can land very slightly below `p_value` instead of at or
  above it; that is the quadrature's own accuracy, not a real crossing.
- **`bonferroni`'s family size is a pair count, not `k_compare`.** Its
  multiplier `m` is the number of estimable pairs that have a finite
  p-value. A pair between two identical grid rows is excluded from that
  count: it is estimable, with zero variance and a real estimate of 0, but
  it has no p-value to rank among the others. `k_compare` is always the
  estimable-mean count, so the two agree only when every pair is both
  estimable and has a p-value. For example, three rows
  `[[1,0,0],[1,0,0],[0,1,0]]` give `k_compare = 3`, since all three means
  are estimable, but `m = 2`, since the repeated pair has no p-value. The
  other two pairs' `bonferroni` p-values are then `min(1, 2 · p_value)`, not
  `min(1, 3 · p_value)`.
- **The symmetric part of `C` is what is used**, matching `lin_hyp`: where
  `V` is asymmetric, `C_ij` is read as `(C_ij + C_ji)/2`, because only the
  symmetric part has a meaning in a variance.

Two more headers back the two sections above without a section of their
own, because each is a handful of free functions with no state.
**`estimability.hpp`** is the aliased-coefficient rule `lin_hyp.hpp` and
`emmeans.hpp` share: `aliased_columns`, `zero_aliased`, and
`row_touches_aliased`. **`p_adjust.hpp`** is the six multiple-testing
methods behind `adjust_p` and behind six of `emmeans`'s own seven
`EmmAdjust` values — every one but `TUKEY`, which needs the studentized
range rather than a plain list of p-values. `emmeans.hpp`'s own
`parse_emm_adjust`/`emm_adjust_name` follow the shape of
`p_adjust.hpp`'s `parse_adjust`/`adjust_name`. Both headers are
DuckDB-free and Eigen-free, like `linalg.hpp`.

## Testing & validation

All ten layers above are unit-tested directly, standalone (no DuckDB),
against golden values — `test/cpp/test_linalg.cpp`, `test/cpp/test_lm_fit.cpp`
(statsmodels oracle), `test/cpp/test_cluster_ids.cpp` (the sort-based
densifying rule `lm_fit` and `glm_fit` share), `test/cpp/test_lin_hyp.cpp`
(statsmodels oracle), `test/cpp/test_optimize.cpp` (classic optimization
test functions), `test/cpp/test_estimability.cpp` and
`test/cpp/test_p_adjust.cpp` (the aliased-coefficient rule and the
multiple-testing methods, each shared with `lin_hyp`), `test/cpp/test_tukey.cpp`
(scipy's `studentized_range`, plus the `k = 2` identity against `pt` and the
large-`df` limit against `NormalCDF`), `test/cpp/test_emmeans.cpp`
(statsmodels' `t_test`), `test/cpp/test_glm_core.cpp` (the link round trips
and derivatives described above, which need no oracle), and
`test/cpp/test_glm_fit.cpp` (statsmodels and R oracles together, since
neither covers the whole surface, plus the `lm_fit` identity at gaussian
with an identity link, which needs no oracle either). Build and run them
with `scripts/run-cpp-tests.sh` (or `make test_cpp`).
