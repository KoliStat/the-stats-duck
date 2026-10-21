# glm_fit design (#22)

Date: 2026-10-20. Status: the binding specification for #22.

`glm_fit` fits a generalized linear model by iteratively reweighted least
squares. It is an aggregate, so `GROUP BY` gives one fit per group, and it
reports coefficients with their covariance so that `lin_hyp` (#20) and
`emmeans` (#21) work against a GLM without either of them changing. That is
the point of this issue: both post-fit helpers are deliberately stateless so
they serve any fit, and until now there has been only one fitter to serve.

This document is the authority for the implementation plan. Where the plan
and this document disagree, this document is right, and the plan should be
corrected rather than reinterpreted.

## Dependency

The kernel consumes `src/include/estimability.hpp`, which was added by #21
and is not on `v0.9` until PR #67 merges. Implementation therefore either
waits for that merge or stacks this branch on `feat/21-emmeans`. Writing
this specification does not depend on it.

## Scope

In scope: four families with their canonical and common alternate links,
prior weights, an offset, classical and robust covariance, rank-deficient
designs, a dispersion estimate where the family has one, and a separation
flag for binomial fits.

| Family | Canonical link | Also supported | Dispersion |
| --- | --- | --- | --- |
| `gaussian` | `identity` | `log`, `inverse` | estimated |
| `binomial` | `logit` | `probit`, `cloglog` | fixed at 1 |
| `poisson` | `log` | `identity`, `sqrt` | fixed at 1 |
| `gamma` | `inverse` | `log`, `identity` | estimated |

Out of scope, deliberately:

- **Negative binomial.** Its dispersion is estimated by maximum likelihood,
  which needs digamma and trigamma. `distributions.hpp` has neither, so they
  would be new numerics with their own generator, goldens and tolerance
  justification — the shape of work that was the largest single task in #21.
  It gets its own issue.
- **Weighted least squares in `lm_core`.** #19's deferred increment B stays
  deferred. `glm_fit` has its own weighted solve inside the IRLS loop and
  does not call `lm_core::fit_lm`.
- **Exact separation detection.** Deciding separation exactly is a linear
  programming feasibility problem and this repository has no LP solver. The
  heuristic below is what ships.
- **Quasi families, user-supplied links, and `etastart`/`mustart` overrides.**
  No caller has asked.

## The SQL surface

```
glm_fit(y, x [, family [, options]]) -> STRUCT
```

An aggregate. One row is one observation, as `docs/fitter_conventions.md`
requires.

| Argument | Type | Notes |
| --- | --- | --- |
| `y` | `DOUBLE` | response |
| `x` | `LIST(DOUBLE)` | predictor values, without a constant column |
| `family` | `VARCHAR` | default `gaussian` |
| `options` | `STRUCT` | all remaining options, by name |

`family` defaults to `gaussian`, which makes `glm_fit(y, [x1, x2])` the same
fit as `lm_fit(y, [x1, x2])`. That equivalence is deliberate and the tests
pin it.

### The options struct

`glm_fit` has seven optional inputs where `lm_fit` has three, and two of
them — `weights` and `offset` — are both per-row DOUBLEs, so they can never
be told apart positionally. The `VARCHAR`-versus-`BOOLEAN` disambiguation
`lm_fit` uses for one slot does not extend to six. So every option is a
named field of one trailing struct.

| Field | Type | Per-row | Default | Notes |
| --- | --- | --- | --- | --- |
| `link` | `VARCHAR` | no | the family's canonical link | |
| `weights` | `DOUBLE` | yes | 1 | prior weights; binomial trial counts |
| `offset` | `DOUBLE` | yes | 0 | added to the linear predictor with a fixed coefficient of 1 |
| `vcov` | `VARCHAR` | no | `const` | the names `lm_core::parse_vcov` accepts |
| `cluster` | `VARCHAR` | yes | none | required by `cr0` and `cr1`, rejected otherwise |
| `intercept` | `BOOLEAN` | no | `true` | prepends a constant column |

**The binomial response convention, stated because it is otherwise
ambiguous.** `y` is the **proportion** of successes and must lie in [0, 1];
`weights` is the number of trials. Plain 0/1 data is then the default case,
with every weight 1. Grouped data passes the proportion as `y` and the trial
count as `weights`, so 7 successes out of 10 is `y = 0.7` with
`weights = 10`. This is R's convention for the proportion form. A count of
successes passed as `y` with `weights` as trials is **not** supported and
fails the [0, 1] domain check, which is the right outcome: silently treating
a 7 as a proportion would be worse.

Four rules about the struct, each of which the implementation must enforce:

- **Fields are looked up by name, never by position.** DuckDB's struct type
  is ordered, so `{'link': 'probit', 'vcov': 'HC1'}` and
  `{'vcov': 'HC1', 'link': 'probit'}` are different types carrying the same
  meaning. Both must work.
- **An unrecognized field name is a bind error** that lists the valid names.
  A silently ignored typo is the failure this rule exists to prevent: a
  caller writing `{'weight': n}` would otherwise get an unweighted fit that
  looks correct.
- **A constant-only field must be constant-foldable.** `link`, `vcov` and
  `intercept` are read once at bind time, so an expression varying per row
  is a bind error.
- **Omitting the struct entirely is the same as passing an empty one.**

Reusing `lm_core::parse_vcov` means `glm_fit` accepts exactly the spellings
`lm_fit` accepts, with the same case-insensitivity — `const`, `none`,
`nonrobust`, `ols`, `iid`, `hc0` through `hc3`, `cr0`, and `cr1` or
`cluster` — and echoes the same canonical labels back.

### Missing data

Listwise drop, matching `lm_fit`. A row leaves the fit when `y`, any element
of the design row, the weight, the offset or the cluster key is NULL, NaN or
infinite. A design row whose length differs from the first one seen fails
the whole group. A group with too few surviving rows is a failed fit, which
is a NULL result and not an error.

### The result STRUCT

Field order is contract and append-only. The type function carries the
comment `// Field order MUST match the writes in Finalize` with an index
comment per field.

| # | Field | Type | Notes |
| --- | --- | --- | --- |
| 0 | `coefficients` | `LIST<STRUCT(term VARCHAR, estimate, std_error, z_statistic, p_value DOUBLE)>` | positional terms, `(Intercept)` first |
| 1 | `n` | `BIGINT` | rows used after the listwise drop |
| 2 | `k` | `BIGINT` | design columns, intercept included |
| 3 | `df_residual` | `BIGINT` | `n − rank` |
| 4 | `deviance` | `DOUBLE` | |
| 5 | `null_deviance` | `DOUBLE` | the intercept-only model's deviance |
| 6 | `dispersion` | `DOUBLE` | 1 for binomial and Poisson |
| 7 | `loglik` | `DOUBLE` | |
| 8 | `aic` | `DOUBLE` | |
| 9 | `iterations` | `BIGINT` | IRLS iterations used |
| 10 | `converged` | `BOOLEAN` | |
| 11 | `separation` | `BOOLEAN` | binomial only; always false otherwise |
| 12 | `family` | `VARCHAR` | echoed canonically |
| 13 | `link` | `VARCHAR` | echoed canonically |
| 14 | `has_intercept` | `BOOLEAN` | |
| 15 | `vcov_type` | `VARCHAR` | the canonical estimator name |
| 16 | `n_clusters` | `BIGINT` | NULL unless CR0 or CR1 |
| 17 | `rank` | `BIGINT` | parameters actually estimated |
| 18 | `cov` | `LIST<DOUBLE>` | row-major k², in `coefficients` order |

Three deliberate differences from `lm_fit`:

**`z_statistic`, not `t_statistic`, for all four families.**
`docs/fitter_conventions.md` requires a z-referenced fitter to name the
field `z_statistic` and forbids a bare `statistic`. Binomial and Poisson
have fixed dispersion, so the normal reference is the right one. R and
statsmodels both switch to a t reference when dispersion is estimated, which
would mean the same field changing its meaning with the family. This spec
keeps one name and one reference across all four. The consequence is that
for Gaussian and gamma fits the p-values are very slightly smaller than R
reports, by the difference between a normal and a t reference at
`n − rank` degrees of freedom. **That difference is a documented deviation
from the oracle and must be stated beside the affected goldens and in the
user-facing documentation.** A caller wanting the t reference has
`df_residual` and can compute it.

**No `r_squared` and no `f_statistic`.** Neither generalizes to a GLM.
`deviance` and `null_deviance` are what a GLM reports, and a reader wanting
a pseudo-R² can form `1 − deviance/null_deviance` themselves rather than
have this function pick one of the several definitions.

**`converged = false` returns the best iterate rather than NULL.** The
conventions doc mandates this for iterative fitters, mirroring
`optimize::Result`, which reports its best point rather than failing. A
data-shaped failure — too few rows, a degenerate design, a domain
violation, fewer than two clusters for CR — is still a NULL group.

## The kernel

`src/include/glm_core.hpp` and `src/glm_core.cpp`, in namespace `statsduck`,
DuckDB-free and Eigen-free at the boundary like `lm_core`. It depends on
`linalg.hpp`, `distributions.hpp`, `estimability.hpp`, and on `lm_core.hpp`
for the `Vcov` enum and its parser **only** — never for fitting.

```cpp
enum class Family { kGaussian, kBinomial, kPoisson, kGamma };
enum class Link { kIdentity, kLogit, kProbit, kCLogLog, kLog, kInverse, kSqrt };

bool parse_family(const std::string &name, Family &out);
bool parse_link(const std::string &name, Link &out);
const char *family_name(Family f);
const char *link_name(Link l);
Link canonical_link(Family f);
bool link_allowed(Family f, Link l);

struct GlmOptions {
	Family family = Family::kGaussian;
	Link link = Link::kIdentity;   // set from canonical_link unless given
	Vcov vcov = Vcov::kConst;
	bool intercept = true;
	int max_iter = 25;
	double tol = 1e-8;
};

struct GlmResult {
	bool ok = false;
	std::string error;
	std::size_t n = 0, k = 0, rank = 0, n_clusters = 0;
	std::size_t df_residual = 0; // n − rank; integral, and the STRUCT's BIGINT
	std::vector<std::string> terms;
	std::vector<double> beta, std_error, z_statistic, p_value;
	std::vector<double> cov;        // row-major k²
	double deviance = 0.0, null_deviance = 0.0, dispersion = 1.0;
	double loglik = 0.0, aic = 0.0;
	int iterations = 0;
	bool converged = false, separation = false;
};

GlmResult fit_glm(const std::vector<double> &y, const linalg::Mat &X_pred,
                  const GlmOptions &opts,
                  const std::vector<double> *weights = nullptr,
                  const std::vector<double> *offset = nullptr,
                  const std::vector<int> *cluster_ids = nullptr);
```

`parse_family` and `parse_link` return `bool` rather than throwing, in the
shape of `lm_core::parse_vcov`, so the caller chooses the exception.
`link_allowed` exists because not every link is defensible for every family:
a `logit` link on a Poisson family is a bind error, not a fit.

### Family and link as free functions

```cpp
double link_eta(Link l, double mu);     // g(mu)
double link_inv(Link l, double eta);    // g⁻¹(eta)
double link_deriv(Link l, double eta);  // dmu/deta
double variance(Family f, double mu);   // V(mu)
```

Pure scalar functions, which is what makes them cheaply testable: a
round-trip `link_inv(link_eta(mu)) == mu` over a grid per link, and
`link_deriv` against a central difference of `link_inv`. Those two tests
need no oracle and catch a class of sign and algebra errors that otherwise
surfaces only as slow or failed convergence, which reads like a tuning
problem rather than a defect.

`link_inv` clamps its output away from the boundary of the family's support
— `mu` into `[eps, 1 − eps]` for binomial, `mu ≥ eps` for Poisson and gamma
— because an unclamped `logit` inverse underflows to exactly 0 at
`eta ≈ −745` and the deviance then returns infinity. The clamp value and the
reason go in the header.

### The IRLS loop

```
eta = X·beta + offset
mu  = link_inv(eta)
z   = eta − offset + (y − mu) / link_deriv(eta)      working response
W   = w_prior · link_deriv(eta)² / variance(mu)      working weights
solve (XᵀWX)·beta = XᵀWz
```

Starting values follow R's `glm.fit`, because they are what make the
awkward cases converge rather than an arbitrary choice:
`(w·y + 0.5)/(w + 1)` for binomial, `y + 0.1` for Poisson, `y` for Gaussian
and gamma.

Convergence is R's criterion, `|dev − dev_prev| / (|dev| + 0.1) < tol`, with
`tol` 1e-8 and at most 25 iterations. When a step produces a non-finite
deviance, a fitted value outside the family's support, or an increased
deviance, the step is halved, at most **25 times per iteration**, matching
R's inner loop. Step halving is what keeps a cloglog fit from diverging.
The first version of this spec said "a probit or cloglog fit"; a probit fit
was then measured never to halve at all, 0 times over 40,000 random fits, so
the claim holds for cloglog only.

**Halving starts at the second pass, not the first.** This paragraph said
otherwise in the first version of this spec, and that was wrong. The
starting `mu` is essentially the saturated model for Poisson and gamma, so
its deviance is at or near zero — measured at exactly 0 for gamma and
Gaussian on this issue's own fixtures — and **every** first step therefore
increases the deviance. Halving on an increase at pass 1 is worse than a
stall: `beta` starts as the zero vector, so halving averages toward zeros
rather than back toward the starting `mu`, which is a different model and
not a shorter step. Running the rule as first written collapsed every
fixture to all-zero coefficients while still reporting `ok = true`,
including the Gaussian identity fit, which is a one-step exact solve. R
reaches the same place from the same structure: its `coefold` is NULL on
the first iteration, which is why it stops there rather than halving.

**The support check is a halving trigger, not something the clamp
provides.** `link_inv` clamps the probability links and the strictly
positive ones, but the identity link is deliberately unclamped, and the
identity link is allowed for Poisson while the log link is allowed for
binomial. Without an explicit per-family validity test on `mu`, a Poisson
identity fit can report `converged = true` with a negative fitted mean and
a binomial log fit with a probability above 1, because the deviance's
`y·ln(y/mu)` term is skipped at `y = 0` and so returns a finite number
instead of NaN. A negative `variance(mu)` then makes a working weight
negative, which reaches the normal matrix and turns the solve into a
minimum-norm solution of an indefinite system. The validity test and a
negative-weight screen are both required.

**Exhausting the halvings does not by itself mean the fit failed.** When
the halved step's relative deviance change is below `tol`, the fit has
converged and reports `converged = true`. The IRLS direction is a strict
descent direction for the deviance while `XᵀWX` is positive definite, so a
step that 25 halvings cannot improve is one whose score is at rounding
level. Measured over 180,000 random fits across nine family and link pairs,
that branch reported convergence 14 times and a pattern search from each
reported `beta` could not improve any of them. Otherwise the fit stops with
`converged = false` and the best iterate so far.

`iterations` counts outer IRLS iterations, not halvings. A fit that
converges on its third pass reports 3 whether or not steps were halved
inside those passes.

The solve uses `linalg::cholesky_solve` on `XᵀWX`, falling back to
`linalg::svd_solve` when the Cholesky reports the matrix is not positive
definite, which mirrors what `lm_core` already does.

**The rank-deficient column set is determined once and held fixed.** This
deviates from R, which re-pivots every iteration. Holding the set fixed is
deterministic and cannot oscillate between iterations, and it is justified
rather than merely convenient: for strictly positive weights
`rank(W^½X) = rank(X)` because `W^½` is invertible, so the aliased set does
not depend on the iteration. The set is computed from X restricted to rows
whose **prior** weight is positive, because a zero-weight row contributes
nothing and can change the rank. Dropped columns follow #36's convention
exactly: the earlier column of a dependent set is kept, the later one is
aliased with NaN in its estimate, standard error, z, p and in the matching
row and column of `cov`, and `terms` still names it.

### Dispersion

Fixed at 1 for binomial and Poisson. For Gaussian and gamma it is the
Pearson statistic,

```
dispersion = Σ w·(y − mu)² / variance(mu) / (n − rank)
```

which for Gaussian reduces to the usual σ². Classical covariance is
`dispersion · (XᵀWX)⁻¹`.

### Robust covariance

The real difference from `lm_core`: a GLM's sandwich meat uses **score**
residuals, not response residuals. The score contribution for observation i
is

```
u_i = x_i · w_i · (y_i − mu_i) · link_deriv(eta_i) / variance(mu_i)
```

- **HC0** sandwiches `Σ u_i u_iᵀ` between `(XᵀWX)⁻¹`.
- **HC1** multiplies HC0 by `n / (n − rank)`.
- **HC2** and **HC3** divide each score by `(1 − h_i)^½` and `(1 − h_i)`
  respectively, where `h_i = w_i^½ · x_iᵀ (XᵀWX)⁻¹ x_i · w_i^½` is the hat
  diagonal of the **weighted** design.
- **CR0** sums the scores within each cluster before the outer product.
- **CR1** applies `[G/(G−1)]·[(N−1)/(N−rank)]`, the same factor `lm_core`
  uses.

At Gaussian with an identity link these reduce to `lm_core`'s formulas
exactly, because `link_deriv` is 1 and `variance` is 1, so `u_i` becomes
`x_i·w_i·(y_i − mu_i)`. That equivalence is a test, not a remark.

Inference degrees of freedom follow `lm_core`'s rule: cluster-robust
covariance references `G − 1` while `df_residual` stays `n − rank`, because
`df_residual` is a property of the fit rather than of the covariance
estimator.

### Log-likelihood and AIC

```
gaussian:  −n/2 · (ln 2π σ² + 1),  σ² = deviance / n
binomial:  Σ [ lchoose(w, w·y) + w·y·ln mu + w·(1 − y)·ln(1 − mu) ]
poisson:   Σ w · [ y·ln mu − mu − lgamma(y + 1) ]
gamma:     Σ with shape 1/dispersion, through lgamma
```

`AIC = −2·loglik + 2·(rank + d)`, where `d` is 1 for Gaussian and gamma,
whose dispersion is an estimated parameter, and 0 for binomial and Poisson.
The Gaussian form matches `lm_core::loglik`, which is the Gaussian MLE value
and is `+inf` at zero deviance; the AIC is then `−inf`, which passes through
as a value rather than becoming NULL, because ±Inf is representable under
the conventions doc.

`LogChoose` already exists in `distributions.hpp`. `std::lgamma` covers the
rest, so this section adds no new special functions.

### Separation

`separation = true` when the family is binomial, the fit did not converge
within `max_iter`, and at least one fitted value lies within 1e-10 of 0 or
1. For the other three families it is always false.

The flag is a heuristic and the documentation says so plainly. It fires on
perfect separation. It can also fire on severe near-separation, which it
cannot distinguish. It exists because perfect separation makes coefficients
diverge, and without it a caller sees an enormous estimate beside an
enormous standard error with no indication of why. That is how separation
reaches a published table.

**The flag also has a blind spot, and this paragraph is the correction of an
earlier version of this spec that did not admit it.** `separation` requires
a fit that did **not** converge. A response with no variation at all — every
`y` equal, which a `GROUP BY` over a rare outcome produces routinely — is a
textbook instance of complete separation, yet it typically **converges**
before `max_iter` and so is never flagged. The reason is that IRLS tests the
deviance rather than the coefficients, and the deviance flattens near the
support boundary long before the coefficients settle. Measured on this
issue's own fixtures: an all-zero binomial response converges at iteration
23 with an intercept standard error around 40311, and `separation` stays
false throughout.

So the promise is narrower than the paragraph above it first implied.
`separation = true` is evidence of trouble. `separation = false` is **not**
evidence of a sound fit, and a caller who wants that assurance must also
look at the magnitude of `std_error`. Both the header and the user-facing
documentation must say this, because a reader who sees only the first
paragraph would conclude that `converged = true` rules out an uninformative
fit, and it does not.

Fixing the blind spot properly means testing the coefficients for divergence
rather than the deviance for flatness, or detecting separation exactly, which
the Scope section rules out for want of a linear-programming solver. Neither
is in this issue.

### Domain violations

Data-shaped failures, so a NULL group and never an error:

- binomial with `y` outside [0, 1];
- Poisson with `y` negative;
- gamma with `y` not strictly positive;
- a negative or non-finite weight;
- a non-finite offset;
- fewer than two clusters under CR0 or CR1;
- `n ≤ rank`, or a design with no surviving rows.

Misuse is a bind error instead: an unknown family, link or `vcov` name, a
link the family does not allow, an unrecognized options field, a non-constant
value in a constant-only field, a cluster-robust `vcov` with no cluster
field, or a cluster field with a non-cluster `vcov`. The last two match what
`lm_fit` already rejects.

## Shared extraction

`DensifyClusters` at `src/lm_fit_function.cpp:351` turns VARCHAR cluster keys
into dense zero-based ids by sorting rather than hashing, which is what keeps
the result independent of hash-table order — the rule
`docs/fitter_conventions.md` states and that #54 exists to fix elsewhere. Both
aggregates need it, so it moves to `src/include/cluster_ids.hpp` and
`lm_fit_function.cpp` moves onto the shared copy. `lm_fit`'s existing cluster
tests cover that refactor.

## Validation

The oracle runs offline in a throwaway virtualenv and the goldens are
committed, per `docs/fitter_conventions.md`. Generator:
`test/cpp/gen_glm_fit_fixtures.py`, using statsmodels' `sm.GLM`, which covers
all four families, every link in the table, and `fit(cov_type=...)` for HC0
through HC3 and clustered — so one oracle serves the whole surface.

### The strongest check needs no oracle

**Gaussian with an identity link must equal `lm_fit` exactly** on the same
data: the coefficients, the full covariance matrix, `df_residual`, and
`dispersion` against `lm_fit`'s σ². Run it for every covariance estimator,
including the robust ones, where the score residuals reduce to response
residuals. `lm_fit` is already statsmodels-validated, so this ties the new
fitter to the old one rather than to a fresh set of constants. If the GLM
sandwich is wrong, this is where it shows.

### Four more oracle-free identities

1. **`null_deviance` equals the deviance of the intercept-only fit** of the
   same family and link, computed by actually fitting it. Catches a wrong
   null model, which a single golden number would not.
2. **Link round-trips and derivatives**, as described under the kernel.
3. **Grouped and ungrouped binomial agree.** The same data as 0/1 rows, and
   collapsed to proportions with `weights` set to the trial counts, must give
   identical coefficients and standard errors. This is the test for the whole
   `weights` path.
4. **Scaling every binomial trial count by a constant** leaves the
   coefficients unchanged and scales the deviance by that constant.

### Tolerance bands

The deviance, `null_deviance`, `loglik`, `aic` and `dispersion` are closed
forms given the fitted values, so 1e-10 relative. P-values are 1e-8
absolute, inheriting `NormalCDF`.

**The band for coefficients and standard errors must be measured, not
assumed.** Both implementations stop when the relative deviance change falls
below 1e-8, and near a quadratic minimum a deviance change of ε corresponds
to a coefficient change of order √ε, so the stopping rule alone guarantees
only about 1e-4. In practice IRLS converges quadratically and the final step
overshoots far past the criterion, so real agreement is much tighter. The
implementer measures the achieved agreement across every fixture, sets the
band from the worst case with margin, and records both the band and the
measurement. A band taken from theory here would be either uselessly loose or
quietly fragile. This is the same obligation the studentized range's node
counts carried in #21.

Where the `z_statistic`-everywhere choice makes a p-value differ from the
oracle for Gaussian and gamma, the generator records the oracle's t-based
value **and** the expected normal-based value, so the deviation is visible in
the fixture rather than hidden in a loosened band.

### Fixtures

One per thing that can break: 0/1 binomial with two predictors; grouped
binomial as proportions with trial counts; Poisson with an offset; gamma with
a log link; Gaussian identity for the `lm_fit` cross-check; a perfectly
separated binomial for the `separation` flag; a rank-deficient design with a
dependent column; and a clustered design for CR1.

### The SQL layer

`test/sql/glm_fit.test`, with floats printed through `printf('%.4f', …)`.

A worked example per family, plus:

- the options struct read correctly with its fields written in a different
  order from the canonical one;
- an unrecognized field name raising with the valid names listed;
- a non-constant value in `link`, `vcov` or `intercept` raising;
- `cr1` with no cluster field, and a cluster field with a non-cluster
  `vcov`, both raising, as `lm_fit` already does;
- each domain violation giving a NULL group rather than an error;
- a multi-group `GROUP BY` where one group fails a domain check and the
  others are unaffected;
- the `lm_fit` equivalence asserted at the SQL layer, since that is where a
  user meets it;
- `converged`, `iterations` and `separation` on the separated fixture.

### The metadata net

`test/sql/function_docs.test` holds three `IN (...)` lists and a count
assertion, at 99 after #21. Adding `glm_fit` makes it 100, and the name goes
in all three lists. The implementation plan gives this its own numbered step,
because in #20 it was missed and then reported as unaffected.

## Files

New:

- `src/include/cluster_ids.hpp`
- `src/include/glm_core.hpp`, `src/glm_core.cpp`
- `src/include/glm_fit_function.hpp`, `src/glm_fit_function.cpp`
- `test/cpp/gen_glm_fit_fixtures.py`, `test/cpp/glm_fit_fixtures.hpp`,
  `test/cpp/test_glm_fit.cpp`
- `test/sql/glm_fit.test`

Modified:

- `src/lm_fit_function.cpp` — onto `cluster_ids.hpp`
- `src/stats_duck_extension.cpp` — register `glm_fit`
- `CMakeLists.txt`, `scripts/run-cpp-tests.sh`
- `test/sql/function_docs.test`
- `README.md`, `docs/kernel_api.md`, `docs/fitter_conventions.md`,
  `CHANGELOG.md`

`docs/fitter_conventions.md` gains the options-struct convention, which is
new and binds future fitters: when a fitter's optional inputs exceed what
positional overloads can disambiguate — in particular when two of them are
per-row values of the same type — they go in one trailing STRUCT whose
fields are read by name, with an unrecognized name a bind error. The page
should also note that `lm_fit` keeps its positional overloads, because a
released position never changes meaning, and that a struct overload could be
added to it later.

## Size

This is a larger spec than #21's. Four families times seven links, six
covariance estimators on a new score-residual formula, an iterative loop
with its own convergence and failure semantics, a new argument shape, and a
separation heuristic. The implementation plan should expect on the order of
eight tasks, and it should sequence them so that each is independently
verifiable: the link functions and their round-trip tests before the loop
that uses them, the loop before the covariance that depends on its weights,
and the `lm_fit` equivalence as early as a Gaussian fit exists, because it
is the check that catches the most for the least.

If the plan cannot keep the tasks independently verifiable, that is a signal
to split this issue rather than to write a longer plan — the natural seam is
binomial and Poisson first, where dispersion is fixed at 1, then Gaussian
and gamma with the dispersion machinery.

## Decisions already taken

- Four families. Negative binomial is deferred to its own issue because of
  digamma and trigamma.
- Gamma's dispersion is the Pearson moment estimator, not maximum
  likelihood, so no new special functions are needed.
- One trailing options STRUCT, fields by name, unknown name a bind error.
- `z_statistic` and a normal reference for all four families, with the
  deviation from R documented.
- A heuristic `separation` flag, binomial only.
- HC0 through HC3 and CR0/CR1 are in scope, using score residuals.
- The rank-deficient set is fixed once rather than re-pivoted per iteration.
- `glm_fit` does not call `lm_core::fit_lm`; #19's WLS increment stays
  deferred.
