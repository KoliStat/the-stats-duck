# glm_fit Implementation Plan (#22)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `glm_fit(y, x [, family [, options]])`, an aggregate that fits a generalized linear model by iteratively reweighted least squares over four families, and reports coefficients with their covariance so `lin_hyp` and `emmeans` work against a GLM unchanged.

**Architecture:** A DuckDB-free kernel in `src/glm_core.cpp` owns the IRLS loop and the family and link dispatch, built on `linalg`, `distributions` and `estimability.hpp`. It does not call `lm_core::fit_lm` — `fit_lm` has no weights and computes R², F and a log-likelihood an inner loop would waste. The SQL layer binds one trailing options STRUCT whose fields are read by name, because `glm_fit` has seven optional inputs and two of them are per-row DOUBLEs that positional overloads can never disambiguate.

**Tech Stack:** C++17, the DuckDB aggregate API, the in-repo `linalg` and `distributions` kernels, Eigen only behind `linalg.cpp`, sqllogictest for the SQL layer, statsmodels as an offline oracle.

**Spec:** `notes/plans/2026-10-20-glm-fit-design.md` — read it alongside this plan. Where the two disagree, the spec is right and this plan is wrong.

## Dependency before Task 2

The kernel consumes `src/include/estimability.hpp`, added by #21 and not on `v0.9` until PR #67 merges. Task 1 does not need it. Before starting Task 2, confirm the header exists:

```bash
test -f src/include/estimability.hpp && echo present || echo "BLOCKED: merge PR #67 or rebase onto feat/21-emmeans"
```

If it is absent, stop and report — do not write a second copy of the aliased-coefficient convention.

## Global Constraints

Every task's requirements implicitly include these. Values are copied from the spec.

- **The binomial response is a proportion.** `y` in [0, 1], `weights` the trial count. 7 of 10 is `y = 0.7, weights = 10`. A count of successes passed as `y` fails the domain check, deliberately.
- **`z_statistic` with a normal reference, for all four families.** This differs from R and statsmodels for Gaussian and gamma, which switch to a t reference when dispersion is estimated. The deviation is documented beside every affected golden and in the user-facing docs. Never rename the field per family.
- **Dispersion** is fixed at 1 for binomial and Poisson; for Gaussian and gamma it is `Σ w·(y−mu)²/variance(mu) / (n − rank)`.
- **The rank-deficient column set is computed once, from X restricted to positive-weight rows, and held fixed** across iterations. Dropped columns follow #36: the earlier column of a dependent set is kept, the later one is aliased with NaN in its estimate, standard error, z, p and in the matching row and column of `cov`, and `terms` still names it.
- **Convergence:** `|dev − dev_prev| / (|dev| + 0.1) < 1e-8`, at most 25 outer iterations, at most 25 step halvings per iteration. `iterations` counts outer passes, not halvings.
- **Starting values follow R's `glm.fit`:** `(w·y + 0.5)/(w + 1)` binomial, `y + 0.1` Poisson, `y` Gaussian and gamma.
- **`converged = false` returns the best iterate, never NULL.** A data-shaped failure is a NULL group and never raises; misuse is a bind error.
- **Covariance is a flat row-major `LIST<DOUBLE>`** of length k², in `coefficients` order, with NaN wherever an aliased term is involved.
- **Result STRUCT field order is contract and append-only**, with the comment `// Field order MUST match the writes in Finalize` and an index comment per field.
- **NaN becomes SQL NULL at the vector layer; ±Inf passes through unchanged.**
- **Two namespaces, one letter apart.** `distributions.hpp` is `stats_duck`; the kernel is `statsduck`.
- **Determinism, on a given platform.** Fixed loop orders, no hashing, no random numbers. Not bit-identical across platforms, because libm differs — see `docs/fitter_conventions.md`.
- **No default-hash `std::string`-keyed `unordered_map` or `unordered_set`** in extension code. It breaks the wasm build at load time. `scripts/check-wasm-string-hash.sh` enforces it in CI.
- **Register with `RegisterDocumented`**, which applies `OnCreateConflict::ALTER_ON_CONFLICT`. Supply a description, a parameter list and an example.
- **Stage files explicitly. Never `git add -A`.** The `duckdb` submodule shows as modified by design and must stay out of every commit.
- **Plain English** in comments, commit messages and docs. Short linear sentences, one idea each, no buzzwords, no engineering slang.

Commands, with absolute paths because the shell's working directory drifts:

```bash
# Fast kernel suites, no DuckDB
bash scripts/run-cpp-tests.sh

# Full build. Check the EXIT STATUS, never the tail of the output: a pipeline
# ending in `tail` returns tail's status, so a compiler failure reads as
# success and you then test a stale binary. That cost real time twice on the
# #21 branch — see notes/engineering/2026-10-verifying-against-a-stale-binary.md
cmake --build "C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release" --config Release --target unittest
echo "build exit: $?"

# SQL suite. The ABSOLUTE path is required: a relative "test/sql/*" resolves
# against the DuckDB source tree and runs its 4000+ core tests instead.
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/test/Release/unittest.exe" \
  "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"
```

Baseline on `v0.9` before any change: C++ 73 + 511 + 113 + 39 = 736 checks in four suites, SQL 2206 assertions in 64 files. If PR #67 merged first, the baseline is 2271 checks in eight suites and 2376 assertions in 66 files — read the actual numbers from a clean run rather than assuming which.

## Review Focus

Five input classes the spec implies, which a reasonable person will meet and which no task's tests would otherwise exercise. Most likely first. Each has its test added to the task that owns the code.

1. **Binomial `y` exactly 0 or 1.** This is *every* plain 0/1 logistic fit, so it is not an edge case. The deviance term `y·ln(y/mu)` is `0·ln 0` at `y = 0`, which is NaN in IEEE unless the term is special-cased to 0. An unguarded deviance makes the single most common use of this function return NULL. Test in Task 3, Step 7.
2. **A prior weight of exactly 0 on some rows.** The spec computes the rank set from positive-weight rows, which implies such rows are excluded — but it never says whether they count in `n` and therefore in `df_residual`. **Ruling for this plan: a zero-weight row is dropped at accumulation, exactly like a NULL, so it is absent from `n`.** Counting a row that influences no estimate would inflate `df_residual` and deflate the dispersion. Test in Task 7, Step 12.
3. **`n ≤ rank` after the listwise drop.** The dispersion divides by `n − rank`, so 1 row with 1 predictor plus an intercept divides by zero and yields ±Inf or NaN across the whole result. Expected: a NULL group, decided before any division. Test in Task 4, Step 11.
4. **A response with no variation** — every `y` equal. For binomial that is perfect separation with no information; for Poisson all-zero counts drive `mu → 0` and the log link to −Inf. Expected: either a NULL group or a finite reported result with `converged` and `separation` set honestly, never NaN in a field that reads as an estimate. Test in Task 6, Step 5.
5. **An offset large enough to saturate the link.** A Poisson offset of 1000 makes `mu = e¹⁰⁰⁰`, which is +Inf before any fitting happens. Expected: a NULL group from the domain check, not Inf propagating into the coefficients and out to the user. Test in Task 7, Step 14.

---

## Task 1: Extract the cluster densifier

`DensifyClusters` turns VARCHAR cluster keys into dense zero-based ids by sorting rather than hashing, which is what keeps the result independent of hash-table order. Both aggregates need it. It takes only `std::vector<std::string>` and returns `std::vector<int>`, so it is already DuckDB-free and belongs in `src/include/`.

This task does not need `estimability.hpp`, so it can run before PR #67 merges.

**Files:**
- Create: `src/include/cluster_ids.hpp`
- Modify: `src/lm_fit_function.cpp:345-368` — delete the private copy, call the shared one
- Modify: `scripts/run-cpp-tests.sh` — one new suite
- Test: `test/cpp/test_cluster_ids.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, in namespace `statsduck`:
  ```cpp
  std::vector<int> densify_clusters(const std::vector<std::string> &keys);
  ```
  Returns one id per key, zero-based, assigned in ascending key order, with equal keys sharing an id. An empty input returns an empty vector.

- [ ] **Step 1: Write the failing test**

Create `test/cpp/test_cluster_ids.cpp`. Copy the `check` harness from `test/cpp/test_estimability.cpp` if it exists, or from `test/cpp/test_lin_hyp.cpp`: a `check(bool, const char *)` counter, a `main` returning non-zero on any failure, and a printed count. Use a bounds-safe accessor for every read — a failing call returns an empty vector, and indexing one turns a clear assertion failure into a crash that hides which assertion failed. That defect appeared three times in this repository's history.

```cpp
// Ids are assigned in ascending key order, not first-seen order, so the
// partition is reproducible whatever order the rows arrived in.
const std::vector<std::string> keys = {"c", "a", "b", "a", "c"};
const std::vector<int> ids = statsduck::densify_clusters(keys);
check(ids.size() == 5, "one id per key");
check(at(ids, 1) == 0 && at(ids, 3) == 0, "both 'a' rows share id 0");
check(at(ids, 2) == 1, "'b' is id 1");
check(at(ids, 0) == 2 && at(ids, 4) == 2, "both 'c' rows share id 2");

// The same partition from a different row order gives the same id per key.
const std::vector<std::string> shuffled = {"a", "c", "b", "c", "a"};
const std::vector<int> ids2 = statsduck::densify_clusters(shuffled);
check(at(ids2, 0) == 0 && at(ids2, 2) == 1 && at(ids2, 1) == 2,
      "ids follow the key, not the position");

// Degenerate shapes.
check(statsduck::densify_clusters({}).empty(), "no keys, no ids");
const std::vector<int> one = statsduck::densify_clusters({"only"});
check(one.size() == 1 && at(one, 0) == 0, "a single key is id 0");
const std::vector<int> same = statsduck::densify_clusters({"x", "x", "x"});
check(same.size() == 3 && at(same, 0) == 0 && at(same, 2) == 0,
      "all-equal keys are one cluster");

// The empty string is a key like any other, and sorts first.
const std::vector<int> emptykey = statsduck::densify_clusters({"b", "", "b"});
check(at(emptykey, 1) == 0 && at(emptykey, 0) == 1, "the empty string is a real key");

// The highest id plus one is the cluster count, which is what fit_lm needs.
int max_id = -1;
for (const int v : ids) { if (v > max_id) max_id = v; }
check(max_id + 1 == 3, "three distinct keys give ids 0..2");
```

- [ ] **Step 2: Register the suite and run it to see it fail**

In `scripts/run-cpp-tests.sh`, add alongside the existing `build_and_run` calls:

```bash
build_and_run test_cluster_ids \
  "$ROOT/test/cpp/test_cluster_ids.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_cluster_ids`, because the header does not exist. The other suites still pass.

- [ ] **Step 3: Write `src/include/cluster_ids.hpp`**

Header-only, `#pragma once`, namespace `statsduck`, including only `<algorithm>`, `<cstddef>`, `<string>` and `<vector>`. No DuckDB header, so a kernel translation unit can include it.

Move the body from `src/lm_fit_function.cpp:351-368` unchanged, renamed `densify_clusters` and marked `inline`. Carry the existing comment across, because it records *why* the function sorts instead of hashing: `std::sort` uses `operator<` and so avoids the libc++ `std::__hash_memory` symbol that a default-hashed `std::string`-keyed map pulls into the wasm build, which is the bug that broke `table_one` in the browser. Add one line saying the label values are irrelevant and only the partition they induce matters.

- [ ] **Step 4: Run the suite to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: every suite passes, with the pre-existing counts unchanged and the new one reporting its own.

- [ ] **Step 5: Move `lm_fit_function.cpp` onto the shared header**

Delete the private `DensifyClusters` and its comment. Include `cluster_ids.hpp` and call `statsduck::densify_clusters` at the one call site, around `src/lm_fit_function.cpp:404`. Change nothing else — this is a move, not a rewrite.

- [ ] **Step 6: Build and run both suites**

Run the build command from Global Constraints and check its exit status, then the SQL suite.
Expected: the SQL suite's assertion count is unchanged, and `test/sql/lm_fit_agg.test` passes in particular, since it is what covers the cluster path.

- [ ] **Step 7: Commit**

```bash
git add src/include/cluster_ids.hpp src/lm_fit_function.cpp test/cpp/test_cluster_ids.cpp scripts/run-cpp-tests.sh
git commit -m "Share the cluster densifier between fitters (#22)"
```

---

## Task 2: Families, links, and their parsers

The pure scalar functions the whole kernel is built on. They need no oracle, and their tests are the cheapest high-value tests in this plan: a wrong link derivative otherwise shows up only as slow or failed convergence, which reads like a tuning problem rather than a defect.

**Files:**
- Create: `src/include/glm_core.hpp` — the enums, the parsers, the four scalar functions
- Create: `src/glm_core.cpp` — their implementations
- Modify: `CMakeLists.txt` — add `src/glm_core.cpp` to `EXTENSION_SOURCES`
- Modify: `scripts/run-cpp-tests.sh` — one new suite, compiling `glm_core.cpp` and `linalg.cpp`
- Test: `test/cpp/test_glm_core.cpp`

**Interfaces:**
- Consumes: `statsduck::linalg::Mat` from `linalg.hpp`; `statsduck::Vcov`, `parse_vcov` and `vcov_name` from `lm_core.hpp`.
- Produces, in namespace `statsduck`:
  ```cpp
  enum class Family { kGaussian, kBinomial, kPoisson, kGamma };
  enum class Link { kIdentity, kLogit, kProbit, kCLogLog, kLog, kInverse, kSqrt };

  bool parse_family(const std::string &name, Family &out);   // false on unknown
  bool parse_link(const std::string &name, Link &out);        // false on unknown
  const char *family_name(Family f);
  const char *link_name(Link l);
  Link canonical_link(Family f);
  bool link_allowed(Family f, Link l);

  double link_eta(Link l, double mu);     // g(mu)
  double link_inv(Link l, double eta);    // g inverse; clamped, see below
  double link_deriv(Link l, double eta);  // dmu/deta
  double variance(Family f, double mu);   // V(mu)
  ```

- [ ] **Step 1: Write the failing test, round-trips first**

Create `test/cpp/test_glm_core.cpp`. The round-trip and derivative checks come before any constant, because a failure in them tells you the algebra is wrong rather than that a number moved.

```cpp
const statsduck::Link links[] = {
    statsduck::Link::kIdentity, statsduck::Link::kLogit, statsduck::Link::kProbit,
    statsduck::Link::kCLogLog, statsduck::Link::kLog, statsduck::Link::kInverse,
    statsduck::Link::kSqrt};

// Round trip. mu is sampled inside every link's domain: the three
// probability links need mu in (0, 1), and log, inverse and sqrt need mu > 0.
for (const statsduck::Link l : links) {
    const bool prob = (l == statsduck::Link::kLogit || l == statsduck::Link::kProbit ||
                       l == statsduck::Link::kCLogLog);
    const double mus_prob[] = {0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999};
    const double mus_pos[] = {1e-6, 0.01, 0.5, 1.0, 3.0, 100.0, 1e6};
    const double *mus = prob ? mus_prob : mus_pos;
    for (std::size_t i = 0; i < 7; i++) {
        const double mu = mus[i];
        const double back = statsduck::link_inv(l, statsduck::link_eta(l, mu));
        check(close_rel(back, mu, 1e-10), "link_inv inverts link_eta");
    }
}

// The derivative against a central difference of link_inv. The band is 1e-6
// relative because a central difference is second-order accurate: with
// h = 1e-5 the truncation error is about h^2 = 1e-10 relative, and the
// subtraction loses about 1e-11, so 1e-6 is loose enough to never be the
// thing that fails while still catching a wrong factor or sign.
for (const statsduck::Link l : links) {
    for (const double eta : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
        if (l == statsduck::Link::kInverse && std::fabs(eta) < 1e-9) continue; // mu = 1/eta
        const double h = 1e-5;
        const double fd = (statsduck::link_inv(l, eta + h) - statsduck::link_inv(l, eta - h)) / (2 * h);
        check(close_rel(statsduck::link_deriv(l, eta), fd, 1e-6),
              "link_deriv matches a central difference of link_inv");
    }
}

// Known closed forms, so a self-consistent but wrong pair cannot pass.
check(close_rel(statsduck::link_eta(statsduck::Link::kLogit, 0.5), 0.0, 0.0), "logit(0.5) = 0");
check(close_rel(statsduck::link_inv(statsduck::Link::kLogit, 0.0), 0.5, 1e-15), "expit(0) = 0.5");
check(close_rel(statsduck::link_deriv(statsduck::Link::kLogit, 0.0), 0.25, 1e-15),
      "d expit/d eta at 0 is 1/4");
check(close_rel(statsduck::link_inv(statsduck::Link::kProbit, 0.0), 0.5, 1e-12), "probit inverse at 0");
check(close_rel(statsduck::link_deriv(statsduck::Link::kIdentity, 7.0), 1.0, 0.0),
      "the identity link's derivative is 1");

// Variance functions.
check(close_rel(statsduck::variance(statsduck::Family::kGaussian, 3.0), 1.0, 0.0), "gaussian V = 1");
check(close_rel(statsduck::variance(statsduck::Family::kPoisson, 3.0), 3.0, 0.0), "poisson V = mu");
check(close_rel(statsduck::variance(statsduck::Family::kBinomial, 0.25), 0.1875, 1e-15),
      "binomial V = mu(1-mu)");
check(close_rel(statsduck::variance(statsduck::Family::kGamma, 3.0), 9.0, 0.0), "gamma V = mu^2");

// Parsing, the canonical links, and what each family allows.
statsduck::Family f;
statsduck::Link l;
check(statsduck::parse_family("binomial", f) && f == statsduck::Family::kBinomial, "binomial parses");
check(statsduck::parse_family("GAUSSIAN", f) && f == statsduck::Family::kGaussian,
      "family parsing is case insensitive, as parse_vcov is");
check(!statsduck::parse_family("negbin", f), "negative binomial is not in this increment");
check(!statsduck::parse_family("", f), "an empty family name is unknown");
check(statsduck::parse_link("cloglog", l) && l == statsduck::Link::kCLogLog, "cloglog parses");
check(!statsduck::parse_link("logaritmic", l), "a misspelled link is unknown");

check(statsduck::canonical_link(statsduck::Family::kGaussian) == statsduck::Link::kIdentity, "gaussian canonical");
check(statsduck::canonical_link(statsduck::Family::kBinomial) == statsduck::Link::kLogit, "binomial canonical");
check(statsduck::canonical_link(statsduck::Family::kPoisson) == statsduck::Link::kLog, "poisson canonical");
check(statsduck::canonical_link(statsduck::Family::kGamma) == statsduck::Link::kInverse, "gamma canonical");

check(statsduck::link_allowed(statsduck::Family::kBinomial, statsduck::Link::kProbit), "binomial allows probit");
check(!statsduck::link_allowed(statsduck::Family::kPoisson, statsduck::Link::kLogit),
      "a probability link on poisson is not allowed");
check(!statsduck::link_allowed(statsduck::Family::kBinomial, statsduck::Link::kInverse),
      "binomial does not allow inverse");
// Every family allows its own canonical link, which is the invariant that
// makes the family-only call form work.
for (const statsduck::Family fam : {statsduck::Family::kGaussian, statsduck::Family::kBinomial,
                                    statsduck::Family::kPoisson, statsduck::Family::kGamma}) {
    check(statsduck::link_allowed(fam, statsduck::canonical_link(fam)),
          "the canonical link is always allowed");
}

// Canonical echo: a name parses to the enum and the enum echoes that name.
for (const char *n : {"gaussian", "binomial", "poisson", "gamma"}) {
    check(statsduck::parse_family(n, f) && std::string(statsduck::family_name(f)) == n,
          "family_name echoes the canonical spelling");
}
for (const char *n : {"identity", "logit", "probit", "cloglog", "log", "inverse", "sqrt"}) {
    check(statsduck::parse_link(n, l) && std::string(statsduck::link_name(l)) == n,
          "link_name echoes the canonical spelling");
}
```

- [ ] **Step 2: Register the suite and run it to see it fail**

```bash
build_and_run test_glm_core \
  "$ROOT/test/cpp/test_glm_core.cpp" \
  "$ROOT/src/glm_core.cpp" \
  "$ROOT/src/linalg.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_glm_core`.

- [ ] **Step 3: Write the header's enums, parsers and declarations**

`src/include/glm_core.hpp`, namespace `statsduck`. The file comment states the scope, that the kernel is DuckDB-free and Eigen-free at its boundary, and that it depends on `lm_core.hpp` for the `Vcov` enum only and never for fitting.

`parse_family` and `parse_link` return `bool` rather than throwing, in the shape of `lm_core::parse_vcov`, so the caller chooses the exception type and message. Both are case-insensitive, matching `parse_vcov`. `link_allowed` rejects a probability link on a non-binomial family and `inverse` on binomial; document the table it encodes.

- [ ] **Step 4: Implement the four scalar functions**

In `src/glm_core.cpp`. The algebra:

| Link | `link_eta(mu)` | `link_inv(eta)` | `link_deriv(eta)` |
| --- | --- | --- | --- |
| identity | `mu` | `eta` | `1` |
| logit | `ln(mu/(1−mu))` | `1/(1+e^−eta)` | `mu(1−mu)` |
| probit | `NormalQuantile(mu)` | `NormalCDF(eta)` | `NormalPDF(eta)` |
| cloglog | `ln(−ln(1−mu))` | `1 − e^(−e^eta)` | `e^(eta − e^eta)` |
| log | `ln(mu)` | `e^eta` | `e^eta` |
| inverse | `1/mu` | `1/eta` | `−1/eta²` |
| sqrt | `√mu` | `eta²` | `2·eta` |

Use `stats_duck::NormalCDF`, `NormalQuantile` and `NormalPDF` from `distributions.hpp` for probit — never a second implementation, per the conventions doc. Remember the namespace is `stats_duck` with an underscore here.

Variance: `1` for Gaussian, `mu(1−mu)` for binomial, `mu` for Poisson, `mu²` for gamma.

**`link_inv` clamps its result away from the boundary of the family's support.** An unclamped logit inverse underflows to exactly 0 around `eta = −745`, and the deviance then returns infinity for a case that should simply be a very small fitted probability. Clamp to `[kMuEps, 1 − kMuEps]` for the probability links and to `≥ kMuEps` for log, inverse and sqrt, with `kMuEps = 1e-10`. State the constant and the reason in the header, and note that the clamp is why `link_inv(link_eta(mu))` round-trips only for `mu` inside the clamped range — the test samples accordingly.

- [ ] **Step 5: Run the test to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: `test_glm_core` passes. If the derivative check fails for exactly one link, the error is in that row of the table above, not in the test.

- [ ] **Step 6: Commit**

```bash
git add src/include/glm_core.hpp src/glm_core.cpp test/cpp/test_glm_core.cpp CMakeLists.txt scripts/run-cpp-tests.sh
git commit -m "Add the glm families, links and their parsers (#22)"
```

---

## Task 3: The oracle, the deviance, the log-likelihood and the dispersion

Closed-form quantities given the fitted values, so they are testable before any fitting exists. This task also builds the generator that every later task consumes, so the oracle's conventions are pinned once and early.

**Files:**
- Create: `test/cpp/gen_glm_fit_fixtures.py`, `test/cpp/glm_fit_fixtures.hpp` (generated)
- Modify: `src/include/glm_core.hpp`, `src/glm_core.cpp` — the four functions below
- Modify: `test/cpp/test_glm_core.cpp` — their tests

**Interfaces:**
- Consumes: `Family`, `variance`, from Task 2.
- Produces, in namespace `statsduck`:
  ```cpp
  // Σ of the per-observation deviance contributions, already doubled.
  double deviance(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                  const std::vector<double> &w);
  double glm_loglik(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                    const std::vector<double> &w, double dispersion);
  double glm_aic(Family f, double loglik, std::size_t rank);
  double pearson_dispersion(Family f, const std::vector<double> &y, const std::vector<double> &mu,
                            const std::vector<double> &w, std::size_t df_residual);
  ```

- [ ] **Step 1: Write the generator**

Create `test/cpp/gen_glm_fit_fixtures.py`, following `test/cpp/gen_lm_fit_fixtures.py` in structure: a module docstring naming the oracle and its version, a printed header file, and self-checking identities that fail loudly if a convention moves.

```python
"""Goldens for glm_fit.

Oracle: statsmodels sm.GLM (statsmodels >= 0.14), run offline in a throwaway
virtualenv. Never a build or CI dependency.

Eight designs, one per thing that can break:
  DS1 binomial 0/1, two predictors, logit
  DS2 binomial grouped: DS1 collapsed to proportions with trial counts
  DS3 poisson with a log link and an offset
  DS4 gamma with a log link
  DS5 gaussian identity, for the lm_fit cross-check
  DS6 binomial perfectly separated, for the separation flag
  DS7 a rank-deficient design whose third column is the sum of the first two
  DS8 a clustered design, for CR1

DELIBERATE DEVIATION, recorded next to the goldens it affects: this project
reports z_statistic against a normal reference for every family, while
statsmodels switches to a t reference when dispersion is estimated. So for
DS4 and DS5 the oracle's p-values differ from ours. Both are emitted: p_oracle
is what statsmodels reports and p_expected is the normal-reference value the
kernel must produce. For binomial and poisson the two are identical.

Self-checks that must hold before anything is written:
  1. DS1 and DS2 give the same coefficients and standard errors, because they
     are the same data in two shapes
  2. null_deviance equals the deviance of the intercept-only fit
  3. DS5's coefficients and covariance match statsmodels OLS exactly
  4. scaling DS2's trial counts by 10 leaves the coefficients unchanged and
     scales the deviance by 10
  5. for binomial and poisson, p_oracle == p_expected
"""
```

Emit, per design: the data, then `beta`, `std_error`, `z_statistic`, `p_oracle`, `p_expected`, `deviance`, `null_deviance`, `dispersion`, `loglik`, `aic`, `rank`, `df_residual`, and the flat row-major `cov` for each covariance estimator the design exercises. Keep 17 significant digits.

- [ ] **Step 2: Run the generator and commit the fixtures**

```bash
python -m venv /tmp/glmenv && /tmp/glmenv/bin/pip install statsmodels
/tmp/glmenv/bin/python test/cpp/gen_glm_fit_fixtures.py > test/cpp/glm_fit_fixtures.hpp
```

Expected: the five self-checks print as passing. **If self-check 1 fails, stop and report** — it means either the grouped-binomial convention is not what the spec says or the fixture is built wrong, and no golden should be trusted until it holds.

- [ ] **Step 3: Write the failing test for the deviance**

Add to `test/cpp/test_glm_core.cpp`. Hand-computable cases first, because they localize a failure better than a fixture does.

```cpp
// Gaussian deviance is the weighted residual sum of squares.
const std::vector<double> y = {1.0, 2.0, 4.0};
const std::vector<double> mu = {1.5, 2.0, 3.0};
const std::vector<double> w = {1.0, 1.0, 1.0};
check(close_rel(statsduck::deviance(statsduck::Family::kGaussian, y, mu, w), 1.25, 1e-15),
      "gaussian deviance is 0.25 + 0 + 1");

// Poisson: 2 Σ w [y ln(y/mu) − (y − mu)]. At y = mu every term is 0.
const std::vector<double> ymu = {2.0, 3.0};
check(close_rel(statsduck::deviance(statsduck::Family::kPoisson, ymu, ymu, {1.0, 1.0}), 0.0, 1e-15),
      "a perfect poisson fit has zero deviance");

// Weights scale the deviance linearly.
const double d1 = statsduck::deviance(statsduck::Family::kGaussian, y, mu, {1.0, 1.0, 1.0});
const double d2 = statsduck::deviance(statsduck::Family::kGaussian, y, mu, {3.0, 3.0, 3.0});
check(close_rel(d2, 3.0 * d1, 1e-15), "the deviance is linear in the weights");

// Binomial at y = 0.5, mu = 0.5 is zero; a mismatch is positive.
check(close_rel(statsduck::deviance(statsduck::Family::kBinomial, {0.5}, {0.5}, {1.0}), 0.0, 1e-15),
      "a perfect binomial fit has zero deviance");
check(statsduck::deviance(statsduck::Family::kBinomial, {0.5}, {0.9}, {1.0}) > 0.0,
      "a binomial mismatch has positive deviance");
```

- [ ] **Step 4: Implement the deviance**

The four families, each already doubled:

```
gaussian:  Σ w·(y − mu)²
binomial:  2 Σ w·[ y·ln(y/mu) + (1 − y)·ln((1 − y)/(1 − mu)) ]
poisson:   2 Σ w·[ y·ln(y/mu) − (y − mu) ]
gamma:     2 Σ w·[ −ln(y/mu) + (y − mu)/mu ]
```

- [ ] **Step 5: Implement the log-likelihood, the AIC and the dispersion**

```
gaussian loglik:  −n/2·(ln 2π σ² + 1),  σ² = deviance/n     // matches lm_core::loglik
binomial loglik:  Σ [ LogChoose(w, w·y) + w·y·ln mu + w·(1−y)·ln(1−mu) ]
poisson loglik:   Σ w·[ y·ln mu − mu − lgamma(y + 1) ]
gamma loglik:     with shape = 1/dispersion, through std::lgamma
```

`LogChoose` is already in `distributions.hpp` at line 863. `std::lgamma` from `<cmath>` covers the rest; this task adds no new special functions.

`glm_aic` is `−2·loglik + 2·(rank + d)`, where `d` is 1 for Gaussian and gamma, whose dispersion is an estimated parameter, and 0 for binomial and Poisson. The Gaussian log-likelihood is `+inf` at zero deviance, so the AIC is `−inf`; that passes through as a value rather than becoming NULL, because ±Inf is representable under the conventions doc.

`pearson_dispersion` returns `Σ w·(y − mu)²/variance(mu) / df_residual` for Gaussian and gamma, and exactly 1.0 for binomial and Poisson regardless of the data — the caller never has to know which families estimate it.

- [ ] **Step 6: Run, then commit**

Run: `bash scripts/run-cpp-tests.sh`

```bash
git add src/include/glm_core.hpp src/glm_core.cpp test/cpp/test_glm_core.cpp \
        test/cpp/gen_glm_fit_fixtures.py test/cpp/glm_fit_fixtures.hpp
git commit -m "Add the glm deviance, log-likelihood, AIC and dispersion (#22)"
```

- [ ] **Step 7: Write the zero-and-one binomial test (Review Focus item 1)**

```cpp
// Every plain 0/1 logistic fit has y exactly 0 and exactly 1, so this is the
// common case and not an edge case. The deviance term y·ln(y/mu) is 0·ln 0 at
// y = 0, which is NaN in IEEE unless the term is special-cased to zero. An
// unguarded deviance makes the most common use of this function return NULL.
const std::vector<double> y01 = {0.0, 1.0, 0.0, 1.0};
const std::vector<double> mu01 = {0.2, 0.8, 0.3, 0.6};
const std::vector<double> w01 = {1.0, 1.0, 1.0, 1.0};
const double d01 = statsduck::deviance(statsduck::Family::kBinomial, y01, mu01, w01);
check(std::isfinite(d01), "a 0/1 binomial deviance is finite");
check(d01 > 0.0, "and positive");

// Hand-computed: −2[ln(0.8) + ln(0.8) + ln(0.7) + ln(0.6)] = 2.5285...
check(close_rel(d01, -2.0 * (std::log(0.8) + std::log(0.8) + std::log(0.7) + std::log(0.6)), 1e-12),
      "at y in {0, 1} the deviance reduces to -2 Σ ln(likelihood)");

// The same guard on the other side, and for a weighted proportion of exactly 0.
check(std::isfinite(statsduck::deviance(statsduck::Family::kBinomial, {0.0}, {0.5}, {10.0})),
      "a zero proportion with weight 10 is finite");
check(std::isfinite(statsduck::deviance(statsduck::Family::kBinomial, {1.0}, {0.5}, {10.0})),
      "a unit proportion with weight 10 is finite");

// Poisson has the same 0·ln 0 term at y = 0, which all-zero count data hits.
const double dp0 = statsduck::deviance(statsduck::Family::kPoisson, {0.0, 0.0}, {0.5, 0.5}, {1.0, 1.0});
check(std::isfinite(dp0), "a poisson deviance at y = 0 is finite");
check(close_rel(dp0, 2.0, 1e-12), "and equals 2 Σ mu when every y is zero");

// The log-likelihood must survive the same inputs.
check(std::isfinite(statsduck::glm_loglik(statsduck::Family::kBinomial, y01, mu01, w01, 1.0)),
      "the 0/1 binomial log-likelihood is finite");
```

- [ ] **Step 8: Run, then commit**

Run: `bash scripts/run-cpp-tests.sh`. If this fails, the deviance needs `y·ln(y/mu)` defined as 0 when `y == 0`, and the symmetric term defined as 0 when `y == 1`.

```bash
git add test/cpp/test_glm_core.cpp src/glm_core.cpp
git commit -m "glm: guard the deviance at a zero or unit response (#22)"
```

---

## Task 4: The IRLS loop with classical covariance

**Files:**
- Modify: `src/include/glm_core.hpp` — `GlmOptions`, `GlmResult`, `fit_glm`
- Modify: `src/glm_core.cpp` — the loop
- Create: `test/cpp/test_glm_fit.cpp`
- Modify: `scripts/run-cpp-tests.sh` — one new suite

**Interfaces:**
- Consumes: everything from Tasks 2 and 3; `linalg::cholesky_solve`, `svd_solve`, `inv_spd`, `independent_columns`, `matvec`; `statsduck::aliased_columns` and `zero_aliased` from `estimability.hpp`.
- Produces, in namespace `statsduck`:
  ```cpp
  struct GlmOptions {
      Family family = Family::kGaussian;
      Link link = Link::kIdentity;      // set from canonical_link unless given
      Vcov vcov = Vcov::kConst;
      bool intercept = true;
      int max_iter = 25;
      double tol = 1e-8;
  };

  struct GlmResult {
      bool ok = false;
      std::string error;
      std::size_t n = 0, k = 0, rank = 0, n_clusters = 0, df_residual = 0;
      std::vector<std::string> terms;
      std::vector<double> beta, std_error, z_statistic, p_value;
      std::vector<double> cov;          // row-major k²
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
  `ok = false` with a populated `error` is a data-shaped failure the caller reports as NULL. `terms` are `(Intercept)`, `x1`, `x2`, … matching `lm_core`.

- [ ] **Step 1: Write the failing test — the `lm_fit` equivalence first**

Create `test/cpp/test_glm_fit.cpp`. This test comes first because it is the one that catches the most for the least: it ties the new fitter to an already statsmodels-validated one rather than to a fresh set of constants.

```cpp
// Gaussian with an identity link IS ordinary least squares. The two fitters
// must agree on every coefficient, every standard error, the full covariance
// matrix, df_residual, and dispersion against sigma squared. If the weighted
// solve or the covariance assembly is wrong, this is where it shows.
statsduck::GlmOptions go;
go.family = statsduck::Family::kGaussian;
go.link = statsduck::Link::kIdentity;
const statsduck::GlmResult g = statsduck::fit_glm(kDs5Y, ds5_design(), go);
statsduck::LmOptions lo;
const statsduck::LmResult o = statsduck::fit_lm(kDs5Y, ds5_design(), lo);

check(g.ok && o.ok, "both fitters succeed on DS5");
check(g.k == o.k && g.rank == o.rank, "same k and rank");
check(g.df_residual == o.df_residual, "same residual df"); // both are std::size_t
for (std::size_t j = 0; j < o.k; j++) {
    check(close_rel(at(g.beta, j), at(o.beta, j), 1e-12), "same coefficient");
    check(close_rel(at(g.std_error, j), at(o.std_error, j), 1e-12), "same standard error");
}
for (std::size_t e = 0; e < o.cov.size(); e++) {
    check(close_rel(at(g.cov, e), at(o.cov, e), 1e-12), "same covariance entry");
}
check(close_rel(g.dispersion, o.sigma * o.sigma, 1e-12), "dispersion is sigma squared");
check(g.converged, "an identity-link gaussian fit converges");
check(g.iterations <= 3, "and does so in very few iterations, since it is linear");
check(!g.separation, "separation is false for a gaussian fit");
```

The band is 1e-12 rather than something looser because Gaussian with identity converges in one step: the working weights are all 1 and the working response is `y`, so IRLS *is* the normal-equations solve. Any larger discrepancy means the two paths differ structurally, not numerically.

- [ ] **Step 2: Register the suite and run it to see it fail**

```bash
build_and_run test_glm_fit \
  "$ROOT/test/cpp/test_glm_fit.cpp" \
  "$ROOT/src/glm_core.cpp" \
  "$ROOT/src/lm_core.cpp" \
  "$ROOT/src/linalg.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_glm_fit`, because `fit_glm` does not exist.

- [ ] **Step 3: Write the structs and the validation**

Add `GlmOptions`, `GlmResult` and the `fit_glm` declaration to the header, documenting the three-outcome contract and that `terms` names aliased columns. In `src/glm_core.cpp`, validate first and return `ok = false` with a message for each data-shaped failure:

- `y` empty, or `y.size()` not equal to `X_pred.rows`;
- a supplied `weights` or `offset` whose length is not `y.size()`;
- `!link_allowed(opts.family, opts.link)`;
- a domain violation: binomial `y` outside [0, 1], Poisson `y` negative, gamma `y` not strictly positive, a negative or non-finite weight, a non-finite offset;
- `opts.vcov` is CR0 or CR1 with no `cluster_ids`, or fewer than two distinct clusters.

- [ ] **Step 4: Assemble the design and the fixed aliased set**

Prepend a constant column when `opts.intercept`, so `X_pred` arrives without one, matching `fit_lm`. Then compute the aliased set **once**, with `linalg::independent_columns` on the design restricted to rows whose prior weight is positive, and hold it for the whole loop. The header states why: for strictly positive weights `rank(W^½X) = rank(X)` because `W^½` is invertible, so the set cannot change between iterations, and fixing it makes the fit deterministic where re-pivoting could oscillate.

- [ ] **Step 5: Implement the loop**

```
mu    = start(family, y, w)                        // R's glm.fit starting values
eta   = link_eta(mu)
dev   = deviance(family, y, mu, w)
repeat up to max_iter:
    z   = eta − offset + (y − mu) / link_deriv(eta)
    W   = w · link_deriv(eta)² / variance(mu)
    solve (XᵀWX)·beta = XᵀWz                      // kept columns only
    eta_new = X·beta + offset
    mu_new  = link_inv(eta_new)
    dev_new = deviance(family, y, mu_new, w)
    halve the step up to 25 times while dev_new is not finite or exceeds dev
    if |dev_new − dev| / (|dev| + 0.1) < tol: converged, stop
```

Use `linalg::cholesky_solve` on `XᵀWX`, falling back to `linalg::svd_solve` when the Cholesky reports the matrix is not positive definite, mirroring `lm_core`. `iterations` counts outer passes, not halvings. On exhausting `max_iter`, return the best iterate with `converged = false` — never NULL, per the conventions doc.

- [ ] **Step 6: Fill in the result**

`dispersion` from `pearson_dispersion`, `cov` as `dispersion · inv_spd(XᵀWX)` for the classical estimator, `std_error` as the square root of its diagonal, `z_statistic` as `beta/std_error`, and `p_value` as `2·(1 − NormalCDF(|z|))`. Aliased columns get NaN in `beta`, `std_error`, `z_statistic`, `p_value` and in their row and column of `cov`, while `terms` still names them.

`null_deviance` comes from **actually fitting the intercept-only model** with the same family, link, weights and offset — not from a closed form. When `opts.intercept` is false the null model has no columns at all, so `mu` is the link inverse of the offset alone.

- [ ] **Step 7: Run the test to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: `test_glm_fit` passes the equivalence block.

- [ ] **Step 8: Add the oracle comparisons and measure the achieved agreement**

Loop over DS1, DS3, DS4 and DS5 from the fixtures, comparing `beta`, `std_error`, `z_statistic`, `p_expected`, `deviance`, `null_deviance`, `dispersion`, `loglik` and `aic`.

**The band for coefficients and standard errors must be measured rather than assumed.** Both implementations stop when the relative deviance change falls below 1e-8, and near a quadratic minimum a deviance change of ε corresponds to a coefficient change of order √ε — so the stopping rule alone guarantees only about 1e-4, while IRLS's quadratic convergence means the final step usually overshoots far past the criterion and real agreement is much tighter. Print the worst relative deviation across every fixture, set the band from it with margin, and record both the band and the measurement in a comment. The closed-form quantities — deviance, `null_deviance`, `loglik`, `aic`, `dispersion` — take 1e-10 relative, and p-values 1e-8 absolute, inheriting `NormalCDF`.

Compare DS4 and DS5 against `p_expected`, not `p_oracle`, and say in a comment that the two differ because this project uses a normal reference where statsmodels uses a t reference when dispersion is estimated.

- [ ] **Step 9: Add the oracle-free identities**

```cpp
// null_deviance is the intercept-only model's deviance, computed two ways.
statsduck::GlmOptions nb;
nb.family = statsduck::Family::kBinomial;
nb.link = statsduck::Link::kLogit;
const statsduck::GlmResult full = statsduck::fit_glm(kDs1Y, ds1_design(), nb);
const statsduck::GlmResult nullfit = statsduck::fit_glm(kDs1Y, linalg::Mat(kDs1Y.size(), 0), nb);
check(nullfit.ok, "the intercept-only fit succeeds");
check(close_rel(full.null_deviance, nullfit.deviance, 1e-10),
      "null_deviance equals the intercept-only fit's deviance");

// Grouped and ungrouped binomial are the same data in two shapes.
const statsduck::GlmResult ungrouped = statsduck::fit_glm(kDs1Y, ds1_design(), nb);
const statsduck::GlmResult grouped = statsduck::fit_glm(kDs2Y, ds2_design(), nb, &kDs2Weights);
for (std::size_t j = 0; j < ungrouped.k; j++) {
    check(close_rel(at(grouped.beta, j), at(ungrouped.beta, j), 1e-8),
          "grouping does not change the coefficients");
    check(close_rel(at(grouped.std_error, j), at(ungrouped.std_error, j), 1e-8),
          "nor the standard errors");
}

// Scaling every trial count scales the deviance and leaves beta alone.
std::vector<double> w10 = kDs2Weights;
for (double &v : w10) { v *= 10.0; }
const statsduck::GlmResult scaled = statsduck::fit_glm(kDs2Y, ds2_design(), nb, &w10);
for (std::size_t j = 0; j < grouped.k; j++) {
    check(close_rel(at(scaled.beta, j), at(grouped.beta, j), 1e-8),
          "scaling the trial counts leaves the coefficients unchanged");
}
check(close_rel(scaled.deviance, 10.0 * grouped.deviance, 1e-8),
      "and scales the deviance by the same factor");

// A rank-deficient design: the later column of a dependent set is aliased.
const statsduck::GlmResult rd = statsduck::fit_glm(kDs7Y, ds7_design(), nb);
check(rd.ok, "a rank-deficient fit is not a failure");
check(rd.rank < rd.k, "the rank is below k");
check(rd.terms.size() == rd.k, "every column is still named");
check(std::isnan(at(rd.beta, rd.k - 1)), "the dependent column is aliased");
check(std::isfinite(at(rd.beta, 1)), "the kept columns are estimated");
check(rd.df_residual == rd.n - rd.rank, "df_residual counts the rank, not k");
```

- [ ] **Step 10: Run, then commit**

Run: `bash scripts/run-cpp-tests.sh`

```bash
git add src/include/glm_core.hpp src/glm_core.cpp test/cpp/test_glm_fit.cpp scripts/run-cpp-tests.sh
git commit -m "Add the glm IRLS loop with classical covariance (#22)"
```

- [ ] **Step 11: Write the n ≤ rank test (Review Focus item 3)**

```cpp
// The dispersion divides by n − rank, so one row with one predictor plus an
// intercept divides by zero and would put Inf or NaN across the whole result.
// The decision has to be made before any division.
linalg::Mat tiny(1, 1);
tiny(0, 0) = 2.0;
statsduck::GlmOptions tg;
tg.family = statsduck::Family::kGaussian;
const statsduck::GlmResult t1 = statsduck::fit_glm({5.0}, tiny, tg);
check(!t1.ok, "one row with an intercept and one predictor is a failed fit");
check(!t1.error.empty(), "and says why");

// Two rows and two parameters is exactly saturated: n == rank, so still no df.
linalg::Mat two(2, 1);
two(0, 0) = 1.0;
two(1, 0) = 2.0;
const statsduck::GlmResult t2 = statsduck::fit_glm({3.0, 7.0}, two, tg);
check(!t2.ok, "a saturated gaussian fit has no residual df and fails");

// Three rows and two parameters is the smallest fit that works.
linalg::Mat three(3, 1);
three(0, 0) = 1.0;
three(1, 0) = 2.0;
three(2, 0) = 4.0;
const statsduck::GlmResult t3 = statsduck::fit_glm({3.0, 7.0, 11.0}, three, tg);
check(t3.ok && t3.df_residual == 1, "three rows and two parameters leaves one df");
check(std::isfinite(t3.dispersion), "and a finite dispersion");

// Binomial and poisson have a fixed dispersion, so they do not divide at all
// — but they still need df_residual to be meaningful, so the rule is uniform.
statsduck::GlmOptions bg;
bg.family = statsduck::Family::kBinomial;
bg.link = statsduck::Link::kLogit;
const statsduck::GlmResult b1 = statsduck::fit_glm({1.0}, tiny, bg);
check(!b1.ok, "the n <= rank rule applies to binomial too");
```

- [ ] **Step 12: Run, then commit**

```bash
git add test/cpp/test_glm_fit.cpp src/glm_core.cpp
git commit -m "glm: fail a fit with no residual degrees of freedom (#22)"
```

---

## Task 5: Robust covariance from score residuals

**Files:**
- Modify: `src/glm_core.cpp` — the covariance assembly
- Modify: `test/cpp/test_glm_fit.cpp`

**Interfaces:**
- Consumes: `fit_glm` from Task 4; `linalg::sandwich`; `statsduck::densify_clusters` from Task 1 is used by the SQL layer, not here — `fit_glm` already takes dense `cluster_ids`.
- Produces: no new signature. `opts.vcov` now changes `cov`, `std_error`, `z_statistic` and `p_value`, and `n_clusters` is populated for CR0 and CR1.

- [ ] **Step 1: Write the failing test — the equivalence, per estimator**

```cpp
// At gaussian with an identity link, link_deriv is 1 and variance is 1, so the
// score u_i = x_i·w_i·(y_i − mu_i) reduces to the OLS residual score. Every
// robust estimator must therefore agree with lm_core's, which is already
// statsmodels-validated. This is the test for the whole sandwich.
const statsduck::Vcov estimators[] = {statsduck::Vcov::kHC0, statsduck::Vcov::kHC1,
                                      statsduck::Vcov::kHC2, statsduck::Vcov::kHC3};
for (const statsduck::Vcov v : estimators) {
    statsduck::GlmOptions go;
    go.family = statsduck::Family::kGaussian;
    go.link = statsduck::Link::kIdentity;
    go.vcov = v;
    statsduck::LmOptions lo;
    lo.vcov = v;
    const statsduck::GlmResult g = statsduck::fit_glm(kDs5Y, ds5_design(), go);
    const statsduck::LmResult o = statsduck::fit_lm(kDs5Y, ds5_design(), lo);
    check(g.ok && o.ok, "both fitters succeed");
    for (std::size_t e = 0; e < o.cov.size(); e++) {
        check(close_rel(at(g.cov, e), at(o.cov, e), 1e-12),
              "the glm sandwich matches lm_core's at gaussian identity");
    }
}

// The same for the cluster-robust pair, which also has to count the clusters.
for (const statsduck::Vcov v : {statsduck::Vcov::kCR0, statsduck::Vcov::kCR1}) {
    statsduck::GlmOptions go;
    go.family = statsduck::Family::kGaussian;
    go.link = statsduck::Link::kIdentity;
    go.vcov = v;
    statsduck::LmOptions lo;
    lo.vcov = v;
    const statsduck::GlmResult g = statsduck::fit_glm(kDs8Y, ds8_design(), go, nullptr, nullptr, &kDs8Clusters);
    const statsduck::LmResult o = statsduck::fit_lm(kDs8Y, ds8_design(), lo, &kDs8Clusters);
    check(g.ok && o.ok, "both cluster fits succeed");
    check(g.n_clusters == o.n_clusters, "same cluster count");
    for (std::size_t e = 0; e < o.cov.size(); e++) {
        check(close_rel(at(g.cov, e), at(o.cov, e), 1e-12), "the cluster sandwich matches");
    }
}

// Changing the estimator never moves the point estimates.
statsduck::GlmOptions c1, c2;
c1.family = c2.family = statsduck::Family::kBinomial;
c1.link = c2.link = statsduck::Link::kLogit;
c2.vcov = statsduck::Vcov::kHC1;
const statsduck::GlmResult p1 = statsduck::fit_glm(kDs1Y, ds1_design(), c1);
const statsduck::GlmResult p2 = statsduck::fit_glm(kDs1Y, ds1_design(), c2);
for (std::size_t j = 0; j < p1.k; j++) {
    check(close_rel(at(p2.beta, j), at(p1.beta, j), 0.0),
          "beta is bit-identical across covariance estimators");
}
check(!close_rel(at(p2.std_error, 1), at(p1.std_error, 1), 1e-9),
      "but the standard errors do differ");
```

- [ ] **Step 2: Run it to see it fail**

Run: `bash scripts/run-cpp-tests.sh`
Expected: the robust cases fail, because `opts.vcov` is still ignored.

- [ ] **Step 3: Implement the score residuals and the sandwich**

The score contribution for observation i is

```
u_i = x_i · w_i · (y_i − mu_i) · link_deriv(eta_i) / variance(mu_i)
```

- **HC0**: `A⁻¹ (Σ u_i u_iᵀ) A⁻¹` where `A = XᵀWX`.
- **HC1**: HC0 times `n / (n − rank)`.
- **HC2**, **HC3**: divide each `u_i` by `(1 − h_i)^½` and `(1 − h_i)` respectively, with `h_i = w_i^½ · x_iᵀ A⁻¹ x_i · w_i^½`, the hat diagonal of the weighted design.
- **CR0**: sum the scores within each cluster before the outer product.
- **CR1**: CR0 times `[G/(G−1)]·[(N−1)/(N−rank)]`.

Use `linalg::sandwich` for the outer products and `linalg::inv_spd` for `A⁻¹`, exactly as `lm_core` does. Set `n_clusters` for CR0 and CR1, and leave it 0 otherwise so the SQL layer writes NULL.

Inference df follow `lm_core`: cluster-robust references `G − 1`, while `df_residual` stays `n − rank` because it is a property of the fit rather than of the estimator. Note that `z_statistic` and its p-value stay normal-referenced under every estimator, per the Global Constraints.

- [ ] **Step 4: Run, then commit**

Run: `bash scripts/run-cpp-tests.sh`. If HC2 or HC3 fails while HC0 and HC1 pass, the hat diagonal is missing a weight factor.

```bash
git add src/glm_core.cpp test/cpp/test_glm_fit.cpp
git commit -m "Add score-residual robust covariance to glm_fit (#22)"
```

- [ ] **Step 5: Add the oracle comparison for a non-Gaussian robust fit**

Compare DS1 with HC1 and DS8 with CR1 against the fixture's `cov` for those estimators, using the band measured in Task 4. The Gaussian equivalence proves the algebra; this proves the score formula is right for a family where `link_deriv` and `variance` are not both 1.

- [ ] **Step 6: Run, then commit**

```bash
git add test/cpp/test_glm_fit.cpp
git commit -m "glm: oracle the robust covariance on a binomial and a clustered fit (#22)"
```

---

## Task 6: Separation and the no-variation response

**Files:**
- Modify: `src/glm_core.cpp` — the separation heuristic
- Modify: `src/include/glm_core.hpp` — document it
- Modify: `test/cpp/test_glm_fit.cpp`

**Interfaces:**
- Consumes: `fit_glm` from Task 4.
- Produces: `GlmResult::separation` is now set.

- [ ] **Step 1: Write the failing test**

```cpp
// DS6 is perfectly separated: a threshold on x1 predicts y exactly, so the
// likelihood has no interior maximum and the coefficients diverge. Without a
// flag the caller sees an enormous estimate beside an enormous standard error
// and nothing saying why — which is how separation reaches a published table.
statsduck::GlmOptions sg;
sg.family = statsduck::Family::kBinomial;
sg.link = statsduck::Link::kLogit;
const statsduck::GlmResult s = statsduck::fit_glm(kDs6Y, ds6_design(), sg);
check(s.ok, "a separated fit still returns a result, not a failure");
check(!s.converged, "it does not converge");
check(s.separation, "and the separation flag is set");
check(s.iterations == sg.max_iter, "it used every iteration");
check(std::isfinite(at(s.beta, 1)), "the reported coefficient is finite, if large");

// The flag is binomial-only. A poisson fit that fails to converge must not
// claim separation, because the heuristic has no meaning there.
statsduck::GlmOptions pg;
pg.family = statsduck::Family::kPoisson;
pg.link = statsduck::Link::kLog;
const statsduck::GlmResult p = statsduck::fit_glm(kDs3Y, ds3_design(), pg, nullptr, &kDs3Offset);
check(p.ok && p.converged, "the poisson fixture converges");
check(!p.separation, "and never reports separation");

// A well-behaved binomial fit does not report separation either.
const statsduck::GlmResult ok1 = statsduck::fit_glm(kDs1Y, ds1_design(), sg);
check(ok1.converged && !ok1.separation, "a clean logistic fit is not separated");
```

- [ ] **Step 2: Run it to see it fail**

Expected: the `s.separation` check fails, because the field is still always false.

- [ ] **Step 3: Implement the heuristic**

`separation = true` when all three hold: the family is binomial, the fit did not converge within `max_iter`, and at least one fitted value lies within 1e-10 of 0 or 1. Always false for the other three families.

Document in the header that this is a heuristic: it fires on perfect separation and can also fire on severe near-separation, which it cannot distinguish. Say why it exists rather than leaving a reader to infer it.

- [ ] **Step 4: Run, then commit**

```bash
git add src/glm_core.cpp src/include/glm_core.hpp test/cpp/test_glm_fit.cpp
git commit -m "Add the binomial separation heuristic to glm_fit (#22)"
```

- [ ] **Step 5: Write the no-variation test (Review Focus item 4)**

```cpp
// A response with no variation. For binomial this is separation with no
// information at all; for poisson, all-zero counts drive mu to zero and the
// log link to minus infinity. Either outcome is acceptable — a failed fit, or
// a finite result with converged and separation set honestly — but a NaN in a
// field that reads as an estimate is not.
const std::vector<double> all_zero(kDs1Y.size(), 0.0);
const statsduck::GlmResult bz = statsduck::fit_glm(all_zero, ds1_design(), sg);
if (bz.ok) {
    for (std::size_t j = 0; j < bz.k; j++) {
        const double b = at(bz.beta, j);
        check(!std::isnan(b), "an all-zero binomial response gives no NaN coefficient");
    }
    check(std::isfinite(bz.deviance), "and a finite deviance");
    check(!bz.converged || bz.separation || std::isfinite(at(bz.std_error, 0)),
          "either it did not converge, or it is flagged, or the errors are finite");
}

const std::vector<double> all_one(kDs1Y.size(), 1.0);
const statsduck::GlmResult bo = statsduck::fit_glm(all_one, ds1_design(), sg);
check(bo.ok || !bo.error.empty(), "an all-one binomial response either fits or says why");

// All-zero poisson counts. mu -> 0 and eta -> -inf under a log link.
const statsduck::GlmResult pz = statsduck::fit_glm(all_zero, ds1_design(), pg);
if (pz.ok) {
    check(std::isfinite(pz.deviance), "an all-zero poisson deviance is finite");
    check(!std::isnan(at(pz.beta, 0)), "and the intercept is not NaN");
}

// A constant gaussian response is a legitimate fit with zero deviance.
const std::vector<double> all_five(kDs5Y.size(), 5.0);
statsduck::GlmOptions gg;
gg.family = statsduck::Family::kGaussian;
const statsduck::GlmResult gc = statsduck::fit_glm(all_five, ds5_design(), gg);
check(gc.ok, "a constant gaussian response fits");
check(close_rel(gc.deviance, 0.0, 1e-20), "with zero deviance");
check(close_rel(at(gc.beta, 0), 5.0, 1e-10), "an intercept of 5");
check(!gc.separation, "and no separation claim");
```

- [ ] **Step 6: Run, then commit**

```bash
git add test/cpp/test_glm_fit.cpp src/glm_core.cpp
git commit -m "glm: pin behaviour on a response with no variation (#22)"
```

---

## Task 7: The SQL aggregate

**Files:**
- Create: `src/include/glm_fit_function.hpp`, `src/glm_fit_function.cpp`
- Modify: `CMakeLists.txt`, `src/stats_duck_extension.cpp`
- Modify: `test/sql/function_docs.test`
- Create: `test/sql/glm_fit.test`

**Interfaces:**
- Consumes: `fit_glm`, `parse_family`, `parse_link`, `family_name`, `link_name`, `canonical_link`, `link_allowed` from Tasks 2–6; `lm_core::parse_vcov` and `vcov_name`; `statsduck::densify_clusters` from Task 1.
- Produces: the SQL aggregate `glm_fit(y, x [, family [, options]])` and `void RegisterGlmFit(ExtensionLoader &loader)`.

Read `src/lm_fit_function.cpp` completely before starting. It is the pattern for the accumulator, the bind data, the result type, `SetD`, and the `GROUP BY` state handling, and it is the file this one should look like.

- [ ] **Step 1: Write the failing SQL test**

Create `test/sql/glm_fit.test`. Start with the equivalence, because it is the claim a user is most likely to check:

```
# name: test/sql/glm_fit.test
# description: generalized linear models (glm_fit)
# group: [stats_duck]

require stats_duck

statement ok
CREATE TABLE ds AS SELECT * FROM (VALUES
  (1.0, 1.0, 2.0), (3.0, 2.0, 1.0), (5.0, 3.0, 4.0),
  (7.0, 4.0, 3.0), (9.0, 5.0, 6.0), (11.0, 6.0, 5.0)
) AS t(y, x1, x2);

# A gaussian fit with an identity link is ordinary least squares, so glm_fit
# and lm_fit must agree. This is the fastest way for a reader to convince
# themselves the new fitter is sound.
query II
SELECT printf('%.10f', (g).coefficients[2].estimate - (l).coefficients[2].estimate),
       printf('%.10f', (g).coefficients[2].std_error - (l).coefficients[2].std_error)
FROM (SELECT glm_fit(y, [x1, x2]) AS g, lm_fit(y, [x1, x2]) AS l FROM ds);
----
0.0000000000	0.0000000000
```

Then a worked example per family, and these cases, each as its own query:

- the options struct with its fields in a **different order** from the canonical one, giving the same fit;
- `{'link': 'probit'}` on a binomial fit, and `link` echoed back canonically;
- `{'vcov': 'HC1'}` changing the standard errors but not the estimates;
- `{'weights': n}` on grouped binomial data matching the ungrouped fit;
- `{'offset': log(exposure)}` on a Poisson fit;
- `{'intercept': false}` dropping the constant term;
- `family` omitted defaulting to `gaussian`, and `family` echoed canonically;
- `'fdr'`-style aliasing is not relevant here, but `vcov := 'cluster'` must echo as `CR1`, matching `lm_fit`;
- `converged`, `iterations` and `separation` on a separated binomial fixture.

Then the misuse cases, each expecting an error with its message fragment: an unknown family; an unknown link; a link the family does not allow; an unknown `vcov`; an unrecognized options field, with the valid names listed; a non-constant expression in `link`, `vcov` or `intercept`; `vcov := 'CR1'` with no cluster field; and a cluster field with a non-cluster `vcov`.

Then the data-shaped failures, each giving a NULL result rather than an error: binomial `y` of 2.0; Poisson `y` of −1.0; gamma `y` of 0.0; a negative weight; too few rows.

- [ ] **Step 2: Write the multi-group test**

Required, not optional. A `GROUP BY` where one group fails a domain check and the others are unaffected is the only shape that catches a failure leaking across groups, and every other query in the file returns a single row.

```
statement ok
CREATE TABLE ds_multi AS SELECT * FROM (VALUES
  ('a', 0.0, 1.0), ('a', 1.0, 2.0), ('a', 0.0, 3.0), ('a', 1.0, 4.0), ('a', 1.0, 5.0),
  ('b', 0.0, 1.0), ('b', 2.0, 2.0), ('b', 1.0, 3.0), ('b', 0.0, 4.0), ('b', 1.0, 5.0),
  ('c', 1.0, 1.0), ('c', 0.0, 2.0), ('c', 1.0, 3.0), ('c', 1.0, 4.0), ('c', 0.0, 5.0)
) AS t(grp, y, x1);

# Group b has a y of 2.0, which is outside binomial's [0, 1] support, so that
# group is NULL while a and c fit normally. A domain failure must not abort the
# query or contaminate its neighbours.
query III
SELECT grp, (g).n IS NULL, (g).converged
FROM (SELECT grp, glm_fit(y, [x1], 'binomial') AS g FROM ds_multi GROUP BY grp)
ORDER BY grp;
----
a	false	true
b	true	NULL
c	false	true
```

- [ ] **Step 3: Run it to see it fail**

Expected: a binder error, because `glm_fit` does not exist.

- [ ] **Step 4: Write the result type and the bind**

In `src/glm_fit_function.cpp`, define `GlmFitResultType()` with the comment `// Field order MUST match the writes in Finalize` and an index comment per field, in the spec's order: `coefficients`, `n`, `k`, `df_residual`, `deviance`, `null_deviance`, `dispersion`, `loglik`, `aic`, `iterations`, `converged`, `separation`, `family`, `link`, `has_intercept`, `vcov_type`, `n_clusters`, `rank`, `cov`. The coefficient row type is `term`, `estimate`, `std_error`, `z_statistic`, `p_value` — **`z_statistic`**, never `t_statistic`.

The bind reads the options struct **by field name**, iterating the struct's children and matching names, never by position: DuckDB's struct type is ordered, so `{'link': 'probit', 'vcov': 'HC1'}` and the same two fields reversed are different types with the same meaning, and both must work. An unrecognized name throws a `BinderException` listing the valid names — a silently ignored `{'weight': n}` would otherwise give an unweighted fit that looks correct. `link`, `vcov` and `intercept` are constant-folded at bind time and a non-constant expression there is a bind error; `weights`, `offset` and `cluster` are per-row and are kept as arguments.

- [ ] **Step 5: Write the accumulator and the finalize**

Buffer the rows as `lm_fit` does — the conventions doc is explicit that leverage-based and score-based estimators do not reduce to streaming moments, so the state holds the raw rows and the maths happens in `Finalize`. Drop a row when `y`, any design element, the weight, the offset or the cluster key is NULL, NaN or infinite, and **also when the prior weight is exactly 0**, per the ruling in Review Focus item 2. A design row whose length differs from the first seen fails the group.

Call `statsduck::densify_clusters` on the buffered keys, then `fit_glm`. A result with `ok = false` writes a NULL STRUCT for that group and never raises. Write every double through the `SetD` NaN-to-NULL helper, and set the `family`, `link` and `vcov_type` VARCHAR children on every row including a NULL one.

- [ ] **Step 6: Register the aggregate**

Three overloads — `(DOUBLE, LIST(DOUBLE))`, `(DOUBLE, LIST(DOUBLE), VARCHAR)` and `(DOUBLE, LIST(DOUBLE), VARCHAR, STRUCT(...))` — through `RegisterDocumented` with `OnCreateConflict::ALTER_ON_CONFLICT`, a description, a parameter list and an example. The description states the four families, that the binomial response is a proportion with `weights` as the trial count, and that `z_statistic` uses a normal reference for every family.

- [ ] **Step 7: Build and run the new file**

Check the build's exit status, then run just `test/sql/glm_fit.test`.

- [ ] **Step 8: Update the documentation metadata net**

`test/sql/function_docs.test` has three `IN (...)` lists and a count assertion. Add `glm_fit` to **all three** in alphabetical position and raise the count by one. **Verify by running the file and reading the count**, not by reasoning about it — in #20 this was missed and then reported as unaffected.

- [ ] **Step 9: Run both suites in full**

Expected: the SQL suite passes with its assertion count risen by the new file's, and the C++ suites are unchanged.

- [ ] **Step 10: Write the `lin_hyp` and `emmeans` tests, which are this issue's whole point**

Both helpers are deliberately stateless so they serve any fit, and until `glm_fit` there was only one fitter to serve. Nothing else in this plan checks that the claim holds, so these two queries are the ones that justify the issue.

```
# A Type III test of one term on a logistic fit. lin_hyp needs no change to
# serve a GLM: it takes the coefficients and the covariance, whatever produced
# them. sigma2 is 1 because glm_fit's cov is already on the right scale, and
# df_residual is the denominator df.
query II
SELECT (h).df_num, printf('%.4f', (h).chisq)
FROM (SELECT lin_hyp([[0, 1]],
                     list_transform(g.coefficients, c -> c.estimate),
                     g.cov, 1.0, g.df_residual) AS h
      FROM (SELECT glm_fit(y, [x1], 'binomial') AS g FROM ds_bin));
----
1	<the value from the run>

# The joint Wald statistic from lin_hyp over every slope must equal the
# statistic from glm_fit's own coefficients, which is the same arithmetic
# reached by two routes.
query I
SELECT printf('%.8f', (h).chisq - pow((g).coefficients[2].estimate / (g).coefficients[2].std_error, 2))
FROM (SELECT glm_fit(y, [x1], 'binomial') AS g,
             lin_hyp([[0, 1]], list_transform(glm_fit(y, [x1], 'binomial').coefficients, c -> c.estimate),
                     glm_fit(y, [x1], 'binomial').cov, 1.0, 1.0) AS h
      FROM ds_bin);
----
0.00000000

# emmeans over a cell-means logistic fit gives marginal means on the LINK
# scale, with Tukey-adjusted pairwise differences. The link scale is the
# correct and only thing emmeans can report, because it is given beta and cov
# and nothing about the link — so the README must say so.
query III
SELECT (e).k_compare, printf('%.4f', (e).means[1].estimate), (e).diffs[1].p_adjusted IS NOT NULL
FROM (SELECT emmeans([[1,0,0],[0,1,0],[0,0,1]],
                     list_transform(g.coefficients, c -> c.estimate),
                     g.cov, g.df_residual) AS e
      FROM (SELECT glm_fit(y, [g1, g2, g3], 'binomial', {'intercept': false}) AS g FROM ds_cells));
----
3	<the value from the run>	true
```

Replace each `<the value from the run>` with what the query actually prints, and check the first against a hand computation: with one slope the Wald chi-square is `(estimate/std_error)²`, which the second query asserts directly.

**One documentation consequence to carry into Task 8:** `emmeans` on a GLM reports means on the **link** scale, not the response scale, because it receives only `beta` and `cov` and knows nothing about the link. A user expecting probabilities from a logistic fit will get log-odds. Say so in the README's `glm_fit` section and in `emmeans`'s, since neither currently mentions the other.

- [ ] **Step 11: Commit**

```bash
git add src/include/glm_fit_function.hpp src/glm_fit_function.cpp CMakeLists.txt \
        src/stats_duck_extension.cpp test/sql/glm_fit.test test/sql/function_docs.test
git commit -m "Expose glm_fit as a SQL aggregate (#22)"
```

- [ ] **Step 12: Write the zero-weight test (Review Focus item 2)**

```
# A prior weight of exactly 0 means the row influences no estimate, so it is
# dropped at accumulation exactly like a NULL and is absent from n. Counting it
# would inflate df_residual and deflate the dispersion.
query III
SELECT (a).n, (b).n, printf('%.10f', (a).deviance - (b).deviance)
FROM (
  SELECT glm_fit(y, [x1], 'gaussian', {'weights': w0}) AS a,
         glm_fit(y, [x1], 'gaussian') AS b
  FROM (SELECT y, x1, CASE WHEN x1 > 3 THEN 0.0 ELSE 1.0 END AS w0 FROM ds)
);
----
3	6	0.0000000000
```

The first fit sees three rows and the second six, and their deviances differ — so assert `n` and compare each against a separate fit of the surviving rows only, rather than against each other. Write that second query too: a fit with zero weights on the rows where `x1 > 3` must equal a fit of `WHERE x1 <= 3` with no weights at all, in `n`, `deviance`, every coefficient and every standard error. That equivalence is the real content of this item.

- [ ] **Step 13: Run, then commit**

```bash
git add test/sql/glm_fit.test src/glm_fit_function.cpp
git commit -m "glm_fit: drop zero-weight rows at accumulation (#22)"
```

- [ ] **Step 14: Write the saturating-offset test (Review Focus item 5)**

```
# A poisson offset of 1000 makes mu = e^1000, which is +Inf before any fitting
# happens. The domain check has to catch it, so the group is NULL rather than
# Inf propagating into the coefficients and out to the user.
query I
SELECT (g).n IS NULL
FROM (SELECT glm_fit(y, [x1], 'poisson', {'offset': 1000.0}) AS g FROM ds);
----
true

# A merely large offset is a legitimate fit, so the check must not be a blunt
# magnitude threshold.
query II
SELECT (g).converged, (g).n
FROM (SELECT glm_fit(y, [x1], 'poisson', {'offset': 2.0}) AS g FROM ds);
----
true	6

# A non-finite offset is dropped as missing data, like a NULL.
query I
SELECT (g).n
FROM (SELECT glm_fit(y, [x1], 'poisson',
                     {'offset': CASE WHEN x1 > 5 THEN 'inf'::DOUBLE ELSE 0.0 END}) AS g FROM ds);
----
5
```

The first case needs a decision the implementer must make and record: either the offset's magnitude is checked up front, or the first `link_inv` produces a non-finite `mu` and the fit fails on that. Either is acceptable; a reported result containing Inf is not.

- [ ] **Step 15: Run, then commit**

```bash
git add test/sql/glm_fit.test src/glm_core.cpp src/glm_fit_function.cpp
git commit -m "glm_fit: a saturating offset is a failed fit, not an infinity (#22)"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`, `docs/kernel_api.md`, `docs/fitter_conventions.md`, `CHANGELOG.md`

- [ ] **Step 1: Add the README row and section**

Add `glm_fit` to the function table, marked **aggregate** the way the `lm_fit` row is. Then a section after the `lm_fit` one with a worked query per family and its real output, the result-field table, and these points:

- the binomial response is a **proportion** with `weights` as the trial count, so 7 of 10 is `y = 0.7, weights = 10`, and a count passed as `y` fails the domain check;
- **`z_statistic` uses a normal reference for all four families.** R and statsmodels switch to a t reference for Gaussian and gamma, where dispersion is estimated, so our p-values are very slightly smaller there. State it plainly and say that `df_residual` is reported for a caller who wants the t reference;
- `converged = false` returns the best iterate rather than NULL, and `iterations` counts outer IRLS passes;
- `separation` is a heuristic, binomial-only, and can fire on severe near-separation as well as perfect separation;
- `deviance` and `null_deviance` replace R² and F, neither of which generalizes;
- the options struct's fields are read by name, so field order does not matter, and an unknown field is an error.

- [ ] **Step 2: Add the kernel_api section**

Document `glm_core.hpp` in the shape of the `lm_core.hpp` section: the signature, a symbol table, and the conventions. Cover the fixed aliased-column set and why it is fixed, the score-residual sandwich and that it reduces to `lm_core`'s at Gaussian identity, the `link_inv` clamp and its constant, and the dispersion rule per family. Add `cluster_ids.hpp` to the layer list and to the testing list.

- [ ] **Step 3: Add the options-struct convention to fitter_conventions**

Under the input conventions, state the rule this issue establishes and that binds future fitters: when a fitter's optional inputs exceed what positional overloads can disambiguate — in particular when two of them are per-row values of the same type — they go in one trailing STRUCT whose fields are read by name, with an unrecognized name a bind error. Say that `lm_fit` keeps its positional overloads because a released position never changes meaning, and that a struct overload could be added to it later. Move `glm_fit` from the page's "next comes" list into the set of functions it binds.

- [ ] **Step 4: Add the CHANGELOG entry**

Cover `glm_fit`, the four families and their links, the options struct, and the robust covariance. State the `z_statistic` deviation from R rather than omitting it. Do not claim the separation flag detects separation exactly — it is a heuristic, and over-claiming in a CHANGELOG is what had to be corrected across four files in #63.

- [ ] **Step 5: Run every documented example**

Paste each SQL example from the README into the CLI and compare its output to what the documentation claims. A documented example that does not run is worse than no example.

```bash
cmake --build "C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release" --config Release --target shell
echo "shell build exit: $?"
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/Release/duckdb.exe" -unsigned \
  -c "LOAD '.../build/release/extension/stats_duck/stats_duck.duckdb_extension'; <query>"
```

Build the **`shell`** target, not just `unittest`: `duckdb.exe` statically links the extension, so a shell built before `glm_fit` existed will silently register nothing and a `LOAD` of a fresh build will not help. That cost real time on the #21 branch — see `notes/engineering/2026-10-verifying-against-a-stale-binary.md`.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/kernel_api.md docs/fitter_conventions.md CHANGELOG.md
git commit -m "Document glm_fit (#22)"
```

---

## Done when

- `bash scripts/run-cpp-tests.sh` passes with the pre-existing suites' counts unchanged and three new suites green.
- The SQL suite passes in full, with `function_docs.test`'s count raised by one and verified by running it.
- Gaussian with an identity link agrees with `lm_fit` to 1e-12 on every covariance estimator, classical and robust.
- The measured coefficient agreement against statsmodels is recorded in `test/cpp/test_glm_fit.cpp` alongside the band it set.
- Every README example has been run against a freshly built `shell` target and matches its documented output.
- No commit contains the `duckdb` submodule.
