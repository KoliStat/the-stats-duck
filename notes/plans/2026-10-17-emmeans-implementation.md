# emmeans Implementation Plan (#21)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `emmeans(L, beta, cov, df [, alpha [, adjust]])`, which reports estimated marginal means for an existing fit with their standard errors and confidence limits, every pairwise difference between them, and a multiplicity adjustment for that family including Tukey.

**Architecture:** The kernel forms θ = L·β and C = L·cov·Lᵀ once and reads every reported number off θ and C, so nothing is inverted. Tukey needs the studentized range distribution, which this plan adds to `distributions.hpp` and exposes as the SQL scalars `ptukey` and `qtukey`. Three pieces currently private to `lin_hyp` and `adjust_p` move to shared headers first, so that one implementation of each convention serves both consumers.

**Tech Stack:** C++17, the DuckDB extension API, the in-repo `linalg` and `distributions` kernels, Eigen only behind `linalg.cpp`, sqllogictest for the SQL layer, and scipy plus statsmodels as offline oracles.

**Spec:** `notes/plans/2026-10-17-emmeans-design.md` — read it alongside this plan. Where the two disagree, the spec is right and this plan is wrong.

## Global Constraints

Every task's requirements implicitly include these. Values are copied from the spec.

- **Apache 2.0 only.** R's `ptukey` is GPL. Do not port, adapt, translate or consult-and-paraphrase that code, in whole or in part. Implement from the published mathematics. The oracle is `scipy.stats.studentized_range`, BSD licensed, offline only.
- **The oracle is never a build or CI dependency.** Generators run in a throwaway virtualenv and the goldens are committed.
- **Two namespaces, one letter apart.** `distributions.hpp` is `stats_duck`; the kernel is `statsduck`. Calls across that boundary need qualifying. `src/lin_hyp.cpp` already does this.
- **`q` means two different things.** In the studentized range it is the quantile; in `emmeans` it is the number of marginal means. Name them `q` and `n_means` respectively in code that touches both.
- **Determinism.** Fixed loop orders, fixed quadrature node counts, no hashing, no random numbers. Identical inputs give bit-identical results on every platform.
- **No default-hash `std::string`-keyed `unordered_map` or `unordered_set`** in extension code. It breaks the wasm build at load time. `scripts/check-wasm-string-hash.sh` enforces this in CI.
- **NaN becomes SQL NULL at the vector layer**, through the `SetD` helper pattern. ±Inf passes through unchanged.
- **Result STRUCT field order is contract and append-only.** The type function carries `// Field order MUST match the writes in Finalize` with an index comment per field.
- **`tol` for negative variances is 1e-12**, relative to `max_j C_jj` for a mean and to `C_ii + C_jj` for a difference.
- **`adjust` defaults to `tukey`.** The canonical name is echoed, so `fdr` comes back as `BH`.
- **Every tolerance band carries a written reason** next to it.
- **Stage files explicitly. Never `git add -A`.** The `duckdb` submodule is expected to be dirty — that is the applied fmt patch, recorded in `patches/`, and it must stay out of every commit.
- **Absolute paths for the SQL suite.** `unittest.exe "test/sql/*"` with a relative path resolves against the DuckDB source tree and runs its 4000+ core tests instead.
- **Plain English in comments, commit messages and documentation.** Short linear sentences, one idea each, no buzzwords, no engineering slang.

Commands, with absolute paths:

```bash
# Fast C++ kernel suites, no DuckDB
bash scripts/run-cpp-tests.sh

# Full extension build
cmake --build "C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release" --config Release --target unittest

# SQL suite
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/test/Release/unittest.exe" \
  "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"
```

Baseline before any change: C++ 736 checks in 4 suites, SQL 2206 assertions in 64 files, zero failures.

## Review Focus

Five input classes the spec implies, which a reasonable person will meet and which no task's tests would otherwise exercise. Most likely first. Each has its test added to the task that owns the code.

1. **Two identical rows in `L`.** A user who builds a grid with a repeated level gets C_ii = C_jj = C_ij, so the pairwise variance is exactly zero by cancellation. Expected: estimate 0, `std_error` 0, NULL t and p, and the pair still present — not a NULL pair, not a negative variance, not a crash. Test in Task 5, Step 13.
2. **`cov` scaled by a large factor.** Units alone moved `df_num` in #20. Here the t statistics and p-values must be unchanged by scaling `cov` by 1e12, and so must `k_compare`, because the variance tolerance is relative. Test in Task 5, Step 15.
3. **Tukey requested with fewer than two estimable means.** `ptukey` is undefined for k < 2. The code must not call it. Expected: no comparisons, `k_compare` reports what it found, adjustment is a no-op, no NaN leaking into a reported field. Test in Task 5, Step 17.
4. **`ptukey` and `qtukey` outside their domain.** k < 2, non-integer or non-positive ν, q ≤ 0, p at exactly 0 or 1, and NaN in any argument. Expected: the documented edge value or NaN, never a hang and never a silently wrong number. Test in Task 3, Step 11.
5. **A fractional, tiny `df` together with an extreme `alpha`.** `df = 0.5` with `alpha = 1e-10` drives `StudentTQuantile` into its tail. Expected: a finite limit or ±Inf, reached in bounded time, never a hang and never a NaN that reads as a valid limit. Test in Task 5, Step 19.

---

## Task 1: Extract the two DuckDB-free helpers

Both helpers are needed by `emmeans` and both exist today as private code. Extract them first, with their own standalone tests, so a regression attaches to the refactor that caused it rather than to new work. This task touches no DuckDB code and verifies in the fast cycle.

**Files:**
- Create: `src/include/estimability.hpp`
- Create: `src/include/p_adjust.hpp`
- Modify: `src/lin_hyp.cpp:59-80` — onto `estimability.hpp`
- Modify: `scripts/run-cpp-tests.sh` — two new suites
- Test: `test/cpp/test_estimability.cpp`, `test/cpp/test_p_adjust.cpp`

**Interfaces:**
- Consumes: `linalg::Mat` from `src/include/linalg.hpp`.
- Produces, in namespace `statsduck`:
  ```cpp
  // estimability.hpp
  std::vector<bool> aliased_columns(const linalg::Mat &V);
  bool row_touches_aliased(const double *row, std::size_t k,
                           const std::vector<bool> &aliased);
  linalg::Mat zero_aliased(const linalg::Mat &V, const std::vector<bool> &aliased);
  std::vector<double> zero_aliased(const std::vector<double> &beta,
                                   const std::vector<bool> &aliased);

  // p_adjust.hpp
  enum class AdjustMethod { BONFERRONI, HOLM, HOCHBERG, BENJAMINI_HOCHBERG,
                            BENJAMINI_YEKUTIELI, NONE };
  bool parse_adjust(const std::string &name, AdjustMethod &out);
  const char *adjust_name(AdjustMethod m);
  void adjust_p(const std::vector<double> &p_in, const std::vector<bool> &valid,
                AdjustMethod method, std::vector<double> &p_out);
  ```

- [ ] **Step 1: Write the failing estimability test**

Create `test/cpp/test_estimability.cpp`. Follow the harness style of `test/cpp/test_lin_hyp.cpp`: a `check(bool, const char*)` counter, a `main` that returns non-zero on any failure, and a printed count. Use a safe accessor rather than indexing a vector that a failing call may have left empty — that defect appeared three times in #33, #36 and #20.

```cpp
// A 3x3 covariance whose middle coefficient is aliased.
linalg::Mat V(3, 3);
for (std::size_t i = 0; i < 3; i++)
    for (std::size_t j = 0; j < 3; j++)
        V(i, j) = 1.0 + static_cast<double>(i * 3 + j);
V(1, 1) = std::numeric_limits<double>::quiet_NaN();

const std::vector<bool> a = statsduck::aliased_columns(V);
check(a.size() == 3, "aliased_columns returns one flag per column");
check(!a[0] && a[1] && !a[2], "only the NaN diagonal is aliased");

// A row with no weight on the aliased column is still estimable.
const double safe[3] = {1.0, 0.0, -1.0};
const double unsafe[3] = {1.0, 1e-300, 0.0};
check(!statsduck::row_touches_aliased(safe, 3, a), "zero weight does not touch");
check(statsduck::row_touches_aliased(unsafe, 3, a), "any non-zero weight touches");

// Zeroing clears the aliased row and column and leaves the rest alone.
const linalg::Mat Vz = statsduck::zero_aliased(V, a);
check(Vz(1, 1) == 0.0 && Vz(0, 1) == 0.0 && Vz(1, 2) == 0.0, "aliased row and column zeroed");
check(Vz(0, 0) == V(0, 0) && Vz(2, 2) == V(2, 2), "other entries untouched");
check(std::isfinite(Vz(1, 1)), "no NaN survives in the working copy");

const std::vector<double> b = {5.0, 7.0, 9.0};
const std::vector<double> bz = statsduck::zero_aliased(b, a);
check(bz.size() == 3 && bz[0] == 5.0 && bz[1] == 0.0 && bz[2] == 9.0, "beta aliased entry zeroed");

// A fully finite covariance aliases nothing.
linalg::Mat F(2, 2);
F(0, 0) = 1.0; F(0, 1) = 0.0; F(1, 0) = 0.0; F(1, 1) = 1.0;
const std::vector<bool> none = statsduck::aliased_columns(F);
check(!none[0] && !none[1], "a finite covariance aliases nothing");
```

- [ ] **Step 2: Register the suite and run it to see it fail**

In `scripts/run-cpp-tests.sh`, add below the existing `build_and_run` calls:

```bash
build_and_run test_estimability \
  "$ROOT/test/cpp/test_estimability.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_estimability`, because `estimability.hpp` does not exist. The other four suites still pass.

- [ ] **Step 3: Write `src/include/estimability.hpp`**

Header-only, `#pragma once`, namespace `statsduck`, including only `linalg.hpp`, `<cmath>`, `<cstddef>` and `<vector>`. Lift the logic from `src/lin_hyp.cpp:59-80` unchanged in behaviour. A column is aliased when its diagonal entry is not finite. `row_touches_aliased` returns true when any aliased column has a non-zero coefficient, using `!= 0.0` rather than a tolerance, because a weight is either present or not and the caller chose it. The file-level comment states that this is the convention #36 introduced, and why zeroing is necessary: IEEE makes `0 · NaN = NaN`, so an aliased entry cannot simply be multiplied by zero.

- [ ] **Step 4: Run the suite to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: all five suites pass. The new one reports its own count; the other four are unchanged at 73, 511, 113 and 39.

- [ ] **Step 5: Refactor `src/lin_hyp.cpp` onto the helper**

Replace the inline aliased-column loop and the two zeroing loops with calls to `aliased_columns`, `row_touches_aliased` and the two `zero_aliased` overloads. Behaviour must not change: `lin_hyp` still treats *any* row touching an aliased column as making the whole hypothesis non-estimable, returning `estimable = false` for the result rather than per row. Keep the existing comment explaining the IEEE reason, or move it to the header and leave a one-line pointer.

- [ ] **Step 6: Run the suite to confirm nothing moved**

Run: `bash scripts/run-cpp-tests.sh`
Expected: `test_lin_hyp` reports exactly `113 checks, 0 failures`. A different number means the refactor changed behaviour and must be reverted rather than accommodated.

- [ ] **Step 7: Commit**

```bash
git add src/include/estimability.hpp src/lin_hyp.cpp test/cpp/test_estimability.cpp scripts/run-cpp-tests.sh
git commit -m "Extract the aliased-coefficient rule into estimability.hpp (#21)"
```

- [ ] **Step 8: Write the failing p_adjust test**

Create `test/cpp/test_p_adjust.cpp`. Each C++ test file in this repository is standalone and compiled on its own, so give this file its own `check`, `close` and `at` helpers rather than expecting them from elsewhere. Copy them from `test/cpp/test_lin_hyp.cpp`.

Use the vector already pinned by `test/sql/adjust_p.test`, so the extraction is held to released behaviour rather than to a fresh opinion:

```cpp
const std::vector<double> p = {0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205};
const std::vector<bool> all_valid(8, true);
std::vector<double> out;

statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::BONFERRONI, out);
const double bonf[8] = {0.008, 0.064, 0.312, 0.328, 0.336, 0.48, 0.592, 1.0};
for (std::size_t i = 0; i < 8; i++)
    check(close(at(out, i), bonf[i], 1e-12), "bonferroni matches the released value");

statsduck::adjust_p(p, all_valid, statsduck::AdjustMethod::HOLM, out);
const double holm[8] = {0.008, 0.056, 0.234, 0.234, 0.234, 0.234, 0.234, 0.234};
for (std::size_t i = 0; i < 8; i++)
    check(close(at(out, i), holm[i], 1e-4), "holm matches the released value");

// The valid mask: an invalid slot comes back NaN and does not change the ranks.
std::vector<bool> mask(8, true);
mask[2] = false;
statsduck::adjust_p(p, mask, statsduck::AdjustMethod::BONFERRONI, out);
check(std::isnan(at(out, 2)), "an invalid slot is NaN");
check(close(at(out, 0), 0.007, 1e-12), "the multiplier counts only valid entries");

// Parsing, including the alias and the canonical echo.
statsduck::AdjustMethod m;
check(statsduck::parse_adjust("fdr", m) && m == statsduck::AdjustMethod::BENJAMINI_HOCHBERG,
      "fdr parses as Benjamini-Hochberg");
check(std::string(statsduck::adjust_name(m)) == "BH", "fdr echoes as BH");
check(statsduck::parse_adjust("none", m) && m == statsduck::AdjustMethod::NONE, "none parses");
check(!statsduck::parse_adjust("tukey", m), "tukey is not a p-list method");
check(!statsduck::parse_adjust("Bonferroni", m), "parsing is case sensitive, as adjust_p already is");
```

The holm band is 1e-4 because `test/sql/adjust_p.test` records those values rounded to four places; the bonferroni values are exact in that file, so they get 1e-12. The masked bonferroni expectation is `0.001 × 7 = 0.007`: seven valid entries, not eight. If the implementation disagrees with any value here, check `test/sql/adjust_p.test` before changing either — that file is the released contract, and released behaviour wins.

- [ ] **Step 9: Register the suite and run it to see it fail**

Add to `scripts/run-cpp-tests.sh`:

```bash
build_and_run test_p_adjust \
  "$ROOT/test/cpp/test_p_adjust.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_p_adjust`.

- [ ] **Step 10: Write `src/include/p_adjust.hpp`**

Move the `AdjustMethod` enum and the body of `AdjustInPlace` from `src/adjust_p_function.cpp:33-160` into the header, renamed `adjust_p`, in namespace `statsduck`. Replace `idx_t` with `std::size_t`. The file includes only `<cmath>`, `<cstddef>`, `<limits>`, `<numeric>`, `<string>`, `<vector>` and `<algorithm>` — no DuckDB header, because the `emmeans` kernel must be able to include it.

Add two functions the existing file does not have:

- `parse_adjust(name, out) -> bool`, in the shape of `lm_core::parse_vcov`: it returns false for an unknown name rather than throwing, so the caller chooses the exception type and the message. It accepts exactly the names `src/adjust_p_function.cpp:42-48` accepts, with the same case sensitivity, and it does **not** accept `tukey`, which cannot be computed from p-values alone.
- `adjust_name(m) -> const char *`, returning the canonical name, so `fdr` maps to `BH`. `emmeans` echoes this.

Keep the behaviour of the moved code byte-identical. This step is a move, not a rewrite.

- [ ] **Step 11: Run the suite to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: six suites pass, with `73`, `511`, `113`, `39` unchanged.

- [ ] **Step 12: Commit**

```bash
git add src/include/p_adjust.hpp test/cpp/test_p_adjust.cpp scripts/run-cpp-tests.sh
git commit -m "Extract the p-value adjustment methods into p_adjust.hpp (#21)"
```

---

## Task 2: Move the DuckDB-coupled consumers onto the shared code

`adjust_p_function.cpp` keeps its binding and loses its private copy of the maths. The nested-list reader comes out of `lin_hyp_function.cpp` so `emmeans_function.cpp` does not re-implement an offset walk. This task needs the full build, which is why it is separate from Task 1.

**Files:**
- Create: `src/include/list_read.hpp`
- Modify: `src/adjust_p_function.cpp` — onto `p_adjust.hpp`
- Modify: `src/lin_hyp_function.cpp:56-85` — onto `list_read.hpp`
- Test: existing `test/sql/adjust_p.test` and `test/sql/lin_hyp.test`, both of which must pass unchanged

**Interfaces:**
- Consumes: `statsduck::adjust_p`, `statsduck::parse_adjust`, `statsduck::adjust_name` from Task 1.
- Produces, in namespace `duckdb` (the file is DuckDB-coupled by design):
  ```cpp
  // list_read.hpp — returns false when the shape is wrong, and fills `error`.
  bool ReadDoubleList(Vector &v, idx_t row, std::vector<double> &out);
  bool ReadDoubleMatrix(Vector &v, idx_t row, statsduck::linalg::Mat &out,
                        std::string &error);
  ```

- [ ] **Step 1: Write `src/include/list_read.hpp`**

Lift `ReadDoubles` and the three-level unwrap from `src/lin_hyp_function.cpp:56-85`. The file-level comment states why it exists: the nested read walks two levels of offsets, and a nested list arriving from `list_transform`, a join or a dictionary-encoded vector can have rows that are out of order or not laid out back to back, so the walk must never assume layout.

`ReadDoubleMatrix` returns false and sets `error` when the rows are ragged, naming the first offending row and both widths. A NULL element anywhere, or a NULL row, returns false with an error saying so — the caller turns that into a NULL result rather than an exception, per the spec.

- [ ] **Step 2: Move `lin_hyp_function.cpp` onto it**

Replace the inline unwrap with `ReadDoubleMatrix` and `ReadDoubleList`. Keep every existing error message exactly as it is: `test/sql/lin_hyp.test` asserts on their text.

- [ ] **Step 3: Move `adjust_p_function.cpp` onto `p_adjust.hpp`**

Delete the private enum and `AdjustInPlace`. Call `statsduck::parse_adjust` and throw the file's existing `InvalidInputException` with its existing message when parsing fails, so the released error text does not change. Call `statsduck::adjust_p` for the arithmetic.

- [ ] **Step 4: Build**

Run the build command from Global Constraints.
Expected: a clean build. A link error here usually means a moved function kept `static` in the header without `inline`.

- [ ] **Step 5: Run the SQL suite**

Run the SQL command from Global Constraints.
Expected: `All tests passed (2206 assertions in 64 test cases)`. Any failure in `adjust_p.test` or `lin_hyp.test` means a behaviour change in a pure move, which must be reverted rather than accommodated.

- [ ] **Step 6: Run the C++ suites too**

Run: `bash scripts/run-cpp-tests.sh`
Expected: six suites pass, unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/include/list_read.hpp src/lin_hyp_function.cpp src/adjust_p_function.cpp
git commit -m "Share the nested-list reader and the p-adjust kernel (#21)"
```

---

## Task 3: The studentized range distribution

The distribution behind Tukey, in `distributions.hpp` with the rest of the CDFs. Independently verifiable, and `emmeans` must not be started until this task's tests pass, because debugging a wrong adjusted p-value is far harder when either layer could be the cause.

**Files:**
- Modify: `src/include/distributions.hpp` — append a studentized range section
- Modify: `scripts/run-cpp-tests.sh` — one new suite
- Create: `test/cpp/gen_tukey_fixtures.py`, `test/cpp/tukey_fixtures.hpp` (generated), `test/cpp/test_tukey.cpp`

**Interfaces:**
- Consumes: `stats_duck::NormalCDF`, `stats_duck::StudentTCDF`, `stats_duck::ChiSquareQuantile`, all already in `distributions.hpp`.
- Produces, in namespace `stats_duck`:
  ```cpp
  double StudentizedRangeCDF(double q, double k, double df);
  double StudentizedRangeQuantile(double p, double k, double df);
  ```

- [ ] **Step 1: Write the generator**

Create `test/cpp/gen_tukey_fixtures.py`, following `test/cpp/gen_lin_hyp_fixtures.py` in structure: a module docstring stating the oracle and its version, a printed header file, and self-checking identities that fail loudly if the oracle's conventions move.

```python
"""Goldens for the studentized range distribution.

Oracle: scipy.stats.studentized_range (scipy >= 1.7), BSD licensed, run
offline. Never a build or CI dependency.

The grid is k in {2,3,4,5,6,10,20} and nu in {1,2,5,10,30,60,120,1000}.
For each pair the CDF is sampled at the oracle's own 0.5, 0.9, 0.95 and
0.99 quantiles, so the sample points sit where the distribution has mass
rather than at arbitrary abscissae.

Self-checks, which must hold before anything is written:
  1. at k = 2, cdf(q) == 2 * t.cdf(q / sqrt(2), nu) - 1
  2. ppf(cdf(q)) == q
  3. cdf is increasing in q and decreasing in k
"""
```

The generator writes `test/cpp/tukey_fixtures.hpp` as a flat array of `{k, nu, q, cdf}` plus a second array of `{k, nu, p, quantile}`. Keep 17 significant digits.

- [ ] **Step 2: Run the generator and commit the fixtures**

```bash
python -m venv /tmp/tukeyenv && /tmp/tukeyenv/bin/pip install scipy
/tmp/tukeyenv/bin/python test/cpp/gen_tukey_fixtures.py > test/cpp/tukey_fixtures.hpp
```

Expected: the four self-checks print as passing, and the header is written. If self-check 1 fails, stop: either scipy's convention has moved or the identity was mis-stated, and both need resolving before any goldens are trusted.

- [ ] **Step 3: Write the failing test, identities first**

Create `test/cpp/test_tukey.cpp`. Put the identity tests before the fixture tests, because a failure in an identity tells you the quadrature is wrong while a fixture failure only tells you a number moved.

```cpp
// Identity 1: at k = 2 the studentized range is the t distribution exactly.
// Q = sqrt(2)*|T|, so P(Q <= q) = 2*F_t(q/sqrt(2)) - 1. This anchors the
// whole double integral against StudentTCDF, which is already validated.
const double nus[] = {1.0, 2.0, 5.0, 10.0, 30.0, 120.0, 1000.0};
for (const double nu : nus) {
    for (const double q : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}) {
        const double want = 2.0 * stats_duck::StudentTCDF(q / std::sqrt(2.0), nu) - 1.0;
        const double got = stats_duck::StudentizedRangeCDF(q, 2.0, nu);
        check(close(got, want, 1e-8), "k = 2 reduces to the t distribution");
    }
}

// Identity 2: as nu grows the outer integral collapses to the normal range,
// and at k = 2 that is 2*Phi(q/sqrt(2)) - 1.
for (const double q : {1.0, 2.0, 4.0, 6.0}) {
    const double want = 2.0 * stats_duck::NormalCDF(q / std::sqrt(2.0)) - 1.0;
    const double got = stats_duck::StudentizedRangeCDF(q, 2.0, 1e9);
    check(close(got, want, 1e-7), "large nu reduces to the normal range");
}

// Identity 3: the quantile inverts the CDF.
for (const double k : {2.0, 3.0, 5.0, 10.0}) {
    for (const double nu : {5.0, 20.0, 60.0}) {
        for (const double p : {0.5, 0.9, 0.95, 0.99}) {
            const double q = stats_duck::StudentizedRangeQuantile(p, k, nu);
            check(close(stats_duck::StudentizedRangeCDF(q, k, nu), p, 1e-8),
                  "quantile inverts the CDF");
        }
    }
}

// Identity 4: monotone increasing in q, decreasing in k.
for (const double nu : {5.0, 30.0}) {
    double prev = -1.0;
    for (const double q : {0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0}) {
        const double cur = stats_duck::StudentizedRangeCDF(q, 4.0, nu);
        check(cur > prev, "CDF increases in q");
        prev = cur;
    }
    double last = 2.0;
    for (const double k : {2.0, 3.0, 4.0, 6.0, 10.0, 20.0}) {
        const double cur = stats_duck::StudentizedRangeCDF(3.0, k, nu);
        check(cur < last, "CDF decreases in k");
        last = cur;
    }
}
```

Then the fixture loops over `tukey_fixtures.hpp`, with the bands from the spec: 1e-8 absolute for the CDF and 1e-6 relative for the quantile. Write the reason for each band as a comment above it, in the words the spec uses.

- [ ] **Step 4: Register the suite and run it to see it fail**

Add to `scripts/run-cpp-tests.sh`:

```bash
build_and_run test_tukey \
  "$ROOT/test/cpp/test_tukey.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_tukey`, since the two functions do not exist.

- [ ] **Step 5: Add the Gauss-Legendre helper**

In `src/include/distributions.hpp`, add a private helper that returns nodes and weights for n-point Gauss-Legendre on [-1, 1], computed by Newton iteration on the Legendre polynomial and cached in a function-local static `std::vector<double>` pair. A function-local static is initialized once, thread-safely, in C++11 and later. Do not use a `std::string`-keyed map to cache by n — that breaks the wasm build. Key the cache by having one function per node count, or by a small fixed array indexed by a compile-time constant.

Add to the file's existing "Numerical references" comment block: Gauss-Legendre quadrature, and Copenhaver and Holland (1988) for the studentized range.

- [ ] **Step 6: Implement the inner integral**

```cpp
// W(w; k) = P(range of k iid standard normals <= w)
//         = k * integral phi(z) * [Phi(z) - Phi(z-w)]^(k-1) dz
//
// The integrand is bounded by phi(z), whose mass beyond |z| = 8 is below
// 1.3e-15, so truncating there costs less than k * 1.3e-15 — three orders
// below the 1e-8 band this function is held to.
inline double NormalRangeCDF(double w, double k) {
    if (!(w > 0.0)) return 0.0;
    if (!(k >= 2.0)) return std::numeric_limits<double>::quiet_NaN();
    // panelled Gauss-Legendre over [-8, 8]
}
```

- [ ] **Step 7: Implement the outer integral**

```cpp
// P(Q <= q; k, nu) = integral f_nu(s) * W(q*s; k) ds, where s = sqrt(chi2_nu/nu).
//
// log f_nu(s) = (nu/2)*log(nu) + (nu-1)*log(s) - nu*s*s/2
//               - (nu/2 - 1)*log(2) - lgamma(nu/2)
// computed in logs because nu^(nu/2) overflows for moderate nu.
//
// The integration range comes from the chi-squared quantile already in this
// header: s_lo = sqrt(ChiSquareQuantile(1e-12, nu)/nu) and s_hi likewise at
// 1 - 1e-12. The mass outside is below 2e-12, which is under the 1e-8 band.
```

Above the ν cutoff, return `NormalRangeCDF(q, k)` directly. Choose the cutoff by measuring the jump at candidate values and picking one where it is below 1e-9; R switches around ν = 25000, which is the right order of magnitude. Record the chosen value, the measured jump, and the reason in a comment.

- [ ] **Step 8: Implement the quantile**

Bracketed bisection on the CDF. Start from `[0, 8]` and double the upper end until the CDF exceeds p, with a hard cap on the doublings so a pathological input cannot loop forever. Then a **fixed** 100 bisection steps, so the result never depends on an adaptive stopping rule that could differ between platforms. State in a comment that 100 steps shrink the bracket far below the CDF's own accuracy, so the CDF is the limiting factor.

- [ ] **Step 9: Run the test to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: `test_tukey` passes. If identity 1 fails while the fixtures pass, the quadrature is subtly wrong and the fixtures were generated from the same wrong assumption — treat the identity as authoritative.

- [ ] **Step 10: Tune the node counts against the band**

Raise the node count until the whole fixture grid and all four identities pass with margin, then stop. Record the chosen counts and the evidence — which case was tightest, and by how much — in a comment. Do not tune to one case and assume the rest: the ν = 1 and k = 20 corners are the hardest, so check them explicitly.

- [ ] **Step 11: Add the domain tests (Review Focus item 4)**

```cpp
const double nan = std::numeric_limits<double>::quiet_NaN();
check(stats_duck::StudentizedRangeCDF(0.0, 3.0, 10.0) == 0.0, "q = 0 gives 0");
check(stats_duck::StudentizedRangeCDF(-1.0, 3.0, 10.0) == 0.0, "negative q gives 0");
check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, 1.0, 10.0)), "k < 2 is NaN");
check(std::isnan(stats_duck::StudentizedRangeCDF(2.0, 3.0, 0.0)), "non-positive df is NaN");
check(std::isnan(stats_duck::StudentizedRangeCDF(nan, 3.0, 10.0)), "NaN q is NaN");
check(stats_duck::StudentizedRangeCDF(1e300, 3.0, 10.0) == 1.0, "huge q saturates at 1");

check(stats_duck::StudentizedRangeQuantile(0.0, 3.0, 10.0) == 0.0, "p = 0 is the support floor");
check(std::isinf(stats_duck::StudentizedRangeQuantile(1.0, 3.0, 10.0)), "p = 1 is +inf");
check(std::isnan(stats_duck::StudentizedRangeQuantile(-0.1, 3.0, 10.0)), "p below 0 is NaN");
check(std::isnan(stats_duck::StudentizedRangeQuantile(1.1, 3.0, 10.0)), "p above 1 is NaN");
check(std::isnan(stats_duck::StudentizedRangeQuantile(0.95, 1.0, 10.0)), "k < 2 is NaN");
```

Each of these must return in bounded time. A hang here is a failure, not a slow test.

- [ ] **Step 12: Run and commit**

Run: `bash scripts/run-cpp-tests.sh`
Expected: seven suites pass.

```bash
git add src/include/distributions.hpp test/cpp/gen_tukey_fixtures.py \
        test/cpp/tukey_fixtures.hpp test/cpp/test_tukey.cpp scripts/run-cpp-tests.sh
git commit -m "Add the studentized range distribution (#21)"
```

---

## Task 4: Expose `ptukey` and `qtukey` as SQL scalars

Two thin registrations. They give the new numerics their own SQL coverage and are useful on their own for looking up critical values.

**Files:**
- Modify: `src/distribution_functions.cpp` — register both
- Modify: `test/sql/function_docs.test` — three lists and the count
- Create: `test/sql/tukey.test`

**Interfaces:**
- Consumes: `stats_duck::StudentizedRangeCDF`, `stats_duck::StudentizedRangeQuantile` from Task 3.
- Produces: the SQL functions `ptukey(q, k, df)` and `qtukey(p, k, df)`, both `DOUBLE`.

- [ ] **Step 1: Write the failing SQL test**

Create `test/sql/tukey.test` in the style of `test/sql/distribution_functions.test`. Include the k = 2 identity at the SQL layer, so it is checked at both layers:

```
# name: test/sql/tukey.test
# description: studentized range distribution (ptukey, qtukey)
# group: [stats_duck]

require stats_duck

# At k = 2 the studentized range is the t distribution: P(Q<=q) = 2*F_t(q/sqrt(2)) - 1
query I
SELECT printf('%.10f', ptukey(3.0, 2, 10.0) - (2 * pt(3.0 / sqrt(2.0), 10.0) - 1));
----
0.0000000000

# A published critical value: q(0.95, 3, 6) is 4.339 to three decimals.
query I
SELECT printf('%.3f', qtukey(0.95, 3, 6.0));
----
4.339

# The quantile inverts the CDF.
query I
SELECT printf('%.8f', ptukey(qtukey(0.95, 4, 20.0), 4, 20.0));
----
0.95000000

# Out of domain gives NULL, not an error.
query III
SELECT ptukey(2.0, 1, 10.0), ptukey(2.0, 3, 0.0), qtukey(1.5, 3, 10.0);
----
NULL	NULL	NULL
```

- [ ] **Step 2: Run it to see it fail**

Run the SQL command from Global Constraints with the path narrowed to the new file.
Expected: a binder error, because `ptukey` does not exist.

- [ ] **Step 3: Register both functions**

In `src/distribution_functions.cpp`, follow the pattern the neighbouring `pt`/`qt` registrations use. Both take three DOUBLE arguments and return DOUBLE. Register through `RegisterDocumented` with `OnCreateConflict::ALTER_ON_CONFLICT`, a description, a parameter list and an example — `CreateInfo` defaults to `ERROR_ON_CONFLICT`, and #50's audit found 95 registrations silently relying on the wrong default. NaN from the kernel becomes NULL at the vector layer, which is what makes the out-of-domain row above return NULL rather than raising.

- [ ] **Step 4: Build and run the new test**

Expected: `test/sql/tukey.test` passes.

- [ ] **Step 5: Update the documentation metadata net**

`test/sql/function_docs.test` has three `IN (...)` lists and a count assertion currently reading `96`. Add `ptukey` and `qtukey` to **all three** lists, in their alphabetical positions, and change the count to `98`. In #20 this file was missed and the omission was then reported as "unaffected", so verify by running the file and reading the count, not by reasoning about it.

- [ ] **Step 6: Run the whole SQL suite**

Expected: `All tests passed`, with the assertion count above 2206 by however many the new file adds.

- [ ] **Step 7: Commit**

```bash
git add src/distribution_functions.cpp test/sql/tukey.test test/sql/function_docs.test
git commit -m "Expose ptukey and qtukey as SQL scalars (#21)"
```

---

## Task 5: The emmeans kernel

**Files:**
- Create: `src/include/emmeans.hpp`, `src/emmeans.cpp`
- Modify: `CMakeLists.txt` — add `src/emmeans.cpp` to `EXTENSION_SOURCES`
- Modify: `scripts/run-cpp-tests.sh` — one new suite
- Create: `test/cpp/gen_emmeans_fixtures.py`, `test/cpp/emmeans_fixtures.hpp` (generated), `test/cpp/test_emmeans.cpp`

**Interfaces:**
- Consumes: `linalg::matvec`, `linalg::sandwich`; `statsduck::aliased_columns`, `row_touches_aliased`, `zero_aliased`; `statsduck::adjust_p`, `AdjustMethod`; `stats_duck::StudentTCDF`, `StudentTQuantile`, `StudentizedRangeCDF`, `StudentizedRangeQuantile`.
- Produces, in namespace `statsduck`:
  ```cpp
  enum class EmmAdjust { TUKEY, BONFERRONI, HOLM, HOCHBERG, BH, BY, NONE };
  bool parse_emm_adjust(const std::string &name, EmmAdjust &out);
  const char *emm_adjust_name(EmmAdjust m);

  struct EmmMean {
      bool estimable = false;
      double estimate = 0.0, std_error = 0.0, t_statistic = 0.0,
             p_value = 0.0, lower_cl = 0.0, upper_cl = 0.0;
  };
  struct EmmDiff {
      std::size_t i = 0, j = 0;   // 0-based here; the SQL layer adds 1
      bool estimable = false;
      double estimate = 0.0, std_error = 0.0, t_statistic = 0.0,
             p_value = 0.0, p_adjusted = 0.0, lower_cl = 0.0, upper_cl = 0.0;
  };
  struct EmmeansResult {
      bool ok = false;              // false means misuse; the caller raises
      std::string error;
      std::vector<EmmMean> means;   // one per row of L, in L's order
      std::vector<EmmDiff> diffs;   // i < j, row-major
      double df_den = 0.0, alpha = 0.0;
      EmmAdjust adjust = EmmAdjust::TUKEY;
      std::size_t k_compare = 0;    // estimable means, the family size
  };
  EmmeansResult emmeans(const linalg::Mat &L, const std::vector<double> &beta,
                        const linalg::Mat &V, double df, double alpha,
                        EmmAdjust adjust);
  ```

  Note the two-level outcome: `ok = false` is misuse, and estimability is per element. There is no top-level `estimable`, unlike `LinHypResult`, because one bad grid row must not discard the others.

  `EmmAdjust` and `statsduck::AdjustMethod` are deliberately two enums. `AdjustMethod` holds the six methods computable from a list of p-values, and `adjust_p` is honest about that — it cannot compute Tukey, which needs the t statistic and the family size. `EmmAdjust` adds `TUKEY` and maps the other six across, one to one:

  ```cpp
  // BONFERRONI -> AdjustMethod::BONFERRONI
  // HOLM       -> AdjustMethod::HOLM
  // HOCHBERG   -> AdjustMethod::HOCHBERG
  // BH         -> AdjustMethod::BENJAMINI_HOCHBERG
  // BY         -> AdjustMethod::BENJAMINI_YEKUTIELI
  // NONE       -> AdjustMethod::NONE
  ```

  `emm_adjust_name` returns `tukey`, `bonferroni`, `holm`, `hochberg`, `BH`, `BY` or `none`, so `fdr` echoes as `BH`. `parse_emm_adjust` accepts `fdr` as an alias for `BH` and is case sensitive, matching `adjust_p`.

- [ ] **Step 1: Write the generator**

Create `test/cpp/gen_emmeans_fixtures.py`, following `gen_lin_hyp_fixtures.py`. The fixture is a one-way design with three groups and cell-means coding, which makes the marginal means the group means and lets the test check them by hand:

```python
"""Goldens for emmeans.

Oracle: statsmodels OLS .t_test, the same oracle gen_lin_hyp_fixtures.py
uses, plus scipy.stats.studentized_range for the Tukey p-values.

Design DS1: three groups, cell-means coded (no intercept), so L = I(3)
and each marginal mean is that group's sample mean.
    g1 = [1, 2, 3]      mean 2
    g2 = [4, 6, 8]      mean 6
    g3 = [5, 5, 8]      mean 6
Within-group SS = 2 + 8 + 6 = 16 on 6 df, so s^2 = 8/3 and every mean has
standard error sqrt(s^2/3) = 0.9428090416, every difference
sqrt(2*s^2/3) = 1.3333333333. Those are checkable without an oracle and
the test asserts them directly.

Each mean comes from t_test(L_row). Each pair comes from
t_test(L_i - L_j), which reaches the pairwise standard error by a route
that does not use C_ii + C_jj - 2*C_ij, so it independently checks the
shortcut the kernel takes.

Self-checks, which must hold before anything is written:
  1. mean[g] equals the sample mean of group g
  2. se[g] equals sqrt(s^2/n_g)
  3. the pairwise se from t_test equals sqrt(C_ii + C_jj - 2*C_ij)
  4. at two means, the Tukey p equals the unadjusted p
"""
```

Also emit a second design with an aliased column, so the estimability path has goldens: a 4-column design whose last column is the sum of two others, fitted with `lm_fit`'s convention that the later column is dropped.

- [ ] **Step 2: Run the generator and commit the fixtures**

```bash
python -m venv /tmp/emmenv && /tmp/emmenv/bin/pip install statsmodels scipy
/tmp/emmenv/bin/python test/cpp/gen_emmeans_fixtures.py > test/cpp/emmeans_fixtures.hpp
```

Expected: the four self-checks print as passing. Self-check 3 is the important one: if it fails, either the sandwich convention or the oracle call is wrong, and no golden should be trusted until it holds.

- [ ] **Step 3: Write the failing test, hand-checkable values first**

Create `test/cpp/test_emmeans.cpp`. Use a bounds-safe accessor for every vector read — a failing call returns empty vectors, and indexing them turns a clear assertion failure into a crash that hides which assertion was wrong. That exact defect appeared in #33, #36 and #20.

```cpp
static double at(const std::vector<double> &v, std::size_t i) {
    return i < v.size() ? v[i] : std::numeric_limits<double>::quiet_NaN();
}

// DS1: three cell means, L = I(3).
linalg::Mat L(3, 3);
for (std::size_t i = 0; i < 3; i++)
    for (std::size_t j = 0; j < 3; j++)
        L(i, j) = (i == j) ? 1.0 : 0.0;

const auto r = statsduck::emmeans(L, beta_ds1, V_ds1, 6.0, 0.05,
                                  statsduck::EmmAdjust::TUKEY);
check(r.ok, "DS1 fits");
check(r.means.size() == 3, "three means");
check(r.diffs.size() == 3, "three pairs from three means");
check(r.k_compare == 3, "all three means are estimable");

// Hand-computable: the means are the group means, the SEs come from s^2 = 8/3.
check(close(r.means[0].estimate, 2.0, 1e-10), "mean 1 is the group mean");
check(close(r.means[1].estimate, 6.0, 1e-10), "mean 2 is the group mean");
check(close(r.means[0].std_error, 0.9428090416, 1e-9), "mean SE is sqrt(s2/n)");
check(close(r.diffs[0].std_error, 1.3333333333, 1e-9), "pair SE is sqrt(2 s2/n)");
check(close(r.diffs[0].estimate, -4.0, 1e-10), "pair 1-2 is 2 - 6");
check(r.diffs[0].i == 0 && r.diffs[0].j == 1, "pairs are 0-based in the kernel");
check(r.diffs[1].i == 0 && r.diffs[1].j == 2, "and enumerated row-major");
check(r.diffs[2].i == 1 && r.diffs[2].j == 2, "third pair is 2-3");
```

Then the fixture loops, with the spec's bands and their reasons as comments.

- [ ] **Step 4: Register the suite and run it to see it fail**

Add to `scripts/run-cpp-tests.sh`:

```bash
build_and_run test_emmeans \
  "$ROOT/test/cpp/test_emmeans.cpp" \
  "$ROOT/src/emmeans.cpp" \
  "$ROOT/src/linalg.cpp"
```

Run: `bash scripts/run-cpp-tests.sh`
Expected: `COMPILE FAILED: test_emmeans`.

- [ ] **Step 5: Write the header**

Create `src/include/emmeans.hpp` with the interface block above. The file comment states the scope and the three things it is not: it does not build the grid, it does not compute Satterthwaite degrees of freedom, and it inverts nothing. Say explicitly that `i` and `j` are 0-based and that the SQL layer adds one, because that is the obvious place for an off-by-one.

- [ ] **Step 6: Implement validation and the one sandwich**

In `src/emmeans.cpp`, validate in this order and return `ok = false` with a message naming the function and the actual sizes: a row width that is not k, a `cov` that is not k × k, a `beta` of length 0, a `df` that is not positive and finite, an `alpha` outside (0, 1). Then take `aliased_columns(V)`, build the zeroed working copies, and form `theta = matvec(L, beta_w)` and `C = sandwich(L, V_w)` once.

Two shapes that look like errors but are not:

- **An `L` with zero rows is not misuse.** It returns `ok = true` with empty `means` and `diffs` and `k_compare` of 0. The SQL layer is what turns an empty `means` into a NULL result, per the spec. Deciding this in the kernel would make an empty grid an exception, which would abort a whole `GROUP BY` for one group whose grid came out empty.
- **A zero row inside `L`** produces a zero row in `C`, which is legitimate — it is the degenerate contrast that estimates zero with zero variance. Do not reject it.

```cpp
// An empty grid is a value, not an error.
const linalg::Mat L0(0, 3);
const auto r0 = statsduck::emmeans(L0, beta_ds1, V_ds1, 6.0, 0.05,
                                   statsduck::EmmAdjust::TUKEY);
check(r0.ok, "an empty L is not misuse");
check(r0.means.empty() && r0.diffs.empty(), "and produces nothing");
check(r0.k_compare == 0, "with a family size of zero");
```

- [ ] **Step 7: Implement the means**

For each row: mark it not estimable when `row_touches_aliased`, when θᵢ is not finite, or when `C_ii` is negative past `tol · max_j C_jj`. Otherwise set the estimate, the standard error as the square root of a `C_ii` clamped at zero, the t statistic and the two-sided p against `StudentTCDF`, and the limits from `StudentTQuantile(1 - alpha/2, df)`. A zero standard error gives a NaN t and a NaN p while keeping the estimate and the limits. The limits are never adjusted.

- [ ] **Step 8: Run the test to see the means pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: the mean assertions pass and the pair assertions fail, since `diffs` is still empty.

- [ ] **Step 9: Implement the pairs and the adjustments**

Enumerate i < j row-major over the estimable means, but **emit a `diffs` element for every pair** so positions stay stable, marking the ones that touch a non-estimable mean as not estimable. Variance is `C_ii + C_jj - 2 * sym(C_ij)` where `sym(C_ij) = 0.5 * (C(i,j) + C(j,i))`. Apply the relative tolerance with scale `C_ii + C_jj`.

Then the adjustment, with `m` the number of estimable pairs and `k_compare` the number of estimable means:

- `TUKEY`: `p_adjusted = 1 - StudentizedRangeCDF(sqrt(2) * fabs(t), k_compare, df)`, and the limit multiplier is `StudentizedRangeQuantile(1 - alpha, k_compare, df) / sqrt(2)`. Guard: when `k_compare < 2`, do not call either function.
- `BONFERRONI`: through `statsduck::adjust_p` for the p-values, and `StudentTQuantile(1 - alpha / (2 * m), df)` for the multiplier.
- `HOLM`, `HOCHBERG`, `BH`, `BY`: through `statsduck::adjust_p` with the estimable pairs as the valid mask, and the limits left at `StudentTQuantile(1 - alpha/2, df)`.
- `NONE`: `p_adjusted` equals `p_value`, limits unadjusted.

- [ ] **Step 10: Run the test to see it pass**

Run: `bash scripts/run-cpp-tests.sh`
Expected: eight suites pass.

- [ ] **Step 11: Commit**

```bash
git add src/include/emmeans.hpp src/emmeans.cpp CMakeLists.txt \
        test/cpp/gen_emmeans_fixtures.py test/cpp/emmeans_fixtures.hpp \
        test/cpp/test_emmeans.cpp scripts/run-cpp-tests.sh
git commit -m "Add the emmeans kernel (#21)"
```

- [ ] **Step 12: Write the aliased-coefficient test**

```cpp
// DS2 is rank deficient: its last column is the sum of two others, so lm_fit
// drops it and cov carries NaN in that row and column.
// A grid row avoiding the aliased column is evaluated; one touching it is not.
const auto r2 = statsduck::emmeans(L_ds2, beta_ds2, V_ds2, df_ds2, 0.05,
                                   statsduck::EmmAdjust::NONE);
check(r2.ok, "a rank-deficient fit is not misuse");
check(r2.means[0].estimable, "a row avoiding the aliased column is estimable");
check(!r2.means[2].estimable, "a row touching it is not");
check(r2.k_compare == 2, "k_compare counts only the estimable means");
check(r2.diffs.size() == 3, "every pair is still present");
check(!r2.diffs[1].estimable, "a pair touching the aliased mean is not estimable");
check(r2.diffs[0].estimable, "a pair between two good means still is");
check(std::isfinite(r2.means[0].estimate), "no NaN leaks from the aliased column");
```

- [ ] **Step 13: Write the duplicated-row test (Review Focus item 1)**

```cpp
// Two identical grid rows. C_ii = C_jj = C_ij, so the pairwise variance is
// exactly zero by cancellation. A user who repeats a level reaches this, so
// it must not be mistaken for a negative variance.
linalg::Mat Ld(2, 3);
Ld(0, 0) = 1.0; Ld(0, 1) = 0.0; Ld(0, 2) = 0.0;
Ld(1, 0) = 1.0; Ld(1, 1) = 0.0; Ld(1, 2) = 0.0;
const auto rd = statsduck::emmeans(Ld, beta_ds1, V_ds1, 6.0, 0.05,
                                   statsduck::EmmAdjust::NONE);
check(rd.ok && rd.diffs.size() == 1, "one pair from two rows");
check(rd.diffs[0].estimable, "an identical pair is estimable, not NULL");
check(rd.diffs[0].estimate == 0.0, "the difference is exactly zero");
check(rd.diffs[0].std_error == 0.0, "and so is its standard error");
check(std::isnan(rd.diffs[0].t_statistic), "t is NaN at a zero standard error");
check(std::isnan(rd.diffs[0].p_value), "and so is p");
check(rd.means[0].estimate == rd.means[1].estimate,
      "identical rows give identical means");
```

Two more assertions belong here, because they are the cheapest possible checks on the adjustment wiring and the spec names both:

```cpp
// With one comparison, bonferroni multiplies by one and changes nothing.
const auto rb1 = statsduck::emmeans(L_two_rows, beta_ds1, V_ds1, 6.0, 0.05,
                                    statsduck::EmmAdjust::BONFERRONI);
check(rb1.diffs.size() == 1, "two means give one pair");
check(rb1.diffs[0].p_adjusted == rb1.diffs[0].p_value,
      "bonferroni over one pair leaves p alone");

// Under none, every adjusted p equals its raw p exactly, for every pair.
const auto rn = statsduck::emmeans(L, beta_ds1, V_ds1, 6.0, 0.05,
                                   statsduck::EmmAdjust::NONE);
for (std::size_t d = 0; d < rn.diffs.size(); d++)
    check(rn.diffs[d].p_adjusted == rn.diffs[d].p_value,
          "none leaves every p alone");
check(std::string(statsduck::emm_adjust_name(rn.adjust)) == "none",
      "the echoed name is canonical");
```

`L_two_rows` is `L`'s first two rows: a 2 by 3 matrix with ones at (0,0) and (1,1).

- [ ] **Step 14: Run, then commit**

Run: `bash scripts/run-cpp-tests.sh`. If the duplicated-row pair comes back not estimable, the tolerance is being applied with the wrong sign or the wrong scale.

```bash
git add test/cpp/test_emmeans.cpp
git commit -m "emmeans: test aliased columns and duplicated grid rows (#21)"
```

- [ ] **Step 15: Write the scale-invariance test (Review Focus item 2)**

```cpp
// Scaling cov by a constant scales every variance by it, so the estimates
// scale by its square root and the t statistics, p-values and k_compare do
// not move at all. In #20 a units change silently moved df_num and the F,
// so this is pinned rather than assumed.
linalg::Mat Vbig = V_ds1;
for (double &v : Vbig.data) v *= 1e12;
const auto rb = statsduck::emmeans(L, beta_ds1, Vbig, 6.0, 0.05,
                                   statsduck::EmmAdjust::TUKEY);
check(rb.ok && rb.k_compare == 3, "k_compare is unaffected by scale");
for (std::size_t i = 0; i < 3; i++) {
    check(close(rb.means[i].estimate, r.means[i].estimate, 1e-10),
          "estimates do not depend on cov's scale");
    check(close(rb.means[i].std_error, r.means[i].std_error * 1e6, 1e-4),
          "standard errors scale by the square root");
}
check(close(rb.diffs[0].t_statistic * 1e6, r.diffs[0].t_statistic, 1e-6),
      "t scales inversely, as it must");
```

- [ ] **Step 16: Run and commit**

```bash
git add test/cpp/test_emmeans.cpp
git commit -m "emmeans: pin scale invariance of the tolerance decision (#21)"
```

- [ ] **Step 17: Write the one-estimable-mean test (Review Focus item 3)**

```cpp
// Tukey with fewer than two estimable means. ptukey is undefined for k < 2,
// so it must not be called at all.
const auto r1 = statsduck::emmeans(L_one_row, beta_ds1, V_ds1, 6.0, 0.05,
                                   statsduck::EmmAdjust::TUKEY);
check(r1.ok, "a single grid row is not misuse");
check(r1.means.size() == 1 && r1.diffs.empty(), "one mean, no pairs");
check(r1.k_compare == 1, "k_compare reports what it found");
check(std::isfinite(r1.means[0].estimate), "the mean is reported normally");
check(std::isfinite(r1.means[0].lower_cl), "and its limits are finite");
```

- [ ] **Step 18: Run and commit**

```bash
git add test/cpp/test_emmeans.cpp
git commit -m "emmeans: no studentized range call below two estimable means (#21)"
```

- [ ] **Step 19: Write the extreme df and alpha test (Review Focus item 5)**

```cpp
// A fractional, tiny df with an extreme alpha drives StudentTQuantile deep
// into its tail. The result may be enormous, but it must be reached in
// bounded time and must be either finite or an infinity, never a NaN that
// reads as a valid limit.
const auto re = statsduck::emmeans(L, beta_ds1, V_ds1, 0.5, 1e-10,
                                   statsduck::EmmAdjust::NONE);
check(re.ok, "a fractional df is accepted");
check(re.df_den == 0.5 && re.alpha == 1e-10, "df and alpha are echoed exactly");
const double lo = re.means[0].lower_cl, hi = re.means[0].upper_cl;
check(!std::isnan(lo) && !std::isnan(hi), "limits are not NaN");
check(lo <= re.means[0].estimate && re.means[0].estimate <= hi,
      "the estimate lies inside its own interval");

// df at the boundary is misuse, not a very wide interval.
const auto rz = statsduck::emmeans(L, beta_ds1, V_ds1, 0.0, 0.05,
                                   statsduck::EmmAdjust::NONE);
check(!rz.ok, "df = 0 is misuse");
check(rz.error.find("df") != std::string::npos, "and the message names df");
```

- [ ] **Step 20: Run and commit**

```bash
git add test/cpp/test_emmeans.cpp
git commit -m "emmeans: pin behaviour at extreme df and alpha (#21)"
```

---

## Task 6: The emmeans SQL function

**Files:**
- Create: `src/include/emmeans_function.hpp`, `src/emmeans_function.cpp`
- Modify: `CMakeLists.txt` — add `src/emmeans_function.cpp` to `EXTENSION_SOURCES`
- Modify: `src/stats_duck_extension.cpp` — call `RegisterEmmeans`
- Modify: `test/sql/function_docs.test` — three lists and the count, 98 to 99
- Create: `test/sql/emmeans.test`

**Interfaces:**
- Consumes: `statsduck::emmeans`, `EmmAdjust`, `parse_emm_adjust`, `emm_adjust_name` from Task 5; `ReadDoubleMatrix`, `ReadDoubleList` from Task 2.
- Produces: the SQL scalar `emmeans(L, beta, cov, df [, alpha [, adjust]])`, returning the STRUCT the spec defines, and `void RegisterEmmeans(ExtensionLoader &loader)`.

- [ ] **Step 1: Write the failing SQL test**

Create `test/sql/emmeans.test`. Build the fixture with `lm_fit` so the test exercises the real producer of `cov`:

```
# name: test/sql/emmeans.test
# description: estimated marginal means (emmeans)
# group: [stats_duck]

require stats_duck

statement ok
CREATE TABLE ds1 AS SELECT * FROM (VALUES
  (1.0, 1, 0, 0), (2.0, 1, 0, 0), (3.0, 1, 0, 0),
  (4.0, 0, 1, 0), (6.0, 0, 1, 0), (8.0, 0, 1, 0),
  (5.0, 0, 0, 1), (5.0, 0, 0, 1), (8.0, 0, 0, 1)
) AS t(y, g1, g2, g3);

# Cell-means coding, so each marginal mean is that group's mean: 2, 6, 6.
query III
SELECT printf('%.4f', (e).means[1].estimate),
       printf('%.4f', (e).means[2].estimate),
       printf('%.4f', (e).means[3].estimate)
FROM (SELECT emmeans([[1,0,0],[0,1,0],[0,0,1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.df_residual) AS e
      FROM (SELECT lm_fit(y, [g1, g2, g3], 'const', false) AS m FROM ds1));
----
2.0000	6.0000	6.0000
```

Cover, each as its own query: the mean standard error 0.9428 and the pair standard error 1.3333; `i` and `j` coming back 1-based; `diffs` having three elements; `k_compare` of 3; `adjust` echoing `tukey` by default and `BH` when given `fdr`; the limits moving between `tukey` and `holm` but `p_value` staying the same in both; `alpha := 0.01` widening the limits; a single-row `L` giving an empty `diffs`; and `ptukey` reproducing the Tukey `p_adjusted` for one pair, so the two layers are tied together in SQL as well as in C++.

Then the misuse cases, each expecting an error with its message fragment: a ragged `L`, a row width that is not k, a `cov` that is not k² long, `df := 0`, `alpha := 1`, and `adjust := 'nope'`.

Then the three NULL paths, which are data-shaped and must **not** raise. The first query establishes the pattern; write each as its own query:

```
# An empty L makes the whole result NULL, not an error and not an empty list.
query I
SELECT emmeans(CAST([] AS DOUBLE[][]), [1.0, 2.0, 3.0],
               [1.0,0,0, 0,1.0,0, 0,0,1.0], 6.0) IS NULL;
----
true

# A NULL element inside L makes the whole result NULL: a missing weight cannot
# be told apart from a zero one, and guessing would change the hypothesis.
query I
SELECT emmeans([[1.0, NULL, 0.0]], [1.0, 2.0, 3.0],
               [1.0,0,0, 0,1.0,0, 0,0,1.0], 6.0) IS NULL;
----
true

# A NULL row of L is the same answer by the same reasoning.
query I
SELECT emmeans([[1.0, 0.0, 0.0], NULL], [1.0, 2.0, 3.0],
               [1.0,0,0, 0,1.0,0, 0,0,1.0], 6.0) IS NULL;
----
true
```

- [ ] **Step 2: Write the multi-row chunk test**

This is required by the spec, not optional. Every other query in the file returns one row, so a wrong offset walk in the output lists would pass the whole suite otherwise.

Group `b` has no observations in its third cell, so that column is all zeros, `lm_fit` drops it as dependent, and the grid row selecting it becomes non-estimable. The three groups therefore report different numbers of estimable means while every `means` list still has three elements.

```
statement ok
CREATE TABLE ds_multi AS SELECT * FROM (VALUES
  ('a', 1.0, 1, 0, 0), ('a', 2.0, 1, 0, 0), ('a', 4.0, 0, 1, 0),
  ('a', 6.0, 0, 1, 0), ('a', 5.0, 0, 0, 1), ('a', 8.0, 0, 0, 1),
  ('b', 2.0, 1, 0, 0), ('b', 3.0, 1, 0, 0), ('b', 7.0, 0, 1, 0),
  ('b', 9.0, 0, 1, 0), ('b', 4.0, 0, 1, 0), ('b', 6.0, 0, 1, 0),
  ('c', 3.0, 1, 0, 0), ('c', 5.0, 1, 0, 0), ('c', 2.0, 0, 1, 0),
  ('c', 4.0, 0, 1, 0), ('c', 9.0, 0, 0, 1), ('c', 7.0, 0, 0, 1)
) AS t(grp, y, g1, g2, g3);

# Three fits in one query. Group b's third cell is empty, so g3 is dropped and
# the grid row selecting it is not estimable. The rows therefore emit different
# numbers of estimable means and of estimable pairs, which a single-row query
# cannot catch when an output offset fails to advance.
query IIII
SELECT grp,
       len((e).means),
       len((e).diffs),
       (e).k_compare
FROM (
  SELECT grp, emmeans([[1,0,0],[0,1,0],[0,0,1]],
                      list_transform(m.coefficients, c -> c.estimate),
                      m.cov, m.df_residual) AS e
  FROM (SELECT grp, lm_fit(y, [g1, g2, g3], 'const', false) AS m
        FROM ds_multi GROUP BY grp)
) ORDER BY grp;
----
a	3	3	3
b	3	3	2
c	3	3	3
```

Then check that the middle group's NULLs land in the right positions rather than merely being present, which is the specific failure an offset bug produces:

```
query III
SELECT printf('%.4f', (e).means[1].estimate),
       (e).means[3].estimate IS NULL,
       (e).diffs[3].estimate IS NULL
FROM (SELECT emmeans([[1,0,0],[0,1,0],[0,0,1]],
                     list_transform(m.coefficients, c -> c.estimate),
                     m.cov, m.df_residual) AS e
      FROM (SELECT lm_fit(y, [g1, g2, g3], 'const', false) AS m
            FROM ds_multi WHERE grp = 'b'));
----
2.5000	true	true
```

- [ ] **Step 3: Run it to see it fail**

Expected: a binder error, because `emmeans` does not exist.

- [ ] **Step 4: Write the result type and the registration**

In `src/emmeans_function.cpp`, define `EmmeansResultType()` carrying the comment `// Field order MUST match the writes in Finalize` and an index comment per field, in the spec's order: `means`, `diffs`, `df_den`, `alpha`, `adjust`, `k_compare`. Two nested STRUCT row types, one for a mean and one for a difference, in the spec's field order.

Register with four, five and six argument overloads so `alpha` and `adjust` are optional, through `RegisterDocumented` with `OnCreateConflict::ALTER_ON_CONFLICT`. Accept `LIST(LIST(DOUBLE))` and cast, since DuckDB infers an integer grid as `INTEGER[][]`.

- [ ] **Step 5: Write the exec function**

Read the arguments with `ReadDoubleMatrix` and `ReadDoubleList`. A NULL argument, a NULL element, or a reader failure gives a NULL result row. A `statsduck::emmeans` result with `ok = false` raises `InvalidInputException` with the kernel's message. Otherwise write the two lists.

Reserve the child list capacity **before** fetching the child vector, because `ListVector::Reserve` can reallocate and a pointer taken earlier becomes stale. Walk the output offsets per row rather than assuming they advance by a fixed amount, since each row contributes a different number of elements. Set every non-finite double through the `SetD` helper so NaN becomes NULL.

Add `i + 1` and `j + 1` when writing the pair indices. The kernel is 0-based and SQL lists are 1-based.

- [ ] **Step 6: Build and run the file**

Expected: `test/sql/emmeans.test` passes, the multi-row test included. A failure only in the multi-row case means the output offset walk is wrong, which is exactly what that test exists to find.

- [ ] **Step 7: Update the documentation metadata net**

Add `emmeans` to all three `IN (...)` lists in `test/sql/function_docs.test` and change the count from `98` to `99`. Verify by running the file.

- [ ] **Step 8: Run both suites in full**

Expected: C++ eight suites green; SQL all passed, with the count risen by the new assertions.

- [ ] **Step 9: Commit**

```bash
git add src/include/emmeans_function.hpp src/emmeans_function.cpp CMakeLists.txt \
        src/stats_duck_extension.cpp test/sql/emmeans.test test/sql/function_docs.test
git commit -m "Expose emmeans as a SQL scalar function (#21)"
```

---

## Task 7: Documentation

**Files:**
- Modify: `README.md` — a function-table row and a section
- Modify: `docs/kernel_api.md` — an `emmeans.hpp` section and the studentized range
- Modify: `docs/fitter_conventions.md` — the matrix-argument rule
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the README row and section**

Add `emmeans` to the function table, marked **scalar** the way the `lm_fit` row marks itself an aggregate. Then a section after the `lin_hyp` one, with a worked query and its output, the result-field table, and five things a reader needs:

- the means' limits are never adjusted, whatever `adjust` says;
- only `tukey` and `bonferroni` move the differences' limits, because the other four have no matching simultaneous interval;
- Tukey is the Tukey-Kramer approximation whenever the standard errors differ, which is to say in any unbalanced design;
- `k_compare` is the family size the adjustment used, so an adjusted p-value can be reproduced;
- **the adjustment assumes the rows of `L` are means.** The studentized range describes the spread of a set of means, and the family here is all pairwise differences among the q rows. A caller who passes rows that are themselves contrasts still gets arithmetic, but the family-wise interpretation no longer holds. The code cannot detect this, so the documentation has to say it.

- [ ] **Step 2: Add the kernel_api section**

Document `emmeans.hpp` in the same shape as the `lin_hyp.hpp` section: the signature, a symbol table, and the conventions. State that nothing is inverted, and that the negative-variance tolerance is relative with a different scale for a mean and for a difference. Add `StudentizedRangeCDF` and `StudentizedRangeQuantile` to the distributions material, including the k = 2 identity, since that is the fastest way for a reader to convince themselves the implementation is right.

- [ ] **Step 3: Add the matrix-argument rule to fitter_conventions**

Under the input conventions, state the rule the spec settles: a matrix a machine produces goes flat and row-major with its dimension inferred, as `cov` does; a matrix a human writes goes nested, as `L` does. Give the reason — a nested literal reads as a matrix, a flat one has to be chunked by eye — and the exception: break it for a matrix argument evaluated once per data row, or one large enough for offsets to matter. Also add `emmeans` to the list of functions the page binds, which currently names `lm_fit`, `glm_fit`, `lin_hyp` and `emmeans` as planned.

- [ ] **Step 4: Add the CHANGELOG entry**

One entry covering `emmeans`, `ptukey` and `qtukey`. Name the Tukey-Kramer approximation rather than implying exactness, and say that the limits are adjusted only under `tukey` and `bonferroni`. Do not claim the covariance is checked more thoroughly than it is — that over-claim had to be corrected across four files in #63.

- [ ] **Step 5: Verify the documented examples actually run**

Paste every SQL example from the README and `kernel_api.md` into the CLI and compare the output to what the documentation claims. A documented example that does not run is worse than no example.

```bash
"C:/Users/massi/source/repos/kolistat/the-stats-duck/build/release/Release/duckdb.exe" \
  -unsigned -c "LOAD 'build/release/extension/stats_duck/stats_duck.duckdb_extension'; <query>"
```

- [ ] **Step 6: Commit**

```bash
git add README.md docs/kernel_api.md docs/fitter_conventions.md CHANGELOG.md
git commit -m "Document emmeans and the studentized range (#21)"
```

---

## Done when

- `bash scripts/run-cpp-tests.sh` reports eight suites passing, with `73`, `511`, `113` and `39` unchanged from the baseline.
- The SQL suite passes in full, with `function_docs.test` asserting `99`.
- Every README and `kernel_api.md` example has been run and matches its documented output.
- No commit contains the `duckdb` submodule.
