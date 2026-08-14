# Fitter API Conventions Standard (#17) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the house conventions that `lm_fit` established into a written, enforceable standard (`docs/fitter_conventions.md`) that every future model-fitting function — `glm_fit` (#22), `lin_hyp` (#20), `emmeans` (#21), and downstream sibling-repo fitters — follows mechanically, closing #17.

**Architecture:** One new public doc in `docs/` (the GH Pages root — finished docs belong there, this plan does not), written section-by-section with each claim verified against the code it describes before commit. A compliance appendix audits the existing function surface against the standard (document-reality-or-flag, never silently aspirational). A final wiring task links the doc from README / `kernel_api.md` / the consuming issues and closes #17.

**Tech Stack:** Markdown; `grep`/`Read` verification against `src/`; `gh` CLI for issue wiring; git on branch `v0.9`.

## Global Constraints

- All commits on branch `v0.9` (the open cycle branch); stage files **explicitly** — never `git add -A` (the `duckdb` submodule is expected-dirty by design: applied fmt patch).
- `docs/` is served by GitHub Pages from `main` — only finished public docs go there.
- This issue changes **zero behavior**: documentation + audit only. A real deviation found during the audit becomes a follow-up issue, not an inline fix.
- The standard documents Increment A's **approved** design (#37) for fields 12–14 (`rank`, `loglik`, `cov`) and rank-based df, explicitly marked *"lands with Increment A"* — the doc must not silently describe unshipped behavior as current.
- statsmodels is the sole oracle, always offline (generator scripts), never a build/CI dependency.
- Where drafted text below asserts behavior of existing code, the task's verify step is authoritative: **the doc follows the code**. If they disagree, correct the doc text and note the delta in the commit message.

---

### Task 1: `docs/fitter_conventions.md` — skeleton + input conventions

**Files:**
- Create: `docs/fitter_conventions.md`

**Interfaces:**
- Consumes: `src/lm_fit_function.cpp` (Update/bind behavior), `test/sql/lm_fit_agg.test` (caller-visible casts)
- Produces: the doc file with stable section anchors `## Input conventions`, later tasks append sections below it

- [ ] **Step 1: Write the skeleton + input-conventions section**

Create `docs/fitter_conventions.md` with exactly this content:

```markdown
# Fitter conventions — the model-fitting API standard

Every SQL-facing model-fitting function in `stats_duck` — today `lm_fit`,
next `glm_fit`, `lin_hyp`, `emmeans`, and fitters in downstream sibling
repos — follows the conventions on this page. They exist so that a new
fitter can be designed mechanically, and so that a consumer who learned one
fitter can predict the others. [`docs/kernel_api.md`](kernel_api.md) covers
the DuckDB-free C++ kernel underneath; this page covers the SQL surface and
the cross-cutting numeric and validation discipline.

Existing non-fitter functions (the hypothesis-test aggregates,
`summary_stats`, …) predate this standard; the appendix records where they
deviate. New fitters follow this page without exception.

## Input conventions

- **One row = one observation.** Fitters are aggregates: `GROUP BY` gives
  per-group fits. Post-fit helpers (`lin_hyp`, `emmeans`) are scalar
  functions consuming a fit's outputs instead.
- **The design row is a `LIST(DOUBLE)` of predictor values.** An aggregate
  sees values, not column names, so coefficients are positional: terms are
  labelled `x1` … `xk` in list order, with `(Intercept)` prepended when
  `add_intercept` is true (the default). Callers pass predictors **without**
  a constant column.
- **Per-row key columns (cluster, subject, strata) are real `VARCHAR`
  arguments.** DuckDB does not implicitly cast integers to `VARCHAR` in this
  position — callers write `id::VARCHAR`. Keys are opaque labels; fitters
  must not depend on their ordering or density.
- **Optional behavior arrives as trailing arguments / new overloads, never
  by reinterpreting an existing position.** Before adding an overload, check
  the *full* positional matrix for ambiguity — the reason WLS (Increment B,
  #19) is deferred is exactly that a weight column and a cluster column must
  not collide in the same position.
- **Missing data: listwise drop.** A row is excluded from the fit when the
  response is NULL/NaN or any element of the design row is NULL/NaN. A group
  where too few rows survive is a failed fit (NULL result), not an error.
```

- [ ] **Step 2: Verify the listwise-drop and cast claims against the code**

Run: `grep -n "IsNull\|isnan\|std::isfinite" src/lm_fit_function.cpp | head -20`
Expected: hits in the Update path skipping NULL/NaN rows. Also run
`grep -n "::VARCHAR" test/sql/lm_fit_agg.test | head -5` — expected: at least one cluster arg cast.
If either claim doesn't hold as drafted (e.g. NaN y is kept, or the list-NULL row errors instead of dropping), rewrite that bullet to match the observed behavior and say so in the commit message.

- [ ] **Step 3: Commit**

```bash
git add docs/fitter_conventions.md
git commit -m "docs: fitter conventions — input conventions (#17)"
```

---

### Task 2: Return-STRUCT standard section

**Files:**
- Modify: `docs/fitter_conventions.md` (append section)

**Interfaces:**
- Consumes: `src/lm_fit_function.cpp:36-52` (`LmFitResultType`, field indices 0–11), #37's approved field plan (12–14)
- Produces: section anchor `## The return STRUCT` with the worked field table Tasks 4–5 reference

- [ ] **Step 1: Append the return-STRUCT section**

Append to `docs/fitter_conventions.md`:

```markdown
## The return STRUCT

- **One STRUCT per fit.** Never parallel columns, never multiple return
  variants for the same fitter.
- **Field order is load-bearing and append-only.** Finalize writes children
  by index; the type function carries the comment
  `// Field order MUST match the writes in Finalize` with a per-field index
  comment (`src/lm_fit_function.cpp:36`). Released field positions never
  change or disappear — new fields append.
- **NaN → SQL NULL at the vector layer** (the `SetD` idiom); ±Inf passes
  through unchanged (`loglik = +inf` at zero RSS is representable).
- **Failure is a value.**
  - *Data-shaped* failure (too few rows, degenerate design, < 2 clusters):
    the group's STRUCT is NULL. Never an exception.
  - *Misuse* (unknown `vcov` name, non-constant option flag): bind-time
    error — the query fails before execution.
  - *Iterative* fitters (`glm_fit` and beyond) additionally expose
    `converged BOOLEAN` and `iterations BIGINT`, returning the best iterate
    with `converged = false` rather than NULL — mirroring
    `optimize::Result` semantics in the kernel.
- **Covariance is a flat row-major `LIST<DOUBLE>`** of length k²,
  `cov[i*k + j] = cov(βᵢ, βⱼ)`, aligned to `coefficients` order (aliased
  positions NaN→NULL once rank-deficient fits land).
- **Degrees of freedom follow the estimated rank.** Fitters report `rank`;
  every df formula uses it (σ², adjusted R², HC/CR small-sample factors,
  t reference). `df_residual = n − rank` is a property of the fit;
  cluster-robust inference references `t(G−1)` without changing
  `df_residual` (statsmodels does the same, silently — here it is stated).
- **p-values and CIs come from the in-repo distribution kernels**
  (`src/include/distributions.hpp`) — never a second CDF implementation.
- **Statistic naming:** t-referenced fitters name the column
  `t_statistic`; z-referenced fitters use `z_statistic`; never a bare
  `statistic` in a fitter STRUCT.

### Worked example — `lm_fit` (the template)

| # | Field | Type | Notes |
|---|-------|------|-------|
| 0 | `coefficients` | `LIST<STRUCT(term VARCHAR, estimate, std_error, t_statistic, p_value DOUBLE)>` | positional terms, `(Intercept)` first |
| 1 | `n` | `BIGINT` | rows used (after listwise drop) |
| 2 | `k` | `BIGINT` | design columns incl. intercept |
| 3 | `df_residual` | `BIGINT` | `n − rank` |
| 4 | `r_squared` | `DOUBLE` | |
| 5 | `adj_r_squared` | `DOUBLE` | |
| 6 | `sigma` | `DOUBLE` | residual SE |
| 7 | `f_statistic` | `DOUBLE` | |
| 8 | `f_p_value` | `DOUBLE` | |
| 9 | `has_intercept` | `BOOLEAN` | |
| 10 | `vcov_type` | `VARCHAR` | canonical name echoed back |
| 11 | `n_clusters` | `BIGINT` | NULL unless CR* |
| 12 | `rank` | `BIGINT` | *lands with Increment A (#37)* |
| 13 | `loglik` | `DOUBLE` | *lands with Increment A (#37)* |
| 14 | `cov` | `LIST<DOUBLE>` | row-major k²; *lands with Increment A (#37)* |
```

- [ ] **Step 2: Verify the field table against the source**

Run: `grep -n "emplace_back" src/lm_fit_function.cpp | head -20`
Expected: fields 0–11 exactly as tabled (names, types, order) — they were read from `LmFitResultType()` on 2026-08-14; re-confirm nothing landed since. If #34/#35 have already merged fields 12–14 by execution time, drop the *"lands with Increment A"* italics for the landed rows instead.

- [ ] **Step 3: Commit**

```bash
git add docs/fitter_conventions.md
git commit -m "docs: fitter conventions — return-STRUCT standard (#17)"
```

---

### Task 3: Numeric portability + validation discipline sections

**Files:**
- Modify: `docs/fitter_conventions.md` (append two sections)

**Interfaces:**
- Consumes: `src/include/portable_string_hash.hpp`, `notes/engineering/2026-06-lm-fit-robust-se-aggregate.md`, `test/cpp/gen_lm_fit_fixtures.py`, `scripts/run-cpp-tests.sh`
- Produces: section anchors `## Numeric & portability discipline`, `## Validation discipline` referenced by Task 5's checklist

- [ ] **Step 1: Append both sections**

Append to `docs/fitter_conventions.md`:

```markdown
## Numeric & portability discipline

- **Deterministic by construction.** Identical inputs (and seed, where an
  explicit seed argument exists) give bit-identical results. Randomness only
  ever enters behind a seed argument (`bootstrap`, sampling functions).
- **The wasm string-hash landmine.** Never put a default-hash
  `std::string`-keyed `std::unordered_map`/`unordered_set` in extension
  code: it pulls in an unexported libc++ symbol (`std::__hash_memory`) and
  breaks the wasm build at load time. Use `PortableStringHash`
  (`src/include/portable_string_hash.hpp`) — or avoid hashing entirely:
  Finalize-side grouping is **sort-based** (the `lm_fit` CR path is the
  precedent).
- **Buffer rows when the estimator needs them.** Leverage-based (HC2/HC3)
  and score-based (CR*, REML-style) estimators don't reduce to fixed
  streaming moments — the aggregate state buffers raw rows and does the
  math in Finalize. Don't contort the math to force a streaming shape.
- **Math lives behind the kernel boundary.** Fitting numerics go in a
  DuckDB-free core TU (`src/lm_core.cpp` pattern) per
  [`kernel_api.md`](kernel_api.md); the `*_function.cpp` file only binds,
  buffers, and writes vectors.

## Validation discipline

- **Oracle offline, goldens committed.** Reference values come from
  statsmodels via a generator script (`test/cpp/gen_*.py`) run in a
  throwaway venv — the oracle is never a build or CI dependency. The
  generator self-checks its own conventions and documents any deliberate
  divergence from the oracle (e.g. conditional vs joint-information SEs).
- **Tolerance bands carry a written rationale.** A loose band is a finding,
  not a shrug — e.g. when the oracle's own optimizer termination noise is
  ~7e-6, SE goldens get a 1e-5 band and a comment saying exactly that.
- **Two test layers, both required.** Standalone C++ goldens
  (`scripts/run-cpp-tests.sh`, no DuckDB) for the kernel/core math, and
  sqllogictest (`test/sql/*.test`) for the SQL surface — float output via
  `printf('%.4f', …)` so results are deterministic text.
- **New-fitter checklist:** generator script + fixtures header → failing
  C++ goldens (incl. failure-as-value cases) → core implementation → SQL
  tests → README function-table row → cross-links here and in
  `kernel_api.md` → CHANGELOG entry.
```

- [ ] **Step 2: Verify the portability claims**

Run: `grep -rn "PortableStringHash" src/ | head -5` — expected: the header definition plus at least one use site (table_one/anova path).
Run: `grep -n "seed" src/bootstrap_function.cpp | head -5` — expected: an explicit seed argument. If `bootstrap` has no seed argument, change that bullet's parenthetical to name only the functions that do.

- [ ] **Step 3: Commit**

```bash
git add docs/fitter_conventions.md
git commit -m "docs: fitter conventions — portability + validation discipline (#17)"
```

---

### Task 4: Compliance audit appendix

**Files:**
- Modify: `docs/fitter_conventions.md` (append appendix)

**Interfaces:**
- Consumes: README function tables (lines ~47–66), `src/*_function.cpp` sources
- Produces: `## Appendix — existing-surface audit` table; any follow-up issues it spawns

- [ ] **Step 1: Enumerate the current function surface**

Run: `ls src/*_function.cpp` and cross-check against the README tables (`grep -n '^| \`' README.md`). Expected inventory (2026-08-14): the ~18 hypothesis-test functions (`ttest_*`, `mann_whitney_u`, `wilcoxon_signed_rank`, `*_test`, `anova_oneway`, `chisq_*`, normality tests, `ks_*`, `sign_test_*`), `summary_stats`, `bootstrap`, the distribution scalars, `lm`/`lm_summary` table functions, `lm_fit`, `read_stat`/COPY writers, `table_one`, VISUALIZE. Only STRUCT-returning statistical functions enter the audit; readers/writers/VISUALIZE are out of scope.

- [ ] **Step 2: Audit each in-scope function and append the appendix**

For each in-scope function read its `*_function.cpp` result-type + Finalize/execute path and fill one row. Append:

```markdown
## Appendix — existing-surface audit (2026-08)

The standard above is binding for new fitters. The pre-standard surface
deviates where noted; deviations are grandfathered (renaming released
fields would break consumers) unless marked with an issue link.

| Function | STRUCT return | NaN→NULL | Missing-data policy | Failure mode | Deviations from the standard |
|----------|---------------|----------|--------------------|--------------|------------------------------|
| `lm_fit` | ✓ (template above) | ✓ | listwise drop | NULL group / bind error | none — it defines the standard |
| … | | | | | |
```

one row per audited function, filled from the source just read — the ellipsis row is scaffolding to delete, not content to keep. Decision rule for the last column: a deviation is worth an issue only if it would surprise a consumer writing generic code across functions (e.g. an error thrown for a data-shaped failure); naming-only drift (`statistic` vs `t_statistic`) is recorded as grandfathered with no issue.

- [ ] **Step 3: File follow-up issues for non-grandfathered deviations**

For each (expected: zero or few): `gh issue create --title "<function>: <deviation>" --label kernel` with a body quoting the relevant standard bullet and the observed behavior. Link the issue number in the appendix row.

- [ ] **Step 4: Commit**

```bash
git add docs/fitter_conventions.md
git commit -m "docs: fitter conventions — existing-surface audit appendix (#17)"
```

---

### Task 5: Wiring + close-out

**Files:**
- Modify: `docs/kernel_api.md` (intro), `README.md` (docs link), `CHANGELOG.md` (`[Unreleased]`)

**Interfaces:**
- Consumes: the finished `docs/fitter_conventions.md`
- Produces: cross-links; #17 closed; #20/#21/#22 pointed at the standard

- [ ] **Step 1: Cross-link from kernel_api.md**

In `docs/kernel_api.md`, after the intro paragraph ending "…calls the kernel directly.", insert:

```markdown
The SQL-facing counterpart of this page is
[`fitter_conventions.md`](fitter_conventions.md) — the conventions every
model-fitting function's SQL surface follows (inputs, the return-STRUCT
standard, validation discipline).
```

- [ ] **Step 2: Link from README**

Run: `grep -n "kernel_api" README.md`. Add a sibling link in the same place (same list/sentence style as the surrounding README text) pointing to `docs/fitter_conventions.md` with the one-line description "conventions for model-fitting functions".

- [ ] **Step 3: CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` (create the section above `## [0.8.0-nothing]` if absent), add:

```markdown
### Added
- `docs/fitter_conventions.md` — the model-fitting API standard: input
  conventions, the return-STRUCT standard, numeric/portability and
  validation discipline, and an audit of the existing surface (#17).
```

- [ ] **Step 4: Commit and push**

```bash
git add docs/kernel_api.md README.md CHANGELOG.md
git commit -m "docs: wire fitter_conventions into README, kernel_api, CHANGELOG (#17)"
git push
```

- [ ] **Step 5: Point the consuming issues at the standard**

```bash
gh issue comment 20 --body "Design note: the fitter API standard (docs/fitter_conventions.md, #17) is now the binding reference for this function's SQL surface — return-STRUCT rules, cov layout (flat row-major LIST<DOUBLE>), df/rank conventions, and the validation checklist."
gh issue comment 21 --body "Design note: the fitter API standard (docs/fitter_conventions.md, #17) is now the binding reference for this function's SQL surface — return-STRUCT rules, cov layout (flat row-major LIST<DOUBLE>), df/rank conventions, and the validation checklist."
gh issue comment 22 --body "Design note: the fitter API standard (docs/fitter_conventions.md, #17) is now the binding reference for glm_fit's SQL surface — note especially the iterative-fitter rule (converged BOOLEAN + iterations BIGINT, best iterate returned with converged=false rather than NULL) and the z_statistic naming."
```

- [ ] **Step 6: Close #17**

```bash
gh issue close 17 --comment "Closed by docs/fitter_conventions.md (branch v0.9): input conventions, the return-STRUCT standard (with the lm_fit field table 0–14 as the worked template), numeric/portability + validation discipline, and an audit appendix over the existing surface. Follow-up issues were filed for any non-grandfathered deviations found by the audit; #20/#21/#22 now reference the standard as binding."
```

---

## Self-review (performed at write time)

- **Spec coverage** — #17's body names six conventions: aggregates over long rows (Task 1), design matrix as `x DOUBLE[]` (Task 1), one STRUCT return incl. the full covariance (Task 2), missing → listwise drop (Task 1), non-convergence via a status field / never throw (Task 2, iterative-fitter rule), reuse the distribution functions for p-values/CIs (Task 2). All covered; the audit (Task 4) and wiring (Task 5) make the standard verifiable and discoverable.
- **Placeholder scan** — every doc section is drafted verbatim in its task; the one intentional scaffold (the appendix's `…` row) is explicitly marked "delete, not content to keep"; verify steps carry concrete commands, expected outputs, and a decision rule for mismatches.
- **Type consistency** — field names/indices in Task 2's table match `src/lm_fit_function.cpp:39-50` (read 2026-08-14) and #37's approved 12–14 plan; section anchors referenced by Tasks 4–5 match the headings introduced in Tasks 1–3.
