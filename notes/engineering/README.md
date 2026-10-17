# Engineering notes

Short writeups about non-obvious decisions and bugs that future contributors
(or future-us) will want to know about. Each note is dated and self-contained;
they are not living documentation, they are postmortems / decision records.

If you hit a subtle bug while developing on stats_duck and the fix was longer
than the diagnosis, please add a note here. It is much easier to find a thing
that has been written down than to re-derive it.

## Inventory

- [`2026-06-volatile-and-rng-bias.md`](2026-06-volatile-and-rng-bias.md) — two
  RNG pitfalls hit while building the `r*` random-sampling family: DuckDB's
  `UnaryExecutor` caches the lambda result for constant-vector inputs (the
  `VOLATILE` flag does not disable this), and MSVC's
  `std::uniform_real_distribution<double>` clips the upper tail. Fixes and
  diagnostic checklist.

- [`2026-06-xpt-reader-quadratic.md`](2026-06-xpt-reader-quadratic.md) — the
  `read_stat()` XPT/SAS/SPSS reader was O(N²) because it re-parsed the file per
  output chunk and ReadStat's `row_offset` reads-and-decodes skipped rows instead
  of seeking (and `bind` read the whole data section for the schema). Fix:
  parse-once buffering + header-only bind; why not a producer thread (WASM).

- [`2026-06-bootstrap-rng-portability.md`](2026-06-bootstrap-rng-portability.md) —
  `bootstrap`'s `seed` promised reproducibility, but `std::uniform_int_distribution`
  is implementation-defined, so the same seed produced different resample streams
  on libstdc++ / libc++ / MSVC. Fix: draw indices from `mt19937_64` directly with
  an unbiased rejection bound. General rule: consume the engine, not the distribution.

- [`2026-06-lm-fit-robust-se-aggregate.md`](2026-06-lm-fit-robust-se-aggregate.md)
  — why `lm_fit` (OLS aggregate with HC0–HC3 SEs) **can't stream**: HC2/HC3
  leverages `hᵢᵢ = xᵢᵀ(X'X)⁻¹xᵢ` and cluster scores don't reduce to sufficient
  statistics, so the state buffers raw rows. Also: building the first
  `LIST<STRUCT>` aggregate output, and why aggregates take positional constants
  (not `:=` named params) — which is what makes its coefficients by-position.

- [`2026-10-lin-hyp-rank-tolerance-and-scale.md`](2026-10-lin-hyp-rank-tolerance-and-scale.md)
  — `lin_hyp`'s joint test reported `df_num` 1 instead of 2, and an F of 9.47
  instead of 132.53, when one predictor was rescaled by 1e3 and another by
  1e-3. `linalg::pinv` decides the numerical rank against `1e-12 · σ_max`, and
  the matrix it was given carries variances on its diagonal, so widely
  different constraint scales read as redundant. Fix: equilibrate C by its
  diagonal first, `C⁻ = D·(D·C·D)⁻·D`. The design spec had mandated the
  unconditioned call. Applies to any kernel that inverts a covariance with
  `pinv`, `emmeans` (#21) included.

- [`2026-06-duckdb-version-must-match-duckdb-wasm.md`](2026-06-duckdb-version-must-match-duckdb-wasm.md)
  — `table_one()` crashed in `@duckdb/duckdb-wasm` with `r/t is not a function`.
  Overturned two hypotheses (the DuckDB version retarget and the `-sWASM_BIGINT`
  flag — `=1` actually LinkErrors, `=0` is required) before `wasm-dis` pinned the
  cause to **one** unresolved import: libc++ `std::__hash_memory`, pulled in by
  `std::unordered_map<std::string>` in anova/chisq and not exported by duckdb-wasm.
  Fixed by an in-extension string hash (`portable_string_hash.hpp`). Includes a
  reusable `@duckdb/duckdb-wasm` Node verification harness.

- [`2026-10-studentized-range-quadrature.md`](2026-10-studentized-range-quadrature.md)
  — the studentized range distribution (#21) passed all 224 scipy goldens, all
  four identities and the k = 2 closed form to 1.2e-15, while carrying a 1.8e-9
  error at k = 20. Why the closed-form case certifies nothing about the hard
  one, why self-convergence against an over-resolved copy of the same rule is
  the check that found it, why nodes beat panels for an entire integrand
  (48 x 2 is 2.3e-15 and 10 x 10 is 1.8e-9 at the same cost), and the branch
  point in `s^(df-1)` that leaves a non-integer df below 4 less accurate than
  the band. Also: `GammaP` stops after 200 terms, so `ChiSquareQuantile` drifts
  above about df = 1e4.
