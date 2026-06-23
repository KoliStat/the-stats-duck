# Engineering notes

Short writeups about non-obvious decisions and bugs that future contributors
(or future-us) will want to know about. Each note is dated and self-contained;
they are not living documentation, they are postmortems / decision records.

If you hit a subtle bug while developing on stats_duck and the fix was longer
than the diagnosis, please add a note here. It is much easier to find a thing
that has been written down than to re-derive it.

## Inventory

- [`2026-06-volatile-and-rng-bias.md`](2026-06-volatile-and-rng-bias.md) — two
  RNG footguns hit while landing the `r*` random-sampling family: DuckDB's
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

- [`2026-06-duckdb-version-must-match-duckdb-wasm.md`](2026-06-duckdb-version-must-match-duckdb-wasm.md)
  — `table_one()` crashed in `@duckdb/duckdb-wasm` with `r/t is not a function`.
  Overturned two hypotheses (the DuckDB version retarget and the `-sWASM_BIGINT`
  flag — `=1` actually LinkErrors, `=0` is required) before `wasm-dis` pinned the
  cause to **one** unresolved import: libc++ `std::__hash_memory`, pulled in by
  `std::unordered_map<std::string>` in anova/chisq and not exported by duckdb-wasm.
  Fixed by an in-extension string hash (`portable_string_hash.hpp`). Includes a
  reusable `@duckdb/duckdb-wasm` Node verification harness.
