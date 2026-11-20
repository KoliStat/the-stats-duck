# The Tukey multiplier cache in emmeans (#72)

Date: 2026-11-19. Branch `feat/72-tukey-multiplier-cache`.

## What was slow

`emmeans` with `adjust = 'tukey'` (the default) puts its pairwise confidence
limits at `qtukey(1 - alpha, k, df) / sqrt(2)` times the standard error.
`StudentizedRangeQuantile` brackets and bisects `StudentizedRangeCDF`, a
double integral (see `2026-10-studentized-range-quadrature.md`), and costs
about 15 ms per call on this machine. It ran once per row. Over 1000 rows of
a five-mean grid it was 15 of the 17.8 s; `'bonferroni'` on the same rows
took 0.08 s. The adjusted p-values cost one CDF evaluation per pair, about
0.3 ms each, which is the rest.

## What changed

The quantile depends only on `(1 - alpha, k_compare, df)`, and a `GROUP BY`
over many groups of equal size asks for the same triple on every row.
`CachedStudentizedRangeQuantile` in `src/emmeans.cpp` keeps the last sixteen
distinct triples with their quantile and returns a hit without the search.
Keys are compared exactly: a different `df` is a different entry, and a NaN
never matches. Replacement is round-robin.

## Why thread_local and not a mutex

DuckDB runs a scalar function on several threads at once, so a plain static
table would be a data race, which is why #21 shipped without the cache. The
two safe shapes are a shared table behind a mutex and a per-thread table. The
per-thread table was chosen: it needs no lock, it cannot race by construction,
and the price is one derivation per worker thread per distinct key instead of
one per query, a few multiples of 15 ms on a query that took seconds. A
`thread_local` of a trivially destructible array is also safe in an extension
loaded as a DLL and in the single-threaded wasm build, where the extension's
`thread_local` random generator already lives.

## Measured

1000 rows, five-mean grid, `df = 20`, `alpha = 0.05`, VS 2022 release build,
the `duckdb.exe` shell with the extension linked in, `.timer on`:

| run | before | after |
|---|---|---|
| `'tukey'`, one triple on every row | 17.75 s | 2.73 to 2.91 s |
| `'bonferroni'`, same rows | 0.087 s | 0.07 s |
| `'tukey'`, a different `df` on every row | 18.44 s | 17.54 s |

The third row is the cache's worst case and shows that its own cost is nil:
sixteen comparisons and one write per miss. One repeat of that run read
21.4 s of wall time with 38.8 s of CPU, twice what one thread can use, and the
next two repeats, with and without `PRAGMA threads = 1`, read 17.5 s with CPU
equal to wall. The outlier was something else on the machine, not the query.
Measure twice before quoting a number.

## Tests

`test/sql/emmeans.test` runs one query over six `(df, alpha)` rows, two of
them repeated, and checks every row's lower limit against `qtukey` computed
directly for that row, so a stale entry served for the wrong key would show.
A second query uses a two-mean grid at a triple seen before with `k = 3`.
`test/cpp/test_emmeans.cpp` does the same at the kernel level. Neither test
can fail before the change: the cache changes no result, only its cost, so
these are regression guards, not the red test of a behaviour change.
