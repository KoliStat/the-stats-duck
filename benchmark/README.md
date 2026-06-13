# Benchmarks

## `bench_xpt_read.sh` — XPT reader scan throughput

Measures how `read_stat()` scales with row count when reading SAS Transport
(`.xpt`) files. It generates files of increasing size with the extension's own
`COPY … TO '*.xpt'` writer, then times a full-scan `count(*)` through the reader
for each size.

```sh
# default: 25k / 50k / 100k / 200k rows, 1 rep each
benchmark/bench_xpt_read.sh

# custom sizes + repetitions (best wall time wins), and value-touching aggregate
ROWS="100000 1000000 10000000" REPS=3 WITH_AGG=1 benchmark/bench_xpt_read.sh
```

It auto-detects the CLI and extension under `build/` (override with `DUCKDB=` /
`STATS_DUCK_EXT=`). Timing comes from the CLI's `.timer on`, so process startup
and the extension `LOAD` are excluded. Generated files land in
`benchmark/.data/` (git-ignored, deleted on exit unless `KEEP_DATA=1`).

The number to watch is the **ratio** column: per doubling of rows, ~2× time is
linear, ~4× is quadratic.

### Real datasets

To benchmark existing `.xpt` files instead of synthetic ones, pass `FILES=` (a
space-separated list) or `DIR=` (scanned for `*.xpt`). Row count is read from the
query, so unknown-size files work. Unreadable files show `ERR`.

```sh
DIR=../icpsr-data benchmark/bench_xpt_read.sh
FILES="/data/study1.xpt /data/study2.xpt" benchmark/bench_xpt_read.sh
```

**Keep licensed / access-controlled data outside the repo working tree.** Point
the runner at a sibling directory (e.g. `../icpsr-data`), not anything under the
repo. Datasets such as ICPSR/openICPSR releases are distributed under a
click-through agreement (confidentiality of research subjects, country-of-concern
sharing restrictions); committing or pushing them is redistribution and a terms
violation. Storing them outside the tree makes an accidental `git add` (or a
typo'd `.gitignore` path, or `git add -f`) structurally impossible — far safer
than relying on the ignore rule for `benchmark/.data/`. Don't paste numbers
derived from such files into committed docs either; the synthetic sweep below is
what belongs in the repo.

A good **public** real-world corpus is the [CDISC SDTM/ADaM pilot
submission](https://github.com/cdisc-org/sdtm-adam-pilot-project) — synthetic
demonstration data, freely redistributable, with realistic clinical-trial
schemas (wide tables, many character columns, real labels/formats). Point the
runner at its dataset folders with `DIR=`. Most datasets are small (≤8k rows,
sub-second), but the long-format ones reach useful scale: `qs.xpt` is 121,749
rows across ~50 columns and reads in **39 s** on the current reader. Its rows are
~7× wider than the synthetic schema (≈265 vs 36 B/row), so at a comparable row
count its per-row cost is ~2× higher (323 µs/row at 122k vs the synthetic 156 at
100k): the O(N²) penalty scales with re-read *bytes*, so wide real tables suffer
more, not less.

Caveat: on the **current O(N²) reader** a large real file is effectively
un-benchmarkable — extrapolating the baseline (`≈ 67 s × (N/200k)²`), 1M rows is
~28 min and 5M rows ~12 h. Massive files are the right tool to confirm the *fix*
scales linearly; for the baseline, the synthetic sweep is what completes.

### Baseline — before any optimization

Windows, MSVC `build/release`, AC power, warm page cache. 5-column schema
(`int→double, double, double, varchar, date`):

| rows | count(*) s | µs/row | ratio |
|-----:|-----------:|-------:|------:|
| 25k  |      1.30  |   52.0 |   —   |
| 50k  |      4.38  |   87.6 | 3.4×  |
| 100k |     15.61  |  156.1 | 3.6×  |
| 200k |     67.29  |  336.5 | 4.3×  |

~4× per doubling, and µs/row roughly doubles per step at the larger sizes: the
scan is **O(N²)**. A 200k-row, 7 MB file takes 67 s. `WITH_AGG=1` (materializing
every cell) costs essentially the same as `count(*)`, which tells you the
re-parse — not value conversion — is the cost. (On battery the CPU throttles and
these run ~1.3–1.4× slower; measure on AC.)

### Why it's quadratic

`ReadStatExecute` (`src/read_stat_function.cpp`) is called once per 2048-row
DuckDB chunk, and **each call re-parses the file from the start**: it inits a
fresh parser, re-opens the file, and uses `readstat_set_row_offset(state.offset)`
to skip to where the previous chunk ended.

The catch is that XPT's row-offset is not a seek. In
`third_party/readstat/src/sas/readstat_xport_read.c` the skip path reads and
decodes every skipped row, only suppressing the value callback:

```c
if (ctx->handle.value && !ctx->variables[i]->skip && !ctx->row_offset) { … }  // emit only at offset 0
…
if (ctx->row_offset) { ctx->row_offset--; }   // else: read the row off the VFS, decode it, discard
else { ctx->parsed_row_count++; }
```

So chunk *k* reads `(k+1)·2048` rows, and a full scan of *N* rows reads/decodes
≈ **N²/4096** rows. The large `sys` time in the baseline is the kernel servicing
all those redundant reads (the file is re-read from byte 0 every chunk; for a
remote `httpfs`/S3 file it would re-fetch). See the reader analysis in the
project notes for the fix options (parse-once buffering; background-thread
producer; projection pushdown via `READSTAT_HANDLER_SKIP_VARIABLE`).

### What "fixed" looks like

Once the file is parsed a single time, the ratio column should settle near
**2.0×** (linear) and µs/row should flatten, letting you push `ROWS` into the
millions without the run blowing up.
