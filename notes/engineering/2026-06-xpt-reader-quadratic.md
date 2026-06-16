# The XPT reader was O(N²): ReadStat's `row_offset` reads, it doesn't seek

*June 2026, while making the `read_stat()` SAS / SPSS / Stata reader performant.*

Reading a SAS Transport (`.xpt`) file with `read_stat()` was pathologically slow
— not "needs tuning" slow, *quadratic* slow. A 200k-row, 7 MB synthetic file took
67 s; the real-world CDISC pilot `qs.xpt` (121,749 rows, ~50 columns) took 39 s.
These files are small. Something was deeply wrong with how we scanned them.

## TL;DR

1. **`read_stat()` re-parsed the whole file from the start for every 2048-row
   output chunk.** DuckDB pulls a table function chunk by chunk; the old
   `ReadStatExecute` answered each pull by spinning up a *fresh* ReadStat parser
   with `row_offset = chunk_start`, `row_limit = STANDARD_VECTOR_SIZE`.
2. **ReadStat's `row_offset` is not a seek.** It reads and decodes every skipped
   row off the VFS and merely suppresses the value callback. So chunk *k* reads
   `(k+1)·2048` rows, and a full scan of *N* rows reads ≈ **N²/4096** rows. Each
   chunk also re-opened the file and re-read its header records — and for an
   httpfs/S3 path, re-fetched the bytes.
3. **Fix: parse once.** Parse the entire file a single time at `InitGlobal` into
   a spillable `ColumnDataCollection`; `Execute` streams chunks out of it.
   O(N²) → O(N). Separately, `bind` was reading the *whole data section* just to
   learn the schema — it now aborts after the header.

## How we found it

`benchmark/bench_xpt_read.sh` times a full-scan `count(*)` over synthetic XPT
files of growing size. The ratio between consecutive sizes is the tell — ~2× per
row doubling is linear, ~4× is quadratic:

| rows | count(*) s | µs/row | ratio |
|-----:|-----------:|-------:|------:|
| 25k  |   1.30 |  52.0 |  —   |
| 50k  |   4.38 |  87.6 | 3.4× |
| 100k |  15.61 | 156.1 | 3.6× |
| 200k |  67.29 | 336.5 | 4.3× |

~4× per doubling, and the per-row cost itself doubles each step. That's N², not a
constant factor. The kernel `sys` time dominated (67 s wall, ~70 s sys at 200k):
the signature of re-reading the same bytes over and over.

## Why `row_offset` doesn't help

The natural assumption is that `readstat_set_row_offset(parser, k)` seeks past the
first *k* rows. It does not. From `third_party/readstat/src/sas/readstat_xport_read.c`,
the per-cell loop:

```c
if (ctx->handle.value && !ctx->variables[i]->skip && !ctx->row_offset) {
    ctx->handle.value(ctx->parsed_row_count, variable, value, ctx->user_ctx); // emit only at offset 0
}
...
if (ctx->row_offset) { ctx->row_offset--; }   // else: the row was read off the VFS and decoded — then dropped
else { ctx->parsed_row_count++; }
```

The row's bytes are read (`xport_read_row`) and each variable is decoded into a
`readstat_value_t`; `row_offset` only gates the *callback*. So "skip to row k"
costs a full read+decode of rows `0..k-1`. With a fresh parser per chunk, the
work is `Σ (k+1)·2048 ≈ N²/4096`.

A second, quieter cost: `readstat_parse_xport` unconditionally calls
`xport_read_data()` whenever the file has columns — it does **not** check whether
a value handler is registered. Our `bind` registered only metadata + variable
handlers (to read the schema) with `row_limit = 0`, and `row_limit = 0` means *no
limit*, not *no rows*. So every `bind` streamed and decoded the entire data
section just to throw it away — one extra full read per query.

## The fix

`ReadStatInitGlobal` now parses the file exactly once into a
`ColumnDataCollection` (allocated against the client context, so it spills to the
buffer manager on native; in WASM it stays resident, which is fine for the data
sizes WASM handles). The ReadStat value handler stages rows into a `DataChunk`
and appends it whenever it fills:

```cpp
while (oi - lc.flushed >= STANDARD_VECTOR_SIZE) {       // crossed a vector boundary
    lc.staging->SetCardinality(STANDARD_VECTOR_SIZE);
    lc.buffer->Append(*lc.append_state, *lc.staging);
    lc.staging->Reset();
    lc.flushed += STANDARD_VECTOR_SIZE;
}
```

`Execute` is now just `InitializeScan` once, then `Scan` per pull. And `bind`
registers a value handler that returns `READSTAT_HANDLER_ABORT` on the first
value — the schema is fully known from the header (every variable handler fires
before any row), so we stop before the data section. `READSTAT_ERROR_USER_ABORT`
is treated as success (an empty file finishes with `READSTAT_OK`, no value seen).

Result: the 200k synthetic scan dropped 67 s → 1.3 s (52×), `qs.xpt` (122k rows)
39 s → 1.1 s (36×), and µs/row went flat at ~6.4 (it had climbed 52 → 336 across
the sweep) — the O(N) signature. 1.6M rows now read in ~15 s; pre-fix that was
~70 min by extrapolation. All 448 reader/round-trip test assertions still pass.

## Trade-offs, and why not threads

Parsing once is **eager**: the whole file is read at `InitGlobal`, before the
first row is returned. That's a regression for one shape — `SELECT * FROM
read_stat('huge.xpt') LIMIT 5` now reads the whole file instead of one chunk —
but XPT is overwhelmingly a full-dataset import format, and the O(N²) full scan
was the case that actually hurt.

The streaming-and-O(N) ideal is a producer thread: run ReadStat start-to-finish
once on a worker, push filled chunks through a bounded queue, let `Execute`
consume, abort on early-out. That keeps `LIMIT` cheap *and* memory constant. We
didn't do it because ReadStat is a push parser (it owns the loop and can't be
suspended without a separate stack), so the producer needs a real thread — and
`wasm_mvp` / `wasm_eh` are single-threaded. It would mean two code paths
(threaded native + `wasm_threads`, buffered fallback for mvp/eh). Parse-once is
the WASM-safe baseline that ships everywhere; the threaded path is a future option
if the eager read ever bites.

## Left on the table (constant-factor wins, now visible)

With the quadratic gone, these finally show up on the meter:

- **Projection pushdown.** ReadStat supports `READSTAT_HANDLER_SKIP_VARIABLE`;
  wiring `column_ids` → skip flags avoids decoding + allocating unselected
  columns (big for wide tables — see `qs.xpt`'s ~50 columns).
- **String fast-path.** `SanitizeUtf8` heap-allocates a `std::string` for every
  string cell; validate in place and `AddString` straight from ReadStat's buffer
  on the (overwhelmingly common) already-valid path.
- **Hoist epoch detection.** `DetectEpochSystem` runs a chain of string compares
  per *temporal cell*; compute it once per column at bind.
