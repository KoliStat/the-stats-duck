# wasm-verify — stats_duck × @duckdb/duckdb-wasm smoke test

Loads the locally-built `stats_duck` **wasm_eh** loadable extension into
`@duckdb/duckdb-wasm@1.32.0` (the async worker EH bundle — the same runtime
bedevere-wise uses) and runs a probe matrix, **including `table_one`**, to catch
WASM-only ABI regressions the native test suite cannot see.

## Why this exists

A loadable extension is an Emscripten *side module*: it imports libc++ and DuckDB
internals from duckdb-wasm's main module. If the extension references a symbol the
main module doesn't export, the Emscripten loader stubs it and the call traps at
runtime as `TypeError: t is not a function` — even though both the native build and
the wasm build succeed. That is exactly how `table_one` (via `anova_oneway` /
`chisq_independence`) crashed on libc++ `std::__hash_memory`. See
`notes/engineering/2026-06-duckdb-version-must-match-duckdb-wasm.md` for the full
diagnosis and `scripts/check-wasm-string-hash.sh` for the regression guard.

## Run

```sh
# 1. build the extension (from the repo root)
PATH="$PWD/scripts/em-shims:$PATH" make wasm_eh

# 2. install + run (from this directory)
cd scripts/wasm-verify
npm ci            # or: npm install
node verify.cjs
```

Expected tail:

```
[summary] baseline-builtin=PASS  scalar-ext=PASS  struct-agg-direct=PASS  anova-direct=PASS  corr_matrix-tablefn=PASS  visualize=PASS  table_one-internalconn=PASS  lm_fit-hc1=PASS  lm_fit-cluster-cr1=PASS  lm_fit-cluster-unnest=PASS
PROBE_DONE
```

The `lm_fit-cluster-*` probes exercise the cluster-robust path (`DensifyClusters` +
the cluster sandwich) in-wasm; they exist to prove the sort-based cluster grouping
imports no `std::__hash_memory` (it would trap as `t is not a function`, exactly
like the original `table_one` hash regression).

Any `FAIL` — especially `t is not a function` — is a real WASM ABI regression.

## Notes

- Keep `@duckdb/duckdb-wasm` pinned to the version bedevere-wise ships (currently
  `1.32.0`); bump both here and the C++ build when it moves.
- Reads `build/wasm_eh/repository/<version>/wasm_eh/stats_duck.duckdb_extension.wasm`
  (version-agnostic — served verbatim over a throwaway localhost http repo).
- `patch-worker.mjs` wraps the duckdb node worker to sanitize a Windows-only colon
  in duckdb-wasm's `~/.duckdb/extensions/<host:port>/…` cache path (illegal on NTFS).
- `node_modules/` and `*.log` are gitignored.
