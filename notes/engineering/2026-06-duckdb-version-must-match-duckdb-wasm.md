# table_one crashes in @duckdb/duckdb-wasm with "r is not a function" — and it is NOT the WASM_BIGINT flag

*June 2026. An investigation that overturned two plausible hypotheses before
landing a one-symbol fix. **Status: FIXED** — `table_one` runs in
`@duckdb/duckdb-wasm@1.32.0`, verified end-to-end (see the harness section).*

`table_one()` works natively but, loaded into `@duckdb/duckdb-wasm@1.32.0`,
throws a bare `TypeError: r is not a function` (the minified variable name varies
— also seen as `t`) from the worker glue. `meta()` and most functions work; only
`table_one` breaks.

## TL;DR (what we proved empirically, in order)

1. **It is not (only) the DuckDB version.** The extension was on DuckDB v1.5.1
   while duckdb-wasm 1.32.0 reports v1.4.3, so we retargeted the whole extension
   to v1.4.3 (submodules, ggsql API ports — see appendix). Native parity
   restored, full suite green. **But the wasm `table_one` crash persisted.**
2. **It is not the `-sWASM_BIGINT` flag.** The standing belief was that
   `-sWASM_BIGINT=0` (legalize i64 → (i32,i32)) was wrong and `=1` (native i64,
   matching duckdb-wasm's BigInt-marshalling worker) would fix it. Empirically:
   - **`=1` does not even load**: `LinkError` at instantiation — *"Import #9
     `env._ZN6duckdb17InternalException...IJyyEE...` imported function does not
     match the expected type"*. duckdb-wasm's dyld hands side modules the
     **legalized** form of duckdb's i64 exports, so a native-i64 extension's
     imports don't match. `=0` is **required** just to link.
   - **`=0` loads but `table_one` still crashes** with `t is not a function`.
3. **The real failure is narrow.** Probing each entry point individually under
   `=0` + v1.4.3:

   | call | uses `std::unordered_map<std::string>` | result |
   |------|:--:|--------|
   | builtin `count`/`sum`            | – | PASS |
   | `pnorm` (ext scalar)             | – | PASS |
   | `summary_stats` (struct agg)     | no | **PASS** |
   | `corr_matrix` (table fn)         | – | PASS |
   | `VISUALIZE … DRAW` (parser ext)  | – | PASS |
   | **`anova_oneway`** (struct agg)  | **yes** | **FAIL `t is not a function`** |
   | **`chisq_independence`**         | **yes** | **FAIL** (same) |
   | `table_one`                      | calls anova/chisq | FAIL |

   The crash correlates exactly with **`std::unordered_map<std::string, …>` /
   `std::string` aggregate state** (`anova_function.cpp:61`,
   `chisq_function.cpp:71`). `summary_stats` is also a struct-returning aggregate
   dispatched the same way but uses plain numeric state — and it works. So this
   is **not** about struct returns, the internal `Connection`, or i64 in general.

## The exact culprit: one unresolved symbol

Disassembling the `=0` extension with `wasm-dis` and subtracting both
duckdb-eh.wasm's exports **and** the extension's own exports (a side module can
satisfy its own `env` imports from its own exports — PIC self-reference) leaves
**exactly one** genuinely unresolved import:

```
_ZNSt3__213__hash_memoryEPKvm  =  std::__1::__hash_memory(const void*, size_t)
```

`std::__hash_memory` is libc++'s byte-hash, called by `std::hash<std::string>`
inside `std::unordered_map<std::string,…>`. duckdb keys its own maps on
`duckdb::string_t` with its own hash, so duckdb-wasm's main module never
instantiates or exports `__hash_memory`. The Emscripten loader stubs the
unresolved import with a throwing placeholder, and the first `std::string` key
hash in anova/chisq calls it → `t is not a function`. `summary_stats` etc. never
hash a `std::string`, so they have nothing stubbed.

## The fix (landed)

Hash `std::string` keys **inside the extension** so `std::hash<std::string>` /
`__hash_memory` is never referenced — `src/include/portable_string_hash.hpp`
(FNV-1a), used as the third template arg of every `std::unordered_map`/`set` with
a `std::string` key: `anova_function.cpp`, `chisq_function.cpp`,
`read_stat_types.cpp`. After this the extension has **zero** unresolved imports;
the harness probe matrix goes all-green (anova/chisq/table_one included) and the
exact repro returns real rows (`p=1.6e-4`, η²`=0.86`). Native suite unchanged
(3878 assertions) — FNV vs std::hash only changes bucketing, not results.

The two paths we *didn't* need: building against duckdb-wasm's exact fork tree,
and static-linking libc++ into the side module. The real problem was a single
libc++ symbol we could just stop depending on.

## Keep `-sWASM_BIGINT=0`

Necessary to load at all (see #2). Removing it reintroduces the instantiation
`LinkError`. CMakeLists.txt documents this inline.

## Reproducing — the Node harness

`scripts/wasm-verify/` (untracked; `npm i` pulls `@duckdb/duckdb-wasm@1.32.0` +
`web-worker`) loads the freshly-built
`build/wasm_eh/.../stats_duck.duckdb_extension.wasm` into the real worker EH
bundle and runs the probe matrix above. Windows gotchas baked into the harness:
duckdb-wasm's node runtime caches fetched extensions to
`~/.duckdb/extensions/<host:port>/…` and a `:` is an illegal Windows path
component, so `patch-worker.mjs` sanitizes the `host:port` colon across the
runtime's fs calls; the extension is served over a throwaway localhost http repo.
`wasm-dis` (binaryen, in emsdk) confirms the import signatures (`=0`: 0 i64
imports; `=1`: 43, e.g. `TemplatedValidityMaskIyE4Copy` as `(i32,i32,i64)`).

## Appendix — the v1.5.1→v1.4.3 retarget (the part that IS fixed)

Necessary for native/wasm version parity even though it didn't fix the wasm
crash. Two internal-API call sites moved:
`ParserExtension::Register(config,ext)` → `config.parser_extensions.push_back(ext)`
(DBConfig field); `ScalarFunction::{Get,Has,Set}ExtraFunctionInfo(...)` → the
`function_info` `shared_ptr<ScalarFunctionInfo>` field (`src/ggsql.cpp` ×1,
`src/ggsql_marks.cpp` ×4). Full native suite passes on v1.4.3.
