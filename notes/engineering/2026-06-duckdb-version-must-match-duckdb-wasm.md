# `table_one` died in the browser with "r is not a function": the extension's DuckDB was newer than duckdb-wasm's

*June 2026, retargeting the extension from DuckDB v1.5.1 back to v1.4.3 so it
loads into the stable `@duckdb/duckdb-wasm`.*

The native extension was flawless — full suite green, `table_one()` returning
clean Table-1 output. Loaded into bedevere-wise (which runs DuckDB in the browser
via `@duckdb/duckdb-wasm`), the *same* `table_one()` query threw a bare
`r is not a function` from deep inside the duckdb-wasm glue. No SQL error, no
binder message — a JavaScript `TypeError`. Scalar stats worked; `table_one` did
not.

## TL;DR

1. **The extension `.wasm` was built against DuckDB v1.5.1; the stable
   duckdb-wasm host bundles DuckDB v1.4.3.** A loadable extension is an
   Emscripten *side module*: it shares the host's function table and imports
   DuckDB internals from the main module. Build it against a different DuckDB
   and those imports line up against an ABI that isn't there.
2. **It surfaces first on the struct-returning path.** `table_one` is a
   `TableFunction` that opens an internal `Connection` and runs SQL calling the
   struct-returning aggregates `summary_stats` / `anova_oneway` /
   `chisq_independence`. Struct return uses an sret-pointer calling convention
   that's the most layout-sensitive thing crossing the module boundary, so the
   mismatch cashes out there as an indirect call into the wrong table slot —
   `r is not a function`. Plain scalar calls happened to still line up, which is
   why only `table_one` broke.
3. **Fix: match the host. Rebuild the whole extension against v1.4.3.** Not a
   shim, not a flag — the DuckDB submodule version *is* the ABI. Move the
   `duckdb` + `extension-ci-tools` submodules to `v1.4.3`, flip the CI
   `duckdb_version` / `ci_tools_version`, port the few internal-API call sites,
   rebuild native + all wasm flavors.

## Why a version bump is an ABI reseat, not a recompile

duckdb-wasm loads `stats_duck.duckdb_extension.wasm` with `dlopen` semantics into
the already-running DuckDB main module. The extension does not statically contain
DuckDB; it *calls into* the host's DuckDB — `Connection`, `ExpressionExecutor`,
the aggregate-function machinery, vector layouts, `LogicalType` internals. All of
that is resolved at load time against the host module's exported symbols and
shared function table. There is no version negotiation beyond the coarse platform
string (`wasm_eh`): if the host is v1.4.3 and the extension was compiled with
v1.5.1 headers, every place a struct grew a field, a vtable reordered, or a
function signature shifted is now a silent mismatch. The loader is happy; the
first call that depends on the changed layout is not.

This is why the symptom is a JS `TypeError` and not a DuckDB exception: the
failure is below DuckDB, at the wasm function-table / sret-ABI level.

## The two source deltas v1.5.1 → v1.4.3

Moving the submodules surfaced exactly two internal-API call sites that the ggsql
(VISUALIZE) layer touches. Both are pure renames of *where the value lives*, not
behavior changes — DuckDB tightened these into accessor methods in 1.5.x and we
were using the 1.5.x form:

| 1.5.x (what we had) | 1.4.3 (what compiles) | What it is |
|---|---|---|
| `ParserExtension::Register(config, ext)` | `config.parser_extensions.push_back(ext)` | `DBConfig` exposes `parser_extensions` as a plain `vector<ParserExtension>` in 1.4.3; the static `Register` helper is a 1.5.x addition. |
| `func.{Get,Has,Set}ExtraFunctionInfo(...)` | the `func.function_info` field (`shared_ptr<ScalarFunctionInfo>`) | 1.4.3 `ScalarFunction` carries `function_info` as a public field; 1.5.x wrapped it in `Get/Has/Set` accessors. `MarkInfo : ScalarFunctionInfo`, so `function_info->Cast<MarkInfo>()` works unchanged. |

`src/ggsql.cpp` (1 site) + `src/ggsql_marks.cpp` (4 sites). No `#if` version
shims — we move wholesale to one DuckDB version, so the call sites just match it.
If we ever bump back up, these are the two to flip back.

## Two things that did *not* change (and why)

- **`-sWASM_BIGINT=0`** (CMakeLists.txt link flag). This legalizes `i64` →
  `(i32, i32)` at the module boundary to match how duckdb-wasm's bundle speaks.
  It's keyed to the *host's* JS↔wasm BigInt convention, not the DuckDB version,
  and 1.4.3's bundle predates native-i64 just like 1.5.1's — so it stays `=0`.
  Getting this wrong is a hard `LinkError` at load, not a runtime `TypeError`.
- **`exclude_archs: 'windows_amd64'`** in the CI pipeline. The MSVC build stays
  excluded: the hardcoded `vcvars64.bat` path in extension-ci-tools'
  `_extension_distribution.yml` (issue #371) is **byte-identical** in v1.4.3, so
  the bump doesn't un-break it. `windows_amd64_mingw` still ships for Windows.

## Local-only footnote: the fmt `_SECURE_SCL` patch

The native MSVC build on this dev box (VS2026 / toolset v145) needs a one-line
patch to `duckdb/third_party/fmt/include/fmt/format.h` — Microsoft removed
`stdext::checked_array_iterator`, and fmt's `#ifdef _SECURE_SCL` takes the
removed-symbol branch because the CRT auto-defines `_SECURE_SCL` to `0` (defined,
but zero). Changed to `#if defined(_SECURE_SCL) && _SECURE_SCL > 0`. This lives in
the submodule working tree and is **not** committed (the parent repo tracks only
the submodule SHA). CI is unaffected: `windows_amd64` (the only MSVC arch) is
excluded, and every other arch is GCC/Clang/Emscripten where `_SECURE_SCL` never
enters. Re-apply by hand if the submodule moves.

## Checklist for the next DuckDB version move

1. Pick the version the **target duckdb-wasm bundle** ships (check
   `@duckdb/duckdb-wasm`'s pinned DuckDB), not the newest DuckDB.
2. Move `duckdb` + `extension-ci-tools` submodules to that tag.
3. `MainDistributionPipeline.yml`: `duckdb_version`, `ci_tools_version`, and the
   `uses: ...@<ver>` reusable-workflow refs (deploy job too).
4. `make release`, fix internal-API deltas at the call sites (no shims), confirm
   the full suite passes — `table_one.test` is the canary for the struct-aggregate
   path.
5. Rebuild **all** wasm flavors (`wasm_mvp` / `wasm_eh` / `wasm_threads`);
   confirm artifacts land under `build/<flavor>/repository/<ver>/<flavor>/` and
   that the emcc link line still carries `-sWASM_BIGINT=0`.
6. Re-check `exclude_archs` against the new `extension-ci-tools` tag — drop
   `windows_amd64` if #371 is fixed there.
