# DuckDB 2.0 Deprecated Calls (#79) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every call that DuckDB 2.0 deprecates from the binding layer, and make the table functions fill their output chunks the way 2.0 expects.

**Architecture:** Two shims go into `src/include/duckdb_compat.hpp`, the one file that knows about DuckDB versions: `compat::ToUnified` and `compat::SetCardinality`. `DataChunk::SetValue` needs no shim, because `Vector::SetValue` is the same call on every line. A shell script compiles each source file against a DuckDB header tree with `g++ -fsyntax-only` and lists the deprecated calls. It is the failing check for the issue, and a CI job keeps it passing.

**Tech Stack:** C++11 on DuckDB 1.4.x and 1.5.x, C++17 on DuckDB 2.0. MSVC (VS 2022) for the builds, TDM-GCC 10.3 for the check script. sqllogictest for the SQL tests. Bash for the script. GitHub Actions for CI.

**Spec:** GitHub issue #79, and the #79 bullet of block 3 in `notes/plans/2026-10-29-v0.10-plan.md`: "a `ToUnified` shim in `duckdb_compat.hpp` for the counted form, and the two `DataChunk` calls handled per site." The section "Findings before the plan" below holds what was measured on 2026-11-23. The plan argues from those findings.

## Findings before the plan

All of this was measured on `v0.10` at `3325c87`, on Windows, with fresh builds.

1. **The baseline.** DuckDB 1.4.3: 3049 assertions in 76 test cases, build with 0 warnings. DuckDB 2.0.0: 3049 assertions in 76 test cases, build with 0 warnings.
2. **The deprecated calls.** `g++ -fsyntax-only -Wdeprecated-declarations` against the 2.0.0 headers reports 126 call sites in `src/`: 73 of `Vector::ToUnifiedFormat(idx_t, UnifiedVectorFormat &)`, 41 of `DataChunk::SetValue`, 12 of `DataChunk::SetCardinality(idx_t)`. Against the 1.4.3 and the 1.5.6 headers it reports none, with `-std=c++11`. The MSVC builds show none of these warnings.
3. **`ToUnifiedFormat`.** On 2.0 the counted form is one line that calls the form without a count (`src/common/types/vector.cpp:454`). Replacing it changes no behaviour. On 1.x only the counted form exists, and it is not `const`.
4. **`DataChunk::SetValue`.** On both lines its body is `data[col_idx].SetValue(index, val)`. Calling `Vector::SetValue` on the column is the same operation.
5. **`DataChunk::SetCardinality` is a real fault on 2.0, not only a warning.** On 2.0 every vector carries its own size. `SetCardinality` sets the row count of the chunk and leaves the vectors alone. `Vector::SetValue` and writes through a raw data pointer do not change a vector's size either. So our table functions hand back a chunk that says N rows while each vector says 0. Query results are right, because the operators that follow use the chunk's count. The engine's check is not satisfied: with `SET debug_verification_mode = 'VERIFY_VECTORS'` the statement fails with `INTERNAL Error: DataChunk::Verify - size mismatch: vector 0 (VARCHAR) has size 0 but chunk has size 3`. Running the whole suite on 2.0 with that setting (`--on-init`) fails 34 of the 76 test files, all with that error. Each of these fails when called alone: `corr_matrix`, `lm`, `lm_summary`, `meta`, `read_stat_metadata`, `table_one`, `VISUALIZE`. `read_stat` passes, because it scans a buffer; its staging chunk has the same mismatch, but nothing checks that chunk.
6. **The replacement is `SetChildCardinality`.** It sets the size of every flat or constant vector and the count of the chunk. `CheckCardinality` only compares, so it throws for a chunk that was filled by index. DuckDB's own table functions that write by index call `SetChildCardinality` (`read_text`, in `src/function/table/direct_file_reader.cpp:194`). DuckDB 1.5.6 carries `SetChildCardinality` as an alias of `SetCardinality`, with a comment that says so. DuckDB 1.4.3 does not have the name.
7. **A throwaway spike confirmed the scope.** With the 12 `SetCardinality` calls routed to `SetChildCardinality` on 2.0, the whole suite passes on 2.0 with vector verification on: 3049 assertions in 76 test cases. Nothing else fails. The spike was discarded.
8. **`statement maybe` works on both test runners.** It accepts success, or an error that contains the expected text. On 1.x `SET debug_verification_mode = ...` fails with `Catalog Error: unrecognized configuration parameter "debug_verification_mode"`. The setting is process-wide on 2.0 (a static), so a test file that switches it on switches it off again at its end.

## Global Constraints

- Branch `feat/79-duckdb2-deprecations`, based on `v0.10` at `3325c87`. Do not push and do not open a pull request. The controller does both.
- The sources compile on three DuckDB lines: 1.4.3 (the `duckdb/` submodule, C++11), 1.5.6 (C++11) and 2.0.0 (C++17). Nothing newer than C++11 outside a `#ifdef STATS_DUCK_DUCKDB_V2` or `#ifdef STATS_DUCK_DUCKDB_VECTOR_V2` branch.
- A difference between DuckDB lines lives in `src/include/duckdb_compat.hpp` and nowhere else. No `#if` on a DuckDB version in any other file.
- No behaviour changes on the 1.x lines. Every existing test keeps its result on both lines.
- TDD. Run the failing check or test and record its output before the production change. Record the passing output after it.
- Never run two builds of the same tree at once, and never run two `unittest.exe` at once. They share a temp directory.
- Never pipe a build through `tail` or `head`. Redirect it to a log file and read the log. Give a build a 10 minute timeout.
- A build must end with 0 warnings: `grep -c warning <log>` prints `0`.
- Stage files by name. Never `git add -A` and never `git add .`. `git status` always shows ` m duckdb`. That is expected. Never stage `duckdb`.
- Every commit message ends with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, or the name of the model that makes the commit.
- Do not change a file's line endings. `*.sh` files use LF.
- Writing style, for code comments, commit messages and documents: short sentences, one idea each. State the fact, not an adjective. No buzzwords (leverage as a verb, seamless, powerful, comprehensive, robust as praise, streamline, first-class, performant, elegant, utilize, facilitate). No slang or metaphors (plumbing, landmine, footgun, gotcha, baked in, under the hood, out of the box, sweep, story). No em-dash chains.

### Build and test commands

Run them from the repository root in Git Bash.

```bash
# DuckDB 1.4.3
cmake --build build/release --config Release --target unittest -- -m:1 > "$LOG" 2>&1
build/release/test/Release/unittest.exe "test/sql/*"
build/release/test/Release/unittest.exe "test/sql/vector_verification.test"

# DuckDB 2.0.0. The test path must be absolute and in Windows form here.
# A relative path runs DuckDB's own tests. A /c/... path is read as an option.
cmake --build build/release-v2.0.0 --config Release --target unittest --parallel 8 > "$LOG" 2>&1
build/release-v2.0.0/test/Release/unittest.exe "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"
build/release-v2.0.0/test/Release/unittest.exe "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/vector_verification.test"

# DuckDB 2.0.0, every test file with vector verification on
build/release-v2.0.0/test/Release/unittest.exe --on-init "SET debug_verification_mode='VERIFY_VECTORS';" "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"
```

A rebuild after a change to `duckdb_compat.hpp` compiles every binding file. It takes about 2 to 5 minutes per line. The header trees for the check script are `duckdb` (1.4.3), `build/duckdb-v1.5.6-include` (1.5.6, headers only) and `build/duckdb-v2.0.0-src` (2.0.0).

## Review Focus

1. A table function result of more than one vector (2,048 rows). The output function is called with a full chunk and then again. Task 2 tests it with a 3,001-row `table_one` result.
2. A `read_stat` file of more than 2,048 rows. The staging chunk is filled, appended, reset and filled again. No test read such a file before. Task 2 adds a 5,000-row round trip.
3. The call after the last row, which returns 0 rows. Every query on a table function ends with one. Task 2's test runs each of them with verification on.
4. A vector that is not flat reaching `compat::ToUnified`: a constant argument, a dictionary vector after a filter, the constant state vector of `OVER ()`. No new test: `test/sql/aggregates_over_window.test`, `test/sql/distribution_functions.test` and `test/sql/nan_inputs.test` cover these, and Task 3 runs them on both lines and with verification on.
5. The 1.x lines. A shim that compiles on 2.0 can fail on C++11 or on a line without the 2.0 names. The check script compiles all three header trees, and the 1.4.3 suite runs after each task.

---

### Task 1: The deprecation check script

**Files:**
- Create: `scripts/check-duckdb-deprecations.sh`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `bash scripts/check-duckdb-deprecations.sh <duckdb-source-dir>`, with the output format and exit codes below. Tasks 2, 3 and 4 run it and read its `FAIL:` or `OK:` line.

The script answers one question: which calls in `src/` does this DuckDB header tree mark as deprecated? It needs no DuckDB build. It also serves as a compile check for a DuckDB line, because a source file that does not compile against the tree makes it fail.

**Behaviour:**

- One argument: a directory that holds `src/include/duckdb.hpp`. A DuckDB source tree qualifies, and so does a directory with only the headers. A relative argument is relative to the caller's directory. With no argument, or when `<dir>/src/include/duckdb.hpp` is missing, print a usage message to stderr and exit 2.
- Work from the repository root (the parent of the script's directory) and compile each `src/*.cpp` by its relative path, so a location prints as `src/<name>.cpp:<line>`.
- The compile command for one file: `$CXX -std=<std> -fsyntax-only -Wdeprecated-declarations -I <dir>/src/include -I src/include -I third_party/eigen -I third_party/readstat/src -I third_party/readstat/src/sas -I third_party/readstat/src/spss -I third_party/readstat/src/stata <file>`. On Windows (`uname -s` starts with `MINGW`, `MSYS` or `CYGWIN`) add `-I third_party/win_iconv`, as `CMakeLists.txt` does. `CXX` defaults to `g++`.
- Pass `<dir>` to the compiler in a form a native Windows compiler accepts. `cygpath -m` gives it when `cygpath` exists.
- `<std>` is `c++17` when `<dir>/src/include/duckdb/common/vector/list_vector.hpp` exists, and `c++11` otherwise. That header exists from DuckDB 2.0 on; it is the test `duckdb_compat.hpp` uses. The environment variable `CXX_STD` overrides the choice.
- Compile up to `$JOBS` files at a time. `JOBS` defaults to the output of `nproc`, or 4 when `nproc` is missing.
- A deprecated call is a compiler line of the form `<path>:<line>:<col>: warning: '<name>' is deprecated...` whose path starts with `src/`. Count each pair of `<name>` and `<path>:<line>` once: a template that is instantiated twice repeats the same warning. A warning located outside `src/` belongs to DuckDB's own headers and is not counted.
- Remove the temporary files on exit. Write `rm -rf -- "${tmp:?}"`, so an empty variable cannot widen the removal.

**Output and exit codes:**

When there is at least one deprecated call, exit 1 and print:

```text
FAIL: 126 deprecated DuckDB call sites in src/ (headers: <dir>/src/include, -std=c++17)

'void duckdb::DataChunk::SetCardinality(duckdb::idx_t)': 12
  src/corr_matrix_function.cpp:304
  src/ggsql.cpp:489
  ...
```

One block per deprecated name, the names sorted, the locations sorted under each name, with `LC_ALL=C`. The quoted name is the compiler's own text.

When a file fails to compile, print `FAIL: <n> source files do not compile (headers: ..., -std=...)`, then the `error:` lines of each such file, and exit 1. Compile errors are reported before deprecated calls, and both can appear in one run.

When nothing is found, exit 0 and print one line:

```text
OK: 39 source files, no deprecated DuckDB call (headers: <dir>/src/include, -std=c++17)
```

- [ ] **Step 1: Record the expected results before writing the script**

These numbers were measured by hand with the same compiler command. They are the test of the script.

| Command | Exit | First line of the output |
|---|---|---|
| `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src` | 1 | `FAIL: 126 deprecated DuckDB call sites in src/ (headers: ..., -std=c++17)` |
| `bash scripts/check-duckdb-deprecations.sh duckdb` | 0 | `OK: 39 source files, no deprecated DuckDB call (headers: ..., -std=c++11)` |
| `bash scripts/check-duckdb-deprecations.sh build/duckdb-v1.5.6-include` | 0 | `OK: 39 source files, no deprecated DuckDB call (headers: ..., -std=c++11)` |
| `CXX_STD=c++11 bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src` | 1 | `FAIL: 32 source files do not compile (headers: ..., -std=c++11)` |
| `bash scripts/check-duckdb-deprecations.sh` | 2 | a usage message on stderr |
| `bash scripts/check-duckdb-deprecations.sh scripts` | 2 | a usage message on stderr |

The fourth command tests the error path: the 2.0 headers need C++17. Seven of the 39 files include no DuckDB header and still compile (`emmeans`, `glm_core`, `glm_fit_core`, `lin_hyp`, `lm_core`, `linalg`, `ttest_compute`), so 32 fail.

On the 2.0.0 tree the three blocks are: `...DataChunk::SetCardinality...` 12, `...DataChunk::SetValue...` 41, `...Vector::ToUnifiedFormat...` 73. Six of the 73 are in headers: `src/include/duckdb_compat.hpp:171` and five in `src/include/list_read.hpp`.

- [ ] **Step 2: Run the first command to see it fail**

Run: `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src`
Expected: `No such file or directory`. The script does not exist yet.

- [ ] **Step 3: Write `scripts/check-duckdb-deprecations.sh`**

Start it with a header comment in the style of `scripts/check-wasm-string-hash.sh`: what the check is for, why MSVC builds do not show these warnings while GCC and Clang do, the usage line, and what it needs (a C++ compiler that knows `-fsyntax-only`, the Eigen and ReadStat submodules, a DuckDB header tree). Use `set -uo pipefail`, as `scripts/run-cpp-tests.sh` does.

- [ ] **Step 4: Run the six commands of Step 1**

Expected: each exit code and first line matches the table. The 2.0.0 run also shows the three blocks with 12, 41 and 73 locations. Each run takes under a minute with 8 jobs.

- [ ] **Step 5: Commit**

```bash
git add scripts/check-duckdb-deprecations.sh
git commit -m "scripts: list the calls a DuckDB header tree marks as deprecated (#79)"
```

Give the commit a body of two or three sentences: what the script does, and that it reports 126 call sites against the 2.0.0 headers and none against 1.4.3 and 1.5.6.

---

### Task 2: Table functions set the size of their output vectors on DuckDB 2.0

**Files:**
- Create: `test/sql/vector_verification.test`
- Modify: `src/include/duckdb_compat.hpp` (a new shim in the Tier 1 section, after `WriteValidity`; one new bullet in the Tier 1 list of the header comment)
- Modify: `src/corr_matrix_function.cpp`, `src/ggsql.cpp`, `src/lm_function.cpp`, `src/meta_function.cpp`, `src/read_stat_function.cpp`, `src/table_one_function.cpp`, `src/ttest_function.cpp`

**Interfaces:**
- Consumes: `scripts/check-duckdb-deprecations.sh` from Task 1.
- Produces: `inline void statsduck::compat::SetCardinality(duckdb::DataChunk &chunk, duckdb::idx_t count)` in `src/include/duckdb_compat.hpp`.

**Background.** On DuckDB 2.0 every vector carries its own size. `DataChunk::SetCardinality`, which 2.0 deprecates, sets the row count of the chunk and leaves the vectors at size 0. `Vector::SetValue` and writes through a raw data pointer do not change the size either. With `SET debug_verification_mode = 'VERIFY_VECTORS'` the engine compares the two and fails with `INTERNAL Error: DataChunk::Verify - size mismatch`. `DataChunk::SetChildCardinality` sets both. DuckDB's own table functions that write by index call it. On 1.x a vector has no size and `SetCardinality` is the whole operation.

- [ ] **Step 1: Write the failing test**

Create `test/sql/vector_verification.test` with exactly this content. The values were read from the 1.4.3 build. Columns are separated by one tab.

```text
# name: test/sql/vector_verification.test
# description: Table functions fill their output chunk so that DuckDB 2.0's
#              vector verification accepts it (#79)
# group: [stats_duck]

require stats_duck

# On DuckDB 2.0 every vector carries its own size, and this debug setting
# makes the engine compare it with the row count of the chunk. The 1.x lines
# do not have the setting. There the SET fails with the message below, which
# "statement maybe" accepts, and the rest of the file runs as plain calls.
statement maybe
SET debug_verification_mode = 'VERIFY_VECTORS';
----
unrecognized configuration parameter

statement ok
CREATE TABLE vv AS
SELECT i::DOUBLE AS x, (2 * i + i % 3)::DOUBLE AS y, CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS g
FROM range(20) t(i);

query TTTIII
SELECT column_name, column_type, kind, n_rows, n_missing, n_distinct FROM meta('vv') ORDER BY column_name
----
g	VARCHAR	categorical	20	0	2
x	DOUBLE	numeric	20	0	20
y	DOUBLE	numeric	20	0	14

query TTI
SELECT row_var, col_var, n FROM corr_matrix('vv', variables := ['x', 'y']) ORDER BY row_var, col_var
----
x	x	20
x	y	20
y	x	20
y	y	20

query T
SELECT term FROM lm('vv', y := 'y', x := ['x']) ORDER BY term
----
(Intercept)
x

query III
SELECT df_model, df_residual, n FROM lm_summary('vv', y := 'y', x := ['x'])
----
1	18	20

query TTTTT
SELECT variable, level, statistic, stratum, display FROM table_one('vv', variables := ['g']) ORDER BY level
----
g	Missing	n (%)	Overall	0 (0.0%)
g	a	n (%)	Overall	10 (50.0%)
g	b	n (%)	Overall	10 (50.0%)

query TT
SELECT column_name, type FROM read_stat_metadata('./test/data/sample.sas7bdat') ORDER BY column_name
----
dtime	DOUBLE
mychar	STRING
mydate	DOUBLE
mylabl	DOUBLE
mynum	DOUBLE
myord	DOUBLE
mytime	DOUBLE

# VISUALIZE returns its one row through a table function.
statement ok
VISUALIZE x AS x, y AS y FROM vv DRAW point

# A result of more than one vector (2,048 rows) takes two calls of the output
# function, the first with a full chunk.
statement ok
CREATE TABLE vv_levels AS SELECT 'L' || lpad(i::VARCHAR, 4, '0') AS c FROM range(3000) t(i);

query II
SELECT count(*), count(DISTINCT level) FROM table_one('vv_levels', variables := ['c'])
----
3001	3001

# read_stat stages the rows of a file in a chunk of its own. A file of more
# than one vector fills that chunk, appends it to the buffer and starts again.
statement ok
COPY (SELECT i AS i, 'r' || i AS s FROM range(5000) t(i)) TO '__TEST_DIR__/vv_big.xpt';

query IIIII
SELECT count(*), CAST(sum(i) AS BIGINT), CAST(min(i) AS BIGINT), CAST(max(i) AS BIGINT), count(DISTINCT s)
FROM read_stat('__TEST_DIR__/vv_big.xpt')
----
5000	12497500	0	4999	5000

# The setting is process-wide on 2.0, so it is switched off again for the
# test files that run after this one.
statement maybe
RESET debug_verification_mode;
----
unrecognized configuration parameter
```

Two of these statements are characterization tests, and that is intended. The `read_stat` round trip of 5,000 rows passes before the change on both lines: nothing verifies the staging chunk. It is there because no test read a file of more than one vector before, and this task changes the two calls on that path. The `VISUALIZE` statement checks only that the statement succeeds, which is what the verification decides.

- [ ] **Step 2: Run the test to verify it fails on 2.0**

The builds in `build/release` and `build/release-v2.0.0` are at the branch base. Do not rebuild before this step.

Run: `build/release-v2.0.0/test/Release/unittest.exe "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/vector_verification.test"`
Expected: FAIL at line 22, the `meta` query, with `INTERNAL Error: DataChunk::Verify - size mismatch: vector 0 (VARCHAR) has size 0 but chunk has size 3`.

Run: `build/release/test/Release/unittest.exe "test/sql/vector_verification.test"`
Expected: `All tests passed (77 assertions in 1 test case)`. DuckDB 1.4.3 has no such setting and no vector sizes, so the file passes there before the change. Record both outputs.

- [ ] **Step 3: Add `compat::SetCardinality` to `src/include/duckdb_compat.hpp`**

Signature: `inline void SetCardinality(duckdb::DataChunk &chunk, duckdb::idx_t count)`, in namespace `statsduck::compat`, in the Tier 1 section after `WriteValidity`. Under `#ifdef STATS_DUCK_DUCKDB_VECTOR_V2` it calls `chunk.SetChildCardinality(count)`. Otherwise it calls `chunk.SetCardinality(count)`.

Give it a `//!` comment that states the contract and the reason: the caller has written rows `[0, count)` into the chunk's vectors by index; on 1.x the chunk alone carries the count; on 2.0 each vector carries a size as well, the deprecated `DataChunk::SetCardinality` leaves it at 0, and `SetChildCardinality` sets both (#79). Add one bullet for it to the Tier 1 list in the comment at the top of the file.

- [ ] **Step 4: Replace the 12 `SetCardinality` calls**

The rule: `X.SetCardinality(N);` becomes `compat::SetCardinality(X, N);`, and `P->SetCardinality(N);` becomes `compat::SetCardinality(*P, N);`. Every one of these files already has `namespace compat = statsduck::compat;`.

| File | Calls |
|---|---|
| `src/corr_matrix_function.cpp` | 1 |
| `src/ggsql.cpp` | 2 |
| `src/lm_function.cpp` | 3 |
| `src/meta_function.cpp` | 1 |
| `src/read_stat_function.cpp` | 3 (two of them on the staging chunk: `lc.staging->...` and `staging....`) |
| `src/table_one_function.cpp` | 1 |
| `src/ttest_function.cpp` | 1 |

In `src/ttest_function.cpp` the call comes before the values are written. Keep that order. `SetChildCardinality` does not touch the data.

- [ ] **Step 5: Replace the 41 `DataChunk::SetValue` calls**

The rule: `X.SetValue(C, R, V);` becomes `X.data[C].SetValue(R, V);`. No shim: `DataChunk::SetValue` has that one line as its body on every DuckDB line.

| File | Calls |
|---|---|
| `src/corr_matrix_function.cpp` | 5 |
| `src/ggsql.cpp` | 2 |
| `src/lm_function.cpp` | 5 |
| `src/meta_function.cpp` | 9 |
| `src/read_stat_function.cpp` | 4 |
| `src/table_one_function.cpp` | 7 |
| `src/ttest_function.cpp` | 9 |

Afterwards `grep -n "SetCardinality(\|\.SetValue(" src/*.cpp` shows only `compat::SetCardinality(` calls and `.data[...].SetValue(` calls.

- [ ] **Step 6: Build both lines and run the new test**

Build 1.4.3 and 2.0.0 with the commands of the Global Constraints, one after the other. Each log must show 0 warnings.

Run the new test on both lines.
Expected on both: `All tests passed (77 assertions in 1 test case)`.

- [ ] **Step 7: Run the suites and the check script**

Run the full suite on 1.4.3, then on 2.0.0, then on 2.0.0 with `--on-init "SET debug_verification_mode='VERIFY_VECTORS';"`.
Expected for all three: `All tests passed (3126 assertions in 77 test cases)`. Before this task the third run failed 34 of 76 files.

Run: `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src`
Expected: `FAIL: 73 deprecated DuckDB call sites in src/`, with one block, for `Vector::ToUnifiedFormat`. No `DataChunk` block is left.

Run: `bash scripts/check-duckdb-deprecations.sh duckdb` and `bash scripts/check-duckdb-deprecations.sh build/duckdb-v1.5.6-include`
Expected: `OK: 39 source files, no deprecated DuckDB call` for both. This is the compile check for the 1.5.6 line, which has no local build.

- [ ] **Step 8: Commit**

```bash
git add test/sql/vector_verification.test src/include/duckdb_compat.hpp src/corr_matrix_function.cpp src/ggsql.cpp src/lm_function.cpp src/meta_function.cpp src/read_stat_function.cpp src/table_one_function.cpp src/ttest_function.cpp
git commit -m "table functions: set the size of the output vectors on DuckDB 2.0 (#79)"
```

Give the commit a body: what was wrong on 2.0 and how it showed, what the shim does on each line, that `DataChunk::SetValue` became `Vector::SetValue`, and what the new test covers.

---

### Task 3: `compat::ToUnified` and its 73 call sites

**Files:**
- Modify: `src/include/duckdb_compat.hpp` (a new shim in the Tier 1 section, placed before `AggregateStates`, which calls it; one new bullet in the Tier 1 list of the header comment)
- Modify: `src/include/list_read.hpp` (5 calls)
- Modify: 21 binding files, listed in Step 3 (67 calls)

**Interfaces:**
- Consumes: `scripts/check-duckdb-deprecations.sh` from Task 1.
- Produces: `inline void statsduck::compat::ToUnified(duckdb::Vector &vector, duckdb::idx_t count, duckdb::UnifiedVectorFormat &format)` in `src/include/duckdb_compat.hpp`.

**Background.** DuckDB 2.0 deprecates `Vector::ToUnifiedFormat(count, format)`, because a vector carries its own size there. On 2.0 the counted form is one line that calls `ToUnifiedFormat(format)`, so replacing it changes no behaviour. On 1.x the counted form is the only one, and it is not `const`. That is why the shim takes a non-const `Vector &`.

- [ ] **Step 1: Run the check to see it fail**

Run: `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src`
Expected: exit 1, `FAIL: 73 deprecated DuckDB call sites in src/`, one block, for `Vector::ToUnifiedFormat`. Record the output. It is the list of sites this task removes.

- [ ] **Step 2: Add `compat::ToUnified` to `src/include/duckdb_compat.hpp`**

Signature: `inline void ToUnified(duckdb::Vector &vector, duckdb::idx_t count, duckdb::UnifiedVectorFormat &format)`, in namespace `statsduck::compat`, in the Tier 1 section, before the `AggregateStates` class. Under `#ifdef STATS_DUCK_DUCKDB_VECTOR_V2` it calls `vector.ToUnifiedFormat(format)` and marks `count` as unused with `(void)count;`. Otherwise it calls `vector.ToUnifiedFormat(count, format)`.

Give it a `//!` comment: what it is, why 2.0 deprecates the counted form, that 2.0's counted form forwards to the other so no behaviour changes, and why the parameter is not `const` (#79). Add one bullet for it to the Tier 1 list in the comment at the top of the file.

- [ ] **Step 3: Replace the 73 calls**

The rule: `R.ToUnifiedFormat(N, F);` becomes `compat::ToUnified(R, N, F);`. `R` is the whole receiver expression, for example `inputs[0]`, `args.data[3]`, `chunk.data[c]` or `compat::Deref(opt_children[...])`. Every call is on one line.

- In the 21 binding files write `compat::ToUnified(...)`. Each of them already has `namespace compat = statsduck::compat;`.
- In `src/include/list_read.hpp` write `statsduck::compat::ToUnified(...)`, as that header spells the namespace in full.
- In `src/include/duckdb_compat.hpp`, inside `AggregateStates`, write `ToUnified(state_vector, count, format_);`.

| File | Calls | File | Calls |
|---|---|---|---|
| `src/adjust_p_function.cpp` | 3 | `src/lm_fit_function.cpp` | 4 |
| `src/anova_function.cpp` | 2 | `src/nonparametric_function.cpp` | 4 |
| `src/auto_bin_function.cpp` | 4 | `src/normality_function.cpp` | 3 |
| `src/bootstrap_function.cpp` | 1 | `src/poibin_function.cpp` | 3 |
| `src/chisq_function.cpp` | 3 | `src/random_sampling_function.cpp` | 6 |
| `src/correlation_function.cpp` | 2 | `src/rank_correlation_function.cpp` | 2 |
| `src/distribution_functions.cpp` | 4 | `src/sas_export_function.cpp` | 2 |
| `src/emmeans_function.cpp` | 3 | `src/sign_test_function.cpp` | 3 |
| `src/glm_fit_function.cpp` | 7 | `src/summary_stats_function.cpp` | 1 |
| `src/ks_test_function.cpp` | 3 | `src/ttest_agg_function.cpp` | 5 |
| `src/lin_hyp_function.cpp` | 2 | `src/include/list_read.hpp` | 5 |
| | | `src/include/duckdb_compat.hpp` | 1 |

Afterwards the only calls of `ToUnifiedFormat` left in `src/` are the two inside `compat::ToUnified`. Check with `grep -rn "ToUnifiedFormat(" src/`; a comment may still name the function.

Do not reformat any file. Keep each changed line within 120 columns by hand.

- [ ] **Step 4: Run the check on the three header trees**

Run: `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src`
Expected: exit 0, `OK: 39 source files, no deprecated DuckDB call (headers: ..., -std=c++17)`.

Run: `bash scripts/check-duckdb-deprecations.sh duckdb` and `bash scripts/check-duckdb-deprecations.sh build/duckdb-v1.5.6-include`
Expected: exit 0 and the `OK:` line with `-std=c++11` for both.

- [ ] **Step 5: Build both lines and run the suites**

Build 1.4.3 and 2.0.0, one after the other. Each log must show 0 warnings.

Run the full suite on 1.4.3, then on 2.0.0, then on 2.0.0 with `--on-init "SET debug_verification_mode='VERIFY_VECTORS';"`.
Expected for all three: `All tests passed (3126 assertions in 77 test cases)`, the numbers Task 2 ended with. This task changes no behaviour, so no number moves.

- [ ] **Step 6: Commit**

Stage `src/include/duckdb_compat.hpp`, `src/include/list_read.hpp` and the 21 binding files by name.

```bash
git commit -m "binding layer: ToUnifiedFormat goes through compat::ToUnified (#79)"
```

Give the commit a body: why 2.0 deprecates the counted form, that 2.0's counted form forwards to the other, the number of call sites, and the result of the check script on the three header trees.

---

### Task 4: The CI job, the changelog and the engineering note

**Files:**
- Modify: `.github/workflows/lint.yml` (one new job)
- Modify: `CHANGELOG.md` (one bullet under `### Changed` and one under `### Fixed`, both in `## [Unreleased]`)
- Create: `notes/engineering/2026-11-duckdb2-vector-sizes.md`
- Modify: `notes/engineering/README.md` (one new entry at the end of the inventory)
- Modify: `notes/engineering/2026-10-duckdb-v2-port.md` (the first bullet of "Left for later")

**Interfaces:**
- Consumes: `scripts/check-duckdb-deprecations.sh` (Task 1), `compat::SetCardinality` (Task 2), `compat::ToUnified` (Task 3), `test/sql/vector_verification.test` (Task 2).
- Produces: nothing that code depends on.

- [ ] **Step 1: Confirm the check passes before wiring it into CI**

Run: `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src`
Expected: exit 0 and the `OK:` line. If it fails, stop and report: Task 3 is not complete.

- [ ] **Step 2: Add the job to `.github/workflows/lint.yml`**

Add this job after `cpp-tests`, with the file's two-space indentation:

```yaml
  duckdb-deprecation-guard:
    name: DuckDB 2.0 deprecated-call guard
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # The sources include Eigen and ReadStat headers. The duckdb submodule
      # is not needed: it is pinned to v1.4.3, and this check wants the v2.0.0
      # headers.
      - name: Check out Eigen and ReadStat
        run: git submodule update --init --depth 1 third_party/eigen third_party/readstat
      # A sparse, blobless clone of the one directory the check reads.
      - name: Fetch the DuckDB v2.0.0 headers
        run: |
          git clone --depth 1 --branch v2.0.0 --filter=blob:none --sparse https://github.com/duckdb/duckdb.git "$RUNNER_TEMP/duckdb-v2"
          git -C "$RUNNER_TEMP/duckdb-v2" sparse-checkout set src/include
      # MSVC builds do not show DuckDB's deprecation warnings, so a deprecated
      # call written on Windows goes unnoticed without this.
      - name: No deprecated DuckDB call in src/
        run: bash scripts/check-duckdb-deprecations.sh "$RUNNER_TEMP/duckdb-v2"
```

The job cannot run locally. Check the YAML by eye against the two existing jobs: same indentation, same key order. The controller reads the job's result on the pull request.

- [ ] **Step 3: Add the two changelog bullets**

Under `## [Unreleased]`, `### Changed`, after the last bullet of that section, add:

```markdown
- **No deprecated DuckDB call is left in the binding layer** (#79). DuckDB 2.0
  deprecates `Vector::ToUnifiedFormat` with a row count, `DataChunk::SetValue`
  and `DataChunk::SetCardinality`. GCC and Clang warned at 126 call sites.
  MSVC builds showed none of them. The calls now go through
  `compat::ToUnified` and `compat::SetCardinality` in
  `src/include/duckdb_compat.hpp`, and through `Vector::SetValue`.
  `scripts/check-duckdb-deprecations.sh <duckdb-dir>` lists every deprecated
  call in `src/` against a DuckDB header tree. It compiles each file with
  `g++ -fsyntax-only`, so it needs no DuckDB build, and it also shows whether
  the sources compile on that DuckDB line. A CI job runs it against the v2.0.0
  headers on every pull request.
```

Under `### Fixed`, between the bullet that starts `- **A quoted boolean for \`add_intercept\` gets the right error** (#70).` and the bullet that starts `- **A SCALE channel named in two cases** (#98).`, add:

```markdown
- **Table functions pass DuckDB 2.0's vector verification** (#79). On DuckDB
  2.0 every vector carries its own size, next to the row count of its chunk.
  `corr_matrix`, `lm`, `lm_summary`, `meta`, `read_stat_metadata`, `table_one`
  and `VISUALIZE` set the row count of their output and left the vectors at
  size 0. Query results were right. With `SET debug_verification_mode =
  'VERIFY_VECTORS'` each of them failed with `INTERNAL Error:
  DataChunk::Verify - size mismatch`, and so did 34 of the 76 test files when
  the suite ran with that setting. They now set both through
  `DataChunk::SetChildCardinality`, as DuckDB's own table functions do. The
  1.x lines are unchanged, because a vector has no size of its own there. The
  new `test/sql/vector_verification.test` runs every table function with the
  verification on, and the whole suite passes with it on DuckDB 2.0.
```

Do not touch any other bullet. Another open pull request adds a bullet after the `(#69)` one, and these two places are away from it.

- [ ] **Step 4: Write the engineering note**

Create `notes/engineering/2026-11-duckdb2-vector-sizes.md`. Follow the shape of `notes/engineering/2026-11-aggregate-states-under-a-window.md`: a title that states the finding, a line `Date: 2026-11-23. Issue #79. Branch \`feat/79-duckdb2-deprecations\`.`, then the sections `## What happened`, `## Why`, `## The fix`, `## How to check`, `## What this does not do`. Title: `# On DuckDB 2.0 a chunk's row count is not the size of its vectors`. Aim for 60 to 100 lines. Use only the facts below; each is measured.

Facts for "What happened":
- The first GCC build of the 2.0 port printed three deprecation warnings. The issue treated them as noise to clean up.
- The count: 126 call sites. 73 `Vector::ToUnifiedFormat(count, format)`, 41 `DataChunk::SetValue`, 12 `DataChunk::SetCardinality`.
- Two of the three are renames with no effect. The third is a fault: with `SET debug_verification_mode = 'VERIFY_VECTORS'` on 2.0, `SELECT * FROM meta('t')` failed with `INTERNAL Error: DataChunk::Verify - size mismatch: vector 0 (VARCHAR) has size 0 but chunk has size 3`. So did `corr_matrix`, `lm`, `lm_summary`, `read_stat_metadata`, `table_one` and `VISUALIZE`. Running the suite with that setting failed 34 of 76 test files.
- Without the setting every result was right, on both lines, and the 2.0 suite passed. That is why the port did not see it.

Facts for "Why":
- On 2.0 a vector carries a size (`Vector::size()`, kept on its buffer). A chunk's `size()` is its own count when one is set, and otherwise the size of its first vector.
- 2.0's `DataChunk::SetCardinality(n)` is deprecated and calls `SetCardinalityUnsafe(n)`: it sets the chunk's count only. The header says so: "this only sets the chunk's cardinality, it does NOT resize the child vectors".
- `Vector::SetValue(index, value)` checks the index against the capacity and writes. It does not change the size. A write through `FlatVector::GetDataMutable` does not either. Only `Vector::Append` and the 2.0 writers do.
- So a table function that writes rows by index and then calls `SetCardinality(n)` returns a chunk with count n and vectors of size 0. The operators after a table function read the chunk's count, so results are right. `DataChunk::Verify` compares the two in `VERIFY_VECTORS` mode and throws.
- `read_stat` did not fail, because its output function scans a `ColumnDataCollection`. Its staging chunk had the same mismatch; `ColumnDataCollection::Append` does not verify.

Facts for "The fix":
- `compat::SetCardinality(chunk, count)`: `SetChildCardinality` on 2.0, `SetCardinality` on 1.x. `SetChildCardinality` sets the size of each flat or constant vector and the chunk's count. It does not touch the data.
- Not `CheckCardinality`, the replacement the deprecation message names first: it only compares the sizes, so it throws for a chunk that was filled by index.
- DuckDB's own code made the same choice: 2.0 calls `SetChildCardinality` in 114 places in `src/` and 15 in `extension/`, among them `read_text` (`src/function/table/direct_file_reader.cpp:194`), which writes by index. DuckDB 1.5.6 carries `SetChildCardinality`, `SetCardinalityUnsafe` and `CheckCardinality` as aliases of `SetCardinality`, with a comment that explains why. DuckDB 1.4.3 has none of them, so the shim keys on the 2.0 test and not on the name.
- `compat::ToUnified(vector, count, format)`: 2.0's counted `ToUnifiedFormat` is one line that calls the other form (`src/common/types/vector.cpp:454`), so this changes nothing. The parameter is a non-const `Vector &` because 1.x's `ToUnifiedFormat` is not `const`.
- `DataChunk::SetValue(col, row, value)` became `chunk.data[col].SetValue(row, value)`: the body of the deprecated function, on every line. No shim.

Facts for "How to check":
- `bash scripts/check-duckdb-deprecations.sh <duckdb-dir>` compiles each `src/*.cpp` with `g++ -fsyntax-only -Wdeprecated-declarations` against `<duckdb-dir>/src/include` and lists the deprecated calls. 39 files take about 20 seconds with 8 jobs. No DuckDB build is needed. The same run is a compile check for a DuckDB line with no local build: the 1.5.6 headers come out of the submodule with `git -C duckdb archive v1.5.6 src/include | tar -x -C build/duckdb-v1.5.6-include`.
- The whole suite with vector verification on, on a 2.0 build: `unittest.exe --on-init "SET debug_verification_mode='VERIFY_VECTORS';" "<absolute path>/test/sql/*"`. It passes since this change. It is the check to run after adding a table function.
- `test/sql/vector_verification.test` keeps a small version of that in the suite. It switches the setting on with `statement maybe`, which accepts success or an error that contains the given text. On 1.x the `SET` fails with `unrecognized configuration parameter` and the file runs as plain calls. The setting is a static on 2.0, so it holds for the whole process; the file resets it at its end.
- MSVC builds do not show these deprecation warnings. GCC and Clang do. On Windows the script is the only local way to see them.

Facts for "What this does not do":
- It does not move the binding layer to 2.0's iterators and writers (`Vector::Values<T>()`, `FlatVector::Writer<T>`), which DuckDB's own guidance prefers over `ToUnifiedFormat`. They do not exist on 1.x. That change waits until the 1.x lines are dropped.
- The CI job pins v2.0.0. A later DuckDB release that deprecates more is seen when the pin moves.

Then add an entry for the note at the end of the inventory in `notes/engineering/README.md`, in the style of the entries there: the file name as a link, then three to five lines that state the finding and the fix.

- [ ] **Step 5: Update the port note**

In `notes/engineering/2026-10-duckdb-v2-port.md`, section "Left for later", replace the first bullet (the one that starts `- v2.0 deprecates the counted` and ends `Tracked in #79.`) with:

```markdown
- Done in #79: v2.0 deprecates the counted `ToUnifiedFormat(count, data)`,
  `DataChunk::SetValue` and `DataChunk::SetCardinality`. The calls were
  replaced at 126 sites. The last of the three was more than a warning: see
  `2026-11-duckdb2-vector-sizes.md`.
```

- [ ] **Step 6: Final verification**

No source file changed in this task, so no rebuild is needed. Run, in this order:

Run: `build/release/test/Release/unittest.exe "test/sql/*"`
Expected: `All tests passed (3126 assertions in 77 test cases)`.

Run: `build/release-v2.0.0/test/Release/unittest.exe "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"`
Expected: `All tests passed (3126 assertions in 77 test cases)`.

Run: `bash scripts/check-duckdb-deprecations.sh build/duckdb-v2.0.0-src`, then with `duckdb`, then with `build/duckdb-v1.5.6-include`.
Expected: the `OK:` line three times.

Run: `bash scripts/check-wasm-string-hash.sh`
Expected: `OK: no std::string-keyed unordered container uses the default hash.`

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/lint.yml CHANGELOG.md notes/engineering/2026-11-duckdb2-vector-sizes.md notes/engineering/README.md notes/engineering/2026-10-duckdb-v2-port.md
git commit -m "CI guard, changelog and note for the DuckDB 2.0 deprecated calls (#79)"
```

Give the commit a body of two or three sentences.
