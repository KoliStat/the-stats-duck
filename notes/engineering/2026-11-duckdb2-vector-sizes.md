# On DuckDB 2.0 a chunk's row count is not the size of its vectors

Date: 2026-11-23. Issue #79. Branch `feat/79-duckdb2-deprecations`.

## What happened

The first GCC build of the 2.0 port printed three deprecation warnings. The
issue treated them as warnings to clean up.

The count was 126 call sites. There were 73 `Vector::ToUnifiedFormat(count,
format)`, 41 `DataChunk::SetValue` and 12 `DataChunk::SetCardinality`.

Two of the three are renames with no effect. The third is a fault. With
`SET debug_verification_mode = 'VERIFY_VECTORS'` on 2.0, `SELECT * FROM
meta('t')` failed with `INTERNAL Error: DataChunk::Verify - size mismatch:
vector 0 (VARCHAR) has size 0 but chunk has size 3`. So did `corr_matrix`,
`lm`, `lm_summary`, `read_stat_metadata`, `table_one` and `VISUALIZE`. Running
the suite with that setting failed 34 of 76 test files.

Without the setting the suite passed on 2.0, as on 1.x, and no test showed a
wrong result. That is why the port did not see it.

## Why

On 2.0 a vector carries a size. `Vector::size()` reads it from the vector's
buffer. A chunk's `size()` is its own count when one is set. Otherwise it is
the size of its first vector.

The 2.0 `DataChunk::SetCardinality(n)` is deprecated and calls
`SetCardinalityUnsafe(n)`. That sets the chunk's count only. The header says
so: "this only sets the chunk's cardinality, it does NOT resize the child
vectors".

`Vector::SetValue(index, value)` writes the value at the index. It does not
change the size. A write through
`FlatVector::GetDataMutable` does not either. Only `Vector::Append` and the 2.0
writers do.

So a table function that writes rows by index and then calls
`SetCardinality(n)` returns a chunk with count n and vectors of size 0. The
tests pass all the same. The expression and aggregate executors take the
chunk's count and do not ask the vectors. `DataChunk::Verify` compares the two
in `VERIFY_VECTORS` mode and throws.

`read_stat` did not fail. Its output function scans a `ColumnDataCollection`.
Its staging chunk had the same mismatch, but `ColumnDataCollection::Append`
does not verify.

## The fix

`compat::SetCardinality(chunk, count)` calls `SetChildCardinality` on 2.0 and
`SetCardinality` on 1.x. `SetChildCardinality` sets the size of each flat or
constant vector and the chunk's count. It does not touch the data.

The fix does not use `CheckCardinality`, the replacement that the deprecation
message names first. It only compares the sizes, so it throws for a chunk that
was filled by index.

DuckDB's own code made the same choice. 2.0 calls `SetChildCardinality` in 113
places in `src/` and 15 in `extension/`. One of them is `read_text`
(`src/function/table/direct_file_reader.cpp:194`), which writes by index.
DuckDB 1.5.6 carries `SetChildCardinality`, `SetCardinalityUnsafe` and
`CheckCardinality` as aliases of `SetCardinality`, with a comment that explains
why. DuckDB 1.4.3 has none of them. So the shim keys on the 2.0 test and not on
the name.

`compat::ToUnified(vector, count, format)` wraps the counted `ToUnifiedFormat`.
On 2.0 that call is one line that calls the other form
(`src/common/types/vector.cpp:454`), so the change has no effect. The
parameter is a non-const `Vector &` because 1.x's `ToUnifiedFormat` is not
`const`.

`DataChunk::SetValue(col, row, value)` became `chunk.data[col].SetValue(row,
value)`. That is the body of the deprecated function on 1.x and on 2.0. There is
no shim.

## How to check

`bash scripts/check-duckdb-deprecations.sh <duckdb-dir>` compiles each
`src/*.cpp` with `g++ -fsyntax-only -Wdeprecated-declarations` against
`<duckdb-dir>/src/include`. It lists the deprecated calls. 39 files take about
20 seconds with 8 jobs. No DuckDB build is needed.

The script parses the compiler's warning lines. In a UTF-8 locale GCC quotes a
name with the typographic quotes U+2018 and U+2019, not with the ASCII
apostrophe. The script sets `LC_ALL=C` for the compiler for that reason. It
also fails when it meets a deprecation warning that it cannot parse, so a
change in the compiler's wording cannot turn the check into a silent pass. The
CI job adds one deprecated call to its own checkout in a second step, and the
script must report it.

The same run is a compile check for a DuckDB line with no local build. The
1.5.6 headers come out of the submodule in two steps: `mkdir -p
build/duckdb-v1.5.6-include`, then `git -C duckdb archive v1.5.6 src/include |
tar -x -C build/duckdb-v1.5.6-include`. The submodule clone needs the `v1.5.6`
tag for that.

Since #106 the CI job runs once for each DuckDB version that the distribution
pipeline builds. `scripts/duckdb-lines.sh` reads the versions from
`MainDistributionPipeline.yml`. The second step also adds a file that does not
compile, and the script must report it. The deprecated call is added on 2.0
and later only, because 1.x does not deprecate it.

The whole suite with vector verification on, on a 2.0 build:
`unittest.exe --on-init "SET debug_verification_mode='VERIFY_VECTORS';"
"<absolute path>/test/sql/*"`. It passes since this change. Run it after
adding a table function.

`test/sql/vector_verification.test` keeps a small version of that in the
suite. It switches the setting on with `statement maybe`, which accepts success
or an error that contains the given text. On 1.x the `SET` fails with
`unrecognized configuration parameter` and the file runs as plain calls. The
setting is a static on 2.0, so it holds for the whole process. The file resets
it at its end.

MSVC builds do not show these deprecation warnings. DuckDB's `CMakeLists.txt`
removes the `/W` level from the MSVC flags (line 1091 at v2.0.0). MSVC then
compiles at its default level 1, and its deprecation warning, C4996, is a level
3 warning. GCC shows them. On Windows the script is the only local way to see
them.

## What this does not do

It does not move the binding layer to 2.0's iterators and writers
(`Vector::Values<T>()`, `FlatVector::Writer<T>`). DuckDB's own guidance prefers
them over `ToUnifiedFormat`. They do not exist on 1.x. That change waits until
the 1.x lines are dropped.

The CI job checks the versions that the distribution pipeline builds (#106). A
later DuckDB release that deprecates more is seen when the pipeline moves to
it.
