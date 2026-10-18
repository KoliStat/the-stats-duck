# Two ways to verify a change against a binary that is not the change

*October 2026, two incidents on the #20/#21 branches.*

Both ran a real command against a real binary and got a confident, wrong
answer, because the binary on disk was not the one the diff produced.

## 1. Piping a build through `tail`/`head` hides a compiler failure

`cmake --build ... | tail -n 40` reports the exit status of `tail`, not of
the build. A failed compile still exits the pipeline with status 0,
because the last command in the pipe ran fine. Whatever binary the last
*successful* build left on disk is what every later command then runs
against, often stale by one or more commits, with no error of its own to
say so.

This is how an earlier task on this branch concluded that DuckDB's
`VALUES` scan evaluates one row per chunk: a specific, wrong claim about
DuckDB's execution model, written into documentation, that took a
reviewer reading the DuckDB source and running an `EXPLAIN` to disprove.

Detect it by never piping a build command into anything. Capture its
output to a file if you need to search it, and check the exit status
right after the build command itself, never after a command downstream of
it:

```bash
cmake --build build/release --config Release --target X > build.log 2>&1
echo "exit: $?"          # the build's own status, checked first
grep -i error build.log  # inspect the log separately, after the check above
```

## 2. `duckdb.exe` statically links `stats_duck`; `LOAD` can be a silent no-op

`build/release/Release/duckdb.exe` comes from the `shell` CMake target,
which links `stats_duck` into the shell binary itself. A `LOAD
'.../stats_duck.duckdb_extension'` against that binary is then redundant
for anything the static copy already provides. Building `unittest`
(a separate static link, into its own test binary) or
`stats_duck_loadable_extension` (the `.duckdb_extension` file on its own)
does **not** rebuild `duckdb.exe`.

If `duckdb.exe` was last built before a function existed, `LOAD` of a
freshly built extension file still succeeds, with no error: DuckDB sees
`stats_duck` already loaded, by name, from the static copy, and the
dynamic module's own registration code runs for nothing the catalog
keeps. The new function then either fails to resolve
(`Catalog Error: ... does not exist`) or, more dangerously, silently
exposes whatever older behavior the static copy happens to carry for a
function of the same name, with nothing to say a newer one was meant to
be there.

Detect it by checking the catalog before issuing any `LOAD` at all:

```sql
SELECT count(*) FROM duckdb_functions() WHERE function_name = '<name>';
```

If that is already nonzero with no `LOAD` run yet, `duckdb.exe` is stale
with respect to that function, whatever is loaded afterward.

Fix it by rebuilding the `shell` target specifically:

```bash
cmake --build build/release --config Release --target shell
```

## The common thread

Both incidents have the same shape: a command that looks like it tests
the new code — a build, a `LOAD` — actually tests an old binary, and
nothing says so on its own. Check the thing that was actually changed
(the build's own exit status; the catalog `LOAD` is supposed to update),
not a downstream proxy for it, before drawing a conclusion from the
output.
