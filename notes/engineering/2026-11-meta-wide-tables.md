# meta() on wide tables: one DISTINCT aggregate per column (#105)

Date: 2026-11-24. Branch `fix/105-meta-wide-tables`.

## What happened

On DuckDB 2.0.0, `SELECT * FROM meta('t')` ended the process when `t` had about
350 columns or more. The shell printed nothing and exited with the Windows
status `0xC00000FD`, a stack overflow. The test runner reported `SIGSEGV - Stack
overflow`. DuckDB 1.4.3 ran the same call on 2,000 columns.

## Why

`meta()` computed its counts with one query: `count(*)`, and for each column
`count(col)` and `count(DISTINCT col)`.

DuckDB 2.0.0 has a new optimizer step, `distinct_aggregate_rewrite`
(`src/optimizer/multi_stage_aggregate_rewriter.cpp`). It puts the input into a
CTE and gives each DISTINCT aggregate its own branch: a group-by on the column
with a count on top. It joins the branches with nested cross products.
`EXPLAIN` for 4 DISTINCT aggregates shows 3 nested `CROSS_PRODUCT` operators. So
the plan is as deep as the number of DISTINCT aggregates.

A recursive walk over that plan runs out of stack at about 350 levels. The main
thread of a Windows program has 1 MB of stack. `EXPLAIN` alone ends the process,
so the failure is in planning. The same query as plain SQL fails without the
extension. Which function overflows was not looked for.

The step has two more costs, and both are reported upstream: planning time
(duckdb/duckdb#26865, closed) and memory (duckdb/duckdb#27754, open). On a
12-row table, 300 DISTINCT aggregates took 2.5 s on 2.0.0 and 0.1 s on 1.4.3.

DuckDB 1.4.3 and 1.5.6 do not have the step. DuckDB 2.0.1 changed it in one
commit, which is about `ORDER BY` inside a DISTINCT aggregate and leaves the
shape of the plan alone. 2.0.1 was not run.

## What does not work

`SET disabled_optimizers = 'distinct_aggregate_rewrite'` avoids the failure:
1,000 DISTINCT aggregates then run in 0.2 s. A user can type it. `meta()`
cannot use it. The setting belongs to the whole database (it has `SetGlobal`
only). Setting it from `meta()`'s internal connection would change the user's
session and every query that runs at that moment.

## The fix

The counts are asked for in batches of 32 columns (`COUNT_BATCH_COLUMNS` in
`src/meta_function.cpp`). A plan is then at most 32 levels deep on every DuckDB
line, and no code tests for the DuckDB version.

Each batch is a query of its own, with its own `count(*)`. A column takes
`n_rows` from the query that counted it. So `n_missing = n_rows - count(col)`
holds on every row, even when another connection changes the table between two
batches. With one `n_rows` for all batches, a delete between two batches could
give a negative `n_missing`.

The numeric query stays one query. It has seven aggregates per numeric column
and no DISTINCT. At 2,000 columns it takes 1.8 s on 2.0.0 and 0.7 s on 1.4.3.

## Why 32

The counts query as plain SQL, on a table of 12 rows. One shell process runs all
batches.

| Columns | Batch size | DuckDB 1.4.3 | DuckDB 2.0.0 |
|---|---|---|---|
| 500 | 25 | 0.1 s | 0.7 s |
| 500 | 50 | 0.1 s | 0.8 s |
| 500 | 100 | 0.1 s | 1.2 s |
| 2,000 | 25 | 0.3 s | 2.4 s |
| 2,000 | 50 | 0.4 s | 3.0 s |
| 2,000 | 100 | 0.2 s | 4.8 s |

The same on a table of 200,000 rows and 100 columns, two runs each.

| Batch size | Queries | DuckDB 1.4.3 | DuckDB 2.0.0 |
|---|---|---|---|
| 100 | 1 | 0.11 s | 0.75 to 0.78 s |
| 50 | 2 | 0.12 s | 0.66 to 0.72 s |
| 32 | 4 | 0.10 to 0.13 s | 0.66 to 0.68 s |
| 25 | 4 | 0.11 to 0.12 s | 0.64 s |
| 10 | 10 | 0.11 to 0.12 s | 0.98 to 1.04 s |

On 2.0.0 the planning time grows faster than the number of DISTINCT aggregates,
so a small batch plans faster. Below about 25 columns the number of queries
starts to cost more than the planning saves. On 1.4.3 the batch size makes no
difference. A table of 32 columns or fewer still gets one query, the same one
as before.

`meta()` itself, on a table of 12 rows, VS 2022 release builds, the `duckdb.exe`
shell with the extension linked in. "Before" is one counts query.

| Columns | 2.0.0 before | 2.0.0 after | 1.4.3 before | 1.4.3 after |
|---|---|---|---|---|
| 300 | 3.5 s | 1.3 s | 0.6 s | 0.4 s |
| 500 | the process ends | 1.7 s | 0.8 s | 0.7 s |
| 1,000 | the process ends | 2.9 s | 1.6 s | 1.1 s |
| 2,000 | not run | 5.9 s | 4.5 s | 1.8 s |

## How it is tested

`test/sql/meta.test` profiles tables of 32, 33, 65 and 501 columns. Each value
column is built so that its counts and its range follow from its number. One
query per table compares every row of `meta()` with those values.

- The 501-column table is the failing test. On 2.0.0 the runner ended there with
  `SIGSEGV - Stack overflow` before the fix. On 1.4.3 it passed before the fix,
  because 1.4.3 never had the fault.
- The 32, 33 and 65-column tables check the edges of a batch. They pass before
  the fix, since one query for all columns is correct too. To see them fail, the
  batch was made one column short on purpose. The 32-column check then reported
  `31 30` instead of `31 31`.

## What this does not do

- It does not report the stack overflow to DuckDB. The reproduction needs no
  extension: a table of 12 rows and 351 columns, and `SELECT count(DISTINCT
  COLUMNS(*)) FROM wide`.
- Two batches can see different states of the table when another connection
  writes in between. Their `n_rows` can then differ. The numeric query and the
  mode queries were already separate queries before this change.
