# An aggregate's state vector is not always flat

Date: 2026-11-09. Issue #69. Branch `fix/69-window-states`.

## What happened

`SELECT lm_fit(y, [x1]) OVER () FROM t` crashed the process with a
segmentation fault. So did `glm_fit` and `ttest_1samp`, which is how #69 was
filed, and so did every other aggregate of the extension once tried:
`anova_oneway`, `pearson_test`, `mann_whitney_u`, `jarque_bera`, all of them.
`OVER (PARTITION BY g)` crashed the same way. `GROUP BY` worked. A framed
window, `OVER (ORDER BY y ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)`, worked.

The issue suspected the STRUCT return type or something a window-capable
aggregate must provide. Neither was it. `ttest_1samp` has a three-number
Welford state with no heap pointer, no destructor and a trivial combine, and
it crashed like the others. That ruled out the combine and destroy paths and
pointed at the one function every aggregate writes by hand: update.

## Why

DuckDB has three ways to feed rows to an aggregate.

- A hash aggregate (`GROUP BY`) calls `update` with a FLAT vector of state
  pointers, one per input row. Our update functions read it with
  `FlatVector::GetData<State *>(state_vector)` and indexed it by row.
- An ungrouped aggregate calls `simple_update`, which takes one state and
  all the rows. None of our aggregates provides it. The planner checks for
  that (`plan_aggregate.cpp`) and routes such an aggregate to the hash
  aggregate even when there are no groups, so the ungrouped path never ran
  our code.
- A window. The segment tree and the naive aggregator fill a flat vector of
  pointers for each frame, so they worked. The constant-frame aggregator,
  which handles `OVER ()` and `OVER (PARTITION BY ...)` without an order or
  a frame, aggregates each partition into a single state. When the aggregate
  has `simple_update` it calls that. When it does not, it hands `update` a
  CONSTANT vector built as `Vector statep(Value::POINTER(0))`, writes the
  partition's state pointer into its one slot, and passes the row count
  (`window_constant_aggregator.cpp`, both on 1.4.3 and 2.0.0).

A constant vector has one entry. Reading it as flat and indexing it by row
walks past that entry from the second row on, and the pointer read there is
whatever follows in memory. DuckDB's own aggregates never see this, because
`AggregateExecutor::UnaryScatter` and its siblings test the vector type first
and route a constant vector to a single state.

## The fix

`statsduck::compat::AggregateStates<State>` in `duckdb_compat.hpp` wraps the
state vector in its unified format and returns the state of input row `i`
through `operator[]`. For a flat vector the selection is the identity; for a
constant vector every row maps to entry 0. The 22 update functions replace
their one `FlatVector::GetData` line with it and keep the rest of their loops,
since `states[i]` still yields a `State *`. Combine and Finalize are
untouched: DuckDB hands those flat vectors on every path, the window
aggregators included (`WindowAggregateStates` allocates a flat pointer
vector).

Two other fixes were possible. A `simple_update` per aggregate would have
made the window path and the ungrouped path call it, at the cost of a second
copy of every update loop. A bind-time rejection, the floor the issue asked
for, would have turned a crash into an error and left the feature out. The
unified-format read is one line per function and makes the window work.

## How the tests pin it

`test/sql/aggregates_over_window.test` compares `agg(...) OVER ()` with the
plain `agg(...)` for 22 aggregates, struct by struct, with `IS NOT DISTINCT
FROM` so NULL fields compare equal, and `OVER (PARTITION BY ...)` with
`GROUP BY` for three of them, joined on the partition key. `bootstrap` only
has to run: its resamples depend on the order the rows arrive in. One block
pins the framed-window path that always worked.

The RED run is worth recording. The sqllogictest runner does not survive a
segmentation fault in the extension, so the failing run has no "Wrong result"
block, just a Catch fatal-signal banner and exit code 139. A crash test is
red when the runner dies, and the exit code is the evidence.

## Rule

An aggregate's `update` must not assume the state vector is flat. Read it
through `compat::AggregateStates`, or through `UnifiedVectorFormat` directly,
and index the states through the selection vector. When a new aggregate is
written by hand rather than through DuckDB's executor templates, run it
under `OVER ()` once before calling it done; `aggregates_over_window.test` is
where that check belongs.
