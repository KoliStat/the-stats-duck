# Porting the binding layer to DuckDB 2.0 without leaving 1.x

Written 2026-10-26, branch `feat/46-duckdb-v2`, off `v0.9`.

DuckDB v2.0.0 went stable on 2026-10-21. Issue #46 predicted that the port
would need the `VISUALIZE` parser moved to the new PEG grammar hooks and the
extension moved to the stable C ABI. Neither was true. This note records
what v2.0 changed, how one source tree now builds on the 1.4, 1.5 and 2.0
lines, the five places where my survey of the code was wrong and the
compiler was right, and the five places where the compiler was satisfied
and the test suite was not.

## What v2.0 changed, measured by building

The extension was built against DuckDB v2.0.0 in an isolated clone. The
first build failed with 1230 errors in 30 of the `*_function.cpp` files.
Every numeric kernel compiled clean: `glm_core`, `glm_fit_core`, `lm_core`,
`emmeans`, `lin_hyp`. That is what the DuckDB-free split in
`docs/kernel_api.md` was for.

Three things changed in shape, not just in name:

| Surface | 1.x | 2.0 |
|---|---|---|
| Aggregate bind callback | `bind(ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)` | `bind(BindAggregateFunctionInput &)` |
| Aggregate Init | one state: `(const AggregateFunction &, data_ptr_t)` | a batch: `(AggregateStateInput &, data_ptr_t *, idx_t)` |
| Parser extension parse callback | `(ParserExtensionInfo *, const string &query)` | `(ParserExtensionInfo *, const vector<SimpleToken> &)` |

Two things were removed outright. `Function::EraseArgument` is gone, and
with it the `original_arguments` bookkeeping it maintained; the serializer
writes an empty vector in that slot now. The `ExecuteWithNulls` executors
are gone, replaced by a protocol where the lambda returns
`optional<RESULT>` and an empty optional means NULL.

Everything else moved or gained a const/mutable split. `ListVector` and
`StructVector` live in `duckdb/common/vector/`. `FlatVector::GetData<T>` is
const and writes go through `GetDataMutable<T>`. `StructVector::GetEntries`
yields `vector<Vector>` instead of `vector<unique_ptr<Vector>>`. The
catalog, column, bind-name and named-parameter APIs are keyed by
`Identifier`, which converts implicitly from a `const char *` but only
explicitly from a `string`. `CopyInfo::table` became a private
`QualifiedName`. `dtime_t::micros` became `.value`.
`ConstantExpression(const Value &)` is deleted in favour of a `Literal`.
`Expression::return_type` and `BoundFunctionExpression::children` went
behind accessors.

Some things did not change. `DBConfig::GetCallbackManager()` and
`ExtensionCallbackManager::Register(ParserExtension)` keep their 1.5
signatures, so the parser registration in `src/ggsql.cpp` needs no edit.
`aggregate_update_t`, `aggregate_combine_t` and `aggregate_destructor_t`
are identical on their full multi-line text. `table_function_bind_t` keeps
its first three parameters; the fourth, `names`, became
`vector<Identifier>`. `FlatVector::SetNull` is untouched, which is why its
56 call sites are not in this port.

## How one tree builds on both lines

`src/include/duckdb_compat.hpp` detects v2 with
`__has_include("duckdb/common/vector/list_vector.hpp")` and defines
`STATS_DUCK_DUCKDB_V2`. This is the approach `src/ggsql.cpp` already used
for the 1.5 parser change. Behind the guard: accessors for the vector layer
(`WriteData`, `ListChild`, `WriteValidity`, `Deref`, `StructChildren`), two
declaration macros for the signature changes (`STATS_DUCK_AGGREGATE_BIND`,
`STATS_DUCK_AGGREGATE_INIT`), a type alias for Finalize's input, and
one-function shims for everything that moved: `EraseArgument`,
`BindArguments`, `Ident` and `Str`, `AddNamedParameter`, the three
`*WithNulls` executors, `TimeMicros`, `ConstantString`, `SetFallible`,
`IsTuple`, and the function-info and bound-expression accessors.

The bind macro is the one worth explaining. A signature change cannot be
bridged by an accessor. On 2.0 the macro declares a thin adapter under the
real name that unpacks the input object and calls `NAMEImpl` with the three
1.x-style parameters, and the body that follows the macro becomes
`NAMEImpl`. On 1.x the macro is just the 1.x signature. Bodies refer to
`context`, `function` and `arguments` by those names on both lines and are
byte-identical across them. The Init macro does the same with a loop.

The executor adapter keeps the 56 call sites' lambda shape
`(args..., ValidityMask &, idx_t)`. On 2.0 each call runs against a one-bit
`ValidityMask` on the stack, which `ValidityMask(validity_t *, idx_t)` wraps
without allocating, and the bit decides the `optional`.

The parse adapter is described further down, because its first version
was wrong in a way only the suite could show.

## The transformer, and where the survey was wrong

The port was applied by scripts in `.port_tools/` (git-ignored): a
vector-layer pass, an eleven-rule pass that asserts the count it expects
per rule and is idempotent against its own output, and two files of one-off
exact-text patches. The counts were set from a survey of the source, and
the compiler disagreed with the survey five times:

- 44 aggregate binds, not 40: four put `static unique_ptr<FunctionData>`
  on the line above the name.
- 22 Finalizes, not 15: seven leave the `AggregateInputData &` parameter
  unnamed.
- One function takes struct children as a parameter, so no `GetEntries`
  binding was in scope for the deref rule.
- One header, `list_read.hpp`, uses the vector helpers; the pass covered
  `src/*.cpp` only.
- My typedef diff compared only the first line of each multi-line typedef.
  That one happened to change nothing, but it was not evidence.

Each is now a rule or a comment in the scripts. The lesson is the one
`docs/fitter_conventions.md` records for test counts: a count derived from
a grep is a claim about the grep, not about the code.

## Running the v2 suite at all: static linking is opt-in on 2.0

On 1.x, `duckdb_extension_load(...)` linked the extension into the
`unittest` binary, so `require stats_duck` at the top of every test file
was satisfied by the binary itself. On 2.0 the configure reports the same
extension as "built but not linked". Linking is decided by a global property
that an extension config has to append to with
`duckdb_extension_statically_link`, and ours never did. The runner's
`require` first asks `ExtensionIsLoaded`, then tries the loadable route,
and on failure emits a `skip-requirement`. Measured, not inferred: the
suite run against the unlinked build exited 0 and printed
`All tests were skipped (total skipped 67)`. A gate that reads the exit
code, or the word "passed", accepts that as green. The assertion count is
the only honest signal.

`extension_config.cmake` now calls
`duckdb_extension_statically_link(stats_duck)` behind `if(COMMAND ...)`,
which 1.x does not define. The v2 configure now reports
`Linked into DuckDB: extensions [stats_duck, core_functions, parquet, json, icu]`,
and the 1.x configure, re-run with the same file, still reports
`[stats_duck, core_functions, parquet]`.

## What the suite found that the compiler could not

With the extension linked, the first real run gave 31 passing files of 67.
The 36 failures had five causes. None of them is a compile error, and each
is a behaviour of 2.0 that the headers do not express.

**A keyword-only parameter without a default is required.** Ten failures,
all table functions called without one of their named parameters:
`read_stat('file')`, `table_one('t', variables := [...])`, `lm`,
`corr_matrix`. The binder printed the signature as
`read_stat(col0 VARCHAR, /, *, format VARCHAR, "encoding" VARCHAR)` and
matched nothing, even with an explicit `::VARCHAR` cast on the argument. On
1.x a named parameter was always optional. On 2.0
`FunctionSignature::AddKeywordOnly(name, type, default_value)` takes an
optional default, and only a parameter with one may be omitted. The compat
shim now passes a NULL of the parameter's type as the default. The binder
then places that NULL into the bind's `named_parameters` map for an
omitted parameter, so a bind reads one as a NULL value rather than a
missing key. The binds already tested `IsNull()`, except `read_stat`'s two
loops over the map, which now skip a NULL value.

**The parse callback must say how many tokens it consumed.** Twenty-one
failures: every `VISUALIZE` test reported DuckDB's own
`syntax error at or near "VISUALIZE"`. The 2.0 result struct carries
`consumed_tokens`. A positive count claims that many leading tokens, zero
means the extension is not taking this input, and a negative count
surfaces the extension's error. The first adapter left the field at its
default of zero, so the parser read every successful parse as a refusal
and threw its original error. Two more properties of the protocol matter.
The token view runs to the end of the whole query, not the statement, and
always ends with an `END_OF_INPUT` sentinel; the statement ends at the
first `TERMINATOR` token, and a statement that owns its terminator claims
`end + 1` tokens, which is what DuckDB's own demo extension does. And the
view carries each token's text and type but not its byte offset, so the
original statement text cannot be sliced back out of the query. The
adapter rebuilds the text from the tokens with conventional spacing. One
space goes between tokens, with four exceptions: none before `, ) ] } ; ::
.` and none after `( [ { . :: :`; none between a word and the `(` that
opens its argument list, or the `:` that starts its type override, unless
the word is a clause keyword such as `AS`, `IN` or `FROM`; and none after
a sign that cannot be a binary operator, because nothing that ends an
operand comes before it. The last two rules were set by the second run,
which had turned `DOMAIN -1 1` into `DOMAIN - 1 1` and `AS :nominal` into
`AS:nominal`. The layer SQL that `VISUALIZE` reports on 2.0 shows this
spacing, not the user's. Comments never reach the view, and a quoted
literal is one token, so neither is affected. One more difference is not
the adapter's to fix: 2.0's tokenizer rejects an unterminated quote before
any parser extension runs, with its own message, where 1.x handed the text
to ours. The two tokenizer tests now expect the words both messages share.

**A scalar function that throws while executing must be marked fallible.**
One visible failure, `adjust_p([0.01], 'bogus')`, which on 2.0 came back as
an INTERNAL error saying the function "is not marked as fallible". 2.0
wraps the execution of every scalar function and rethrows an error from an
unmarked one as internal, because dictionary expression caching, filter
pushdown and `TRY` rely on the flag. Three of our scalar functions throw on
bad input at execution time: `adjust_p`, `lin_hyp` and `emmeans`. They now
call `SetFallible()` on 2.0. 1.x has the same flag as a public field,
`errors`, with the same default of "cannot error"; what reads it differs by
version, and declaring the truth on both lines is the safe choice. DuckDB
1.4.3, 1.4.5 and 1.5.6 all keep the field public; 1.5.6 was checked by
reading its header, since no 1.5 tree is built locally. The distribution,
Poisson-binomial and random-sampling functions do not throw and stay
unmarked.

**`row(...)` builds a TUPLE.** One failure in `glm_fit`'s tests:
`row('probit')` as the options argument was rejected as "not a STRUCT"
where the test expected "needs named fields". 2.0 gives an unnamed struct
its own type id, `LogicalTypeId::TUPLE`, with the same physical layout as
`STRUCT`; `typeof(row('probit'))` prints `TUPLE(VARCHAR)`. The options
check now accepts a TUPLE and lets the existing unnamed-fields check reject
it, which on 2.0 is exactly what `StructType::IsUnnamed` tests. The error
text is unchanged across lines.

**The arrow lambda is a binder error.** Three visible failures, in the
bootstrap, emmeans and lin_hyp tests, plus four sites in the glm_fit tests
behind an earlier failure: `list_transform(xs, c -> c.estimate)` fails on
2.0 with "Deprecated lambda arrow (->) detected" under the default
`lambda_syntax` setting. The keyword form, `lambda c: c.estimate`, is
accepted by 1.4 as well, so the 34 sites in the tests and the 5 in the
README were rewritten to it. The four test files pass on the 1.4.3 binary
with the same assertion counts as before (26, 164, 263 and 58).

The arrow-lambda rewrite is a change to the tests and the README. The
other four are changes to the extension, all behind the compat guard, so
the 1.x lines see only the fallible mark and a function that returns
false.

## What GCC found that MSVC did not

Every local build here is MSVC. The pipeline does not trigger on a pull
request into `v0.9` (it runs on pushes to `main` and pull requests into
`main`), so the first GCC build of the port was a manual dispatch on the
branch. It failed on the 1.5.6 and 2.0 lines at the same two lines of
`src/ttest_function.cpp` before the run was cancelled. The port had
templated three named-parameter helpers on the map type, so that one body
serves the 1.x `named_parameter_map_t` and the 2.0 `named_argument_map_t`.
Inside such a template `it->second.GetValue<double>()` is a call through a
dependent name, and the standard requires the `template` keyword there:
`it->second.template GetValue<double>()`. MSVC accepts the form without
it. GCC rejects it. The same run printed three deprecation warnings that
MSVC never shows; they are listed in #79. A syntax-only pass with the
local GCC 10.3 over every binding file, against the 2.0.0 headers and
the 1.4.3 headers, now reports zero errors, and the same pass on the
unfixed file reports the six error lines CI printed, so the pass is a
real check and not a silent one.

The second run's LTS job failed on `windows_amd64` before compiling
anything of ours. The vcpkg commit that extension-ci-tools `v1.4-andium`
pins builds zlib with a MSYS2 runtime package that the MSYS2 mirrors no
longer serve, and the download fails with HTTP 404. The 2.0 ci-tools pin
a vcpkg release from June 2026 that built every Windows target in the
same run, so the LTS job now passes that commit explicitly. The
community registry's own LTS workflow uses the 1.4 default and meets the
same download; our pin guards our code, not their build.

## Running the v2 suite on Windows

DuckDB v2.0.0's own `tools/cpp/tests/test_cpp_api_*.cpp` do not compile
under MSVC (201 errors: a private `cxx::Instance` constructor reached from
the tests). They are built into `unittest` by `test/api/CMakeLists.txt:10`.
In the scratch clone that line was commented out; nothing of ours depends
on it. This is DuckDB's problem, not the port's, and it only affects
building their test binary on Windows.

Two build-tooling traps cost a cycle each and are worth a line. Piping
`cmake --build` through `tail` reports `tail`'s exit status, which
`notes/engineering/2026-10-verifying-against-a-stale-binary.md` already
records; every stage here writes `$?` to a file instead. And an MSBuild
switch written `/m:1` after `cmake --build ... --` arrives at MSBuild as
`m:1`, because Git Bash converts what looks like a path; MSBuild then fails
with "Only one project can be specified". The dash form `-m:1` is left
alone.

## Verification

Same tree, three DuckDB lines, each stage's exit code recorded to a file
rather than read through a pipe:

| Line | Extension | `unittest` | SQL suite | Kernel suite |
|---|---|---|---|---|
| v1.4.3 (shipping) | 0 errors | 0 errors | 2639 assertions / 67 cases | 3269 checks / 11 suites |
| v1.4.5 (LTS) | 0 errors | 0 errors | 2639 assertions / 67 cases | — |
| v2.0.0 | 0 errors | 0 errors | 2639 assertions / 67 cases | — |

The 1.x counts are identical to before the port, and all three suite
runs come after the fixes the suite itself found. The kernel suite was
last run on the tree before those fixes; no kernel file changed since.

## Left for later

- v2.0 deprecates the counted `ToUnifiedFormat(count, data)`, which this
  tree calls at 66 sites, and `DataChunk::SetValue` and
  `DataChunk::SetCardinality`. MSVC is silent; GCC warns. A
  `compat::ToUnified` shim would remove the first with no behaviour
  change; the other two need a look per site. Tracked in #79.
- `ListVector::GetEntry` is deprecated on 2.0 but still present; the
  compat header does not call it on that line.
- The layer SQL on 2.0 carries reconstructed spacing. Exact text would
  need the token offsets, which the 2.0 parse callback does not pass; an
  upstream request to add `offset` to `SimpleToken` would remove the
  reconstruction.
- The PEG grammar hooks (`Register(shared_ptr<GrammarExtension>)`) exist
  and would let `VISUALIZE` become real grammar productions. An upgrade,
  not a requirement, and still #46's.
- The t-test table functions in `src/ttest_function.cpp`
  (`RegisterTTest1Samp` and its siblings) compile on both lines but are
  never registered; only the aggregate forms are. Dead code, found while
  looking for a counterexample to the named-parameter rule.
