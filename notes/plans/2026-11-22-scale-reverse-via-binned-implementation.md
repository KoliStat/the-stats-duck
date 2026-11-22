# SCALE SETTING reverse, VIA and BINNED (#98) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept upstream ggsql's `SCALE ... SETTING reverse => true`, `SCALE ... VIA <transform>` and `SCALE BINNED <channel>` in VISUALIZE, and emit the matching Vega-Lite scale properties.

**Architecture:** The hand parser in `src/ggsql_grammar.cpp` serves DuckDB 1.4.3 directly. It serves DuckDB 2.0.0 through the token adapter in `src/ggsql.cpp`, which rebuilds the statement text from DuckDB's tokens. Each new option becomes one or more `ScaleSpec` entries, which `ApplyScales` injects into the channel's JSON object, as `TO`, `ZERO`, `DOMAIN` and `LABEL` do today. `VIA` and `BINNED` also make their channel quantitative, through a derived list of type overrides. Two combinations that Vega-Lite cannot draw as upstream means them are rejected when the spec is compiled.

**Tech Stack:** DuckDB extension in C++; sqllogictest files under `test/sql/` run by `unittest.exe`; Vega-Lite v5 JSON output.

**Spec:** GitHub issue #98 (`gh issue view 98`): the design Massimo approved in chat on 2026-11-22, plus the rules added while planning. Upstream reference: ggsql v0.5.2, `doc/syntax/clause/scale.qmd`, `doc/syntax/scale/type/continuous.qmd` and `doc/syntax/scale/type/binned.qmd`. Vega-Lite reference: v5.23.0, `scaleType()` in `src/compile/scale/type.ts` and `channelSupportScaleType()` in `src/scale.ts`.

## Global Constraints

- Work on branch `feat/98-scale-reverse-via-binned`, which is checked out. Do not push, open a pull request or switch branches.
- After every task, both DuckDB lines pass the whole SQL suite. Run every command from the repository root in Git Bash.
  - 1.4.3 build: `cmake --build build/release --config Release --target unittest -- -m:1`
  - 1.4.3 suite: `build/release/test/Release/unittest.exe "test/sql/*"`
  - 2.0.0 build: `cmake --build build/release-v2.0.0 --config Release --target unittest --parallel 8`
  - 2.0.0 suite: `build/release-v2.0.0/test/Release/unittest.exe "C:/Users/massi/source/repos/kolistat/the-stats-duck/test/sql/*"`. Keep this Windows form of the path. A `/c/Users/...` path reaches the runner unconverted and is read as its `-c` option. A relative path runs DuckDB's own tests instead of ours.
  - One file: the same runners with the file's path in place of `test/sql/*`.
  - Baseline before Task 1: 2913 assertions in 75 test cases on both lines. The runner counts a `query` block as 2 assertions and a `statement` block as 1.
  - Do not pipe a build into `tail` or `head`, since that hides its exit code. Redirect to a log file and check `$?`.
- Builds have 0 warnings (MSVC, VS 2022).
- `git status` shows ` m duckdb`. That is expected: an applied patch in the submodule. Never stage it, and never report it.
- Stage files by name. Never use `git add -A` or `git add .`.
- A commit message has one plain subject line, a short body, and this last line: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Write code comments, docs, test comments and commit messages in plain English. Use short linear sentences, one idea each, and state facts instead of praise. No buzzwords (leverage as a verb, seamless, powerful, robust as praise, utilize, facilitate) and no slang (plumbing, gotcha, under the hood).
- sqllogictest format:
  - A blank line ends a statement, so write each statement on one line.
  - `statement error` is followed by `----` and the expected message. The runner matches it as a substring of the actual error.
  - `query TT` output is the spec, one tab, and the layer-SQL map.
- Follow the style of the files you touch: tabs, `IEqual()` for case-insensitive keyword checks, `StringUtil` helpers, and lookup tables as static arrays scanned in order (like `kChannelAliases`). Never add a `std::unordered_map` keyed by `std::string`, because it breaks the wasm build.
- Keywords, setting names, transform names and scale-type names are case-insensitive, like the rest of the grammar. Error messages quote the user's tokens as written. Where this plan gives a message's text, that text is exact.

## Review Focus

1. **Glued arrow.** `reverse=>true` written without spaces gives the same spec on 1.4.3 and 2.0.0. DuckDB 2.0's tokenizer splits it into three tokens, as probed on 2026-11-22. Our own tokenizer, which 1.4.3 uses, keeps it as one. Task 1 pins this.
2. **Nominal channels.** `VIA` or `BINNED` on `color`, `fill` or `stroke`, which are nominal by default, must come out `quantitative`. Otherwise Vega-Lite drops the scale with a warning. Tasks 2 and 3 pin this.
3. **Histograms.**
   - `SCALE BINNED x` must not write a second `"bin"` key.
   - `SCALE x VIA log` fails with a message that says why.
   - `SCALE y VIA log`, the count axis, works.
   Tasks 2 and 3 pin these.
4. **Several layers.** Every layer that maps the channel gets the same scale and type. Task 2 pins this.
5. **Mixed case.** `via LOG`, `SETTING REVERSE => FALSE` and `scale binned color` are accepted like every other keyword. Each task pins one.

## Files

| File | Change |
|---|---|
| `src/ggsql_grammar.cpp` | `Tokenize` emits `=>`. The SCALE loop takes `SETTING`, `VIA` and a leading `BINNED`. |
| `src/include/ggsql_grammar.hpp` | `ScaleSpec` gains `via` and `binned`; `sub_object = ""` means a property of the channel itself. |
| `src/ggsql.cpp` | Adds `EffectiveTypeOverrides` and `CheckBinnedTransforms`. `ApplyScales` writes direct properties. Three words join `IsClauseKeyword`. |
| `test/sql/ggsql_visualize_scale_reverse_via_binned.test` | New; Tasks 2 and 3 extend it. |
| `test/sql/ggsql_compat_upstream.test` | One heatmap block moves to the passing section. |
| `test/sql/ggsql_visualize_titles.test` | The pinned operator list, line 119. |
| `docs/visualize.md`, `README.md`, `CHANGELOG.md`, `notes/engineering/` | Docs, the changelog bullet and an engineering note. |

---

### Task 1: `SETTING reverse => true|false`

**Files:**
- Modify: `src/ggsql_grammar.cpp`: `Tokenize`, plus the SCALE clause loop and its header comment (about lines 624-700).
- Modify: `src/ggsql.cpp`: add `"SETTING"` to the word list in `IsClauseKeyword`.
- Create: `test/sql/ggsql_visualize_scale_reverse_via_binned.test`
- Modify: `test/sql/ggsql_compat_upstream.test`, `test/sql/ggsql_visualize_titles.test:119`
- Modify: `docs/visualize.md`, section 7.

**Interfaces:**
- Consumes: `ScaleSpec {aesthetic, sub_object = "scale", property, value_json}` in `src/include/ggsql_grammar.hpp`. `ApplyScales` merges the entries of one channel and sub-object into one JSON object, in insertion order.
- Produces: the `=>` token from `Tokenize`; the SCALE option `SETTING`; the test file and its conventions, which Tasks 2 and 3 extend; and the operator-list messages, which Task 2 extends.

- [ ] **Step 1: Write the failing tests.** Create the file with this header and table:

```
# name: test/sql/ggsql_visualize_scale_reverse_via_binned.test
# description: SCALE ... SETTING reverse, SCALE ... VIA <transform> and SCALE BINNED, from upstream ggsql (#98, #58 item 6)
# group: [stats_duck]

require stats_duck

statement ok
CREATE TABLE penguins AS SELECT * FROM (VALUES (39.1, 18.7, 3750, 'Adelie'), (39.5, 17.4, 3800, 'Adelie'), (44.0, 18.0, 5000, 'Gentoo')) AS t(bill_len, bill_dep, body_mass, species);
```

The tables below give only the channels a test changes. Each query writes out the whole expected line in the file:

- Statements on `VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point ...` expect:
  `{"$schema":"https://vega.github.io/schema/vega-lite/v5.json","data":{"name":"layer_0"},"mark":"point","encoding":{"x":{"field":"x","type":"quantitative"},"y":<Y>}}` TAB `{layer_0='SELECT bill_len AS x, bill_dep AS y FROM penguins'}`
- Statements with a third mapping `<e> AS color` end the encoding with `,"color":<C>}}` and the layer SQL with `, <e> AS color FROM penguins'}`.
- `<Y>` is `{"field":"y","type":"quantitative"}` unless the table says otherwise.

| statement tail | changed channel |
|---|---|
| `SCALE y SETTING reverse => true` | `<Y>` = `{"field":"y","type":"quantitative","scale":{"reverse":true}}` |
| `SCALE y SETTING reverse=>true` | same as above |
| `SCALE y SETTING REVERSE => FALSE` | `<Y>` = `{"field":"y","type":"quantitative","scale":{"reverse":false}}` |
| with `species AS color`: `SCALE color TO viridis SETTING reverse => true` | `<C>` = `{"field":"color","type":"nominal","scale":{"scheme":"viridis","reverse":true}}` |

Errors, as `statement error` blocks on `VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point SCALE y <tail>`:

| tail | expected message |
|---|---|
| `SETTING breaks => 5` | `Unsupported SCALE setting 'breaks' (supported: reverse)` |
| `SETTING reverse => true, breaks => 5` | `Unsupported SCALE setting 'breaks' (supported: reverse)` |
| `SETTING reverse => maybe` | `Expected true or false after 'SETTING reverse =>', got 'maybe'` |
| `SETTING reverse true` | `Expected '=>' after 'SETTING reverse'` |
| `SETTING reverse =>` | `Expected a value after 'SETTING reverse =>'` |
| `SETTING` | `Expected a setting name after 'SETTING'` |
| `SETTING reverse => true,` | `Expected a setting name after ','` |

- [ ] **Step 2: Run the file on 1.4.3 and see it fail.** Expected: the first query fails with `Unknown SCALE operator 'SETTING'`.

- [ ] **Step 3: Implement.**
  - `Tokenize`: outside quotes and parentheses, `=>` is a token of its own, like `,`. DuckDB 2.0's tokenizer already splits `reverse=>true` into `reverse`, `=>` and `true`; this makes 1.4.3 do the same.
  - The SCALE loop takes a new option, `SETTING <name> => <value> [, <name> => <value>]...`. The list continues while the next token is `,`.
  - The only setting name is `reverse`, in any case. Its value is `true` or `false`, in any case, and becomes `ScaleSpec{aesthetic, "scale", "reverse", "true"|"false"}`.
  - Any other name, and every malformed form, fails with the Step 1 message.
  - Both operator-list messages gain SETTING. They become `Unknown SCALE operator '<op>' (expected TO / ZERO / DOMAIN / LABEL / SETTING)` and `Expected SCALE operator (TO / ZERO / DOMAIN / LABEL / SETTING) after aesthetic`. Update the loop's header comment to match.
  - `IsClauseKeyword` in `src/ggsql.cpp`: add `"SETTING"`. The list's comment says it covers the VISUALIZE grammar. Nothing changes in behaviour, because no `(`, `:` or sign follows `SETTING`.

- [ ] **Step 4: Follow-on test edits.**
  - `test/sql/ggsql_visualize_titles.test:119` now pins `Unknown SCALE operator 'BOGUS' (expected TO / ZERO / DOMAIN / LABEL / SETTING)`.
  - `test/sql/ggsql_compat_upstream.test`: the heatmap block that pins `Unknown SCALE operator 'SETTING'` now runs.
    - Move it to the end of the "Passing" section as a `statement ok` block.
    - Give it the comment `# gallery/examples/heatmap.qmd, reversed magma fill (SETTING reverse, #98)`.
    - Keep the statement text as it is.
    - The passing count becomes 12 of 55. The file does not record that count, so do not add it.

- [ ] **Step 5: Docs.** In `docs/visualize.md` section 7:
  - Add the row `| \`SETTING reverse => true\|false\` | reverse the direction of the scale |` to the option table.
  - After the table, add one paragraph that says three things. `SETTING` takes upstream ggsql's `name => value` form, with commas between settings. `reverse` is the only setting so far, and another name is an error. `=>` may be written with or without spaces.

- [ ] **Step 6: Verify.** The new file and the whole suite pass on 1.4.3 and 2.0.0. Expected: 2929 assertions in 76 test cases. That is the baseline plus 16: the table, 4 queries and 7 errors.

- [ ] **Step 7: Commit.** Stage `src/ggsql_grammar.cpp src/ggsql.cpp test/sql/ggsql_visualize_scale_reverse_via_binned.test test/sql/ggsql_compat_upstream.test test/sql/ggsql_visualize_titles.test docs/visualize.md`. Subject: `VISUALIZE: accept SCALE ... SETTING reverse => true (#98)`.

---

### Task 2: `VIA <transform>`

**Files:**
- Modify: `src/include/ggsql_grammar.hpp`: `ScaleSpec` gains `via`.
- Modify: `src/ggsql_grammar.cpp`: the SCALE loop takes `VIA`.
- Modify: `src/ggsql.cpp`: add `EffectiveTypeOverrides`, use it in `Compile`, and add `"VIA"` to `IsClauseKeyword`.
- Modify: `test/sql/ggsql_visualize_scale_reverse_via_binned.test`, `test/sql/ggsql_visualize_titles.test:119`
- Modify: `docs/visualize.md`, section 7.

**Interfaces:**
- Consumes: Task 1's SCALE loop, operator-list messages and test file.
- Produces:
  - `string via;` on `ScaleSpec`. It holds the transform name in lower case. It is set on every entry a `VIA` produced, and empty otherwise.
  - `vector<TypeOverride> EffectiveTypeOverrides(const VisualizeStatement &stmt)` in `src/ggsql.cpp`, next to `ApplyTypeOverrides` and with the same linkage. It returns `stmt.type_overrides`, followed by one `{aesthetic, "quantitative"}` for each aesthetic that has an entry with a non-empty `via` and no entry in `stmt.type_overrides`. It adds one entry per aesthetic, in first-seen order.
  - `Compile` passes this list to `ApplyTypeOverrides` on both the single-layer and the multi-layer path. Task 3 extends `EffectiveTypeOverrides`.

Transforms. Every entry has `sub_object = "scale"` and `via` = the name. `value_json` is JSON text: `"\"log\""`, `"2"` and so on.

| name | entries pushed, in order |
|---|---|
| `linear` | `type` = `"linear"` |
| `log` | `type` = `"log"` |
| `log2` | `type` = `"log"`, `base` = `2` |
| `ln` | `type` = `"log"`, `base` = `2.718281828459045` |
| `sqrt` | `type` = `"sqrt"` |
| `square` | `type` = `"pow"`, `exponent` = `2` |
| `asinh`, `pseudo_log`, `pseudo_log2`, `pseudo_ln` | `type` = `"symlog"` |

- [ ] **Step 1: Write the failing tests.** Append to the test file, using the conventions from Task 1.

| statement tail on `VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point` | `<Y>` |
|---|---|
| `SCALE y VIA log` | `{"field":"y","type":"quantitative","scale":{"type":"log"}}` |
| `SCALE y VIA log2` | `{"field":"y","type":"quantitative","scale":{"type":"log","base":2}}` |
| `SCALE y VIA ln` | `{"field":"y","type":"quantitative","scale":{"type":"log","base":2.718281828459045}}` |
| `SCALE y VIA sqrt` | `{"field":"y","type":"quantitative","scale":{"type":"sqrt"}}` |
| `SCALE y VIA square` | `{"field":"y","type":"quantitative","scale":{"type":"pow","exponent":2}}` |
| `SCALE y VIA asinh` | `{"field":"y","type":"quantitative","scale":{"type":"symlog"}}` |
| `SCALE y VIA pseudo_ln` | `{"field":"y","type":"quantitative","scale":{"type":"symlog"}}` |
| `SCALE y via LINEAR` | `{"field":"y","type":"quantitative","scale":{"type":"linear"}}` |
| `SCALE y VIA log ZERO false SETTING reverse => true` | `{"field":"y","type":"quantitative","scale":{"type":"log","zero":false,"reverse":true}}` |

More queries:
- `VISUALIZE bill_len AS x, bill_dep AS y, body_mass AS color FROM penguins DRAW point SCALE color VIA log`. `<C>` = `{"field":"color","type":"quantitative","scale":{"type":"log"}}`; the channel was nominal.
- `VISUALIZE bill_len AS x, bill_dep AS y, body_mass AS color:nominal FROM penguins DRAW point SCALE color VIA log`. `<C>` = `{"field":"color","type":"nominal","scale":{"type":"log"}}`; the `:type` suffix takes precedence. The layer SQL carries no suffix: `{layer_0='SELECT bill_len AS x, bill_dep AS y, body_mass AS color FROM penguins'}`.
- Two layers: `VISUALIZE bill_len AS x, bill_dep AS y, body_mass AS color FROM penguins DRAW point DRAW line SCALE color VIA sqrt` expects
  `{"$schema":"https://vega.github.io/schema/vega-lite/v5.json","layer":[{"data":{"name":"layer_0"},"mark":"point","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"color":{"field":"color","type":"quantitative","scale":{"type":"sqrt"}}}},{"data":{"name":"layer_1"},"mark":"line","encoding":{"x":{"field":"x","type":"quantitative"},"y":{"field":"y","type":"quantitative"},"color":{"field":"color","type":"quantitative","scale":{"type":"sqrt"}}}}]}` TAB `{layer_0='SELECT bill_len AS x, bill_dep AS y, body_mass AS color FROM penguins', layer_1='SELECT * FROM (SELECT bill_len AS x, bill_dep AS y, body_mass AS color FROM penguins) ORDER BY x'}`
- The histogram's count axis: `VISUALIZE bill_len AS x FROM penguins DRAW histogram SCALE y VIA log` expects
  `{"$schema":"https://vega.github.io/schema/vega-lite/v5.json","data":{"name":"layer_0"},"mark":"bar","encoding":{"x":{"field":"x","type":"quantitative","bin":true},"y":{"aggregate":"count","type":"quantitative","scale":{"type":"log"}}}}` TAB `{layer_0='SELECT bill_len AS x FROM penguins'}`

Errors:

| statement | expected message |
|---|---|
| `... DRAW point SCALE y VIA` | `Expected a transform name after 'VIA'` |
| `... DRAW point SCALE y VIA exp` | `Unsupported SCALE transform 'exp' (supported: linear, log, log2, ln, sqrt, square, asinh, pseudo_log, pseudo_log2, pseudo_ln)` |
| `... DRAW point SCALE y VIA symlog` | the same message with `'symlog'`. That is Vega-Lite's name, not upstream's. |
| `VISUALIZE bill_len AS x, bill_dep AS y, species AS shape FROM penguins DRAW point SCALE shape VIA log` | `SCALE shape cannot take VIA: Vega-Lite has no continuous scale for shape` |

- [ ] **Step 2: Run the file on 1.4.3 and see the new blocks fail.** Expected: `Unknown SCALE operator 'VIA'`.

- [ ] **Step 3: Implement.**
  - The SCALE loop takes `VIA <name>`. A name not in the table fails with the message above.
  - The canonical aesthetics `shape`, `linetype`, `text` and `tooltip` take no `VIA`; they fail with `SCALE <aesthetic> cannot take VIA: Vega-Lite has no continuous scale for <aesthetic>`. In Vega-Lite, `shape` and `strokeDash` accept only discrete scales, and `text` and `tooltip` have no scale.
  - Add `EffectiveTypeOverrides` as specified, and use it in `Compile`. Vega-Lite drops an explicit scale type that does not fit the field's type, and `color`, `fill` and `stroke` are nominal in our channel table. Say this in one comment at the function.
  - The operator lists gain VIA: `(expected TO / ZERO / DOMAIN / LABEL / VIA / SETTING)` and `Expected SCALE operator (TO / ZERO / DOMAIN / LABEL / VIA / SETTING) after aesthetic`. Update the loop's header comment and `test/sql/ggsql_visualize_titles.test:119` to match.
  - `IsClauseKeyword`: add `"VIA"`.

- [ ] **Step 4: Docs.** In `docs/visualize.md` section 7, add the row `| \`VIA <transform>\` | transform the scale (table below) |`. Below the option table, add a two-column table, `VIA` and Vega-Lite scale:
  - `linear` → `linear`
  - `log`, `log2`, `ln` → `log` with base 10, 2 and e
  - `sqrt` → `sqrt`
  - `square` → `pow` with exponent 2
  - `asinh`, `pseudo_log`, `pseudo_log2`, `pseudo_ln` → `symlog`

  After that table, add these notes:
  - The curve of `symlog` is close to upstream's but not the same.
  - Upstream's `exp`, `exp2`, `exp10`, `integer`, `date`, `datetime` and `time` are not supported yet.
  - `VIA` makes the channel quantitative, because Vega-Lite applies a continuous scale only to a quantitative channel. A `:type` suffix on the mapping takes precedence.
  - `shape`, `linetype`, `text` and `tooltip` take no `VIA`.

- [ ] **Step 5: Verify.** The whole suite passes on both lines. Expected: 2959 assertions in 76 test cases, which adds 13 queries and 4 errors.

- [ ] **Step 6: Commit.** Stage `src/include/ggsql_grammar.hpp src/ggsql_grammar.cpp src/ggsql.cpp test/sql/ggsql_visualize_scale_reverse_via_binned.test test/sql/ggsql_visualize_titles.test docs/visualize.md`. Subject: `VISUALIZE: accept SCALE ... VIA <transform> (#98)`.

---

### Task 3: `SCALE BINNED <channel>`, and the closing docs

**Files:**
- Modify: `src/include/ggsql_grammar.hpp`: `ScaleSpec` gains `binned`; document `sub_object = ""`.
- Modify: `src/ggsql_grammar.cpp`: SCALE reads a scale-type word before the aesthetic.
- Modify: `src/ggsql.cpp`: direct properties in `ApplyScales`, `EffectiveTypeOverrides`, a new `CheckBinnedTransforms`, and `"BINNED"` in `IsClauseKeyword`.
- Modify: `test/sql/ggsql_visualize_scale_reverse_via_binned.test`
- Modify: `docs/visualize.md` (section 7 and the grammar reference), `README.md` (grammar block near line 1024, and the paragraph after it), `CHANGELOG.md`
- Create: `notes/engineering/2026-11-scale-reverse-via-binned.md`. Modify: `notes/engineering/README.md`.

**Interfaces:**
- Consumes: `ScaleSpec::via` and `EffectiveTypeOverrides` from Task 2.
- Produces:
  - `bool binned = false;` on `ScaleSpec`.
  - `SCALE BINNED <aesthetic>` pushes `{aesthetic, sub_object = "", property = "bin", value_json = "true", binned = true}` before the clause's own options. So `"bin"` comes before the channel's sub-objects.
  - `sub_object = ""` means a property of the channel object itself.
  - `void CheckBinnedTransforms(const VisualizeStatement &stmt)` in `src/ggsql.cpp`, called in `Compile` right after the check for an empty layer list.

Behaviour:
- **Parser.**
  - After `SCALE`, the word `BINNED`, in any case, comes before the aesthetic. A `BINNED` clause may have no options.
  - Without `BINNED`, the "Expected SCALE operator ..." error stays.
  - `CONTINUOUS`, `DISCRETE`, `ORDINAL` or `IDENTITY` in that place, in any case, fails with `SCALE <word> is not supported (supported scale type: BINNED)`, with the word as written.
- **`ApplyScales`.**
  - An entry with `sub_object == ""` is written as `,"<property>":<value>` inside the channel object, at its insertion position among the channel's sub-objects.
  - A direct property is written at most once per channel. It is skipped when the channel object already has that key, as a histogram's `x` has `"bin":true`, or when an earlier entry wrote it.
- **`EffectiveTypeOverrides`.** An aesthetic with a `binned` entry is treated like one with a `via`.
- **`CheckBinnedTransforms`.** It throws `InvalidInputException` in two cases:
  - An aesthetic has a `binned` entry and a `via` other than `linear`, even from a separate SCALE clause: `visualize: SCALE BINNED <aesthetic> cannot take VIA <via>: Vega-Lite bins on a linear scale`
  - A layer's mark is `histogram`, and `x` has a `via` other than `linear`: `visualize: SCALE x VIA <via> cannot apply to the histogram's x, which Vega-Lite bins on a linear scale; transform the column in the mapping instead`

- [ ] **Step 1: Write the failing tests.** Append to the test file, using the conventions from Task 1. In the first five rows below, the third mapping is `body_mass AS color`.

| statement tail | `<C>` |
|---|---|
| `SCALE BINNED color` | `{"field":"color","type":"quantitative","bin":true}` |
| `scale binned color TO viridis` | `{"field":"color","type":"quantitative","bin":true,"scale":{"scheme":"viridis"}}` |
| `SCALE BINNED color VIA linear` | `{"field":"color","type":"quantitative","bin":true,"scale":{"type":"linear"}}` |
| `SCALE BINNED color SCALE BINNED color` | `{"field":"color","type":"quantitative","bin":true}` |
| with `body_mass AS color:ordinal`: `SCALE BINNED color` | `{"field":"color","type":"ordinal","bin":true}` |

More queries:
- `VISUALIZE bill_len AS x, bill_dep AS y FROM penguins DRAW point SCALE BINNED color` leaves the spec as if there were no SCALE clause, because the channel is unmapped.
- `VISUALIZE bill_len AS x FROM penguins DRAW histogram SCALE BINNED x` expects
  `{"$schema":"https://vega.github.io/schema/vega-lite/v5.json","data":{"name":"layer_0"},"mark":"bar","encoding":{"x":{"field":"x","type":"quantitative","bin":true},"y":{"aggregate":"count","type":"quantitative"}}}` TAB `{layer_0='SELECT bill_len AS x FROM penguins'}`

Errors:

| statement | expected message |
|---|---|
| `... DRAW point SCALE BINNED` | `Expected aesthetic name after 'SCALE BINNED'` |
| `... DRAW point SCALE DISCRETE color` | `SCALE DISCRETE is not supported (supported scale type: BINNED)` |
| `... DRAW point SCALE ordinal color` | `SCALE ordinal is not supported (supported scale type: BINNED)` |
| `... DRAW point SCALE y` | `Expected SCALE operator (TO / ZERO / DOMAIN / LABEL / VIA / SETTING) after aesthetic` |
| with `body_mass AS color`: `SCALE BINNED color VIA log` | `SCALE BINNED color cannot take VIA log: Vega-Lite bins on a linear scale` |
| with `body_mass AS color`: `SCALE BINNED color SCALE color VIA sqrt` | `SCALE BINNED color cannot take VIA sqrt` |
| `VISUALIZE bill_len AS x FROM penguins DRAW histogram SCALE x VIA log` | `SCALE x VIA log cannot apply to the histogram's x` |

- [ ] **Step 2: Run the file on 1.4.3 and see the new blocks fail.**

- [ ] **Step 3: Implement** the behaviour above, and add `"BINNED"` to `IsClauseKeyword`.

- [ ] **Step 4: Docs.**
  - **Section 7 of `docs/visualize.md`.** Add a paragraph on `SCALE BINNED <channel>` that covers these points:
    - It sets Vega-Lite's `"bin": true` on the channel.
    - It needs no further option, and it takes the others (`SCALE BINNED color TO viridis`).
    - Like `VIA`, it makes the channel quantitative unless the mapping gives a type.
    - A histogram bins its `x` already, so `SCALE BINNED x` adds nothing there.
    - Vega-Lite makes bins of equal width in the data's own units. So a binned channel takes no `VIA` other than `linear`, and neither does a histogram's `x`. To bin on a log scale, transform the column in the mapping (`log10(body_mass) AS x`).
    - Upstream's other scale types (`CONTINUOUS`, `DISCRETE`, `ORDINAL`, `IDENTITY`) are not supported yet.
  - **Grammar references.** In `docs/visualize.md` and in `README.md`, replace the SCALE line of the grammar block with these two lines:

```
[SCALE [BINNED] <channel> {TO <scheme> | ZERO true|false | DOMAIN <lo> <hi> | LABEL '<text>'
                         | VIA <transform> | SETTING reverse => true|false}*]*
```

    Under the docs block, add the bullet `**SCALE:** without \`BINNED\`, a \`SCALE\` needs at least one option.`
  - **`README.md`.** In the paragraph after the grammar block that begins "Multiple `SCALE` options may be **stacked**", add: `` `SETTING reverse => true`, `VIA <transform>` and `SCALE BINNED <channel>` follow upstream ggsql; `docs/visualize.md` lists the transforms and the rules. ``
  - **`CHANGELOG.md`.** Under `## [Unreleased]`, `### Added`, append after the #94 bullet:

```
- **`SCALE ... SETTING reverse => true`, `SCALE ... VIA <transform>` and
  `SCALE BINNED <channel>`** (#98, Tier 1 item 6 of #58). The three follow
  upstream ggsql's syntax. `reverse` sets the scale's `reverse` property.
  `VIA` maps upstream's transform names to Vega-Lite scale types: `log`,
  `log2` and `ln` to a log scale with base 10, 2 and e, `sqrt` to a
  square-root scale, `square` to a power scale with exponent 2, `asinh` and
  the three `pseudo_log` names to `symlog`, and `linear`. `BINNED` sets
  `"bin": true` on the channel and needs no further option. `VIA` and
  `BINNED` make the channel quantitative unless its mapping carries a
  `:type` suffix, because Vega-Lite drops a continuous scale on a nominal
  channel. `shape`, `linetype`, `text` and `tooltip` take no `VIA`. Vega-Lite
  bins in the data's own units, so a binned channel, a histogram's `x`
  included, takes no transform other than `linear`. `SETTING` takes the
  `name => value` form; `reverse` is its only setting so far. One more
  statement of `test/sql/ggsql_compat_upstream.test` passes: 12 of 55.
```

  - **Engineering note.** Create `notes/engineering/2026-11-scale-reverse-via-binned.md` in the style of its neighbours, and add an index entry to `notes/engineering/README.md` in the existing format. It records four findings:
    1. `=>` written with spaces is one token on both DuckDB lines. Glued `reverse=>true` was three tokens on 2.0 and one on 1.4.3, so `Tokenize` now splits it.
    2. Vega-Lite drops an explicit scale type that does not fit the field's type (`scaleType()` in `src/compile/scale/type.ts`, v5.23.0). That is why `VIA` and `BINNED` retype the channel, and why an explicit `:type` wins.
    3. Vega-Lite's bin transform makes bins of equal width in data units, while upstream bins in transformed units. That is why a binned channel takes no transform, the histogram's `x` included.
    4. Vega's `symlog` is `sign(x) * log1p(|x| / C)` with `C = 1`. `asinh` and upstream's pseudo-log transforms share its two regimes, linear near zero and logarithmic far from it, but not its curve in between.

- [ ] **Step 5: Verify.** The whole suite passes on both lines. Expected: 2980 assertions in 76 test cases, which adds 7 queries and 7 errors.

- [ ] **Step 6: Commit.** Stage the source files, the test file, `docs/visualize.md`, `README.md`, `CHANGELOG.md` and both notes files. Subject: `VISUALIZE: accept SCALE BINNED <channel> (#98)`.
