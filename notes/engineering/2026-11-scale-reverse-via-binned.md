# SCALE SETTING, VIA and BINNED (#98)

Date: 2026-11-23. Branch `feat/98-scale-reverse-via-binned`.

## What changed

`SCALE` gained three options from upstream ggsql. `SETTING reverse => true`
sets the scale's `reverse` property. `VIA <transform>` sets the Vega-Lite
scale type. `SCALE BINNED <channel>` sets `"bin": true` on the channel. Four
findings came out of the work. Each one explains a rule in the code.

## 1. The `=>` arrow is one token or three

Written with spaces, `reverse => true` is the same three tokens on both
DuckDB lines. Written without spaces, `reverse=>true` is one token on
1.4.3, where our own tokenizer runs, and three tokens on 2.0.0, where
DuckDB's tokenizer feeds the adapter. The two lines would then give two
different specs for one statement. `Tokenize` now emits `=>` as its own
token, so both lines see the same three. A test pins the glued form.

## 2. Vega-Lite drops a scale type that does not fit the field

`scaleType()` in `src/compile/scale/type.ts` (Vega-Lite v5.23.0) checks the
explicit scale type against the field's type. A `log` scale on a nominal
field does not fit, so Vega-Lite ignores it and logs a warning. `color`,
`fill` and `stroke` are nominal in our channel table. A `SCALE color VIA log`
would therefore do nothing. `VIA` and `BINNED` now retype the channel as
quantitative. A `:type` suffix on the mapping wins over that, because the
user wrote it.

## 3. Vega-Lite bins in data units, upstream bins in transformed units

Upstream's docs describe `VIA` as a transform applied before the data is
mapped (`doc/syntax/scale/type/continuous.qmd`). They do not say whether
binning comes before or after the transform. The rule here rests on
Vega-Lite alone. Its bin transform runs on the raw values and gives bins of
equal width in data units. The `log` scale then only changes how those bins
are drawn, so on a log axis the bars have unequal widths. A reader can
easily take them for bins of equal width on the log scale. So a binned
channel takes no `VIA` other than `linear`, and neither does a histogram's
`x`. Both cases fail with a message that says why. To bin on a log scale,
transform the column in the mapping, for example `log10(body_mass) AS x`.

## 4. `symlog` is close to upstream's pseudo-log, not equal

Vega's `symlog` is `sign(x) * log1p(|x| / C)` with `C = 1`. `asinh` and
upstream's three pseudo-log transforms share its two regimes: linear near
zero and logarithmic far from it. The curve between the two regimes differs.
The four names all map to `symlog`, and `docs/visualize.md` says so.

## 5. `VIA` changes the scale, not the computed values

Vega-Lite computes the density, the violin's density, the regression and
the loess on the raw values. The summary stat averages raw values in the
layer SQL. `VIA` changes only the scale that draws the result. ggplot2
transforms before it computes. `docs/visualize.md` and the CHANGELOG say so.
To get the ggplot2 result, transform the column in the mapping.

Open question: should the density's `x` and the violin's `y` reject a
non-linear `VIA`, as the histogram's `x` does? Nothing rejects it today.
