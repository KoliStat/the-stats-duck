# One tokenizer decides which statements are ours

*November 2026, issue #86, found by the code review of PR #85.*

A DuckDB parser extension is asked about a statement only after DuckDB's own
parser has rejected it. The extension then has to decide whether the
statement is ours, because the answer picks the error message the user sees:
ours, or DuckDB's syntax error at the first keyword. Until #86 that decision
was made by a private byte scan in `src/ggsql.cpp`, while the grammar in
`src/ggsql_grammar.cpp` tokenized the same text its own way. The two
disagreed at the edges, and each edge was a wrong error message.

## What went wrong

- The byte scan did not skip SQL comments. `WITH c AS (...) -- it's filtered`
  followed by `VISUALISE ...` was handed to DuckDB, because the apostrophe in
  the comment opened a string that never closed. A `/* note ( */` failed the
  same way through the parenthesis counter.
- The scan looked for the first depth-0 word `visualize` anywhere after the
  `WITH`. A statement that was not ours, with the word in a column alias or
  a comment, was claimed and got an extension error. A CTE named `visualize`
  truncated the `WITH` clause at the name.
- The direct form accepted only whitespace or `;` after the keyword, while
  the tokenizer also splits on `--`, `/*` and `,`. `VISUALISE--chart` on one
  line fell through to DuckDB.
- The keyword pair was spelled out in four places. The `VISUALISE` spelling
  (#84) had to be added to each of them, with a byte-index special case in
  the scan.

## The fix

The claim is a function of the grammar, `IsVisualizeStatement`, built on the
grammar's `Tokenize`. The same function, `ClaimTokens`, serves the parser, so
the claim and the parse cannot drift again. Four decisions in it are worth
recording.

1. **Walk the `WITH` block by its structure, not by searching for the
   keyword.** A CTE list has a fixed shape: a name, anything up to `AS` (a
   column list, `USING KEY`), anything up to the body (`MATERIALIZED`, `NOT
   MATERIALIZED`), the parenthesised body, and a comma before the next one.
   The token after the list decides ownership. A CTE named `visualize` is a
   name. A `WITH ... SELECT` is DuckDB's whatever words follow at depth 0.
2. **A malformed `WITH` followed by the keyword is ours.** When the walk does
   not find the shape above but a keyword token follows anyway, the user
   wrote a `VISUALIZE` statement with a broken `WITH` clause. The error names
   the `WITH` clause. DuckDB's error would name the keyword, which is the one
   part the user got right.
3. **A tokenizer error does not decide ownership by itself.** The tokens
   read so far do. A direct statement with an unterminated quote later on is
   ours, and the parser names the quote. A `WITH` statement with the quote
   inside a CTE body never exposes the keyword as a token, so it stays with
   DuckDB, whose own message names the quote.
4. **The body is the first token after `AS` that contains a `(`, not one
   that starts with it.** On DuckDB 2.0 the parser extension receives tokens
   without positions. The adapter in `ggsql.cpp` rebuilds the text with
   conventional spacing and glues a word to a following `(` unless the word
   is a clause keyword. `MATERIALIZED` was not in that list, so the body
   arrived as `MATERIALIZED(SELECT ...)`. The keyword is in the list now, and
   the walk tolerates the glued form anyway.

## What the 2.0 token adapter cannot do

Because the tokens carry no positions, `VISUALIZE(n AS x)` and
`VISUALIZE (n AS x)` are the same text on 2.0. On 1.x the glued form is one
token to our tokenizer and the parser says "must be followed by a space"; on
2.0 the grammar sees `(n AS x)` as the first mapping and names that. Both
are our errors, and both lines claim the statement, but a test cannot pin
one message for both. `test/sql/ggsql_visualize_claim.test` pins the error
type for the glued form and the exact mapping error for the spaced form.

The same limit applies to every negative test of the claim: a statement that
is not ours must be invalid SQL to reach the extension at all, and the token
DuckDB names in its syntax error differs between the 1.x and 2.0 parsers.
Those tests use a misspelt `SELECT` and pin `Parser Error` only.

## Rule

One tokenizer, one claim function, shared by the extension and the parser.
When the trailing `SELECT ... VISUALISE x, y DRAW point` form of #58 lands,
it extends `ClaimTokens` with a walk over a complete `SELECT`; it does not
add a second scanner.
