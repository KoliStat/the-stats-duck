#!/usr/bin/env bash
#
# Regression guard for the WASM build.
#
# A std::string-keyed std::unordered_map / unordered_set with the DEFAULT hash
# lowers std::hash<std::string> to libc++'s std::__hash_memory. @duckdb/duckdb-wasm's
# main module does NOT export that symbol (it keys its own maps on duckdb::string_t),
# so the loadable extension's import is stubbed with a throwing placeholder and the
# first key-hash traps at runtime as "TypeError: t is not a function" — even though
# the native build and the wasm build both compile cleanly. It once crashed
# table_one via anova_oneway / chisq_independence.
#
# Such containers must use stats_duck::PortableStringHash as their hash argument.
# See src/include/portable_string_hash.hpp and
# notes/engineering/2026-06-duckdb-version-must-match-duckdb-wasm.md.
#
# Exits non-zero if any src/ line declares such a container without PortableStringHash.
# (string_t-keyed maps and std::map (tree, not hashed) are correctly NOT flagged.)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# An unordered container whose key is std::string / `string` (NOT string_t):
#   key is `string` followed by optional space then ',' (map) or '>' (set).
PATTERN='unordered_(map|set|multimap|multiset)<[[:space:]]*(std::)?string[[:space:]]*[,>]'

violations="$(grep -rnE "$PATTERN" "$ROOT/src" --include='*.cpp' --include='*.hpp' \
  | grep -v 'PortableStringHash' || true)"

if [[ -n "$violations" ]]; then
  echo "FAIL: std::string-keyed unordered container without stats_duck::PortableStringHash"
  echo
  echo "$violations"
  echo
  echo "The default std::hash<std::string> pulls in libc++ std::__hash_memory, which"
  echo "@duckdb/duckdb-wasm does not export -> the WASM build crashes at runtime with"
  echo "'t is not a function'. Pass stats_duck::PortableStringHash as the hash template"
  echo "argument (src/include/portable_string_hash.hpp)."
  exit 1
fi

echo "OK: no std::string-keyed unordered container uses the default hash."
