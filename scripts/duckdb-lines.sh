#!/usr/bin/env bash
#
# Print the DuckDB versions that the distribution pipeline builds, as one JSON
# list on one line:
#
#   ["v1.4.5","v1.5.6","v2.0.0"]
#
# The versions are the values of the "duckdb_version:" keys in
# .github/workflows/MainDistributionPipeline.yml. The compile check in
# .github/workflows/lint.yml runs once for each of them. So the two workflows
# cannot name different versions.
#
# Usage:
#   bash scripts/duckdb-lines.sh [<workflow-file>]
#
# Exit codes: 0 at least one version, 1 no version, a key that cannot be read,
# or a missing file.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
file="${1:-$ROOT/.github/workflows/MainDistributionPipeline.yml}"

if [[ ! -f "$file" ]]; then
  echo "duckdb-lines: no such file: $file" >&2
  exit 1
fi

# A key is a line that starts with "duckdb_version:" after spaces only, so a
# commented line is not a key. The value may be quoted, and a comment may
# follow it.
key='^[[:space:]]*duckdb_version:'
value='[[:space:]]*["'"'"']?v[0-9]+\.[0-9]+\.[0-9]+["'"'"']?[[:space:]]*(#.*)?$'

keys="$(grep -cE "$key" "$file" || true)"
read_keys="$(grep -cE "$key$value" "$file" || true)"

# Every key must be read. A key in a form this script does not know would
# otherwise drop one DuckDB line from the check without a message.
if [[ "$keys" -ne "$read_keys" ]]; then
  echo "duckdb-lines: $((keys - read_keys)) duckdb_version lines in $file are not of the form vX.Y.Z:" >&2
  grep -E "$key" "$file" | grep -vE "$key$value" | sed 's/^/  /' >&2
  exit 1
fi
if [[ "$keys" -eq 0 ]]; then
  echo "duckdb-lines: no duckdb_version key in $file" >&2
  exit 1
fi

# -V sorts by version, so v1.9.2 comes before v1.10.0.
versions="$(grep -E "$key$value" "$file" | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | LC_ALL=C sort -u -V)"

printf '['
sep=''
while IFS= read -r version; do
  printf '%s"%s"' "$sep" "$version"
  sep=','
done <<<"$versions"
printf ']\n'
