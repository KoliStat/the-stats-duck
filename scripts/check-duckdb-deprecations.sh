#!/usr/bin/env bash
#
# List the calls in src/ that a DuckDB header tree marks as deprecated.
#
# DuckDB 2.0 marks some calls as deprecated. GCC prints a warning for each
# use. MSVC builds do not show these warnings, so on Windows nobody sees
# them. DuckDB's CMakeLists.txt removes the /W level from the MSVC flags (line
# 1091 at v2.0.0, line 641 at v1.4.3). MSVC then compiles at its default
# level 1. Its deprecation warning, C4996, is a level 3 warning.
#
# This script compiles every src/*.cpp with -fsyntax-only and
# -Wdeprecated-declarations, and collects the warnings located in src/.
# Warnings located in DuckDB's own headers are not counted.
#
# The script needs no DuckDB build. It also works as a compile check for one
# DuckDB line: a source file that does not compile against the headers makes
# it fail.
#
# Usage:
#   bash scripts/check-duckdb-deprecations.sh <duckdb-source-dir>
#
# <duckdb-source-dir> holds src/include/duckdb.hpp. A full DuckDB source tree
# qualifies, and so does a directory with only the headers.
#
# Environment:
#   CXX      C++ compiler, default g++
#   CXX_STD  language standard, default c++17 for DuckDB 2.0 headers and
#            c++11 for older ones
#   JOBS     parallel compiles, default the output of nproc, or 4
#
# Needs: a C++ compiler that knows -fsyntax-only, the Eigen and ReadStat
# submodules under third_party/, and a DuckDB header tree.
#
# Exit codes: 0 no deprecated call, 1 deprecated calls or compile errors,
# 2 bad usage.

set -uo pipefail

usage() {
  echo "usage: bash scripts/check-duckdb-deprecations.sh <duckdb-source-dir>" >&2
  echo "  <duckdb-source-dir> must contain src/include/duckdb.hpp" >&2
}

if [[ $# -ne 1 || ! -f "$1/src/include/duckdb.hpp" ]]; then
  usage
  exit 2
fi

# Resolve the argument before the script changes directory.
duck_dir="$(cd "$1" && pwd)"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

CXX="${CXX:-g++}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# A native Windows compiler does not understand /c/... paths.
inc_dir="$duck_dir/src/include"
if command -v cygpath >/dev/null 2>&1; then
  inc_dir="$(cygpath -m "$inc_dir")"
fi

# The 2.0 headers need C++17. duckdb_compat.hpp tests for the same header.
if [[ -n "${CXX_STD:-}" ]]; then
  STD="$CXX_STD"
elif [[ -f "$duck_dir/src/include/duckdb/common/vector/list_vector.hpp" ]]; then
  STD="c++17"
else
  STD="c++11"
fi

# CMakeLists.txt adds win_iconv on Windows.
win_inc=""
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) win_inc="third_party/win_iconv" ;;
esac

tmp="$(mktemp -d)" || exit 2
trap 'rm -rf -- "${tmp:?}"' EXIT
mkdir "$tmp/out"

shopt -s nullglob
files=(src/*.cpp)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no src/*.cpp files found under $ROOT" >&2
  exit 2
fi

export CXX STD inc_dir win_inc tmp
# The compiler needs this too, not only sort. In a UTF-8 locale GCC quotes a
# name with the typographic quotes U+2018 and U+2019. The sed pattern below
# looks for the ASCII apostrophe.
export LC_ALL=C

compile_one() {
  local f="$1" name flags
  name="$(basename "$f")"
  flags=(-std="$STD" -fsyntax-only -Wdeprecated-declarations
    -I "$inc_dir" -I src/include -I third_party/eigen
    -I third_party/readstat/src -I third_party/readstat/src/sas
    -I third_party/readstat/src/spss -I third_party/readstat/src/stata)
  if [[ -n "$win_inc" ]]; then
    flags+=(-I "$win_inc")
  fi
  "$CXX" "${flags[@]}" "$f" >"$tmp/out/$name.log" 2>&1
  echo $? >"$tmp/out/$name.rc"
}
export -f compile_one

printf '%s\0' "${files[@]}" | xargs -0 -n 1 -P "$JOBS" bash -c 'compile_one "$1"' _

header="headers: $inc_dir, -std=$STD"
status=0

# Source files that do not compile.
failed=()
for f in "${files[@]}"; do
  name="$(basename "$f")"
  rc="$(cat "$tmp/out/$name.rc" 2>/dev/null || echo 1)"
  if [[ "$rc" != "0" ]]; then
    failed+=("$name")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAIL: ${#failed[@]} source files do not compile ($header)"
  for name in "${failed[@]}"; do
    echo
    echo "src/$name:"
    # A missing or crashed compiler leaves no error: line. Show the log then.
    if ! grep -E 'error:' "$tmp/out/$name.log" 2>/dev/null; then
      head -n 3 "$tmp/out/$name.log" 2>/dev/null || true
    fi
  done
  status=1
fi

# Deprecated calls. Each line becomes "<name><TAB><path>:<line>".
# sort -u counts a name at one location once, even when a template repeats it.
cat "$tmp"/out/*.log 2>/dev/null |
  sed -n -E "s/^(src\/[^:]+):([0-9]+):[0-9]+: warning: ('.*') is deprecated.*\$/\3	\1:\2/p" \
    >"$tmp/parsed.txt"
sort -u "$tmp/parsed.txt" >"$tmp/calls.txt"

# Every deprecation warning in src/ must be parsed. A change in the wording of
# the compiler's message would otherwise end in a silent pass.
cat "$tmp"/out/*.log 2>/dev/null |
  grep -E '^src/[^:]+:[0-9]+:[0-9]+: warning: .*\[-Wdeprecated-declarations\]' \
    >"$tmp/tagged.txt" || true
tagged="$(wc -l <"$tmp/tagged.txt" | tr -d ' ')"
parsed="$(sed -n -E "s/^(src\/[^:]+):([0-9]+):[0-9]+: warning: ('.*') is deprecated.*\$/\3/p" "$tmp/tagged.txt" | wc -l | tr -d ' ')"
if [[ "$tagged" -gt "$parsed" ]]; then
  if [[ $status -ne 0 ]]; then
    echo
  fi
  echo "FAIL: $((tagged - parsed)) deprecation warnings in src/ could not be parsed ($header)"
  # Show the tagged lines that the sed pattern does not parse.
  grep -v -E "^(src/[^:]+):([0-9]+):[0-9]+: warning: ('.*') is deprecated" "$tmp/tagged.txt" |
    head -n 5 | sed 's/^/  /'
  status=1
fi

if [[ -s "$tmp/calls.txt" ]]; then
  total="$(wc -l <"$tmp/calls.txt" | tr -d ' ')"
  if [[ $status -ne 0 ]]; then
    echo
  fi
  echo "FAIL: $total deprecated DuckDB call sites in src/ ($header)"
  # One block per name, with its count.
  cut -f1 "$tmp/calls.txt" | sort -u | while IFS= read -r name; do
    n="$(awk -F '\t' -v k="$name" '$1 == k' "$tmp/calls.txt" | wc -l | tr -d ' ')"
    echo
    echo "$name: $n"
    awk -F '\t' -v k="$name" '$1 == k { print "  " $2 }' "$tmp/calls.txt" | sort
  done
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "OK: ${#files[@]} source files, no deprecated DuckDB call ($header)"
fi
exit $status
