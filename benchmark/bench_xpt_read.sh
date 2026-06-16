#!/usr/bin/env bash
# benchmark/bench_xpt_read.sh — XPT reader scan benchmark for stats_duck.
#
# Two modes:
#   * synthetic (default): generates XPT files of increasing row count with the
#     extension's own COPY ... TO '*.xpt' writer, then times a full-scan
#     count(*) through read_stat() for each. The ratio between consecutive sizes
#     exposes complexity — ~2x per doubling is linear, ~4x is quadratic.
#   * real (FILES= or DIR=): times count(*) over existing .xpt files you supply.
#     Row count is read from the query itself, so unknown-size files are fine.
#
# Timing uses the CLI's `.timer on` (per-statement), so process startup and the
# extension LOAD are excluded. After the first pass files sit in the OS page
# cache, so this measures warm-cache CPU + syscall cost — the algorithm, not disk.
#
# ── Using real datasets ──────────────────────────────────────────────────────
# Point this at files that live OUTSIDE the repo working tree, e.g.
#   DIR=../icpsr-data benchmark/bench_xpt_read.sh
# Do NOT copy access-controlled / licensed data (e.g. ICPSR) into the repo: even
# with benchmark/.data/ git-ignored, keeping such files outside the tree is the
# only way to make an accidental commit structurally impossible. Never commit or
# push them.
#
# Usage:
#   benchmark/bench_xpt_read.sh
#   ROWS="25000 50000 100000" REPS=3 benchmark/bench_xpt_read.sh
#   WITH_AGG=1 benchmark/bench_xpt_read.sh
#   FILES="/data/a.xpt /data/b.xpt" benchmark/bench_xpt_read.sh
#   DIR=../icpsr-data REPS=1 benchmark/bench_xpt_read.sh
#
# Env:
#   ROWS            synthetic mode: row counts (default "25000 50000 100000 200000")
#   FILES           real mode: space-separated existing .xpt files
#   DIR             real mode: directory to scan for *.xpt
#   REPS            timed reps per file; best (min) wall time is reported (default 1)
#   WITH_AGG        synthetic only: also time an aggregate touching every cell (default 0)
#   DUCKDB          duckdb CLI (auto-detected under build/ if unset)
#   STATS_DUCK_EXT  extension path (auto-detected under build/ if unset)
#   KEEP_DATA       synthetic only: keep generated files (default 0 — removed on exit)

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

ROWS="${ROWS:-25000 50000 100000 200000}"
REPS="${REPS:-1}"
WITH_AGG="${WITH_AGG:-0}"
FILES="${FILES:-}"
DIR="${DIR:-}"
DATA_DIR="benchmark/.data"

detect() { for c in "$@"; do [ -e "$c" ] && { echo "$c"; return 0; }; done; return 1; }

DUCKDB="${DUCKDB:-$(detect \
  build/release/Release/duckdb.exe build/release/duckdb \
  build/mingw_release/duckdb.exe build/debug/Debug/duckdb.exe || true)}"
STATS_DUCK_EXT="${STATS_DUCK_EXT:-$(detect \
  build/release/extension/stats_duck/stats_duck.duckdb_extension \
  build/mingw_release/extension/stats_duck/stats_duck.duckdb_extension || true)}"

[ -n "$DUCKDB" ] && [ -e "$DUCKDB" ] || { echo "error: no duckdb CLI found; set DUCKDB=" >&2; exit 1; }
[ -n "$STATS_DUCK_EXT" ] && [ -e "$STATS_DUCK_EXT" ] || { echo "error: no stats_duck extension found; set STATS_DUCK_EXT=" >&2; exit 1; }

echo "duckdb:    $DUCKDB"
echo "extension: $STATS_DUCK_EXT"
echo "reps:      $REPS"
echo

filesize() { stat -c %s "$1" 2>/dev/null || wc -c <"$1"; }

# Normalize an absolute path for the (possibly native-Windows) duckdb CLI. Under
# MSYS/Git-Bash the shell hands out /c/... paths the native CLI can't open;
# cygpath -m rewrites them to C:/... Relative paths and non-MSYS systems pass
# through untouched (cygpath is absent on Linux/macOS).
winpath() {
  case "$1" in
    /*) command -v cygpath >/dev/null 2>&1 && cygpath -m "$1" || echo "$1" ;;
    *)  echo "$1" ;;
  esac
}

# Run count(*) once via .timer/.mode list; echo "ROWS REAL_SECONDS".
# .mode list + .headers off makes the result a bare integer we can grep.
# tr -d '\r' strips the CLI's CRLF line endings so the integer match is robust.
# A read failure (unsupported/corrupt file) yields rows=ERR rather than a count.
measure() {
  local f="$1" out rows t wf
  wf=$(winpath "$f")
  out=$(printf "LOAD '%s';\n.headers off\n.mode list\n.timer on\nSELECT count(*) FROM read_stat('%s');\n" \
        "$STATS_DUCK_EXT" "$wf" | "$DUCKDB" -unsigned 2>&1 | tr -d '\r') || true
  rows=$(printf '%s\n' "$out" | grep -E '^[0-9]+$' | tail -1)
  t=$(printf '%s\n' "$out" | sed -n 's/.*Run Time (s): real \([0-9.]*\).*/\1/p' | tail -1)
  if [ -z "$rows" ] && printf '%s' "$out" | grep -qi "error"; then rows="ERR"; fi
  echo "${rows:-NA} ${t:-NA}"
}

# measure REPS times; echo "ROWS BEST_SECONDS".
best() {
  local f="$1" rows="" best="" r res rr tt
  for ((r=0; r<REPS; r++)); do
    res=$(measure "$f"); rr=${res% *}; tt=${res#* }
    rows="$rr"
    if [ "$tt" != "NA" ] && { [ -z "$best" ] || awk "BEGIN{exit !($tt<$best)}"; }; then best="$tt"; fi
  done
  echo "$rows ${best:-NA}"
}

us_per_row() { awk "BEGIN{ if(\"$1\"==\"NA\"||$1==0||\"$2\"==\"NA\"){print \"NA\"}else{printf \"%.2f\",$2*1e6/$1} }"; }

# ── real mode ────────────────────────────────────────────────────────────────
if [ -n "$FILES$DIR" ]; then
  list=()
  for f in $FILES; do list+=("$f"); done
  if [ -n "$DIR" ]; then
    shopt -s nullglob
    for f in "$DIR"/*.xpt "$DIR"/*.XPT; do list+=("$f"); done
    shopt -u nullglob
  fi
  [ ${#list[@]} -gt 0 ] || { echo "error: no .xpt files in FILES/DIR" >&2; exit 1; }

  printf "%-32s %-9s %-12s %-10s %-9s\n" "file" "MB" "rows" "count_s" "us/row"
  printf -- "-------------------------------- --------- ------------ ---------- ---------\n"
  for f in "${list[@]}"; do
    [ -e "$f" ] || { echo "skip (missing): $f" >&2; continue; }
    mb=$(awk "BEGIN{printf \"%.1f\", $(filesize "$f")/1048576}")
    res=$(best "$f"); rows=${res% *}; t=${res#* }
    printf "%-32s %-9s %-12s %-10s %-9s\n" "$(basename "$f")" "$mb" "$rows" "$t" "$(us_per_row "$rows" "$t")"
  done
  exit 0
fi

# ── synthetic mode ───────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"
cleanup() { [ "${KEEP_DATA:-0}" = "1" ] || rm -rf "$DATA_DIR"; }
trap cleanup EXIT

# 5-column schema: int->double, double, low-card double, varchar, date.
# Exercises numeric, string (UTF-8 sanitize + heap copy) and date (per-cell
# epoch detection) paths. Names <=8 chars / uppercase per XPT v5 rules.
GEN="SELECT i AS ID, (i*1.5)::DOUBLE AS X, (i%7)::DOUBLE AS GRP, \
('S'||(i%1000))::VARCHAR AS NM, (DATE '2000-01-01' + (i%3650)::INTEGER)::DATE AS DT \
FROM range"

echo "rows:      $ROWS   with_agg: $WITH_AGG"
echo
printf "%-10s %-9s %-10s %-11s %-8s\n" "rows" "file_MB" "count_s" "us/row" "ratio"
printf -- "---------- --------- ---------- ----------- --------\n"

prev=""
for n in $ROWS; do
  f="$DATA_DIR/n$n.xpt"
  "$DUCKDB" -unsigned -c "LOAD '$STATS_DUCK_EXT'; COPY ($GEN($n) t(i)) TO '$f';" >/dev/null
  mb=$(awk "BEGIN{printf \"%.1f\", $(filesize "$f")/1048576}")
  res=$(best "$f"); t=${res#* }
  ratio="-"
  [ -n "$prev" ] && [ "$t" != "NA" ] && ratio=$(awk "BEGIN{printf \"%.2fx\", $t/$prev}")
  printf "%-10s %-9s %-10s %-11s %-8s\n" "$n" "$mb" "$t" "$(us_per_row "$n" "$t")" "$ratio"
  prev="$t"
  if [ "$WITH_AGG" = "1" ]; then
    out=$(printf "LOAD '%s';\n.timer on\nSELECT sum(X), max(DT), sum(length(NM)) FROM read_stat('%s');\n" \
          "$STATS_DUCK_EXT" "$f" | "$DUCKDB" -unsigned 2>&1) || true
    at=$(printf '%s\n' "$out" | sed -n 's/.*Run Time (s): real \([0-9.]*\).*/\1/p' | tail -1)
    printf "%-10s %-9s %-10s %-11s %-8s\n" "  +agg" "" "${at:-NA}" "" ""
  fi
done
