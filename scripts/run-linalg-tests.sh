#!/usr/bin/env bash
#
# Build + run the standalone statsduck::linalg C++ unit tests (Epic 0.1).
#
# linalg is deliberately DuckDB-free, so it's tested directly — no duckdb, no
# CMake, no extension build. Compiles test/cpp/test_linalg.cpp + src/linalg.cpp
# (which #includes Eigen) into a single binary and runs it.
#
# Needs: a C++17 compiler ($CXX, default g++) and the third_party/eigen submodule
# (git submodule update --init --depth 1 third_party/eigen).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-g++}"
OUT="$ROOT/build/test_linalg"

if [[ ! -f "$ROOT/third_party/eigen/Eigen/Dense" ]]; then
  echo "ERROR: third_party/eigen not checked out. Run:" >&2
  echo "  git submodule update --init --depth 1 third_party/eigen" >&2
  exit 2
fi

mkdir -p "$ROOT/build"
echo "Compiling linalg tests with $CXX ..."
"$CXX" -std=c++17 -O2 -Wall \
  -I "$ROOT/src/include" \
  -I "$ROOT/third_party/eigen" \
  "$ROOT/test/cpp/test_linalg.cpp" \
  "$ROOT/src/linalg.cpp" \
  -o "$OUT"

"$OUT"
