#!/usr/bin/env bash
#
# Build + run the standalone statsduck C++ unit tests.
#
# These exercise the DuckDB-free numeric layers directly — no duckdb, no CMake,
# no extension build:
#   - linalg kernel        (Epic 0.1): test_linalg.cpp + linalg.cpp
#   - lm_fit regression core (Epic 1.1): test_lm_fit.cpp + lm_core.cpp + linalg.cpp
# Both link Eigen (via linalg.cpp); lm_core also pulls in the header-only
# distributions.hpp. Each suite returns 0 on all-pass.
#
# Needs: a C++17 compiler ($CXX, default g++) and the third_party/eigen submodule
# (git submodule update --init --depth 1 third_party/eigen).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -O2 -Wall -I "$ROOT/src/include" -I "$ROOT/third_party/eigen")

if [[ ! -f "$ROOT/third_party/eigen/Eigen/Dense" ]]; then
  echo "ERROR: third_party/eigen not checked out. Run:" >&2
  echo "  git submodule update --init --depth 1 third_party/eigen" >&2
  exit 2
fi

mkdir -p "$ROOT/build"
rc=0

build_and_run() {
  local name="$1"; shift
  local out="$ROOT/build/$name"
  echo "Compiling $name with $CXX ..."
  if ! "$CXX" "${CXXFLAGS[@]}" "$@" -o "$out"; then
    echo "  COMPILE FAILED: $name" >&2
    rc=1
    return
  fi
  if ! "$out"; then
    echo "  TESTS FAILED: $name" >&2
    rc=1
  fi
}

build_and_run test_linalg \
  "$ROOT/test/cpp/test_linalg.cpp" \
  "$ROOT/src/linalg.cpp"

build_and_run test_lm_fit \
  "$ROOT/test/cpp/test_lm_fit.cpp" \
  "$ROOT/src/lm_core.cpp" \
  "$ROOT/src/linalg.cpp"

if [[ $rc -ne 0 ]]; then
  echo "C++ unit tests FAILED." >&2
else
  echo "All C++ unit tests passed."
fi
exit $rc
