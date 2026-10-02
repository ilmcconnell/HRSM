#!/usr/bin/env bash
# Build native macOS (Apple Silicon) LGAR and EnKF_LGAR_20_003002 binaries from the bundled source zips.
# The committed LGAR/LGAR and LGAR/EnKF_LGAR_* files are Linux x86-64 builds and cannot run on macOS.
set -euo pipefail

LGAR_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$LGAR_DIR/build"
OUT_DIR="$LGAR_DIR/macos-$(uname -m)"
CXX="${CXX:-clang++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2}"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$OUT_DIR"
unzip -q "$LGAR_DIR/LGARTOKF_original_signalSite.zip" -d "$BUILD_DIR"
unzip -q "$LGAR_DIR/LGARTOKF_EnKF_signalSite.zip" -d "$BUILD_DIR"

# The EnKF source was last built as the 100-member variant; the notebook uses 20 members
# (model error SDs 0.03 / 0.02 are already set in the source).
ENKF_SRC="$BUILD_DIR/LGARTOKF_EnKF_signalSite"
sed -i '' 's/int N_ensemble = 100;/int N_ensemble = 20;/' "$ENKF_SRC/bmi_main_lgar.cxx"
grep -q 'int N_ensemble = 20;' "$ENKF_SRC/bmi_main_lgar.cxx" || { echo "failed to set N_ensemble = 20" >&2; exit 1; }

build() {
  local src_dir="$1" out="$2"
  echo "building $out"
  (cd "$src_dir" && $CXX $CXXFLAGS \
    bmi_main_lgar.cxx \
    src/aet.cxx src/bmi_lgar.cxx src/lgar.cxx src/linked_list.cxx \
    src/mem_funcs.cxx src/soil_funcs.cxx src/util_funcs.cxx \
    giuh/giuh.cxx \
    -Iinclude -Ibmi -Igiuh -o "$out")
  chmod u+x "$out"
}

build "$BUILD_DIR/LGARTOKF_original_signalSite" "$OUT_DIR/LGAR"
build "$ENKF_SRC" "$OUT_DIR/EnKF_LGAR_20_003002"

file "$OUT_DIR"/*
