#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=${BUILD_DIR:-"$project_dir/build"}
: "${ASCENDC_CANN_PACKAGE_PATH:?Set ASCENDC_CANN_PACKAGE_PATH to the CANN toolkit root}"
compiler=${ASCENDC_CXX_COMPILER:-}
if [ -z "$compiler" ] && [ -x "$ASCENDC_CANN_PACKAGE_PATH/compiler/bin/bisheng" ]; then
  compiler="$ASCENDC_CANN_PACKAGE_PATH/compiler/bin/bisheng"
fi
cmake_args="-DASCENDC_CANN_PACKAGE_PATH=$ASCENDC_CANN_PACKAGE_PATH"
if [ -n "$compiler" ]; then
  cmake_args="$cmake_args -DCMAKE_CXX_COMPILER=$compiler"
fi
cmake -S "$project_dir" -B "$build_dir" $cmake_args
cmake --build "$build_dir" --target greater_ascendc -j"${JOBS:-1}"
run_file=$(find "$build_dir" -type f -name 'custom_*.run' -print -quit)
if [ -z "$run_file" ]; then
  echo "[ERROR] Ascend C build did not produce custom_*.run under $build_dir" >&2
  exit 1
fi
mkdir -p "$project_dir/build_out"
find "$project_dir/build_out" -maxdepth 1 -type f -name 'custom_*.run' -delete
cp "$run_file" "$project_dir/build_out/"
printf 'Generated %s\n' "$project_dir/build_out/$(basename "$run_file")"
