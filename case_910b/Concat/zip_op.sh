#!/bin/bash
set -eu

op_name=${1:?"usage: $0 <operator-name>"}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
op_dir="$script_dir/../${op_name}"
zip_root="$script_dir/${op_name}_zip"
archive="$script_dir/${op_name}.zip"

test -d "$op_dir/op_host"
test -d "$op_dir/op_kernel"
test -n "$(find "$op_dir/build_out" -maxdepth 1 -type f -name 'custom_*.run' -print -quit 2>/dev/null)"

rm -rf "$zip_root" "$archive"
mkdir -p "$zip_root"
cp -R "$op_dir/op_host" "$zip_root/"
cp -R "$op_dir/op_kernel" "$zip_root/"
find "$op_dir/build_out" -maxdepth 1 -type f -name 'custom_*.run' -exec cp {} "$zip_root/" \;

if command -v zip >/dev/null 2>&1; then
    (cd "$script_dir" && zip -qr "$archive" "${op_name}_zip")
else
    python3 - "$script_dir" "$op_name" <<'PY'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
name = sys.argv[2]
source = root / f"{name}_zip"
archive = root / f"{name}.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
    for path in source.rglob("*"):
        if path.is_file():
            output.write(path, path.relative_to(root))
PY
fi

printf 'Created %s\n' "$archive"
