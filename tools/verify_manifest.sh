#!/usr/bin/env bash
# Checks the files in tests/fixtures/ against the sha256 table of MANIFEST.md (both ways: no missing, no extra files).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../tests/fixtures"
# rows: | `file` | `sha256` |   ->   "sha256  file"
list="$(sed -n 's/^| `\([^`]*\)` | `\([0-9a-f]\{64\}\)` |$/\2  \1/p' MANIFEST.md)"
[[ -n "$list" ]] || { echo "error: no sha256 table in MANIFEST.md" >&2; exit 1; }
echo "$list" | sha256sum -c --quiet -
extra="$(comm -13 <(echo "$list" | awk '{print $2}' | LC_ALL=C sort) <(ls | grep -v '^MANIFEST.md$' | LC_ALL=C sort))"
[[ -z "$extra" ]] || { echo "error: files not in the manifest: $extra" >&2; exit 1; }
big="$(find . -type f -size +20M)"
[[ -z "$big" ]] || { echo "error: fixture larger than 20 MB: $big" >&2; exit 1; }
echo "manifest OK ($(echo "$list" | wc -l) files)"
