#!/usr/bin/env bash
# Full golden run from a clean tests/fixtures/ (MANIFEST.md is kept). Order matters: similarity reads the others.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rm -f tests/fixtures/*.safetensors tests/fixtures/*.json tests/fixtures/*.wav
for c in checkpoint frontend htsat text roberta classify similarity; do
  echo "== $c"
  tools/run_golden.sh "$c" 2> >(grep -v -E 'cache|0it' >&2)
done
