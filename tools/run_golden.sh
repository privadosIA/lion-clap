#!/usr/bin/env bash
# Runs tools/dump_reference.py <subcommand> inside the reference image, offline, CPU.
# Usage: tools/run_golden.sh <checkpoint|frontend|htsat|text|roberta|classify|similarity|noop> [args...]
# Env:   CLAP_WEIGHTS  host dir with hub/models--laion--clap-htsat-unfused (default ~/Proyectos/mediaOrquestator/models/music/clap)
#        CLAP_CKPT_OUT host dir where model.safetensors is written, outside the repo (default ~/Proyectos/mediaOrquestator/models/music/clap-safetensors)
#        FIXTURES_DIR  output dir of the fixtures inside the repo (default tests/fixtures)
set -euo pipefail

IMAGE="${CLAP_IMAGE:-mediaorquestator/clap:1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="${CLAP_WEIGHTS:-$HOME/Proyectos/mediaOrquestator/models/music/clap}"
CKPT_OUT="${CLAP_CKPT_OUT:-$HOME/Proyectos/mediaOrquestator/models/music/clap-safetensors}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <checkpoint|frontend|htsat|text|roberta|classify|similarity|noop> [args...]" >&2
  exit 2
fi
if [[ ! -d "$WEIGHTS" ]]; then
  echo "error: weights directory not found: $WEIGHTS (set CLAP_WEIGHTS)" >&2
  exit 3
fi
case "$CKPT_OUT/" in
  "$REPO"/*) echo "error: CLAP_CKPT_OUT must be outside the repository: $CKPT_OUT" >&2; exit 2 ;;
esac
mkdir -p "$CKPT_OUT"

exec docker run --rm --network none \
  -v "$WEIGHTS":/ckpt:ro \
  -v "$REPO":/work \
  -v "$CKPT_OUT":/out \
  -e TORCHDYNAMO_DISABLE=1 -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 \
  --user "$(id -u):$(id -g)" \
  --entrypoint python "$IMAGE" -I /work/tools/dump_reference.py "$@"
