#!/usr/bin/env bash
# Starts the LionClap HTTP service (src/LionClap.Service), building it first when it is not built.
# Run it from anywhere: it finds the repository from its own location. See docs/http-service.md.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run-service.sh [options]

Starts lionclap-service in the foreground (Ctrl+C or SIGTERM stops it). Every option has an environment variable;
an option wins over the variable.

  --model DIR        where the weights are: the Hugging Face directory of the Python container (hub/models--laion--clap-htsat-unfused/...)
                     or a directory with model.safetensors / pytorch_model.bin, plus tokenizer.json     CLAP_CKPT   (default: ~/models/clap)
  --port N           port, on all interfaces                          CLAP_PORT   (default: 8789)
  --device D         cpu, cuda or cuda:N                              CLAP_DEVICE (default: cpu)
  --token-file FILE  file whose first line is the bearer token        (or CLAP_TOKEN; no token leaves the API open)
  --version V        the version string /health reports               CLAP_VERSION (default: 1)
  --build            build again even if the service is already built
  -h, --help         this text

A cuda device builds the service with the CUDA backend (-p:TorchBackend=cuda-linux), any other with the CPU one; if the
last build used the other backend, it is built again. The first build takes a few minutes.
Exit codes: 0 stopped normally, 2 wrong usage or a setting that cannot be used, anything else: the service's own.
USAGE
}

die() { echo "error: $*" >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${CLAP_CKPT:-$HOME/models/clap}"
port="${CLAP_PORT:-8789}"
device="${CLAP_DEVICE:-cpu}"
version="${CLAP_VERSION:-1}"
token="${CLAP_TOKEN:-}"
rebuild=0

need_value() { [[ $# -ge 2 ]] || die "option $1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       need_value "$@"; model="$2"; shift 2 ;;
    --port)        need_value "$@"; port="$2"; shift 2 ;;
    --device)      need_value "$@"; device="$2"; shift 2 ;;
    --version)     need_value "$@"; version="$2"; shift 2 ;;
    --token-file)  need_value "$@"; [[ -r "$2" ]] || die "--token-file: cannot read '$2'"
                   IFS= read -r token < "$2" || true; shift 2 ;;
    --build)       rebuild=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || die "the port must be a number from 1 to 65535, got '$port'"
[[ "$device" =~ ^(cpu|cuda|cuda:[0-9]+)$ ]] || die "the device must be cpu, cuda or cuda:N, got '$device'"
command -v dotnet >/dev/null || die "dotnet is not installed (the .NET 8 runtime or SDK is needed)"

# The weights: warn (the service itself reports it too at /health). Formats and layouts are the ones of the service's locator.
found=0
[[ -f "$model/model.safetensors" || -f "$model/pytorch_model.bin" ]] && found=1
if (( ! found )); then
  for snapshot in "$model"/hub/models--laion--clap-htsat-unfused/snapshots/*/; do
    [[ -f "${snapshot}model.safetensors" || -f "${snapshot}pytorch_model.bin" ]] && found=1
  done
fi
if (( ! found )); then
  echo "warning: no CLAP weights found in $model (model.safetensors or pytorch_model.bin, flat or in hub/models--laion--clap-htsat-unfused/snapshots/*)" >&2
  echo "         the service starts, but /health says it is not ready." >&2
fi

# Build when needed, with the backend the device asks for.
backend=cpu; [[ "$device" == cuda* ]] && backend=cuda-linux
out="$root/src/LionClap.Service/bin/Release/net8.0"
dll="${LIONCLAP_SERVICE_DLL:-$out/lionclap-service.dll}"
marker="$out/.torch-backend"
if [[ -z "${LIONCLAP_SERVICE_DLL:-}" ]] && { (( rebuild )) || [[ ! -f "$dll" ]] || [[ "$(cat "$marker" 2>/dev/null)" != "$backend" ]]; }; then
  echo "building the service (backend: $backend) ..." >&2
  dotnet build "$root/src/LionClap.Service" -c Release -p:TorchBackend="$backend" -v quiet -nologo >&2
  echo "$backend" > "$marker"
fi
[[ -f "$dll" ]] || die "the service is not built: $dll does not exist"

export CLAP_CKPT="$model" CLAP_PORT="$port" CLAP_DEVICE="$device" CLAP_VERSION="$version"
[[ -n "$token" ]] && export CLAP_TOKEN="$token"

exec dotnet "$dll"
