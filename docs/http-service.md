# The HTTP service

`lionclap-service` (project `src/LionClap.Service`) scores an audio clip against text tags with CLAP. It is the same interface as the Python service `/opt/server.py` of the image `mediaorquestator/clap:1`, so the orchestrator (mediaOrquestator) uses it as an **external engine** (a host and a port) with no change on its side, and any other program can use it with `curl` or an HTTP client. Open `http://HOST:PORT/docs` for the interactive page (Swagger UI, bundled, works offline); the machine-readable description is `http://HOST:PORT/openapi.yaml`.

It speaks plain HTTP and is meant for localhost or a trusted network. Put a reverse proxy in front of it if it must be reached from elsewhere, and set a token.

## Start it

```bash
scripts/run-service.sh --model ~/Proyectos/mediaOrquestator/models/music/clap --port 8789
# lionclap service 1 listening on :8789 (weights: /home/me/.../clap)
```

or, without the script:

```bash
dotnet build src/LionClap.Service -c Release                 # add -p:TorchBackend=cuda-linux for a GPU build
CLAP_CKPT=~/Proyectos/mediaOrquestator/models/music/clap dotnet src/LionClap.Service/bin/Release/net8.0/lionclap-service.dll
```

| variable | default | meaning |
|---|---|---|
| `CLAP_CKPT` | `/ckpt` | where the weights are (see below) |
| `CLAP_TOKEN` | empty | bearer token; empty leaves the API open |
| `CLAP_PORT` | `8789` | port, on all interfaces |
| `CLAP_VERSION` | `1` | the version string `/health` reports |
| `CLAP_DEVICE` | `cpu` | `cpu`, `cuda` or `cuda:N` (a CUDA device needs a CUDA build) |

A wrong value stops the service at start with a message that names the variable (exit code 2).

### The weights

`CLAP_CKPT` is the directory the Python container mounts at `/ckpt`, unchanged: the Hugging Face cache `hub/models--laion--clap-htsat-unfused/snapshots/<revision>/` holding `pytorch_model.bin`, `config.json` and `tokenizer.json`. The service reads the PyTorch file itself (no Python, no conversion step). A flat directory also works, with `model.safetensors` (preferred when both formats are there) or `pytorch_model.bin`, plus `tokenizer.json` and, optionally, `config.json`. The weights are never inside the repository. Nothing is downloaded.

Tests that need them read `CLAP_HF_DIR` (the Hugging Face directory) and `CLAP_CKPT` (a directory made by the golden run, `tools/golden_all.sh`).

### The start script

`scripts/run-service.sh` starts it for you, from any directory, building it first when it is not built:

```bash
scripts/run-service.sh --model DIR --port 8789                                  # CPU
scripts/run-service.sh --model DIR --device cuda:0 --token-file ~/.config/lionclap/token
scripts/run-service.sh --help                                                   # every option and variable
```

Every option has the environment variable of the table above (the option wins). A `cuda` device builds the service with the CUDA backend and any other with the CPU one; if the last build used the other backend it builds again (`--build` forces it). `--token-file` reads the token from the first line of a file, so it does not appear in the process list. It warns when it finds no weights in `--model`. It runs in the foreground; Ctrl+C or SIGTERM stops it (exit code 0).

### As a systemd service

`scripts/lionclap-service.service` is a user unit that runs the script and restarts it on failure; its header lists the commands (copy it to `~/.config/systemd/user/`, put the settings in `~/.config/lionclap/service.env`, `systemctl --user enable --now lionclap-service`). Build the service once by hand first, since the first build takes minutes and systemd would give up waiting; `loginctl enable-linger` keeps it running without a login session.

## The calls

All bodies are JSON (UTF-8). With a token, add `-H "Authorization: Bearer $TOKEN"` to every call below.

### Is it ready: `GET /health`

```bash
curl -s localhost:8789/health
# {"ready":true,"version":"1","busy":false,"queued":0,"gpu":null,"missing":[]}
```

`ready` is true when the weights and `tokenizer.json` are found and the device works (a GPU is not needed, unlike the Python image). `missing` lists what is wrong: `models--laion--clap-htsat-unfused` when no weights are found, `tokenizer.json`, or why a CUDA device cannot be used. `busy` says whether a classification is running and `queued` how many wait. `gpu` is `null` on CPU; on CUDA it is `{"name", "free_mb", "total_mb"}` read from `nvidia-smi` when available. Health answers at once even while a classification runs, and it never loads the model.

### Score a clip: `POST /classify`

```bash
WAV=$(base64 -w0 song.wav)
curl -s -X POST localhost:8789/classify -H 'Content-Type: application/json' -d "{
  \"wav_base64\": \"$WAV\",
  \"labels\": {\"instrument\": [\"piano\", \"guitar\", \"drums\", \"violin\", \"synthesizer\", \"flute\"],
               \"style\": [\"ambient\", \"dance\", \"orchestral\", \"acoustic\"]},
  \"template\": \"A recording of {} music.\", \"top\": 2 }"
# {"seconds":3.0,"categories":{"instrument":[{"tag":"synthesizer","score":0.12},{"tag":"piano","score":0.106}],
#                              "style":[{"tag":"orchestral","score":0.117},{"tag":"dance","score":0.038}]}}
```

| field | meaning | limits | default |
|---|---|---|---|
| `wav_base64` | a WAV file, base64 (standard alphabet, padded, no line breaks): PCM 8/16/24/32 bit or float 32/64 bit, any rate, any channels | longer than 1 s; body up to 40,000,000 bytes | required |
| `labels` | `{category: [tag, ...]}`; the answer keeps the order of the categories | at least one category; at most 200 tags per category (the first 200 are used) | required |
| `template` | the sentence each tag is put into; `{}` is replaced by the tag | must contain `{}` | `This audio is a {} song.` |
| `top` | tags returned per category | clamped to 1–10; `0` means the default | `3` |

The audio is mixed to mono, resampled to 48 kHz by linear interpolation, and cut into up to three windows of 10 s starting at 0, 5 s and 10 s (each used only if longer than 1 s). The audio vector is the mean of the normalised embeddings of the windows. Each tag, put into the template, is embedded and normalised; the **score** is the dot product with the audio vector, rounded to three decimals, and the best `top` of each category come back in descending order. `seconds` is the duration at 48 kHz rounded to one decimal.

Scores are similarity values, not probabilities (measured on songs of known style the reference service did not tell genres apart reliably): treat the service as an experiment.

One classification runs at a time; the others wait in arrival order. The model is loaded on the first classification (about a second on a modern CPU for the 600 MB of weights, then about 0.15 s for a short clip).

### Status codes

| code | when | body |
|---|---|---|
| 200 | classified | `{seconds, categories}` |
| 400 | not JSON or not an object; no `labels`; a template without `{}`; `wav_base64` not base64 or not a readable WAV; audio of one second or less | `{"error": "…"}` |
| 401 | a token is configured and the call lacks it | `{"error": "unauthorized"}` |
| 404 | any other path or method | `{"error": "not found"}` |
| 413 | body larger than 40,000,000 bytes (by `Content-Length` or counted while reading) | `{"error": "the request is too large"}` |
| 500 | an unexpected failure (for example a corrupt weights file) | `{"error": "…"}` |
| 503 | the weights, tokenizer or device are not available | `{"error": "the weights are missing", "missing": […]}` |

`/docs`, `/docs/*` and `/openapi.yaml` never need the token.

## Differences from the Python service

The paths, bodies, status codes, defaults and limits are the same; for the golden requests the answers are identical (same tags, same order, same scores). The differences are where `server.py` fails or is lax:

- Audio of one second or less *after resampling* is a `400` (the Python service crashes with an empty list of windows when it lands exactly on one second).
- A body that is valid JSON but not an object is a `400` (it raises there and drops the connection); unexpected failures are a `500` with the message.
- `{}` is the only placeholder of the template (Python calls `str.format`, so other brace groups, or two `{}`, fail there).
- A tag that is not a string is accepted when it is a number or a boolean (turned to text like Python's `str`); other types are a `400`.
- `ready` does not require a GPU.
- Error *texts* are this service's own words; the statuses and the `{"error": …}` shape are the same.

## GPU

The CPU is the verified path: every number in `docs/audio-frontend.md`, `docs/htsat-audio-encoder.md` and `docs/roberta-text-encoder.md` was measured there. `--device cuda` runs the same code on the GPU with a CUDA build (`-p:TorchBackend=cuda-linux`); its numerics are only checked by the opt-in test (`CLAP_DEVICE=cuda dotnet test …`). Without a CUDA device or build, `/health` says `ready:false` and the reason in `missing`.
