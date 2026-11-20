# LionClap.Net

C# port (.NET 8 + [TorchSharp](https://github.com/dotnet/TorchSharp)) of [LAION-CLAP](https://github.com/LAION-AI/CLAP), inference only. It embeds audio and text into the same 512-dimensional space, so you can score how well an audio clip matches a list of text tags ("piano", "orchestral", "a recording of ambient music"…). It generates nothing.

It uses the Hugging Face weights [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused) (HTSAT-tiny audio encoder + RoBERTa-base text encoder, two projection MLPs, L2 normalisation), following the `transformers` `ClapModel`. **No Python at runtime**: the library, the service, the tests and the build are C# and shell only.

## What is in it

| part | where | what |
|---|---|---|
| Audio frontend | `src/LionClap.Net/Audio/` | WAV decoder, linear resampler to 48 kHz, STFT + log-mel (Slaney filters), three 10 s windows |
| HTSAT audio branch | `src/LionClap.Net/Audio/Htsat/` | Swin-style audio transformer with the audio projection |
| RoBERTa text branch | `src/LionClap.Net/Text/` | text encoder, projection and tokenizer ([Tokenizers.DotNet](https://github.com/ThinkTankAI/Tokenizers.DotNet)) |
| Weights | `src/LionClap.Net/Weights/` | reads `model.safetensors` or the Hugging Face `pytorch_model.bin` directly, no conversion step |
| Classification | `src/LionClap.Net/Classification/` | mean of the window embeddings against the tag embeddings, per category |
| HTTP service | `src/LionClap.Service/` | `GET /health`, `POST /classify`, Swagger UI at `/docs`; drop-in for the Python `/opt/server.py` of `mediaorquestator/clap:1` |

## Parity with the reference

Every stage is tested against fixtures produced by the Python reference (the Docker image `mediaorquestator/clap:1`, run once, offline). Criterion: fp32 `max_abs ≤ 1e-4` and `cos_sim ≥ 0.9999`. Measured on CPU:

| stage | max_abs | cos_sim |
|---|---|---|
| log-mel `input_features` | ≤ 7.6e-6 | 1.000000 |
| audio encoder (pooled `[5, 768]`) | ≤ 2.8e-6 | 1.000000 |
| text encoder (pooled `[5, 768]`) | ≤ 1.3e-6 | 1.000000 |

On the golden requests the service answers with the same tags, the same order and the same scores as the Python one. Details in [`docs/`](docs).

## Quick start

You need the .NET 8 SDK (or newer) and the weights directory (the Hugging Face cache holding `pytorch_model.bin`, `config.json` and `tokenizer.json`, or a flat directory with `model.safetensors` and `tokenizer.json`). The weights are never in this repository and nothing is downloaded.

```bash
# build (TorchBackend: cpu | cuda-linux | cuda-windows)
dotnet build LionClap.Net.sln -p:TorchBackend=cpu

# start the service
scripts/run-service.sh --model /path/to/clap --port 8789
```

Then open `http://localhost:8789/docs`, or:

```bash
WAV=$(base64 -w0 song.wav)
curl -s -X POST localhost:8789/classify -H 'Content-Type: application/json' -d "{
  \"wav_base64\": \"$WAV\",
  \"labels\": {\"instrument\": [\"piano\", \"guitar\", \"drums\", \"violin\"],
               \"style\": [\"ambient\", \"dance\", \"orchestral\", \"acoustic\"]},
  \"template\": \"A recording of {} music.\", \"top\": 2 }"
# {"seconds":3.0,"categories":{"instrument":[{"tag":"synthesizer","score":0.12},…],"style":[…]}}
```

Scores are cosine similarities, not probabilities: use them to rank tags of one clip, not as absolute confidence. Full reference (variables, token, systemd unit, status codes, differences from the Python service) in [`docs/http-service.md`](docs/http-service.md).

## Tests

```bash
dotnet test LionClap.Net.sln -p:TorchBackend=cpu
```

Tests that need the weights are skipped, with the reason, unless `CLAP_CKPT` and `CLAP_HF_DIR` are set; `CLAP_DEVICE=cuda` (plus a CUDA build) enables the opt-in GPU parity test. See [`CLAUDE.md`](CLAUDE.md) for the exact variables and commands.

## Repository layout

```
src/LionClap.Net       library (namespace LionClap)
src/LionClap.Service   ASP.NET Core service, openapi.yaml and Swagger UI embedded (works offline)
tests/                 xUnit tests and the golden fixtures (checked with tools/verify_manifest.sh)
tools/                 golden run: regenerates the fixtures inside the reference container
scripts/               service start script and systemd unit
docs/                  one document per component, and the DocFX site configuration
openspec/              specs and change history (spec-driven workflow)
```

The shared code lives in the sibling repository `musicgen-core` (`~/Proyectos/musicgen-core`, or `-p:MusicGenCoreDir=...`); it is not part of the solution.

## Documentation

- [`docs/audio-frontend.md`](docs/audio-frontend.md)
- [`docs/htsat-audio-encoder.md`](docs/htsat-audio-encoder.md)
- [`docs/roberta-text-encoder.md`](docs/roberta-text-encoder.md)
- [`docs/http-service.md`](docs/http-service.md)

These articles and the API reference of the library, generated from its XML comments (`///`), are built as a static site with [DocFX](https://dotnet.github.io/docfx/) (pinned as a local tool in `.config/dotnet-tools.json`):

```bash
dotnet tool restore
TorchBackend=cpu dotnet docfx docs/docfx.json --serve   # builds docs/_site and serves it on http://localhost:8080
```

`docs/api` and `docs/_site` are generated and ignored by git. The home page is `docs/index.md`. The CI builds the site on every push with `--warningsAsErrors`, so a broken `<see cref>` or a broken link between articles fails that step; the site is not published anywhere.

## Not ported

PANN, training, evaluation and loss code, the `enable_fusion` variant, the BERT/BART text encoders, and the `630k-audioset-best.pt` checkpoint of the `laion_clap` repository.

## GPU

The CPU path is the verified one. `--device cuda` runs the same code with a CUDA build (`-p:TorchBackend=cuda-linux`); it is only covered by the opt-in parity test.

## Related repositories

Depends on `musicgen-core` (shared code) and follows the original LAION-CLAP. Used by `mediaOrquestator`, which starts it as a service.

| Repository | What it is |
|---|---|
| [`mediaOrquestator`](https://github.com/privadosIA/mediaOrquestator) | The server: the HTTP API and its contract, the pipelines (video, images, music), the run history and the music services it starts. |
| [`heartmula`](https://github.com/privadosIA/heartmula) | C# (.NET 8 + TorchSharp) port of [heartlib](https://github.com/HeartMuLa/heartlib): the HeartMuLa song generator, run by the server as a service. |
| [`DiffRhythm2`](https://github.com/privadosIA/DiffRhythm2) | C# port of [DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2): a diffusion model that writes a whole song. |
| [`lion-clap`](https://github.com/privadosIA/lion-clap) (this repository) | C# port of [LAION-CLAP](https://github.com/LAION-AI/CLAP): audio and text embeddings, used to recognise songs. |
| [`musicgen-core`](https://github.com/privadosIA/musicgen-core) | The TorchSharp building blocks shared by the three ports. |
| [`forms-lib`](https://github.com/privadosIA/forms-lib) | `@core/dynamic-forms`, the Angular library that draws forms and tables from JSON. |

The links are to the public copies of these repositories. The third-party tools, models and libraries they build on, with their licenses, are listed in [THIRD_PARTY.md](https://github.com/privadosIA/mediaOrquestator/blob/main/THIRD_PARTY.md).

## License

Apache 2.0 (see [LICENSE](LICENSE)). This is a port of work by other people; see the origin and licences of the original and of the weights in the sections above and, for the whole set, in the mediaOrquestator `dotnet/THIRD_PARTY.md`.

This licence covers the code of this repository. The model weights (`laion/clap-htsat-unfused`) are downloaded from their publisher and are under their own licence: see the model card.

## Maintenance

Maintained by one person in their free time: issues and pull requests are welcome, with no promised response time. See [CONTRIBUTING.md](CONTRIBUTING.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
