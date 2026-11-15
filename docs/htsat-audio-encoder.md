# HTSAT audio encoder and audio embedding

`LionClap.Audio.Htsat.ClapAudioBranch` is the audio side of `ClapModel` (HF `laion/clap-htsat-unfused`, unfused, inference only): log-mel features in, the 768-d pooled encoder output and the 512-d L2-normalised audio embedding out. It reproduces transformers 4.47.1 `modeling_clap.py` (`ClapAudioEncoder` + `audio_projection`), not laion_clap. No Python at run time.

```csharp
using var branch = ClapAudioBranch.Load("/path/to/dir");           // dir: model.safetensors (+ config.json); CPU by default
using var features = new LogMelExtractor().ExtractBatch(waves);     // [B, 1, 1001, 64], see audio-frontend.md
using var result = branch.Forward(features);                        // result.Pooled [B, 768], result.Embeddings [B, 512]
```

`Forward` runs under `no_grad` in eval mode; the caller disposes the result. The branch accepts `[B, 1, T, 64]` with `1 ≤ T ≤ 1024` (shorter inputs are stretched to 1024 like the reference); anything else throws `ArgumentException` naming the problem. A configuration with `enable_fusion` is rejected when `config.json` is read.

## Pipeline

1. **Image**: batch norm over the 64 mel bands (running statistics), time stretched to 1024 with bicubic interpolation (`align_corners=True`), then the four 256-frame slices are stacked along frequency → `[B, 1, 256, 256]`.
2. **Patch embedding**: 4×4 convolution, stride 4, LayerNorm → `[B, 4096, 96]`.
3. **Four Swin stages** (depths 2/2/6/2, dims 96/192/384/768, heads 4/8/16/32, window 8): pre-norm window attention with the relative position bias, odd blocks shifted by 4 with the −100 mask (no shift in the last stage, whose grid is a single window), pre-norm MLP with exact GELU; patch merging (concatenate the four 2×2 sub-grids, LayerNorm, Linear 4C→2C) after the first three stages.
4. **Final LayerNorm and mean over the 64 tokens** → pooled `[B, 768]`.
5. **Projection** `768→512`, ReLU, `512→512`, then L2 normalisation → `[B, 512]`.

Module and field names equal the checkpoint keys (`audio_model.audio_encoder.layers.N.blocks.M.attention.self.query.weight`, …), so `StrictLoader` loads without a mapping. Loading checks keys and shapes from the headers alone and throws one `InvalidDataException` listing every missing key, extra `audio_*` key and shape mismatch before touching any weight; the 207 text and `logit_scale_*` tensors are not read. Buffers (batch-norm statistics, relative position indices) are loaded with the weights.

## Checkpoint for the tests

Tests that need weights are marked `[RequiresCheckpoint]` and read `CLAP_CKPT`: a directory with `model.safetensors` and `config.json`, written by the golden run (`tools/golden_all.sh`, or `tools/run_golden.sh checkpoint`), never inside the repo. Without it they are reported as skipped with the reason.

```bash
CLAP_CKPT=~/Proyectos/mediaOrquestator/models/music/clap-safetensors dotnet test LionClap.Net.sln -p:TorchBackend=cpu
```

## Parity (CPU fp32, against the golden run in `tests/fixtures/`)

| check | limit | measured |
|---|---|---|
| pooled `[5, 768]` vs `audio_encoder_output` (max_abs / cos_sim) | ≤ 1e-4 / ≥ 0.9999 | ≤ 2.8e-6 / 1.000000 |
| embeddings `[5, 512]` vs `audio_embeddings` | ≤ 1e-4 / ≥ 0.9999 | ≤ 2.7e-7 / 1.000000 |
| waveform → `LogMelExtractor` → embedding | ≤ 1e-4 / ≥ 0.9999 | ≤ 2.5e-7 / 1.000000 |
| embedding norm | within 1e-6 of 1 | 6e-8 |
| item alone vs inside the batch | ≤ 1e-4 | ≤ 3.6e-7 |
| intermediates, each stage fed the reference activation before it | ≤ 1e-4 · max(1, peak) | ≤ 7.6e-5 absolute (stage2, peak 94) |
| intermediates, whole chain | ≤ 1e-4 · max(1, peak) | ≤ 1.3e-5 of the peak (stage2: 1.2e-3 absolute, peak 94; stage3: 1.6e-3, peak 141) |

The intermediate limit is relative to the reference peak because the late stages have outliers of ~100 and ~140 (mean |x| 0.4 and 0.9), where one fp32 ulp is already 1e-5 and six blocks accumulate to about a hundred of them; the pooled output and the embeddings, which are what CLAP uses, keep the absolute 1e-4 and sit three orders of magnitude inside it. The torch behind the reference is 2.7.0 and TorchSharp-cpu 0.107.0 bundles libtorch 2.10.0.

## Not here

Fusion path (`enable_fusion`, `mel_conv2d`, `ClapAudioAFFBlock`), padding paths for grids that are not multiples of the window (not reachable in HTSAT-tiny), attentions/hidden-states outputs, dropout and drop-path, GPU or half-precision validation, the text branch and the service.
