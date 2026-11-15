# RoBERTa text encoder and text embedding

`LionClap.Text.ClapTextBranch` is the text side of `ClapModel` (HF `laion/clap-htsat-unfused`, unfused checkpoint, inference only): token ids in, the 768-d pooled encoder output and the 512-d L2-normalised text embedding out. It reproduces transformers 4.47.1 `modeling_clap.py` (`ClapTextModel` + `text_projection`). No Python at run time.

```csharp
using var tokenizer = new ClapTextTokenizer("/path/to/dir/tokenizer.json");
using var branch = ClapTextBranch.Load("/path/to/dir");        // dir: model.safetensors (+ config.json); CPU by default

using var result = branch.Forward(tokenizer.EncodeBatch(texts));          // pads to the longest text, never truncates (server.py)
using var fixedLength = branch.Forward(tokenizer.EncodeBatch(texts, 77)); // pads and cuts to 77, keeping the end token (golden fixtures)
// result.Pooled [B, 768], result.Embeddings [B, 512]; the caller disposes the result
```

`Forward(ids, mask)` takes int64 tensors `[B, L]` with `1 ≤ L ≤ 512`. It rejects, with a message that names the value, ids that are not 2-d or not int64, a mask of another shape or with values other than 0/1, an empty batch, a length of 0 or above 512, an id outside the vocabulary, and a row whose mask is all zero. A configuration with a non-absolute position embedding or `is_decoder` is rejected when `config.json` is read.

## Tokenizer

`ClapTextTokenizer` loads the checkpoint's `tokenizer.json` (RoBERTa byte-level BPE with the start/end post-processor) through `Tokenizers.DotNet` 1.4.1, the Rust library behind Python's `tokenizers`, so the ids equal the reference ones including `<s>` (0) and `</s>` (2). The package and its per-platform native runtime come from `Directory.Build.props` (selected from the OS and architecture; `-p:TokenizerRuntime=<rid>` forces one). Padding and truncation are done in C#:

| call | behavior |
|---|---|
| `EncodeBatch(texts)` | pad with `<pad>` (1) to the longest text; **no truncation** (as `server.py`: `padding=True`); throws above 512 tokens |
| `EncodeBatch(texts, maxLength)` | pad every row to `maxLength` and cut longer ones, keeping the end token (as the fixtures: `max_length=77, truncation`) |

For texts of at most 77 tokens the two modes give the same embedding (the attention mask removes the padding); they differ for longer texts. Which one the service uses by default is decided in the service change.

## Pipeline

1. **Embeddings**: word + token-type (all zeros) + position embeddings, LayerNorm with **eps 1e-12**. Positions are not `0..L−1`: `cumsum(ids != pad) · (ids != pad) + pad`, so the first token has position 2 and padding keeps position 1.
2. **12 post-norm layers**: self-attention (`q·kᵀ/8 + mask`, softmax, `·v`) with the additive mask (0 where attended, `float32.min` where padded), then `LayerNorm(dense(·) + residual)`; feed-forward `768→3072`, exact GELU, `3072→768`, `LayerNorm(· + residual)`.
3. **Pooler**: first token, `dense`, tanh → pooled `[B, 768]`.
4. **Projection** `768→512`, ReLU, `512→512` (the same `ClapProjectionLayer` as the audio branch), then L2 normalisation → `[B, 512]`.

Module and field names equal the checkpoint keys (`text_model.encoder.layer.N.attention.self.query.weight`, `…output.LayerNorm.weight`, …), so `StrictLoader` loads without a mapping. Loading checks keys and shapes from the headers alone and throws one `InvalidDataException` listing every missing key, extra `text_*` key and shape mismatch before touching any weight; the 242 audio and `logit_scale_*` tensors are not read. The two int64 buffers (`position_ids`, `token_type_ids`) are loaded with the weights.

## Checkpoint for the tests

Gated tests (`[RequiresCheckpoint]`) read `CLAP_CKPT`: a directory with `model.safetensors`, `config.json` and `tokenizer.json`, written by the golden run (`tools/golden_all.sh`, or `tools/run_golden.sh checkpoint`), never inside the repo. Without it they are reported as skipped with the reason.

```bash
CLAP_CKPT=~/Proyectos/mediaOrquestator/models/music/clap-safetensors dotnet test LionClap.Net.sln -p:TorchBackend=cpu
```

## Parity (CPU fp32, against the golden run in `tests/fixtures/`)

| check | limit | measured |
|---|---|---|
| token ids and masks, 5 texts (length 77) | equal | equal |
| pooled `[5, 768]` vs `text_pooler_output` (max_abs / cos_sim) | ≤ 1e-4 / ≥ 0.9999 | ≤ 1.3e-6 / 1.000000 |
| embeddings `[5, 512]` vs `text_embeddings` | ≤ 1e-4 / ≥ 0.9999 | ≤ 1.7e-7 / 1.000000 |
| embedding norm | within 1e-6 of 1 | 6e-8 |
| same text unpadded / padded to 77 / inside a longer batch | ≤ 1e-4 | ≤ 1.3e-7 |
| item alone vs inside the batch (embeddings / pooled) | ≤ 1e-4 | ≤ 1.9e-7 / ≤ 2.0e-6 |
| intermediates, embeddings + 12 layers + pooler (whole chain) | ≤ 1e-4 · max(1, peak) | ≤ 8.6e-6 absolute (≤ 4.6e-7 of the peak) |
| audio embeddings (from waveforms) · text embeddings (from strings) vs `similarity` | ≤ 1e-4 | 3.1e-7 |

The torch behind the reference is 2.7.0 and TorchSharp-cpu 0.107.0 bundles libtorch 2.10.0; the layers have no outliers like the late audio stages, so the intermediates would pass even with the absolute 1e-4.

## Not here

Relative position embeddings, decoder/cache and cross-attention paths, head masks and attention outputs, dropout, GPU or half-precision validation, the use of `logit_scale_*`, prompt templates and the service.
