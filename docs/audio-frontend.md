# Audio frontend

`LionClap.Audio.LogMelExtractor` turns a mono 48 kHz waveform into the log-mel `input_features` that the CLAP audio encoder (HF `laion/clap-htsat-unfused`) consumes. It reproduces transformers 4.47.1 `ClapFeatureExtractor` with `truncation=rand_trunc`, `padding=repeatpad` (not laion_clap/torchlibrosa). No Python, no file read at run time.

```csharp
var extractor = new LogMelExtractor();                    // build once, thread-safe
float[] f = extractor.Extract(samples);                   // [1001 × 64] row-major, float32
float[] g = extractor.Extract(twelveSeconds, offset: 48000);   // cut: the 10 s that start at the offset
Tensor t = extractor.ExtractBatch(new[] { a, b, c });     // [B, 1, 1001, 64] float32, CPU
```

## Pipeline

1. **Window** (480000 samples = 10 s). Longer input: the 480000 samples from `offset` (default 0; an offset that does not fit, or a non-zero one on input that is not longer, throws `ArgumentOutOfRangeException`). Shorter input: repeated `⌊480000/len⌋` whole times, then zero-padded on the right (`repeatpad`). Empty input throws `ArgumentException`. There is no random crop: the caller chooses the offset.
2. **STFT** in double precision: periodic Hann window of 1024, hop 480, centered frames over the signal reflect-padded by 512 on both sides → 1001 frames, bins 0..512, power = re² + im². Iterative radix-2 FFT (`Audio/Fft.cs`, `Audio/PowerStft.cs`).
3. **Mel**: Slaney bank `[513, 64]` (50–14000 Hz, Slaney scale and area normalisation) computed in `Audio/MelFilterBank.cs` with the formulas of transformers' `mel_filter_bank`.
4. **dB**: `10·log10(max(x, 1e-10))`, so silence is exactly -100. Rounded to float32 once, at the end.

## Parity (against `tests/fixtures/frontend.safetensors`)

| check | limit | measured |
|---|---|---|
| `input_features`, 5 cases (max_abs / cos_sim) | ≤ 1e-4 / ≥ 0.9999 | ≤ 7.6e-6 / 1.000000 |
| mel bank vs `mel_filters_slaney` | ≤ 1e-6 | 9.0e-10 |
| FFT / STFT frames vs naive DFT (relative) | ≤ 1e-9 | ~1e-15 |

The measured max_abs is one float32 ulp at 64–128 dB: rounding, not a modelling difference. The HTK bank (`mel_filters_htk`) differs by 1.0, so the wrong scale cannot pass. Run: `dotnet test LionClap.Net.sln -p:TorchBackend=cpu --filter "FullyQualifiedName~LogMelExtractorTests|FullyQualifiedName~MelFilterBankTests|FullyQualifiedName~StftTests"`.

## Not here

WAV decoding, resampling to 48 kHz and the 3×10 s windowing of `server.py` belong to the service layer; the fusion path (`enable_fusion`, 4-channel mel) is not ported.
