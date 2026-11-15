#!/usr/bin/env python3
"""Golden run for LionClap.Net: dumps the reference fixtures of laion/clap-htsat-unfused.

Runs inside mediaorquestator/clap:1 via tools/run_golden.sh (offline, CPU). See
openspec/changes/golden-reference-fixtures. Subcommands: checkpoint, frontend, htsat, text, roberta, classify, similarity.
"""

import argparse
import os
import shutil
import sys

SEED = 1234
CKPT = os.environ.get("CLAP_CKPT", "/ckpt")
OUT = os.environ.get("CLAP_OUT", "/out")
FIXTURES = os.environ.get("FIXTURES_DIR", "/work/tests/fixtures")
HUB = "models--laion--clap-htsat-unfused"


def setup():
    """Single-thread deterministic setup; prints the versions the manifest records."""
    import numpy
    import safetensors
    import torch
    import transformers

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)
    print(f"python {sys.version.split()[0]}")
    print(f"torch {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"numpy {numpy.__version__}")
    print(f"safetensors {safetensors.__version__}")


def require_weights():
    """Fails before writing anything if the snapshot is not mounted; names the missing path."""
    snaps = os.path.join(CKPT, "hub", HUB, "snapshots")
    if not os.path.isdir(snaps) or not os.listdir(snaps):
        sys.exit(f"error: weights not found: {snaps} (mount the clap directory at {CKPT})")


def cmd_noop(_args):
    pass


def snapshot_dir():
    snaps = os.path.join(CKPT, "hub", HUB, "snapshots")
    return os.path.join(snaps, sorted(os.listdir(snaps))[-1])


def cmd_checkpoint(_args):
    """pytorch_model.bin -> /out/model.safetensors (fp32, HF key names), then verifies it against the loaded state_dict."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import ClapModel

    if not os.path.isdir(OUT):
        sys.exit(f"error: output directory not found: {OUT} (mount it at {OUT})")
    model = ClapModel.from_pretrained(snapshot_dir()).eval()
    sd = {k: v.detach().to(torch.float32).contiguous().clone() for k, v in model.state_dict().items()}
    path = os.path.join(OUT, "model.safetensors")
    save_file(sd, path, metadata={"format": "pt"})

    back = load_file(path)
    if set(back) != set(sd):
        sys.exit(f"error: key mismatch: only in state_dict {sorted(set(sd) - set(back))[:5]}, only in file {sorted(set(back) - set(sd))[:5]}")
    for k, v in sd.items():
        if back[k].shape != v.shape or not torch.equal(back[k], v):
            sys.exit(f"error: tensor differs after round trip: {k}")
    print(f"checkpoint: {len(sd)} tensors, {sum(v.numel() for v in sd.values())} parameters -> {path}")
    print(f"checkpoint: snapshot {os.path.basename(snapshot_dir())}")
    for name in ("config.json", "tokenizer.json"):  # next to the weights: the C# side reads them from the same directory
        shutil.copyfile(os.path.join(snapshot_dir(), name), os.path.join(OUT, name))


SR = 48000
MAX_SAMPLES = 10 * SR


def make_waveforms():
    """The 5 deterministic inputs: (name, source float64 waveform, cut offset in samples). Seeded, no files."""
    import numpy as np

    rng = np.random.default_rng(SEED)

    def norm(x, peak=0.5):
        return x * (peak / np.max(np.abs(x)))

    t = np.arange(10 * SR) / SR
    tone = norm(sum(a * np.sin(2 * np.pi * 440.0 * h * t) for h, a in ((1, 1.0), (2, 0.5), (3, 0.25))))

    spec = np.fft.rfft(rng.standard_normal(12 * SR))  # 12 s source, cut to 10 s at a fixed offset
    f = np.fft.rfftfreq(12 * SR, 1 / SR)
    spec[(f < 200) | (f > 8000)] = 0
    noise = norm(np.fft.irfft(spec, 12 * SR))

    n = int(4.3 * SR)
    t = np.arange(n) / SR
    k = (8000.0 - 100.0) / 4.3
    chirp = norm(np.sin(2 * np.pi * (100.0 * t + 0.5 * k * t * t)))
    fade = int(0.1 * SR)
    chirp[:fade] *= np.linspace(0, 1, fade)
    chirp[-fade:] *= np.linspace(1, 0, fade)

    clicks = np.zeros(7 * SR)
    tc = np.arange(int(0.005 * SR)) / SR
    burst = np.exp(-tc / 0.001) * np.sin(2 * np.pi * 2000.0 * tc)
    for start in range(0, 7 * SR - len(burst), SR // 4):
        clicks[start : start + len(burst)] += burst
    clicks = norm(clicks)

    n = int(1.5 * SR)
    t = np.arange(n) / SR
    mix = norm(np.sin(2 * np.pi * 220.0 * t) + 0.3 * np.sin(2 * np.pi * 1320.0 * t) + 0.1 * rng.standard_normal(n))

    return [("tone", tone, 0), ("noise", noise, 24000), ("chirp", chirp, 0), ("clicks", clicks, 0), ("mix", mix, 0)]


def write_json(path, obj):
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def cmd_frontend(_args):
    """Waveforms, mel filter banks, input_features (/work/tests/fixtures/frontend*) and audio embeddings."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import ClapModel, ClapProcessor
    from transformers.audio_utils import spectrogram, window_function

    os.makedirs(FIXTURES, exist_ok=True)
    snap = snapshot_dir()
    model = ClapModel.from_pretrained(snap).eval()
    fe = ClapProcessor.from_pretrained(snap).feature_extractor

    cases = []
    waves = []
    for name, source, offset in make_waveforms():
        w = source[offset : offset + MAX_SAMPLES] if len(source) > MAX_SAMPLES else source
        w = w.astype(np.float32)  # what the C# side reads; the extractor casts it to float64 itself
        assert len(w) <= MAX_SAMPLES, name  # never the rand_trunc branch (it needs len > max_length)
        cases.append({"name": name, "samples": len(w), "source_samples": len(source), "cut_offset": offset})
        waves.append(w)
    assert any(len(w) < MAX_SAMPLES for w in waves) and any(len(w) == MAX_SAMPLES for w in waves)

    out = fe(waves, sampling_rate=SR, return_tensors="pt")
    feats = out["input_features"]
    print(f"frontend: input_features {tuple(feats.shape)} {feats.dtype}, is_longer {out['is_longer'].flatten().tolist()}")
    assert not out["is_longer"].any()
    feats = feats.to(torch.float32)

    # which bank reproduces input_features: recompute with the library's own spectrogram() for each bank
    def recompute(w, bank):
        x = w.astype(np.float64)
        if len(x) < MAX_SAMPLES:
            x = np.pad(np.tile(x, int(MAX_SAMPLES / len(x))), (0, MAX_SAMPLES - len(x) * int(MAX_SAMPLES / len(x))))
        return spectrogram(x, window_function(1024, "hann"), frame_length=1024, hop_length=480, power=2.0,
                           mel_filters=bank, log_mel="dB").T

    ref = out["input_features"].numpy()
    diff = {}
    for label, bank in (("slaney", fe.mel_filters_slaney), ("htk", fe.mel_filters)):
        diff[label] = max(float(np.abs(recompute(w, bank) - ref[i, 0]).max()) for i, w in enumerate(waves))
    print(f"frontend: max_abs recomputed vs extractor: {diff}")
    assert diff["slaney"] == 0.0 and diff["htk"] > 0.0, "slaney must reproduce input_features exactly"

    # independent plain-numpy STFT (informational: what the C# frontend must match)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)
    def plain(w):
        x = w.astype(np.float64)
        if len(x) < MAX_SAMPLES:
            x = np.pad(np.tile(x, int(MAX_SAMPLES / len(x))), (0, MAX_SAMPLES - len(x) * int(MAX_SAMPLES / len(x))))
        x = np.pad(x, 512, mode="reflect")
        idx = np.arange(1001)[:, None] * 480 + np.arange(1024)[None, :]
        p = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        return 10 * np.log10(np.maximum(p @ fe.mel_filters_slaney, 1e-10))
    plain_diff = max(float(np.abs(plain(w) - ref[i, 0]).max()) for i, w in enumerate(waves))
    print(f"frontend: plain numpy STFT vs extractor max_abs {plain_diff:.3e}")

    with torch.no_grad():
        pooled = model.audio_model(input_features=feats, is_longer=out["is_longer"]).pooler_output
        projected = model.audio_projection(pooled)
        emb = F.normalize(projected, dim=-1)
        assert torch.allclose(emb, model.get_audio_features(input_features=feats, is_longer=out["is_longer"]), atol=1e-6)
    norms = emb.norm(dim=-1)
    assert torch.all((norms - 1).abs() < 1e-6), norms

    tensors = {"mel_filters_slaney": torch.tensor(fe.mel_filters_slaney, dtype=torch.float32).contiguous(),
               "mel_filters_htk": torch.tensor(fe.mel_filters, dtype=torch.float32).contiguous(),
               "input_features": feats.contiguous()}
    for i, w in enumerate(waves):
        tensors[f"waveform_{i}"] = torch.from_numpy(w.copy())
    save_file(tensors, os.path.join(FIXTURES, "frontend.safetensors"))
    save_file({"audio_encoder_output": pooled.contiguous(), "audio_projected": projected.contiguous(),
               "audio_embeddings": emb.contiguous()}, os.path.join(FIXTURES, "audio_embeddings.safetensors"))
    write_json(os.path.join(FIXTURES, "frontend_cases.json"), {
        "cases": cases, "sampling_rate": SR, "max_samples": MAX_SAMPLES, "mel_bank_used": "slaney",
        "padding": "repeatpad", "truncation": "rand_trunc (never taken: every waveform has at most max_samples)",
        "max_abs_recomputed_slaney": diff["slaney"], "max_abs_recomputed_htk": diff["htk"],
        "max_abs_plain_numpy_stft": plain_diff,
    })
    print(f"frontend: audio embeddings {tuple(emb.shape)}, norms in [{norms.min():.8f}, {norms.max():.8f}]")


TEXTS = [
    "dog",
    "This audio is a rock song.",
    "A slow, melancholic piano ballad with soft female vocals, recorded in a large reverberant hall, "
    "followed by a long crescendo of strings and brass that builds up for several minutes until the whole "
    "orchestra plays together in a triumphant finale, then fades away into a quiet, gentle ending with a "
    "solo flute repeating the opening theme once more before the final silence, "
    "while the audience slowly begins to applaud and the recording engineer lowers the faders one by one",
    "Música tradicional española: guitarra flamenca, palmas y cante jondo — ¡olé! 日本の音楽 🎵",
    "Wait... what?! (loud) \"bass\"; [drop] #1 -- 120 bpm / 4/4 & synth_lead: it's 50% louder!!!",
]
MAX_TEXT = 77


def cmd_text(_args):
    """Token ids/mask (text_cases.json, cross-checked with `tokenizers`), pooler_output and text embeddings."""
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import ClapModel, ClapProcessor

    os.makedirs(FIXTURES, exist_ok=True)
    snap = snapshot_dir()
    model = ClapModel.from_pretrained(snap).eval()
    tok = ClapProcessor.from_pretrained(snap).tokenizer
    enc = tok(TEXTS, padding="max_length", truncation=True, max_length=MAX_TEXT, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    # independent check: the `tokenizers` library over tokenizer.json (no special-casing of padding: we pad ourselves)
    ref = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    ref.enable_truncation(max_length=MAX_TEXT)
    ref.enable_padding(length=MAX_TEXT, pad_id=tok.pad_token_id, pad_token=tok.pad_token)
    for i, e in enumerate(ref.encode_batch(TEXTS)):
        if e.ids != ids[i].tolist() or e.attention_mask != mask[i].tolist():
            sys.exit(f"error: tokenizer mismatch on case {i}: {e.ids[:12]} vs {ids[i].tolist()[:12]}")
    lengths = mask.sum(dim=1).tolist()
    assert lengths[2] == MAX_TEXT and ids[2, -1].item() == tok.eos_token_id, "long case must be cut to 77 and end with eos"
    print(f"text: lengths {lengths}, long case ends with eos {tok.eos_token_id}")

    with torch.no_grad():
        pooled = model.text_model(input_ids=ids, attention_mask=mask).pooler_output
        projected = model.text_projection(pooled)
        emb = F.normalize(projected, dim=-1)
        assert torch.allclose(emb, model.get_text_features(input_ids=ids, attention_mask=mask), atol=1e-6)
    norms = emb.norm(dim=-1)
    assert torch.all((norms - 1).abs() < 1e-6), norms

    write_json(os.path.join(FIXTURES, "text_cases.json"), {
        "max_length": MAX_TEXT, "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
        "cases": [{"text": t, "length": lengths[i], "input_ids": ids[i].tolist(), "attention_mask": mask[i].tolist()}
                  for i, t in enumerate(TEXTS)]})
    save_file({"text_pooler_output": pooled.contiguous(), "text_projected": projected.contiguous(),
               "text_embeddings": emb.contiguous()}, os.path.join(FIXTURES, "text_embeddings.safetensors"))
    print(f"text: embeddings {tuple(emb.shape)}, norms in [{norms.min():.8f}, {norms.max():.8f}]")


def cmd_similarity(_args):
    """5x5 cosine matrix (audio rows, text columns) from the stored embeddings, plus logit_scale_a/t."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    audio = load_file(os.path.join(FIXTURES, "audio_embeddings.safetensors"))["audio_embeddings"]
    text = load_file(os.path.join(FIXTURES, "text_embeddings.safetensors"))["text_embeddings"]
    sim = (audio @ text.T).contiguous()
    assert sim.shape == (5, 5)
    with safe_open(os.path.join(OUT, "model.safetensors"), "pt") as f:  # stored scales, no need to build the model
        scale_a, scale_t = f.get_tensor("logit_scale_a").reshape(1).clone(), f.get_tensor("logit_scale_t").reshape(1).clone()
    save_file({"similarity": sim, "logit_scale_a": scale_a, "logit_scale_t": scale_t},
              os.path.join(FIXTURES, "similarity.safetensors"))
    back = load_file(os.path.join(FIXTURES, "similarity.safetensors"))["similarity"]
    err = (back.double() - audio.double() @ text.double().T).abs().max().item()
    assert err < 1e-6, err
    print(f"similarity: max_abs vs audio @ text.T (float64) {err:.3e}; logit_scale_a {scale_a.item():.6f} (exp {scale_a.exp().item():.4f}), "
          f"logit_scale_t {scale_t.item():.6f}")
    print("similarity:", [[round(v, 3) for v in r] for r in sim.tolist()])


def cmd_htsat(_args):
    """Intermediate activations of the HF audio encoder for case 0 (tone) -> htsat_intermediates.safetensors."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import ClapModel

    frontend = load_file(os.path.join(FIXTURES, "frontend.safetensors"))
    embeddings = load_file(os.path.join(FIXTURES, "audio_embeddings.safetensors"))
    feats = frontend["input_features"][0:1].contiguous()  # [1, 1, 1001, 64]

    model = ClapModel.from_pretrained(snapshot_dir()).eval()
    enc = model.audio_model.audio_encoder
    taps = {}

    def tap(name, pick):
        def hook(_module, args, output):
            taps[name] = pick(args, output).detach().clone().contiguous()
        return hook

    first = lambda _a, o: o[0]  # layers and blocks return tuples: (hidden_states, ...)
    enc.patch_embed.register_forward_hook(tap("image", lambda a, _o: a[0]))  # batch norm + reshape_mel2img output
    enc.patch_embed.register_forward_hook(tap("patch_embed", lambda _a, o: o))
    enc.layers[0].blocks[0].register_forward_hook(tap("stage0_block0", first))
    enc.layers[0].blocks[1].register_forward_hook(tap("stage0_block1", first))
    for i, layer in enumerate(enc.layers):
        layer.register_forward_hook(tap(f"stage{i}", first))  # after patch merging where the stage has it
    enc.norm.register_forward_hook(tap("norm", lambda _a, o: o))

    with torch.no_grad():
        pooled = model.audio_model(input_features=feats, is_longer=torch.zeros(1, 1, dtype=torch.bool)).pooler_output

    expect = {"image": (1, 1, 256, 256), "patch_embed": (1, 4096, 96), "stage0_block0": (1, 4096, 96),
              "stage0_block1": (1, 4096, 96), "stage0": (1, 1024, 192), "stage1": (1, 256, 384),
              "stage2": (1, 64, 768), "stage3": (1, 64, 768), "norm": (1, 64, 768)}
    for k, shape in expect.items():
        assert tuple(taps[k].shape) == shape, (k, tuple(taps[k].shape), shape)
        print(f"htsat: {k} {tuple(taps[k].shape)}")

    # consistency with the other fixtures: the pooled output is the mean over the final tokens
    diff = (taps["norm"].mean(dim=1) - embeddings["audio_encoder_output"][0:1]).abs().max().item()
    print(f"htsat: mean(norm) vs audio_encoder_output[0]: max_abs {diff:.3e}")
    assert diff <= 1e-5, diff
    assert torch.equal(pooled, taps["norm"].mean(dim=1)) or (pooled - taps["norm"].mean(dim=1)).abs().max().item() <= 1e-6

    out = dict(taps)
    out["input_features"] = feats
    save_file(out, os.path.join(FIXTURES, "htsat_intermediates.safetensors"))
    print(f"htsat: wrote {len(out)} tensors")


def cmd_roberta(_args):
    """Intermediate activations of the HF text encoder for text cases 1 and 2 -> roberta_intermediates.safetensors."""
    import json
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import ClapModel

    cases = json.load(open(os.path.join(FIXTURES, "text_cases.json"), encoding="utf-8"))["cases"]
    pick = [1, 2]  # 9 tokens padded to 77, and the 77-token cut
    ids = torch.tensor([cases[i]["input_ids"] for i in pick], dtype=torch.long)
    mask = torch.tensor([cases[i]["attention_mask"] for i in pick], dtype=torch.long)
    reference = load_file(os.path.join(FIXTURES, "text_embeddings.safetensors"))["text_pooler_output"][pick]

    model = ClapModel.from_pretrained(snapshot_dir()).eval()
    text = model.text_model
    taps = {}

    def tap(name, pick_output):
        def hook(_module, _args, output):
            taps[name] = pick_output(output).detach().clone().contiguous()
        return hook

    text.embeddings.register_forward_hook(tap("embeddings", lambda o: o))
    for i, layer in enumerate(text.encoder.layer):
        layer.register_forward_hook(tap(f"layer{i:02d}", lambda o: o[0]))  # layers return tuples
    text.pooler.register_forward_hook(tap("pooler", lambda o: o))

    with torch.no_grad():
        pooled = text(input_ids=ids, attention_mask=mask).pooler_output

    assert len(taps) == 14, sorted(taps)
    for k, v in taps.items():
        shape = (2, 768) if k == "pooler" else (2, 77, 768)
        assert tuple(v.shape) == shape, (k, tuple(v.shape))
    print(f"roberta: {len(taps) - 1} activations of [2, 77, 768] (embeddings and 12 layers) plus the pooler [2, 768]")
    diff = (taps["pooler"] - reference).abs().max().item()
    print(f"roberta: pooler vs stored text_pooler_output rows {pick}: max_abs {diff:.3e}")
    assert diff <= 1e-5, diff
    assert torch.equal(pooled, taps["pooler"])

    save_file(taps, os.path.join(FIXTURES, "roberta_intermediates.safetensors"))
    print(f"roberta: wrote {len(taps)} tensors")


SERVICE_SR = 48000


def make_service_inputs():
    """The four WAV inputs of the service fixtures: (file name, samples [n] or [n, 2], rate, soundfile subtype). Seeded, no files read."""
    import numpy as np

    rng = np.random.default_rng(SEED)

    t = np.arange(12 * 16000) / 16000.0  # 12 s: three windows of the service; upsampled from 16 kHz
    a = 0.4 * np.sin(2 * np.pi * (220.0 * t + 30.0 * t * t)) + 0.2 * np.sin(2 * np.pi * 880.0 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    a = np.clip(a + 0.02 * rng.standard_normal(len(t)), -1, 1)

    t = np.arange(int(3 * 22050)) / 22050.0  # stereo with different content per channel
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t) * np.exp(-3.0 * (t % 0.5))
    noise = rng.standard_normal(len(t))
    right = 0.4 * np.convolve(noise, np.ones(8) / 8, mode="same")
    b = np.stack([left, right], axis=1)

    n = 2 * SERVICE_SR  # float32 WAV at 48 kHz: no resampling
    t = np.arange(n) / float(SERVICE_SR)
    c = 0.25 * np.convolve(rng.standard_normal(n), np.ones(4) / 4, mode="same") + 0.2 * np.sin(2 * np.pi * 330.0 * t)

    t = np.arange(int(1.5 * 44100)) / 44100.0  # 24-bit at 44.1 kHz: 1.5 s, repeat-padded by the extractor
    d = 0.5 * np.sin(2 * np.pi * (200.0 * t + 933.0 * t * t)) + 0.15 * np.sin(2 * np.pi * 1200.0 * t)

    return [
        ("tone_12s_16k_mono_pcm16.wav", a, 16000, "PCM_16"),
        ("mix_3s_22k_stereo_pcm16.wav", b, 22050, "PCM_16"),
        ("noise_2s_48k_mono_float32.wav", c, SERVICE_SR, "FLOAT"),
        ("chirp_1p5s_44k_mono_pcm24.wav", d, 44100, "PCM_24"),
    ]


SERVICE_REQUESTS = [
    {"labels": {"genre": ["rock", "jazz", "classical", "electronic", "hip hop", "folk", "metal", "reggae"],
                "mood": ["happy", "sad", "nostalgic", "calm", "mysterious"]}},
    {"labels": {"instrument": ["piano", "guitar", "drums", "violin", "synthesizer", "flute"],
                "style": ["ambient", "dance", "orchestral", "acoustic"]},
     "template": "A recording of {} music.", "top": 5},
    {"labels": {"genre": ["pop", "blues", "country", "techno"], "mood": ["romantic", "angry", "peaceful"], "tempo": ["fast", "slow"]},
     "top": 0},
    {"labels": {"genre": ["pop", "música clásica", "jazz", "electronic", "folk", "metal", "reggae", "latin", "opera", "disco", "funk", "ambient"]},
     "top": 50},
]


def write_float32_wav(path, samples, rate):
    """Mono IEEE-float WAV written by hand: libsndfile adds a PEAK chunk with the current time, which would make the file differ between runs."""
    import struct
    import numpy as np

    data = np.asarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)  # WAVE_FORMAT_IEEE_FLOAT, mono, 32 bit
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def cmd_classify(_args):
    """WAV inputs, requests and the exact responses of the reference service (`/opt/server.py` Engine.classify) + the embeddings behind them."""
    import base64
    import io
    import json
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, "/opt")
    import server  # the reference service, imported not copied

    os.makedirs(FIXTURES, exist_ok=True)
    engine = server.Engine()
    cases, tensors, too_close = [], {}, []
    for i, ((name, data, rate, subtype), request) in enumerate(zip(make_service_inputs(), SERVICE_REQUESTS)):
        path = os.path.join(FIXTURES, name)
        if subtype == "FLOAT":
            write_float32_wav(path, data, rate)
        else:
            sf.write(path, data, rate, subtype=subtype)
        raw = open(path, "rb").read()
        body = dict(request, wav_base64=base64.b64encode(raw).decode())
        response = engine.classify(body)  # the reference's own code, offline, on this container's CPU

        # the same steps again, keeping the tensors the response is made of
        wav, file_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        resampled = None
        if file_rate != SERVICE_SR:
            n = int(len(wav) * SERVICE_SR / file_rate)
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype("float32")
            resampled = wav
        window = 10 * SERVICE_SR
        segs = [wav[s : s + window] for s in [0, window // 2, window] if len(wav[s : s + window]) > SERVICE_SR]
        with torch.no_grad():
            ai = engine.proc(audios=segs, sampling_rate=SERVICE_SR, return_tensors="pt")
            audio = engine.model.get_audio_features(**ai)
            audio = (audio / audio.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)
            template = str(request.get("template") or "This audio is a {} song.")
            top = max(1, min(int(request.get("top") or 3), 10))
            mine = {}
            for c, (category, tags) in enumerate(request["labels"].items()):
                ti = engine.proc(text=[template.format(t) for t in tags], return_tensors="pt", padding=True)
                text = engine.model.get_text_features(**ti)
                text = text / text.norm(dim=-1, keepdim=True)
                sim = (audio @ text.T)[0]
                best = sim.topk(min(top, len(tags)))
                mine[category] = [{"tag": tags[j], "score": round(v, 3)} for v, j in zip(best.values.tolist(), best.indices.tolist())]
                tensors[f"text_{i}_{c}"] = text.contiguous()
                ordered = sorted(sim.tolist(), reverse=True)
                gap = min(x - y for x, y in zip(ordered, ordered[1:]))
                ranking = sorted(zip(sim.tolist(), tags), reverse=True)
                print(f"classify:   case {i} {category}: smallest gap between adjacent scores {gap:.2e}  " + ", ".join(f"{t} {v:.4f}" for v, t in ranking))
                if gap < 2e-3:
                    too_close.append(f"case {i} {category} ({gap:.2e})")
        assert mine == response["categories"], (i, mine, response["categories"])
        assert response["seconds"] == round(len(wav) / SERVICE_SR, 1)
        tensors[f"audio_{i}"] = audio[0].contiguous()
        if resampled is not None and i == 3:
            tensors["resampled_3"] = torch.from_numpy(resampled.copy())
        print(f"classify: case {i} {name}: rate {file_rate}, {len(segs)} window(s), {response['seconds']} s, "
              + "; ".join(f"{k}: {v[0]['tag']} {v[0]['score']}" for k, v in response["categories"].items()))
        cases.append({"id": i, "wav": name, "sample_rate": file_rate, "channels": int(data.shape[1]) if data.ndim == 2 else 1,
                      "subtype": subtype, "windows": len(segs), "request": request, "response": response})

    assert not too_close, "adjacent scores closer than 2e-3 (the order would depend on float noise); change the labels: " + ", ".join(too_close)
    with open(os.path.join(FIXTURES, "classify_cases.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, indent=2, ensure_ascii=False)  # insertion order kept: categories come back in request order
        f.write("\n")
    save_file(tensors, os.path.join(FIXTURES, "classify_embeddings.safetensors"))
    print(f"classify: wrote {len(cases)} cases and {len(tensors)} tensors")


COMMANDS = {
    "checkpoint": (cmd_checkpoint, "convert pytorch_model.bin to model.safetensors in /out"),
    "frontend": (cmd_frontend, "mel filters, waveforms, input_features, audio embeddings"),
    "htsat": (cmd_htsat, "intermediate activations of the audio encoder for case 0"),
    "text": (cmd_text, "token ids, pooler_output, text embeddings"),
    "roberta": (cmd_roberta, "intermediate activations of the text encoder for cases 1 and 2"),
    "classify": (cmd_classify, "service fixtures: WAV inputs, requests, exact responses of the reference service"),
    "similarity": (cmd_similarity, "5x5 similarity matrix and logit scales"),
    "noop": (cmd_noop, "only prints the environment (reproducibility check)"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="{checkpoint,frontend,htsat,text,roberta,classify,similarity}")
    for name, (fn, help_) in COMMANDS.items():
        sub.add_parser(name, help=help_ if name != "noop" else argparse.SUPPRESS).set_defaults(fn=fn)
    args = p.parse_args(argv)
    if args.command != "noop":
        require_weights()
    setup()
    args.fn(args)


if __name__ == "__main__":
    main()
