using System.Text.Json;

namespace LionClap.Audio.Htsat;

/// <summary>
/// The <c>audio_config</c> of <c>laion/clap-htsat-unfused</c> (HTSAT-tiny). The defaults are that checkpoint's values; only the fields
/// the unfused inference path uses are kept. A configuration that enables the fusion path is rejected.
/// </summary>
public sealed record ClapAudioConfig
{
    public int SpecSize { get; init; } = 256;
    public int NumMelBins { get; init; } = 64;
    public int PatchSize { get; init; } = 4;
    public int PatchStride { get; init; } = 4;
    public int PatchEmbedInputChannels { get; init; } = 1;
    public int PatchEmbedsHiddenSize { get; init; } = 96;
    public int[] Depths { get; init; } = [2, 2, 6, 2];
    public int[] NumAttentionHeads { get; init; } = [4, 8, 16, 32];
    public int WindowSize { get; init; } = 8;
    public double MlpRatio { get; init; } = 4.0;
    public bool QkvBias { get; init; } = true;
    public double LayerNormEps { get; init; } = 1e-5;
    public bool EnablePatchLayerNorm { get; init; } = true;
    public int HiddenSize { get; init; } = 768;
    public int ProjectionDim { get; init; } = 512;

    /// <summary>Time frames the encoder accepts at most: <c>spec_size · (spec_size / num_mel_bins)</c> (1024).</summary>
    public int MaxFrames => SpecSize * FreqRatio;

    /// <summary>How many time slices are folded into the frequency axis of the square "image" (4).</summary>
    public int FreqRatio => SpecSize / NumMelBins;

    /// <summary>Reads the <c>audio_config</c> object of an HF <c>config.json</c>; missing fields keep the defaults.</summary>
    public static ClapAudioConfig FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        if (!doc.RootElement.TryGetProperty("audio_config", out var a))
            throw new InvalidDataException($"{path} has no audio_config");

        if (Bool(a, "enable_fusion", false))
            throw new NotSupportedException("audio_config.enable_fusion is true: the fusion path is not supported (use the unfused checkpoint)");
        if (Str(a, "hidden_act", "gelu") != "gelu" || Str(a, "projection_hidden_act", "relu") != "relu")
            throw new NotSupportedException("only hidden_act=gelu and projection_hidden_act=relu are supported");

        var d = new ClapAudioConfig();
        return new ClapAudioConfig
        {
            SpecSize = Int(a, "spec_size", d.SpecSize),
            NumMelBins = Int(a, "num_mel_bins", d.NumMelBins),
            PatchSize = Int(a, "patch_size", d.PatchSize),
            PatchStride = Int(a, "patch_stride", d.PatchStride),
            PatchEmbedInputChannels = Int(a, "patch_embed_input_channels", d.PatchEmbedInputChannels),
            PatchEmbedsHiddenSize = Int(a, "patch_embeds_hidden_size", d.PatchEmbedsHiddenSize),
            Depths = Ints(a, "depths", d.Depths),
            NumAttentionHeads = Ints(a, "num_attention_heads", d.NumAttentionHeads),
            WindowSize = Int(a, "window_size", d.WindowSize),
            MlpRatio = a.TryGetProperty("mlp_ratio", out var m) ? m.GetDouble() : d.MlpRatio,
            QkvBias = Bool(a, "qkv_bias", d.QkvBias),
            LayerNormEps = a.TryGetProperty("layer_norm_eps", out var e) ? e.GetDouble() : d.LayerNormEps,
            EnablePatchLayerNorm = Bool(a, "enable_patch_layer_norm", d.EnablePatchLayerNorm),
            HiddenSize = Int(a, "hidden_size", d.HiddenSize),
            ProjectionDim = Int(a, "projection_dim", d.ProjectionDim),
        };
    }

    // patch_size / patch_stride are an int or a [h, w] pair in HF configs; only square values are supported.
    private static int Int(JsonElement o, string name, int fallback)
    {
        if (!o.TryGetProperty(name, out var v)) return fallback;
        if (v.ValueKind != JsonValueKind.Array) return v.GetInt32();
        var pair = v.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        if (pair.Distinct().Count() != 1) throw new NotSupportedException($"{name} must be square, got [{string.Join(", ", pair)}]");
        return pair[0];
    }

    private static int[] Ints(JsonElement o, string name, int[] fallback) =>
        o.TryGetProperty(name, out var v) ? v.EnumerateArray().Select(x => x.GetInt32()).ToArray() : fallback;

    private static bool Bool(JsonElement o, string name, bool fallback) =>
        o.TryGetProperty(name, out var v) ? v.GetBoolean() : fallback;

    private static string Str(JsonElement o, string name, string fallback) =>
        o.TryGetProperty(name, out var v) ? v.GetString() ?? fallback : fallback;
}
