using System.Text.Json;

namespace LionClap.Text;

/// <summary>
/// The <c>text_config</c> of <c>laion/clap-htsat-unfused</c> (RoBERTa-base). The defaults are that checkpoint's values; only what the
/// inference path uses is kept. Relative position embeddings and decoder mode are rejected when a configuration is read.
/// </summary>
public sealed record ClapTextConfig
{
    public int VocabSize { get; init; } = 50265;
    public int HiddenSize { get; init; } = 768;
    public int NumHiddenLayers { get; init; } = 12;
    public int NumAttentionHeads { get; init; } = 12;
    public int IntermediateSize { get; init; } = 3072;
    public int MaxPositionEmbeddings { get; init; } = 514;
    public int TypeVocabSize { get; init; } = 1;
    public int PadTokenId { get; init; } = 1;
    public int BosTokenId { get; init; } = 0;
    public int EosTokenId { get; init; } = 2;
    /// <summary>1e-12 in the text branch (the audio branch uses 1e-5).</summary>
    public double LayerNormEps { get; init; } = 1e-12;
    public int ProjectionDim { get; init; } = 512;

    /// <summary>Longest input in tokens: the position table has <c>pad_token_id + 1</c> unusable rows (positions start after the padding index).</summary>
    public int MaxTokens => MaxPositionEmbeddings - PadTokenId - 1;

    /// <summary>Reads the <c>text_config</c> object of an HF <c>config.json</c>; missing fields keep the defaults.</summary>
    public static ClapTextConfig FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        if (!doc.RootElement.TryGetProperty("text_config", out var t))
            throw new InvalidDataException($"{path} has no text_config");

        if (Str(t, "position_embedding_type", "absolute") != "absolute")
            throw new NotSupportedException("only position_embedding_type=absolute is supported");
        if (Bool(t, "is_decoder", false))
            throw new NotSupportedException("text_config.is_decoder is true: decoder mode is not supported");
        if (Str(t, "hidden_act", "gelu") != "gelu" || Str(t, "projection_hidden_act", "relu") != "relu")
            throw new NotSupportedException("only hidden_act=gelu and projection_hidden_act=relu are supported");

        var d = new ClapTextConfig();
        return new ClapTextConfig
        {
            VocabSize = Int(t, "vocab_size", d.VocabSize),
            HiddenSize = Int(t, "hidden_size", d.HiddenSize),
            NumHiddenLayers = Int(t, "num_hidden_layers", d.NumHiddenLayers),
            NumAttentionHeads = Int(t, "num_attention_heads", d.NumAttentionHeads),
            IntermediateSize = Int(t, "intermediate_size", d.IntermediateSize),
            MaxPositionEmbeddings = Int(t, "max_position_embeddings", d.MaxPositionEmbeddings),
            TypeVocabSize = Int(t, "type_vocab_size", d.TypeVocabSize),
            PadTokenId = Int(t, "pad_token_id", d.PadTokenId),
            BosTokenId = Int(t, "bos_token_id", d.BosTokenId),
            EosTokenId = Int(t, "eos_token_id", d.EosTokenId),
            LayerNormEps = t.TryGetProperty("layer_norm_eps", out var e) ? e.GetDouble() : d.LayerNormEps,
            ProjectionDim = Int(t, "projection_dim", d.ProjectionDim),
        };
    }

    private static int Int(JsonElement o, string name, int fallback) => o.TryGetProperty(name, out var v) ? v.GetInt32() : fallback;

    private static bool Bool(JsonElement o, string name, bool fallback) => o.TryGetProperty(name, out var v) ? v.GetBoolean() : fallback;

    private static string Str(JsonElement o, string name, string fallback) => o.TryGetProperty(name, out var v) ? v.GetString() ?? fallback : fallback;
}
