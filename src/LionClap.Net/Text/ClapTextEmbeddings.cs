using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Text;

/// <summary>
/// RoBERTa embeddings (<c>ClapTextEmbeddings</c>): word + token-type + position embeddings, then LayerNorm (eps 1e-12). Field names are the checkpoint
/// keys under <c>text_model.embeddings.</c>. Positions are not <c>0..L−1</c>: padding keeps position <c>pad_token_id</c> and the real tokens count
/// from <c>pad_token_id + 1</c>.
/// </summary>
public sealed class ClapTextEmbeddings : nn.Module<Tensor, Tensor>
{
    private readonly Embedding word_embeddings;
    private readonly Embedding position_embeddings;
    private readonly Embedding token_type_embeddings;
    private readonly LayerNorm LayerNorm;
    private readonly ClapTextConfig _config;

    public ClapTextEmbeddings(ClapTextConfig config) : base(nameof(ClapTextEmbeddings))
    {
        _config = config;
        word_embeddings = nn.Embedding(config.VocabSize, config.HiddenSize, padding_idx: config.PadTokenId);
        position_embeddings = nn.Embedding(config.MaxPositionEmbeddings, config.HiddenSize, padding_idx: config.PadTokenId);
        token_type_embeddings = nn.Embedding(config.TypeVocabSize, config.HiddenSize);
        LayerNorm = nn.LayerNorm(config.HiddenSize, config.LayerNormEps);
        RegisterComponents();
        // persistent buffers of the checkpoint; the position ids are always derived from the ids, the token types are the stored zeros
        register_buffer("position_ids", arange(config.MaxPositionEmbeddings, dtype: ScalarType.Int64).unsqueeze(0));
        register_buffer("token_type_ids", zeros(1, config.MaxPositionEmbeddings, dtype: ScalarType.Int64));
    }

    /// <summary>
    /// Position of every token: <c>cumsum(ids != pad) · (ids != pad) + pad</c>. For ids <c>[0, 5, 6, 2, 1, 1]</c> and pad 1 that is
    /// <c>[2, 3, 4, 5, 1, 1]</c>.
    /// </summary>
    public static Tensor PositionIds(Tensor ids, int padTokenId)
    {
        using var attended = ids.ne(padTokenId).to(ScalarType.Int32);
        using var running = attended.cumsum(1).to(ScalarType.Int32);
        using var counted = running * attended;
        return counted.to(ScalarType.Int64) + padTokenId;
    }

    /// <summary><c>[B, L]</c> int64 ids → <c>[B, L, 768]</c>.</summary>
    public override Tensor forward(Tensor input_ids)
    {
        TextInputs.Validate(input_ids, null, _config);
        var (rows, length) = (input_ids.shape[0], input_ids.shape[1]);
        using var positions = PositionIds(input_ids, _config.PadTokenId);
        using var tokenTypes = get_buffer("token_type_ids").narrow(1, 0, length).expand(rows, length);

        using var words = word_embeddings.forward(input_ids);
        using var types = token_type_embeddings.forward(tokenTypes);
        using var sum = words + types;
        using var places = position_embeddings.forward(positions);
        using var total = sum + places;
        return LayerNorm.forward(total);
    }
}
