using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Text;

/// <summary>Checks of the token ids and attention mask given to the text branch. Every message names the offending value.</summary>
public static class TextInputs
{
    /// <summary>Throws <see cref="ArgumentException"/> when the ids (and the mask, when given) cannot be encoded.</summary>
    public static void Validate(Tensor ids, Tensor? mask, ClapTextConfig config)
    {
        ArgumentNullException.ThrowIfNull(ids);
        if (ids.dim() != 2) throw new ArgumentException($"input_ids must have 2 dimensions [batch, length], got {ids.dim()}", nameof(ids));
        if (ids.dtype != ScalarType.Int64) throw new ArgumentException($"input_ids must be int64, got {ids.dtype}", nameof(ids));
        var (rows, length) = (ids.shape[0], ids.shape[1]);
        if (rows < 1) throw new ArgumentException("input_ids must have at least one row", nameof(ids));
        if (length < 1 || length > config.MaxTokens)
            throw new ArgumentException($"input_ids must have 1 to {config.MaxTokens} tokens per row, got {length}", nameof(ids));

        using var _ = no_grad();
        var lowest = ids.min().item<long>();
        var highest = ids.max().item<long>();
        if (lowest < 0) throw new ArgumentException($"token id {lowest} is outside the vocabulary (size {config.VocabSize})", nameof(ids));
        if (highest >= config.VocabSize) throw new ArgumentException($"token id {highest} is outside the vocabulary (size {config.VocabSize})", nameof(ids));

        if (mask is null) return;
        if (!mask.shape.SequenceEqual(ids.shape))
            throw new ArgumentException($"attention_mask has shape [{string.Join(", ", mask.shape)}], the ids have [{string.Join(", ", ids.shape)}]", nameof(mask));
        using var notBinary = mask.ne(0).logical_and(mask.ne(1));
        if (notBinary.any().item<bool>()) throw new ArgumentException("attention_mask values must be 0 or 1", nameof(mask));
        using var perRow = mask.to(ScalarType.Int64).sum(dim: 1);
        using var empty = perRow.eq(0);
        if (empty.any().item<bool>())
            throw new ArgumentException($"attention_mask row {empty.nonzero()[0, 0].item<long>()} is all zero: every text needs at least one attended token", nameof(mask));
    }
}
