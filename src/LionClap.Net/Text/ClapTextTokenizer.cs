using Tokenizers.DotNet;

namespace LionClap.Text;

/// <summary>Token ids and attention masks of a batch, row-major <c>[rows, length]</c>.</summary>
public sealed record TokenBatch(long[] Ids, long[] Mask, int Rows, int Length);

/// <summary>
/// The checkpoint's tokenizer: <c>tokenizer.json</c> (RoBERTa byte-level BPE with the start/end post-processor) through
/// <c>Tokenizers.DotNet</c>, the Rust library that Python's <c>tokenizers</c> uses, so the ids are the reference ones. Padding and truncation
/// are done here: <see cref="EncodeBatch(IReadOnlyList{string})"/> pads to the longest text and never truncates (what <c>server.py</c> does);
/// <see cref="EncodeBatch(IReadOnlyList{string}, int)"/> pads every row to a fixed length and cuts longer ones, keeping the end token
/// (what the golden fixtures do with 77).
/// </summary>
public sealed class ClapTextTokenizer : IDisposable
{
    private readonly Tokenizer _tokenizer;
    private readonly object _gate = new(); // the native API does not document whether it is thread-safe
    private readonly ClapTextConfig _config;

    public ClapTextTokenizer(string tokenizerJsonPath, ClapTextConfig? config = null)
    {
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"The tokenizer was not found at {tokenizerJsonPath}", tokenizerJsonPath);
        _tokenizer = new Tokenizer(tokenizerJsonPath);
        _config = config ?? new ClapTextConfig();
    }

    /// <summary>Ids of one text with the special tokens, untruncated.</summary>
    public long[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        uint[] ids;
        lock (_gate) ids = _tokenizer.Encode(text);
        return ids.Select(i => (long)i).ToArray();
    }

    /// <summary>Pads to the longest text of the batch; throws if one is longer than the model accepts (it does not cut silently).</summary>
    public TokenBatch EncodeBatch(IReadOnlyList<string> texts)
    {
        var rows = Rows(texts);
        var longest = rows.Max(r => r.Length);
        if (longest > _config.MaxTokens)
            throw new ArgumentException($"a text has {longest} tokens, more than the {_config.MaxTokens} the model accepts; pass a maximum length to cut it", nameof(texts));
        return Pack(rows, longest);
    }

    /// <summary>Pads every row to <paramref name="maxLength"/> and cuts longer ones to it, keeping the end token.</summary>
    public TokenBatch EncodeBatch(IReadOnlyList<string> texts, int maxLength)
    {
        if (maxLength < 2 || maxLength > _config.MaxTokens)
            throw new ArgumentOutOfRangeException(nameof(maxLength), maxLength, $"the maximum length must be between 2 and {_config.MaxTokens}");
        var rows = Rows(texts).Select(r => r.Length <= maxLength ? r : r[..(maxLength - 1)].Append(_config.EosTokenId).ToArray()).ToArray();
        return Pack(rows, maxLength);
    }

    private long[][] Rows(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (texts.Count == 0) throw new ArgumentException("at least one text is needed", nameof(texts));
        return texts.Select(Encode).ToArray();
    }

    private TokenBatch Pack(long[][] rows, int length)
    {
        var ids = new long[rows.Length * length];
        var mask = new long[rows.Length * length];
        for (var r = 0; r < rows.Length; r++)
        {
            for (var i = 0; i < length; i++)
            {
                var real = i < rows[r].Length;
                ids[r * length + i] = real ? rows[r][i] : _config.PadTokenId;
                mask[r * length + i] = real ? 1 : 0;
            }
        }
        return new TokenBatch(ids, mask, rows.Length, length);
    }

    public void Dispose() => _tokenizer.Dispose();
}
