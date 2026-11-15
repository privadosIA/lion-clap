using System.Globalization;
using System.Text;
using System.Text.Json;
using LionClap.Audio;
using LionClap.Audio.Htsat;
using LionClap.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Classification;

/// <summary>Embeds a 48 kHz clip (<see cref="ClapAudioEmbedder"/> in production, a fake in tests).</summary>
public interface IClipEmbedder
{
    ClipEmbedding Embed(float[] clip48k);
}

/// <summary>Embeds texts into unit-norm vectors (<see cref="ClapTextEmbedder"/> in production).</summary>
public interface ITextEmbedder
{
    float[][] Embed(IReadOnlyList<string> texts);
}

/// <summary>The text branch behind <see cref="ITextEmbedder"/>: tokenised to the longest text of the batch without truncation (the reference's <c>padding=True</c>).</summary>
public sealed class ClapTextEmbedder(ClapTextBranch branch, ClapTextTokenizer tokenizer) : ITextEmbedder
{
    public float[][] Embed(IReadOnlyList<string> texts)
    {
        using var result = branch.Forward(tokenizer.EncodeBatch(texts));
        var flat = result.Embeddings.to(ScalarType.Float32).cpu().data<float>().ToArray();
        var dim = (int)result.Embeddings.shape[1];
        return Enumerable.Range(0, texts.Count).Select(i => flat[(i * dim)..((i + 1) * dim)]).ToArray();
    }
}

/// <summary>A classification request after defaults and limits: categories in the order given, each with at most 200 tags.</summary>
public sealed record ClassifyRequest(IReadOnlyList<KeyValuePair<string, IReadOnlyList<string>>> Labels, string Template, int Top)
{
    public const string DefaultTemplate = "This audio is a {} song.";
    public const int DefaultTop = 3, MaxTop = 10, MaxTagsPerCategory = 200;

    /// <summary>Applies the reference's rules: no template means the default, <c>top</c> 0 or absent means 3 and is clamped to 1–10, tags beyond the first 200 of a category are ignored.</summary>
    public static ClassifyRequest Create(IEnumerable<KeyValuePair<string, IReadOnlyList<string>>> labels, string? template = null, int? top = null)
    {
        var capped = labels.Select(c => new KeyValuePair<string, IReadOnlyList<string>>(c.Key, c.Value.Take(MaxTagsPerCategory).ToList())).ToList();
        return new ClassifyRequest(capped, string.IsNullOrEmpty(template) ? DefaultTemplate : template, ClampTop(top));
    }

    public static int ClampTop(int? top) => Math.Max(1, Math.Min(top is null or 0 ? DefaultTop : top.Value, MaxTop));
}

public sealed record ScoredTag(string Tag, double Score);

public sealed record ClassifyResult(double Seconds, IReadOnlyList<KeyValuePair<string, IReadOnlyList<ScoredTag>>> Categories)
{
    /// <summary>The JSON of the reference service: <c>{"seconds": 12.0, "categories": {"genre": [{"tag": "rock", "score": 0.179}, …]}}</c>, categories in request order.</summary>
    public byte[] ToJson()
    {
        using var stream = new MemoryStream();
        using (var w = new Utf8JsonWriter(stream))
        {
            w.WriteStartObject();
            w.WritePropertyName("seconds");
            w.WriteRawValue(Seconds.ToString("0.0", CultureInfo.InvariantCulture)); // 12.0, as the reference prints it
            w.WriteStartObject("categories");
            foreach (var (category, tags) in Categories)
            {
                w.WriteStartArray(category);
                foreach (var t in tags)
                {
                    w.WriteStartObject();
                    w.WriteString("tag", t.Tag);
                    w.WriteNumber("score", t.Score);
                    w.WriteEndObject();
                }
                w.WriteEndArray();
            }
            w.WriteEndObject();
            w.WriteEndObject();
        }
        return stream.ToArray();
    }
}

/// <summary>
/// What the reference service does with embeddings: the audio vector (mean of the window embeddings) is compared with the embedding of each tag put into
/// the template, per category; the best <c>top</c> come back ordered by score, rounded to three decimals.
/// </summary>
public sealed class ClapClassifier(IClipEmbedder audio, ITextEmbedder text)
{
    public ClassifyResult Classify(float[] clip48k, ClassifyRequest request)
    {
        var vector = audio.Embed(clip48k).Vector;
        var categories = new List<KeyValuePair<string, IReadOnlyList<ScoredTag>>>();
        foreach (var (category, tags) in request.Labels)
        {
            if (tags.Count == 0) { categories.Add(new(category, [])); continue; }
            var embeddings = text.Embed(tags.Select(t => request.Template.Replace("{}", t, StringComparison.Ordinal)).ToList());
            var scores = embeddings.Select(e => Dot(vector, e)).ToArray();
            var best = Enumerable.Range(0, tags.Count)
                .OrderByDescending(i => scores[i]).ThenBy(i => i)       // ties by index: torch.topk does not promise an order for them
                .Take(Math.Min(request.Top, tags.Count))
                .Select(i => new ScoredTag(tags[i], Math.Round(scores[i], 3, MidpointRounding.ToEven)))
                .ToList();
            categories.Add(new(category, best));
        }
        var seconds = Math.Round(clip48k.Length / (double)ClapAudioEmbedder.SampleRate, 1, MidpointRounding.ToEven);
        return new ClassifyResult(seconds, categories);
    }

    private static double Dot(float[] a, float[] b)
    {
        if (a.Length != b.Length) throw new InvalidOperationException($"the audio and text embeddings differ in size ({a.Length} vs {b.Length})");
        double sum = 0;
        for (var i = 0; i < a.Length; i++) sum += (double)a[i] * b[i];
        return sum;
    }
}
