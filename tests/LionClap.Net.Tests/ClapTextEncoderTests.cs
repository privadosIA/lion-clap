using LionClap.Text;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class ClapTextEncoderTests(ITestOutputHelper output)
{
    private static readonly ClapTextConfig Config = new();

    private static Tensor Stored(string name) => Fixtures.Load("roberta_intermediates.safetensors")[name];

    /// <summary>Intermediates: cos_sim ≥ 0.9999 and max_abs ≤ 1e-4 · max(1, peak |reference|).</summary>
    private void AssertIntermediate(string what, Tensor got, Tensor expected)
    {
        var r = Parity.Compare(got, expected);
        var peak = expected.abs().max().item<float>();
        var limit = 1e-4 * Math.Max(1.0, peak);
        output.WriteLine($"{what}: {r} peak|ref|={peak:F3} max_abs/peak={r.MaxAbs / peak:E2} limit={limit:E2}");
        Assert.True(r.MaxAbs <= limit && r.CosSim >= 0.9999, $"{what}: {r} (limit {limit:E2})");
    }

    [Fact]
    public void ExtendedMaskIsZeroWhereAttendedAndTheLowestFloatWherePadded()
    {
        using var mask = tensor(new long[] { 1, 1, 0, 1, 0, 0 }, [2, 3]);
        using var extended = TextMasks.Extend(mask);
        Assert.Equal(new long[] { 2, 1, 1, 3 }, extended.shape);
        var v = extended.data<float>().ToArray();
        Assert.Equal(new[] { 0f, 0f, float.MinValue, 0f, float.MinValue, float.MinValue }, v);

        using var all = TextMasks.Extend(ones(1, 4, dtype: ScalarType.Int64));
        Assert.All(all.data<float>().ToArray(), x => Assert.Equal(0f, x));
    }

    [Fact]
    public void PaddedKeysGetNoAttentionWeight()
    {
        using var attention = new ClapTextSelfAttention(Config);
        attention.eval();
        using var _ = no_grad();
        using var hidden = randn(1, 5, 768);
        using var mask = TextMasks.Extend(tensor(new long[] { 1, 1, 1, 0, 0 }, [1, 5]));
        using var withPadding = attention.forward(hidden, mask);
        // the attended positions must not change when the padded positions hold different values
        using var changed = hidden.clone();
        changed[0, 3..5] = randn(2, 768) * 10;
        using var again = attention.forward(changed, mask);
        Assert.True((withPadding[0, 0..3] - again[0, 0..3]).abs().max().item<float>() < 1e-5f);
    }

    [Fact]
    public void ParameterAndBufferCensusOfTheTextModel()
    {
        using var model = new ClapTextModel(Config);
        Assert.Equal(199, model.named_parameters().Count()); // embeddings 5 + 12 layers × 16 + pooler 2
        Assert.Equal(2, model.named_buffers().Count());       // position_ids, token_type_ids
        var names = model.named_parameters().Select(p => p.name).ToList();
        Assert.Contains("encoder.layer.11.attention.output.LayerNorm.weight", names);
        Assert.Contains("encoder.layer.0.attention.self.query.bias", names);
        Assert.Contains("pooler.dense.weight", names);
        Assert.Contains("embeddings.LayerNorm.bias", names);
    }

    [RequiresCheckpoint]
    public void Layer0MatchesFromTheReferenceEmbeddings()
    {
        using var layer = Checkpoint.Load(new ClapTextLayer(Config), "text_model.encoder.layer.0.");
        using var _ = no_grad();
        var (ids, mask) = TextCases.Tensors(1, 2);
        using var idsOwner = ids;
        using var maskOwner = mask;
        using var extended = TextMasks.Extend(mask);
        using var result = layer.forward(Stored("embeddings"), extended);
        AssertIntermediate("layer00 (from the reference embeddings)", result, Stored("layer00"));
    }

    [RequiresCheckpoint]
    public void EachLayerAndThePoolerMatchFromTheReferenceActivationBeforeIt()
    {
        using var _ = no_grad();
        var (ids, mask) = TextCases.Tensors(1, 2);
        using var idsOwner = ids;
        using var maskOwner = mask;
        using var extended = TextMasks.Extend(mask);
        for (var i = 0; i < 12; i++)
        {
            using var layer = Checkpoint.Load(new ClapTextLayer(Config), $"text_model.encoder.layer.{i}.");
            using var result = layer.forward(Stored(i == 0 ? "embeddings" : $"layer{i - 1:00}"), extended);
            AssertIntermediate($"layer{i:00}", result, Stored($"layer{i:00}"));
        }
        using var pooler = Checkpoint.Load(new ClapTextPooler(Config), "text_model.pooler.");
        using var pooled = pooler.forward(Stored("layer11"));
        AssertIntermediate("pooler", pooled, Stored("pooler"));
    }

    [RequiresCheckpoint]
    public void WholeModelReproducesEveryStoredActivation()
    {
        using var model = Checkpoint.Load(new ClapTextModel(Config), "text_model.");
        using var _ = no_grad();
        var (ids, mask) = TextCases.Tensors(1, 2);
        using var idsOwner = ids;
        using var maskOwner = mask;

        var compared = new List<string>();
        var failures = new List<string>();
        using var pooled = model.Encode(ids, mask, (name, t) =>
        {
            var expected = Stored(name);
            var r = Parity.Compare(t, expected);
            var peak = expected.abs().max().item<float>();
            var limit = 1e-4 * Math.Max(1.0, peak);
            output.WriteLine($"chain {name}: {r} peak|ref|={peak:F3} max_abs/peak={r.MaxAbs / peak:E2} limit={limit:E2}");
            if (r.MaxAbs > limit || r.CosSim < 0.9999) failures.Add($"{name}: {r} (limit {limit:E2})");
            compared.Add(name);
        });
        Assert.Equal(14, compared.Count);
        Assert.Equal("embeddings", compared[0]);
        Assert.Equal("pooler", compared[^1]);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }
}
