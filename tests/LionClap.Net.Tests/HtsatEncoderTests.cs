using LionClap.Audio.Htsat;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class HtsatEncoderTests(ITestOutputHelper output)
{
    private static readonly ClapAudioConfig Config = new();
    private const string Prefix = "audio_model.audio_encoder.";

    private static Tensor Intermediate(string name) => Fixtures.Load("htsat_intermediates.safetensors")[name];

    private Parity.Result AssertParity(string what, Tensor got, Tensor expected)
    {
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{what}: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, $"{what}: {r}");
        return r;
    }

    /// <summary>Intermediate activations: cos_sim ≥ 0.9999 and max_abs ≤ 1e-4 · max(1, peak |reference|) (late stages have outliers around 100).</summary>
    private (Parity.Result r, double peak) AssertIntermediate(string what, Tensor got, Tensor expected)
    {
        var r = Parity.Compare(got, expected);
        var peak = expected.abs().max().item<float>();
        var limit = 1e-4 * Math.Max(1.0, peak);
        output.WriteLine($"{what}: {r} peak|ref|={peak:F3} max_abs/peak={r.MaxAbs / peak:E2} limit={limit:E2}");
        Assert.True(r.MaxAbs <= limit && r.CosSim >= 0.9999, $"{what}: {r} (limit {limit:E2})");
        return (r, peak);
    }

    [Fact]
    public void ParametersAndBuffersHaveTheCheckpointLayout()
    {
        using var encoder = new ClapAudioEncoder(Config);
        var parameters = encoder.named_parameters().Select(p => p.name).ToList();
        var buffers = encoder.named_buffers().Select(b => b.name).ToList();
        Assert.Equal(221, parameters.Count);   // 12 blocks × 17, 3 merges × 3, patch embed 4, batch norm 2, final norm 2
        Assert.Equal(15, buffers.Count);       // 12 relative_position_index + running_mean/var + num_batches_tracked
        Assert.DoesNotContain(parameters, n => n.StartsWith("layers.3.downsample"));  // the last stage has no patch merging
        Assert.Contains("layers.2.blocks.5.attention.self.relative_position_bias_table", parameters);
        Assert.Contains("layers.0.downsample.reduction.weight", parameters);
        Assert.Contains("patch_embed.proj.weight", parameters);
        Assert.Contains("batch_norm.running_var", buffers);
        Assert.Contains("layers.3.blocks.1.attention.self.relative_position_index", buffers);
    }

    [Fact]
    public void StagesChangeTheTokenGridAsExpected()
    {
        using var _ = no_grad();
        using var merge = new ClapAudioStage(Config, 96, (16, 16), 2, 4, merge: true);
        merge.eval();
        using var x = randn(1, 256, 96);
        using var y = merge.forward(x);
        Assert.Equal(new long[] { 1, 64, 192 }, y.shape);

        using var last = new ClapAudioStage(Config, 96, (8, 8), 2, 4, merge: false);
        last.eval();
        using var z = last.forward(randn(1, 64, 96));
        Assert.Equal(new long[] { 1, 64, 96 }, z.shape);
    }

    [Fact]
    public void PatchMergingTakesTheFourSubGridsInTheReferenceOrder()
    {
        using var _ = no_grad();
        using var merging = new ClapAudioPatchMerging(1, (2, 2));
        foreach (var (name, p) in merging.named_parameters())
            if (name == "reduction.weight") p.copy_(tensor(new float[] { 0, 1, 0, 0, 0, 0, 1, 0 }, [2, 4])); // pick the 2nd and 3rd concatenated values

        // grid [[0, 1], [2, 3]] → concatenation [x(0,0), x(1,0), x(0,1), x(1,1)] = [0, 2, 1, 3], normalised by mean 1.5 and variance 1.25
        using var x = tensor(new float[] { 0, 1, 2, 3 }, [1, 4, 1]);
        using var y = merging.forward(x);
        var scale = Math.Sqrt(1.25 + 1e-5);
        Assert.Equal(new long[] { 1, 1, 2 }, y.shape);
        Assert.Equal((2 - 1.5) / scale, y[0, 0, 0].item<float>(), 5);
        Assert.Equal((1 - 1.5) / scale, y[0, 0, 1].item<float>(), 5);
    }

    [RequiresCheckpoint]
    public void EachStageMatchesFromTheReferenceActivationBeforeIt()
    {
        var configs = new (string input, string expected)[] { ("patch_embed", "stage0"), ("stage0", "stage1"), ("stage1", "stage2"), ("stage2", "stage3") };
        using var _ = no_grad();
        for (var i = 0; i < 4; i++)
        {
            using var stage = Checkpoint.Load(new ClapAudioStage(Config, 96 << i, (64 >> i, 64 >> i), Config.Depths[i], Config.NumAttentionHeads[i], merge: i < 3),
                $"{Prefix}layers.{i}.");
            using var result = stage.forward(Intermediate(configs[i].input));
            AssertIntermediate(configs[i].expected, result, Intermediate(configs[i].expected));
        }
    }

    [RequiresCheckpoint]
    public void FinalNormMatchesFromTheReferenceLastStage()
    {
        using var norm = Checkpoint.Load(TorchSharp.torch.nn.LayerNorm(768, Config.LayerNormEps), Prefix + "norm.");
        using var _ = no_grad();
        using var result = norm.forward(Intermediate("stage3"));
        AssertIntermediate("norm", result, Intermediate("norm"));
    }

    [RequiresCheckpoint]
    public void WholeEncoderReproducesEveryStoredIntermediateAndThePooledOutput()
    {
        using var encoder = Checkpoint.Load(new ClapAudioEncoder(Config), Prefix);
        using var _ = no_grad();
        var stored = Fixtures.Load("htsat_intermediates.safetensors");
        var compared = new List<string>();
        var failures = new List<string>();
        using var pooled = encoder.Encode(stored["input_features"], (name, t) =>
        {
            if (!stored.TryGetValue(name, out var stored_t)) return; // the fixture keeps the first two blocks of stage 0 and every stage output
            var peak = stored_t.abs().max().item<float>();
            var limit = 1e-4 * Math.Max(1.0, peak);
            var r = Parity.Compare(t, stored_t);
            output.WriteLine($"chain {name}: {r} peak|ref|={peak:F3} max_abs/peak={r.MaxAbs / peak:E2} limit={limit:E2}");
            if (r.MaxAbs > limit || r.CosSim < 0.9999) failures.Add($"{name}: {r} (limit {limit:E2})");
            compared.Add(name);
        });
        Assert.Equal(new[] { "image", "patch_embed", "stage0_block0", "stage0_block1", "stage0", "stage1", "stage2", "stage3", "norm" }, compared);
        Assert.True(failures.Count == 0, string.Join("; ", failures));

        var reference = Fixtures.Load("audio_embeddings.safetensors")["audio_encoder_output"][0..1];
        AssertParity("pooled (case 0)", pooled, reference);
    }

    [RequiresCheckpoint]
    public void PooledOutputMatchesForTheFiveFixtureCases()
    {
        using var encoder = Checkpoint.Load(new ClapAudioEncoder(Config), Prefix);
        using var _ = no_grad();
        var input = Fixtures.Load("frontend.safetensors")["input_features"];
        var expected = Fixtures.Load("audio_embeddings.safetensors")["audio_encoder_output"];
        using var pooled = encoder.forward(input);
        Assert.Equal(new long[] { 5, 768 }, pooled.shape);
        for (var i = 0; i < 5; i++) AssertParity($"pooled case {i}", pooled[i], expected[i]);
        AssertParity("pooled batch", pooled, expected);
    }
}
