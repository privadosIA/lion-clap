using LionClap.Audio;
using LionClap.Audio.Htsat;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

/// <summary>The real branch is built once per class, and only when a gated test runs.</summary>
public sealed class BranchFixture : IDisposable
{
    private readonly Lazy<ClapAudioBranch> _branch = new(() => ClapAudioBranch.Load(Checkpoint.Dir!));

    public ClapAudioBranch Branch => _branch.Value;

    public void Dispose()
    {
        if (_branch.IsValueCreated) _branch.Value.Dispose();
    }
}

public class HtsatEmbeddingsTests(BranchFixture fixture, ITestOutputHelper output) : IClassFixture<BranchFixture>
{
    private static Tensor FrontendInput => Fixtures.Load("frontend.safetensors")["input_features"];

    private static Tensor ReferenceEmbeddings => Fixtures.Load("audio_embeddings.safetensors")["audio_embeddings"];

    private void AssertEmbedding(string what, Tensor got, Tensor expected)
    {
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{what}: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, $"{what}: {r}");
    }

    [RequiresCheckpoint]
    public void FixtureFeaturesGiveTheReferenceEmbeddingsWithUnitNorm()
    {
        using var result = fixture.Branch.Forward(FrontendInput);
        Assert.Equal(new long[] { 5, 768 }, result.Pooled.shape);
        Assert.Equal(new long[] { 5, 512 }, result.Embeddings.shape);

        AssertEmbedding("pooled batch", result.Pooled, Fixtures.Load("audio_embeddings.safetensors")["audio_encoder_output"]);
        for (var i = 0; i < 5; i++) AssertEmbedding($"embedding case {i}", result.Embeddings[i], ReferenceEmbeddings[i]);
        AssertEmbedding("embedding batch", result.Embeddings, ReferenceEmbeddings);

        using var norms = result.Embeddings.norm(dim: 1);
        var worst = (norms - 1).abs().max().item<float>();
        output.WriteLine($"unit norm: worst |norm-1| = {worst:E2}");
        Assert.True(worst <= 1e-6, $"{worst:E2}");
    }

    [RequiresCheckpoint]
    public void WaveformsThroughTheFrontendGiveTheReferenceEmbeddings()
    {
        var frontend = Fixtures.Load("frontend.safetensors");
        var waves = Enumerable.Range(0, 5).Select(i => frontend[$"waveform_{i}"].data<float>().ToArray()).ToList();
        using var features = new LogMelExtractor().ExtractBatch(waves);
        using var result = fixture.Branch.Forward(features);

        for (var i = 0; i < 5; i++) AssertEmbedding($"waveform case {i}", result.Embeddings[i], ReferenceEmbeddings[i]);
        AssertEmbedding("waveforms batch", result.Embeddings, ReferenceEmbeddings);
    }

    [RequiresCheckpoint]
    public void EachItemAloneAgreesWithItsBatchResult()
    {
        var input = FrontendInput;
        using var batch = fixture.Branch.Forward(input);
        for (var i = 0; i < 5; i++)
        {
            using var item = input[i..(i + 1)].contiguous();
            using var alone = fixture.Branch.Forward(item);
            var worst = (alone.Embeddings - batch.Embeddings[i..(i + 1)]).abs().max().item<float>();
            var worstPooled = (alone.Pooled - batch.Pooled[i..(i + 1)]).abs().max().item<float>();
            output.WriteLine($"item {i}: embeddings {worst:E2}, pooled {worstPooled:E2}");
            Assert.True(worst <= 1e-4 && worstPooled <= 1e-4, $"item {i}: {worst:E2} / {worstPooled:E2}");
        }
    }

    [RequiresCheckpoint]
    public void TheSameInputGivesBitIdenticalOutputs()
    {
        var input = FrontendInput;
        using var first = fixture.Branch.Forward(input);
        using var second = fixture.Branch.Forward(input);
        Assert.Equal(0f, (first.Pooled - second.Pooled).abs().max().item<float>());
        Assert.Equal(0f, (first.Embeddings - second.Embeddings).abs().max().item<float>());
    }

    [RequiresCheckpoint]
    public void ShorterTimeAxesAreAccepted()
    {
        using var short100 = FrontendInput[0..1, .., 0..100, ..].contiguous(); // 100 frames are stretched to 1024 like the reference does
        using var result = fixture.Branch.Forward(short100);
        Assert.Equal(new long[] { 1, 512 }, result.Embeddings.shape);
        Assert.True(float.IsFinite(result.Embeddings.sum().item<float>()));
    }
}
