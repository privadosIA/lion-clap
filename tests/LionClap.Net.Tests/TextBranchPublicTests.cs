using LionClap.Audio;
using LionClap.Audio.Htsat;
using LionClap.Text;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

/// <summary>The real text branch and tokenizer, built once per class and only when a gated test runs.</summary>
public sealed class TextBranchFixture : IDisposable
{
    private readonly Lazy<ClapTextBranch> _branch = new(() => ClapTextBranch.Load(Checkpoint.Dir!));
    private readonly Lazy<ClapTextTokenizer> _tokenizer = new(() => new ClapTextTokenizer(Path.Combine(Checkpoint.Dir!, "tokenizer.json")));

    public ClapTextBranch Branch => _branch.Value;

    public ClapTextTokenizer Tokenizer => _tokenizer.Value;

    public void Dispose()
    {
        if (_branch.IsValueCreated) _branch.Value.Dispose();
        if (_tokenizer.IsValueCreated) _tokenizer.Value.Dispose();
    }
}

public class TextPublicBehaviorTests(TextBranchFixture fixture, ITestOutputHelper output) : IClassFixture<TextBranchFixture>
{
    private static IReadOnlyList<string> Texts => TextCases.All.Select(c => c.Text).ToList();

    private static Tensor Stored(string file, string name) => Fixtures.Load(file)[name];

    private void AssertClose(string what, Tensor got, Tensor expected, double limit = 1e-4)
    {
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{what}: {r}");
        Assert.True(r.MaxAbs <= limit && r.CosSim >= 0.9999, $"{what}: {r}");
    }

    [RequiresCheckpoint]
    public void StoredIdsGiveTheReferencePooledOutputAndEmbeddingsWithUnitNorm()
    {
        var (ids, mask) = TextCases.Tensors(0, 1, 2, 3, 4);
        using var idsOwner = ids;
        using var maskOwner = mask;
        using var result = fixture.Branch.Forward(ids, mask);
        Assert.Equal(new long[] { 5, 768 }, result.Pooled.shape);
        Assert.Equal(new long[] { 5, 512 }, result.Embeddings.shape);

        var pooled = Stored("text_embeddings.safetensors", "text_pooler_output");
        var embeddings = Stored("text_embeddings.safetensors", "text_embeddings");
        for (var i = 0; i < 5; i++)
        {
            AssertClose($"pooled text {i}", result.Pooled[i], pooled[i]);
            AssertClose($"embedding text {i}", result.Embeddings[i], embeddings[i]);
        }
        AssertClose("pooled batch", result.Pooled, pooled);
        AssertClose("embedding batch", result.Embeddings, embeddings);

        using var norms = result.Embeddings.norm(dim: 1);
        var worst = (norms - 1).abs().max().item<float>();
        output.WriteLine($"unit norm: worst |norm-1| = {worst:E2}");
        Assert.True(worst <= 1e-6, $"{worst:E2}");
    }

    [RequiresCheckpoint]
    public void RawStringsGiveTheReferenceEmbeddings()
    {
        using var result = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(Texts, 77));
        AssertClose("embeddings from strings", result.Embeddings, Stored("text_embeddings.safetensors", "text_embeddings"));
    }

    [RequiresCheckpoint]
    public void PaddingDoesNotChangeTheEmbedding()
    {
        var text = TextCases.All[1].Text;
        using var alone = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(new[] { text }));            // unpadded (9 tokens)
        using var padded = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(new[] { text }, 77));      // padded to 77
        using var mixed = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(new[] { text, TextCases.All[2].Text })); // padded to the longer text
        AssertClose("unpadded vs padded 77", alone.Embeddings, padded.Embeddings);
        AssertClose("unpadded vs inside a longer batch", alone.Embeddings, mixed.Embeddings[0..1]);
    }

    [RequiresCheckpoint]
    public void EachItemAloneAgreesWithItsBatchResult()
    {
        using var batch = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(Texts, 77));
        for (var i = 0; i < 5; i++)
        {
            using var alone = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(new[] { Texts[i] }, 77));
            var worst = (alone.Embeddings - batch.Embeddings[i..(i + 1)]).abs().max().item<float>();
            var worstPooled = (alone.Pooled - batch.Pooled[i..(i + 1)]).abs().max().item<float>();
            output.WriteLine($"item {i}: embeddings {worst:E2}, pooled {worstPooled:E2}");
            Assert.True(worst <= 1e-4 && worstPooled <= 1e-4, $"item {i}: {worst:E2} / {worstPooled:E2}");
        }
    }

    [RequiresCheckpoint]
    public void TheSameInputGivesBitIdenticalOutputs()
    {
        var batch = fixture.Tokenizer.EncodeBatch(Texts, 77);
        using var first = fixture.Branch.Forward(batch);
        using var second = fixture.Branch.Forward(batch);
        Assert.Equal(0f, (first.Pooled - second.Pooled).abs().max().item<float>());
        Assert.Equal(0f, (first.Embeddings - second.Embeddings).abs().max().item<float>());
    }

    [RequiresCheckpoint]
    public void ATextOf512TokensWorksAnd513IsRejected()
    {
        var huge = string.Join(' ', Enumerable.Repeat("sound", 600));
        using var result = fixture.Branch.Forward(fixture.Tokenizer.EncodeBatch(new[] { huge }, 512));
        Assert.Equal(new long[] { 1, 512 }, result.Embeddings.shape);
        Assert.True(float.IsFinite(result.Embeddings.sum().item<float>()));
        Assert.True(Math.Abs(result.Embeddings.norm().item<float>() - 1) <= 1e-6);

        using var ids = zeros(1, 513, dtype: ScalarType.Int64);
        using var mask = ones(1, 513, dtype: ScalarType.Int64);
        var ex = Assert.Throws<ArgumentException>(() => fixture.Branch.Forward(ids, mask));
        Assert.Contains("512", ex.Message);
    }
}

public class CrossModalTests(TextBranchFixture text, ITestOutputHelper output) : IClassFixture<TextBranchFixture>
{
    [RequiresCheckpoint]
    public void WaveformsAndStringsReproduceTheStoredSimilarityMatrix()
    {
        using var audio = ClapAudioBranch.Load(Checkpoint.Dir!);
        var frontend = Fixtures.Load("frontend.safetensors");
        var waves = Enumerable.Range(0, 5).Select(i => frontend[$"waveform_{i}"].data<float>().ToArray()).ToList();
        using var features = new LogMelExtractor().ExtractBatch(waves);
        using var a = audio.Forward(features);
        using var t = text.Branch.Forward(text.Tokenizer.EncodeBatch(TextCases.All.Select(c => c.Text).ToList(), 77));

        using var similarity = a.Embeddings.matmul(t.Embeddings.transpose(0, 1));
        var expected = Fixtures.Load("similarity.safetensors")["similarity"];
        var r = Parity.Compare(similarity, expected);
        output.WriteLine($"similarity 5x5: {r}");
        Assert.Equal(new long[] { 5, 5 }, similarity.shape);
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, r.ToString());
    }
}
