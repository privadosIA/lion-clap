using LionClap.Text;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class ClapTextEmbeddingsTests(ITestOutputHelper output)
{
    private static readonly ClapTextConfig Config = new();

    private static Tensor Ids(long[,] rows) => tensor(rows);

    [Fact]
    public void PositionsStartAfterThePaddingIndexAndPaddingKeepsIt()
    {
        using var ids = Ids(new long[,] { { 0, 5, 6, 2, 1, 1 } });            // trailing padding
        using var trailing = ClapTextEmbeddings.PositionIds(ids, 1);
        Assert.Equal(new long[] { 2, 3, 4, 5, 1, 1 }, trailing.data<long>().ToArray());

        using var interior = Ids(new long[,] { { 0, 5, 1, 6, 2 } });          // a pad inside does not consume a position
        using var inner = ClapTextEmbeddings.PositionIds(interior, 1);
        Assert.Equal(new long[] { 2, 3, 1, 4, 5 }, inner.data<long>().ToArray());

        using var leading = Ids(new long[,] { { 1, 1, 0, 5, 2 } });           // left padding
        using var left = ClapTextEmbeddings.PositionIds(leading, 1);
        Assert.Equal(new long[] { 1, 1, 2, 3, 4 }, left.data<long>().ToArray());
    }

    [Fact]
    public void ValidationNamesTheProblem()
    {
        var mask = ones(1, 3, dtype: ScalarType.Int64);
        using var ok = Ids(new long[,] { { 0, 5, 2 } });
        TextInputs.Validate(ok, mask, Config); // accepted

        Assert.Contains("2 dimensions", Assert.Throws<ArgumentException>(() => TextInputs.Validate(zeros(3, dtype: ScalarType.Int64), null, Config)).Message);
        Assert.Contains("int64", Assert.Throws<ArgumentException>(() => TextInputs.Validate(zeros(1, 3, dtype: ScalarType.Int32), null, Config)).Message);
        Assert.Contains("at least one row", Assert.Throws<ArgumentException>(() => TextInputs.Validate(zeros(0, 3, dtype: ScalarType.Int64), null, Config)).Message);
        Assert.Contains("1 to 512", Assert.Throws<ArgumentException>(() => TextInputs.Validate(zeros(1, 0, dtype: ScalarType.Int64), null, Config)).Message);
        Assert.Contains("513", Assert.Throws<ArgumentException>(() => TextInputs.Validate(zeros(1, 513, dtype: ScalarType.Int64), null, Config)).Message);

        using var high = Ids(new long[,] { { 0, 50265, 2 } });
        var ex = Assert.Throws<ArgumentException>(() => TextInputs.Validate(high, null, Config));
        Assert.Contains("50265", ex.Message);
        Assert.Contains("vocabulary", ex.Message);
        using var negative = Ids(new long[,] { { 0, -3, 2 } });
        Assert.Contains("-3", Assert.Throws<ArgumentException>(() => TextInputs.Validate(negative, null, Config)).Message);

        var wrongShape = Assert.Throws<ArgumentException>(() => TextInputs.Validate(ok, ones(1, 4, dtype: ScalarType.Int64), Config));
        Assert.Contains("[1, 4]", wrongShape.Message);
        Assert.Contains("[1, 3]", wrongShape.Message);
        Assert.Contains("0 or 1", Assert.Throws<ArgumentException>(() => TextInputs.Validate(ok, full(new long[] { 1, 3 }, 2, dtype: ScalarType.Int64), Config)).Message);
        Assert.Contains("row 0 is all zero", Assert.Throws<ArgumentException>(() => TextInputs.Validate(ok, zeros(1, 3, dtype: ScalarType.Int64), Config)).Message);
    }

    [Fact]
    public void ShapeIsBatchByLengthByHidden()
    {
        using var embeddings = new ClapTextEmbeddings(Config);
        embeddings.eval();
        using var _ = no_grad();
        using var ids = Ids(new long[,] { { 0, 5, 6, 2, 1 }, { 0, 7, 2, 1, 1 } });
        using var result = embeddings.forward(ids);
        Assert.Equal(new long[] { 2, 5, 768 }, result.shape);
    }

    [RequiresCheckpoint]
    public void EmbeddingsOutputOfTheGoldenCasesMatchesTheReference()
    {
        using var embeddings = Checkpoint.Load(new ClapTextEmbeddings(Config), "text_model.embeddings.");
        using var _ = no_grad();
        var (ids, _) = TextCases.Tensors(1, 2);
        using var idsOwner = ids;
        using var result = embeddings.forward(ids);
        var expected = Fixtures.Load("roberta_intermediates.safetensors")["embeddings"];

        var r = Parity.Compare(result, expected);
        var peak = expected.abs().max().item<float>();
        var limit = 1e-4 * Math.Max(1.0, peak);
        output.WriteLine($"embeddings: {r} peak|ref|={peak:F3} limit={limit:E2}");
        Assert.True(r.MaxAbs <= limit && r.CosSim >= 0.9999, $"{r} (limit {limit:E2})");
    }
}
