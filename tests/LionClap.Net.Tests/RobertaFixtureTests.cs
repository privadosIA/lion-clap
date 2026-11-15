namespace LionClap.Tests;

public class RobertaFixtureTests
{
    [Fact]
    public void IntermediatesHaveTheExpectedShapes()
    {
        var f = Fixtures.Load("roberta_intermediates.safetensors");
        var expected = new List<string> { "embeddings", "pooler" };
        expected.AddRange(Enumerable.Range(0, 12).Select(i => $"layer{i:00}"));
        Assert.Equal(expected.OrderBy(k => k), f.Keys.OrderBy(k => k));
        foreach (var k in expected)
            Assert.Equal(k == "pooler" ? new long[] { 2, 768 } : new long[] { 2, 77, 768 }, f[k].shape);
    }

    [Fact]
    public void PoolerEqualsTheStoredPooledOutputOfCases1And2()
    {
        var pooler = Fixtures.Load("roberta_intermediates.safetensors")["pooler"];
        var stored = Fixtures.Load("text_embeddings.safetensors")["text_pooler_output"];
        for (var row = 0; row < 2; row++)
        {
            var got = pooler[row].data<float>().ToArray();
            var want = stored[row + 1].data<float>().ToArray();
            Assert.True(got.Zip(want, (a, b) => Math.Abs(a - b)).Max() <= 1e-5);
        }
    }
}
