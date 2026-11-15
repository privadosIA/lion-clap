namespace LionClap.Tests;

public class HtsatFixtureTests
{
    [Fact]
    public void IntermediatesHaveTheExpectedShapesAndMatchTheFrontendInput()
    {
        var f = Fixtures.Load("htsat_intermediates.safetensors");
        var expected = new Dictionary<string, long[]>
        {
            ["input_features"] = [1, 1, 1001, 64],
            ["image"] = [1, 1, 256, 256],
            ["patch_embed"] = [1, 4096, 96],
            ["stage0_block0"] = [1, 4096, 96],
            ["stage0_block1"] = [1, 4096, 96],
            ["stage0"] = [1, 1024, 192],
            ["stage1"] = [1, 256, 384],
            ["stage2"] = [1, 64, 768],
            ["stage3"] = [1, 64, 768],
            ["norm"] = [1, 64, 768],
        };
        Assert.Equal(expected.Keys.OrderBy(k => k), f.Keys.OrderBy(k => k));
        foreach (var (k, shape) in expected) Assert.Equal(shape, f[k].shape);

        var frontend = Fixtures.Load("frontend.safetensors")["input_features"][0].contiguous().data<float>().ToArray();
        Assert.Equal(frontend, f["input_features"].contiguous().data<float>().ToArray());
    }

    [Fact]
    public void MeanOfTheFinalTokensIsThePooledOutput()
    {
        var norm = Fixtures.Load("htsat_intermediates.safetensors")["norm"]; // [1, 64, 768]
        var pooled = Fixtures.Load("audio_embeddings.safetensors")["audio_encoder_output"][0].data<float>().ToArray();
        var mean = norm.mean([1L]).contiguous().data<float>().ToArray();
        var worst = mean.Zip(pooled, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(worst <= 1e-5, $"max_abs {worst:E3}");
    }
}
