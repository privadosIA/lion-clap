using LionClap.Audio.Htsat;

namespace LionClap.Tests;

public class ClapAudioConfigTests
{
    private static string TempConfig(string audioConfigJson)
    {
        var path = Path.Combine(TestTemp.Dir(), "config.json");
        File.WriteAllText(path, $"{{\"audio_config\": {audioConfigJson}}}");
        return path;
    }

    [Fact]
    public void DefaultsDescribeHtsatTiny()
    {
        var c = new ClapAudioConfig();
        Assert.Equal(1024, c.MaxFrames);
        Assert.Equal(4, c.FreqRatio);
        Assert.Equal(96 * 8, c.HiddenSize);
    }

    [Fact]
    public void FusionConfigurationIsRejected()
    {
        var ex = Assert.Throws<NotSupportedException>(() => ClapAudioConfig.FromFile(TempConfig("{\"enable_fusion\": true}")));
        Assert.Contains("fusion", ex.Message);
    }

    [Fact]
    public void PatchStrideAsPairIsReadAndNonSquareIsRejected()
    {
        Assert.Equal(4, ClapAudioConfig.FromFile(TempConfig("{\"patch_stride\": [4, 4]}")).PatchStride);
        Assert.Throws<NotSupportedException>(() => ClapAudioConfig.FromFile(TempConfig("{\"patch_stride\": [4, 2]}")));
    }

    [Fact]
    public void MissingAudioConfigIsReported() =>
        Assert.Throws<InvalidDataException>(() => ClapAudioConfig.FromFile(TempConfigRaw("{}")));

    private static string TempConfigRaw(string json)
    {
        var path = Path.Combine(TestTemp.Dir(), "config.json");
        File.WriteAllText(path, json);
        return path;
    }

    [RequiresCheckpoint]
    public void DefaultsEqualTheRealConfigJson()
    {
        var real = ClapAudioConfig.FromFile(Checkpoint.ConfigFile);
        var d = new ClapAudioConfig();
        Assert.Equal(d.SpecSize, real.SpecSize);
        Assert.Equal(d.NumMelBins, real.NumMelBins);
        Assert.Equal(d.PatchSize, real.PatchSize);
        Assert.Equal(d.PatchStride, real.PatchStride);
        Assert.Equal(d.PatchEmbedsHiddenSize, real.PatchEmbedsHiddenSize);
        Assert.Equal(d.Depths, real.Depths);
        Assert.Equal(d.NumAttentionHeads, real.NumAttentionHeads);
        Assert.Equal(d.WindowSize, real.WindowSize);
        Assert.Equal(d.MlpRatio, real.MlpRatio);
        Assert.Equal(d.QkvBias, real.QkvBias);
        Assert.Equal(d.LayerNormEps, real.LayerNormEps);
        Assert.Equal(d.EnablePatchLayerNorm, real.EnablePatchLayerNorm);
        Assert.Equal(d.HiddenSize, real.HiddenSize);
        Assert.Equal(d.ProjectionDim, real.ProjectionDim);
    }
}
