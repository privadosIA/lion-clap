using LionClap.Audio;

namespace LionClap.Tests;

public class LinearResamplerTests
{
    [Theory]
    [InlineData(192000, 16000, 576000)]
    [InlineData(66150, 22050, 144000)]
    [InlineData(66150, 44100, 72000)]
    [InlineData(100, 44100, 108)]       // int(100 · 48000 / 44100) = int(108.84) = 108
    [InlineData(1, 8000, 6)]
    public void LengthIsTheTruncatedRatio(int samples, int rate, int expected)
    {
        Assert.Equal(expected, LinearResampler.OutputLength(samples, rate, 48000));
        Assert.Equal(expected, LinearResampler.Resample(new float[samples], rate, 48000).Length);
    }

    [Fact]
    public void ARampIsReproducedExactlyByLinearInterpolation()
    {
        var ramp = Enumerable.Range(0, 1000).Select(i => (float)i).ToArray();
        var up = LinearResampler.Resample(ramp, 16000, 48000);
        Assert.Equal(3000, up.Length);
        var step = 999.0 / 2999.0;                                  // numpy.linspace(0, 999, 3000)
        for (var i = 0; i < up.Length; i += 137) Assert.Equal((float)(i * step), up[i], 3);
        Assert.Equal(0f, up[0]);
        Assert.Equal(999f, up[^1]);                                 // the end point is exact
    }

    [Fact]
    public void TheSameRateReturnsACopy()
    {
        var x = new float[] { 1, 2, 3 };
        var y = LinearResampler.Resample(x, 48000, 48000);
        Assert.Equal(x, y);
        Assert.NotSame(x, y);
    }

    [Fact]
    public void DownsamplingKeepsTheFirstAndLastSamples()
    {
        var x = Enumerable.Range(0, 96000).Select(i => (float)Math.Sin(i * 0.01)).ToArray();
        var y = LinearResampler.Resample(x, 96000, 48000);
        Assert.Equal(48000, y.Length);
        Assert.Equal(x[0], y[0]);
        Assert.Equal(x[^1], y[^1]);
    }

    [Fact]
    public void AnEmptyInputGivesAnEmptyOutput() => Assert.Empty(LinearResampler.Resample([], 16000, 48000));
}
