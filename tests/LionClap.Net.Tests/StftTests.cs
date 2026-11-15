using LionClap.Audio;
using Xunit.Abstractions;

namespace LionClap.Tests;

public class StftTests(ITestOutputHelper output)
{
    private static double[] RandomSignal(int n, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
    }

    private static (double re, double im) NaiveDft(double[] x, int k)
    {
        double re = 0, im = 0;
        for (var n = 0; n < x.Length; n++)
        {
            var a = -2.0 * Math.PI * ((long)k * n % x.Length) / x.Length;
            re += x[n] * Math.Cos(a);
            im += x[n] * Math.Sin(a);
        }
        return (re, im);
    }

    [Fact]
    public void FftEqualsNaiveDft()
    {
        var x = RandomSignal(1024, 1);
        var re = (double[])x.Clone();
        var im = new double[1024];
        new Fft(1024).Transform(re, im);

        double worst = 0, scale = 0;
        for (var k = 0; k < 1024; k++)
        {
            var (er, ei) = NaiveDft(x, k);
            worst = Math.Max(worst, Math.Max(Math.Abs(re[k] - er), Math.Abs(im[k] - ei)));
            scale = Math.Max(scale, Math.Max(Math.Abs(er), Math.Abs(ei)));
        }
        output.WriteLine($"fft vs naive dft: max_abs={worst:E3} relative to max={worst / scale:E3}");
        Assert.True(worst / scale <= 1e-9);
    }

    [Fact]
    public void FftRejectsSizesThatAreNotPowersOfTwo() => Assert.Throws<ArgumentException>(() => new Fft(1000));

    [Fact]
    public void TenSecondsGiveTheReferenceFrameCount()
    {
        Assert.Equal(1001, PowerStft.FrameCount(480000));
        var power = new PowerStft().Power(RandomSignal(480000, 2));
        Assert.Equal(1001 * 513, power.Length);
    }

    [Theory]
    [InlineData(0)]    // starts in the left reflection
    [InlineData(1)]
    [InlineData(500)]
    [InlineData(1000)] // ends in the right reflection
    public void FramesEqualNaiveDftOfTheReflectPaddedWindow(int frame)
    {
        var x = RandomSignal(480000, 3);
        var power = new PowerStft().Power(x);

        var padded = new double[x.Length + 1024];
        for (var i = 0; i < 512; i++) padded[i] = x[512 - i];
        Array.Copy(x, 0, padded, 512, x.Length);
        for (var j = 0; j < 512; j++) padded[512 + x.Length + j] = x[x.Length - 2 - j];

        var win = Enumerable.Range(0, 1024).Select(i => 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / 1024)).ToArray();
        var seg = Enumerable.Range(0, 1024).Select(i => padded[frame * 480 + i] * win[i]).ToArray();
        double worst = 0, scale = 0;
        for (var k = 0; k <= 512; k += 7)
        {
            var (er, ei) = NaiveDft(seg, k);
            var expected = er * er + ei * ei;
            worst = Math.Max(worst, Math.Abs(power[frame * 513 + k] - expected));
            scale = Math.Max(scale, expected);
        }
        output.WriteLine($"frame {frame}: max_abs={worst:E3} relative to max={worst / scale:E3}");
        Assert.True(worst / scale <= 1e-9);
    }
}
