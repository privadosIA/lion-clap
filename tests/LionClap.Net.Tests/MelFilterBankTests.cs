using LionClap.Audio;
using Xunit.Abstractions;

namespace LionClap.Tests;

public class MelFilterBankTests(ITestOutputHelper output)
{
    private static double MaxAbs(double[] got, float[] expected)
    {
        Assert.Equal(expected.Length, got.Length);
        var max = 0.0;
        for (var i = 0; i < got.Length; i++) max = Math.Max(max, Math.Abs(got[i] - expected[i]));
        return max;
    }

    private static float[] Fixture(string name)
    {
        var t = Fixtures.Load("frontend.safetensors")[name];
        Assert.Equal(new long[] { 513, 64 }, t.shape);
        return t.data<float>().ToArray();
    }

    [Fact]
    public void EqualsTheReferenceSlaneyBank()
    {
        var diff = MaxAbs(MelFilterBank.Create(), Fixture("mel_filters_slaney"));
        output.WriteLine($"mel bank vs mel_filters_slaney: max_abs={diff:E3}");
        Assert.True(diff <= 1e-6, $"max_abs {diff:E3}");
    }

    [Fact]
    public void DiffersClearlyFromTheHtkBank()
    {
        var diff = MaxAbs(MelFilterBank.Create(), Fixture("mel_filters_htk"));
        output.WriteLine($"mel bank vs mel_filters_htk: max_abs={diff:E3}");
        Assert.True(diff > 1e-2, $"max_abs {diff:E3}");
    }

    [Fact]
    public void ScaleConversionsAreInverse()
    {
        foreach (var hz in new[] { 0.0, 50.0, 999.0, 1000.0, 4321.0, 14000.0 })
            Assert.Equal(hz, MelFilterBank.MelToHertz(MelFilterBank.HertzToMel(hz)), 9);
    }
}
