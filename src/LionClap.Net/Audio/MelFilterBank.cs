namespace LionClap.Audio;

/// <summary>
/// The Slaney mel filter bank of the CLAP feature extractor, computed in double precision with the formulas of
/// transformers' <c>mel_filter_bank(norm="slaney", mel_scale="slaney")</c>. Row-major <c>[bins, mels]</c> (513 × 64).
/// </summary>
public static class MelFilterBank
{
    public const int SampleRate = 48000;
    public const int Bins = 513;
    public const int Mels = 64;
    public const double MinFrequency = 50.0;
    public const double MaxFrequency = 14000.0;

    private const double MinLogHertz = 1000.0;
    private const double MinLogMel = 15.0;

    /// <summary>Hz → mel, Slaney scale: linear below 1 kHz, logarithmic above.</summary>
    public static double HertzToMel(double hz) =>
        hz >= MinLogHertz ? MinLogMel + Math.Log(hz / MinLogHertz) * (27.0 / Math.Log(6.4)) : 3.0 * hz / 200.0;

    /// <summary>mel → Hz, the inverse of <see cref="HertzToMel"/>.</summary>
    public static double MelToHertz(double mel) =>
        mel >= MinLogMel ? MinLogHertz * Math.Exp(Math.Log(6.4) / 27.0 * (mel - MinLogMel)) : 200.0 * mel / 3.0;

    /// <summary>Builds the bank: triangular filters between the mel-spaced edges, each scaled by 2 / (upper edge − lower edge).</summary>
    public static double[] Create()
    {
        var edges = Linspace(HertzToMel(MinFrequency), HertzToMel(MaxFrequency), Mels + 2);
        for (var i = 0; i < edges.Length; i++) edges[i] = MelToHertz(edges[i]);
        var fft = Linspace(0.0, SampleRate / 2, Bins);

        var bank = new double[Bins * Mels];
        for (var m = 0; m < Mels; m++)
        {
            var lower = edges[m];
            var center = edges[m + 1];
            var upper = edges[m + 2];
            var enorm = 2.0 / (upper - lower);
            for (var b = 0; b < Bins; b++)
            {
                var down = -(lower - fft[b]) / (center - lower);
                var up = (upper - fft[b]) / (upper - center);
                bank[b * Mels + m] = Math.Max(0.0, Math.Min(down, up)) * enorm;
            }
        }
        return bank;
    }

    /// <summary>numpy.linspace with the end point included (the last value is exactly <paramref name="stop"/>).</summary>
    private static double[] Linspace(double start, double stop, int count)
    {
        var r = new double[count];
        var step = (stop - start) / (count - 1);
        for (var i = 0; i < count; i++) r[i] = start + i * step;
        r[count - 1] = stop;
        return r;
    }
}
