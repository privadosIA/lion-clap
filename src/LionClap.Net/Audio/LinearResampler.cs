namespace LionClap.Audio;

/// <summary>
/// The resampling of the reference service: linear interpolation (<c>numpy.interp</c>) of the whole signal onto <c>int(n · target / rate)</c> points spread evenly
/// from the first to the last sample (<c>numpy.linspace(0, n−1, m)</c>), computed in double precision and rounded to float32. It is not a band-limited resampler,
/// and it is what the Python service does, so the numbers agree.
/// </summary>
public static class LinearResampler
{
    /// <summary>Number of samples after resampling <paramref name="samples"/> from <paramref name="rate"/> to <paramref name="targetRate"/>.</summary>
    public static int OutputLength(int samples, int rate, int targetRate) => checked((int)((double)((long)samples * targetRate) / rate));

    public static float[] Resample(ReadOnlySpan<float> samples, int rate, int targetRate)
    {
        if (samples.Length == 0) return [];
        if (rate == targetRate) return samples.ToArray();

        var length = OutputLength(samples.Length, rate, targetRate);
        var result = new float[length];
        if (length == 0) return result;
        var last = samples.Length - 1;
        var step = length > 1 ? (double)last / (length - 1) : 0.0;          // numpy.linspace: start + i · step, the end point set exactly
        for (var i = 0; i < length; i++)
        {
            var x = i == length - 1 && length > 1 ? last : i * step;
            var j = (int)Math.Floor(x);
            if (j >= last || x <= j)
            {
                result[i] = samples[Math.Min(j, last)];                      // numpy.interp returns the sample itself on an exact hit (x <= j is x == j, because j = floor(x) <= x) and at the end
                continue;
            }
            double left = samples[j], right = samples[j + 1];
            result[i] = (float)((right - left) / 1.0 * (x - j) + left);       // slope · (x − xp[j]) + fp[j], with xp[j + 1] − xp[j] = 1
        }
        return result;
    }
}
