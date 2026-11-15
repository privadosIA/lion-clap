namespace LionClap.Audio;

/// <summary>
/// Power spectrogram as transformers' <c>spectrogram(..., power=2.0)</c> computes it for CLAP: periodic Hann window of 1024,
/// hop 480, centered frames over a signal reflect-padded by 512 on both sides, bins 0..512.
/// </summary>
internal sealed class PowerStft
{
    public const int FftSize = 1024;
    public const int Hop = 480;
    public const int Bins = FftSize / 2 + 1;

    private readonly Fft _fft = new(FftSize);
    private readonly double[] _window = new double[FftSize];

    public PowerStft()
    {
        for (var i = 0; i < FftSize; i++) _window[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / FftSize);
    }

    public static int FrameCount(int samples) => 1 + samples / Hop;

    /// <summary>Row-major <c>[frames, 513]</c> power values. The signal must be longer than 512 samples (reflect padding).</summary>
    public double[] Power(ReadOnlySpan<double> signal)
    {
        var n = signal.Length;
        const int pad = FftSize / 2;
        if (n <= pad) throw new ArgumentException($"the signal needs more than {pad} samples", nameof(signal));
        var frames = FrameCount(n);
        var power = new double[frames * Bins];
        var re = new double[FftSize];
        var im = new double[FftSize];
        for (var t = 0; t < frames; t++)
        {
            var first = t * Hop - pad; // index into the unpadded signal of the frame's first sample
            for (var i = 0; i < FftSize; i++)
            {
                var k = first + i;
                if (k < 0) k = -k;                    // reflect around the first sample
                else if (k >= n) k = 2 * (n - 1) - k; // reflect around the last sample
                re[i] = signal[k] * _window[i];
                im[i] = 0.0;
            }
            _fft.Transform(re, im);
            var row = t * Bins;
            for (var b = 0; b < Bins; b++) power[row + b] = re[b] * re[b] + im[b] * im[b];
        }
        return power;
    }
}
