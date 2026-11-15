using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Audio;

/// <summary>
/// Waveform → the log-mel <c>input_features</c> of the CLAP audio encoder, as transformers' <c>ClapFeatureExtractor</c> computes them
/// (<c>truncation=rand_trunc</c>, <c>padding=repeatpad</c>, Slaney bank), all in double precision and rounded to float32 at the end.
/// The random crop of <c>rand_trunc</c> does not exist here: a waveform longer than 10 s is cut at an explicit offset.
/// Immutable and thread-safe.
/// </summary>
public sealed class LogMelExtractor
{
    public const int SampleRate = MelFilterBank.SampleRate;
    public const int MaxSamples = 10 * SampleRate;
    public const int Frames = 1001;
    public const int Mels = MelFilterBank.Mels;
    public const int FeatureCount = Frames * Mels;

    private const double MelFloor = 1e-10;

    private readonly PowerStft _stft = new();
    private readonly double[] _bank = MelFilterBank.Create();
    private readonly int[] _firstBin = new int[Mels];
    private readonly int[] _endBin = new int[Mels];

    public LogMelExtractor()
    {
        // Each triangular filter is non-zero on a few bins only; summing just those adds the same terms in the same order.
        for (var m = 0; m < Mels; m++)
        {
            _firstBin[m] = PowerStft.Bins;
            _endBin[m] = 0;
            for (var b = 0; b < PowerStft.Bins; b++)
            {
                if (_bank[b * Mels + m] == 0.0) continue;
                _firstBin[m] = Math.Min(_firstBin[m], b);
                _endBin[m] = b + 1;
            }
        }
    }

    /// <summary>
    /// Log-mel features of one waveform (mono, 48 kHz): row-major <c>[1001, 64]</c>, as <c>input_features[0, 0]</c>.
    /// A waveform longer than 10 s is cut to the 10 s that start at <paramref name="offset"/>; a shorter one is repeated and zero-padded.
    /// </summary>
    public float[] Extract(ReadOnlySpan<float> samples, int offset = 0)
    {
        var signal = Window(samples, offset);
        var power = _stft.Power(signal);

        var features = new float[FeatureCount];
        for (var t = 0; t < Frames; t++)
        {
            var row = t * PowerStft.Bins;
            for (var m = 0; m < Mels; m++)
            {
                var sum = 0.0;
                for (var b = _firstBin[m]; b < _endBin[m]; b++) sum += power[row + b] * _bank[b * Mels + m];
                features[t * Mels + m] = (float)(10.0 * Math.Log10(Math.Max(sum, MelFloor)));
            }
        }
        return features;
    }

    /// <summary>Features of several waveforms as one float32 CPU tensor <c>[B, 1, 1001, 64]</c>, in the order given.</summary>
    public Tensor ExtractBatch(IReadOnlyList<float[]> waveforms, IReadOnlyList<int>? offsets = null)
    {
        ArgumentNullException.ThrowIfNull(waveforms);
        if (waveforms.Count == 0) throw new ArgumentException("at least one waveform is needed", nameof(waveforms));
        if (offsets != null && offsets.Count != waveforms.Count)
            throw new ArgumentException("offsets must have one entry per waveform", nameof(offsets));

        var all = new float[waveforms.Count * FeatureCount];
        for (var i = 0; i < waveforms.Count; i++)
            Extract(waveforms[i], offsets?[i] ?? 0).CopyTo(all, i * FeatureCount);
        return tensor(all, new long[] { waveforms.Count, 1, Frames, Mels });
    }

    /// <summary>The 480000 samples (as double) the spectrogram is computed on: cut, as is, or repeat-padded.</summary>
    private static double[] Window(ReadOnlySpan<float> samples, int offset)
    {
        var length = samples.Length;
        if (length == 0) throw new ArgumentException("the waveform is empty", nameof(samples));
        if (length > MaxSamples)
        {
            if (offset < 0 || offset > length - MaxSamples)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, $"the {MaxSamples}-sample window must fit in the {length}-sample waveform (0..{length - MaxSamples})");
        }
        else if (offset != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), offset, $"a waveform of at most {MaxSamples} samples is not cut: the offset must be 0");
        }

        var signal = new double[MaxSamples];
        if (length >= MaxSamples)
        {
            for (var i = 0; i < MaxSamples; i++) signal[i] = samples[offset + i];
            return signal;
        }
        var repeats = MaxSamples / length; // whole repetitions; the rest stays zero (repeatpad)
        for (var r = 0; r < repeats; r++)
            for (var i = 0; i < length; i++) signal[r * length + i] = samples[i];
        return signal;
    }
}
