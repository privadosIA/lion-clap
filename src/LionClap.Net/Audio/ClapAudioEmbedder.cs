using LionClap.Audio.Htsat;
using LionClap.Classification;
using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Audio;

/// <summary>The embedding of a whole clip: the mean of the L2-normalised embeddings of its windows (so its norm is at most 1), and how many windows there were.</summary>
public sealed record ClipEmbedding(float[] Vector, int Windows);

/// <summary>
/// The audio side of the reference service: a clip at 48 kHz is cut into up to three windows of 10 s that start at 0, 5 s and 10 s (each kept only when it is longer
/// than 1 s), each window goes through the frontend and the audio branch, and the clip embedding is the mean of the normalised window embeddings.
/// </summary>
public sealed class ClapAudioEmbedder(ClapAudioBranch branch, LogMelExtractor extractor) : IClipEmbedder
{
    public const int SampleRate = LogMelExtractor.SampleRate;
    public const int WindowSamples = 10 * SampleRate;

    /// <summary>Converts decoded audio to the 48 kHz mono signal the windows are cut from.</summary>
    public static float[] To48k(DecodedAudio audio) => LinearResampler.Resample(audio.Mono, audio.SampleRate, SampleRate);

    /// <summary>The windows of a 48 kHz clip. Throws <see cref="ArgumentException"/> when no window is longer than one second.</summary>
    public static IReadOnlyList<float[]> Windows(float[] clip)
    {
        var windows = new List<float[]>();
        foreach (var start in new[] { 0, WindowSamples / 2, WindowSamples })
        {
            if (start >= clip.Length) continue;
            var length = Math.Min(WindowSamples, clip.Length - start);
            if (length > SampleRate) windows.Add(clip.AsSpan(start, length).ToArray());
        }
        if (windows.Count == 0) throw new ArgumentException("the audio is shorter than one second");
        return windows;
    }

    public ClipEmbedding Embed(float[] clip48k)
    {
        var windows = Windows(clip48k);
        using var features = extractor.ExtractBatch(windows);
        using var result = branch.Forward(features);
        using var mean = result.Embeddings.mean([0L]);
        return new ClipEmbedding(mean.to(ScalarType.Float32).cpu().data<float>().ToArray(), windows.Count);
    }
}
