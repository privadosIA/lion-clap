using System.Text.Json;
using LionClap.Audio;
using LionClap.Audio.Htsat;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class WindowTests
{
    [Fact]
    public void ATwelveSecondClipHasThreeWindows()
    {
        var windows = ClapAudioEmbedder.Windows(new float[12 * 48000]);
        Assert.Equal(new[] { 480000, 336000, 96000 }, windows.Select(w => w.Length)); // 0–10 s, 5–12 s, 10–12 s
    }

    [Theory]
    [InlineData(2 * 48000, 1)]          // shorter than a window: one
    [InlineData(10 * 48000, 2)]         // 0–10 s and 5–10 s; the third start (10 s) is at the end
    [InlineData(11 * 48000, 2)]         // 10–11 s is exactly one second: not longer than 1 s
    [InlineData(11 * 48000 + 1, 3)]
    [InlineData(48000 + 1, 1)]
    public void WindowsAreKeptOnlyWhenLongerThanOneSecond(int samples, int expected) =>
        Assert.Equal(expected, ClapAudioEmbedder.Windows(new float[samples]).Count);

    [Theory]
    [InlineData(0)]
    [InlineData(1000)]
    [InlineData(48000)]                 // exactly one second
    public void OneSecondOrLessIsRejected(int samples)
    {
        var ex = Assert.Throws<ArgumentException>(() => ClapAudioEmbedder.Windows(new float[samples]));
        Assert.Contains("shorter than one second", ex.Message);
    }

    [Fact]
    public void AudioThatLandsOnOneSecondAfterResamplingIsRejected()
    {
        var audio = new DecodedAudio(new float[44100], 44100, 1); // 1.0 s at 44.1 kHz → 48000 samples
        Assert.Throws<ArgumentException>(() => ClapAudioEmbedder.Windows(ClapAudioEmbedder.To48k(audio)));
    }

    [Fact]
    public void WindowsHoldTheRightSamples()
    {
        var clip = Enumerable.Range(0, 12 * 48000).Select(i => (float)i).ToArray();
        var windows = ClapAudioEmbedder.Windows(clip);
        Assert.Equal(0f, windows[0][0]);
        Assert.Equal(5 * 48000f, windows[1][0]);
        Assert.Equal(10 * 48000f, windows[2][0]);
        Assert.Equal(clip[^1], windows[2][^1]);
    }
}

public class ClapAudioEmbedderTests(BranchFixture fixture, ITestOutputHelper output) : IClassFixture<BranchFixture>
{
    private static JsonElement Cases() => JsonDocument.Parse(File.ReadAllText(Fixtures.PathOf("classify_cases.json"))).RootElement.GetProperty("cases");

    [RequiresCheckpoint]
    public void TheResampledGoldenSignalMatchesWithin1e6()
    {
        var chirp = WavDecoder.Decode(File.ReadAllBytes(Fixtures.PathOf("chirp_1p5s_44k_mono_pcm24.wav")));
        var ours = ClapAudioEmbedder.To48k(chirp);
        var golden = Fixtures.Load("classify_embeddings.safetensors")["resampled_3"].data<float>().ToArray();
        Assert.Equal(golden.Length, ours.Length);
        var worst = ours.Zip(golden, (a, b) => Math.Abs(a - b)).Max();
        output.WriteLine($"resampled 24-bit golden signal: {golden.Length} samples, max_abs {worst:E2}");
        Assert.True(worst <= 1e-6, $"{worst:E2}");
    }

    [RequiresCheckpoint]
    public void WindowAveragedEmbeddingsOfTheFourGoldenInputsMatch()
    {
        var embedder = new ClapAudioEmbedder(fixture.Branch, new LogMelExtractor());
        var golden = Fixtures.Load("classify_embeddings.safetensors");
        foreach (var c in Cases().EnumerateArray())
        {
            var i = c.GetProperty("id").GetInt32();
            var audio = WavDecoder.Decode(File.ReadAllBytes(Fixtures.PathOf(c.GetProperty("wav").GetString()!)));
            var embedding = embedder.Embed(ClapAudioEmbedder.To48k(audio));

            Assert.Equal(c.GetProperty("windows").GetInt32(), embedding.Windows);
            var r = Parity.Compare(tensor(embedding.Vector), golden[$"audio_{i}"]);
            output.WriteLine($"case {i} {c.GetProperty("wav").GetString()}: {embedding.Windows} window(s), {r}");
            Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, $"case {i}: {r}");
        }
    }

    [RequiresCheckpoint]
    public void TheTwelveSecondInputUsesThreeWindows()
    {
        var embedder = new ClapAudioEmbedder(fixture.Branch, new LogMelExtractor());
        var audio = WavDecoder.Decode(File.ReadAllBytes(Fixtures.PathOf("tone_12s_16k_mono_pcm16.wav")));
        Assert.Equal(3, embedder.Embed(ClapAudioEmbedder.To48k(audio)).Windows);
    }
}
