using LionClap.Audio;
using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;

namespace LionClap.Tests;

public class LogMelExtractorTests(ITestOutputHelper output)
{
    private static readonly string[] Names = ["tone", "noise", "chirp", "clicks", "mix"];
    private static readonly LogMelExtractor Extractor = new();

    private static float[] Waveform(int i) => Fixtures.Load("frontend.safetensors")[$"waveform_{i}"].data<float>().ToArray();

    private static float[] Expected(int i) =>
        Fixtures.Load("frontend.safetensors")["input_features"][i, 0].contiguous().data<float>().ToArray();

    private static Parity.Result Compare(float[] got, float[] expected) =>
        Parity.Compare(torch.tensor(got), torch.tensor(expected));

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void MatchesTheReferenceInputFeatures(int i)
    {
        var r = Compare(Extractor.Extract(Waveform(i)), Expected(i));
        output.WriteLine($"{Names[i]}: {r}");
        Assert.True(r.MaxAbs <= 1e-4, r.ToString());
        Assert.True(r.CosSim >= 0.9999, r.ToString());
    }

    [Fact]
    public void RepeatPaddedTailShowsTheFloor()
    {
        var f = Extractor.Extract(Waveform(2)); // chirp, 206400 samples repeated twice, then zeros from sample 412800
        for (var m = 0; m < LogMelExtractor.Mels; m++)
            Assert.Equal(-100f, f[(LogMelExtractor.Frames - 1) * LogMelExtractor.Mels + m]);
    }

    [Fact]
    public void VeryShortInputWorks()
    {
        var f = Extractor.Extract(new[] { 0.25f });
        Assert.Equal(LogMelExtractor.FeatureCount, f.Length);
        Assert.All(f, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void CutInsideALongerBufferEqualsTheFixtureCase()
    {
        var w = Waveform(1);
        var rng = new Random(7);
        var buffer = new float[24000 + w.Length + 72000];
        for (var i = 0; i < buffer.Length; i++) buffer[i] = (float)(rng.NextDouble() - 0.5);
        w.CopyTo(buffer, 24000);

        var r = Compare(Extractor.Extract(buffer, 24000), Expected(1));
        output.WriteLine($"noise cut at 24000: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, r.ToString());
    }

    [Fact]
    public void DefaultOffsetIsTheFirstWindow()
    {
        var rng = new Random(8);
        var twelveSeconds = Enumerable.Range(0, 12 * LogMelExtractor.SampleRate).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        Assert.Equal(Extractor.Extract(twelveSeconds.AsSpan(0, LogMelExtractor.MaxSamples)), Extractor.Extract(twelveSeconds));
    }

    [Fact]
    public void OffsetThatDoesNotFitIsRejected()
    {
        var twelveSeconds = new float[12 * LogMelExtractor.SampleRate];
        Assert.Throws<ArgumentOutOfRangeException>(() => Extractor.Extract(twelveSeconds, 100000));
        Assert.Throws<ArgumentOutOfRangeException>(() => Extractor.Extract(twelveSeconds, -1));
        Assert.Equal(LogMelExtractor.FeatureCount, Extractor.Extract(twelveSeconds, 96000).Length); // the last offset that fits
    }

    [Fact]
    public void OffsetOnAWaveformThatIsNotLongerIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Extractor.Extract(new float[1000], 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => Extractor.Extract(new float[LogMelExtractor.MaxSamples], 1));
    }

    [Fact]
    public void EmptyWaveformIsRejected() => Assert.Throws<ArgumentException>(() => Extractor.Extract(ReadOnlySpan<float>.Empty));

    [Fact]
    public void TheInputIsNotModified()
    {
        var w = Waveform(2);
        var copy = (float[])w.Clone();
        Extractor.Extract(w);
        Assert.Equal(copy, w);
    }

    [Fact]
    public void BatchMatchesTheFixtureAndTheItemsAlone()
    {
        var waves = Enumerable.Range(0, 5).Select(Waveform).ToList();
        using var batch = Extractor.ExtractBatch(waves);
        Assert.Equal(new long[] { 5, 1, 1001, 64 }, batch.shape);
        Assert.Equal(torch.ScalarType.Float32, batch.dtype);

        var expected = Fixtures.Load("frontend.safetensors")["input_features"];
        var r = Parity.Compare(batch, expected);
        output.WriteLine($"batch of 5: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, r.ToString());

        for (var i = 0; i < 5; i++)
            Assert.Equal(Extractor.Extract(waves[i]), batch[i, 0].contiguous().data<float>().ToArray());
    }

    [Fact]
    public void BatchRejectsEmptyListAndMismatchedOffsets()
    {
        Assert.Throws<ArgumentException>(() => Extractor.ExtractBatch(new List<float[]>()));
        Assert.Throws<ArgumentException>(() => Extractor.ExtractBatch(new List<float[]> { new float[10] }, new[] { 0, 0 }));
    }
}
