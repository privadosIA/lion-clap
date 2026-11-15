using LionClap.Audio;

namespace LionClap.Tests;

public class WavDecoderTests
{
    private static float[] Mono(WavBuilder b) => WavDecoder.Decode(b.Build()).Mono;

    [Fact]
    public void Pcm16IsScaledByTheLibsndfileFactor()
    {
        var audio = WavDecoder.Decode(new WavBuilder { Bits = 16, Rate = 16000, Data = WavBuilder.Pcm16(0, 16384, -16384, -32768, 32767) }.Build());
        Assert.Equal(new[] { 0f, 0.5f, -0.5f, -1f, 32767 / 32768f }, audio.Mono);
        Assert.Equal((16000, 1), (audio.SampleRate, audio.Channels));
    }

    [Fact]
    public void Pcm8IsUnsigned() =>
        Assert.Equal(new[] { -1f, 0f, 127 / 128f }, Mono(new WavBuilder { Bits = 8, Data = [0, 128, 255] }));

    [Fact]
    public void Pcm24IsSignExtended() =>
        Assert.Equal(new[] { 0.5f, -0.5f, -1f, 8388607 / 8388608f }, Mono(new WavBuilder { Bits = 24, Data = WavBuilder.Pcm24(0x400000, -0x400000, -0x800000, 0x7FFFFF) }));

    [Fact]
    public void Pcm32IsScaledByTwoToThe31() =>
        Assert.Equal(new[] { 0.25f, -1f }, Mono(new WavBuilder { Bits = 32, Data = WavBuilder.Pcm32(0x20000000, int.MinValue) }));

    [Fact]
    public void FloatFormatsAreTakenAsTheyAre()
    {
        Assert.Equal(new[] { 0.1f, -0.7f, 1.5f }, Mono(new WavBuilder { Tag = 3, Bits = 32, Data = WavBuilder.Float32(0.1f, -0.7f, 1.5f) }));
        Assert.Equal(new[] { 0.1f, -0.7f }, Mono(new WavBuilder { Tag = 3, Bits = 64, Data = WavBuilder.Float64(0.1, -0.7) }));
    }

    [Fact]
    public void ChannelsAreAveragedInFloat32()
    {
        var audio = WavDecoder.Decode(new WavBuilder { Channels = 2, Tag = 3, Bits = 32, Data = WavBuilder.Float32(0.5f, 0.25f, -0.5f, 0.25f) }.Build());
        Assert.Equal(new[] { 0.375f, -0.125f }, audio.Mono);
        Assert.Equal(2, audio.Channels);

        var three = WavDecoder.Decode(new WavBuilder { Channels = 3, Bits = 16, Data = WavBuilder.Pcm16(3000, 6000, 9000) }.Build());
        Assert.Equal((3000 / 32768f + 6000 / 32768f + 9000 / 32768f) / 3f, three.Mono[0]);
    }

    [Theory]
    [InlineData(1, 16)]
    [InlineData(3, 32)]
    public void ExtensibleHeadersUseTheSubFormat(ushort tag, ushort bits)
    {
        var data = tag == 1 ? WavBuilder.Pcm16(16384, -16384) : WavBuilder.Float32(0.5f, -0.5f);
        Assert.Equal(new[] { 0.5f, -0.5f }, Mono(new WavBuilder { Extensible = true, Tag = tag, Bits = bits, Data = data }));
    }

    [Fact]
    public void UnknownChunksAreSkippedIncludingOddSizedOnes()
    {
        var b = new WavBuilder { Data = WavBuilder.Pcm16(16384, -16384) };
        b.ChunksBeforeData.Add(("LIST", [1, 2, 3]));            // odd size: one pad byte follows
        b.ChunksBeforeData.Add(("PEAK", new byte[16]));
        Assert.Equal(new[] { 0.5f, -0.5f }, Mono(b));
    }

    [Theory]
    [InlineData(0xFFFFFFFFu)]
    [InlineData(0u)]
    [InlineData(1000000u)]
    public void ADataSizeThatIsUnknownOrTooLargeMeansToTheEndOfTheFile(uint sizeField) =>
        Assert.Equal(new[] { 0.5f, -0.5f }, Mono(new WavBuilder { Data = WavBuilder.Pcm16(16384, -16384), DataSizeField = sizeField }));

    [Fact]
    public void ATruncatedFileDropsTheLastPartialFrame()
    {
        var file = new WavBuilder { Channels = 2, Data = WavBuilder.Pcm16(16384, 16384, -16384, -16384, 8192) }.Build(); // 5 samples = 2 frames + 1
        Assert.Equal(new[] { 0.5f, -0.5f }, WavDecoder.Decode(file).Mono);
    }

    [Fact]
    public void ThePadByteOfAnOddDataChunkDoesNotAddASample() =>
        Assert.Equal(new[] { -1f, 0f, 127 / 128f }, Mono(new WavBuilder { Bits = 8, Data = [0, 128, 255] })); // 3 bytes of data, 1 pad byte

    [Theory]
    [InlineData("riff", "not RIFF")]
    [InlineData("wave", "not WAVE")]
    [InlineData("nofmt", "no fmt")]
    [InlineData("nodata", "no data")]
    [InlineData("datafirst", "before the fmt")]
    [InlineData("zerochannels", "zero channels")]
    [InlineData("adpcm", "unsupported")]
    [InlineData("12bit", "unsupported")]
    [InlineData("noframes", "no complete sample frame")]
    public void BrokenFilesAreRejectedWithTheCause(string kind, string _)
    {
        var b = new WavBuilder { Data = WavBuilder.Pcm16(1, 2, 3, 4) };
        switch (kind)
        {
            case "riff": b.Riff = "RIFX"; break;
            case "wave": b.Wave = "AVI "; break;
            case "nofmt": b.OmitFmt = true; break;
            case "nodata": b.OmitData = true; break;
            case "datafirst": b.DataFirst = true; break;
            case "zerochannels": b.Channels = 0; break;
            case "adpcm": b.Tag = 2; break;
            case "12bit": b.Bits = 12; break;
            case "noframes": b.Data = [1]; break;
        }
        var ex = Assert.Throws<WavException>(() => WavDecoder.Decode(b.Build()));
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
    }

    [Fact]
    public void MessagesNameTheProblem()
    {
        Assert.Contains("RIFF", Assert.Throws<WavException>(() => WavDecoder.Decode(new WavBuilder { Riff = "RIFX" }.Build())).Message);
        Assert.Contains("fmt", Assert.Throws<WavException>(() => WavDecoder.Decode(new WavBuilder { OmitFmt = true }.Build())).Message);
        Assert.Contains("12 bits", Assert.Throws<WavException>(() => WavDecoder.Decode(new WavBuilder { Bits = 12, Data = [1, 2, 3, 4] }.Build())).Message);
        Assert.Contains("too short", Assert.Throws<WavException>(() => WavDecoder.Decode(new byte[5])).Message);
        Assert.Throws<WavException>(() => WavDecoder.Decode([]));
    }

    [Fact]
    public void AnArbitraryNonWavByteStringIsRejected()
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes("this is definitely not a wav file, just some text that is long enough");
        Assert.Throws<WavException>(() => WavDecoder.Decode(bytes));
    }

    [Fact]
    public void TheGoldenFilesDecodeToTheirDeclaredShape()
    {
        var expected = new (string File, int Rate, int Channels, int Frames)[]
        {
            ("tone_12s_16k_mono_pcm16.wav", 16000, 1, 192000),
            ("mix_3s_22k_stereo_pcm16.wav", 22050, 2, 66150),
            ("noise_2s_48k_mono_float32.wav", 48000, 1, 96000),
            ("chirp_1p5s_44k_mono_pcm24.wav", 44100, 1, 66150),
        };
        foreach (var (file, rate, channels, frames) in expected)
        {
            var audio = WavDecoder.Decode(File.ReadAllBytes(Fixtures.PathOf(file)));
            Assert.Equal((rate, channels, frames), (audio.SampleRate, audio.Channels, audio.Mono.Length));
            Assert.All(audio.Mono, v => Assert.InRange(v, -1f, 1f));
        }
    }
}
