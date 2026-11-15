using LionClap.Audio.Htsat;
using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class HtsatInputStageTests(ITestOutputHelper output)
{
    private static readonly ClapAudioConfig Config = new();

    private static Tensor Intermediate(string name) => Fixtures.Load("htsat_intermediates.safetensors")[name];

    private void AssertParity(string what, Tensor got, Tensor expected)
    {
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{what}: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, $"{what}: {r}");
    }

    [RequiresCheckpoint]
    public void ImageAndPatchEmbeddingMatchTheReference()
    {
        using var encoder = Checkpoint.Load(new ClapAudioEncoder(Config), "audio_model.audio_encoder.");
        using var _ = no_grad();
        var input = Intermediate("input_features");

        using var image = encoder.MelToImage(input);
        Assert.Equal(new long[] { 1, 1, 256, 256 }, image.shape);
        AssertParity("image", image, Intermediate("image"));

        using var tokens = encoder.Embed(input);
        Assert.Equal(new long[] { 1, 4096, 96 }, tokens.shape);
        AssertParity("patch_embed", tokens, Intermediate("patch_embed"));
    }

    [RequiresCheckpoint]
    public void PatchEmbeddingAloneMatchesFromTheReferenceImage()
    {
        using var encoder = Checkpoint.Load(new ClapAudioEncoder(Config), "audio_model.audio_encoder.");
        using var _ = no_grad();
        using var tokens = encoder.EmbedImage(Intermediate("image"));
        AssertParity("patch_embed from the reference image", tokens, Intermediate("patch_embed"));
    }

    [Fact]
    public void FoldPlacesTimeSlicesAlongTheFrequencyAxis()
    {
        // 1024 frames: no stretch; image[b, 0, k*64 + f, t] = x[b, 0, k*256 + t, f]
        using var x = arange(2 * 1024 * 64, dtype: ScalarType.Float32).reshape(2, 1, 1024, 64);
        using var image = ClapAudioEncoder.FoldToImage(x, Config);
        Assert.Equal(new long[] { 2, 1, 256, 256 }, image.shape);
        foreach (var (b, k, f, t) in new[] { (0, 0, 0, 0), (1, 3, 63, 255), (0, 2, 17, 100), (1, 1, 5, 7) })
            Assert.Equal(x[b, 0, k * 256 + t, f].item<float>(), image[b, 0, k * 64 + f, t].item<float>());
    }

    [Fact]
    public void ShorterInputIsStretchedToTheFullImage()
    {
        using var x = randn(1, 1, 1001, 64);
        using var image = ClapAudioEncoder.FoldToImage(x, Config);
        Assert.Equal(new long[] { 1, 1, 256, 256 }, image.shape);
        using var one = randn(1, 1, 1, 64); // a single frame is accepted too
        using var imageOne = ClapAudioEncoder.FoldToImage(one, Config);
        Assert.Equal(new long[] { 1, 1, 256, 256 }, imageOne.shape);
    }

    [Theory]
    [InlineData(3, "4 dimensions")]
    [InlineData(2, "1 channel")]
    [InlineData(80, "64 mel")]
    [InlineData(1025, "1024")]
    [InlineData(0, "1024")]
    public void InvalidInputIsRejectedWithAMessage(int selector, string expectedInMessage)
    {
        using var encoder = new ClapAudioEncoder(Config);
        using var _ = no_grad();
        using var input = selector switch
        {
            3 => zeros(1, 1001, 64),
            2 => zeros(1, 2, 100, 64),
            80 => zeros(1, 1, 100, 80),
            1025 => zeros(1, 1, 1025, 64),
            _ => zeros(1, 1, 0, 64),
        };
        var ex = Assert.Throws<ArgumentException>(() => encoder.MelToImage(input));
        Assert.Contains(expectedInMessage, ex.Message);
    }

    [Fact]
    public void FullLengthInputSkipsTheStretch()
    {
        using var encoder = new ClapAudioEncoder(Config);
        using var _ = no_grad();
        using var input = randn(1, 1, 1024, 64);
        using var image = encoder.MelToImage(input);
        Assert.Equal(new long[] { 1, 1, 256, 256 }, image.shape);
        using var tokens = encoder.EmbedImage(image);
        Assert.Equal(new long[] { 1, 4096, 96 }, tokens.shape);
    }
}
