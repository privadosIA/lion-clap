using LionClap.Audio.Htsat;
using MusicGen.Core.Testing;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class HtsatBlockTests(ITestOutputHelper output)
{
    private static readonly ClapAudioConfig Config = new();
    private const string Stage0 = "audio_model.audio_encoder.layers.0.blocks.";

    private static Tensor Intermediate(string name) => Fixtures.Load("htsat_intermediates.safetensors")[name];

    private void AssertParity(string what, Tensor got, Tensor expected)
    {
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{what}: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, $"{what}: {r}");
    }

    [RequiresCheckpoint]
    public void Stage0Block0MatchesFromTheReferencePatchEmbedding()
    {
        using var block = Checkpoint.Load(new ClapAudioLayer(Config, 96, (64, 64), 4, shift: 0), Stage0 + "0.");
        using var _ = no_grad();
        using var result = block.forward(Intermediate("patch_embed"));
        AssertParity("stage0_block0", result, Intermediate("stage0_block0"));
    }

    [RequiresCheckpoint]
    public void Stage0Block1ShiftedAndMaskedMatchesFromTheReferenceBlock0()
    {
        using var block = Checkpoint.Load(new ClapAudioLayer(Config, 96, (64, 64), 4, shift: 4), Stage0 + "1.");
        Assert.Equal(4, block.Shift);
        using var _ = no_grad();
        using var result = block.forward(Intermediate("stage0_block0"));
        AssertParity("stage0_block1", result, Intermediate("stage0_block1"));
    }

    [Fact]
    public void ShiftIsDroppedWhenTheGridFitsInOneWindow()
    {
        using var last = new ClapAudioLayer(Config, 768, (8, 8), 32, shift: 4);
        Assert.Equal(0, last.Shift);
        using var first = new ClapAudioLayer(Config, 96, (64, 64), 4, shift: 4);
        Assert.Equal(4, first.Shift);
    }

    [Fact]
    public void GridsThatWouldNeedPaddingAreRejected() =>
        Assert.Throws<NotSupportedException>(() => new ClapAudioLayer(Config, 96, (60, 64), 4, shift: 0));

    [Fact]
    public void BlockKeepsTheShapeAndIsIndependentPerBatchItem()
    {
        using var block = new ClapAudioLayer(Config, 96, (16, 16), 4, shift: 4);
        block.eval();
        using var _ = no_grad();
        using var x = randn(3, 256, 96);
        using var batch = block.forward(x);
        Assert.Equal(x.shape, batch.shape);
        using var alone = block.forward(x[1..2]);
        Assert.True((batch[1..2] - alone).abs().max().item<float>() < 1e-5f);
    }
}
