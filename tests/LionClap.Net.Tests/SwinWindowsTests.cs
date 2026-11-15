using LionClap.Audio.Htsat;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class SwinWindowsTests
{
    [Fact]
    public void PartitionThenReverseIsTheIdentity()
    {
        using var x = randn(2, 16, 24, 5);
        using var windows = SwinWindows.Partition(x, 8);
        Assert.Equal(new long[] { 2 * 2 * 3, 8, 8, 5 }, windows.shape);
        using var back = SwinWindows.Reverse(windows, 8, 16, 24);
        Assert.Equal(0f, (x - back).abs().max().item<float>());
    }

    [Fact]
    public void WindowsAreOrderedByBatchThenRowThenColumn()
    {
        using var x = arange(2 * 16 * 24 * 1, dtype: ScalarType.Float32).reshape(2, 16, 24, 1);
        using var windows = SwinWindows.Partition(x, 8);
        // window number = b·(2·3) + wr·3 + wc
        foreach (var (b, wr, wc, i, j) in new[] { (0, 0, 0, 0, 0), (0, 1, 2, 3, 5), (1, 0, 1, 7, 7), (1, 1, 2, 0, 4) })
            Assert.Equal(x[b, wr * 8 + i, wc * 8 + j, 0].item<float>(), windows[b * 6 + wr * 3 + wc, i, j, 0].item<float>());
    }

    [Fact]
    public void PartitionRejectsMapsThatDoNotDivide()
    {
        using var x = zeros(1, 12, 16, 3);
        Assert.Throws<ArgumentException>(() => SwinWindows.Partition(x, 8));
    }

    [Fact]
    public void RelativePositionIndexHasTheExpectedStructure()
    {
        using var index = SwinWindows.RelativePositionIndex(8);
        Assert.Equal(new long[] { 64, 64 }, index.shape);
        var v = index.data<long>().ToArray();
        Assert.Equal(0, v.Min());
        Assert.Equal(224, v.Max()); // (2·8−1)² − 1 rows in the bias table: 225
        for (var i = 0; i < 64; i++)
        {
            Assert.Equal(112, v[i * 64 + i]);             // same position: (7·15 + 7)
            for (var j = 0; j < 64; j++) Assert.Equal(224, v[i * 64 + j] + v[j * 64 + i]); // opposite offsets mirror
        }
        Assert.Equal(111, v[0 * 64 + 1]); // one column to the left: (7·15 + 6)
    }

    [Fact]
    public void ShiftMaskHasOnlyZeroAndMinusHundred()
    {
        using var mask = SwinWindows.ShiftMask(64, 64, 8, 4);
        Assert.Equal(new long[] { 64, 64, 64 }, mask.shape);
        Assert.All(mask.data<float>().ToArray(), v => Assert.True(v == 0f || v == -100f));
    }

    [Theory]
    [InlineData(0, 4096)]       // top-left window: one region, nothing is masked
    [InlineData(7, 2048)]       // top-right: two regions of 32 positions → 64·32 zero entries
    [InlineData(63, 1024)]      // bottom-right: four regions of 16 positions → 64·16 zero entries
    public void ShiftMaskCountsMatchTheRegionsOfEachWindow(int window, int zeros)
    {
        using var mask = SwinWindows.ShiftMask(64, 64, 8, 4);
        using var w = mask[window];
        Assert.Equal(zeros, w.eq(0f).sum().item<long>());
        using var t = w.transpose(0, 1);
        Assert.Equal(0f, (w - t).abs().max().item<float>()); // symmetric
        for (var i = 0; i < 64; i++) Assert.Equal(0f, w[i, i].item<float>()); // a position always sees itself
    }

    [Fact]
    public void ShiftMaskRejectsAnUnshiftedBlock() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => SwinWindows.ShiftMask(64, 64, 8, 0));

    [RequiresCheckpoint]
    public void ComputedIndexEqualsTheCheckpointBufferOfEveryBlock()
    {
        using var expected = SwinWindows.RelativePositionIndex(8);
        var depths = new[] { 2, 2, 6, 2 };
        var checkedBlocks = 0;
        for (var layer = 0; layer < depths.Length; layer++)
            for (var block = 0; block < depths[layer]; block++)
            {
                using var stored = Checkpoint.Index.Read($"audio_model.audio_encoder.layers.{layer}.blocks.{block}.attention.self.relative_position_index");
                Assert.Equal(expected.shape, stored.shape);
                Assert.Equal(expected.data<long>().ToArray(), stored.to(ScalarType.Int64).data<long>().ToArray());
                checkedBlocks++;
            }
        Assert.Equal(12, checkedBlocks);
    }
}
