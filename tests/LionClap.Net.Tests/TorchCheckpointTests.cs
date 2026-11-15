using LionClap.Weights;
using MusicGen.Core.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

/// <summary>A read-only stream that counts what is read, to prove that opening a checkpoint only reads headers.</summary>
internal sealed class CountingStream(Stream inner) : Stream
{
    public long BytesRead { get; private set; }
    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => inner.Length;
    public override long Position { get => inner.Position; set => inner.Position = value; }
    public override int Read(byte[] buffer, int offset, int count) { var n = inner.Read(buffer, offset, count); BytesRead += n; return n; }
    public override long Seek(long offset, SeekOrigin origin) => inner.Seek(offset, origin);
    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

public class TorchCheckpointTests(ITestOutputHelper output)
{
    private static string Temp() => TestTemp.Dir();

    private static string Write(Action<TorchFileBuilder>? configure = null, Action<TorchFileBuilder>? tensors = null)
    {
        var b = new TorchFileBuilder();
        (tensors ?? (t => t.Add("w", new float[] { 1, 2, 3 }, 3)))(b);
        configure?.Invoke(b);
        return b.WriteTo(Temp());
    }

    [Fact]
    public void ReadsFloatAndIntTensorsExactlyAndInFileOrder()
    {
        var path = Write(tensors: b => b
            .Add("a.weight", new float[] { 1.5f, -2, 3.25f, 4, 5, 6 }, 2, 3)
            .Add("b.ids", new long[] { 7, 8, 9, 10 }, 1, 4)
            .Add("scale", new float[] { 2.9f }));                         // a scalar: shape [], stride []
        var ckpt = TorchCheckpoint.Open(path);

        Assert.Equal(new[] { "a.weight", "b.ids", "scale" }, ckpt.Tensors.Keys);
        Assert.Equal("F32", ckpt.Tensors["a.weight"].Dtype);
        Assert.Equal(new long[] { 2, 3 }, ckpt.Tensors["a.weight"].Shape);
        Assert.Equal("I64", ckpt.Tensors["b.ids"].Dtype);
        Assert.Equal(new long[] { 1, 4 }, ckpt.Tensors["b.ids"].Shape);
        Assert.Empty(ckpt.Tensors["scale"].Shape);

        using var a = ckpt.Read("a.weight");
        Assert.Equal(new float[] { 1.5f, -2, 3.25f, 4, 5, 6 }, a.data<float>().ToArray());
        using var ids = ckpt.Read("b.ids");
        Assert.Equal(new long[] { 7, 8, 9, 10 }, ids.data<long>().ToArray());
        using var asFloat = ckpt.Read("b.ids", ScalarType.Float32);
        Assert.Equal(new float[] { 7, 8, 9, 10 }, asFloat.data<float>().ToArray());
        using var scale = ckpt.Read("scale");
        Assert.Equal(2.9f, scale.item<float>());
    }

    [Fact]
    public void TensorsThatShareAStorageAreReadAtTheirOffsets()
    {
        var shared = new float[] { 0, 1, 2, 3, 4, 5, 6, 7 };
        var data = new byte[32];
        Buffer.BlockCopy(shared, 0, data, 0, 32);
        var b = new TorchFileBuilder()
            .AddRaw(new("first", "F32", new long[] { 4 }, data, "0", Offset: 0))
            .AddRaw(new("second", "F32", new long[] { 4 }, data, "0", Offset: 4));
        var ckpt = TorchCheckpoint.Open(b.WriteTo(Temp()));
        using var second = ckpt.Read("second");
        Assert.Equal(new float[] { 4, 5, 6, 7 }, second.data<float>().ToArray());
    }

    [Fact]
    public void DimensionsOfSizeOneMayHaveAnyStride()
    {
        var b = new TorchFileBuilder().AddRaw(new("w", "F32", new long[] { 1, 4 }, new byte[16], "0", Stride: new long[] { 999, 1 }));
        Assert.Equal(new long[] { 1, 4 }, TorchCheckpoint.Open(b.WriteTo(Temp())).Tensors["w"].Shape);
    }

    [Fact]
    public void OpeningReadsOnlyHeaders()
    {
        var big = new float[1_000_000]; // 4 MB of tensor data that must not be read
        var path = Write(tensors: b => b.Add("big", big, 1000, 1000).Add("small", new float[] { 1 }, 1));
        using var counting = new CountingStream(File.OpenRead(path));
        var ckpt = TorchCheckpoint.Open(counting, path);
        output.WriteLine($"file {counting.Length} bytes, read {counting.BytesRead}");
        Assert.True(counting.BytesRead < 32 * 1024, $"read {counting.BytesRead} bytes");
        Assert.Equal(2, ckpt.Tensors.Count);
    }

    [Fact]
    public void ANonexistentFileIsReported() =>
        Assert.Throws<FileNotFoundException>(() => TorchCheckpoint.Open(Path.Combine(Temp(), "nope.bin")));

    [Theory]
    [InlineData("text")]
    [InlineData("empty")]
    [InlineData("truncated")]
    public void FilesThatAreNotZipArchivesAreRejected(string kind)
    {
        var dir = Temp();
        var path = Path.Combine(dir, "x.bin");
        switch (kind)
        {
            case "text": File.WriteAllText(path, new string('a', 500)); break;
            case "empty": File.WriteAllBytes(path, []); break;
            default: File.WriteAllBytes(path, new TorchFileBuilder().Add("w", new float[] { 1 }, 1).Build()[..60]); break;
        }
        var ex = Assert.ThrowsAny<InvalidDataException>(() => TorchCheckpoint.Open(path));
        Assert.Contains("not a readable PyTorch checkpoint", ex.Message);
    }

    [Fact]
    public void ATruncatedArchiveIsRejected()
    {
        var full = new TorchFileBuilder().Add("w", new float[1000], 1000).Build();
        var path = Path.Combine(Temp(), "cut.bin");
        File.WriteAllBytes(path, full[..(full.Length - 300)]); // the end of the file (central directory) is gone
        Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(path));
    }

    [Fact]
    public void ACompressedEntryIsRejected()
    {
        var ex = Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(Write(b => b.Compressed = true)));
        Assert.Contains("compressed", ex.Message);
    }

    [Fact]
    public void Zip64IsRejected() => Assert.Throws<NotSupportedException>(() => TorchCheckpoint.Open(Write(b => b.Zip64 = true)));

    [Fact]
    public void BigEndianIsRejected() => Assert.Throws<NotSupportedException>(() => TorchCheckpoint.Open(Write(b => b.BigEndian = true)));

    [Fact]
    public void ArchivesWithoutExactlyOnePickleAreRejected()
    {
        Assert.Contains("no data.pkl", Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(Write(b => b.NoPickle = true))).Message);
        Assert.Contains("more than one", Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(Write(b => b.TwoPickles = true))).Message);
    }

    [Theory]
    [InlineData("os system")]
    [InlineData("builtins eval")]
    [InlineData("subprocess Popen")]
    public void AGlobalOutsideTheAllowedSetIsRejectedByName(string global)
    {
        var ex = Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(Write(b => b.ForbiddenGlobal = global)));
        Assert.Contains(global.Replace(' ', '.'), ex.Message);
        Assert.Contains("not allowed", ex.Message);
    }

    [Fact]
    public void AnUnsupportedOpcodeIsRejectedWithItsPosition()
    {
        var ex = Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(Write(b => b.ExtraOpcode = (byte)'b'))); // BUILD
        Assert.Contains("0x62", ex.Message);
        Assert.Contains("byte", ex.Message);
    }

    [Fact]
    public void AStorageTypeOtherThanFloat32OrInt64IsRejected()
    {
        var ex = Assert.Throws<NotSupportedException>(() => TorchCheckpoint.Open(Write(b => b.StorageClassOverride = "torch HalfStorage")));
        Assert.Contains("HalfStorage", ex.Message);
    }

    [Fact]
    public void ANonContiguousTensorIsRejected()
    {
        var b = new TorchFileBuilder().AddRaw(new("w", "F32", new long[] { 2, 3 }, new byte[24], "0", Stride: new long[] { 1, 2 })); // transposed
        var ex = Assert.Throws<NotSupportedException>(() => TorchCheckpoint.Open(b.WriteTo(Temp())));
        Assert.Contains("not contiguous", ex.Message);
    }

    [Fact]
    public void ATensorPointingPastItsStorageIsRejected()
    {
        var b = new TorchFileBuilder().AddRaw(new("w", "F32", new long[] { 4 }, new byte[16], "0", Offset: 2)); // elements 2..5 of a 4-element storage
        var ex = Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(b.WriteTo(Temp())));
        Assert.Contains("past the end", ex.Message);
    }

    [Fact]
    public void AStorageSizeThatDisagreesWithThePickleIsRejected()
    {
        var b = new TorchFileBuilder().AddRaw(new("w", "F32", new long[] { 4 }, new byte[16], "0", StorageElements: 10));
        Assert.Throws<InvalidDataException>(() => TorchCheckpoint.Open(b.WriteTo(Temp())));
    }

    [RequiresBothCheckpoints]
    public void TheRealBinHasTheSame447TensorsAsTheConvertedCheckpoint()
    {
        var bin = TorchCheckpoint.Open(Checkpoint.RealBin!);
        var reference = Checkpoint.Index;
        Assert.Equal(447, bin.Tensors.Count);
        Assert.Equal(reference.Tensors.Keys.OrderBy(k => k, StringComparer.Ordinal), bin.Tensors.Keys.OrderBy(k => k, StringComparer.Ordinal));

        // the converted checkpoint keeps every tensor in float32, the .bin keeps the 15 integer buffers (indices, token ids, counters) in int64
        var integerNames = new[] { "relative_position_index", "num_batches_tracked", "position_ids", "token_type_ids" };
        int identical = 0, integers = 0;
        foreach (var (key, info) in bin.Tensors)
        {
            var expected = reference.Tensors[key];
            Assert.Equal(expected.Shape, info.Shape);
            using var a = bin.Read(key);
            using var b = reference.Read(key);
            if (info.Dtype == "F32")
            {
                Assert.Equal("F32", expected.Dtype);
                Assert.True(a.bytes.SequenceEqual(b.bytes), $"{key} differs");
                identical++;
            }
            else
            {
                Assert.Equal("I64", info.Dtype);
                Assert.Contains(integerNames, n => key.EndsWith(n, StringComparison.Ordinal));
                using var asFloat = a.to(ScalarType.Float32);          // the golden file holds these as float32
                Assert.True(asFloat.bytes.SequenceEqual(b.bytes), $"{key} differs");
                integers++;
            }
        }
        output.WriteLine($"{identical} float32 tensors bit-identical, {integers} int64 buffers equal to their float32 copies");
        Assert.Equal(447, identical + integers);
        Assert.Equal(15, integers);
    }

    [RequiresBothCheckpoints]
    public void OpeningTheRealBinReadsOnlyHeaders()
    {
        using var counting = new CountingStream(File.OpenRead(Checkpoint.RealBin!));
        var bin = TorchCheckpoint.Open(counting, Checkpoint.RealBin!);
        output.WriteLine($"file {counting.Length / 1024 / 1024} MB, read {counting.BytesRead / 1024} KB");
        Assert.Equal(447, bin.Tensors.Count);
        Assert.True(counting.BytesRead < 2 * 1024 * 1024, $"read {counting.BytesRead} bytes");
    }
}
