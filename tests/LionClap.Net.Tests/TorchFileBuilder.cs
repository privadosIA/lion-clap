using System.Buffers.Binary;
using System.Text;

namespace LionClap.Tests;

/// <summary>Builds small files in the layout <c>torch.save</c> writes (a zip of stored entries, a protocol-2 pickle of a state_dict), with switches to break them.</summary>
public sealed class TorchFileBuilder
{
    public sealed record Tensor(string Name, string Dtype, long[] Shape, byte[] Data, string StorageKey, long Offset = 0, long[]? Stride = null, long? StorageElements = null);

    private readonly List<Tensor> _tensors = new();
    private readonly Dictionary<string, byte[]> _storages = new();

    public string StorageClass(string dtype) => dtype == "F32" ? "torch FloatStorage" : "torch LongStorage";

    // switches
    public string? ForbiddenGlobal { get; set; }
    public string? StorageClassOverride { get; set; }
    public byte? ExtraOpcode { get; set; }
    public bool Compressed { get; set; }
    public bool TwoPickles { get; set; }
    public bool NoPickle { get; set; }
    public bool Zip64 { get; set; }
    public bool BigEndian { get; set; }
    public int? TruncateTo { get; set; }

    public TorchFileBuilder Add(string name, float[] values, params long[] shape)
    {
        var data = new byte[values.Length * 4];
        for (var i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(i * 4), values[i]);
        return AddRaw(new Tensor(name, "F32", shape, data, _storages.Count.ToString()));
    }

    public TorchFileBuilder Add(string name, long[] values, params long[] shape)
    {
        var data = new byte[values.Length * 8];
        for (var i = 0; i < values.Length; i++) BinaryPrimitives.WriteInt64LittleEndian(data.AsSpan(i * 8), values[i]);
        return AddRaw(new Tensor(name, "I64", shape, data, _storages.Count.ToString()));
    }

    public TorchFileBuilder AddRaw(Tensor t)
    {
        _tensors.Add(t);
        if (!_storages.ContainsKey(t.StorageKey)) _storages[t.StorageKey] = t.Data;
        return this;
    }

    public byte[] Build()
    {
        var entries = new List<(string Name, byte[] Data)>();
        var pickle = Pickle();
        if (!NoPickle) entries.Add(("pytorch_model/data.pkl", pickle));
        if (TwoPickles) entries.Add(("other/data.pkl", pickle));
        foreach (var (key, data) in _storages) entries.Add(($"pytorch_model/data/{key}", data));
        entries.Add(("pytorch_model/version", Encoding.ASCII.GetBytes("3\n")));
        if (BigEndian) entries.Add(("pytorch_model/byteorder", Encoding.ASCII.GetBytes("big")));

        using var ms = new MemoryStream();
        var offsets = new List<long>();
        foreach (var (name, data) in entries)
        {
            offsets.Add(ms.Position);
            var nameBytes = Encoding.UTF8.GetBytes(name);
            var padding = (int)((64 - (ms.Position + 30 + nameBytes.Length + 4) % 64) % 64); // torch aligns the data with the extra field
            var header = new byte[30];
            BinaryPrimitives.WriteUInt32LittleEndian(header, 0x04034b50);
            BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(26), (ushort)nameBytes.Length);
            BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(28), (ushort)(4 + padding));
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(18), (uint)data.Length);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(22), (uint)data.Length);
            ms.Write(header);
            ms.Write(nameBytes);
            var extra = new byte[4 + padding];
            BinaryPrimitives.WriteUInt16LittleEndian(extra, 0x4646);
            BinaryPrimitives.WriteUInt16LittleEndian(extra.AsSpan(2), (ushort)padding);
            ms.Write(extra);
            ms.Write(data);
        }

        var directoryStart = ms.Position;
        for (var i = 0; i < entries.Count; i++)
        {
            var nameBytes = Encoding.UTF8.GetBytes(entries[i].Name);
            var h = new byte[46];
            BinaryPrimitives.WriteUInt32LittleEndian(h, 0x02014b50);
            BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(10), (ushort)(Compressed ? 8 : 0));
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(20), (uint)entries[i].Data.Length);
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(24), (uint)entries[i].Data.Length);
            BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(28), (ushort)nameBytes.Length);
            BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(42), Zip64 ? 0xFFFFFFFF : (uint)offsets[i]);
            ms.Write(h);
            ms.Write(nameBytes);
        }
        var end = new byte[22];
        BinaryPrimitives.WriteUInt32LittleEndian(end, 0x06054b50);
        BinaryPrimitives.WriteUInt16LittleEndian(end.AsSpan(10), (ushort)entries.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(end.AsSpan(12), (uint)(ms.Position - directoryStart));
        BinaryPrimitives.WriteUInt32LittleEndian(end.AsSpan(16), (uint)directoryStart);
        ms.Write(end);

        var bytes = ms.ToArray();
        return TruncateTo is { } n ? bytes[..n] : bytes;
    }

    public string WriteTo(string directory, string name = "pytorch_model.bin")
    {
        var path = Path.Combine(directory, name);
        File.WriteAllBytes(path, Build());
        return path;
    }

    // ---- pickle (protocol 2, the opcodes torch emits) -----------------------------------------------------------------------------------

    private byte[] Pickle()
    {
        var p = new MemoryStream();
        void Op(byte b) => p.WriteByte(b);
        void Str(string s) { Op((byte)'X'); var b = Encoding.UTF8.GetBytes(s); p.Write(BitConverter.GetBytes(b.Length)); p.Write(b); }
        void Int(long v) { Op((byte)'J'); p.Write(BitConverter.GetBytes((int)v)); }
        void Global(string text) { Op((byte)'c'); p.Write(Encoding.ASCII.GetBytes(text.Replace(' ', '\n') + "\n")); }
        void Ints(long[] values) { Op((byte)'('); foreach (var v in values) Int(v); Op((byte)'t'); }

        Op(0x80); Op(2);
        Op((byte)'}');
        Op((byte)'(');
        foreach (var t in _tensors)
        {
            Str(t.Name);
            Global(ForbiddenGlobal ?? "torch._utils _rebuild_tensor_v2");
            Op((byte)'(');
            Op((byte)'(');
            Str("storage");
            Global(StorageClassOverride ?? StorageClass(t.Dtype));
            Str(t.StorageKey);
            Str("cpu");
            Int(t.StorageElements ?? t.Data.Length / (t.Dtype == "F32" ? 4 : 8));
            Op((byte)'t');
            Op((byte)'Q');
            Int(t.Offset);
            Ints(t.Shape);
            Ints(t.Stride ?? RowMajor(t.Shape));
            Op(0x89);
            Global("collections OrderedDict");
            Op((byte)')');
            Op((byte)'R');
            Op((byte)'t');
            Op((byte)'R');
            if (ExtraOpcode is { } extra) Op(extra);
        }
        Op((byte)'u');
        Op((byte)'.');
        return p.ToArray();
    }

    private static long[] RowMajor(long[] shape)
    {
        var stride = new long[shape.Length];
        long s = 1;
        for (var i = shape.Length - 1; i >= 0; i--) { stride[i] = s; s *= shape[i]; }
        return stride;
    }
}
