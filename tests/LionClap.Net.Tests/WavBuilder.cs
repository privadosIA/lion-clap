using System.Buffers.Binary;
using System.Text;

namespace LionClap.Tests;

/// <summary>Builds WAV files byte by byte, so the decoder is tested against known sample values and unusual layouts.</summary>
public sealed class WavBuilder
{
    public ushort Tag { get; set; } = 1;
    public ushort Channels { get; set; } = 1;
    public uint Rate { get; set; } = 48000;
    public ushort Bits { get; set; } = 16;
    public bool Extensible { get; set; }
    public byte[] Data { get; set; } = [];
    public List<(string Id, byte[] Body)> ChunksBeforeData { get; } = new();
    public uint? DataSizeField { get; set; }
    public bool OmitFmt { get; set; }
    public bool OmitData { get; set; }
    public bool DataFirst { get; set; }
    public string Riff { get; set; } = "RIFF";
    public string Wave { get; set; } = "WAVE";

    public static byte[] Pcm16(params short[] samples)
    {
        var b = new byte[samples.Length * 2];
        for (var i = 0; i < samples.Length; i++) BinaryPrimitives.WriteInt16LittleEndian(b.AsSpan(i * 2), samples[i]);
        return b;
    }

    public static byte[] Pcm24(params int[] samples)
    {
        var b = new byte[samples.Length * 3];
        for (var i = 0; i < samples.Length; i++) { b[i * 3] = (byte)samples[i]; b[i * 3 + 1] = (byte)(samples[i] >> 8); b[i * 3 + 2] = (byte)(samples[i] >> 16); }
        return b;
    }

    public static byte[] Pcm32(params int[] samples)
    {
        var b = new byte[samples.Length * 4];
        for (var i = 0; i < samples.Length; i++) BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(i * 4), samples[i]);
        return b;
    }

    public static byte[] Float32(params float[] samples)
    {
        var b = new byte[samples.Length * 4];
        for (var i = 0; i < samples.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(i * 4), samples[i]);
        return b;
    }

    public static byte[] Float64(params double[] samples)
    {
        var b = new byte[samples.Length * 8];
        for (var i = 0; i < samples.Length; i++) BinaryPrimitives.WriteDoubleLittleEndian(b.AsSpan(i * 8), samples[i]);
        return b;
    }

    private static byte[] Chunk(string id, byte[] body, uint? sizeField = null)
    {
        var bytes = new List<byte>(Encoding.ASCII.GetBytes(id));
        bytes.AddRange(BitConverter.GetBytes(sizeField ?? (uint)body.Length));
        bytes.AddRange(body);
        if (body.Length % 2 == 1) bytes.Add(0); // chunks are padded to an even size
        return bytes.ToArray();
    }

    public byte[] Build()
    {
        var fmt = new List<byte>();
        fmt.AddRange(BitConverter.GetBytes(Extensible ? (ushort)0xFFFE : Tag));
        fmt.AddRange(BitConverter.GetBytes(Channels));
        fmt.AddRange(BitConverter.GetBytes(Rate));
        fmt.AddRange(BitConverter.GetBytes(Rate * Channels * (uint)(Bits / 8)));
        fmt.AddRange(BitConverter.GetBytes((ushort)(Channels * (Bits / 8))));
        fmt.AddRange(BitConverter.GetBytes(Bits));
        if (Extensible)
        {
            fmt.AddRange(BitConverter.GetBytes((ushort)22));      // extension size
            fmt.AddRange(BitConverter.GetBytes(Bits));            // valid bits
            fmt.AddRange(BitConverter.GetBytes(0u));              // channel mask
            fmt.AddRange(BitConverter.GetBytes(Tag));             // sub-format GUID: the format tag first
            fmt.AddRange(new byte[] { 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71 });
        }

        var body = new List<byte>(Encoding.ASCII.GetBytes(Wave));
        if (DataFirst && !OmitData) body.AddRange(Chunk("data", Data, DataSizeField));
        if (!OmitFmt) body.AddRange(Chunk("fmt ", fmt.ToArray()));
        foreach (var (id, chunk) in ChunksBeforeData) body.AddRange(Chunk(id, chunk));
        if (!DataFirst && !OmitData) body.AddRange(Chunk("data", Data, DataSizeField));

        var file = new List<byte>(Encoding.ASCII.GetBytes(Riff));
        file.AddRange(BitConverter.GetBytes((uint)body.Count));
        file.AddRange(body);
        return file.ToArray();
    }
}
