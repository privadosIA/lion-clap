using System.Buffers.Binary;

namespace LionClap.Audio;

/// <summary>A WAV file that cannot be decoded. The message says why.</summary>
public sealed class WavException(string message) : Exception(message);

/// <summary>The audio of a WAV file mixed down to mono float32 in [-1, 1], with the rate and channel count it had.</summary>
public sealed record DecodedAudio(float[] Mono, int SampleRate, int Channels);

/// <summary>
/// A WAV (RIFF/WAVE) decoder for what the service accepts: integer PCM of 8 (unsigned), 16, 24 and 32 bits and IEEE float of 32 and 64 bits, also behind the
/// extensible header, with any number of channels, which are averaged to mono in float32. Samples are scaled as libsndfile does (<c>int16 / 32768</c>, …), the
/// chunks are walked by size (odd sizes are padded, unknown chunks skipped) and a <c>data</c> size that is zero, <c>0xFFFFFFFF</c> or larger than the file means
/// "to the end of the file"; a final partial frame is dropped.
/// </summary>
public static class WavDecoder
{
    private const ushort Pcm = 1, Float = 3, Extensible = 0xFFFE;

    public static DecodedAudio Decode(ReadOnlySpan<byte> file)
    {
        if (file.Length < 12) throw new WavException("the data is too short to be a WAV file");
        if (!file[..4].SequenceEqual("RIFF"u8)) throw new WavException("it does not start with RIFF");
        if (!file.Slice(8, 4).SequenceEqual("WAVE"u8)) throw new WavException("the RIFF container is not WAVE");

        var position = 12;
        ushort tag = 0, bits = 0, channels = 0;
        uint rate = 0;
        var haveFormat = false;
        while (position + 8 <= file.Length)
        {
            var id = file.Slice(position, 4);
            var size = BinaryPrimitives.ReadUInt32LittleEndian(file.Slice(position + 4, 4));
            var body = position + 8;

            if (id.SequenceEqual("fmt "u8))
            {
                if (size < 16 || body + 16 > file.Length) throw new WavException("the fmt chunk is truncated");
                tag = BinaryPrimitives.ReadUInt16LittleEndian(file[body..]);
                channels = BinaryPrimitives.ReadUInt16LittleEndian(file[(body + 2)..]);
                rate = BinaryPrimitives.ReadUInt32LittleEndian(file[(body + 4)..]);
                bits = BinaryPrimitives.ReadUInt16LittleEndian(file[(body + 14)..]);
                if (tag == Extensible)
                {
                    if (size < 40 || body + 26 > file.Length) throw new WavException("the extensible fmt chunk is truncated");
                    tag = BinaryPrimitives.ReadUInt16LittleEndian(file[(body + 24)..]); // first two bytes of the sub-format GUID
                }
                haveFormat = true;
            }
            else if (id.SequenceEqual("data"u8))
            {
                if (!haveFormat) throw new WavException("the data chunk comes before the fmt chunk");
                var available = (long)file.Length - body;
                var length = size == 0 || size == 0xFFFFFFFF || size > available ? available : size;
                return Convert(file.Slice(body, (int)length), tag, bits, channels, rate);
            }

            var next = body + (long)size + (size & 1); // chunks are padded to an even size
            if (next > file.Length) break;
            position = (int)next;
        }
        throw new WavException(haveFormat ? "there is no data chunk" : "there is no fmt chunk");
    }

    private static DecodedAudio Convert(ReadOnlySpan<byte> data, ushort tag, ushort bits, ushort channels, uint rate)
    {
        if (channels == 0) throw new WavException("the file declares zero channels");
        if (rate == 0 || rate > int.MaxValue) throw new WavException($"the file declares an unusable sample rate ({rate})");
        var supported = (tag == Pcm && bits is 8 or 16 or 24 or 32) || (tag == Float && bits is 32 or 64);
        if (!supported) throw new WavException($"unsupported WAV encoding (format {tag}, {bits} bits): only PCM 8/16/24/32 and float 32/64 are decoded");

        var bytes = bits / 8;
        var frames = data.Length / (bytes * channels);
        if (frames == 0) throw new WavException("the data chunk holds no complete sample frame");

        var mono = new float[frames];
        var frame = bytes * channels;
        for (var f = 0; f < frames; f++)
        {
            var sum = 0f;
            for (var c = 0; c < channels; c++) sum += Sample(data.Slice(f * frame + c * bytes, bytes), tag, bits);
            mono[f] = channels == 1 ? sum : sum / channels; // float32 sum and division, as numpy's mean over float32
        }
        return new DecodedAudio(mono, (int)rate, channels);
    }

    private static float Sample(ReadOnlySpan<byte> s, ushort tag, ushort bits)
    {
        if (tag == Float) return bits == 32 ? BinaryPrimitives.ReadSingleLittleEndian(s) : (float)BinaryPrimitives.ReadDoubleLittleEndian(s);
        return bits switch
        {
            8 => (s[0] - 128) / 128f,                                              // unsigned
            16 => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
            24 => ((s[0] | (s[1] << 8) | (s[2] << 16)) << 8 >> 8) / 8388608f,      // sign-extended 24-bit
            _ => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
        };
    }
}
