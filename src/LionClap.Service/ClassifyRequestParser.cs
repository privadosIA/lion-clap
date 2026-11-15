using System.Globalization;
using System.Text.Json;
using LionClap.Audio;
using LionClap.Classification;

namespace LionClap.Service;

/// <summary>A request the service refuses; the message goes to the client in a <c>400 {"error": …}</c>.</summary>
internal sealed class BadRequestException(string message) : Exception(message);

/// <summary>A request ready to run: the labels, template and top, and the audio already decoded, mixed to mono and resampled to 48 kHz.</summary>
internal sealed record ParsedClassify(ClassifyRequest Request, float[] Clip48k);

/// <summary>Turns the JSON body of <c>POST /classify</c> into a <see cref="ParsedClassify"/>, checking in the order of the reference service: labels, template, top, then the audio.</summary>
internal static class ClassifyRequestParser
{
    public static JsonDocument ParseJson(ReadOnlySpan<byte> body)
    {
        JsonDocument doc;
        try { doc = JsonDocument.Parse(body.IsEmpty ? "{}"u8.ToArray() : body.ToArray()); }
        catch (JsonException e) { throw new BadRequestException($"the body is not valid JSON: {e.Message}"); }
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
        {
            doc.Dispose();
            throw new BadRequestException("the body must be a JSON object");
        }
        return doc;
    }

    public static ParsedClassify Parse(JsonElement body)
    {
        var labels = Labels(body);
        var template = Template(body);
        var top = Top(body);
        var clip = Audio(body);
        return new ParsedClassify(ClassifyRequest.Create(labels, template, top), clip);
    }

    private static List<KeyValuePair<string, IReadOnlyList<string>>> Labels(JsonElement body)
    {
        if (!body.TryGetProperty("labels", out var labels) || labels.ValueKind != JsonValueKind.Object || !labels.EnumerateObject().Any())
            throw new BadRequestException("labels: an object {category: [tags]} is needed");
        var result = new List<KeyValuePair<string, IReadOnlyList<string>>>();
        foreach (var category in labels.EnumerateObject())
        {
            if (category.Value.ValueKind != JsonValueKind.Array)
                throw new BadRequestException($"labels.{category.Name}: a list of tags is needed");
            result.Add(new(category.Name, category.Value.EnumerateArray().Select(t => Tag(category.Name, t)).ToList()));
        }
        return result;
    }

    // the reference turns every tag with str(): numbers and booleans are accepted the same way, anything else is refused
    private static string Tag(string category, JsonElement tag) => tag.ValueKind switch
    {
        JsonValueKind.String => tag.GetString()!,
        JsonValueKind.Number => tag.GetRawText(),
        JsonValueKind.True => "True",
        JsonValueKind.False => "False",
        JsonValueKind.Null => "None",
        _ => throw new BadRequestException($"labels.{category}: every tag must be a string"),
    };

    private static string? Template(JsonElement body)
    {
        if (!body.TryGetProperty("template", out var t)) return null;
        string template;
        switch (t.ValueKind)
        {
            case JsonValueKind.String: template = t.GetString()!; break;
            case JsonValueKind.Null or JsonValueKind.False: return null;
            case JsonValueKind.Number when t.GetDouble() == 0: return null;
            default: throw new BadRequestException("template must be a string");
        }
        if (template.Length == 0) return null;
        if (!template.Contains("{}", StringComparison.Ordinal)) throw new BadRequestException("template needs a {} where the tag goes");
        return template;
    }

    private static int? Top(JsonElement body)
    {
        if (!body.TryGetProperty("top", out var t)) return null;
        long value;
        switch (t.ValueKind)
        {
            case JsonValueKind.Null or JsonValueKind.False: return null;
            case JsonValueKind.True: value = 1; break;
            case JsonValueKind.Number: value = (long)Math.Clamp(Math.Truncate(t.GetDouble()), -1e9, 1e9); break;
            case JsonValueKind.String when t.GetString()!.Length == 0: return null;
            case JsonValueKind.String when long.TryParse(t.GetString()!.Trim(), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var parsed): value = parsed; break;
            default: throw new BadRequestException("top must be an integer");
        }
        return (int)Math.Clamp(value, int.MinValue, int.MaxValue);
    }

    private static float[] Audio(JsonElement body)
    {
        var text = body.TryGetProperty("wav_base64", out var w) && w.ValueKind == JsonValueKind.String ? w.GetString()! : "";
        var bytes = StrictBase64(text);
        DecodedAudio audio;
        try { audio = WavDecoder.Decode(bytes); }
        catch (WavException e) { throw new BadRequestException($"wav_base64 is not a readable WAV: {e.Message}"); }

        if (audio.Mono.Length < audio.SampleRate) throw new BadRequestException("the audio is shorter than one second");
        var clip = ClapAudioEmbedder.To48k(audio);
        if (clip.Length <= ClapAudioEmbedder.SampleRate) throw new BadRequestException("the audio is shorter than one second");
        return clip;
    }

    // base64.b64decode(validate=True): only the alphabet, padding at the end and a length that is a multiple of four
    private static byte[] StrictBase64(string text)
    {
        if (text.Length % 4 != 0) throw new BadRequestException("wav_base64 is not valid base64");
        var pad = text.EndsWith("==", StringComparison.Ordinal) ? 2 : text.EndsWith('=') ? 1 : 0;
        for (var i = 0; i < text.Length - pad; i++)
        {
            var c = text[i];
            if (!(c is >= 'A' and <= 'Z' or >= 'a' and <= 'z' or >= '0' and <= '9' or '+' or '/'))
                throw new BadRequestException("wav_base64 is not valid base64");
        }
        try { return Convert.FromBase64String(text); }
        catch (FormatException) { throw new BadRequestException("wav_base64 is not valid base64"); }
    }
}
