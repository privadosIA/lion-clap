using System.Text.Json;

namespace LionClap.Tests;

/// <summary>The five reference texts of <c>text_cases.json</c> with their ids and masks (length 77).</summary>
public static class TextCases
{
    public sealed record Case(string Text, int Length, long[] Ids, long[] Mask);

    public static IReadOnlyList<Case> All { get; } = Load();

    /// <summary>Ids and mask of the chosen cases as int64 tensors <c>[n, 77]</c>; the caller disposes them.</summary>
    public static (TorchSharp.torch.Tensor Ids, TorchSharp.torch.Tensor Mask) Tensors(params int[] cases)
    {
        var ids = cases.SelectMany(i => All[i].Ids).ToArray();
        var mask = cases.SelectMany(i => All[i].Mask).ToArray();
        return (TorchSharp.torch.tensor(ids, [cases.Length, 77]), TorchSharp.torch.tensor(mask, [cases.Length, 77]));
    }

    private static List<Case> Load()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Fixtures.PathOf("text_cases.json")));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(
            c.GetProperty("text").GetString()!,
            c.GetProperty("length").GetInt32(),
            c.GetProperty("input_ids").EnumerateArray().Select(x => x.GetInt64()).ToArray(),
            c.GetProperty("attention_mask").EnumerateArray().Select(x => x.GetInt64()).ToArray())).ToList();
    }
}
