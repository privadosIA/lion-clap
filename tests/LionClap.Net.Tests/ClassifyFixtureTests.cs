using System.Text.Json;

namespace LionClap.Tests;

public class ClassifyFixtureTests
{
    private static JsonElement Cases()
    {
        var doc = JsonDocument.Parse(File.ReadAllText(Fixtures.PathOf("classify_cases.json")));
        return doc.RootElement.GetProperty("cases");
    }

    [Fact]
    public void EveryCaseReferencesAnExistingWavAndKeepsTheRequestCategoryOrder()
    {
        var cases = Cases().EnumerateArray().ToList();
        Assert.Equal(4, cases.Count);
        foreach (var c in cases)
        {
            Assert.True(File.Exists(Fixtures.PathOf(c.GetProperty("wav").GetString()!)));
            var requested = c.GetProperty("request").GetProperty("labels").EnumerateObject().Select(p => p.Name).ToList();
            var answered = c.GetProperty("response").GetProperty("categories").EnumerateObject().Select(p => p.Name).ToList();
            Assert.Equal(requested, answered);
        }
    }

    [Fact]
    public void TheFourInputsDifferInRateChannelsAndFormat()
    {
        var cases = Cases().EnumerateArray().ToList();
        Assert.Equal(new[] { 16000, 22050, 48000, 44100 }, cases.Select(c => c.GetProperty("sample_rate").GetInt32()));
        Assert.Equal(new[] { 1, 2, 1, 1 }, cases.Select(c => c.GetProperty("channels").GetInt32()));
        Assert.Equal(new[] { "PCM_16", "PCM_16", "FLOAT", "PCM_24" }, cases.Select(c => c.GetProperty("subtype").GetString()));
        Assert.Equal(3, cases[0].GetProperty("windows").GetInt32());
    }

    [Fact]
    public void EmbeddingFixtureHasOneAudioVectorPerCaseAndTheResampledSignal()
    {
        var f = Fixtures.Load("classify_embeddings.safetensors");
        for (var i = 0; i < 4; i++) Assert.Equal(new long[] { 512 }, f[$"audio_{i}"].shape);
        Assert.Equal(new long[] { 72000 }, f["resampled_3"].shape); // 1.5 s at 48 kHz
        Assert.Equal(new long[] { 8, 512 }, f["text_0_0"].shape);
    }
}
