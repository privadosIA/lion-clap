using System.Text;
using System.Text.Json;
using LionClap.Audio;
using LionClap.Classification;
using Xunit.Abstractions;

namespace LionClap.Tests;

internal sealed class FakeClip(float[] vector, int windows = 1) : IClipEmbedder
{
    public ClipEmbedding Embed(float[] clip48k) => new(vector, windows);
}

/// <summary>Maps the text it is given (the tag already inside the template) to a vector, and remembers what it was asked.</summary>
internal sealed class FakeText(Func<string, float[]> map) : ITextEmbedder
{
    public List<string> Asked { get; } = new();
    public float[][] Embed(IReadOnlyList<string> texts)
    {
        Asked.AddRange(texts);
        return texts.Select(map).ToArray();
    }
}

public class ClapClassifierTests(ITestOutputHelper output)
{
    private static readonly float[] Clip = new float[48000 * 2];

    private static IReadOnlyList<KeyValuePair<string, IReadOnlyList<string>>> Labels(params (string Category, string[] Tags)[] categories) =>
        categories.Select(c => new KeyValuePair<string, IReadOnlyList<string>>(c.Category, c.Tags)).ToList();

    /// <summary>A classifier whose audio vector is [1] and where the score of a tag is looked up by name.</summary>
    private static (ClapClassifier Classifier, FakeText Text) With(Dictionary<string, float> scores) =>
        (new ClapClassifier(new FakeClip([1f]), new FakeText(t => [scores[ScoreKey(t)]])), null!);

    private static string ScoreKey(string templated) => templated.Replace("This audio is a ", "").Replace(" song.", "");

    private static ClassifyResult Run(Dictionary<string, float> scores, ClassifyRequest request, float[]? clip = null, FakeText? text = null)
    {
        text ??= new FakeText(t => [scores[ScoreKey(t)]]);
        return new ClapClassifier(new FakeClip([1f]), text).Classify(clip ?? Clip, request);
    }

    [Fact]
    public void TagsComeBackOrderedByDescendingScore()
    {
        var scores = new Dictionary<string, float> { ["a"] = 0.9f, ["b"] = 0.5f, ["c"] = 0.7f };
        var result = Run(scores, ClassifyRequest.Create(Labels(("x", ["a", "b", "c"]))));
        Assert.Equal(new[] { "a", "c", "b" }, result.Categories[0].Value.Select(t => t.Tag));
    }

    [Fact]
    public void TiesKeepTheRequestOrder()
    {
        var scores = new Dictionary<string, float> { ["a"] = 0.5f, ["b"] = 0.5f, ["c"] = 0.5f, ["d"] = 0.9f };
        var result = Run(scores, ClassifyRequest.Create(Labels(("x", ["a", "b", "c", "d"]))));
        Assert.Equal(new[] { "d", "a", "b" }, result.Categories[0].Value.Select(t => t.Tag));
    }

    [Theory]
    [InlineData(null, 3)]
    [InlineData(0, 3)]
    [InlineData(1, 1)]
    [InlineData(5, 5)]
    [InlineData(10, 10)]
    [InlineData(50, 10)]
    [InlineData(-2, 1)]
    public void TopIsClampedAndZeroMeansTheDefault(int? top, int expected) => Assert.Equal(expected, ClassifyRequest.ClampTop(top));

    [Fact]
    public void FewerTagsThanTopReturnsAllOfThem()
    {
        var scores = new Dictionary<string, float> { ["a"] = 0.1f, ["b"] = 0.2f };
        Assert.Equal(2, Run(scores, ClassifyRequest.Create(Labels(("x", ["a", "b"])), top: 50)).Categories[0].Value.Count);
    }

    [Fact]
    public void OnlyTheFirst200TagsOfACategoryAreUsed()
    {
        var tags = Enumerable.Range(0, 300).Select(i => $"t{i}").ToArray();
        var scores = tags.ToDictionary(t => t, t => t == "t250" ? 0.99f : 0.1f);   // the best tag is beyond the cap
        var text = new FakeText(t => [scores[ScoreKey(t)]]);
        var result = Run(scores, ClassifyRequest.Create(Labels(("x", tags)), top: 10), text: text);
        Assert.Equal(200, text.Asked.Count);
        Assert.DoesNotContain(result.Categories[0].Value, t => t.Tag == "t250");
    }

    [Fact]
    public void AnEmptyCategoryGivesAnEmptyListAndIsNotEmbedded()
    {
        var scores = new Dictionary<string, float> { ["a"] = 0.3f };
        var text = new FakeText(t => [scores[ScoreKey(t)]]);
        var result = Run(scores, ClassifyRequest.Create(Labels(("empty", []), ("full", ["a"]))), text: text);
        Assert.Empty(result.Categories[0].Value);
        Assert.Single(result.Categories[1].Value);
        Assert.Single(text.Asked);
    }

    [Fact]
    public void CategoriesKeepTheOrderOfTheRequest()
    {
        var scores = new Dictionary<string, float> { ["a"] = 0.3f };
        var result = Run(scores, ClassifyRequest.Create(Labels(("zeta", ["a"]), ("alpha", ["a"]), ("mid", ["a"]))));
        Assert.Equal(new[] { "zeta", "alpha", "mid" }, result.Categories.Select(c => c.Key));
    }

    [Fact]
    public void TheTagIsPutInTheTemplate()
    {
        var text = new FakeText(_ => [0.5f]);
        new ClapClassifier(new FakeClip([1f]), text).Classify(Clip, ClassifyRequest.Create(Labels(("x", ["rock", "hip hop"])), template: "A recording of {} music."));
        Assert.Equal(new[] { "A recording of rock music.", "A recording of hip hop music." }, text.Asked);

        var defaults = new FakeText(_ => [0.5f]);
        new ClapClassifier(new FakeClip([1f]), defaults).Classify(Clip, ClassifyRequest.Create(Labels(("x", ["rock"])), template: ""));
        Assert.Equal(new[] { "This audio is a rock song." }, defaults.Asked);
    }

    [Fact]
    public void ScoresAreRoundedToThreeDecimals()
    {
        var scores = new Dictionary<string, float> { ["a"] = 2f / 3f, ["b"] = 0.12349f, ["c"] = -0.0004f };
        var result = Run(scores, ClassifyRequest.Create(Labels(("x", ["a", "b", "c"]))));
        Assert.Equal(new[] { 0.667, 0.123, -0.0 }, result.Categories[0].Value.Select(t => t.Score));
    }

    [Theory]
    [InlineData(72000, 1.5)]
    [InlineData(100000, 2.1)]
    [InlineData(576000, 12.0)]
    public void SecondsAreRoundedToOneDecimal(int samples, double expected) =>
        Assert.Equal(expected, Run(new Dictionary<string, float> { ["a"] = 0.1f }, ClassifyRequest.Create(Labels(("x", ["a"]))), new float[samples]).Seconds);

    [Fact]
    public void ADimensionMismatchIsReported()
    {
        var classifier = new ClapClassifier(new FakeClip([1f, 0f]), new FakeText(_ => [1f]));
        Assert.Throws<InvalidOperationException>(() => classifier.Classify(Clip, ClassifyRequest.Create(Labels(("x", ["a"])))));
    }

    [Fact]
    public void TheJsonIsTheReferenceShapeWithCategoriesInOrderAndAsciiEscapes()
    {
        var result = new ClassifyResult(12.0, [
            new("genre", [new ScoredTag("rock", 0.2), new ScoredTag("música", 0.179)]),
            new("empty", []),
        ]);
        var json = Encoding.UTF8.GetString(result.ToJson());
        output.WriteLine(json);
        Assert.Equal("{\"seconds\":12.0,\"categories\":{\"genre\":[{\"tag\":\"rock\",\"score\":0.2},{\"tag\":\"m\\u00FAsica\",\"score\":0.179}],\"empty\":[]}}", json);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal("música", doc.RootElement.GetProperty("categories").GetProperty("genre")[1].GetProperty("tag").GetString());
    }
}

public class ClapClassifierGoldenTests(BranchFixture audio, TextBranchFixture text, ITestOutputHelper output) : IClassFixture<BranchFixture>, IClassFixture<TextBranchFixture>
{
    [RequiresCheckpoint]
    public void TheFourGoldenRequestsGiveTheReferenceResponses()
    {
        var classifier = new ClapClassifier(new ClapAudioEmbedder(audio.Branch, new LogMelExtractor()), new ClapTextEmbedder(text.Branch, text.Tokenizer));
        using var doc = JsonDocument.Parse(File.ReadAllText(Fixtures.PathOf("classify_cases.json")));
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            var request = c.GetProperty("request");
            var labels = request.GetProperty("labels").EnumerateObject()
                .Select(p => new KeyValuePair<string, IReadOnlyList<string>>(p.Name, p.Value.EnumerateArray().Select(x => x.GetString()!).ToList())).ToList();
            var parsed = ClassifyRequest.Create(labels,
                request.TryGetProperty("template", out var t) ? t.GetString() : null,
                request.TryGetProperty("top", out var top) ? top.GetInt32() : null);

            var clip = ClapAudioEmbedder.To48k(WavDecoder.Decode(File.ReadAllBytes(Fixtures.PathOf(c.GetProperty("wav").GetString()!))));
            var result = classifier.Classify(clip, parsed);

            var expected = c.GetProperty("response");
            Assert.Equal(expected.GetProperty("seconds").GetDouble(), result.Seconds);
            var expectedCategories = expected.GetProperty("categories").EnumerateObject().ToList();
            Assert.Equal(expectedCategories.Select(p => p.Name), result.Categories.Select(x => x.Key));

            double worst = 0;
            for (var k = 0; k < expectedCategories.Count; k++)
            {
                var want = expectedCategories[k].Value.EnumerateArray().ToList();
                var got = result.Categories[k].Value;
                Assert.Equal(want.Select(w => w.GetProperty("tag").GetString()), got.Select(g => g.Tag));
                for (var j = 0; j < want.Count; j++) worst = Math.Max(worst, Math.Abs(want[j].GetProperty("score").GetDouble() - got[j].Score));
            }
            output.WriteLine($"case {c.GetProperty("id").GetInt32()} {c.GetProperty("wav").GetString()}: largest score difference {worst:E2}");
            Assert.True(worst <= 1.1e-3, $"largest score difference {worst:E2}");
        }
    }
}
