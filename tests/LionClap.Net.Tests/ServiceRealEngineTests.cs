using System.Net;
using System.Text;
using System.Text.Json;
using Xunit.Abstractions;

namespace LionClap.Tests;

/// <summary>The service with the real model over the Python container's directory: the golden requests must give the reference service's answers.</summary>
public class ServiceRealEngineTests(ITestOutputHelper output)
{
    private static JsonElement Cases() => JsonDocument.Parse(File.ReadAllText(Fixtures.PathOf("classify_cases.json"))).RootElement.GetProperty("cases");

    private static string BodyOf(JsonElement c)
    {
        var request = c.GetProperty("request");
        var wav = Convert.ToBase64String(File.ReadAllBytes(Fixtures.PathOf(c.GetProperty("wav").GetString()!)));
        var sb = new StringBuilder($"{{\"wav_base64\": \"{wav}\", \"labels\": {request.GetProperty("labels").GetRawText()}");
        if (request.TryGetProperty("template", out var t)) sb.Append($", \"template\": {t.GetRawText()}");
        if (request.TryGetProperty("top", out var top)) sb.Append($", \"top\": {top.GetRawText()}");
        return sb.Append('}').ToString();
    }

    private static double AssertSameAsReference(JsonElement c, JsonElement answer)
    {
        var expected = c.GetProperty("response");
        Assert.Equal(expected.GetProperty("seconds").GetDouble(), answer.GetProperty("seconds").GetDouble());
        var want = expected.GetProperty("categories").EnumerateObject().ToList();
        var got = answer.GetProperty("categories").EnumerateObject().ToList();
        Assert.Equal(want.Select(p => p.Name), got.Select(p => p.Name));
        double worst = 0;
        for (var k = 0; k < want.Count; k++)
        {
            var w = want[k].Value.EnumerateArray().ToList();
            var g = got[k].Value.EnumerateArray().ToList();
            Assert.Equal(w.Select(x => x.GetProperty("tag").GetString()), g.Select(x => x.GetProperty("tag").GetString()));
            for (var j = 0; j < w.Count; j++) worst = Math.Max(worst, Math.Abs(w[j].GetProperty("score").GetDouble() - g[j].GetProperty("score").GetDouble()));
        }
        return worst;
    }

    [RequiresHfDir]
    public async Task TheGoldenRequestsOverHttpGiveTheReferenceServicesAnswers()
    {
        await using var s = RunningService.Real(Checkpoint.HfDir!);

        var health = await RunningService.Json(await s.Http.GetAsync("/health"));
        Assert.True(health.GetProperty("ready").GetBoolean());
        Assert.Empty(health.GetProperty("missing").EnumerateArray());
        Assert.Equal(JsonValueKind.Null, health.GetProperty("gpu").ValueKind);
        Assert.False(s.Engine!.IsLoaded);                       // health never loads the model

        foreach (var c in Cases().EnumerateArray())
        {
            var response = await s.Post(BodyOf(c));
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            var worst = AssertSameAsReference(c, await RunningService.Json(response));
            output.WriteLine($"case {c.GetProperty("id").GetInt32()} {c.GetProperty("wav").GetString()}: largest score difference {worst:E2}");
            Assert.True(worst <= 1.1e-3, $"{worst:E2}");
        }
        Assert.True(s.Engine.IsLoaded);                         // loaded by the first classification and kept
    }

    [RequiresHfDir]
    public async Task TwoSimultaneousRequestsAreBothAnsweredCorrectly()
    {
        await using var s = RunningService.Real(Checkpoint.HfDir!);
        var cases = Cases().EnumerateArray().ToList();
        var a = s.Post(BodyOf(cases[1]));
        var b = s.Post(BodyOf(cases[2]));
        var answers = await Task.WhenAll(a, b);
        Assert.All(answers, r => Assert.Equal(HttpStatusCode.OK, r.StatusCode));
        Assert.True(AssertSameAsReference(cases[1], await RunningService.Json(answers[0])) <= 1.1e-3);
        Assert.True(AssertSameAsReference(cases[2], await RunningService.Json(answers[1])) <= 1.1e-3);
        Assert.False(s.Gate.Busy);
    }

    [Fact]
    public async Task ADirectoryWithoutWeightsIsNotReadyAndClassifyAnswers503()
    {
        await using var s = RunningService.Real(TestTemp.Dir());
        var health = await RunningService.Json(await s.Http.GetAsync("/health"));
        Assert.False(health.GetProperty("ready").GetBoolean());
        Assert.Equal("models--laion--clap-htsat-unfused", health.GetProperty("missing")[0].GetString());

        var response = await s.Post(RunningService.Body());
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.False(s.Engine!.IsLoaded);
    }

    [Fact]
    public async Task ABrokenCheckpointIsA500AndTheNextRequestTriesAgain()
    {
        var dir = TestTemp.Dir();
        File.WriteAllText(Path.Combine(dir, "model.safetensors"), "not a safetensors file");
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), "{}");
        await using var s = RunningService.Real(dir);

        Assert.True((await RunningService.Json(await s.Http.GetAsync("/health"))).GetProperty("ready").GetBoolean()); // the files exist: health does not open them
        var first = await s.Post(RunningService.Body());
        Assert.Equal(HttpStatusCode.InternalServerError, first.StatusCode);
        Assert.False(s.Engine!.IsLoaded);
        Assert.Equal(HttpStatusCode.InternalServerError, (await s.Post(RunningService.Body())).StatusCode); // not remembered as loaded
    }
}
