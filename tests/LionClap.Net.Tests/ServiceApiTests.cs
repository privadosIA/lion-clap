using System.Net;
using System.Text.Json;
using LionClap.Service;
using LionClap.Weights;

namespace LionClap.Tests;

public class ServiceApiTests
{
    [Fact]
    public async Task HealthReportsReadinessVersionAndLoad()
    {
        await using var s = new RunningService();
        var response = await s.Http.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType!.MediaType);
        var body = await RunningService.Json(response);
        Assert.True(body.GetProperty("ready").GetBoolean());
        Assert.Equal("7", body.GetProperty("version").GetString());
        Assert.False(body.GetProperty("busy").GetBoolean());
        Assert.Equal(0, body.GetProperty("queued").GetInt32());
        Assert.Equal(JsonValueKind.Null, body.GetProperty("gpu").ValueKind);
        Assert.Empty(body.GetProperty("missing").EnumerateArray());
    }

    [Fact]
    public async Task WithoutWeightsHealthIsNotReadyAndClassifyAnswers503WithTheSameList()
    {
        var absent = new ClapCheckpointFiles(null, WeightsFormat.None, null, null, [ClapCheckpointLocator.HubId]);
        await using var s = new RunningService(locate: () => absent);

        var health = await RunningService.Json(await s.Http.GetAsync("/health"));
        Assert.False(health.GetProperty("ready").GetBoolean());
        Assert.Equal(ClapCheckpointLocator.HubId, health.GetProperty("missing")[0].GetString());

        var response = await s.Post(RunningService.Body());
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        var body = await RunningService.Json(response);
        Assert.Equal("the weights are missing", body.GetProperty("error").GetString());
        Assert.Equal(ClapCheckpointLocator.HubId, body.GetProperty("missing")[0].GetString());
        Assert.Empty(s.Classifier.Calls);
    }

    [Fact]
    public async Task AnUnavailableDeviceIsReportedAsNotReady()
    {
        await using var s = new RunningService(deviceProblem: () => "device cuda is not available");
        var health = await RunningService.Json(await s.Http.GetAsync("/health"));
        Assert.False(health.GetProperty("ready").GetBoolean());
        Assert.Contains("device cuda", health.GetProperty("missing")[0].GetString());
        Assert.Equal(HttpStatusCode.ServiceUnavailable, (await s.Post(RunningService.Body())).StatusCode);
    }

    [Fact]
    public async Task ClassifyAnswers200WithSecondsAndCategoriesInTheOrderOfTheRequest()
    {
        await using var s = new RunningService();
        var response = await s.Post(RunningService.Body(seconds: 2, labels: "{\"zeta\": [\"a\", \"b\"], \"alpha\": [\"c\"]}", extra: ", \"top\": 1, \"template\": \"Music: {}\""));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType!.MediaType);
        var body = await RunningService.Json(response);
        Assert.Equal(2.0, body.GetProperty("seconds").GetDouble());
        Assert.Equal(new[] { "zeta", "alpha" }, body.GetProperty("categories").EnumerateObject().Select(p => p.Name));
        Assert.Equal("a", body.GetProperty("categories").GetProperty("zeta")[0].GetProperty("tag").GetString());
        Assert.Equal(1, body.GetProperty("categories").GetProperty("zeta").GetArrayLength());

        var call = Assert.Single(s.Classifier.Calls);
        Assert.Equal(96000, call.Clip.Length);
        Assert.Equal(("Music: {}", 1), (call.Request.Template, call.Request.Top));
    }

    [Fact]
    public async Task AudioAtAnotherRateIsResampledTo48kBeforeClassifying()
    {
        await using var s = new RunningService();
        Assert.Equal(HttpStatusCode.OK, (await s.Post(RunningService.Body(seconds: 3, rate: 16000))).StatusCode);
        Assert.Equal(144000, Assert.Single(s.Classifier.Calls).Clip.Length);
    }

    public static IEnumerable<object[]> BadBodies()
    {
        var wav = RunningService.Body();
        yield return ["{not json", "not valid JSON"];
        yield return ["[1, 2]", "JSON object"];
        yield return ["{}", "labels"];
        yield return ["{\"labels\": []}", "labels"];
        yield return ["{\"labels\": {}}", "labels"];
        yield return [RunningService.Body(labels: "{\"genre\": \"rock\"}"), "labels.genre"];
        yield return [RunningService.Body(labels: "{\"genre\": [{\"a\": 1}]}"), "every tag must be a string"];
        yield return [RunningService.Body(extra: ", \"template\": \"This is rock\""), "needs a {}"];
        yield return [RunningService.Body(extra: ", \"template\": [1]"), "template must be a string"];
        yield return [RunningService.Body(extra: ", \"top\": \"many\""), "top must be an integer"];
        yield return ["{\"wav_base64\": \"AAA\", \"labels\": {\"g\": [\"a\"]}}", "not valid base64"];
        yield return ["{\"wav_base64\": \"AAAA AAAA\", \"labels\": {\"g\": [\"a\"]}}", "not valid base64"];
        yield return ["{\"wav_base64\": \"AAAA\", \"labels\": {\"g\": [\"a\"]}}", "not a readable WAV"];
        yield return ["{\"labels\": {\"g\": [\"a\"]}}", "not a readable WAV"];
        yield return [RunningService.Body(seconds: 0.5), "shorter than one second"];
        yield return [RunningService.Body(seconds: 1.0), "shorter than one second"];
    }

    [Theory]
    [MemberData(nameof(BadBodies))]
    public async Task InvalidRequestsAre400WithAMessageThatNamesTheProblem(string body, string expectedInMessage)
    {
        await using var s = new RunningService();
        var response = await s.Post(body);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(expectedInMessage, (await RunningService.Json(response)).GetProperty("error").GetString());
        Assert.Empty(s.Classifier.Calls);
    }

    [Fact]
    public async Task OtherPathsAndMethodsAre404()
    {
        await using var s = new RunningService();
        foreach (var (method, path) in new[] { ("GET", "/nope"), ("POST", "/health"), ("GET", "/classify"), ("DELETE", "/classify"), ("GET", "/") })
        {
            var response = await s.Http.SendAsync(new HttpRequestMessage(new HttpMethod(method), path));
            Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
            Assert.Equal("not found", (await RunningService.Json(response)).GetProperty("error").GetString());
        }
    }

    [Fact]
    public async Task ATrailingSlashIsIgnored()
    {
        await using var s = new RunningService();
        Assert.Equal(HttpStatusCode.OK, (await s.Http.GetAsync("/health/")).StatusCode);
    }

    [Fact]
    public async Task ABodyLargerThanTheLimitIsRefusedByItsContentLength()
    {
        await using var s = new RunningService();
        var status = await s.Raw("POST /classify HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 40000001\r\n\r\n");
        Assert.Contains("413", status);
        Assert.Empty(s.Classifier.Calls);
    }

    [Fact]
    public async Task ABodyLargerThanTheLimitIsRefusedWhileItIsCounted()
    {
        await using var s = new RunningService();
        var status = await s.Raw("POST /classify HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n", async stream =>
        {
            var chunk = new byte[1 << 20];
            Array.Fill(chunk, (byte)'a');
            var header = System.Text.Encoding.ASCII.GetBytes($"{chunk.Length:x}\r\n");
            for (var i = 0; i < 42; i++)
            {
                await stream.WriteAsync(header);
                await stream.WriteAsync(chunk);
                await stream.WriteAsync("\r\n"u8.ToArray());
            }
        });
        Assert.Contains("413", status);
        Assert.Empty(s.Classifier.Calls);
    }

    [Fact]
    public async Task AnUnexpectedFailureIs500AndTheServiceKeepsWorking()
    {
        await using var s = new RunningService();
        s.Classifier.Fail = new InvalidOperationException("the model exploded");
        var response = await s.Post(RunningService.Body());
        Assert.Equal(HttpStatusCode.InternalServerError, response.StatusCode);
        Assert.Contains("the model exploded", (await RunningService.Json(response)).GetProperty("error").GetString());

        s.Classifier.Fail = null;
        Assert.Equal(HttpStatusCode.OK, (await s.Post(RunningService.Body())).StatusCode);
        Assert.False(s.Gate.Busy);
    }
}

public class ServiceAuthenticationTests
{
    [Fact]
    public async Task WithATokenEveryCallNeedsIt()
    {
        await using var s = new RunningService(token: "secret");
        foreach (var token in new string?[] { null, "wrong", "secre", "secret2" })
        {
            var response = await s.Post(RunningService.Body(), token);
            Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
            Assert.Equal("unauthorized", (await RunningService.Json(response)).GetProperty("error").GetString());
        }
        Assert.Equal(HttpStatusCode.Unauthorized, (await s.Http.GetAsync("/health")).StatusCode);
        Assert.Empty(s.Classifier.Calls);

        Assert.Equal(HttpStatusCode.OK, (await s.Post(RunningService.Body(), "secret")).StatusCode);
        var health = new HttpRequestMessage(HttpMethod.Get, "/health");
        health.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", "secret");
        Assert.Equal(HttpStatusCode.OK, (await s.Http.SendAsync(health)).StatusCode);
    }

    [Fact]
    public async Task TheDocumentationDoesNotNeedTheToken()
    {
        await using var s = new RunningService(token: "secret");
        Assert.Equal(HttpStatusCode.OK, (await s.Http.GetAsync("/docs")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await s.Http.GetAsync("/openapi.yaml")).StatusCode);
    }

    [Fact]
    public async Task WithoutATokenTheApiIsOpen()
    {
        await using var s = new RunningService();
        Assert.Equal(HttpStatusCode.OK, (await s.Post(RunningService.Body())).StatusCode);
    }
}

public class ServiceQueueTests
{
    [Fact]
    public async Task RequestsRunOneAtATimeInArrivalOrderAndHealthAnswersMeanwhile()
    {
        await using var s = new RunningService();
        s.Classifier.Hold = true;

        var first = s.Post(RunningService.Body(seconds: 2, labels: "{\"first\": [\"a\"]}"));
        Assert.True(await s.Classifier.Started.WaitAsync(TimeSpan.FromSeconds(10)));   // the first is running
        var second = s.Post(RunningService.Body(seconds: 3, labels: "{\"second\": [\"b\"]}"));
        var third = s.Post(RunningService.Body(seconds: 4, labels: "{\"third\": [\"c\"]}"));

        JsonElement health = default;
        for (var i = 0; i < 200; i++)
        {
            health = await RunningService.Json(await s.Http.GetAsync("/health"));     // answers at once although a request is running
            if (health.GetProperty("queued").GetInt32() == 2) break;
            await Task.Delay(25);
        }
        Assert.True(health.GetProperty("busy").GetBoolean());
        Assert.Equal(2, health.GetProperty("queued").GetInt32());
        Assert.False(first.IsCompleted);

        s.Classifier.Hold = false;
        s.Classifier.Finish.Release(3);
        foreach (var r in await Task.WhenAll(first, second, third)) Assert.Equal(HttpStatusCode.OK, r.StatusCode);

        Assert.Equal(1, s.Classifier.MaxConcurrent);
        Assert.Equal(new[] { "first", "second", "third" }, s.Classifier.Calls.Select(c => c.Request.Labels[0].Key));
        var after = await RunningService.Json(await s.Http.GetAsync("/health"));
        Assert.Equal((false, 0), (after.GetProperty("busy").GetBoolean(), after.GetProperty("queued").GetInt32()));
    }
}

public class ServiceDocsTests
{
    [Fact]
    public async Task TheDocumentationPageAndEverythingItLoadsAreServedByTheService()
    {
        await using var s = new RunningService();
        var page = await s.Http.GetAsync("/docs");
        Assert.Equal(HttpStatusCode.OK, page.StatusCode);
        Assert.Equal("text/html", page.Content.Headers.ContentType!.MediaType);
        var html = await page.Content.ReadAsStringAsync();
        Assert.Contains("swagger-ui", html);
        Assert.DoesNotContain("http://", html.Replace("http://www.w3.org", ""));      // nothing is fetched from the network
        Assert.DoesNotContain("https://", html);

        foreach (var reference in new[] { "/docs/swagger-ui/swagger-ui-bundle.js", "/docs/swagger-ui/swagger-ui.css", "/docs/initializer.js", "/docs/swagger-ui/favicon-32x32.png" })
        {
            Assert.Contains(reference, html + await (await s.Http.GetAsync("/docs/initializer.js")).Content.ReadAsStringAsync());
            var asset = await s.Http.GetAsync(reference);
            Assert.Equal(HttpStatusCode.OK, asset.StatusCode);
            Assert.True(asset.Content.Headers.ContentLength > 100, reference);
        }
        Assert.True((await s.Http.GetAsync("/docs/swagger-ui/swagger-ui-bundle.js")).Content.Headers.ContentLength > 1_000_000);

        var initializer = await (await s.Http.GetAsync("/docs/initializer.js")).Content.ReadAsStringAsync();
        Assert.Contains("/openapi.yaml", initializer);
    }

    [Fact]
    public async Task TheOpenApiDescriptionIsServedAsYaml()
    {
        await using var s = new RunningService();
        var response = await s.Http.GetAsync("/openapi.yaml");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/yaml", response.Content.Headers.ContentType!.MediaType);
        Assert.Contains("\nopenapi: 3.0.3", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task HeadWorksAndPathTraversalDoesNot()
    {
        await using var s = new RunningService();
        var head = await s.Http.SendAsync(new HttpRequestMessage(HttpMethod.Head, "/docs"));
        Assert.Equal(HttpStatusCode.OK, head.StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await s.Http.GetAsync("/docs/../openapi.yaml/../x")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await s.Http.GetAsync("/docs/missing.js")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await s.Http.SendAsync(new HttpRequestMessage(HttpMethod.Post, "/docs"))).StatusCode);
    }
}

public class ClassifyRequestParserTests
{
    private static ParsedClassify Parse(string body)
    {
        using var doc = ClassifyRequestParser.ParseJson(System.Text.Encoding.UTF8.GetBytes(body));
        return ClassifyRequestParser.Parse(doc.RootElement);
    }

    [Theory]
    [InlineData("", 3)]
    [InlineData(", \"top\": 0", 3)]
    [InlineData(", \"top\": null", 3)]
    [InlineData(", \"top\": false", 3)]
    [InlineData(", \"top\": \"\"", 3)]
    [InlineData(", \"top\": 50", 10)]
    [InlineData(", \"top\": 5", 5)]
    [InlineData(", \"top\": \"7\"", 7)]
    [InlineData(", \"top\": 2.9", 2)]
    [InlineData(", \"top\": true", 1)]
    [InlineData(", \"top\": -4", 1)]
    [InlineData(", \"top\": 99999999999", 10)]
    public void TopFollowsPythonsIntRulesAndIsClamped(string extra, int expected) =>
        Assert.Equal(expected, Parse(RunningService.Body(extra: extra)).Request.Top);

    [Theory]
    [InlineData("", "This audio is a {} song.")]
    [InlineData(", \"template\": \"\"", "This audio is a {} song.")]
    [InlineData(", \"template\": null", "This audio is a {} song.")]
    [InlineData(", \"template\": \"Sounds like {}\"", "Sounds like {}")]
    public void TheTemplateDefaultsWhenAbsentOrEmpty(string extra, string expected) =>
        Assert.Equal(expected, Parse(RunningService.Body(extra: extra)).Request.Template);

    [Fact]
    public void TagsThatAreNumbersOrBooleansAreTurnedToTextLikeStr()
    {
        var tags = Parse(RunningService.Body(labels: "{\"x\": [\"a\", 7, 2.5, true, false, null]}")).Request.Labels[0].Value;
        Assert.Equal(new[] { "a", "7", "2.5", "True", "False", "None" }, tags);
    }

    [Fact]
    public void CategoriesKeepTheirOrderAndAtMost200TagsAreKept()
    {
        var many = string.Join(",", Enumerable.Range(0, 300).Select(i => $"\"t{i}\""));
        var request = Parse(RunningService.Body(labels: $"{{\"zeta\": [{many}], \"alpha\": []}}")).Request;
        Assert.Equal(new[] { "zeta", "alpha" }, request.Labels.Select(l => l.Key));
        Assert.Equal(200, request.Labels[0].Value.Count);
        Assert.Empty(request.Labels[1].Value);
    }

    [Fact]
    public void AnEmptyBodyIsAnEmptyObjectAndNeedsLabels()
    {
        using var doc = ClassifyRequestParser.ParseJson(ReadOnlySpan<byte>.Empty);
        Assert.Contains("labels", Assert.Throws<BadRequestException>(() => ClassifyRequestParser.Parse(doc.RootElement)).Message);
    }
}
