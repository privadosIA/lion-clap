using LionClap.Classification;
using LionClap.Service;

namespace LionClap.Tests;

public class OpenApiContractTests
{
    private static readonly Dictionary<string, string> Doc = YamlOutline.Read(Docs.OpenApi);

    private static IEnumerable<string> Keys(string prefix) =>
        Doc.Keys.Where(k => k.StartsWith(prefix + ".", StringComparison.Ordinal)).Select(k => k[(prefix.Length + 1)..].Split('.')[0]).Distinct();

    [Fact]
    public void PathsAndMethodsAreExactlyTheRoutesTheServiceAnswers()
    {
        var methods = new[] { "get", "post", "put", "delete", "patch", "head" };
        var documented = Doc.Keys
            .Select(k => k.Split('.'))
            .Where(p => p.Length >= 3 && p[0] == "paths" && p[1].StartsWith('/') && methods.Contains(p[2]))
            .Select(p => $"{p[2].ToUpperInvariant()} {p[1]}")
            .Distinct();
        Assert.Equal(ServiceApp.RouteTable.OrderBy(r => r), documented.OrderBy(r => r));
    }

    [Theory]
    [InlineData("/health", "get", new[] { "200", "401" })]
    [InlineData("/classify", "post", new[] { "200", "400", "401", "413", "500", "503" })]
    public void EveryCallDocumentsTheStatusCodesItCanAnswer(string path, string method, string[] expected)
    {
        var codes = Keys($"paths.{path}.{method}.responses").OrderBy(c => c).ToArray();
        Assert.Equal(expected, codes);
    }

    [Fact]
    public void TheDocumentedLimitsAndDefaultsAreTheOnesTheServiceEnforces()
    {
        const string request = "components.schemas.ClassifyRequest.properties";
        Assert.Equal(ClassifyRequest.DefaultTop.ToString(), Doc[$"{request}.top.default"]);
        Assert.Equal("1", Doc[$"{request}.top.minimum"]);
        Assert.Equal(ClassifyRequest.MaxTop.ToString(), Doc[$"{request}.top.maximum"]);
        Assert.Equal(ClassifyRequest.MaxTagsPerCategory.ToString(), Doc[$"{request}.labels.additionalProperties.maxItems"]);
        Assert.Equal(ClassifyRequest.DefaultTemplate, Doc[$"{request}.template.default"]);
        Assert.Equal(ServiceApp.MaxBodyBytes.ToString(), Doc["info.x-max-body-bytes"]);
        Assert.Contains("40,000,000 bytes", Docs.OpenApi);
        Assert.Contains("This audio is a {} song.", Docs.OpenApi);
    }

    [Fact]
    public void TheSchemasListTheFieldsTheServiceSends()
    {
        Assert.Equal(new[] { "busy", "gpu", "missing", "queued", "ready", "version" }, Keys("components.schemas.Health.properties").OrderBy(k => k));
        Assert.Equal(new[] { "labels", "template", "top", "wav_base64" }, Keys("components.schemas.ClassifyRequest.properties").OrderBy(k => k));
        Assert.Equal(new[] { "categories", "seconds" }, Keys("components.schemas.ClassifyResponse.properties").OrderBy(k => k));
        Assert.Equal(new[] { "score", "tag" }, Keys("components.schemas.TagScore.properties").OrderBy(k => k));
        Assert.Equal(new[] { "error", "missing" }, Keys("components.schemas.WeightsMissing.properties").OrderBy(k => k));
        Assert.Equal(new[] { "free_mb", "name", "total_mb" }, Keys("components.schemas.Health.properties.gpu.properties").OrderBy(k => k));
    }

    [Fact]
    public void TheBearerSchemeIsDeclaredAndTheDocumentIsOpenApi3()
    {
        Assert.Equal("3.0.3", Doc["openapi"]);
        Assert.Equal("bearer", Doc["components.securitySchemes.bearerAuth.scheme"]);
        Assert.Equal("http", Doc["components.securitySchemes.bearerAuth.type"]);
    }
}
