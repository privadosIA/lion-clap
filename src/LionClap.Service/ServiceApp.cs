using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Server.Kestrel.Core;

namespace LionClap.Service;

/// <summary>The HTTP interface of the service: the two calls of the Python service, with its paths, bodies and status codes.</summary>
internal sealed class ServiceApp
{
    /// <summary>The largest body accepted (the reference's limit: about three minutes of 48 kHz mono WAV in base64).</summary>
    public const int MaxBodyBytes = 40_000_000;

    /// <summary>The calls this class answers (keep in step with <see cref="Handle"/>; a test compares it with the OpenAPI description).</summary>
    public static readonly string[] RouteTable = { "GET /health", "POST /classify" };

    private readonly ServiceSettings _settings;
    private readonly IClassifier _classifier;
    private readonly HealthProbe _health;
    private readonly FifoGate _gate;
    private readonly byte[] _expectedAuth;

    public ServiceApp(ServiceSettings settings, IClassifier classifier, HealthProbe health, FifoGate gate)
    {
        _settings = settings;
        _classifier = classifier;
        _health = health;
        _gate = gate;
        _expectedAuth = Encoding.UTF8.GetBytes("Bearer " + settings.Token);
    }

    /// <summary>Builds the web application listening on <paramref name="url"/>.</summary>
    public static WebApplication Build(ServiceApp api, string url)
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls(url);
        builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = null); // the body is counted here against MaxBodyBytes (Kestrel's own cap is 30 MB)
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        var app = builder.Build();
        app.Run(api.Handle);
        return app;
    }

    private async Task Handle(HttpContext ctx)
    {
        try
        {
            var path = ctx.Request.Path.Value ?? "/";
            if (path.Length > 1) path = path.TrimEnd('/');
            var method = ctx.Request.Method;

            if (Docs.TryServe(ctx, path, method) is { } doc) { await doc; return; }

            if (_settings.Token.Length > 0 && !Authorized(ctx)) { await Json(ctx, 401, new ErrorBody("unauthorized")); return; }

            if (method == "GET" && path == "/health") { await Json(ctx, 200, _health.Get()); return; }
            if (method == "POST" && path == "/classify") { await Classify(ctx); return; }
            await Json(ctx, 404, new ErrorBody("not found"));
        }
        catch (Exception e) when (!ctx.Response.HasStarted)
        {
            await Json(ctx, 500, new ErrorBody(e.Message));
        }
    }

    private bool Authorized(HttpContext ctx)
    {
        var given = Encoding.UTF8.GetBytes(ctx.Request.Headers.Authorization.ToString());
        return given.Length == _expectedAuth.Length && CryptographicOperations.FixedTimeEquals(given, _expectedAuth);
    }

    private async Task Classify(HttpContext ctx)
    {
        if (ctx.Request.ContentLength > MaxBodyBytes) { await Json(ctx, 413, new ErrorBody("the request is too large")); return; }
        var buffer = new MemoryStream();
        var chunk = new byte[64 * 1024];
        int n;
        while ((n = await ctx.Request.Body.ReadAsync(chunk)) > 0)
        {
            buffer.Write(chunk, 0, n);
            if (buffer.Length > MaxBodyBytes) { await Json(ctx, 413, new ErrorBody("the request is too large")); return; }
        }

        JsonDocument? doc = null;
        try
        {
            try { doc = ClassifyRequestParser.ParseJson(buffer.GetBuffer().AsSpan(0, (int)buffer.Length)); }
            catch (BadRequestException e) { await Json(ctx, 400, new ErrorBody(e.Message)); return; }

            var missing = _health.Missing();
            if (missing.Count > 0) { await Json(ctx, 503, new MissingBody("the weights are missing", missing)); return; }

            ParsedClassify parsed;
            try { parsed = ClassifyRequestParser.Parse(doc.RootElement); }
            catch (BadRequestException e) { await Json(ctx, 400, new ErrorBody(e.Message)); return; }

            var result = await _gate.RunAsync(() => _classifier.Classify(parsed.Clip48k, parsed.Request));
            var bytes = result.ToJson();
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            ctx.Response.ContentLength = bytes.Length;
            await ctx.Response.Body.WriteAsync(bytes);
        }
        finally { doc?.Dispose(); }
    }

    // ---- JSON ----------------------------------------------------------------------------------------------------------------------

    private static readonly JsonSerializerOptions JsonOptions = new() { DefaultIgnoreCondition = JsonIgnoreCondition.Never };

    internal static async Task Json<T>(HttpContext ctx, int status, T body)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(body, JsonOptions);
        ctx.Response.StatusCode = status;
        ctx.Response.ContentType = "application/json";
        ctx.Response.ContentLength = bytes.Length;
        await ctx.Response.Body.WriteAsync(bytes);
    }

    internal sealed record ErrorBody([property: JsonPropertyName("error")] string Error);

    internal sealed record MissingBody([property: JsonPropertyName("error")] string Error, [property: JsonPropertyName("missing")] IReadOnlyList<string> Missing);
}

/// <summary>The documentation endpoints (<c>/docs</c>, <c>/openapi.yaml</c>), served from the executable itself, without a token.</summary>
internal static class Docs
{
    private static readonly Assembly Assembly = typeof(Docs).Assembly;

    private static string ContentType(string name) => Path.GetExtension(name) switch
    {
        ".html" => "text/html; charset=utf-8",
        ".js" => "text/javascript; charset=utf-8",
        ".css" => "text/css; charset=utf-8",
        ".png" => "image/png",
        ".yaml" => "application/yaml; charset=utf-8",
        _ => "text/plain; charset=utf-8",
    };

    /// <summary>The text of the OpenAPI description.</summary>
    public static string OpenApi
    {
        get
        {
            using var s = Assembly.GetManifestResourceStream("openapi.yaml")!;
            using var r = new StreamReader(s, Encoding.UTF8);
            return r.ReadToEnd();
        }
    }

    public static Task? TryServe(HttpContext ctx, string path, string method)
    {
        string? resource = path switch
        {
            "/openapi.yaml" => "openapi.yaml",
            "/docs" => "docs/index.html",
            _ when path.StartsWith("/docs/", StringComparison.Ordinal) && !path.Contains("..") => "docs/" + path["/docs/".Length..],
            _ => null,
        };
        if (resource is null || (method != "GET" && method != "HEAD")) return null;
        var stream = Assembly.GetManifestResourceStream(resource);
        if (stream is null) return null;
        return Send(ctx, stream, resource);
    }

    private static async Task Send(HttpContext ctx, Stream stream, string resource)
    {
        await using (stream)
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = ContentType(resource);
            ctx.Response.ContentLength = stream.Length;
            if (ctx.Request.Method != "HEAD") await stream.CopyToAsync(ctx.Response.Body);
        }
    }
}
