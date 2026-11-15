using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using LionClap.Classification;
using LionClap.Service;
using LionClap.Weights;
using Microsoft.AspNetCore.Builder;

namespace LionClap.Tests;

/// <summary>A classifier that needs no model: it answers a fixed result, remembers what it was asked and can be held until a test lets it go.</summary>
internal sealed class FakeClassifier : IClassifier
{
    public readonly List<(float[] Clip, ClassifyRequest Request)> Calls = new();
    public readonly SemaphoreSlim Started = new(0);
    public readonly SemaphoreSlim Finish = new(0);
    public volatile bool Hold;
    public volatile int Concurrent, MaxConcurrent;
    public Exception? Fail;

    public ClassifyResult Classify(float[] clip48k, ClassifyRequest request)
    {
        lock (Calls) Calls.Add((clip48k, request));
        var now = Interlocked.Increment(ref Concurrent);
        MaxConcurrent = Math.Max(MaxConcurrent, now);
        try
        {
            Started.Release();
            if (Fail is { } f) throw f;
            if (Hold) Finish.Wait();
            return new ClassifyResult(Math.Round(clip48k.Length / 48000.0, 1), request.Labels
                .Select(c => new KeyValuePair<string, IReadOnlyList<ScoredTag>>(c.Key, c.Value.Take(request.Top).Select((t, i) => new ScoredTag(t, 0.5 - i * 0.1)).ToList())).ToList());
        }
        finally { Interlocked.Decrement(ref Concurrent); }
    }
}

/// <summary>The service running in this process on a free port, with a fake classifier and a health probe that does not touch the disk.</summary>
internal sealed class RunningService : IAsyncDisposable
{
    public FakeClassifier Classifier { get; }
    public FifoGate Gate { get; } = new();
    public HttpClient Http { get; }
    public int Port { get; }
    private readonly WebApplication _app;
    private readonly IDisposable? _owned;

    public static ClapCheckpointFiles Ready = new("/w/model.safetensors", WeightsFormat.SafeTensors, null, "/w/tokenizer.json", []);

    public RunningService(string token = "", Func<ClapCheckpointFiles>? locate = null, Func<string?>? deviceProblem = null)
        : this(new FakeClassifier(), "/ckpt", token, locate ?? (() => Ready), deviceProblem ?? (() => null), null) { }

    /// <summary>The service with the real model over a checkpoint directory (the real locator, the real device check).</summary>
    public static RunningService Real(string checkpointDirectory, string token = "")
    {
        var engine = new ClapEngine(new ServiceSettings(checkpointDirectory, token, 0, "7", "cpu"));
        return new RunningService(engine, checkpointDirectory, token, null, null, engine);
    }

    private RunningService(IClassifier classifier, string checkpointDirectory, string token, Func<ClapCheckpointFiles>? locate, Func<string?>? deviceProblem, IDisposable? owned)
    {
        Classifier = (classifier as FakeClassifier)!;
        Engine = classifier as ClapEngine;
        _owned = owned;
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        Port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();

        var settings = new ServiceSettings(checkpointDirectory, token, Port, "7", "cpu");
        var health = new HealthProbe(settings, Gate, TimeSpan.FromMilliseconds(1));
        if (locate is not null) health = new HealthProbe(settings, Gate, TimeSpan.FromMilliseconds(1)) { Locate = locate, DeviceProblem = deviceProblem ?? (() => null), Gpu = () => null };
        _app = ServiceApp.Build(new ServiceApp(settings, classifier, health, Gate), $"http://127.0.0.1:{Port}");
        _app.StartAsync().GetAwaiter().GetResult();
        Http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{Port}"), Timeout = TimeSpan.FromSeconds(120) };
    }

    public ClapEngine? Engine { get; }

    /// <summary>A valid request body: <paramref name="seconds"/> of a 48 kHz mono 16-bit sine, and the given labels.</summary>
    public static string Body(double seconds = 2, string labels = "{\"genre\": [\"rock\", \"jazz\", \"folk\"], \"mood\": [\"calm\"]}", string extra = "", int rate = 48000)
    {
        var n = (int)(seconds * rate);
        var samples = Enumerable.Range(0, n).Select(i => (short)(8000 * Math.Sin(i * 0.05))).ToArray();
        var wav = new WavBuilder { Rate = (uint)rate, Data = WavBuilder.Pcm16(samples) }.Build();
        return $"{{\"wav_base64\": \"{Convert.ToBase64String(wav)}\", \"labels\": {labels}{extra}}}";
    }

    public Task<HttpResponseMessage> Post(string body, string? token = null)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/classify") { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        if (token is not null) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        return Http.SendAsync(request);
    }

    public static async Task<JsonElement> Json(HttpResponseMessage response)
    {
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return doc.RootElement.Clone();
    }

    /// <summary>Sends raw bytes to the service over TCP and returns the status line (for bodies that HttpClient would not let us send).</summary>
    public async Task<string> Raw(string head, Func<NetworkStream, Task>? body = null)
    {
        using var client = new TcpClient();
        await client.ConnectAsync(IPAddress.Loopback, Port);
        var stream = client.GetStream();
        await stream.WriteAsync(Encoding.ASCII.GetBytes(head));
        if (body is not null)
        {
            try { await body(stream); } catch (IOException) { } // the service may answer and close before we finish sending
        }
        var buffer = new byte[256];
        var read = await stream.ReadAsync(buffer);
        return Encoding.ASCII.GetString(buffer, 0, read).Split("\r\n")[0];
    }

    public async ValueTask DisposeAsync()
    {
        Http.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        _owned?.Dispose();
    }
}
