using LionClap.Service;

ServiceSettings settings;
try { settings = ServiceSettings.FromEnvironment(); }
catch (SettingsException e)
{
    Console.Error.WriteLine($"error: {e.Message}");
    return 2;
}

using var engine = new ClapEngine(settings);
var gate = new FifoGate();
var api = new ServiceApp(settings, engine, new HealthProbe(settings, gate), gate);
var app = ServiceApp.Build(api, $"http://0.0.0.0:{settings.Port}");
await app.StartAsync();
Console.WriteLine($"lionclap service {settings.Version} listening on :{settings.Port} (weights: {settings.CheckpointDirectory})");
await app.WaitForShutdownAsync();
return 0;
