using System.Globalization;

namespace LionClap.Service;

/// <summary>A setting that cannot be used. The message names the environment variable.</summary>
internal sealed class SettingsException(string message) : Exception(message);

/// <summary>What the service is told by its environment, with the variable names of the Python service (plus <c>CLAP_DEVICE</c>).</summary>
/// <param name="CheckpointDirectory">CLAP_CKPT: where the weights are (the Python container's /ckpt layout, or a directory with model.safetensors).</param>
/// <param name="Token">CLAP_TOKEN: bearer token; empty leaves the API open.</param>
/// <param name="Port">CLAP_PORT: the port to listen on, on all interfaces.</param>
/// <param name="Version">CLAP_VERSION: the version string health reports.</param>
/// <param name="Device">CLAP_DEVICE: cpu, cuda or cuda:N.</param>
internal sealed record ServiceSettings(string CheckpointDirectory, string Token, int Port, string Version, string Device)
{
    public const string CkptVariable = "CLAP_CKPT", TokenVariable = "CLAP_TOKEN", PortVariable = "CLAP_PORT", VersionVariable = "CLAP_VERSION", DeviceVariable = "CLAP_DEVICE";

    public static ServiceSettings FromEnvironment() => From(Environment.GetEnvironmentVariable);

    public static ServiceSettings From(Func<string, string?> get)
    {
        string Text(string name, string fallback) => get(name) is { Length: > 0 } v ? v : fallback;

        var port = 8789;
        if (get(PortVariable) is { Length: > 0 } p
            && (!int.TryParse(p, NumberStyles.None, CultureInfo.InvariantCulture, out port) || port < 1 || port > 65535))
            throw new SettingsException($"{PortVariable} must be a port number from 1 to 65535, got '{p}'");

        var device = Text(DeviceVariable, "cpu");
        if (!IsDevice(device)) throw new SettingsException($"{DeviceVariable} must be cpu, cuda or cuda:N, got '{device}'");

        return new ServiceSettings(Text(CkptVariable, "/ckpt"), get(TokenVariable) ?? "", port, Text(VersionVariable, "1"), device.ToLowerInvariant());
    }

    public static bool IsDevice(string d) =>
        d.Equals("cpu", StringComparison.OrdinalIgnoreCase) || d.Equals("cuda", StringComparison.OrdinalIgnoreCase)
        || (d.StartsWith("cuda:", StringComparison.OrdinalIgnoreCase) && int.TryParse(d.AsSpan(5), NumberStyles.None, CultureInfo.InvariantCulture, out _));
}
