using LionClap.Service;

namespace LionClap.Tests;

public class ServiceSettingsTests
{
    private static ServiceSettings From(params (string Name, string Value)[] env) =>
        ServiceSettings.From(name => env.FirstOrDefault(e => e.Name == name).Value);

    [Fact]
    public void DefaultsAreTheReferenceServices()
    {
        var s = From();
        Assert.Equal(("/ckpt", "", 8789, "1", "cpu"), (s.CheckpointDirectory, s.Token, s.Port, s.Version, s.Device));
    }

    [Fact]
    public void EveryVariableIsRead()
    {
        var s = From(("CLAP_CKPT", "/models/clap"), ("CLAP_TOKEN", "secret"), ("CLAP_PORT", "9000"), ("CLAP_VERSION", "7"), ("CLAP_DEVICE", "CUDA:1"));
        Assert.Equal(("/models/clap", "secret", 9000, "7", "cuda:1"), (s.CheckpointDirectory, s.Token, s.Port, s.Version, s.Device));
    }

    [Fact]
    public void EmptyValuesFallBackToTheDefaults()
    {
        var s = From(("CLAP_CKPT", ""), ("CLAP_PORT", ""), ("CLAP_VERSION", ""), ("CLAP_DEVICE", ""));
        Assert.Equal(("/ckpt", 8789, "1", "cpu"), (s.CheckpointDirectory, s.Port, s.Version, s.Device));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("70000")]
    [InlineData("-5")]
    [InlineData("80a")]
    [InlineData("8 0")]
    public void ABadPortNamesTheVariableAndTheValue(string port)
    {
        var ex = Assert.Throws<SettingsException>(() => From(("CLAP_PORT", port)));
        Assert.Contains("CLAP_PORT", ex.Message);
        Assert.Contains($"'{port}'", ex.Message);
    }

    [Theory]
    [InlineData("gpu")]
    [InlineData("cuda:")]
    [InlineData("cuda:x")]
    [InlineData("cuda:-1")]
    [InlineData("cpu:0")]
    public void ABadDeviceNamesTheVariable(string device)
    {
        var ex = Assert.Throws<SettingsException>(() => From(("CLAP_DEVICE", device)));
        Assert.Contains("CLAP_DEVICE", ex.Message);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("cuda")]
    [InlineData("cuda:0")]
    [InlineData("CPU")]
    public void ValidDevicesAreAccepted(string device) => Assert.Equal(device.ToLowerInvariant(), From(("CLAP_DEVICE", device)).Device);
}
