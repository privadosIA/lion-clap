using MusicGen.Core.Testing;
using TorchSharp;

namespace LionClap.Tests;

/// <summary>Access to tests/fixtures (golden run of the reference implementation, versioned in the repo), with checksum verification from MANIFEST.md.</summary>
public static class Fixtures
{
    private static readonly FixtureDirectory Directory_ = FixtureDirectory.Find();

    public static string Dir => Directory_.Dir;

    public static string PathOf(string file) => Directory_.PathOf(file);

    public static Dictionary<string, torch.Tensor> Load(string file) => Directory_.Load(file);
}
