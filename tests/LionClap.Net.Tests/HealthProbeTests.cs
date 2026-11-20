using LionClap.Service;

namespace LionClap.Tests;

public class HealthProbeTests
{
    [Fact]
    public void NvidiaSmiIsNeverResolvedThroughThePath()
    {
        // Either the machine has it in a standard place (an absolute path to an existing file) or the service reports no card.
        var found = HealthProbe.FindNvidiaSmi();
        Assert.True(found is null || (Path.IsPathRooted(found) && File.Exists(found)), found);
    }

    [Fact]
    public void TheCardLineOfNvidiaSmiIsParsed()
    {
        var gpu = HealthProbe.ParseSmi("NVIDIA GeForce RTX 3090, 23000, 24576\n");
        Assert.NotNull(gpu);
        Assert.Equal(("NVIDIA GeForce RTX 3090", 23000L, 24576L), (gpu!.Name, gpu.FreeMb, gpu.TotalMb));
    }
}
