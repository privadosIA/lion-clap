using LionClap.Audio.Htsat;
using LionClap.Text;

namespace LionClap.Tests;

/// <summary>Temporary directories of the tests. They live under one folder per test process, which is deleted when the process ends, so repeated runs do not fill /tmp.</summary>
public static class TestTemp
{
    private static readonly Lazy<string> Root = new(() =>
    {
        var root = Path.Combine(Path.GetTempPath(), "lionclap-tests", $"{Environment.ProcessId}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        AppDomain.CurrentDomain.ProcessExit += (_, _) => { try { Directory.Delete(root, true); } catch (IOException) { } catch (UnauthorizedAccessException) { } };
        return root;
    });

    public static string Dir(string? name = null)
    {
        var path = Path.Combine(Root.Value, name ?? Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}

/// <summary>Configurations with the structure of the real branches (12 blocks / 12 layers, same parameter names) and tiny sizes, so crafted checkpoints stay small.</summary>
public static class SmallConfigs
{
    public static readonly ClapTextConfig Text = new()
    {
        VocabSize = 100,
        HiddenSize = 32,
        NumHiddenLayers = 12,
        NumAttentionHeads = 4,
        IntermediateSize = 64,
        MaxPositionEmbeddings = 34,
        ProjectionDim = 16,
    };

    public static readonly ClapAudioConfig Audio = new()
    {
        PatchEmbedsHiddenSize = 8,
        NumAttentionHeads = [2, 2, 2, 2],
        HiddenSize = 64,
        ProjectionDim = 16,
    };
}
