namespace LionClap.Weights;

public enum WeightsFormat { None, SafeTensors, PyTorch }

/// <summary>Where the pieces of a CLAP checkpoint are. <see cref="Missing"/> is empty exactly when the weights and the tokenizer were found.</summary>
public sealed record ClapCheckpointFiles(string? Weights, WeightsFormat Format, string? Config, string? Tokenizer, IReadOnlyList<string> Missing)
{
    public bool Ready => Missing.Count == 0;
}

/// <summary>
/// Finds the weights, <c>config.json</c> and <c>tokenizer.json</c> of <c>laion/clap-htsat-unfused</c> in the layouts in use: a flat directory with
/// <c>model.safetensors</c> or with <c>pytorch_model.bin</c> (safetensors wins), the Hugging Face cache <c>hub/models--laion--clap-htsat-unfused/snapshots/&lt;revision&gt;/</c>
/// that the Python container mounts at <c>/ckpt</c> (the last revision, in name order, that holds weights), or a path to the weights file itself.
/// </summary>
public static class ClapCheckpointLocator
{
    public const string HubId = "models--laion--clap-htsat-unfused";
    public const string SafeTensorsName = "model.safetensors", PyTorchName = "pytorch_model.bin", ConfigName = "config.json", TokenizerName = "tokenizer.json";

    public static ClapCheckpointFiles Locate(string path)
    {
        if (File.Exists(path))
        {
            var format = path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) ? WeightsFormat.SafeTensors : WeightsFormat.PyTorch;
            return Describe(Path.GetDirectoryName(Path.GetFullPath(path))!, path, format);
        }
        if (!Directory.Exists(path)) return new ClapCheckpointFiles(null, WeightsFormat.None, null, null, [HubId]);

        if (WeightsIn(path) is { } flat) return Describe(path, flat.File, flat.Format);

        var snapshots = Path.Combine(path, "hub", HubId, "snapshots");
        if (Directory.Exists(snapshots))
        {
            foreach (var revision in Directory.GetDirectories(snapshots).OrderBy(d => d, StringComparer.Ordinal).Reverse())
            {
                if (WeightsIn(revision) is { } found) return Describe(revision, found.File, found.Format);
            }
        }
        return new ClapCheckpointFiles(null, WeightsFormat.None, null, null, [HubId]);
    }

    private static (string File, WeightsFormat Format)? WeightsIn(string directory)
    {
        var safe = Path.Combine(directory, SafeTensorsName);
        if (File.Exists(safe)) return (safe, WeightsFormat.SafeTensors);
        var bin = Path.Combine(directory, PyTorchName);
        return File.Exists(bin) ? (bin, WeightsFormat.PyTorch) : null;
    }

    private static ClapCheckpointFiles Describe(string directory, string weights, WeightsFormat format)
    {
        var config = Path.Combine(directory, ConfigName);
        var tokenizer = Path.Combine(directory, TokenizerName);
        var missing = new List<string>();
        if (!File.Exists(tokenizer)) missing.Add(TokenizerName);
        return new ClapCheckpointFiles(weights, format, File.Exists(config) ? config : null, File.Exists(tokenizer) ? tokenizer : null, missing);
    }
}
