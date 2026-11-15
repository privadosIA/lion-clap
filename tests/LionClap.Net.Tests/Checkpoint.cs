using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Tests;

/// <summary>
/// The converted checkpoint, taken from CLAP_CKPT: a directory with <c>model.safetensors</c> and <c>config.json</c> (written by the golden run,
/// never inside the repo). Tests that need it are marked <see cref="RequiresCheckpointAttribute"/> and are reported as skipped without it.
/// </summary>
public static class Checkpoint
{
    public static readonly CheckpointEnvironment Env = new("CLAP_CKPT");

    private static readonly Lazy<CheckpointIndex> Index_ = new(() => CheckpointIndex.Open(Path.Combine(Dir!, "model.safetensors")));

    public static string? Dir => Env.Dir;

    public static string ConfigFile => Path.Combine(Dir!, "config.json");

    public static CheckpointIndex Index => Index_.Value;

    /// <summary>The Hugging Face directory of the Python container (CLAP_HF_DIR), when it exists.</summary>
    public static string? HfDir => Environment.GetEnvironmentVariable("CLAP_HF_DIR") is { Length: > 0 } d && Directory.Exists(d) ? d : null;

    /// <summary>The <c>pytorch_model.bin</c> of the newest snapshot under <see cref="HfDir"/>, or null.</summary>
    public static string? RealBin => HfDir is null ? null
        : Directory.Exists(Path.Combine(HfDir, "hub", "models--laion--clap-htsat-unfused", "snapshots"))
            ? Directory.GetFiles(Path.Combine(HfDir, "hub", "models--laion--clap-htsat-unfused", "snapshots"), "pytorch_model.bin", SearchOption.AllDirectories).OrderBy(f => f, StringComparer.Ordinal).LastOrDefault()
            : null;

    /// <summary>
    /// Copies into <paramref name="module"/> the tensors stored under <paramref name="keyPrefix"/> (e.g. <c>audio_model.audio_encoder.patch_embed.</c>).
    /// Every parameter must exist in the checkpoint; buffers are loaded when they exist there. Used to test one module at a time.
    /// </summary>
    public static T Load<T>(T module, string keyPrefix) where T : nn.Module
    {
        using var _ = no_grad();
        foreach (var (name, p) in module.named_parameters())
        {
            using var w = Index.Read(keyPrefix + name, p.dtype);
            p.copy_(w);
        }
        foreach (var (name, b) in module.named_buffers())
        {
            if (!Index.Tensors.ContainsKey(keyPrefix + name)) continue;
            using var w = Index.Read(keyPrefix + name);
            b.copy_(w);
        }
        module.eval();
        return module;
    }
}

/// <summary>[Fact] that is skipped (with the reason) unless CLAP_CKPT points to an existing directory.</summary>
public sealed class RequiresCheckpointAttribute : RequiresCheckpointFactAttribute
{
    public RequiresCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>
/// Also needs CLAP_HF_DIR: the Hugging Face directory the Python container mounts at /ckpt (it holds <c>hub/models--laion--clap-htsat-unfused/snapshots/*/pytorch_model.bin</c>),
/// to compare the PyTorch file with the converted checkpoint.
/// </summary>
public sealed class RequiresBothCheckpointsAttribute : Xunit.FactAttribute
{
    public RequiresBothCheckpointsAttribute() => Skip = Checkpoint.Env.NoCheckpointReason ?? (Checkpoint.RealBin is null ? "CLAP_HF_DIR not set or it has no pytorch_model.bin" : null);
}

/// <summary>[Fact] that needs the Python container's Hugging Face directory (CLAP_HF_DIR), the layout the service reads in production.</summary>
public sealed class RequiresHfDirAttribute : Xunit.FactAttribute
{
    public RequiresHfDirAttribute() => Skip = Checkpoint.RealBin is null ? "CLAP_HF_DIR not set or it has no pytorch_model.bin" : null;
}

/// <summary>[Fact] that reruns the embedding parity on the GPU: it needs CLAP_CKPT, CLAP_DEVICE=cuda (or cuda:N) and a CUDA build of the service libraries.</summary>
public sealed class RequiresCudaCheckpointAttribute : Xunit.FactAttribute
{
    public RequiresCudaCheckpointAttribute()
    {
        var device = Environment.GetEnvironmentVariable("CLAP_DEVICE")?.ToLowerInvariant() ?? "cpu";
        Skip = Checkpoint.Env.NoCheckpointReason
            ?? (!device.StartsWith("cuda") ? "CLAP_DEVICE is not cuda: the GPU parity test is opt-in"
            : !TorchSharp.torch.cuda.is_available() ? "no CUDA device or CUDA build (build with -p:TorchBackend=cuda-linux)" : null);
    }
}
