using LionClap.Weights;
using MusicGen.Core.Weights;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Audio.Htsat;

/// <summary>Projection MLP (<c>ClapProjectionLayer</c>): <c>linear1</c> → ReLU → <c>linear2</c>.</summary>
public sealed class ClapProjectionLayer : nn.Module<Tensor, Tensor>
{
    private readonly Linear linear1;
    private readonly Linear linear2;

    public ClapProjectionLayer(int hiddenSize, int projectionDim) : base(nameof(ClapProjectionLayer))
    {
        linear1 = nn.Linear(hiddenSize, projectionDim);
        linear2 = nn.Linear(projectionDim, projectionDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var hidden = linear1.forward(x);
        using var activated = nn.functional.relu(hidden);
        return linear2.forward(activated);
    }
}

/// <summary><c>ClapAudioModel</c>: only wraps the encoder (key prefix <c>audio_model.audio_encoder.</c>).</summary>
public sealed class ClapAudioModel : nn.Module<Tensor, Tensor>
{
    private readonly ClapAudioEncoder audio_encoder;

    public ClapAudioModel(ClapAudioConfig config) : base(nameof(ClapAudioModel))
    {
        audio_encoder = new ClapAudioEncoder(config);
        RegisterComponents();
    }

    public ClapAudioEncoder Encoder => audio_encoder;

    public override Tensor forward(Tensor inputFeatures) => audio_encoder.forward(inputFeatures);
}

/// <summary>The pooled encoder output <c>[B, 768]</c> and the L2-normalised embedding <c>[B, 512]</c>. The caller disposes both.</summary>
public readonly record struct ClapAudioFeatures(Tensor Pooled, Tensor Embeddings) : IDisposable
{
    public void Dispose()
    {
        Pooled.Dispose();
        Embeddings.Dispose();
    }
}

/// <summary>
/// The audio branch of <c>ClapModel</c> (unfused, inference only): HTSAT encoder, projection MLP and L2 normalisation, as
/// <c>get_audio_features</c>. Field names are the checkpoint keys (<c>audio_model.*</c>, <c>audio_projection.*</c>).
/// </summary>
public sealed class ClapAudioBranch : nn.Module
{
    private readonly ClapAudioModel audio_model;
    private readonly ClapProjectionLayer audio_projection;

    public ClapAudioBranch(ClapAudioConfig? config = null) : base(nameof(ClapAudioBranch))
    {
        Config = config ?? new ClapAudioConfig();
        audio_model = new ClapAudioModel(Config);
        audio_projection = new ClapProjectionLayer(Config.HiddenSize, Config.ProjectionDim);
        RegisterComponents();
        eval();
    }

    public ClapAudioConfig Config { get; }

    /// <summary>
    /// Builds the branch on <paramref name="device"/> (CPU by default) from <paramref name="checkpoint"/>: a directory in any layout that
    /// <see cref="ClapCheckpointLocator"/> understands (flat with <c>model.safetensors</c> or <c>pytorch_model.bin</c>, or the Hugging Face cache of the Python
    /// container), or the weights file itself. The configuration is read from the <c>config.json</c> beside the weights when it exists, otherwise the defaults are used.
    /// </summary>
    public static ClapAudioBranch Load(string checkpoint, Device? device = null)
    {
        var files = ClapCheckpointLocator.Locate(checkpoint);
        if (files.Weights is null)
            throw new FileNotFoundException($"No CLAP weights found in {checkpoint} (missing: {string.Join(", ", files.Missing)})", checkpoint);
        var branch = new ClapAudioBranch(files.Config is { } config ? ClapAudioConfig.FromFile(config) : null);
        if (files.Format == WeightsFormat.PyTorch) branch.LoadWeights(TorchCheckpoint.Open(files.Weights), device ?? CPU);
        else branch.LoadWeights(CheckpointIndex.Open(files.Weights), device ?? CPU);
        return branch;
    }

    /// <summary>
    /// Strict load of every <c>audio_*</c> tensor (parameters and buffers). Keys and shapes are checked with the headers alone: a missing key, an extra
    /// audio key or another shape throws one <see cref="InvalidDataException"/> that lists them all, before any weight is touched. Text and
    /// <c>logit_scale_*</c> keys are outside the prefix and are not read.
    /// </summary>
    public void LoadWeights(CheckpointIndex checkpoint, Device device)
    {
        StrictLoader.Load(this, checkpoint, device, new StrictLoadOptions("The checkpoint does not match the audio branch:")
        {
            Prefix = "audio_",
            StripPrefix = "",         // the model keys keep their "audio_model." / "audio_projection." names
            IncludeBuffers = true,    // batch-norm statistics, and the (integer) relative position indices
            ReadDtype = ScalarType.Float32,
            UnexpectedLabel = "audio checkpoint keys",
        });
        eval();
    }

    /// <summary>
    /// The same strict load from a PyTorch <c>pytorch_model.bin</c> (<see cref="TorchCheckpointLoader"/>): identical checks, error format and resulting weights
    /// as the safetensors path.
    /// </summary>
    public void LoadWeights(TorchCheckpoint checkpoint, Device device)
    {
        TorchCheckpointLoader.Load(this, checkpoint, device, "audio_", "The checkpoint does not match the audio branch:", "audio checkpoint keys");
        eval();
    }

    /// <summary>Embeds log-mel features <c>[B, 1, T, 64]</c> (T ≤ 1024): pooled <c>[B, 768]</c> and normalised embeddings <c>[B, 512]</c>.</summary>
    public ClapAudioFeatures Forward(Tensor inputFeatures)
    {
        using var _ = no_grad();
        var device = parameters().First().device;
        var input = OnDevice(inputFeatures, device);
        try
        {
            var pooled = audio_model.forward(input);
            using var projected = audio_projection.forward(pooled);
            var embeddings = nn.functional.normalize(projected, p: 2.0, dim: -1, eps: 1e-12);
            return new ClapAudioFeatures(pooled, embeddings);
        }
        finally
        {
            if (!ReferenceEquals(input, inputFeatures)) input.Dispose();
        }
    }

    private static Tensor OnDevice(Tensor t, Device device) =>
        t.device_type == device.type && t.device_index == device.index ? t : t.to(device);
}
