using LionClap.Weights;
using LionClap.Audio.Htsat;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Text;

/// <summary>The pooled encoder output <c>[B, 768]</c> and the L2-normalised embedding <c>[B, 512]</c>. The caller disposes both.</summary>
public readonly record struct ClapTextFeatures(Tensor Pooled, Tensor Embeddings) : IDisposable
{
    public void Dispose()
    {
        Pooled.Dispose();
        Embeddings.Dispose();
    }
}

/// <summary>
/// The text branch of <c>ClapModel</c> (unfused checkpoint, inference only): RoBERTa encoder, pooler, projection MLP and L2 normalisation, as
/// <c>get_text_features</c>. Field names are the checkpoint keys (<c>text_model.*</c>, <c>text_projection.*</c>).
/// </summary>
public sealed class ClapTextBranch : nn.Module
{
    private readonly ClapTextModel text_model;
    private readonly ClapProjectionLayer text_projection;

    public ClapTextBranch(ClapTextConfig? config = null) : base(nameof(ClapTextBranch))
    {
        Config = config ?? new ClapTextConfig();
        text_model = new ClapTextModel(Config);
        text_projection = new ClapProjectionLayer(Config.HiddenSize, Config.ProjectionDim);
        RegisterComponents();
        eval();
    }

    public ClapTextConfig Config { get; }

    /// <summary>
    /// Builds the branch on <paramref name="device"/> (CPU by default) from <paramref name="checkpoint"/>: a directory in any layout that
    /// <see cref="ClapCheckpointLocator"/> understands (flat with <c>model.safetensors</c> or <c>pytorch_model.bin</c>, or the Hugging Face cache of the Python
    /// container), or the weights file itself. The configuration is read from the <c>config.json</c> beside the weights when it exists, otherwise the defaults are used.
    /// </summary>
    public static ClapTextBranch Load(string checkpoint, Device? device = null)
    {
        var files = ClapCheckpointLocator.Locate(checkpoint);
        if (files.Weights is null)
            throw new FileNotFoundException($"No CLAP weights found in {checkpoint} (missing: {string.Join(", ", files.Missing)})", checkpoint);
        var branch = new ClapTextBranch(files.Config is { } config ? ClapTextConfig.FromFile(config) : null);
        if (files.Format == WeightsFormat.PyTorch) branch.LoadWeights(TorchCheckpoint.Open(files.Weights), device ?? CPU);
        else branch.LoadWeights(CheckpointIndex.Open(files.Weights), device ?? CPU);
        return branch;
    }

    /// <summary>
    /// Strict load of every <c>text_*</c> tensor (parameters and buffers). Keys and shapes are checked with the headers alone: a missing key, an extra
    /// text key or another shape throws one <see cref="InvalidDataException"/> that lists them all, before any weight is touched. Audio and
    /// <c>logit_scale_*</c> keys are outside the prefix and are not read.
    /// </summary>
    public void LoadWeights(CheckpointIndex checkpoint, Device device)
    {
        StrictLoader.Load(this, checkpoint, device, new StrictLoadOptions("The checkpoint does not match the text branch:")
        {
            Prefix = "text_",
            StripPrefix = "",         // the model keys keep their "text_model." / "text_projection." names
            IncludeBuffers = true,    // position_ids and token_type_ids (int64) round-trip through float32 exactly
            ReadDtype = ScalarType.Float32,
            UnexpectedLabel = "text checkpoint keys",
        });
        eval();
    }

    /// <summary>
    /// The same strict load from a PyTorch <c>pytorch_model.bin</c> (<see cref="TorchCheckpointLoader"/>): identical checks, error format and resulting weights
    /// as the safetensors path.
    /// </summary>
    public void LoadWeights(TorchCheckpoint checkpoint, Device device)
    {
        TorchCheckpointLoader.Load(this, checkpoint, device, "text_", "The checkpoint does not match the text branch:", "text checkpoint keys");
        eval();
    }

    /// <summary>Embeds token ids and attention mask <c>[B, L]</c> (int64, L ≤ 512): pooled <c>[B, 768]</c> and normalised embeddings <c>[B, 512]</c>. Inputs on another device are moved to the module's.</summary>
    public ClapTextFeatures Forward(Tensor inputIds, Tensor attentionMask)
    {
        using var _ = no_grad();
        var device = parameters().First().device;
        var ids = OnDevice(inputIds, device);
        var mask = OnDevice(attentionMask, device);
        try
        {
            var pooled = text_model.forward(ids, mask);
            using var projected = text_projection.forward(pooled);
            var embeddings = nn.functional.normalize(projected, p: 2.0, dim: -1, eps: 1e-12);
            return new ClapTextFeatures(pooled, embeddings);
        }
        finally
        {
            if (!ReferenceEquals(ids, inputIds)) ids.Dispose();
            if (!ReferenceEquals(mask, attentionMask)) mask.Dispose();
        }
    }

    /// <summary>Convenience for a tokenised batch: builds the int64 tensors (on <paramref name="device"/>, or the module's when null) and embeds them.</summary>
    public ClapTextFeatures Forward(TokenBatch batch, Device? device = null)
    {
        using var ids = tensor(batch.Ids, [batch.Rows, batch.Length], device: device ?? parameters().First().device);
        using var mask = tensor(batch.Mask, [batch.Rows, batch.Length], device: device ?? parameters().First().device);
        return Forward(ids, mask);
    }

    private static Tensor OnDevice(Tensor t, Device device) =>
        t.device_type == device.type && t.device_index == device.index ? t : t.to(device);
}
