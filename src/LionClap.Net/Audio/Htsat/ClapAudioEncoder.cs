using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Audio.Htsat;

/// <summary>
/// The HTSAT (Swin-tiny) audio encoder of CLAP, unfused (<c>ClapAudioEncoder</c>). Field names are the checkpoint keys under
/// <c>audio_model.audio_encoder.</c>. <see cref="forward"/> maps log-mel <c>[B, 1, T, 64]</c> (T ≤ 1024) to the pooled output <c>[B, 768]</c>.
/// </summary>
public sealed class ClapAudioEncoder : nn.Module<Tensor, Tensor>
{
    private readonly BatchNorm2d batch_norm;
    private readonly ClapAudioPatchEmbed patch_embed;
    private readonly ModuleList<ClapAudioStage> layers;
    private readonly LayerNorm norm;
    private readonly ClapAudioConfig _config;

    public ClapAudioEncoder(ClapAudioConfig config) : base(nameof(ClapAudioEncoder))
    {
        if (config.Depths.Length != config.NumAttentionHeads.Length)
            throw new ArgumentException("depths and num_attention_heads must have one entry per stage", nameof(config));
        _config = config;
        batch_norm = nn.BatchNorm2d(config.NumMelBins);
        patch_embed = new ClapAudioPatchEmbed(config);

        var stages = config.Depths.Length;
        var grid = (int)patch_embed.GridSize;
        layers = nn.ModuleList(Enumerable.Range(0, stages).Select(i => new ClapAudioStage(
            config, config.PatchEmbedsHiddenSize << i, (grid >> i, grid >> i), config.Depths[i], config.NumAttentionHeads[i], merge: i < stages - 1)).ToArray());
        norm = nn.LayerNorm(config.PatchEmbedsHiddenSize << (stages - 1), config.LayerNormEps);
        RegisterComponents();
    }

    /// <summary>Pooled output <c>[B, 768]</c>.</summary>
    public override Tensor forward(Tensor inputFeatures) => Encode(inputFeatures, null);

    /// <summary>
    /// Runs the encoder. <paramref name="tap"/>, when given, receives the intermediate activations under the names of the golden fixture
    /// (<c>image</c>, <c>patch_embed</c>, <c>stage{i}_block{j}</c>, <c>stage{i}</c>, <c>norm</c>); the tensors are only valid during the call.
    /// </summary>
    public Tensor Encode(Tensor inputFeatures, Action<string, Tensor>? tap)
    {
        using var image = MelToImage(inputFeatures);
        tap?.Invoke("image", image);
        var hidden = patch_embed.forward(image);
        tap?.Invoke("patch_embed", hidden);

        for (var i = 0; i < layers.Count; i++)
        {
            var next = layers[i].Forward(hidden, tap, $"stage{i}");
            hidden.Dispose();
            hidden = next;
            tap?.Invoke($"stage{i}", hidden);
        }
        using (hidden)
        {
            using var normalized = norm.forward(hidden);
            tap?.Invoke("norm", normalized);
            // the reference re-arranges the final map and averages it with an adaptive pool: a mean over the tokens, whatever their order
            return normalized.mean([1L]);
        }
    }

    /// <summary>
    /// Log-mel <c>[B, 1, T, 64]</c> (T ≤ 1024) → batch norm over the mel bands (running statistics) → the square image <c>[B, 1, 256, 256]</c>
    /// that the patch embedding consumes (<c>reshape_mel2img</c>).
    /// </summary>
    public Tensor MelToImage(Tensor inputFeatures)
    {
        Validate(inputFeatures);
        using var swapped = inputFeatures.transpose(1, 3);         // [B, 64, T, 1]: the batch norm runs over the mel axis
        using var normalized = batch_norm.forward(swapped);
        using var back = normalized.transpose(1, 3);               // [B, 1, T, 64]
        return FoldToImage(back, _config);
    }

    /// <summary>Input stage: <see cref="MelToImage"/> and the patch embedding → tokens <c>[B, 4096, 96]</c>.</summary>
    public Tensor Embed(Tensor inputFeatures)
    {
        using var image = MelToImage(inputFeatures);
        return patch_embed.forward(image);
    }

    /// <summary>The patch embedding alone, for an image already built by <see cref="MelToImage"/>.</summary>
    public Tensor EmbedImage(Tensor image) => patch_embed.forward(image);

    /// <summary>
    /// A time axis shorter than 1024 is stretched to 1024 (bicubic, <c>align_corners</c>), then the axis is cut in <c>freq_ratio</c> slices
    /// that are stacked along the frequency axis: <c>image[b, 0, k·64 + f, t] = x[b, 0, k·256 + t, f]</c>.
    /// </summary>
    internal static Tensor FoldToImage(Tensor x, ClapAudioConfig config)
    {
        var stretched = x.shape[2] < config.MaxFrames
            ? nn.functional.interpolate(x, size: [config.MaxFrames, x.shape[3]], mode: InterpolationMode.Bicubic, align_corners: true)
            : x;
        try
        {
            var (b, c, frames, mels) = (stretched.shape[0], stretched.shape[1], stretched.shape[2], stretched.shape[3]);
            var ratio = config.FreqRatio;
            using var sliced = stretched.reshape(b, c * ratio, frames / ratio, mels);
            using var swapped = sliced.permute(0, 1, 3, 2).contiguous();
            return swapped.reshape(b, c, mels * ratio, frames / ratio);
        }
        finally
        {
            if (!ReferenceEquals(stretched, x)) stretched.Dispose();
        }
    }

    private void Validate(Tensor x)
    {
        if (x.dim() != 4)
            throw new ArgumentException($"input_features must have 4 dimensions [batch, 1, time, {_config.NumMelBins}], got {x.dim()}", nameof(x));
        if (x.shape[1] != 1)
            throw new ArgumentException($"input_features must have 1 channel, got {x.shape[1]}", nameof(x));
        if (x.shape[3] != _config.NumMelBins)
            throw new ArgumentException($"input_features must have {_config.NumMelBins} mel bands, got {x.shape[3]}", nameof(x));
        if (x.shape[2] < 1 || x.shape[2] > _config.MaxFrames)
            throw new ArgumentException($"input_features must have 1 to {_config.MaxFrames} time frames, got {x.shape[2]}", nameof(x));
        if (x.shape[0] < 1)
            throw new ArgumentException("input_features must have at least one item", nameof(x));
    }
}
