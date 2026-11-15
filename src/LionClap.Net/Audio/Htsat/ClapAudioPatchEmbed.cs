using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Audio.Htsat;

/// <summary>Patch embedding (<c>ClapAudioPatchEmbed</c>, unfused): 4×4 convolution with stride 4, flatten to tokens, LayerNorm. Field names are the checkpoint keys.</summary>
public sealed class ClapAudioPatchEmbed : nn.Module<Tensor, Tensor>
{
    private readonly Conv2d proj;
    private readonly LayerNorm norm;
    private readonly long _size;
    private readonly long _stride;

    public ClapAudioPatchEmbed(ClapAudioConfig config) : base(nameof(ClapAudioPatchEmbed))
    {
        _size = config.SpecSize;
        _stride = config.PatchStride;
        proj = nn.Conv2d(config.PatchEmbedInputChannels, config.PatchEmbedsHiddenSize, config.PatchSize, stride: config.PatchStride,
            padding: (config.PatchSize - config.PatchStride) / 2);
        norm = nn.LayerNorm(config.PatchEmbedsHiddenSize, config.LayerNormEps);
        RegisterComponents();
    }

    /// <summary>Tokens per side of the patch grid (64 for a 256×256 image).</summary>
    public long GridSize => _size / _stride;

    /// <summary><c>[B, C, 256, 256]</c> image → <c>[B, grid², hidden]</c> tokens.</summary>
    public override Tensor forward(Tensor image)
    {
        if (image.shape[2] != _size || image.shape[3] != _size)
            throw new ArgumentException($"input image size ({image.shape[2]}×{image.shape[3]}) does not match the model ({_size}×{_size})", nameof(image));
        using var x = proj.forward(image);
        using var tokens = x.flatten(2).transpose(1, 2);
        return norm.forward(tokens);
    }
}
