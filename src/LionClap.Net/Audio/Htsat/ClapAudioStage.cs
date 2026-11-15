using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Audio.Htsat;

/// <summary>Patch merging between stages (<c>ClapAudioPatchMerging</c>): the four 2×2 sub-grids are concatenated, normalised and reduced 4C → 2C.</summary>
public sealed class ClapAudioPatchMerging : nn.Module<Tensor, Tensor>
{
    private readonly Linear reduction;
    private readonly LayerNorm norm;
    private readonly long _height;
    private readonly long _width;

    public ClapAudioPatchMerging(int dim, (int height, int width) resolution) : base(nameof(ClapAudioPatchMerging))
    {
        if (resolution.height % 2 != 0 || resolution.width % 2 != 0)
            throw new NotSupportedException($"a {resolution.height}×{resolution.width} grid has an odd side: padding is not needed by HTSAT-tiny and not ported");
        (_height, _width) = resolution;
        reduction = nn.Linear(4 * dim, 2 * dim, hasBias: false);
        norm = nn.LayerNorm(4 * dim); // default eps 1e-5, as in HF
        RegisterComponents();
    }

    /// <summary><c>[B, H·W, C]</c> → <c>[B, H·W/4, 2C]</c>.</summary>
    public override Tensor forward(Tensor x)
    {
        var (batch, channels) = (x.shape[0], x.shape[2]);
        using var grid = x.reshape(batch, _height, _width, channels);
        using var p0 = Sub(grid, 0, 0);
        using var p1 = Sub(grid, 1, 0);
        using var p2 = Sub(grid, 0, 1);
        using var p3 = Sub(grid, 1, 1);
        using var joined = cat([p0, p1, p2, p3], -1);
        using var flat = joined.reshape(batch, -1, 4 * channels);
        using var normed = norm.forward(flat);
        return reduction.forward(normed);
    }

    // grid[:, row::2, col::2, :]
    private static Tensor Sub(Tensor grid, int row, int col) =>
        grid.index(TensorIndex.Colon, TensorIndex.Slice(row, null, 2), TensorIndex.Slice(col, null, 2), TensorIndex.Colon);
}

/// <summary>One Swin stage (<c>ClapAudioStage</c>): <c>blocks</c> (even ones plain, odd ones shifted by window/2) and, except in the last stage, <c>downsample</c>.</summary>
public sealed class ClapAudioStage : nn.Module<Tensor, Tensor>
{
    private readonly ModuleList<ClapAudioLayer> blocks;
    private readonly ClapAudioPatchMerging? downsample;

    public ClapAudioStage(ClapAudioConfig config, int dim, (int height, int width) resolution, int depth, int heads, bool merge) : base(nameof(ClapAudioStage))
    {
        blocks = nn.ModuleList(Enumerable.Range(0, depth)
            .Select(i => new ClapAudioLayer(config, dim, resolution, heads, shift: i % 2 == 0 ? 0 : config.WindowSize / 2)).ToArray());
        if (merge) downsample = new ClapAudioPatchMerging(dim, resolution);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden) => Forward(hidden, null, "");

    /// <summary>
    /// Runs the stage. <paramref name="tap"/>, when given, receives <c>{prefix}_block{i}</c> after each block; the tensors are only valid during the call.
    /// Returns the output after patch merging (when the stage has it).
    /// </summary>
    public Tensor Forward(Tensor hidden, Action<string, Tensor>? tap, string prefix)
    {
        var current = hidden;
        for (var i = 0; i < blocks.Count; i++)
        {
            var next = blocks[i].forward(current);
            if (!ReferenceEquals(current, hidden)) current.Dispose();
            current = next;
            tap?.Invoke($"{prefix}_block{i}", current);
        }
        if (downsample is null) return current;
        using var _ = ReferenceEquals(current, hidden) ? null : current;
        return downsample.forward(current);
    }
}
