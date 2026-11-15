using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Audio.Htsat;

/// <summary>Window attention with the relative position bias (<c>ClapAudioSelfAttention</c>). Field names are the checkpoint keys.</summary>
public sealed class ClapAudioSelfAttention : nn.Module
{
    private readonly Linear query;
    private readonly Linear key;
    private readonly Linear value;
    private readonly Parameter relative_position_bias_table;
    private readonly int _heads;
    private readonly int _headSize;
    private readonly int _window;

    public ClapAudioSelfAttention(ClapAudioConfig config, int dim, int heads, int window) : base(nameof(ClapAudioSelfAttention))
    {
        if (dim % heads != 0) throw new ArgumentException($"the hidden size ({dim}) is not a multiple of the number of attention heads ({heads})");
        (_heads, _headSize, _window) = (heads, dim / heads, window);
        query = nn.Linear(dim, dim, hasBias: config.QkvBias);
        key = nn.Linear(dim, dim, hasBias: config.QkvBias);
        value = nn.Linear(dim, dim, hasBias: config.QkvBias);
        relative_position_bias_table = nn.Parameter(zeros((2 * window - 1) * (2 * window - 1), heads));
        RegisterComponents();
        register_buffer("relative_position_index", SwinWindows.RelativePositionIndex(window));
    }

    /// <summary><paramref name="hidden"/> <c>[windows·B, w², C]</c>; <paramref name="mask"/> <c>[windows, w², w²]</c> or null.</summary>
    public Tensor forward(Tensor hidden, Tensor? mask)
    {
        var (batch, n, channels) = (hidden.shape[0], hidden.shape[1], hidden.shape[2]);
        using var q = Heads(query.forward(hidden), batch, n);
        using var k = Heads(key.forward(hidden), batch, n);
        using var v = Heads(value.forward(hidden), batch, n);

        using var kt = k.transpose(-1, -2);
        using var raw = matmul(q, kt);
        var scores = raw / Math.Sqrt(_headSize);
        using (scores)
        {
            using var table = relative_position_bias_table.index_select(0, get_buffer("relative_position_index").reshape(-1));
            using var bias = table.reshape(n, n, _heads).permute(2, 0, 1).contiguous().unsqueeze(0);
            var biased = scores + bias;
            using (biased)
            {
                Tensor probs;
                if (mask is null)
                {
                    probs = softmax(biased, -1);
                }
                else
                {
                    var windows = mask.shape[0];
                    using var grouped = biased.reshape(batch / windows, windows, _heads, n, n);
                    using var masked = grouped + mask.unsqueeze(1).unsqueeze(0);
                    using var flat = masked.reshape(-1, _heads, n, n);
                    probs = softmax(flat, -1);
                }
                using (probs)
                {
                    using var context = matmul(probs, v);
                    using var merged = context.permute(0, 2, 1, 3).contiguous();
                    return merged.reshape(batch, n, channels);
                }
            }
        }
    }

    private Tensor Heads(Tensor x, long batch, long n)
    {
        using (x) { using var split = x.reshape(batch, n, _heads, _headSize); return split.permute(0, 2, 1, 3); }
    }
}

internal sealed class ClapAudioSelfOutput : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;
    public ClapAudioSelfOutput(int dim) : base(nameof(ClapAudioSelfOutput)) { dense = nn.Linear(dim, dim); RegisterComponents(); }
    public override Tensor forward(Tensor x) => dense.forward(x);
}

/// <summary><c>ClapAudioAttention</c>: <c>self</c> and the output projection <c>output.dense</c>.</summary>
public sealed class ClapAudioAttention : nn.Module
{
    private readonly ClapAudioSelfAttention self;
    private readonly ClapAudioSelfOutput output;

    public ClapAudioAttention(ClapAudioConfig config, int dim, int heads, int window) : base(nameof(ClapAudioAttention))
    {
        self = new ClapAudioSelfAttention(config, dim, heads, window);
        output = new ClapAudioSelfOutput(dim);
        RegisterComponents();
    }

    public Tensor forward(Tensor hidden, Tensor? mask)
    {
        using var attended = self.forward(hidden, mask);
        return output.forward(attended);
    }
}

internal sealed class ClapAudioIntermediate : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;
    public ClapAudioIntermediate(int dim, int hidden) : base(nameof(ClapAudioIntermediate)) { dense = nn.Linear(dim, hidden); RegisterComponents(); }

    public override Tensor forward(Tensor x)
    {
        using var projected = dense.forward(x);
        return nn.functional.gelu(projected); // exact (erf) GELU, as ACT2FN["gelu"]
    }
}

internal sealed class ClapAudioOutput : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;
    public ClapAudioOutput(int hidden, int dim) : base(nameof(ClapAudioOutput)) { dense = nn.Linear(hidden, dim); RegisterComponents(); }
    public override Tensor forward(Tensor x) => dense.forward(x);
}

/// <summary>
/// One Swin block (<c>ClapAudioLayer</c>) over a fixed token grid: pre-norm window attention (shifted when <c>shift &gt; 0</c>) and pre-norm MLP,
/// both residual. Input and output are <c>[B, H·W, C]</c>. Dropout and drop-path do not exist at inference.
/// </summary>
public sealed class ClapAudioLayer : nn.Module<Tensor, Tensor>
{
    private readonly LayerNorm layernorm_before;
    private readonly ClapAudioAttention attention;
    private readonly LayerNorm layernorm_after;
    private readonly ClapAudioIntermediate intermediate;
    private readonly ClapAudioOutput output;
    private readonly int _height;
    private readonly int _width;
    private readonly int _window;
    private readonly int _shift;
    private readonly MaskCache _mask = new();

    /// <summary>Holds the shift mask outside the registered fields (it is derived from shapes, not a weight).</summary>
    private sealed class MaskCache { public Tensor? Value; }

    public ClapAudioLayer(ClapAudioConfig config, int dim, (int height, int width) resolution, int heads, int shift) : base(nameof(ClapAudioLayer))
    {
        (_height, _width) = resolution;
        _window = config.WindowSize;
        _shift = shift;
        // set_shift_and_window_size: a grid that fits in one window is not shifted
        if (Math.Min(_height, _width) <= _window)
        {
            if (Math.Min(_height, _width) < _window)
                throw new NotSupportedException($"a {_height}×{_width} grid is smaller than the {_window}×{_window} window: not needed by HTSAT-tiny and not ported");
            _shift = 0;
        }
        if (_height % _window != 0 || _width % _window != 0)
            throw new NotSupportedException($"a {_height}×{_width} grid needs padding to {_window}×{_window} windows: not needed by HTSAT-tiny and not ported");

        layernorm_before = nn.LayerNorm(dim, config.LayerNormEps);
        attention = new ClapAudioAttention(config, dim, heads, _window);
        layernorm_after = nn.LayerNorm(dim, config.LayerNormEps);
        intermediate = new ClapAudioIntermediate(dim, (int)(config.MlpRatio * dim));
        output = new ClapAudioOutput((int)(config.MlpRatio * dim), dim);
        RegisterComponents();
    }

    public int Shift => _shift;

    public override Tensor forward(Tensor hidden)
    {
        var (batch, channels) = (hidden.shape[0], hidden.shape[2]);
        using var normed = layernorm_before.forward(hidden);
        using var grid = normed.reshape(batch, _height, _width, channels);
        using var shifted = _shift > 0 ? grid.roll([-_shift, -_shift], [1, 2]) : grid.alias();
        using var windows = SwinWindows.Partition(shifted, _window);
        using var tokens = windows.reshape(-1, _window * _window, channels);

        using var attended = attention.forward(tokens, _shift > 0 ? Mask(hidden) : null);
        using var attendedWindows = attended.reshape(-1, _window, _window, channels);
        using var merged = SwinWindows.Reverse(attendedWindows, _window, _height, _width);
        using var restored = _shift > 0 ? merged.roll([_shift, _shift], [1, 2]) : merged.alias();
        using var flat = restored.reshape(batch, _height * _width, channels);

        using var afterAttention = hidden + flat;
        using var normedAfter = layernorm_after.forward(afterAttention);
        using var hiddenMlp = intermediate.forward(normedAfter);
        using var mlp = output.forward(hiddenMlp);
        return afterAttention + mlp;
    }

    private Tensor Mask(Tensor like)
    {
        if (_mask.Value is { } cached && cached.device_type == like.device_type && cached.dtype == like.dtype) return cached;
        _mask.Value?.Dispose();
        _mask.Value = SwinWindows.ShiftMask(_height, _width, _window, _shift).to(like.dtype, like.device);
        return _mask.Value;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _mask.Value?.Dispose();
        base.Dispose(disposing);
    }
}
