using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Text;

/// <summary>The additive attention mask of the reference: 0 where a key is attended, the lowest float32 where it is padding. Shape <c>[B, 1, 1, L]</c>.</summary>
public static class TextMasks
{
    public static Tensor Extend(Tensor attentionMask)
    {
        using var asFloat = attentionMask.to(ScalarType.Float32);
        using var inverted = 1.0f - asFloat;
        using var keyed = inverted.unsqueeze(1).unsqueeze(1);
        return keyed * float.MinValue;
    }
}

/// <summary>Multi-head self-attention (<c>ClapTextSelfAttention</c>): <c>q·kᵀ / sqrt(head) + mask</c>, softmax, <c>·v</c>. Field names are the checkpoint keys.</summary>
public sealed class ClapTextSelfAttention : nn.Module
{
    private readonly Linear query;
    private readonly Linear key;
    private readonly Linear value;
    private readonly int _heads;
    private readonly int _headSize;

    public ClapTextSelfAttention(ClapTextConfig config) : base(nameof(ClapTextSelfAttention))
    {
        if (config.HiddenSize % config.NumAttentionHeads != 0)
            throw new ArgumentException($"the hidden size ({config.HiddenSize}) is not a multiple of the number of attention heads ({config.NumAttentionHeads})");
        (_heads, _headSize) = (config.NumAttentionHeads, config.HiddenSize / config.NumAttentionHeads);
        query = nn.Linear(config.HiddenSize, config.HiddenSize);
        key = nn.Linear(config.HiddenSize, config.HiddenSize);
        value = nn.Linear(config.HiddenSize, config.HiddenSize);
        RegisterComponents();
    }

    /// <summary><paramref name="hidden"/> <c>[B, L, C]</c>, <paramref name="extendedMask"/> <c>[B, 1, 1, L]</c>.</summary>
    public Tensor forward(Tensor hidden, Tensor extendedMask)
    {
        var (batch, length, channels) = (hidden.shape[0], hidden.shape[1], hidden.shape[2]);
        using var q = Heads(query.forward(hidden), batch, length);
        using var k = Heads(key.forward(hidden), batch, length);
        using var v = Heads(value.forward(hidden), batch, length);

        using var kt = k.transpose(-1, -2);
        using var raw = matmul(q, kt);
        using var scaled = raw / Math.Sqrt(_headSize);
        using var masked = scaled + extendedMask;
        using var probs = softmax(masked, -1);
        using var context = matmul(probs, v);
        using var merged = context.permute(0, 2, 1, 3).contiguous();
        return merged.reshape(batch, length, channels);
    }

    private Tensor Heads(Tensor x, long batch, long length)
    {
        using (x) { using var split = x.reshape(batch, length, _heads, _headSize); return split.permute(0, 2, 1, 3); }
    }
}

/// <summary>Post-norm residual projection (<c>ClapTextSelfOutput</c>): <c>LayerNorm(dense(x) + residual)</c>.</summary>
public sealed class ClapTextSelfOutput : nn.Module
{
    private readonly Linear dense;
    private readonly LayerNorm LayerNorm;

    public ClapTextSelfOutput(ClapTextConfig config) : base(nameof(ClapTextSelfOutput))
    {
        dense = nn.Linear(config.HiddenSize, config.HiddenSize);
        LayerNorm = nn.LayerNorm(config.HiddenSize, config.LayerNormEps);
        RegisterComponents();
    }

    public Tensor forward(Tensor hidden, Tensor residual)
    {
        using var projected = dense.forward(hidden);
        using var sum = projected + residual;
        return LayerNorm.forward(sum);
    }
}

/// <summary><c>ClapTextAttention</c>: <c>self</c> and <c>output</c>.</summary>
public sealed class ClapTextAttention : nn.Module
{
    private readonly ClapTextSelfAttention self;
    private readonly ClapTextSelfOutput output;

    public ClapTextAttention(ClapTextConfig config) : base(nameof(ClapTextAttention))
    {
        self = new ClapTextSelfAttention(config);
        output = new ClapTextSelfOutput(config);
        RegisterComponents();
    }

    public Tensor forward(Tensor hidden, Tensor extendedMask)
    {
        using var attended = self.forward(hidden, extendedMask);
        return output.forward(attended, hidden);
    }
}

/// <summary><c>ClapTextIntermediate</c>: <c>Linear → GELU</c> (exact, erf).</summary>
public sealed class ClapTextIntermediate : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;

    public ClapTextIntermediate(ClapTextConfig config) : base(nameof(ClapTextIntermediate))
    {
        dense = nn.Linear(config.HiddenSize, config.IntermediateSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var projected = dense.forward(x);
        return nn.functional.gelu(projected);
    }
}

/// <summary><c>ClapTextOutput</c>: <c>LayerNorm(dense(x) + residual)</c>.</summary>
public sealed class ClapTextOutput : nn.Module
{
    private readonly Linear dense;
    private readonly LayerNorm LayerNorm;

    public ClapTextOutput(ClapTextConfig config) : base(nameof(ClapTextOutput))
    {
        dense = nn.Linear(config.IntermediateSize, config.HiddenSize);
        LayerNorm = nn.LayerNorm(config.HiddenSize, config.LayerNormEps);
        RegisterComponents();
    }

    public Tensor forward(Tensor hidden, Tensor residual)
    {
        using var projected = dense.forward(hidden);
        using var sum = projected + residual;
        return LayerNorm.forward(sum);
    }
}

/// <summary>One post-norm encoder layer (<c>ClapTextLayer</c>): attention block, then feed-forward block, each with its residual and LayerNorm.</summary>
public sealed class ClapTextLayer : nn.Module
{
    private readonly ClapTextAttention attention;
    private readonly ClapTextIntermediate intermediate;
    private readonly ClapTextOutput output;

    public ClapTextLayer(ClapTextConfig config) : base(nameof(ClapTextLayer))
    {
        attention = new ClapTextAttention(config);
        intermediate = new ClapTextIntermediate(config);
        output = new ClapTextOutput(config);
        RegisterComponents();
    }

    public Tensor forward(Tensor hidden, Tensor extendedMask)
    {
        using var attended = attention.forward(hidden, extendedMask);
        using var inner = intermediate.forward(attended);
        return output.forward(inner, attended);
    }
}
