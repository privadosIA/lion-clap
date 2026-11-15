using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LionClap.Text;

/// <summary>The stack of encoder layers (<c>ClapTextEncoder</c>, key <c>layer.N</c>).</summary>
public sealed class ClapTextEncoder : nn.Module
{
    private readonly ModuleList<ClapTextLayer> layer;

    public ClapTextEncoder(ClapTextConfig config) : base(nameof(ClapTextEncoder))
    {
        layer = nn.ModuleList(Enumerable.Range(0, config.NumHiddenLayers).Select(_ => new ClapTextLayer(config)).ToArray());
        RegisterComponents();
    }

    /// <summary>Runs every layer. <paramref name="tap"/>, when given, receives <c>layer00</c>, <c>layer01</c>, … after each; tensors are only valid during the call.</summary>
    public Tensor forward(Tensor hidden, Tensor extendedMask, Action<string, Tensor>? tap = null)
    {
        var current = hidden;
        for (var i = 0; i < layer.Count; i++)
        {
            var next = layer[i].forward(current, extendedMask);
            if (!ReferenceEquals(current, hidden)) current.Dispose();
            current = next;
            tap?.Invoke($"layer{i:00}", current);
        }
        return current;
    }
}

/// <summary>The pooler (<c>ClapTextPooler</c>): the first token through <c>dense</c> and tanh.</summary>
public sealed class ClapTextPooler : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;

    public ClapTextPooler(ClapTextConfig config) : base(nameof(ClapTextPooler))
    {
        dense = nn.Linear(config.HiddenSize, config.HiddenSize);
        RegisterComponents();
    }

    /// <summary><c>[B, L, C]</c> → <c>[B, C]</c>.</summary>
    public override Tensor forward(Tensor hidden)
    {
        using var first = hidden[.., 0];
        using var projected = dense.forward(first);
        return tanh(projected);
    }
}

/// <summary>
/// <c>ClapTextModel</c>: embeddings, encoder and pooler. Field names are the checkpoint keys under <c>text_model.</c>.
/// <see cref="Encode"/> maps ids and attention mask <c>[B, L]</c> to the pooled output <c>[B, 768]</c>.
/// </summary>
public sealed class ClapTextModel : nn.Module
{
    private readonly ClapTextEmbeddings embeddings;
    private readonly ClapTextEncoder encoder;
    private readonly ClapTextPooler pooler;
    private readonly ClapTextConfig _config;

    public ClapTextModel(ClapTextConfig config) : base(nameof(ClapTextModel))
    {
        _config = config;
        embeddings = new ClapTextEmbeddings(config);
        encoder = new ClapTextEncoder(config);
        pooler = new ClapTextPooler(config);
        RegisterComponents();
    }

    public Tensor forward(Tensor inputIds, Tensor attentionMask) => Encode(inputIds, attentionMask, null);

    /// <summary>
    /// Runs the model. <paramref name="tap"/>, when given, receives the intermediate activations under the names of the golden fixture
    /// (<c>embeddings</c>, <c>layer00</c>…<c>layer11</c>, <c>pooler</c>); the tensors are only valid during the call.
    /// </summary>
    public Tensor Encode(Tensor inputIds, Tensor attentionMask, Action<string, Tensor>? tap)
    {
        TextInputs.Validate(inputIds, attentionMask, _config);
        using var extended = TextMasks.Extend(attentionMask);
        using var embedded = embeddings.forward(inputIds);
        tap?.Invoke("embeddings", embedded);
        using var hidden = encoder.forward(embedded, extended, tap);
        var pooled = pooler.forward(hidden);
        tap?.Invoke("pooler", pooled);
        return pooled;
    }
}
