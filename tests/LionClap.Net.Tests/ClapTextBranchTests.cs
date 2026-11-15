using LionClap.Text;
using MusicGen.Core.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class ClapTextBranchTests(ITestOutputHelper output)
{
    /// <summary>Same structure as RoBERTa-base (12 layers) with tiny sizes, so crafted checkpoints stay small.</summary>
    private static readonly ClapTextConfig Small = new()
    {
        VocabSize = 100,
        HiddenSize = 32,
        NumHiddenLayers = 12,
        NumAttentionHeads = 4,
        IntermediateSize = 64,
        MaxPositionEmbeddings = 34,
        ProjectionDim = 16,
    };

    private static string WriteCheckpoint(ClapTextBranch branch, Action<Dictionary<string, Tensor>> edit)
    {
        var state = branch.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        // keys outside the text branch must be ignored by the loader
        state["audio_model.audio_encoder.norm.weight"] = zeros(3);
        state["logit_scale_t"] = zeros(1);
        edit(state);
        var path = Path.Combine(TestTemp.Dir(), "model.safetensors");
        SafeTensorsWriter.Save(path, state);
        return path;
    }

    [Fact]
    public void CensusMatchesTheCheckpointLayout()
    {
        using var branch = new ClapTextBranch(Small);
        Assert.Equal(203, branch.named_parameters().Count()); // text_model 199 + projection 4
        Assert.Equal(2, branch.named_buffers().Count());
        Assert.Equal(205, branch.state_dict().Count);          // the 205 text tensors of the checkpoint
        Assert.Contains("text_model.embeddings.position_ids", branch.state_dict().Keys);
        Assert.Contains("text_projection.linear2.bias", branch.state_dict().Keys);
    }

    [Fact]
    public void ABrokenCheckpointIsRejectedWithEveryDiscrepancyAndNothingIsChanged()
    {
        using var source = new ClapTextBranch(Small);
        var path = WriteCheckpoint(source, state =>
        {
            state.Remove("text_model.pooler.dense.bias");                       // missing
            state["text_model.encoder.layer.0.bogus.weight"] = zeros(2);        // extra text key
            state["text_projection.linear1.weight"] = zeros(7, 7);              // wrong shape
        });

        using var target = new ClapTextBranch(Small);
        var before = target.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(CheckpointIndex.Open(path), CPU));
        output.WriteLine(ex.Message);
        Assert.Contains("text_model.pooler.dense.bias", ex.Message);
        Assert.Contains("text_model.encoder.layer.0.bogus.weight", ex.Message);
        Assert.Contains("text_projection.linear1.weight", ex.Message);

        foreach (var (key, value) in target.state_dict())
            Assert.Equal(0.0, (value.to(ScalarType.Float64) - before[key].to(ScalarType.Float64)).abs().max().item<double>());
    }

    [Fact]
    public void ACompleteCheckpointLoadsAndForeignKeysAreIgnored()
    {
        using var source = new ClapTextBranch(Small);
        var path = WriteCheckpoint(source, _ => { });
        using var target = new ClapTextBranch(Small);
        target.LoadWeights(CheckpointIndex.Open(path), CPU);

        using var ids = tensor(new long[,] { { 0, 5, 6, 2, 1 } });
        using var mask = tensor(new long[,] { { 1, 1, 1, 1, 0 } });
        using var a = source.Forward(ids, mask);
        using var b = target.Forward(ids, mask);
        Assert.Equal(0f, (a.Embeddings - b.Embeddings).abs().max().item<float>()); // same weights, same result
    }

    [Fact]
    public void ForwardRejectsInvalidInputsWithAMessage()
    {
        using var branch = new ClapTextBranch(Small);
        using var ids = tensor(new long[,] { { 0, 150, 2 } });
        using var mask = ones(1, 3, dtype: ScalarType.Int64);
        var ex = Assert.Throws<ArgumentException>(() => branch.Forward(ids, mask));
        Assert.Contains("150", ex.Message);
        Assert.Contains("100", ex.Message);
    }

    [RequiresCheckpoint]
    public void TheRealCheckpointLoadsEveryTextTensorAndIgnoresTheRest()
    {
        using var branch = ClapTextBranch.Load(Checkpoint.Dir!);
        var textKeys = Checkpoint.Index.Tensors.Keys.Count(k => k.StartsWith("text_"));
        Assert.Equal(205, textKeys);
        Assert.Equal(textKeys, branch.state_dict().Count); // none left unloaded, none invented
        var ignored = Checkpoint.Index.Tensors.Keys.Count(k => !k.StartsWith("text_"));
        output.WriteLine($"text tensors {textKeys}, ignored {ignored}");
        Assert.Equal(447 - 205, ignored);
    }
}
