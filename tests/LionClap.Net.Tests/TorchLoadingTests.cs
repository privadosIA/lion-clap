using LionClap.Audio.Htsat;
using LionClap.Text;
using LionClap.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class TorchLoadingTests(ITestOutputHelper output)
{
    /// <summary>Same structure as RoBERTa-base (12 layers, 203 parameters) with tiny sizes.</summary>
    private static readonly ClapTextConfig SmallText = new()
    {
        VocabSize = 100,
        HiddenSize = 32,
        NumHiddenLayers = 12,
        NumAttentionHeads = 4,
        IntermediateSize = 64,
        MaxPositionEmbeddings = 34,
        ProjectionDim = 16,
    };

    /// <summary>Same structure as HTSAT-tiny (12 blocks) with tiny widths.</summary>
    private static readonly ClapAudioConfig SmallAudio = new()
    {
        PatchEmbedsHiddenSize = 8,
        NumAttentionHeads = [2, 2, 2, 2],
        HiddenSize = 64,
        ProjectionDim = 16,
    };

    /// <summary>Writes the module's state in the layout of a <c>pytorch_model.bin</c> (float tensors as float32, the integer buffers as int64), after <paramref name="edit"/>.</summary>
    private static string WriteBin(nn.Module module, Action<Dictionary<string, TorchFileBuilder.Tensor>>? edit = null)
    {
        var entries = new Dictionary<string, TorchFileBuilder.Tensor>();
        var i = 0;
        foreach (var (name, t) in module.state_dict())
        {
            using var cpu = t.detach().cpu();
            var key = (i++).ToString();
            if (cpu.dtype == ScalarType.Int64)
            {
                var values = cpu.data<long>().ToArray();
                var data = new byte[values.Length * 8];
                Buffer.BlockCopy(values, 0, data, 0, data.Length);
                entries[name] = new(name, "I64", cpu.shape, data, key);
            }
            else
            {
                var values = cpu.to(ScalarType.Float32).data<float>().ToArray();
                var data = new byte[values.Length * 4];
                Buffer.BlockCopy(values, 0, data, 0, data.Length);
                entries[name] = new(name, "F32", cpu.shape, data, key);
            }
        }
        edit?.Invoke(entries);
        var builder = new TorchFileBuilder();
        foreach (var e in entries.Values) builder.AddRaw(e);
        return builder.WriteTo(TestTemp.Dir());
    }

    private static TorchFileBuilder.Tensor Floats(string name, long[] shape, string key) =>
        new(name, "F32", shape, new byte[shape.Aggregate(1L, (a, b) => a * b) * 4], key);

    private static void AssertSameState(nn.Module a, nn.Module b)
    {
        var sa = a.state_dict();
        var sb = b.state_dict();
        Assert.Equal(sa.Keys.OrderBy(k => k), sb.Keys.OrderBy(k => k));
        foreach (var (k, v) in sa) Assert.True(v.bytes.SequenceEqual(sb[k].bytes), $"{k} differs");
    }

    [Fact]
    public void TextBranchLoadsACompleteBinIgnoringForeignKeysAndEqualsTheSource()
    {
        manual_seed(1);
        using var source = new ClapTextBranch(SmallText);
        var bin = WriteBin(source, e => { e["audio_model.audio_encoder.norm.weight"] = Floats("audio_model.audio_encoder.norm.weight", [3], "900"); e["logit_scale_a"] = Floats("logit_scale_a", [], "901"); });

        using var target = new ClapTextBranch(SmallText);
        target.LoadWeights(TorchCheckpoint.Open(bin), CPU);
        AssertSameState(source, target);

        using var ids = tensor(new long[,] { { 0, 5, 6, 2, 1 } });
        using var mask = tensor(new long[,] { { 1, 1, 1, 1, 0 } });
        using var a = source.Forward(ids, mask);
        using var b = target.Forward(ids, mask);
        Assert.Equal(0f, (a.Embeddings - b.Embeddings).abs().max().item<float>());
    }

    [Fact]
    public void AudioBranchLoadsACompleteBinAndEqualsTheSource()
    {
        manual_seed(2);
        using var source = new ClapAudioBranch(SmallAudio);
        var bin = WriteBin(source, e => e["text_model.pooler.dense.bias"] = Floats("text_model.pooler.dense.bias", [3], "900"));
        using var target = new ClapAudioBranch(SmallAudio);
        target.LoadWeights(TorchCheckpoint.Open(bin), CPU);
        AssertSameState(source, target);
    }

    [Fact]
    public void ABrokenTextBinIsRejectedWithEveryDiscrepancyAndNothingIsChanged()
    {
        using var source = new ClapTextBranch(SmallText);
        var bin = WriteBin(source, e =>
        {
            e.Remove("text_model.pooler.dense.bias");                                                       // missing
            e["text_model.encoder.layer.0.bogus.weight"] = Floats("text_model.encoder.layer.0.bogus.weight", [2], "900"); // extra text key
            e["text_projection.linear1.weight"] = Floats("text_projection.linear1.weight", [7, 7], "901");  // wrong shape
        });

        using var target = new ClapTextBranch(SmallText);
        var before = target.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(TorchCheckpoint.Open(bin), CPU));
        output.WriteLine(ex.Message);
        Assert.Contains("The checkpoint does not match the text branch", ex.Message);
        Assert.Contains("text_model.pooler.dense.bias", ex.Message);
        Assert.Contains("text_model.encoder.layer.0.bogus.weight", ex.Message);
        Assert.Contains("text_projection.linear1.weight", ex.Message);
        foreach (var (key, value) in target.state_dict()) Assert.True(value.bytes.SequenceEqual(before[key].bytes), $"{key} changed");
    }

    [Fact]
    public void ABrokenAudioBinIsRejectedWithEveryDiscrepancyAndNothingIsChanged()
    {
        using var source = new ClapAudioBranch(SmallAudio);
        var bin = WriteBin(source, e =>
        {
            e.Remove("audio_model.audio_encoder.norm.bias");
            e["audio_model.audio_encoder.bogus.weight"] = Floats("audio_model.audio_encoder.bogus.weight", [2], "900");
            e["audio_projection.linear1.weight"] = Floats("audio_projection.linear1.weight", [7, 7], "901");
        });
        using var target = new ClapAudioBranch(SmallAudio);
        var before = target.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(TorchCheckpoint.Open(bin), CPU));
        Assert.Contains("audio_model.audio_encoder.norm.bias", ex.Message);
        Assert.Contains("audio_model.audio_encoder.bogus.weight", ex.Message);
        Assert.Contains("audio_projection.linear1.weight", ex.Message);
        foreach (var (key, value) in target.state_dict()) Assert.True(value.bytes.SequenceEqual(before[key].bytes), $"{key} changed");
    }

    [RequiresBothCheckpoints]
    public void TheAudioBranchFromTheRealBinEqualsTheOneFromSafetensors()
    {
        using var fromBin = new ClapAudioBranch();
        fromBin.LoadWeights(TorchCheckpoint.Open(Checkpoint.RealBin!), CPU);
        using var fromSafetensors = ClapAudioBranch.Load(Checkpoint.Dir!);
        AssertSameState(fromBin, fromSafetensors);

        var input = Fixtures.Load("frontend.safetensors")["input_features"];
        using var a = fromBin.Forward(input);
        using var b = fromSafetensors.Forward(input);
        Assert.Equal(0f, (a.Embeddings - b.Embeddings).abs().max().item<float>());
        output.WriteLine($"audio: {fromBin.state_dict().Count} tensors bit-identical from either format; embeddings equal");
    }

    [RequiresBothCheckpoints]
    public void TheTextBranchFromTheRealBinEqualsTheOneFromSafetensors()
    {
        using var fromBin = new ClapTextBranch();
        fromBin.LoadWeights(TorchCheckpoint.Open(Checkpoint.RealBin!), CPU);
        using var fromSafetensors = ClapTextBranch.Load(Checkpoint.Dir!);
        AssertSameState(fromBin, fromSafetensors);

        var (ids, mask) = TextCases.Tensors(0, 1, 2, 3, 4);
        using var idsOwner = ids;
        using var maskOwner = mask;
        using var a = fromBin.Forward(ids, mask);
        using var b = fromSafetensors.Forward(ids, mask);
        Assert.Equal(0f, (a.Embeddings - b.Embeddings).abs().max().item<float>());
        output.WriteLine($"text: {fromBin.state_dict().Count} tensors bit-identical from either format; embeddings equal");
    }
}
