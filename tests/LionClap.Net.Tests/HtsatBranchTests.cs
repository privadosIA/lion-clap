using LionClap.Audio.Htsat;
using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace LionClap.Tests;

public class HtsatBranchTests(ITestOutputHelper output)
{
    /// <summary>Writes a checkpoint made of the branch's own state, with the given edits, to a temporary file.</summary>
    private static string WriteCheckpoint(ClapAudioBranch branch, Action<Dictionary<string, Tensor>> edit)
    {
        var state = branch.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        // keys outside the audio branch must be ignored by the loader
        state["text_model.embeddings.word_embeddings.weight"] = zeros(3, 4);
        state["logit_scale_a"] = zeros(1);
        edit(state);
        var path = Path.Combine(TestTemp.Dir(), "model.safetensors");
        SafeTensorsWriter.Save(path, state);
        return path;
    }

    [Fact]
    public void ABrokenCheckpointIsRejectedWithEveryDiscrepancyAndNothingIsChanged()
    {
        using var source = new ClapAudioBranch(SmallConfigs.Audio);
        var path = WriteCheckpoint(source, state =>
        {
            state.Remove("audio_model.audio_encoder.norm.bias");                                   // missing
            state["audio_model.audio_encoder.bogus.weight"] = zeros(2);                            // extra audio key
            state["audio_projection.linear1.weight"] = zeros(7, 7);                                // wrong shape
        });

        using var target = new ClapAudioBranch(SmallConfigs.Audio);
        var before = target.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(CheckpointIndex.Open(path), CPU));
        output.WriteLine(ex.Message);
        Assert.Contains("audio_model.audio_encoder.norm.bias", ex.Message);
        Assert.Contains("audio_model.audio_encoder.bogus.weight", ex.Message);
        Assert.Contains("audio_projection.linear1.weight", ex.Message);

        foreach (var (key, value) in target.state_dict())
            Assert.Equal(0.0, (value.to(ScalarType.Float64) - before[key].to(ScalarType.Float64)).abs().max().item<double>());
    }

    [Fact]
    public void ACompleteCheckpointLoadsAndForeignKeysAreIgnored()
    {
        using var source = new ClapAudioBranch(SmallConfigs.Audio);
        var path = WriteCheckpoint(source, _ => { });

        using var target = new ClapAudioBranch(SmallConfigs.Audio);
        var result = StrictLoaderResult(target, path);
        Assert.Equal(source.state_dict().Count, result);
    }

    private static int StrictLoaderResult(ClapAudioBranch target, string path)
    {
        target.LoadWeights(CheckpointIndex.Open(path), CPU);
        return target.state_dict().Count;
    }

    [Fact]
    public void AnInputWithAWrongMelCountIsRejected()
    {
        using var branch = new ClapAudioBranch(SmallConfigs.Audio);
        using var input = zeros(1, 1, 100, 80);
        var ex = Assert.Throws<ArgumentException>(() => branch.Forward(input));
        Assert.Contains("64", ex.Message);
    }

    [RequiresCheckpoint]
    public void TheRealCheckpointLoadsEveryAudioTensor()
    {
        using var branch = ClapAudioBranch.Load(Checkpoint.Dir!);
        var audioKeys = Checkpoint.Index.Tensors.Keys.Count(k => k.StartsWith("audio_"));
        Assert.Equal(240, audioKeys);                       // 236 encoder tensors + 4 projection tensors
        Assert.Equal(audioKeys, branch.state_dict().Count); // none left unloaded, none invented
        var ignored = Checkpoint.Index.Tensors.Keys.Count(k => !k.StartsWith("audio_"));
        output.WriteLine($"audio tensors {audioKeys}, ignored {ignored}");
        Assert.Equal(447 - 240, ignored);
    }
}
