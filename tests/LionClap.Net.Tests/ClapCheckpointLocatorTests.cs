using LionClap.Audio.Htsat;
using LionClap.Text;
using LionClap.Weights;

namespace LionClap.Tests;

public class ClapCheckpointLocatorTests
{
    private static string Dir() => TestTemp.Dir();

    private static string Touch(string directory, string name)
    {
        var path = Path.Combine(directory, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "x");
        return path;
    }

    private static string Snapshots(string root, string revision) => Path.Combine(root, "hub", ClapCheckpointLocator.HubId, "snapshots", revision);

    [Fact]
    public void FlatSafetensorsDirectory()
    {
        var d = Dir();
        Touch(d, "model.safetensors"); Touch(d, "config.json"); Touch(d, "tokenizer.json");
        var files = ClapCheckpointLocator.Locate(d);
        Assert.Equal((WeightsFormat.SafeTensors, true), (files.Format, files.Ready));
        Assert.Equal(Path.Combine(d, "model.safetensors"), files.Weights);
        Assert.Equal(Path.Combine(d, "config.json"), files.Config);
        Assert.Equal(Path.Combine(d, "tokenizer.json"), files.Tokenizer);
    }

    [Fact]
    public void FlatPyTorchDirectory_AlsoWhenTheDirectoryIsASnapshot()
    {
        var d = Dir();
        Touch(d, "pytorch_model.bin"); Touch(d, "config.json"); Touch(d, "tokenizer.json");
        var files = ClapCheckpointLocator.Locate(d);
        Assert.Equal((WeightsFormat.PyTorch, true), (files.Format, files.Ready));
    }

    [Fact]
    public void SafetensorsWinsWhenBothFormatsArePresent()
    {
        var d = Dir();
        Touch(d, "pytorch_model.bin"); Touch(d, "model.safetensors"); Touch(d, "tokenizer.json");
        Assert.Equal(WeightsFormat.SafeTensors, ClapCheckpointLocator.Locate(d).Format);
    }

    [Fact]
    public void HuggingFaceCacheLayoutOfThePythonContainer()
    {
        var root = Dir();
        var snap = Snapshots(root, "8fa0f1c6");
        Touch(snap, "pytorch_model.bin"); Touch(snap, "config.json"); Touch(snap, "tokenizer.json");
        var files = ClapCheckpointLocator.Locate(root);
        Assert.Equal((WeightsFormat.PyTorch, true), (files.Format, files.Ready));
        Assert.Equal(Path.Combine(snap, "pytorch_model.bin"), files.Weights);
        Assert.Equal(Path.Combine(snap, "tokenizer.json"), files.Tokenizer);
    }

    [Fact]
    public void TheLastRevisionThatHoldsWeightsIsUsed()
    {
        var root = Dir();
        var older = Snapshots(root, "aaaa"); var newer = Snapshots(root, "bbbb"); var empty = Snapshots(root, "cccc");
        Touch(older, "pytorch_model.bin"); Touch(older, "tokenizer.json");
        Touch(newer, "pytorch_model.bin"); Touch(newer, "tokenizer.json");
        Directory.CreateDirectory(empty);                                    // the newest by name has no weights: skipped
        Assert.Equal(Path.Combine(newer, "pytorch_model.bin"), ClapCheckpointLocator.Locate(root).Weights);
    }

    [Fact]
    public void FlatLayoutTakesPrecedenceOverTheCache()
    {
        var root = Dir();
        Touch(root, "model.safetensors"); Touch(root, "tokenizer.json");
        Touch(Snapshots(root, "aaaa"), "pytorch_model.bin");
        Assert.Equal(Path.Combine(root, "model.safetensors"), ClapCheckpointLocator.Locate(root).Weights);
    }

    [Fact]
    public void APathToTheWeightsFileUsesItsDirectory()
    {
        var d = Dir();
        var weights = Touch(d, "model.safetensors"); Touch(d, "tokenizer.json");
        var files = ClapCheckpointLocator.Locate(weights);
        Assert.Equal(weights, files.Weights);
        Assert.True(files.Ready);
    }

    [Fact]
    public void NothingUsableIsReportedWithWhatIsMissing()
    {
        var nonexistent = ClapCheckpointLocator.Locate(Path.Combine(Dir(), "nope"));
        Assert.False(nonexistent.Ready);
        Assert.Null(nonexistent.Weights);
        Assert.Equal(new[] { ClapCheckpointLocator.HubId }, nonexistent.Missing);

        Assert.Equal(new[] { ClapCheckpointLocator.HubId }, ClapCheckpointLocator.Locate(Dir()).Missing); // empty directory

        var noTokenizer = Dir();
        Touch(noTokenizer, "model.safetensors");
        var files = ClapCheckpointLocator.Locate(noTokenizer);
        Assert.False(files.Ready);
        Assert.NotNull(files.Weights);
        Assert.Equal(new[] { "tokenizer.json" }, files.Missing);
    }

    [Fact]
    public void LoadingFromADirectoryWithoutWeightsFailsNamingWhatIsMissing()
    {
        var ex = Assert.Throws<FileNotFoundException>(() => ClapAudioBranch.Load(Dir()));
        Assert.Contains(ClapCheckpointLocator.HubId, ex.Message);
        Assert.Throws<FileNotFoundException>(() => ClapTextBranch.Load(Dir()));
    }

    [RequiresBothCheckpoints]
    public void ThePythonContainersDirectoryAndTheConvertedOneAreBothLocated()
    {
        var hf = ClapCheckpointLocator.Locate(Checkpoint.HfDir!);
        Assert.Equal((WeightsFormat.PyTorch, true), (hf.Format, hf.Ready));
        Assert.NotNull(hf.Config);
        Assert.NotNull(hf.Tokenizer);

        var converted = ClapCheckpointLocator.Locate(Checkpoint.Dir!);
        Assert.Equal((WeightsFormat.SafeTensors, true), (converted.Format, converted.Ready));
    }

    [RequiresBothCheckpoints]
    public void BothBranchesLoadFromThePythonContainersDirectory()
    {
        using var audio = ClapAudioBranch.Load(Checkpoint.HfDir!);
        using var text = ClapTextBranch.Load(Checkpoint.HfDir!);
        Assert.Equal(240, audio.state_dict().Count);
        Assert.Equal(205, text.state_dict().Count);
    }
}
