using MusicGen.Core.Testing;

namespace LionClap.Tests;

public class FixturesTests
{
    [Fact]
    public void FrontendFixtureIsReadable()
    {
        var f = Fixtures.Load("frontend.safetensors");
        Assert.Contains("input_features", f.Keys);
        Assert.Equal(new long[] { 5, 1, 1001, 64 }, f["input_features"].shape);
        Assert.Equal(new long[] { 513, 64 }, f["mel_filters_slaney"].shape);
    }

    [Fact]
    public void EveryFixtureIsVerifiedAgainstTheManifest()
    {
        var files = Directory.GetFiles(Fixtures.Dir).Select(Path.GetFileName).Where(n => n != "MANIFEST.md").ToList();
        Assert.NotEmpty(files);
        foreach (var file in files)
            Assert.Null(Record.Exception(() => Fixtures.PathOf(file!))); // PathOf throws InvalidDataException on a mismatch
    }

    [Fact]
    public void ATamperedFixtureIsRejectedByName()
    {
        var tmp = new DirectoryInfo(TestTemp.Dir());
        try
        {
            File.Copy(Path.Combine(Fixtures.Dir, "MANIFEST.md"), Path.Combine(tmp.FullName, "MANIFEST.md"));
            File.Copy(Path.Combine(Fixtures.Dir, "text_cases.json"), Path.Combine(tmp.FullName, "text_cases.json"));
            File.AppendAllText(Path.Combine(tmp.FullName, "text_cases.json"), " ");

            var ex = Assert.Throws<InvalidDataException>(() => new FixtureDirectory(tmp.FullName).PathOf("text_cases.json"));
            Assert.Contains("text_cases.json", ex.Message);
        }
        finally { tmp.Delete(true); }
    }
}
