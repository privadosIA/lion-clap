using LionClap.Text;

namespace LionClap.Tests;

/// <summary>The tokenizer is built once per class, and only when a gated test runs (tokenizer.json comes from CLAP_CKPT).</summary>
public sealed class TokenizerFixture : IDisposable
{
    private readonly Lazy<ClapTextTokenizer> _tokenizer = new(() => new ClapTextTokenizer(Path.Combine(Checkpoint.Dir!, "tokenizer.json")));

    public ClapTextTokenizer Tokenizer => _tokenizer.Value;

    public void Dispose()
    {
        if (_tokenizer.IsValueCreated) _tokenizer.Value.Dispose();
    }
}

public class ClapTextTokenizerTests(TokenizerFixture fixture) : IClassFixture<TokenizerFixture>
{
    private static IReadOnlyList<string> Texts => TextCases.All.Select(c => c.Text).ToList();

    [RequiresCheckpoint]
    public void FixedLengthEqualsTheStoredIdsAndMasks()
    {
        var batch = fixture.Tokenizer.EncodeBatch(Texts, 77);
        Assert.Equal((5, 77), (batch.Rows, batch.Length));
        for (var r = 0; r < 5; r++)
        {
            Assert.Equal(TextCases.All[r].Ids, batch.Ids[(r * 77)..((r + 1) * 77)]);
            Assert.Equal(TextCases.All[r].Mask, batch.Mask[(r * 77)..((r + 1) * 77)]);
        }
        Assert.Equal(2, batch.Ids[2 * 77 + 76]); // the long text is cut to 77 and ends with the end token
        Assert.Equal(1L, batch.Mask[2 * 77 + 76]);
    }

    [RequiresCheckpoint]
    public void LongestModeEqualsTheFixedRowsCutAtTheLongest()
    {
        var shorter = new[] { 0, 1, 3, 4 };
        var batch = fixture.Tokenizer.EncodeBatch(shorter.Select(i => TextCases.All[i].Text).ToList());
        var longest = shorter.Max(i => TextCases.All[i].Length);
        Assert.Equal((4, longest), (batch.Rows, batch.Length));
        for (var r = 0; r < 4; r++)
        {
            Assert.Equal(TextCases.All[shorter[r]].Ids[..longest], batch.Ids[(r * longest)..((r + 1) * longest)]);
            Assert.Equal(TextCases.All[shorter[r]].Mask[..longest], batch.Mask[(r * longest)..((r + 1) * longest)]);
        }
    }

    [RequiresCheckpoint]
    public void EmptyTextIsStartAndEndOnly() => Assert.Equal(new long[] { 0, 2 }, fixture.Tokenizer.Encode(""));

    [RequiresCheckpoint]
    public void MoreThanTheModelAcceptsIsRejectedNotCut()
    {
        var huge = string.Join(' ', Enumerable.Repeat("sound", 600));
        var ex = Assert.Throws<ArgumentException>(() => fixture.Tokenizer.EncodeBatch(new[] { huge }));
        Assert.Contains("512", ex.Message);
        var cut = fixture.Tokenizer.EncodeBatch(new[] { huge }, 512); // explicit maximum: cut, end token kept
        Assert.Equal(2, cut.Ids[511]);
        Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Tokenizer.EncodeBatch(new[] { huge }, 513));
    }

    [Fact]
    public void AMissingTokenizerFileIsReported()
    {
        var path = Path.Combine(TestTemp.Dir(), "missing.json");
        Assert.Throws<FileNotFoundException>(() => new ClapTextTokenizer(path));
    }
}
