using LionClap.Text;

namespace LionClap.Tests;

public class ClapTextConfigTests
{
    private static string TempConfig(string textConfigJson)
    {
        var path = Path.Combine(TestTemp.Dir(), "config.json");
        File.WriteAllText(path, $"{{\"text_config\": {textConfigJson}}}");
        return path;
    }

    [Fact]
    public void DefaultsDescribeRobertaBase()
    {
        var c = new ClapTextConfig();
        Assert.Equal(512, c.MaxTokens);          // 514 position rows, the first two unusable
        Assert.Equal(64, c.HiddenSize / c.NumAttentionHeads);
        Assert.Equal(1e-12, c.LayerNormEps);
    }

    [Theory]
    [InlineData("{\"position_embedding_type\": \"relative_key\"}", "absolute")]
    [InlineData("{\"is_decoder\": true}", "decoder")]
    [InlineData("{\"hidden_act\": \"relu\"}", "gelu")]
    public void UnsupportedConfigurationsAreRejected(string json, string expectedInMessage)
    {
        var ex = Assert.Throws<NotSupportedException>(() => ClapTextConfig.FromFile(TempConfig(json)));
        Assert.Contains(expectedInMessage, ex.Message);
    }

    [Fact]
    public void MissingTextConfigIsReported()
    {
        var path = Path.Combine(TestTemp.Dir(), "config.json");
        File.WriteAllText(path, "{}");
        Assert.Throws<InvalidDataException>(() => ClapTextConfig.FromFile(path));
    }

    [RequiresCheckpoint]
    public void DefaultsEqualTheRealConfigJson()
    {
        var real = ClapTextConfig.FromFile(Checkpoint.ConfigFile);
        Assert.Equal(new ClapTextConfig(), real);
        Assert.Equal(1e-12, real.LayerNormEps);
    }
}
