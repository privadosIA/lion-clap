using LionClap.Audio.Htsat;
using LionClap.Text;
using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;

namespace LionClap.Tests;

/// <summary>
/// Opt-in: the same embedding parity as the CPU tests, with the branches on the GPU (<c>CLAP_DEVICE=cuda</c>, a CUDA build, <c>CLAP_CKPT</c>). The CPU is the verified path of the
/// project; this shows whether the GPU stays inside the same limits (max_abs 1e-4, cos_sim 0.9999).
/// </summary>
public class GpuParityTests(ITestOutputHelper output)
{
    private static torch.Device Device => torch.device(Environment.GetEnvironmentVariable("CLAP_DEVICE")!.ToLowerInvariant());

    [RequiresCudaCheckpoint]
    public void AudioEmbeddingsOnTheGpuMatchTheReference()
    {
        using var branch = ClapAudioBranch.Load(Checkpoint.Dir!, Device);
        using var result = branch.Forward(Fixtures.Load("frontend.safetensors")["input_features"]);
        var r = Parity.Compare(result.Embeddings, Fixtures.Load("audio_embeddings.safetensors")["audio_embeddings"]);
        output.WriteLine($"audio on {Device}: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, r.ToString());
    }

    [RequiresCudaCheckpoint]
    public void TextEmbeddingsOnTheGpuMatchTheReference()
    {
        using var branch = ClapTextBranch.Load(Checkpoint.Dir!, Device);
        var (ids, mask) = TextCases.Tensors(0, 1, 2, 3, 4);
        using var idsOwner = ids;
        using var maskOwner = mask;
        using var result = branch.Forward(ids, mask);
        var r = Parity.Compare(result.Embeddings, Fixtures.Load("text_embeddings.safetensors")["text_embeddings"]);
        output.WriteLine($"text on {Device}: {r}");
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, r.ToString());
    }
}
