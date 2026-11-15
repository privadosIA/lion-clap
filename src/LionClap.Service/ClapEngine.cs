using LionClap.Audio;
using LionClap.Audio.Htsat;
using LionClap.Classification;
using LionClap.Text;
using LionClap.Weights;
using TorchSharp;

namespace LionClap.Service;

/// <summary>
/// The real model behind <see cref="IClassifier"/>: both branches, the tokenizer and the frontend, loaded from the checkpoint directory on the first
/// classification (as the reference service does) and kept for the next ones. A failed load is not remembered: the next request tries again.
/// Calls are serialised by the caller (<see cref="FifoGate"/>); loading is also guarded here.
/// </summary>
internal sealed class ClapEngine(ServiceSettings settings) : IClassifier, IDisposable
{
    private readonly object _load = new();
    private ClapClassifier? _classifier;
    private ClapAudioBranch? _audio;
    private ClapTextBranch? _text;
    private ClapTextTokenizer? _tokenizer;

    public bool IsLoaded { get { lock (_load) return _classifier is not null; } }

    public ClassifyResult Classify(float[] clip48k, ClassifyRequest request) => Ensure().Classify(clip48k, request);

    private ClapClassifier Ensure()
    {
        lock (_load)
        {
            if (_classifier is not null) return _classifier;

            var files = ClapCheckpointLocator.Locate(settings.CheckpointDirectory);
            if (!files.Ready) throw new InvalidOperationException($"the weights are missing: {string.Join(", ", files.Missing)}");
            var device = settings.Device == "cpu" ? torch.CPU : torch.device(settings.Device);

            ClapAudioBranch? audio = null;
            ClapTextBranch? text = null;
            ClapTextTokenizer? tokenizer = null;
            try
            {
                audio = ClapAudioBranch.Load(settings.CheckpointDirectory, device);
                text = ClapTextBranch.Load(settings.CheckpointDirectory, device);
                tokenizer = new ClapTextTokenizer(files.Tokenizer!, text.Config);
                (_audio, _text, _tokenizer) = (audio, text, tokenizer);
                return _classifier = new ClapClassifier(new ClapAudioEmbedder(audio, new LogMelExtractor()), new ClapTextEmbedder(text, tokenizer));
            }
            catch
            {
                audio?.Dispose();
                text?.Dispose();
                tokenizer?.Dispose();
                throw;
            }
        }
    }

    public void Dispose()
    {
        lock (_load)
        {
            _audio?.Dispose();
            _text?.Dispose();
            _tokenizer?.Dispose();
            (_audio, _text, _tokenizer, _classifier) = (null, null, null, null);
        }
    }
}
