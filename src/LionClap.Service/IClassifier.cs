using LionClap.Classification;

namespace LionClap.Service;

/// <summary>What the HTTP layer needs from the model: classify a 48 kHz mono clip. It is only called by one request at a time.</summary>
internal interface IClassifier
{
    ClassifyResult Classify(float[] clip48k, ClassifyRequest request);
}
