# LionClap.Net

C# port (.NET 8 + TorchSharp) of LAION-CLAP, inference only. It embeds audio and text into the same 512-dimensional space, so you can score how well an audio clip matches a list of text tags. It uses the Hugging Face weights `laion/clap-htsat-unfused` and needs no Python at runtime.

## Articles

- [Audio frontend](audio-frontend.md): WAV decoding, resampling to 48 kHz and the log-mel features.
- [HTSAT audio encoder](htsat-audio-encoder.md): the audio branch and the audio embedding.
- [RoBERTa text encoder](roberta-text-encoder.md): the text branch and the text embedding.
- [The HTTP service](http-service.md): `GET /health` and `POST /classify`, a drop-in for the Python service.

## API reference

The public API of the `LionClap` library, generated from the XML comments of the source:

- [LionClap.Audio](api/LionClap.Audio.yml): WAV decoder, resampler, log-mel frontend and the audio embedder (with `LionClap.Audio.Htsat`, the HTSAT branch).
- [LionClap.Text](api/LionClap.Text.yml): the tokenizer and the RoBERTa text branch.
- [LionClap.Classification](api/LionClap.Classification.yml): scoring of tags against a clip.
- [LionClap.Weights](api/LionClap.Weights.yml): reading and strict loading of the checkpoints.
