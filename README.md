# Silero VAD

A voice activity detection (VAD) plugin for [OpenVoiceOS](https://github.com/OpenVoiceOS) that uses the [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model. VAD tells the listener when a chunk of audio contains speech, so the assistant knows when the user stops talking.

The plugin registers two components:

- `ovos-vad-plugin-silero`, a `VADEngine` that OVOS uses to detect silence during listening.
- `ovos-ww-verifier-silero`, a wake word verifier that checks whether the audio that leads up to a wake word detection contains enough speech to confirm the activation.

Both use the same bundled Silero ONNX model, or a custom model path you provide. Input audio must be 16 kHz, 16-bit mono PCM.

## Install

```bash
pip install ovos-vad-plugin-silero
```

## Usage

Set `ovos-vad-plugin-silero` as the VAD module in your listener config:

```json
{
    "listener": {
        "VAD": {
            "module": "ovos-vad-plugin-silero",
            "ovos-vad-plugin-silero": {
                "model": "/optional/path/to/model.onnx",
                "threshold": 0.2
            }
        }
    }
}
```

- `model` is optional. If you leave it out, the plugin uses the Silero ONNX model bundled with the package.
- `threshold` is the minimum speech probability (0 to 1) for a chunk to count as speech. The default is 0.2.

To use the wake word verifier instead, set `ovos-ww-verifier-silero` as a wake word verifier plugin. The verifier scores the audio the listener keeps before the detection in 32 ms frames. It confirms the activation when one second of those frames holds enough speech. Silence before or after the word does not dilute the result. The wake word must be inside that kept audio: with the default 15 chunks, that is 0.96 s at 64 ms chunks and 1.92 s at 128 ms chunks. It takes these options:

- `model` is the path to a custom Silero ONNX model.
- `threshold` is the minimum speech probability (0 to 1) for a frame to count as speech. The default is 0.5. The stock ovos-config sets 0.1 for this verifier, and that value also applies per frame.
- `min_speech_ratio` is the share of frames (0 to 1) in the best window that must be speech to confirm the activation. The default is 0.15.
- `window_seconds` is the length of that window in seconds. The default is 1.0.

## Related projects

- [ovos-dinkum-listener](https://github.com/OpenVoiceOS/ovos-dinkum-listener), the OVOS listener that loads VAD plugins like this one.
- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager), defines the `VADEngine` and `HotWordVerifier` plugin templates this package implements.
- [silero-vad](https://github.com/snakers4/silero-vad), the upstream VAD model this plugin runs.

## License

Apache-2.0
