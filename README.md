# Silero VAD

A voice activity detection (VAD) plugin for [OpenVoiceOS](https://github.com/OpenVoiceOS) that uses the [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model. VAD tells the listener when a chunk of audio contains speech, so the assistant knows when the user stops talking.

The plugin registers two components:

- `ovos-vad-plugin-silero`, a `VADEngine` that OVOS uses to detect silence during listening.
- `ovos-ww-verifier-silero`, a wake word verifier that checks whether the audio right after a wake word detection contains enough speech to confirm the activation.

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

To use the wake word verifier instead, set `ovos-ww-verifier-silero` as a wake word verifier plugin. It takes the same `model` and `threshold` options. The verifier default threshold is 0.1.

## Related projects

- [ovos-dinkum-listener](https://github.com/OpenVoiceOS/ovos-dinkum-listener), the OVOS listener that loads VAD plugins like this one.
- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager), defines the `VADEngine` and `HotWordVerifier` plugin templates this package implements.
- [silero-vad](https://github.com/snakers4/silero-vad), the upstream VAD model this plugin runs.

## License

Apache-2.0
