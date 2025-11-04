from os.path import join, dirname
from typing import Optional, Dict, Any

import numpy as np
import onnxruntime
from ovos_plugin_manager.templates.hotwords import HotWordVerifier
from ovos_plugin_manager.templates.vad import VADEngine


class SileroVoiceActivityDetector:
    """Detects speech/silence using Silero VAD.

    https://github.com/snakers4/silero-vad
    """

    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.session.intra_op_num_threads = 1
        self.session.inter_op_num_threads = 1

        self.reset()

    def reset(self):
        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def __call__(self, audio_array: np.ndarray, sample_rate: int = 16000):
        """Return probability of speech in audio [0-1].

        Audio must be 16Khz 16-bit mono PCM.
        """
        if len(audio_array.shape) == 1:
            # Add batch dimension
            audio_array = np.expand_dims(audio_array, 0)

        if len(audio_array.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {audio_array.dim()}"
            )

        if audio_array.shape[0] > 1:
            raise ValueError("Onnx model does not support batching")

        if sample_rate != 16000:
            raise ValueError("Only 16Khz audio is supported")

        ort_inputs = {
            "input": audio_array.astype(np.float32),
            "h0": self._h,
            "c0": self._c,
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs

        out = out.squeeze(2)[:, 1]  # make output type match JIT analog

        return out


class SileroVAD(VADEngine):
    def __init__(self, config=None, sample_rate=None):
        super().__init__(config, sample_rate)
        model = self.config.get("model") or join(dirname(__file__), "silero_vad.onnx")
        self.vad_threshold = self.config.get("threshold", 0.2)
        self.vad = SileroVoiceActivityDetector(model)

    def reset(self):
        self.vad.reset()

    def is_silence(self, chunk):
        # return True or False
        audio_array = np.frombuffer(chunk, dtype=np.int16)
        return self.vad(audio_array)[0] < self.vad_threshold


class SileroVADVerifier(HotWordVerifier):
    """
    A HotWordVerifier plugin that uses Silero VAD to confirm that the audio
    chunk following a wake word detection contains sufficient human speech.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vad_model_path = self.config.get("model") or join(dirname(__file__), "silero_vad.onnx")
        # Probability threshold for audio to be considered "speech"
        self.vad_threshold = self.config.get("threshold", 0.1)
        self.vad = SileroVoiceActivityDetector(self.vad_model_path)

    def verify(self, chunk: bytes) -> bool:
        """
        Verifies the audio chunk: requires a minimum ratio of VAD frames
        to be classified as speech to confirm the activation.
        
        Args:
            chunk: Audio bytes in 16-bit PCM format at 16kHz sample rate
            
        Returns:
            True if the audio contains sufficient speech, False otherwise
        """
        try:
            audio_array_int16 = np.frombuffer(chunk, dtype=np.int16)
            prob = self.vad(audio_array_int16)[0]
            return prob >= self.vad_threshold
        except (ValueError, Exception) as e:
            # Log the error and return False to reject invalid audio
            return False