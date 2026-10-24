"""The Silero wake word verifier must judge the wake word, not the silence a
listener keeps before it.

The listener hands the verifier its last ``wakeword_chunks_to_save`` chunks.
At the default 128 ms chunk that is 1.92 s, and most of it can be silence
before a short wake word, or after it when the engine fires late. The speech segment here is chosen by signal energy,
independently of Silero, so the expected answer does not come from the model
under test.
"""
from __future__ import annotations

import unittest
import wave
from os.path import dirname, join

import numpy as np

from ovos_vad_plugin_silero import SileroVADVerifier

SR = 16000
FIXTURE = join(dirname(__file__), "fixtures", "command.wav")


def loudest_segment(audio: np.ndarray, seconds: float) -> np.ndarray:
    """Return the `seconds`-long window of `audio` with the most energy."""
    n = int(seconds * SR)
    energy = np.convolve(audio.astype(np.float64) ** 2, np.ones(n), mode="valid")
    start = int(np.argmax(energy))
    return audio[start:start + n]


class TestSileroVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with wave.open(FIXTURE) as wav:
            assert wav.getframerate() == SR and wav.getsampwidth() == 2 and wav.getnchannels() == 1
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        cls.word = loudest_segment(audio, 0.5)  # about the length of a wake word
        cls.verifier = SileroVADVerifier(config={})

    def floor(self, seconds: float, dbfs: float | None = None) -> np.ndarray:
        """Digital silence, or white noise at `dbfs` like a real microphone floor."""
        n = int(seconds * SR)
        if dbfs is None:
            return np.zeros(n, dtype=np.int16)
        rng = np.random.default_rng(0)
        return (rng.standard_normal(n) * 32767 * 10 ** (dbfs / 20)).astype(np.int16)

    def buffer(self, lead_seconds: float, tail_seconds: float = 0.0, dbfs: float | None = None) -> bytes:
        return np.concatenate([self.floor(lead_seconds, dbfs), self.word,
                               self.floor(tail_seconds, dbfs)]).tobytes()

    def test_wake_word_after_silence_is_accepted(self):
        # 0.46 s and 1.42 s of lead make the 0.96 s and 1.92 s windows.
        for dbfs in (None, -60):
            for lead in (0.0, 0.46, 1.0, 1.42):
                with self.subTest(lead_seconds=lead, dbfs=dbfs):
                    self.assertTrue(self.verifier.verify(self.buffer(lead, dbfs=dbfs)))

    def test_late_detection_is_accepted(self):
        # The engine can fire up to about a second after the word ends.
        for tail in (0.76, 1.1, 1.42):
            with self.subTest(tail_seconds=tail):
                self.assertTrue(self.verifier.verify(self.buffer(1.42 - tail, tail, dbfs=-60)))

    def test_silence_is_rejected(self):
        for seconds in (0.96, 1.92):
            with self.subTest(seconds=seconds):
                silence = np.zeros(int(seconds * SR), dtype=np.int16)
                self.assertFalse(self.verifier.verify(silence.tobytes()))

    def test_low_noise_is_rejected(self):
        rng = np.random.default_rng(0)
        for seconds in (0.96, 1.92):
            with self.subTest(seconds=seconds):
                noise = (rng.standard_normal(int(seconds * SR)) * 200).astype(np.int16)
                self.assertFalse(self.verifier.verify(noise.tobytes()))

    def test_result_does_not_depend_on_the_previous_call(self):
        speech = self.buffer(1.42)
        silence = np.zeros(int(1.92 * SR), dtype=np.int16).tobytes()
        self.assertFalse(self.verifier.verify(silence))
        self.assertTrue(self.verifier.verify(speech))
        self.assertFalse(self.verifier.verify(silence))
        self.assertTrue(self.verifier.verify(speech))

    def test_audio_shorter_than_one_frame_is_rejected(self):
        self.assertFalse(self.verifier.verify(self.word[:100].tobytes()))


if __name__ == "__main__":
    unittest.main()
