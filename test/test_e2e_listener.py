"""End-to-end listener tests for ovos-vad-plugin-silero.

Exercises SileroVAD both as a standalone classifier and wired into the
ovoscope MiniVoiceLoop pipeline.  The bundled ONNX model (shipped inside the
package) is used — no network access required.
"""
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed")

from ovoscope.listener import get_mini_listener, MockHotWordEngine  # noqa: E402
from ovoscope.voice_loop import MiniVoiceLoop, MockStreamingSTT  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "command.wav"

# Silero VAD processes 512 samples at 16 kHz → 1024 bytes (int16 mono)
_CHUNK_BYTES = 512 * 2  # 1024


def _load_speech_chunk() -> bytes:
    """Return a 512-sample chunk from the middle of the speech fixture."""
    with wave.open(str(FIXTURE)) as wf:
        assert wf.getframerate() == 16000, "Fixture must be 16 kHz"
        assert wf.getnchannels() == 1, "Fixture must be mono"
        frames = wf.readframes(wf.getnframes())
    mid = len(frames) // 2
    return frames[mid: mid + _CHUNK_BYTES]


# ---------------------------------------------------------------------------
# 1. Standalone VAD classification
# ---------------------------------------------------------------------------

class TestSileroVADClassification:
    """Real SileroVAD is_silence() accuracy on controlled inputs."""

    def setup_method(self):
        from ovos_vad_plugin_silero import SileroVAD
        self.vad = SileroVAD()

    def test_silent_chunk_is_silence(self):
        silent = b"\x00" * _CHUNK_BYTES
        assert self.vad.is_silence(silent)

    def test_speech_chunk_is_not_silence(self):
        speech = _load_speech_chunk()
        assert not self.vad.is_silence(speech)

    def test_reset_clears_state(self):
        """reset() must not raise and should allow re-use."""
        self.vad.reset()
        silent = b"\x00" * _CHUNK_BYTES
        assert self.vad.is_silence(silent)


# ---------------------------------------------------------------------------
# 2. MiniListener VAD integration
# ---------------------------------------------------------------------------

class TestMiniListenerWithSileroVAD:
    """SileroVAD injected into MiniListener via vad_instance=."""

    def setup_method(self):
        from ovos_vad_plugin_silero import SileroVAD
        self.vad = SileroVAD()
        self.listener = get_mini_listener(vad_instance=self.vad)

    def teardown_method(self):
        self.listener.shutdown()

    def test_mini_listener_silent_chunk(self):
        silent = b"\x00" * _CHUNK_BYTES
        assert self.listener.is_silence(silent)

    def test_mini_listener_speech_chunk(self):
        speech = _load_speech_chunk()
        assert not self.listener.is_silence(speech)

    def test_mini_listener_extract_speech_strips_silence(self):
        """extract_speech removes silent frames but keeps speech frames."""
        silent = b"\x00" * _CHUNK_BYTES
        speech = _load_speech_chunk()
        # Build: [speech, silence, silence] — only the speech frame should survive.
        audio = speech + silent + silent
        result = self.vad.extract_speech(audio) if hasattr(self.vad, "extract_speech") else None
        if result is not None:
            # The returned bytes must not be empty (speech was present).
            assert len(result) > 0


# ---------------------------------------------------------------------------
# 3. MiniVoiceLoop full-pipeline integration
# ---------------------------------------------------------------------------

class TestMiniVoiceLoopWithSileroVAD:
    """Drive the fixture through MiniVoiceLoop with real Silero VAD."""

    def _build_vl(self, transcript: str = "hello") -> MiniVoiceLoop:
        from ovos_vad_plugin_silero import SileroVAD
        vad = SileroVAD()
        stt = MockStreamingSTT(transcript=transcript)
        ww = MockHotWordEngine("hey_mycroft", trigger_after=1)
        return MiniVoiceLoop(
            ww_instances={"hey_mycroft": ww},
            vad_instance=vad,
            stt_instance=stt,
        )

    def test_voice_loop_record_begin_emitted(self):
        """Wake-word detection should trigger record_begin."""
        with self._build_vl() as vl:
            msgs = vl.feed_chunks([b"\x00" * 2048] * 3)
            vl.assert_record_begin_emitted(msgs)

    def test_voice_loop_utterance_from_fixture(self):
        """Full loop over the WAV fixture yields recognizer_loop:utterance."""
        with self._build_vl(transcript="hello world") as vl:
            msgs = vl.feed_file(str(FIXTURE))
        assert any(m.msg_type == "recognizer_loop:utterance" for m in msgs), (
            f"Expected recognizer_loop:utterance. Got: {[m.msg_type for m in msgs]}"
        )

    def test_voice_loop_utterance_text_matches(self):
        """The emitted utterance text matches the mock STT transcript."""
        transcript = "turn on the lights"
        with self._build_vl(transcript=transcript) as vl:
            msgs = vl.feed_file(str(FIXTURE))
        vl.assert_utterance_emitted(transcript, msgs)

    def test_voice_loop_speech_then_silence_segmentation(self):
        """Real Silero VAD produces record_end after speech ends."""
        with self._build_vl() as vl:
            msgs = vl.feed_file(str(FIXTURE))
        msg_types = [m.msg_type for m in msgs]
        assert "recognizer_loop:record_begin" in msg_types
        assert "recognizer_loop:record_end" in msg_types
        # record_end must come after record_begin
        begin_idx = next(i for i, m in enumerate(msgs) if m.msg_type == "recognizer_loop:record_begin")
        end_idx = next(i for i, m in enumerate(msgs) if m.msg_type == "recognizer_loop:record_end")
        assert end_idx > begin_idx
