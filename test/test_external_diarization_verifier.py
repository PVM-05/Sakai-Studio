import sys
from unittest.mock import MagicMock, patch
import pytest

from src.ai_engines.speaker_diarization import ExternalDiarizationVerifier
from src.core.config import app_config


class DummyBlock:
    def __init__(self, start_time="00:00:01,000", end_time="00:00:04,000", speaker="spk_0", speaker_id="spk_0"):
        self.start_time = start_time
        self.end_time = end_time
        self.speaker = speaker
        self.speaker_id = speaker_id
        self.speaker_embedding = None
        self.review_needed = False
        self.confidence_level = "high"


class DummyTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def test_verifier_fallback_when_disabled():
    app_config.mmoTTSDeepDiarization.value = False
    sample_campp = {"spk_0": {"name": "Nhan vat 1"}}
    res = ExternalDiarizationVerifier.verify_or_fallback(
        audio_path="dummy.wav",
        campp_speakers=sample_campp,
        exact_speakers=2,
    )
    assert res == sample_campp


def test_verifier_is_available():
    available = ExternalDiarizationVerifier.is_available()
    assert isinstance(available, bool)


def test_verifier_fallback_when_audio_not_exists():
    app_config.mmoTTSDeepDiarization.value = True
    sample_campp = {"spk_0": {"name": "Nhan vat 1"}}
    res = ExternalDiarizationVerifier.verify_or_fallback(
        audio_path="non_existent_audio_file_123456.wav",
        campp_speakers=sample_campp,
        exact_speakers=2,
    )
    assert res == sample_campp


def test_verifier_fallback_when_not_available(tmp_path):
    app_config.mmoTTSDeepDiarization.value = True
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_bytes(b"dummy audio content")
    sample_campp = {"spk_0": {"name": "Nhan vat 1"}}

    with patch.object(ExternalDiarizationVerifier, "is_available", return_value=False):
        res = ExternalDiarizationVerifier.verify_or_fallback(
            audio_path=str(dummy_audio),
            campp_speakers=sample_campp,
        )
        assert res == sample_campp


def test_verifier_fallback_when_pipeline_none(tmp_path):
    app_config.mmoTTSDeepDiarization.value = True
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_bytes(b"dummy audio content")
    sample_campp = {"spk_0": {"name": "Nhan vat 1"}}

    mock_pyannote = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = None
    mock_pyannote.Pipeline = mock_pipeline_cls

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_pyannote}):
        with patch.object(ExternalDiarizationVerifier, "is_available", return_value=True):
            res = ExternalDiarizationVerifier.verify_or_fallback(
                audio_path=str(dummy_audio),
                campp_speakers=sample_campp,
            )
            assert res == sample_campp


def test_verifier_fallback_on_exception_and_vram_cleanup(tmp_path):
    app_config.mmoTTSDeepDiarization.value = True
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_bytes(b"dummy audio content")
    sample_campp = {"spk_0": {"name": "Nhan vat 1"}}

    mock_pyannote = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("CUDA OOM error during model load")
    mock_pyannote.Pipeline = mock_pipeline_cls

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_pyannote}):
        with patch.object(ExternalDiarizationVerifier, "is_available", return_value=True):
            res = ExternalDiarizationVerifier.verify_or_fallback(
                audio_path=str(dummy_audio),
                campp_speakers=sample_campp,
            )
            assert res == sample_campp


def test_verifier_pipeline_success_with_hungarian_mapping(tmp_path):
    app_config.mmoTTSDeepDiarization.value = True
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_bytes(b"dummy audio content")

    blk1 = DummyBlock(start_time="00:00:00,000", end_time="00:00:05,000", speaker="spk_0", speaker_id="spk_0")
    blk2 = DummyBlock(start_time="00:00:06,000", end_time="00:00:10,000", speaker="spk_1", speaker_id="spk_1")
    blocks = [blk1, blk2]

    class MockProfile:
        def __init__(self, speaker_id):
            self.speaker_id = speaker_id
            self.embedding = None

    sample_campp = {
        "spk_0": MockProfile("spk_0"),
        "spk_1": MockProfile("spk_1"),
    }

    mock_diarization = MagicMock()
    # It returns (turn, _, speaker)
    mock_diarization.itertracks.return_value = [
        (DummyTurn(0.0, 5.0), None, "SPEAKER_00"),
        (DummyTurn(6.0, 10.0), None, "SPEAKER_01"),
    ]

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_diarization

    mock_pyannote = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance
    mock_pyannote.Pipeline = mock_pipeline_cls

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_pyannote}):
        with patch.object(ExternalDiarizationVerifier, "is_available", return_value=True):
            res = ExternalDiarizationVerifier.verify_or_fallback(
                audio_path=str(dummy_audio),
                campp_speakers=sample_campp,
                exact_speakers=2,
                subtitle_blocks=blocks,
            )
            assert res == sample_campp
            # Check pipeline call params
            mock_pipeline_instance.assert_called_once_with(str(dummy_audio), num_speakers=2)
            # Check speaker reassignment
            assert blk1.speaker in ["spk_0", "spk_1"]
            assert blk2.speaker in ["spk_0", "spk_1"]
