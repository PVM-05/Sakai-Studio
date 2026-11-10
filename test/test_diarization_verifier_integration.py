"""Kiểm thử tích hợp toàn diện cho bộ kiểm chứng phân đoạn ExternalDiarizationVerifier."""

import os
import sys
from unittest.mock import MagicMock, patch

# Đảm bảo không bị gián đoạn nếu môi trường chạy chưa cài đặt pyannote.audio
if "pyannote" not in sys.modules:
    sys.modules["pyannote"] = MagicMock()
if "pyannote.audio" not in sys.modules:
    sys.modules["pyannote.audio"] = MagicMock()

from src.ai_engines.speaker_diarization import ExternalDiarizationVerifier
from src.core.config import app_config


class MockTurn:
    """Đoạn hội thoại giả lập chứa mốc thời gian bắt đầu và kết thúc."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class MockAnnotation:
    """Đối tượng kết quả phân đoạn giả lập trả về danh sách lượt nói."""

    def __init__(self, turns=None):
        if turns is None:
            self.turns = [
                (MockTurn(0.0, 5.0), None, "SPEAKER_00"),
                (MockTurn(5.0, 10.0), None, "SPEAKER_01"),
            ]
        else:
            self.turns = turns

    def itertracks(self, yield_label: bool = True):
        return self.turns


class MockSubtitleBlock:
    """Khối phụ đề giả lập dùng để kiểm chứng việc gán lại nhãn người nói."""

    def __init__(
        self,
        start_time: str = "00:00:00,000",
        end_time: str = "00:00:05,000",
        speaker: str = "spk_0",
        speaker_id: str = "spk_0",
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.speaker = speaker
        self.speaker_id = speaker_id
        self.speaker_embedding = None
        self.review_needed = False
        self.confidence_level = "high"


def test_verifier_pipeline_none_fallback():
    """Khi nạp pipeline thất bại trả về None, hệ thống an toàn hoàn trả dữ liệu CAM++."""
    app_config.mmoTTSDeepDiarization.value = True
    campp_data = {"spk_0": {"name": "Gốc"}}

    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=None):
        res = ExternalDiarizationVerifier.verify_or_fallback(
            audio_path=__file__,
            campp_speakers=campp_data,
        )
        assert res == campp_data


def test_verifier_exception_network_fallback():
    """Khi xảy ra ngoại lệ như mất mạng hoặc lỗi bộ nhớ GPU, hệ thống an toàn hoàn trả CAM++."""
    app_config.mmoTTSDeepDiarization.value = True
    campp_data = {"spk_0": {"name": "Gốc"}}

    with patch("pyannote.audio.Pipeline.from_pretrained", side_effect=RuntimeError("Lỗi kết nối mạng")):
        res = ExternalDiarizationVerifier.verify_or_fallback(
            audio_path=__file__,
            campp_speakers=campp_data,
        )
        assert res == campp_data


def test_verifier_successful_mock_execution():
    """Khi pipeline phân đoạn thành công, hệ thống ánh xạ nhãn người nói và cập nhật danh sách phụ đề."""
    app_config.mmoTTSDeepDiarization.value = True
    campp_data = {
        "spk_0": {"name": "Người nói 1", "speaker_id": "spk_0"},
        "spk_1": {"name": "Người nói 2", "speaker_id": "spk_1"},
    }

    blk1 = MockSubtitleBlock(
        start_time="00:00:00,000",
        end_time="00:00:05,000",
        speaker="spk_0",
        speaker_id="spk_0",
    )
    blk2 = MockSubtitleBlock(
        start_time="00:00:05,000",
        end_time="00:00:10,000",
        speaker="spk_1",
        speaker_id="spk_1",
    )
    blocks = [blk1, blk2]

    mock_pipe = MagicMock()
    mock_pipe.return_value = MockAnnotation()

    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipe):
        res = ExternalDiarizationVerifier.verify_or_fallback(
            audio_path=__file__,
            campp_speakers=campp_data,
            exact_speakers=2,
            subtitle_blocks=blocks,
        )

        assert res == campp_data
        mock_pipe.assert_called_once_with(__file__, num_speakers=2)
        assert blk1.speaker in ["spk_0", "spk_1"]
        assert blk2.speaker in ["spk_0", "spk_1"]


def test_verifier_quality_gate_rejection():
    """Khi cổng kiểm soát chất lượng từ chối do chênh lệch người nói, hệ thống giữ nguyên nhãn gốc CAM++."""
    app_config.mmoTTSDeepDiarization.value = True
    campp_data = {"spk_0": {"name": "Người nói duy nhất", "speaker_id": "spk_0"}}

    blk = MockSubtitleBlock(
        start_time="00:00:00,000",
        end_time="00:00:02,000",
        speaker="spk_0",
        speaker_id="spk_0",
    )
    blocks = [blk]

    # Giả lập pyannote nhận diện ra 4 người nói trong khi CAM++ chỉ có 1 người (độ lệch > 2)
    annotation_4_spk = MockAnnotation(
        turns=[
            (MockTurn(0.0, 2.0), None, "SPEAKER_00"),
            (MockTurn(2.0, 4.0), None, "SPEAKER_01"),
            (MockTurn(4.0, 6.0), None, "SPEAKER_02"),
            (MockTurn(6.0, 8.0), None, "SPEAKER_03"),
        ]
    )

    mock_pipe = MagicMock()
    mock_pipe.return_value = annotation_4_spk

    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipe):
        res = ExternalDiarizationVerifier.verify_or_fallback(
            audio_path=__file__,
            campp_speakers=campp_data,
            exact_speakers=None,
            subtitle_blocks=blocks,
        )

        # Cổng chất lượng từ chối và hoàn trả dữ liệu CAM++ ban đầu
        assert res == campp_data
        # Nhãn của phụ đề không bị thay đổi
        assert blk.speaker == "spk_0"


def test_verifier_disabled_passthrough():
    """Khi tắt tính năng phân đoạn chuyên sâu, hệ thống trả về kết quả CAM++ lập tức mà không tải mô hình."""
    app_config.mmoTTSDeepDiarization.value = False
    campp_data = {"spk_0": {"name": "Gốc"}}

    with patch("pyannote.audio.Pipeline.from_pretrained") as mock_from_pretrained:
        res = ExternalDiarizationVerifier.verify_or_fallback(
            audio_path=__file__,
            campp_speakers=campp_data,
        )

        assert res == campp_data
        mock_from_pretrained.assert_not_called()
