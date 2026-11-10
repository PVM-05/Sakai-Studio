"""Bài kiểm thử các trường cấu hình phân đoạn chuyên sâu trong QConfig."""

from src.core.config import app_config


def test_deep_diarization_config_defaults():
    """Kiểm tra sự tồn tại và giá trị mặc định của các trường cấu hình."""
    assert hasattr(app_config, "mmoTTSDeepDiarization")
    assert hasattr(app_config, "mmoTTSDeepDiarizationEngine")
    assert hasattr(app_config, "mmoTTSDeepDiarizationModelDir")

    # Kiểm tra giá trị mặc định
    assert app_config.mmoTTSDeepDiarization.value is False
    assert app_config.mmoTTSDeepDiarizationEngine.value == "pyannote"
    assert app_config.mmoTTSDeepDiarizationModelDir.value == ""


def test_deep_diarization_config_mutation():
    """Kiểm tra khả năng thay đổi giá trị và khôi phục cấu hình."""
    app_config.mmoTTSDeepDiarization.value = True
    assert app_config.mmoTTSDeepDiarization.value is True

    app_config.mmoTTSDeepDiarizationEngine.value = "nemotron"
    assert app_config.mmoTTSDeepDiarizationEngine.value == "nemotron"

    # Khôi phục trạng thái ban đầu
    app_config.mmoTTSDeepDiarization.value = False
    app_config.mmoTTSDeepDiarizationEngine.value = "pyannote"
