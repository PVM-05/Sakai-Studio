# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tự động cho các cải tiến OmniVoice:
1. Lưu giữ và truyền đúng lời thoại mẫu giọng nhân vật (ref_text), không bị ép thành dấu chấm "."
2. Không cưỡng ép duration trong generate_single để giữ âm sắc tự nhiên 100%, tự động co giãn WSOLA khi cần
3. Thuật toán WSOLA tương quan chuẩn hóa không gây méo tiếng, không NaN, biên độ an toàn
4. Gán giọng gợi ý tự động đồng bộ cả hồ sơ nhân vật và danh sách câu thoại phụ đề
5. Nút nghe thử an toàn không gây lỗi libshiboken và phục hồi biểu tượng chính xác
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QApplication
from src.ai_engines.speaker_diarization import CharacterVoiceProfile
from src.ai_engines.translator import SubtitleBlock
from src.ai_engines.tts_engine import TTSEngine, VoiceClonePrompt


def _ensure_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if sys.argv else [""])
    return app


def test_character_voice_profile_ref_text():
    prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Người nói 1",
        source_audio_path="test.wav",
        ref_text="Xin chào các bạn",
    )
    assert prof.ref_text == "Xin chào các bạn"
    d = prof.to_dict()
    assert d["ref_text"] == "Xin chào các bạn"

    restored = CharacterVoiceProfile.from_dict(d)
    assert restored.ref_text == "Xin chào các bạn"


def test_set_reference_voice_does_not_pass_dot_when_empty():
    engine = TTSEngine(auto_extract_ref=False, mode="clone", load_asr=False)
    mock_model = MagicMock()
    engine.model = mock_model
    engine.load_model = MagicMock()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    try:
        engine.set_reference_voice(tmp_wav, ref_text=None)
        assert mock_model.create_voice_clone_prompt.called
        call_kwargs = mock_model.create_voice_clone_prompt.call_args[1]
        assert call_kwargs["ref_text"] is None or call_kwargs["ref_text"] != "."
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def test_get_or_create_prompt_does_not_pass_dot():
    engine = TTSEngine(auto_extract_ref=False, mode="clone", load_asr=False)
    mock_model = MagicMock()
    engine.model = mock_model
    engine.load_model = MagicMock()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    try:
        engine._get_or_create_prompt_for_audio(tmp_wav, cache_key="test_key", ref_text=None)
        assert mock_model.create_voice_clone_prompt.called
        call_kwargs = mock_model.create_voice_clone_prompt.call_args[1]
        assert call_kwargs["ref_text"] is None or call_kwargs["ref_text"] != "."
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def test_generate_single_natural_generation_without_forced_duration():
    engine = TTSEngine(auto_extract_ref=False, mode="design", instruct="male, young adult")
    mock_model = MagicMock()
    sr = 24000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
    mock_audio = 0.4 * np.sin(2 * np.pi * 300 * t)
    mock_model.generate.return_value = [mock_audio]

    engine.model = mock_model
    engine.load_model = MagicMock()

    # Yêu cầu target_duration=1.2s -> audio dài 1.5s sẽ được WSOLA co nén về khớp target_duration
    res = engine.generate_single("Chào buổi sáng", target_duration=1.2)
    assert len(res) > 0
    # generate không bị ép tham số duration
    call_kwargs = mock_model.generate.call_args[1]
    assert "duration" not in call_kwargs
    # Kết quả sau WSOLA co ngắn lại gần 1.2s
    out_dur = len(res) / float(sr)
    assert 1.15 <= out_dur <= 1.25


def test_safe_time_stretch_normalized_correlation():
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    audio = (0.4 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    # Co nén 1.2x
    faster = TTSEngine._safe_time_stretch(audio, 1.20, sample_rate=sr)
    assert not np.isnan(faster).any()
    assert not np.isinf(faster).any()
    assert len(faster) < len(audio)
    assert np.max(np.abs(faster)) <= 0.85

    # Giãn dài 0.85x
    slower = TTSEngine._safe_time_stretch(audio, 0.85, sample_rate=sr)
    assert not np.isnan(slower).any()
    assert not np.isinf(slower).any()
    assert len(slower) > len(audio)
    assert np.max(np.abs(slower)) <= 0.85


def test_apply_recommended_voice_syncs_profiles_and_blocks():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG

    interface = OmniVoiceInterface()
    preset_choice = list(VIETNAMESE_VOICE_CATALOG.keys())[0]

    prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Người nói 1",
        method="clone",
        source_audio_path="old_ref.wav",
    )
    interface.speaker_profiles = {"Người nói 1": prof}
    blk = SubtitleBlock(
        index=1,
        start_time=0.0,
        end_time=2.0,
        text="Xin chào",
        translated_text="Xin chào",
        speaker="Người nói 1",
        voice_ref_path="old_ref.wav",
    )
    interface.subtitle_blocks = [blk]

    interface._apply_recommended_voice("Người nói 1", preset_choice)

    assert prof.method == "preset"
    assert prof.preset_name == preset_choice
    assert prof.source_audio_path == ""
    # Các khối phụ đề của nhân vật cũng phải được cập nhật
    assert blk.voice_ref_path == ""
    assert blk.voice_instruct == VIETNAMESE_VOICE_CATALOG[preset_choice]["instruct"]
    assert blk.is_speaker_locked is True


def test_safe_audition_button_reset():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from qfluentwidgets import PushButton, FluentIcon

    interface = OmniVoiceInterface()
    btn = PushButton("Nghe")
    btn.setProperty("default_text", "Nghe")
    btn.setProperty("default_icon", FluentIcon.PLAY)

    interface._active_audition_button = btn
    interface._reset_audition_button()

    assert interface._active_audition_button is None
    assert btn.text() == "Nghe"
    assert btn.isEnabled() is True


def test_audition_queue_empty_graceful():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    interface = OmniVoiceInterface()
    interface._audition_queue = []
    interface._audition_all_active = True
    interface._play_next_audition_in_queue()

    assert interface._audition_all_active is False
    assert interface._current_audition_speaker is None


def test_can_switch_to_step_logic():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    interface = OmniVoiceInterface()
    interface.subtitle_blocks = []
    interface.current_video_path = ""
    interface.speaker_profiles = {}

    # Bước 0 (Nạp dữ liệu) luôn cho phép
    assert interface._can_switch_to_step(0) is True

    # Bước 1 (Quản lý nhân vật) bị chặn nếu hoàn toàn không có dữ liệu
    assert interface._can_switch_to_step(1) is False

    # Khi có video hợp lệ, Bước 1 được mở khóa
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_video = f.name
    try:
        interface.current_video_path = tmp_video
        assert interface._can_switch_to_step(1) is True
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)

    interface.current_video_path = ""

    # Khi có hồ sơ nhân vật, Bước 1 được mở khóa
    interface.speaker_profiles = {"Người nói 1": {}}
    assert interface._can_switch_to_step(1) is True
    interface.speaker_profiles = {}

    # Bước 3 (Phòng thu / Studio) luôn cho phép chuyển sang
    assert interface._can_switch_to_step(3) is True

    # Bước 2 và Bước 4 bắt buộc phải có phụ đề
    assert interface._can_switch_to_step(2) is False
    assert interface._can_switch_to_step(4) is False

    interface.subtitle_blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.0, text="A")
    ]
    assert interface._can_switch_to_step(1) is True
    assert interface._can_switch_to_step(2) is True
    assert interface._can_switch_to_step(4) is True


def test_step_buttons_disabled_during_running_task():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    interface = OmniVoiceInterface()
    assert len(interface.step_buttons) == 5

    # Khi bắt đầu tác vụ nền: tất cả các nút bước phải bị vô hiệu hóa
    interface._set_task_running_state(True)
    for btn in interface.step_buttons:
        assert btn.isEnabled() is False

    # Khi tác vụ nền kết thúc: tất cả các nút bước phải được kích hoạt lại
    interface._set_task_running_state(False)
    for btn in interface.step_buttons:
        assert btn.isEnabled() is True


def test_generate_single_passes_ref_text_to_audio_prompt():
    engine = TTSEngine(auto_extract_ref=False, mode="clone", load_asr=False)
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.zeros(sr, dtype=np.float32)]
    engine.model = mock_model

    engine._get_or_create_prompt_for_audio = MagicMock(return_value=MagicMock())

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    try:
        engine.generate_single(
            "Câu cần đọc",
            override_mode="clone",
            override_ref_audio=tmp_wav,
            override_ref_text="Lời thoại mẫu chuẩn",
        )
        assert engine._get_or_create_prompt_for_audio.called
        call_kwargs = engine._get_or_create_prompt_for_audio.call_args[1]
        assert call_kwargs.get("ref_text") == "Lời thoại mẫu chuẩn"
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def test_no_parentheses_in_key_ui_text():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    interface = OmniVoiceInterface()

    # Nút thu gọn thiết lập giọng mặc định
    toggle_text = interface.btn_toggle_default_voice_settings.text()
    assert "(" not in toggle_text and ")" not in toggle_text
    assert toggle_text == "Tùy Chọn Giọng Mặc Định Dự Phòng"

    # Mục lựa chọn phần cứng
    hw_items = [
        interface.combo_hardware_selector.itemText(i)
        for i in range(interface.combo_hardware_selector.count())
    ]
    for item in hw_items:
        assert "(" not in item and ")" not in item


def test_vietnamese_number_normalization():
    from src.ai_engines.tts_engine import normalize_vietnamese_numbers_and_text

    # Số nguyên đơn giản và số năm
    assert normalize_vietnamese_numbers_and_text("Năm 2024") == "Năm hai nghìn không trăm hai mươi tư"
    assert normalize_vietnamese_numbers_and_text("Có 15 người") == "Có mười lăm người"
    assert normalize_vietnamese_numbers_and_text("Số 0") == "Số không"
    assert normalize_vietnamese_numbers_and_text("Tầng 21") == "Tầng hai mươi mốt"

    # Số thập phân và phần trăm
    assert normalize_vietnamese_numbers_and_text("Tăng 3.5%") == "Tăng ba phẩy năm phần trăm"
    assert normalize_vietnamese_numbers_and_text("Đạt 100%") == "Đạt một trăm phần trăm"

    # Giữ nguyên thẻ biểu cảm
    assert normalize_vietnamese_numbers_and_text("Chào bạn [laughter] năm 2025") == "Chào bạn [laughter] năm hai nghìn không trăm hai mươi lăm"


def test_synthesize_blocks_never_creates_consistent_prompt_for_design():
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock, patch

    engine = TTSEngine(auto_extract_ref=False, mode="design")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr, dtype=np.float32) * 0.3]
    engine.model = mock_model

    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=2.0, text="Câu thứ nhất", speaker="Người nói 1"),
        SubtitleBlock(index=2, start_time=2.5, end_time=4.5, text="Câu thứ hai", speaker="Người nói 1"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        with patch("src.ai_engines.tts_engine.is_audio_quality_acceptable_for_voice_lock", return_value=(True, "OK")):
            report = engine.synthesize_blocks(blocks, total_duration=5.0, output_path=tmp_out)
            # Tuyệt đối không sinh khóa consistent_Người nói 1 để ép các câu sau phải clone câu 1
            assert "consistent_Người nói 1" not in engine._speaker_prompt_cache
            assert report.status in ("success", "partial")
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_cache_key_includes_quality_parameters():
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(auto_extract_ref=False, resume_cache_dir=os.path.abspath("tmp_test_cache"))
    p1 = engine._compute_block_cache_path(0, "Chào bạn", "spk_1", None, "instruct_1", 2.0)
    assert p1 is not None

    # Khi đổi guidance_scale, đường dẫn cache phải đổi khác hoàn toàn
    engine.guidance_scale = 3.5
    p2 = engine._compute_block_cache_path(0, "Chào bạn", "spk_1", None, "instruct_1", 2.0)
    assert p1 != p2

    # Khi đổi num_step, đường dẫn cache phải đổi khác hoàn toàn
    engine.num_step = 16
    p3 = engine._compute_block_cache_path(0, "Chào bạn", "spk_1", None, "instruct_1", 2.0)
    assert p2 != p3


def test_synthesize_blocks_normalizes_master_peak():
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    engine = TTSEngine(auto_extract_ref=False, mode="design")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    # Giả lập âm thanh sinh ra có biên độ cực lớn 1.5 (vượt ngưỡng clipping)
    mock_model.generate.return_value = [np.ones(sr, dtype=np.float32) * 1.5]
    engine.model = mock_model

    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.0, text="Kiểm tra âm lượng", speaker="Người nói 1"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        report = engine.synthesize_blocks(blocks, total_duration=2.0, output_path=tmp_out)
        import soundfile as sf
        data, _ = sf.read(tmp_out)
        peak = float(np.max(np.abs(data)))
        # Biên độ đỉnh tối đa không bao giờ được vượt quá 0.891 (-1.0 dBFS)
        assert peak <= 0.892
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_single_unified_voice_mode_ui():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from src.core.config import config
    config.set(config.mmoTTSMultiSpeakerMode, "single")

    interface = OmniVoiceInterface()
    assert hasattr(interface, "chk_single_unified_voice")
    assert hasattr(interface, "combo_single_unified_preset")
    assert hasattr(interface, "btn_preview_single_unified")

    # Mặc định bật chế độ một giọng đọc đồng nhất để đơn giản hóa thao tác
    assert interface.chk_single_unified_voice.isChecked() is True
    assert interface.combo_single_unified_preset.count() == len(VIETNAMESE_VOICE_CATALOG)

    # Khi bật chế độ 1 giọng, cho phép chuyển thẳng sang Bước 3 (Biên tập phụ đề) nếu có phụ đề
    interface.subtitle_blocks = [SubtitleBlock(index=1, start_time=0.0, end_time=1.0, text="Chào bạn")]
    assert interface._can_switch_to_step(2) is True


def test_casting_table_pure_ai_presets_and_audition():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG

    interface = OmniVoiceInterface()
    prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Người nói 1",
        method="preset",
        preset_name=list(VIETNAMESE_VOICE_CATALOG.keys())[0],
    )
    interface.speaker_profiles = {"Người nói 1": prof}
    interface._update_casting_table()

    # Bảng phân vai phải có 6 cột chuẩn mực, loại bỏ cột mẫu giọng tệp ngoài
    assert interface.casting_table.columnCount() == 6
    header_labels = [interface.casting_table.horizontalHeaderItem(c).text() for c in range(6)]
    assert "Mẫu Giọng Nguồn" not in header_labels
    assert "Giọng Đọc Trí Tuệ Nhân Tạo" in header_labels
    assert header_labels == [
        "Mã Nhân Vật",
        "Tên Nhân Vật",
        "Số Câu Thoại",
        "Giọng Đọc Trí Tuệ Nhân Tạo",
        "Trạng Thái",
        "Thao Tác",
    ]








def test_single_unified_voice_consistency_across_blocks():
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    preset_name = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
    expected_instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

    engine = TTSEngine(
        auto_extract_ref=False,
        mode="design",
        multi_speaker_mode="single",
        instruct=expected_instruct,
        preset_name=preset_name,
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr * 4, dtype=np.float32) * 0.5]
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model

    # Tạo 2 khối phụ đề với nhãn người nói khác nhau
    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Câu thoại thứ nhất", speaker="Người nói 1", voice_instruct=expected_instruct),
        SubtitleBlock(index=2, start_time=2.0, end_time=3.5, text="Câu thoại thứ hai", speaker="Người nói 2", voice_instruct=expected_instruct),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        recorded_calls = []
        original_gen = engine.generate_single

        def mock_generate_single(*args, **kwargs):
            recorded_calls.append(kwargs)
            return original_gen(*args, **kwargs)

        with patch.object(engine, "generate_single", side_effect=mock_generate_single):
            report = engine.synthesize_blocks(blocks, total_duration=4.0, output_path=tmp_out)

        assert report.status == "success"
        assert len(recorded_calls) == 2
        for call_kw in recorded_calls:
            # Xác nhận chạy chế độ mỏ neo clone đồng nhất khóa trọn vẹn âm sắc
            assert call_kw.get("override_mode") == "clone"
            assert call_kw.get("override_prompt") is mock_prompt
            assert "anchor_" in os.path.basename(call_kw.get("override_ref_audio", ""))

        # Cả 2 khối phụ đề phải dùng chung đúng cùng một mẫu prompt mỏ neo
        assert recorded_calls[0]["override_prompt"] is recorded_calls[1]["override_prompt"]
        assert recorded_calls[0]["override_ref_audio"] == recorded_calls[1]["override_ref_audio"]
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_no_destructive_audio_truncation_when_duration_exceeds_slot():
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock
    import soundfile as sf

    engine = TTSEngine(auto_extract_ref=False, mode="design", multi_speaker_mode="single")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    # Giả lập sinh ra âm thanh 2.0s nhưng khoảng cách đến câu tiếp chỉ là 0.8s
    mock_model.generate.return_value = [np.ones(int(2.0 * sr), dtype=np.float32) * 0.4]
    mock_model.create_voice_clone_prompt.return_value = MagicMock()
    engine.model = mock_model

    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=0.8, text="Câu nói dài", speaker="Người nói 1"),
        SubtitleBlock(index=2, start_time=1.0, end_time=1.8, text="Câu tiếp theo", speaker="Người nói 1"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        report = engine.synthesize_blocks(blocks, total_duration=3.0, output_path=tmp_out)
        assert report.status == "success"
        assert report.stretched_count >= 1
        data, file_sr = sf.read(tmp_out)
        # Tổng độ dài file phải chứa đầy đủ cả 2 câu thoại mà không bị cắt cụt về 0.8s
        assert len(data) >= int(2.0 * file_sr)
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_vietnamese_duration_estimation_safe_floor():
    try:
        from src.ai_engines.omnivoice.models.omnivoice import OmniVoice
        mock_model = OmniVoice.__new__(OmniVoice)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Bỏ qua kiểm thử mô hình OmniVoice khi môi trường không cài đặt PyTorch hoặc transformers.")

    mock_estimator = MagicMock()
    mock_estimator.estimate_duration.return_value = 50.0
    mock_model.duration_estimator = mock_estimator

    sample_vietnamese = "Đây là một câu kiểm tra thời lượng tiếng Việt tương đối dài để bảo đảm hệ thống không bao giờ bị cắt cụt các từ cuối cùng."
    tokens = mock_model._estimate_target_tokens(sample_vietnamese, None, None, speed=1.0)
    words = len(sample_vietnamese.split())
    assert tokens >= int(words * 6.0)


def test_preview_unified_voice_uses_first_srt_line_when_loaded():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu phụ đề thực tế số 1 từ tệp phim"),
        SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:06,000", text="Câu phụ đề thứ hai"),
    ]

    captured_args = {}
    def mock_launch_worker(task_type, **kwargs):
        captured_args["task_type"] = task_type
        captured_args.update(kwargs)

    ui._launch_worker = mock_launch_worker
    ui._preview_unified_voice()

    assert captured_args.get("task_type") == "vn_catalog_preview"
    # Phải đọc câu phụ đề đầu tiên trong tệp, không được đọc câu mẫu catalog cứng
    assert captured_args.get("text") == "Câu phụ đề thực tế số 1 từ tệp phim"


def test_audition_character_voice_uses_subtitle_line_when_no_exact_speaker_match():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Lời thoại trong phim của nhân vật"),
    ]
    ui.speaker_profiles = {
        "Nam - Miền Bắc Truyền Cảm": {
            "method": "preset",
            "preset_name": "Nam - Miền Bắc Truyền Cảm",
            "voice_instruct": "male, northern vietnamese, emotional",
        }
    }

    captured_args = {}
    def mock_launch_worker(task_type, **kwargs):
        captured_args["task_type"] = task_type
        captured_args.update(kwargs)

    ui._launch_worker = mock_launch_worker
    ui._audition_character_voice("Nam - Miền Bắc Truyền Cảm")

    assert captured_args.get("task_type") == "character_audition"
    # Phải đọc lời thoại thực tế trong tệp phụ đề, không được dùng câu Xin chào thử nghiệm
    assert captured_args.get("text") == "Lời thoại trong phim của nhân vật"


def test_synthesize_blocks_single_mode_fallback_to_design_when_anchor_fails():
    """Kiểm tra khi tạo mỏ neo thất bại (hoặc không đạt chuẩn chất lượng), hệ thống tự động lùi về chế độ thiết kế giọng trực tiếp từng câu (fallback design) để đảm bảo không bị gián đoạn tiến trình."""
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock, patch

    engine = TTSEngine(auto_extract_ref=False, mode="design", multi_speaker_mode="single", instruct="male, young adult")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    # Âm thanh ngắn dưới 2.5s sẽ bị kiểm tra chất lượng từ chối, gây fallback về design
    mock_model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.5]
    mock_model.create_voice_clone_prompt.return_value = MagicMock()
    engine.model = mock_model

    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Câu phụ đề một"),
        SubtitleBlock(index=2, start_time=2.0, end_time=3.5, text="Câu phụ đề hai"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        recorded_calls = []
        original_gen = engine.generate_single

        def mock_generate_single(*args, **kwargs):
            recorded_calls.append(kwargs)
            return original_gen(*args, **kwargs)

        with patch.object(engine, "generate_single", side_effect=mock_generate_single):
            report = engine.synthesize_blocks(blocks, total_duration=4.0, output_path=tmp_out)

        assert report.status == "success"
        assert len(recorded_calls) == 2
        for call_kw in recorded_calls:
            # Khi mỏ neo không thành công, hệ thống lùi về thiết kế giọng trực tiếp từng câu
            assert call_kw.get("override_mode") == "design"
            assert call_kw.get("override_prompt") is None
            assert call_kw.get("override_instruct") == "male, young adult"
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_unified_voice_srt_without_speaker_uses_design_preset():
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    preset_name = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
    instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
    engine = TTSEngine(
        auto_extract_ref=False,
        mode="design",
        multi_speaker_mode="single",
        instruct=instruct,
        preset_name=preset_name,
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = [np.ones(24000 * 4, dtype=np.float32) * 0.5]
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model

    # Phụ đề hoàn toàn không có nhãn người nói
    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Câu không có speaker một", speaker="", voice_instruct=instruct),
        SubtitleBlock(index=2, start_time=2.0, end_time=3.5, text="Câu không có speaker hai", speaker="", voice_instruct=instruct),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        recorded_calls = []
        original_gen = engine.generate_single

        def mock_generate_single(*args, **kwargs):
            recorded_calls.append(kwargs)
            return original_gen(*args, **kwargs)

        with patch.object(engine, "generate_single", side_effect=mock_generate_single):
            report = engine.synthesize_blocks(blocks, total_duration=4.0, output_path=tmp_out)

        assert report.status == "success"
        assert len(recorded_calls) == 2
        for call_kw in recorded_calls:
            assert call_kw.get("override_mode") == "clone"
            assert call_kw.get("override_prompt") is mock_prompt
            assert "anchor_" in os.path.basename(call_kw.get("override_ref_audio", ""))
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_unified_voice_overrides_stale_clone_profiles():
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from unittest.mock import MagicMock

    preset_name = "Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện"
    instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        dummy_sample = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        # Hồ sơ nhân vật cũ còn lưu method=clone và tệp âm thanh mẫu cũ
        engine = TTSEngine(
            auto_extract_ref=False,
            mode="design",
            multi_speaker_mode="single",
            instruct=instruct,
            preset_name=preset_name,
        )
        engine.load_model = MagicMock()
        mock_model = MagicMock()
        mock_model.generate.return_value = [np.ones(24000 * 4, dtype=np.float32) * 0.5]
        mock_prompt = MagicMock()
        mock_model.create_voice_clone_prompt.return_value = mock_prompt
        engine.model = mock_model

        old_prompt_mock = MagicMock()
        prof = CharacterVoiceProfile(
            speaker_id="spk_old",
            display_name="Nhân vật cũ",
            method="clone",
            source_audio_path=dummy_sample,
        )
        engine.speaker_profiles = {"Nhân vật cũ": prof}
        engine.current_prompt = old_prompt_mock  # Giả lập còn sót prompt clone cũ

        blocks = [
            SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Câu của nhân vật cũ", speaker="Nhân vật cũ", voice_instruct=instruct),
        ]

        recorded_calls = []
        original_gen = engine.generate_single

        def mock_generate_single(*args, **kwargs):
            recorded_calls.append(kwargs)
            return original_gen(*args, **kwargs)

        with patch.object(engine, "generate_single", side_effect=mock_generate_single):
            report = engine.synthesize_blocks(blocks, total_duration=2.0, output_path=tmp_out)

        assert report.status == "success"
        assert len(recorded_calls) == 1
        call_kw = recorded_calls[0]
        # Tuyệt đối không được sử dụng lại mẫu clone cũ của nhân vật cũ
        assert call_kw.get("override_mode") == "clone"
        assert call_kw.get("override_prompt") is mock_prompt
        assert call_kw.get("override_prompt") is not old_prompt_mock
        assert call_kw.get("override_ref_audio") != dummy_sample
        assert "anchor_" in os.path.basename(call_kw.get("override_ref_audio", ""))
    finally:
        for p in (dummy_sample, tmp_out):
            if os.path.exists(p):
                os.remove(p)


def test_unified_voice_switching_presets_invalidates_cache():
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG

    with tempfile.TemporaryDirectory() as cache_dir:
        ins_a = VIETNAMESE_VOICE_CATALOG["Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"]["instruct"]
        ins_b = VIETNAMESE_VOICE_CATALOG["Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện"]["instruct"]

        engine_a = TTSEngine(mode="design", multi_speaker_mode="single", instruct=ins_a, resume_cache_dir=cache_dir)
        engine_b = TTSEngine(mode="design", multi_speaker_mode="single", instruct=ins_b, resume_cache_dir=cache_dir)

        path_a = engine_a._compute_block_cache_path(0, "Nội dung câu nói", "unified", None, ins_a, None, method_or_mode="design")
        path_b = engine_b._compute_block_cache_path(0, "Nội dung câu nói", "unified", None, ins_b, None, method_or_mode="design")

        # Khóa cache của Preset A và Preset B phải hoàn toàn khác biệt
        assert path_a != path_b


def test_pre_export_check_is_read_only_and_unified_voice_syncs_profiles():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    ui = OmniVoiceInterface()
    ui.chk_single_unified_voice.setChecked(True)
    ui.combo_single_unified_preset.setCurrentIndex(1)
    chosen_preset = ui.combo_single_unified_preset.currentText().strip()
    expected_instruct = VIETNAMESE_VOICE_CATALOG[chosen_preset]["instruct"]

    # Giả lập phụ đề chưa được gán instruct hoặc còn sót ref_path
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu thử nghiệm", speaker="Chưa phân vai", voice_instruct="", voice_ref_path="old.wav"),
    ]

    # Giả lập hồ sơ nhân vật dạng đối tượng và dạng từ điển
    obj_prof = CharacterVoiceProfile(
        speaker_id="spk_obj",
        display_name="Nhân vật đối tượng",
        method="clone",
        source_audio_path="old_obj.wav",
        custom_audio_path="custom_obj.wav",
        voice_clone_prompt_path="prompt_obj.pt",
        ref_text="Mẫu cũ",
        status="review_needed",
    )
    dict_prof = {
        "speaker_id": "spk_dict",
        "display_name": "Nhân vật từ điển",
        "method": "clone",
        "source_audio_path": "old_dict.wav",
        "ref_audio_path": "old_dict.wav",
        "custom_audio_path": "custom_dict.wav",
        "voice_clone_prompt_path": "prompt_dict.pt",
        "ref_text": "Mẫu cũ",
        "status": "review_needed",
    }
    ui.speaker_profiles = {"spk_obj": obj_prof, "spk_dict": dict_prof}

    # 1. Hàm kiểm tra chất lượng trước khi xuất phải hoàn toàn ĐỌC, KHÔNG làm thay đổi dữ liệu
    issues = ui._run_pre_export_quality_check()
    errors = [i for i in issues if i["severity"] == "error"]
    assert len(errors) == 0
    # Phụ đề chưa bị thay đổi qua bước kiểm tra
    assert ui.subtitle_blocks[0].voice_instruct == ""
    assert ui.subtitle_blocks[0].voice_ref_path == "old.wav"
    assert ui.subtitle_blocks[0].speaker == "Chưa phân vai"

    # 2. Bước chọn chế độ một giọng đọc phải BẢO TOÀN toàn diện cả phụ đề và mọi kiểu hồ sơ nhân vật gốc
    ui._apply_unified_voice_to_all_subtitles()
    assert ui.subtitle_blocks[0].speaker == "Chưa phân vai"
    assert ui.subtitle_blocks[0].voice_ref_path == "old.wav"
    assert ui.subtitle_blocks[0].is_speaker_locked is False

    # Kiểm tra hồ sơ nhân vật dạng đối tượng không bị xóa đè
    assert obj_prof.method == "clone"
    assert obj_prof.source_audio_path == "old_obj.wav"
    assert getattr(obj_prof, "custom_audio_path", "") == "custom_obj.wav"
    assert getattr(obj_prof, "voice_clone_prompt_path", "") == "prompt_obj.pt"

    # Kiểm tra hồ sơ nhân vật dạng từ điển không bị xóa đè
    assert dict_prof["method"] == "clone"
    assert dict_prof["source_audio_path"] == "old_dict.wav"
    assert dict_prof["ref_audio_path"] == "old_dict.wav"
    assert dict_prof["custom_audio_path"] == "custom_dict.wav"

    # Thu thập tham số tổng hợp phải chuyển sang chế độ single và áp dụng instruct mỏ neo
    params = ui._collect_engine_params()
    assert params["multi_speaker_mode"] == "single"
    assert params["mode"] == "design"
    assert params["instruct"] == expected_instruct
    assert params["preset_name"] == chosen_preset
    assert dict_prof["ref_text"] == "Mẫu cũ"
    assert dict_prof["status"] == "review_needed"

    # 3. Tham số thu thập phải phản ánh đúng chế độ design, instruct và mỏ neo
    assert params["multi_speaker_mode"] == "single"
    assert params["mode"] == "design"
    assert params["instruct"] == expected_instruct
    assert params["ref_audio_path"] is None


def test_synthesize_blocks_model_generate_exact_kwargs_and_translated_text():
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    preset_name = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
    expected_instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

    engine = TTSEngine(
        auto_extract_ref=False,
        mode="design",
        multi_speaker_mode="single",
        instruct=expected_instruct,
        preset_name=preset_name,
        normalize_text=False,
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr * 4, dtype=np.float32) * 0.5]
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model

    # Khối 1: Có cả translated_text và text -> phải ưu tiên tuyệt đối translated_text
    # Khối 2: translated_text để trống -> dùng text gốc
    # Khối 3: translated_text chỉ toàn khoảng trắng -> lùi về text gốc
    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Original text one", translated_text="Lời dịch tiếng Việt một", speaker="Nhân vật A"),
        SubtitleBlock(index=2, start_time=2.0, end_time=3.5, text="Lời thoại gốc không có bản dịch", translated_text="", speaker=""),
        SubtitleBlock(index=3, start_time=4.0, end_time=5.5, text="Lời thoại với bản dịch khoảng trắng", translated_text="   ", speaker="Nhân vật B"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        report = engine.synthesize_blocks(blocks, total_duration=6.0, output_path=tmp_out)
        assert report.status == "success"

        # 4 lượt thử mỏ neo (4 seeds) + 3 lượt sinh khối câu thoại = 7 lượt gọi đến model.generate()
        assert mock_model.generate.call_count == 7
        calls = mock_model.generate.call_args_list

        # Lượt 0..3: Thử nghiệm tạo mẫu mỏ neo
        for c in calls[:4]:
            assert c.kwargs.get("instruct") == expected_instruct

        # Lượt 4: Phải đọc translated_text "Lời dịch tiếng Việt một" với prompt mỏ neo đồng nhất
        assert calls[4].kwargs.get("text") == "Lời dịch tiếng Việt một"
        assert calls[4].kwargs.get("voice_clone_prompt") is mock_prompt

        # Lượt 5: Phải đọc text gốc "Lời thoại gốc không có bản dịch" với prompt mỏ neo đồng nhất
        assert calls[5].kwargs.get("text") == "Lời thoại gốc không có bản dịch"
        assert calls[5].kwargs.get("voice_clone_prompt") is mock_prompt

        # Lượt 6: Phải lùi về text gốc "Lời thoại với bản dịch khoảng trắng" với prompt mỏ neo đồng nhất
        assert calls[6].kwargs.get("text") == "Lời thoại với bản dịch khoảng trắng"
        assert calls[6].kwargs.get("voice_clone_prompt") is mock_prompt
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_synthesize_blocks_nameless_and_none_speaker_exact_preset_kwargs():
    """Kiểm tra khối phụ đề không có tên người nói (speaker=None hoặc speaker='') nhận đúng prompt mỏ neo của preset vào model.generate."""
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    preset_name = "Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện"
    expected_instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

    engine = TTSEngine(
        auto_extract_ref=False,
        mode="design",
        multi_speaker_mode="single",
        instruct=expected_instruct,
        preset_name=preset_name,
        normalize_text=False,
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr * 4, dtype=np.float32) * 0.3]
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model

    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=2.0, text="Câu một không gán người nói", speaker=None),
        SubtitleBlock(index=2, start_time=2.5, end_time=4.5, text="Câu hai có chuỗi người nói rỗng", speaker=""),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        report = engine.synthesize_blocks(blocks, total_duration=5.0, output_path=tmp_out)
        assert report.status == "success"
        # 4 lượt thử mỏ neo + 2 lượt sinh câu thoại = 6 lượt gọi
        assert mock_model.generate.call_count == 6

        for call in mock_model.generate.call_args_list[4:]:
            assert call.kwargs.get("voice_clone_prompt") is mock_prompt
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_synthesize_blocks_ignores_old_clone_profiles_in_single_unified_mode():
    """Kiểm tra chế độ một giọng đồng nhất hoàn toàn phớt lờ các hồ sơ clone cũ và không gọi tạo prompt từ tệp mẫu cũ."""
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    preset_name = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
    expected_instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

    engine = TTSEngine(
        auto_extract_ref=False,
        mode="design",
        multi_speaker_mode="single",
        instruct=expected_instruct,
        preset_name=preset_name,
        normalize_text=False,
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr * 4, dtype=np.float32) * 0.4]
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model
    engine._get_or_create_prompt_for_audio = MagicMock()

    # Cung cấp hồ sơ clone cũ trong speaker_profiles
    engine.speaker_profiles = {
        "Nhân vật A": {
            "method": "clone",
            "source_audio_path": "old_sample_a.wav",
            "ref_audio_path": "old_sample_a.wav",
            "voice_clone_prompt_path": "old_prompt_a.pt",
        }
    }

    blocks = [
        SubtitleBlock(
            index=1,
            start_time=0.0,
            end_time=2.0,
            text="Lời thoại nhân vật cũ",
            speaker="Nhân vật A",
            voice_ref_path="old_sample_a.wav",
        )
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        report = engine.synthesize_blocks(blocks, total_duration=3.0, output_path=tmp_out)
        assert report.status == "success"

        # Kiểm tra _get_or_create_prompt_for_audio TUYỆT ĐỐI không được gọi đến tệp mẫu cũ
        assert engine._get_or_create_prompt_for_audio.call_count == 0

        # Kiểm tra khối câu thoại được sinh bằng prompt mỏ neo đồng nhất, không dùng clone tệp cũ
        # 4 lượt thử mỏ neo + 1 lượt sinh câu thoại = 5 lượt gọi
        assert mock_model.generate.call_count == 5
        block_kwargs = mock_model.generate.call_args_list[4].kwargs
        assert block_kwargs.get("voice_clone_prompt") is mock_prompt
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_synthesize_blocks_cache_invalidation_when_switching_presets():
    """Kiểm tra khi chuyển từ Preset A sang Preset B trong cùng thư mục cache, pipeline không dùng lại âm thanh của Preset A."""
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as cache_dir:
        preset_a = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
        preset_b = "Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện"
        ins_a = VIETNAMESE_VOICE_CATALOG[preset_a]["instruct"]
        ins_b = VIETNAMESE_VOICE_CATALOG[preset_b]["instruct"]

        blocks = [
            SubtitleBlock(index=1, start_time=0.0, end_time=2.0, text="Câu kiểm tra bộ nhớ đệm", speaker="Người nói 1")
        ]

        # 1. Chạy với Preset A
        engine_a = TTSEngine(
            auto_extract_ref=False,
            mode="design",
            multi_speaker_mode="single",
            instruct=ins_a,
            preset_name=preset_a,
            resume_cache_dir=cache_dir,
            normalize_text=False,
        )
        engine_a.load_model = MagicMock()
        mock_model_a = MagicMock()
        sr = 24000
        mock_model_a.generate.return_value = [np.ones(sr * 4, dtype=np.float32) * 0.1]
        mock_prompt_a = MagicMock()
        mock_model_a.create_voice_clone_prompt.return_value = mock_prompt_a
        engine_a.model = mock_model_a

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_a = f.name

        try:
            rep_a = engine_a.synthesize_blocks(blocks, total_duration=3.0, output_path=out_a)
            assert rep_a.status == "success"
            # 4 lượt thử mỏ neo A + 1 lượt sinh block = 5
            assert mock_model_a.generate.call_count == 5
        finally:
            if os.path.exists(out_a):
                os.remove(out_a)

        # 2. Chạy với Preset B trên cùng thư mục cache -> Phải tạo mới âm thanh (cache miss), không được nhận cache của A
        engine_b = TTSEngine(
            auto_extract_ref=False,
            mode="design",
            multi_speaker_mode="single",
            instruct=ins_b,
            preset_name=preset_b,
            resume_cache_dir=cache_dir,
            normalize_text=False,
        )
        engine_b.load_model = MagicMock()
        mock_model_b = MagicMock()
        mock_model_b.generate.return_value = [np.ones(sr * 4, dtype=np.float32) * 0.8]
        mock_prompt_b = MagicMock()
        mock_model_b.create_voice_clone_prompt.return_value = mock_prompt_b
        engine_b.model = mock_model_b

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_b = f.name

        try:
            rep_b = engine_b.synthesize_blocks(blocks, total_duration=3.0, output_path=out_b)
            assert rep_b.status == "success"
            # mock_model_b PHẢI được gọi vì khóa cache khác biệt hoàn toàn giữa Preset A và B
            # 4 lượt thử mỏ neo B + 1 lượt sinh block = 5
            assert mock_model_b.generate.call_count == 5

            # 3. Chạy lại lần nữa với Preset B -> Phải trúng cache (cache hit), không cần gọi generate lại cho block
            mock_model_b.reset_mock()
            rep_b2 = engine_b.synthesize_blocks(blocks, total_duration=3.0, output_path=out_b)
            assert rep_b2.status == "success"
            # Anchor đã có trong file cache pt/wav nên mock_model_b.generate không cần chạy lại
            assert mock_model_b.generate.call_count == 0
        finally:
            if os.path.exists(out_b):
                os.remove(out_b)


def test_normalize_vietnamese_numbers_and_text_special_cases():
    """Kiểm tra hàm chuẩn hóa số xử lý đúng số bắt đầu bằng 0, phiên bản, ngày tháng và mã định danh."""
    from src.ai_engines.tts_engine import normalize_vietnamese_numbers_and_text

    # 1. Số bắt đầu bằng số 0 (Mã số / Số điện thoại) phải đọc từng chữ số
    assert normalize_vietnamese_numbers_and_text("0123") == "không một hai ba"
    assert normalize_vietnamese_numbers_and_text("0912345678") == "không chín một hai ba bốn năm sáu bảy tám"

    # 2. Phiên bản có dấu chấm
    assert "v một chấm hai chấm ba" in normalize_vietnamese_numbers_and_text("Phiên bản v1.2.3")

    # 3. Ngày tháng
    res_date = normalize_vietnamese_numbers_and_text("Ngày 29/10/2026")
    assert "hai mươi chín" in res_date and "mười" in res_date and "hai nghìn không trăm hai mươi sáu" in res_date

    # 4. Giờ phút
    assert "mười bốn giờ ba mươi phút" in normalize_vietnamese_numbers_and_text("Họp lúc 14h30")

    # 5. Phân cách hàng nghìn
    assert "năm mươi nghìn đồng" in normalize_vietnamese_numbers_and_text("50.000 VNĐ")

    # 6. Số nguyên và số thập phân thông thường
    assert normalize_vietnamese_numbers_and_text("100") == "một trăm"
    assert "một phẩy năm" in normalize_vietnamese_numbers_and_text("1.5")


def test_generate_single_in_design_mode_never_receives_clone_prompt_even_if_passed():
    """Kiểm tra chế độ design tuyệt đối không nhận prompt nhân bản kể cả khi tham số override_prompt được truyền vào."""
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from unittest.mock import MagicMock

    preset_name = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
    expected_instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

    engine = TTSEngine(auto_extract_ref=False, mode="design", instruct=expected_instruct)
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.2]
    engine.model = mock_model

    fake_prompt = MagicMock()
    # Truyền override_prompt giả mạo vào generate_single khi mode là design
    engine.generate_single("Chào bạn", override_mode="design", override_prompt=fake_prompt)

    assert mock_model.generate.call_count == 1
    call_kwargs = mock_model.generate.call_args.kwargs
    assert call_kwargs.get("instruct") == expected_instruct
    assert "voice_clone_prompt" not in call_kwargs or call_kwargs["voice_clone_prompt"] is None


def test_pre_export_quality_check_validates_real_audio_file():
    """Kiểm tra hàm kiểm định trước khi xuất phát hiện đúng tệp âm thanh lỗi, im lặng hoặc clipping."""
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    import soundfile as sf

    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu chuẩn", speaker="Người nói 1")
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Tệp âm thanh không tồn tại
        missing_path = os.path.join(tmp_dir, "non_existent.wav")
        issues_missing = ui._run_pre_export_quality_check(check_audio_path=missing_path)
        assert any("không tồn tại" in i["message"] for i in issues_missing)

        # 2. Tệp âm thanh hoàn toàn im lặng (RMS = 0)
        silent_path = os.path.join(tmp_dir, "silent.wav")
        sf.write(silent_path, np.zeros(24000, dtype=np.float32), 24000)
        issues_silent = ui._run_pre_export_quality_check(check_audio_path=silent_path)
        assert any("im lặng" in i["message"] for i in issues_silent)

        # 3. Tệp âm thanh đạt ngưỡng cực đại (Clipping / peak = 1.0)
        clip_path = os.path.join(tmp_dir, "clipping.wav")
        sf.write(clip_path, np.ones(24000, dtype=np.float32), 24000)
        issues_clip = ui._run_pre_export_quality_check(check_audio_path=clip_path)
        assert any("ngưỡng cực đại" in i["message"] for i in issues_clip)

        # 4. Tệp âm thanh hợp lệ (biên độ an toàn 0.5)
        valid_path = os.path.join(tmp_dir, "valid.wav")
        sf.write(valid_path, np.ones(24000, dtype=np.float32) * 0.5, 24000)
        issues_valid = ui._run_pre_export_quality_check(check_audio_path=valid_path)
        audio_issues = [i for i in issues_valid if "Tệp âm thanh" in i["message"] or "ngưỡng cực đại" in i["message"]]
        assert len(audio_issues) == 0


def test_synthesize_blocks_reports_overflow_and_respects_strict_sync_strategy():
    """Kiểm tra hệ thống báo cáo chính xác danh sách câu tràn mốc thời lượng và hỗ trợ chế độ strict_sync."""
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    sr = 24000
    # Tạo đoạn âm thanh giả lập dài 3 giây
    fake_long_audio = np.ones(int(3.0 * sr), dtype=np.float32) * 0.4

    # Phụ đề có mốc thời gian chỉ khả dụng 1.0 giây (từ 0.0s đến 1.0s, câu tiếp theo bắt đầu lúc 1.0s)
    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.0, text="Câu dài bị tràn thời gian", speaker=""),
        SubtitleBlock(index=2, start_time=1.0, end_time=2.5, text="Câu kế tiếp", speaker=""),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        # 1. Chế độ mặc định preserve_speech: co giãn 1.25x và báo cáo câu tràn vào report.overflow_lines
        engine_preserve = TTSEngine(auto_extract_ref=False, mode="design", overflow_strategy="preserve_speech")
        engine_preserve.load_model = MagicMock()
        mock_m1 = MagicMock()
        mock_m1.generate.return_value = [fake_long_audio]
        engine_preserve.model = mock_m1

        rep_p = engine_preserve.synthesize_blocks(blocks, total_duration=3.0, output_path=tmp_out)
        assert rep_p.status == "success"
        assert len(rep_p.overflow_lines) >= 1
        assert rep_p.overflow_lines[0]["line"] == 1
        assert rep_p.overflow_lines[0]["overflow_sec"] > 0

        # 2. Chế độ strict_sync: co giãn tối đa và cắt khớp mốc thời gian không đè câu tiếp theo
        engine_strict = TTSEngine(auto_extract_ref=False, mode="design", overflow_strategy="strict_sync")
        engine_strict.load_model = MagicMock()
        mock_m2 = MagicMock()
        mock_m2.generate.return_value = [fake_long_audio]
        engine_strict.model = mock_m2

        rep_s = engine_strict.synthesize_blocks(blocks, total_duration=3.0, output_path=tmp_out)
        assert rep_s.status == "success"
        assert len(rep_s.overflow_lines) >= 1
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_vietnamese_phone_number_digit_by_digit():
    from src.ai_engines.tts_engine import normalize_vietnamese_numbers_and_text
    # Kiểm tra số điện thoại có dấu chấm hoặc gạch ngang được đọc từng chữ số
    res1 = normalize_vietnamese_numbers_and_text("Liên hệ số 0987.123.456 để được tư vấn.")
    assert "không chín tám bảy một hai ba bốn năm sáu" in res1

    res2 = normalize_vietnamese_numbers_and_text("Hotline: 091-234-5678")
    assert "không chín một hai ba bốn năm sáu bảy tám" in res2


def test_preprocess_vietnamese_adds_sentence_closing_punctuation():
    from src.ai_engines.tts_engine import preprocess_vietnamese_tts_text
    # Đảm bảo câu không có dấu kết thúc được tự động bổ sung dấu chấm để hạ giọng trọn vẹn
    res = preprocess_vietnamese_tts_text("Hôm nay trời rất đẹp")
    assert res.endswith(".")
    # Câu đã có dấu câu không bị nhân đôi
    res_excl = preprocess_vietnamese_tts_text("Tuyệt vời quá!")
    assert res_excl.endswith("!")
    assert not res_excl.endswith("!.")


def test_trim_silence_padding_preserves_60ms_boundary():
    from src.ai_engines.tts_engine import TTSEngine
    sr = 24000
    # Tạo đoạn âm thanh: 1 giây im lặng, 1 giây âm thanh, 1 giây im lặng
    silence_1s = np.zeros(sr, dtype=np.float32)
    signal_1s = np.ones(sr, dtype=np.float32) * 0.5
    raw_audio = np.concatenate([silence_1s, signal_1s, silence_1s])

    trimmed = TTSEngine._trim_silence_padding(raw_audio, threshold_db=-45.0, sample_rate=sr)
    # 60ms = 1440 samples mỗi đầu -> tổng độ dài khoảng sr + 2 * 1440 = 26880
    expected_margin = int(0.060 * sr)
    assert len(trimmed) >= sr + expected_margin


def test_generate_single_sets_deterministic_seed_for_preset():
    from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
    from unittest.mock import MagicMock
    import zlib

    engine = TTSEngine(auto_extract_ref=False, mode="design")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.2]
    engine.model = mock_model

    instruct = VIETNAMESE_VOICE_CATALOG["Nữ Trẻ Năng Động - Video Ngắn Và Giải Trí"]["instruct"]
    expected_seed = (zlib.crc32(instruct.encode("utf-8")) % 100000) + 42

    mock_torch = MagicMock()
    with patch("src.ai_engines.tts_engine.torch", mock_torch):
        engine.generate_single("Câu kiểm tra hạt ngẫu nhiên", override_mode="design", override_instruct=instruct)
        mock_torch.manual_seed.assert_called_with(expected_seed)


def test_resolve_playlist_entry_url_youtube_and_bilibili():
    from src.desktop.ui.ytdlp_interface import resolve_playlist_entry_url

    # 1. YouTube dạng youtu.be có query danh sách phát
    entry_yt = {"url": "https://youtu.be/dQw4w9WgXcQ?list=PL1234567890"}
    res_yt = resolve_playlist_entry_url(entry_yt)
    assert res_yt == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # 2. Bilibili tuyển tập tập 3
    entry_bili = {"id": "BV1py4y1c71k", "page": 3}
    res_bili = resolve_playlist_entry_url(entry_bili, parent_url="https://www.bilibili.com/video/BV1py4y1c71k")
    assert res_bili == "https://www.bilibili.com/video/BV1py4y1c71k?p=3"


def test_on_analysis_success_handles_none_entries_gracefully():
    _ensure_qapp()
    from src.desktop.ui.ytdlp_interface import YtdlpInterface

    ytdlp = YtdlpInterface()
    # Danh sách phát chứa video bình thường và video bị xóa (None)
    mock_info = {
        "_type": "playlist",
        "title": "Danh sách tuyển tập thử nghiệm",
        "uploader": "Kênh mẫu",
        "entries": [
            {"id": "v1", "title": "Video 1", "duration": 120},
            None,
            {"id": "v2", "title": "Video 2", "duration": 180},
        ]
    }
    # Gọi trực tiếp slot phân tích thành công
    ytdlp.on_analysis_success(mock_info)
    assert ytdlp.is_playlist is True
    assert len(ytdlp.playlist_entries) == 2
    assert ytdlp.playlist_table.rowCount() == 2


def test_reset_all_voices_triggers_speaker_detection_when_video_present():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    interface = OmniVoiceInterface()
    interface.chk_single_unified_voice.setChecked(True)
    interface.subtitle_blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Câu thoại mẫu", voice_instruct="male, instruct")
    ]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_video = f.name

    try:
        interface.current_video_path = tmp_video
        mock_detect = MagicMock()
        interface._run_auto_detect_speakers = mock_detect

        # Gọi hàm đặt lại giọng
        interface._reset_all_character_voices()

        # Kiểm tra chế độ 1 giọng đọc đã được tắt để nhường chỗ cho giọng gốc
        assert interface.chk_single_unified_voice.isChecked() is False
        assert interface.subtitle_blocks[0].voice_instruct == ""
        # Đã kích hoạt trích xuất lại giọng gốc từ video
        mock_detect.assert_called_once()
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)


def test_on_speakers_detected_preserves_video_source_audio_samples():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.translator import SubtitleBlock

    interface = OmniVoiceInterface()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sample_wav = f.name

    try:
        interface.subtitle_blocks = [
            SubtitleBlock(index=1, start_time=0.0, end_time=1.5, text="Câu thoại của nhân vật A", speaker="Nhân vật A")
        ]
        detected = {
            "Nhân vật A": CharacterVoiceProfile(
                speaker_id="spk_0",
                display_name="Nhân vật A",
                method="clone",
                source_audio_path=sample_wav,
            )
        }
        interface._on_speakers_detected(detected)

        # Kiểm tra hồ sơ nhân vật giữ nguyên phương thức clone và đường dẫn tệp âm thanh gốc từ video
        assert "Nhân vật A" in interface.speaker_profiles
        prof = interface.speaker_profiles["Nhân vật A"]
        assert prof.method == "clone"
        assert prof.source_audio_path == sample_wav
        assert interface.subtitle_blocks[0].voice_ref_path == sample_wav
        assert interface.subtitle_blocks[0].voice_instruct == ""
    finally:
        if os.path.exists(sample_wav):
            os.remove(sample_wav)


def test_user_locked_profile_preserved_on_diarization_rerun():
    """Kiểm tra hồ sơ nhân vật được người dùng khóa (user_locked=True) được bảo toàn qua các lần nhận diện lại."""
    engine = TTSEngine(mode="clone")
    engine.speaker_profiles = {
        "Nhân vật Đã Khóa": CharacterVoiceProfile(
            speaker_id="spk_locked",
            display_name="Nhân vật Đã Khóa",
            method="preset",
            preset_name="Nữ Thanh Niên",
            status="locked",
            user_locked=True,
        ),
        "Người nói 1": CharacterVoiceProfile(
            speaker_id="spk_0",
            display_name="Người nói 1",
            method="clone",
            status="locked",
            user_locked=False,
        ),
    }

    mock_coordinator = MagicMock()
    mock_coordinator.last_provider = "CPUExecutionProvider"
    mock_coordinator.process_video_subtitles.return_value = {
        "Người nói 1": CharacterVoiceProfile(
            speaker_id="spk_0",
            display_name="Người nói 1 (Mới)",
            method="clone",
            status="locked",
            user_locked=False,
        )
    }

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào")
    with patch("src.ai_engines.speaker_diarization.SpeakerDiarizationCoordinator", return_value=mock_coordinator), \
         patch("src.ai_engines.tts_engine.VoiceSeparator._extract_audio", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("os.path.exists", return_value=True):
        engine.detect_speakers_from_video("fake_video.mp4", [b1], max_speakers=2)

    assert "Nhân vật Đã Khóa" in engine.speaker_profiles
    p_locked = engine.speaker_profiles["Nhân vật Đã Khóa"]
    preset_val = p_locked.preset_name if hasattr(p_locked, "preset_name") else p_locked.get("preset_name")
    assert preset_val == "Nữ Thanh Niên"
    p_new = engine.speaker_profiles["Người nói 1"]
    name_val = p_new.display_name if hasattr(p_new, "display_name") else p_new.get("display_name")
    assert name_val == "Người nói 1 (Mới)"


def test_srt_prefix_regex_ignores_generic_words_and_timestamps():
    """Kiểm tra phát hiện người nói từ SRT bỏ qua ghi chú chung (Lưu ý:) và mốc giờ (12:30:)."""
    engine = TTSEngine(mode="clone")
    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Lưu ý: Vui lòng không làm ồn trong phòng thu.")
    b2 = SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "12:30: Cuộc hẹn bắt đầu.")
    b3 = SubtitleBlock(3, "00:00:07,000", "00:00:09,000", "Hôm nay - chúng ta cùng thảo luận.")
    b4 = SubtitleBlock(4, "00:00:10,000", "00:00:12,000", "[Đạo Diễn]: Chuẩn bị bấm máy!")

    profiles = engine.detect_speakers_from_srt_text([b1, b2, b3, b4])

    assert b1.text.startswith("Lưu ý:")
    assert b2.text.startswith("12:30:")
    assert b3.text.startswith("Hôm nay -")
    assert b4.text == "Chuẩn bị bấm máy!"
    assert b4.speaker == "Người nói 1"


def test_config_sync_items_and_aliases():
    """Kiểm tra đồng bộ các mục cấu hình TTS MMO và alias tương thích hai chiều."""
    from src.core.config import cfg

    assert hasattr(cfg, "mmoTTSResumeCacheDir")
    assert hasattr(cfg, "mmoTTSOverflowStrategy")
    assert hasattr(cfg, "mmoTTSEnableAiVocalSeparation")
    assert hasattr(cfg, "mmoTTSAIVocalSeparation")

    engine = TTSEngine.from_app_config()
    assert engine.overflow_strategy in ("extend", "stretch", "strict_sync", "preserve_speech")


def test_preserve_speech_cursor_drift():
    """Kiểm tra chiến lược preserve_speech bù trôi mốc thời gian an toàn không cắt đuôi câu."""
    import soundfile as sf
    engine = TTSEngine(auto_extract_ref=False, mode="design", multi_speaker_mode="single", overflow_strategy="preserve_speech")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.sampling_rate = sr
    mock_model.generate.return_value = [np.ones(sr * 3, dtype=np.float32) * 0.4]
    mock_model.create_voice_clone_prompt.return_value = MagicMock()
    engine.model = mock_model

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu một thoại dài hơn thời lượng khả dụng.")
    b2 = SubtitleBlock(2, "00:00:02,100", "00:00:03,000", "Câu hai kế tiếp.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_wav = tf.name

    try:
        rep = engine.synthesize_blocks([b1, b2], total_duration=8.0, output_path=out_wav)
        assert rep.status in ("success", "partial")
        assert rep.succeeded_count == 2
        assert os.path.exists(out_wav)
        data, _ = sf.read(out_wav)
        assert len(data) > int(2.0 * sr)
    finally:
        if os.path.exists(out_wav):
            os.remove(out_wav)


def test_camplus_voice_consistency_refinement_loop():
    """Kiểm tra vòng kiểm định tính nhất quán âm sắc CAM++ phát hiện câu lệch chuẩn và tái tạo."""
    from src.ai_engines.tts_engine import DubbingReport
    engine = TTSEngine(auto_extract_ref=False, mode="design", multi_speaker_mode="single")
    engine.speaker_model_path = "fake_camplus.onnx"
    sr = 24000
    engine.sample_rate = sr

    normal_audio = np.ones(sr * 2, dtype=np.float32) * 0.5
    outlier_audio = np.ones(sr * 2, dtype=np.float32) * -0.5

    items = [
        {"index": 0, "audio": normal_audio, "start_sample": 0, "spk_key": "spk_1", "text": "Câu 1", "mode": "design", "prompt": None, "ref": None, "instruct": "test"},
        {"index": 1, "audio": normal_audio, "start_sample": sr * 3, "spk_key": "spk_1", "text": "Câu 2", "mode": "design", "prompt": None, "ref": None, "instruct": "test"},
        {"index": 2, "audio": outlier_audio, "start_sample": sr * 6, "spk_key": "spk_1", "text": "Câu 3", "mode": "design", "prompt": None, "ref": None, "instruct": "test"},
    ]
    full_audio = np.zeros(sr * 10, dtype=np.float32)
    rep = DubbingReport(total_blocks=3)

    mock_session = MagicMock()
    # Giả lập embedding: câu 0 và 1 có vector [1, 0], câu 2 có vector [0, 1] (cosin với centroid thấp < 0.65)
    def fake_extract(session, audio, sample_rate):
        if float(audio[0]) > 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    improved_audio = np.ones(sr * 2, dtype=np.float32) * 0.8
    engine.generate_single = MagicMock(return_value=improved_audio)

    with patch("os.path.exists", return_value=True), \
         patch("src.ai_engines.speaker_embedding_onnx.create_speaker_session", return_value=mock_session), \
         patch("src.ai_engines.speaker_embedding_onnx.extract_speaker_embedding", side_effect=fake_extract), \
         patch("src.ai_engines.speaker_embedding_onnx.close_speaker_session"):
        engine._verify_and_refine_voice_consistency(items, full_audio, rep, min_sim=0.65)

    assert rep.inconsistent_retried_count >= 1


def test_campplus_model_sha256_verification(tmp_path):
    """Kiểm tra mã băm SHA-256 của mô hình CAM++ được đăng ký và xác thực toàn vẹn."""
    from src.ai_engines.model_hashes import KNOWN_WEIGHT_HASHES, verify_weight_file

    assert "campplus.onnx" in KNOWN_WEIGHT_HASHES
    assert KNOWN_WEIGHT_HASHES["campplus.onnx"] == "dd1740aa1e1ffa3895f96aef2166b8af2bb2ad09c00769dd275ee36aef6a2a7f"

    # Tạo tệp giả lập có mã băm sai
    bad_model = tmp_path / "campplus.onnx"
    bad_model.write_bytes(b"corrupted_model_content_12345")
    assert not verify_weight_file(bad_model, "campplus.onnx")


def test_extract_audio_sample_rate_and_clone_high_quality(tmp_path):
    """Kiểm tra trích xuất âm thanh gốc 24 kHz và cắt mẫu nhân vật giữ tần số lấy mẫu cao."""
    from src.ai_engines.voice_separator import VoiceSeparator
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator, CharacterVoiceProfile
    import soundfile as sf

    sep = VoiceSeparator()
    # Kiểm tra chữ ký hàm _extract_audio nhận sample_rate
    assert sep._extract_audio.__code__.co_varnames[3] == "sample_rate"

    # Giả lập quá trình process_video_subtitles với clone_audio_path 24 kHz
    sr_diar = 16000
    sr_clone = 24000
    diar_wav = tmp_path / "diar_16k.wav"
    clone_wav = tmp_path / "clone_24k.wav"

    d_data = np.ones(sr_diar * 4, dtype=np.float32) * 0.3
    c_data = np.ones(sr_clone * 4, dtype=np.float32) * 0.3
    sf.write(str(diar_wav), d_data, sr_diar)
    sf.write(str(clone_wav), c_data, sr_clone)

    coord = SpeakerDiarizationCoordinator(allow_mock=True)
    b1 = SubtitleBlock(1, "00:00:00,500", "00:00:03,500", "Xin chào mọi người")

    profiles = coord.process_video_subtitles(
        video_path=str(diar_wav),
        blocks=[b1],
        temp_dir=str(tmp_path),
        clone_audio_path=str(clone_wav),
    )

    assert len(profiles) > 0
    spk_key = list(profiles.keys())[0]
    sample_file = profiles[spk_key].source_audio_path
    assert os.path.exists(sample_file)
    info = sf.info(sample_file)
    # Tần số lấy mẫu của tệp cắt mẫu phải đạt chuẩn 24 kHz của OmniVoice
    assert info.samplerate == sr_clone


def test_anchor_prompt_f0_gender_alignment_preference():
    """Kiểm tra bộ tạo mỏ neo chấm điểm ưu tiên mẫu có cao độ F0 khớp với giới tính yêu cầu."""
    engine = TTSEngine(mode="design", instruct="female, young adult, high pitch")
    engine.load_model = MagicMock()
    mock_model = MagicMock()

    sr = 24000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False, dtype=np.float32)
    # Mẫu 1: F0 = 130 Hz (giọng nam, lệch giới tính)
    male_wave = 0.4 * np.sin(2 * np.pi * 130 * t)
    # Mẫu 2: F0 = 220 Hz (giọng nữ, chuẩn giới tính)
    female_wave = 0.4 * np.sin(2 * np.pi * 220 * t)

    mock_model.generate.side_effect = [[male_wave], [female_wave]]
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model

    with patch("os.path.exists", return_value=False):
        prompt, wav_path = engine._build_anchor_prompt(
            key="test_gender_f0",
            instruct="female, young adult",
            sample_text="Xin chào",
            tries=2,
        )

    assert prompt is not None
    # Xác nhận tệp wav mỏ neo được ghi từ mẫu giọng nữ đạt điểm cao hơn
    saved_data, _ = sf.read(wav_path)
    c_feat = engine._extract_acoustic_features(saved_data, sr)
    assert c_feat is not None
    measured_f0 = float(c_feat[0] * 200.0)
    assert measured_f0 > 180.0


def test_synthesize_blocks_forwards_all_clustering_parameters(tmp_path):
    """Kiểm tra synthesize_blocks truyền đầy đủ exact_speakers và clustering_sensitivity."""
    engine = TTSEngine(
        mode="design",
        multi_speaker_mode="auto_video_cluster",
        max_speakers=3,
        exact_speakers=3,
        clustering_sensitivity="high",
        enable_ai_vocal_separation=False,
    )
    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Chào bạn")

    engine.detect_speakers_from_video = MagicMock(return_value={})
    engine.load_model = MagicMock()
    engine.model = MagicMock()
    engine.model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.25]
    engine._build_anchor_prompt = MagicMock(return_value=(MagicMock(), "fake_anchor.wav"))

    fake_video = str(tmp_path / "fake.mp4")
    Path(fake_video).write_bytes(b"dummy")
    out_audio = str(tmp_path / "out.wav")

    with patch("os.path.exists", return_value=True), \
         patch("src.ai_engines.tts_engine.TTSEngine._has_audio_stream", return_value=False):
        engine.synthesize_blocks([b1], total_duration=5.0, output_path=out_audio, video_path=fake_video)

    assert engine.detect_speakers_from_video.called
    kwargs = engine.detect_speakers_from_video.call_args[1]
    assert kwargs.get("exact_speakers") == 3
    assert kwargs.get("clustering_sensitivity") == "high"
    assert kwargs.get("enable_ai_vocal_separation") is False


def test_soft_knee_limiter_preserves_unclipped_speech():
    """Kiểm tra bộ nén soft-knee limiter không dìm nhỏ các câu thoại bình thường khi có một đỉnh âm lượng lớn."""
    sr = 24000
    engine = TTSEngine(mode="design")
    engine.load_model = MagicMock()

    # Tạo câu 1 có biên độ bình thường (0.6) và câu 2 có một đỉnh lớn (1.4)
    normal_seg = np.ones(sr * 2, dtype=np.float32) * 0.6
    spiked_seg = np.ones(sr * 2, dtype=np.float32) * 0.6
    spiked_seg[100:150] = 1.4

    engine.generate_single = MagicMock(side_effect=[normal_seg, spiked_seg])
    engine._build_anchor_prompt = MagicMock(return_value=(MagicMock(), "fake_anchor.wav"))

    b1 = SubtitleBlock(1, "00:00:00,500", "00:00:02,500", "Câu một")
    b2 = SubtitleBlock(2, "00:00:03,000", "00:00:05,000", "Câu hai")

    out_file = "test_limiter_out.wav"
    try:
        with patch("src.ai_engines.tts_engine.TTSEngine._has_audio_stream", return_value=False):
            report = engine.synthesize_blocks([b1, b2], total_duration=6.0, output_path=out_file)
        assert os.path.exists(out_file)
        data, _ = sf.read(out_file)
        # Đỉnh cao nhất được nén an toàn <= 0.892
        assert np.max(np.abs(data)) <= 0.892
        # Đoạn có thoại của câu một (từ 0.6s đến 2.0s) vẫn giữ biên độ khỏe (không bị tụt thảm hại)
        speech_slice = data[int(0.6 * sr):int(2.0 * sr)]
        assert np.mean(np.abs(speech_slice)) > 0.55
    finally:
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except Exception:
                pass


def test_unassigned_speaker_inherits_neighbor_before_fallback(tmp_path):
    """Kiểm tra câu chưa xác định người nói được kế thừa người nói của câu liền trước thay vì chuyển sang giọng ngẫu nhiên."""
    engine = TTSEngine(auto_extract_ref=False, multi_speaker_mode="multi")
    engine.load_model = MagicMock()
    b1 = SubtitleBlock(1, "00:00:00,000", "00:00:02,000", "Câu một của Nam", speaker="Người nói 1")
    b2 = SubtitleBlock(2, "00:00:02,500", "00:00:04,500", "Câu hai không rõ", speaker="Chưa xác định")

    mock_prompt_1 = MagicMock()
    engine._speaker_prompt_cache["character_Người nói 1"] = mock_prompt_1
    engine.speaker_profiles = {"Người nói 1": {"voice_instruct": "male, young adult"}}

    captured_kwargs = []
    t = np.linspace(0, 1.0, 24000, endpoint=False, dtype=np.float32)
    sine_audio = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    def mock_gen(text, **kwargs):
        captured_kwargs.append(kwargs)
        return sine_audio

    engine.generate_single = MagicMock(side_effect=mock_gen)

    out_wav = str(tmp_path / "test_inherit.wav")
    with patch("src.ai_engines.tts_engine.TTSEngine._has_audio_stream", return_value=False):
        report = engine.synthesize_blocks([b1, b2], total_duration=5.0, output_path=out_wav)

    assert report.succeeded_count == 2
    assert captured_kwargs[1]["override_prompt"] == mock_prompt_1


def test_speaker_prefix_detection_rules():
    """Kiểm tra quy tắc nhận diện tiền tố người nói: viết hoa/tên riêng nhận diện, từ thời gian/chú ý bị bỏ qua."""
    engine = TTSEngine(auto_extract_ref=False)
    b1 = SubtitleBlock(1, "00:00:00,000", "00:00:02,000", "[Bác sĩ] Xin chào bạn")
    b2 = SubtitleBlock(2, "00:00:02,500", "00:00:04,500", "Hôm nay: thời tiết đẹp")
    b3 = SubtitleBlock(3, "00:00:05,000", "00:00:07,000", "NAM: Đi thôi")
    b4 = SubtitleBlock(4, "00:00:07,500", "00:00:09,500", "chú ý: cẩn thận đường trơn")

    profiles = engine.detect_speakers_from_srt_text([b1, b2, b3, b4])
    assert "Bác sĩ" in profiles or "Người nói 1" in profiles
    assert any("NAM" in p.get("description", "") for p in profiles.values())
    assert "Hôm nay" not in profiles and "hôm nay" not in profiles
    assert "Chú ý" not in profiles and "chú ý" not in profiles


def test_multilingual_anchor_sample_text():
    """Kiểm tra mẫu mỏ neo đa ngôn ngữ sinh đúng câu theo ngôn ngữ thiết lập."""
    engine = TTSEngine(auto_extract_ref=False)
    engine.language = "en"
    en_sample = engine._get_anchor_sample_text()
    assert "hello everyone" in en_sample.lower() or "timbre" in en_sample.lower()

    engine.language = "vi"
    vi_sample = engine._get_anchor_sample_text()
    assert len(vi_sample) > 20

    engine.language = "zh"
    zh_sample = engine._get_anchor_sample_text()
    assert len(zh_sample) > 5


def test_single_clone_fails_early_when_no_audio(tmp_path):
    """Kiểm tra chế độ clone 1 giọng báo lỗi rõ ràng khi không có tệp âm thanh mẫu."""
    engine = TTSEngine(auto_extract_ref=False, mode="clone", multi_speaker_mode="single")
    engine.ref_audio_path = "duong_dan_khong_ton_tai_12345.wav"
    engine.current_prompt = None
    engine.load_model = MagicMock()

    b1 = SubtitleBlock(1, "00:00:00,000", "00:00:02,000", "Câu một")
    out_wav = str(tmp_path / "test_fail.wav")
    report = engine.synthesize_blocks([b1], total_duration=3.0, output_path=out_wav)

    assert report.status == "failed"
    assert "Không tìm thấy" in report.message or "không thể" in report.message


def test_normalize_vietnamese_large_numbers_and_version_strings():
    """Kiểm tra đọc đúng số tiền hàng triệu, số nghìn và phân biệt với chuỗi phiên bản."""
    from src.ai_engines.tts_engine import normalize_vietnamese_numbers_and_text

    t1 = normalize_vietnamese_numbers_and_text("Chi phí là 1.000.000 đồng")
    assert "một triệu đồng" in t1
    assert "chấm không" not in t1

    t2 = normalize_vietnamese_numbers_and_text("Có 2.500 người tham gia")
    assert "hai nghìn năm trăm người" in t2

    t3 = normalize_vietnamese_numbers_and_text("Phiên bản v1.2.3 đã ra mắt")
    assert "chấm" in t3


def test_agglomerative_cluster_embeddings_scipy_accuracy():
    """Kiểm tra phân cụm thứ bậc SciPy gom đúng 2 cụm vector tách biệt."""
    from src.ai_engines.speaker_diarization import agglomerative_cluster_embeddings

    # Tạo 2 nhóm vector trực giao rõ rệt
    c1 = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    c2 = np.tile([0.0, 1.0, 0.0, 0.0], (4, 1))
    embs = np.vstack([c1, c2])

    labels = agglomerative_cluster_embeddings(embs, exact_k=2)
    assert len(labels) == 8
    assert len(np.unique(labels)) == 2
    assert np.all(labels[:4] == labels[0])
    assert np.all(labels[4:] == labels[4])
    assert labels[0] != labels[4]


def test_download_speaker_embedding_model_file_path_handling(tmp_path):
    """Kiểm tra hàm tải mô hình xử lý an toàn khi tham số là đường dẫn tệp thay vì thư mục."""
    from src.ai_engines.speaker_embedding_onnx import download_speaker_embedding_model

    file_target = tmp_path / "models" / "campplus.onnx"
    # Giả lập tệp mô hình đã tồn tại sẵn
    file_target.parent.mkdir(parents=True, exist_ok=True)
    file_target.write_bytes(b"RIFF" + b"0" * 2000)

    with patch("src.ai_engines.model_hashes.verify_weight_file", return_value=True):
        res = download_speaker_embedding_model(file_target)
        assert Path(res).is_file()
        assert not (file_target.parent / "campplus.onnx").is_dir()


def test_preprocess_audio_resamples_24k_to_16k():
    """Kiểm tra _preprocess_audio tự động lấy mẫu lại từ 24 kHz về 16 kHz an toàn."""
    from src.ai_engines.speaker_embedding_onnx import _preprocess_audio

    # Tạo tín hiệu 24 kHz dài 1 giây (24000 mẫu)
    audio_24k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
    processed = _preprocess_audio(audio_24k, sample_rate=24000)

    # Đầu ra phải tương ứng 16000 mẫu (hoặc tối thiểu 0.25s)
    assert len(processed) >= 15000 and len(processed) <= 17000
    assert np.all(np.isfinite(processed))


def test_generate_single_respects_custom_seed():
    """Kiểm tra generate_single chấp nhận tham số hạt số ngẫu nhiên riêng biệt khi tái tạo."""
    import torch
    engine = TTSEngine(auto_extract_ref=False, instruct="instruct gốc")
    engine.load_model = MagicMock()
    engine.model = MagicMock()
    engine.model.generate = MagicMock(return_value=[np.zeros(24000, dtype=np.float32)])

    captured_seed = None

    def fake_manual_seed(seed):
        nonlocal captured_seed
        captured_seed = seed

    with patch("torch.manual_seed", side_effect=fake_manual_seed):
        engine.generate_single("Kiểm tra hạt số", seed=777888)

    assert captured_seed == 777888


def test_reset_speaker_locked_prompt_removes_anchor_files_on_disk(tmp_path):
    """Kiểm tra reset_speaker_locked_prompt dọn sạch các tệp mỏ neo đĩa của nhân vật."""
    engine = TTSEngine(auto_extract_ref=False)
    engine.voice_prompt_cache_dir = str(tmp_path)

    wav_file = tmp_path / "anchor_spk1_test.wav"
    pt_file = tmp_path / "anchor_spk1_test.pt"
    wav_file.write_bytes(b"wavdata")
    pt_file.write_bytes(b"ptdata")

    engine._speaker_prompt_cache["char_Người nói 1"] = MagicMock()
    engine._speaker_anchor_files["char_Người nói 1"] = {str(wav_file), str(pt_file)}

    engine.reset_speaker_locked_prompt("Người nói 1")

    assert not wav_file.exists()
    assert not pt_file.exists()
    assert "char_Người nói 1" not in engine._speaker_prompt_cache


def test_compute_block_cache_path_includes_language_and_ref_audio(tmp_path):
    """Kiểm tra khóa cache phụ thuộc vào ngôn ngữ và tệp âm thanh mẫu."""
    engine = TTSEngine(auto_extract_ref=False, mode="clone")
    engine.resume_cache_dir = str(tmp_path)
    engine.ref_audio_path = str(tmp_path / "ref_a.wav")
    (tmp_path / "ref_a.wav").write_bytes(b"123")

    engine.language = "vi"
    p_vi = engine._compute_block_cache_path(0, "Chào", "unified", None, None, 2.0)

    engine.language = "en"
    p_en = engine._compute_block_cache_path(0, "Chào", "unified", None, None, 2.0)

    assert p_vi != p_en


def test_synthesize_blocks_single_auto_mode_builds_anchor_prompt(tmp_path):
    """Kiểm tra chế độ một giọng với mode='auto' tạo và khóa mỏ neo thống nhất."""
    engine = TTSEngine(auto_extract_ref=False, mode="auto", multi_speaker_mode="single")
    engine.load_model = MagicMock()

    mock_prompt = MagicMock()
    anchor_wav = str(tmp_path / "anchor_auto.wav")
    engine._build_anchor_prompt = MagicMock(return_value=(mock_prompt, anchor_wav))

    captured_calls = []

    def fake_generate_single(text, **kwargs):
        captured_calls.append(kwargs)
        return np.zeros(24000, dtype=np.float32)

    engine.generate_single = fake_generate_single

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một")
    b2 = SubtitleBlock(2, "00:00:03,500", "00:00:05,500", "Câu hai")
    out_wav = str(tmp_path / "out_auto.wav")

    engine.synthesize_blocks([b1, b2], total_duration=6.0, output_path=out_wav)

    assert len(captured_calls) == 2
    for c in captured_calls:
        assert c.get("override_mode") == "clone"
        assert c.get("override_prompt") is mock_prompt
        assert c.get("override_ref_audio") == anchor_wav


def test_switching_between_single_and_multi_speaker_mode_preserves_identities():
    """Xác minh chuyển đổi giữa chế độ một giọng và đa nhân vật bảo toàn 100% dữ liệu gốc."""
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    ui = OmniVoiceInterface()
    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu một", speaker="Người nói 1", voice_ref_path="char1.wav")
    b2 = SubtitleBlock(2, "00:00:02,500", "00:00:03,500", "Câu hai", speaker="Người nói 2", voice_ref_path="char2.wav")
    ui.subtitle_blocks = [b1, b2]

    p1 = CharacterVoiceProfile(speaker_id="spk_1", display_name="Người nói 1", method="clone", source_audio_path="char1.wav")
    p2 = CharacterVoiceProfile(speaker_id="spk_2", display_name="Người nói 2", method="clone", source_audio_path="char2.wav")
    ui.speaker_profiles = {"Người nói 1": p1, "Người nói 2": p2}

    # Chọn chế độ một giọng đồng nhất
    ui.combo_multi_speaker_mode.setCurrentIndex(0)
    assert ui._get_selected_multi_speaker_mode() == "single"
    assert not ui.container_single_voice.isHidden()
    assert ui.container_multi_voice.isHidden()

    # Dữ liệu phụ đề và hồ sơ nhân vật KHÔNG bị ghi đè
    assert ui.subtitle_blocks[0].speaker == "Người nói 1"
    assert ui.subtitle_blocks[0].voice_ref_path == "char1.wav"
    assert ui.subtitle_blocks[1].speaker == "Người nói 2"
    assert ui.subtitle_blocks[1].voice_ref_path == "char2.wav"
    assert ui.speaker_profiles["Người nói 1"].source_audio_path == "char1.wav"

    # Chuyển lại chế độ đa nhân vật
    ui.combo_multi_speaker_mode.setCurrentIndex(1)
    assert ui._get_selected_multi_speaker_mode() == "auto_video_cluster"
    assert ui.container_single_voice.isHidden()
    assert not ui.container_multi_voice.isHidden()

    # Dữ liệu phân vai vẫn còn nguyên vẹn 100%
    assert ui.subtitle_blocks[0].speaker == "Người nói 1"
    assert ui.subtitle_blocks[0].voice_ref_path == "char1.wav"
    assert ui.subtitle_blocks[1].speaker == "Người nói 2"
    assert ui.subtitle_blocks[1].voice_ref_path == "char2.wav"
    assert len(ui.speaker_profiles) == 2


def test_drag_and_drop_pt_vs_audio_file(tmp_path):
    """Xác minh kéo thả tệp .pt được nhận diện là đặc trưng giọng đã mã hóa, không phải tệp WAV."""
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QDropEvent
    from PySide6.QtCore import QPointF, Qt

    ui = OmniVoiceInterface()
    pt_file = str(tmp_path / "voice_prompt.pt")
    with open(pt_file, "wb") as f:
        f.write(b"mock_prompt_bytes")

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(pt_file)])
    event = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)

    ui.dropEvent(event)
    assert ui.current_ref_audio_path == pt_file
    assert "đã mã hóa" in ui.lbl_ref_status.text()
    assert ui.btn_play_ref_audio.isEnabled() is False

    # Kéo thả tệp WAV thông thường
    wav_file = str(tmp_path / "sample.wav")
    with open(wav_file, "wb") as f:
        f.write(b"mock_wav_bytes")

    mime_wav = QMimeData()
    mime_wav.setUrls([QUrl.fromLocalFile(wav_file)])
    event_wav = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime_wav, Qt.LeftButton, Qt.NoModifier)

    ui.dropEvent(event_wav)
    assert ui.current_ref_audio_path == wav_file
    assert "Tệp âm thanh mẫu đang nạp" in ui.lbl_ref_status.text()
    assert ui.btn_play_ref_audio.isEnabled() is True


def test_master_detection_button_dynamic_state_and_label(tmp_path):
    """Xác minh nút nhận diện chính tại Bước 2 chuyển đổi nhãn và trạng thái chính xác."""
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    ui = OmniVoiceInterface()
    ui.current_video_path = ""
    ui.subtitle_blocks = []
    ui.speaker_profiles = {}
    ui._update_identify_button_state()

    # Chưa có dữ liệu: nút bị khóa
    assert ui.btn_auto_detect_speakers.isEnabled() is False
    assert ui.btn_auto_detect_speakers.text() == "Tự Động Nhận Diện Nhân Vật"

    # Có phụ đề: nút mở khóa
    ui.subtitle_blocks = [SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Chào")]
    ui._update_identify_button_state()
    assert ui.btn_auto_detect_speakers.isEnabled() is True
    assert ui.btn_auto_detect_speakers.text() == "Tự Động Nhận Diện Nhân Vật"

    # Đã có nhân vật phân vai: nút đổi nhãn thành Phân Tích Lại Nhân Vật
    ui.speaker_profiles = {"Người nói 1": {"speaker_id": "spk_1"}}
    ui._update_identify_button_state()
    assert ui.btn_auto_detect_speakers.isEnabled() is True
    assert ui.btn_auto_detect_speakers.text() == "Phân Tích Lại Nhân Vật"


def test_dubbing_report_quality_scoring_and_to_dict():
    """Xác minh các chỉ số chất lượng mới trong DubbingReport và phương thức to_dict."""
    from src.ai_engines.tts_engine import DubbingReport

    report = DubbingReport(
        status="success",
        total_blocks=10,
        succeeded_count=10,
        failed_count=0,
        stretched_count=2,
        inconsistent_retried_count=1,
        speaker_consistency_score=0.92,
        fallback_count=0,
    )
    # Tính compressed_ratio và quality_score
    total_ok = max(1, report.succeeded_count)
    report.compressed_ratio = round(float(report.stretched_count) / float(total_ok), 4)
    calc_score = 100.0 - min(30.0, report.compressed_ratio * 30.0) - min(20.0, max(0.0, (1.0 - report.speaker_consistency_score) * 40.0))
    report.quality_score = round(calc_score, 1)

    assert report.compressed_ratio == 0.2
    assert report.quality_score > 85.0
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["speaker_consistency_score"] == 0.92
    assert d["compressed_ratio"] == 0.2
    assert d["quality_score"] == report.quality_score
    assert d["fallback_count"] == 0


def test_speaker_profiles_project_persistence(tmp_path):
    """Xác minh lưu và tải hồ sơ phân vai dự án dạng JSON với đầy đủ thuộc tính và user_locked."""
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(allow_mock=True)
    engine.speaker_profiles = {
        "Người nói 1": {
            "name": "Người nói 1",
            "method": "preset",
            "preset_name": "Nam Trầm",
            "voice_instruct": "male, calm",
            "source_audio_path": "",
            "ref_text": "",
            "user_locked": True,
            "description": "Nhân vật chính",
        },
        "Người nói 2": {
            "name": "Người nói 2",
            "method": "clone",
            "preset_name": "",
            "voice_instruct": "",
            "source_audio_path": "sample.wav",
            "ref_text": "Chào bạn",
            "user_locked": False,
            "description": "Nhân vật phụ",
        },
    }

    json_path = str(tmp_path / "project_profiles.json")
    assert engine.save_speaker_profiles(json_path) is True
    assert Path(json_path).is_file()

    # Tạo engine mới và tải lại
    engine2 = TTSEngine(allow_mock=True)
    engine2.speaker_profiles = {
        "Người nói 1": {
            "name": "Người nói 1",
            "method": "clone",
            "user_locked": True,  # Người dùng đã khóa ở session mới
        }
    }
    loaded = engine2.load_speaker_profiles(json_path, merge=True)
    assert "Người nói 1" in loaded
    assert "Người nói 2" in loaded
    # user_locked của Người nói 1 được bảo vệ
    assert loaded["Người nói 1"]["method"] == "clone"
    assert loaded["Người nói 2"]["user_locked"] is False


def test_advanced_parameters_reset_to_default():
    """Xác minh nút đặt lại mặc định và trạng thái thu gọn mặc định của tham số nâng cao."""
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    ui = OmniVoiceInterface()
    # Khối nội dung kỹ thuật ban đầu phải thu gọn
    assert not ui.adv_content_widget.isVisible()

    # Thay đổi tham số
    ui.spin_num_step.setValue(12)
    ui.spin_guidance.setValue(3.5)
    ui.spin_t_shift.setValue(0.5)

    # Đặt lại mặc định
    ui._reset_advanced_parameters_to_default()
    assert ui.spin_num_step.value() == 32
    assert ui.spin_guidance.value() == 2.0
    assert ui.spin_t_shift.value() == 0.10


def test_subtitle_table_dual_audition_and_filter():
    """Xác minh bảng phụ đề có nút nghe gốc, nút nghe thử câu đang chọn và bộ lọc chỉ hiện dòng cần duyệt."""
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from PySide6.QtWidgets import QPushButton

    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Chào bạn", speaker="Người nói 1", is_speaker_locked=True),
        SubtitleBlock(2, "00:00:02,500", "00:00:04,000", "Cần kiểm tra câu này", speaker="", confidence_level="review_needed"),
    ]
    ui._populate_subtitle_table()

    # Kiểm tra ô cột 5 có nút nghe gốc QPushButton
    btn_listen = ui.sub_table.cellWidget(0, 5)
    assert isinstance(btn_listen, QPushButton)
    assert hasattr(ui, "btn_preview_selected_row")
    assert ui.btn_preview_selected_row.text() == "Nghe Thử Câu Đang Chọn"

    # Kiểm tra bộ lọc "Chỉ hiện dòng cần duyệt"
    ui.combo_sub_filter.setCurrentText("Chỉ hiện dòng cần duyệt")
    assert ui.sub_table.isRowHidden(0) is True  # Đã khóa -> ẩn
    assert ui.sub_table.isRowHidden(1) is False  # Cần kiểm tra -> hiện

    # Chuyển về "Tất cả câu thoại"
    ui.combo_sub_filter.setCurrentText("Tất cả câu thoại")
    assert ui.sub_table.isRowHidden(0) is False
    assert ui.sub_table.isRowHidden(1) is False


def test_gender_and_age_group_decoupling():
    """Xác minh tách biệt hoàn toàn giữa giới tính và nhóm tuổi, cùng bộ phân loại âm học đa chiều."""
    from src.ai_engines.tts_engine import AcousticClassifier, VoiceRecommendationEngine, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    # 1. Giọng trẻ em (F0 > 240 Hz, centroid cao): age_group='child', gender='female' hoặc 'male'
    child_f0 = [255.0, 260.0, 268.0, 252.0]
    res_child = AcousticClassifier.classify(child_f0, centroid=2400.0, iqr_f0=12.0)
    assert res_child["age_group"] == "child"
    assert res_child["gender"] in ("female", "male")
    assert res_child["pitch_level"] in ("high", "very_high")
    assert res_child["style_tag"] == "hoat_hinh"

    # Kiểm tra gợi ý giọng cho trẻ em: phải khớp với preset Thiếu Nhi / Em Bé với điểm số cao (> 70)
    prof_child = CharacterVoiceProfile(
        speaker_id="spk_child",
        display_name="Bé Na",
        gender=res_child["gender"],
        age_group="child",
        pitch_level=res_child["pitch_level"],
    )
    score_child, desc_child = VoiceRecommendationEngine.calculate_voice_match(
        prof_child,
        VIETNAMESE_VOICE_CATALOG["Thiếu Nhi Hồn Nhiên - Phim Hoạt Hình"],
    )
    assert score_child >= 60
    assert "nhóm tuổi phù hợp" in desc_child

    # 2. Giọng nữ trầm ở vùng F0 giao thoa (F0 = 168 Hz, centroid cao, formants sáng)
    female_deep_f0 = [165.0, 170.0, 168.0, 169.0]
    res_f_deep = AcousticClassifier.classify(female_deep_f0, centroid=1700.0, iqr_f0=14.0)
    assert res_f_deep["gender"] == "female"
    assert res_f_deep["age_group"] in ("middle_aged", "young_adult")

    # 3. Giọng nam cao ở vùng F0 giao thoa (F0 = 165 Hz, centroid thấp, dải trầm áp đảo)
    male_high_f0 = [162.0, 165.0, 168.0, 164.0]
    # 8 dải năng lượng: dải thấp (0, 1) áp đảo dải cao
    male_bands = np.array([3.0, 2.5, 1.2, 1.0, 0.8, 0.4, 0.3, 0.2], dtype=np.float32)
    res_m_high = AcousticClassifier.classify(male_high_f0, centroid=1100.0, iqr_f0=14.0, band_energies=male_bands)
    assert res_m_high["gender"] == "male"
    assert res_m_high["age_group"] in ("young_adult", "teenager")

    # 4. Giọng nam rất trầm / người cao tuổi (F0 = 98 Hz)
    res_elder = AcousticClassifier.classify([95.0, 98.0, 102.0], centroid=950.0, iqr_f0=15.0)
    assert res_elder["gender"] == "male"
    assert res_elder["age_group"] == "elderly"
    assert res_elder["pitch_level"] == "very_low"


def test_vietnamese_accent_classification():
    """Xác minh bộ nhận diện phương ngữ tiếng Việt (Bắc, Trung, Nam) qua từ vựng phụ đề."""
    from src.ai_engines.tts_engine import VietnameseAccentClassifier

    # Miền Nam
    south_subs = [
        "Anh đi đâu hổng nói cho tía má hay nè, làm tui lo quá nghen",
        "Con cá bự ghê, ăn xỉn luôn đó nha",
    ]
    acc_s, conf_s = VietnameseAccentClassifier.classify(south_subs)
    assert acc_s == "nam"
    assert conf_s >= 0.70

    # Miền Trung
    central_subs = [
        "Mô tê răng rứa, mi mần chi mà chừ mới về rứa hè",
        "Cái đọi ni đẹp quá chi rứa",
    ]
    acc_c, conf_c = VietnameseAccentClassifier.classify(central_subs)
    assert acc_c == "trung"
    assert conf_c >= 0.70

    # Miền Bắc
    north_subs = [
        "Bố mẹ bảo thế à, thế thì để con lấy thìa ăn chè nhé",
        "Dạ vâng, hôm nay trời đẹp nhỉ",
    ]
    acc_n, conf_n = VietnameseAccentClassifier.classify(north_subs)
    assert acc_n == "bac"
    assert conf_n >= 0.70

    # Câu thông thường không có từ khẩu ngữ đặc trưng -> Mặc định miền Bắc chuẩn 0.50
    plain_subs = ["Hôm nay là một ngày thật tuyệt vời để bắt đầu công việc mới."]
    acc_p, conf_p = VietnameseAccentClassifier.classify(plain_subs)
    assert acc_p == "bac"
    assert conf_p == 0.50


def test_global_voice_allocator_non_colliding():
    """Xác minh thuật toán GlobalVoiceAllocator đảm bảo 100% không có 2 nhân vật nào bị trùng giọng."""
    from src.ai_engines.tts_engine import GlobalVoiceAllocator, VIETNAMESE_VOICE_CATALOG, verify_speaker_acoustic_distinction
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.translator import SubtitleBlock

    # 4 nhân vật có đặc tính âm học khác nhau:
    # Nhân vật 1: Nam trầm miền Nam
    p1 = CharacterVoiceProfile(speaker_id="spk_1", display_name="Bác Ba Phi", gender="male", age_group="middle_aged", pitch_level="low", vietnamese_accent="nam", accent_confidence=0.8)
    # Nhân vật 2: Nữ thanh niên miền Bắc
    p2 = CharacterVoiceProfile(speaker_id="spk_2", display_name="Cô Thắm", gender="female", age_group="young_adult", pitch_level="high", vietnamese_accent="bac", accent_confidence=0.8)
    # Nhân vật 3: Trẻ em miền Bắc
    p3 = CharacterVoiceProfile(speaker_id="spk_3", display_name="Bé Bo", gender="female", age_group="child", pitch_level="high", vietnamese_accent="bac", accent_confidence=0.7)
    # Nhân vật 4: Nữ trầm miền Trung
    p4 = CharacterVoiceProfile(speaker_id="spk_4", display_name="Mệ Huế", gender="female", age_group="middle_aged", pitch_level="low", vietnamese_accent="trung", accent_confidence=0.9)

    profiles = [p1, p2, p3, p4]
    allocations = GlobalVoiceAllocator.allocate_distinct_voices(profiles)

    # 1. Xác nhận 4 nhân vật được gán 4 giọng hoàn toàn KHÁC NHAU
    assigned_presets = list(allocations.values())
    assert len(assigned_presets) == 4
    assert len(set(assigned_presets)) == 4, "Tồn tại giọng bị trùng lặp giữa các nhân vật!"

    # 2. Xác nhận tính tối ưu: Bé Bo (trẻ em) được gán đúng giọng thiếu nhi
    assert "Thiếu Nhi" in allocations["Bé Bo"] or "Em Bé" in allocations["Bé Bo"]

    # 3. Xác nhận Mệ Huế (miền Trung) được gán đúng giọng Nữ Trầm Truyền Cảm (accent=trung)
    assert allocations["Mệ Huế"] == "Nữ Trầm Truyền Cảm - Đọc Truyện Đêm Khuya"

    # 4. Xác nhận kiểm tra độ tách biệt âm học báo cáo thành công không trùng lặp
    dist_rep = verify_speaker_acoustic_distinction(profiles)
    assert dist_rep["all_distinct"] is True
    assert dist_rep["duplicate_count"] == 0


def test_extract_chunk_f0_accuracy():
    """Xác minh hàm extract_chunk_f0 tính toán chính xác cao độ F0 từ âm thanh."""
    from src.ai_engines.speaker_diarization import extract_chunk_f0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # 1. Âm 120 Hz (giọng nam)
    audio_120 = (0.5 * np.sin(2 * np.pi * 120.0 * t)).astype(np.float32)
    f0_male = extract_chunk_f0(audio_120, sr)
    assert f0_male is not None
    assert abs(f0_male - 120.0) < 5.0, f"Kỳ vọng ~120 Hz nhưng đo được {f0_male}"

    # 2. Âm 220 Hz (giọng nữ)
    audio_220 = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    f0_female = extract_chunk_f0(audio_220, sr)
    assert f0_female is not None
    assert abs(f0_female - 220.0) < 5.0, f"Kỳ vọng ~220 Hz nhưng đo được {f0_female}"

    # 3. Khoảng lặng / âm rất nhỏ
    silence = np.zeros(sr, dtype=np.float32)
    assert extract_chunk_f0(silence, sr) is None


def test_autonomous_speaker_count_single_speaker_never_split():
    """Xác minh video 1 người nói (thuyết minh, độc thoại) không bao giờ bị tách nhầm thành nhiều người nói."""
    from src.ai_engines.speaker_diarization import (
        estimate_optimal_clusters,
        agglomerative_cluster_embeddings,
    )

    rng = np.random.default_rng(123)
    dim = 192
    center = rng.normal(0, 1, dim)
    center /= np.linalg.norm(center)

    # 12 câu thoại của 1 người nói duy nhất với độ biến thiên tự nhiên (có câu cảm xúc d ~ 0.40)
    samples = []
    for _ in range(12):
        pert = rng.normal(0, 0.12, dim)
        v = center + pert
        samples.append(v / np.linalg.norm(v))
    embs = np.array(samples, dtype=np.float32)

    # Tự động ước lượng K mà không có bất kỳ cấu hình nào từ người dùng
    k_est = estimate_optimal_clusters(embs, threshold=0.40)
    assert k_est == 1, f"Kỳ vọng 1 người nói nhưng thuật toán đoán {k_est}"

    labels = agglomerative_cluster_embeddings(embs, threshold=0.40)
    assert len(np.unique(labels)) == 1, "Một người nói bị phân mảnh thành nhiều cụm!"
    assert np.all(labels == 0)


def test_autonomous_speaker_count_five_speakers_without_max_limit():
    """Xác minh tự động phát hiện chính xác 5 nhân vật mà không bị giới hạn trần 4 người nói."""
    from src.ai_engines.speaker_diarization import (
        estimate_optimal_clusters,
        agglomerative_cluster_embeddings,
    )

    rng = np.random.default_rng(456)
    dim = 192

    # Tạo 5 tâm cụm trực giao/tách biệt rõ
    centers = []
    for _ in range(5):
        c = rng.normal(0, 1, dim)
        for prev_c in centers:
            c -= np.dot(c, prev_c) * prev_c
        c /= np.linalg.norm(c)
        centers.append(c)

    all_embs = []
    for c in centers:
        for _ in range(4):
            v = c + rng.normal(0, 0.04, dim)
            all_embs.append(v / np.linalg.norm(v))
    embeddings = np.array(all_embs, dtype=np.float32)

    # Không chỉ định max_k (chế độ tự động hoàn toàn)
    k_est = estimate_optimal_clusters(embeddings, max_k=None, threshold=0.40)
    assert k_est == 5, f"Kỳ vọng phát hiện đúng 5 nhân vật nhưng ra {k_est}"

    labels = agglomerative_cluster_embeddings(embeddings, max_k=None, threshold=0.40)
    assert len(np.unique(labels)) == 5, f"Kỳ vọng 5 cụm nhưng thu được {len(np.unique(labels))}"


def test_f0_assisted_distance_prevents_gender_merging():
    """Xác minh khoảng cách F0 ngăn chặn việc gộp nhầm nhân vật nam và nữ khi vector đặc trưng ở vùng ranh giới."""
    from src.ai_engines.speaker_diarization import (
        estimate_optimal_clusters,
        agglomerative_cluster_embeddings,
    )

    rng = np.random.default_rng(789)
    dim = 192

    # Hai tâm cụm ở cự ly ranh giới (cosine distance ~ 0.35)
    c1 = rng.normal(0, 1, dim)
    c1 /= np.linalg.norm(c1)
    c2 = c1 + rng.normal(0, 0.60, dim)
    c2 /= np.linalg.norm(c2)

    # 4 câu thoại của người nam (F0 ~ 110 Hz) và 4 câu của người nữ (F0 ~ 220 Hz)
    embs = []
    f0s = []
    for _ in range(4):
        v = c1 + rng.normal(0, 0.03, dim)
        embs.append(v / np.linalg.norm(v))
        f0s.append(110.0)
    for _ in range(4):
        v = c2 + rng.normal(0, 0.03, dim)
        embs.append(v / np.linalg.norm(v))
        f0s.append(220.0)

    embeddings = np.array(embs, dtype=np.float32)

    # Có F0 hỗ trợ: tách biệt hoàn toàn 2 người nói
    k_est = estimate_optimal_clusters(embeddings, threshold=0.40, sample_f0s=f0s)
    assert k_est == 2, f"Kỳ vọng 2 người nói nhưng ra {k_est}"

    labels = agglomerative_cluster_embeddings(embeddings, threshold=0.40, sample_f0s=f0s)
    assert len(np.unique(labels)) == 2
    assert labels[0] != labels[4]


def test_ui_speaker_count_mode_toggle_and_param_collection():
    """Xác minh giao diện người dùng chuyển đổi chế độ số lượng nhân vật và thu thập tham số chính xác."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if sys.argv else [""])
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    ui = OmniVoiceInterface()

    # Mặc định: Tự động nhận diện hoàn toàn theo giọng nói thực tế (idx = 0)
    assert ui.combo_speaker_count_mode.currentIndex() == 0
    assert ui.lbl_max_spk.isHidden() is True
    assert ui.spin_max_speakers.isHidden() is True

    params = ui._collect_engine_params()
    assert params["speaker_count_mode"] == "auto"
    assert params["max_speakers"] is None
    assert params["exact_speakers"] is None

    # Chuyển sang: Cố định chính xác số lượng nhân vật (idx = 1)
    ui.combo_speaker_count_mode.setCurrentIndex(1)
    assert ui.lbl_max_spk.isHidden() is False
    assert ui.spin_max_speakers.isHidden() is False
    assert "chính xác" in ui.lbl_max_spk.text()
    ui.spin_max_speakers.setValue(3)

    params = ui._collect_engine_params()
    assert params["speaker_count_mode"] == "exact"
    assert params["max_speakers"] == 3
    assert params["exact_speakers"] == 3

    # Chuyển sang: Tự động ước lượng có giới hạn trần (idx = 2)
    ui.combo_speaker_count_mode.setCurrentIndex(2)
    assert ui.lbl_max_spk.isHidden() is False
    assert ui.spin_max_speakers.isHidden() is False
    assert "tối đa" in ui.lbl_max_spk.text()
    ui.spin_max_speakers.setValue(5)

    params = ui._collect_engine_params()
    assert params["speaker_count_mode"] == "capped"
    assert params["max_speakers"] == 5
    assert params["exact_speakers"] is None








