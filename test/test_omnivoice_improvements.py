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
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr, dtype=np.float32) * 0.5]
    mock_model.create_voice_clone_prompt.return_value = MagicMock()
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
            # Xác nhận rõ rệt chế độ design và đúng instruct của preset đã chọn
            assert call_kw.get("override_mode") == "design"
            assert call_kw.get("override_prompt") is None
            assert call_kw.get("override_ref_audio") is None
            assert call_kw.get("override_instruct") == expected_instruct

        # Kiểm tra trực tiếp tham số truyền vào model.generate()
        assert mock_model.generate.call_count == 2
        for call_item in mock_model.generate.call_args_list:
            gen_kwargs = call_item.kwargs
            assert gen_kwargs.get("instruct") == expected_instruct
            assert "voice_clone_prompt" not in gen_kwargs or gen_kwargs["voice_clone_prompt"] is None
            assert "ref_audio" not in gen_kwargs or gen_kwargs["ref_audio"] is None
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
    from src.ai_engines.omnivoice.models.omnivoice import OmniVoice
    # Kiểm tra phương thức _estimate_target_tokens có floor an toàn cho tiếng Việt
    mock_model = OmniVoice.__new__(OmniVoice)
    mock_estimator = MagicMock()
    mock_estimator.estimate_duration.return_value = 50.0
    mock_model.duration_estimator = mock_estimator

    sample_vietnamese = "Đây là một câu kiểm tra thời lượng tiếng Việt tương đối dài để bảo đảm hệ thống không bao giờ bị cắt cụt các từ cuối cùng."
    tokens = mock_model._estimate_target_tokens(sample_vietnamese, None, None, speed=1.0)
    words = len(sample_vietnamese.split())
    assert tokens >= int(words * 8.5)


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


def test_synthesize_blocks_single_mode_pure_design_no_anchor_clone():
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    engine = TTSEngine(auto_extract_ref=False, mode="design", multi_speaker_mode="single", instruct="male, young adult")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.5]
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
            # Ở chế độ thiết kế giọng đơn nhất, phải chạy thuần túy mode design và KHÔNG dùng prompt clone mỏ neo
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
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.5]
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
            assert call_kw.get("override_mode") == "design"
            assert call_kw.get("override_prompt") is None
            assert call_kw.get("override_ref_audio") is None
            assert call_kw.get("override_instruct") == instruct
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
        )
        engine.load_model = MagicMock()
        mock_model = MagicMock()
        mock_model.generate.return_value = [np.ones(24000, dtype=np.float32) * 0.5]
        engine.model = mock_model

        prof = CharacterVoiceProfile(
            speaker_id="spk_old",
            display_name="Nhân vật cũ",
            method="clone",
            source_audio_path=dummy_sample,
        )
        engine.speaker_profiles = {"Nhân vật cũ": prof}
        engine.current_prompt = MagicMock()  # Giả lập còn sót prompt clone cũ

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
        # Tuyệt đối không được sử dụng lại mẫu clone cũ
        assert call_kw.get("override_mode") == "design"
        assert call_kw.get("override_prompt") is None
        assert call_kw.get("override_ref_audio") is None
        assert call_kw.get("override_instruct") == instruct
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

    # 2. Bước áp dụng chủ đích phải đồng bộ toàn diện cả phụ đề và mọi kiểu hồ sơ nhân vật
    ui._apply_unified_voice_to_all_subtitles()
    assert ui.subtitle_blocks[0].voice_instruct == expected_instruct
    assert ui.subtitle_blocks[0].voice_ref_path == ""
    assert ui.subtitle_blocks[0].speaker == chosen_preset
    assert ui.subtitle_blocks[0].is_speaker_locked is True

    # Kiểm tra đồng bộ hồ sơ nhân vật dạng đối tượng
    assert obj_prof.method == "preset"
    assert obj_prof.preset_name == chosen_preset
    assert obj_prof.voice_instruct == expected_instruct
    assert obj_prof.source_audio_path == ""
    assert getattr(obj_prof, "custom_audio_path", "") == ""
    assert getattr(obj_prof, "voice_clone_prompt_path", "") == ""
    assert getattr(obj_prof, "ref_text", "") == ""
    assert obj_prof.status == "locked"

    # Kiểm tra đồng bộ hồ sơ nhân vật dạng từ điển
    assert dict_prof["method"] == "preset"
    assert dict_prof["preset_name"] == chosen_preset
    assert dict_prof["voice_instruct"] == expected_instruct
    assert dict_prof["instruct"] == expected_instruct
    assert dict_prof["source_audio_path"] == ""
    assert dict_prof["ref_audio_path"] == ""
    assert dict_prof["custom_audio_path"] == ""
    assert dict_prof["voice_clone_prompt_path"] == ""
    assert dict_prof["ref_text"] == ""
    assert dict_prof["status"] == "locked"

    # 3. Tham số thu thập phải phản ánh đúng chế độ design, instruct và đồng bộ profile
    params = ui._collect_engine_params()
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
        normalize_text=False,
    )
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr, dtype=np.float32) * 0.5]
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

        # Kiểm tra 3 lượt gọi đến model.generate()
        assert mock_model.generate.call_count == 3
        calls = mock_model.generate.call_args_list

        # Lượt 1: Phải đọc translated_text "Lời dịch tiếng Việt một"
        assert calls[0].kwargs.get("text") == "Lời dịch tiếng Việt một"
        assert calls[0].kwargs.get("instruct") == expected_instruct
        assert "voice_clone_prompt" not in calls[0].kwargs or calls[0].kwargs["voice_clone_prompt"] is None

        # Lượt 2: Phải đọc text gốc "Lời thoại gốc không có bản dịch"
        assert calls[1].kwargs.get("text") == "Lời thoại gốc không có bản dịch"
        assert calls[1].kwargs.get("instruct") == expected_instruct
        assert "voice_clone_prompt" not in calls[1].kwargs or calls[1].kwargs["voice_clone_prompt"] is None

        # Lượt 3: Phải lùi về text gốc "Lời thoại với bản dịch khoảng trắng"
        assert calls[2].kwargs.get("text") == "Lời thoại với bản dịch khoảng trắng"
        assert calls[2].kwargs.get("instruct") == expected_instruct
        assert "voice_clone_prompt" not in calls[2].kwargs or calls[2].kwargs["voice_clone_prompt"] is None
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
