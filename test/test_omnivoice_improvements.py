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
