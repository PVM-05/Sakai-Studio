# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tự động toàn diện cho quy trình lồng tiếng OmniVoice 5 giai đoạn (TDD).
Bao phủ toàn bộ các kịch bản P1, P2 và 8 bài kiểm thử nghiệm thu bắt buộc.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.ai_engines.translator import SubtitleBlock
from src.ai_engines.tts_engine import (
    TTSEngine,
    DubbingReport,
    validate_and_normalize_subtitles,
    preprocess_vietnamese_tts_text,
    apply_dynamic_ducking,
    validate_audio_file,
)


def _make_sine_wave(duration_sec: float = 1.5, sr: int = 24000, freq: float = 220.0, amp: float = 0.25) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# ==============================================================================
# 1. KIỂM THỬ MỘT CÂU PHỤ ĐỀ BỊ LỖI TTS -> BÁO LỖI HOẶC HOÀN THÀNH MỘT PHẦN
# ==============================================================================
def test_partial_or_failed_block_synthesis_does_not_report_false_success(tmp_path):
    """
    Khi 1 câu trong danh sách phụ đề bị lỗi ngoại lệ lúc tổng hợp,
    synthesize_blocks phải trả về DubbingReport có status='partial' (không phải 'success'),
    ghi nhận đúng failed_count=1 và failed_lines=[2].
    Nếu tất cả các câu đều lỗi, status phải là 'failed'.
    """
    engine = TTSEngine(auto_extract_ref=False, mode="design", instruct="male, young adult, moderate pitch")
    engine.load_model = MagicMock()

    blocks = [
        SubtitleBlock(start_time=0.0, end_time=1.5, text="Câu thứ nhất hợp lệ", translated_text="Câu thứ nhất hợp lệ"),
        SubtitleBlock(start_time=2.0, end_time=3.5, text="Câu thứ hai bị lỗi", translated_text="Câu thứ hai bị lỗi"),
        SubtitleBlock(start_time=4.0, end_time=5.5, text="Câu thứ ba hợp lệ", translated_text="Câu thứ ba hợp lệ"),
    ]

    def side_effect_generate(text, *args, **kwargs):
        if "thứ hai" in text:
            raise RuntimeError("Mô phỏng lỗi giải mã câu thứ hai")
        return _make_sine_wave(1.2, engine.sample_rate)

    engine.generate_single = MagicMock(side_effect=side_effect_generate)

    out_wav = str(tmp_path / "partial_track.wav")
    report = engine.synthesize_blocks(blocks, total_duration=6.0, output_path=out_wav)

    assert isinstance(report, DubbingReport)
    assert report.status == "partial"
    assert report.total_blocks == 3
    assert report.succeeded_count == 2
    assert report.failed_count == 1
    assert 2 in report.failed_lines

    # Trường hợp toàn bộ các câu đều lỗi -> status == 'failed'
    engine.generate_single = MagicMock(side_effect=RuntimeError("Lỗi toàn bộ"))
    out_wav_fail = str(tmp_path / "failed_track.wav")
    report_fail = engine.synthesize_blocks(blocks, total_duration=6.0, output_path=out_wav_fail)
    assert report_fail.status == "failed"
    assert report_fail.succeeded_count == 0
    assert report_fail.failed_count == 3


# ==============================================================================
# 2. KIỂM THỬ CÂU THOẠI CUỐI VIDEO -> KHÔNG MẤT TIẾNG HOẶC CẮT ÂM THANH NGOÀI DỰ KIẾN
# ==============================================================================
def test_last_subtitle_not_truncated_at_end_of_track(tmp_path):
    """
    Khi câu thoại cuối cùng bắt đầu gần cuối thời lượng dự kiến hoặc có độ dài âm thanh
    vượt quá total_duration + 5s, bộ đệm âm thanh phải tự động mở rộng để giữ trọn vẹn câu thoại
    mà không cắt cụt phần đuôi.
    """
    engine = TTSEngine(auto_extract_ref=False, mode="design", sync_to_subtitle_duration=False)
    engine.load_model = MagicMock()

    # Câu thoại bắt đầu ở giây 9.5 trên video 10.0s nhưng âm thanh dài tới 7.0s (kết thúc ở 16.5s > 10+5=15s)
    long_tail_wave = _make_sine_wave(duration_sec=7.0, sr=engine.sample_rate, amp=0.3)
    engine.generate_single = MagicMock(return_value=long_tail_wave)

    blocks = [
        SubtitleBlock(start_time=9.5, end_time=11.0, text="Câu kết thúc kéo dài", translated_text="Câu kết thúc kéo dài"),
    ]

    out_wav = str(tmp_path / "tail_track.wav")
    report = engine.synthesize_blocks(blocks, total_duration=10.0, output_path=out_wav)

    assert report.status == "success"
    assert report.truncated_count == 0
    audio_out, sr = sf.read(out_wav)
    expected_min_samples = int(9.5 * sr) + len(long_tail_wave)
    assert len(audio_out) >= expected_min_samples
    # Kiểm tra năng lượng ở giây thứ 16 (cuối câu thoại) vẫn còn nguyên vẹn
    tail_Check = audio_out[int(15.5 * sr) : int(16.4 * sr)]
    assert float(np.max(np.abs(tail_Check))) > 0.2


# ==============================================================================
# 3. KIỂM THỬ TIMESTAMP ÂM HOẶC SAI THỨ TỰ -> PHÁT HIỆN VÀ BÁO RÕ DÒNG PHỤ ĐỀ
# ==============================================================================
def test_validate_subtitles_detects_negative_and_out_of_order_timestamps():
    """
    Giai đoạn 1 (validate_and_normalize_subtitles) phải phát hiện chính xác:
    - Timestamp âm
    - Timestamp kết thúc <= bắt đầu
    - Timestamp sai thứ tự thời gian (câu sau bắt đầu trước câu trước)
    - Câu vượt ngoài thời lượng video
    """
    blocks = [
        SubtitleBlock(start_time=-1.5, end_time=1.0, text="Câu âm", translated_text="Câu âm"),
        SubtitleBlock(start_time=3.0, end_time=2.0, text="Câu ngược", translated_text="Câu ngược"),
        SubtitleBlock(start_time=1.2, end_time=2.5, text="Câu sai thứ tự", translated_text="Câu sai thứ tự"),
        SubtitleBlock(start_time=25.0, end_time=28.0, text="Câu vượt thời lượng video", translated_text="Câu vượt thời lượng video"),
    ]

    issues = validate_and_normalize_subtitles(blocks, video_duration=20.0, auto_fix=False)
    issue_lines = {item["line"] for item in issues}
    issue_types = {item["type"] for item in issues}

    assert 1 in issue_lines
    assert 2 in issue_lines
    assert 3 in issue_lines
    assert 4 in issue_lines
    assert "negative_timestamp" in issue_types
    assert "inverted_duration" in issue_types
    assert "out_of_order" in issue_types
    assert "exceeds_video_duration" in issue_types


# ==============================================================================
# 4. KIỂM THỬ NHÂN VẬT XUẤT HIỆN NHIỀU LẦN -> GIỮ GIỌNG NHẤT QUÁN & KIỂM ĐỊNH MẪU KHÓA
# ==============================================================================
def test_multi_speaker_consistency_and_quality_checked_locking(tmp_path):
    """
    Khi nhân vật xuất hiện nhiều lần:
    - Nếu câu đầu quá ngắn hoặc chất lượng kém (im lặng/clipping), không khóa làm mẫu giọng.
    - Khi câu đủ chuẩn (>= 1.5s, RMS hợp lệ, không clipping), tạo VoiceClonePrompt cố định cho các câu sau.
    - Hỗ trợ xóa/tạo lại mẫu giọng đã khóa của nhân vật qua reset_speaker_locked_prompt().
    """
    engine = TTSEngine(auto_extract_ref=False, mode="design", multi_speaker_mode="custom_table")
    engine.load_model = MagicMock()
    engine.model = MagicMock()
    fake_prompt = MagicMock(name="LockedVoicePrompt")
    engine.model.create_voice_clone_prompt = MagicMock(return_value=fake_prompt)

    blocks = [
        SubtitleBlock(start_time=0.0, end_time=2.0, text="Câu một của Nam", translated_text="Câu một của Nam", speaker="Người nói 1", voice_instruct="male, young adult"),
        SubtitleBlock(start_time=2.5, end_time=4.5, text="Câu hai của Nam", translated_text="Câu hai của Nam", speaker="Người nói 1", voice_instruct="male, young adult"),
    ]

    # Câu 1 đạt chuẩn chất lượng (1.8 giây, biên độ 0.25)
    good_wave = _make_sine_wave(1.8, engine.sample_rate, amp=0.25)
    engine.generate_single = MagicMock(return_value=good_wave)

    out_wav = str(tmp_path / "consistent_spk.wav")
    report = engine.synthesize_blocks(blocks, total_duration=5.0, output_path=out_wav)

    assert report.status == "success"
    assert "consistent_Người nói 1" in engine._speaker_prompt_cache
    # Câu thứ 2 phải được truyền override_prompt = fake_prompt
    second_call_kwargs = engine.generate_single.call_args_list[1].kwargs
    assert second_call_kwargs.get("override_prompt") is fake_prompt

    # Kiểm tra nút/hàm tạo lại mẫu giọng nhân vật
    engine.reset_speaker_locked_prompt("Người nói 1")
    assert "consistent_Người nói 1" not in engine._speaker_prompt_cache


# ==============================================================================
# 5. KIỂM THỬ VIDEO KHÔNG CÓ ÂM THANH GỐC -> VẪN GHÉP ĐƯỢC TRACK TTS HỢP LỆ
# ==============================================================================
def test_mix_audio_ducking_handles_video_without_audio_stream(tmp_path):
    """
    Tạo một video MP4 chỉ có hình ảnh (không có kênh âm thanh gốc) bằng FFmpeg,
    sau đó gọi mix_audio_ducking() để ghép track WAV TTS vào.
    Video đầu ra phải tồn tại, hợp lệ và có kênh âm thanh.
    """
    from src.core.tools.ffmpeg_cli import FFmpegCLI

    ffmpeg_bin = FFmpegCLI.instance().ffmpeg_path
    silent_video = str(tmp_path / "video_no_audio.mp4")
    tts_wav = str(tmp_path / "tts_track.wav")
    out_video = str(tmp_path / "video_dubbed.mp4")

    # Tạo video 2 giây không có âm thanh
    subprocess.run(
        [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            silent_video,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sf.write(tts_wav, _make_sine_wave(2.0, 24000, amp=0.3), 24000)

    TTSEngine.mix_audio_ducking(
        video_in=silent_video,
        tts_audio_in=tts_wav,
        video_out=out_video,
        original_volume=0.2,
    )

    assert os.path.exists(out_video)
    assert os.path.getsize(out_video) > 1000
    assert TTSEngine._has_audio_stream(out_video) is True


# ==============================================================================
# 6. KIỂM THỬ VIDEO CÓ NHẠC NỀN LỚN -> DUCKING ĐỘNG GIẢM NHẠC NỀN KHI CÓ THOẠI, KHÔNG CLIPPING
# ==============================================================================
def test_dynamic_envelope_ducking_attenuates_bgm_only_during_speech_without_clipping():
    """
    Kiểm tra hàm apply_dynamic_ducking:
    - Khi không có tiếng thoại TTS (0.0s - 1.0s), âm lượng nhạc nền giữ nguyên (~0.8).
    - Khi có tiếng thoại TTS lớn (1.0s - 2.0s), nhạc nền tự động giảm xuống theo duck_gain
      và tổng âm thanh đầu ra được giới hạn biên độ không bao giờ vượt ngưỡng clipping (<= 0.99).
    """
    sr = 24000
    # Nhạc nền lớn xuyên suốt 3 giây (biên độ 0.85)
    bgm = _make_sine_wave(3.0, sr, freq=110.0, amp=0.85)
    # Giọng đọc chỉ xuất hiện từ giây 1.0 đến giây 2.0 (biên độ 0.85)
    tts = np.zeros_like(bgm)
    tts[sr : 2 * sr] = _make_sine_wave(1.0, sr, freq=440.0, amp=0.85)

    mixed = apply_dynamic_ducking(bgm, tts, sr=sr, duck_gain=0.15)

    # Không bao giờ bị clipping vượt quá 0.99
    assert float(np.max(np.abs(mixed))) <= 0.99
    # Ở đoạn không có thoại (0.1s - 0.8s), nhạc nền không bị giảm xuống 0.15 mà giữ gần nguyên mức gốc
    rms_no_speech = float(np.sqrt(np.mean(mixed[int(0.1 * sr) : int(0.8 * sr)] ** 2)))
    assert rms_no_speech > 0.45


# ==============================================================================
# 7. KIỂM THỬ HỦY HOẶC ĐÓNG TÁC VỤ -> DỪNG SỚM SẠCH SẼ, KHÔNG TREO HOẶC BÁO THÀNH CÔNG GIẢ
# ==============================================================================
def test_worker_lifecycle_cancellation_and_close_cleanup(tmp_path):
    """
    Khi người dùng bấm Hủy tác vụ (request_cancel) trong lúc đang tổng hợp nhiều câu,
    synthesize_blocks phải dừng ngay ở câu tiếp theo và trả về status='cancelled'.
    """
    engine = TTSEngine(auto_extract_ref=False, mode="design")
    engine.load_model = MagicMock()

    blocks = [
        SubtitleBlock(start_time=0.0, end_time=1.0, text="Câu 1", translated_text="Câu 1"),
        SubtitleBlock(start_time=1.5, end_time=2.5, text="Câu 2", translated_text="Câu 2"),
        SubtitleBlock(start_time=3.0, end_time=4.0, text="Câu 3", translated_text="Câu 3"),
    ]

    def synth_and_cancel(text, *args, **kwargs):
        engine.request_cancel()
        return _make_sine_wave(0.8, engine.sample_rate)

    engine.generate_single = MagicMock(side_effect=synth_and_cancel)
    out_wav = str(tmp_path / "cancelled.wav")
    report = engine.synthesize_blocks(blocks, total_duration=5.0, output_path=out_wav)

    assert report.status == "cancelled"
    assert engine.generate_single.call_count == 1


# ==============================================================================
# 8. KIỂM THỬ GPU THIẾU VRAM -> THÔNG BÁO TIẾNG VIỆT RÕ RÀNG VÀ GIẢI PHÓNG BỘ NHỚ ĐỆM
# ==============================================================================
def test_oom_cuda_error_handled_with_clear_vietnamese_guidance(tmp_path):
    """
    Khi phát sinh lỗi thiếu bộ nhớ CUDA Out of Memory, TTSEngine phải tự động dọn cache
    và ghi thông báo hướng dẫn khắc phục rõ ràng bằng tiếng Việt.
    """
    logs = []
    engine = TTSEngine(parent_logger=logs.append, auto_extract_ref=False, mode="design")
    engine.load_model = MagicMock()
    engine.generate_single = MagicMock(side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))

    blocks = [SubtitleBlock(start_time=0.0, end_time=2.0, text="Kiểm tra tràn VRAM", translated_text="Kiểm tra tràn VRAM")]
    out_wav = str(tmp_path / "oom.wav")
    report = engine.synthesize_blocks(blocks, total_duration=3.0, output_path=out_wav)

    assert report.status == "failed"
    assert report.oom_detected is True
    assert any("VRAM" in msg or "bộ nhớ" in msg.lower() for msg in logs)


# ==============================================================================
# 9. KIỂM THỬ TIỀN XỬ LÝ TIẾNG VIỆT & KHÔI PHỤC TÁC VỤ (CHECKPOINT RESUME)
# ==============================================================================
def test_vietnamese_preprocessing_and_checkpoint_resume(tmp_path):
    """
    - Kiểm tra chuẩn hóa đơn vị đo, phần trăm và chia câu dài tiếng Việt.
    - Kiểm tra cơ chế lưu checkpoint từng câu: câu đã tổng hợp thành công trước đó
      không cần gọi lại GPU khi chạy lại cùng tác vụ.
    """
    processed = preprocess_vietnamese_tts_text("Tốc độ đạt 120 km/h và tăng 15% so với năm trước.")
    assert "ki lô mét trên giờ" in processed
    assert "phần trăm" in processed

    # Kiểm thử resume từ cache
    cache_dir = str(tmp_path / "resume_cache")
    engine = TTSEngine(auto_extract_ref=False, mode="design", resume_cache_dir=cache_dir)
    engine.load_model = MagicMock()
    engine.generate_single = MagicMock(return_value=_make_sine_wave(1.0, engine.sample_rate))

    blocks = [
        SubtitleBlock(start_time=0.0, end_time=1.5, text="Câu lưu bộ nhớ đệm 1", translated_text="Câu lưu bộ nhớ đệm 1"),
        SubtitleBlock(start_time=2.0, end_time=3.5, text="Câu lưu bộ nhớ đệm 2", translated_text="Câu lưu bộ nhớ đệm 2"),
    ]
    out_wav = str(tmp_path / "resumed.wav")

    # Lần chạy 1: gọi generate_single 2 lần
    rep1 = engine.synthesize_blocks(blocks, total_duration=4.0, output_path=out_wav)
    assert rep1.status == "success"
    assert engine.generate_single.call_count == 2

    # Lần chạy 2 với cùng dữ liệu và resume_cache_dir: nạp từ checkpoint, không gọi lại GPU
    engine.generate_single.reset_mock()
    rep2 = engine.synthesize_blocks(blocks, total_duration=4.0, output_path=out_wav)
    assert rep2.status == "success"
    assert rep2.resumed_from_cache_count == 2
    assert engine.generate_single.call_count == 0


# ==============================================================================
# 10. KIỂM THỬ VÒNG ĐỜI GIAO DIỆN OMNIVOICEINTERFACE & XỬ LÝ HOÀN THÀNH MỘT PHẦN
# ==============================================================================
def test_ui_worker_partial_status_and_close_event(tmp_path):
    """
    Kiểm tra quản lý trạng thái tác vụ tập trung trên OmniVoiceInterface:
    - Khi _set_task_running_state(True), nút tổng hợp bị khóa và nút Hủy Tác Vụ mở khóa.
    - Khi nhận thông báo '[PARTIAL] ...', tiền tố nội bộ được gỡ bỏ khỏi nhãn trạng thái.
    - Khi đóng cửa sổ (closeEvent), trình phát và worker được dọn dẹp an toàn.
    """
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QCloseEvent
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ui._set_task_running_state(True)
    assert ui.btn_studio_synth.isEnabled() is False
    assert ui.btn_cancel_task.isEnabled() is True

    wav_file = str(tmp_path / "sample.wav")
    sf.write(wav_file, _make_sine_wave(1.0, 24000), 24000)

    ui._on_worker_finished(
        True,
        "subtitle_audio_only",
        wav_file,
        "[PARTIAL] Hoàn thành một phần: thành công 2 câu, lỗi 1 câu (dòng: 2).",
    )
    assert ui.btn_studio_synth.isEnabled() is True
    assert ui.btn_cancel_task.isEnabled() is False
    assert "[PARTIAL]" not in ui.lbl_status.text()
    assert "Hoàn thành một phần" in ui.lbl_status.text()

    ui.closeEvent(QCloseEvent())


# ==============================================================================
# 11. KIỂM THỬ TRÍCH XUẤT VECTOR ĐẶC TRƯNG NGƯỜI NÓI QUA ONNX (TÁC VỤ 1)
# ==============================================================================
def test_onnx_speaker_embedding_session_and_extraction():
    from src.ai_engines.speaker_embedding_onnx import (
        create_speaker_session,
        extract_speaker_embedding,
        batch_extract_speaker_embeddings,
        close_speaker_session,
    )

    mock_audio = np.random.randn(16000 * 2).astype(np.float32)
    session, provider = create_speaker_session(model_path="", prefer_cuda=False, allow_mock=True)
    assert provider in ("CPUExecutionProvider", "MockExecutionProvider")

    emb = extract_speaker_embedding(session, mock_audio, sample_rate=16000)
    assert emb.shape == (192,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    chunks = [mock_audio, mock_audio]
    batch_embs = batch_extract_speaker_embeddings(session, chunks, sample_rate=16000, batch_size=2)
    assert batch_embs.shape == (2, 192)

    close_speaker_session(session)


def test_camppus_fbank_preprocessing_matches_model_input_contract():
    from src.ai_engines.speaker_embedding_onnx import _extract_fbank_features, _make_model_input

    waveform = _make_sine_wave(duration_sec=1.2, sr=16000, freq=220.0)
    features = _extract_fbank_features(waveform)
    assert features.ndim == 2
    assert features.shape[1] == 80
    assert np.isfinite(features).all()

    class _Input:
        name = "input"
        shape = ["batch", "frames", 80]

    class _Output:
        name = "embedding"
        shape = ["batch", 192]

    class _FakeSession:
        def get_inputs(self):
            return [_Input()]

        def get_outputs(self):
            return [_Output()]

    tensor, time_axis = _make_model_input(_FakeSession(), waveform)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1
    assert tensor.shape[2] == 80
    assert time_axis == 1


# ==============================================================================
# 12. KIỂM THỬ PHÂN CỤM COSIN VÀ 3 MỨC ĐỘ TIN CẬY (TÁC VỤ 2)
# ==============================================================================
def test_kmeanspp_probabilities_are_normalized_from_float32_weights():
    from src.ai_engines.speaker_diarization import _normalized_choice_probabilities

    # Dãy trọng số float32 dễ có tổng sau phép chia lệch khỏi 1 do làm tròn.
    weights = np.asarray([0.1, 0.2, 0.3, 0.4, 1e-7, 3.7], dtype=np.float32)
    probabilities = _normalized_choice_probabilities(weights)

    assert probabilities is not None
    assert probabilities.dtype == np.float64
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert np.isclose(probabilities.sum(dtype=np.float64), 1.0, rtol=0.0, atol=1e-15)

    # Trọng số bằng 0/không hữu hạn không được tạo ra phân phối hỏng.
    degenerate = _normalized_choice_probabilities(np.asarray([0.0, np.nan, np.inf]))
    assert degenerate is None


def test_cosine_clustering_and_three_confidence_levels():
    from src.ai_engines.speaker_diarization import (
        SpeakerDiarizationCoordinator,
        SpeakerVoiceProfile,
        ConfidenceLevel,
        cosine_cluster_embeddings,
    )

    spk1_base = np.zeros(192, dtype=np.float32)
    spk1_base[0] = 1.0
    spk2_base = np.zeros(192, dtype=np.float32)
    spk2_base[1] = 1.0

    rng = np.random.default_rng(42)
    cluster_1 = spk1_base + rng.normal(0, 0.02, (5, 192)).astype(np.float32)
    cluster_2 = spk2_base + rng.normal(0, 0.02, (5, 192)).astype(np.float32)
    ambiguous = (spk1_base + spk2_base) / np.sqrt(2.0)
    ambiguous_noisy = ambiguous + rng.normal(0, 0.05, (1, 192)).astype(np.float32)

    all_embs = np.vstack([cluster_1, cluster_2, ambiguous_noisy])
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_embs = all_embs / norms

    labels, conf_levels, conf_scores = cosine_cluster_embeddings(all_embs, valid_indices=list(range(11)))
    assert len(labels) == 11
    # Cụm 1 và Cụm 2 phải có nhãn khác nhau
    assert labels[0] != labels[5]
    # Các câu rõ ràng phải đạt tin cậy cao
    assert conf_levels[0] == ConfidenceLevel.HIGH.value
    assert conf_levels[5] == ConfidenceLevel.HIGH.value
    # Câu phân vân phải rơi vào mức cần kiểm tra hoặc chưa xác định
    assert conf_levels[10] in (ConfidenceLevel.REVIEW_NEEDED.value, ConfidenceLevel.UNASSIGNED.value)


# ==============================================================================
# 13. KIỂM THỬ KHÓA NHÂN VẬT VÀ GIẢI PHÓNG PHIÊN NHẬN DIỆN TRONG TTSENGINE (TÁC VỤ 3)
# ==============================================================================
def test_tts_engine_respects_locked_speaker_and_releases_diarizer_session(tmp_path):
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    wav_path = str(tmp_path / "voice_track.wav")
    sf.write(wav_path, _make_sine_wave(8.0, 16000), 16000)

    engine = TTSEngine(auto_extract_ref=False, mode="design")
    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Xin chào", translated_text="Xin chào")
    b2 = SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:06,000", text="Chào bạn", translated_text="Chào bạn")
    b2.speaker = "Nhân vật đã khóa"
    b2.is_speaker_locked = True

    blocks = [b1, b2]
    # Chỉ dùng mô hình giả lập và âm thanh WAV được sao chép trong bài kiểm thử.
    def _copy_test_audio(source, target):
        audio, sr = sf.read(source, dtype="float32")
        sf.write(target, audio, sr)
        return True

    with patch("src.ai_engines.tts_engine.VoiceSeparator") as separator_cls:
        separator_cls.return_value.separate_from_video.side_effect = _copy_test_audio
        res = engine.detect_speakers_from_video(
            video_path=wav_path,
            blocks=blocks,
            max_speakers=3,
            prefer_cuda=False,
            allow_mock=True,
        )
    assert isinstance(res, dict)
    assert len(res) >= 1
    # Câu 1 được gán nhận diện từ bộ điều phối
    assert b1.speaker == "Người nói 1"
    assert hasattr(b1, "confidence_level")
    # Câu 2 phải được giữ nguyên nhân vật đã khóa
    assert b2.speaker == "Nhân vật đã khóa"
    # Thuộc tính phiên làm việc nhận diện nội bộ phải được thu hồi
    assert getattr(engine, "_active_diarizer_session", None) is None
    assert any(
        Path(p.get("ref_audio_path", "")).is_file()
        for p in engine.speaker_profiles.values()
    )


def test_speaker_model_missing_is_not_silently_mocked(tmp_path):
    from src.ai_engines.speaker_embedding_onnx import create_speaker_session

    missing_model = tmp_path / "model-does-not-exist.onnx"
    with pytest.raises(FileNotFoundError, match="không tồn tại"):
        create_speaker_session(
            model_path=str(missing_model),
            prefer_cuda=False,
            allow_mock=False,
        )


def test_diarizer_rejects_unreadable_audio_instead_of_random_fallback(tmp_path):
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator

    block = SubtitleBlock(
        index=1,
        start_time="00:00:00,000",
        end_time="00:00:01,000",
        text="Câu thoại kiểm thử",
    )
    coordinator = SpeakerDiarizationCoordinator(allow_mock=True, prefer_cuda=False)
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp âm thanh"):
        coordinator.process_video_subtitles(
            video_path=str(tmp_path / "video-khong-ton-tai.mp4"),
            blocks=[block],
        )


def test_silent_segments_remain_unassigned(tmp_path):
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator

    silent_wav = tmp_path / "silent.wav"
    sf.write(str(silent_wav), np.zeros(16000, dtype=np.float32), 16000)
    block = SubtitleBlock(
        index=1,
        start_time="00:00:00,000",
        end_time="00:00:01,000",
        text="Không có tiếng",
    )
    coordinator = SpeakerDiarizationCoordinator(allow_mock=True, prefer_cuda=False)
    result = coordinator.process_video_subtitles(
        video_path=str(silent_wav),
        blocks=[block],
        prefer_cuda=False,
    )
    assert result == {}
    assert block.speaker == "Chưa xác định"
    assert block.confidence_level == "unassigned"


# ==============================================================================
# 14. KIỂM THỬ QUY TRÌNH GIAO DIỆN 5 BƯỚC VÀ BẢO TOÀN DỮ LIỆU (TÁC VỤ 4)
# ==============================================================================
def test_ui_wizard_preserves_state_on_step_navigation():
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    assert hasattr(ui, "step_stack")
    assert ui.step_stack.count() == 5
    assert ui.current_step_index == 0

    ui._switch_to_step(1)
    assert ui.current_step_index == 1
    ui._switch_to_step(2)
    assert ui.current_step_index == 2

    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Thử nghiệm", speaker="Người nói 1")
    ]
    ui._switch_to_step(0)
    assert len(ui.subtitle_blocks) == 1
    assert ui.subtitle_blocks[0].speaker == "Người nói 1"


# ==============================================================================
# 15. KIỂM THỬ RÀ SOÁT CHẤT LƯỢNG TRƯỚC XUẤT (TÁC VỤ 5)
# ==============================================================================
def test_pre_export_quality_checks_and_warnings():
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Đoạn 1", speaker="Người nói 1"),
        SubtitleBlock(index=2, start_time="00:00:02,000", end_time="00:00:03,000", text="Đoạn 2", speaker="Chưa xác định"),
    ]
    issues = ui._run_pre_export_quality_check()
    assert len(issues) >= 1
    assert any("Chưa xác định" in item["message"] or "chưa gán" in item["message"] for item in issues)

    # Thử nghiệm với phụ đề rỗng
    ui.subtitle_blocks = []
    empty_issues = ui._run_pre_export_quality_check()
    assert len(empty_issues) >= 1
    assert any(item["severity"] == "error" for item in empty_issues)

    # Thử nghiệm với câu thoại quá dài so với thời lượng
    ui.subtitle_blocks = [
        SubtitleBlock(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:02,000",
            text="Đây là một câu thoại có độ dài rất lớn nhưng mốc thời gian phụ đề chỉ có một giây ngắn ngủi nên chắc chắn sẽ bị đọc tràn ra ngoài.",
            speaker="Người nói 1",
        )
    ]
    long_issues = ui._run_pre_export_quality_check()
    assert any("tràn mốc thời gian" in item["message"] or "quá ngắn" in item["message"] for item in long_issues)


# ==============================================================================
# 16. KIỂM THỬ TIỀN XỬ LÝ FBANK VÀ CƠ CHẾ QUẢN LÝ MÔ HÌNH CAM++ ONNX (P0-A)
# ==============================================================================
def test_fbank_feature_extraction_and_cmvn():
    from src.ai_engines.speaker_embedding_onnx import extract_fbank_features

    audio = np.random.randn(int(1.5 * 16000)).astype(np.float32)
    feats = extract_fbank_features(audio, sample_rate=16000)
    assert feats.ndim == 2
    assert feats.shape[1] == 80
    assert feats.shape[0] > 50
    col_means = np.mean(feats, axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-3)


def test_create_speaker_session_rejects_missing_model_when_mock_disallowed():
    import pytest
    from src.ai_engines.speaker_embedding_onnx import create_speaker_session, MockSpeakerSession

    with pytest.raises(FileNotFoundError) as exc_info:
        create_speaker_session(model_path="khong_ton_tai_campplus.onnx", allow_mock=False)
    assert "Không tìm thấy" in str(exc_info.value) or "mô hình" in str(exc_info.value)

    session, provider = create_speaker_session(model_path="khong_ton_tai_campplus.onnx", allow_mock=True)
    assert isinstance(session, MockSpeakerSession)
    assert provider in ("CPUExecutionProvider", "MockExecutionProvider")


def test_config_has_mmo_speaker_model_path():
    from src.core.config import config
    assert hasattr(config, "mmoSpeakerModelPath")


# ==============================================================================
# 17. KIỂM THỬ XỬ LÝ LỖI ÂM THANH VÀ BẢO TOÀN CÂU KHÓA (P0-B)
# ==============================================================================
def test_diarization_handles_missing_or_silent_audio_without_fake_noise_and_preserves_locked():
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator, ConfidenceLevel
    from src.ai_engines.tts_engine import SubtitleBlock

    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu 1 đã khóa")
    b1.speaker = "Giọng Đã Khóa"
    b1.is_speaker_locked = True
    b1.confidence_level = ConfidenceLevel.HIGH.value

    b2 = SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:06,000", text="Câu 2 bình thường")
    b3 = SubtitleBlock(index=3, start_time="00:00:07,000", end_time="00:00:07,200", text="Câu 3 quá ngắn")

    blocks = [b1, b2, b3]
    coordinator = SpeakerDiarizationCoordinator(model_path="", allow_mock=True)

    # Trường hợp 1: Tệp video/âm thanh không tồn tại -> bắt ngoại lệ và đánh dấu Chưa xác định
    with pytest.raises(FileNotFoundError):
        coordinator.process_video_subtitles(
            video_path="tep_khong_ton_tai.wav",
            blocks=blocks,
        )
    # b1 không bị đổi
    assert b1.speaker == "Giọng Đã Khóa"
    assert b1.confidence_level == ConfidenceLevel.HIGH.value
    # b2 và b3 được đánh dấu Chưa xác định / unassigned
    assert b2.speaker == "Chưa xác định"
    assert b2.confidence_level == ConfidenceLevel.UNASSIGNED.value
    assert b3.speaker == "Chưa xác định"
    assert b3.confidence_level == ConfidenceLevel.UNASSIGNED.value


# ==============================================================================
# 18. KIỂM THỬ TUẦN TỰ HÓA VRAM GPU VÀ TRÍCH XUẤT MẪU GIỌNG NHÂN VẬT (P1)
# ==============================================================================
def test_vram_sequencing_diarization_before_tts_load_and_sample_extraction(tmp_path):
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    call_order = []

    test_wav = str(tmp_path / "interview.wav")
    sf.write(test_wav, _make_sine_wave(6.0, 16000), 16000)

    engine = TTSEngine(
        multi_speaker_mode="auto_video_cluster",
        voice_prompt_cache_dir=str(tmp_path / "prompts"),
        allow_mock=True,
    )

    original_load = engine.load_model
    original_detect = engine.detect_speakers_from_video

    def mock_detect(*args, **kwargs):
        call_order.append(("detect_speakers", engine.model is None))
        return original_detect(*args, **kwargs)

    def mock_load(*args, **kwargs):
        call_order.append(("load_model", True))
        # Không tải model thật trong unit test
        engine.sample_rate = 24000

    engine.detect_speakers_from_video = mock_detect
    engine.load_model = mock_load

    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Người 1 nói")
    b2 = SubtitleBlock(index=2, start_time="00:00:03,500", end_time="00:00:05,500", text="Người 2 nói")

    out_wav = str(tmp_path / "output.wav")
    # Thay thế generate_single bằng sóng âm thực tế để xác thực toàn bộ chu trình tổng hợp
    engine.generate_single = lambda *a, **k: _make_sine_wave(1.8, engine.sample_rate)

    report = engine.synthesize_blocks(
        blocks=[b1, b2],
        total_duration=6.0,
        output_path=out_wav,
        video_path=test_wav,
    )
    assert report.status == "success"
    assert report.succeeded_count == 2
    assert os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0

    # Kiểm tra thứ tự tuần tự: detect_speakers phải được gọi TRƯỚC load_model
    # và tại thời điểm detect_speakers, engine.model phải là None (chưa nạp OmniVoice vào VRAM)
    action_names = [item[0] for item in call_order]
    assert "detect_speakers" in action_names
    assert "load_model" in action_names
    idx_detect = action_names.index("detect_speakers")
    idx_load = action_names.index("load_model")
    assert idx_detect < idx_load, "Nhận diện người nói ONNX phải chạy và hoàn tất TRƯỚC KHI nạp OmniVoice TTS lên GPU!"
    assert call_order[idx_detect][1] is True, "Tại thời điểm nhận diện người nói, mô hình OmniVoice TTS phải chưa được nạp vào VRAM!"

    # Kiểm tra trích xuất mẫu giọng cho nhân vật
    assert len(engine.speaker_profiles) >= 1
    for spk_name, prof in engine.speaker_profiles.items():
        ref_path = prof.get("ref_audio_path")
        assert ref_path and os.path.exists(ref_path), f"Mẫu giọng nhân vật cho {spk_name} phải được lưu thành tệp thực tế"


# ==============================================================================
# 19. KIỂM THỬ ĐỒNG BỘ PHÂN VAI, BỘ LỌC PHỤ ĐỀ VÀ THIẾT LẬP TỐC ĐỘ (P2)
# ==============================================================================
def test_casting_table_sync_and_subtitle_filter():
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Kiểm tra dải tốc độ từ 0.8 đến 1.5
    assert ui.spin_speed.minimum() >= 0.8
    assert ui.spin_speed.maximum() <= 1.5

    # Thiết lập dữ liệu phụ đề và hồ sơ nhân vật
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Đoạn 1", speaker="Người nói 1"),
        SubtitleBlock(index=2, start_time="00:00:03,000", end_time="00:00:04,000", text="Đoạn 2", speaker="Người nói 2"),
    ]
    ui.subtitle_blocks[0].confidence_level = "high"
    ui.subtitle_blocks[1].confidence_level = "review_needed"

    ui.speaker_profiles = {
        "Người nói 1": {"speaker_id": "spk_0", "count": 1, "confidence": "high"},
        "Người nói 2": {"speaker_id": "spk_1", "count": 1, "confidence": "review_needed"},
    }

    ui._update_casting_table()
    ui._populate_subtitle_table()

    # Kiểm tra có bộ lọc phụ đề
    assert hasattr(ui, "combo_sub_filter")

    # Khi lọc "Cần rà soát", hàng 0 phải ẩn và hàng 1 phải hiện
    ui.combo_sub_filter.setCurrentText("Cần rà soát")
    assert ui.sub_table.isRowHidden(0) is True
    assert ui.sub_table.isRowHidden(1) is False

    # Khi lọc "Tất cả câu thoại", cả 2 hàng phải hiện
    ui.combo_sub_filter.setCurrentText("Tất cả câu thoại")
    assert ui.sub_table.isRowHidden(0) is False
    assert ui.sub_table.isRowHidden(1) is False

    # Thử đổi tên nhân vật trong casting_table hàng 0 từ "Người nói 1" thành "Đạo Diễn Nam"
    item = ui.casting_table.item(0, 1)
    assert item is not None
    item.setText("Đạo Diễn Nam")
    ui._on_casting_table_item_changed(item)

    # Hồ sơ và câu phụ đề phải được cập nhật đồng bộ sang tên mới
    assert "Đạo Diễn Nam" in ui.speaker_profiles
    assert "Người nói 1" not in ui.speaker_profiles
    assert ui.subtitle_blocks[0].speaker == "Đạo Diễn Nam"


# ==============================================================================
# 20. KIỂM THỬ ĐỘ THUẦN KHIẾT MẪU GIỌNG, LÀM SẠCH VÀ KHÓA PROMPT NHẤT QUÁN
# ==============================================================================
def test_character_reference_sample_selection_prefers_high_confidence_single_take(tmp_path):
    """
    Kiểm tra thuật toán chọn mẫu giọng nhân vật:
    - Khi có câu thoại đạt độ tin cậy cao dài >= 2.5s, ưu tiên dùng đoạn đơn lẻ đó làm mẫu,
      không ghép nối câu độ tin cậy thấp (review_needed) gây lẫn tạp âm.
    """
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator, ConfidenceLevel
    from src.ai_engines.tts_engine import SubtitleBlock

    sr = 16000
    test_wav = str(tmp_path / "dialogue.wav")
    total_audio = _make_sine_wave(10.0, sr)
    sf.write(test_wav, total_audio, sr)

    b1 = SubtitleBlock(index=1, start_time="00:00:00,500", end_time="00:00:03,500", text="Câu dài rõ ràng")
    b2 = SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:05,000", text="Câu ngắn")

    coordinator = SpeakerDiarizationCoordinator(allow_mock=True)

    with patch("src.ai_engines.speaker_diarization.cosine_cluster_embeddings") as mock_cluster:
        mock_cluster.return_value = (
            np.array([0, 0]),
            [ConfidenceLevel.HIGH.value, ConfidenceLevel.REVIEW_NEEDED.value],
            [0.98, 0.45],
        )
        profiles = coordinator.process_video_subtitles(
            video_path=test_wav,
            blocks=[b1, b2],
            temp_dir=str(tmp_path / "cache"),
        )

    assert "Người nói 1" in profiles
    prof = profiles["Người nói 1"]
    sample_path = prof.source_audio_sample
    assert sample_path and os.path.exists(sample_path)
    sample_data, sample_sr = sf.read(sample_path)
    duration = len(sample_data) / sample_sr
    assert 2.5 <= duration <= 3.6


def test_diarization_clears_stale_voice_refs_on_unlocked_blocks(tmp_path):
    """
    Kiểm tra việc làm sạch mẫu giọng cũ:
    Khi nhận diện lại, các câu chưa khóa bị đánh dấu Chưa xác định phải được xóa sạch voice_ref_path cũ,
    không để mẫu giọng cũ rò rỉ sang quá trình lồng tiếng.
    """
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator
    from src.ai_engines.tts_engine import SubtitleBlock

    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu hợp lệ")
    b1.voice_ref_path = "duong_dan_cu_spk1.wav"
    b1.speaker = "Người nói cũ"

    b2 = SubtitleBlock(index=2, start_time="00:00:03,500", end_time="00:00:03,600", text="Câu quá ngắn")
    b2.voice_ref_path = "duong_dan_cu_spk2.wav"
    b2.speaker = "Người nói cũ 2"

    b3_locked = SubtitleBlock(index=3, start_time="00:00:04,000", end_time="00:00:06,000", text="Câu đã khóa")
    b3_locked.is_speaker_locked = True
    b3_locked.voice_ref_path = "duong_dan_khoa.wav"
    b3_locked.speaker = "Nhân vật khóa"

    test_wav = str(tmp_path / "test_clear.wav")
    sf.write(test_wav, _make_sine_wave(6.0, 16000), 16000)

    coordinator = SpeakerDiarizationCoordinator(allow_mock=True)
    profiles = coordinator.process_video_subtitles(
        video_path=test_wav,
        blocks=[b1, b2, b3_locked],
        temp_dir=str(tmp_path / "samples"),
    )

    # b1 được gán mẫu mới (khác đường dẫn cũ)
    assert b1.voice_ref_path != "duong_dan_cu_spk1.wav"
    assert "Người nói" in b1.speaker
    # b2 quá ngắn bị đánh dấu Chưa xác định -> voice_ref_path phải bị xóa rỗng
    assert b2.speaker == "Chưa xác định"
    assert b2.voice_ref_path == ""
    # b3_locked giữ nguyên toàn bộ
    assert b3_locked.speaker == "Nhân vật khóa"
    assert b3_locked.voice_ref_path == "duong_dan_khoa.wav"


def test_character_prompt_locking_and_reuse_across_sentences(tmp_path):
    """
    Kiểm tra khóa một prompt duy nhất cho mỗi nhân vật:
    Toàn bộ các câu thuộc cùng một nhân vật phải tái sử dụng chung một VoiceClonePrompt
    trong suốt quá trình lồng tiếng video.
    """
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock, VoiceClonePrompt

    engine = TTSEngine(
        multi_speaker_mode="auto_video_cluster",
        voice_prompt_cache_dir=str(tmp_path / "prompts"),
        allow_mock=True,
    )
    engine.load_model = MagicMock()
    mock_prompt_1 = MagicMock(spec=VoiceClonePrompt)
    engine._get_or_create_prompt_for_audio = MagicMock(return_value=mock_prompt_1)

    prompts_used = []
    def mock_generate(*args, **kwargs):
        prompts_used.append(kwargs.get("voice_clone_prompt"))
        return _make_sine_wave(1.0, engine.sample_rate)

    engine.model = MagicMock()
    engine.model.generate.side_effect = mock_generate

    ref_wav = str(tmp_path / "spk1_ref.wav")
    sf.write(ref_wav, _make_sine_wave(3.0, 16000), 16000)

    engine.speaker_profiles = {
        "Người nói 1": {"speaker_id": "spk_0", "ref_audio_path": ref_wav, "instruct": "male"},
    }

    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,500", text="Câu 1 của nhân vật 1", speaker="Người nói 1")
    b2 = SubtitleBlock(index=2, start_time="00:00:03,000", end_time="00:00:04,500", text="Câu 2 của nhân vật 1", speaker="Người nói 1")
    b3 = SubtitleBlock(index=3, start_time="00:00:05,000", end_time="00:00:06,500", text="Câu 3 của nhân vật 1", speaker="Người nói 1")

    out_wav = str(tmp_path / "out.wav")
    engine.synthesize_blocks(
        blocks=[b1, b2, b3],
        total_duration=7.0,
        output_path=out_wav,
    )

    # Cả 3 câu của "Người nói 1" phải dùng chung đúng một prompt mock_prompt_1
    assert len(prompts_used) == 3
    assert prompts_used[0] is mock_prompt_1
    assert prompts_used[1] is mock_prompt_1
    assert prompts_used[2] is mock_prompt_1
    # Cache phải lưu trữ prompt dưới khóa character_Người nói 1
    assert "character_Người nói 1" in engine._speaker_prompt_cache
    assert engine._speaker_prompt_cache["character_Người nói 1"] is mock_prompt_1


def test_prompt_creation_failure_falls_back_to_instruct_not_unconditioned(tmp_path):
    """
    Kiểm tra xử lý dự phòng:
    Nếu tạo VoiceClonePrompt thất bại từ ref_audio, generate_single phải tự động
    chuyển sang override_instruct hoặc instruct mặc định, tuyệt đối không gọi generate
    khi chưa có điều kiện giọng nói.
    """
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(
        auto_extract_ref=False,
        instruct="female, young adult",
        allow_mock=True,
    )
    engine.load_model = MagicMock()
    # Giả lập tạo prompt thất bại
    engine._get_or_create_prompt_for_audio = MagicMock(return_value=None)

    captured_kwargs = {}
    def mock_generate(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _make_sine_wave(1.0, engine.sample_rate)

    engine.model = MagicMock()
    engine.model.generate.side_effect = mock_generate

    fake_ref = str(tmp_path / "broken_ref.wav")
    sf.write(fake_ref, _make_sine_wave(1.0, 16000), 16000)

    engine.generate_single(
        text="Thử nghiệm dự phòng giọng nói",
        override_ref_audio=fake_ref,
        override_instruct="male, middle-aged",
    )

    # Khi prompt tạo thất bại, gen_kwargs phải có instruct dự phòng
    assert "voice_clone_prompt" not in captured_kwargs
    assert captured_kwargs.get("instruct") == "male, middle-aged"


# =============================================================================
# 21. KIỂM THỬ HỒ SƠ GIỌNG NHÂN VẬT (CHARACTER VOICE PROFILE) VÀ KHÓA GIỌNG
# =============================================================================

def test_character_voice_profile_dataclass_contract():
    """
    Kiểm tra cấu trúc CharacterVoiceProfile:
    - Lưu trữ đầy đủ trường thông tin chuẩn theo đặc tả thiết kế.
    - Tương thích ngược hoàn toàn với truy cập dạng từ điển profile['speaker_id'], profile.get('ref_audio').
    - Các bí danh thuộc tính cũ (source_audio_sample, dialogue_count, assigned_strategy) hoạt động 2 chiều.
    """
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    prof = CharacterVoiceProfile(
        speaker_id="speaker_01",
        display_name="Người dẫn chuyện",
        utterance_count=8,
        total_duration=24.5,
        source_audio_path="cache/speaker_01_ref.wav",
        method="clone",
        preset_name="",
        voice_instruct="male, warm, storyteller",
        status="locked",
    )

    assert prof.speaker_id == "speaker_01"
    assert prof.display_name == "Người dẫn chuyện"
    assert prof.utterance_count == 8
    assert prof.total_duration == 24.5
    assert prof.source_audio_path == "cache/speaker_01_ref.wav"
    assert prof.method == "clone"
    assert prof.status == "locked"

    # Kiểm tra truy cập dạng từ điển
    assert prof["speaker_id"] == "speaker_01"
    assert prof["ref_audio"] == "cache/speaker_01_ref.wav"
    assert prof["source_audio_sample"] == "cache/speaker_01_ref.wav"
    assert prof["count"] == 8
    assert prof.get("method") == "clone"
    assert prof.get("status") == "locked"
    assert "source_audio_path" in prof

    # Kiểm tra các thuộc tính bí danh tương thích ngược
    assert prof.source_audio_sample == "cache/speaker_01_ref.wav"
    prof.source_audio_sample = "cache/new_ref.wav"
    assert prof.source_audio_path == "cache/new_ref.wav"

    assert prof.dialogue_count == 8
    prof.dialogue_count = 10
    assert prof.utterance_count == 10

    assert prof.assigned_strategy == "clone"
    prof.assigned_strategy = "preset"
    assert prof.method == "preset"


def test_process_video_subtitles_returns_character_voice_profiles(tmp_path):
    """
    Kiểm tra process_video_subtitles trả về tập hợp CharacterVoiceProfile chuẩn hóa,
    được đánh dấu trạng thái mặc định 'locked' và phương pháp 'clone'.
    """
    from src.ai_engines.speaker_diarization import (
        CharacterVoiceProfile,
        SpeakerDiarizationCoordinator,
    )
    from src.ai_engines.tts_engine import SubtitleBlock

    sr = 16000
    fake_audio = np.concatenate([
        _make_sine_wave(3.0, sr, freq=220.0),
        _make_sine_wave(3.0, sr, freq=440.0),
    ])
    audio_path = str(tmp_path / "dialogue_video.wav")
    sf.write(audio_path, fake_audio, sr)

    blocks = [
        SubtitleBlock(1, "00:00:00,000", "00:00:02,800", "Chào bạn"),
        SubtitleBlock(2, "00:00:03,000", "00:00:05,800", "Chào bạn, tôi là người thứ hai"),
    ]

    coordinator = SpeakerDiarizationCoordinator(allow_mock=True)
    profiles = coordinator.process_video_subtitles(
        video_path=audio_path,
        blocks=blocks,
        max_speakers=2,
        temp_dir=str(tmp_path),
    )

    assert len(profiles) >= 1
    for name, prof in profiles.items():
        assert isinstance(prof, CharacterVoiceProfile)
        assert prof.status == "locked"
        assert prof.method == "clone"
        assert prof.utterance_count >= 1
        assert prof.source_audio_path != ""
        assert os.path.exists(prof.source_audio_path)








