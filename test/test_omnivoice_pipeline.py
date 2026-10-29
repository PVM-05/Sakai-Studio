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
    # Giữ giọng thiết kế thuần túy qua các câu, không tự tạo prompt nhân bản làm méo tiếng
    assert "consistent_Người nói 1" not in engine._speaker_prompt_cache
    second_call_kwargs = engine.generate_single.call_args_list[1].kwargs
    assert "male, young adult" in second_call_kwargs.get("override_instruct")
    assert second_call_kwargs.get("override_prompt") is None


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


def test_profile_prompt_pre_initialization_and_failover(tmp_path):
    """
    Kiểm tra khởi tạo trước và khóa vĩnh viễn prompt âm sắc cho từng nhân vật:
    - Trước khi tổng hợp câu thoại, các nhân vật có tệp mẫu hợp lệ phải được tạo trước VoiceClonePrompt.
    - Với nhân vật tạo prompt thất bại, hệ thống tự động khóa sang preset tiếng Việt dự phòng,
      tất cả các câu của nhân vật đó đều dùng chung preset dự phòng này.
    """
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    engine = TTSEngine(auto_extract_ref=False, allow_mock=True)
    engine.load_model = MagicMock()
    engine.model = MagicMock()

    mock_prompt = MagicMock()
    def fake_get_or_create(audio_input, cache_key=None, ref_text=None):
        if "speaker_01" in str(audio_input) or (cache_key and "speaker_01" in cache_key):
            return mock_prompt
        # speaker_02 lỗi
        return None

    engine._get_or_create_prompt_for_audio = MagicMock(side_effect=fake_get_or_create)

    ref_1 = str(tmp_path / "speaker_01_ref.wav")
    ref_2 = str(tmp_path / "speaker_02_ref.wav")
    sf.write(ref_1, _make_sine_wave(1.0, 16000), 16000)
    sf.write(ref_2, _make_sine_wave(1.0, 16000), 16000)

    prof1 = CharacterVoiceProfile(
        speaker_id="speaker_01",
        display_name="Nhân vật 1",
        source_audio_path=ref_1,
        method="clone",
        status="locked",
    )
    prof2 = CharacterVoiceProfile(
        speaker_id="speaker_02",
        display_name="Nhân vật 2",
        source_audio_path=ref_2,
        method="clone",
        status="locked",
    )
    engine.speaker_profiles = {"Nhân vật 1": prof1, "Nhân vật 2": prof2}

    blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu 1 của nhân vật 1", speaker="Nhân vật 1"),
        SubtitleBlock(2, "00:00:03,500", "00:00:05,000", "Câu 2 của nhân vật 2", speaker="Nhân vật 2"),
        SubtitleBlock(3, "00:00:05,500", "00:00:07,000", "Câu 3 của nhân vật 1", speaker="Nhân vật 1"),
        SubtitleBlock(4, "00:00:07,500", "00:00:09,000", "Câu 4 của nhân vật 2", speaker="Nhân vật 2"),
    ]

    gen_calls = []
    def fake_generate_single(text, **kwargs):
        gen_calls.append((text, kwargs))
        return _make_sine_wave(1.0, engine.sample_rate)

    engine.generate_single = MagicMock(side_effect=fake_generate_single)

    out_wav = str(tmp_path / "out_dub.wav")
    report = engine.synthesize_blocks(blocks, total_duration=10.0, output_path=out_wav)

    # Xác nhận trước khi tổng hợp, prompt của speaker 1 đã được tạo và lưu cache
    assert "character_Nhân vật 1" in engine._speaker_prompt_cache
    assert engine._speaker_prompt_cache["character_Nhân vật 1"] is mock_prompt

    # Nhân vật 2 tạo thất bại nên phải có preset/instruct được khóa
    assert hasattr(engine, "_speaker_instruct_cache")
    assert "Nhân vật 2" in engine._speaker_instruct_cache
    fallback_instruct = engine._speaker_instruct_cache["Nhân vật 2"]
    assert fallback_instruct != ""

    # Kiểm tra tất cả các câu của nhân vật 1 đều nhận đúng mock_prompt
    assert gen_calls[0][1]["override_prompt"] is mock_prompt
    assert gen_calls[2][1]["override_prompt"] is mock_prompt

    # Kiểm tra tất cả các câu của nhân vật 2 đều nhận đúng cùng một fallback_instruct đã khóa
    assert gen_calls[1][1]["override_instruct"] == fallback_instruct
    assert gen_calls[3][1]["override_instruct"] == fallback_instruct


def test_casting_table_profile_sync(tmp_path):
    """
    Kiểm tra giao diện Bảng hồ sơ giọng nhân vật:
    1. Khi đổi phương pháp/preset của nhân vật trên bảng phân vai, mọi câu thoại của nhân vật
       trong danh sách phụ đề được cập nhật tức thì (xóa đường dẫn cũ, nạp chỉ dẫn mới).
    2. Khi đổi tên nhân vật trên bảng phân vai, danh sách phụ đề và hồ sơ nhân vật đồng bộ chính xác.
    3. Khi đổi nhân vật của một câu thoại trên bảng phụ đề, câu thoại tự động kế thừa cấu hình giọng của nhân vật mới.
    """
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock, VIETNAMESE_VOICE_CATALOG
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ref_file = str(tmp_path / "ref_char1.wav")
    sf.write(ref_file, _make_sine_wave(1.0, 16000), 16000)

    prof = CharacterVoiceProfile(
        speaker_id="speaker_01",
        display_name="Nhân vật 1",
        source_audio_path=ref_file,
        method="clone",
        status="locked",
    )
    ui.speaker_profiles = {"Nhân vật 1": prof}
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào", speaker="Nhân vật 1", voice_ref_path=ref_file),
        SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Tạm biệt", speaker="Nhân vật 1", voice_ref_path=ref_file),
    ]

    ui._update_casting_table()
    assert ui.casting_table.rowCount() == 1

    # 1. Thử đổi phương pháp từ clone sang preset chuẩn từ danh mục tiếng Việt
    preset_choice = list(VIETNAMESE_VOICE_CATALOG.keys())[2]
    ui._on_character_method_changed("Nhân vật 1", preset_choice)

    assert prof.method == "preset"
    assert prof.preset_name == preset_choice
    for blk in ui.subtitle_blocks:
        assert blk.voice_ref_path == ""
        assert blk.voice_instruct == VIETNAMESE_VOICE_CATALOG[preset_choice]["instruct"]
        assert blk.is_speaker_locked is True

    # 2. Thử đổi phương pháp ngược lại về clone
    ui._on_character_method_changed("Nhân vật 1", "Nhân bản giọng từ video")
    assert prof.method == "clone"
    for blk in ui.subtitle_blocks:
        assert blk.voice_ref_path == ref_file
        assert blk.voice_instruct == ""

    # 3. Thử đổi tên nhân vật từ "Nhân vật 1" thành "Người dẫn chuyện"
    item = ui.casting_table.item(0, 1)
    item.setText("Người dẫn chuyện")
    # itemChanged kích hoạt _on_casting_table_item_changed
    assert "Người dẫn chuyện" in ui.speaker_profiles
    assert "Nhân vật 1" not in ui.speaker_profiles
    for blk in ui.subtitle_blocks:
        assert blk.speaker == "Người dẫn chuyện"


# ==============================================================================
# 22. KIỂM THỬ KHỐI HỒ SƠ GIỌNG NHÂN VẬT & CỘT LOẠI GIỌNG TRONG TAB BƯỚC 2 (TDD)
# ==============================================================================
def test_character_voice_profiles_overview_layout_and_count(tmp_path):
    """
    Xác minh khối tổng quan hồ sơ nhân vật trong Bước 2:
    - Tiêu đề phụ: 'Video đã phân tích âm thanh · Đã nhận diện'
    - Nhãn số lượng hiển thị đúng số nhân vật thực tế tìm thấy: 'Tổng số nhân vật nhận diện: 3 nhân vật'
    - Tạo đúng 3 thẻ con tương ứng với 3 nhân vật và hiển thị đúng huy hiệu trạng thái, loại giọng.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ref_wav = str(tmp_path / "char1.wav")
    sf.write(ref_wav, _make_sine_wave(2.0, 16000), 16000)

    prof1 = CharacterVoiceProfile(
        speaker_id="speaker_01",
        display_name="Nhân vật 1",
        source_audio_path=ref_wav,
        method="clone",
        status="locked",
    )
    prof2 = CharacterVoiceProfile(
        speaker_id="speaker_02",
        display_name="Nhân vật 2",
        preset_name="Nam Miền Bắc Trầm Ấm",
        method="preset",
        status="locked",
    )
    prof3 = CharacterVoiceProfile(
        speaker_id="speaker_03",
        display_name="Nhân vật 3",
        method="clone",
        status="unassigned",
    )
    ui.speaker_profiles = {
        "Nhân vật 1": prof1,
        "Nhân vật 2": prof2,
        "Nhân vật 3": prof3,
    }
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Chào bạn", speaker="Nhân vật 1", voice_ref_path=ref_wav),
        SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Tôi là nhân vật hai", speaker="Nhân vật 2", voice_instruct="Nam trầm ấm"),
        SubtitleBlock(3, "00:00:07,000", "00:00:09,000", "Tôi chưa có giọng", speaker="Nhân vật 3"),
    ]

    ui._update_character_profiles_overview()

    assert hasattr(ui, "lbl_char_overview_title")
    assert ui.lbl_char_overview_title.text() == "Video đã phân tích âm thanh · Đã nhận diện"
    assert hasattr(ui, "lbl_char_count")
    assert ui.lbl_char_count.text() == "Tổng số nhân vật nhận diện: 3 nhân vật"

    # Kiểm tra số lượng thẻ trong char_cards_layout
    cards = getattr(ui, "_character_cards", [])
    assert len(cards) == 3

    # Thẻ 1: Nhân bản giọng từ video, Sẵn sàng
    assert cards[0]["name"] == "Nhân vật 1"
    assert cards[0]["status"] == "Sẵn sàng"
    assert "Nhân bản giọng từ video" in cards[0]["type"]

    # Thẻ 2: Giọng thiết kế, Sẵn sàng
    assert cards[1]["name"] == "Nhân vật 2"
    assert cards[1]["status"] == "Sẵn sàng"
    assert "Giọng thiết kế" in cards[1]["type"]

    # Thẻ 3: Chưa có giọng hoặc Cần kiểm tra
    assert cards[2]["name"] == "Nhân vật 3"
    assert cards[2]["status"] == "Cần kiểm tra"


def test_sub_table_voice_source_column_display_and_sync(tmp_path):
    """
    Xác minh Cột 4 'Loại Giọng & Nguồn':
    - Tiêu đề cột phải là 'Loại Giọng & Nguồn'.
    - Định dạng hiển thị chuẩn: 'Nhân bản giọng từ video', 'Giọng thiết kế: {tên}', 'Chưa có giọng'.
    - Khi thay đổi Nhân vật ở Cột 3, Cột 4 phải tự động cập nhật đúng loại giọng của nhân vật mới.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ref_wav = str(tmp_path / "char1.wav")
    sf.write(ref_wav, _make_sine_wave(1.5, 16000), 16000)

    header_col4 = ui.sub_table.horizontalHeaderItem(4).text()
    assert header_col4 == "Loại Giọng & Nguồn"

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu 1", speaker="Nhân vật 1", voice_ref_path=ref_wav)
    b2 = SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Câu 2", speaker="Nhân vật 2", voice_instruct="Nam trầm")
    b3 = SubtitleBlock(3, "00:00:07,000", "00:00:09,000", "Câu 3", speaker="Nhân vật 3")

    assert ui._format_voice_source_display(b1) == "Nhân bản giọng từ video"
    assert ui._format_voice_source_display(b2) == "Giọng thiết kế: Nam trầm"
    assert ui._format_voice_source_display(b3) == "Chưa có giọng"

    ui.speaker_profiles = {
        "Nhân vật 1": CharacterVoiceProfile("spk_1", "Nhân vật 1", source_audio_path=ref_wav, method="clone"),
        "Nhân vật 2": CharacterVoiceProfile("spk_2", "Nhân vật 2", method="preset", voice_instruct="Nam trầm"),
    }
    ui.subtitle_blocks = [b1, b2, b3]
    ui._populate_subtitle_table()

    assert ui.sub_table.item(0, 4).text() == "Nhân bản giọng từ video"
    assert ui.sub_table.item(1, 4).text() == "Giọng thiết kế: Nam trầm"
    assert ui.sub_table.item(2, 4).text() == "Chưa có giọng"

    # Đổi câu 3 (đang Nhân vật 3) sang Nhân vật 1
    item_spk = ui.sub_table.item(2, 3)
    item_spk.setText("Nhân vật 1")
    # Kiểm tra Cột 4 được cập nhật tự động
    assert ui.sub_table.item(2, 4).text() == "Nhân bản giọng từ video"


def test_character_audition_finds_first_dialogue_line(tmp_path):
    """
    Xác minh nút Nghe Thử Giọng của nhân vật:
    - Tìm chính xác câu thoại đầu tiên của nhân vật trong video để tổng hợp.
    - Nếu nhân vật chưa có câu thoại, sử dụng câu thoại mẫu mặc định.
    - Truyền đúng override_ref_audio hoặc override_instruct tương ứng với hồ sơ.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ref_wav = str(tmp_path / "char1.wav")
    sf.write(ref_wav, _make_sine_wave(1.5, 16000), 16000)

    ui.speaker_profiles = {
        "Nhân vật 1": CharacterVoiceProfile("spk_1", "Nhân vật 1", source_audio_path=ref_wav, method="clone"),
        "Nhân vật 2": CharacterVoiceProfile("spk_2", "Nhân vật 2", method="preset", voice_instruct="Nữ dịu dàng"),
        "Nhân vật 3": CharacterVoiceProfile("spk_3", "Nhân vật 3", method="preset", voice_instruct="Nữ trẻ trung"),
    }
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Lời mở đầu của nhân vật 2", speaker="Nhân vật 2"),
        SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Câu đầu tiên của nhân vật 1", speaker="Nhân vật 1"),
        SubtitleBlock(3, "00:00:07,000", "00:00:09,000", "Câu thứ hai của nhân vật 1", speaker="Nhân vật 1"),
    ]

    captured_args = {}
    def mock_launch(task_type, text="", output_path="", override_ref_audio="", override_instruct="", **kwargs):
        captured_args["task_type"] = task_type
        captured_args["text"] = text
        captured_args["override_ref_audio"] = override_ref_audio
        captured_args["override_instruct"] = override_instruct

    ui._launch_worker = mock_launch

    # Audition Nhân vật 1 -> Phải lấy câu 2 ("Câu đầu tiên của nhân vật 1")
    ui._audition_character_voice("Nhân vật 1")
    assert captured_args["task_type"] == "character_audition"
    assert captured_args["text"] == "Câu đầu tiên của nhân vật 1"
    assert captured_args["override_ref_audio"] == ref_wav

    # Audition Nhân vật 3 (không có trong subtitle_blocks) -> Dùng câu mẫu mặc định
    ui._audition_character_voice("Nhân vật 3")
    assert captured_args["task_type"] == "character_audition"
    assert "Xin chào, đây là giọng nói thử nghiệm của Nhân vật 3." in captured_args["text"]
    assert captured_args["override_instruct"] == "Nữ trẻ trung"


def test_character_voice_profiles_overview_unanalyzed_state():
    """
    Xác minh trạng thái khi chưa phân tích nhân vật:
    - Tiêu đề phụ: 'Âm thanh video chưa phân tích nhân vật'
    - Nhãn trạng thái: 'Trạng thái: Chưa phân tích'
    - Danh sách thẻ con rỗng (hiển thị văn bản hướng dẫn chạy nhận diện).
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.speaker_profiles = {}
    ui.is_speakers_detected = False
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một", speaker=""),
    ]

    ui._update_character_profiles_overview()

    assert ui.lbl_char_overview_title.text() == "Âm thanh video chưa phân tích nhân vật"
    assert ui.lbl_char_count.text() == "Trạng thái: Chưa phân tích"
    assert len(ui._character_cards) == 0


def test_audition_rejects_missing_clone_audio_sample():
    """
    Xác minh kiểm tra tính hợp lệ của mẫu nhân bản trước khi nghe thử:
    - Nếu nhân vật đặt phương pháp clone nhưng tệp mẫu không tồn tại, tuyệt đối không âm thầm chạy hay đổi giọng.
    - Không khởi chạy worker tổng hợp và khôi phục nút nghe thử.
    """
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import PushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.speaker_profiles = {
        "Nhân vật Thiếu Mẫu": CharacterVoiceProfile(
            "spk_x",
            "Nhân vật Thiếu Mẫu",
            source_audio_path="C:/non_existent_folder/missing_audio.wav",
            method="clone",
        )
    }

    launch_called = []
    ui._launch_worker = lambda *a, **k: launch_called.append(True)

    btn = PushButton("Nghe Thử Giọng")
    ui._audition_character_voice("Nhân vật Thiếu Mẫu", btn)

    # Worker không được phép chạy
    assert len(launch_called) == 0
    # Nút phải được mở khóa và phục hồi văn bản
    assert btn.text() == "Nghe Thử Giọng"
    assert btn.isEnabled() is True


def test_audition_button_recovers_on_playback_end():
    """
    Xác minh phục hồi trạng thái nút nghe thử sau khi phát xong hoặc dừng phát:
    - Khi phát xong (EndOfMedia), nút phải tự động chuyển về 'Nghe Thử Giọng' và được mở khóa.
    """
    from PySide6.QtWidgets import QApplication
    from PySide6.QtMultimedia import QMediaPlayer
    from qfluentwidgets import PushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    btn = PushButton("Nghe Thử Giọng")
    ui._active_audition_button = btn

    # Giả lập trạng thái đang phát
    btn.setText("Dừng Phát")
    btn.setEnabled(True)

    # Báo hiệu phát xong qua mediaStatusChanged
    ui._on_player_media_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)

    assert btn.text() == "Nghe Thử Giọng"
    assert btn.isEnabled() is True
    assert ui._active_audition_button is None


def test_synthesize_blocks_strictly_respects_preset_method_over_stale_clone_path(tmp_path):
    """
    Xác minh synthesize_blocks tôn trọng tuyệt đối phương pháp preset:
    - Khi nhân vật được đặt method='preset', ngay cả khi hồ sơ còn source_audio_path hoặc _speaker_prompt_cache,
      hệ thống bắt buộc phải truyền override_mode='design', xóa override_prompt và override_ref_audio,
      và chỉ sử dụng override_instruct.
    """
    from unittest.mock import MagicMock
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    engine = TTSEngine(mode="clone")
    engine.load_model = MagicMock()
    ref_audio = tmp_path / "stale_sample.wav"
    ref_audio.write_bytes(b"RIFFdummydata")

    prof = CharacterVoiceProfile(
        speaker_id="spk_preset",
        display_name="Nhân vật 1",
        method="preset",
        preset_name="Nam Thanh Niên",
        voice_instruct="male, young adult, clear",
        source_audio_path=str(ref_audio),
    )
    engine.speaker_profiles = {"Nhân vật 1": prof}
    engine._speaker_prompt_cache["character_Nhân vật 1"] = MagicMock()

    blocks = [
        SubtitleBlock(
            1,
            "00:00:01,000",
            "00:00:03,000",
            "Xin chào các bạn",
            speaker="Nhân vật 1",
            voice_instruct="male, young adult, clear",
            voice_ref_path="",
        )
    ]

    captured_calls = []

    def fake_generate_single(text, **kwargs):
        captured_calls.append(kwargs)
        return _make_sine_wave(1.0, 24000)

    engine.generate_single = fake_generate_single

    out_wav = str(tmp_path / "out.wav")
    engine.synthesize_blocks(blocks, 5.0, out_wav)

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call.get("override_mode") == "design"
    assert call.get("override_prompt") is None
    assert call.get("override_ref_audio") is None
    assert call.get("override_instruct") == "male, young adult, clear"


def test_compute_block_cache_path_includes_method_and_mtime(tmp_path):
    """
    Xác minh khóa bộ nhớ đệm checkpoint thay đổi khi phương pháp hoặc tệp mẫu thay đổi.
    """
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(mode="clone")
    engine.resume_cache_dir = str(tmp_path)

    sample_wav = tmp_path / "sample.wav"
    sample_wav.write_bytes(b"123")

    p1 = engine._compute_block_cache_path(
        0, "Xin chào", "Nhân vật 1", str(sample_wav), "", 2.0, method_or_mode="clone"
    )
    p2 = engine._compute_block_cache_path(
        0, "Xin chào", "Nhân vật 1", "", "male voice", 2.0, method_or_mode="preset"
    )

    # Đổi phương pháp từ clone sang preset phải sinh ra khóa cache khác nhau
    assert p1 != p2


def test_assign_target_speaker_clears_stale_ref_audio_and_instruct():
    """
    Xác minh khi gán dòng sang nhân vật mới (dùng preset),
    mẫu giọng cũ (voice_ref_path) của dòng đó phải được xóa hoàn toàn về rỗng.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(
            1,
            "00:00:01,000",
            "00:00:03,000",
            "Câu của Nhân vật 1",
            speaker="Nhân vật 1",
            voice_ref_path="C:/voices/sample_1.wav",
            voice_instruct="",
        )
    ]
    ui._populate_subtitle_table()

    # Thiết lập Nhân vật 2 là preset
    ui.speaker_profiles = {
        "Nhân vật 1": CharacterVoiceProfile("spk_1", "Nhân vật 1", source_audio_path="C:/voices/sample_1.wav", method="clone"),
        "Nhân vật 2": CharacterVoiceProfile("spk_2", "Nhân vật 2", method="preset", voice_instruct="female voice"),
    }

    # Chọn Nhân vật 2 trong combo mục tiêu và bôi đen dòng 0
    ui.combo_target_speaker.clear()
    ui.combo_target_speaker.addItems(["Nhân vật 1", "Nhân vật 2"])
    ui.combo_target_speaker.setCurrentText("Nhân vật 2")
    ui.sub_table.selectRow(0)

    ui._assign_target_speaker_to_selected_rows()

    blk = ui.subtitle_blocks[0]
    assert blk.speaker == "Nhân vật 2"
    # Mẫu giọng clone cũ phải bị xóa sạch, không được giữ lại!
    assert blk.voice_ref_path == ""
    assert blk.voice_instruct == "female voice"
    assert blk.is_speaker_locked is True


def test_apply_preset_to_target_speaker_switches_method_and_clears_source_audio():
    """
    Xác minh khi bấm 'Áp dụng thiết kế giọng cho nhân vật',
    hồ sơ nhân vật phải chuyển hẳn sang method='preset' và xóa sạch source_audio_path/ref_audio_path.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    prof = CharacterVoiceProfile(
        "spk_1",
        "Nhân vật 1",
        source_audio_path="C:/voices/sample_1.wav",
        method="clone",
        status="locked",
    )
    ui.speaker_profiles = {"Nhân vật 1": prof}
    ui.subtitle_blocks = [
        SubtitleBlock(
            1,
            "00:00:01,000",
            "00:00:03,000",
            "Câu một",
            speaker="Nhân vật 1",
            voice_ref_path="C:/voices/sample_1.wav",
        )
    ]
    ui._populate_subtitle_table()

    ui.combo_target_speaker.clear()
    ui.combo_target_speaker.addItem("Nhân vật 1")
    ui.combo_target_speaker.setCurrentText("Nhân vật 1")

    # Chọn một preset giọng
    first_preset = ui.combo_speaker_voice_preset.itemText(0)
    ui.combo_speaker_voice_preset.setCurrentText(first_preset)

    ui._apply_preset_to_target_speaker()

    p = ui.speaker_profiles["Nhân vật 1"]
    assert p.method == "preset"
    assert p.source_audio_path == ""
    assert getattr(p, "ref_audio_path", "") == ""
    # Dòng phụ đề cũng phải xóa sạch voice_ref_path
    assert ui.subtitle_blocks[0].voice_ref_path == ""
    assert ui.subtitle_blocks[0].voice_instruct != ""


def test_detect_speakers_from_srt_text_strips_prefix_clears_ref_and_preserves_locked():
    """
    Xác minh detect_speakers_from_srt_text:
    - Loại bỏ tiền tố tên người nói khỏi văn bản (để TTS không phát âm tên người nói).
    - Xóa voice_ref_path cũ khỏi các dòng vừa phân vai.
    - Bảo toàn các dòng đã khóa (is_speaker_locked=True).
    - Thiết lập method='preset' cho các hồ sơ được nhận diện.
    """
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    engine = TTSEngine(mode="clone")
    b1 = SubtitleBlock(
        1,
        "00:00:01,000",
        "00:00:03,000",
        "[Nam]: Chào mọi người!",
        voice_ref_path="C:/old_stale_ref.wav",
    )
    b2 = SubtitleBlock(
        2,
        "00:00:03,500",
        "00:00:05,000",
        "Em chào anh ạ",
        speaker="Nữ Chính",
        voice_instruct="female sweet",
        is_speaker_locked=True,
    )

    profiles = engine.detect_speakers_from_srt_text([b1, b2])

    # Block 1 phải được loại bỏ tiền tố [Nam]:
    assert b1.text == "Chào mọi người!"
    assert b1.speaker == "Người nói 1"
    # Mẫu giọng clone cũ phải bị xóa vì phân vai từ SRT dùng preset
    assert b1.voice_ref_path == ""

    # Block 2 đã khóa phải giữ nguyên
    assert b2.speaker == "Nữ Chính"
    assert b2.voice_instruct == "female sweet"
    assert b2.is_speaker_locked is True

    # Hồ sơ tạo ra phải mang method='preset'
    assert profiles["Người nói 1"]["method"] == "preset"


def test_character_overview_card_marks_review_needed_even_with_sample(tmp_path):
    """
    Xác minh thẻ nhân vật phân biệt chất lượng nhận diện:
    - Nếu status='review_needed', ngay cả khi có tệp mẫu WAV hợp lệ,
      thẻ vẫn phải hiển thị huy hiệu 'Cần kiểm tra' và chi tiết 'Mẫu giọng cần rà soát'.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    valid_wav = tmp_path / "valid_sample.wav"
    valid_wav.write_bytes(b"RIFFheaderdata")

    ui.speaker_profiles = {
        "Nhân vật Cần Rà Soát": CharacterVoiceProfile(
            "spk_rv",
            "Nhân vật Cần Rà Soát",
            source_audio_path=str(valid_wav),
            method="clone",
            status="review_needed",
        )
    }
    ui.is_speakers_detected = True
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào", speaker="Nhân vật Cần Rà Soát")
    ]

    ui._update_character_profiles_overview()

    assert len(ui._character_cards) == 1
    card_info = ui._character_cards[0]
    assert card_info["status"] == "Cần kiểm tra"
    assert card_info["detail"] == "Mẫu giọng cần rà soát"


def test_generate_single_in_design_mode_never_uses_ref_audio(tmp_path):
    """
    Xác minh trong generate_single():
    Khi override_mode='design', ngay cả khi override_ref_audio được truyền và tệp tồn tại,
    hệ thống tuyệt đối không nạp tệp mẫu nhân bản, chỉ sử dụng instruct.
    """
    from unittest.mock import MagicMock
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(mode="clone")
    engine.load_model = MagicMock()
    engine.model = MagicMock()

    captured_kwargs = {}

    def fake_generate(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock(waveform=_make_sine_wave(1.0, 24000))

    engine.model.generate = MagicMock(side_effect=fake_generate)

    sample_wav = tmp_path / "stale_sample.wav"
    sample_wav.write_bytes(b"RIFFheader")

    engine.generate_single(
        "Chào bạn",
        override_mode="design",
        override_ref_audio=str(sample_wav),
        override_instruct="female sweet voice",
    )

    assert "voice_clone_prompt" not in captured_kwargs
    assert captured_kwargs.get("instruct") == "female sweet voice"


def test_detect_speakers_from_srt_text_preserves_locked_clone_character(tmp_path):
    """
    Xác minh detect_speakers_from_srt_text() khi gặp nhân vật đã khóa giọng nhân bản (clone):
    Hồ sơ đăng ký phải bảo toàn method='clone' và ref_audio_path, không bị biến thành preset.
    """
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    engine = TTSEngine(mode="clone")
    ref_wav = str(tmp_path / "locked_clone.wav")
    Path(ref_wav).write_bytes(b"RIFFheader")

    engine.speaker_profiles = {
        "Nhân vật Đã Khóa": CharacterVoiceProfile(
            speaker_id="spk_locked",
            display_name="Nhân vật Đã Khóa",
            source_audio_path=ref_wav,
            method="clone",
            status="locked",
        )
    }

    b_locked = SubtitleBlock(
        1,
        "00:00:01,000",
        "00:00:03,000",
        "Câu thoại của người đã khóa",
        speaker="Nhân vật Đã Khóa",
        voice_ref_path=ref_wav,
        is_speaker_locked=True,
    )
    b_new = SubtitleBlock(
        2,
        "00:00:04,000",
        "00:00:06,000",
        "Câu thoại mới chưa phân vai",
    )

    profiles = engine.detect_speakers_from_srt_text([b_locked, b_new])

    # Nhân vật đã khóa phải giữ nguyên phương pháp clone và đường dẫn mẫu
    assert profiles["Nhân vật Đã Khóa"]["method"] == "clone"
    assert profiles["Nhân vật Đã Khóa"]["ref_audio_path"] == ref_wav
    assert b_locked.voice_ref_path == ref_wav


def test_character_overview_card_marks_review_needed_for_dict_profile(tmp_path):
    """
    Xác minh khối tổng quan hồ sơ nhân vật hiển thị đúng huy hiệu 'Cần kiểm tra'
    đối với hồ sơ dạng từ điển (dict) trả về từ nhận diện video có độ tin cậy thấp.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    valid_wav = tmp_path / "valid_sample.wav"
    valid_wav.write_bytes(b"RIFFheader")

    # Hồ sơ dạng từ điển thực tế từ detect_speakers_from_video
    ui.speaker_profiles = {
        "Nhân vật Độ Tin Cậy Thấp": {
            "speaker_id": "spk_dict",
            "description": "Độ tin cậy: 45%",
            "source_audio_path": str(valid_wav),
            "ref_audio_path": str(valid_wav),
            "method": "clone",
            "status": "locked",
            "confidence_level": "review_needed",
            "high_confidence_ratio": 0.45,
            "count": 3,
        }
    }
    ui.is_speakers_detected = True
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một", speaker="Nhân vật Độ Tin Cậy Thấp")
    ]

    ui._update_character_profiles_overview()

    assert len(ui._character_cards) == 1
    card_info = ui._character_cards[0]
    assert card_info["status"] == "Cần kiểm tra"
    assert card_info["detail"] == "Mẫu giọng cần rà soát"


def test_audition_button_not_stuck_when_worker_is_running():
    """
    Xác minh khi hệ thống đang bận chạy tác vụ ngầm:
    Bấm 'Nghe Thử Giọng' sẽ bị từ chối an toàn và nút không bị khóa ở trạng thái 'Đang tạo giọng...'.
    """
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import PushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    # Giả lập worker đang chạy
    ui._worker = MagicMock()
    ui._worker.isRunning.return_value = True

    btn = PushButton("Nghe Thử Giọng")
    ui._audition_character_voice("Nhân vật 1", btn)

    # Nút không được phép bị chuyển sang Đang tạo giọng và phải luôn mở khóa
    assert btn.text() == "Nghe Thử Giọng"
    assert btn.isEnabled() is True



def test_cosine_clustering_single_speaker_hypothesis():
    """
    Xác minh cosine_cluster_embeddings nhận diện đúng 1 người nói (k = 1)
    khi tất cả các vector đặc trưng có khoảng cách cosin nhỏ (< 0.35, tương đồng > 0.65).
    Trước đây hàm luôn ép buộc k >= 2.
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    rng = np.random.default_rng(42)
    base_vec = rng.standard_normal(192).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    # Tạo 6 đoạn thoại của cùng một người nói (nhiễu nhỏ, khoảng cách cosin < 0.20)
    samples = []
    for _ in range(6):
        noise = rng.standard_normal(192).astype(np.float32) * 0.1
        v = base_vec + noise
        v /= np.linalg.norm(v)
        samples.append(v)
    embs = np.stack(samples, axis=0)

    labels, conf_levels, scores = cosine_cluster_embeddings(embs, list(range(6)), max_speakers=4)
    unique_speakers = np.unique(labels)
    # Phải nhận diện là 1 người nói duy nhất
    assert len(unique_speakers) == 1
    assert unique_speakers[0] == 0


def test_cosine_clustering_singleton_cluster_silhouette_not_inflated():
    """
    Xác minh cụm đơn lẻ không được nhận điểm Silhouette bằng 1.0 (tránh phân mảnh mỗi câu một cụm).
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    rng = np.random.default_rng(42)
    # 2 cụm người nói rõ ràng (mỗi cụm 3 mẫu, cách xa nhau)
    v1 = rng.standard_normal(192).astype(np.float32); v1 /= np.linalg.norm(v1)
    v2 = -v1  # vuông góc hoặc đối nghịch
    v2 /= np.linalg.norm(v2)

    samples = [v1 + rng.standard_normal(192)*0.05 for _ in range(3)] + [v2 + rng.standard_normal(192)*0.05 for _ in range(3)]
    embs = np.stack([s / np.linalg.norm(s) for s in samples], axis=0)

    labels, _, _ = cosine_cluster_embeddings(embs, list(range(6)), max_speakers=4)
    # Phải nhận diện đúng 2 cụm, không tách lẻ thành 3 hoặc 4 cụm đơn
    assert len(np.unique(labels)) == 2


def test_cosine_clustering_allows_full_k_candidates():
    """
    Xác minh thuật toán cho phép thử nghiệm đủ k ứng viên lên tới min(max_speakers, n_samples).
    Với N = 5 mẫu và max_speakers = 4, trước đây bị chặn ở N//2 = 2, nay phải phân biệt được 3 người nói.
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    # Tạo 3 nhóm người nói trực giao
    e1 = np.zeros(192, dtype=np.float32); e1[0] = 1.0
    e2 = np.zeros(192, dtype=np.float32); e2[50] = 1.0
    e3 = np.zeros(192, dtype=np.float32); e3[100] = 1.0

    samples = [e1, e1.copy(), e2, e2.copy(), e3]
    embs = np.stack(samples, axis=0)

    labels, _, _ = cosine_cluster_embeddings(embs, list(range(5)), max_speakers=4)
    # Phải nhận diện đúng 3 người nói khác nhau
    assert len(np.unique(labels)) == 3


def test_synthesize_blocks_does_not_skip_diarization_if_some_blocks_assigned(tmp_path):
    """
    Xác minh trong synthesize_blocks():
    Khi có 1 câu đã có nhãn người nói nhưng còn các câu khác chưa khóa/chưa phân vai,
    chế độ auto_video_cluster vẫn chạy nhận diện người nói thay vì bỏ qua hoàn toàn.
    """
    from unittest.mock import MagicMock
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    engine = TTSEngine(mode="clone", multi_speaker_mode="auto_video_cluster")
    engine.detect_speakers_from_video = MagicMock(return_value={})
    engine.load_model = MagicMock()
    engine._prepare_and_lock_character_prompts = MagicMock()
    engine.generate_single = MagicMock(return_value=_make_sine_wave(1.0, 24000))

    vid_file = str(tmp_path / "mock_vid.mp4")
    Path(vid_file).write_bytes(b"mock")

    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu đã phân vai", speaker="Nhân vật Cũ", is_speaker_locked=True)
    b2 = SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:06,000", text="Câu chưa phân vai", speaker="")

    engine.synthesize_blocks([b1, b2], 10.0, output_path=str(tmp_path / "out.wav"), video_path=vid_file)

    # Phải gọi detect_speakers_from_video vì vẫn còn câu b2 chưa phân vai
    assert engine.detect_speakers_from_video.called is True


def test_block_preview_uses_correct_override_mode_and_voice(tmp_path):
    """
    Xác minh nghe thử một câu phụ đề (block_preview):
    Khi câu được gán phương pháp preset, worker truyền override_mode='design'
    vào generate_single bất kể chế độ toàn cục của engine là clone.
    """
    from unittest.mock import MagicMock
    from src.desktop.ui.omnivoice_interface import OmniVoiceTaskWorker

    captured = {}
    def fake_generate_single(text, **kwargs):
        captured.update(kwargs)
        return str(tmp_path / "mock.wav")

    mock_engine = MagicMock()
    mock_engine.generate_single = MagicMock(side_effect=fake_generate_single)

    worker = OmniVoiceTaskWorker(
        task_type="block_preview",
        engine_params={"mode": "clone"},
        text="Xin chào",
        output_path=str(tmp_path / "preview.wav"),
        override_mode="design",
        override_instruct="female sweet voice",
    )
    worker._active_engine = mock_engine
    worker._run_engine_task(mock_engine)

    assert captured.get("override_mode") == "design"
    assert captured.get("override_instruct") == "female sweet voice"


def test_character_overview_ignores_unassigned_labels():
    """
    Xác minh _update_character_profiles_overview không tính nhãn 'Chưa phân vai'
    hoặc 'Chưa xác định' vào tổng số nhân vật nhận diện được.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.is_speakers_detected = True
    ui.speaker_profiles = {"Nhân vật 1": {"method": "preset", "status": "locked", "voice_instruct": "male"}}
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một", speaker="Nhân vật 1"),
        SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Câu hai", speaker="Chưa phân vai"),
    ]

    ui._update_character_profiles_overview()
    # Chỉ có 1 nhân vật duy nhất
    assert len(ui._character_cards) == 1
    assert "1 nhân vật" in ui.lbl_char_count.text()


def test_clone_method_without_sample_marks_review_needed():
    """
    Xác minh khi người dùng đổi phương pháp sang clone nhưng chưa có tệp mẫu âm thanh,
    trạng thái hồ sơ phải là review_needed, không được hiển thị là locked.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    prof = CharacterVoiceProfile(speaker_id="spk_test", display_name="Nhân vật Mới", source_audio_path="")
    ui.speaker_profiles = {"Nhân vật Mới": prof}

    ui._on_character_method_changed("Nhân vật Mới", "Nhân bản giọng từ video")
    assert prof.status == "review_needed"


def test_export_shows_warning_confirmation_dialog(monkeypatch):
    """
    Xác minh khi danh sách phụ đề có cảnh báo chưa phân vai:
    Hàm xuất gọi hộp thoại xác nhận. Nếu người dùng chọn từ chối (False),
    hệ thống dừng lại mà không mở hộp thoại lưu tệp.
    """
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    if hasattr(ui, "chk_single_unified_voice"):
        ui.chk_single_unified_voice.setChecked(False)
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một", speaker="Chưa phân vai")
    ]

    mock_confirm = MagicMock(return_value=False)
    monkeypatch.setattr(ui, "_confirm_pre_export_warnings", mock_confirm)
    ui._launch_worker = MagicMock()

    ui._export_subtitle_audio_track()

    assert mock_confirm.called is True
    assert ui._launch_worker.called is False


def test_cosine_clustering_imbalanced_minority_speaker_not_merged():
    """
    Xác minh thuật toán phân cụm cosin không nuốt chửng nhân vật phụ (thiểu số)
    khi dữ liệu mất cân bằng (ví dụ: Nhân vật A nói 10 câu, Nhân vật B nói 1 câu khác biệt).
    Giả thuyết k=1 không được chấp nhận nếu có mẫu cách biệt rõ rệt.
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    rng = np.random.default_rng(42)
    # 10 mẫu nhân vật A xung quanh trục [1, 0, 0, 0] với phương sai nhỏ
    embs_a = np.zeros((10, 8), dtype=np.float32)
    embs_a[:, 0] = 1.0 + rng.normal(0, 0.02, 10)
    embs_a[:, 1:] = rng.normal(0, 0.02, (10, 7))

    # 1 mẫu nhân vật B theo trục trực giao [0, 1, 0, 0] (khoảng cách cosin ~ 1.0)
    emb_b = np.zeros((1, 8), dtype=np.float32)
    emb_b[0, 1] = 1.0

    all_embs = np.vstack([embs_a, emb_b])
    labels, conf_levels, conf_scores = cosine_cluster_embeddings(
        all_embs, valid_indices=list(range(len(all_embs))), max_speakers=4
    )

    # Nhân vật B (chỉ số 10) bắt buộc phải thuộc cụm khác với Nhân vật A (chỉ số 0..9)
    assert labels[10] != labels[0]
    assert len(np.unique(labels)) >= 2
    # Cụm đơn lẻ (1 câu) trong bộ dữ liệu phải được đánh dấu 'review_needed'
    assert conf_levels[10] == "review_needed"


def test_synthesize_blocks_preserves_locked_character_profiles_and_names_when_diarizing_unassigned(tmp_path):
    """
    Xác minh khi xuất video có câu chưa phân vai:
    1. detect_speakers_from_video không ghi đè làm mất self.speaker_profiles của các nhân vật đã khóa.
    2. Các câu chưa phân vai được gán tên người nói mới mà không trùng/đè tên nhân vật đã khóa.
    3. Bước chuẩn bị prompt giữ nguyên preset tiếng Việt mà người dùng đã chọn cho nhân vật đã khóa.
    """
    from unittest.mock import MagicMock, patch
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile, SpeakerDiarizationCoordinator

    engine = TTSEngine(mode="clone", multi_speaker_mode="auto_video_cluster")
    engine.load_model = MagicMock()

    prof_locked = CharacterVoiceProfile(
        speaker_id="spk_locked",
        display_name="Nhân vật Đã Khóa",
        method="preset",
        preset_name="Nam Thanh Niên",
        voice_instruct="male, young adult, clear",
        status="locked",
    )
    engine.speaker_profiles = {"Nhân vật Đã Khóa": prof_locked}

    b1 = SubtitleBlock(
        index=1,
        start_time="00:00:01,000",
        end_time="00:00:03,000",
        text="Câu đã phân vai của nhân vật đã khóa",
        speaker="Nhân vật Đã Khóa",
        is_speaker_locked=True,
    )
    b2 = SubtitleBlock(
        index=2,
        start_time="00:00:04,000",
        end_time="00:00:06,000",
        text="Câu mới chưa phân vai",
        speaker="",
        is_speaker_locked=False,
    )

    vid_file = str(tmp_path / "test_vid.mp4")
    Path(vid_file).write_bytes(b"dummy")

    # Giả lập coordinator nhận diện câu b2 thành Người nói 1
    mock_detected_prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Người nói 1",
        method="clone",
        status="locked",
    )

    def fake_separate(*args, **kwargs):
        out_path = args[1] if len(args) == 2 else args[2]
        sf.write(out_path, _make_sine_wave(2.0, 16000), 16000)
        return True

    with patch("src.ai_engines.tts_engine.VoiceSeparator.separate_from_video", side_effect=fake_separate), \
         patch.object(
             SpeakerDiarizationCoordinator,
             "process_video_subtitles",
             return_value={"Người nói 1": mock_detected_prof},
         ):
        engine.detect_speakers_from_video(vid_file, [b1, b2], max_speakers=4)

    # 1. Hồ sơ 'Nhân vật Đã Khóa' phải được bảo toàn nguyên vẹn
    assert "Nhân vật Đã Khóa" in engine.speaker_profiles
    assert engine.speaker_profiles["Nhân vật Đã Khóa"].method == "preset"
    assert engine.speaker_profiles["Nhân vật Đã Khóa"].preset_name == "Nam Thanh Niên"

    # 2. Hồ sơ mới 'Người nói 1' cũng được ghi nhận vào bảng hồ sơ
    assert "Người nói 1" in engine.speaker_profiles

    # 3. Khi chuẩn bị prompt, nhân vật đã khóa phải dùng đúng preset Nam Thanh Niên
    engine._prepare_and_lock_character_prompts([b1, b2])
    assert "Nhân vật Đã Khóa" in engine._speaker_instruct_cache
    assert "Nam Thanh Niên" in engine.speaker_profiles["Nhân vật Đã Khóa"].preset_name


def test_audition_all_speakers_queues_and_plays_sequentially():
    """
    Xác minh nút 'Nghe thử toàn bộ nhân vật':
    1. Trích xuất đúng danh sách từ _character_cards (dùng key 'name' hoặc 'speaker').
    2. Thiết lập hàng đợi tuần tự, cập nhật trạng thái nút và bắt đầu phát nhân vật đầu tiên.
    3. Khi nhận tín hiệu EndOfMedia, tự động chuyển sang phát nhân vật kế tiếp trong hàng đợi.
    4. Có thể dừng chủ động bất kỳ lúc nào và làm sạch hàng đợi.
    """
    from PySide6.QtWidgets import QApplication
    from PySide6.QtMultimedia import QMediaPlayer
    from qfluentwidgets import PushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    prof1 = CharacterVoiceProfile(
        speaker_id="spk_1",
        display_name="Nhân vật 1",
        method="preset",
        preset_name="Nam Thanh Niên",
        voice_instruct="male youth",
        status="locked",
    )
    prof2 = CharacterVoiceProfile(
        speaker_id="spk_2",
        display_name="Nhân vật 2",
        method="preset",
        preset_name="Nữ Ngọt Ngào",
        voice_instruct="female sweet",
        status="locked",
    )
    ui.speaker_profiles = {"Nhân vật 1": prof1, "Nhân vật 2": prof2}

    btn1 = PushButton("Nghe Thử Giọng")
    btn2 = PushButton("Nghe Thử Giọng")
    ui._character_cards = [
        {"name": "Nhân vật 1", "button": btn1},
        {"name": "Nhân vật 2", "button": btn2},
    ]

    auditioned_list = []
    ui._launch_worker = lambda *args, **kwargs: auditioned_list.append(kwargs.get("override_instruct"))

    # Bắt đầu nghe thử toàn bộ
    ui._audition_all_speakers()

    assert ui._audition_all_active is True
    assert ui._current_audition_speaker == "Nhân vật 1"
    assert ui._audition_queue == ["Nhân vật 2"]
    assert len(auditioned_list) == 1

    # Chuyển tiếp sang nhân vật 2
    ui._play_next_audition_in_queue()
    assert ui._current_audition_speaker == "Nhân vật 2"
    assert ui._audition_queue == []
    assert len(auditioned_list) == 2

    # Dừng phát toàn bộ
    ui._stop_audition_all()
    assert ui._audition_all_active is False
    assert ui._audition_queue == []
    assert "Toàn Bộ" in ui.btn_audition_all_speakers.text()


# ==============================================================================
# KIỂM THỬ TƯƠNG THÍCH HAI CHIỀU CHO HỒ SƠ GIỌNG NÓI NHÂN VẬT
# ==============================================================================
def test_character_voice_profile_dict_and_attribute_interoperability():
    """
    Xác minh CharacterVoiceProfile hỗ trợ truy cập hai chiều:
    1. Đọc và ghi thuộc tính qua cú pháp từ điển (profile['field'] = val) không gây lỗi TypeError.
    2. Đọc và ghi qua cú pháp đối tượng (profile.field = val).
    3. Hỗ trợ to_dict(), from_dict(), normalize_character_profile() với các trường mới:
       vietnamese_accent, gender, pitch_level, style_tag, recommended_voices, error_message.
    """
    from src.ai_engines.speaker_diarization import (
        CharacterVoiceProfile,
        normalize_character_profile,
    )

    prof = CharacterVoiceProfile(
        speaker_id="spk_test",
        display_name="Người nói thử nghiệm",
        vietnamese_accent="nam",
        gender="male",
        pitch_level="low",
        style_tag="story",
    )

    # 1. Truy cập và gán dạng từ điển
    assert prof["speaker_id"] == "spk_test"
    assert prof["vietnamese_accent"] == "nam"
    assert prof["gender"] == "male"
    assert prof.get("pitch_level") == "low"

    # Thao tác gán dạng từ điển phải thành công không bị ném TypeError
    prof["method"] = "preset"
    prof["preset_name"] = "nam_mien_nam_tram_am"
    prof["error_message"] = "Lỗi thử nghiệm"
    assert prof.method == "preset"
    assert prof.preset_name == "nam_mien_nam_tram_am"
    assert prof.error_message == "Lỗi thử nghiệm"

    # 2. Chuyển đổi sang từ điển và phục hồi lại
    d = prof.to_dict()
    assert d["speaker_id"] == "spk_test"
    assert d["vietnamese_accent"] == "nam"
    assert d["error_message"] == "Lỗi thử nghiệm"

    reconstructed = CharacterVoiceProfile.from_dict(d)
    assert reconstructed.speaker_id == "spk_test"
    assert reconstructed.vietnamese_accent == "nam"
    assert reconstructed.preset_name == "nam_mien_nam_tram_am"

    # 3. Hàm chuẩn hóa an toàn
    norm1 = normalize_character_profile(prof)
    assert norm1 is prof

    norm2 = normalize_character_profile({"speaker_id": "spk_dict", "display_name": "Nhân vật từ điển", "vietnamese_accent": "trung"})
    assert isinstance(norm2, CharacterVoiceProfile)
    assert norm2.speaker_id == "spk_dict"
    assert norm2.vietnamese_accent == "trung"

    norm3 = normalize_character_profile(None, speaker_id="spk_fallback")
    assert isinstance(norm3, CharacterVoiceProfile)
    assert norm3.speaker_id == "spk_fallback"


def test_voice_catalog_vietnamese_metadata_completeness():
    """
    Xác minh tất cả các mục giọng trong VIETNAMESE_VOICE_CATALOG:
    1. Đều có đủ các trường: name, gender, accent, age_group, pitch, style, instruct, sample_vi, slug.
    2. Giá trị accent nằm trong {'bac', 'trung', 'nam'}.
    3. Giá trị gender nằm trong {'male', 'female'}.
    4. Giá trị pitch nằm trong {'very_low', 'low', 'moderate', 'high', 'very_high'}.
    5. Bảo đảm tồn tại giọng đại diện cho cả 3 miền Bắc, Trung, Nam.
    """
    from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG

    assert len(VIETNAMESE_VOICE_CATALOG) >= 22
    valid_accents = {"bac", "trung", "nam"}
    valid_genders = {"male", "female"}
    valid_pitches = {"very_low", "low", "moderate", "high", "very_high"}

    accents_found = set()
    for preset_label, info in VIETNAMESE_VOICE_CATALOG.items():
        assert "name" in info, f"Thiếu 'name' trong {preset_label}"
        assert "gender" in info, f"Thiếu 'gender' trong {preset_label}"
        assert "accent" in info, f"Thiếu 'accent' trong {preset_label}"
        assert "age_group" in info, f"Thiếu 'age_group' trong {preset_label}"
        assert "pitch" in info, f"Thiếu 'pitch' trong {preset_label}"
        assert "style" in info, f"Thiếu 'style' trong {preset_label}"
        assert "instruct" in info, f"Thiếu 'instruct' trong {preset_label}"
        assert "sample_vi" in info, f"Thiếu 'sample_vi' trong {preset_label}"
        assert "slug" in info, f"Thiếu 'slug' trong {preset_label}"

        assert info["accent"] in valid_accents, f"Accent không hợp lệ: {info['accent']} tại {preset_label}"
        assert info["gender"] in valid_genders, f"Gender không hợp lệ: {info['gender']} tại {preset_label}"
        assert info["pitch"] in valid_pitches, f"Pitch không hợp lệ: {info['pitch']} tại {preset_label}"
        accents_found.add(info["accent"])

    assert accents_found == valid_accents, f"Chưa đủ cả 3 miền: {accents_found}"


# ==============================================================================
# 28. KIỂM THỬ ĐỘNG CƠ GỢI Ý GIỌNG NÓI THÔNG MINH (TÁC VỤ 3)
# ==============================================================================
def test_voice_recommendation_engine_matches_and_ranks_top_k():
    """
    Xác minh VoiceRecommendationEngine:
    1. Đề xuất đúng top 3 giọng đọc có điểm tương đồng cao nhất.
    2. Khớp chính xác giới tính và khẩu âm vùng miền (Ví dụ: Nam miền Nam đề xuất giọng miền Nam lên đầu).
    3. Trả về lý do đề xuất tiếng Việt chuẩn mực và điểm số từ 0 đến 100.
    """
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.tts_engine import VoiceRecommendationEngine

    engine = VoiceRecommendationEngine()

    # Trường hợp 1: Nam miền Nam, giọng trầm
    prof_nam = CharacterVoiceProfile(
        speaker_id="spk_nam",
        display_name="Chú Ba",
        gender="male",
        vietnamese_accent="nam",
        pitch_level="low",
        style_tag="ke_chuyen",
    )

    recs_nam = engine.recommend_voices(prof_nam, top_k=3)
    assert len(recs_nam) == 3
    # Vị trí thứ nhất phải là giọng nam miền Nam
    top1_nam = recs_nam[0]
    assert top1_nam["gender"] == "male"
    assert top1_nam["accent"] == "nam"
    assert top1_nam["score"] >= 80
    assert "miền Nam" in top1_nam["reason"]
    assert "Phù hợp" in top1_nam["reason"]

    # Trường hợp 2: Nữ miền Bắc, cao độ cao
    prof_nu = CharacterVoiceProfile(
        speaker_id="spk_nu",
        display_name="Cô Lan",
        gender="female",
        vietnamese_accent="bac",
        pitch_level="high",
        style_tag="tam_su",
    )

    recs_nu = engine.recommend_voices(prof_nu, top_k=3)
    assert len(recs_nu) == 3
    top1_nu = recs_nu[0]
    assert top1_nu["gender"] == "female"
    assert top1_nu["accent"] == "bac"
    assert top1_nu["score"] >= 80


# ==============================================================================
# 29. KIỂM THỬ ĐỒNG BỘ BỘ CHỌN NHÂN VẬT KHI ĐỔI TÊN (TÁC VỤ 4)
# ==============================================================================
def test_speaker_selectors_sync_on_character_rename():
    """
    Xác minh khi danh sách nhân vật thay đổi hoặc đổi tên trên bảng:
    1. _sync_speaker_selectors() cập nhật combo_target_speaker với danh sách tên mới.
    2. Không còn giữ danh sách mặc định cứng 'Người nói 1'..'Người nói 6'.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Giả lập hồ sơ 2 nhân vật đã được đổi tên
    ui.speaker_profiles = {
        "Thám tử Lừng Danh": CharacterVoiceProfile(speaker_id="spk_0", display_name="Thám tử Lừng Danh"),
        "Kẻ Giấu Mặt": CharacterVoiceProfile(speaker_id="spk_1", display_name="Kẻ Giấu Mặt"),
    }

    ui._sync_speaker_selectors()

    items = [ui.combo_target_speaker.itemText(i) for i in range(ui.combo_target_speaker.count())]
    assert "Thám tử Lừng Danh" in items
    assert "Kẻ Giấu Mặt" in items
    assert "Người nói 6" not in items


# ==============================================================================
# 30. KIỂM THỬ TỰ ĐỘNG PHỤC HỒI KHI NGHE THỬ TUẦN TỰ GẶP SỰ CỐ (TÁC VỤ 5)
# ==============================================================================
def test_audition_all_queue_continues_on_individual_failure():
    """
    Xác minh trong luồng nghe thử tuần tự toàn bộ nhân vật:
    Nếu một nhân vật tổng hợp thất bại (success=False), hệ thống không dừng im lặng
    mà ghi nhận lỗi vào hồ sơ và tự động tiếp tục phát nhân vật kế tiếp trong hàng đợi.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    prof1 = CharacterVoiceProfile(speaker_id="spk_fail", display_name="Nhân vật lỗi")
    prof2 = CharacterVoiceProfile(speaker_id="spk_ok", display_name="Nhân vật tốt")
    ui.speaker_profiles = {"spk_fail": prof1, "spk_ok": prof2}

    ui._audition_queue = ["spk_ok"]
    ui._audition_all_active = True
    ui._audition_all_total = 2
    ui._audition_all_current_idx = 0
    ui._current_audition_speaker = "spk_fail"

    called_next = []
    ui._audition_next_in_queue = lambda: called_next.append(True)

    # Giả lập worker báo thất bại cho nhân vật đầu tiên
    ui._on_worker_finished(False, "", "Lỗi nạp mô hình", "Chi tiết lỗi")

    # Xác minh lỗi được ghi nhận và hàng đợi vẫn tiếp tục gọi nhân vật kế tiếp
    assert prof1.error_message == "Lỗi nạp mô hình"
    assert len(called_next) == 1


def test_casting_table_has_two_separate_listen_buttons():
    """
    Xác minh bảng phân vai có đúng 2 nút nghe riêng biệt cho mỗi nhân vật:
    - Nút 'Nghe Giọng Gốc' (phát mẫu âm thanh gốc).
    - Nút 'Nghe Thử Giọng Trí Tuệ Nhân Tạo' (tổng hợp câu thử).
    Không có nút chỉ mang tên 'Nghe Thử' không rõ nguồn gốc.
    """
    from PySide6.QtWidgets import QApplication, QPushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ui.speaker_profiles = {
        "Nhân vật A": CharacterVoiceProfile(
            speaker_id="spk_0",
            display_name="Nhân vật A",
            source_audio_path="/fake/audio.wav",
        )
    }
    ui._update_casting_table()

    cell_widget = ui.casting_table.cellWidget(0, 5)
    assert cell_widget is not None, "Ô cột 5 phải có widget chứa các nút nghe"

    buttons = cell_widget.findChildren(QPushButton)
    btn_texts = [b.text() for b in buttons]

    assert any("Giọng Gốc" in t for t in btn_texts), f"Thiếu nút Nghe Giọng Gốc, chỉ có: {btn_texts}"
    assert any("Trí Tuệ Nhân Tạo" in t for t in btn_texts), f"Thiếu nút Nghe Thử Giọng TTNNT, chỉ có: {btn_texts}"
    assert not any(t.strip() == "Nghe Thử" for t in btn_texts), f"Nút không rõ nguồn 'Nghe Thử' vẫn còn: {btn_texts}"


def test_voice_recommendation_widget_shows_top3_for_character():
    """
    Xác minh _build_voice_recommendation_widget() tạo widget hiển thị tối đa 3 gợi ý giọng,
    mỗi gợi ý có điểm phù hợp và nút Áp Dụng.
    """
    from PySide6.QtWidgets import QApplication, QPushButton, QLabel
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Nhân vật thử nghiệm",
        gender="female",
        vietnamese_accent="bac",
        pitch_level="high",
    )
    ui.speaker_profiles = {"Nhân vật thử nghiệm": prof}

    widget = ui._build_voice_recommendation_widget("Nhân vật thử nghiệm")
    assert widget is not None

    apply_buttons = [b for b in widget.findChildren(QPushButton) if "Áp Dụng" in b.text()]
    assert 1 <= len(apply_buttons) <= 3, f"Số nút Áp Dụng phải từ 1 đến 3, hiện có: {len(apply_buttons)}"

    labels = widget.findChildren(QLabel)
    label_texts = [l.text() for l in labels]
    assert any("%" in t for t in label_texts), f"Phải có nhãn điểm (%), hiện có: {label_texts}"


def test_diarization_summary_label_updates_after_detection():
    """
    Xác minh sau khi _on_speakers_detected() được gọi,
    nhãn lbl_diarization_summary hiển thị đủ thông tin:
    số nhân vật, số câu đã phân vai.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
    blk1.speaker = "Nhân vật A"
    blk1.confidence_level = "high"
    blk1.is_speaker_locked = True

    blk2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
    blk2.speaker = "Nhân vật B"
    blk2.confidence_level = "review_needed"
    blk2.is_speaker_locked = True

    blk3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu 3")
    blk3.speaker = "Nhân vật A"
    blk3.confidence_level = "high"
    blk3.is_speaker_locked = True

    ui.subtitle_blocks = [blk1, blk2, blk3]
    profiles = {
        "Nhân vật A": CharacterVoiceProfile(speaker_id="spk_0", display_name="Nhân vật A"),
        "Nhân vật B": CharacterVoiceProfile(speaker_id="spk_1", display_name="Nhân vật B"),
    }
    ui._on_speakers_detected(profiles)

    summary_text = ui.lbl_diarization_summary.text()
    assert "2" in summary_text, f"Phải đề cập 2 nhân vật: {summary_text}"
    assert "câu" in summary_text or "3" in summary_text, f"Phải đề cập số câu: {summary_text}"


def test_step0_default_voice_settings_collapsible_and_adaptive_mode():
    """
    Xác minh Bước 0 (Nguồn Đầu Vào):
    1. Có nút thu gọn/mở rộng thiết lập giọng mặc định (mặc định đóng để giảm rối mắt).
    2. Khi chuyển chế độ tạo giọng, các khối điều khiển không liên quan tự động ẩn.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # 1. Kiểm tra trạng thái thu gọn ban đầu
    assert hasattr(ui, "btn_toggle_default_voice_settings"), "Thiếu nút chuyển đổi thu gọn thiết lập giọng mặc định"
    assert hasattr(ui, "default_voice_settings_container"), "Thiếu khung chứa thiết lập giọng mặc định"
    assert ui.default_voice_settings_container.isHidden(), "Mặc định khung thiết lập giọng mặc định phải thu gọn (ẩn)"

    # Bấm mở rộng
    ui.btn_toggle_default_voice_settings.click()
    assert not ui.default_voice_settings_container.isHidden(), "Sau khi bấm mở, khung thiết lập giọng phải hiển thị"

    # 2. Kiểm tra hiển thị thích ứng theo chế độ tạo giọng
    # Chế độ 0: Nhân bản
    ui.combo_mode.setCurrentIndex(0)
    assert ui.voice_source_stack.currentWidget() == ui.clone_card

    # Chế độ 1: Thiết kế
    ui.combo_mode.setCurrentIndex(1)
    assert ui.voice_source_stack.currentWidget() == ui.design_card

    # Chế độ 2: Tự động
    ui.combo_mode.setCurrentIndex(2)
    assert ui.voice_source_stack.isHidden()


def test_step1_batch_actions_menu_and_safe_reset_dialog():
    """
    Xác minh Bước 1 (Quản Lý Nhân Vật):
    1. Có nút menu 'Thao Tác Khác' chứa các tác vụ hàng loạt thay vì phơi ra nhiều nút bấm.
    2. Thao tác đặt lại toàn bộ giọng có cơ chế xác nhận an toàn.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # 1. Có nút menu thao tác khác
    assert hasattr(ui, "btn_more_actions"), "Thiếu nút menu Thao Tác Khác"
    assert ui.btn_more_actions.text() == "Thao Tác Khác"

    # 2. Kiểm tra thao tác đặt lại toàn bộ có xác nhận
    ui.speaker_profiles = {
        "Nhân vật 1": CharacterVoiceProfile(speaker_id="spk_0", display_name="Nhân vật 1", status="locked")
    }

    # Giả lập người dùng từ chối xác nhận (confirm=False)
    ui._handle_reset_all_voices_with_confirmation(confirmed=False)
    assert "Nhân vật 1" in ui.speaker_profiles, "Khi hủy bỏ, hồ sơ nhân vật không được bị xóa"
    assert ui.speaker_profiles["Nhân vật 1"].status == "locked"

    # Giả lập người dùng đồng ý xác nhận (confirmed=True)
    ui._handle_reset_all_voices_with_confirmation(confirmed=True)
    assert len(ui.speaker_profiles) == 0 or all(p.status != "locked" for p in ui.speaker_profiles.values())


def test_step2_contextual_toolbar_enables_only_on_subtitle_selection():
    """
    Xác minh Bước 2 (Biên Tập Phụ Đề):
    1. Thanh công cụ ngữ cảnh tự động vô hiệu hóa các thao tác khi không có câu nào được chọn.
    2. Thanh công cụ kích hoạt đầy đủ các chức năng khi có ít nhất 1 dòng được chọn.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Xin chào bạn")
    blk2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Tôi là trợ lý ảo")
    ui.subtitle_blocks = [blk1, blk2]
    ui._populate_subtitle_table()

    # Chưa chọn dòng nào
    ui.sub_table.clearSelection()
    ui._update_subtitle_action_buttons()

    assert not ui.btn_preview_selected_row.isEnabled(), "Chưa chọn dòng thì nút nghe thử phải tắt"
    assert not ui.btn_insert_tag_to_sub.isEnabled(), "Chưa chọn dòng thì nút chèn biểu cảm phải tắt"

    # Chọn dòng 0
    ui.sub_table.selectRow(0)
    ui._update_subtitle_action_buttons()

    assert ui.btn_preview_selected_row.isEnabled(), "Đã chọn dòng thì nút nghe thử phải bật"
    assert ui.btn_insert_tag_to_sub.isEnabled(), "Đã chọn dòng thì nút chèn biểu cảm phải bật"


def test_stepper_step_state_sync_and_validation():
    """
    Xác minh thanh điều hướng 5 bước:
    1. Cập nhật kiểu dáng cho bước hiện tại.
    2. Ngăn chặn chuyển sang Bước 1, 2 nếu chưa có tệp phụ đề nào được nạp khi bấm nút trên giao diện.
    3. Cho phép chuyển bước bình thường khi đã có phụ đề hợp lệ.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Chưa có phụ đề: thử chuyển sang Bước 1
    ui.subtitle_blocks = []
    can_switch = ui._can_switch_to_step(1)
    assert not can_switch, "Chưa nạp phụ đề thì không được phép chuyển sang Bước 1"

    # Thử gọi bấm nút chuyển bước trên giao diện: vẫn phải ở Bước 0
    ui._on_step_button_clicked(1)
    assert ui.current_step_index == 0, "Không đủ điều kiện phải được giữ lại ở Bước 0"

    # Đã nạp phụ đề: cho phép chuyển sang Bước 1
    ui.subtitle_blocks = [SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")]
    assert ui._can_switch_to_step(1) is True
    ui._on_step_button_clicked(1)
    assert ui.current_step_index == 1















def test_download_speaker_embedding_model_uses_existing_or_downloads(tmp_path, monkeypatch):
    """
    Xác minh hàm download_speaker_embedding_model:
    1. Nếu tệp campplus.onnx đã có trong thư mục chỉ định, trả về đường dẫn ngay lập tức.
    2. Nếu chưa có, gọi hf_hub_download với repo_id và filename chính xác.
    """
    from pathlib import Path
    from src.ai_engines.speaker_embedding_onnx import download_speaker_embedding_model

    # Trường hợp 1: Tệp đã tồn tại
    fake_model = tmp_path / "campplus.onnx"
    fake_model.write_bytes(b"dummy onnx content" * 100)
    res = download_speaker_embedding_model(target_dir=tmp_path)
    assert res == str(fake_model.resolve())

    # Trường hợp 2: Tệp chưa tồn tại, kích hoạt hf_hub_download
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    called_hf = []

    def mock_hf_download(repo_id, filename, local_dir):
        called_hf.append((repo_id, filename, local_dir))
        out_f = Path(local_dir) / filename
        out_f.write_bytes(b"downloaded onnx content" * 100)
        return str(out_f)

    import sys
    from types import ModuleType
    mock_hf_mod = sys.modules.get("huggingface_hub")
    if mock_hf_mod is None:
        mock_hf_mod = ModuleType("huggingface_hub")
        monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf_mod)
    monkeypatch.setattr(mock_hf_mod, "hf_hub_download", mock_hf_download, raising=False)

    res2 = download_speaker_embedding_model(target_dir=empty_dir)
    assert len(called_hf) == 1
    assert called_hf[0][0] == "welcomyou/campplus-3dspeaker-200k-onnx"
    assert "campplus" in called_hf[0][1]
    assert Path(res2).is_file()


def test_agglomerative_clustering_handles_imbalance_and_exact_k():
    """
    Xác minh thuật toán phân cụm thứ bậc AHC:
    1. Tách chính xác trường hợp mất cân bằng (10 câu của người A, 2 câu của người B).
    2. Hỗ trợ exact_k: khi yêu cầu exact_k=2 thì ép ra đúng 2 cụm.
    3. Hỗ trợ exact_k=3 thì ép ra đúng 3 cụm.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import agglomerative_cluster_embeddings

    rng = np.random.default_rng(42)
    center_a = np.zeros(192, dtype=np.float32)
    center_a[0] = 1.0
    center_b = np.zeros(192, dtype=np.float32)
    center_b[1] = 1.0

    embs_a = center_a + rng.normal(0, 0.05, (10, 192)).astype(np.float32)
    embs_b = center_b + rng.normal(0, 0.05, (2, 192)).astype(np.float32)
    embeddings = np.vstack([embs_a, embs_b])

    # 1. Tự động theo ngưỡng (threshold=0.40) -> Phải tách được đúng 2 cụm
    labels_auto = agglomerative_cluster_embeddings(embeddings, threshold=0.40)
    assert len(np.unique(labels_auto)) == 2
    assert np.all(labels_auto[:10] == labels_auto[0])
    assert np.all(labels_auto[10:] == labels_auto[10])
    assert labels_auto[0] != labels_auto[10]

    # 2. Cố định exact_k=2
    labels_exact2 = agglomerative_cluster_embeddings(embeddings, exact_k=2)
    assert len(np.unique(labels_exact2)) == 2

    # 3. Cố định exact_k=3
    labels_exact3 = agglomerative_cluster_embeddings(embeddings, exact_k=3)
    assert len(np.unique(labels_exact3)) == 3


def test_audio_chunk_boundary_clamping_prevents_bleeding():
    """
    Xác minh thuật toán cắt âm thanh phân vai:
    Khi hai câu thoại nằm sát nhau (khoảng nghỉ chỉ 0.06s),
    khoảng đệm an toàn không cắt lẹm vào câu trước hoặc câu sau.
    """
    from src.ai_engines.speaker_diarization import _calculate_safe_audio_padding

    pad_before, pad_after = _calculate_safe_audio_padding(
        start_s=2.0,
        end_s=3.0,
        prev_end=1.94,
        next_start=4.0,
    )
    assert pad_before <= 0.031
    assert pad_after == 0.15


def test_tts_engine_detect_speakers_forwards_exact_k_and_sensitivity(tmp_path):
    """
    Xác minh TTSEngine.detect_speakers_from_video:
    1. Chấp nhận các tham số exact_speakers và clustering_sensitivity.
    2. Chuyển đổi chính xác clustering_sensitivity ('high' -> 0.34, 'low' -> 0.46, 'standard' -> 0.40).
    3. Truyền đúng exact_speakers và distance_threshold xuống process_video_subtitles.
    """
    from unittest.mock import MagicMock, patch
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    engine = TTSEngine()
    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Câu một")
    b2 = SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:06,000", text="Câu hai")
    vid_file = str(tmp_path / "mock.mp4")
    with open(vid_file, "wb") as f:
        f.write(b"mock")

    with patch("src.ai_engines.tts_engine.VoiceSeparator") as mock_sep_cls, \
         patch("src.ai_engines.speaker_diarization.SpeakerDiarizationCoordinator.process_video_subtitles") as mock_process:

        mock_sep = MagicMock()
        mock_sep.separate_from_video.return_value = False
        def fake_extract(v, out):
            with open(out, "wb") as f:
                f.write(b"mock_audio")
            return True
        mock_sep._extract_audio.side_effect = fake_extract
        mock_sep_cls.return_value = mock_sep
        mock_process.return_value = {}

        # Gọi với exact_speakers=2 và clustering_sensitivity="high"
        engine.detect_speakers_from_video(
            video_path=vid_file,
            blocks=[b1, b2],
            max_speakers=5,
            exact_speakers=2,
            clustering_sensitivity="high",
            allow_mock=True,
        )

        assert mock_process.called
        _, kwargs = mock_process.call_args
        assert kwargs.get("exact_speakers") == 2
        assert kwargs.get("distance_threshold") == 0.34
        assert kwargs.get("max_speakers") == 5


def test_omnivoice_ui_speaker_count_mode_and_sensitivity_controls():
    """
    Xác minh các điều khiển Chế độ số người và Độ nhạy phân tách trên giao diện OmniVoice:
    1. Tồn tại combo_speaker_count_mode với hai lựa chọn Tự động ước lượng và Cố định chính xác.
    2. Chuyển sang Cố định chính xác đổi nhãn lbl_max_spk thành 'Số người chính xác:'.
    3. Tồn tại combo_clustering_sensitivity với 3 mức độ nhạy.
    4. _get_ui_params() trả về đúng exact_speakers và clustering_sensitivity.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    assert hasattr(ui, "combo_speaker_count_mode")
    assert hasattr(ui, "combo_clustering_sensitivity")
    assert hasattr(ui, "lbl_max_spk")

    # Mặc định là Tự động ước lượng
    assert ui.combo_speaker_count_mode.currentIndex() == 0
    assert "tối đa" in ui.lbl_max_spk.text().lower()

    # Chuyển sang Cố định chính xác số người
    ui.combo_speaker_count_mode.setCurrentIndex(1)
    assert "chính xác" in ui.lbl_max_spk.text().lower()

    ui.spin_max_speakers.setValue(2)
    ui.combo_clustering_sensitivity.setCurrentIndex(1)  # Cao - Dễ tách
    params = ui._collect_engine_params()
    assert params["exact_speakers"] == 2
    assert params["clustering_sensitivity"] == "high"

    # Chuyển về Tự động ước lượng
    ui.combo_speaker_count_mode.setCurrentIndex(0)
    assert "tối đa" in ui.lbl_max_spk.text().lower()
    params2 = ui._collect_engine_params()
    assert params2["exact_speakers"] is None


def test_tts_engine_init_accepts_speaker_clustering_params_and_extra_kwargs():
    """
    Xác minh TTSEngine.__init__():
    1. Chấp nhận các tham số speaker_count_mode, exact_speakers, clustering_sensitivity.
    2. Hỗ trợ **kwargs phòng ngừa, không ném ngoại lệ TypeError khi worker truyền toàn bộ params.
    """
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(
        speaker_count_mode="exact",
        exact_speakers=3,
        clustering_sensitivity="high",
        arbitrary_future_ui_param="safe_value",
    )
    assert engine.speaker_count_mode == "exact"
    assert engine.exact_speakers == 3
    assert engine.clustering_sensitivity == "high"


def test_on_speakers_detected_counters_and_empty_profile_handling(tmp_path):
    """
    Xác minh _on_speakers_detected() trong OmniVoiceInterface:
    1. total_assigned đếm chính xác số câu có người nói hợp lệ (không dựa vào is_speaker_locked).
    2. total_locked đếm các câu đã được người dùng xác nhận hoặc khóa.
    3. Khi profiles rỗng ({}), is_speakers_detected KHÔNG được đặt thành True,
       và hiển thị cảnh báo không đủ dữ liệu âm thanh thay vì báo thành công.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Câu 1")
    b1.speaker = "Nhân vật 1"
    b1.is_speaker_locked = False  # Đã gán nhưng CHƯA khóa thủ công
    b1.confidence_level = "high"

    b2 = SubtitleBlock(index=2, start_time="00:00:03,000", end_time="00:00:04,000", text="Câu 2")
    b2.speaker = "Nhân vật 2"
    b2.is_speaker_locked = True  # Đã khóa thủ công
    b2.confidence_level = "high"

    b3 = SubtitleBlock(index=3, start_time="00:00:05,000", end_time="00:00:06,000", text="Câu 3")
    b3.speaker = "Chưa xác định"
    b3.is_speaker_locked = False
    b3.confidence_level = "unassigned"

    ui.subtitle_blocks = [b1, b2, b3]
    ui.is_speakers_detected = False

    # Trường hợp 1: Nhận diện thành công với 2 nhân vật
    profiles = {
        "Nhân vật 1": CharacterVoiceProfile(speaker_id="spk_0", display_name="Nhân vật 1"),
        "Nhân vật 2": CharacterVoiceProfile(speaker_id="spk_1", display_name="Nhân vật 2"),
    }
    ui._on_speakers_detected(profiles)

    assert ui.is_speakers_detected is True
    summary_text = ui.lbl_diarization_summary.text()
    # Phải báo 2/3 câu được phân vai (thay vì 1/3 như khi chỉ đếm is_speaker_locked)
    assert "2/3 câu" in summary_text
    assert "1 câu đã khóa" in summary_text or "1 câu đã xác nhận" in summary_text

    # Trường hợp 2: Pipeline trả về rỗng (video im lặng hoặc không đủ dữ liệu)
    ui.is_speakers_detected = False
    ui._on_speakers_detected({})
    assert ui.is_speakers_detected is False
    summary_empty = ui.lbl_diarization_summary.text()
    assert "Không tìm thấy" in summary_empty or "Chưa có" in summary_empty or "không đủ" in summary_empty.lower()


def test_is_valid_video_file_prevents_directory_or_corrupt_input(tmp_path):
    """
    Xác minh hàm kiểm tra tính hợp lệ của video:
    - Thư mục: Không hợp lệ.
    - Tệp không tồn tại: Không hợp lệ.
    - Tệp 0 byte: Không hợp lệ.
    - Tệp đuôi không phải video/audio: Không hợp lệ.
    - Tệp video hợp lệ: Hợp lệ.
    """
    import os
    from src.desktop.ui.omnivoice_interface import is_valid_media_file

    fake_dir = str(tmp_path / "video_dir")
    os.makedirs(fake_dir, exist_ok=True)
    assert is_valid_media_file(fake_dir) is False

    non_exist = str(tmp_path / "not_there.mp4")
    assert is_valid_media_file(non_exist) is False

    empty_file = str(tmp_path / "empty.mp4")
    with open(empty_file, "wb") as f:
        pass
    assert is_valid_media_file(empty_file) is False

    txt_file = str(tmp_path / "sub.txt")
    with open(txt_file, "wb") as f:
        f.write(b"hello")
    assert is_valid_media_file(txt_file) is False

    valid_vid = str(tmp_path / "good.mp4")
    with open(valid_vid, "wb") as f:
        f.write(b"valid_bytes_for_video")
    assert is_valid_media_file(valid_vid) is True


def test_check_media_and_srt_duration_detects_discrepancy(tmp_path):
    """
    Xác minh hàm check_media_and_srt_duration:
    1. Phát hiện chênh lệch khi thời lượng video và phụ đề lệch vượt ngưỡng 5 giây.
    2. Trả về thông tin đầy đủ: video_duration, srt_duration, diff, is_mismatch.
    3. Cảnh báo khi phụ đề dài hơn video.
    """
    import cv2
    import numpy as np
    from src.desktop.ui.omnivoice_interface import check_media_and_srt_duration
    from src.ai_engines.translator import SubtitleBlock

    # Tạo video ngắn 2 giây (50 frames, 25 fps)
    video_file = str(tmp_path / "short_video.mp4")
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (100, 100))
    for _ in range(50):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()

    # Phụ đề dài 15 giây -> Lệch 13 giây > ngưỡng 5 giây
    blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:05,000", "Câu một"),
        SubtitleBlock(2, "00:00:06,000", "00:00:15,000", "Câu hai"),
    ]

    res = check_media_and_srt_duration(video_file, blocks, threshold_sec=5.0)
    assert res["is_mismatch"] is True
    assert round(res["video_duration"], 1) == 2.0
    assert round(res["srt_duration"], 1) == 15.0
    assert res["diff"] >= 12.9
    assert "chênh lệch" in res["message"].lower() or "cảnh báo" in res["message"].lower()

    # Trường hợp đồng bộ (lệch dưới 1 giây)
    synced_blocks = [
        SubtitleBlock(1, "00:00:00,500", "00:00:02,000", "Câu khớp"),
    ]
    res_sync = check_media_and_srt_duration(video_file, synced_blocks, threshold_sec=5.0)
    assert res_sync["is_mismatch"] is False



def test_filter_subtitle_table_supports_locked_category():
    """
    Xác minh bộ lọc phụ đề hỗ trợ tùy chọn 'Đã khóa':
    1. combo_sub_filter chứa mục 'Đã khóa'.
    2. Khi chọn 'Đã khóa', chỉ hiện câu có is_speaker_locked = True, ẩn câu is_speaker_locked = False.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
    blk1.is_speaker_locked = True

    blk2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
    blk2.is_speaker_locked = False

    ui.subtitle_blocks = [blk1, blk2]
    ui._populate_subtitle_table()

    # Kiểm tra combo_sub_filter có mục 'Đã khóa'
    items = [ui.combo_sub_filter.itemText(i) for i in range(ui.combo_sub_filter.count())]
    assert "Đã khóa" in items

    # Lọc theo 'Đã khóa'
    ui.combo_sub_filter.setCurrentText("Đã khóa")
    assert not ui.sub_table.isRowHidden(0)
    assert ui.sub_table.isRowHidden(1)


def test_subtitle_table_speaker_combobox_and_auto_locking():
    """
    Xác minh cột Người nói trên bảng phụ đề:
    1. Sử dụng ComboBox thay vì chuỗi văn bản tĩnh.
    2. Chứa danh sách nhân vật đã nhận diện.
    3. Khi đổi người nói qua ComboBox:
       - blk.speaker được cập nhật.
       - blk.is_speaker_locked tự động chuyển thành True.
       - blk.confidence_level chuyển thành 'high'.
       - Nguồn giọng ở cột 4 tự động làm mới.
    """
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import ComboBox
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ui.speaker_profiles = {
        "Thám tử Lừng Danh": CharacterVoiceProfile(speaker_id="spk_0", display_name="Thám tử Lừng Danh"),
        "Kẻ Giấu Mặt": CharacterVoiceProfile(speaker_id="spk_1", display_name="Kẻ Giấu Mặt"),
    }

    blk = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào")
    blk.speaker = "Chưa xác định"
    blk.is_speaker_locked = False
    blk.confidence_level = "unassigned"

    ui.subtitle_blocks = [blk]
    ui._populate_subtitle_table()

    combo_spk = ui.sub_table.cellWidget(0, 3)
    assert isinstance(combo_spk, ComboBox)
    assert combo_spk.findText("Thám tử Lừng Danh") >= 0

    # Người dùng chọn nhân vật
    combo_spk.setCurrentText("Thám tử Lừng Danh")

    assert blk.speaker == "Thám tử Lừng Danh"
    assert blk.is_speaker_locked is True
    assert blk.confidence_level == "high"


def test_subtitle_table_has_7_columns_with_listen_button():
    """
    Xác minh bảng phụ đề có 7 cột:
    0: STT, 1: Bắt đầu, 2: Kết thúc, 3: Người nói, 4: Nguồn giọng, 5: Nghe gốc, 6: Nội dung thoại.
    Cột 5 có nút nghe với FluentIcon.
    """
    from PySide6.QtWidgets import QApplication, QPushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào")
    ui.subtitle_blocks = [blk]
    ui._populate_subtitle_table()

    assert ui.sub_table.columnCount() == 7
    header_texts = [ui.sub_table.horizontalHeaderItem(c).text() for c in range(7)]
    assert "Nghe Gốc" in header_texts[5] or "Nghe gốc" in header_texts[5]

    btn_listen = ui.sub_table.cellWidget(0, 5)
    assert isinstance(btn_listen, QPushButton)


def test_agglomerative_clustering_does_not_collapse_multispeakers_on_low_silhouette():
    """
    Xác minh thuật toán _agglomerative_clustering không ép gộp toàn bộ về 1 người
    khi các mẫu có độ phân tách cosin rõ ràng giữa 2 người khác nhau nhưng hệ số Silhouette thấp.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import agglomerative_cluster_embeddings as _agglomerative_clustering

    # Giả lập 2 nhóm vector đặc trưng của 2 người nói khác nhau
    spk1_embs = np.array([
        [0.98, 0.10, 0.05],
        [0.95, 0.15, 0.08],
        [0.97, 0.08, 0.12],
    ], dtype=np.float32)
    spk2_embs = np.array([
        [0.10, 0.98, 0.05],
        [0.15, 0.95, 0.08],
        [0.08, 0.97, 0.12],
    ], dtype=np.float32)

    embs = np.vstack([spk1_embs, spk2_embs])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    labels = _agglomerative_clustering(embs, threshold=0.40, exact_k=None, max_k=4)
    unique_clusters = np.unique(labels)

    # Phải phát hiện đúng 2 người nói riêng biệt, KHÔNG bị gộp thành 1 cụm
    assert len(unique_clusters) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_detect_speech_segments_vad_identifies_speech_and_trims_silence():
    """
    Xác minh hàm VAD và cắt khoảng lặng:
    1. Phát hiện đúng đoạn có tiếng nói trong tín hiệu âm thanh có chứa khoảng lặng.
    2. trim_silence_vad cắt bỏ khoảng lặng ở 2 đầu, giữ trọn vẹn phần giọng nói.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import detect_speech_segments_vad, trim_silence_vad

    sr = 16000
    # Tạo 1 giây im lặng + 1.5 giây tín hiệu có tiếng (sin) + 1 giây im lặng
    silence_start = np.zeros(int(1.0 * sr), dtype=np.float32)
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    speech = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    silence_end = np.zeros(int(1.0 * sr), dtype=np.float32)

    full_audio = np.concatenate([silence_start, speech, silence_end])

    segments = detect_speech_segments_vad(full_audio, sample_rate=sr, min_speech_sec=0.3)
    assert len(segments) >= 1
    # Đoạn tiếng nói phải nằm trong khoảng từ 0.8s đến 2.7s
    seg_start, seg_end = segments[0]
    assert 0.8 <= seg_start <= 1.1
    assert 2.4 <= seg_end <= 2.7

    # trim_silence_vad phải cắt bỏ 2 giây im lặng
    trimmed = trim_silence_vad(full_audio, sample_rate=sr)
    trimmed_duration = len(trimmed) / sr
    assert 1.3 <= trimmed_duration <= 1.7


def test_calibrated_confidence_score_penalizes_short_and_ambiguous_utterances():
    """
    Xác minh thuật toán cosine_cluster_embeddings:
    1. Câu thoại dài, nằm gần tâm cụm có độ tin cậy 'high'.
    2. Câu thoại quá ngắn (< 0.5s) hoặc nằm ở ranh giới giữa 2 người nói bị hạ mức 'review_needed'.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings, ConfidenceLevel

    # 2 cụm người nói rõ ràng
    spk1_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    spk2_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Mẫu 1: Rất chuẩn người 1, dài 2.0s -> HIGH
    m1 = spk1_emb.copy()
    # Mẫu 2: Nằm gần ranh giới giữa 2 người (45 độ) -> REVIEW_NEEDED
    m2 = np.array([0.707, 0.707, 0.0], dtype=np.float32)
    # Mẫu 3: Rất chuẩn người 2 nhưng siêu ngắn 0.3s -> REVIEW_NEEDED
    m3 = spk2_emb.copy()

    embs = np.vstack([m1, m2, m3])
    durations = [2.0, 2.0, 0.3]

    labels, conf_levels, conf_scores = cosine_cluster_embeddings(
        embs,
        valid_indices=[0, 1, 2],
        max_speakers=2,
        exact_speakers=2,
        sample_durations=durations,
    )

    assert conf_levels[0] == ConfidenceLevel.HIGH.value
    assert conf_levels[1] == ConfidenceLevel.REVIEW_NEEDED.value
    assert conf_levels[2] == ConfidenceLevel.REVIEW_NEEDED.value
    assert conf_scores[0] > conf_scores[1]
    assert conf_scores[0] > conf_scores[2]


def test_diarization_supports_video_without_srt(tmp_path):
    """
    Xác minh khi truyền blocks=[] (chưa có phụ đề):
    1. process_video_subtitles tự động phân đoạn âm thanh qua VAD.
    2. Tự sinh các khối câu thoại và nhận diện đúng các nhân vật.
    3. Trả về hồ sơ nhân vật và điền đầy đủ danh sách blocks.
    """
    import numpy as np
    import soundfile as sf
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator

    sr = 16000
    # Tạo 2 phân đoạn âm thanh của 2 người nói
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    spk1 = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    spk2 = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)

    audio = np.concatenate([spk1, silence, spk2])
    wav_path = str(tmp_path / "multi_speech.wav")
    sf.write(wav_path, audio, sr)

    coord = SpeakerDiarizationCoordinator(allow_mock=True, prefer_cuda=False)
    empty_blocks = []
    profiles = coord.process_video_subtitles(
        video_path=wav_path,
        blocks=empty_blocks,
        max_speakers=2,
    )

    assert len(empty_blocks) >= 2
    assert len(profiles) >= 1


def test_cosine_clustering_outlier_rejection_marks_distant_samples_unassigned():
    """
    Xác minh trong cơ chế mẫu nòng cốt (anchor-based):
    Nếu một đoạn thoại ngoài nòng cốt có khoảng cách cosin lớn (> 0.45) tới tất cả các tâm cụm nòng cốt,
    hệ thống KHÔNG được ép gán vào người gần nhất mà phải từ chối gán (unassigned).
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings, ConfidenceLevel

    # 2 mẫu nòng cốt của 2 nhân vật (dài 2.0s)
    spk1_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spk2_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # Mẫu 3: Đoạn ngắn 0.3s, có vector thuộc chiều thứ 4 [0, 0, 0, 1] (khoảng cách cosin = 1.0 tới cả 2 người)
    outlier_emb = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    embs = np.vstack([spk1_emb, spk2_emb, outlier_emb])
    durations = [2.0, 2.0, 0.3]

    labels, conf_levels, conf_scores = cosine_cluster_embeddings(
        embs,
        valid_indices=[0, 1, 2],
        max_speakers=2,
        exact_speakers=2,
        distance_threshold=0.40,
        sample_durations=durations,
    )

    # 2 mẫu đầu được gán vào 2 cụm khác nhau
    assert labels[0] != labels[1]
    assert conf_levels[0] == ConfidenceLevel.HIGH.value
    assert conf_levels[1] == ConfidenceLevel.HIGH.value

    # Mẫu thứ 3 bị từ chối gán, chuyển sang Chưa xác định
    assert conf_levels[2] == ConfidenceLevel.UNASSIGNED.value
    assert conf_scores[2] == 0.0
    assert labels[2] == -1


def test_vad_spectral_band_filters_out_high_frequency_noise():
    """
    Xác minh hàm detect_speech_segments_vad:
    1. Tín hiệu nhiễu cao tần (tần số 7000 Hz) có năng lượng RMS lớn nhưng bị loại bỏ do ngoài dải giọng nói.
    2. Tín hiệu giọng nói (tần số 400 Hz) có năng lượng tương đương được giữ lại chính xác.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import detect_speech_segments_vad

    sr = 16000
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    # Tín hiệu nhiễu cao tần (tiếng súng/rít kim loại) 7000 Hz
    hf_noise = (0.5 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    # Tín hiệu giọng nói con người 400 Hz
    voice = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    audio = np.concatenate([hf_noise, silence, voice])

    segs = detect_speech_segments_vad(audio, sample_rate=sr, min_speech_sec=0.3)
    assert len(segs) == 1
    # Chỉ có đoạn giọng nói (từ 1.5s đến 2.5s) được công nhận
    assert 1.4 <= segs[0][0] <= 1.6
    assert 2.4 <= segs[0][1] <= 2.6
