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



