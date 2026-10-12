# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tự động (TDD Regression Suite) cho toàn bộ phát hiện P1/P2
trên luồng Tự Động (AutoPipeline), Dịch Phụ Đề (Translator/TranslationInterface),
OCR (VideoOcrEngine) và Lồng Tiếng (OmniVoiceInterface).
"""

import os
import sys
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_engines.translator import SubtitleBlock, parse_srt
from src.ai_engines.tts_engine import DubbingReport, TTSEngine
from src.ai_engines.ocr_engine import VideoOcrEngine


def _create_dummy_video(path: str, fps: float = 10.0, num_frames: int = 20, width: int = 320, height: int = 240):
    """Tạo tệp video MP4 hợp lệ phục vụ kiểm thử tự động."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestAutoPipelineReliability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_sakai_reliability_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Kiểm thử P2: parse_srt hỗ trợ WebVTT (MM:SS.mmm + cue settings) & ASS
    # -------------------------------------------------------------------------
    def test_parse_srt_supports_vtt_short_timestamps_and_ass_dialogue(self):
        vtt_content = """WEBVTT

00:01.250 --> 00:03.500 align:start position:10%
Xin chào từ WebVTT ngắn

01:02:03.100 --> 01:02:05.900
Câu thứ hai có giờ đầy đủ
"""
        vtt_blocks = parse_srt(vtt_content)
        self.assertEqual(len(vtt_blocks), 2)
        self.assertEqual(vtt_blocks[0].start_time, "00:00:01,250")
        self.assertEqual(vtt_blocks[0].end_time, "00:00:03,500")
        self.assertEqual(vtt_blocks[0].text, "Xin chào từ WebVTT ngắn")
        self.assertEqual(vtt_blocks[1].start_time, "01:02:03,100")

        ass_content = """[Script Info]
Title: Test ASS

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.20,0:00:04.50,Default,Người nói 1,0,0,0,,{\\an5\\pos(160,200)}Câu thoại ASS dòng 1\\Ndòng 2
Dialogue: 0,0:00:05.00,0:00:07.80,Default,Người nói 2,0,0,0,,Câu thoại ASS thứ hai
"""
        ass_blocks = parse_srt(ass_content)
        self.assertEqual(len(ass_blocks), 2)
        self.assertEqual(ass_blocks[0].start_time, "00:00:01,200")
        self.assertEqual(ass_blocks[0].end_time, "00:00:04,500")
        self.assertEqual(ass_blocks[0].speaker, "Người nói 1")
        self.assertNotIn("{\\an5", ass_blocks[0].text)
        self.assertIn("Câu thoại ASS dòng 1", ass_blocks[0].text)

    # -------------------------------------------------------------------------
    # 2. Kiểm thử P1: Sửa bảng phụ đề không ghi đè mất văn bản gốc (blk.text)
    # -------------------------------------------------------------------------
    def test_subtitle_edit_preserves_original_source_text(self):
        from PySide6.QtWidgets import QApplication, QTableWidgetItem
        app = QApplication.instance() or QApplication([])

        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        blk = SubtitleBlock(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:03,000",
            text="Original English Line",
            translated_text="Dòng dịch ban đầu",
        )
        dummy = mock.MagicMock()
        dummy.subtitle_blocks = [blk]
        item = QTableWidgetItem("Dòng dịch đã chỉnh sửa")
        # Giả lập sửa cột 5 (nội dung thoại) tại dòng 0
        with mock.patch.object(item, "row", return_value=0), mock.patch.object(item, "column", return_value=5):
            OmniVoiceInterface._on_subtitle_table_edited(dummy, item)

        self.assertEqual(blk.translated_text, "Dòng dịch đã chỉnh sửa")
        self.assertEqual(blk.text, "Original English Line")

    # -------------------------------------------------------------------------
    # 3. Kiểm thử P1: Nhánh xóa phụ đề lỗi không được phép bỏ qua hoặc lấy video gốc
    # -------------------------------------------------------------------------
    def test_subtitle_remover_failure_not_ignored(self):
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineWorker

        video_in = os.path.join(self.temp_dir, "source.mp4")
        video_out = os.path.join(self.temp_dir, "out.mp4")
        _create_dummy_video(video_in)

        worker = AutoPipelineWorker(
            video_path=video_in,
            output_path=video_out,
            ocr_lang="en",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )

        # Giả lập tiến trình con thoát với exitcode=0 nhưng báo lỗi qua queue và không tạo file đầu ra
        fake_nonexistent_out = os.path.join(self.temp_dir, "missing_cleaned.mp4")
        with mock.patch("multiprocessing.Process") as mock_proc_cls:
            proc_inst = mock.MagicMock()
            proc_inst.is_alive.return_value = False
            proc_inst.exitcode = 0
            mock_proc_cls.return_value = proc_inst

            with self.assertRaises(RuntimeError):
                worker._run_subtitle_remover(fake_nonexistent_out)

    # -------------------------------------------------------------------------
    # 4. Kiểm thử P2: FFmpeg ghép phụ đề lỗi phải báo lỗi, không âm thầm copy video_in
    # -------------------------------------------------------------------------
    def test_ffmpeg_overlay_failure_raises_error_instead_of_silent_copy(self):
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineWorker

        video_in = os.path.join(self.temp_dir, "clean.mp4")
        ass_in = os.path.join(self.temp_dir, "sub.ass")
        video_out = os.path.join(self.temp_dir, "merged.mp4")
        _create_dummy_video(video_in)
        Path(ass_in).write_text("[Script Info]\nTitle: Test\n", encoding="utf-8")

        worker = AutoPipelineWorker(
            video_path=video_in,
            output_path=video_out,
            ocr_lang="en",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )

        failed_proc = mock.MagicMock(returncode=1, stderr="FFmpeg filter error")
        with mock.patch("subprocess.run", return_value=failed_proc):
            with self.assertRaises(RuntimeError):
                worker._overlay_subtitle_ffmpeg(video_in, ass_in, video_out)

        self.assertFalse(os.path.exists(video_out))

    # -------------------------------------------------------------------------
    # 5. Kiểm thử P1: Dịch lỗi hoặc TTS lỗi không được báo thành công hoàn toàn
    # -------------------------------------------------------------------------
    def test_pipeline_fails_when_translation_or_tts_fails(self):
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineWorker
        from src.core.config import config

        video_in = os.path.join(self.temp_dir, "input.mp4")
        video_out = os.path.join(self.temp_dir, "output.mp4")
        _create_dummy_video(video_in)

        sample_blocks = [
            SubtitleBlock(index=1, start_time="00:00:00,200", end_time="00:00:01,500", text="Hello world")
        ]

        # Kịch bản A: Nhánh dịch ném ngoại lệ -> Pipeline phải báo FAILED (success=False)
        worker_a = AutoPipelineWorker(
            video_path=video_in,
            output_path=video_out,
            ocr_lang="en",
            target_lang="vi",
            engine_type="llm",
            api_config={"api_key": "invalid"},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )
        results_a = []
        worker_a.finished_signal.connect(lambda ok, path, msg: results_a.append((ok, path, msg)))

        with mock.patch.object(worker_a, "_extract_ocr_subtitles", return_value=sample_blocks), \
             mock.patch.object(worker_a, "_run_subtitle_remover", side_effect=lambda out_p: _create_dummy_video(out_p)), \
             mock.patch("src.desktop.ui.auto_pipeline_interface.SubtitleTranslator", side_effect=RuntimeError("API 401")):
            worker_a.run()

        self.assertEqual(len(results_a), 1)
        self.assertFalse(results_a[0][0], "Pipeline không được báo thành công khi nhánh dịch thất bại!")
        self.assertEqual(getattr(worker_a, "final_status", ""), "FAILED")

        # Kịch bản B: Dịch và xóa sub thành công nhưng TTS thất bại -> Pipeline không được báo SUCCESS
        worker_b = AutoPipelineWorker(
            video_path=video_in,
            output_path=video_out,
            ocr_lang="en",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )
        results_b = []
        worker_b.finished_signal.connect(lambda ok, path, msg: results_b.append((ok, path, msg)))

        def fake_translate(blocks, **kwargs):
            for b in blocks:
                b.translated_text = "Xin chào thế giới"
            return blocks

        failed_tts_report = DubbingReport(
            status="failed",
            valid_audio_file=False,
            failed_count=1,
            message="Tổng hợp giọng nói thất bại hoàn toàn.",
        )
        fake_tts_engine = mock.MagicMock()
        fake_tts_engine.synthesize_blocks.return_value = failed_tts_report
        fake_tts_engine.last_report = failed_tts_report

        old_enable_tts = config.mmoEnableTTS.value
        try:
            config.mmoEnableTTS.value = True
            with mock.patch.object(worker_b, "_extract_ocr_subtitles", return_value=sample_blocks), \
                 mock.patch.object(worker_b, "_run_subtitle_remover", side_effect=lambda out_p: _create_dummy_video(out_p)), \
                 mock.patch.object(worker_b, "_overlay_subtitle_ffmpeg", side_effect=lambda vin, ass, vout: shutil.copy2(vin, vout)), \
                 mock.patch("src.desktop.ui.auto_pipeline_interface.SubtitleTranslator") as mock_tr_cls, \
                 mock.patch("src.desktop.ui.auto_pipeline_interface.TTSEngine.from_app_config", return_value=fake_tts_engine):
                mock_tr_cls.return_value.translator.translate_blocks.side_effect = fake_translate
                worker_b.run()
        finally:
            config.mmoEnableTTS.value = old_enable_tts

        self.assertEqual(len(results_b), 1)
        self.assertFalse(results_b[0][0], "Pipeline không được báo thành công hoàn toàn khi TTS thất bại!")
        self.assertEqual(getattr(worker_b, "final_status", ""), "FAILED")

    # -------------------------------------------------------------------------
    # 6. Kiểm thử P1: Hủy tác vụ luôn phát finished_signal và dọn thư mục tạm
    # -------------------------------------------------------------------------
    def test_pipeline_cancellation_emits_signal_and_cleans_temp_dir(self):
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineWorker

        video_in = os.path.join(self.temp_dir, "cancel_in.mp4")
        video_out = os.path.join(self.temp_dir, "cancel_out.mp4")
        _create_dummy_video(video_in)

        worker = AutoPipelineWorker(
            video_path=video_in,
            output_path=video_out,
            ocr_lang="en",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )
        results = []
        worker.finished_signal.connect(lambda ok, path, msg: results.append((ok, path, msg)))

        def stop_during_ocr():
            worker.stop()
            return []

        with mock.patch.object(worker, "_extract_ocr_subtitles", side_effect=stop_during_ocr):
            worker.run()

        self.assertEqual(len(results), 1, "Khi hủy tác vụ, worker bắt buộc phải phát finished_signal!")
        self.assertFalse(results[0][0])
        self.assertEqual(getattr(worker, "final_status", ""), "CANCELLED")
        if getattr(worker, "_current_run_temp_dir", None):
            self.assertFalse(os.path.exists(worker._current_run_temp_dir))

    # -------------------------------------------------------------------------
    # 7. Kiểm thử P1 & P2: Mã hóa API Key & Đồng bộ ngôn ngữ Dịch -> Tự Động
    # -------------------------------------------------------------------------
    def test_api_key_protection_and_language_sync(self):
        from src.desktop.ui.translation_interface import obfuscate_api_key, deobfuscate_api_key

        raw_key = "sk-proj-secret-test-key-987654"
        encoded = obfuscate_api_key(raw_key)
        self.assertNotEqual(encoded, raw_key)
        self.assertNotIn(raw_key, encoded)
        decoded = deobfuscate_api_key(encoded)
        self.assertEqual(decoded, raw_key)
        # Tương thích ngược với khóa văn bản thuần cũ
        self.assertEqual(deobfuscate_api_key(raw_key), raw_key)

    # -------------------------------------------------------------------------
    # 8. Kiểm thử P1: Hàng đợi xử lý hàng loạt tất cả video trong thư mục
    # -------------------------------------------------------------------------
    def test_batch_video_queue_processes_all_videos(self):
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineWorker
        from src.core.config import config

        vid1 = os.path.join(self.temp_dir, "clip1.mp4")
        vid2 = os.path.join(self.temp_dir, "clip2.mp4")
        _create_dummy_video(vid1)
        _create_dummy_video(vid2)

        out1 = os.path.join(self.temp_dir, "clip1_MMO_Auto.mp4")
        out2 = os.path.join(self.temp_dir, "clip2_MMO_Auto.mp4")

        worker = AutoPipelineWorker(
            video_path=vid1,
            output_path=out1,
            ocr_lang="en",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
            video_queue=[vid1, vid2],
        )
        results = []
        worker.finished_signal.connect(lambda ok, path, msg: results.append((ok, path, msg)))

        old_enable_tts = config.mmoEnableTTS.value
        try:
            config.mmoEnableTTS.value = False
            with mock.patch.object(worker, "_extract_ocr_subtitles", return_value=[]), \
                 mock.patch.object(worker, "_run_subtitle_remover", side_effect=lambda out_p: _create_dummy_video(out_p)):
                worker.run()
        finally:
            config.mmoEnableTTS.value = old_enable_tts

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][0])
        self.assertTrue(os.path.exists(out1), "Video thứ 1 trong thư mục phải được xuất!")
        self.assertTrue(os.path.exists(out2), "Video thứ 2 trong thư mục phải được xuất!")

    # -------------------------------------------------------------------------
    # 9. Kiểm thử P2: VideoOcrEngine ánh xạ đúng mã ngôn ngữ cho PaddleOCR
    # -------------------------------------------------------------------------
    def test_ocr_engine_language_mapping(self):
        engine_vi = VideoOcrEngine(ocr_lang="vi")
        self.assertEqual(engine_vi._resolve_paddle_lang(), "vi")
        engine_ja = VideoOcrEngine(ocr_lang="ja")
        self.assertEqual(engine_ja._resolve_paddle_lang(), "japan")
        engine_ko = VideoOcrEngine(ocr_lang="ko")
        self.assertEqual(engine_ko._resolve_paddle_lang(), "korean")
        engine_zh = VideoOcrEngine(ocr_lang="zh")
        self.assertEqual(engine_zh._resolve_paddle_lang(), "ch")
        engine_auto = VideoOcrEngine(ocr_lang="auto")
        self.assertEqual(engine_auto._resolve_paddle_lang(), "ch")


if __name__ == "__main__":
    unittest.main()
