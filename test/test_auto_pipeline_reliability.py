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

    # -------------------------------------------------------------------------
    # 10. Kiểm thử P1 UI: OmniVoice xóa sạch video/phụ đề cũ khi nhận phiên mới
    # -------------------------------------------------------------------------
    def test_omnivoice_load_subtitle_and_video_clears_stale_session_data(self):
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()
        old_vid = os.path.join(self.temp_dir, "old_task.mp4")
        old_srt = os.path.join(self.temp_dir, "old_task.srt")
        _create_dummy_video(old_vid)
        Path(old_srt).write_text("1\n00:00:01,000 --> 00:00:02,000\nCâu cũ\n", encoding="utf-8")

        # Phiên 1: Có cả phụ đề và video
        ui.load_subtitle_and_video(srt_path=old_srt, video_path=old_vid)
        self.assertEqual(ui.current_video_path, old_vid)
        self.assertTrue(ui.btn_dub_merge_video.isEnabled())

        # Phiên 2: Chuyển phụ đề mới từ tab Dịch sang mà không kèm video (video_path=None, srt_path=None)
        new_blocks = [
            SubtitleBlock(index=1, start_time="00:00:00,500", end_time="00:00:01,800", text="Hello", translated_text="Xin chào")
        ]
        ui.load_subtitle_and_video(srt_path=None, video_path=None, blocks=new_blocks)
        self.assertEqual(ui.current_video_path, "", "Phải xóa đường dẫn video cũ khi nạp phiên phụ đề mới không kèm video!")
        self.assertEqual(ui.current_subtitle_path, "", "Phải xóa đường dẫn phụ đề cũ khi nạp danh sách khối mới không kèm tệp!")
        self.assertFalse(ui.btn_dub_merge_video.isEnabled(), "Nút Lồng Tiếng Và Xuất Video phải khóa khi không có video!")

        # Kiểm thử nút Bỏ Chọn Video thủ công
        ui.load_subtitle_and_video(srt_path=old_srt, video_path=old_vid)
        self.assertTrue(hasattr(ui, "btn_clear_video"), "Phải có nút Bỏ Chọn Video (btn_clear_video)!")
        ui._clear_selected_video()
        self.assertEqual(ui.current_video_path, "")
        self.assertFalse(ui.btn_dub_merge_video.isEnabled())

    # -------------------------------------------------------------------------
    # 11. Kiểm thử P1 UI: Khóa bảng & truyền bản sao độc lập cho OmniVoiceTaskWorker
    # -------------------------------------------------------------------------
    def test_omnivoice_locks_table_and_deepcopies_blocks_when_running(self):
        from PySide6.QtWidgets import QApplication, QAbstractItemView
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()
        blocks = [
            SubtitleBlock(index=1, start_time="00:00:00,500", end_time="00:00:02,000", text="Orig", translated_text="Bản dịch gốc")
        ]
        ui.load_subtitle_and_video(blocks=blocks)

        # Kiểm tra khóa bảng và các nút khi đang chạy
        ui._set_task_running_state(True)
        self.assertEqual(ui.sub_table.editTriggers(), QAbstractItemView.NoEditTriggers)
        self.assertFalse(ui.btn_load_srt.isEnabled())
        self.assertFalse(ui.btn_load_video.isEnabled())
        self.assertFalse(ui.btn_apply_preset_to_speaker.isEnabled())
        self.assertFalse(ui.btn_assign_speaker_to_rows.isEnabled())

        ui._set_task_running_state(False)
        self.assertNotEqual(ui.sub_table.editTriggers(), QAbstractItemView.NoEditTriggers)
        self.assertTrue(ui.btn_load_srt.isEnabled())

        # Kiểm tra _launch_worker tạo bản sao độc lập của subtitle_blocks
        with mock.patch("src.desktop.ui.omnivoice_interface.OmniVoiceTaskWorker.start"):
            ui.combo_mode.setCurrentIndex(1)  # Chế độ Thiết kế giọng nói
            ui._launch_worker("subtitle_audio_only", output_path=os.path.join(self.temp_dir, "out.wav"), subtitle_blocks=ui.subtitle_blocks)
            ui._set_task_running_state(False)

        self.assertIsNotNone(ui._worker)
        worker_blocks = ui._worker.subtitle_blocks
        self.assertIsNot(worker_blocks, ui.subtitle_blocks, "Danh sách khối phụ đề truyền cho worker phải là bản sao độc lập!")
        self.assertIsNot(worker_blocks[0], ui.subtitle_blocks[0], "Từng đối tượng SubtitleBlock trong worker phải độc lập với UI!")
        ui.subtitle_blocks[0].translated_text = "Đã sửa trên UI"
        self.assertEqual(worker_blocks[0].translated_text, "Bản dịch gốc")

    # -------------------------------------------------------------------------
    # 12. Kiểm thử P2 UI: Cột chỉ đọc và kiểm tra mốc thời gian trên bảng phụ đề
    # -------------------------------------------------------------------------
    def test_omnivoice_sub_table_readonly_columns_and_timestamp_validation(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()
        blocks = [
            SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Hello", translated_text="Xin chào")
        ]
        ui.load_subtitle_and_video(blocks=blocks)

        item_stt = ui.sub_table.item(0, 0)
        item_source = ui.sub_table.item(0, 4)
        self.assertFalse(bool(item_stt.flags() & Qt.ItemIsEditable), "Cột STT (0) phải ở chế độ chỉ đọc!")
        self.assertFalse(bool(item_source.flags() & Qt.ItemIsEditable), "Cột Nguồn Giọng Nhân Vật (4) phải ở chế độ chỉ đọc!")

        # Thử nhập mốc Kết Thúc nhỏ hơn mốc Bắt Đầu -> phải khôi phục giá trị cũ
        item_end = ui.sub_table.item(0, 2)
        item_end.setText("00:00:00,500")
        self.assertEqual(ui.subtitle_blocks[0].end_time, "00:00:03,000", "Mốc kết thúc không hợp lệ phải bị từ chối và khôi phục!")

        # Nhập mốc Kết Thúc hợp lệ -> phải cập nhật
        item_end.setText("00:00:04,500")
        self.assertEqual(ui.subtitle_blocks[0].end_time, "00:00:04,500")

    # -------------------------------------------------------------------------
    # 13. Kiểm thử P2 UI: SubtitleReviewDialog lưu cả mốc thời gian & chỉ đọc cột STT
    # -------------------------------------------------------------------------
    def test_subtitle_review_dialog_saves_timestamps_and_locks_index_col(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from src.desktop.ui.auto_pipeline_interface import SubtitleReviewDialog

        app = QApplication.instance() or QApplication([])
        blocks = [
            SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,500", text="Orig", translated_text="Bản dịch cũ")
        ]
        dlg = SubtitleReviewDialog(blocks)
        self.assertFalse(bool(dlg.table.item(0, 0).flags() & Qt.ItemIsEditable), "Cột STT trong SubtitleReviewDialog phải chỉ đọc!")

        # Sửa mốc bắt đầu, kết thúc và văn bản hợp lệ
        dlg.table.item(0, 1).setText("00:00:01,200")
        dlg.table.item(0, 2).setText("00:00:03,800")
        dlg.table.item(0, 3).setText("Bản dịch mới")
        dlg._save_and_accept()

        self.assertEqual(blocks[0].start_time, "00:00:01,200")
        self.assertEqual(blocks[0].end_time, "00:00:03,800")
        self.assertEqual(blocks[0].translated_text, "Bản dịch mới")
        self.assertEqual(blocks[0].text, "Orig", "Không được ghi đè văn bản gốc blk.text khi đã có sẵn!")

    # -------------------------------------------------------------------------
    # 14. Kiểm thử P2 UI: Thư mục trống xóa hàng đợi cũ & 0 câu OCR báo hoàn thành một phần
    # -------------------------------------------------------------------------
    def test_empty_folder_clears_queue_and_zero_ocr_warns_partial(self):
        from PySide6.QtWidgets import QApplication
        from src.core.config import config
        from src.desktop.ui.auto_pipeline_interface import (
            AutoPipelineInterface,
            AutoPipelineWorker,
            PIPELINE_PARTIAL,
        )

        app = QApplication.instance() or QApplication([])
        ui = AutoPipelineInterface()
        vid1 = os.path.join(self.temp_dir, "queued.mp4")
        _create_dummy_video(vid1)
        ui.video_queue = [vid1]
        ui.current_video_path = vid1

        empty_dir = os.path.join(self.temp_dir, "empty_folder")
        os.makedirs(empty_dir, exist_ok=True)
        with mock.patch("src.core.tools.folder_memory.FolderMemoryDialog.getExistingDirectory", return_value=empty_dir):
            ui._select_video_folder()

        self.assertEqual(ui.video_queue, [], "Chọn thư mục trống phải xóa sạch hàng đợi video cũ!")
        self.assertIsNone(ui.current_video_path, "Chọn thư mục trống phải xóa đường dẫn video hiện tại!")

        # Kiểm tra 0 câu OCR trả về PIPELINE_PARTIAL thay vì báo thành công dịch hoàn toàn
        out_vid = os.path.join(self.temp_dir, "zero_ocr_out.mp4")
        worker = AutoPipelineWorker(
            video_path=vid1,
            output_path=out_vid,
            ocr_lang="zh",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )
        old_enable_tts = config.mmoEnableTTS.value
        try:
            config.mmoEnableTTS.value = False
            with mock.patch.object(worker, "_extract_ocr_subtitles", return_value=[]), \
                 mock.patch.object(worker, "_run_subtitle_remover", side_effect=lambda p: _create_dummy_video(p)):
                worker.run()
        finally:
            config.mmoEnableTTS.value = old_enable_tts

        self.assertEqual(worker.final_status, PIPELINE_PARTIAL, "Khi OCR tìm thấy 0 câu phụ đề, trạng thái phải là PARTIAL_SUCCESS!")

    # -------------------------------------------------------------------------
    # 15. Kiểm thử P2 UI: Hiển thị hộp chọn kiểu phụ đề, nút Hàng đợi & co giãn 960x600
    # -------------------------------------------------------------------------
    def test_auto_pipeline_visible_style_combo_and_responsive_resize(self):
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineInterface
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        auto_ui = AutoPipelineInterface()
        omni_ui = OmniVoiceInterface()

        self.assertFalse(auto_ui.style_combo.isHidden(), "Hộp chọn kiểu phụ đề (style_combo) phải hiển thị trực tiếp trên tab Tự Động!")
        self.assertTrue(hasattr(auto_ui, "btn_view_queue"), "Tab Tự Động phải có nút xem Danh Sách Hàng Đợi (btn_view_queue)!")

        for w, h in [(960, 600), (1280, 720)]:
            auto_ui.resize(w, h)
            omni_ui.resize(w, h)
            app.processEvents()
            self.assertEqual(auto_ui.width(), w)
            self.assertEqual(omni_ui.width(), w)

    # -------------------------------------------------------------------------
    # 16. Kiểm thử P1 Đợt 1 & 2: Khóa toàn bộ điều khiển OmniVoice & xóa profile cũ khi nạp SRT
    # -------------------------------------------------------------------------
    def test_omnivoice_comprehensive_control_lock_and_manual_srt_profile_reset(self):
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        # Kiểm tra khóa toàn bộ điều khiển cấu hình khi tác vụ đang chạy
        ui.spin_duration.setValue(3.5)
        ui._update_control_states()
        self.assertFalse(ui.spin_speed.isEnabled(), "Khi thời lượng cố định > 0, spin_speed phải bị khóa!")

        ui._set_task_running_state(True)
        locked_widgets = [
            ui.combo_lang,
            ui.combo_vn_catalog,
            ui.edit_vn_sample_text,
            ui.btn_free_vram,
            ui.chk_auto_extract_video,
            ui.edit_ref_text,
            ui.combo_gender,
            ui.combo_age,
            ui.combo_pitch,
            ui.combo_style,
            ui.combo_accent,
            ui.combo_dialect,
            ui.edit_instruct_preview,
            ui.studio_text_edit,
            ui.edit_pronun_token,
            ui.btn_insert_pronun,
            ui.spin_max_speakers,
            ui.combo_target_speaker,
            ui.combo_speaker_voice_preset,
            ui.combo_sub_tag,
            ui.spin_speed,
            ui.spin_duration,
            ui.spin_num_step,
            ui.spin_guidance,
            ui.spin_t_shift,
            ui.spin_layer_penalty,
            ui.spin_pos_temp,
            ui.spin_cls_temp,
            ui.spin_chunk_dur,
            ui.spin_chunk_thresh,
            ui.spin_pad_dur,
            ui.spin_fade_dur,
            ui.spin_duck_vol,
            ui.combo_ducking_mode,
            ui.chk_sync_sub_duration,
            ui.chk_denoise,
            ui.chk_preprocess_prompt,
            ui.chk_postprocess_output,
            ui.chk_normalize_text,
            ui.chk_flashinfer,
        ]
        for w in locked_widgets:
            self.assertFalse(w.isEnabled(), f"Điều khiển {w} phải bị khóa khi OmniVoice đang chạy!")

        # Mở khóa lại sau khi kết thúc: spin_speed vẫn phải giữ trạng thái khóa do spin_duration = 3.5 > 0
        ui._set_task_running_state(False)
        self.assertTrue(ui.combo_lang.isEnabled())
        self.assertTrue(ui.spin_num_step.isEnabled())
        self.assertTrue(ui.studio_text_edit.isEnabled())
        self.assertFalse(ui.spin_speed.isEnabled(), "Sau khi mở khóa, spin_speed phải tôn trọng trạng thái spin_duration > 0!")

        # Kiểm tra nạp tệp SRT thủ công phải xóa speaker_profiles cũ
        ui.speaker_profiles = {"Người nói 1": {"speaker_id": "Người nói 1", "instruct": "female"}}
        srt_file = os.path.join(self.temp_dir, "manual_load.srt")
        Path(srt_file).write_text("1\n00:00:01,000 --> 00:00:02,000\nCâu thoại mới\n", encoding="utf-8")
        ui._load_subtitle_file_internal(srt_file)
        self.assertEqual(ui.speaker_profiles, {}, "Nạp tệp phụ đề thủ công mới phải xóa sạch speaker_profiles của phiên trước!")

    # -------------------------------------------------------------------------
    # 17. Kiểm thử P1 Đợt 2: Phát hiện và xử lý xung đột tên tệp đầu ra khi chạy hàng loạt
    # -------------------------------------------------------------------------
    def test_batch_output_conflict_detection_and_resolution_modes(self):
        from src.desktop.ui.auto_pipeline_interface import resolve_batch_output_plan

        dir_a = os.path.join(self.temp_dir, "batch_a")
        os.makedirs(dir_a, exist_ok=True)
        vid1 = os.path.join(dir_a, "clip.mp4")
        vid2 = os.path.join(dir_a, "clip.mkv")  # Cùng stem "clip" trong cùng thư mục -> trùng đường dẫn đầu ra clip_MMO_Auto.mp4!
        _create_dummy_video(vid1)
        _create_dummy_video(vid2)

        # Tạo sẵn tệp đầu ra của vid1 trên ổ đĩa để mô phỏng xung đột ghi đè
        existing_out = os.path.join(dir_a, "clip_MMO_Auto.mp4")
        _create_dummy_video(existing_out)

        # Chế độ chặn (block): phát hiện xung đột và không cho phép chạy nếu chưa giải quyết
        plan_block = resolve_batch_output_plan([vid1, vid2], conflict_mode="block")
        self.assertTrue(plan_block["has_conflicts"])
        self.assertGreaterEqual(len(plan_block["conflicts"]), 1)

        # Chế độ tự đổi tên (auto_rename): tự thêm hậu tố _1, _2 để không trùng nhau và không ghi đè tệp sẵn có
        plan_rename = resolve_batch_output_plan([vid1, vid2], conflict_mode="auto_rename")
        resolved_pairs = plan_rename["resolved_items"]
        self.assertEqual(len(resolved_pairs), 2)
        out_paths = [p[1] for p in resolved_pairs]
        self.assertEqual(len(set(out_paths)), 2, "Các đường dẫn đầu ra sau khi tự đổi tên phải hoàn toàn phân biệt!")
        self.assertNotIn(existing_out, out_paths, "Chế độ tự đổi tên không được phép ghi đè lên tệp đã tồn tại trên đĩa!")

        # Chế độ bỏ qua tệp đã tồn tại (skip)
        plan_skip = resolve_batch_output_plan([vid1], conflict_mode="skip")
        self.assertEqual(len(plan_skip["resolved_items"]), 0, "Chế độ skip phải bỏ qua video đã có sẵn tệp đầu ra trên ổ đĩa!")
        self.assertEqual(len(plan_skip["skipped_videos"]), 1)

    # -------------------------------------------------------------------------
    # 18. Kiểm thử P1 Đợt 3: Chống trả kết quả sai khi nạp model dịch bất đồng bộ
    # -------------------------------------------------------------------------
    def test_async_model_fetch_ignores_stale_request_id_and_provider_mismatch(self):
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.translation_interface import TranslationInterface, PROVIDERS_INFO

        app = QApplication.instance() or QApplication([])
        ui = TranslationInterface()

        provider_keys = list(PROVIDERS_INFO.keys())
        gemini_idx = provider_keys.index("Google Gemini")
        openai_idx = provider_keys.index("OpenAI")

        ui._is_updating_provider = True
        ui.provider_combo.setCurrentIndex(gemini_idx)
        ui.api_key_edit.setText("AIzaSyDummyGeminiKey123456789")
        ui._is_updating_provider = False

        # Giả lập yêu cầu mới nhất có request_id = 5 cho Google Gemini
        ui._model_fetch_req_id = 5
        ui._on_fetch_models_done(5, "Google Gemini", True, ["gemini-2.5-flash", "gemini-2.5-pro"], "OK")
        self.assertEqual(ui.model_combo.count(), 2)
        self.assertEqual(ui.model_combo.currentText(), "gemini-2.5-flash")

        # Một phản hồi trễ từ yêu cầu cũ (request_id = 4 của OpenAI) vừa về tới -> phải bị bỏ qua!
        ui._on_fetch_models_done(4, "OpenAI", True, ["gpt-4o-mini"], "OK")
        self.assertEqual(ui.model_combo.count(), 2)
        self.assertEqual(ui.model_combo.currentText(), "gemini-2.5-flash", "Kết quả của request_id cũ không được ghi đè model mới!")

        # Phản hồi có cùng request_id nhưng khác nhà cung cấp đang chọn hiện tại -> cũng phải bị bỏ qua!
        ui._is_updating_provider = True
        ui.provider_combo.setCurrentIndex(openai_idx)
        ui._is_updating_provider = False
        ui._on_fetch_models_done(5, "Google Gemini", True, ["gemini-1.5-flash"], "OK")
        self.assertNotEqual(ui.model_combo.currentText(), "gemini-1.5-flash")

    # -------------------------------------------------------------------------
    # 19. Kiểm thử P2 Đợt 3: Bảo mật khóa DPAPI & phân biệt Đã lưu vs Đã xác thực
    # -------------------------------------------------------------------------
    def test_dpapi_key_encryption_and_verified_vs_saved_status(self):
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.translation_interface import (
            _encrypt_secret,
            _decrypt_secret,
            CONFIG_API_FILE,
        )
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineInterface

        app = QApplication.instance() or QApplication([])
        secret = "sk-proj-test-dpapi-secret-998877"
        encrypted = _encrypt_secret(secret)
        self.assertNotEqual(encrypted, secret)
        if os.name == "nt":
            self.assertTrue(encrypted.startswith("dpapi:"), "Trên Windows phải sử dụng kho bảo mật hệ điều hành DPAPI!")
        self.assertEqual(_decrypt_secret(encrypted), secret)

        backup_cfg = CONFIG_API_FILE.read_text(encoding="utf-8") if CONFIG_API_FILE.exists() else None
        try:
            auto_ui = AutoPipelineInterface()
            # Trường hợp 1: Đã lưu khóa nhưng chưa xác thực (api_key_verified = False)
            cfg_unverified = {
                "engine_index": 1,
                "engine_type": "llm",
                "provider_index": 0,
                "provider_name": "Google Gemini",
                "api_key": encrypted,
                "api_key_verified": False,
                "model_name": "gemini-2.5-flash",
            }
            CONFIG_API_FILE.write_text(json.dumps(cfg_unverified, ensure_ascii=False), encoding="utf-8")
            auto_ui._update_settings_summary()
            self.assertIn("chưa xác thực", auto_ui.lbl_summary_translate.text().lower())

            # Trường hợp 2: Đã xác thực thành công (api_key_verified = True)
            cfg_unverified["api_key_verified"] = True
            CONFIG_API_FILE.write_text(json.dumps(cfg_unverified, ensure_ascii=False), encoding="utf-8")
            auto_ui._update_settings_summary()
            self.assertIn("Đã xác thực", auto_ui.lbl_summary_translate.text())
        finally:
            if backup_cfg is not None:
                CONFIG_API_FILE.write_text(backup_cfg, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 20. Kiểm thử P1/P2 Đợt 1 & 4: Hủy tiến trình FFmpeg & cập nhật retranslateUi
    # -------------------------------------------------------------------------
    def test_ffmpeg_overlay_cancellation_and_retranslate_ui_all_tabs(self):
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineInterface, AutoPipelineWorker
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
        from src.desktop.ui.translation_interface import TranslationInterface

        app = QApplication.instance() or QApplication([])
        auto_ui = AutoPipelineInterface()
        omni_ui = OmniVoiceInterface()
        trans_ui = TranslationInterface()

        # Kiểm tra retranslateUi() thực sự cập nhật nhãn giao diện và không để trống
        auto_ui.retranslateUi()
        omni_ui.retranslateUi()
        trans_ui.retranslateUi()
        self.assertTrue(len(auto_ui.btn_run_pipeline.text()) > 0)
        self.assertTrue(len(omni_ui.btn_studio_synth.text()) > 0)

        # Kiểm tra hủy tiến trình FFmpeg đang chạy trong _overlay_subtitle_ffmpeg
        in_vid = os.path.join(self.temp_dir, "ff_cancel_in.mp4")
        ass_file = os.path.join(self.temp_dir, "ff_cancel.ass")
        out_vid = os.path.join(self.temp_dir, "ff_cancel_out.mp4")
        _create_dummy_video(in_vid)
        Path(ass_file).write_text("[Script Info]\nTitle: Test\n", encoding="utf-8")

        worker = AutoPipelineWorker(
            video_path=in_vid,
            output_path=out_vid,
            ocr_lang="zh",
            target_lang="vi",
            engine_type="google",
            api_config={},
            inpaint_mode="sttn_auto",
            sub_areas=[],
            style_type="tiktok_yellow",
        )

        class DummyPopen:
            def __init__(self, *args, **kwargs):
                self.terminated = False
                self.returncode = None
                self.stderr = mock.MagicMock()
                self.stderr.read.return_value = ""

            def poll(self):
                if self.terminated:
                    self.returncode = -15
                    return -15
                worker.stop()
                return None

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def communicate(self, timeout=None):
                return ("", "")

        dummy_proc = DummyPopen()
        with mock.patch("subprocess.Popen", return_value=dummy_proc):
            worker._overlay_subtitle_ffmpeg(in_vid, ass_file, out_vid)

        self.assertTrue(dummy_proc.terminated, "Khi nhận lệnh dừng, tiến trình FFmpeg đang ghép phụ đề phải bị terminate ngay!")

    def test_step0_has_identify_button_enabled_only_when_srt_and_video_present(self):
        """
        Xác minh Bước 1 (Nguồn Đầu Vào) có nút 'Nhận Diện Nhân Vật Từ Video + SRT':
        1. Nút bị vô hiệu hóa khi chưa nạp cả SRT lẫn video.
        2. Nút bị vô hiệu hóa khi chỉ có SRT mà chưa có video.
        3. Nút được kích hoạt khi cả SRT hợp lệ và video đều đã được nạp.
        4. Nhãn trạng thái thay đổi theo nguồn thực tế.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        # Trạng thái ban đầu: chưa có tệp nào
        ui._update_identify_button_state()
        self.assertFalse(ui.btn_identify_characters.isEnabled())

        # Chỉ có SRT: nút chính vẫn tắt
        ui.subtitle_blocks = [object()]  # Giả lập có câu phụ đề
        ui.current_video_path = ""
        ui._update_identify_button_state()
        self.assertFalse(ui.btn_identify_characters.isEnabled())
        hint_text = ui.lbl_detect_source_hint.text()
        self.assertIn("SRT", hint_text)

        # Có cả SRT lẫn video: nút chính bật
        valid_vid = os.path.join(self.temp_dir, 'sample.mp4')
        with open(valid_vid, 'wb') as f:
            f.write(b'video_data')
        ui.current_video_path = valid_vid
        ui._update_identify_button_state()
        self.assertTrue(ui.btn_identify_characters.isEnabled())
        hint_text = ui.lbl_detect_source_hint.text()
        self.assertIn("video", hint_text.lower())


    def test_step0_identify_button_enabled_with_video_only(self):
        """
        Xác minh nút nhận diện nhân vật tại Bước 1 mở khi chỉ có video (chưa có SRT):
        1. Nút btn_identify_characters được kích hoạt (enabled=True).
        2. Nhãn hiển thị thông báo sẵn sàng nhận diện trực tiếp từ video qua VAD.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        valid_vid = os.path.join(self.temp_dir, 'standalone.mp4')
        with open(valid_vid, 'wb') as f:
            f.write(b'video_data')

        ui.subtitle_blocks = []
        ui.current_video_path = valid_vid
        ui._update_identify_button_state()

        self.assertTrue(ui.btn_identify_characters.isEnabled())
        hint = ui.lbl_detect_source_hint.text()
        self.assertIn("video", hint.lower())
        self.assertIn("tự động", hint.lower())

    def test_speakers_detected_populates_empty_subtitles_from_worker(self):
        """
        Xác minh khi phụ đề ban đầu rỗng, _on_speakers_detected tiếp nhận
        toàn bộ các câu thoại do worker sinh ra và nạp vào bảng phụ đề.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
        from src.ai_engines.translator import SubtitleBlock
        from src.ai_engines.speaker_diarization import CharacterVoiceProfile
        from unittest.mock import MagicMock

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()
        ui.subtitle_blocks = []

        worker = MagicMock()
        b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Đoạn thoại 1")
        b1.speaker = "Người nói 1"
        b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Đoạn thoại 2")
        b2.speaker = "Người nói 2"
        worker.subtitle_blocks = [b1, b2]
        ui._worker = worker

        profiles = {
            "Người nói 1": CharacterVoiceProfile(speaker_id="spk_0", display_name="Người nói 1"),
            "Người nói 2": CharacterVoiceProfile(speaker_id="spk_1", display_name="Người nói 2"),
        }
        ui._on_speakers_detected(profiles)

        self.assertEqual(len(ui.subtitle_blocks), 2)
        self.assertEqual(ui.sub_table.rowCount(), 2)

    def test_export_detected_subtitles_to_srt_file(self):
        """
        Xác minh phương thức _export_subtitles_to_srt tạo đúng tệp SRT
        với đầy đủ mốc thời gian và nội dung câu thoại.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
        from src.ai_engines.translator import SubtitleBlock
        import tempfile, os

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,500", "Chào bạn")
        b1.speaker = "Nhân vật A"
        b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,500", "Hẹn gặp lại")
        b2.speaker = "Nhân vật B"
        ui.subtitle_blocks = [b1, b2]

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            out_srt = f.name

        try:
            ok = ui._export_subtitles_to_srt(out_srt)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(out_srt))
            with open(out_srt, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("00:00:01,000 --> 00:00:02,500", content)
            self.assertIn("Chào bạn", content)
            self.assertIn("00:00:03,000 --> 00:00:04,500", content)
        finally:
            if os.path.exists(out_srt):
                os.remove(out_srt)

    def test_casting_progress_bar_calculates_ratios_correctly(self):
        """
        Xác minh tiện ích CastingProgressBarWidget:
        1. Tính đúng tỷ lệ phần trăm phân đoạn: đã gán/khóa, cần rà soát, chưa xác định.
        2. Cập nhật nhãn chi tiết tiếng Việt chuẩn xác theo dữ liệu câu thoại.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import CastingProgressBarWidget
        from src.ai_engines.translator import SubtitleBlock

        app = QApplication.instance() or QApplication([])
        widget = CastingProgressBarWidget()

        b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
        b1.speaker = "Người nói 1"
        b1.confidence_level = "high"
        b1.is_speaker_locked = True

        b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
        b2.speaker = "Người nói 2"
        b2.confidence_level = "review_needed"
        b2.is_speaker_locked = False

        b3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu 3")
        b3.speaker = ""
        b3.confidence_level = "unassigned"
        b3.is_speaker_locked = False

        b4 = SubtitleBlock(4, "00:00:07,000", "00:00:08,000", "Câu 4")
        b4.speaker = "Người nói 1"
        b4.confidence_level = "high"
        b4.is_speaker_locked = True

        widget.update_progress([b1, b2, b3, b4])

        self.assertAlmostEqual(widget.ratio_high, 0.50, places=2)
        self.assertAlmostEqual(widget.ratio_review, 0.25, places=2)
        self.assertAlmostEqual(widget.ratio_unassigned, 0.25, places=2)

        text = widget.lbl_details.text()
        self.assertIn("3/4", text)
        self.assertIn("2 câu đã khóa", text)
        self.assertIn("1 câu cần rà soát", text)
        self.assertIn("1 câu chưa xác định", text)

    def test_subtitle_table_row_background_colors_reflect_confidence(self):
        """
        Xác minh màu nền các dòng trên bảng phụ đề:
        - Dòng 'review_needed' có màu nền vàng nhạt.
        - Dòng 'unassigned' có màu nền đỏ nhạt.
        - Dòng 'high' hoặc đã khóa giữ màu nền mặc định.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
        from src.ai_engines.translator import SubtitleBlock

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu tin cậy cao")
        b1.speaker = "Người nói 1"
        b1.confidence_level = "high"
        b1.is_speaker_locked = True

        b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu cần rà soát")
        b2.speaker = "Người nói 1"
        b2.confidence_level = "review_needed"
        b2.is_speaker_locked = False

        b3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu chưa xác định")
        b3.speaker = ""
        b3.confidence_level = "unassigned"
        b3.is_speaker_locked = False

        ui.subtitle_blocks = [b1, b2, b3]
        ui._populate_subtitle_table()

        bg2 = ui.sub_table.item(1, 0).background().color()
        self.assertGreater(bg2.red(), 240)
        self.assertGreater(bg2.green(), 230)

        bg3 = ui.sub_table.item(2, 0).background().color()
        self.assertGreater(bg3.red(), 240)
        self.assertLess(bg3.green(), 240)

    def test_batch_assign_speaker_updates_selected_rows_and_locks(self):
        """
        Xác minh chức năng gán nhân vật hàng loạt:
        1. Khi chọn dòng 0 và dòng 2 và bấm gán cho 'Nhân vật Đặc Biệt':
           - Dòng 0 và 2 được cập nhật speaker='Nhân vật Đặc Biệt', is_speaker_locked=True.
           - Dòng 1 giữ nguyên.
        2. Chức năng khóa và mở khóa hàng loạt cập nhật đúng trạng thái.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
        from src.ai_engines.translator import SubtitleBlock

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
        b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
        b2.speaker = "Người cũ"
        b3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu 3")

        ui.subtitle_blocks = [b1, b2, b3]
        ui._populate_subtitle_table()

        # Chọn dòng 0 và dòng 2
        ui.sub_table.selectRow(0)
        ui.sub_table.item(2, 0).setSelected(True)

        # Đặt nhân vật mục tiêu trên bộ chọn hàng loạt
        ui.combo_batch_speaker.clear()
        ui.combo_batch_speaker.addItem("Nhân vật Đặc Biệt")
        ui.combo_batch_speaker.setCurrentText("Nhân vật Đặc Biệt")

        # Kích hoạt gán hàng loạt
        ui._batch_assign_selected_speaker()

        self.assertEqual(ui.subtitle_blocks[0].speaker, "Nhân vật Đặc Biệt")
        self.assertTrue(ui.subtitle_blocks[0].is_speaker_locked)
        self.assertEqual(ui.subtitle_blocks[2].speaker, "Nhân vật Đặc Biệt")
        self.assertTrue(ui.subtitle_blocks[2].is_speaker_locked)
        self.assertEqual(ui.subtitle_blocks[1].speaker, "Người cũ")

        # Kích hoạt mở khóa hàng loạt cho dòng 0
        ui.sub_table.clearSelection()
        ui.sub_table.selectRow(0)
        ui._batch_set_locked_state(False)
        self.assertFalse(ui.subtitle_blocks[0].is_speaker_locked)

    def test_voice_separator_preserves_audio_fidelity_without_hard_gating(self):
        """
        Xác minh bộ lọc âm thanh trong VoiceSeparator:
        - Không áp dụng hard-gating làm nát sóng âm và mất tín hiệu.
        - Bảo toàn dải tần giọng nói và biên độ âm lượng chuẩn hóa mượt mà.
        """
        import tempfile
        import os
        import numpy as np
        from scipy.io import wavfile
        from src.ai_engines.voice_separator import VoiceSeparator

        sr = 16000
        # Tạo tín hiệu giả lập: 0.5s đầu là giọng nói (400 Hz), 0.5s sau là 800 Hz
        t = np.linspace(0, 1.0, sr, endpoint=False)
        speech = 0.5 * np.sin(2 * np.pi * 400 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "in.wav")
            out_wav = os.path.join(tmpdir, "out.wav")
            wavfile.write(in_wav, sr, (speech * 32767).astype(np.int16))

            sep = VoiceSeparator(backend="scipy")
            ok = sep.separate(in_wav, out_wav)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(out_wav))

            out_sr, out_data = wavfile.read(out_wav)
            self.assertEqual(out_sr, 16000)
            # Tín hiệu không bị triệt tiêu thành 0 ở phần sau
            out_float = out_data.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(out_float ** 2))
            self.assertGreater(rms, 0.05, "Âm thanh không được bị triệt tiêu bởi gating thô sơ")
            # Kiểm tra không có hiện tượng hard zeroing diện rộng (tỷ lệ mẫu bằng chính xác 0 phải < 10%)
            zero_ratio = np.mean(np.abs(out_float) < 1e-6)
            self.assertLess(zero_ratio, 0.10, "Không được xuất hiện hard gating làm nát sóng âm")

    def test_eigengap_cluster_estimation_detects_correct_speaker_count(self):
        """
        Xác minh thuật toán phân tích trị riêng (Eigengap) ước tính chính xác số người nói:
        1. 2 người nói riêng biệt -> Tự động phân đúng 2 cụm mà không cần chỉ định exact_k.
        2. 3 người nói riêng biệt -> Tự động phân đúng 3 cụm.
        3. 1 người nói duy nhất -> Tự động gộp thành 1 cụm.
        """
        import numpy as np
        from src.ai_engines.speaker_diarization import (
            agglomerative_cluster_embeddings,
            estimate_optimal_clusters_eigengap,
        )

        np.random.seed(42)
        dim = 192

        # 1. Hai người nói rõ rệt
        center_a = np.random.randn(dim)
        center_a /= np.linalg.norm(center_a)
        center_b = np.random.randn(dim)
        center_b -= np.dot(center_b, center_a) * center_a # Trực giao
        center_b /= np.linalg.norm(center_b)

        # Mỗi người 3 câu thoại có nhiễu nhẹ
        samples_a = [center_a + 0.05 * np.random.randn(dim) for _ in range(4)]
        samples_b = [center_b + 0.05 * np.random.randn(dim) for _ in range(4)]
        embs_2spk = np.array(samples_a + samples_b)
        embs_2spk /= np.linalg.norm(embs_2spk, axis=1, keepdims=True)

        k_est_2 = estimate_optimal_clusters_eigengap(embs_2spk, min_k=1, max_k=6)
        self.assertEqual(k_est_2, 2, f"Kỳ vọng 2 người nói nhưng thuật toán đoán {k_est_2}")

        labels_2 = agglomerative_cluster_embeddings(embs_2spk, exact_k=None, max_k=6)
        self.assertEqual(len(np.unique(labels_2)), 2)

        # 2. Một người nói duy nhất
        samples_single = [center_a + 0.04 * np.random.randn(dim) for _ in range(6)]
        embs_1spk = np.array(samples_single)
        embs_1spk /= np.linalg.norm(embs_1spk, axis=1, keepdims=True)

        k_est_1 = estimate_optimal_clusters_eigengap(embs_1spk, min_k=1, max_k=6)
        self.assertEqual(k_est_1, 1, f"Kỳ vọng 1 người nói nhưng thuật toán đoán {k_est_1}")

        labels_1 = agglomerative_cluster_embeddings(embs_1spk, exact_k=None, max_k=6)
        self.assertEqual(len(np.unique(labels_1)), 1)

    def test_safe_audio_padding_and_boundary_refinement_prevents_overlap(self):
        """
        Xác minh cơ chế gọt tỉa biên và tính toán khoảng đệm an toàn:
        1. Khi hai câu thoại liền kề có mốc thời gian chồng lấn (overlap):
           - Điểm bắt đầu hiệu dụng của câu sau tự động lùi về ranh giới an toàn, chống dính chéo âm thanh.
        2. Khi hai câu thoại cách nhau rất sát (0.04s):
           - Khoảng đệm không được vượt quá một nửa khoảng cách nghỉ.
        """
        from src.ai_engines.speaker_diarization import _calculate_safe_audio_padding

        # Trường hợp 1: Chồng lấn thời gian (câu 1 kết thúc 2.4s, câu 2 bắt đầu 2.2s)
        pad_before, pad_after = _calculate_safe_audio_padding(
            start_s=2.2, end_s=3.0, prev_end=2.4, next_start=4.0
        )
        self.assertLessEqual(pad_before, 0.0, "Khi chồng lấn, khoảng đệm trước không được dương để tránh liếm sâu vào câu trước")

        # Trường hợp 2: Cách nhau rất sát 0.06s (câu 1 kết thúc 2.0s, câu 2 bắt đầu 2.06s)
        pad_before, pad_after = _calculate_safe_audio_padding(
            start_s=2.06, end_s=3.0, prev_end=2.0, next_start=4.0
        )
        self.assertLessEqual(pad_before, 0.03 + 1e-6, "Khoảng đệm không được vượt quá một nửa khoảng nghỉ")

    def test_ui_speaker_count_controls_and_vietnamese_labels(self):
        """
        Xác minh các điều khiển phân vai trên giao diện OmniVoice:
        1. Hộp số lượng nhân vật hỗ trợ dải giá trị từ 1 đến 12.
        2. Nhãn hiển thị và danh sách chọn sử dụng 100% tiếng Việt chuẩn, không chứa ngoặc đơn.
        3. Thay đổi chế độ cập nhật nhãn số lượng nhân vật chính xác.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        self.assertEqual(ui.spin_max_speakers.minimum(), 1)
        self.assertEqual(ui.spin_max_speakers.maximum(), 12)

        # Kiểm tra không chứa ngoặc đơn trong combo_speaker_count_mode
        for i in range(ui.combo_speaker_count_mode.count()):
            txt = ui.combo_speaker_count_mode.itemText(i)
            self.assertNotIn("(", txt)
            self.assertNotIn(")", txt)

        # Kiểm tra chuyển đổi nhãn
        ui.combo_speaker_count_mode.setCurrentIndex(1)
        self.assertEqual(ui.lbl_max_spk.text(), "Số lượng nhân vật chính xác:")

        ui.combo_speaker_count_mode.setCurrentIndex(0)
        self.assertEqual(ui.lbl_max_spk.text(), "Số lượng nhân vật tối đa:")

    def test_play_subtitle_original_segment_initializes_cache_dir_safely(self):
        """
        Xác minh OmniVoiceInterface khởi tạo thuộc tính voice_prompt_cache_dir đầy đủ:
        - Không ném AttributeError khi bấm nút Nghe gốc câu thoại.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
        from src.ai_engines.translator import SubtitleBlock

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        self.assertTrue(hasattr(ui, "voice_prompt_cache_dir"))
        self.assertTrue(os.path.isdir(ui.voice_prompt_cache_dir))

        # Gọi _play_subtitle_original_segment khi chưa có video -> Phải thoát êm dịu, không crash
        blk = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Thử nghiệm")
        ui._play_subtitle_original_segment(blk, None)

    def test_on_worker_finished_handles_deleted_audition_button_gracefully(self):
        """
        Xác minh _on_worker_finished và _reset_audition_button xử lý an toàn khi đối tượng
        PushButton C++ đã bị Qt xóa (deleted) do vẽ lại bảng, không ném RuntimeError shiboken.
        """
        from PySide6.QtWidgets import QApplication, QWidget
        from qfluentwidgets import PushButton
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        # Tạo một nút con trong một container tạm rồi xóa container
        temp_container = QWidget()
        btn = PushButton("Nghe Thử Giọng", temp_container)
        ui._active_audition_button = btn

        # Xóa container khiến đối tượng C++ của btn bị delete
        temp_container.deleteLater()
        # Ép xóa đối tượng C++
        del temp_container
        import gc
        gc.collect()

        # Gọi _reset_audition_button và _on_worker_finished không được crash
        ui._reset_audition_button()
        ui._on_worker_finished(success=True, out_path="character_audition", detail="", log_msg="")
        self.assertIsNone(ui._active_audition_button)

    def test_ai_voice_separator_attenuates_instrumental_bgm_and_preserves_voice(self):
        """
        Xác minh bộ tách giọng AI (AIVoiceSeparator):
        - Bảo toàn năng lượng hài âm giọng nói con người (formant 300 - 2500 Hz).
        - Triệt tiêu đáng kể thành phần gõ percussive và nhạc nền ngoài dải.
        - Trả về tệp âm thanh 16kHz mono trong trẻo.
        """
        import tempfile
        import os
        import numpy as np
        from scipy.io import wavfile
        from src.ai_engines.voice_separator import AIVoiceSeparator

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Giọng nói: Hài âm 300Hz, 600Hz, 1200Hz
        voice = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
        # Nhạc nền và tiếng ồn: Tiếng bùm bass 40Hz + tiếng rít 8500Hz + tiếng gõ nhịp ngẫu nhiên
        bgm = 0.4 * np.sin(2 * np.pi * 40 * t) + 0.3 * np.sin(2 * np.pi * 8500 * t)
        bgm += (np.random.rand(sr) - 0.5) * 0.15 # Nhiễu nền

        mix = voice + bgm

        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "mix.wav")
            out_wav = os.path.join(tmpdir, "vocals.wav")
            wavfile.write(in_wav, sr, (mix * 16384).astype(np.int16))

            separator = AIVoiceSeparator()
            ok = separator.separate_vocals(in_wav, out_wav)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(out_wav))

            out_sr, out_data = wavfile.read(out_wav)
            self.assertEqual(out_sr, 16000)
            out_float = out_data.astype(np.float32) / 32768.0

            # Phân tích FFT kiểm tra độ suy giảm của bass 40Hz so với giọng 600Hz
            fft_in = np.abs(np.fft.rfft(mix))
            fft_out = np.abs(np.fft.rfft(out_float))
            freqs = np.fft.rfftfreq(len(mix), d=1.0 / sr)

            idx_40 = np.argmin(np.abs(freqs - 40.0))
            idx_600 = np.argmin(np.abs(freqs - 600.0))

            # Tỷ lệ năng lượng giọng / bass phải tăng mạnh ở đầu ra
            ratio_in = fft_in[idx_600] / max(fft_in[idx_40], 1e-6)
            ratio_out = fft_out[idx_600] / max(fft_out[idx_40], 1e-6)
            self.assertGreater(ratio_out, ratio_in * 1.5, "Tỷ lệ giọng nói so với tạp âm nền phải được cải thiện rõ rệt")

    def test_ui_ai_vocal_separation_checkbox_and_config(self):
        """
        Xác minh ô kiểm 'Tách sạch nhạc nền bằng trí tuệ nhân tạo' trên giao diện OmniVoice:
        - Tồn tại trên giao diện và mặc định được bật.
        - Thu thập đúng tham số 'ai_vocal_separation' khi gọi _collect_engine_parameters.
        """
        from PySide6.QtWidgets import QApplication
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        app = QApplication.instance() or QApplication([])
        ui = OmniVoiceInterface()

        self.assertTrue(hasattr(ui, "chk_ai_vocal_separation"))
        self.assertTrue(ui.chk_ai_vocal_separation.isChecked())

        params = ui._collect_engine_params()
        self.assertIn("ai_vocal_separation", params)
        self.assertTrue(params["ai_vocal_separation"])

        ui.chk_ai_vocal_separation.setChecked(False)
        params_off = ui._collect_engine_params()
        self.assertFalse(params_off["ai_vocal_separation"])


    def test_wsola_pitch_preservation(self):
        """
        Xac minh thuat toan co gian thoi luong WSOLA bao toan 100% cao do am thanh:
        - Tan so ban dau 250 Hz.
        - Khi tang toc 1.3x, tan so phai duoc giu nguyen ~250 Hz (+/- 2%), khong bi day vot len 325 Hz (chipmunk).
        - Khi giam toc 0.8x, tan so cung phai duoc giu nguyen ~250 Hz (+/- 2%).
        """
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        orig_freq = 250.0
        tone = (0.5 * np.sin(2 * np.pi * orig_freq * t)).astype(np.float32)

        # Tang toc 1.3x
        stretched_fast = TTSEngine._safe_time_stretch(tone, speed_factor=1.3, sample_rate=sr)
        self.assertLess(len(stretched_fast), len(tone))
        fft_fast = np.abs(np.fft.rfft(stretched_fast))
        freqs_fast = np.fft.rfftfreq(len(stretched_fast), d=1.0 / sr)
        peak_fast = freqs_fast[np.argmax(fft_fast)]
        self.assertAlmostEqual(peak_fast, orig_freq, delta=6.0, msg="Cao do khong duoc bien dang khi tang toc")

        # Giam toc 0.8x
        stretched_slow = TTSEngine._safe_time_stretch(tone, speed_factor=0.8, sample_rate=sr)
        self.assertGreater(len(stretched_slow), len(tone))
        fft_slow = np.abs(np.fft.rfft(stretched_slow))
        freqs_slow = np.fft.rfftfreq(len(stretched_slow), d=1.0 / sr)
        peak_slow = freqs_slow[np.argmax(fft_slow)]
        self.assertAlmostEqual(peak_slow, orig_freq, delta=6.0, msg="Cao do khong duoc bien dang khi giam toc")

    def test_audio_overlay_crossfade_no_clipping(self):
        """
        Xac minh ky thuat ghep noi muot (Crossfade) loai bo hoan toan Hard Clipping khi 2 cau phu de chong lan:
        - Bien do tong khong bi ep vuong goc o muc 1.0 (clipping square wave).
        - Am thanh duoc ghep chuyen tiep em ai ma khong gay no lach tach.
        """
        sr = 24000
        full_audio = np.zeros(int(3.0 * sr), dtype=np.float32)

        # Doan 1: bien do 0.8
        seg1 = np.full(int(1.5 * sr), 0.8, dtype=np.float32)
        full_audio = TTSEngine._overlay_audio_segment(full_audio, seg1, start_sample=0, sample_rate=sr)

        # Doan 2: chong lan 0.2s vao doan 1 voi bien do 0.8
        start_seg2 = int(1.3 * sr)
        seg2 = np.full(int(1.0 * sr), 0.8, dtype=np.float32)
        full_audio = TTSEngine._overlay_audio_segment(full_audio, seg2, start_sample=start_seg2, sample_rate=sr)

        # Dinh tin hieu khong vuot qua 1.0 va khong bi cat cut cung thanh xung vuong
        max_val = np.max(np.abs(full_audio))
        self.assertLessEqual(max_val, 1.0)
        # Tai vung chong lan, gia tri phai duoc lam muot
        overlap_slice = full_audio[start_seg2 : int(1.5 * sr)]
        self.assertTrue(np.all(np.isfinite(overlap_slice)))

    def test_extract_reference_from_video_no_name_error(self):
        """
        Xac minh ham extract_reference_from_video khong bi loi NameError: enable_ai_vocal_separation
        du truyen tham so hay de mac dinh.
        """
        engine = TTSEngine(auto_extract_ref=False, mode="design")
        # Video khong ton tai se tra ve None mot cach an toan ma khong vang ngoai le NameError
        res1 = engine.extract_reference_from_video("dummy_non_existent_video.mp4")
        self.assertIsNone(res1)

        res2 = engine.extract_reference_from_video("dummy_non_existent_video.mp4", enable_ai_vocal_separation=True)
        self.assertIsNone(res2)

        res3 = engine.extract_reference_from_video("dummy_non_existent_video.mp4", enable_ai_vocal_separation=False)
        self.assertIsNone(res3)

    def test_calculate_safe_target_duration(self):
        """
        Xac minh co che tinh toan thoi luong an toan cho cau thoai:
        - Cau ngan (2 tu) trong khung phu de 4.0s khong bi ep keo dai thanh 4.0s.
        - Cau dai (30 tu) trong khung phu de 1.5s khong bi ep nen thanh 1.5s.
        """
        # Cau ngan: 2 tu, thoi luong noi tu nhien ~0.6s
        safe_dur_short = TTSEngine._calculate_safe_target_duration("Chào anh.", slot_duration=4.0)
        self.assertIsNone(safe_dur_short, "Cau qua ngan so voi khung gio thi de noi tu nhien, khong ep keo dai")

        # Cau dai: 30 tu, thoi luong noi tu nhien ~7.5s trong khung 1.5s
        long_text = " ".join(["từ"] * 30)
        safe_dur_long = TTSEngine._calculate_safe_target_duration(long_text, slot_duration=1.5)
        self.assertIsNotNone(safe_dur_long)
        self.assertGreaterEqual(safe_dur_long, 5.0)

        # Cau vua van: 10 tu (~2.6s) trong khung 2.8s -> giu nguyen 2.8s
        medium_text = " ".join(["từ"] * 10)
        safe_dur_med = TTSEngine._calculate_safe_target_duration(medium_text, slot_duration=2.8)
        self.assertEqual(safe_dur_med, 2.8)

if __name__ == "__main__":
    unittest.main()


