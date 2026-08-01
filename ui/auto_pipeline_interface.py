# -*- coding: utf-8 -*-
"""
Tab Trang Chủ - Tự Động (Full MMO Auto-Pipeline).
Luồng xử lý tối giản: Input Video -> OCR Detection -> Text + Time -> (Translation || Subtitle Remove) -> Merge -> Output Video.
Giao diện tối giản: Hiển thị thiết lập hiện tại + Nút tùy chỉnh chi tiết để chuyển sang các tab tính năng phụ.
"""

from __future__ import annotations

import os
import sys
import cv2
import time
import json
import threading
import multiprocessing
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QShowEvent
from qfluentwidgets import (
    CardWidget, PushButton, PrimaryPushButton, ComboBox, TitleLabel,
    BodyLabel, CaptionLabel, InfoBar, FluentIcon, ProgressBar, TextEdit,
    SubtitleLabel, CheckBox
)

from backend.config import config, tr
from backend.translator import SubtitleTranslator, SubtitleBlock
from backend.tools.subtitle_exporter import SubtitleExporter
from backend.tools.folder_memory import FolderMemoryDialog
from backend.ocr_engine import VideoOcrEngine, SubtitleSegment
from backend.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from ui.component.video_display_component import VideoDisplayComponent
from ui.translation_interface import CONFIG_API_FILE, PROVIDERS_INFO

# Lịch sử cấu hình Pipeline
CONFIG_PIPELINE_FILE = Path(__file__).parent.parent / "config" / "auto_pipeline_config.json"


def _remover_process_worker(queue, video_path, output_path, options):
    """Top-level worker function for subtitle removal process."""
    sr = None
    try:
        import traceback
        from backend.main import SubtitleRemover
        sr = SubtitleRemover(video_path, True)
        sr.video_out_path = output_path
        for key, val in options.items():
            setattr(sr, key, val)
        sr.add_progress_listener(
            lambda progress, isFinished, frame_no=0: SubtitleRemoverRemoteCall.remote_call_update_progress(
                queue, progress, isFinished, frame_no
            )
        )
        sr.append_output = lambda *args: SubtitleRemoverRemoteCall.remote_call_append_log(queue, list(args))
        sr.manage_process = lambda pid: SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
        sr.update_preview_with_comp = lambda *args: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(
            queue, list(args)
        )
        sr.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
    finally:
        if sr:
            sr.isFinished = True
            sr.vsf_running = False
        SubtitleRemoverRemoteCall.remote_call_finish(queue)


class SubtitleReviewDialog(QDialog):
    """Hộp thoại Duyệt & Chỉnh Sửa Phụ Đề Nhanh trước khi Ghép Video trong Luồng Tự Động."""

    def __init__(self, blocks: list[SubtitleBlock], parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Duyệt & Chỉnh Sửa Phụ Đề Nhanh (Luồng Tự Động)")
        self.resize(760, 500)
        self.blocks = blocks

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        layout.addWidget(SubtitleLabel("Duyệt câu chữ phụ đề đã dịch (Bạn có thể chỉnh sửa nội dung trực tiếp):", self))

        self.table = QTableWidget(len(blocks), 4, self)
        self.table.setHorizontalHeaderLabels(["STT", "Bắt Đầu", "Kết Thúc", "Văn Bản Phụ Đề Dịch"])
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 110)
        self.table.horizontalHeader().setStretchLastSection(True)

        for row, blk in enumerate(blocks):
            self.table.setItem(row, 0, QTableWidgetItem(str(blk.index)))
            self.table.setItem(row, 1, QTableWidgetItem(blk.start_time))
            self.table.setItem(row, 2, QTableWidgetItem(blk.end_time))
            txt_item = QTableWidgetItem(blk.translated_text or blk.text)
            self.table.setItem(row, 3, txt_item)

        layout.addWidget(self.table, 1)

        btn_box = QHBoxLayout()
        btn_continue = PrimaryPushButton("Lưu & Tiếp Tục Render Video", self)
        btn_continue.setIcon(FluentIcon.PLAY)
        btn_continue.clicked.connect(self._save_and_accept)
        btn_box.addStretch(1)
        btn_box.addWidget(btn_continue)

        layout.addLayout(btn_box)

    def _save_and_accept(self):
        for row, blk in enumerate(self.blocks):
            item = self.table.item(row, 3)
            if item:
                txt = item.text()
                blk.translated_text = txt
                blk.text = txt
        self.accept()


class AutoPipelineWorker(QThread):
    """Worker Thread chạy 1-Click luồng MMO Auto Pipeline (Chạy song song Dịch + Xóa sub)."""
    progress_signal = Signal(int, str)
    step_signal = Signal(int)
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, str)
    review_requested_signal = Signal(object, object)  # (extracted_blocks, threading.Event)

    def __init__(
        self,
        video_path: str,
        output_path: str,
        ocr_lang: str,
        target_lang: str,
        engine_type: str,
        api_config: dict,
        inpaint_mode: str,
        sub_areas: list,
        style_type: str,
        pause_for_review: bool = False,
        parent=None
    ):
        super().__init__(parent=parent)
        self.video_path = video_path
        self.output_path = output_path
        self.ocr_lang = ocr_lang
        self.target_lang = target_lang
        self.engine_type = engine_type
        self.api_config = api_config
        self.inpaint_mode = inpaint_mode
        self.sub_areas = sub_areas
        self.style_type = style_type
        self.pause_for_review = pause_for_review
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            # =============================================================
            # BƯỚC 1: TRÍCH XUẤT OCR TỪ VIDEO (Text + Time)
            # =============================================================
            self.step_signal.emit(1)
            self.progress_signal.emit(5, "Bước 1/3: Đang trích xuất OCR phụ đề...")
            self.log_signal.emit(f"🔍 BẮT ĐẦU TRÍCH XUẤT OCR: {os.path.basename(self.video_path)}")

            extracted_blocks = self._extract_ocr_subtitles()
            if self._is_stopped:
                return

            self.log_signal.emit(f"✅ Đã phát hiện {len(extracted_blocks)} câu phụ đề (Text + Time).")

            # =============================================================
            # BƯỚC 2: CHẠY SONG SONG (Translation || Subtitle Remove)
            # =============================================================
            self.step_signal.emit(2)
            self.progress_signal.emit(30, "Bước 2/3: Chạy song song Dịch AI & Xóa phụ đề gốc...")
            self.log_signal.emit("⚡ KHỞI CHẠY 2 NHÁNH SONG SONG: [Dịch phụ đề AI] || [Xóa phụ đề gốc (Inpaint)]")

            cleaned_video_path = str(Path(self.output_path).parent / f"{Path(self.video_path).stem}_cleaned_temp.mp4")

            translate_error = [None]
            remover_error = [None]

            def task_translation():
                try:
                    if extracted_blocks:
                        engine_name_map = {
                            "gguf": f"Local GGUF ({self.api_config.get('model_name', 'Qwen2.5')})",
                            "marian": "Local MarianMT",
                            "llm": f"{self.api_config.get('provider_name', 'AI')} ({self.api_config.get('model_name', '')})",
                            "google": "Free Google Translate"
                        }
                        display_engine_name = engine_name_map.get(self.engine_type, self.engine_type.upper())
                        self.log_signal.emit(f"  └─► [Nhánh Dịch AI] Đang dịch {len(extracted_blocks)} câu bằng {display_engine_name}...")
                        translator = SubtitleTranslator(
                            engine=self.engine_type,
                            api_key=self.api_config.get("api_key"),
                            api_base=self.api_config.get("base_url"),
                            model=self.api_config.get("model_name", "gpt-4o-mini"),
                        )

                        if self.engine_type == "llm" and self.api_config.get("custom_prompt"):
                            if hasattr(translator.translator, 'custom_prompt'):
                                translator.translator.custom_prompt = self.api_config.get("custom_prompt") + "\nRespond ONLY with a JSON array of translated strings matching the input array order."

                        translator.translator.translate_blocks(
                            extracted_blocks,
                            source_lang=self.ocr_lang,
                            target_lang=self.target_lang,
                            progress_callback=lambda p, msg: self.log_signal.emit(f"  └─► [Nhánh Dịch AI] Tiến độ: {p}% ({msg})")
                        )
                        self.log_signal.emit("  └─► [Nhánh Dịch AI] ✅ Hoàn tất dịch phụ đề sang ngôn ngữ mới!")
                except Exception as ex:
                    translate_error[0] = ex
                    self.log_signal.emit(f"  └─► [Nhánh Dịch AI] ❌ Lỗi: {ex}")

            def task_subtitle_removal():
                try:
                    self.log_signal.emit(f"  └─► [Nhánh Xóa Sub] Đang khởi chạy mô hình inpaint ({self.inpaint_mode})...")
                    self._run_subtitle_remover(cleaned_video_path)
                    self.log_signal.emit("  └─► [Nhánh Xóa Sub] ✅ Hoàn tất xóa phụ đề gốc khỏi video!")
                except Exception as ex:
                    remover_error[0] = ex
                    self.log_signal.emit(f"  └─► [Nhánh Xóa Sub] ❌ Lỗi: {ex}")

            t_trans = threading.Thread(target=task_translation, daemon=True)
            t_rem = threading.Thread(target=task_subtitle_removal, daemon=True)

            t_trans.start()
            t_rem.start()

            # Wait for both branches to complete
            while t_trans.is_alive() or t_rem.is_alive():
                if self._is_stopped:
                    return
                time.sleep(0.5)

            if translate_error[0]:
                self.log_signal.emit(f"⚠️ Cảnh báo dịch thuật: {translate_error[0]}")
            if remover_error[0]:
                raise remover_error[0]

            # Pause for user review if requested
            if self.pause_for_review and extracted_blocks:
                self.log_signal.emit("⏸️ Luồng Tự Động tạm dừng để người dùng duyệt & chỉnh sửa phụ đề...")
                evt = threading.Event()
                self.review_requested_signal.emit(extracted_blocks, evt)
                evt.wait()
                if self._is_stopped:
                    return
                self.log_signal.emit("▶️ Đã hoàn thành duyệt phụ đề. Tiếp tục render video!")

            # =============================================================
            # BƯỚC 3: MERGE (Cleaned Video + Translated Subtitle -> Output)
            # =============================================================
            self.step_signal.emit(3)
            self.progress_signal.emit(85, "Bước 3/3: Đang ghép phụ đề mới và render Output Video...")
            self.log_signal.emit("🧩 BẮT ĐẦU MERGE: Ghép phụ đề mới vào Video...")

            ass_path = str(Path(self.output_path).parent / f"{Path(self.video_path).stem}_translated_temp.ass")
            sub_items = []
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            for block in extracted_blocks:
                start_parts = block.start_time.replace(',', '.').split(':')
                end_parts = block.end_time.replace(',', '.').split(':')
                s_sec = float(start_parts[0])*3600 + float(start_parts[1])*60 + float(start_parts[2])
                e_sec = float(end_parts[0])*3600 + float(end_parts[1])*60 + float(end_parts[2])
                sub_items.append({
                    "start_frame": int(s_sec * fps),
                    "end_frame": int(e_sec * fps),
                    "translated_text": block.translated_text if block.translated_text else block.text
                })

            SubtitleExporter.export_ass(ass_path, sub_items, fps, style_type=self.style_type)

            input_video_for_merge = cleaned_video_path if (os.path.exists(cleaned_video_path) and os.path.getsize(cleaned_video_path) > 0) else self.video_path

            if sub_items:
                self._overlay_subtitle_ffmpeg(input_video_for_merge, ass_path, self.output_path)
            elif os.path.exists(input_video_for_merge) and os.path.getsize(input_video_for_merge) > 0:
                import shutil
                shutil.copy(input_video_for_merge, self.output_path)

            for tmp in [cleaned_video_path, ass_path]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

            self.progress_signal.emit(100, "Hoàn thành toàn bộ luồng Tự Động MMO!")
            self.log_signal.emit(f"🎉 ĐÃ XUẤT OUTPUT VIDEO THÀNH CÔNG TẠI: {self.output_path}")
            self.finished_signal.emit(True, self.output_path, "Đã hoàn thành xuất video tự động thành công!")

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.log_signal.emit(f"❌ LỖI LUỒNG TỰ ĐỘNG: {e}\n{err_msg}")
            self.finished_signal.emit(False, "", str(e))

    def _extract_ocr_subtitles(self) -> list[SubtitleBlock]:
        """Trích xuất phụ đề từ video bằng VideoOcrEngine thống nhất."""
        engine = VideoOcrEngine(
            ocr_mode="auto",
            ocr_lang=self.ocr_lang,
            use_typo_map=True,
            use_whisper_fallback=True,
            use_voice_separation=False,
            similarity_threshold=0.7,
        )
        segments = engine.extract_subtitles(
            video_path=self.video_path,
            sub_areas=self.sub_areas,
            progress_callback=lambda pct, msg: self.log_signal.emit(f"🔍 [OCR] {msg}"),
        )
        # Convert SubtitleSegment to SubtitleBlock for compatibility
        from backend.translator import SubtitleBlock
        blocks = []
        for seg in segments:
            blocks.append(SubtitleBlock(
                index=seg.index,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
            ))
        return blocks

    def _run_subtitle_remover(self, output_path: str):
        options = {
            "sub_areas": self.sub_areas if self.sub_areas else [],
            "inpaint_mode": self.inpaint_mode,
        }

        remote_call = SubtitleRemoverRemoteCall()
        remote_call.register_update_progress_callback(
            lambda p, is_fin, f_no: self.progress_signal.emit(
                int(30 + (p / 100.0) * 50), f"Bước 2/3: Đang xóa phụ đề khung hình {f_no}..."
            )
        )

        p = multiprocessing.Process(
            target=_remover_process_worker,
            args=(remote_call.queue, self.video_path, output_path, options)
        )
        p.start()

        while p.is_alive():
            if self._is_stopped:
                p.terminate()
                break
            time.sleep(0.5)

        p.join()
        if p.exitcode and p.exitcode != 0:
            raise RuntimeError(f"Subtitle remover process exited with code {p.exitcode}")

    def _overlay_subtitle_ffmpeg(self, video_in: str, ass_in: str, video_out: str):
        import subprocess
        import shutil
        from backend.tools.ffmpeg_cli import FFmpegCLI
        ffmpeg_bin = FFmpegCLI.instance().ffmpeg_path

        clean_ass = os.path.abspath(ass_in).replace('\\', '/').replace(':', '\\:')
        cmd = [
            ffmpeg_bin, "-y",
            "-i", os.path.abspath(video_in),
            "-vf", f"subtitles='{clean_ass}'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            os.path.abspath(video_out)
        ]

        if os.path.exists(video_out) and os.path.getsize(video_out) == 0:
            try:
                os.remove(video_out)
            except Exception:
                pass

        res = subprocess.run(cmd, capture_output=True, text=True)

        if not os.path.exists(video_out) or os.path.getsize(video_out) == 0:
            self.log_signal.emit("⚠️ FFmpeg burn subtitle gặp sự cố. Tiến hành sao chép video gốc làm tệp đầu ra...")
            if os.path.exists(video_out):
                try:
                    os.remove(video_out)
                except Exception:
                    pass
            shutil.copy(os.path.abspath(video_in), os.path.abspath(video_out))


class AutoPipelineInterface(QWidget):
    """
    Tab Trang Chủ - Tự Động (Giao diện tối giản chuẩn MMO).
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("AutoPipelineInterface")
        self.parent = parent
        self.current_video_path = None
        self.pipeline_worker: AutoPipelineWorker | None = None

        self._init_ui()
        self._load_pipeline_config()
        self._connect_signals()
        self._update_settings_summary()

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        self._update_settings_summary()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 18, 24, 18)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # 1. Header Banner Tối Giản
        # -------------------------------------------------------------
        header_card = CardWidget(self)
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title = TitleLabel("Tự Động", self)
        desc = CaptionLabel(
            "Quy trình tự động hóa 1-Click: Input Video ➔ OCR (Text+Time) ➔ [Dịch AI & Xóa Sub (Song song)] ➔ Merge ➔ Output Video", self
        )
        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        main_layout.addWidget(header_card)

        # -------------------------------------------------------------
        # 2. Minimalist Feature Settings Overview Card
        # -------------------------------------------------------------
        config_card = CardWidget(self)
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(16, 14, 16, 14)
        config_layout.setSpacing(10)

        # Row 1: Video File Selection & Jump to Downloader
        row1 = QHBoxLayout()
        self.btn_select_video = PushButton("Chọn Video Nguồn", self)
        self.btn_select_video.setIcon(FluentIcon.VIDEO)
        self.btn_select_folder = PushButton("Chọn Thư Mục Video", self)
        self.btn_select_folder.setIcon(FluentIcon.FOLDER)
        
        self.lbl_video_status = BodyLabel("Đã nạp: Chưa chọn video nào", self)
        self.lbl_video_status.setStyleSheet("color: #666666; font-style: italic;")

        self.btn_nav_ytdlp = PushButton("Tải Video", self)
        self.btn_nav_ytdlp.setIcon(FluentIcon.DOWNLOAD)

        row1.addWidget(self.btn_select_video)
        row1.addWidget(self.btn_select_folder)
        row1.addWidget(self.lbl_video_status, 1)
        row1.addWidget(self.btn_nav_ytdlp)
        config_layout.addLayout(row1)

        # Divider line
        div1 = QWidget(self)
        div1.setFixedHeight(1)
        div1.setStyleSheet("background-color: rgba(0,0,0,0.08);")
        config_layout.addWidget(div1)

        # Row 2: OCR & Translation Status + Jump to Translate Tab
        row2 = QHBoxLayout()
        lbl_trans_title = BodyLabel("Dịch thuật & OCR:", self)
        lbl_trans_title.setStyleSheet("font-weight: bold;")
        self.lbl_summary_translate = CaptionLabel("Thiết lập: OCR (Tự động) ➔ Dịch (Miễn phí)", self)
        self.lbl_summary_translate.setStyleSheet("background: rgba(0,120,212,0.1); padding: 4px 10px; border-radius: 6px; color: #0078d4;")

        self.btn_nav_translate = PushButton("Chỉnh sửa chi tiết", self)
        self.btn_nav_translate.setIcon(FluentIcon.SETTING)

        row2.addWidget(lbl_trans_title)
        row2.addWidget(self.lbl_summary_translate, 1)
        row2.addWidget(self.btn_nav_translate)
        config_layout.addLayout(row2)

        # Divider line
        div2 = QWidget(self)
        div2.setFixedHeight(1)
        div2.setStyleSheet("background-color: rgba(0,0,0,0.08);")
        config_layout.addWidget(div2)

        # Row 3: Subtitle Remover & Hardware Status + Jump to Remover Tab
        row3 = QHBoxLayout()
        lbl_remove_title = BodyLabel("Xóa Phụ Đề Video:", self)
        lbl_remove_title.setStyleSheet("font-weight: bold;")
        self.lbl_summary_remover = CaptionLabel("Thiết lập: STTN Auto | GPU Acceleration: BẬT", self)
        self.lbl_summary_remover.setStyleSheet("background: rgba(16,124,65,0.1); padding: 4px 10px; border-radius: 6px; color: #107c41;")

        self.btn_nav_remover = PushButton("Chỉnh sửa chi tiết", self)
        self.btn_nav_remover.setIcon(FluentIcon.SETTING)

        row3.addWidget(lbl_remove_title)
        row3.addWidget(self.lbl_summary_remover, 1)
        row3.addWidget(self.btn_nav_remover)
        config_layout.addLayout(row3)

        main_layout.addWidget(config_card)

        # Hidden Combo Controls (maintaining internal backend options compatibility)
        self.ocr_lang_combo = ComboBox(self)
        self.ocr_lang_combo.addItems(["Tự động phát hiện", "Tiếng Trung", "Tiếng Anh", "Tiếng Việt"])
        self.target_lang_combo = ComboBox(self)
        self.target_lang_combo.addItems(["Tiếng Việt", "Tiếng Anh", "Tiếng Trung", "Tiếng Nhật", "Tiếng Hàn"])
        self.engine_combo = ComboBox(self)
        self.engine_combo.addItems(["Google Translate (Miễn phí)", "OpenAI / Gemini / Claude API Key"])
        self.inpaint_combo = ComboBox(self)
        self.inpaint_combo.addItems(["STTN Auto (Khuyên dùng)", "Lama", "Propainter", "OpenCV Rapid"])
        self.style_combo = ComboBox(self)
        self.style_combo.addItems(["TikTok Vàng Nổi Bật", "Sub Trắng Viền Đen Chữ Nổi"])

        for hidden_w in [self.ocr_lang_combo, self.target_lang_combo, self.engine_combo, self.inpaint_combo, self.style_combo]:
            hidden_w.setVisible(False)

        # -------------------------------------------------------------
        # 3. Workspace: Video Player & Parallel Step Monitor Console
        # -------------------------------------------------------------
        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(12)

        # Left: Video Player Component
        self.video_display = VideoDisplayComponent(self)
        workspace_layout.addWidget(self.video_display, 3)

        # Right: Log & Step Monitor Card
        monitor_card = CardWidget(self)
        monitor_layout = QVBoxLayout(monitor_card)
        monitor_layout.setContentsMargins(14, 12, 14, 12)
        monitor_layout.setSpacing(10)

        monitor_layout.addWidget(SubtitleLabel("Sơ Đồ & Tiến Trình Pipeline Auto", self))

        # Visual Parallel Step Indicator Badges
        self.step_label_1 = CaptionLabel("▶ Bước 1: OCR Detection (Trích xuất Text + Time)", self)
        self.step_label_2 = CaptionLabel("▶ Bước 2: [Song song] Dịch AI & Xóa phụ đề gốc (Inpaint)", self)
        self.step_label_3 = CaptionLabel("▶ Bước 3: Merge phụ đề mới & Xuất Video", self)

        for lbl in [self.step_label_1, self.step_label_2, self.step_label_3]:
            monitor_layout.addWidget(lbl)

        # Realtime Log Console
        self.log_console = TextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("Nhật ký xử lý luồng tự động sẽ hiển thị tại đây...")
        monitor_layout.addWidget(self.log_console, 1)

        workspace_layout.addWidget(monitor_card, 2)
        main_layout.addLayout(workspace_layout, 1)

        # -------------------------------------------------------------
        # 4. Bottom Action Card
        # -------------------------------------------------------------
        action_card = CardWidget(self)
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(14, 10, 14, 10)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setValue(0)
        action_layout.addWidget(self.progress_bar, 1)

        self.pause_review_checkbox = CheckBox("Dừng duyệt & sửa phụ đề trước khi ghép video", self)
        action_layout.addWidget(self.pause_review_checkbox)

        self.status_label = BodyLabel("Sẵn sàng chạy luồng tự động.", self)
        action_layout.addWidget(self.status_label)

        self.btn_run_pipeline = PrimaryPushButton("Bắt đầu Luồng Tự Động (1-Click)", self)
        self.btn_run_pipeline.setIcon(FluentIcon.PLAY)
        action_layout.addWidget(self.btn_run_pipeline)

        main_layout.addWidget(action_card)

    def _connect_signals(self):
        self.btn_select_video.clicked.connect(self._select_video_file)
        self.btn_select_folder.clicked.connect(self._select_video_folder)
        self.btn_run_pipeline.clicked.connect(self._start_pipeline)

        # Connect Navigation Buttons to jump to sub-feature tabs
        self.btn_nav_translate.clicked.connect(self._jump_to_translate_tab)
        self.btn_nav_remover.clicked.connect(self._jump_to_remover_tab)
        self.btn_nav_ytdlp.clicked.connect(self._jump_to_ytdlp_tab)

    def _jump_to_translate_tab(self):
        main_win = self.window()
        if hasattr(main_win, 'toolsInterface'):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(2)

    def _jump_to_remover_tab(self):
        main_win = self.window()
        if hasattr(main_win, 'toolsInterface'):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(3)

    def _jump_to_ytdlp_tab(self):
        main_win = self.window()
        if hasattr(main_win, 'toolsInterface'):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(0)

    def showEvent(self, event):
        super().showEvent(event)
        self._update_settings_summary()

    def _update_settings_summary(self):
        """Đọc và hiển thị các thiết lập hiện tại lên giao diện tối giản."""
        # 1. Cấu hình Tab Dịch Phụ Đề
        if CONFIG_API_FILE.exists():
            try:
                data = json.loads(CONFIG_API_FILE.read_text(encoding="utf-8"))
                engine_idx = data.get("engine_index", 0)
                if engine_idx == 0:
                    self.lbl_summary_translate.setText("OCR: Auto ➔ Dịch AI: Google Translate (Miễn phí)")
                else:
                    provider_idx = data.get("provider_index", 0)
                    provider_keys = list(PROVIDERS_INFO.keys())
                    provider_name = data.get("provider_name") or (provider_keys[provider_idx] if 0 <= provider_idx < len(provider_keys) else "GGUF Model")
                    model_name = data.get("model_name", "Qwen2.5 / Local Model")
                    engine_type = data.get("engine_type", "")
                    if not engine_type:
                        if provider_name == "GGUF Model":
                            engine_type = "gguf"
                        elif provider_name == "MarianMT":
                            engine_type = "marian"
                        elif data.get("api_key"):
                            engine_type = "llm"
                        else:
                            engine_type = "google"

                    if engine_type == "gguf":
                        self.lbl_summary_translate.setText(f"OCR: Auto ➔ Dịch Local GGUF: {model_name} |  Offline Engine")
                    elif engine_type == "marian":
                        self.lbl_summary_translate.setText("OCR: Auto ➔ Dịch Local MarianMT |  Offline Engine")
                    else:
                        has_key = bool(data.get("api_key", "").strip())
                        key_str = "🔑 Key hợp lệ" if has_key else "⚠️ Chưa nạp Key"
                        self.lbl_summary_translate.setText(f"OCR: Auto ➔ Dịch AI: {provider_name} ({model_name}) | {key_str}")
            except Exception:
                self.lbl_summary_translate.setText("OCR: Auto ➔ Dịch AI: Google Translate")
        else:
            self.lbl_summary_translate.setText("OCR: Auto ➔ Dịch AI: Google Translate)")

        # 2. Cấu hình Tab Xóa Sub & Phần cứng
        from backend.config import config as main_cfg
        hw_enabled = getattr(main_cfg.hardwareAcceleration, 'value', True)
        hw_str = "GPU CUDA/DirectML: BẬT" if hw_enabled else "CPU Mode"

        raw_mode = getattr(main_cfg.inpaintMode, 'value', 'sttn_auto')
        cfg_file = Path("config") / "auto_pipeline_config.json"
        if cfg_file.exists():
            try:
                auto_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                if "inpaint_mode" in auto_cfg and auto_cfg["inpaint_mode"]:
                    raw_mode = auto_cfg["inpaint_mode"]
            except Exception:
                pass

        if hasattr(raw_mode, 'value'):
            raw_mode = raw_mode.value
        if hasattr(raw_mode, 'value'):
            raw_mode = raw_mode.value

        mode_str = str(raw_mode).lower().replace('-', '_')
        if 'lama' in mode_str:
            mode_code = 'lama'
        elif 'propainter' in mode_str:
            mode_code = 'propainter'
        elif 'opencv' in mode_str:
            mode_code = 'opencv'
        else:
            mode_code = 'sttn_auto'

        inpaint_display_map = {
            "sttn_auto": "STTN Auto (Khuyên dùng)",
            "lama": "Lama Inpaint",
            "propainter": "Propainter",
            "opencv": "OpenCV Rapid"
        }
        inpaint_text = inpaint_display_map.get(mode_code, mode_code)
        self.lbl_summary_remover.setText(f"Mô hình Xóa: {inpaint_text} | Tăng tốc: {hw_str}")

        combo_map = {'sttn_auto': 0, 'lama': 1, 'propainter': 2, 'opencv': 3}
        self.inpaint_combo.setCurrentIndex(combo_map.get(mode_code, 0))

    def _select_video_file(self):
        filepath, _ = FolderMemoryDialog.getOpenFileName(
            self,
            "Chọn tệp Video nguồn",
            filter_str="Tệp Video (*.mp4 *.mkv *.avi *.mov *.flv *.webm);;Tất cả tệp (*)",
            category="video"
        )
        if filepath:
            self.current_video_path = filepath
            self.lbl_video_status.setText(f"Đã nạp Video: {os.path.basename(filepath)}")
            self.lbl_video_status.setStyleSheet("color: #107c41; font-weight: bold;")
            self.video_display.set_video_path(filepath)
            
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.video_display.set_video_parameters(w, h, fps, fc)
                ret, frame = cap.read()
                if ret:
                    self.video_display.update_video_display(frame)
                cap.release()

            InfoBar.success("Đã nạp Video", f"Tệp: {os.path.basename(filepath)}", parent=self, duration=2500)

    def _select_video_folder(self):
        folder = FolderMemoryDialog.getExistingDirectory(self, "Chọn thư mục chứa Video", category="video")
        if folder:
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm'))]
            if files:
                first_video = os.path.join(folder, files[0])
                self._select_video_file_by_path(first_video)
                self.lbl_video_status.setText(f"Thư mục ({len(files)} videos): {os.path.basename(folder)}")
                self.lbl_video_status.setStyleSheet("color: #107c41; font-weight: bold;")
                InfoBar.success("Đã nạp thư mục hàng loạt", f"Phát hiện {len(files)} video!", parent=self, duration=3000)

    def _select_video_file_by_path(self, filepath: str):
        if os.path.exists(filepath):
            self.current_video_path = filepath
            self.video_display.set_video_path(filepath)
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.video_display.set_video_parameters(w, h, fps, fc)
                ret, frame = cap.read()
                if ret:
                    self.video_display.update_video_display(frame)
                cap.release()

    def _start_pipeline(self):
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            InfoBar.warning("Cảnh báo", "Vui lòng chọn tệp Video nguồn trước khi chạy luồng tự động!", parent=self, duration=3000)
            return

        if self.pipeline_worker and self.pipeline_worker.isRunning():
            return

        api_config = {}
        if CONFIG_API_FILE.exists():
            try:
                api_config = json.loads(CONFIG_API_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        ocr_map = {"Tự động phát hiện": "auto", "Tiếng Trung": "zh", "Tiếng Anh": "en", "Tiếng Việt": "vi"}
        tgt_map = {"Tiếng Việt": "vi", "Tiếng Anh": "en", "Tiếng Trung": "zh", "Tiếng Nhật": "ja", "Tiếng Hàn": "ko"}
        inpaint_map = {"STTN Auto (Khuyên dùng)": "sttn_auto", "Lama": "lama", "Propainter": "propainter", "OpenCV Rapid": "opencv"}
        style_map = {"TikTok Vàng Nổi Bật": "tiktok_yellow", "Sub Trắng Viền Đen Chữ Nổi": "clean_white"}

        ocr_lang = ocr_map.get(self.ocr_lang_combo.currentText(), "auto")
        target_lang = tgt_map.get(self.target_lang_combo.currentText(), "vi")
        # Xác định engine_type chuẩn xác từ config đã đồng bộ
        engine_idx = api_config.get("engine_index", 0)
        if engine_idx == 0:
            engine_type = "google"
        else:
            engine_type = api_config.get("engine_type", "")
            if not engine_type:
                provider_idx = api_config.get("provider_index", 0)
                provider_keys = list(PROVIDERS_INFO.keys())
                p_name = api_config.get("provider_name") or (provider_keys[provider_idx] if 0 <= provider_idx < len(provider_keys) else "")
                if p_name == "GGUF Model":
                    engine_type = "gguf"
                elif p_name == "MarianMT":
                    engine_type = "marian"
                elif api_config.get("api_key"):
                    engine_type = "llm"
        # Sync inpaint mode from config
        from backend.config import config as main_cfg
        raw_mode = getattr(main_cfg.inpaintMode, 'value', 'sttn_auto')
        cfg_file = Path("config") / "auto_pipeline_config.json"
        if cfg_file.exists():
            try:
                auto_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                if "inpaint_mode" in auto_cfg and auto_cfg["inpaint_mode"]:
                    raw_mode = auto_cfg["inpaint_mode"]
            except Exception:
                pass

        if hasattr(raw_mode, 'value'):
            raw_mode = raw_mode.value
        if hasattr(raw_mode, 'value'):
            raw_mode = raw_mode.value

        mode_str = str(raw_mode).lower().replace('-', '_')
        if 'lama' in mode_str:
            inpaint_mode = 'lama'
        elif 'propainter' in mode_str:
            inpaint_mode = 'propainter'
        elif 'opencv' in mode_str:
            inpaint_mode = 'opencv'
        else:
            inpaint_mode = 'sttn_auto'
        style_type = style_map.get(self.style_combo.currentText(), "tiktok_yellow")

        vd_p = Path(self.current_video_path)
        output_path = str(vd_p.parent / f"{vd_p.stem}_MMO_Auto.mp4")

        sub_areas = self.video_display.get_selection_coordinates()

        self.log_console.clear()
        self.progress_bar.setValue(0)
        self.btn_run_pipeline.setEnabled(False)

        self.pipeline_worker = AutoPipelineWorker(
            video_path=self.current_video_path,
            output_path=output_path,
            ocr_lang=ocr_lang,
            target_lang=target_lang,
            engine_type=engine_type,
            api_config=api_config,
            inpaint_mode=inpaint_mode,
            sub_areas=sub_areas,
            style_type=style_type,
            pause_for_review=self.pause_review_checkbox.isChecked(),
            parent=self
        )

        self.pipeline_worker.progress_signal.connect(self._on_progress)
        self.pipeline_worker.step_signal.connect(self._on_step_changed)
        self.pipeline_worker.log_signal.connect(self._on_log)
        self.pipeline_worker.finished_signal.connect(self._on_finished)
        self.pipeline_worker.review_requested_signal.connect(self._on_review_requested)
        self.pipeline_worker.finished.connect(self.pipeline_worker.deleteLater)

        self.pipeline_worker.start()

    def _on_review_requested(self, blocks, evt):
        dialog = SubtitleReviewDialog(blocks, parent=self)
        dialog.exec()
        evt.set()

    @Slot(int, str)
    def _on_progress(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    @Slot(int)
    def _on_step_changed(self, step: int):
        labels = [self.step_label_1, self.step_label_2, self.step_label_3]
        for i, lbl in enumerate(labels, 1):
            if i == step:
                lbl.setStyleSheet("font-weight: bold; color: #0078d4;")
            elif i < step:
                lbl.setStyleSheet("color: #107c41;")
            else:
                lbl.setStyleSheet("color: #666666;")

    @Slot(str)
    def _on_log(self, msg: str):
        self.log_console.append(msg)

    @Slot(bool, str, str)
    def _on_finished(self, success: bool, output_file: str, message: str):
        self.btn_run_pipeline.setEnabled(True)
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("Hoàn thành luồng tự động!")
            InfoBar.success("Thành công", f"Đã xuất Video hoàn chỉnh tại: {os.path.basename(output_file)}", parent=self, duration=4000)
        else:
            self.status_label.setText("Lỗi khi xử lý luồng.")
            InfoBar.error("Lỗi Luồng Tự Động", message, parent=self, duration=4500)

    def _save_pipeline_config(self):
        try:
            cfg = {
                "ocr_lang_index": self.ocr_lang_combo.currentIndex(),
                "target_lang_index": self.target_lang_combo.currentIndex(),
                "inpaint_index": self.inpaint_combo.currentIndex(),
                "style_index": self.style_combo.currentIndex(),
            }
            config_path = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "config" / "auto_pipeline_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_pipeline_config(self):
        try:
            config_path = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "config" / "auto_pipeline_config.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                if 0 <= cfg.get("ocr_lang_index", 0) < self.ocr_lang_combo.count():
                    self.ocr_lang_combo.setCurrentIndex(cfg["ocr_lang_index"])
                if 0 <= cfg.get("target_lang_index", 0) < self.target_lang_combo.count():
                    self.target_lang_combo.setCurrentIndex(cfg["target_lang_index"])
                if 0 <= cfg.get("inpaint_index", 0) < self.inpaint_combo.count():
                    self.inpaint_combo.setCurrentIndex(cfg["inpaint_index"])
                if 0 <= cfg.get("style_index", 0) < self.style_combo.count():
                    self.style_combo.setCurrentIndex(cfg["style_index"])
        except Exception:
            pass

    def retranslateUi(self):
        pass
