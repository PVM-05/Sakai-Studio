# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog
from PySide6.QtCore import Slot, Signal, Qt, QThread
from PySide6 import QtWidgets
from qfluentwidgets import (PushButton, PrimaryPushButton, CardWidget, FluentIcon, InfoBar,
                           ComboBoxSettingCard, PushSettingCard)

from ui.component.video_display_component import VideoDisplayComponent
from ui.component.task_list_component import TaskListComponent, TaskStatus, TaskOptions
from backend.config import config, tr
from backend.tools.subtitle_exporter import SubtitleExporter
from backend.ocr_engine import VideoOcrEngine


class FastSrtExtractThread(QThread):
    progress_signal = Signal(float, str)
    finished_signal = Signal(str, bool, str)  # (output_path, success, message)

    def __init__(self, video_path: str, sub_areas: list, save_dir: str = "", ocr_mode: str = "auto", ocr_lang: str = "vi", parent=None):
        super().__init__(parent=parent)
        self.video_path = video_path
        self.sub_areas = sub_areas
        self.save_dir = save_dir
        self.ocr_mode = ocr_mode
        self.ocr_lang = ocr_lang
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.finished_signal.emit("", False, "Tệp video không tồn tại.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit("", False, "Không thể mở tệp video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        engine = VideoOcrEngine(
            ocr_mode=self.ocr_mode,
            ocr_lang=self.ocr_lang,
            use_typo_map=True,
            use_whisper_fallback=True
        )

        def progress_callback(pct, msg):
            if self._is_stopped:
                engine.stop()
            self.progress_signal.emit(pct, msg)

        try:
            segments = engine.extract_subtitles(self.video_path, self.sub_areas, progress_callback)
        except Exception as e:
            self.finished_signal.emit("", False, f"Lỗi trích xuất: {e}")
            return

        if self._is_stopped:
            self.finished_signal.emit("", False, "Đã hủy trích xuất.")
            return

        if not segments:
            self.finished_signal.emit("", False, "Không phát hiện được chữ phụ đề hay giọng nói trong video.")
            return

        items = []
        for seg in segments:
            items.append({
                'start_frame': seg.start_frame,
                'end_frame': seg.end_frame,
                'text': seg.text
            })

        # Đường dẫn file SRT đầu ra
        video_p = Path(self.video_path)
        srt_name = f"{video_p.stem}.srt"
        if self.save_dir and os.path.isdir(self.save_dir):
            out_srt_path = os.path.join(self.save_dir, srt_name)
        else:
            out_srt_path = str(video_p.with_suffix('.srt'))

        try:
            SubtitleExporter.export_srt(out_srt_path, items, fps)
            self.finished_signal.emit(out_srt_path, True, f"Trích xuất thành công {len(items)} câu phụ đề.")
        except Exception as e:
            self.finished_signal.emit("", False, f"Lỗi ghi tệp phụ đề: {e}")


class ExtractorInterface(QWidget):
    """Thẻ chuyên biệt cho việc Trích xuất Phụ đề Video (Chuẩn VSE)."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ExtractorInterface")
        self.video_path = None
        self.last_exported_srt_path = None
        self._extract_thread = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Cột bên trái: Khung trình phát video
        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.set_player_mode("extractor")
        main_layout.addWidget(self.video_display_component, 2)

        # Cột bên phải: Cài đặt và Nút bấm hành động
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        settings_card = CardWidget(self)
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(16, 16, 16, 16)
        settings_layout.setSpacing(10)

        # 1. Chế độ nhận diện phụ đề
        self.mode_card = ComboBoxSettingCard(
            configItem=config.ocrMode,
            icon=FluentIcon.SPEED_HIGH,
            title="Chế độ nhận diện phụ đề",
            content="Lựa chọn mức độ ưu tiên tốc độ xử lý hoặc độ chính xác",
            texts=["Tự động chọn mô hình tối ưu", "Ưu tiên tốc độ xử lý nhanh", "Ưu tiên độ chính xác cao nhất"],
            parent=settings_card
        )
        settings_layout.addWidget(self.mode_card)

        # 2. Ngôn ngữ nhận diện phụ đề
        self.lang_card = ComboBoxSettingCard(
            configItem=config.ocrLanguage,
            icon=FluentIcon.LANGUAGE,
            title="Ngôn ngữ phụ đề nhận diện",
            content="Chọn ngôn ngữ của phụ đề hoặc giọng nói có trong video",
            texts=["Tiếng Việt", "Tiếng Anh", "Tiếng Trung", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Pháp", "Tiếng Đức", "Tiếng Nga", "Tiếng Tây Ban Nha"],
            parent=settings_card
        )
        settings_layout.addWidget(self.lang_card)

        # 3. Thư mục lưu file SRT
        self.save_dir_card = PushSettingCard(
            text="Chọn thư mục",
            icon=FluentIcon.FOLDER,
            title="Thư mục lưu tệp phụ đề",
            content="Mặc định lưu cùng vị trí với tệp video gốc",
            parent=settings_card
        )
        self.save_dir_card.clicked.connect(self._on_choose_save_dir)
        settings_layout.addWidget(self.save_dir_card)

        # 4. Quản lý quy tắc sửa lỗi & Watermark (typoMap.json)
        self.typo_card = PushSettingCard(
            text="Mở tệp quy tắc",
            icon=FluentIcon.EDIT,
            title="Bộ lọc quảng cáo và sửa lỗi chính tả",
            content="Tự động lọc các từ quảng cáo hoặc sửa từ OCR viết sai theo quy tắc",
            parent=settings_card
        )
        self.typo_card.clicked.connect(self._on_open_typo_map)
        settings_layout.addWidget(self.typo_card)

        right_layout.addWidget(settings_card)

        # Danh sách tệp video xử lý hàng loạt
        task_container = CardWidget(self)
        task_layout = QHBoxLayout(task_container)
        task_layout.setContentsMargins(0, 0, 0, 0)
        self.task_list_component = TaskListComponent(self)
        self.task_list_component.task_selected.connect(self._on_task_selected)
        task_layout.addWidget(self.task_list_component)
        right_layout.addWidget(task_container, 1)

        # Bảng nút bấm điều khiển
        button_container = CardWidget(self)
        button_layout = QGridLayout(button_container)
        button_layout.setContentsMargins(12, 12, 12, 12)
        button_layout.setSpacing(8)

        self.btn_open = PushButton("Mở Video", self)
        self.btn_open.setIcon(FluentIcon.FOLDER)
        self.btn_open.clicked.connect(self._open_file)
        button_layout.addWidget(self.btn_open, 0, 0)

        self.btn_add_area = PushButton("Thêm vùng chọn", self)
        self.btn_add_area.setIcon(FluentIcon.ADD)
        self.btn_add_area.clicked.connect(self._add_area)
        button_layout.addWidget(self.btn_add_area, 0, 1)

        self.btn_extract = PrimaryPushButton("Trích xuất SRT", self)
        self.btn_extract.setIcon(FluentIcon.DOCUMENT)
        self.btn_extract.clicked.connect(self._start_extraction)
        button_layout.addWidget(self.btn_extract, 1, 0)

        self.btn_jump_translate = PushButton("Dịch SRT", self)
        self.btn_jump_translate.setIcon(FluentIcon.CHAT)
        self.btn_jump_translate.setEnabled(False)
        self.btn_jump_translate.clicked.connect(self._jump_to_translate)
        button_layout.addWidget(self.btn_jump_translate, 1, 1)

        self.btn_save_config = PrimaryPushButton("Lưu Cài Đặt", self)
        self.btn_save_config.setIcon(FluentIcon.SAVE)
        self.btn_save_config.clicked.connect(self._save_extractor_config)
        button_layout.addWidget(self.btn_save_config, 2, 0, 1, 2)

        right_layout.addWidget(button_container)
        main_layout.addLayout(right_layout, 1)

    def _open_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn tệp Video để trích xuất phụ đề",
            "",
            "Tệp Video (*.mp4 *.mkv *.avi *.mov *.flv *.webm);;Tất cả tệp (*)"
        )
        if files:
            for f in files:
                self.task_list_component.add_task(f)
            self._load_video(files[0])

    def _on_task_selected(self, index, file_path):
        self._load_video(file_path)

    def _load_video(self, video_path):
        if not video_path or not os.path.exists(video_path):
            return
        self.video_path = video_path
        self.video_display_component.set_video_path(video_path)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.video_display_component.set_video_parameters(w, h, fps, fc)
            ret, frame = cap.read()
            if ret:
                self.video_display_component.update_video_display(frame)
            cap.release()

    def _add_area(self):
        self.video_display_component.add_default_selection()

    def _on_choose_save_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục lưu tệp phụ đề SRT",
            config.srtSaveDirectory.value or ""
        )
        if directory:
            config.set(config.srtSaveDirectory, directory)
            self.save_dir_card.setContent(directory)

    def _save_extractor_config(self):
        try:
            cfg_file = Path("config") / "auto_pipeline_config.json"
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if cfg_file.exists():
                try:
                    data = json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    data = {}

            data["ocr_lang"] = self.lang_card.comboBox.currentText()
            data["ocr_mode"] = self.mode_card.comboBox.currentText()
            cfg_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            InfoBar.success(
                "Đã Lưu Cài Đặt Trích Xuất",
                f"Ngôn ngữ OCR: {data['ocr_lang']}. Tab Tự Động đã được đồng bộ!",
                parent=self,
                duration=3500
            )
        except Exception as e:
            InfoBar.error("Lỗi", f"Không thể lưu cài đặt trích xuất: {e}", parent=self, duration=3000)

    def _on_open_typo_map(self):
        typo_file = os.path.join("config", "typoMap.json")
        if not os.path.exists(typo_file):
            os.makedirs("config", exist_ok=True)
            with open(typo_file, "w", encoding="utf-8") as f:
                f.write('{\n\t"l\'m": "I\'m"\n}')
        try:
            os.startfile(os.path.abspath(typo_file))
        except Exception as e:
            InfoBar.error("Lỗi mở tệp", str(e), parent=self, duration=3000)

    def _start_extraction(self):
        if not self.video_path or not os.path.exists(self.video_path):
            InfoBar.warning("Cảnh báo", "Vui lòng chọn tệp video trước khi trích xuất!", parent=self, duration=3000)
            return

        if self._extract_thread and self._extract_thread.isRunning():
            InfoBar.info("Thông báo", "Đang trong quá trình trích xuất phụ đề...", parent=self, duration=3000)
            return

        idx = self.mode_card.comboBox.currentIndex()
        ocr_mode = "fast" if idx == 1 else ("precise" if idx == 2 else "auto")

        lang_idx = self.lang_card.comboBox.currentIndex()
        lang_codes = ["vi", "en", "ch", "japan", "korean", "french", "german", "ru", "es"]
        ocr_lang = lang_codes[lang_idx] if lang_idx < len(lang_codes) else "vi"

        sub_areas = self.video_display_component.selection_rects
        save_dir = config.srtSaveDirectory.value or ""

        self.btn_extract.setEnabled(False)
        self.btn_extract.setText("Đang trích xuất...")

        InfoBar.info("Đang xử lý", "Đang bắt đầu nhận diện chữ và trích xuất phụ đề...", parent=self, duration=3000)

        self._extract_thread = FastSrtExtractThread(self.video_path, sub_areas, save_dir, ocr_mode, ocr_lang, parent=self)
        self._extract_thread.finished_signal.connect(self._on_extraction_finished)
        self._extract_thread.finished.connect(self._extract_thread.deleteLater)
        self._extract_thread.start()

    def _on_extraction_finished(self, out_srt_path: str, success: bool, message: str):
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Trích xuất SRT")

        if success and out_srt_path:
            self.last_exported_srt_path = out_srt_path
            self.btn_jump_translate.setEnabled(True)
            InfoBar.success(
                "Hoàn thành trích xuất",
                f"{message}\nTệp phụ đề đã lưu tại: {os.path.basename(out_srt_path)}",
                parent=self,
                duration=4500
            )
        else:
            InfoBar.error("Thất bại", message, parent=self, duration=3500)

    def _jump_to_translate(self):
        main_win = self.window()
        if hasattr(main_win, 'toolsInterface'):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(2)
            if hasattr(self, 'last_exported_srt_path') and self.last_exported_srt_path:
                main_win.toolsInterface.open_srt_in_translator(self.last_exported_srt_path)
