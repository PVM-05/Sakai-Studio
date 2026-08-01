# -*- coding: utf-8 -*-
"""
Giao diện Dịch Thuật Phụ Đề SRT / VTT / ASS Chuyên Nghiệp.
Tích hợp nạp Video / Phụ đề, Trình phát Video Preview, Trích phụ đề OCR tự động & Đồng bộ thời gian.
"""

from __future__ import annotations

import os
import cv2
import json
import threading
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTableWidgetItem,
    QHeaderView, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from qfluentwidgets import (
    CardWidget, PushButton, ComboBox, EditableComboBox, TableWidget, TitleLabel,
    BodyLabel, InfoBar, FluentIcon, ProgressBar, PrimaryPushButton, LineEdit,
    PasswordLineEdit, CaptionLabel, SubtitleLabel
)

from backend.translator import SubtitleTranslator, parse_srt, blocks_to_srt, SubtitleBlock, fetch_accessible_models
from backend.tools.subtitle_exporter import frame_to_timestamp_srt
from backend.tools.folder_memory import FolderMemoryDialog
from ui.component.video_display_component import VideoDisplayComponent
from backend.ocr_engine import VideoOcrEngine

# File cấu hình lưu API Settings người dùng
CONFIG_API_FILE = Path(__file__).parent.parent / "config" / "translation_api_config.json"

LANG_MAP_SOURCE = {
    "Tự động phát hiện": "auto",
    "Tiếng Trung": "zh",
    "Tiếng Anh": "en",
    "Tiếng Nhật": "ja",
    "Tiếng Hàn": "ko",
    "Tiếng Pháp": "fr",
    "Tiếng Đức": "de",
    "Tiếng Nga": "ru",
    "Tiếng Tây Ban Nha": "es",
    "Tiếng Việt": "vi",
}

LANG_MAP_TARGET = {
    "Tiếng Việt": "vi",
    "Tiếng Anh": "en",
    "Tiếng Trung": "zh",
    "Tiếng Nhật": "ja",
    "Tiếng Hàn": "ko",
    "Tiếng Pháp": "fr",
    "Tiếng Đức": "de",
    "Tiếng Tây Ban Nha": "es",
}

PROVIDERS_INFO = {
    "Google Gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"],
        "key_prefix": ["AIzaSy", "AQ.Ab8RN"],
        "is_local": False,
    },
    "OpenAI": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o3-mini", "gpt-3.5-turbo"],
        "key_prefix": ["sk-proj-", "sk-admin-"],
        "is_local": False,
    },
    "Anthropic Claude": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
        "key_prefix": ["sk-ant-"],
        "is_local": False,
    },
    "DeepSeek AI": {
        "label": "DeepSeek AI",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "key_prefix": ["deepseek", "ds-"],
        "is_local": False,
    },
    "GGUF Model": {
        "label": "GGUF Model",
        "base_url": "local_gguf",
        "models": ["Tự động quét file .gguf trong models/"],
        "key_prefix": [],
        "is_local": True,
        "is_gguf": True,
    },
    "MarianMT": {
        "label": "MarianMT",
        "base_url": "local",
        "models": ["Tự động chọn theo Ngôn Ngữ Nguồn (Anh/Trung -> Việt)"],
        "key_prefix": [],
        "is_local": True,
    },
    "Custom Endpoint": {
        "label": "Custom Endpoint",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "claude-3-5-sonnet-latest", "deepseek-chat"],
        "key_prefix": [],
        "is_local": False,
    },
}


class VideoOcrThread(QThread):
    """Worker Thread chạy trích xuất OCR phụ đề từ tệp Video ngầm."""
    progress_signal = Signal(float, str)
    finished_signal = Signal(list)  # Emits list[SubtitleBlock]

    def __init__(self, video_path: str, sub_areas: list, parent=None):
        super().__init__(parent=parent)
        self.video_path = video_path
        self.sub_areas = sub_areas
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        blocks: list[SubtitleBlock] = []
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit(blocks)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        engine = VideoOcrEngine(
            use_typo_map=True,
            use_whisper_fallback=True
        )

        def progress_callback(pct, msg):
            if self._is_stopped:
                engine.stop()
            self.progress_signal.emit(pct, msg)

        try:
            segments = engine.extract_subtitles(self.video_path, self.sub_areas, progress_callback)
        except Exception:
            self.finished_signal.emit(blocks)
            return
            
        if self._is_stopped:
            self.finished_signal.emit([])
            return
            
        if segments:
            for idx, seg in enumerate(segments, start=1):
                blocks.append(SubtitleBlock(idx, seg.start_time, seg.end_time, seg.text))

        self.finished_signal.emit(blocks)


class TranslationInterface(QWidget):
    """
    Giao diện Chuyên biệt cho Dịch Phụ Đề SRT & Trích xuất Video.
    """
    progress_signal = Signal(float, str)
    finished_signal = Signal(bool, str)
    _fetch_models_done_signal = Signal(bool, list, str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("TranslationInterface")
        self.current_srt_path = None
        self.current_video_path = None
        self.batch_folder = None
        self.subtitle_blocks: list[SubtitleBlock] = []
        self._is_translating = False
        self._is_updating_provider = False
        self.ocr_thread: VideoOcrThread | None = None

        self._init_ui()
        self._load_saved_api_config()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # -------------------------------------------------------------
        # 1. Header Title Card
        # -------------------------------------------------------------
        title_card = CardWidget(self)
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(16, 14, 16, 14)

        header_title = TitleLabel("Dịch Phụ Đề & Trích Xuất Video", self)
        header_desc = CaptionLabel(
            "Công cụ chuyên biệt dịch phụ đề SRT/VTT/ASS & Trích xuất phụ đề từ Video. Hỗ trợ dịch tự động bằng các mô hình AI.", self
        )
        title_layout.addWidget(header_title)
        title_layout.addWidget(header_desc)
        main_layout.addWidget(title_card)

        # -------------------------------------------------------------
        # 2. Main Control Settings Card
        # -------------------------------------------------------------
        control_card = CardWidget(self)
        control_layout = QGridLayout(control_card)
        control_layout.setContentsMargins(16, 14, 16, 14)
        control_layout.setSpacing(12)

        # Row 0: Select File & Video OCR
        self.file_button = PushButton("Chọn tệp Phụ đề", self)
        self.file_button.setIcon(FluentIcon.FOLDER)
        control_layout.addWidget(self.file_button, 0, 0)

        self.btn_extract_ocr = PushButton("Trích phụ đề từ Video", self)
        self.btn_extract_ocr.setIcon(FluentIcon.EDIT)
        self.btn_extract_ocr.setEnabled(False)
        control_layout.addWidget(self.btn_extract_ocr, 0, 1)

        # Language Selectors
        self.src_lang_label = BodyLabel("Ngôn ngữ nguồn:", self)
        control_layout.addWidget(self.src_lang_label, 0, 2, Qt.AlignRight | Qt.AlignVCenter)
        self.src_lang_combo = ComboBox(self)
        self.src_lang_combo.addItems(list(LANG_MAP_SOURCE.keys()))
        control_layout.addWidget(self.src_lang_combo, 0, 3)

        self.tgt_lang_label = BodyLabel("Ngôn ngữ đích:", self)
        control_layout.addWidget(self.tgt_lang_label, 0, 4, Qt.AlignRight | Qt.AlignVCenter)
        self.tgt_lang_combo = ComboBox(self)
        self.tgt_lang_combo.addItems(list(LANG_MAP_TARGET.keys()))
        control_layout.addWidget(self.tgt_lang_combo, 0, 5)

        # Row 1: Engine & Action
        self.engine_label = BodyLabel("Mô hình dịch:", self)
        control_layout.addWidget(self.engine_label, 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.engine_combo = ComboBox(self)
        self.engine_combo.addItems([
            "Google Translate",
            "Mô hình AI tùy chỉnh "
        ])
        control_layout.addWidget(self.engine_combo, 1, 1, 1, 3)

        self.btn_toggle_api = PushButton("Cấu hình AI", self)
        self.btn_toggle_api.setIcon(FluentIcon.SETTING)
        control_layout.addWidget(self.btn_toggle_api, 1, 4)

        self.translate_btn = PrimaryPushButton("Bắt đầu dịch", self)
        self.translate_btn.setIcon(FluentIcon.PLAY)
        control_layout.addWidget(self.translate_btn, 1, 5)

        main_layout.addWidget(control_card)

        # -------------------------------------------------------------
        # 3. Expandable AI API Key & Model Config Card
        # -------------------------------------------------------------
        self.api_config_card = CardWidget(self)
        api_layout = QGridLayout(self.api_config_card)
        api_layout.setContentsMargins(16, 14, 16, 14)
        api_layout.setSpacing(10)

        # Provider Selector Row
        api_layout.addWidget(BodyLabel("Nhà cung cấp AI:"), 0, 0, Qt.AlignRight | Qt.AlignVCenter)

        provider_box = QHBoxLayout()
        self.provider_combo = ComboBox(self)
        for key, info in PROVIDERS_INFO.items():
            self.provider_combo.addItem(info["label"])
        provider_box.addWidget(self.provider_combo)
        provider_box.addStretch()

        api_layout.addLayout(provider_box, 0, 1, 1, 3)

        # API Key Row
        self.api_key_label = BodyLabel("API Key:")
        api_layout.addWidget(self.api_key_label, 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        
        self.key_box_widget = QWidget(self)
        key_box = QHBoxLayout(self.key_box_widget)
        key_box.setContentsMargins(0, 0, 0, 0)
        self.api_key_edit = PasswordLineEdit(self)
        self.api_key_edit.setPlaceholderText("Nhập API Key")
        key_box.addWidget(self.api_key_edit, 1)

        self.btn_verify_key = PrimaryPushButton("Xác thực & Nạp Key", self)
        self.btn_verify_key.setIcon(FluentIcon.ACCEPT)
        key_box.addWidget(self.btn_verify_key)

        api_layout.addWidget(self.key_box_widget, 1, 1)

        # Model Name Dropdown
        api_layout.addWidget(BodyLabel("Tên Model:"), 1, 2, Qt.AlignRight | Qt.AlignVCenter)
        self.model_combo = ComboBox(self)
        self.model_combo.setPlaceholderText("Vui lòng xác thực API Key...")
        self.model_combo.setEnabled(False)
        api_layout.addWidget(self.model_combo, 1, 3)

        # Custom Prompt Instruction
        api_layout.addWidget(BodyLabel("Prompt tùy chỉnh:"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.prompt_edit = LineEdit(self)
        self.prompt_edit.setPlaceholderText("Ví dụ: Dịch phụ đề chuẩn ngữ cảnh video, tự nhiên và mượt mà.")
        api_layout.addWidget(self.prompt_edit, 2, 1, 1, 3)

        # Save Settings Button
        self.btn_save_config = PrimaryPushButton("Lưu Cài Đặt", self)
        self.btn_save_config.setIcon(FluentIcon.SAVE)
        self.btn_save_config.clicked.connect(lambda: self._save_api_config(show_toast=True))
        api_layout.addWidget(self.btn_save_config, 3, 1, 1, 3)

        self.api_config_card.setVisible(False)
        main_layout.addWidget(self.api_config_card)

        # -------------------------------------------------------------
        # 4. Video Preview Player Card (Dynamic Visible for Video)
        # -------------------------------------------------------------
        self.video_card = CardWidget(self)
        video_card_layout = QVBoxLayout(self.video_card)
        video_card_layout.setContentsMargins(12, 12, 12, 12)
        
        self.video_display = VideoDisplayComponent(self)
        self.video_display.set_player_mode("translator")
        video_card_layout.addWidget(self.video_display)

        self.video_card.setVisible(False)
        main_layout.addWidget(self.video_card)

        # -------------------------------------------------------------
        # 5. Subtitle Editor Table Card
        # -------------------------------------------------------------
        self.table_card = CardWidget(self)
        table_layout = QVBoxLayout(self.table_card)
        table_layout.setContentsMargins(12, 12, 12, 12)
        table_layout.setSpacing(8)

        # Search Bar
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(BodyLabel("Tìm kiếm câu phụ đề:"))
        self.search_edit = LineEdit(self)
        self.search_edit.setPlaceholderText("Nhập từ khóa cần lọc...")
        filter_layout.addWidget(self.search_edit)
        table_layout.addLayout(filter_layout)

        # Table Widget
        self.table = TableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "STT", "Mốc thời gian", "Văn bản gốc", "Văn bản dịch (Nhấp đúp để sửa)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table_layout.addWidget(self.table)

        main_layout.addWidget(self.table_card, 1)

        # -------------------------------------------------------------
        # 6. Progress & Export Action Bar
        # -------------------------------------------------------------
        export_card = CardWidget(self)
        export_layout = QHBoxLayout(export_card)
        export_layout.setContentsMargins(12, 10, 12, 10)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setValue(0)
        export_layout.addWidget(self.progress_bar, 1)

        self.status_label = BodyLabel("Sẵn sàng.", self)
        export_layout.addWidget(self.status_label)

        self.pause_btn = PushButton("Tạm dừng", self)
        self.pause_btn.setIcon(FluentIcon.PAUSE)
        self.pause_btn.setVisible(False)
        export_layout.addWidget(self.pause_btn)

        self.stop_btn = PushButton("Dừng dịch", self)
        self.stop_btn.setIcon(FluentIcon.CANCEL)
        self.stop_btn.setVisible(False)
        export_layout.addWidget(self.stop_btn)

        self.export_srt_button = PushButton("Lưu file phụ đề dịch", self)
        self.export_srt_button.setIcon(FluentIcon.SAVE)
        export_layout.addWidget(self.export_srt_button)

        main_layout.addWidget(export_card)

    def _connect_signals(self):
        self.file_button.clicked.connect(self._open_single_file)
        self.btn_extract_ocr.clicked.connect(self._start_video_ocr_extraction)
        self.btn_toggle_api.clicked.connect(self._toggle_api_card)
        self.btn_verify_key.clicked.connect(self._verify_and_load_api_key_with_toast)

        self.src_lang_combo.currentIndexChanged.connect(self._save_api_config)
        self.tgt_lang_combo.currentIndexChanged.connect(self._save_api_config)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.engine_combo.currentIndexChanged.connect(self._save_api_config)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.api_key_edit.textChanged.connect(self._on_api_key_typed)
        self.api_key_edit.textChanged.connect(self._save_api_config)
        self.model_combo.currentIndexChanged.connect(self._save_api_config)
        self.prompt_edit.textChanged.connect(self._save_api_config)

        self.translate_btn.clicked.connect(self._start_translation)
        self.pause_btn.clicked.connect(self._toggle_pause_translation)
        self.stop_btn.clicked.connect(self._stop_translation)
        self.search_edit.textChanged.connect(self._filter_table)
        self.table.cellChanged.connect(self._on_table_cell_changed)
        self.table.itemSelectionChanged.connect(self._on_table_row_selected)
        self.export_srt_button.clicked.connect(self._export_srt)

        self.progress_signal.connect(self._update_progress)
        self.finished_signal.connect(self._on_translation_finished)
        self._fetch_models_done_signal.connect(self._on_fetch_models_done)

    def _toggle_api_card(self):
        is_vis = not self.api_config_card.isVisible()
        self.api_config_card.setVisible(is_vis)
        if is_vis:
            self._on_provider_changed(self.provider_combo.currentIndex())

    def _on_engine_changed(self, index: int):
        if index == 1:
            self.api_config_card.setVisible(True)
        self._on_provider_changed(self.provider_combo.currentIndex())

    def _detect_provider_from_key(self, key: str) -> str | None:
        key = key.strip()
        if not key or len(key) < 6:
            return None

        if (key.startswith("AIzaSy") or key.startswith("AQ.") or key.startswith("AQ-") or key.startswith("projects/")) and len(key) >= 15:
            return "Google Gemini"
        elif (key.startswith("sk-ant-api03-") or key.startswith("sk-ant-admin-") or key.startswith("sk-ant-")) and len(key) >= 12:
            return "Anthropic Claude"
        elif (key.startswith("sk-proj-") or key.startswith("sk-admin-") or key.startswith("sk-svcacct-") or key.startswith("sess-")) and len(key) >= 12:
            return "OpenAI"
        elif key.startswith("sk-or-v1-") or key.startswith("gsk_") or (key.startswith("sk-") and len(key) >= 15):
            return "OpenAI"
        elif key.startswith("deepseek") or key.startswith("ds-"):
            return "DeepSeek AI"
        elif any(loc in key.lower() for loc in ["localhost", "127.0.0.1", "http://", "https://", "ollama"]):
            return "Ollama"

        if len(key) >= 10:
            provider_keys = list(PROVIDERS_INFO.keys())
            curr_idx = self.provider_combo.currentIndex()
            if 0 <= curr_idx < len(provider_keys):
                return provider_keys[curr_idx]

        return "Google Gemini"

    def _verify_and_load_api_key_with_toast(self):
        key = self.api_key_edit.text().strip()
        if not key:
            InfoBar.warning("Cảnh báo", "Vui lòng nhập API Key trước khi xác thực!", parent=self, duration=3000)
            self.model_combo.clear()
            self.model_combo.setPlaceholderText("Vui lòng nhập API Key...")
            self.model_combo.setEnabled(False)
            return

        detected = self._detect_provider_from_key(key)
        if not detected:
            InfoBar.error("Lỗi API Key", "API Key không đúng định dạng nhận diện", parent=self, duration=3500)
            self.model_combo.clear()
            self.model_combo.setPlaceholderText("API Key không đúng định dạng...")
            self.model_combo.setEnabled(False)
            return

        provider_keys = list(PROVIDERS_INFO.keys())
        if detected in provider_keys:
            idx = provider_keys.index(detected)
            self.provider_combo.setCurrentIndex(idx)

        ok, models, msg = fetch_accessible_models(detected, key)
        self.model_combo.clear()
        if ok and models:
            self.model_combo.setEnabled(True)
            self.model_combo.addItems(models)
            self.model_combo.setCurrentIndex(0)
            InfoBar.success("Xác thực thành công", f"Đã kết nối {detected}! Nạp thành công {len(models)} model.", parent=self, duration=3500)
        else:
            self.model_combo.setPlaceholderText(msg)
            self.model_combo.setEnabled(False)
            InfoBar.error("Xác thực thất bại", msg, parent=self, duration=4000)

        self._save_api_config()

    def _on_provider_changed(self, index: int):
        if self._is_updating_provider:
            return

        provider_keys = list(PROVIDERS_INFO.keys())
        if 0 <= index < len(provider_keys):
            p_key = provider_keys[index]
            info = PROVIDERS_INFO[p_key]
            self._is_updating_provider = True
            self.model_combo.clear()

            if info.get("is_gguf"):
                from backend.translator import get_available_gguf_models
                self.api_key_label.setVisible(False)
                self.key_box_widget.setVisible(False)
                self.model_combo.setEnabled(True)
                ggufs = get_available_gguf_models()
                if ggufs:
                    self.model_combo.addItems([p.name for p in ggufs])
                else:
                    self.model_combo.addItem("Tự động tải Qwen2.5 GGUF khi bắt đầu Dịch")
                self.model_combo.setCurrentIndex(0)
            elif info.get("is_local"):
                self.api_key_label.setVisible(False)
                self.key_box_widget.setVisible(False)
                self.model_combo.setEnabled(True)
                self.model_combo.addItems(info["models"])
                self.model_combo.setCurrentIndex(0)
            else:
                self.api_key_label.setVisible(True)
                self.key_box_widget.setVisible(True)
                self.api_key_edit.setPlaceholderText("Nhập API Key")
                key = self.api_key_edit.text().strip()
                detected = self._detect_provider_from_key(key)
                if key and (detected == p_key or p_key == "Custom Endpoint"):
                    ok, models, msg = fetch_accessible_models(p_key, key)
                    if ok and models:
                        self.model_combo.setEnabled(True)
                        self.model_combo.addItems(models)
                        self.model_combo.setCurrentIndex(0)
                    else:
                        self.model_combo.setPlaceholderText(msg)
                        self.model_combo.setEnabled(False)
                else:
                    self.model_combo.setPlaceholderText("Vui lòng nhập API Key hợp lệ...")
                    self.model_combo.setEnabled(False)

            self._is_updating_provider = False
            self._save_api_config()

    @Slot(bool, list, str)
    def _on_fetch_models_done(self, ok: bool, models: list, msg: str):
        self.model_combo.clear()
        if ok and models:
            self.model_combo.setEnabled(True)
            self.model_combo.addItems(models)
            self.model_combo.setCurrentIndex(0)
        else:
            self.model_combo.setPlaceholderText(msg)
            self.model_combo.setEnabled(False)

    def _on_api_key_typed(self, key_text: str):
        if hasattr(self, '_debounce_timer') and self._debounce_timer is not None:
            self._debounce_timer.stop()
            
        provider_keys = list(PROVIDERS_INFO.keys())
        curr_idx = self.provider_combo.currentIndex()
        if 0 <= curr_idx < len(provider_keys):
            curr_p_key = provider_keys[curr_idx]
            if PROVIDERS_INFO.get(curr_p_key, {}).get("is_local"):
                # If currently selected provider is Local, do not auto-switch via API key
                return

        key = key_text.strip()
        detected_provider = self._detect_provider_from_key(key)

        if not detected_provider:
            self.model_combo.clear()
            self.model_combo.setPlaceholderText("API Key không đúng định dạng...")
            self.model_combo.setEnabled(False)
            return

        provider_keys = list(PROVIDERS_INFO.keys())
        if detected_provider in provider_keys:
            idx = provider_keys.index(detected_provider)
            if self.provider_combo.currentIndex() != idx:
                self.provider_combo.setCurrentIndex(idx)
            else:
                def _fetch_worker():
                    try:
                        ok, models, msg = fetch_accessible_models(detected_provider, key)
                        self._fetch_models_done_signal.emit(ok, models, msg)
                    except Exception:
                        self._fetch_models_done_signal.emit(False, [], "Lỗi kết nối")

                self.model_combo.clear()
                self.model_combo.setPlaceholderText("Đang kiểm tra API Key...")
                self.model_combo.setEnabled(False)
                from PySide6.QtCore import QTimer
                self._debounce_timer = QTimer(self)
                self._debounce_timer.setSingleShot(True)
                self._debounce_timer.timeout.connect(lambda: threading.Thread(target=_fetch_worker, daemon=True).start())
                self._debounce_timer.start(500)  # 500ms debounce

    def _open_single_file(self):
        filepath, _ = FolderMemoryDialog.getOpenFileName(
            self,
            "Chọn tệp Video hoặc Phụ đề",
            filter_str="Tất cả định dạng (*.mp4 *.mkv *.avi *.mov *.flv *.webm *.srt *.vtt *.ass);;Tệp Video (*.mp4 *.mkv *.avi *.mov *.flv *.webm);;Tệp Phụ đề (*.srt *.vtt *.ass);;Tất cả tệp (*)",
            category="subtitle"
        )
        if filepath:
            ext = os.path.splitext(filepath)[-1].lower()
            if ext in ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm'):
                # Mở Video File
                self.current_video_path = filepath
                self.video_card.setVisible(True)
                self.btn_extract_ocr.setEnabled(True)

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

                InfoBar.success(
                    "Đã mở tệp Video",
                    f"Tệp: {os.path.basename(filepath)}. Bấm 'Trích phụ đề từ Video' để quét chữ OCR!",
                    parent=self,
                    duration=3500,
                )
            else:
                self.load_subtitle_file(filepath)

    def load_subtitle_file(self, filepath: str):
        """Nạp trực tiếp tệp phụ đề từ đường dẫn"""
        if not filepath or not os.path.exists(filepath):
            return
        self.current_srt_path = filepath
        self.video_card.setVisible(False)
        self.btn_extract_ocr.setEnabled(False)

        try:
            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            self.subtitle_blocks = parse_srt(content)
            self._populate_table(self.subtitle_blocks)
            InfoBar.success(
                "Đã nạp tệp phụ đề",
                f"Tệp {os.path.basename(filepath)} ({len(self.subtitle_blocks)} câu)",
                parent=self,
                duration=3500,
            )
        except Exception as e:
            InfoBar.error("Lỗi nạp phụ đề", str(e), parent=self, duration=3500)

    def _start_video_ocr_extraction(self):
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            InfoBar.warning("Cảnh báo", "Vui lòng mở một tệp Video trước khi trích phụ đề!", parent=self, duration=3000)
            return

        if self.ocr_thread and self.ocr_thread.isRunning():
            return

        sub_areas = self.video_display.get_selection_coordinates()
        self.btn_extract_ocr.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Đang trích xuất phụ đề OCR từ Video...")

        self.ocr_thread = VideoOcrThread(self.current_video_path, sub_areas, parent=self)
        self.ocr_thread.progress_signal.connect(self._update_progress)
        self.ocr_thread.finished_signal.connect(self._on_ocr_extraction_finished)
        self.ocr_thread.finished.connect(self.ocr_thread.deleteLater)
        self.ocr_thread.start()

    @Slot(list)
    def _on_ocr_extraction_finished(self, blocks: list[SubtitleBlock]):
        self.btn_extract_ocr.setEnabled(True)
        self.progress_bar.setValue(100)
        self.subtitle_blocks = blocks
        self._populate_table(blocks)

        if blocks:
            self.status_label.setText(f"Trích xuất thành công {len(blocks)} câu phụ đề!")
            InfoBar.success("Trích xuất hoàn tất", f"Đã trích xuất {len(blocks)} câu phụ đề từ Video sẵn sàng dịch!", parent=self, duration=3500)
        else:
            self.status_label.setText("Không tìm thấy chữ phụ đề trong vùng quét.")
            InfoBar.warning("Thông báo", "Không phát hiện chữ phụ đề trong video.", parent=self, duration=3000)

    def _on_table_row_selected(self):
        """Khi chọn câu phụ đề trong bảng, tự động nhảy video tới mốc thời gian đó."""
        if not self.video_card.isVisible() or not self.current_video_path:
            return

        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self.subtitle_blocks):
            block = self.subtitle_blocks[row]
            # Parse start time HH:MM:SS,mmm
            try:
                parts = block.start_time.replace(',', '.').split(':')
                s_sec = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
                fps = self.video_display.fps if hasattr(self.video_display, 'fps') and self.video_display.fps else 30.0
                frame_no = max(1, int(s_sec * fps))
                self.video_display.video_slider.setValue(frame_no)
                self.video_display.sync_audio_position(frame_no)
            except Exception:
                pass

    def _populate_table(self, blocks: list[SubtitleBlock]):
        self.table.blockSignals(True)
        self.table.setRowCount(len(blocks))
        for row, block in enumerate(blocks):
            item_idx = QTableWidgetItem(str(block.index))
            item_idx.setFlags(item_idx.flags() ^ Qt.ItemIsEditable)

            item_time = QTableWidgetItem(f"{block.start_time} --> {block.end_time}")
            item_time.setFlags(item_time.flags() ^ Qt.ItemIsEditable)

            item_orig = QTableWidgetItem(block.text)
            item_orig.setFlags(item_orig.flags() ^ Qt.ItemIsEditable)

            item_trans = QTableWidgetItem(block.translated_text if block.translated_text else "")

            self.table.setItem(row, 0, item_idx)
            self.table.setItem(row, 1, item_time)
            self.table.setItem(row, 2, item_orig)
            self.table.setItem(row, 3, item_trans)
        self.table.blockSignals(False)

    def _filter_table(self, query: str):
        query = query.lower().strip()
        for row in range(self.table.rowCount()):
            orig = self.table.item(row, 2).text().lower() if self.table.item(row, 2) else ""
            trans = self.table.item(row, 3).text().lower() if self.table.item(row, 3) else ""
            match = (query in orig) or (query in trans)
            self.table.setRowHidden(row, not match if query else False)

    def _on_table_cell_changed(self, row: int, column: int):
        if column == 3 and 0 <= row < len(self.subtitle_blocks):
            new_text = self.table.item(row, 3).text()
            self.subtitle_blocks[row].translated_text = new_text

    def _start_translation(self):
        if self._is_translating:
            return

        if not self.subtitle_blocks and not self.current_srt_path:
            InfoBar.warning("Cảnh báo", "Vui lòng mở tệp phụ đề hoặc trích xuất phụ đề từ Video trước khi dịch!", parent=self, duration=3000)
            return

        src_selected = self.src_lang_combo.currentText()
        tgt_selected = self.tgt_lang_combo.currentText()

        source_lang = LANG_MAP_SOURCE.get(src_selected, "auto")
        target_lang = LANG_MAP_TARGET.get(tgt_selected, "vi")

        engine_idx = self.engine_combo.currentIndex()
        engine_type = "llm" if engine_idx == 1 else "google"

        provider_keys = list(PROVIDERS_INFO.keys())
        p_key = provider_keys[self.provider_combo.currentIndex()] if 0 <= self.provider_combo.currentIndex() < len(provider_keys) else "Google Gemini"
        p_info = PROVIDERS_INFO.get(p_key, {})
        is_local = p_info.get("is_local", False)

        if p_key in ("MarianMT", "Local NMT GPU (Offline)"):
            engine_type = "local_nmt"
        elif p_key == "GGUF Model":
            engine_type = "gguf"
            from backend.translator import get_available_gguf_models
            ggufs = get_available_gguf_models()
            selected_model_name = self.model_combo.currentText().strip()
            matched = [p for p in ggufs if p.name == selected_model_name]
            if matched:
                model_name = str(matched[0])
            elif ggufs:
                model_name = str(ggufs[0])
            else:
                model_name = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

        api_key = self.api_key_edit.text().strip()
        api_base = p_info.get("base_url", "https://api.openai.com/v1")
        model_name = self.model_combo.currentText().strip() or "gemini-1.5-flash"
        custom_prompt = self.prompt_edit.text().strip()

        if engine_type == "llm" and not is_local and not api_key:
            InfoBar.warning("Cần API Key", "Vui lòng nhập API Key để dùng mô hình AI Online!", parent=self, duration=3500)
            self.api_config_card.setVisible(True)
            return

        self._is_translating = True
        self._is_paused = False
        self._is_stopped = False
        self._populate_table(self.subtitle_blocks)
        self.translate_btn.setEnabled(False)
        self.pause_btn.setVisible(True)
        self.pause_btn.setText("Tạm dừng")
        self.pause_btn.setIcon(FluentIcon.PAUSE)
        self.stop_btn.setVisible(True)
        self.progress_bar.setValue(0)

        def _cancel_check():
            if self._is_stopped:
                return True
            if self._is_paused:
                return "paused"
            return False

        def _worker():
            try:
                translator = SubtitleTranslator(
                    engine=engine_type,
                    api_key=api_key if engine_type == "llm" else None,
                    api_base=api_base if engine_type == "llm" else None,
                    model=model_name,
                )

                if engine_type in ("llm", "gguf") and custom_prompt and hasattr(translator.translator, 'custom_prompt'):
                    translator.translator.custom_prompt = custom_prompt + "\nRespond ONLY with a JSON array of translated strings matching the input array order."

                translator.translator.translate_blocks(
                    self.subtitle_blocks,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    progress_callback=lambda p, m: self.progress_signal.emit(p, m),
                    cancel_callback=_cancel_check,
                )
                if self._is_stopped:
                    self.finished_signal.emit(True, "Đã dừng dịch thuật theo yêu cầu!")
                else:
                    self.finished_signal.emit(True, "Dịch thành công tất cả phụ đề!")
            except Exception as e:
                self.finished_signal.emit(False, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_table_live(self):
        """Cập nhật trực tiếp nội dung văn bản dịch lên bảng phụ đề trong khi đang dịch."""
        if not self.subtitle_blocks:
            return
        row_count = self.table.rowCount()
        if row_count != len(self.subtitle_blocks):
            self._populate_table(self.subtitle_blocks)
            return

        self.table.blockSignals(True)
        for row, b in enumerate(self.subtitle_blocks):
            if b.translated_text:
                item = self.table.item(row, 3)
                if item and item.text() != b.translated_text:
                    item.setText(b.translated_text)
        self.table.blockSignals(False)

    def _toggle_pause_translation(self):
        if not self._is_translating:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.pause_btn.setText("Tiếp tục")
            self.pause_btn.setIcon(FluentIcon.PLAY)
            self.status_label.setText("Đã tạm dừng dịch thuật.")
        else:
            self.pause_btn.setText("Tạm dừng")
            self.pause_btn.setIcon(FluentIcon.PAUSE)
            self.status_label.setText("Đang tiếp tục dịch...")

    def _stop_translation(self):
        if not self._is_translating:
            return
        self._is_stopped = True
        self.status_label.setText("Đang dừng dịch thuật...")

    @Slot(float, str)
    def _update_progress(self, val: float, msg: str):
        self.progress_bar.setValue(int(val * 100))
        self.status_label.setText(msg)
        self._update_table_live()

    @Slot(bool, str)
    def _on_translation_finished(self, success: bool, message: str):
        self._is_translating = False
        self._is_paused = False
        self._is_stopped = False
        self.translate_btn.setEnabled(True)
        self.pause_btn.setVisible(False)
        self.stop_btn.setVisible(False)

        if success:
            if "dừng dịch" in message.lower():
                self.status_label.setText("Đã dừng dịch.")
                InfoBar.warning("Thông báo", message, parent=self, duration=3500)
            else:
                self.progress_bar.setValue(100)
                self.status_label.setText("Hoàn thành dịch thuật!")
                InfoBar.success("Thành công", "Đã dịch xong toàn bộ tệp phụ đề!", parent=self, duration=3500)
            self._populate_table(self.subtitle_blocks)
        else:
            self.status_label.setText("Lỗi khi dịch.")
            InfoBar.error("Lỗi Dịch Thuật", message, parent=self, duration=4000)

    def _export_srt(self):
        if not self.subtitle_blocks:
            InfoBar.warning("Cảnh báo", "Không có dữ liệu phụ đề để xuất!", parent=self, duration=2500)
            return

        default_name = "translated.srt"
        if self.current_srt_path:
            p = Path(self.current_srt_path)
            default_name = f"{p.stem}_translated.srt"
        elif self.current_video_path:
            p = Path(self.current_video_path)
            default_name = f"{p.stem}_translated.srt"

        filepath, _ = FolderMemoryDialog.getSaveFileName(
            self, "Lưu file phụ đề dịch", default_filename=default_name, filter_str="Phụ đề (*.srt)", category="subtitle"
        )
        if filepath:
            content = blocks_to_srt(self.subtitle_blocks, use_translated=True)
            Path(filepath).write_text(content, encoding="utf-8")
            InfoBar.success("Đã lưu", f"Đã xuất tệp phụ đề tại: {os.path.basename(filepath)}", parent=self, duration=3000)

    def _save_api_config(self, show_toast=False):
        try:
            CONFIG_API_FILE.parent.mkdir(parents=True, exist_ok=True)
            engine_idx = self.engine_combo.currentIndex()
            provider_idx = self.provider_combo.currentIndex()
            provider_keys = list(PROVIDERS_INFO.keys())
            provider_name = provider_keys[provider_idx] if 0 <= provider_idx < len(provider_keys) else ""

            if engine_idx == 0:
                engine_type = "google"
            elif provider_name == "GGUF Model":
                engine_type = "gguf"
            elif provider_name == "MarianMT":
                engine_type = "marian"
            else:
                engine_type = "llm"

            model_name = self.model_combo.currentText().strip()
            data = {
                "src_lang_index": self.src_lang_combo.currentIndex(),
                "tgt_lang_index": self.tgt_lang_combo.currentIndex(),
                "engine_index": engine_idx,
                "provider_index": provider_idx,
                "engine_type": engine_type,
                "provider_name": provider_name,
                "api_key": self.api_key_edit.text().strip(),
                "model_name": model_name,
                "custom_prompt": self.prompt_edit.text().strip(),
            }
            CONFIG_API_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            if show_toast:
                engine_label = f"Local GGUF ({model_name})" if engine_type == "gguf" else (
                    "Local MarianMT" if engine_type == "marian" else (
                        f"AI API ({provider_name} - {model_name})" if engine_type == "llm" else "Google Translate"
                    )
                )
                InfoBar.success(
                    "Đã Lưu & Đồng Bộ Cài Đặt Dịch",
                    f"Mô hình dịch: {engine_label}. Tab Tự Động đã được đồng bộ!",
                    parent=self,
                    duration=3500
                )
        except Exception as e:
            if show_toast:
                InfoBar.error("Lỗi", f"Không thể lưu cài đặt dịch: {e}", parent=self, duration=3000)

    def _load_saved_api_config(self):
        if CONFIG_API_FILE.exists():
            try:
                data = json.loads(CONFIG_API_FILE.read_text(encoding="utf-8"))
                if "src_lang_index" in data and 0 <= data["src_lang_index"] < self.src_lang_combo.count():
                    self.src_lang_combo.setCurrentIndex(data["src_lang_index"])
                if "tgt_lang_index" in data and 0 <= data["tgt_lang_index"] < self.tgt_lang_combo.count():
                    self.tgt_lang_combo.setCurrentIndex(data["tgt_lang_index"])
                if "engine_index" in data and 0 <= data["engine_index"] < self.engine_combo.count():
                    self.engine_combo.setCurrentIndex(data["engine_index"])
                if "provider_index" in data and 0 <= data["provider_index"] < self.provider_combo.count():
                    self.provider_combo.setCurrentIndex(data["provider_index"])
                if "api_key" in data and data["api_key"]:
                    self.api_key_edit.setText(data["api_key"])
                    self._on_api_key_typed(data["api_key"])
                if "model_name" in data and data["model_name"]:
                    idx = self.model_combo.findText(data["model_name"])
                    if idx >= 0:
                        self.model_combo.setCurrentIndex(idx)
                if "custom_prompt" in data and data["custom_prompt"]:
                    self.prompt_edit.setText(data["custom_prompt"])
                self._on_provider_changed(self.provider_combo.currentIndex())
            except Exception:
                pass

    def retranslateUi(self):
        pass
