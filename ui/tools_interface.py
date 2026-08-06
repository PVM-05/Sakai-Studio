# -*- coding: utf-8 -*-
"""
Tab Công Cụ — Hub chứa 4 công cụ con chuyên sâu với sub-navigation dạng Pivot.
1. Tải Video
2. Trích Xuất Phụ Đề
3. Dịch & Sửa Phụ Đề
4. Xóa Phụ Đề Video
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from PySide6.QtCore import Qt
from qfluentwidgets import Pivot, FluentIcon

from ui.ytdlp_interface import YtdlpInterface
from ui.extractor_interface import ExtractorInterface
from ui.translation_interface import TranslationInterface
from ui.home_interface import HomeInterface
from backend.config import tr


class ToolsInterface(QWidget):
    """Giao diện gộp 4 công cụ chuyên sâu với thanh chuyển sub-tab mượt mà."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ToolsInterface")
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Thanh Sub-Navigation Pivot ở trên cùng
        self.pivot = Pivot(self)
        
        # 2. QStackedWidget chứa 4 giao diện con
        self.stacked_widget = QStackedWidget(self)

        # Khởi tạo 4 sub-interface
        self.ytdlpInterface = YtdlpInterface(self)
        self.ytdlpInterface.setObjectName("YtdlpSubInterface")

        self.extractorInterface = ExtractorInterface(self)
        self.extractorInterface.setObjectName("ExtractorSubInterface")

        self.translationInterface = TranslationInterface(self)
        self.translationInterface.setObjectName("TranslationSubInterface")

        self.homeInterface = HomeInterface(self)
        self.homeInterface.setObjectName("HomeSubInterface")

        # Thêm các sub-interface vào stacked widget
        self.stacked_widget.addWidget(self.ytdlpInterface)
        self.stacked_widget.addWidget(self.extractorInterface)
        self.stacked_widget.addWidget(self.translationInterface)
        self.stacked_widget.addWidget(self.homeInterface)

        # Thêm các item vào Pivot bar
        self._add_sub_tab(self.ytdlpInterface.objectName(), "Tải Video", 0, icon=FluentIcon.DOWNLOAD)
        self._add_sub_tab(self.extractorInterface.objectName(), "Trích Xuất Phụ Đề", 1, icon=FluentIcon.DOCUMENT)
        self._add_sub_tab(self.translationInterface.objectName(), "Dịch Phụ Đề", 2, icon=FluentIcon.LANGUAGE)
        self._add_sub_tab(self.homeInterface.objectName(), "Xóa Phụ Đề Video", 3, icon=FluentIcon.ERASE_TOOL)

        # Đặt tab mặc định là Trích Xuất Phụ Đề (index 1) hoặc Tải Video (index 0)
        self.pivot.setCurrentItem(self.ytdlpInterface.objectName())
        self.stacked_widget.setCurrentIndex(0)

        main_layout.addWidget(self.pivot, 0, Qt.AlignLeft)
        main_layout.addWidget(self.stacked_widget, 1)

    def _add_sub_tab(self, route_key: str, text: str, index: int, icon=None):
        """Thêm 1 tab con vào Pivot bar với icon chuẩn Fluent."""
        self.pivot.addItem(
            routeKey=route_key,
            text=text,
            onClick=lambda: self.stacked_widget.setCurrentIndex(index),
            icon=icon
        )

    def switch_to_sub_tab(self, index: int):
        """Chuyển sang sub-tab theo index (0=yt-dlp, 1=extractor, 2=translation, 3=remover)."""
        if 0 <= index < self.stacked_widget.count():
            widget = self.stacked_widget.widget(index)
            if widget:
                self.pivot.setCurrentItem(widget.objectName())
                self.stacked_widget.setCurrentIndex(index)

    def open_video_in_remover(self, filepath: str):
        """Mở video đã tải trong công cụ xóa phụ đề."""
        self.switch_to_sub_tab(3)
        if hasattr(self.homeInterface, 'open_downloaded_video'):
            self.homeInterface.open_downloaded_video(filepath)

    def open_srt_in_translator(self, srt_path: str):
        """Mở file SRT đã trích xuất trong công cụ dịch phụ đề."""
        self.switch_to_sub_tab(2)
        if hasattr(self.translationInterface, 'load_subtitle_file'):
            self.translationInterface.load_subtitle_file(srt_path)

    def retranslateUi(self):
        # Update Pivot tab labels
        try:
            self.pivot.item(self.ytdlpInterface.objectName()).setText("Tải Video")
            self.pivot.item(self.extractorInterface.objectName()).setText("Trích Xuất Phụ Đề")
            self.pivot.item(self.translationInterface.objectName()).setText("Dịch Phụ Đề")
            self.pivot.item(self.homeInterface.objectName()).setText("Xóa Phụ Đề Video")
        except Exception:
            pass
