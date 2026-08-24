# -*- coding: utf-8 -*-
"""
Cấu Hình MMO (Kiếm tiền online)
Chứa các bộ lọc và thủ thuật chống gậy bản quyền (Copyright bypass)
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from qfluentwidgets import (ScrollArea, ExpandLayout, SettingCardGroup, 
                            SwitchSettingCard, RangeSettingCard, ComboBoxSettingCard,
                            TitleLabel, FluentIcon, SettingCard, DoubleSpinBox)
from src.core.config import config, tr

class DoubleSpinBoxSettingCard(SettingCard):
    """ Custom Setting Card with DoubleSpinBox for Float values """
    def __init__(self, icon, title, content, configItem=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.spinBox = DoubleSpinBox(self)
        self.spinBox.setMinimumWidth(120)
        self.spinBox.setRange(0.1, 5.0)
        self.spinBox.setSingleStep(0.05)
        self.spinBox.setDecimals(2)
        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        
        if self.configItem:
            self.spinBox.setValue(self.configItem.value)
            self.spinBox.valueChanged.connect(self._onValueChanged)
            
    def _onValueChanged(self, value):
        if self.configItem:
            config.set(self.configItem, value)


class MMOSettingInterface(ScrollArea):
    """ Giao diện Cấu Hình MMO """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        self.setObjectName("MMOSettingInterface")
        
        self._init_ui()
        self._init_settings()

    def _init_ui(self):
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.scrollWidget.setObjectName("scrollWidget")
        
        # Tiêu đề
        self.title = TitleLabel("Cấu Hình MMO (Chống bản quyền)", self)
        self.expandLayout.addWidget(self.title)

    def _init_settings(self):
        self.mmoGroup = SettingCardGroup("Lách Bản Quyền Cơ Bản", self.scrollWidget)
        
        # 1. Lật ngang video
        self.flipCard = SwitchSettingCard(
            FluentIcon.SYNC,
            "Lật Ngang Video",
            "Tự động lật ngược video theo chiều ngang để đánh lừa hệ thống nhận diện.",
            configItem=config.mmoHorizontalFlip,
            parent=self.mmoGroup
        )
        
        # 2. Tốc độ
        self.speedCard = DoubleSpinBoxSettingCard(
            FluentIcon.SPEED_OFF,
            "Tốc Độ Video & Âm Thanh",
            "Tăng hoặc giảm tốc độ nhẹ (ví dụ: 1.05 hoặc 0.95) để tránh khớp mã hash video.",
            configItem=config.mmoSpeedMultiplier,
            parent=self.mmoGroup
        )
        
        # 3. Cắt viền (Crop)
        self.cropCard = RangeSettingCard(
            config.mmoCropPercentage,
            FluentIcon.CUT,
            "Cắt Viền (Crop %)",
            "Cắt bớt viền xung quanh khung hình (1% - 50%) để lách thuật toán so khớp khung hình.",
            self.mmoGroup
        )
        
        # 4. Phong cách phụ đề
        self.subStyleCard = ComboBoxSettingCard(
            config.mmoSubStyle,
            FluentIcon.FONT,
            "Phong Cách Phụ Đề Chèn Lại",
            "Định dạng màu chữ, viền chữ đặc trưng (dành riêng cho nền tảng video ngắn).",
            texts=["TikTok Vàng", "YouTube Trắng", "Netflix Cổ Điển", "Tùy Chỉnh"],
            parent=self.mmoGroup
        )
        
        # 5. Smart Tracking Mask
        self.smartMaskCard = SwitchSettingCard(
            FluentIcon.EDIT,
            "Mặt Nạ Văn Bản AI (Smart Mask)",
            "Chỉ xóa đúng các điểm ảnh thuộc nét chữ thay vì xóa cả mảng, giữ nguyên 80% chi tiết nền video.",
            configItem=config.mmoSmartTextMask,
            parent=self.mmoGroup
        )
        
        self.mmoGroup.addSettingCard(self.flipCard)
        self.mmoGroup.addSettingCard(self.speedCard)
        self.mmoGroup.addSettingCard(self.cropCard)
        self.mmoGroup.addSettingCard(self.subStyleCard)
        self.mmoGroup.addSettingCard(self.smartMaskCard)
        
        self.expandLayout.addWidget(self.mmoGroup)

        # TTS & Audio Ducking Group
        self.ttsGroup = SettingCardGroup("Lồng Tiếng AI (Offline TTS)", self.scrollWidget)
        
        self.enableTTSCard = SwitchSettingCard(
            FluentIcon.MICROPHONE,
            "Bật Lồng Tiếng AI (OmniVoice Local)",
            "Hệ thống sẽ tự động tổng hợp giọng nói từ phụ đề và đè lên video. Sử dụng GPU cục bộ.",
            configItem=config.mmoEnableTTS,
            parent=self.ttsGroup
        )
        
        self.duckingCard = DoubleSpinBoxSettingCard(
            FluentIcon.VOLUME,
            "Âm Lượng Video Gốc (Audio Ducking)",
            "Chỉnh nhỏ tiếng gốc khi có giọng AI (0.15 = 15% âm lượng ban đầu).",
            configItem=config.mmoAudioDuckingVolume,
            parent=self.ttsGroup
        )
        self.duckingCard.spinBox.setRange(0.0, 1.0)
        self.duckingCard.spinBox.setSingleStep(0.05)
        
        self.ttsGroup.addSettingCard(self.enableTTSCard)
        self.ttsGroup.addSettingCard(self.duckingCard)
        
        self.expandLayout.addWidget(self.ttsGroup)

    def retranslateUi(self):
        """ Cập nhật text khi đổi ngôn ngữ (nếu cần) """
        pass
