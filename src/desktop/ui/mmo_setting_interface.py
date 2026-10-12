# -*- coding: utf-8 -*-
"""
Cấu hình sản xuất video tự động và lồng tiếng trí tuệ nhân tạo OmniVoice trên GPU.
Tuân thủ nghiêm ngặt quy tắc thiết kế giao diện tiếng Việt chuẩn mực, tự động xuống dòng và không dùng từ tiếng Anh trong ngoặc đơn.
"""

from __future__ import annotations

import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from qfluentwidgets import (
    ScrollArea,
    ExpandLayout,
    SettingCardGroup,
    SwitchSettingCard,
    RangeSettingCard,
    ComboBoxSettingCard,
    ComboBox,
    TitleLabel,
    FluentIcon,
    SettingCard,
    DoubleSpinBox,
    PushSettingCard,
    LineEdit,
    PrimaryPushButton,
    InfoBar,
)
from src.core.config import config, tr
from src.ai_engines.tts_engine import TTSEngine


class DoubleSpinBoxSettingCard(SettingCard):
    """Thẻ cài đặt tùy chỉnh sử dụng ô nhập số thực."""

    def __init__(self, icon, title, content, configItem=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.spinBox = DoubleSpinBox(self)
        self.spinBox.setMinimumWidth(130)
        self.spinBox.setRange(0.1, 5.0)
        self.spinBox.setSingleStep(0.05)
        self.spinBox.setDecimals(2)
        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)

        if self.configItem:
            self.spinBox.setValue(float(self.configItem.value))
            self.spinBox.valueChanged.connect(self._onValueChanged)

    def _onValueChanged(self, value):
        if self.configItem:
            config.set(self.configItem, float(value))


class LineEditSettingCard(SettingCard):
    """Thẻ cài đặt tùy chỉnh sử dụng ô nhập văn bản một dòng."""

    def __init__(self, icon, title, content, placeholder="", configItem=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.lineEdit = LineEdit(self)
        self.lineEdit.setMinimumWidth(260)
        self.lineEdit.setPlaceholderText(placeholder)
        self.lineEdit.setClearButtonEnabled(True)
        self.hBoxLayout.addWidget(self.lineEdit, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)

        if self.configItem:
            self.lineEdit.setText(str(self.configItem.value or ""))
            self.lineEdit.textChanged.connect(self._onTextChanged)

    def _onTextChanged(self, text: str):
        if self.configItem:
            config.set(self.configItem, text.strip())


class TTSPreviewCard(SettingCard):
    """Thẻ chọn giọng đọc mẫu tiếng Việt và nghe thử trực tiếp trên GPU."""

    previewRequested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(
            FluentIcon.PLAY,
            "Chọn Giọng Đọc Mẫu Và Nghe Thử Tiếng Việt",
            "Chọn một trong 22 giọng đọc mẫu tiếng Việt hoặc dùng cấu hình hiện tại để tổng hợp và nghe thử trực tiếp trên card đồ họa.",
            parent,
        )
        self.voicePresetCombo = ComboBox(self)
        self.voicePresetCombo.setMinimumWidth(230)
        self.voicePresetCombo.addItem("Theo cấu hình hiện tại")
        for label in TTSEngine.VIETNAMESE_VOICE_CATALOG.keys():
            self.voicePresetCombo.addItem(label)
        self.voicePresetCombo.currentTextChanged.connect(self._onPresetChanged)

        self.previewEdit = LineEdit(self)
        self.previewEdit.setMinimumWidth(240)
        self.previewEdit.setPlaceholderText("Nhập câu tiếng Việt cần nghe thử...")
        self.previewEdit.setText("Xin chào, đây là giọng lồng tiếng tự nhiên được tổng hợp trực tiếp trên card đồ họa.")

        self.previewBtn = PrimaryPushButton("Nghe Thử Tiếng Việt", self)
        self.previewBtn.setIcon(FluentIcon.VOLUME)
        self.previewBtn.clicked.connect(self._onPreviewClicked)

        self.hBoxLayout.addWidget(self.voicePresetCombo, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(8)
        self.hBoxLayout.addWidget(self.previewEdit, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(8)
        self.hBoxLayout.addWidget(self.previewBtn, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)

    def _onPresetChanged(self, label: str):
        preset = TTSEngine.VIETNAMESE_VOICE_CATALOG.get(label)
        if preset and preset.get("sample_vi"):
            self.previewEdit.setText(preset["sample_vi"])
            if preset.get("instruct"):
                config.set(config.mmoTTSInstructCustom, preset["instruct"])

    def _onPreviewClicked(self):
        label = self.voicePresetCombo.currentText()
        preset = TTSEngine.VIETNAMESE_VOICE_CATALOG.get(label)
        instruct = preset.get("instruct", "") if preset else ""
        self.previewRequested.emit(self.previewEdit.text().strip(), instruct)


class TTSPreviewWorker(QThread):
    """Luồng chạy nền tổng hợp âm thanh nghe thử tiếng Việt hoặc lưu mẫu giọng nói .pt."""

    finishedSignal = Signal(bool, str, str)

    def __init__(self, action: str = "preview", text: str = "", instruct: str = "", save_path: str = "", parent=None):
        super().__init__(parent=parent)
        self.action = action
        self.text = text
        self.instruct = instruct
        self.save_path = save_path

    def run(self):
        try:
            engine = TTSEngine.from_app_config()
            if self.action == "save_prompt":
                ok = engine.save_voice_prompt(self.save_path)
                engine.unload_model()
                if ok:
                    self.finishedSignal.emit(True, self.save_path, "Đã trích xuất và lưu tệp đặc trưng giọng mẫu thành công.")
                else:
                    self.finishedSignal.emit(False, "", "Không thể tạo đặc trưng giọng mẫu. Vui lòng chọn tệp âm thanh giọng mẫu hợp lệ trước.")
            else:
                if self.instruct:
                    engine.tts_mode = "design"
                    engine.instruct_custom = self.instruct
                    engine.language_id = "vi"
                elif not engine.language_id or engine.language_id == "auto":
                    engine.language_id = "vi"
                out_wav = engine.synthesize_preview(self.text)
                engine.unload_model()
                if out_wav and os.path.exists(out_wav):
                    self.finishedSignal.emit(True, out_wav, "Đã tổng hợp giọng đọc tiếng Việt nghe thử thành công.")
                else:
                    self.finishedSignal.emit(False, "", "Không thể tạo tệp âm thanh nghe thử.")
        except Exception as e:
            self.finishedSignal.emit(False, "", str(e))


class MMOSettingInterface(ScrollArea):
    """Giao diện cấu hình chống bản quyền và lồng tiếng trí tuệ nhân tạo."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)
        self._preview_worker: TTSPreviewWorker | None = None
        self._audio_output = QAudioOutput(self)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)

        self.setObjectName("MMOSettingInterface")

        self._init_ui()
        self._init_settings()
        self._apply_word_wrap()

    def _init_ui(self):
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.scrollWidget.setObjectName("scrollWidget")
        self.expandLayout.setSpacing(20)
        self.expandLayout.setContentsMargins(28, 20, 28, 28)

        self.title = TitleLabel("Cấu Hình Sản Xuất Video Và Lồng Tiếng", self)
        self.expandLayout.addWidget(self.title)

    def _init_settings(self):
        # =====================================================================
        # NHÓM 1: BIẾN ĐỔI KHUNG HÌNH & PHỤ ĐỀ CHỐNG BẢN QUYỀN
        # =====================================================================
        self.mmoGroup = SettingCardGroup("Biến Đổi Video Chống Bản Quyền", self.scrollWidget)

        self.flipCard = SwitchSettingCard(
            FluentIcon.SYNC,
            "Lật Ngang Khung Hình Video",
            "Tự động lật ngược video theo chiều ngang để thay đổi bố cục hình ảnh gốc.",
            configItem=config.mmoHorizontalFlip,
            parent=self.mmoGroup,
        )

        self.speedCard = DoubleSpinBoxSettingCard(
            FluentIcon.SPEED_OFF,
            "Tốc Độ Phát Video Và Âm Thanh",
            "Tăng hoặc giảm nhẹ tốc độ phát (ví dụ 1.05 hoặc 0.95) để thay đổi mã nhận diện thời gian.",
            configItem=config.mmoSpeedMultiplier,
            parent=self.mmoGroup,
        )

        self.cropCard = RangeSettingCard(
            config.mmoCropPercentage,
            FluentIcon.CUT,
            "Tỷ Lệ Cắt Viền Khung Hình",
            "Cắt bớt phần trăm viền xung quanh khung hình từ 1% đến 50% để tránh trùng khớp khung hình.",
            self.mmoGroup,
        )

        self.subStyleCard = ComboBoxSettingCard(
            config.mmoSubStyle,
            FluentIcon.FONT,
            "Phong Cách Phụ Đề Chèn Lại",
            "Định dạng màu sắc và viền chữ nổi bật dành cho các nền tảng video ngắn.",
            texts=["Phong cách Vàng Nổi Bật", "Phong cách Trắng Viền Đen", "Phong cách Điện Ảnh Cổ Điển", "Tùy Chỉnh"],
            parent=self.mmoGroup,
        )

        self.smartMaskCard = SwitchSettingCard(
            FluentIcon.EDIT,
            "Mặt Nạ Bám Sát Nét Chữ",
            "Chỉ xóa đúng các điểm ảnh thuộc nét chữ phụ đề thay vì xóa cả khung chữ nhật, giữ lại tối đa chi tiết nền.",
            configItem=config.mmoSmartTextMask,
            parent=self.mmoGroup,
        )

        self.mmoGroup.addSettingCard(self.flipCard)
        self.mmoGroup.addSettingCard(self.speedCard)
        self.mmoGroup.addSettingCard(self.cropCard)
        self.mmoGroup.addSettingCard(self.subStyleCard)
        self.mmoGroup.addSettingCard(self.smartMaskCard)
        self.expandLayout.addWidget(self.mmoGroup)

        # =====================================================================
        # NHÓM 2: CẤU HÌNH CHÍNH LỒNG TIẾNG TRÍ TUỆ NHÂN TẠO
        # =====================================================================
        self.ttsGroup = SettingCardGroup("Lồng Tiếng Trí Tuệ Nhân Tạo Trên Card Đồ Họa", self.scrollWidget)

        self.enableTTSCard = SwitchSettingCard(
            FluentIcon.MICROPHONE,
            "Bật Lồng Tiếng Tự Động",
            "Tổng hợp giọng nói đa ngôn ngữ trực tiếp trên card đồ họa bằng mô hình khuếch tán và tự động hòa trộn vào video.",
            configItem=config.mmoEnableTTS,
            parent=self.ttsGroup,
        )

        self.ttsModeCard = ComboBoxSettingCard(
            config.mmoTTSMode,
            FluentIcon.PEOPLE,
            "Chế Độ Tạo Giọng Đọc",
            "Chọn nhân bản theo giọng mẫu, thiết kế giọng mới theo thuộc tính nhân vật, hoặc để mô hình tự chọn giọng.",
            texts=["Nhân Bản Giọng Nói Từ Mẫu", "Thiết Kế Giọng Nói Theo Thuộc Tính", "Giọng Đọc Tự Động"],
            parent=self.ttsGroup,
        )

        self.duckingCard = DoubleSpinBoxSettingCard(
            FluentIcon.VOLUME,
            "Âm Lượng Âm Thanh Gốc Khi Lồng Tiếng",
            "Hạ nhỏ âm lượng gốc của video để làm nổi bật giọng đọc mới (0.15 tương ứng 15% âm lượng ban đầu).",
            configItem=config.mmoAudioDuckingVolume,
            parent=self.ttsGroup,
        )
        self.duckingCard.spinBox.setRange(0.0, 1.0)
        self.duckingCard.spinBox.setSingleStep(0.05)

        self.ttsSpeedCard = DoubleSpinBoxSettingCard(
            FluentIcon.SPEED_OFF,
            "Tốc Độ Phát Ngôn",
            "Điều chỉnh tốc độ đọc câu phụ đề từ 0.5 đến 2.0 (1.0 là tốc độ tiêu chuẩn).",
            configItem=config.mmoTTSSpeed,
            parent=self.ttsGroup,
        )
        self.ttsSpeedCard.spinBox.setRange(0.5, 2.0)
        self.ttsSpeedCard.spinBox.setSingleStep(0.05)

        self.syncDurationCard = SwitchSettingCard(
            FluentIcon.HISTORY,
            "Tự Động Khớp Thời Lượng Phụ Đề",
            "Tự động điều chỉnh thời lượng phát âm của từng câu khớp chính xác với khoảng thời gian hiển thị phụ đề.",
            configItem=config.mmoTTSSyncDuration,
            parent=self.ttsGroup,
        )

        self.ttsGroup.addSettingCard(self.enableTTSCard)
        self.ttsGroup.addSettingCard(self.ttsModeCard)
        self.ttsGroup.addSettingCard(self.duckingCard)
        self.ttsGroup.addSettingCard(self.ttsSpeedCard)
        self.ttsGroup.addSettingCard(self.syncDurationCard)
        self.expandLayout.addWidget(self.ttsGroup)

        # =====================================================================
        # NHÓM 3: NHÂN BẢN GIỌNG NÓI (VOICE CLONING)
        # =====================================================================
        self.cloneGroup = SettingCardGroup("Thiết Lập Nhân Bản Giọng Nói", self.scrollWidget)

        self.autoExtractRefCard = SwitchSettingCard(
            FluentIcon.VIDEO,
            "Tự Động Trích Xuất Giọng Mẫu Từ Video Nguồn",
            "Tự động tách nhạc nền và lấy 8 giây giọng nói rõ nhất từ chính video đầu vào để nhân bản giọng gốc.",
            configItem=config.mmoTTSAutoExtractRef,
            parent=self.cloneGroup,
        )

        ref_path = config.mmoTTSRefAudioPath.value
        self.refAudioCard = PushSettingCard(
            "Chọn Tệp Giọng Mẫu",
            FluentIcon.MUSIC,
            "Tệp Âm Thanh Hoặc Đặc Trưng Giọng Mẫu",
            ref_path if ref_path else "Chọn tệp âm thanh 3 đến 10 giây hoặc tệp đặc trưng giọng đã lưu để nhân bản.",
            parent=self.cloneGroup,
        )
        self.refAudioCard.clicked.connect(self._choose_ref_audio)

        self.refTextCard = LineEditSettingCard(
            FluentIcon.DOCUMENT,
            "Nội Dung Văn Bản Của Giọng Mẫu",
            "Nhập lời thoại của tệp giọng mẫu để tăng độ chính xác. Nếu để trống, hệ thống sẽ tự động nhận diện.",
            placeholder="Để trống để tự động nhận diện lời thoại giọng mẫu...",
            configItem=config.mmoTTSRefText,
            parent=self.cloneGroup,
        )

        self.savePromptCard = PushSettingCard(
            "Lưu Đặc Trưng Giọng",
            FluentIcon.SAVE,
            "Lưu Mẫu Giọng Đã Mã Hóa Để Tái Sử Dụng",
            "Trích xuất đặc trưng từ tệp âm thanh mẫu và lưu thành tệp riêng giúp các lần chạy sau nạp tức thì.",
            parent=self.cloneGroup,
        )
        self.savePromptCard.clicked.connect(self._save_voice_prompt_dialog)

        self.cloneGroup.addSettingCard(self.autoExtractRefCard)
        self.cloneGroup.addSettingCard(self.refAudioCard)
        self.cloneGroup.addSettingCard(self.refTextCard)
        self.cloneGroup.addSettingCard(self.savePromptCard)
        self.expandLayout.addWidget(self.cloneGroup)

        # =====================================================================
        # NHÓM 4: THIẾT KẾ GIỌNG NÓI THEO THUỘC TÍNH (VOICE DESIGN)
        # =====================================================================
        self.designGroup = SettingCardGroup("Thiết Kế Giọng Nói Theo Thuộc Tính", self.scrollWidget)

        self.genderCard = ComboBoxSettingCard(
            config.mmoTTSGender,
            FluentIcon.PEOPLE,
            "Giới Tính Giọng Đọc",
            "Chỉ định giới tính cho giọng đọc được thiết kế tự động.",
            texts=["Tự động", "Giọng Nam", "Giọng Nữ"],
            parent=self.designGroup,
        )

        self.ageCard = ComboBoxSettingCard(
            config.mmoTTSAge,
            FluentIcon.CALENDAR,
            "Độ Tuổi Giọng Đọc",
            "Chọn độ tuổi mong muốn của người phát ngôn.",
            texts=["Tự động", "Trẻ em", "Thiếu niên", "Thanh niên", "Trung niên", "Cao tuổi"],
            parent=self.designGroup,
        )

        self.pitchCard = ComboBoxSettingCard(
            config.mmoTTSPitch,
            FluentIcon.VOLUME,
            "Cao Độ Giọng Đọc",
            "Điều chỉnh độ trầm bổng của âm sắc giọng nói.",
            texts=["Tự động", "Rất trầm", "Trầm", "Trung bình", "Cao", "Rất cao"],
            parent=self.designGroup,
        )

        self.styleCard = ComboBoxSettingCard(
            config.mmoTTSStyle,
            FluentIcon.MICROPHONE,
            "Phong Cách Phát Âm",
            "Chọn phong cách phát ngôn bình thường hoặc giọng thì thầm.",
            texts=["Bình thường", "Giọng thì thầm"],
            parent=self.designGroup,
        )

        self.customInstructCard = LineEditSettingCard(
            FluentIcon.EDIT,
            "Thuộc Tính Giọng Bổ Sung",
            "Nhập thêm thuộc tính khẩu âm hoặc phương ngữ cách nhau bằng dấu phẩy nếu cần.",
            placeholder="Nhập thuộc tính bổ sung...",
            configItem=config.mmoTTSInstructCustom,
            parent=self.designGroup,
        )

        self.designGroup.addSettingCard(self.genderCard)
        self.designGroup.addSettingCard(self.ageCard)
        self.designGroup.addSettingCard(self.pitchCard)
        self.designGroup.addSettingCard(self.styleCard)
        self.designGroup.addSettingCard(self.customInstructCard)
        self.expandLayout.addWidget(self.designGroup)

        # =====================================================================
        # NHÓM 5: THAM SỐ KHUẾCH TÁN CHUYÊN SÂU & NGHE THỬ
        # =====================================================================
        self.advancedTTSGroup = SettingCardGroup("Tham Số Giải Mã Khuếch Tán Và Kiểm Thử", self.scrollWidget)

        self.numStepCard = RangeSettingCard(
            config.mmoTTSNumStep,
            FluentIcon.SPEED_HIGH,
            "Số Bước Giải Mã Khuếch Tán",
            "Số bước lặp mở mặt nạ âm thanh từ 4 đến 64 (mặc định 32 cho chất lượng cao, giảm xuống 16 để tăng gấp đôi tốc độ).",
            self.advancedTTSGroup,
        )

        self.guidanceCard = DoubleSpinBoxSettingCard(
            FluentIcon.ALIGNMENT,
            "Hệ Số Dẫn Hướng Phân Loại",
            "Mức độ bám sát đặc trưng giọng mẫu và thuộc tính từ 0.0 đến 4.0 (mặc định 2.0).",
            configItem=config.mmoTTSGuidanceScale,
            parent=self.advancedTTSGroup,
        )
        self.guidanceCard.spinBox.setRange(0.0, 4.0)
        self.guidanceCard.spinBox.setSingleStep(0.1)

        self.denoiseCard = SwitchSettingCard(
            FluentIcon.BROOM,
            "Khử Nhiễu Âm Thanh Khi Tổng Hợp",
            "Kích hoạt thẻ khử nhiễu nội bộ của mô hình để tạo ra giọng nói trong trẻo hơn.",
            configItem=config.mmoTTSDenoise,
            parent=self.advancedTTSGroup,
        )

        self.preprocessPromptCard = SwitchSettingCard(
            FluentIcon.CUT,
            "Tiền Xử Lý Cắt Khoảng Lặng Giọng Mẫu",
            "Tự động loại bỏ các đoạn im lặng thừa trong tệp âm thanh mẫu và chuẩn hóa dấu câu trước khi nhân bản.",
            configItem=config.mmoTTSPreprocessPrompt,
            parent=self.advancedTTSGroup,
        )

        self.postprocessOutputCard = SwitchSettingCard(
            FluentIcon.COMPLETED,
            "Hậu Xử Lý Làm Mượt Âm Thanh Đầu Ra",
            "Tự động cắt khoảng lặng dài và làm mượt biên độ đầu cuối để tránh tiếng nổ lách tách khi ghép câu.",
            configItem=config.mmoTTSPostprocessOutput,
            parent=self.advancedTTSGroup,
        )

        self.normalizeTextCard = SwitchSettingCard(
            FluentIcon.FONT,
            "Chuẩn Hóa Chữ Số Và Ngày Tháng",
            "Tự động chuyển đổi các con số trong câu phụ đề sang dạng chữ đọc tương ứng trước khi phát âm.",
            configItem=config.mmoTTSNormalizeText,
            parent=self.advancedTTSGroup,
        )

        self.flashInferCard = SwitchSettingCard(
            FluentIcon.DEVELOPER_TOOLS,
            "Tăng Tốc Đồ Họa Chuyên Sâu",
            "Sử dụng nhân tính toán tối ưu và đồ thị phần cứng trên card đồ họa để tăng tốc độ tổng hợp lên gấp 2 lần.",
            configItem=config.mmoTTSFlashInfer,
            parent=self.advancedTTSGroup,
        )

        self.previewCard = TTSPreviewCard(self.advancedTTSGroup)
        self.previewCard.previewRequested.connect(self._run_preview_synthesis)

        self.advancedTTSGroup.addSettingCard(self.numStepCard)
        self.advancedTTSGroup.addSettingCard(self.guidanceCard)
        self.advancedTTSGroup.addSettingCard(self.denoiseCard)
        self.advancedTTSGroup.addSettingCard(self.preprocessPromptCard)
        self.advancedTTSGroup.addSettingCard(self.postprocessOutputCard)
        self.advancedTTSGroup.addSettingCard(self.normalizeTextCard)
        self.advancedTTSGroup.addSettingCard(self.flashInferCard)
        self.advancedTTSGroup.addSettingCard(self.previewCard)
        self.expandLayout.addWidget(self.advancedTTSGroup)

        # Kết nối tín hiệu tự động khóa/mở khóa các công cụ liên quan
        self.enableTTSCard.checkedChanged.connect(self._update_tts_states)
        self.autoExtractRefCard.checkedChanged.connect(self._update_tts_states)
        self.syncDurationCard.checkedChanged.connect(self._on_sync_duration_changed)
        self.ttsModeCard.comboBox.currentIndexChanged.connect(self._update_tts_states)
        self._update_tts_states()

    def _apply_word_wrap(self):
        """Đảm bảo tất cả các thẻ cài đặt tự động xuống dòng và điều chỉnh chiều cao theo độ dài văn bản."""
        for card in self.findChildren(SettingCard):
            if hasattr(card, "titleLabel"):
                card.titleLabel.setWordWrap(True)
            if hasattr(card, "contentLabel"):
                card.contentLabel.setWordWrap(True)
            card.setMinimumHeight(72)

    def _choose_ref_audio(self):
        """Mở hộp thoại chọn tệp âm thanh hoặc tệp đặc trưng giọng mẫu tùy chỉnh."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn Tệp Giọng Mẫu Hoặc Đặc Trưng Giọng",
            "",
            "Tệp Giọng Mẫu (*.wav *.mp3 *.m4a *.flac *.ogg *.pt);;Tất cả tệp (*.*)",
        )
        if file_path:
            config.set(config.mmoTTSRefAudioPath, file_path)
            self.refAudioCard.setContent(file_path)
            InfoBar.success("Đã chọn giọng mẫu", os.path.basename(file_path), parent=self, duration=2500)

    def _save_voice_prompt_dialog(self):
        """Lưu đặc trưng giọng mẫu hiện tại ra tệp .pt để tái sử dụng nhanh."""
        ref_path = config.mmoTTSRefAudioPath.value
        if not ref_path or not os.path.exists(ref_path):
            InfoBar.warning(
                "Chưa có tệp giọng mẫu",
                "Vui lòng chọn tệp âm thanh giọng mẫu trước khi lưu đặc trưng giọng.",
                parent=self,
                duration=3000,
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu Tệp Đặc Trưng Giọng Mẫu",
            os.path.splitext(os.path.basename(ref_path))[0] + ".pt",
            "Tệp Đặc Trưng Giọng (*.pt)",
        )
        if not save_path:
            return

        self.savePromptCard.button.setEnabled(False)
        InfoBar.info("Đang xử lý", "Đang mã hóa đặc trưng giọng mẫu trên card đồ họa...", parent=self, duration=3000)

        self._preview_worker = TTSPreviewWorker(action="save_prompt", save_path=save_path, parent=self)
        self._preview_worker.finishedSignal.connect(self._on_worker_finished)
        self._preview_worker.start()

    def _run_preview_synthesis(self, text: str, instruct: str = ""):
        """Chạy tổng hợp âm thanh nghe thử tiếng Việt trên luồng nền."""
        if self._preview_worker and self._preview_worker.isRunning():
            return

        if instruct and hasattr(self, "customInstructCard"):
            self.customInstructCard.lineEdit.setText(instruct)

        self.previewCard.previewBtn.setEnabled(False)
        self.previewCard.previewBtn.setText("Đang Tổng Hợp...")
        InfoBar.info("Đang tổng hợp giọng đọc tiếng Việt", "Hệ thống đang chạy mô hình trên card đồ họa, vui lòng đợi...", parent=self, duration=3000)

        self._preview_worker = TTSPreviewWorker(action="preview", text=text, instruct=instruct, parent=self)
        self._preview_worker.finishedSignal.connect(self._on_worker_finished)
        self._preview_worker.start()

    def _on_worker_finished(self, success: bool, file_path: str, message: str):
        self.previewCard.previewBtn.setEnabled(True)
        self.previewCard.previewBtn.setText("Nghe Thử Tiếng Việt")
        self.savePromptCard.button.setEnabled(True)

        if success:
            InfoBar.success("Hoàn thành", message, parent=self, duration=3500)
            if file_path.lower().endswith(".wav") and os.path.exists(file_path):
                self._media_player.stop()
                self._media_player.setSource(QUrl.fromLocalFile(file_path))
                self._audio_output.setVolume(1.0)
                self._media_player.play()
        else:
            InfoBar.error("Lỗi xử lý giọng nói", message, parent=self, duration=5000)

    def _update_tts_states(self, *args):
        """
        Khi một tính năng tự động được bật, các công cụ điều chỉnh thủ công liên quan tự động vô hiệu hóa để tránh xung đột.
        """
        tts_enabled = self.enableTTSCard.isChecked()
        mode = config.mmoTTSMode.value

        # Nhóm cài đặt chung
        self.ttsModeCard.setEnabled(tts_enabled)
        self.duckingCard.setEnabled(tts_enabled)
        self.ttsSpeedCard.setEnabled(tts_enabled and not self.syncDurationCard.isChecked())
        self.syncDurationCard.setEnabled(tts_enabled)

        # Nhóm nhân bản giọng nói (chỉ bật khi chế độ là "clone")
        is_clone_mode = tts_enabled and (mode == "clone")
        self.autoExtractRefCard.setEnabled(is_clone_mode)
        auto_extract = self.autoExtractRefCard.isChecked()
        self.refAudioCard.setEnabled(is_clone_mode and not auto_extract)
        self.refTextCard.setEnabled(is_clone_mode and not auto_extract)
        self.savePromptCard.setEnabled(is_clone_mode and not auto_extract)

        # Nhóm thiết kế giọng nói (chỉ bật khi chế độ là "design")
        is_design_mode = tts_enabled and (mode == "design")
        self.genderCard.setEnabled(is_design_mode)
        self.ageCard.setEnabled(is_design_mode)
        self.pitchCard.setEnabled(is_design_mode)
        self.styleCard.setEnabled(is_design_mode)
        self.customInstructCard.setEnabled(is_design_mode)

        # Nhóm tham số nâng cao
        for card in [
            self.numStepCard,
            self.guidanceCard,
            self.denoiseCard,
            self.preprocessPromptCard,
            self.postprocessOutputCard,
            self.normalizeTextCard,
            self.flashInferCard,
            self.previewCard,
        ]:
            card.setEnabled(tts_enabled)

    def _on_sync_duration_changed(self, checked: bool):
        self.ttsSpeedCard.setEnabled(self.enableTTSCard.isChecked() and not checked)

    def retranslateUi(self):
        """Cập nhật văn bản khi đổi ngôn ngữ."""
        pass
