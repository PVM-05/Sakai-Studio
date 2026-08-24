import os
import json
from pathlib import Path
from PySide6 import QtWidgets, QtCore
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, RangeSettingCard,
                          PushSettingCard, PrimaryPushSettingCard, OptionsSettingCard,
                          FolderListSettingCard, HyperlinkCard, ColorSettingCard, 
                          CustomColorSettingCard, SettingCard, ComboBox, SwitchButton, 
                          BodyLabel, SubtitleLabel, CaptionLabel, TransparentToolButton)
from PySide6.QtCore import Qt
from src.core.config import config, tr, HARDWARD_ACCELERATION_OPTION
from src.core.tools.constant import InpaintMode, SubtitleDetectMode

class SettingInterface(QtWidgets.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(12)
        
        # Helper to safely map config values to combo index
        def get_idx(val, options):
            try: return list(options).index(val)
            except: return 0
            
        def create_label(text):
            label = BodyLabel(text, parent)
            label.setMinimumHeight(26)
            return label

        # ==========================================
        # Khối 1: Cấu hình Trí tuệ Nhân tạo (AI Config)
        # ==========================================
        ai_card = CardWidget(parent)
        ai_layout = QtWidgets.QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(16, 16, 16, 16)
        ai_layout.setSpacing(12)
        
        ai_title = SubtitleLabel("Cấu hình Trí tuệ Nhân tạo", parent)
        ai_layout.addWidget(ai_title)
        
        ai_form = QtWidgets.QFormLayout()
        ai_form.setContentsMargins(0, 0, 0, 0)
        ai_form.setSpacing(12)
        ai_layout.addLayout(ai_form)

        # Inpaint Mode
        self.inpaint_combo = ComboBox(parent)
        self.inpaint_combo.addItems([list(tr['InpaintMode'].values())[i] for i,_ in enumerate(config.inpaintMode.validator.options)])
        self.inpaint_combo.setCurrentIndex(get_idx(config.inpaintMode.value, config.inpaintMode.validator.options))
        self.inpaint_combo.currentIndexChanged.connect(lambda idx: config.set(config.inpaintMode, list(config.inpaintMode.validator.options)[idx]))
        config.inpaintMode.valueChanged.connect(lambda val: self.inpaint_combo.setCurrentIndex(get_idx(val, config.inpaintMode.validator.options)))
        self.inpaint_combo.setToolTip("Chọn mô hình AI xóa chữ (LaMa, STTN, ProPainter, OpenCV)")
        ai_form.addRow(create_label("Mô hình xóa chữ:"), self.inpaint_combo)

        # Subtitle Detect Mode
        self.subtitle_detect_model_combo = ComboBox(parent)
        self.subtitle_detect_model_combo.addItems([list(tr['SubtitleDetectMode'].values())[i] for i,_ in enumerate(config.subtitleDetectMode.validator.options)])
        self.subtitle_detect_model_combo.setCurrentIndex(get_idx(config.subtitleDetectMode.value, config.subtitleDetectMode.validator.options))
        self.subtitle_detect_model_combo.currentIndexChanged.connect(lambda idx: config.set(config.subtitleDetectMode, list(config.subtitleDetectMode.validator.options)[idx]))
        config.subtitleDetectMode.valueChanged.connect(lambda val: self.subtitle_detect_model_combo.setCurrentIndex(get_idx(val, config.subtitleDetectMode.validator.options)))
        ai_form.addRow(create_label("Mô hình quét OCR:"), self.subtitle_detect_model_combo)

        # OCR Language
        self.ocr_language_combo = ComboBox(parent)
        self.ocr_language_combo.addItems(["Tự động phát hiện", "Tiếng Việt", "Tiếng Anh", "Tiếng Trung", "Tiếng Nhật", "Tiếng Hàn"])
        self.ocr_language_combo.setCurrentIndex(get_idx(config.ocrLanguage.value, config.ocrLanguage.validator.options))
        self.ocr_language_combo.currentIndexChanged.connect(lambda idx: config.set(config.ocrLanguage, list(config.ocrLanguage.validator.options)[idx]))
        config.ocrLanguage.valueChanged.connect(lambda val: self.ocr_language_combo.setCurrentIndex(get_idx(val, config.ocrLanguage.validator.options)))
        ai_form.addRow(create_label("Ngôn ngữ chữ:"), self.ocr_language_combo)

        # OCR Speed
        self.ocr_mode_combo = ComboBox(parent)
        self.ocr_mode_combo.addItems(["Tự động (Cân bằng)", "Nhanh (Bỏ qua nhiều frame)", "Chính xác (Quét kỹ từng frame)"])
        self.ocr_mode_combo.setCurrentIndex(get_idx(config.ocrMode.value, config.ocrMode.validator.options))
        self.ocr_mode_combo.currentIndexChanged.connect(lambda idx: config.set(config.ocrMode, list(config.ocrMode.validator.options)[idx]))
        config.ocrMode.valueChanged.connect(lambda val: self.ocr_mode_combo.setCurrentIndex(get_idx(val, config.ocrMode.validator.options)))
        ai_form.addRow(create_label("Tốc độ quét OCR:"), self.ocr_mode_combo)

        # SAM 2
        self.sam2_switch = SwitchButton(parent)
        self.sam2_switch.setOnText("Bật")
        self.sam2_switch.setOffText("Tắt")
        self.sam2_switch.setChecked(config.sam2Refine.value)
        self.sam2_switch.checkedChanged.connect(lambda checked: config.set(config.sam2Refine, checked))
        config.sam2Refine.valueChanged.connect(self.sam2_switch.setChecked)
        self.sam2_switch.setToolTip("Sử dụng Segment Anything Model 2 để cắt mặt nạ ôm sát viền đối tượng, giảm vùng xóa thừa")
        ai_form.addRow(create_label("Tinh chỉnh bằng SAM 2:"), self.sam2_switch)

        self.addWidget(ai_card)

        # ==========================================
        # Khối 2: Tùy chỉnh Mặt nạ (Mask Settings)
        # ==========================================
        mask_card = CardWidget(parent)
        mask_layout = QtWidgets.QVBoxLayout(mask_card)
        mask_layout.setContentsMargins(16, 16, 16, 16)
        mask_layout.setSpacing(12)
        
        mask_title = SubtitleLabel("Tùy chỉnh Mặt nạ", parent)
        mask_layout.addWidget(mask_title)

        mask_form = QtWidgets.QFormLayout()
        mask_form.setContentsMargins(0, 0, 0, 0)
        mask_form.setSpacing(12)
        mask_layout.addLayout(mask_form)

        # Mask Type
        self.mask_type_combo = ComboBox(parent)
        self.mask_type_combo.addItems(["Nét chữ", "Hộp chữ nhật"])
        self.mask_type_combo.setCurrentIndex(get_idx(config.maskType.value, config.maskType.validator.options))
        self.mask_type_combo.currentIndexChanged.connect(lambda idx: config.set(config.maskType, list(config.maskType.validator.options)[idx]))
        config.maskType.valueChanged.connect(lambda val: self.mask_type_combo.setCurrentIndex(get_idx(val, config.maskType.validator.options)))
        mask_form.addRow(create_label("Kiểu mặt nạ:"), self.mask_type_combo)

        # Auto Tighten
        self.auto_tighten_switch = SwitchButton(parent)
        self.auto_tighten_switch.setOnText("Bật")
        self.auto_tighten_switch.setOffText("Tắt")
        self.auto_tighten_switch.setChecked(config.autoTighten.value)
        self.auto_tighten_switch.checkedChanged.connect(lambda checked: config.set(config.autoTighten, checked))
        config.autoTighten.valueChanged.connect(self.auto_tighten_switch.setChecked)
        self.auto_tighten_switch.setToolTip("Tự động phân tích và co hẹp mặt nạ ôm khít chữ khi vẽ khung")
        mask_form.addRow(create_label("Tự động ôm khít:"), self.auto_tighten_switch)

        # Mask Dilation
        dilation_layout = QtWidgets.QHBoxLayout()
        self.dilation_slider = Slider(Qt.Horizontal, parent)
        self.dilation_slider.setRange(0, 50)
        self.dilation_slider.setValue(config.maskDilation.value)
        self.dilation_label = CaptionLabel(f"{config.maskDilation.value} px")
        self.dilation_slider.valueChanged.connect(lambda val: self.dilation_label.setText(f"{val} px"))
        self.dilation_slider.valueChanged.connect(lambda val: config.set(config.maskDilation, val))
        config.maskDilation.valueChanged.connect(self.dilation_slider.setValue)
        dilation_layout.addWidget(self.dilation_slider)
        dilation_layout.addWidget(self.dilation_label)
        mask_form.addRow(create_label("Giãn nở:"), dilation_layout)

        # Auto Mask Dilation
        self.auto_dilation_switch = SwitchButton(parent)
        self.auto_dilation_switch.setOnText("Bật")
        self.auto_dilation_switch.setOffText("Tắt")
        self.auto_dilation_switch.setChecked(config.autoMaskDilation.value)
        self.auto_dilation_switch.checkedChanged.connect(lambda checked: config.set(config.autoMaskDilation, checked))
        config.autoMaskDilation.valueChanged.connect(self.auto_dilation_switch.setChecked)
        self.auto_dilation_switch.setToolTip("Tự động tính toán độ giãn nở mặt nạ dựa trên chiều cao của chữ/logo")
        mask_form.addRow(create_label("Tự động giãn nở:"), self.auto_dilation_switch)

        # Mask Feather
        feather_layout = QtWidgets.QHBoxLayout()
        self.feather_slider = Slider(Qt.Horizontal, parent)
        self.feather_slider.setRange(0, 30)
        self.feather_slider.setValue(config.maskFeather.value)
        self.feather_label = CaptionLabel(f"{config.maskFeather.value} px")
        self.feather_slider.valueChanged.connect(lambda val: self.feather_label.setText(f"{val} px"))
        self.feather_slider.valueChanged.connect(lambda val: config.set(config.maskFeather, val))
        config.maskFeather.valueChanged.connect(self.feather_slider.setValue)
        feather_layout.addWidget(self.feather_slider)
        feather_layout.addWidget(self.feather_label)
        mask_form.addRow(create_label("Làm mềm:"), feather_layout)

        # Mask Fade Padding
        fade_padding_layout = QtWidgets.QHBoxLayout()
        self.fade_padding_slider = Slider(Qt.Horizontal, parent)
        self.fade_padding_slider.setRange(0, 15)
        self.fade_padding_slider.setValue(config.maskFadePadding.value)
        self.fade_padding_label = CaptionLabel(f"{config.maskFadePadding.value} frames")
        self.fade_padding_slider.valueChanged.connect(lambda val: self.fade_padding_label.setText(f"{val} frames"))
        self.fade_padding_slider.valueChanged.connect(lambda val: config.set(config.maskFadePadding, val))
        config.maskFadePadding.valueChanged.connect(self.fade_padding_slider.setValue)
        fade_padding_layout.addWidget(self.fade_padding_slider)
        fade_padding_layout.addWidget(self.fade_padding_label)
        mask_form.addRow(create_label("Bù trừ khung hình mờ:"), fade_padding_layout)

        self.addWidget(mask_card)

        # ==========================================
        # Khối 3: Công cụ & Tương tác (Tools)
        # ==========================================
        tools_card = CardWidget(parent)
        tools_layout = QtWidgets.QVBoxLayout(tools_card)
        tools_layout.setContentsMargins(16, 16, 16, 16)
        tools_layout.setSpacing(12)
        tools_layout.addWidget(SubtitleLabel("Công cụ & Tương tác", parent))



        # Moving Subtitle Tracking
        self.moving_subtitle_card = PushButton("Bám đuổi phụ đề di chuyển", parent)
        self.moving_subtitle_card.setIcon(FluentIcon.RUN if hasattr(FluentIcon, 'RUN') else FluentIcon.VIDEO)
        self.moving_subtitle_card.setToolTip("Khoanh vùng chữ/logo ở 1 khung hình rồi bấm để bám đuổi")
        tools_layout.addWidget(self.moving_subtitle_card)

        # Brush & Mask Tools
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(8)
        
        robot_icon = FluentIcon.ROBOT if hasattr(FluentIcon, 'ROBOT') else FluentIcon.SEARCH
        brush_icon = FluentIcon.BRUSH if hasattr(FluentIcon, 'BRUSH') else FluentIcon.EDIT
        red_icon = FluentIcon.VIEW if hasattr(FluentIcon, 'VIEW') else FluentIcon.SEARCH
        clear_icon = FluentIcon.DELETE if hasattr(FluentIcon, 'DELETE') else FluentIcon.CLOSE

        self.auto_detect_frame_btn = TransparentToolButton(robot_icon, parent)
        self.auto_detect_frame_btn.setToolTip("Quét OCR 1 khung hình hiện tại")
        
        self.brush_mode_btn = TransparentToolButton(brush_icon, parent)
        self.brush_mode_btn.setToolTip("Chế độ vẽ cọ / Khoanh hộp")
        
        self.red_mask_btn = TransparentToolButton(red_icon, parent)
        self.red_mask_btn.setToolTip("Xem Lớp phủ Đỏ")
        
        self.clear_brush_btn = TransparentToolButton(clear_icon, parent)
        self.clear_brush_btn.setToolTip("Xóa nét cọ")

        btn_layout.addWidget(self.auto_detect_frame_btn)
        btn_layout.addWidget(self.brush_mode_btn)
        btn_layout.addWidget(self.red_mask_btn)
        btn_layout.addWidget(self.clear_brush_btn)
        btn_layout.addStretch()
        
        tools_layout.addLayout(btn_layout)

        self.addWidget(tools_card)

        # ==========================================
        # Khối 4: Cài đặt nâng cao
        # ==========================================
        self.open_advanced_card = PushSettingCard(
            text=tr["Setting"].get("OpenAdvancedSetting", "Mở Cài Đặt"),
            icon=FluentIcon.SETTING,
            title="Cài đặt nâng cao",
            content="Cấu hình chuyên sâu mô hình",
            parent=parent
        )
        self.open_advanced_card.clicked.connect(self._on_open_advanced_settings)
        self.addWidget(self.open_advanced_card)

        self.addStretch(1)

    def _on_open_advanced_settings(self):
        w = self.parentWidget()
        while w and not hasattr(w, 'advancedSettingInterface'):
            w = w.parentWidget()
        if w and hasattr(w, 'advancedSettingInterface'):
            w.switchTo(w.advancedSettingInterface)

    def set_inpaint_mode_enabled(self, enabled):
        self.inpaint_combo.setEnabled(enabled)

    def _on_choose_save_directory(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self.parentWidget(),
            tr["Setting"].get("SaveDirectory", "Chọn Thư Mục Lưu Video"),
            config.saveDirectory.value
        )
        if folder:
            config.set(config.saveDirectory, folder)
            try:
                self.save_directory.setContent(folder)
            except AttributeError:
                pass



    def _on_open_advanced_settings(self):
        """Chuyển sang trang Cài Đặt Nâng Cao khi click vào thẻ cài đặt nâng cao"""
        w = self.parentWidget()
        while w and not hasattr(w, 'advancedSettingInterface'):
            w = w.parentWidget()
        if w and hasattr(w, 'advancedSettingInterface'):
            w.switchTo(w.advancedSettingInterface)



    def set_inpaint_mode_enabled(self, enabled):
        """启用或禁用 inpaint 模式下拉框"""
        self.inpaint_combo.setEnabled(enabled)

    def reset_setting(self):
        """重置所有设置为默认值"""
        # 这里需要实现重置逻辑
        pass

    def update_gpu_info(self):
        """Cập nhật động thông tin GPU và hiệu năng đang chạy dựa trên cài đặt"""
        import torch
        
        # Nếu đang ở chế độ tự động tối ưu hóa hiệu năng
        is_auto = config.autoHardwareTuning.value
        
        if is_auto:
            if not torch.cuda.is_available():
                max_load_pp, max_load_sttn = 15, 20
                performance_tier = "Tự động - CPU Only"
            else:
                try:
                    device_idx = torch.cuda.current_device()
                    total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
                    if total_vram >= 16:
                        max_load_pp, max_load_sttn = 75, 100
                        performance_tier = "Tự động - Cao cấp"
                    elif total_vram >= 10:
                        max_load_pp, max_load_sttn = 55, 70
                        performance_tier = "Tự động - Trung cấp"
                    elif total_vram >= 6:
                        max_load_pp, max_load_sttn = 40, 50
                        performance_tier = "Tự động - Phổ thông"
                    else:
                        max_load_pp, max_load_sttn = 25, 30
                        performance_tier = "Tự động - Thấp"
                except Exception:
                    max_load_pp, max_load_sttn = 50, 50
                    performance_tier = "Tự động - Không xác định"
        else:
            # Nếu người dùng tắt tự động tối ưu hóa, lấy thông số thực tế của thanh trượt
            max_load_pp = config.propainterMaxLoadNum.value
            max_load_sttn = config.sttnMaxLoadNum.value
            performance_tier = "Thủ công"

        # Lấy thông tin thiết bị đồ họa
        gpu_title = "Thiết bị: CPU Only"
        if torch.cuda.is_available():
            try:
                device_idx = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device_idx)
                total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
                gpu_title = f"GPU: {gpu_name} ({total_vram:.1f} GB VRAM)"
            except Exception:
                gpu_title = "GPU CUDA Detected"

        gpu_content = f"Chế độ: {performance_tier} | Cấu hình: ProPainter: {max_load_pp} frames, STTN: {max_load_sttn} frames."
        self.gpu_info_card.setTitle(gpu_title)
        self.gpu_info_card.setContent(gpu_content)

        # Cảnh báo VRAM nếu chạy Tracking + ProPainter trên máy yếu
        is_moving_tracking = getattr(config, 'movingSubtitleTracking', None) and config.movingSubtitleTracking.value
        inpaint_mode = getattr(config, 'inpaintMode', None) and config.inpaintMode.value
        mode_val = getattr(inpaint_mode, 'value', inpaint_mode)
        if torch.cuda.is_available() and is_moving_tracking and mode_val == "propainter":
            try:
                device_idx = torch.cuda.current_device()
                total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
                if total_vram < 6.0:
                    from qfluentwidgets import InfoBar, InfoBarPosition
                    InfoBar.warning(
                        title="Cảnh báo VRAM",
                        content="Bạn đang bật tính năng Xóa phụ đề di chuyển cùng với ProPainter. Máy bạn có VRAM dưới 6GB, có thể gây tràn bộ nhớ. Vui lòng cân nhắc tắt tính năng hoặc giảm MaxLoadNum.",
                        orient=QtCore.Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=5000,
                        parent=self
                    )
            except Exception:
                pass

    def retranslateUi(self):
        """Cập nhật lại văn bản hiển thị trên các SettingCard khi đổi ngôn ngữ nóng"""
        
        # Cập nhật combo inpaint mode (block signals để tránh kích hoạt thay đổi cấu hình)
        self.inpaint_combo.blockSignals(True)
        current_inpaint_idx = self.inpaint_combo.currentIndex()
        self.inpaint_combo.clear()
        self.inpaint_combo.addItems([list(tr['InpaintMode'].values())[i] for i,_ in enumerate(config.inpaintMode.validator.options)])
        self.inpaint_combo.setCurrentIndex(current_inpaint_idx)
        self.inpaint_combo.blockSignals(False)
        self.inpaint_combo.setToolTip(tr["Setting"].get("InpaintModeTooltip", "Select inpaint model"))
        
        # Cập nhật combo subtitle detect mode
        self.subtitle_detect_model_combo.blockSignals(True)
        current_detect_idx = self.subtitle_detect_model_combo.currentIndex()
        self.subtitle_detect_model_combo.clear()
        self.subtitle_detect_model_combo.addItems([list(tr['SubtitleDetectMode'].values())[i] for i,_ in enumerate(config.subtitleDetectMode.validator.options)])
        self.subtitle_detect_model_combo.setCurrentIndex(current_detect_idx)
        self.subtitle_detect_model_combo.blockSignals(False)
        self.subtitle_detect_model_combo.setToolTip(tr["Setting"].get("SubtitleDetectModeTooltip", "Select OCR model"))
        
        # Cập nhật combo mask type
        self.mask_type_combo.blockSignals(True)
        current_mask_idx = self.mask_type_combo.currentIndex()
        self.mask_type_combo.clear()
        self.mask_type_combo.addItems([tr['Setting'].get('MaskTypeStroke', 'Nét chữ'), tr['Setting'].get('MaskTypeBox', 'Hộp chữ nhật')])
        self.mask_type_combo.setCurrentIndex(current_mask_idx)
        self.mask_type_combo.blockSignals(False)
        self.mask_type_combo.setToolTip(tr["Setting"].get("MaskTypeTooltip", "Mask type tooltip"))
        
        # Cập nhật combo OCR Language
        if hasattr(self, 'ocr_language_combo'):
            self.ocr_language_combo.setToolTip(tr["Setting"].get("OcrLanguageTooltip", "Chọn ngôn ngữ OCR"))
            self.ocr_language_combo.blockSignals(True)
            current_ocr_idx = self.ocr_language_combo.currentIndex()
            self.ocr_language_combo.clear()
            self.ocr_language_combo.addItems([list(tr['OcrLanguage'].values())[i] for i, _ in enumerate(config.ocrLanguage.validator.options)])
            self.ocr_language_combo.setCurrentIndex(current_ocr_idx)
            self.ocr_language_combo.blockSignals(False)
        
        self.gpu_info_card.setToolTip(tr["Setting"].get("GpuInfoTooltip", "GPU information"))
        self.update_gpu_info()

        # Cập nhật lại tự động xuống dòng sau khi đổi ngôn ngữ
        # Cải tiến WordWrap cho nhãn cài đặt và Fix chiều cao ComboBox bị ép trong QFormLayout
        for child in self.parentWidget().findChildren(QtWidgets.QWidget) if self.parentWidget() else []:
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)
            if isinstance(child, ComboBox):
                child.setMinimumHeight(34)

    def _on_inpaint_mode_changed(self, idx):
        try:
            import json
            from pathlib import Path
            from src.core.config import config as main_cfg
            main_cfg.save()

            raw_mode = getattr(main_cfg.inpaintMode, 'value', 'sttn_auto')
            if hasattr(raw_mode, 'value'):
                raw_mode = raw_mode.value
            if hasattr(raw_mode, 'value'):
                raw_mode = raw_mode.value
            inpaint_str = str(raw_mode).lower().replace('-', '_')

            if 'lama' in inpaint_str:
                mode_code = 'lama'
            elif 'propainter' in inpaint_str:
                mode_code = 'propainter'
            elif 'opencv' in inpaint_str:
                mode_code = 'opencv'
            else:
                mode_code = 'sttn_auto'

            cfg_file = Path("config") / "auto_pipeline_config.json"
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if cfg_file.exists():
                try:
                    data = json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    data = {}

            data["inpaint_mode"] = mode_code
            cfg_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass