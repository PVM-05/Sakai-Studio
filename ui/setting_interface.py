from PySide6 import QtWidgets
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, RangeSettingCard,
                          PushSettingCard, PrimaryPushSettingCard, OptionsSettingCard,
                          FolderListSettingCard, HyperlinkCard, ColorSettingCard, 
                          CustomColorSettingCard, SettingCard)
from backend.config import config, tr, HARDWARD_ACCELERATION_OPTION
from backend.tools.constant import InpaintMode, SubtitleDetectMode

class SettingInterface(QtWidgets.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        
        # 1. 界面语言设置
        self.interface_combo = ComboBoxSettingCard(
            configItem=config.interface,
            icon=FluentIcon.LANGUAGE,
            title=tr["SubtitleExtractorGUI"]["InterfaceLanguage"],
            content="",
            parent=parent,
            texts=config.intefaceTexts.keys(),
        )
        self.interface_combo.setToolTip("Chọn ngôn ngữ hiển thị cho giao diện phần mềm.")
        self.addWidget(self.interface_combo)
        
        # 2. 处理模式设置 (Mô hình AI xóa chữ)
        self.inpaint_mode_combo = ComboBoxSettingCard(
            configItem=config.inpaintMode,
            icon=FluentIcon.GLOBE,
            title=tr["SubtitleExtractorGUI"]["InpaintMode"],
            content="",
            parent=parent,
            texts=[list(tr['InpaintMode'].values())[i] for i,_ in enumerate(config.inpaintMode.validator.options)],
        )
        self.inpaint_mode_combo.setToolTip(
            "Chọn mô hình trí tuệ nhân tạo để xóa phụ đề:\n"
            "- LaMa: Mô hình tốt nhất cho ảnh tĩnh, hoạt động rất nhanh.\n"
            "- STTN: Mô hình video trung cấp, đảm bảo tính liên kết thời gian tốt.\n"
            "- ProPainter: Mô hình video cao cấp nhất, khử nhấp nháy vượt trội.\n"
            "- OpenCV: Sử dụng thuật toán xử lý ảnh truyền thống, tốc độ cực nhanh nhưng chất lượng cơ bản."
        )
        self.addWidget(self.inpaint_mode_combo)

        # 3. Mô hình OCR phát hiện phụ đề
        self.subtitle_detect_model_combo = ComboBoxSettingCard(
            configItem=config.subtitleDetectMode,
            icon=FluentIcon.SEARCH,
            title=tr["SubtitleExtractorGUI"]["SubtitleDetectMode"],
            content="",
            parent=parent,
            texts=[list(tr['SubtitleDetectMode'].values())[i] for i,_ in enumerate(config.subtitleDetectMode.validator.options)],
        )
        self.subtitle_detect_model_combo.setToolTip(
            "Chọn mô hình OCR phát hiện phụ đề:\n"
            "- Server: Mô hình PP-OCRv5 Server có độ chính xác rất cao, khuyên dùng.\n"
            "- Mobile: Mô hình PP-OCRv5 Mobile dung lượng nhẹ, tốc độ nhanh hơn nhưng dễ bị sót chữ hơn."
        )
        self.addWidget(self.subtitle_detect_model_combo)

        # 4. 选用的 Mask 类型 (Mask Type)
        self.mask_type_combo = ComboBoxSettingCard(
            configItem=config.maskType,
            icon=FluentIcon.BROOM,
            title="Kiểu mặt nạ xóa chữ (Mask Type)",
            content="Chọn phương pháp tạo mặt nạ phụ đề để xóa",
            parent=parent,
            texts=["Nét chữ", "Hộp chữ nhật"]
        )
        self.mask_type_combo.setToolTip(
            "Chọn phương pháp che phủ dòng chữ:\n"
            "- Nét chữ (Stroke): Mặt nạ bám khít theo từng nét vẽ của chữ. Giúp giữ nguyên vẹn tối đa nền gốc xung quanh.\n"
            "- Hộp chữ nhật (Box): Che phủ toàn bộ hộp chữ nhật chứa chữ. Xóa sạch 100% nhưng vùng cần inpaint lớn hơn, dễ gây mờ nền."
        )
        self.addWidget(self.mask_type_combo)



        # 5. GPU / VRAM Info Card
        self.gpu_info_card = SettingCard(
            icon=FluentIcon.INFO,
            title="Thiết bị: Đang quét...",
            content="Đang tối ưu hóa cấu hình hiệu năng...",
            parent=parent
        )
        self.gpu_info_card.setToolTip("Hiển thị thông tin tên GPU đồ họa, dung lượng VRAM thực tế và hạn mức số khung hình được phân bổ tối đa cho việc xử lý đồng thời.")
        self.addWidget(self.gpu_info_card)

        # 7. Thẻ Mở Cài Đặt Nâng Cao
        self.open_advanced_card = PushSettingCard(
            text=tr["Setting"].get("OpenAdvancedSetting", "Mở Cài Đặt"),
            icon=FluentIcon.SETTING,
            title="Cài đặt nâng cao",
            content="Mở toàn bộ tùy chọn cấu hình chi tiết (Tăng tốc GPU, Poisson, Feathering, Mask, Auto-Tighten...)",
            parent=parent
        )
        self.open_advanced_card.clicked.connect(self._on_open_advanced_settings)
        self.addWidget(self.open_advanced_card)

        # Listen to config changes to dynamically update GPU / VRAM card info
        config.autoHardwareTuning.valueChanged.connect(self.update_gpu_info)
        config.propainterMaxLoadNum.valueChanged.connect(self.update_gpu_info)
        config.sttnMaxLoadNum.valueChanged.connect(self.update_gpu_info)
        self.update_gpu_info()

        # Listen to Auto-Tighten changes to disable Mask Type if ON
        if hasattr(config, 'autoTightenMask'):
            config.autoTightenMask.valueChanged.connect(self._on_auto_tighten_changed)
            self._on_auto_tighten_changed(config.autoTightenMask.value)



        # Cho phép các nhãn mô tả tự động xuống dòng
        from PySide6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)

        # 如果硬件加速选项被禁用, 设置硬件加速为False并只读
        if not HARDWARD_ACCELERATION_OPTION and hasattr(self, 'hardware_acceleration'):
            self.hardware_acceleration.switchButton.setChecked(False)
            self.hardware_acceleration.switchButton.setEnabled(False)
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationNO"])
            config.set(config.hardwareAcceleration, False)
        # 添加一些空间
        self.addStretch(1)
    
    def _on_auto_tighten_changed(self, value):
        self.mask_type_combo.setDisabled(value)

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
        self.inpaint_mode_combo.comboBox.setEnabled(enabled)

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

    def retranslateUi(self):
        """Cập nhật lại văn bản hiển thị trên các SettingCard khi đổi ngôn ngữ nóng"""
        self.interface_combo.setTitle(tr["SubtitleExtractorGUI"]["InterfaceLanguage"])
        self.interface_combo.setToolTip(tr["Setting"].get("InterfaceLanguageTooltip", "Select interface language"))
        
        # Cập nhật combo inpaint mode (block signals để tránh kích hoạt thay đổi cấu hình)
        self.inpaint_mode_combo.comboBox.blockSignals(True)
        current_inpaint_idx = self.inpaint_mode_combo.comboBox.currentIndex()
        self.inpaint_mode_combo.setTitle(tr["SubtitleExtractorGUI"]["InpaintMode"])
        self.inpaint_mode_combo.comboBox.clear()
        self.inpaint_mode_combo.comboBox.addItems([list(tr['InpaintMode'].values())[i] for i,_ in enumerate(config.inpaintMode.validator.options)])
        self.inpaint_mode_combo.comboBox.setCurrentIndex(current_inpaint_idx)
        self.inpaint_mode_combo.comboBox.blockSignals(False)
        self.inpaint_mode_combo.setToolTip(tr["Setting"].get("InpaintModeTooltip", "Select inpaint model"))
        
        # Cập nhật combo subtitle detect mode
        self.subtitle_detect_model_combo.comboBox.blockSignals(True)
        current_detect_idx = self.subtitle_detect_model_combo.comboBox.currentIndex()
        self.subtitle_detect_model_combo.setTitle(tr["SubtitleExtractorGUI"]["SubtitleDetectMode"])
        self.subtitle_detect_model_combo.comboBox.clear()
        self.subtitle_detect_model_combo.comboBox.addItems([list(tr['SubtitleDetectMode'].values())[i] for i,_ in enumerate(config.subtitleDetectMode.validator.options)])
        self.subtitle_detect_model_combo.comboBox.setCurrentIndex(current_detect_idx)
        self.subtitle_detect_model_combo.comboBox.blockSignals(False)
        self.subtitle_detect_model_combo.setToolTip(tr["Setting"].get("SubtitleDetectModeTooltip", "Select OCR model"))
        
        # Cập nhật combo mask type
        self.mask_type_combo.comboBox.blockSignals(True)
        current_mask_idx = self.mask_type_combo.comboBox.currentIndex()
        self.mask_type_combo.setTitle(tr["Setting"].get("MaskType", "Kiểu mặt nạ xóa chữ (Mask Type)"))
        self.mask_type_combo.setContent(tr["Setting"].get("MaskTypeDesc", "Chọn phương pháp tạo mặt nạ phụ đề để xóa"))
        self.mask_type_combo.comboBox.clear()
        self.mask_type_combo.comboBox.addItems([tr['Setting'].get('MaskTypeStroke', 'Nét chữ'), tr['Setting'].get('MaskTypeBox', 'Hộp chữ nhật')])
        self.mask_type_combo.comboBox.setCurrentIndex(current_mask_idx)
        self.mask_type_combo.comboBox.blockSignals(False)
        self.mask_type_combo.setToolTip(tr["Setting"].get("MaskTypeTooltip", "Mask type tooltip"))
        
        self.gpu_info_card.setToolTip(tr["Setting"].get("GpuInfoTooltip", "GPU information"))
        self.update_gpu_info()

        # Cập nhật lại tự động xuống dòng sau khi đổi ngôn ngữ
        from PySide6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)