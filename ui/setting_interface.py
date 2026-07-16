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
        
        # 界面语言设置
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
        
        # 处理模式设置
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

        # 是否启用硬件加速
        self.hardware_acceleration = SwitchSettingCard(
            configItem=config.hardwareAcceleration,
            icon=FluentIcon.SPEED_HIGH, 
            title=tr["Setting"]["HardwareAcceleration"],
            content=tr["Setting"]["HardwareAccelerationDesc"],
            parent=parent
        )
        self.hardware_acceleration.setToolTip("Bật hoặc Tắt tăng tốc đồ họa phần cứng GPU (CUDA hoặc DirectML).")
        self.addWidget(self.hardware_acceleration)

        # 是否启用 Poisson Blending
        self.poisson_blending = SwitchSettingCard(
            configItem=config.poissonBlending,
            icon=FluentIcon.BRUSH, 
            title="Poisson Blending (泊松融合)",
            content="Hòa trộn mượt mà bằng thuật toán Poisson Blending",
            parent=parent
        )
        self.poisson_blending.setToolTip("Sử dụng thuật toán Poisson Blending để hòa trộn mượt mà biên giao thoa giữa vùng được xóa và video gốc, loại bỏ vệt cắt răng cưa.")
        self.addWidget(self.poisson_blending)

        # 是否启用 Temporal Smoothing
        self.temporal_smoothing = SwitchSettingCard(
            configItem=config.temporalSmoothing,
            icon=FluentIcon.MOVIE, 
            title="Temporal Smoothing (Lọc mượt thời gian)",
            content="Khử nhấp nháy, rung hạt nền bằng bộ lọc thích ứng chuyển động",
            parent=parent
        )
        self.temporal_smoothing.setToolTip("Khử hiện tượng nhấp nháy hoặc rung hạt nhiễu (flickering) ở vùng inpaint bằng cách nội suy trung bình trọng số thích ứng chuyển động với các khung hình lân cận.")
        self.addWidget(self.temporal_smoothing)

        # Whether to sharpen inpainted area
        self.sharpen_inpainted_area = SwitchSettingCard(
            configItem=config.sharpenInpaintedArea,
            icon=FluentIcon.EDIT,
            title="Làm nét vùng xóa (Sharpen Inpainted Area)",
            content="Làm nét nhẹ vùng nền sau khi xóa phụ đề",
            parent=parent
        )
        self.sharpen_inpainted_area.setToolTip("Áp dụng bộ lọc Unsharp Mask làm nét cục bộ vùng ảnh sau khi xóa để bù đắp lại các chi tiết bị mờ do quá trình nội suy AI tạo ra.")
        self.addWidget(self.sharpen_inpainted_area)

        # 选用的 Mask 类型 (Mask Type)
        self.mask_type_combo = ComboBoxSettingCard(
            configItem=config.maskType,
            icon=FluentIcon.BROOM,
            title="Kiểu mặt nạ xóa chữ (Mask Type)",
            content="Chọn phương pháp tạo mặt nạ phụ đề để xóa",
            parent=parent,
            texts=["Nét chữ (Stroke - Sạch nhất, không ghost)", "Hộp chữ nhật (Box - Kiểu cũ)"]
        )
        self.mask_type_combo.setToolTip(
            "Chọn phương pháp che phủ dòng chữ:\n"
            "- Nét chữ (Stroke): Mặt nạ bám khít theo từng nét vẽ của chữ. Giúp giữ nguyên vẹn tối đa nền gốc xung quanh.\n"
            "- Hộp chữ nhật (Box): Che phủ toàn bộ hộp chữ nhật chứa chữ. Xóa sạch 100% nhưng vùng cần inpaint lớn hơn, dễ gây mờ nền."
        )
        self.addWidget(self.mask_type_combo)

        # GPU / VRAM Info Card
        self.gpu_info_card = SettingCard(
            icon=FluentIcon.INFO,
            title="Thiết bị: Đang quét...",
            content="Đang tối ưu hóa cấu hình hiệu năng...",
            parent=parent
        )
        self.gpu_info_card.setToolTip("Hiển thị thông tin tên GPU đồ họa, dung lượng VRAM thực tế và hạn mức số khung hình được phân bổ tối đa cho việc xử lý đồng thời.")
        self.addWidget(self.gpu_info_card)

        # Listen to config changes to dynamically update GPU / VRAM card info
        config.autoHardwareTuning.valueChanged.connect(self.update_gpu_info)
        config.propainterMaxLoadNum.valueChanged.connect(self.update_gpu_info)
        config.sttnMaxLoadNum.valueChanged.connect(self.update_gpu_info)
        self.update_gpu_info()

        # 如果硬件加速选项被禁用, 设置硬件加速为False并只读
        if not HARDWARD_ACCELERATION_OPTION:
            self.hardware_acceleration.switchButton.setChecked(False)
            self.hardware_acceleration.switchButton.setEnabled(False)
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationNO"])
            config.set(config.hardwareAcceleration, False)
        # 添加一些空间
        self.addStretch(1)
    
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
                        performance_tier = "Tự động - Thấp (Low VRAM)"
                except Exception:
                    max_load_pp, max_load_sttn = 50, 50
                    performance_tier = "Tự động - Không xác định"
        else:
            # Nếu người dùng tắt tự động tối ưu hóa, lấy thông số thực tế của thanh trượt
            max_load_pp = config.propainterMaxLoadNum.value
            max_load_sttn = config.sttnMaxLoadNum.value
            performance_tier = "Thủ công (Manual)"

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
        self.interface_combo.setToolTip("Chọn ngôn ngữ hiển thị cho giao diện phần mềm.")
        
        # Cập nhật combo inpaint mode (block signals để tránh kích hoạt thay đổi cấu hình)
        self.inpaint_mode_combo.comboBox.blockSignals(True)
        current_inpaint_idx = self.inpaint_mode_combo.comboBox.currentIndex()
        self.inpaint_mode_combo.setTitle(tr["SubtitleExtractorGUI"]["InpaintMode"])
        self.inpaint_mode_combo.comboBox.clear()
        self.inpaint_mode_combo.comboBox.addItems([list(tr['InpaintMode'].values())[i] for i,_ in enumerate(config.inpaintMode.validator.options)])
        self.inpaint_mode_combo.comboBox.setCurrentIndex(current_inpaint_idx)
        self.inpaint_mode_combo.comboBox.blockSignals(False)
        self.inpaint_mode_combo.setToolTip(
            "Chọn mô hình trí tuệ nhân tạo để xóa phụ đề:\n"
            "- LaMa: Mô hình tốt nhất cho ảnh tĩnh, hoạt động rất nhanh.\n"
            "- STTN: Mô hình video trung cấp, đảm bảo tính liên kết thời gian tốt.\n"
            "- ProPainter: Mô hình video cao cấp nhất, khử nhấp nháy vượt trội.\n"
            "- OpenCV: Sử dụng thuật toán xử lý ảnh truyền thống, tốc độ cực nhanh nhưng chất lượng cơ bản."
        )
        
        # Cập nhật combo subtitle detect mode
        self.subtitle_detect_model_combo.comboBox.blockSignals(True)
        current_detect_idx = self.subtitle_detect_model_combo.comboBox.currentIndex()
        self.subtitle_detect_model_combo.setTitle(tr["SubtitleExtractorGUI"]["SubtitleDetectMode"])
        self.subtitle_detect_model_combo.comboBox.clear()
        self.subtitle_detect_model_combo.comboBox.addItems([list(tr['SubtitleDetectMode'].values())[i] for i,_ in enumerate(config.subtitleDetectMode.validator.options)])
        self.subtitle_detect_model_combo.comboBox.setCurrentIndex(current_detect_idx)
        self.subtitle_detect_model_combo.comboBox.blockSignals(False)
        self.subtitle_detect_model_combo.setToolTip(
            "Chọn mô hình OCR phát hiện phụ đề:\n"
            "- Server: Mô hình PP-OCRv5 Server có độ chính xác rất cao, khuyên dùng.\n"
            "- Mobile: Mô hình PP-OCRv5 Mobile dung lượng nhẹ, tốc độ nhanh hơn nhưng dễ bị sót chữ hơn."
        )
        
        self.hardware_acceleration.setTitle(tr["Setting"]["HardwareAcceleration"])
        if not HARDWARD_ACCELERATION_OPTION:
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationNO"])
        else:
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationDesc"])
        self.hardware_acceleration.setToolTip("Bật hoặc Tắt tăng tốc đồ họa phần cứng GPU (CUDA hoặc DirectML).")
        
        self.poisson_blending.setTitle("Poisson Blending (泊松融合)")
        self.poisson_blending.setContent(tr["Setting"]["PoissonBlendingDesc"])
        self.poisson_blending.setToolTip(tr["Setting"]["PoissonBlendingTooltip"])
        
        self.temporal_smoothing.setTitle(tr["Setting"]["TemporalSmoothing"])
        self.temporal_smoothing.setContent(tr["Setting"]["TemporalSmoothingDesc"])
        self.temporal_smoothing.setToolTip(tr["Setting"]["TemporalSmoothingTooltip"])
        
        self.sharpen_inpainted_area.setTitle(tr["Setting"]["SharpenInpaintedArea"])
        self.sharpen_inpainted_area.setContent(tr["Setting"]["SharpenInpaintedAreaDesc"])
        self.sharpen_inpainted_area.setToolTip(tr["Setting"]["SharpenInpaintedAreaTooltip"])
        
        # Cập nhật combo mask type
        self.mask_type_combo.comboBox.blockSignals(True)
        current_mask_idx = self.mask_type_combo.comboBox.currentIndex()
        self.mask_type_combo.setTitle(tr["Setting"]["MaskType"])
        self.mask_type_combo.setContent(tr["Setting"]["MaskTypeDesc"])
        self.mask_type_combo.comboBox.clear()
        self.mask_type_combo.comboBox.addItems([tr['Setting']['MaskTypeStroke'], tr['Setting']['MaskTypeBox']])
        self.mask_type_combo.comboBox.setCurrentIndex(current_mask_idx)
        self.mask_type_combo.comboBox.blockSignals(False)
        self.mask_type_combo.setToolTip(
            "Chọn phương pháp che phủ dòng chữ:\n"
            "- Nét chữ (Stroke): Mặt nạ bám khít theo từng nét vẽ của chữ. Giúp giữ nguyên vẹn tối đa nền gốc xung quanh.\n"
            "- Hộp chữ nhật (Box): Che phủ toàn bộ hộp chữ nhật chứa chữ. Xóa sạch 100% nhưng vùng cần inpaint lớn hơn, dễ gây mờ nền."
        )
        
        self.gpu_info_card.setToolTip("Hiển thị thông tin tên GPU đồ họa, dung lượng VRAM thực tế và hạn mức số khung hình được phân bổ tối đa cho việc xử lý đồng thời.")
        self.update_gpu_info()