import os
import json
from pathlib import Path
from PySide6 import QtWidgets, QtCore
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
        
        # 1. Xóa phần cấu hình ngôn ngữ ở đây vì đã được chuyển vào cài đặt nâng cao (Hệ thống)
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
        self.inpaint_mode_combo.comboBox.currentIndexChanged.connect(self._on_inpaint_mode_changed)
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

        # 3.5. Ngôn ngữ phụ đề OCR
        lang_icon = FluentIcon.LANGUAGE if hasattr(FluentIcon, 'LANGUAGE') else FluentIcon.GLOBE
        self.ocr_language_combo = ComboBoxSettingCard(
            configItem=config.ocrLanguage,
            icon=lang_icon,
            title="Ngôn ngữ phụ đề OCR",
            content="Tự động phát hiện hoặc chọn ngôn ngữ để AI quét OCR nhận diện chính xác nhất",
            parent=parent,
            texts=["Tự động phát hiện", "Tiếng Việt", "Tiếng Anh", "Tiếng Trung", "Tiếng Nhật", "Tiếng Hàn"]
        )
        self.ocr_language_combo.setToolTip(
            "Chọn ngôn ngữ của phụ đề trong video để mô hình AI tối ưu hóa độ chính xác:\n"
            "- Tự động phát hiện: Phân tích và tự động nhận diện ngôn ngữ chữ.\n"
            "- Tiếng Việt / Tiếng Anh / Tiếng Trung / Tiếng Nhật / Tiếng Hàn: Nạp bộ từ điển tối ưu chuyên biệt cho ngôn ngữ đã chọn."
        )
        self.addWidget(self.ocr_language_combo)

        # 4. 选用的 Mask 类型 (Mask Type)
        self.mask_type_combo = ComboBoxSettingCard(
            configItem=config.maskType,
            icon=FluentIcon.BROOM,
            title="Kiểu mặt nạ xóa chữ",
            content="Chọn phương pháp tạo mặt nạ phụ đề để xóa",
            parent=parent,
            texts=["Nét chữ", "Hộp chữ nhật"]
        )
        self.mask_type_combo.setToolTip(
            "Chọn phương pháp che phủ dòng chữ:\n"
            "- Nét chữ: Mặt nạ bám khít theo từng nét vẽ của chữ. Giúp giữ nguyên vẹn tối đa nền gốc xung quanh.\n"
            "- Hộp chữ nhật: Che phủ toàn bộ hộp chữ nhật chứa chữ. Xóa sạch 100% nhưng vùng cần inpaint lớn hơn, dễ gây mờ nền."
        )
        self.addWidget(self.mask_type_combo)

        # 4.5. Công tắc Tự động nhận diện chữ theo từng frame (ON / OFF)
        robot_icon = FluentIcon.ROBOT if hasattr(FluentIcon, 'ROBOT') else FluentIcon.SEARCH
        self.auto_detect_switch = SwitchSettingCard(
            icon=robot_icon,
            title="Tự động nhận diện chữ",
            content="Bật: Tự động quét OCR và xóa theo từng frame khi chạy | Tắt: Sử dụng khoanh vùng thủ công",
            configItem=config.autoDetectTextFrameByFrame,
            parent=parent
        )
        self.auto_detect_switch.setToolTip(
            "Chế độ Tự động: Khi chạy video, AI sẽ tự động phát hiện và xóa phụ đề trên từng khung hình.\n"
            "Chế độ Thủ công: Xóa theo các ô chữ cố định (bạn vẫn có thể dùng nút 'Quét AI 1 frame' bên dưới để tự động khoanh vùng thử)."
        )
        self.addWidget(self.auto_detect_switch)

        # 4.5.2. Nút bấm Xóa phụ đề di chuyển (Interactive Tracking)
        run_icon = FluentIcon.RUN if hasattr(FluentIcon, 'RUN') else FluentIcon.VIDEO
        self.moving_subtitle_card = PrimaryPushSettingCard(
            text="Nhận Diện",
            icon=run_icon,
            title="Xóa phụ đề di chuyển",
            content="Bám đuổi và xóa phụ đề động, cuộn chữ hoặc logo di chuyển theo thời gian",
            parent=parent
        )
        self.moving_subtitle_card.setToolTip(
            "Tính năng Bám đuổi phụ đề động:\n"
            "- Khoanh vùng logo ở khung hình 1 rồi bấm 'Nhận Diện'.\n"
            "- AI sẽ tự động bám sát vị trí di chuyển qua từng khung hình."
        )
        self.addWidget(self.moving_subtitle_card)

        # 4.6. Bộ công cụ cọ vẽ & Xem trước Red Mask
        from PySide6.QtCore import Qt
        from qfluentwidgets import TransparentToolButton
        
        brush_icon = FluentIcon.BRUSH if hasattr(FluentIcon, 'BRUSH') else FluentIcon.EDIT
        self.mask_tools_card = SettingCard(
            icon=brush_icon,
            title="Công cụ vẽ mặt nạ",
            content="Tùy chỉnh chế độ vẽ, xem trước lớp đỏ và dọn dẹp cọ",
            parent=parent
        )
        self.mask_tools_card.setToolTip("Điều khiển cọ vẽ tự do, khoanh hộp và lớp xem trước Red Mask")

        # Nút Quét AI 1 frame (Hỗ trợ khoanh vùng thủ công)
        self.auto_detect_frame_btn = TransparentToolButton(robot_icon, self.mask_tools_card)
        self.auto_detect_frame_btn.setToolTip("Quét AI tự động khoanh vùng chữ trên 1 khung hình hiện tại (dùng để sửa/xóa thủ công)")
        self.auto_detect_frame_btn.setCursor(Qt.PointingHandCursor)

        # Nút Chế độ vẽ (Khoanh Hộp ↔ Cọ Vẽ)
        self.brush_mode_btn = TransparentToolButton(brush_icon, self.mask_tools_card)
        self.brush_mode_btn.setToolTip("Chế độ: Khoanh Hộp (Mặc định)")
        self.brush_mode_btn.setCursor(Qt.PointingHandCursor)

        # Nút Red Mask
        red_icon = FluentIcon.VIEW if hasattr(FluentIcon, 'VIEW') else FluentIcon.SEARCH
        self.red_mask_btn = TransparentToolButton(red_icon, self.mask_tools_card)
        self.red_mask_btn.setToolTip("Bật / Tắt xem trước Lớp Phủ Đỏ")
        self.red_mask_btn.setCursor(Qt.PointingHandCursor)

        # Nút Xóa cọ
        clear_icon = FluentIcon.DELETE if hasattr(FluentIcon, 'DELETE') else FluentIcon.CLOSE
        self.clear_brush_btn = TransparentToolButton(clear_icon, self.mask_tools_card)
        self.clear_brush_btn.setToolTip("Xóa tất cả nét cọ vẽ tự do")
        self.clear_brush_btn.setCursor(Qt.PointingHandCursor)

        self.mask_tools_card.hBoxLayout.addWidget(self.auto_detect_frame_btn)
        self.mask_tools_card.hBoxLayout.addWidget(self.brush_mode_btn)
        self.mask_tools_card.hBoxLayout.addWidget(self.red_mask_btn)
        self.mask_tools_card.hBoxLayout.addWidget(self.clear_brush_btn)
        self.mask_tools_card.hBoxLayout.addSpacing(12)
        self.addWidget(self.mask_tools_card)





        # 7. Thẻ Mở Cài Đặt Nâng Cao
        self.open_advanced_card = PushSettingCard(
            text=tr["Setting"].get("OpenAdvancedSetting", "Mở Cài Đặt"),
            icon=FluentIcon.SETTING,
            title="Cài đặt nâng cao",
            content="Mở toàn bộ tùy chọn cấu hình chi tiết",
            parent=parent
        )
        self.open_advanced_card.clicked.connect(self._on_open_advanced_settings)
        self.addWidget(self.open_advanced_card)



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
        self.mask_type_combo.setTitle(tr["Setting"].get("MaskType", "Kiểu mặt nạ xóa chữ"))
        self.mask_type_combo.setContent(tr["Setting"].get("MaskTypeDesc", "Chọn phương pháp tạo mặt nạ phụ đề để xóa"))
        self.mask_type_combo.comboBox.clear()
        self.mask_type_combo.comboBox.addItems([tr['Setting'].get('MaskTypeStroke', 'Nét chữ'), tr['Setting'].get('MaskTypeBox', 'Hộp chữ nhật')])
        self.mask_type_combo.comboBox.setCurrentIndex(current_mask_idx)
        self.mask_type_combo.comboBox.blockSignals(False)
        self.mask_type_combo.setToolTip(tr["Setting"].get("MaskTypeTooltip", "Mask type tooltip"))
        
        # Cập nhật combo OCR Language
        if hasattr(self, 'ocr_language_combo'):
            self.ocr_language_combo.setTitle(tr["Setting"].get("OcrLanguageTitle", "Ngôn ngữ phụ đề OCR"))
            self.ocr_language_combo.setContent(tr["Setting"].get("OcrLanguageDesc", "Tự động phát hiện hoặc chọn ngôn ngữ để AI quét OCR nhận diện chính xác nhất"))
            self.ocr_language_combo.setToolTip(tr["Setting"].get("OcrLanguageTooltip", "Chọn ngôn ngữ OCR"))
            if hasattr(self.ocr_language_combo, 'comboBox') and hasattr(config, 'ocrLanguage'):
                self.ocr_language_combo.comboBox.blockSignals(True)
                current_ocr_idx = self.ocr_language_combo.comboBox.currentIndex()
                self.ocr_language_combo.comboBox.clear()
                self.ocr_language_combo.comboBox.addItems([list(tr['OcrLanguage'].values())[i] for i, _ in enumerate(config.ocrLanguage.validator.options)])
                self.ocr_language_combo.comboBox.setCurrentIndex(current_ocr_idx)
                self.ocr_language_combo.comboBox.blockSignals(False)
        
        self.gpu_info_card.setToolTip(tr["Setting"].get("GpuInfoTooltip", "GPU information"))
        self.update_gpu_info()

        # Cập nhật lại tự động xuống dòng sau khi đổi ngôn ngữ
        from PySide6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)

    def _on_inpaint_mode_changed(self, idx):
        try:
            import json
            from pathlib import Path
            from backend.config import config as main_cfg
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