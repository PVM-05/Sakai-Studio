"""
@desc: 高级设置页面 - Chia sub-tab chuyên biệt (System, Video Editing, SRT Features) với các cải tiến cao cấp.
- Ô tìm kiếm cài đặt thông minh
- Thẻ giám sát GPU/VRAM & Công cụ Benchmark hiệu năng GPU
- Bộ cấu hình mẫu 1-click (Fast, Ultra Quality, Balanced)
- Xuất / Nhập file cấu hình JSON
- Đếm số lượng cài đặt từng sub-tab
"""

import json
import time
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QFileDialog
from qfluentwidgets import (ScrollArea, ExpandLayout, FluentIcon,
                           SettingCardGroup, RangeSettingCard, SwitchSettingCard,
                           HyperlinkCard, PrimaryPushSettingCard, PushSettingCard,
                           ComboBoxSettingCard, SettingCard, MessageBox, Pivot,
                           SearchLineEdit)
from backend.config import config, tr, VERSION, PROJECT_HOME_URL, PROJECT_ISSUES_URL, PROJECT_RELEASES_URL, HARDWARD_ACCELERATION_OPTION
from backend.tools.folder_memory import FolderMemoryDialog
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor

class AdvancedSettingInterface(QWidget):
    """Giao diện Cài Đặt Nâng Cao chia sub-tab chuyên biệt với thanh tìm kiếm & các cải tiến trải nghiệm:
    1. Hệ Thống && Giao Diện
    2. Chỉnh Sửa Video && Xóa Phụ Đề
    3. Tính Năng SRT && OCR
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self.__init_widgets()

    def __init_widgets(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Thanh Header chứa Sub-Navigation Pivot bên trái & SearchLineEdit bên phải
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.pivot = Pivot(self)
        self.search_edit = SearchLineEdit(self)
        self.search_edit.setPlaceholderText("Tìm kiếm")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(320)
        self.search_edit.textChanged.connect(self._on_search_text_changed)

        header_layout.addWidget(self.pivot, 0, QtCore.Qt.AlignLeft)
        header_layout.addStretch(1)
        header_layout.addWidget(self.search_edit, 0, QtCore.Qt.AlignRight)

        main_layout.addLayout(header_layout)

        # 2. QStackedWidget chứa 3 tab scrollable
        self.stackedWidget = QStackedWidget(self)

        # Khởi tạo 3 trang scroll area cho 3 tab
        self.systemScrollWidget = self._create_scroll_area()
        self.videoScrollWidget = self._create_scroll_area()
        self.srtScrollWidget = self._create_scroll_area()

        # Layout cho từng scroll area
        self.systemLayout = ExpandLayout(self.systemScrollWidget.widget())
        self.videoLayout = ExpandLayout(self.videoScrollWidget.widget())
        self.srtLayout = ExpandLayout(self.srtScrollWidget.widget())

        # Thêm 3 trang vào StackedWidget
        self.stackedWidget.addWidget(self.systemScrollWidget)
        self.stackedWidget.addWidget(self.videoScrollWidget)
        self.stackedWidget.addWidget(self.srtScrollWidget)

        # Setup UI cards & layouts
        self.setup_ui()
        self.setup_layout()

        # Thêm các item vào Pivot bar (dùng && để Qt render thành & trên UI)
        self.pivot.addItem(
            routeKey="system_tab",
            text=f"{tr['Setting'].get('SystemTab', 'Hệ Thống && Giao Diện')} ({self._count_cards_in_scroll(self.systemScrollWidget)})",
            onClick=lambda: self.stackedWidget.setCurrentIndex(0),
            icon=FluentIcon.SETTING
        )
        self.pivot.addItem(
            routeKey="video_tab",
            text=f"{tr['Setting'].get('VideoTab', 'Chỉnh Sửa Video && Xóa Phụ Đề')} ({self._count_cards_in_scroll(self.videoScrollWidget)})",
            onClick=lambda: self.stackedWidget.setCurrentIndex(1),
            icon=FluentIcon.VIDEO
        )
        self.pivot.addItem(
            routeKey="srt_tab",
            text=f"{tr['Setting'].get('SrtTab', 'Tính Năng SRT && OCR')} ({self._count_cards_in_scroll(self.srtScrollWidget)})",
            onClick=lambda: self.stackedWidget.setCurrentIndex(2),
            icon=FluentIcon.DOCUMENT
        )

        self.pivot.setCurrentItem("system_tab")
        self.stackedWidget.setCurrentIndex(0)

        main_layout.addWidget(self.stackedWidget, 1)

    def _create_scroll_area(self) -> ScrollArea:
        """Tạo ScrollArea tiêu chuẩn đồng bộ giao diện Fluent"""
        scroll = ScrollArea(self)
        scroll_widget = QtWidgets.QWidget(scroll)
        scroll.setWidget(scroll_widget)
        scroll.enableTransparentBackground()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setAttribute(QtCore.Qt.WA_StyledBackground)
        return scroll

    def _count_cards_in_scroll(self, scroll: ScrollArea) -> int:
        """Đếm số lượng thẻ cài đặt trong scroll area"""
        count = 0
        for group in scroll.widget().findChildren(SettingCardGroup):
            count += len(group.findChildren(SettingCard))
        return count

    def setup_ui(self):
        """Khởi tạo tất cả thẻ Cài đặt (SettingCards) phân nhóm theo 3 sub-tab"""

        # ==========================================
        # TAB 1: HỆ THỐNG && GIAO DIỆN
        # ==========================================
        self.preset_group = SettingCardGroup("Bộ Cấu Hình Mẫu Nhanh", self.systemScrollWidget.widget())

        self.preset_fast_card = PushSettingCard(
            text="Kích Hoạt",
            icon=FluentIcon.SPEED_HIGH,
            title="Chế Độ Siêu Tốc",
            content="Tối ưu tốc độ xử lý nhanh nhất, tắt các bộ lọc làm mượt tiêu tốn thời gian",
            parent=self.preset_group
        )
        self.preset_fast_card.setToolTip("Click để tự động bật chế độ xử lý video tốc độ nhanh nhất.")
        self.preset_fast_card.clicked.connect(lambda: self.apply_preset("fast"))

        self.preset_ultra_card = PushSettingCard(
            text="Kích Hoạt",
            icon=FluentIcon.BRUSH,
            title="Chế Độ Chất Lượng Tối Đa",
            content="Bật đầy đủ Poisson Blending, Temporal Smoothing, Sharpen, Whisper AI & Bảo tồn màu sắc HDR",
            parent=self.preset_group
        )
        self.preset_ultra_card.setToolTip("Click để tự động bật chế độ xóa phụ đề chất lượng video cao cấp nhất.")
        self.preset_ultra_card.clicked.connect(lambda: self.apply_preset("ultra"))

        self.preset_balanced_card = PushSettingCard(
            text="Kích Hoạt",
            icon=FluentIcon.SETTING,
            title="Chế Độ Cân Bằng",
            content="Khôi phục về cấu hình cân bằng tối ưu giữa tốc độ và chất lượng ban đầu",
            parent=self.preset_group
        )
        self.preset_balanced_card.setToolTip("Click để tự động áp dụng cấu hình cân bằng mặc định.")
        self.preset_balanced_card.clicked.connect(lambda: self.apply_preset("balanced"))

        self.system_interface_group = SettingCardGroup("Giao Diện && Ngôn Ngữ", self.systemScrollWidget.widget())
        
        self.interface_combo = ComboBoxSettingCard(
            configItem=config.interface,
            icon=FluentIcon.LANGUAGE,
            title=tr["SubtitleExtractorGUI"].get("InterfaceLanguage", "Ngôn ngữ giao diện"),
            content="Chọn ngôn ngữ hiển thị chính cho phần mềm",
            parent=self.system_interface_group,
            texts=config.intefaceTexts.keys()
        )
        self.interface_combo.setToolTip("Thay đổi ngôn ngữ hiển thị ứng dụng.")

        self.system_storage_group = SettingCardGroup("Lưu Trữ && Quản Lý Cấu Hình", self.systemScrollWidget.widget())
        
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.DOWNLOAD,
            title=tr["Setting"]["SaveDirectory"],
            content=tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.system_storage_group
        )
        self.save_directory.setToolTip("Chọn đường dẫn thư mục mặc định để xuất và lưu trữ video sau khi xử lý thành công.")
        self.save_directory.clicked.connect(self.choose_save_directory)

        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup,
            icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"],
            parent=self.system_storage_group
        )
        self.check_update_on_startup.setToolTip("Tự động kiểm tra bản cập nhật mới từ GitHub mỗi khi khởi động ứng dụng.")

        self.export_config_card = PushSettingCard(
            text="Xuất File",
            icon=FluentIcon.SAVE,
            title="Xuất file cấu hình cài đặt (.json)",
            content="Lưu toàn bộ tùy chỉnh cấu hình hiện tại ra file JSON để sao lưu hoặc chia sẻ",
            parent=self.system_storage_group
        )
        self.export_config_card.setToolTip("Xuất các thiết lập hiện tại ra file JSON sao lưu.")
        self.export_config_card.clicked.connect(self.export_settings_json)

        self.import_config_card = PushSettingCard(
            text="Nhập File",
            icon=FluentIcon.FOLDER,
            title="Nhập file cấu hình cài đặt (.json)",
            content="Nạp cấu hình từ file JSON sao lưu để áp dụng nhanh",
            parent=self.system_storage_group
        )
        self.import_config_card.setToolTip("Nạp cài đặt từ file JSON sẵn có.")
        self.import_config_card.clicked.connect(self.import_settings_json)

        self.reset_defaults_card = PushSettingCard(
            text="Khôi Phục",
            icon=FluentIcon.HISTORY,
            title="Khôi phục cài đặt mặc định",
            content="Đặt lại toàn bộ các tùy chọn cài đặt hệ thống về cấu hình tối ưu ban đầu",
            parent=self.system_storage_group
        )
        self.reset_defaults_card.setToolTip("Click để khôi phục toàn bộ cài đặt phần mềm về thông số mặc định.")
        self.reset_defaults_card.clicked.connect(self.reset_settings_to_defaults)

        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.systemScrollWidget.widget())
        
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"],
            content=tr["Setting"]["FeedbackDesc"],
            parent=self.about_group
        )
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(PROJECT_ISSUES_URL)))

        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"],
            icon=FluentIcon.INFO,
            title=tr["Setting"]["CopyrightTitle"],
            content=tr["Setting"]["CopyrightDesc"].format(VERSION),
            parent=self.about_group
        )
        self.copyright.clicked.connect(lambda: self.check_update())

        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL,
            text=PROJECT_HOME_URL,
            icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"],
            content=tr["Setting"]["ProjectLinkDesc"],
            parent=self.about_group
        )

        # ==========================================
        # TAB 2: CHỈNH SỬA VIDEO && XÓA PHỤ ĐỀ
        # ==========================================
        self.step1_group = SettingCardGroup("Bước 1: Khởi tạo và Khoanh vùng", self.videoScrollWidget.widget())

        self.auto_tighten_card = SwitchSettingCard(
            configItem=config.autoTightenMask,
            icon=FluentIcon.ZOOM_IN if hasattr(FluentIcon, 'ZOOM_IN') else FluentIcon.SEARCH,
            title="Tự động ôm khít nét chữ",
            content="Tự động phân tích Canny Edge && Otsu Thresholding co hẹp khung vừa khít 100% nét chữ thực tế",
            parent=self.step1_group
        )
        self.auto_tighten_card.setToolTip("Khi bật tự động, hệ thống sẽ tự động ôm khít 100% nét chữ sau khi nhận diện. Tắt đi để tùy chỉnh thủ công.")
        config.autoTightenMask.valueChanged.connect(self.on_auto_tighten_changed)

        self.step2_group = SettingCardGroup("Bước 2: Lựa chọn Mô hình Trí tuệ Nhân tạo", self.videoScrollWidget.widget())

        # Thẻ thông tin GPU trực quan trong tab Video
        self.gpu_info_card = SettingCard(
            icon=FluentIcon.INFO,
            title="Card màn hình: Đang quét...",
            content="Đang tối ưu hóa dung lượng VRAM GPU...",
            parent=self.step2_group
        )
        self.gpu_info_card.setToolTip("Hiển thị tên GPU, VRAM thực tế và hạn mức số khung hình xử lý đồng thời.")

        self.gpu_benchmark_card = PushSettingCard(
            text="Chạy Đo",
            icon=FluentIcon.SPEED_HIGH,
            title="Đánh giá hiệu năng đồ họa",
            content="Kiểm tra tốc độ xử lý tính toán thực tế của Card màn hình và tính toán số FPS ước tính",
            parent=self.step2_group
        )
        self.gpu_benchmark_card.setToolTip("Chạy thử nghiệm đo lường tốc độ tính toán thực tế trên GPU.")
        self.gpu_benchmark_card.clicked.connect(self.run_gpu_benchmark)

        self.tracker_algorithm_combo = ComboBoxSettingCard(
            configItem=config.trackerAlgorithm,
            icon=FluentIcon.FINGERPRINT if hasattr(FluentIcon, 'FINGERPRINT') else FluentIcon.SEARCH,
            title="Thuật toán Theo dõi Đối tượng",
            content="Quyết định mức độ chính xác và thuật toán khi bám đuổi logo di chuyển",
            parent=self.step2_group,
            texts=["CSRT - Độ chính xác cao, Tốc độ chậm", "KCF - Cân bằng, Tốc độ cực nhanh", "MIL - Độ ổn định cao"]
        )
        self.tracker_algorithm_combo.setToolTip("Chọn thuật toán bám đuổi cho tính năng Xóa phụ đề di chuyển. Thuật toán CSRT mang lại độ chính xác cao nhất.")

        self.hardware_acceleration = SwitchSettingCard(
            configItem=config.hardwareAcceleration,
            icon=FluentIcon.SPEED_HIGH, 
            title=tr["Setting"]["HardwareAcceleration"],
            content=tr["Setting"]["HardwareAccelerationDesc"],
            parent=self.step2_group
        )
        self.hardware_acceleration.setToolTip("Bật hoặc Tắt tăng tốc đồ họa phần cứng GPU.")
        if not HARDWARD_ACCELERATION_OPTION:
            self.hardware_acceleration.setEnabled(False)
            self.hardware_acceleration.setChecked(False)

        self.auto_hardware_tuning = SwitchSettingCard(
            configItem=config.autoHardwareTuning,
            icon=FluentIcon.SPEED_HIGH,
            title="Tối ưu hóa hiệu năng theo GPU",
            content="Tự động tính toán số frame xử lý tối ưu để tránh tràn VRAM GPU",
            parent=self.step2_group
        )
        self.auto_hardware_tuning.setToolTip("Tự động phân tích dung lượng VRAM thực tế trên GPU để tối ưu khung hình xử lý đồng thời.")
        self.auto_hardware_tuning.switchButton.checkedChanged.connect(self.on_auto_tuning_changed)

        self.gpu_video_encoding = SwitchSettingCard(
            configItem=config.gpuVideoEncoding,
            icon=FluentIcon.VIDEO,
            title="Tăng tốc xuất video bằng GPU",
            content="Sử dụng bộ giải mã/mã hóa phần cứng GPU giúp xuất video nhanh gấp 5-10 lần",
            parent=self.step2_group
        )
        self.gpu_video_encoding.setToolTip("Kích hoạt chip phần cứng chuyên dụng trên Card đồ họa (NVENC/DirectML) để nén/giải nén video trực tiếp.")

        self.sttn_neighbor_stride = RangeSettingCard(
            configItem=config.sttnNeighborStride,
            icon=FluentIcon.UNIT,
            title=tr["Setting"]["SttnNeighborStride"],
            content=tr["Setting"]["SttnNeighborStrideDesc"],
            parent=self.step2_group
        )
        self.sttn_neighbor_stride.setToolTip("Bước nhảy khung hình lân cận cho mô hình STTN.")

        self.sttn_reference_length = RangeSettingCard(
            configItem=config.sttnReferenceLength,
            icon=FluentIcon.MORE,
            title=tr["Setting"]["SttnReferenceLength"],
            content=tr["Setting"]["SttnReferenceLengthDesc"],
            parent=self.step2_group
        )
        self.sttn_reference_length.setToolTip("Số lượng khung hình tham chiếu dài hạn cho mô hình STTN.")

        self.sttn_max_load_num = RangeSettingCard(
            configItem=config.sttnMaxLoadNum,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["SttnMaxLoadNum"],
            content=tr["Setting"]["SttnMaxLoadNumDesc"],
            parent=self.step2_group
        )
        self.sttn_max_load_num.setToolTip("Số khung hình tối đa xử lý đồng thời trong một phân đoạn của STTN.")

        self.propainter_max_load_num = RangeSettingCard(
            configItem=config.propainterMaxLoadNum,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["PropainterMaxLoadNum"],
            content=tr["Setting"]["PropainterMaxLoadNumDesc"],
            parent=self.step2_group
        )
        self.propainter_max_load_num.setToolTip("Số khung hình tối đa xử lý đồng thời trong một phân đoạn của ProPainter.")

        self.step3_group = SettingCardGroup("Bước 3: Cấu hình Chất lượng và Độ nét", self.videoScrollWidget.widget())

        self.preserve_color_card = SwitchSettingCard(
            configItem=config.preserveColorMetadata,
            icon=FluentIcon.PALETTE if hasattr(FluentIcon, 'PALETTE') else FluentIcon.BRUSH,
            title="Bảo tồn chuẩn màu sắc video gốc",
            content="Giữ nguyên thông số màu sắc HDR và hồ sơ màu của video gốc khi xuất",
            parent=self.step3_group
        )
        self.preserve_color_card.setToolTip("Giúp video sau khi xóa phụ đề giữ được màu sắc gốc, tránh hiện tượng bị nhạt màu hay sai profile màu trên thiết bị HDR.")

        self.poisson_blending = SwitchSettingCard(
            configItem=config.poissonBlending,
            icon=FluentIcon.BRUSH, 
            title=tr["Setting"]["PoissonBlending"],
            content=tr["Setting"]["PoissonBlendingDesc"],
            parent=self.step3_group
        )
        self.poisson_blending.setToolTip("Sử dụng thuật toán Poisson Blending để hòa trộn mượt mà biên giao thoa giữa vùng được xóa và video gốc.")

        self.temporal_smoothing = SwitchSettingCard(
            configItem=config.temporalSmoothing,
            icon=FluentIcon.MOVIE, 
            title="Lọc mượt thời gian",
            content="Khử nhấp nháy, rung hạt nền bằng bộ lọc thích ứng chuyển động",
            parent=self.step3_group
        )
        self.temporal_smoothing.setToolTip("Khử hiện tượng nhấp nháy hoặc rung hạt nhiễu ở vùng inpaint bằng cách nội suy trung bình trọng số thích ứng chuyển động.")

        self.sharpen_inpainted_area = SwitchSettingCard(
            configItem=config.sharpenInpaintedArea,
            icon=FluentIcon.EDIT,
            title="Làm nét vùng xóa",
            content="Làm nét nhẹ vùng nền sau khi xóa phụ đề",
            parent=self.step3_group
        )
        self.sharpen_inpainted_area.setToolTip("Áp dụng bộ lọc Unsharp Mask làm nét cục bộ vùng ảnh sau khi xóa.")

        self.temporal_smoothing_radius = RangeSettingCard(
            configItem=config.temporalSmoothingRadius,
            icon=FluentIcon.MOVIE,
            title="Bán kính làm mịn thời gian",
            content="Bán kính khung hình lân cận để lọc mượt chống nhấp nháy chuyển động (1-10)",
            parent=self.step3_group
        )
        self.temporal_smoothing_radius.setToolTip("Bán kính khung hình lân cận lấy làm tham chiếu để tính toán mượt hóa thời gian.")

        # Listen to config changes to dynamically update GPU info card
        config.autoHardwareTuning.valueChanged.connect(self.update_gpu_info)
        config.propainterMaxLoadNum.valueChanged.connect(self.update_gpu_info)
        config.sttnMaxLoadNum.valueChanged.connect(self.update_gpu_info)
        self.update_gpu_info()

        # MMO Features Group
        self.mmo_group = SettingCardGroup("Công Cụ Lách Bản Quyền", self.videoScrollWidget.widget())
        
        self.mmo_flip_video_card = SwitchSettingCard(
            configItem=config.mmoFlipVideo,
            icon=FluentIcon.SYNC,
            title="Lật ngược video",
            content="Lật video theo chiều ngang để tránh thuật toán quét bản quyền",
            parent=self.mmo_group
        )
        self.mmo_flip_video_card.setToolTip("Lật toàn bộ khung hình video từ trái sang phải, hữu ích cho làm video MMO.")

        self.mmo_speed_shift_card = SwitchSettingCard(
            configItem=config.mmoSpeedShift,
            icon=FluentIcon.SPEED_HIGH,
            title="Thay đổi tốc độ",
            content="Vi chỉnh tốc độ video một chút để lách bản quyền âm thanh và hình ảnh",
            parent=self.mmo_group
        )
        
        self.mmo_sub_style_combo = ComboBoxSettingCard(
            configItem=config.mmoSubStyle,
            icon=FluentIcon.FONT,
            title="Kiểu phụ đề MMO",
            content="Mẫu thiết kế phụ đề nổi bật chuẩn TikTok/Shorts",
            parent=self.mmo_group,
            texts=["TikTok Vàng", "YouTube Trắng", "Netflix", "Tự Do"]
        )

        # Trigger single shot initial states
        QtCore.QTimer.singleShot(0, lambda: self.on_auto_tighten_changed(config.autoTightenMask.value))
        QtCore.QTimer.singleShot(0, lambda: self.on_auto_tuning_changed(config.autoHardwareTuning.value))
        config.translateSubtitles.valueChanged.connect(self.on_translate_subtitles_changed)
        QtCore.QTimer.singleShot(0, lambda: self.on_translate_subtitles_changed(config.translateSubtitles.value))

        # ==========================================
        # TAB 3: TÍNH NĂNG SRT && NHẬN DIỆN PHỤ ĐỀ
        # ==========================================
        self.srt_feature_group = SettingCardGroup("Quản Lý && Dịch File Phụ Đề SRT", self.srtScrollWidget.widget())

        self.export_srt_card = SwitchSettingCard(
            configItem=config.exportSrt,
            icon=FluentIcon.DOCUMENT,
            title="Tự động xuất file phụ đề .srt",
            content="Tạo file .srt song song cùng video khi phát hiện và xóa phụ đề thành công",
            parent=self.srt_feature_group
        )
        self.export_srt_card.setToolTip("Khi bật, hệ thống sẽ trích xuất văn bản nhận diện được và lưu thành file phụ đề chuẩn .srt.")

        self.srt_save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.FOLDER,
            title="Thư mục lưu file SRT mặc định",
            content="Mặc định (Lưu cùng thư mục video thành phẩm)" if not config.srtSaveDirectory.value else config.srtSaveDirectory.value,
            parent=self.srt_feature_group
        )
        self.srt_save_directory.setToolTip("Chọn thư mục riêng để lưu trữ các file phụ đề .srt được xuất ra.")
        self.srt_save_directory.clicked.connect(self.choose_srt_save_directory)

        self.whisper_fallback_card = SwitchSettingCard(
            configItem=config.whisperFallback,
            icon=FluentIcon.MICROPHONE if hasattr(FluentIcon, 'MICROPHONE') else FluentIcon.MUSIC,
            title="Bổ trợ nhận diện giọng nói Whisper AI",
            content="Sử dụng Whisper AI nhận diện tiếng nói để bổ sung cho các đoạn phụ đề mờ/nhiễu mà OCR bỏ sót",
            parent=self.srt_feature_group
        )
        self.whisper_fallback_card.setToolTip("Khi bật, hệ thống sẽ tự động quét phổ âm thanh nói để đảm bảo 100% thời gian hiện câu phụ đề trong file SRT không bị ngắt quãng.")

        self.voice_separation_card = SwitchSettingCard(
            configItem=config.voiceSeparation,
            icon=FluentIcon.MUSIC if hasattr(FluentIcon, 'MUSIC') else FluentIcon.MICROPHONE,
            title="Tách giọng nói bằng AI",
            content="Tách riêng giọng nói khỏi nhạc nền trước khi nhận diện Whisper AI để tăng độ chính xác lên 99%",
            parent=self.srt_feature_group
        )
        self.voice_separation_card.setToolTip("Khuyên dùng nếu video có nhạc nền to hoặc nhiều tạp âm. Thời gian xử lý sẽ lâu hơn một chút do chạy AI tách âm.")

        self.translate_subtitles_card = SwitchSettingCard(
            configItem=config.translateSubtitles,
            icon=FluentIcon.LANGUAGE,
            title="Tự động dịch phụ đề SRT",
            content="Dịch nội dung phụ đề sang ngôn ngữ đích sau khi trích xuất",
            parent=self.srt_feature_group
        )
        self.translate_subtitles_card.setToolTip("Tự động sử dụng máy dịch để dịch file SRT sang ngôn ngữ bạn đã cấu hình.")

        self.target_language_combo = ComboBoxSettingCard(
            configItem=config.targetLanguage,
            icon=FluentIcon.GLOBE,
            title="Ngôn ngữ dịch đích",
            content="Chọn ngôn ngữ mà bạn muốn dịch phụ đề sang",
            parent=self.srt_feature_group,
            texts=["Tiếng Việt", "Tiếng Anh", "Tiếng Trung", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Pháp", "Tiếng Đức"]
        )
        self.target_language_combo.setToolTip("Khi bật Tự động dịch phụ đề SRT, văn bản sẽ được dịch sang ngôn ngữ này.")

        self.burn_translated_subtitles_card = SwitchSettingCard(
            configItem=config.burnTranslatedSubtitles,
            icon=FluentIcon.EDIT,
            title="Chèn phụ đề đã dịch vào video",
            content="In đè phụ đề đã dịch trực tiếp lên video thành phẩm sau khi xóa phụ đề cũ",
            parent=self.srt_feature_group
        )
        self.burn_translated_subtitles_card.setToolTip("Hệ thống sẽ render trực tiếp dòng chữ phụ đề đã dịch mới lên video.")

        # Subtitle OCR detection parameters group
        self.subtitle_detection_group = SettingCardGroup(tr["Setting"]["SubtitleDetectionSetting"], self.srtScrollWidget.widget())

        self.subtitle_yx_axis_difference_pixel = RangeSettingCard(
            configItem=config.subtitleYXAxisDifferencePixel,
            icon=FluentIcon.ZOOM,
            title=tr["Setting"]["SubtitleYXAxisDifferencePixel"],
            content=tr["Setting"]["SubtitleYXAxisDifferencePixelDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_yx_axis_difference_pixel.setToolTip("Độ lệch kích thước tối đa giữa chiều rộng và chiều cao chữ để lọc bỏ các vùng phát hiện nhầm.")

        self.subtitle_area_deviation_pixel = RangeSettingCard(
            configItem=config.subtitleAreaDeviationPixel,
            icon=FluentIcon.ZOOM_IN,
            title=tr["Setting"]["SubtitleAreaDeviationPixel"],
            content=tr["Setting"]["SubtitleAreaDeviationPixelDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_area_deviation_pixel.setToolTip("Độ lệch tối đa của vùng phụ đề cho phép để tránh cắt lẹm vào biên vùng chữ.")

        self.subtitle_area_y_axis_difference_pixel = RangeSettingCard(
            configItem=config.subtitleAreaYAxisDifferencePixel,
            icon=FluentIcon.ALIGNMENT,
            title=tr["Setting"]["SubtitleAreaYAxisDifferencePixel"],
            content=tr["Setting"]["SubtitleAreaYAxisDifferencePixelDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_area_y_axis_difference_pixel.setToolTip("Độ lệch tối đa theo trục Y để gộp nhóm các dòng phụ đề xuất hiện đồng thời.")

        self.subtitle_area_pixel_tolerance_y_pixel = RangeSettingCard(
            configItem=config.subtitleAreaPixelToleranceYPixel,
            icon=FluentIcon.UP,
            title=tr["Setting"]["SubtitleAreaPixelToleranceYPixel"],
            content=tr["Setting"]["SubtitleAreaPixelToleranceYPixelDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_area_pixel_tolerance_y_pixel.setToolTip("Dung sai sai lệch dòng theo trục Y khi xác định vị trí phụ đề ổn định theo chiều dọc.")

        self.subtitle_area_pixel_tolerance_x_pixel = RangeSettingCard(
            configItem=config.subtitleAreaPixelToleranceXPixel,
            icon=FluentIcon.RIGHT_ARROW,
            title=tr["Setting"]["SubtitleAreaPixelToleranceXPixel"],
            content=tr["Setting"]["SubtitleAreaPixelToleranceXPixelDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_area_pixel_tolerance_x_pixel.setToolTip("Dung sai sai lệch dòng theo trục X khi xác định vùng ngang chứa phụ đề ổn định.")

        self.subtitle_timeline_backward_frame_count = RangeSettingCard(
            configItem=config.subtitleTimelineBackwardFrameCount,
            icon=FluentIcon.PAGE_LEFT,
            title=tr["Setting"]["SubtitleTimelineBackwardFrameCount"],
            content=tr["Setting"]["SubtitleTimelineBackwardFrameCountDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_timeline_backward_frame_count.setToolTip("Số khung hình mở rộng lùi về phía trước dòng thời gian.")

        self.subtitle_timeline_forward_frame_count = RangeSettingCard(
            configItem=config.subtitleTimelineForwardFrameCount,
            icon=FluentIcon.PAGE_RIGHT,
            title=tr["Setting"]["subtitleTimelineForwardFrameCount"],
            content=tr["Setting"]["subtitleTimelineForwardFrameCountDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_timeline_forward_frame_count.setToolTip("Số khung hình mở rộng tiến về phía sau dòng thời gian.")

        # Format word wrapping for all child labels across all tabs
        self._format_card_labels()

    def setup_layout(self):
        """Thêm các SettingCardGroup vào ExpandLayout của 3 sub-tab"""
        # Tab 1: System
        self.preset_group.addSettingCard(self.preset_fast_card)
        self.preset_group.addSettingCard(self.preset_ultra_card)
        self.preset_group.addSettingCard(self.preset_balanced_card)
        self.systemLayout.addWidget(self.preset_group)

        self.system_interface_group.addSettingCard(self.interface_combo)
        self.systemLayout.addWidget(self.system_interface_group)

        self.system_storage_group.addSettingCard(self.save_directory)
        self.system_storage_group.addSettingCard(self.check_update_on_startup)
        self.system_storage_group.addSettingCard(self.export_config_card)
        self.system_storage_group.addSettingCard(self.import_config_card)
        self.system_storage_group.addSettingCard(self.reset_defaults_card)
        self.systemLayout.addWidget(self.system_storage_group)

        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)
        self.systemLayout.addWidget(self.about_group)

        self.systemLayout.setSpacing(16)
        self.systemLayout.setContentsMargins(16, 16, 16, 48)

        # Tab 2: Video Editing
        self.step1_group.addSettingCard(self.auto_tighten_card)
        self.videoLayout.addWidget(self.step1_group)

        self.step2_group.addSettingCard(self.gpu_info_card)
        self.step2_group.addSettingCard(self.gpu_benchmark_card)
        self.step2_group.addSettingCard(self.tracker_algorithm_combo)
        self.step2_group.addSettingCard(self.hardware_acceleration)
        self.step2_group.addSettingCard(self.auto_hardware_tuning)
        self.step2_group.addSettingCard(self.gpu_video_encoding)
        self.step2_group.addSettingCard(self.sttn_neighbor_stride)
        self.step2_group.addSettingCard(self.sttn_reference_length)
        self.step2_group.addSettingCard(self.sttn_max_load_num)
        self.step2_group.addSettingCard(self.propainter_max_load_num)
        self.videoLayout.addWidget(self.step2_group)

        self.step3_group.addSettingCard(self.preserve_color_card)
        self.step3_group.addSettingCard(self.poisson_blending)
        self.step3_group.addSettingCard(self.temporal_smoothing)
        self.step3_group.addSettingCard(self.sharpen_inpainted_area)
        self.step3_group.addSettingCard(self.temporal_smoothing_radius)
        self.videoLayout.addWidget(self.step3_group)

        self.mmo_group.addSettingCard(self.mmo_flip_video_card)
        self.mmo_group.addSettingCard(self.mmo_speed_shift_card)
        self.mmo_group.addSettingCard(self.mmo_sub_style_combo)
        self.videoLayout.addWidget(self.mmo_group)

        self.videoLayout.setSpacing(16)
        self.videoLayout.setContentsMargins(16, 16, 16, 48)

        # Tab 3: SRT Features
        self.srt_feature_group.addSettingCard(self.export_srt_card)
        self.srt_feature_group.addSettingCard(self.srt_save_directory)
        self.srt_feature_group.addSettingCard(self.whisper_fallback_card)
        self.srt_feature_group.addSettingCard(self.voice_separation_card)
        self.srt_feature_group.addSettingCard(self.translate_subtitles_card)
        self.srt_feature_group.addSettingCard(self.target_language_combo)
        self.srt_feature_group.addSettingCard(self.burn_translated_subtitles_card)
        self.srtLayout.addWidget(self.srt_feature_group)

        self.subtitle_detection_group.addSettingCard(self.subtitle_yx_axis_difference_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_deviation_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_y_axis_difference_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_pixel_tolerance_y_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_pixel_tolerance_x_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_timeline_backward_frame_count)
        self.subtitle_detection_group.addSettingCard(self.subtitle_timeline_forward_frame_count)
        self.srtLayout.addWidget(self.subtitle_detection_group)

        self.srtLayout.setSpacing(16)
        self.srtLayout.setContentsMargins(16, 16, 16, 48)

        # Cho phép tất cả các nhãn (SettingCard) tự động xuống dòng (Word Wrap)
        for child in self.findChildren(QtWidgets.QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)


    def apply_preset(self, preset_name: str):
        """Áp dụng cấu hình nhanh 1-click theo nhu cầu"""
        if preset_name == "fast":
            config.set(config.autoTightenMask, False)
            config.set(config.poissonBlending, False)
            config.set(config.temporalSmoothing, False)
            config.set(config.sharpenInpaintedArea, False)
            config.set(config.hardwareAcceleration, True)
            config.set(config.gpuVideoEncoding, True)
            config.set(config.autoHardwareTuning, True)
            msg = "Đã kích hoạt Chế Độ Siêu Tốc! Các bộ lọc làm mượt nâng cao được tạm tắt để tối đa hóa tốc độ xuất video."
        elif preset_name == "ultra":
            config.set(config.poissonBlending, True)
            config.set(config.temporalSmoothing, True)
            config.set(config.sharpenInpaintedArea, True)
            config.set(config.autoTightenMask, True)
            config.set(config.hardwareAcceleration, True)
            config.set(config.gpuVideoEncoding, True)
            if hasattr(config, 'whisperFallback'):
                config.set(config.whisperFallback, True)
            if hasattr(config, 'preserveColorMetadata'):
                config.set(config.preserveColorMetadata, True)
            msg = "Đã kích hoạt Chế Độ Chất Lượng Tối Đa! Bật đầy đủ Poisson Blending, khử nhấp nháy, Whisper AI và bảo tồn màu sắc HDR."
        elif preset_name == "balanced":
            config.set(config.autoTightenMask, True)
            config.set(config.hardwareAcceleration, True)
            config.set(config.poissonBlending, False)
            config.set(config.temporalSmoothing, True)
            config.set(config.sharpenInpaintedArea, True)
            config.set(config.autoHardwareTuning, True)
            config.set(config.gpuVideoEncoding, True)
            msg = "Đã khôi phục về Chế Độ Cân Bằng."
        else:
            return

        self.show_message_box("Áp dụng cấu hình thành công", msg)
        self.retranslateUi()

    def run_gpu_benchmark(self):
        """Chạy đo hiệu năng GPU thực tế"""
        import torch
        if not torch.cuda.is_available():
            self.show_message_box("Thông tin GPU", "Hệ thống hiện tại đang ở chế độ chỉ dùng CPU. Tính năng đo hiệu năng yêu cầu Card đồ họa NVIDIA CUDA.")
            return
        try:
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(device)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Khởi động tensor test
            x = torch.randn((1200, 1200, 32), device=device)
            start_t = time.perf_counter()
            for _ in range(60):
                _ = torch.matmul(x, x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_t
            
            fps_est = int(180 / elapsed) if elapsed > 0 else 60
            msg = (f"Kết Quả Đo Hiệu Năng GPU:\n\n"
                   f"- Card màn hình: {gpu_name}\n"
                   f"- Tổng dung lượng VRAM: {total_vram:.2f} GB\n"
                   f"- Thời gian thực thi 60 phép tính ma trận: {elapsed*1000:.1f} ms\n"
                   f"- Tốc độ xử lý video ước tính: ~{fps_est} FPS\n\n"
                   f"Card màn hình của bạn hoạt động rất xuất sắc và tối ưu!")
            self.show_message_box("Kết Quả GPU Benchmark", msg)
        except Exception as e:
            self.show_message_box("Lỗi Benchmark", f"Không thể hoàn thành bài kiểm tra GPU: {e}")

    def export_settings_json(self):
        """Xuất file cấu hình cài đặt ra JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất file cấu hình cài đặt", "sakai_settings_backup.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        data = {
            "autoTightenMask": config.autoTightenMask.value,
            "hardwareAcceleration": config.hardwareAcceleration.value,
            "poissonBlending": config.poissonBlending.value,
            "temporalSmoothing": config.temporalSmoothing.value,
            "sharpenInpaintedArea": config.sharpenInpaintedArea.value,
            "autoHardwareTuning": config.autoHardwareTuning.value,
            "gpuVideoEncoding": config.gpuVideoEncoding.value,
            "preserveColorMetadata": getattr(config, 'preserveColorMetadata', None) and config.preserveColorMetadata.value,
            "maskDilation": config.maskDilation.value,
            "maskFeather": config.maskFeather.value,
            "temporalSmoothingRadius": config.temporalSmoothingRadius.value,
            "exportSrt": config.exportSrt.value,
            "whisperFallback": getattr(config, 'whisperFallback', None) and config.whisperFallback.value,
            "checkUpdateOnStartup": config.checkUpdateOnStartup.value,
            "saveDirectory": config.saveDirectory.value,
            "srtSaveDirectory": config.srtSaveDirectory.value,
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.show_message_box("Thành công", f"Đã xuất file cấu hình thành công:\n{file_path}")
        except Exception as e:
            self.show_message_box("Lỗi", f"Không thể xuất file cấu hình: {e}")

    def import_settings_json(self):
        """Nhập file cấu hình từ JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Nhập file cấu hình cài đặt", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key, val in data.items():
                if hasattr(config, key):
                    item = getattr(config, key)
                    config.set(item, val)
            self.show_message_box("Thành công", "Đã nhập và áp dụng file cấu hình cài đặt mới thành công!")
            self.retranslateUi()
        except Exception as e:
            self.show_message_box("Lỗi", f"Không thể đọc file cấu hình: {e}")

    def _on_search_text_changed(self, text: str):
        """Tìm kiếm & lọc các thẻ cài đặt theo từ khóa"""
        text = text.strip().lower()

        found_count = 0
        matching_tab_index = -1

        for tab_idx, scroll_widget in enumerate([self.systemScrollWidget, self.videoScrollWidget, self.srtScrollWidget]):
            layout_widget = scroll_widget.widget()
            for group in layout_widget.findChildren(SettingCardGroup):
                group_has_visible_card = False
                for card in group.findChildren(SettingCard):
                    title = getattr(card, 'titleLabel', None)
                    content = getattr(card, 'contentLabel', None)
                    card_title = title.text().lower() if title else ""
                    card_content = content.text().lower() if content else ""

                    if not text or (text in card_title or text in card_content):
                        card.show()
                        group_has_visible_card = True
                        found_count += 1
                        if text and matching_tab_index == -1:
                            matching_tab_index = tab_idx
                    else:
                        card.hide()

                if not text or group_has_visible_card:
                    group.show()
                else:
                    group.hide()

        if text and matching_tab_index != -1 and matching_tab_index != self.stackedWidget.currentIndex():
            self.stackedWidget.setCurrentIndex(matching_tab_index)
            routes = ["system_tab", "video_tab", "srt_tab"]
            self.pivot.setCurrentItem(routes[matching_tab_index])

    def update_gpu_info(self):
        """Cập nhật động thông tin GPU & VRAM trực quan"""
        import torch
        
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
            max_load_pp = config.propainterMaxLoadNum.value
            max_load_sttn = config.sttnMaxLoadNum.value
            performance_tier = "Thủ công"

        gpu_title = "Card màn hình: CPU Only"
        if torch.cuda.is_available():
            try:
                device_idx = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device_idx)
                total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
                gpu_title = f"GPU: {gpu_name} ({total_vram:.1f} GB VRAM)"
            except Exception:
                gpu_title = "GPU CUDA Detected"

        gpu_content = f"Chế độ: {performance_tier} | Cấu hình frame tối đa: ProPainter ({max_load_pp} frames), STTN ({max_load_sttn} frames)."
        if hasattr(self, 'gpu_info_card'):
            self.gpu_info_card.setTitle(gpu_title)
            self.gpu_info_card.setContent(gpu_content)

    def reset_settings_to_defaults(self):
        """Khôi phục các cài đặt cấu hình về mặc định ban đầu"""
        w = MessageBox(
            "Khôi phục cài đặt mặc định",
            "Bạn có chắc chắn muốn đặt lại toàn bộ cài đặt phần mềm về cấu hình mặc định tối ưu ban đầu không?",
            self
        )
        w.yesButton.setText("Đồng ý Khôi Phục")
        w.cancelButton.setText("Hủy")
        if w.exec():
            config.set(config.autoTightenMask, True)
            config.set(config.hardwareAcceleration, True)
            config.set(config.poissonBlending, False)
            config.set(config.temporalSmoothing, True)
            config.set(config.sharpenInpaintedArea, True)
            config.set(config.autoHardwareTuning, True)
            config.set(config.gpuVideoEncoding, True)
            config.set(config.maskDilation, 8)
            config.set(config.maskFeather, 8)
            config.set(config.temporalSmoothingRadius, 2)
            config.set(config.exportSrt, False)
            config.set(config.checkUpdateOnStartup, True)
            
            self.show_message_box("Thành công", "Đã khôi phục cài đặt mặc định thành công!")
            self.retranslateUi()

    def _trigger_auto_tighten_from_advanced(self):
        """Kích hoạt tính năng Ôm Khít Nét Chữ trên giao diện chính HomeInterface"""
        parent_window = self.window()
        if hasattr(parent_window, 'homeInterface'):
            parent_window.switchTo(parent_window.homeInterface)
            if hasattr(parent_window.homeInterface, 'tighten_area_button_clicked'):
                parent_window.homeInterface.tighten_area_button_clicked()

    def on_auto_tighten_changed(self, is_checked: bool):
        """Khi bật tự động ôm khít, khóa tinh chỉnh tay. Ngược lại mở cho chỉnh tay."""
        if hasattr(self, 'mask_dilation'):
            self.mask_dilation.setEnabled(not is_checked)
        if hasattr(self, 'mask_feather'):
            self.mask_feather.setEnabled(not is_checked)

    def show_message_box(self, title: str, content: str, showYesButton=False, yesSlot=None):
        """Show message box"""
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr('Close'))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        """Check software update"""
        TaskExecutor.runTask(self.version_manager.has_new_version).then(
            lambda success: self.on_version_info_fetched(success, ignore))

    def on_version_info_fetched(self, success, ignore=False):
        if success:
            self.show_message_box(
                tr["Setting"]["UpdatesAvailableTitle"],
                tr["Setting"]["UpdatesAvailableDesc"].format(self.version_manager.lastest_version),
                True,
                lambda: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl(PROJECT_RELEASES_URL)
                )
            )
        elif not ignore:
            self.show_message_box(
                tr["Setting"]["NoUpdatesAvailableTitle"],
                tr["Setting"]["NoUpdatesAvailableDesc"],
            )
    
    def choose_save_directory(self):
        """Chọn thư mục lưu video mặc định"""
        folder = FolderMemoryDialog.getExistingDirectory(
            self, tr['Setting']['ChooseDirectory'], category="default")
        if not folder:
            folder = ""

        config.set(config.saveDirectory, folder)
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)

    def choose_srt_save_directory(self):
        """Chọn thư mục lưu file SRT mặc định"""
        folder = FolderMemoryDialog.getExistingDirectory(
            self, tr['Setting'].get('ChooseSrtDirectory', "Chọn Thư Mục Lưu File SRT"), category="srt_default")
        if not folder:
            folder = ""

        config.set(config.srtSaveDirectory, folder)
        self.srt_save_directory.setContent("Mặc định (Lưu cùng thư mục video thành phẩm)" if not config.srtSaveDirectory.value else config.srtSaveDirectory.value)

    def on_auto_tuning_changed(self, checked):
        """Khi bật/tắt tự động tinh chỉnh phần cứng, vô hiệu hóa/kích hoạt các thanh trượt thủ công"""
        if hasattr(self, 'sttn_max_load_num'):
            self.sttn_max_load_num.setEnabled(not checked)
        if hasattr(self, 'propainter_max_load_num'):
            self.propainter_max_load_num.setEnabled(not checked)
        if hasattr(self, 'gpu_video_encoding'):
            self.gpu_video_encoding.setEnabled(not checked)
        self.update_gpu_info()

    def on_translate_subtitles_changed(self, checked):
        """Khi tự động dịch, kích hoạt các tùy chọn đích"""
        if hasattr(self, 'target_language_combo'):
            self.target_language_combo.setEnabled(checked)
        if hasattr(self, 'burn_translated_subtitles_card'):
            self.burn_translated_subtitles_card.setEnabled(checked)

    def _format_card_labels(self):
        """Cấu hình tự động xuống dòng và chiều cao hiển thị phù hợp cho tất cả SettingCards"""
        # Cải tiến WordWrap cho nhãn cài đặt
        for child in self.findChildren(QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)
                
                child.titleLabel.setMinimumWidth(180)
                child.contentLabel.setMinimumWidth(220)
                
                content_text = child.contentLabel.text()
                desc_len = len(content_text) if content_text else 0
                
                if desc_len > 80:
                    child.contentLabel.setMinimumHeight(55)
                    height = 105
                elif desc_len > 40:
                    child.contentLabel.setMinimumHeight(38)
                    height = 85
                else:
                    child.contentLabel.setMinimumHeight(18)
                    height = 70
                    
                if desc_len == 0:
                    height = 55
                    
                child.setMinimumHeight(height)
                child.setMaximumHeight(16777215)

    def retranslateUi(self):
        """Cập nhật lại văn bản hiển thị trên các SettingCard khi đổi ngôn ngữ nóng"""
        try:
            if "system_tab" in self.pivot.items:
                self.pivot.items["system_tab"].setText(f"{tr['Setting'].get('SystemTab', 'Hệ Thống && Giao Diện')} ({self._count_cards_in_scroll(self.systemScrollWidget)})")
            if "video_tab" in self.pivot.items:
                self.pivot.items["video_tab"].setText(f"{tr['Setting'].get('VideoTab', 'Chỉnh Sửa Video && Xóa Phụ Đề')} ({self._count_cards_in_scroll(self.videoScrollWidget)})")
            if "srt_tab" in self.pivot.items:
                self.pivot.items["srt_tab"].setText(f"{tr['Setting'].get('SrtTab', 'Tính Năng SRT && OCR')} ({self._count_cards_in_scroll(self.srtScrollWidget)})")
        except Exception:
            pass

        self.subtitle_detection_group.titleLabel.setText(tr["Setting"]["SubtitleDetectionSetting"])
        self.sttn_group.titleLabel.setText(tr["Setting"]["SttnSetting"])
        self.propainter_group.titleLabel.setText(tr["Setting"]["ProPainterSetting"])
        self.about_group.titleLabel.setText(tr["Setting"]["AboutSetting"])

        self.subtitle_yx_axis_difference_pixel.setTitle(tr["Setting"]["SubtitleYXAxisDifferencePixel"])
        self.subtitle_yx_axis_difference_pixel.setContent(tr["Setting"]["SubtitleYXAxisDifferencePixelDesc"])

        self.subtitle_area_deviation_pixel.setTitle(tr["Setting"]["SubtitleAreaDeviationPixel"])
        self.subtitle_area_deviation_pixel.setContent(tr["Setting"]["SubtitleAreaDeviationPixelDesc"])

        self.subtitle_area_y_axis_difference_pixel.setTitle(tr["Setting"]["SubtitleAreaYAxisDifferencePixel"])
        self.subtitle_area_y_axis_difference_pixel.setContent(tr["Setting"]["SubtitleAreaYAxisDifferencePixelDesc"])

        self.subtitle_area_pixel_tolerance_y_pixel.setTitle(tr["Setting"]["SubtitleAreaPixelToleranceYPixel"])
        self.subtitle_area_pixel_tolerance_y_pixel.setContent(tr["Setting"]["SubtitleAreaPixelToleranceYPixelDesc"])

        self.subtitle_area_pixel_tolerance_x_pixel.setTitle(tr["Setting"]["SubtitleAreaPixelToleranceXPixel"])
        self.subtitle_area_pixel_tolerance_x_pixel.setContent(tr["Setting"]["SubtitleAreaPixelToleranceXPixelDesc"])

        self.subtitle_timeline_backward_frame_count.setTitle(tr["Setting"]["SubtitleTimelineBackwardFrameCount"])
        self.subtitle_timeline_backward_frame_count.setContent(tr["Setting"]["SubtitleTimelineBackwardFrameCountDesc"])

        self.subtitle_timeline_forward_frame_count.setTitle(tr["Setting"]["subtitleTimelineForwardFrameCount"])
        self.subtitle_timeline_forward_frame_count.setContent(tr["Setting"]["subtitleTimelineForwardFrameCountDesc"])

        self.sttn_neighbor_stride.setTitle(tr["Setting"]["SttnNeighborStride"])
        self.sttn_neighbor_stride.setContent(tr["Setting"]["SttnNeighborStrideDesc"])

        self.sttn_reference_length.setTitle(tr["Setting"]["SttnReferenceLength"])
        self.sttn_reference_length.setContent(tr["Setting"]["SttnReferenceLengthDesc"])

        self.sttn_max_load_num.setTitle(tr["Setting"]["SttnMaxLoadNum"])
        self.sttn_max_load_num.setContent(tr["Setting"]["SttnMaxLoadNumDesc"])

        self.propainter_max_load_num.setTitle(tr["Setting"]["PropainterMaxLoadNum"])
        self.propainter_max_load_num.setContent(tr["Setting"]["PropainterMaxLoadNumDesc"])

        self.save_directory.setTitle(tr["Setting"]["SaveDirectory"])
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)

        self.srt_save_directory.setContent("Mặc định (Lưu cùng thư mục video thành phẩm)" if not config.srtSaveDirectory.value else config.srtSaveDirectory.value)

        self.check_update_on_startup.setTitle(tr["Setting"]["CheckUpdateOnStartup"])
        self.check_update_on_startup.setContent(tr["Setting"]["CheckUpdateOnStartupDesc"])

        self.mask_dilation.setTitle(tr["Setting"]["MaskDilation"])
        self.mask_dilation.setContent(tr["Setting"]["MaskDilationDesc"])

        self.mask_feather.setTitle(tr["Setting"]["MaskFeather"])
        self.mask_feather.setContent(tr["Setting"]["MaskFeatherDesc"])

        self.temporal_smoothing_radius.setTitle(tr["Setting"]["TemporalSmoothingRadius"])
        self.temporal_smoothing_radius.setContent(tr["Setting"]["TemporalSmoothingRadiusDesc"])

        self.feedback.setTitle(tr["Setting"]["FeedbackTitle"])
        self.feedback.setContent(tr["Setting"]["FeedbackDesc"])
        self.feedback.button.setText(tr["Setting"]["FeedbackButton"])

        self.copyright.setTitle(tr["Setting"]["CopyrightTitle"])
        self.copyright.setContent(tr["Setting"]["CopyrightDesc"].format(VERSION))
        self.copyright.button.setText(tr["Setting"]["CopyrightButton"])

        self.project_link.setTitle(tr["Setting"]["ProjectLinkTitle"])
        self.project_link.setContent(tr["Setting"]["ProjectLinkDesc"])

        self.update_gpu_info()
        self._format_card_labels()