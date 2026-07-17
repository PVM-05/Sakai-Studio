"""
@desc: 高级设置页面
"""

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (ScrollArea, ExpandLayout, CardWidget, SubtitleLabel,
                           FluentIcon, NavigationWidget, NavigationItemPosition,
                           SettingCardGroup, RangeSettingCard, SwitchSettingCard,
                           HyperlinkCard, PrimaryPushSettingCard, PushSettingCard,
                           MessageBox)
from backend.config import config, tr, VERSION, PROJECT_HOME_URL, PROJECT_ISSUES_URL, PROJECT_RELEASES_URL
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor

class AdvancedSettingInterface(ScrollArea):
    """高级设置页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self.__init_widgets()

    def __init_widgets(self):
        # 创建滚动内容的容器
        self.scrollWidget = QtWidgets.QWidget(self)
        self.expandLayout = ExpandLayout(self.scrollWidget)
        
        # 设置滚动区域属性
        self.setWidget(self.scrollWidget)
        self.enableTransparentBackground()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # 设置滚动区域样式以适应主题
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        
        # 设置UI
        self.setup_ui()
        self.setup_layout()

    def setup_layout(self):
        self.subtitle_detection_group.addSettingCard(self.subtitle_yx_axis_difference_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_deviation_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_y_axis_difference_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_pixel_tolerance_y_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_area_pixel_tolerance_x_pixel)
        self.subtitle_detection_group.addSettingCard(self.subtitle_timeline_backward_frame_count)
        self.subtitle_detection_group.addSettingCard(self.subtitle_timeline_forward_frame_count)
        self.expandLayout.addWidget(self.subtitle_detection_group)

        self.sttn_group.addSettingCard(self.sttn_neighbor_stride)
        self.sttn_group.addSettingCard(self.sttn_reference_length)
        self.sttn_group.addSettingCard(self.sttn_max_load_num)
        self.expandLayout.addWidget(self.sttn_group)

        self.propainter_group.addSettingCard(self.propainter_max_load_num)
        self.expandLayout.addWidget(self.propainter_group)

        self.advanced_group.addSettingCard(self.save_directory)
        self.advanced_group.addSettingCard(self.check_update_on_startup)
        self.advanced_group.addSettingCard(self.auto_hardware_tuning)
        self.advanced_group.addSettingCard(self.gpu_video_encoding)
        self.advanced_group.addSettingCard(self.mask_dilation)
        self.advanced_group.addSettingCard(self.mask_feather)
        self.advanced_group.addSettingCard(self.temporal_smoothing_radius)
        self.expandLayout.addWidget(self.advanced_group)

        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)
        
        self.expandLayout.addWidget(self.about_group)
        self.expandLayout.setSpacing(16)
        self.expandLayout.setContentsMargins(16, 16, 16, 48)
        
    def setup_ui(self):
        """设置UI"""
        # 字幕检测设置组
        self.subtitle_detection_group = SettingCardGroup(tr["Setting"]["SubtitleDetectionSetting"], self.scrollWidget)
        # STTN设置组
        self.sttn_group = SettingCardGroup(tr["Setting"]["SttnSetting"], self.scrollWidget)
        # Propainter设置组
        self.propainter_group = SettingCardGroup(tr["Setting"]["ProPainterSetting"], self.scrollWidget)
        # 高级设置组
        self.advanced_group = SettingCardGroup(tr["Setting"]["AdvancedSetting"], self.scrollWidget)
        # 关于设置组
        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.scrollWidget)
        
        self.subtitle_yx_axis_difference_pixel = RangeSettingCard(
            configItem=config.subtitleYXAxisDifferencePixel,
            icon=FluentIcon.ZOOM,
            title=tr["Setting"]["SubtitleYXAxisDifferencePixel"],
            content=tr["Setting"]["SubtitleYXAxisDifferencePixelDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_yx_axis_difference_pixel.setToolTip("Độ lệch kích thước tối đa giữa chiều rộng và chiều cao chữ để lọc bỏ các vùng phát hiện nhầm không phải phụ đề.")
        
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
        self.subtitle_timeline_backward_frame_count.setToolTip("Số khung hình mở rộng lùi về phía trước dòng thời gian (mặc định: 3 frames) giúp xử lý triệt để hiệu ứng chữ bắt đầu hiện (fade-in).")

        self.subtitle_timeline_forward_frame_count = RangeSettingCard(
            configItem=config.subtitleTimelineForwardFrameCount,
            icon=FluentIcon.PAGE_RIGHT,
            title=tr["Setting"]["subtitleTimelineForwardFrameCount"],
            content=tr["Setting"]["subtitleTimelineForwardFrameCountDesc"],
            parent=self.subtitle_detection_group
        )
        self.subtitle_timeline_forward_frame_count.setToolTip("Số khung hình mở rộng tiến về phía sau dòng thời gian (mặc định: 3 frames) giúp xử lý triệt để hiệu ứng chữ mờ dần biến mất (fade-out).")

        self.sttn_neighbor_stride = RangeSettingCard(
            configItem=config.sttnNeighborStride,
            icon=FluentIcon.UNIT,
            title=tr["Setting"]["SttnNeighborStride"],
            content=tr["Setting"]["SttnNeighborStrideDesc"],
            parent=self.sttn_group
        )
        self.sttn_neighbor_stride.setToolTip("Bước nhảy khung hình lân cận cho mô hình STTN. Giá trị nhỏ hơn sẽ mượt mà hơn nhưng tính toán chậm hơn.")

        self.sttn_reference_length = RangeSettingCard(
            configItem=config.sttnReferenceLength,
            icon=FluentIcon.MORE,
            title=tr["Setting"]["SttnReferenceLength"],
            content=tr["Setting"]["SttnReferenceLengthDesc"],
            parent=self.sttn_group
        )
        self.sttn_reference_length.setToolTip("Số lượng khung hình tham chiếu dài hạn cho mô hình STTN nhằm đồng bộ chi tiết bối cảnh toàn video.")

        self.sttn_max_load_num = RangeSettingCard(
            configItem=config.sttnMaxLoadNum,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["SttnMaxLoadNum"],
            content=tr["Setting"]["SttnMaxLoadNumDesc"],
            parent=self.sttn_group
        )
        self.sttn_max_load_num.setToolTip("Số khung hình tối đa xử lý đồng thời trong một phân đoạn của STTN. Chỉ có hiệu lực khi tắt tính năng Tự động tối ưu GPU.")

        self.propainter_max_load_num = RangeSettingCard(
            configItem=config.propainterMaxLoadNum,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["PropainterMaxLoadNum"],
            content=tr["Setting"]["PropainterMaxLoadNumDesc"],
            parent=self.propainter_group
        )
        self.propainter_max_load_num.setToolTip("Số khung hình tối đa xử lý đồng thời trong một phân đoạn của ProPainter. Chỉ có hiệu lực khi tắt tính năng Tự động tối ưu GPU.")

        # 视频保存路径
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.DOWNLOAD,
            title=tr["Setting"]["SaveDirectory"],
            content=tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.advanced_group
        )
        self.save_directory.setToolTip("Chọn đường dẫn thư mục mặc định để xuất và lưu trữ video sau khi xử lý thành công.")
        self.save_directory.clicked.connect(self.choose_save_directory)

        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup,
            icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"],
            parent=self.advanced_group
        )
        self.check_update_on_startup.setToolTip("Tự động kiểm tra bản cập nhật mới từ GitHub mỗi khi bạn khởi động ứng dụng.")

        self.auto_hardware_tuning = SwitchSettingCard(
            configItem=config.autoHardwareTuning,
            icon=FluentIcon.SPEED_HIGH,
            title="Tối ưu hóa hiệu năng theo GPU (Auto Hardware Tuning)",
            content="Tự động tính toán số frame xử lý tối ưu để tránh tràn VRAM GPU",
            parent=self.advanced_group
        )
        self.auto_hardware_tuning.setToolTip("Tự động phân tích dung lượng bộ nhớ VRAM thực tế trên card màn hình của bạn để tính toán số khung hình xử lý đồng thời tối ưu nhất, ngăn chặn lỗi tràn VRAM (Out of Memory).")
        self.auto_hardware_tuning.switchButton.checkedChanged.connect(self.on_auto_tuning_changed)
        QtCore.QTimer.singleShot(0, lambda: self.on_auto_tuning_changed(config.autoHardwareTuning.value))

        self.gpu_video_encoding = SwitchSettingCard(
            configItem=config.gpuVideoEncoding,
            icon=FluentIcon.VIDEO,
            title="Tăng tốc xuất video bằng GPU (GPU Hardware Encoding)",
            content="Sử dụng bộ giải mã/mã hóa phần cứng GPU giúp xuất video nhanh gấp 5-10 lần",
            parent=self.advanced_group
        )
        self.gpu_video_encoding.setToolTip("Kích hoạt chip phần cứng chuyên dụng trên Card NVIDIA (NVENC) để nén và giải nén video trực tiếp, tăng tốc xuất video thành phẩm lên gấp 5 đến 10 lần mà không tiêu tốn tài nguyên CPU.")

        self.mask_dilation = RangeSettingCard(
            configItem=config.maskDilation,
            icon=FluentIcon.ZOOM_IN,
            title="Độ giãn nở mặt nạ (Mask Dilation)",
            content="Độ rộng phần giãn nở bao phủ xung quanh viền phụ đề (0-50 px)",
            parent=self.advanced_group
        )
        self.mask_dilation.setToolTip("Độ giãn nở ra phía ngoài của mặt nạ chữ. Nếu video bị sót lại viền đen của chữ hoặc bóng mờ sau khi xóa, hãy tăng thông số này lên (khuyên dùng: 8-12px).")

        self.mask_feather = RangeSettingCard(
            configItem=config.maskFeather,
            icon=FluentIcon.BRUSH,
            title="Làm mềm biên mặt nạ (Mask Feathering)",
            content="Độ làm mềm mượt biên mặt nạ tránh gãy pixel răng cưa (0-30 px)",
            parent=self.advanced_group
        )
        self.mask_feather.setToolTip("Độ mờ nhòe biên mặt nạ (Alpha Feathering). Giúp làm mượt chuyển tiếp giữa khu vực được lấp đầy nền mới và video gốc, loại bỏ vệt cắt cứng sắc cạnh.")

        self.temporal_smoothing_radius = RangeSettingCard(
            configItem=config.temporalSmoothingRadius,
            icon=FluentIcon.MOVIE,
            title="Bán kính làm mịn thời gian (Temporal Radius)",
            content="Bán kính khung hình lân cận để lọc mượt chống nhấp nháy chuyển động (1-10)",
            parent=self.advanced_group
        )
        self.temporal_smoothing_radius.setToolTip("Bán kính số khung hình liền trước/sau được lấy làm tham chiếu để tính toán mượt hóa thời gian. Số càng lớn lọc nhấp nháy càng tốt nhưng có thể gây bóng mờ trên cảnh chuyển động rất nhanh.")

        # 添加反馈链接
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"],
            content=tr["Setting"]["FeedbackDesc"],
            parent=self.about_group
        )
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl(PROJECT_ISSUES_URL)
        ))
        # 添加版权信息
        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["CopyrightTitle"],
            content=tr["Setting"]["CopyrightDesc"].format(VERSION),
            parent=self.about_group
        )
        self.copyright.clicked.connect(lambda: self.check_update())
        # 添加项目链接
        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL,
            text=PROJECT_HOME_URL,
            icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"],
            content=tr["Setting"]["ProjectLinkDesc"],
            parent=self.about_group
        )

        # Cho phép các nhãn mô tả tự động xuống dòng và co giãn chiều cao theo độ dài chữ
        from PySide6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)
                content_text = child.contentLabel.text()
                if content_text:
                    height = 90 if len(content_text) > 40 else 70
                else:
                    height = 50
                child.setMinimumHeight(height)
                child.setMaximumHeight(height)

    def show_message_box(self, title: str, content: str, showYesButton=False, yesSlot=None):
        """ show message box """
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr('Close'))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        """ check software update

        Parameters
        ----------
        ignore: bool
            ignore message box when no updates are available
        """
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
        """选择保存目录"""
        last_save_directory = "./" if not config.saveDirectory.value else config.saveDirectory.value
        folder = QFileDialog.getExistingDirectory(
            self, tr['Setting']['ChooseDirectory'], last_save_directory)
        if not folder:
            folder = ""

        config.set(config.saveDirectory, folder)
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)

    def on_auto_tuning_changed(self, checked):
        """Khi bật/tắt tự động tinh chỉnh phần cứng, ta vô hiệu hóa/kích hoạt các thanh trượt thủ công"""
        self.sttn_max_load_num.setEnabled(not checked)
        self.propainter_max_load_num.setEnabled(not checked)

    def retranslateUi(self):
        """Cập nhật lại văn bản hiển thị trên các SettingCard khi đổi ngôn ngữ nóng"""
        self.subtitle_detection_group.titleLabel.setText(tr["Setting"]["SubtitleDetectionSetting"])
        self.sttn_group.titleLabel.setText(tr["Setting"]["SttnSetting"])
        self.propainter_group.titleLabel.setText(tr["Setting"]["ProPainterSetting"])
        self.advanced_group.titleLabel.setText(tr["Setting"]["AdvancedSetting"])
        self.about_group.titleLabel.setText(tr["Setting"]["AboutSetting"])

        self.subtitle_yx_axis_difference_pixel.setTitle(tr["Setting"]["SubtitleYXAxisDifferencePixel"])
        self.subtitle_yx_axis_difference_pixel.setContent(tr["Setting"]["SubtitleYXAxisDifferencePixelDesc"])
        self.subtitle_yx_axis_difference_pixel.setToolTip("Độ lệch kích thước tối đa giữa chiều rộng và chiều cao chữ để lọc bỏ các vùng phát hiện nhầm không phải phụ đề.")

        self.subtitle_area_deviation_pixel.setTitle(tr["Setting"]["SubtitleAreaDeviationPixel"])
        self.subtitle_area_deviation_pixel.setContent(tr["Setting"]["SubtitleAreaDeviationPixelDesc"])
        self.subtitle_area_deviation_pixel.setToolTip("Độ lệch tối đa của vùng phụ đề cho phép để tránh cắt lẹm vào biên vùng chữ.")

        self.subtitle_area_y_axis_difference_pixel.setTitle(tr["Setting"]["SubtitleAreaYAxisDifferencePixel"])
        self.subtitle_area_y_axis_difference_pixel.setContent(tr["Setting"]["SubtitleAreaYAxisDifferencePixelDesc"])
        self.subtitle_area_y_axis_difference_pixel.setToolTip("Độ lệch tối đa theo trục Y để gộp nhóm các dòng phụ đề xuất hiện đồng thời.")

        self.subtitle_area_pixel_tolerance_y_pixel.setTitle(tr["Setting"]["SubtitleAreaPixelToleranceYPixel"])
        self.subtitle_area_pixel_tolerance_y_pixel.setContent(tr["Setting"]["SubtitleAreaPixelToleranceYPixelDesc"])
        self.subtitle_area_pixel_tolerance_y_pixel.setToolTip("Dung sai sai lệch dòng theo trục Y khi xác định vị trí phụ đề ổn định theo chiều dọc.")

        self.subtitle_area_pixel_tolerance_x_pixel.setTitle(tr["Setting"]["SubtitleAreaPixelToleranceXPixel"])
        self.subtitle_area_pixel_tolerance_x_pixel.setContent(tr["Setting"]["SubtitleAreaPixelToleranceXPixelDesc"])
        self.subtitle_area_pixel_tolerance_x_pixel.setToolTip("Dung sai sai lệch dòng theo trục X khi xác định vùng ngang chứa phụ đề ổn định.")

        self.subtitle_timeline_backward_frame_count.setTitle(tr["Setting"]["SubtitleTimelineBackwardFrameCount"])
        self.subtitle_timeline_backward_frame_count.setContent(tr["Setting"]["SubtitleTimelineBackwardFrameCountDesc"])
        self.subtitle_timeline_backward_frame_count.setToolTip("Số khung hình mở rộng lùi về phía trước dòng thời gian (mặc định: 3 frames) giúp xử lý triệt để hiệu ứng chữ bắt đầu hiện (fade-in).")

        self.subtitle_timeline_forward_frame_count.setTitle(tr["Setting"]["subtitleTimelineForwardFrameCount"])
        self.subtitle_timeline_forward_frame_count.setContent(tr["Setting"]["subtitleTimelineForwardFrameCountDesc"])
        self.subtitle_timeline_forward_frame_count.setToolTip("Số khung hình mở rộng tiến về phía sau dòng thời gian (mặc định: 3 frames) giúp xử lý triệt để hiệu ứng chữ mờ dần biến mất (fade-out).")

        self.sttn_neighbor_stride.setTitle(tr["Setting"]["SttnNeighborStride"])
        self.sttn_neighbor_stride.setContent(tr["Setting"]["SttnNeighborStrideDesc"])
        self.sttn_neighbor_stride.setToolTip("Bước nhảy khung hình lân cận cho mô hình STTN. Giá trị nhỏ hơn sẽ mượt mà hơn nhưng tính toán chậm hơn.")

        self.sttn_reference_length.setTitle(tr["Setting"]["SttnReferenceLength"])
        self.sttn_reference_length.setContent(tr["Setting"]["SttnReferenceLengthDesc"])
        self.sttn_reference_length.setToolTip("Số lượng khung hình tham chiếu dài hạn cho mô hình STTN nhằm đồng bộ chi tiết bối cảnh toàn video.")

        self.sttn_max_load_num.setTitle(tr["Setting"]["SttnMaxLoadNum"])
        self.sttn_max_load_num.setContent(tr["Setting"]["SttnMaxLoadNumDesc"])
        self.sttn_max_load_num.setToolTip("Số khung hình tối đa xử lý đồng thời trong một phân đoạn của STTN. Chỉ có hiệu lực khi tắt tính năng Tự động tối ưu GPU.")

        self.propainter_max_load_num.setTitle(tr["Setting"]["PropainterMaxLoadNum"])
        self.propainter_max_load_num.setContent(tr["Setting"]["PropainterMaxLoadNumDesc"])
        self.propainter_max_load_num.setToolTip("Số khung hình tối đa xử lý đồng thời trong một phân đoạn của ProPainter. Chỉ có hiệu lực khi tắt tính năng Tự động tối ưu GPU.")

        self.save_directory.setTitle(tr["Setting"]["SaveDirectory"])
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)
        self.save_directory.setToolTip("Chọn đường dẫn thư mục mặc định để xuất và lưu trữ video sau khi xử lý thành công.")

        self.check_update_on_startup.setTitle(tr["Setting"]["CheckUpdateOnStartup"])
        self.check_update_on_startup.setContent(tr["Setting"]["CheckUpdateOnStartupDesc"])
        self.check_update_on_startup.setToolTip("Tự động kiểm tra bản cập nhật mới từ GitHub mỗi khi bạn khởi động ứng dụng.")

        self.mask_dilation.setTitle(tr["Setting"]["MaskDilation"])
        self.mask_dilation.setContent(tr["Setting"]["MaskDilationDesc"])
        self.mask_dilation.setToolTip(tr["Setting"]["MaskDilationTooltip"])

        self.mask_feather.setTitle(tr["Setting"]["MaskFeather"])
        self.mask_feather.setContent(tr["Setting"]["MaskFeatherDesc"])
        self.mask_feather.setToolTip(tr["Setting"]["MaskFeatherTooltip"])

        self.temporal_smoothing_radius.setTitle(tr["Setting"]["TemporalSmoothingRadius"])
        self.temporal_smoothing_radius.setContent(tr["Setting"]["TemporalSmoothingRadiusDesc"])
        self.temporal_smoothing_radius.setToolTip(tr["Setting"]["TemporalSmoothingRadiusTooltip"])

        self.feedback.setTitle(tr["Setting"]["FeedbackTitle"])
        self.feedback.setContent(tr["Setting"]["FeedbackDesc"])
        self.feedback.button.setText(tr["Setting"]["FeedbackButton"])

        self.copyright.setTitle(tr["Setting"]["CopyrightTitle"])
        self.copyright.setContent(tr["Setting"]["CopyrightDesc"].format(VERSION))
        self.copyright.button.setText(tr["Setting"]["CopyrightButton"])

        self.project_link.setTitle(tr["Setting"]["ProjectLinkTitle"])
        self.project_link.setContent(tr["Setting"]["ProjectLinkDesc"])

        # Cập nhật lại chiều cao các thẻ cài đặt sau khi đổi ngôn ngữ
        from PySide6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if hasattr(child, 'contentLabel') and hasattr(child, 'titleLabel'):
                child.contentLabel.setWordWrap(True)
                child.titleLabel.setWordWrap(True)
                content_text = child.contentLabel.text()
                if content_text:
                    height = 90 if len(content_text) > 40 else 70
                else:
                    height = 50
                child.setMinimumHeight(height)
                child.setMaximumHeight(height)