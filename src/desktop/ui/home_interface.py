import os
import cv2
import numpy as np
import json
from pathlib import Path
import threading
import multiprocessing
import time
import traceback
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Slot, QRect, Signal, Qt, QThread, QTimer
from PySide6 import QtWidgets, QtCore
from datetime import datetime
from qfluentwidgets import (PushButton, PrimaryPushButton, CardWidget, TextEdit, FluentIcon, InfoBar, InfoBarPosition, ScrollArea, SubtitleLabel, CaptionLabel)
from src.desktop.ui.setting_interface import SettingInterface
from src.desktop.ui.component.video_display_component import VideoDisplayComponent
from src.desktop.ui.component.task_list_component import TaskListComponent, TaskStatus, TaskOptions
from src.desktop.ui.icon.my_fluent_icon import MyFluentIcon
from src.core.config import config, tr
from src.core.tools.constant import InpaintMode, SubtitleDetectMode
from src.core.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from src.core.tools.process_manager import ProcessManager
from src.core.tools.common_tools import get_readable_path, is_image_file, read_image
from src.core.tools.subtitle_exporter import SubtitleExporter
from src.ai_engines.ocr_engine import VideoOcrEngine





# FastSrtExtractThread has been refactored and moved to ui/extractor_interface.py to prevent code duplication.

class HomeInterface(QWidget):
    progress_signal = Signal(int, bool, int)
    append_log_signal = Signal(list)
    update_preview_with_comp_signal = Signal(list)
    task_error_signal = Signal(object)
    toggle_buttons_signal = Signal(bool)  # True=显示运行按钮, False=显示停止按钮
    task_status_signal = Signal(int, object)  # (task_index, TaskStatus)
    select_task_signal = Signal(int)  # task_index
    # Tín hiệu trả kết quả inpaint preview từ background thread về UI thread
    mask_preview_result_signal = Signal(object, object)  # (result_frame, error_info)
    # Tín hiệu trả khung hình tua bất đồng bộ (background thread -> UI thread)
    async_seek_signal = Signal(object, int)
    # Tín hiệu trả kết quả nhận diện chữ AI (background thread -> UI thread)
    auto_detect_result_signal = Signal(object)
    track_motion_finished_signal = Signal(object)
    srt_exported_signal = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("HomeInterface")
        # 初始化一些变量
        self.video_path = None
        self.video_cap = None
        self.fps = None
        self.frame_count = None
        self.frame_width = None
        self.frame_height = None
        self.se = None  # 后台字幕提取器

        # 字幕区域参数
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        # 添加自动滚动控制标志
        self.auto_scroll = True
        self._stop_event = threading.Event()  # 线程安全的停止信号
        self._worker_thread = None
        self.running_process = None
        self._saved_inpaint_mode = None  # 保存图片锁定前的 inpaint 模式
        self._video_cap_lock = threading.Lock()  # 保护 video_cap 的线程锁
        self.is_queue_paused = False  # Trạng thái tạm dừng hàng đợi

        # 当前正在处理的任务索引
        self.current_processing_task_index = -1
        self.last_exported_srt_path = None

        # Biến trạng thái tua mượt liên tục & RAM Frame Cache
        self._is_seeking_frame = False
        self._current_seeking_value = None
        self._cache_lock = threading.RLock()
        self._frame_cache = {}  # {frame_no: frame_ndarray}
        self._preload_thread = None
        self._preload_cancel = None
        self._cached_rapid_engine = None
        self._cached_paddle_engine = None

        self.__init_widgets()
        self.progress_signal.connect(self.update_progress)
        self.append_log_signal.connect(self.append_log)
        self.update_preview_with_comp_signal.connect(self.update_preview_with_comp)
        self.task_error_signal.connect(self.on_task_error)
        self.toggle_buttons_signal.connect(self._toggle_buttons)
        self.task_status_signal.connect(lambda idx, status: self.task_list_component.update_task_status(idx, status))
        self.select_task_signal.connect(self.task_list_component.select_task)
        # Kết nối signal trả kết quả xem trước (background thread → UI thread)
        self.mask_preview_result_signal.connect(self._on_mask_preview_result)
        self.async_seek_signal.connect(self._apply_seek_frame)
        self.auto_detect_result_signal.connect(self._apply_auto_detected_rects)
        self.track_motion_finished_signal.connect(self._on_track_motion_finished)
        self.srt_exported_signal.connect(self._on_srt_exported)

    def __init_widgets(self):
        """创建主页面"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 左侧视频区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        
        # 创建视频显示组件
        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.set_player_mode("remover")
        self.video_display_component.ab_sections_changed.connect(self.ab_sections_changed)
        self.video_display_component.selections_changed.connect(self.selections_changed)
        self.video_display_component.auto_detect_clicked.connect(self._on_auto_detect_clicked)
        left_layout.addWidget(self.video_display_component)
        
        # Kết nối tín hiệu thanh trượt đỏ (kéo chuột liên tục chuẩn YouTube)
        self.video_display = self.video_display_component.video_display
        self.video_slider = self.video_display_component.video_slider
        self.video_slider.valueChanged.connect(self.slider_changed)
        self.video_slider.sliderMoved.connect(self.slider_changed)
        self.video_slider.sliderReleased.connect(lambda: self.slider_changed(self.video_slider.value()))
        
        # 输出文本区域
        self.output_text = TextEdit()
        self.output_text.setMinimumHeight(150)
        self.output_text.setReadOnly(True)
        self.output_text.document().setDocumentMargin(10)        
        # 连接滚动条值变化信号
        self.output_text.verticalScrollBar().valueChanged.connect(self.on_scroll_change)
        
        # Hộp chứa log đã được gỡ bỏ theo yêu cầu
        # output_container = CardWidget(self)
        # output_layout = QVBoxLayout()
        # output_layout.setContentsMargins(0, 0, 0, 0)
        # output_layout.addWidget(self.output_text)
        # output_container.setLayout(output_layout)
        # left_layout.addWidget(output_container)

        main_layout.addLayout(left_layout, 2)

        # 右侧设置区域
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # 设置容器 (Bảng cài đặt nền trắng gốc sạch đẹp)
        settings_container = CardWidget(self)
        self.setting_interface = SettingInterface(settings_container)
        settings_container.setLayout(self.setting_interface)
        
        # Bọc SettingInterface vào ScrollArea để tránh vỡ UI khi màn hình nhỏ
        scroll_area = ScrollArea(self)
        scroll_area.setWidget(settings_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        right_layout.addWidget(scroll_area)
        
        # Kết nối các công cụ từ SettingInterface (bên phải) sang VideoDisplayComponent
        if hasattr(self.setting_interface, 'auto_detect_frame_btn'):
            self.setting_interface.auto_detect_frame_btn.clicked.connect(self._on_auto_detect_clicked)
        elif hasattr(self.setting_interface, 'auto_detect_card'):
            self.setting_interface.auto_detect_card.clicked.connect(self._on_auto_detect_clicked)
            
        if hasattr(self.setting_interface, 'moving_subtitle_card'):
            self.setting_interface.moving_subtitle_card.clicked.connect(self._on_track_motion_clicked)

        if hasattr(self.setting_interface, 'brush_mode_btn'):
            self.video_display_component.brush_mode_btn = self.setting_interface.brush_mode_btn
            self.setting_interface.brush_mode_btn.clicked.connect(self.video_display_component.toggle_draw_mode)
        if hasattr(self.setting_interface, 'red_mask_btn'):
            self.video_display_component.red_mask_btn = self.setting_interface.red_mask_btn
            self.setting_interface.red_mask_btn.clicked.connect(self.video_display_component.toggle_red_mask)
        if hasattr(self.setting_interface, 'clear_brush_btn'):
            self.setting_interface.clear_brush_btn.clicked.connect(self.video_display_component.clear_freehand_strokes)
        
        # Kết nối sự kiện bật/tắt công tắc Tự động (Tracking)
        if getattr(config, 'movingSubtitleTracking', None):
            config.movingSubtitleTracking.valueChanged.connect(self._update_manual_tools_state)
        QtCore.QTimer.singleShot(100, lambda: self._update_manual_tools_state(None))
        
        # 添加任务列表容器
        task_list_container = CardWidget(self)
        task_list_layout = QHBoxLayout()
        task_list_layout.setContentsMargins(0, 0, 0, 0)
        task_list_layout.setSpacing(0)
        self.task_list_component = TaskListComponent(self)
        self.task_list_component.task_selected.connect(self.on_task_selected)
        self.task_list_component.task_deleted.connect(self.on_task_deleted)
        task_list_layout.addWidget(self.task_list_component)
        task_list_container.setLayout(task_list_layout)
        right_layout.addWidget(task_list_container, 1)  # 占满剩余空间
        
        # Thẻ chứa các nút bấm hành động (Layout 2 Cột Cân Bằng, Thiết Kế Chuẩn Fluent UI)
        button_container = CardWidget(self)
        button_layout = QGridLayout(button_container)
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(10)
        
        # Hàng 0: Mở Video & Thêm Vùng Khoanh Chọn
        self.file_button = PushButton(tr['SubtitleExtractorGUI']['Open'], self)
        self.file_button.setIcon(FluentIcon.FOLDER)
        self.file_button.clicked.connect(self.open_file)
        button_layout.addWidget(self.file_button, 0, 0)
        
        self.add_area_button = PushButton(tr['Setting']['AddArea'], self)
        self.add_area_button.setIcon(FluentIcon.ADD)
        self.add_area_button.setToolTip(tr['Setting']['AddAreaTooltip'])
        self.add_area_button.clicked.connect(self.add_area_button_clicked)
        button_layout.addWidget(self.add_area_button, 0, 1)
        
        # Đã chuyển nút Nhận Diện Chuyển Động Logo sang SettingInterface (bên phải)
        
        # Hàng 2: Xem Trước Mask & Bắt Đầu Xóa (Primary Accent Button)
        self.mask_preview_button = PushButton(tr['Setting']['MaskPreview'], self)
        self.mask_preview_button.setIcon(FluentIcon.VIEW)
        self.mask_preview_button.setToolTip(tr['Setting']['MaskPreviewTooltip'])
        self.mask_preview_button.clicked.connect(self.mask_preview_button_clicked)
        self.mask_preview_button.pressed.connect(self._on_mask_preview_pressed)
        self.mask_preview_button.released.connect(self._on_mask_preview_released)
        button_layout.addWidget(self.mask_preview_button, 2, 0)
        
        self.run_button = PrimaryPushButton(tr['SubtitleExtractorGUI']['Run'], self)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.clicked.connect(self.run_button_clicked)
        button_layout.addWidget(self.run_button, 2, 1)
        
        self.stop_button = PushButton(tr['SubtitleExtractorGUI']['Stop'], self)
        self.stop_button.setIcon(MyFluentIcon.Stop)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        button_layout.addWidget(self.stop_button, 2, 1)
        
        self.pause_resume_button = PushButton(tr['Setting'].get('PauseQueue', "Tạm dừng hàng đợi"), self)
        self.pause_resume_button.setIcon(FluentIcon.PAUSE)
        self.pause_resume_button.setVisible(False)
        self.pause_resume_button.clicked.connect(self.pause_resume_button_clicked)
        button_layout.addWidget(self.pause_resume_button, 0, 0)
        
        # Hàng 3: Lưu Cài Đặt (Phủ ngang 2 Cột)
        self.btn_save_config = PushButton("Lưu Cài Đặt", self)
        self.btn_save_config.setIcon(FluentIcon.SAVE)
        self.btn_save_config.clicked.connect(self.save_remover_config)
        button_layout.addWidget(self.btn_save_config, 3, 0, 1, 2)

        right_layout.addWidget(button_container)

        main_layout.addLayout(right_layout, 1)
    
    def save_remover_config(self):
        try:
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

            display_names = {
                "sttn_auto": "STTN Auto (Khuyên dùng)",
                "lama": "Lama Inpaint",
                "propainter": "Propainter",
                "opencv": "OpenCV Rapid"
            }
            mode_label = display_names.get(mode_code, mode_code)

            InfoBar.success(
                "Đã Lưu Cài Đặt Xóa Sub",
                f"Mô hình xóa: {mode_label}. Tab Tự Động đã được đồng bộ!",
                position=InfoBarPosition.TOP, parent=self,
                duration=3500
            )
        except Exception as e:
            InfoBar.error("Lỗi", f"Không thể lưu cài đặt xóa sub: {e}", position=InfoBarPosition.TOP, parent=self, duration=3000)

    def on_scroll_change(self, value):
        """监控滚动条位置变化"""
        scrollbar = self.output_text.verticalScrollBar()
        # 如果滚动到底部，启用自动滚动
        if value == scrollbar.maximum():
            self.auto_scroll = True
        # 如果用户向上滚动，禁用自动滚动
        elif self.auto_scroll and value < scrollbar.maximum():
            self.auto_scroll = False

    
    def slider_changed(self, value):
        # Reset preview state khi người dùng tua video
        self._preview_original_frame = None
        self._preview_inpainted_frame = None
        if hasattr(self, 'mask_preview_button') and self.mask_preview_button.text() != tr['Setting']['MaskPreview']:
            self.mask_preview_button.setText(tr['Setting']['MaskPreview'])
            self.mask_preview_button.setToolTip(tr['Setting']['MaskPreviewTooltip'])

        # Nếu video đang PLAY (đang phát), không giải mã bằng OpenCV CPU để tránh xung đột với QMediaPlayer
        if hasattr(self, 'video_display_component') and hasattr(self.video_display_component, 'media_player'):
            from PySide6.QtMultimedia import QMediaPlayer
            if getattr(self.video_display_component.media_player, 'playbackState', None):
                if self.video_display_component.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    return

        # Lưu vị trí tua cần chuyển tới
        self._pending_seek_value = value

        # Cập nhật ngay giao diện đồng hồ thời gian & Đồng bộ GPU Hardware Video Player (1ms YouTube-style Hardware Seek)
        if hasattr(self, 'video_display_component'):
            self.video_display_component.update_time_display()
            self.video_display_component.sync_audio_position(value)

        # 🚀 KIỂM TRA RAM FRAME CACHE HỖ TRỢ TUA TỨC THÌ (0ms Latency Hit)
        exact_hit = False
        if hasattr(self, '_frame_cache') and self._frame_cache:
            with self._cache_lock:
                if value in self._frame_cache:
                    self._apply_seek_frame(self._frame_cache[value], value, is_exact=True)
                    exact_hit = True
                else:
                    # Hiển thị tạm frame gần nhất nếu khoảng cách ngắn (<= 25 frames) để UI luôn phản hồi 0ms
                    keys = self._frame_cache.keys()
                    nearest_k = min(keys, key=lambda k: abs(k - value))
                    if abs(nearest_k - value) <= 25:
                        self._apply_seek_frame(self._frame_cache[nearest_k], value, is_exact=False)

        # Nếu chưa có exact frame trong RAM cache -> Kích hoạt giải mã ngầm cho đúng frame 'value'!
        if not exact_hit:
            self._do_async_seek()

    def _do_async_seek(self):
        target_value = getattr(self, '_pending_seek_value', None)
        if target_value is None:
            return

        # Nếu frame đã có trong RAM cache -> Hiển thị 0ms tức thì
        if hasattr(self, '_frame_cache') and target_value in self._frame_cache:
            self._apply_seek_frame(self._frame_cache[target_value], target_value, is_exact=True)

        if getattr(self, '_is_seeking_frame', False):
            return
            
        self._is_seeking_frame = True

        def _bg_seek_worker():
            while True:
                # Lấy vị trí target_value mới nhất mà con trỏ chuột đang chỉ tới
                cur_target = getattr(self, '_pending_seek_value', None)
                if cur_target is None:
                    break

                # Nếu frame này đã sẵn sàng trong RAM cache -> Đưa ra UI ngay
                if hasattr(self, '_frame_cache') and self._frame_cache:
                    with self._cache_lock:
                        if cur_target in self._frame_cache:
                            self.async_seek_signal.emit(self._frame_cache[cur_target], cur_target)
                            if getattr(self, '_pending_seek_value', None) == cur_target:
                                break
                            continue

                frame = None
                try:
                    with self._video_cap_lock:
                        if self.video_cap is not None and self.video_cap.isOpened():
                            target_frame_idx = max(0, cur_target - 1)
                            try:
                                current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                            except Exception:
                                current_pos = -1

                            diff = target_frame_idx - current_pos
                            # Tua ngắn (1-60 frames) dùng read() siêu tốc <4ms per batch
                            if 0 <= diff <= 60:
                                for step in range(1, max(1, diff + 1)):
                                    ret, read_frame = self.video_cap.read()
                                    if not ret:
                                        break
                                    # Lưu tất cả các frame trung gian được đọc vào RAM cache để các cú nhích tiếp theo tua 0ms tức thì
                                    frame_no = current_pos + step + 1
                                    if hasattr(self, '_frame_cache'):
                                        with self._cache_lock:
                                            self._frame_cache[frame_no] = read_frame.copy()
                                    frame = read_frame
                            else:
                                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                                ret, frame = self.video_cap.read()
                                if ret and hasattr(self, '_frame_cache'):
                                    with self._cache_lock:
                                        self._frame_cache[cur_target] = frame.copy()

                            if not ret:
                                frame = None
                except Exception:
                    pass
                
                # Gửi frame giải mã về UI thread để hiển thị lập tức
                self.async_seek_signal.emit(frame if frame is not None else -1, cur_target)

                # Nếu con trỏ chuột dừng kéo (vị trí target không đổi) -> Hoàn tất tua, thoát luồng
                if getattr(self, '_pending_seek_value', None) == cur_target:
                    break

            self._is_seeking_frame = False

        threading.Thread(target=_bg_seek_worker, daemon=True).start()

    @Slot(object, int)
    def _apply_seek_frame(self, frame, value, is_exact=True):
        if frame is not None and isinstance(frame, np.ndarray):
            # Chỉ lưu vào RAM Cache khi là frame chính xác vừa giải mã
            if is_exact and hasattr(self, '_frame_cache'):
                with self._cache_lock:
                    self._frame_cache[value] = frame.copy()
                    if len(self._frame_cache) > 800:
                        oldest_keys = list(self._frame_cache.keys())[:150]
                        for k in oldest_keys:
                            self._frame_cache.pop(k, None)

            self.video_display_component.set_dragger_enabled(True)

            # Cập nhật ô khoanh vùng bám đuổi logo di chuyển theo thời gian thực khi rê timeline
            if hasattr(self, '_tracked_sub_list') and self._tracked_sub_list and value in self._tracked_sub_list:
                video_rects = self._tracked_sub_list[value]
                display_rects = [(r[2], r[3], r[0], r[1]) for r in video_rects]
                if hasattr(self.video_display_component, 'selection_rects'):
                    preview_rects = self.video_display_component.video_coordinates_to_preview_coordinates(display_rects)
                    self.video_display_component.selection_rects = preview_rects

            self.update_preview(frame)
            # Ép cập nhật lại màn hình hiển thị tức thì theo vị trí con trỏ chuột
            if hasattr(self.video_display_component, 'video_display') and self.video_display_component.video_display:
                self.video_display_component.video_display.repaint()

        if hasattr(self, 'video_display_component'):
            self.video_display_component.sync_audio_position(value)

    def _on_track_motion_clicked(self):
        if getattr(self, '_is_tracked', False):
            # Hủy bám đuổi
            self._is_tracked = False
            self._tracked_sub_list = {}
            get_current_task_index = self.task_list_component.get_current_task_index()
            if get_current_task_index >= 0:
                self.task_list_component.update_task_option(get_current_task_index, 'tracked_sub_list', {})
            config.set(config.movingSubtitleTracking, False)
            if hasattr(self.setting_interface, 'moving_subtitle_card'):
                self.setting_interface.moving_subtitle_card.setText("Nhận Diện Logo")
            
            # Trả lại box rỗng hoặc cũ
            self.update_preview(self.current_frame)
            
            InfoBar.success(
                title="Đã hủy bám đuổi",
                content="Đã xóa dữ liệu bám đuổi chuyển động cho video này.",
                position=InfoBarPosition.TOP, parent=self
            )
            return

        if not self.video_path or not os.path.exists(self.video_path):
            InfoBar.warning(
                title="Chưa mở video",
                content="Vui lòng mở một tệp video trước khi thực hiện nhận diện chuyển động!",
                position=InfoBarPosition.TOP, parent=self
            )
            return

        # Lấy vùng khoanh chọn từ video_display_component nếu sub_areas chưa đồng bộ
        if hasattr(self.video_display_component, 'selection_rects') and self.video_display_component.selection_rects:
            self.sub_areas = self.video_display_component.preview_coordinates_to_video_coordinates(self.video_display_component.selection_rects)

        is_invalid_area = False
        if not hasattr(self, 'sub_areas') or not self.sub_areas or len(self.sub_areas) == 0:
            is_invalid_area = True
        elif len(self.sub_areas) == 1 and hasattr(self, 'frame_width') and self.frame_width:
            ymin, ymax, xmin, xmax = self.sub_areas[0]
            if xmin == 0 and ymin == 0 and xmax >= self.frame_width - 1 and ymax >= self.frame_height - 1:
                is_invalid_area = True

        if is_invalid_area:
            InfoBar.warning(
                title="Bắt buộc chọn vùng logo di chuyển",
                content="Bạn phải dùng chuột khoanh vùng chứa logo di chuyển trên khung hình trước khi bấm 'Nhận Diện'!",
                position=InfoBarPosition.TOP, parent=self,
                duration=5000
            )
            return

        if hasattr(self.setting_interface, 'moving_subtitle_card'):
            self.setting_interface.moving_subtitle_card.setEnabled(False)
            self.setting_interface.moving_subtitle_card.setText("Đang theo dõi...")

        InfoBar.info(
            title="Đang nhận diện chuyển động logo",
            content="Hệ thống đang tự động bám đuổi vị trí logo qua tất cả khung hình trong video...",
            duration=4000,
            position=InfoBarPosition.TOP, parent=self
        )

        current_frame = 1
        if hasattr(self.video_display_component, 'video_slider'):
            current_frame = self.video_display_component.video_slider.value()

        def _bg_track():
            try:
                from src.core.tools.object_tracker import ObjectTracker
                tracker = ObjectTracker(self.video_path, self.sub_areas, start_frame=current_frame)
                tracked_dict = tracker.find_subtitle_frame_no(sub_remover=self)
                self.track_motion_finished_signal.emit(tracked_dict)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Lỗi nhận diện chuyển động: {e}")
                self.track_motion_finished_signal.emit({})

        threading.Thread(target=_bg_track, daemon=True).start()

    @Slot(object)
    def _on_track_motion_finished(self, tracked_dict):
        if tracked_dict and len(tracked_dict) > 0:
            self._is_tracked = True
            if hasattr(self.video_display_component, 'set_tracked_dict'):
                self.video_display_component.set_tracked_dict(tracked_dict)
            if hasattr(self.setting_interface, 'moving_subtitle_card'):
                self.setting_interface.moving_subtitle_card.setEnabled(True)
                self.setting_interface.moving_subtitle_card.setText("Đã Nhận Diện (Hủy)")

            self._tracked_sub_list = tracked_dict
            config.set(config.movingSubtitleTracking, True)

            get_current_task_index = self.task_list_component.get_current_task_index()
            if get_current_task_index >= 0:
                self.task_list_component.update_task_option(get_current_task_index, 'tracked_sub_list', tracked_dict)

            InfoBar.success(
                title="Nhận diện chuyển động hoàn tất!",
                content=f"Đã bám đuổi logo thành công trên {len(tracked_dict)} khung hình! Hãy bấm nút 'Bắt Đầu' để khởi chạy xóa toàn bộ video và xuất tệp.",
                duration=6000,
                position=InfoBarPosition.TOP, parent=self
            )
        else:
            self._is_tracked = False
            config.set(config.movingSubtitleTracking, False)
            if hasattr(self.setting_interface, 'moving_subtitle_card'):
                self.setting_interface.moving_subtitle_card.setEnabled(True)
                self.setting_interface.moving_subtitle_card.setText("Nhận Diện Logo")
            
            InfoBar.error(
                title="Không nhận diện được chuyển động",
                content="Thuật toán không thể tự động bám đuổi vị trí. Vui lòng kiểm tra lại vùng khoanh chọn.",
                duration=5000,
                position=InfoBarPosition.TOP, parent=self
            )

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, '_session_checked', False):
            self._session_checked = True
            QtCore.QTimer.singleShot(300, self._check_session_recovery)

    def _check_session_recovery(self):
        if hasattr(self, 'task_list_component') and self.task_list_component.has_saved_session():
            from qfluentwidgets import MessageDialog
            dialog = MessageDialog(
                title="Khôi phục phiên làm việc",
                content="Phát hiện danh sách nhiệm vụ chưa hoàn thành từ lần làm việc trước. Bạn có muốn khôi phục không?",
                parent=self
            )
            dialog.yesButton.setText("Khôi phục")
            dialog.cancelButton.setText("Bỏ qua")
            if dialog.exec():
                self.task_list_component.load_session()
                InfoBar.success(
                    title="Khôi phục thành công",
                    content="Đã tải lại toàn bộ danh sách video và cấu hình trước đó.",
                    position=InfoBarPosition.TOP, parent=self,
                    duration=3000
                )

    def _update_manual_tools_state(self, _=None):
        """Khóa/Mở khóa các công cụ vẽ thủ công khi Bật/Tắt chế độ Tự động Tracking"""
        is_auto = getattr(config, 'movingSubtitleTracking', None) and config.movingSubtitleTracking.value

        # 1. Bật/Tắt các nút bấm trong bảng Cài Đặt (SettingInterface)
        if hasattr(self, 'setting_interface'):
            if hasattr(self.setting_interface, 'auto_detect_frame_btn'):
                self.setting_interface.auto_detect_frame_btn.setEnabled(not is_auto)
            if hasattr(self.setting_interface, 'brush_mode_btn'):
                self.setting_interface.brush_mode_btn.setEnabled(not is_auto)
            if hasattr(self.setting_interface, 'clear_brush_btn'):
                self.setting_interface.clear_brush_btn.setEnabled(not is_auto)
            if hasattr(self.setting_interface, 'mask_type_combo'):
                self.setting_interface.mask_type_combo.setEnabled(not is_auto)
            if hasattr(self.setting_interface, 'moving_subtitle_card'):
                self.setting_interface.moving_subtitle_card.setEnabled(not is_auto)

        # 2. Bật/Tắt khả năng kéo vẽ trên trình phát video
        if hasattr(self, 'video_display_component'):
            self.video_display_component.set_dragger_enabled(not is_auto)

        # 3. Hiển thị thông báo hướng dẫn cho người dùng (Chỉ khi có thay đổi từ user)
        if _ is not None and isinstance(_, bool):
            if is_auto:
                InfoBar.info(
                    title="Chế độ Tự động",
                    content="Tính năng tự động đang bật. Đã khóa các công cụ vẽ thủ công để tránh xung đột.",
                    position=InfoBarPosition.TOP, parent=self,
                    duration=3000
                )
            else:
                InfoBar.info(
                    title="Chế độ Thủ công",
                    content="Đã mở lại bộ công cụ cọ vẽ, khoanh vùng và nút Quét AI 1 frame.",
                    position=InfoBarPosition.TOP, parent=self,
                    duration=3000
                )

    def _on_auto_detect_clicked(self):
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            InfoBar.warning(
                title="Cảnh báo",
                content="Vui lòng tải một video trước khi thực hiện tự động nhận diện chữ.",
                position=InfoBarPosition.TOP, parent=self,
                duration=3000
            )
            return

        selected_lang = getattr(config, 'ocrLanguage', None) and config.ocrLanguage.value
        lang_display = {
            'auto': 'Tự động phát hiện',
            'vi': 'Tiếng Việt',
            'en': 'Tiếng Anh',
            'ch': 'Tiếng Trung',
            'japan': 'Tiếng Nhật',
            'korean': 'Tiếng Hàn'
        }.get(selected_lang, 'Tự động phát hiện')

        InfoBar.info(
            title="Đang xử lý AI OCR",
            content=f"Đang quét tìm kiếm vị trí các dòng phụ đề (Ngôn ngữ: {lang_display})...",
            position=InfoBarPosition.TOP, parent=self,
            duration=3000
        )

        current_frame_no = 1
        if hasattr(self.video_display_component, 'video_slider'):
            current_frame_no = self.video_display_component.video_slider.value()

        def _bg_detect():
            detected_rects = []
            try:
                # Dùng frame hiện tại chính xác trên màn hình thay vì cap.set() để tránh bị lệch khung hình do Keyframe
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    import cv2
                    cap = cv2.VideoCapture(self.video_path)
                    if current_frame_no > 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no - 1)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret or frame is None:
                        return
                
                h, w = frame.shape[:2]
                import re
                
                try:
                    from src.core.config import config
                    from src.core.tools.constant import SubtitleDetectMode
                    mode = config.subtitleDetectMode.value
                    mode_val = mode.value if hasattr(mode, 'value') else str(mode)

                    if mode_val == SubtitleDetectMode.RAPID_OCR.value:
                        if self._cached_rapid_engine is None:
                            from rapidocr_onnxruntime import RapidOCR
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.hardwareAcceleration.value else ['CPUExecutionProvider']
                            self._cached_rapid_engine = RapidOCR(providers=providers)
                        result_rapid, _ = self._cached_rapid_engine(frame)
                        result = []
                        if result_rapid:
                            for line in result_rapid:
                                box, text, score = line
                                result.append((box, text, score))
                    elif mode_val in [SubtitleDetectMode.PADDLE_OCR.value, SubtitleDetectMode.PP_OCRv5_MOBILE.value, SubtitleDetectMode.PP_OCRv5_SERVER.value]:
                        if self._cached_paddle_engine is None:
                            from src.ai_engines.paddle_compat import build_paddleocr
                            if mode_val == SubtitleDetectMode.PADDLE_OCR.value:
                                self._cached_paddle_engine = build_paddleocr(
                                    lang="ch",
                                    device="gpu" if config.hardwareAcceleration.value else "cpu",
                                )
                            else:
                                from src.core.tools.model_config import ModelConfig
                                model_config = ModelConfig()
                                self._cached_paddle_engine = build_paddleocr(
                                    lang="ch",
                                    device="gpu" if config.hardwareAcceleration.value else "cpu",
                                    text_detection_model_dir=model_config.DET_MODEL_DIR,
                                    text_recognition_model_dir=model_config.REC_MODEL_DIR,
                                )
                        
                        result = []
                        if hasattr(self._cached_paddle_engine, "ocr"):
                            res = self._cached_paddle_engine.ocr(frame, det=True, rec=True)
                            if res and isinstance(res, (list, tuple)) and len(res) > 0 and res[0]:
                                for item in res[0]:
                                    if item and len(item) == 2:
                                        box = item[0]
                                        txt = item[1][0] if isinstance(item[1], (list, tuple)) and len(item[1]) > 0 else ""
                                        score = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 1.0
                                        result.append((box, txt, score))
                        else:
                            # Fallback if somehow .ocr() is missing (unlikely)
                            from src.ai_engines.paddle_compat import extract_paddle_boxes
                            boxes = extract_paddle_boxes(self._cached_paddle_engine, frame, threshold=0.3)
                            if boxes:
                                for (x1, y1, x2, y2) in boxes:
                                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                                    result.append((poly, "TEXT", 1.0))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[AutoDetectText] OCR failed: {e}")
                    result = []

                if result:
                    for dt_box, text, score in result:
                        clean_text = text.strip() if text else ""
                        if not clean_text:
                            continue

                        # 1. 🛑 LỌC ICON GIẢ CHỮ VÀ HUD GAME (ví dụ: OOOOOO, 000000, ......, ||||||)
                        if len(set(clean_text)) == 1 and len(clean_text) >= 3:
                            continue
                        if re.match(r'^[O0o.|\-_=*#~]+$', clean_text):
                            continue

                        # 2. 🛑 LỌC CON SỐ CÔ ĐỘC / CHỈ SỐ VẬT PHẨM GAME (ví dụ: 64, 32, 16 trên ô đồ Minecraft)
                        if clean_text.isdigit() and len(clean_text) <= 3:
                            continue

                        # 3. 🎯 BỘ LỌC NGÔN NGỮ NGHIÊM NGẶT (STRICT LANGUAGE FILTER)
                        # Tách riêng từng hệ chữ để lọc chính xác
                        hanzi_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', clean_text)
                        kana_chars = re.findall(r'[\u3040-\u30ff]', clean_text)
                        hangul_chars = re.findall(r'[\uac00-\ud7af]', clean_text)
                        
                        if selected_lang in ['vi', 'en']:
                            # Tiếng Việt / Tiếng Anh: loại bỏ tất cả chữ Trung/Nhật/Hàn
                            if len(hanzi_chars) > 0 or len(kana_chars) > 0 or len(hangul_chars) > 0:
                                continue
                            if score < 0.8:
                                continue

                        elif selected_lang == 'ch':
                            # Tiếng Trung: chỉ lấy ô có chữ Hán, loại bỏ Kana thuần và Hangul thuần
                            if len(hanzi_chars) == 0:
                                continue
                            if score < 0.7:
                                continue

                        elif selected_lang == 'korean':
                            # Tiếng Hàn: chỉ lấy ô có chữ Hangul
                            if len(hangul_chars) == 0:
                                continue
                            if score < 0.7:
                                continue

                        elif selected_lang == 'japan':
                            # Tiếng Nhật: lấy ô có Kana hoặc Kanji, loại bỏ Hangul thuần
                            if len(kana_chars) == 0 and len(hanzi_chars) == 0:
                                continue
                            if len(hangul_chars) > 0 and len(kana_chars) == 0:
                                continue
                            if score < 0.7:
                                continue

                        xs = [pt[0] for pt in dt_box]
                        ys = [pt[1] for pt in dt_box]
                        xmin, xmax = int(min(xs)), int(max(xs))
                        ymin, ymax = int(min(ys)), int(max(ys))
                        
                        # Mở rộng lề rộng hơn (40px ngang, 20px dọc) để bao phủ sạch sẽ 
                        # toàn bộ khung nền (background pill) của phụ đề trên các video ngắn (TikTok/Shorts).
                        xmin = max(0, xmin - 40)
                        xmax = min(w, xmax + 40)
                        ymin = max(0, ymin - 20)
                        ymax = min(h, ymax + 20)

                        detected_rects.append((ymin, ymax, xmin, xmax))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[AutoDetectText] OCR Error: {e}")

            self.auto_detect_result_signal.emit(detected_rects)

        threading.Thread(target=_bg_detect, daemon=True).start()

    @Slot(object)
    def _apply_auto_detected_rects(self, detected_rects):
        if not detected_rects:
            InfoBar.warning(
                title="Không tìm thấy phụ đề",
                content="Không phát hiện dòng chữ nào trên khung hình hiện tại.",
                position=InfoBarPosition.TOP, parent=self,
                duration=3000
            )
            return

        self.sub_areas = detected_rects

        # Chuyển đổi sang tọa độ tỷ lệ preview (0..1) của widget
        widget_rects = self.video_display_component.video_coordinates_to_preview_coordinates(detected_rects)
        self.video_display_component.set_selection_rects(widget_rects)
        self.video_display_component.selected_indices = set(range(len(widget_rects)))
        
        if config.autoTighten.value:
            # Nếu bật Auto Tighten, gọi hàm để tự động ôm khít lại các khung vừa phát hiện
            self._is_tightening = True
            try:
                self.tighten_area_button_clicked()
            finally:
                self._is_tightening = False
        else:
            # Cập nhật khung vẽ trực quan trên giao diện
            self.video_display_component.update_preview_with_rect()
            
            # Đồng bộ vào options của task hiện tại
            current_idx = self.task_list_component.get_current_task_index()
            if current_idx >= 0:
                self.task_list_component.update_task_option(current_idx, TaskOptions.SUB_AREAS, detected_rects)

        InfoBar.success(
            title="Nhận diện thành công",
            content=f"Đã tự động khoanh vùng {len(detected_rects)} vị trí chữ trên khung hình!",
            position=InfoBarPosition.TOP, parent=self,
            duration=4000
        )

    def ab_sections_changed(self, ab_sections):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.AB_SECTIONS, ab_sections)

    def selections_changed(self, selections):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
            
        # Tính năng Tự động ôm khít chữ (Auto Tighten)
        if selections and config.autoTighten.value and not getattr(self, '_is_tightening', False):
            self._is_tightening = True
            try:
                self.tighten_area_button_clicked()
                # Sau khi tighten xong, set_selections sẽ phát lại selections_changed
                # Nên ta return để chu kỳ kế tiếp update với kích thước mới
                return
            finally:
                self._is_tightening = False
                
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, selections)


    def on_task_selected(self, index, file_path):
        """处理任务被选中事件
        
        Args:
            index: 任务索引
            file_path: 文件路径
        """
        # 加载选中的视频进行预览
        self.load_video(file_path)
        ab_sections = self.task_list_component.get_task_option(index, TaskOptions.AB_SECTIONS, [])
        self.video_display_component.set_ab_sections(ab_sections)
        selections = self.task_list_component.get_task_option(index, TaskOptions.SUB_AREAS, [])
        if len(selections) <= 0:
            self.video_display_component.load_selections_from_config()
        else:
            self.video_display_component.set_selection_rects(selections)
            
        tracked_list = self.task_list_component.get_task_option(index, 'tracked_sub_list', {})
        if tracked_list and len(tracked_list) > 0:
            self._is_tracked = True
            # Chuyển đổi key sang kiểu int để đảm bảo lấy đúng index khung hình từ timeline khi đã bị convert sang string bởi JSON
            self._tracked_sub_list = {int(k): v for k, v in tracked_list.items()}
            config.set(config.movingSubtitleTracking, True)
            if hasattr(self.setting_interface, 'moving_subtitle_card'):
                self.setting_interface.moving_subtitle_card.setText("Đã Nhận Diện (Hủy)")
        else:
            self._is_tracked = False
            self._tracked_sub_list = {}
            config.set(config.movingSubtitleTracking, False)
            if hasattr(self.setting_interface, 'moving_subtitle_card'):
                self.setting_interface.moving_subtitle_card.setText("Nhận Diện Logo")
    
    def on_task_deleted(self, index):
        """处理任务被删除事件
        
        Args:
            index: 任务索引
        """
        # 如果删除的是正在处理的任务，则需要更新状态
        if index == self.current_processing_task_index:
            self.current_processing_task_index = -1
        
        task = self.task_list_component.get_task(0)
        if task:
            # 如果还有任务，选中第一个
            self.task_list_component.select_task(0)

    def update_preview(self, frame):
        self.current_frame = frame
        # 先缩放图像
        resized_frame = self._img_resize(frame)

        # 设置视频参数
        self.video_display_component.set_video_parameters(
            self.frame_width, self.frame_height, 
            self.scaled_width if hasattr(self, 'scaled_width') else None,
            self.scaled_height if hasattr(self, 'scaled_height') else None,
            self.border_left if hasattr(self, 'border_left') else 0,
            self.border_top if hasattr(self, 'border_top') else 0,
            self.fps if self.fps is not None else 30,
        )
        
        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame)

    def _img_resize(self, image):
        height, width = image.shape[:2]
        
        video_preview_width = self.video_display_component.video_preview_width
        video_preview_height = self.video_display_component.video_preview_height
        # 计算等比缩放后的尺寸
        target_ratio = video_preview_width / video_preview_height
        image_ratio = width / height
        
        if image_ratio > target_ratio:
            # 宽度适配，高度按比例缩放
            new_width = video_preview_width
            new_height = int(new_width / image_ratio)
            top_border = (video_preview_height - new_height) // 2
            bottom_border = video_preview_height - new_height - top_border
            left_border = 0
            right_border = 0
        else:
            # 高度适配，宽度按比例缩放
            new_height = video_preview_height
            new_width = int(new_height * image_ratio)
            left_border = (video_preview_width - new_width) // 2
            right_border = video_preview_width - new_width - left_border
            top_border = 0
            bottom_border = 0
        
        # 先缩放图像
        resized = cv2.resize(image, (new_width, new_height))
        
        # 添加黑边以填充到目标尺寸
        padded = cv2.copyMakeBorder(
            resized, 
            top_border, bottom_border, 
            left_border, right_border, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # 保存边框信息，用于坐标转换
        self.border_left = left_border / video_preview_width
        self.border_right = right_border / video_preview_width
        self.border_top = top_border / video_preview_height
        self.border_bottom = bottom_border / video_preview_height
        self.original_width = width
        self.original_height = height
        self.is_vertical = width < height
        self.scaled_width = new_width / video_preview_width
        self.scaled_height = new_height / video_preview_height
        
        return padded

    def stop_button_clicked(self):
        try:
            self._stop_event.set()
            running_process = self.running_process
            if running_process:
                ProcessManager.instance().terminate_by_process(running_process)
            # Dọn dẹp file tạm chưa hoàn chỉnh nếu bị dừng giữa chừng
            if self.current_processing_task_index >= 0:
                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.PENDING)
                task_item = self.task_list_component.get_task(self.current_processing_task_index)
                if task_item and getattr(task_item, 'output_path', None) and os.path.exists(task_item.output_path):
                    try:
                        os.remove(task_item.output_path)
                    except Exception:
                        pass
        finally:
            self.running_process = None
            self.run_button.setVisible(True)
            self.stop_button.setVisible(False)
            if hasattr(self, 'video_display_component'):
                self.video_display_component.set_controls_enabled(True)

    @Slot(bool)
    def _toggle_buttons(self, show_run):
        """Chuyển đổi trạng thái hiển thị của các nút bấm"""
        self.run_button.setVisible(show_run)
        self.stop_button.setVisible(not show_run)
        self.file_button.setVisible(show_run)
        if hasattr(self, 'pause_resume_button'):
            self.pause_resume_button.setVisible(not show_run)
        if hasattr(self, 'video_display_component'):
            self.video_display_component.set_controls_enabled(show_run)
            
        # Cải tiến: Khóa/Mở khóa toàn bộ thẻ cấu hình bên dưới khi hệ thống đang chạy
        if hasattr(self, 'setting_interface'):
            self.setting_interface.setEnabled(show_run)
            
        if show_run:
            self.is_queue_paused = False
            if hasattr(self, 'pause_resume_button'):
                self.pause_resume_button.setText(tr['Setting'].get('PauseQueue', "Tạm dừng hàng đợi"))
                self.pause_resume_button.setIcon(FluentIcon.PAUSE)

    def run_button_clicked(self):
        if not self.task_list_component.get_pending_tasks():
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return

        try:
            # 获取所有待执行的任务
            pending_tasks = self.task_list_component.get_pending_tasks()
            if not pending_tasks:
                return

            self._stop_event.clear()
            self.toggle_buttons_signal.emit(False)
            # 开启后台线程处理视频
            def task():
                try:
                    # Giải phóng VRAM tiến trình chính trước khi chạy hàng đợi
                    try:
                        from src.core.main import ModelCacheManager
                        ModelCacheManager.clear()
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    while not self._stop_event.is_set():
                        try:
                            # Nếu hàng đợi bị tạm dừng, chờ cho đến khi được tiếp tục hoặc dừng hẳn
                            while self.is_queue_paused and not self._stop_event.is_set():
                                time.sleep(0.5)

                            if self._stop_event.is_set():
                                break

                            pending_tasks = self.task_list_component.get_pending_tasks()
                            if not pending_tasks:
                                break
                            pending_task = pending_tasks[0]
                            # 更新当前处理的任务索引
                            self.current_processing_task_index, task_item = pending_task
                            if not self.load_video(task_item.path):
                                self.append_log_signal.emit([tr['SubtitleExtractorGUI']['OpenVideoFailed'].format(task_item.path)])
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)
                                continue

                            # Bỏ check chế độ Tự động nhận diện chữ theo từng frame vì OCR quét mặc định toàn màn hình nếu không có tọa độ.
                            self.append_log_signal.emit(["[Chế độ Xóa Phụ Đề] Đã bắt đầu xử lý xóa phụ đề."])

                            # Kiểm tra chế độ Xóa phụ đề di chuyển (Moving Subtitle Tracking)
                            is_moving_sub = getattr(config, 'movingSubtitleTracking', None) and config.movingSubtitleTracking.value
                            subtitle_areas = self.task_list_component.get_task_option(self.current_processing_task_index, TaskOptions.SUB_AREAS, [])
                            tracked_list = self.task_list_component.get_task_option(self.current_processing_task_index, 'tracked_sub_list', {})
                            
                            if is_moving_sub:
                                is_full_screen = False
                                if subtitle_areas and len(subtitle_areas) == 1 and hasattr(self, 'frame_width') and self.frame_width:
                                    ymin, ymax, xmin, xmax = subtitle_areas[0]
                                    if xmin == 0 and ymin == 0 and xmax >= self.frame_width - 1 and ymax >= self.frame_height - 1:
                                        is_full_screen = True

                                if not subtitle_areas or len(subtitle_areas) == 0 or is_full_screen or not tracked_list:
                                    self.append_log_signal.emit(["[Cảnh báo] Chưa xác định vùng logo di chuyển! Vui lòng dùng chuột khoanh vùng logo và bấm 'Nhận Diện' trước khi bắt đầu."])
                                    InfoBar.warning(
                                        title="Chưa xác định vùng logo di chuyển",
                                        content="Bạn phải dùng chuột khoanh vùng chứa logo di chuyển và bấm 'Nhận Diện' trước khi bấm 'Bắt Đầu'!",
                                        position=InfoBarPosition.TOP, parent=self,
                                        duration=5000
                                    )
                                    self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.IDLE)
                                    break
                                else:
                                    self.append_log_signal.emit(["[Chế độ Xóa phụ đề di chuyển] AI đang bám đuổi và tự động cập nhật vị trí phụ đề động trên từng khung hình."])
                            else:
                                if not subtitle_areas or len(subtitle_areas) <= 0:
                                    subtitle_areas = [(0, self.frame_height, 0, self.frame_width)]
                                    self.task_list_component.update_task_option(self.current_processing_task_index, TaskOptions.SUB_AREAS, subtitle_areas)

                            self.video_display_component.save_selections_to_config()

                            # 更新任务状态为运行中
                            self.task_list_component.update_task_progress(self.current_processing_task_index, 1)

                            # 选中当前任务
                            self.select_task_signal.emit(self.current_processing_task_index)

                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None

                            self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.PROCESSING)
                            self.start_time = time.time()
                            options = {}
                            for key in task_item.options:
                                value = task_item.options[key]
                                if key == TaskOptions.SUB_AREAS.value:
                                    value = self.video_display_component.preview_coordinates_to_video_coordinates(value)
                                options[key] = value
                            if 'tracked_sub_list' not in options and hasattr(self, '_tracked_sub_list') and self._tracked_sub_list:
                                options['tracked_sub_list'] = self._tracked_sub_list
                            # Xác định output path trước để UI có thể track được
                            if not task_item.output_path:
                                import pathlib
                                vd_name = pathlib.Path(task_item.path).stem
                                ext = os.path.splitext(task_item.path)[-1]
                                if is_image_file(task_item.path):
                                    pic_dir = os.path.join(os.path.dirname(task_item.path), 'no_sub')
                                    out_path = os.path.join(pic_dir, f'{vd_name}{ext}')
                                else:
                                    out_path = os.path.abspath(os.path.join(os.path.dirname(task_item.path), f'{vd_name}_no_sub.mp4'))
                                task_item.output_path = out_path
                            output_path = task_item.output_path
                            process = self.run_subtitle_remover_process(task_item.path, output_path, options)

                            # 检查是否在处理过程中被停止
                            if self._stop_event.is_set():
                                break

                            # 更新任务状态为已完成
                            task_obj = self.task_list_component.get_task(self.current_processing_task_index)
                            if process.exitcode == 0 and task_obj and task_obj.status == TaskStatus.PROCESSING:
                                self.progress_signal.emit(100, True, 0)
                                task_obj.output_path = output_path
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.COMPLETED)
                                
                                # Kiểm tra xem file SRT có được tạo ra không
                                custom_dir = getattr(config, 'srtSaveDirectory', None)
                                save_dir = custom_dir.value if (custom_dir and custom_dir.value and os.path.isdir(custom_dir.value)) else None
                                if save_dir:
                                    import pathlib
                                    expected_srt = os.path.join(save_dir, pathlib.Path(output_path).with_suffix('.srt').name)
                                else:
                                    import pathlib
                                    expected_srt = str(pathlib.Path(output_path).with_suffix('.srt'))
                                
                                if os.path.exists(expected_srt):
                                    self.last_exported_srt_path = expected_srt
                                    self.srt_exported_signal.emit(expected_srt)
                            else:
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)

                        except Exception as e:
                            print(e)
                            self.append_log_signal.emit([f"Error: {e}"])
                            # 更新任务状态为失败
                            if self.current_processing_task_index >= 0:
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)
                            continue
                        finally:
                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None
                            time.sleep(1)
                finally:
                    self.toggle_buttons_signal.emit(True)
                    # Giải phóng VRAM tiến trình chính sau khi hoàn thành hàng đợi
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            self._worker_thread = threading.Thread(target=task, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            print(traceback.format_exc())
            self.append_log_signal.emit([f"Error: {e}"])
            self.toggle_buttons_signal.emit(True)

    @staticmethod
    def remover_process(queue, video_path, output_path, options):
        """
        在子进程中执行字幕提取的函数
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 选项
        """
        sr = None
        try:
            from src.core.main import GUISubtitleRemover
            from src.core.interface.worker_interface import ProcessCallback

            class RemoteCallback(ProcessCallback):
                def log_message(self, *args):
                    SubtitleRemoverRemoteCall.remote_call_append_log(queue, list(args))
                def update_progress(self, progress: int, is_finished: bool, frame_no: int = 0):
                    SubtitleRemoverRemoteCall.remote_call_update_progress(queue, progress, is_finished, frame_no)
                def on_error(self, err: Exception):
                    SubtitleRemoverRemoteCall.remote_call_catch_error(queue, err)
                def manage_process(self, pid: int):
                    SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
                def update_preview(self, *args):
                    SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(queue, list(args))

            sr = GUISubtitleRemover(video_path, True, callback=RemoteCallback())
            sr.video_out_path = output_path
            for key in options:
                setattr(sr, key, options[key])

            sr.run()
        except Exception as e:
            traceback.print_exc()
            SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
        finally:
            if sr:
                sr.isFinished = True
                sr.vsf_running = False
            SubtitleRemoverRemoteCall.remote_call_finish(queue)
            

    # 修改run_subtitle_remover_process方法
    def run_subtitle_remover_process(self, video_path, output_path, options):
        """
        使用多进程执行字幕提取，并等待进程完成
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 任务选项
        """
        subtitle_remover_remote_caller = SubtitleRemoverRemoteCall()
        subtitle_remover_remote_caller.register_update_progress_callback(self.progress_signal.emit)
        subtitle_remover_remote_caller.register_log_callback(self.append_log_signal.emit)
        subtitle_remover_remote_caller.register_update_preview_with_comp_callback(self.update_preview_with_comp_signal.emit)
        subtitle_remover_remote_caller.register_error_callback(self.task_error_signal.emit)
        process = multiprocessing.Process(
            target=HomeInterface.remover_process,
            args=(subtitle_remover_remote_caller.queue, video_path, output_path, options)
        )
        try:
            if self._stop_event.is_set():
                return process
            process.start()
            ProcessManager.instance().add_process(process)
            self.running_process = process
            process.join()
            print(f"Process exited with code {process.exitcode}")
        finally:
            subtitle_remover_remote_caller.stop()
        return process

    @Slot()
    def processing_finished(self):
        pending_tasks = self.task_list_component.get_pending_tasks()
        if pending_tasks:
            # 还有待执行任务, 忽略
            return
        # 处理完成后恢复界面可用性
        self.run_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.se = None
        if hasattr(self, 'video_display_component'):
            self.video_display_component.set_controls_enabled(True)
        # 重置视频滑块
        self.video_slider.setValue(1)
        # 重置当前处理任务索引
        self.current_processing_task_index = -1

    def update_progress(self, progress_total, isFinished, frame_no=0):
        try:
            if frame_no > 0 and hasattr(self, 'frame_count') and self.frame_count and self.frame_count > 0:
                pos = min(self.frame_count, max(1, frame_no))
                if pos != self.video_slider.value():
                    self.video_slider.blockSignals(True)
                    self.video_slider.setValue(pos)
                    self.video_slider.blockSignals(False)
                    self.video_display_component.update_time_display()
            
            # Tính toán FPS và ETA
            fps = 0.0
            eta_str = ""
            if hasattr(self, 'start_time') and self.start_time > 0:
                elapsed = time.time() - self.start_time
                current_frame = int(progress_total / 100 * self.frame_count)
                if elapsed > 1.0 and current_frame > 0:
                    fps = current_frame / elapsed
                    remaining_frames = self.frame_count - current_frame
                    if fps > 0:
                        eta_seconds = int(remaining_frames / fps)
                        m, s = divmod(eta_seconds, 60)
                        h, m = divmod(m, 60)
                        if h > 0:
                            eta_str = f"{h:02d}:{m:02d}:{s:02d}"
                        else:
                            eta_str = f"{m:02d}:{s:02d}"

            # Cập nhật tiến trình và thông tin tốc độ
            if self.current_processing_task_index >= 0:
                progress_text = f"{progress_total}%"
                if fps > 0 and eta_str:
                    progress_text += f" ({fps:.1f} FPS, {eta_str})"
                progress_item = self.task_list_component.table.item(self.current_processing_task_index, 1)
                if progress_item:
                    progress_item.setText(progress_text)
                self.task_list_component.tasks[self.current_processing_task_index].progress = progress_total
            
            # 检查是否完成
            if isFinished:
                self.processing_finished()
        except Exception as e:
            # 捕获 any exception, 防止 crash
            print(f"更新进度时出错: {str(e)}")

    @Slot(list)
    def append_log(self, log):
        self.append_output(*log)

    def append_output(self, *args):
        """添加文本到输出区域并控制滚动
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        # 将所有参数转换为字符串并用空格连接
        text = ' '.join(str(arg) for arg in args).rstrip()
        timestamp = datetime.now().strftime('%H:%M:%S')
        # 转义HTML特殊字符
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # 根据内容判断消息类型并着色
        if '错误' in text or 'Error' in text or '失败' in text or 'Failed' in text:
            color = '#e74c3c'
        elif '成功' in text or '完成' in text or 'Success' in text or 'Finished' in text:
            color = '#27ae60'
        elif '警告' in text or 'Warning' in text:
            color = '#f39c12'
        else:
            color = '#2980b9'
        html = f'<span style="color:#888;">[{timestamp}]</span> <span style="color:{color};">{escaped}</span><br>'
        self.output_text.append(html)
        print(*args)  # 保持原始的 print 行为
        # 如果启用了自动滚动，则滚动到底部
        if self.auto_scroll:
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(list)
    def update_preview_with_comp(self, args):
        """Cập nhật preview khi đang xử lý video (hiển thị frame gốc bên trái và frame đã xóa bên phải)"""
        frame_ori = args[0]
        frame_comp = args[1]
        if len(args) >= 3 and args[2] > 0 and hasattr(self, 'frame_count') and self.frame_count:
            current_frame = min(self.frame_count, max(1, args[2]))
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(current_frame)
            self.video_slider.blockSignals(False)

        # Hiển thị trực tiếp frame kết quả đã xóa sub (giữ nguyên tỷ lệ video gốc)
        resized_frame = self._img_resize(frame_comp)
        self.video_display_component.update_video_display(resized_frame, draw_selection=False)
        self.video_display_component.set_dragger_enabled(False)
        self.video_display_component.update_time_display()

    def _on_mask_preview_pressed(self):
        """Giữ nút preview để xem ảnh Gốc"""
        if hasattr(self, '_preview_original_frame') and self._preview_original_frame is not None:
            resized_orig = self._img_resize(self._preview_original_frame)
            self.video_display_component.update_video_display(resized_orig, draw_selection=True)

    def _on_mask_preview_released(self):
        """Nhả nút preview để xem ảnh Đã Xóa AI"""
        if hasattr(self, '_preview_inpainted_frame') and self._preview_inpainted_frame is not None:
            resized_inpaint = self._img_resize(self._preview_inpainted_frame)
            self.video_display_component.update_video_display(resized_inpaint, draw_selection=False)

    @Slot(str)
    def _on_srt_exported(self, expected_srt: str):
        if hasattr(self, 'setting_interface') and hasattr(self.setting_interface, 'jump_to_translate_card'):
            self.setting_interface.jump_to_translate_card.button.setEnabled(True)

    @Slot(object)
    def on_task_error(self, e):
        self.append_output(tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(str(e)))
        if self.current_processing_task_index >= 0:
            self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)

    def load_video(self, video_path):
        self.video_path = video_path
        with self._video_cap_lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
        # 如果是图片文件，直接走图片加载路径
        if is_image_file(video_path):
            return self.load_as_picture(video_path)
        with self._video_cap_lock:
            self.video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
            if not self.video_cap.isOpened():
                self.video_cap = None
                return self.load_as_picture(video_path)
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.release()
                self.video_cap = None
                return self.load_as_picture(video_path)
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)

        if hasattr(self, '_frame_cache'):
            with self._cache_lock:
                self._frame_cache.clear()
                if frame is not None:
                    self._frame_cache[1] = frame.copy()

        self.update_preview(frame)
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.fps = self.fps
        self.video_display_component.set_video_path(video_path)
        self.video_display_component.update_time_display()
        self.video_display_component.set_controls_enabled(True)
        self.video_display_component.set_dragger_enabled(True)
        # 视频模式下恢复用户原始的 inpaint 模式选择
        self._unlock_inpaint_mode()
        
        # 🚀 Kích hoạt luồng chạy ngầm nạp trước cache khung hình mẫu toàn video (Ultra-fast 0ms Timeline Scrubbing)
        self._start_bg_frame_preload(video_path)
        return True

    def _start_bg_frame_preload(self, video_path):
        if getattr(self, '_preload_cancel', None) is not None:
            self._preload_cancel.set()

        self._preload_cancel = threading.Event()
        cancel_evt = self._preload_cancel

        def _worker():
            try:
                cap = cv2.VideoCapture(get_readable_path(video_path))
                if not cap.isOpened():
                    return
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = 5 if fc < 3000 else 10
                curr = 1
                while not cancel_evt.is_set() and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if curr % step == 0 or curr == 1:
                        if hasattr(self, '_frame_cache'):
                            with self._cache_lock:
                                if curr not in self._frame_cache:
                                    self._frame_cache[curr] = frame
                    curr += 1
                cap.release()
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True)
        self._preload_thread = t
        t.start()

    def load_as_picture(self, path):
        if not is_image_file(path):
            return False
        self.video_path = path
        self.video_cap = None
        frame = read_image(get_readable_path(path))
        if frame is None:
            return False
        self.frame_count = 1
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.fps = 1
        self.update_preview(frame)
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        # Chế độ ảnh: khóa inpaint mode sang LAMA
        self._lock_inpaint_mode_to_lama()
        return True

    def _lock_inpaint_mode_to_lama(self):
        """Chế độ ảnh: khóa inpaint mode về LAMA"""
        if self._saved_inpaint_mode is None:
            self._saved_inpaint_mode = config.inpaintMode.value
        config.set(config.inpaintMode, InpaintMode.LAMA)
        self.setting_interface.set_inpaint_mode_enabled(False)

    def _unlock_inpaint_mode(self):
        """Chế độ video: khôi phục inpaint mode ban đầu của người dùng"""
        if self._saved_inpaint_mode is not None:
            config.set(config.inpaintMode, self._saved_inpaint_mode)
            self._saved_inpaint_mode = None
        self.setting_interface.set_inpaint_mode_enabled(True)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        return True

    def add_area_button_clicked(self):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            InfoBar.warning(
                title=tr['TaskList']['Warning'],
                content=tr['SubtitleExtractorGUI']['PleaseSelectTask'],
                position=InfoBarPosition.TOP, parent=self,
                duration=3000
            )
            return
        
        # Thêm một vùng chọn mặc định
        self.video_display_component.add_default_selection()
        # Lưu các vùng chọn mới nhất vào nhiệm vụ
        selections = self.video_display_component.selection_rects
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, selections)

    def open_adv_setting_clicked(self):
        """Chuyển trực tiếp sang tab Cài Đặt Nâng Cao"""
        parent_window = self.window()
        if hasattr(parent_window, 'advancedSettingInterface'):
            parent_window.switchTo(parent_window.advancedSettingInterface)

    def tighten_area_button_clicked(self):
        """Tự động phân tích đường viền chữ bên trong khung chọn và co hẹp ôm khít nét chữ"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            InfoBar.warning(
                title=tr['TaskList']['Warning'],
                content=tr['SubtitleExtractorGUI']['OpenVideoFirst'],
                position=InfoBarPosition.TOP, parent=self,
                duration=3000
            )
            return

        selections = self.video_display_component.selection_rects
        if not selections:
            InfoBar.warning(
                title=tr['TaskList']['Warning'],
                content=tr['SubtitleExtractorGUI']['PleaseSelectSubtitleArea'],
                position=InfoBarPosition.TOP, parent=self,
                duration=3000
            )
            return

        frame_h, frame_w = self.current_frame.shape[:2]
        new_selections = []

        for rect in selections:
            ymin_r, ymax_r, xmin_r, xmax_r = rect
            y1 = max(0, int(ymin_r * frame_h))
            y2 = min(frame_h, int(ymax_r * frame_h))
            x1 = max(0, int(xmin_r * frame_w))
            x2 = min(frame_w, int(xmax_r * frame_w))

            if y2 - y1 < 5 or x2 - x1 < 5:
                new_selections.append(rect)
                continue

            crop = self.current_frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Sử dụng Canny edge detection để bắt cạnh chữ
            edges = cv2.Canny(gray, 50, 150)
            
            # Khử nhiễu biên (tránh việc background bị cắt ở viền tạo thành cạnh)
            edges[0:2, :] = 0
            edges[-2:, :] = 0
            edges[:, 0:2] = 0
            edges[:, -2:] = 0

            # Sử dụng hình thái học để nối liền các nét đứt của chữ
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            pts = np.argwhere(closed > 0)
            if len(pts) > 10:
                y_min_c, x_min_c = pts.min(axis=0)
                y_max_c, x_max_c = pts.max(axis=0)

                pad = 1 # Chỉnh pad = 1 để ôm khít tuyệt đối
                ny1 = max(0, y1 + y_min_c - pad)
                ny2 = min(frame_h, y1 + y_max_c + pad)
                nx1 = max(0, x1 + x_min_c - pad)
                nx2 = min(frame_w, x1 + x_max_c + pad)

                new_selections.append((ny1 / frame_h, ny2 / frame_h, nx1 / frame_w, nx2 / frame_w))
            else:
                new_selections.append(rect)

        self.video_display_component.set_selection_rects(new_selections)
        self.video_display_component.update_preview_with_rect()
        
        # FIX BUG-04: Save tightened selections back to task options for backend processing
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index >= 0:
            self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, new_selections)

        InfoBar.success(
            title="Thành công",
            content="Đã tự động co hẹp và ôm khít 100% nét chữ!",
            position=InfoBarPosition.TOP, parent=self,
            duration=3000
        )

    def mask_preview_button_clicked(self):
        """Xử lý sự kiện xem trước kết quả xóa bằng thuật toán đã chọn"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            InfoBar.warning(
                title="Chưa mở video",
                content="Vui lòng tải tệp video hoặc hình ảnh trước khi thực hiện xem trước.",
                position=InfoBarPosition.TOP, parent=self,
                duration=3000
            )
            return

        # 1. Lấy tọa độ khoanh vùng (rects) và nét vẽ tự do (strokes)
        selections = self.video_display_component.preview_coordinates_to_video_coordinates(
            self.video_display_component.selection_rects
        )
        strokes = getattr(self.video_display_component, 'freehand_strokes', [])
        
        if (not selections or len(selections) == 0) and (not strokes or len(strokes) == 0):
            InfoBar.warning(
                title="Chưa khoanh vùng xóa",
                content="Vui lòng khoanh vùng chữ cần xóa hoặc dùng cọ vẽ trước khi bấm Xem Trước.",
                position=InfoBarPosition.TOP, parent=self,
                duration=3500
            )
            return

        # 2. Tạo mặt nạ tổng hợp
        from src.core.tools.inpaint_tools import create_combined_mask
        
        mask_size = (self.frame_height, self.frame_width)
        dilation = config.maskDilation.value
        feather = config.maskFeather.value
        
        # Chuyển đổi từ format UI (ymin, ymax, xmin, xmax) sang format mask (xmin, xmax, ymin, ymax)
        mask_coords = [(xmin, xmax, ymin, ymax) for (ymin, ymax, xmin, xmax) in selections]
        
        # Luôn sử dụng mask tổng hợp kết hợp cả hình hộp và nét cọ
        combined_mask = create_combined_mask(mask_size, coords_list=mask_coords, strokes_list=strokes, dilation=dilation, feather_pixels=feather, frame=self.current_frame)
            
        # 3. Sao chép các giá trị cần thiết trước khi vào thread
        preview_frame = self.current_frame.copy()
        inpaint_mode = config.inpaintMode.value
        sharpen_enabled = config.sharpenInpaintedArea.value
        
        self.append_output(f"Xem trước kết quả xóa (mô hình: {inpaint_mode.name})...")
        self.mask_preview_button.setEnabled(False)
        self.mask_preview_button.setText("Đang xử lý...")

        # 4. Chạy inpaint trong background thread để tránh đơ UI
        def _run_inpaint():
            import time as _time
            _t0 = _time.time()
            inpainted_frame = None
            try:
                import torch
                from src.ai_engines.inpaint.lama_inpaint import LamaInpaint
                from src.ai_engines.inpaint.opencv_inpaint import OpenCVInpaint
                from src.core.tools.model_config import ModelConfig
                from src.core.main import ModelCacheManager
                from src.core.tools.hardware_accelerator import HardwareAccelerator
                
                model_config = ModelConfig()
                device = HardwareAccelerator.instance().device
                self.append_log_signal.emit([f"Thiết bị xử lý: {device} ({_time.time()-_t0:.1f}s)"])

                # --- Tự động thu nhỏ và chuẩn hóa kích thước frame (bội số của 16) để tiết kiệm VRAM và tránh lỗi mô hình AI ---
                MAX_PREVIEW_DIM = 1080  # Giới hạn chiều lớn nhất cho preview
                work_frame = preview_frame
                work_mask = combined_mask
                orig_h, orig_w = preview_frame.shape[:2]
                
                # Xác định kích thước mục tiêu
                if max(orig_h, orig_w) > MAX_PREVIEW_DIM:
                    scale_factor = MAX_PREVIEW_DIM / max(orig_h, orig_w)
                    new_w = int(orig_w * scale_factor)
                    new_h = int(orig_h * scale_factor)
                else:
                    new_w = orig_w
                    new_h = orig_h
                
                # Ép buộc kích thước chiều dài và rộng phải chia hết cho 16
                new_w = max(16, (new_w // 16) * 16)
                new_h = max(16, (new_h // 16) * 16)
                
                if new_w != orig_w or new_h != orig_h:
                    work_frame = cv2.resize(preview_frame, (new_w, new_h))
                    if work_mask.ndim == 3:
                        work_mask = cv2.resize(combined_mask, (new_w, new_h))
                    else:
                        work_mask = cv2.resize(combined_mask, (new_w, new_h))
                    self.append_log_signal.emit([f"Chuẩn hóa kích thước frame từ {orig_w}×{orig_h} → {new_w}×{new_h} để tương thích mô hình AI"])

                def _do_inpaint(dev):
                    """Thực hiện inpaint trên thiết bị chỉ định, trả về frame đã xử lý"""
                    # Removed unused nonlocal
                    result = None
                    
                    if inpaint_mode == InpaintMode.OPENCV:
                        # OpenCV inpaint yêu cầu mặt nạ nhị phân 8-bit đơn kênh (uint8)
                        mask_uint8 = work_mask
                        if mask_uint8.dtype != np.uint8:
                            mask_uint8 = (mask_uint8 * 255).astype(np.uint8) if mask_uint8.max() <= 1.0 else mask_uint8.astype(np.uint8)
                        if mask_uint8.ndim == 3:
                            mask_uint8 = mask_uint8[:, :, 0]
                        inpainter = OpenCVInpaint()
                        result = inpainter.inpaint(work_frame, mask_uint8)
                    elif inpaint_mode == InpaintMode.LAMA:
                        self.append_log_signal.emit([f"Đang nạp mô hình LAMA trên {dev}..."])
                        def load_lama():
                            model_path = os.path.join(model_config.LAMA_MODEL_DIR, 'big-lama.pt')
                            return LamaInpaint(dev, model_path)
                        lama_model = ModelCacheManager.get_model(f"lama_{dev}", load_lama)
                        self.append_log_signal.emit([f"Mô hình LAMA sẵn sàng ({_time.time()-_t0:.1f}s). Đang xóa phụ đề..."])
                        result = lama_model.inpaint(work_frame, work_mask)
                    elif inpaint_mode == InpaintMode.STTN_DET:
                        from src.ai_engines.inpaint.sttn_det_inpaint import STTNDetInpaint
                        self.append_log_signal.emit([f"Đang nạp mô hình STTN_DET trên {dev}..."])
                        def load_sttn_det():
                            return STTNDetInpaint(dev, model_config.STTN_DET_MODEL_PATH)
                        sttn_model = ModelCacheManager.get_model(f"sttn_det_{dev}", load_sttn_det)
                        self.append_log_signal.emit([f"Mô hình STTN_DET sẵn sàng ({_time.time()-_t0:.1f}s). Đang xóa phụ đề trên 5 frame..."])
                        frames = [work_frame.copy() for _ in range(5)]
                        results = sttn_model(frames, work_mask)
                        result = results[2]
                    elif inpaint_mode == InpaintMode.STTN_AUTO:
                        from src.ai_engines.inpaint.sttn_auto_inpaint import STTNInpaint
                        self.append_log_signal.emit([f"Đang nạp mô hình STTN_AUTO trên {dev}..."])
                        def load_sttn_auto():
                            return STTNInpaint(dev, model_config.STTN_AUTO_MODEL_PATH)
                        sttn_model = ModelCacheManager.get_model(f"sttn_auto_{dev}", load_sttn_auto)
                        self.append_log_signal.emit([f"Mô hình STTN_AUTO sẵn sàng ({_time.time()-_t0:.1f}s). Đang xóa phụ đề trên 5 frame..."])
                        frames = [work_frame.copy() for _ in range(5)]
                        results = sttn_model(work_mask, input_frames=frames)
                        result = results[2]
                    elif inpaint_mode == InpaintMode.PROPAINTER:
                        from src.ai_engines.inpaint.propainter_inpaint import PropainterInpaint
                        self.append_log_signal.emit([f"Đang nạp mô hình ProPainter trên {dev}..."])
                        def load_propainter():
                            return PropainterInpaint(dev, model_config.PROPAINTER_MODEL_DIR)
                        propainter_model = ModelCacheManager.get_model(f"propainter_{dev}", load_propainter)
                        self.append_log_signal.emit([f"Mô hình ProPainter sẵn sàng ({_time.time()-_t0:.1f}s). Đang xóa phụ đề trên 5 frame..."])
                        frames = [work_frame.copy() for _ in range(5)]
                        results = propainter_model.inpaint(frames, work_mask)
                        result = results[2]
                    else:
                        self.append_log_signal.emit(["Mô hình không xác định, dùng LAMA làm mặc định..."])
                        def load_lama_fb():
                            model_path = os.path.join(model_config.LAMA_MODEL_DIR, 'big-lama.pt')
                            return LamaInpaint(dev, model_path)
                        lama_model = ModelCacheManager.get_model(f"lama_{dev}", load_lama_fb)
                        result = lama_model.inpaint(work_frame, work_mask)
                    return result

                # Thử GPU trước, nếu OOM thì fallback sang CPU
                try:
                    inpainted_frame = _do_inpaint(device)
                except torch.cuda.OutOfMemoryError:
                    self.append_log_signal.emit(["GPU hết VRAM! Đang giải phóng bộ nhớ và chuyển sang CPU..."])
                    torch.cuda.empty_cache()
                    cpu_device = torch.device("cpu")
                    inpainted_frame = _do_inpaint(cpu_device)

                # --- Phóng to kết quả về kích thước gốc nếu kích thước đã thay đổi ---
                if inpainted_frame is not None and (inpainted_frame.shape[1] != orig_w or inpainted_frame.shape[0] != orig_h):
                    inpainted_frame = cv2.resize(inpainted_frame, (orig_w, orig_h))
                    
                # Áp dụng bộ lọc tái tạo vân bề mặt nếu cấu hình bật
                if sharpen_enabled and inpainted_frame is not None:
                    gray_mask = combined_mask.astype(np.float32)
                    if gray_mask.max() > 1.0:
                        gray_mask = gray_mask / 255.0
                    if gray_mask.ndim == 2:
                        gray_mask = gray_mask[:, :, np.newaxis]
                    smoothed = cv2.bilateralFilter(inpainted_frame, d=5, sigmaColor=50, sigmaSpace=50)
                    details = cv2.subtract(inpainted_frame, smoothed)
                    sharpened = cv2.addWeighted(inpainted_frame, 1.0, details, 1.8, 0)
                    h, w = inpainted_frame.shape[:2]
                    noise = np.random.normal(0, 2.0, (h, w, 3)).astype(np.float32)
                    sharpened_with_noise = (sharpened.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
                    inpainted_frame = (inpainted_frame * (1.0 - gray_mask) + sharpened_with_noise * gray_mask).clip(0, 255).astype(np.uint8)

                self.append_log_signal.emit([f"Xem trước hoàn tất trong {_time.time()-_t0:.1f} giây!"])
                # Lưu kết quả vào biến instance (tránh truyền numpy array qua Signal)
                self._preview_result_frame = inpainted_frame
                self._preview_error_info = None
                self.mask_preview_result_signal.emit(True, None)

            except Exception as e:
                traceback.print_exc()
                self.append_log_signal.emit([f"Lỗi khi xem trước: {e}"])
                # Lưu lỗi vào biến instance
                self._preview_result_frame = preview_frame
                self._preview_error_info = (e, combined_mask)
                self.mask_preview_result_signal.emit(False, None)

        threading.Thread(target=_run_inpaint, daemon=True).start()

    @Slot(object, object)
    def _on_mask_preview_result(self, success, _unused):
        """Nhận kết quả inpaint từ background thread và cập nhật UI"""
        try:
            self.mask_preview_button.setEnabled(True)
            self.mask_preview_button.setText(tr['Setting'].get('MaskPreview', 'Xem trước kết quả'))
            
            result_frame = getattr(self, '_preview_result_frame', None)
            error_info = getattr(self, '_preview_error_info', None)
            
            if result_frame is None:
                self.append_output(tr['SubtitleExtractorGUI']['PreviewResultError'])
                return
            
            self.append_output(tr['SubtitleExtractorGUI']['UpdatingDisplayLog'].format(result_frame.shape))
            
            if error_info is None:
                # Hiển thị trực tiếp 100% kết quả AI đã xóa sạch phụ đề trên toàn bộ khung hình
                inpainted_draw = result_frame.copy()

                resized_preview = self._img_resize(inpainted_draw)
                self.video_display_component.update_video_display(resized_preview, draw_selection=False)
                self.video_display_component.video_display.repaint()
                self.append_output(tr['SubtitleExtractorGUI']['PreviewSuccessLog'])
                
                InfoBar.success(
                    title="Xem trước hoàn tất (5 frame)",
                    content="Đã hiển thị bản xem trước 5 khung hình thử nghiệm. Để tiến hành xóa toàn bộ video và xuất tệp, vui lòng bấm nút 'Bắt Đầu'.",
                    duration=6000,
                    position=InfoBarPosition.TOP, parent=self
                )
            else:
                e, combined_mask = error_info
                self.append_output(tr['SubtitleExtractorGUI']['PreviewFailedLog'].format(e))
                # Fallback: vẽ đè màu đỏ lên vùng mặt nạ
                red_overlay = np.zeros_like(result_frame)
                red_overlay[:, :] = [0, 0, 255]  # BGR
                mask_bool = combined_mask > 0
                fallback_frame = result_frame.copy()
                if np.any(mask_bool):
                    fallback_frame[mask_bool] = cv2.addWeighted(result_frame, 0.4, red_overlay, 0.6, 0)[mask_bool]
                resized_preview = self._img_resize(fallback_frame)
                self.video_display_component.update_video_display(resized_preview, draw_selection=False)
                self.video_display_component.video_display.repaint()
                InfoBar.warning(
                    title=tr['Setting']['MaskPreview'],
                    content=tr['SubtitleExtractorGUI']['PreviewFallbackContent'],
                    duration=3500,
                    position=InfoBarPosition.TOP, parent=self
                )
        except Exception as ex:
            traceback.print_exc()
            self.append_output(tr['SubtitleExtractorGUI']['PreviewError'].format(ex))
            self.mask_preview_button.setEnabled(True)
            self.mask_preview_button.setText(tr['Setting'].get('MaskPreview', 'Xem trước kết quả'))

    def pause_resume_button_clicked(self):
        # Đổi trạng thái tạm dừng hàng đợi
        self.is_queue_paused = not self.is_queue_paused
        
        # Cập nhật nhãn và icon của nút tương ứng
        if self.is_queue_paused:
            self.pause_resume_button.setText(tr['Setting'].get('ResumeQueue', "Tiếp tục hàng đợi"))
            self.pause_resume_button.setIcon(FluentIcon.PLAY)
            self.append_output(tr['SubtitleExtractorGUI']['QueuePausedLog'])
            InfoBar.warning(
                title=tr['SubtitleExtractorGUI']['PauseQueueTitle'],
                content=tr['SubtitleExtractorGUI']['PauseQueueContent'],
                duration=3500,
                position=InfoBarPosition.TOP, parent=self
            )
        else:
            self.pause_resume_button.setText(tr['Setting'].get('PauseQueue', "Tạm dừng hàng đợi"))
            self.pause_resume_button.setIcon(FluentIcon.PAUSE)
            self.append_output(tr['SubtitleExtractorGUI']['QueueResumedLog'])
            InfoBar.success(
                title=tr['SubtitleExtractorGUI']['ResumeQueueTitle'],
                content=tr['SubtitleExtractorGUI']['ResumeQueueContent'],
                duration=3000,
                position=InfoBarPosition.TOP, parent=self
            )

    def open_file(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            tr['SubtitleExtractorGUI']['Open'],
            "",
            "All Files (*.*);;Video Files (*.mp4 *.flv *.wmv *.avi *.mkv *.mov);;Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if files:
            files_loaded = []
            # 倒序打开, 确保第一个视频截图显示在屏幕上
            for path in reversed(files):
                if self.load_video(path):
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {path}")
                    files_loaded.append(path)
                else:
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoFailed']}: {path}")
            # 正序添加, 确保任务列表顺序一致
            for path in reversed(files_loaded):
                # 添加到任务列表
                self.task_list_component.add_task(path)
                index = max(0, self.task_list_component.find_task_index_by_path(path))
                self.task_list_component.select_task(index)

    def closeEvent(self, event):
        """窗口关闭时断开信号连接并清理资源"""
        try:
            # 通知 worker 线程停止
            self._stop_event.set()
            # 终止子进程
            ProcessManager.instance().terminate_all()
            # 等待 worker 线程结束（最多5秒）
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5)

            # 断开信号连接
            try:
                self.progress_signal.disconnect(self.update_progress)
            except (RuntimeError, TypeError):
                pass
            try:
                self.append_log_signal.disconnect(self.append_log)
            except (RuntimeError, TypeError):
                pass
            try:
                self.update_preview_with_comp_signal.disconnect(self.update_preview_with_comp)
            except (RuntimeError, TypeError):
                pass
            try:
                self.task_error_signal.disconnect(self.on_task_error)
            except (RuntimeError, TypeError):
                pass
            try:
                self.toggle_buttons_signal.disconnect(self._toggle_buttons)
            except (RuntimeError, TypeError):
                pass
            try:
                self.video_display_component.video_slider.valueChanged.disconnect(self.slider_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.video_display_component.ab_sections_changed.disconnect(self.ab_sections_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.video_display_component.selections_changed.disconnect(self.selections_changed)
            except (RuntimeError, TypeError):
                pass
            # 释放视频资源
            with self._video_cap_lock:
                if self.video_cap:
                    self.video_cap.release()
                    self.video_cap = None
        except Exception as e:
            print(f"Error during close window:", e)
        super().closeEvent(event)

    def retranslateUi(self):
        """Cập nhật giao diện của HomeInterface khi đổi ngôn ngữ nóng"""
        self.file_button.setText(tr['SubtitleExtractorGUI']['Open'])
        self.add_area_button.setText(tr['Setting']['AddArea'])
        self.add_area_button.setToolTip(tr['Setting']['AddAreaTooltip'])
        self.mask_preview_button.setText(tr['Setting']['MaskPreview'])
        self.mask_preview_button.setToolTip(tr['Setting']['MaskPreviewTooltip'])
        self.run_button.setText(tr['SubtitleExtractorGUI']['Run'])
        self.stop_button.setText(tr['SubtitleExtractorGUI']['Stop'])
        if self.is_queue_paused:
            self.pause_resume_button.setText(tr['Setting'].get('ResumeQueue', "Tiếp tục hàng đợi"))
        else:
            self.pause_resume_button.setText(tr['Setting'].get('PauseQueue', "Tạm dừng hàng đợi"))
        self.setting_interface.retranslateUi()
        self.task_list_component.retranslateUi()
        self.video_display_component.retranslateUi()
    
    def open_downloaded_video(self, filepath):
        self._session_checked = True
        if self.load_video(filepath):
            self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {filepath}")
            self.task_list_component.add_task(filepath)
            index = max(0, self.task_list_component.find_task_index_by_path(filepath))
            self.task_list_component.select_task(index)

    