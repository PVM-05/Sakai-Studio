import os
import cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu
from PySide6.QtCore import Qt, Signal, QRect, QRectF, QObject, QEvent, QUrl, QPoint
from PySide6.QtGui import QAction, QShortcut, QCursor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6 import QtCore, QtWidgets, QtGui 
from qfluentwidgets import qconfig, CardWidget, HollowHandleStyle, TransparentToolButton, FluentIcon

from backend.config import config, tr


def format_seconds_to_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_sec = int(seconds)
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def make_white_icon(fluent_icon):
    if hasattr(fluent_icon, 'icon'):
        return fluent_icon.icon(color=QtGui.QColor(255, 255, 255))
    return fluent_icon

class VideoDisplayComponent(QWidget):
    """视频显示组件，包含视频预览和选择框功能"""
    
    # 定义信号
    selections_changed = Signal(list)  # 选择框变化信号
    ab_sections_changed = Signal(list)  # AB分区变化信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # 初始化变量
        self.is_drawing = False
        self.selection_rect = (0, 0, 0, 0)  # 当前正在绘制或调整的选区 (ymin, ymax, xmin, xmax)
        self.selection_rects = []  # 存储多个选区，每个元素为 (ymin, ymax, xmin, xmax)
        self.active_selection_index = -1  # 当前活动选区的索引
        self.drag_start_pos = None
        self.resize_edge = None
        self.edge_size = 10  # 调整大小的边缘区域
        self.enable_mouse_events = True  # 控制是否启用鼠标事件
        
        # AB分区标记相关变量
        self.ab_sections = []  # 存储AB分区标记 [range(start, end), ...]
        self.current_ab_start = -1  # 当前AB分区的起点
        
        # 创建右键菜单
        self.__init_context_menu()
        
        # 获取屏幕大小
        screen = QtWidgets.QApplication.primaryScreen().size()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        
        # 设置视频预览区域大小（根据屏幕宽度动态调整）
        self.video_preview_width = 960
        self.video_preview_height = self.video_preview_width * 9 // 16
        if self.screen_width // 2 < 960:
            self.video_preview_width = 640
            self.video_preview_height = self.video_preview_width * 9 // 16
            
        # 视频相关参数
        self.frame_width = None
        self.frame_height = None
        self.scaled_width = None
        self.scaled_height = None
        self.border_left = 0
        self.border_top = 0
        self.fps = 30

        # Timer phát video tự động
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self._on_playback_tick)

        # QMediaPlayer & QAudioOutput phát âm thanh đồng bộ
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.media_player.positionChanged.connect(self._on_audio_position_changed)
        self.video_source_path = None

        # 🔴 Red Mask Overlay & 🖌️ Freehand Brush Masking
        self.show_red_mask = False
        self.draw_mode = 'box'  # 'box' hoặc 'brush'
        self.brush_size = 20
        self.freehand_strokes = []  # list các stroke {'radius': int, 'points': [(x_ratio, y_ratio)...]}
        self.current_stroke = None

        self.__init_widgets()
        self.__init_shotcuts()
        
    def __init_widgets(self):
        """初始化组件"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频预览区域和进度条容器
        self.video_container = CardWidget(self)
        self.video_container.setObjectName('videoContainer')
        video_layout = QVBoxLayout()
        video_layout.setSpacing(0)
        video_layout.setContentsMargins(2, 2, 2, 2)
        video_layout.setAlignment(Qt.AlignCenter)
        
        # 创建内部黑色背景容器
        self.black_container = QWidget(self)
        self.black_container.setObjectName('blackContainer')
        self.black_container.setStyleSheet("""
            #blackContainer {
                background-color: black;
                border-radius: 10px;
                border: 0px solid transparent;
            }
        """)
        black_layout = QVBoxLayout()
        black_layout.setContentsMargins(0, 0, 0, 0)
        black_layout.setSpacing(0)
        black_layout.setAlignment(Qt.AlignCenter)
        
        # 视频显示标签
        self.video_display = QtWidgets.QLabel()
        self.video_display.setStyleSheet("""
            background-color: black;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            border: 0px solid transparent;
        """)
        self.video_display.setMinimumSize(480, 270)
        
        self.video_display.setMouseTracking(True)
        self.video_display.setScaledContents(True)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.mousePressEvent = self.selection_mouse_press
        self.video_display.mouseMoveEvent = self.selection_mouse_move
        self.video_display.mouseReleaseEvent = self.selection_mouse_release
        
        # 视频滑块 - Thanh trượt Đỏ chuẩn YouTube
        self.video_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(1)
        self.video_slider.setFixedHeight(16)
        self.video_slider.setMaximum(100)
        self.video_slider.setValue(1)
        self.video_slider.setCursor(Qt.PointingHandCursor)
        self.video_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(255, 255, 255, 0.25);
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #FF0000;
                border-radius: 2px;
            }
            QSlider::add-page:horizontal {
                background: rgba(255, 255, 255, 0.25);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FF0000;
                width: 12px;
                height: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #FF0000;
                width: 14px;
                height: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
        """)
        self.video_slider.valueChanged.connect(self.update_time_display)
        
        # Khung chứa hiển thị video linh hoạt theo tỉ lệ khung hình (Aspect Ratio)
        self.video_display.setObjectName('videoDisplay')
        self.ratio_container = QWidget()
        ratio_layout = QVBoxLayout(self.ratio_container)
        ratio_layout.setContentsMargins(0, 0, 0, 0)
        ratio_layout.setAlignment(Qt.AlignCenter)
        ratio_layout.addWidget(self.video_display)

        self.target_aspect_ratio = 9.0 / 16.0
        self.ratio_container.setMinimumWidth(240)

        black_layout.addWidget(self.ratio_container, 0, Qt.AlignCenter)

        outer_self = self
        class RatioEventFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Resize:
                    aspect = getattr(outer_self, 'target_aspect_ratio', 9.0 / 16.0)
                    if aspect <= 1.0:  # Landscape or Square
                        new_h = int(obj.width() * aspect)
                        if new_h > 480:
                            new_h = 480
                        obj.setFixedHeight(new_h)
                return False

        ratio_filter = RatioEventFilter(self.ratio_container)
        self.ratio_container.installEventFilter(ratio_filter)

        # 进度条和滑块容器 (Bảng điều khiển dạng YouTube Player)
        control_container = QWidget(self)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(8, 4, 8, 6)
        control_layout.setSpacing(4)
        control_layout.addWidget(self.video_slider)
        
        # Thanh nút bấm & Thời gian chuẩn YouTube
        player_bar_layout = QtWidgets.QHBoxLayout()
        player_bar_layout.setContentsMargins(4, 2, 4, 2)
        player_bar_layout.setSpacing(6)

        btn_qss = """
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
                min-width: 26px;
                min-height: 26px;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            QToolButton:disabled {
                opacity: 0.3;
            }
        """

        # 1. Nút Phát / Tạm dừng
        self.play_btn = TransparentToolButton(make_white_icon(FluentIcon.PLAY), self)
        self.play_btn.setToolTip("Phát / Tạm dừng")
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setStyleSheet(btn_qss)
        self.play_btn.clicked.connect(self.toggle_play_pause)

        # 2. Nút Tua lùi 5s
        self.rewind_btn = TransparentToolButton(make_white_icon(FluentIcon.LEFT_ARROW), self)
        self.rewind_btn.setToolTip("Tua lùi 5s")
        self.rewind_btn.setCursor(Qt.PointingHandCursor)
        self.rewind_btn.setStyleSheet(btn_qss)
        self.rewind_btn.clicked.connect(self.seek_rewind)

        # 3. Nút Tua tới 5s
        self.forward_btn = TransparentToolButton(make_white_icon(FluentIcon.RIGHT_ARROW), self)
        self.forward_btn.setToolTip("Tua tới 5s")
        self.forward_btn.setCursor(Qt.PointingHandCursor)
        self.forward_btn.setStyleSheet(btn_qss)
        self.forward_btn.clicked.connect(self.seek_forward)

        # 4. Nút Dừng và về đầu
        self.stop_btn = TransparentToolButton(make_white_icon(FluentIcon.CANCEL if hasattr(FluentIcon, 'CANCEL') else FluentIcon.CLOSE), self)
        self.stop_btn.setToolTip("Dừng và về đầu")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(btn_qss)
        self.stop_btn.clicked.connect(self.stop_playback)

        # 5. Đồng hồ Thời gian chuẩn YouTube: 00:00 / 01:30
        self.time_display_label = QtWidgets.QLabel("00:00 / 00:00")
        self.time_display_label.setStyleSheet("color: #EEEEEE; font-weight: 500; font-size: 13px; font-family: 'Segoe UI', Arial, sans-serif; padding-left: 6px;")

        # 6. Nút Nút Tắt/Mở Loa âm lượng chuẩn YouTube
        vol_icon = FluentIcon.VOLUME_LEVEL if hasattr(FluentIcon, 'VOLUME_LEVEL') else FluentIcon.MUSIC
        self.volume_btn = TransparentToolButton(make_white_icon(vol_icon), self)
        self.volume_btn.setToolTip("Tắt / Bật tiếng")
        self.volume_btn.setCursor(Qt.PointingHandCursor)
        self.volume_btn.setStyleSheet(btn_qss)
        self.volume_btn.clicked.connect(self.toggle_mute)

        # 7. Thanh cuộn âm lượng 0-100%
        self.volume_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(75)
        self.volume_slider.setCursor(Qt.PointingHandCursor)
        self.volume_slider.setToolTip("Âm lượng: 100%")
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #FFFFFF;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal:hover {
                background: #FF0000;
            }
        """)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)

        # 8. Nút Bật/Tắt Xem Trước Red Mask
        red_icon = FluentIcon.VIEW if hasattr(FluentIcon, 'VIEW') else FluentIcon.SEARCH
        self.red_mask_btn = TransparentToolButton(make_white_icon(red_icon), self)
        self.red_mask_btn.setToolTip("Bật / Tắt xem trước Red Mask (Lớp phủ đỏ)")
        self.red_mask_btn.setCursor(Qt.PointingHandCursor)
        self.red_mask_btn.setStyleSheet(btn_qss)
        self.red_mask_btn.clicked.connect(self.toggle_red_mask)

        # 9. Nút Chế độ vẽ: Khoanh Hộp ↔ Cọ Vẽ Tự Do
        brush_icon = FluentIcon.BRUSH if hasattr(FluentIcon, 'BRUSH') else FluentIcon.EDIT
        self.brush_mode_btn = TransparentToolButton(make_white_icon(brush_icon), self)
        self.brush_mode_btn.setToolTip("Chế độ: Khoanh Hộp (Mặc định)")
        self.brush_mode_btn.setCursor(Qt.PointingHandCursor)
        self.brush_mode_btn.setStyleSheet(btn_qss)
        self.brush_mode_btn.clicked.connect(self.toggle_draw_mode)

        # 10. Nút Xóa tất cả nét cọ
        clear_icon = FluentIcon.DELETE if hasattr(FluentIcon, 'DELETE') else FluentIcon.CLOSE
        self.clear_brush_btn = TransparentToolButton(make_white_icon(clear_icon), self)
        self.clear_brush_btn.setToolTip("Xóa tất cả nét cọ vẽ tự do")
        self.clear_brush_btn.setCursor(Qt.PointingHandCursor)
        self.clear_brush_btn.setStyleSheet(btn_qss)
        self.clear_brush_btn.clicked.connect(self.clear_freehand_strokes)

        # Badge số Frame ở góc phải (kiểu HD badge)
        self.frame_badge_label = QtWidgets.QLabel("1 / 100")
        self.frame_badge_label.setStyleSheet("color: #CCCCCC; background-color: rgba(255, 255, 255, 0.12); border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold; font-family: Consolas, monospace;")

        # Xếp thứ tự chuẩn YouTube: Play ➔ Tua lùi ➔ Tua tới ➔ Dừng ➔ Đồng hồ ➔ Loa ➔ Slider ➔ Red Mask ➔ Brush ➔ Xóa Cọ
        player_bar_layout.addWidget(self.play_btn)
        player_bar_layout.addWidget(self.rewind_btn)
        player_bar_layout.addWidget(self.forward_btn)
        player_bar_layout.addWidget(self.stop_btn)
        player_bar_layout.addWidget(self.time_display_label)
        player_bar_layout.addSpacing(10)
        player_bar_layout.addWidget(self.volume_btn)
        player_bar_layout.addWidget(self.volume_slider)
        player_bar_layout.addSpacing(10)
        player_bar_layout.addWidget(self.red_mask_btn)
        player_bar_layout.addWidget(self.brush_mode_btn)
        player_bar_layout.addWidget(self.clear_brush_btn)
        player_bar_layout.addStretch(1)
        player_bar_layout.addWidget(self.frame_badge_label)

        control_layout.addLayout(player_bar_layout)

        control_container.setLayout(control_layout)
        control_container.setStyleSheet("""
            background-color: #0F0F0F;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;        
        """)
        black_layout.addWidget(control_container)
        
        self.black_container.setLayout(black_layout)
        video_layout.addWidget(self.black_container)
        self.video_container.setLayout(video_layout)
        main_layout.addWidget(self.video_container)

        # Chưa chọn video thì vô hiệu hóa nút bấm
        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled: bool):
        """Bật / Tắt bộ điều khiển video khi có / chưa có video"""
        self.play_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.rewind_btn.setEnabled(enabled)
        self.forward_btn.setEnabled(enabled)
        self.volume_btn.setEnabled(enabled)
        self.volume_slider.setEnabled(enabled)
        self.red_mask_btn.setEnabled(enabled)
        self.brush_mode_btn.setEnabled(enabled)
        self.clear_brush_btn.setEnabled(enabled)
        self.video_slider.setEnabled(enabled)

    def set_player_mode(self, mode: str):
        """Cấu hình giao diện trình phát phù hợp với từng tab chức năng:
        - 'remover': Đầy đủ trình phát + Cọ vẽ tự do + Red Mask + Khung chọn
        - 'extractor': Trình phát + Khung chọn khoanh vùng phụ đề (Không có cọ vẽ/Red Mask)
        - 'translator' / 'player': Trình phát video tiêu chuẩn (Không vẽ/khoanh vùng)
        """
        self.player_mode = mode
        if mode == "remover":
            self.red_mask_btn.setVisible(True)
            self.brush_mode_btn.setVisible(True)
            self.clear_brush_btn.setVisible(True)
            self.enable_mouse_events = True
        elif mode == "extractor":
            self.red_mask_btn.setVisible(False)
            self.brush_mode_btn.setVisible(False)
            self.clear_brush_btn.setVisible(False)
            self.enable_mouse_events = True
            self.draw_mode = 'box'
        else:
            self.red_mask_btn.setVisible(False)
            self.brush_mode_btn.setVisible(False)
            self.clear_brush_btn.setVisible(False)
            self.enable_mouse_events = False
            self.draw_mode = 'box'

    def toggle_mute(self):
        """Bật / Tắt tiếng preview video"""
        if self.volume_slider.value() > 0:
            self._last_volume = self.volume_slider.value()
            self.volume_slider.setValue(0)
        else:
            restore_val = getattr(self, '_last_volume', 80)
            self.volume_slider.setValue(restore_val if restore_val > 0 else 80)

    def on_volume_changed(self, value: int):
        """Thay đổi âm lượng âm thanh (0-100)"""
        vol_float = max(0.0, min(1.0, value / 100.0))
        if hasattr(self, 'audio_output'):
            try:
                self.audio_output.setVolume(vol_float)
            except Exception:
                pass
        self.volume_slider.setToolTip(f"Âm lượng: {value}%")
        if value == 0:
            mute_icon = FluentIcon.MUTE if hasattr(FluentIcon, 'MUTE') else FluentIcon.CANCEL
            self.volume_btn.setIcon(make_white_icon(mute_icon))
            self.volume_btn.setToolTip("Bật tiếng")
        else:
            vol_icon = FluentIcon.VOLUME_LEVEL if hasattr(FluentIcon, 'VOLUME_LEVEL') else FluentIcon.MUSIC
            self.volume_btn.setIcon(make_white_icon(vol_icon))
            self.volume_btn.setToolTip("Tắt tiếng")

    def set_video_path(self, path: str):
        """Cài đặt đường dẫn tệp video nguồn để phát âm thanh đồng bộ"""
        self.video_source_path = path
        if path and os.path.exists(path) and not path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
            try:
                self.media_player.stop()
                self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
            except Exception as e:
                print(f"Lỗi nạp âm thanh video: {e}")
        else:
            try:
                self.media_player.stop()
                self.media_player.setSource(QUrl())
            except Exception:
                pass

    def sync_audio_position(self, frame_no=None):
        """Đồng bộ vị trí phát âm thanh theo khung hình hiện tại"""
        if frame_no is None:
            frame_no = self.video_slider.value()
        fps_val = self.fps if hasattr(self, 'fps') and self.fps and self.fps > 0 else 30
        target_ms = int((frame_no - 1) / fps_val * 1000)
        try:
            if abs(self.media_player.position() - target_ms) > 300:
                self.media_player.setPosition(target_ms)
        except Exception:
            pass

    def toggle_play_pause(self):
        """Phát hoặc Tạm dừng video kèm âm thanh"""
        if not self.play_btn.isEnabled() or self.video_slider.maximum() <= 1:
            return
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()
            try:
                self.media_player.pause()
            except Exception:
                pass
            self.play_btn.setIcon(make_white_icon(FluentIcon.PLAY))
            self.play_btn.setToolTip("Phát / Tạm dừng")
        else:
            if self.video_slider.value() >= self.video_slider.maximum():
                self.video_slider.setValue(1)
            fps_val = self.fps if hasattr(self, 'fps') and self.fps and self.fps > 0 else 30
            interval_ms = max(15, int(1000 / fps_val))

            # Đồng bộ vị trí âm thanh
            current_frame = self.video_slider.value()
            current_ms = int((current_frame - 1) / fps_val * 1000)
            try:
                self.media_player.setPosition(current_ms)
                self.media_player.play()
            except Exception as e:
                print(f"Lỗi phát âm thanh: {e}")

            self.playback_timer.start(interval_ms)
            self.play_btn.setIcon(make_white_icon(FluentIcon.PAUSE if hasattr(FluentIcon, 'PAUSE') else FluentIcon.CANCEL))
            self.play_btn.setToolTip("Tạm dừng")

    def stop_playback(self):
        """Dừng video và quay về khung hình đầu tiên"""
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()
        try:
            self.media_player.stop()
            self.media_player.setPosition(0)
        except Exception:
            pass
        self.play_btn.setIcon(make_white_icon(FluentIcon.PLAY))
        self.play_btn.setToolTip("Phát / Tạm dừng")
        self.video_slider.setValue(1)

    def seek_rewind(self):
        """Tua lùi 5 giây"""
        if not self.rewind_btn.isEnabled() or self.video_slider.maximum() <= 1:
            return
        fps_val = self.fps if hasattr(self, 'fps') and self.fps and self.fps > 0 else 30
        step = max(1, int(5 * fps_val))
        current_val = self.video_slider.value()
        new_val = max(self.video_slider.minimum(), current_val - step)
        self.video_slider.setValue(new_val)
        self.sync_audio_position(new_val)

    def seek_forward(self):
        """Tua tới 5 giây"""
        if not self.forward_btn.isEnabled() or self.video_slider.maximum() <= 1:
            return
        fps_val = self.fps if hasattr(self, 'fps') and self.fps and self.fps > 0 else 30
        step = max(1, int(5 * fps_val))
        current_val = self.video_slider.value()
        new_val = min(self.video_slider.maximum(), current_val + step)
        self.video_slider.setValue(new_val)
        self.sync_audio_position(new_val)

    def _on_audio_position_changed(self, position_ms: int):
        """Đồng bộ mượt mờ 100% không bị lag âm thanh bằng clock thực tế của QMediaPlayer"""
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            fps_val = self.fps if hasattr(self, 'fps') and self.fps and self.fps > 0 else 30
            target_frame = max(1, min(self.video_slider.maximum(), int(position_ms / 1000.0 * fps_val) + 1))
            if abs(self.video_slider.value() - target_frame) >= 1:
                self.video_slider.setValue(target_frame)

    def _on_playback_tick(self):
        """Tiến tới khung hình tiếp theo khi đang phát"""
        # Nếu media_player đang phát âm thanh, việc đồng bộ vị trí chuẩn do _on_audio_position_changed đảm nhiệm
        if hasattr(self, 'media_player') and self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            return
        current_val = self.video_slider.value()
        max_val = self.video_slider.maximum()
        if current_val < max_val:
            self.video_slider.setValue(current_val + 1)
        else:
            self.stop_playback()

    def update_time_display(self):
        """Cập nhật đồng hồ thời gian chuẩn YouTube (MM:SS / MM:SS) và Badge Khung hình"""
        current_frame = self.video_slider.value()
        total_frames = max(1, self.video_slider.maximum())
        fps_val = self.fps if hasattr(self, 'fps') and self.fps and self.fps > 0 else 30

        current_sec = (current_frame - 1) / fps_val
        total_sec = total_frames / fps_val

        curr_str = format_seconds_to_hms(current_sec)
        total_str = format_seconds_to_hms(total_sec)

        # Rút gọn 00:MM:SS -> MM:SS kiểu YouTube
        if curr_str.startswith("00:"):
            curr_str = curr_str[3:]
        if total_str.startswith("00:"):
            total_str = total_str[3:]

        self.time_display_label.setText(f"{curr_str} / {total_str}")
        if hasattr(self, 'frame_badge_label'):
            self.frame_badge_label.setText(f"{current_frame} / {total_frames}")

    def __init_shotcuts(self):
        """初始化快捷键"""
        self.shortcut_ab_start = QShortcut(QtGui.QKeySequence("["), self)
        self.shortcut_ab_start.activated.connect(self.__handle_mark_for_ab_start)
        self.shortcut_ab_start.setContext(Qt.ApplicationShortcut)

        self.shortcut_ab_end = QShortcut(QtGui.QKeySequence("]"), self)
        self.shortcut_ab_end.activated.connect(self.__handle_mark_for_ab_end)
        self.shortcut_ab_end.setContext(Qt.ApplicationShortcut)

        self.shortcut_ab_delete = QShortcut(QtGui.QKeySequence("\\"), self)
        self.shortcut_ab_delete.activated.connect(self.__handle_delete_ab_section)
        self.shortcut_ab_delete.setContext(Qt.ApplicationShortcut)

        self.shortcut_delete_selection = QShortcut(QtGui.QKeySequence.Delete, self)
        self.shortcut_delete_selection.activated.connect(self.__handle_delete_selection)
        self.shortcut_delete_selection.setContext(Qt.ApplicationShortcut)

        self.shortcut_clear_all_selections = QShortcut(QtGui.QKeySequence("Ctrl+Delete"), self)
        self.shortcut_clear_all_selections.activated.connect(self.clear_selections)
        self.shortcut_clear_all_selections.setContext(Qt.ApplicationShortcut)

        # 添加左右键控制slider的快捷键
        self.shortcut_right = QShortcut(QtGui.QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.__adjust_slider_value(self.fps))
        self.shortcut_right.setContext(Qt.ApplicationShortcut)
        
        self.shortcut_left = QShortcut(QtGui.QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.__adjust_slider_value(-self.fps))
        self.shortcut_left.setContext(Qt.ApplicationShortcut)
        
        # 添加Ctrl+左右键控制slider的快捷键
        self.shortcut_ctrl_right = QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.shortcut_ctrl_right.activated.connect(lambda: self.__adjust_slider_value(self.fps*5))
        self.shortcut_ctrl_right.setContext(Qt.ApplicationShortcut)
        
        self.shortcut_ctrl_left = QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.shortcut_ctrl_left.activated.connect(lambda: self.__adjust_slider_value(-self.fps*5))
        self.shortcut_ctrl_left.setContext(Qt.ApplicationShortcut)
        
        # 添加Shift+左右键控制slider的快捷键
        self.shortcut_shift_right = QShortcut(QtGui.QKeySequence("Shift+Right"), self)
        self.shortcut_shift_right.activated.connect(lambda: self.__adjust_slider_value(1))
        self.shortcut_shift_right.setContext(Qt.ApplicationShortcut)
        
        self.shortcut_shift_left = QShortcut(QtGui.QKeySequence("Shift+Left"), self)
        self.shortcut_shift_left.activated.connect(lambda: self.__adjust_slider_value(-1))
        self.shortcut_shift_left.setContext(Qt.ApplicationShortcut)

    def update_video_display(self, frame, draw_selection=True):
        """Cập nhật khung hình xem trước chuẩn tỉ lệ không méo."""
        if frame is None:
            return

        target_w = getattr(self, 'video_preview_width', 640)
        target_h = getattr(self, 'video_preview_height', 360)
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Chuyển đổi OpenCV BGR sang QImage và QPixmap hiển thị lên QLabel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(image)
        
        # 创建带圆角的图像
        rounded_pix = QtGui.QPixmap(pix.size())
        rounded_pix.fill(Qt.transparent)  # 填充透明背景
        
        painter = QtGui.QPainter(rounded_pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)  # 抗锯齿
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        
        # 创建圆角路径
        path = QtGui.QPainterPath()
        rect = QRectF(0, 0, pix.width(), pix.height())
        
        # 手动创建只有左上和右上圆角的路径
        radius = 8
        path.moveTo(radius, 0)
        path.lineTo(pix.width() - radius, 0)
        path.arcTo(pix.width() - radius * 2, 0, radius * 2, radius * 2, 90, -90)
        path.lineTo(pix.width(), pix.height())
        path.lineTo(0, pix.height())
        path.lineTo(0, radius)
        path.arcTo(0, 0, radius * 2, radius * 2, 180, -90)
        path.closeSubpath()
        
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)
        painter.end()
        
        # 保存当前的pixmap用于绘制选择框
        self.current_pixmap = rounded_pix.copy()
        
        self.video_display.setPixmap(rounded_pix)
            
        # 更新视频显示
        self.update_preview_with_rect(draw_selection=draw_selection)
    
    def toggle_red_mask(self):
        """Bật / Tắt xem trước Red Mask mờ"""
        self.show_red_mask = not self.show_red_mask
        if self.show_red_mask:
            self.red_mask_btn.setToolTip("Tắt xem trước Red Mask (Đang bật)")
        else:
            self.red_mask_btn.setToolTip("Bật xem trước Red Mask (Lớp phủ đỏ)")
        self.update_preview_with_rect()

    def toggle_draw_mode(self):
        """Chuyển đổi qua lại giữa Chế độ Khoanh Hộp ↔ Vẽ Cọ Tự Do"""
        if self.draw_mode == 'box':
            self.draw_mode = 'brush'
            self.brush_mode_btn.setToolTip("Chế độ: Vẽ Cọ Tự Do (Đang bật)")
            self.video_display.setCursor(Qt.CrossCursor)
        else:
            self.draw_mode = 'box'
            self.brush_mode_btn.setToolTip("Chế độ: Khoanh Hộp (Mặc định)")
            self.video_display.setCursor(Qt.ArrowCursor)
        self.update_preview_with_rect()

    def clear_freehand_strokes(self):
        """Xóa tất cả các nét cọ vẽ tự do"""
        self.freehand_strokes.clear()
        self.current_stroke = None
        self.update_preview_with_rect()

    def update_preview_with_rect(self, rect=None, draw_selection=True):
        """更新带有选择框的预览"""
        if not hasattr(self, 'current_pixmap') or self.current_pixmap is None:
            return
            
        # 如果提供了新的矩形，使用它
        if rect is not None and self.active_selection_index >= 0:
            self.selection_rects[self.active_selection_index] = rect
            
        # 创建一个副本用于绘制
        pixmap_copy = self.current_pixmap.copy()
        painter = QtGui.QPainter(pixmap_copy)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        pixmap_size = self.current_pixmap.size()
        pw, ph = pixmap_size.width(), pixmap_size.height()

        # 🔴 1. VẼ RED MASK OVERLAY NẾU ĐANG BẬT [👁️ Mask Đỏ]
        if self.show_red_mask:
            red_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 110))
            red_pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 160), 2)
            painter.setBrush(red_brush)
            painter.setPen(red_pen)

            # Vẽ phủ màu đỏ lên các hộp chữ nhật
            for r in self.selection_rects:
                ymin, ymax, xmin, xmax = r
                pixel_rect = QRect(int(xmin * pw), int(ymin * ph), int((xmax - xmin) * pw), int((ymax - ymin) * ph))
                painter.drawRect(pixel_rect)

            # Vẽ phủ màu đỏ lên nét cọ tự do
            all_strokes = list(self.freehand_strokes)
            if getattr(self, 'current_stroke', None) and self.current_stroke.get('points'):
                all_strokes.append(self.current_stroke)

            for stroke in all_strokes:
                pts = stroke.get('points', [])
                if not pts:
                    continue
                r_val = stroke.get('radius', 20)
                thick = max(6, int(r_val)) if r_val > 1 else max(6, int(r_val * pw * 2))
                stroke_pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 140), thick, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(stroke_pen)
                
                if len(pts) == 1:
                    px, py = int(pts[0][0] * pw), int(pts[0][1] * ph)
                    painter.drawPoint(px, py)
                else:
                    for i in range(len(pts) - 1):
                        p1 = QPoint(int(pts[i][0] * pw), int(pts[i][1] * ph))
                        p2 = QPoint(int(pts[i + 1][0] * pw), int(pts[i + 1][1] * ph))
                        painter.drawLine(p1, p2)

        # 🖌️ 2. VẼ CÁC NÉT CỌ TỰ DO (COLORFUL STROKES NẾU KHÔNG BẬT RED MASK)
        if not self.show_red_mask:
            all_strokes = list(self.freehand_strokes)
            if getattr(self, 'current_stroke', None) and self.current_stroke.get('points'):
                all_strokes.append(self.current_stroke)

            for stroke in all_strokes:
                pts = stroke.get('points', [])
                if not pts:
                    continue
                r_val = stroke.get('radius', 20)
                thick = max(6, int(r_val)) if r_val > 1 else max(6, int(r_val * pw * 2))
                stroke_pen = QtGui.QPen(QtGui.QColor(0, 230, 255, 180), thick, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(stroke_pen)
                
                if len(pts) == 1:
                    px, py = int(pts[0][0] * pw), int(pts[0][1] * ph)
                    painter.drawPoint(px, py)
                else:
                    for i in range(len(pts) - 1):
                        p1 = QPoint(int(pts[i][0] * pw), int(pts[i][1] * ph))
                        p2 = QPoint(int(pts[i + 1][0] * pw), int(pts[i + 1][1] * ph))
                        painter.drawLine(p1, p2)

        # 3. VẼ HỘP CHỮ NHẬT KHUNG VIỀN SÁNG
        if draw_selection:
            painter.setBrush(Qt.NoBrush)
            for i, rect in enumerate(self.selection_rects):
                if i == self.active_selection_index:
                    pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
                else:
                    pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
                painter.setPen(pen)
                ymin, ymax, xmin, xmax = rect
                pixel_rect = QRect(int(xmin * pw), int(ymin * ph), int((xmax - xmin) * pw), int((ymax - ymin) * ph))
                painter.drawRect(pixel_rect)
            
            if self.is_drawing and any(self.selection_rect):
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
                painter.setPen(pen)
                ymin, ymax, xmin, xmax = self.selection_rect
                pixel_rect = QRect(int(xmin * pw), int(ymin * ph), int((xmax - xmin) * pw), int((ymax - ymin) * ph))
                painter.drawRect(pixel_rect)
            
        # 绘制AB分区标记
        total_frames = self.video_slider.maximum()
        if total_frames > 0 and self.ab_sections:
            # 在视频显示区域下方5像素处绘制AB分区标记
            ab_rect_height = 5
            ab_rect_y = pixmap_copy.height() - ab_rect_height
            
            # 设置半透明白色画刷
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, 128))  # 半透明白色
            
            # 计算可用宽度（考虑左右边距）
            left_margin = 15
            right_margin = 15
            available_width = pixmap_copy.width() - left_margin - right_margin
            
            for section_range in self.ab_sections:
                # 计算相对位置
                start_x = left_margin + int((section_range.start / total_frames) * available_width)
                end_x = left_margin + int((section_range.stop / total_frames) * available_width)
                
                # 绘制AB分区矩形
                painter.drawRect(start_x, ab_rect_y, end_x - start_x, ab_rect_height)
        
        # 绘制current_ab_start的高亮竖线
        if self.current_ab_start >= 0 and total_frames > 0:
            # 计算可用宽度（考虑左右边距）
            left_margin = 15
            right_margin = 15
            available_width = pixmap_copy.width() - left_margin - right_margin
            
            # 计算current_ab_start的相对位置
            start_x = left_margin + int((self.current_ab_start / total_frames) * available_width)
            
            # 设置高亮白色画笔
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255))  # 纯白色
            pen.setWidth(2)
            painter.setPen(pen)
            
            # 绘制高亮竖线，高度为5像素
            ab_line_height = 5
            ab_line_y = pixmap_copy.height() - ab_line_height
            painter.drawLine(start_x, ab_line_y, start_x, pixmap_copy.height())
        
        painter.end()
        
        # 更新显示
        self.video_display.setPixmap(pixmap_copy)
    
    def selection_mouse_press(self, event):
        """鼠标按下事件处理"""
        if not self.enable_mouse_events:
            return
        
        # 右键点击显示上下文菜单
        if event.button() == Qt.RightButton:
            self.context_menu.exec_(event.globalPos())
            return
        
        video_display_width = self.video_display.width()
        video_display_height = self.video_display.height()

        pos = event.pos()
        y_ratio = max(0, min(1, (pos.y() - self.border_top) / video_display_height if video_display_height > 0 else 0))
        x_ratio = max(0, min(1, (pos.x() - self.border_left) / video_display_width if video_display_width > 0 else 0))

        # 🖌️ CHẾ ĐỘ VẼ CỌ TỰ DO
        if getattr(self, 'draw_mode', 'box') == 'brush':
            self.is_drawing_stroke = True
            self.current_stroke = {
                'radius': getattr(self, 'brush_size', 20),
                'points': [(x_ratio, y_ratio)]
            }
            self.update_preview_with_rect()
            return

        # 开始绘制新选区 (Ctrl + Click)
        if event.modifiers() & Qt.ControlModifier:
            self.selection_rect = (y_ratio, y_ratio, x_ratio, x_ratio)
            self.drag_start_pos = (y_ratio, x_ratio)
            self.resize_edge = None
            self.active_selection_index = -1
            return
        
        # 双击重置所有选区
        if event.type() == QEvent.MouseButtonDblClick:
            self.clear_selections()
            return
        
        # 检查是否点击了已有选区
        pos = event.pos()
        y_ratio = (pos.y() - self.border_top) / video_display_height if video_display_height > 0 else 0
        x_ratio = (pos.x() - self.border_left) / video_display_width if video_display_width > 0 else 0
        
        clicked_index = -1
        for i, rect in enumerate(self.selection_rects):
            # 将比例坐标转换为像素坐标用于检测
            ymin, ymax, xmin, xmax = rect
            pixel_rect = QRect(
                int(xmin * video_display_width) + self.border_left,
                int(ymin * video_display_height) + self.border_top,
                int((xmax - xmin) * video_display_width),
                int((ymax - ymin) * video_display_height)
            )
            
            # 检查是否在选区边缘（用于调整大小）
            if self.is_on_rect_edge(pos, pixel_rect):
                clicked_index = i
                self.active_selection_index = i
                self.resize_edge = self.get_resize_edge(pos, pixel_rect)
                self.drag_start_pos = (y_ratio, x_ratio)
                self.update_preview_with_rect()
                return
            # 检查是否在选区内部（用于移动）
            elif pixel_rect.contains(pos):
                clicked_index = i
                self.active_selection_index = i
                self.resize_edge = "move"
                self.drag_start_pos = (y_ratio, x_ratio)
                self.update_preview_with_rect()
                return
        
        # 如果没有点击任何选区，开始绘制新选区
        if clicked_index == -1:
            self.is_drawing = True
            self.selection_rect = (y_ratio, y_ratio, x_ratio, x_ratio)
            self.drag_start_pos = (y_ratio, x_ratio)
            self.resize_edge = None
            self.active_selection_index = -1

    def is_on_rect_edge(self, pos, pixel_rect):
        """检查点是否在矩形边缘
        注意：这里的pixel_rect是已经转换为像素坐标的QRect对象
        """
        # 右下角
        if abs(pos.x() - pixel_rect.right()) <= self.edge_size and abs(pos.y() - pixel_rect.bottom()) <= self.edge_size:
            return True
        # 右上角
        elif abs(pos.x() - pixel_rect.right()) <= self.edge_size and abs(pos.y() - pixel_rect.top()) <= self.edge_size:
            return True
        # 左下角
        elif abs(pos.x() - pixel_rect.left()) <= self.edge_size and abs(pos.y() - pixel_rect.bottom()) <= self.edge_size:
            return True
        # 左上角
        elif abs(pos.x() - pixel_rect.left()) <= self.edge_size and abs(pos.y() - pixel_rect.top()) <= self.edge_size:
            return True
        # 左边缘
        elif abs(pos.x() - pixel_rect.left()) <= self.edge_size and pixel_rect.top() <= pos.y() <= pixel_rect.bottom():
            return True
        # 右边缘
        elif abs(pos.x() - pixel_rect.right()) <= self.edge_size and pixel_rect.top() <= pos.y() <= pixel_rect.bottom():
            return True
        # 上边缘
        elif abs(pos.y() - pixel_rect.top()) <= self.edge_size and pixel_rect.left() <= pos.x() <= pixel_rect.right():
            return True
        # 下边缘
        elif abs(pos.y() - pixel_rect.bottom()) <= self.edge_size and pixel_rect.left() <= pos.x() <= pixel_rect.right():
            return True
        return False

    def get_resize_edge(self, pos, rect):
        """获取调整大小的边缘类型"""
        # 右下角
        if abs(pos.x() - rect.right()) <= self.edge_size and abs(pos.y() - rect.bottom()) <= self.edge_size:
            return "bottomright"
        # 右上角
        elif abs(pos.x() - rect.right()) <= self.edge_size and abs(pos.y() - rect.top()) <= self.edge_size:
            return "topright"
        # 左下角
        elif abs(pos.x() - rect.left()) <= self.edge_size and abs(pos.y() - rect.bottom()) <= self.edge_size:
            return "bottomleft"
        # 左上角
        elif abs(pos.x() - rect.left()) <= self.edge_size and abs(pos.y() - rect.top()) <= self.edge_size:
            return "topleft"
        # 左边缘
        elif abs(pos.x() - rect.left()) <= self.edge_size and rect.top() <= pos.y() <= rect.bottom():
            return "left"
        # 右边缘
        elif abs(pos.x() - rect.right()) <= self.edge_size and rect.top() <= pos.y() <= rect.bottom():
            return "right"
        # 上边缘
        elif abs(pos.y() - rect.top()) <= self.edge_size and rect.left() <= pos.x() <= rect.right():
            return "top"
        # 下边缘
        elif abs(pos.y() - rect.bottom()) <= self.edge_size and rect.left() <= pos.x() <= rect.right():
            return "bottom"
        return None

    def selection_mouse_move(self, event):
        """鼠标移动事件处理"""
        if not self.enable_mouse_events:
            return
        
        video_display_width = self.video_display.width()
        video_display_height = self.video_display.height()
        
        pos = event.pos()
        y_ratio = (pos.y() - self.border_top) / video_display_height if video_display_height > 0 else 0
        x_ratio = (pos.x() - self.border_left) / video_display_width if video_display_width > 0 else 0
        
        # 限制比例值在0-1范围内
        y_ratio = max(0, min(1, y_ratio))
        x_ratio = max(0, min(1, x_ratio))
        
        # 🖌️ VẼ NÉT CỌ TỰ DO KHI RÊ CHUỘT
        if getattr(self, 'is_drawing_stroke', False) and getattr(self, 'current_stroke', None):
            self.current_stroke['points'].append((x_ratio, y_ratio))
            self.update_preview_with_rect()
            return

        # 根据不同的操作模式处理鼠标移动
        if self.is_drawing:  # 绘制新选择框
            start_y, _, start_x, _ = self.selection_rect
            self.selection_rect = (start_y, y_ratio, start_x, x_ratio)
            self.update_preview_with_rect()
        elif self.resize_edge and self.active_selection_index >= 0:  # 调整选择框大小或位置
            ymin, ymax, xmin, xmax = self.selection_rects[self.active_selection_index]
            start_y, start_x = self.drag_start_pos
            
            if self.resize_edge == "move":
                dy = y_ratio - start_y
                dx = x_ratio - start_x
                new_ymin = max(0, min(1 - (ymax - ymin), ymin + dy))
                new_ymax = min(1, max(new_ymin + (ymax - ymin), new_ymin))
                new_xmin = max(0, min(1 - (xmax - xmin), xmin + dx))
                new_xmax = min(1, max(new_xmin + (xmax - xmin), new_xmin))
                
                self.selection_rects[self.active_selection_index] = (new_ymin, new_ymax, new_xmin, new_xmax)
                self.drag_start_pos = (y_ratio, x_ratio)
            else:
                if "left" in self.resize_edge:
                    xmin = min(xmax - 0.01, x_ratio)
                if "right" in self.resize_edge:
                    xmax = max(xmin + 0.01, x_ratio)
                if "top" in self.resize_edge:
                    ymin = min(ymax - 0.01, y_ratio)
                if "bottom" in self.resize_edge:
                    ymax = max(ymin + 0.01, y_ratio)
                
                xmin = max(0, min(xmin, 1))
                xmax = max(0, min(xmax, 1))
                ymin = max(0, min(ymin, 1))
                ymax = max(0, min(ymax, 1))
                
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                
                self.selection_rects[self.active_selection_index] = (ymin, ymax, xmin, xmax)
            
            self.update_preview_with_rect()
        else:
            self.update_cursor_shape(pos)
    
    def selection_mouse_release(self, event):
        """鼠标释放事件处理"""
        if not self.enable_mouse_events:
            return
            
        # 🖌️ KẾT THÚC VẼ NÉT CỌ TỰ DO
        if getattr(self, 'is_drawing_stroke', False) and getattr(self, 'current_stroke', None):
            self.is_drawing_stroke = False
            if len(self.current_stroke['points']) > 0:
                self.freehand_strokes.append(self.current_stroke)
            self.current_stroke = None
            self.update_preview_with_rect()
            self.selections_changed.emit(self.selection_rects)
            return

        # 结束绘制或调整
        if self.is_drawing:
            # 标准化选择框（确保ymin < ymax, xmin < xmax）
            ymin, ymax, xmin, xmax = self.selection_rect
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            
            # 更新标准化后的选区
            self.selection_rect = (ymin, ymax, xmin, xmax)
            
            # 如果选择框有效（不是点击），添加到选区列表
            # 使用比例值计算宽度和高度
            width_ratio = abs(xmax - xmin)
            height_ratio = abs(ymax - ymin)
            
            # 转换为像素大小进行判断
            pixel_width = width_ratio * self.video_display.width()
            pixel_height = height_ratio * self.video_display.height()
            
            if pixel_width > 5 and pixel_height > 5:
                self.selection_rects.append(self.selection_rect)
                self.active_selection_index = len(self.selection_rects) - 1
                
                # 发送选择框变化信号
                self.selections_changed.emit(self.selection_rects)
            
            self.is_drawing = False
            self.selection_rect = (0, 0, 0, 0)  # 重置为空选区
        elif self.resize_edge and self.active_selection_index >= 0:
            # 标准化选择框
            ymin, ymax, xmin, xmax = self.selection_rects[self.active_selection_index]
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            
            # 更新标准化后的选区
            self.selection_rects[self.active_selection_index] = (ymin, ymax, xmin, xmax)
                        
            # 发送选择框变化信号
            self.selections_changed.emit(self.selection_rects)
            
            self.resize_edge = None
        
    def update_cursor_shape(self, pos):
        """根据鼠标位置更新光标形状"""
        video_display_height = self.video_display.height()
        video_display_width = self.video_display.width()
        
        # 如果有活动选区，优先检查活动选区
        if self.active_selection_index >= 0 and self.active_selection_index < len(self.selection_rects):
            # 获取活动选区
            ymin, ymax, xmin, xmax = self.selection_rects[self.active_selection_index]
            
            # 确保坐标规范化
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            
            # 将比例坐标转换为像素坐标
            pixel_rect = QRect(
                round(xmin * video_display_width) + self.border_left,
                round(ymin * video_display_height) + self.border_top,
                round((xmax - xmin) * video_display_width),
                round((ymax - ymin) * video_display_height)
            )
            
            # 检查鼠标是否在选择框边缘
            if self.is_on_rect_edge(pos, pixel_rect):
                # 根据边缘类型设置光标
                edge_type = self.get_resize_edge(pos, pixel_rect)
                if edge_type == "left" or edge_type == "right":
                    self.video_display.setCursor(Qt.SizeHorCursor)
                    return
                elif edge_type == "top" or edge_type == "bottom":
                    self.video_display.setCursor(Qt.SizeVerCursor)
                    return
                elif edge_type == "topleft" or edge_type == "bottomright":
                    self.video_display.setCursor(Qt.SizeFDiagCursor)
                    return
                elif edge_type == "topright" or edge_type == "bottomleft":
                    self.video_display.setCursor(Qt.SizeBDiagCursor)
                    return
            elif pixel_rect.contains(pos):
                self.video_display.setCursor(Qt.SizeAllCursor)
                return
        
        # 如果没有活动选区或鼠标不在活动选区上，检查所有其他选区
        for rect in self.selection_rects:
            # 获取选区坐标
            ymin, ymax, xmin, xmax = rect
            
            # 确保坐标规范化
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            
            # 将比例坐标转换为像素坐标
            pixel_rect = QRect(
                round(xmin * video_display_width) + self.border_left,
                round(ymin * video_display_height) + self.border_top,
                round((xmax - xmin) * video_display_width),
                round((ymax - ymin) * video_display_height)
            )
            
            # 检查鼠标是否在选择框边缘
            if self.is_on_rect_edge(pos, pixel_rect):
                # 根据边缘类型设置光标
                edge_type = self.get_resize_edge(pos, pixel_rect)
                if edge_type == "left" or edge_type == "right":
                    self.video_display.setCursor(Qt.SizeHorCursor)
                    return
                elif edge_type == "top" or edge_type == "bottom":
                    self.video_display.setCursor(Qt.SizeVerCursor)
                    return
                elif edge_type == "topleft" or edge_type == "bottomright":
                    self.video_display.setCursor(Qt.SizeFDiagCursor)
                    return
                elif edge_type == "topright" or edge_type == "bottomleft":
                    self.video_display.setCursor(Qt.SizeBDiagCursor)
                    return
            # 检查鼠标是否在选择框内部
            elif pixel_rect.contains(pos):
                self.video_display.setCursor(Qt.SizeAllCursor)
                return
        
        # 如果鼠标不在任何选区上，设置为默认光标
        self.video_display.setCursor(Qt.ArrowCursor)
    
    def set_video_parameters(self, frame_width, frame_height, 
                             scaled_width=None, scaled_height=None, 
                             border_left=0, border_top=0, 
                             fps=30):
        """Cài đặt thông số video và tự động tính toán tỷ lệ khung hình (Aspect Ratio) chính xác không bị méo."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height
        self.border_left = border_left
        self.border_top = border_top
        self.fps = fps if fps and fps > 0 else 30

        if frame_width and frame_height and frame_width > 0 and frame_height > 0:
            aspect = float(frame_height) / float(frame_width)
            self.target_aspect_ratio = aspect

            # Tự động tính kích thước xem trước tỉ lệ 1:1 chuẩn xác với mọi video (16:9, 9:16, 1:1, 4:3, 21:9...)
            if aspect > 1.0:  # Video Dọc (Vertical Video: TikTok, Shorts 9:16)
                self.video_preview_height = 420
                self.video_preview_width = max(200, int(420 / aspect))
            elif aspect == 1.0:  # Video Vuông (Square Video 1:1)
                self.video_preview_height = 360
                self.video_preview_width = 360
            else:  # Video Ngang (Landscape Video 16:9, 4:3, 21:9)
                self.video_preview_width = 680
                self.video_preview_height = int(680 * aspect)

            if hasattr(self, 'ratio_container') and self.ratio_container:
                self.ratio_container.setFixedWidth(self.video_preview_width)
                self.ratio_container.setFixedHeight(self.video_preview_height)
            if hasattr(self, 'video_display') and self.video_display:
                self.video_display.setFixedSize(self.video_preview_width, self.video_preview_height)
    
    def get_selection_coordinates(self):
        """Lấy danh sách tọa độ các vùng chọn (ymin, ymax, xmin, xmax)."""
        if hasattr(self, 'selection_rects') and self.selection_rects:
            return list(self.selection_rects)
        if hasattr(self, 'selection_rect') and any(self.selection_rect):
            return [self.selection_rect]
        return []

    def set_mask_mode(self, mode):
        """Thiết lập chế độ chọn mask: 'rect' (ô vuông) hoặc 'brush' (cọ vẽ tự do)"""
        self.mask_mode = mode
        if mode == "brush":
            self.video_display.setCursor(Qt.CrossCursor)
        else:
            self.video_display.setCursor(Qt.ArrowCursor)
        self.update_preview_with_rect()

    def set_brush_size(self, size):
        """Thiết lập đường kính cọ vẽ (pixels)"""
        self.brush_size = max(2, min(200, size))

    def clear_brush_strokes(self):
        """Xóa tất cả các nét cọ vẽ"""
        self.brush_strokes = []
        self.update_preview_with_rect()

    def set_split_view(self, enabled, after_frame=None):
        """Bật/Tắt chế độ so sánh Split-View trước/sau"""
        self.split_view_enabled = enabled
        if after_frame is not None:
            self.set_after_frame(after_frame)
        self.update_preview_with_rect()

    def set_after_frame(self, frame):
        """Nạp khung hình đã inpaint/xóa để hiển thị so sánh Split-View"""
        if frame is None:
            self.after_pixmap = None
            return
        try:
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb_frame.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            self.after_pixmap = QtGui.QPixmap.fromImage(qimg)
        except Exception:
            self.after_pixmap = None
        self.update_preview_with_rect()
    
    def set_selection_rects(self, rects):
        """设置选择框"""
        self.selection_rects = rects
        self.selection_rect = rects[-1] if rects else (0, 0, 0, 0)
        self.active_selection_index = len(rects) - 1
        self.update_preview_with_rect()
        
    def add_default_selection(self):
        """Thêm một vùng chọn mặc định ở nửa dưới màn hình"""
        default_rect = (0.8, 0.9, 0.15, 0.85)
        self.selection_rects.append(default_rect)
        self.active_selection_index = len(self.selection_rects) - 1
        self.selections_changed.emit(self.selection_rects)
        self.update_preview_with_rect()
        return True
    
    def load_selections_from_config(self):
        """从配置中加载选择框的相对位置和大小"""
        # 从配置中读取选择框的相对位置和大小
        areas_str = config.subtitleSelectionAreas.value
        
        # 检查配置值是否有效
        if not areas_str:
            return False

        # 清空现有选区
        self.selection_rects = []
        
        # 解析配置字符串
        areas = areas_str.split(";")
        for area in areas:
            try:
                parts = area.split(",")
                ymin, ymax, xmin, xmax = map(float, parts)
                self.selection_rects.append((ymin, ymax, xmin, xmax))
            except ValueError:
                continue
        
        # 如果有选区，设置最后一个为活动选区
        if self.selection_rects:
            self.active_selection_index = len(self.selection_rects) - 1
        else:
            self.active_selection_index = -1
        self.selections_changed.emit(self.selection_rects)

        # 更新预览
        self.update_preview_with_rect()
        
        return len(self.selection_rects) > 0
    
    def preview_coordinates_to_video_coordinates(self, preview_selection_rects):
        """获取选择框在原始视频中的坐标"""
        if getattr(self, 'scaled_width', None) is None or not self.scaled_width or not self.scaled_height:
            return []
        
        selection_rects = []
        video_display_height = self.video_display.height()
        video_display_width = self.video_display.width()
        if not video_display_width or not video_display_height:
            return []
        for rect in preview_selection_rects:
            ymin, ymax, xmin, xmax = rect
                
            # 调整选择框坐标，考虑黑边偏移
            x_adjusted = max(0, xmin - self.border_left)
            y_adjusted = max(0, ymin - self.border_top)
            
            # 如果选择框超出了实际视频区域，需要调整宽度和高度
            w_adjusted = min((xmax - xmin), self.scaled_width - x_adjusted)
            h_adjusted = min((ymax - ymin), self.scaled_height - y_adjusted)
            # 转换为原始视频坐标
            scale_x = self.frame_width / (self.scaled_width * video_display_width)
            scale_y = self.frame_height / (self.scaled_height * video_display_height)

            # 使用round代替int，避免精度丢失
            xmin = round(x_adjusted * scale_x * video_display_width)
            xmax = round((x_adjusted + w_adjusted) * scale_x * video_display_width)
            ymin = round(y_adjusted * scale_y * video_display_height)
            ymax = round((y_adjusted + h_adjusted) * scale_y * video_display_height)
            
            # 确保坐标在有效范围内
            xmin = max(0, min(xmin, self.frame_width))
            xmax = max(0, min(xmax, self.frame_width))
            ymin = max(0, min(ymin, self.frame_height))
            ymax = max(0, min(ymax, self.frame_height))
            
            # 确保xmin < xmax, ymin < ymax
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
                
            selection_rects.append((ymin, ymax, xmin, xmax))
        return selection_rects

    def set_dragger_enabled(self, enabled):
        """设置拖动器是否可用"""
        self.enable_mouse_events = enabled
        self.video_display.setMouseTracking(enabled)
        self.video_display.setCursor(Qt.ArrowCursor)

    def save_selections_to_config(self):
        """保存所有选择框的相对位置和大小"""
        areas_str_parts = []
        
        for rect in self.selection_rects:
            ymin, ymax, xmin, xmax = rect
            # 直接使用比例值，四舍五入到4位小数
            areas_str_parts.append(f"{round(ymin,4)},{round(ymax,4)},{round(xmin,4)},{round(xmax,4)}")
        
        # 更新配置
        config.subtitleSelectionAreas.value = ";".join(areas_str_parts)
        if len(config.subtitleSelectionAreas.value) <= 0:
            config.subtitleSelectionAreas.value = config.subtitleSelectionAreas.defaultValue
        qconfig.save()
    
    def get_selection_rects(self):
        """获取所有选区"""
        return self.selection_rects
    
    def clear_selections(self):
        """清除所有选区与 cọ vẽ"""
        self.selection_rects = []
        self.brush_strokes = []
        self.active_selection_index = -1
        self.update_preview_with_rect()
        self.selections_changed.emit(self.selection_rects)

    def __handle_delete_selection(self):
        """处理删除当前选区的逻辑"""
        try:
            if self.active_selection_index >= 0 and self.selection_rects:
                # 删除当前活跃选区
                self.selection_rects.pop(self.active_selection_index)
                
                # 如果还有其他选区，将最后一个选区设为活跃选区
                if self.selection_rects:
                    self.active_selection_index = len(self.selection_rects) - 1
                else:
                    self.active_selection_index = -1
                
                # 更新显示
                self.update_preview_with_rect()
                
                # 发送选区变化信号
                self.selections_changed.emit(self.selection_rects)
                return True
            return False
        finally:
            # 获取当前鼠标位置
            global_pos = QCursor.pos()
            pos = self.video_display.mapFromGlobal(global_pos)
            self.update_cursor_shape(pos)

    def __handle_mark_for_ab_start(self):
        """处理标记AB分区起点的逻辑"""
        current_frame = self.video_slider.value()
        if current_frame >= 0:
            # 检查是否需要调整已有区间
            adjusted = False
            for i, section_range in enumerate(self.ab_sections):
                if current_frame in section_range:
                    # 调整已有区间的起点
                    self.ab_sections[i] = range(current_frame, section_range.stop)
                    adjusted = True
                    break
            
            if not adjusted:
                # 记录新的AB分区起点
                self.current_ab_start = current_frame
            
            # 更新显示
            self.update_preview_with_rect()
            return True
        return False

    def __handle_mark_for_ab_end(self):
        """处理标记AB分区终点的逻辑"""
        current_frame = self.video_slider.value()
        if current_frame >= 0 and self.current_ab_start >= 0:
            # 检查是否需要调整已有区间
            adjusted = False
            for i, section_range in enumerate(self.ab_sections):
                if current_frame in section_range:
                    # 调整已有区间的终点
                    self.ab_sections[i] = range(section_range.start, current_frame + 1)
                    adjusted = True
                    break
            
            if not adjusted and self.current_ab_start != current_frame:
                # 添加新的AB分区
                self.ab_sections.append(range(self.current_ab_start, current_frame + 1))
                self.current_ab_start = -1  # 重置起点
                self.ab_sections_changed.emit(self.ab_sections)
            
            # 更新显示
            self.update_preview_with_rect()
            return True
        return False

    def __handle_delete_ab_section(self):
        """处理删除当前AB区块的逻辑"""
        current_frame = self.video_slider.value()
        if current_frame >= 0 and self.ab_sections:
            # 查找当前帧所在的AB区块
            for i, section_range in enumerate(self.ab_sections):
                if current_frame in section_range:
                    # 删除该AB区块
                    self.ab_sections.pop(i)
                    
                    # 如果当前有标记的起点，且在被删除的区块内，重置起点
                    if self.current_ab_start in section_range:
                        self.current_ab_start = -1
                    
                    # 发送AB区块变化信号
                    self.ab_sections_changed.emit(self.ab_sections)
                    
                    # 更新显示
                    self.update_preview_with_rect()
                    return True
        return False
    
    def __adjust_slider_value(self, delta):
        """调整视频滑块的值"""
        current_value = self.video_slider.value()
        max_value = self.video_slider.maximum()
        new_value = current_value + int(delta)
        
        # 确保新值在有效范围内
        if new_value < self.video_slider.minimum():
            new_value = self.video_slider.minimum()
        elif new_value > max_value:
            new_value = max_value
            
        # 设置新值
        self.video_slider.setValue(new_value)

    def eventFilter(self, obj, event):
        """事件过滤器，处理键盘事件"""
        if event.type() == QEvent.KeyPress:
            # 处理退格键和删除键
            if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
                if self.__handle_delete_selection():
                    return True
        # 对于其他事件，继续传递给父类处理
        return super().eventFilter(obj, event)

    def __init_context_menu(self):
        """初始化右键菜单"""
        self.context_menu = QMenu(self)
        
        # 设定区块起点动作
        self.action_mark_ab_start = QAction(tr['SubtitleExtractorGUI']['MarkABStart'], self)
        self.action_mark_ab_start.setShortcut("[")
        self.action_mark_ab_start.triggered.connect(self.__handle_mark_for_ab_start)
        self.context_menu.addAction(self.action_mark_ab_start)
        
        # 设定区块终点动作
        self.action_mark_ab_end = QAction(tr['SubtitleExtractorGUI']['MarkABEnd'], self)
        self.action_mark_ab_end.setShortcut("]")
        self.action_mark_ab_end.triggered.connect(self.__handle_mark_for_ab_end)
        self.context_menu.addAction(self.action_mark_ab_end)

        self.action_mark_ab_delete = QAction(tr['SubtitleExtractorGUI']['DeleteABSection'], self)
        self.action_mark_ab_delete.setShortcut("\\")
        self.action_mark_ab_delete.triggered.connect(self.__handle_delete_ab_section)
        self.context_menu.addAction(self.action_mark_ab_delete)

        self.action_delete_selection = QAction(tr['SubtitleExtractorGUI']['DeleteSelection'], self)
        self.action_delete_selection.setShortcut("DELETE")
        self.action_delete_selection.triggered.connect(self.__handle_delete_selection)
        self.context_menu.addAction(self.action_delete_selection)

        self.action_clear_all_selections = QAction(tr['SubtitleExtractorGUI'].get('ClearAllSelections', 'Clear All Selections'), self)
        self.action_clear_all_selections.setShortcut("Ctrl+Delete")
        self.action_clear_all_selections.triggered.connect(self.clear_selections)
        self.context_menu.addAction(self.action_clear_all_selections)

    def get_ab_sections(self):
        """获取AB分区标记"""
        return self.ab_sections

    def set_ab_sections(self, sections):
        """设置AB分区标记"""
        self.ab_sections = sections
        self.update_preview_with_rect()

    def clear_ab_sections(self):
        """清除所有AB分区标记"""
        self.ab_sections = []
        self.current_ab_start = -1
        self.update_preview_with_rect()

    def retranslateUi(self):
        """Cập nhật ngôn ngữ của context menu khi đổi ngôn ngữ nóng"""
        self.action_mark_ab_start.setText(tr['SubtitleExtractorGUI']['MarkABStart'])
        self.action_mark_ab_end.setText(tr['SubtitleExtractorGUI']['MarkABEnd'])
        self.action_mark_ab_delete.setText(tr['SubtitleExtractorGUI']['DeleteABSection'])
        self.action_delete_selection.setText(tr['SubtitleExtractorGUI']['DeleteSelection'])
        self.action_clear_all_selections.setText(tr['SubtitleExtractorGUI'].get('ClearAllSelections', 'Clear All Selections'))

    def closeEvent(self, event):
        """窗口关闭时断开信号连接"""
        try:
            # 断开信号连接
            self.shortcut_ab_start.activated.disconnect(self.__handle_mark_for_ab_start)
            self.shortcut_ab_end.activated.disconnect(self.__handle_mark_for_ab_end)
            self.shortcut_ab_delete.activated.disconnect(self.__handle_delete_ab_section)
            self.action_mark_ab_start.triggered.disconnect(self.__handle_mark_for_ab_start)
            self.action_mark_ab_end.triggered.disconnect(self.__handle_mark_for_ab_end)
            self.action_mark_ab_delete.triggered.disconnect(self.__handle_delete_ab_section)
            self.action_delete_selection.triggered.disconnect(self.__handle_delete_selection)
            self.shortcut_delete_selection.activated.disconnect(self.__handle_delete_selection)
            self.action_clear_all_selections.triggered.disconnect(self.clear_selections)
            self.shortcut_clear_all_selections.activated.disconnect(self.clear_selections)
        except Exception as e:
            print(f"Error during close window:", e)
        super().closeEvent(event)