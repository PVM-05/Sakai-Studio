# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao（原作者） / 改写：Jason Eric
@Time    : 2023/4/1 6:07 下午（原始时间）
@FileName: gui.py
@desc: 字幕去除器图形化界面（由 PySimpleGUI 改写为 PySide6）
"""
import sys
import os

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='.*onnxruntime.*')
warnings.filterwarnings('ignore', message='.*Unsupported Windows version.*')
import configparser
import cv2
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;auto'
import multiprocessing
from PySide6.QtCore import Qt, QTranslator
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QApplication, QFrame, QStackedWidget, QHBoxLayout, QLabel
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, setThemeColor, OptionsConfigItem,
                          OptionsValidator, SubtitleLabel, HollowHandleStyle, qconfig, ConfigItem, QConfig,
                          NavigationWidget, NavigationItemPosition, isDarkTheme, InfoBar)

from qframelesswindow.utils import getSystemAccentColor
from src.core.config import config, tr, VERSION
from src.core.tools.theme_listener import SystemThemeListener
from src.core.tools.process_manager import ProcessManager
from src.desktop.ui.auto_pipeline_interface import AutoPipelineInterface
from src.desktop.ui.advanced_setting_interface import AdvancedSettingInterface
from src.desktop.ui.tools_interface import ToolsInterface
from src.desktop.ui.home_interface import HomeInterface
from src.desktop.ui.extractor_interface import ExtractorInterface
from src.desktop.ui.ytdlp_interface import YtdlpInterface
from src.desktop.ui.mmo_setting_interface import MMOSettingInterface
from src.desktop.ui.translation_interface import TranslationInterface


class SubtitleExtractorGUI(FluentWindow): 
    def __init__(self):
        super().__init__()
        # Bật hiệu ứng Mica (nếu Windows hỗ trợ) để tăng độ thẩm mỹ theo taste-skill
        self.setMicaEffectEnabled(True)
        # Thiết lập màu sắc chủ đạo rực rỡ hiện đại (Vibrant Blue/Indigo)
        setTheme(Theme.AUTO)
        setThemeColor('#6366F1', save=True)

        # Khởi tạo hệ thống theme listener để đồng bộ nếu cần
        self.themeListener = SystemThemeListener(self)
        self.themeListener.start()
 
        # 设置窗口图标
        self.setWindowIcon(QtGui.QIcon("design/vsr.ico"))
        self.setWindowTitle("Sakai Studio v" + VERSION)
        
        # Đặt kích thước tối thiểu an toàn hợp lệ
        self.setMinimumSize(960, 600)
        
        # Đảm bảo TitleBar hiển thị chuẩn, nằm trên cùng và di chuyển được cửa sổ
        if hasattr(self, 'titleBar'):
            self.titleBar.raise_()
            self.titleBar.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

        # 创建界面布局
        self._create_layout()
        self._connectSignalToSlot()
        self._lazy_check_update()
        
        # Tự động mở toàn màn hình
        self.showMaximized()

    def _lazy_check_update(self):
        """ 延迟检查更新 """
        if not config.checkUpdateOnStartup.value:
            return
        self.check_update_timer = QtCore.QTimer(self)
        self.check_update_timer.setSingleShot(True)
        self.check_update_timer.timeout.connect(lambda: self.advancedSettingInterface.check_update(ignore=True))
        self.check_update_timer.start(2000)

    def _connectSignalToSlot(self):
        config.appRestartSig.connect(self._showRestartTooltip)
        config.interface.valueChanged.connect(self.change_language)

    def change_language(self, language):
        """Thay đổi ngôn ngữ nóng tức thì không cần khởi động lại ứng dụng"""
        try:
            # 1. Nạp lại file dịch thuật tương ứng từ backend/interface/
            translation_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface', f"{language}.ini")
            if os.path.exists(translation_file):
                tr.read(translation_file, encoding='utf-8')
            
            # 2. Cập nhật tiêu đề cửa sổ chính
            self.setWindowTitle("Sakai Studio v" + VERSION)
            
            # 3. Cập nhật tiêu đề các tab phụ trong Navigation Bar
            try:
                self.navigationInterface.widget(self.advancedSettingInterface.objectName()).setText(tr['Setting']['AdvancedSetting'])
            except Exception as e:
                print("Lỗi đổi nhãn Navigation:", e)
            
            # 4. Gọi cập nhật giao diện nóng cho các component con
            if hasattr(self.homeInterface, 'retranslateUi'):
                self.homeInterface.retranslateUi()
            if hasattr(self.ytdlpInterface, 'retranslateUi'):
                self.ytdlpInterface.retranslateUi()
            self.advancedSettingInterface.retranslateUi()
            
            # 5. Hiển thị thông báo góc màn hình đổi thành công
            InfoBar.success(
                title=tr['SubtitleExtractorGUI']['Title'],
                content="Thay đổi ngôn ngữ thành công!",
                duration=3000,
                parent=self
            )
        except Exception as e:
            print(f"Lỗi khi chuyển đổi ngôn ngữ nóng: {str(e)}")

    def _showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            'Cập nhật thành công',
            'Cấu hình sẽ có hiệu lực sau khi khởi động lại',
            duration=5000,
            parent=self
        )

    def _create_layout(self):
        # 1. Tab Trang Chủ - Luồng Tự Động Tất-Cả-Trong-Một (MMO Auto-Pipeline)
        self.autoPipelineInterface = AutoPipelineInterface(self)
        self.autoPipelineInterface.setObjectName("AutoPipelineInterface")
        
        # 2. Tab Công Cụ Chuyên Sâu (Hub gộp 4 sub-tools: Tải video, Trích xuất, Dịch sub, Xóa sub)
        self.toolsInterface = ToolsInterface(self)
        self.toolsInterface.setObjectName("ToolsInterface")
        
        # Shortcuts / Properties tương thích ngược
        self.homeInterface = self.toolsInterface.homeInterface
        self.extractorInterface = self.toolsInterface.extractorInterface
        self.translationInterface = self.toolsInterface.translationInterface
        self.ytdlpInterface = self.toolsInterface.ytdlpInterface

        # 3. Tab Cấu Hình Nâng Cao (Hệ thống)
        self.advancedSettingInterface = AdvancedSettingInterface(self)
        self.advancedSettingInterface.setObjectName("AdvancedSettingInterface")
        
        # 4. Tab Cấu Hình MMO
        self.mmoSettingInterface = MMOSettingInterface(self)
        self.mmoSettingInterface.setObjectName("MMOSettingInterface")
        
        # Bổ sung 4 tab chính vào Navigation Bar
        self.addSubInterface(self.autoPipelineInterface, FluentIcon.HOME, "Tự Động")
        self.addSubInterface(self.toolsInterface, FluentIcon.DEVELOPER_TOOLS, "Công Cụ")
        self.addSubInterface(self.mmoSettingInterface, FluentIcon.MARKET, "Cấu Hình MMO")
        self.addSubInterface(self.advancedSettingInterface, FluentIcon.SETTING, tr['Setting']['AdvancedSetting'], NavigationItemPosition.BOTTOM)

    def open_video_in_remover(self, filepath):
        """Chuyen sang Tab Cong Cu -> Sub-tab Xoa Sub va mo video da tai"""
        self.switchTo(self.toolsInterface)
        self.toolsInterface.open_video_in_remover(filepath)

    def closeEvent(self, event):
        """程序关闭时保存窗口位置并清理资源"""
        self.save_window_position()
        ProcessManager.instance().terminate_all()
        super().closeEvent(event)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

    def save_window_position(self):
        """Lưu vị trí cửa sổ an toàn (tránh tọa độ âm khi Maximize)"""
        try:
            if self.isMaximized():
                rect = self.normalGeometry()
            else:
                rect = self.geometry()
            config.set(config.windowX, max(0, rect.x()))
            config.set(config.windowY, max(0, rect.y()))
            config.set(config.windowW, max(800, rect.width()))
            config.set(config.windowH, max(600, rect.height()))
        except Exception:
            pass

    def load_window_position(self):
        """Luôn luôn mở rộng toàn màn hình (Full Screen / Maximized) mỗi khi mở ứng dụng"""
        try:
            self.showMaximized()
        except Exception as e:
            print(f"Lỗi mở toàn màn hình: {e}")
            self.showMaximized()

    def center_window(self):
        """Đặt cửa sổ nằm đúng chính giữa màn hình làm việc"""
        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w = min(1280, screen_rect.width() - 40)
        h = min(800, screen_rect.height() - 40)
        self.resize(w, h)
        x = screen_rect.left() + (screen_rect.width() - w) // 2
        y = screen_rect.top() + (screen_rect.height() - h) // 2
        self.move(x, y)

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 检测Ctrl+C组合键
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            print("\n程序被用户中断(Ctrl+C)，正在退出...")
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    
    # Đặt font hệ thống mặc định hợp lệ để tiêu diệt triệt để lỗi QFont::setPointSize: Point size <= 0 (-1)
    default_font = QtGui.QFont("Segoe UI", 9)
    default_font.setPointSize(9)
    default_font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(default_font)

    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    window = SubtitleExtractorGUI()
    # 先设置透明, 再显示, 否则会有闪烁的效果
    window.setWindowOpacity(0.0)
    window.showMaximized()
    # 使用动画效果逐渐显示窗口
    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)  # 300毫秒的动画
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()
    app.exec()