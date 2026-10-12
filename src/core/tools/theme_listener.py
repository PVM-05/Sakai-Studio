# -*- coding: utf-8 -*-
import threading
from PySide6 import QtCore
from qfluentwidgets import setTheme, qconfig, Theme
import darkdetect


class SystemThemeListener(QtCore.QObject):
    """Bộ lắng nghe thay đổi giao diện sáng/tối của hệ điều hành an toàn luồng."""

    systemThemeChanged = QtCore.Signal()
    _themeDetected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._thread: threading.Thread | None = None
        self._themeDetected.connect(self._applyThemeOnMainThread)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._listenLoop,
            name="SystemThemeListenerDaemon",
            daemon=True,
        )
        self._thread.start()

    def _listenLoop(self):
        try:
            darkdetect.listener(self._onThemeChanged)
        except Exception:
            pass

    def _onThemeChanged(self, theme: str):
        if theme:
            self._themeDetected.emit(theme)

    @QtCore.Slot(str)
    def _applyThemeOnMainThread(self, theme_str: str):
        theme = Theme.DARK if theme_str.lower() == "dark" else Theme.LIGHT
        setTheme(theme)
        if qconfig.themeMode.value != Theme.AUTO or theme == qconfig.theme:
            return

        qconfig.theme = Theme.AUTO
        qconfig._cfg.themeChanged.emit(Theme.AUTO)
        self.systemThemeChanged.emit()