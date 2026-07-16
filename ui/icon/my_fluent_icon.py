import os
import sys
from enum import Enum

from qfluentwidgets import getIconColor, Theme, FluentIconBase


class MyFluentIcon(FluentIconBase, Enum):
    Stop = "stop"

    def path(self, theme=Theme.AUTO):
        relative_path = f'ui/icon/{self.value}_{getIconColor(theme)}.svg'
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path).replace('\\', '/')
        # Fallback to project root directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, relative_path).replace('\\', '/')
