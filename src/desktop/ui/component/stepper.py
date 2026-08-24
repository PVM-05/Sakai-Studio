# -*- coding: utf-8 -*-
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from qfluentwidgets import BodyLabel, IconWidget, FluentIcon

class StepperWidget(QWidget):
    """
    Component Stepper hiển thị các bước tuần tự một cách trực quan.
    """
    def __init__(self, steps: list[str], parent=None):
        super().__init__(parent=parent)
        self.steps = steps
        self.current_step = 0
        
        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(12)
        
        self.step_items = []
        for i, step_text in enumerate(self.steps, 1):
            item_widget = QWidget(self)
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(12)
            
            icon_widget = IconWidget(FluentIcon.PAGE_RIGHT, item_widget)
            icon_widget.setFixedSize(18, 18)
            
            label = BodyLabel(f"Bước {i}: {step_text}", item_widget)
            
            item_layout.addWidget(icon_widget)
            item_layout.addWidget(label)
            item_layout.addStretch(1)
            
            self.vBoxLayout.addWidget(item_widget)
            self.step_items.append({'icon': icon_widget, 'label': label})
            
        self.set_current_step(1) # Khởi tạo mặc định
        
    def set_current_step(self, step_index: int, status: str = "running"):
        """
        Cập nhật trạng thái hiển thị của các bước.
        - Các bước trước step_index sẽ có màu xanh lá (hoàn thành).
        - Bước hiện tại (step_index) sẽ có màu xanh dương nếu running, hoặc đỏ nếu error.
        - Các bước sau sẽ bị làm mờ.
        (Lưu ý: step_index tính từ 1)
        """
        self.current_step = step_index
        
        for i, item in enumerate(self.step_items, 1):
            icon_w: IconWidget = item['icon']
            lbl: BodyLabel = item['label']
            
            if i < step_index:
                # Đã hoàn thành
                icon_w.setIcon(FluentIcon.COMPLETED)
                # Dùng QColor thay vì theme color string để chắc chắn
                lbl.setStyleSheet("color: #107c41; font-weight: normal;")
            elif i == step_index:
                # Đang xử lý
                if status == "error":
                    icon_w.setIcon(FluentIcon.CANCEL)
                    lbl.setStyleSheet("color: #d13438; font-weight: bold;")
                else:
                    icon_w.setIcon(FluentIcon.SYNC)
                    lbl.setStyleSheet("color: #0078d4; font-weight: bold;")
            else:
                # Chờ xử lý
                icon_w.setIcon(FluentIcon.INFO)
                lbl.setStyleSheet("color: #666666; font-weight: normal;")

    def set_error_step(self, step_index: int):
        self.set_current_step(step_index, "error")
