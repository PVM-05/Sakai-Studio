"""
Cấu hình kiểm thử tự động pytest toàn cục cho Sakai Studio.
Đảm bảo môi trường kiểm thử không bao giờ bị chặn bởi các hộp thoại đồ họa hệ điều hành.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def prevent_blocking_dialogs(monkeypatch):
    """
    Tự động vô hiệu hóa các hộp thoại mở/lưu tệp tin và thông báo xác nhận đồ họa
    để ngăn chặn triệt để tình trạng kiểm thử bị treo (hang) trên môi trường CI và máy trạm.
    """
    try:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        # Trả về đường dẫn giả lập hợp lệ khi hàm lưu/mở tệp được gọi
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("/tmp/mock_output.wav", "WAV"))
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("/tmp/mock_input.srt", "SRT"))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "/tmp/mock_dir")
        monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
        monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    except Exception:
        pass
