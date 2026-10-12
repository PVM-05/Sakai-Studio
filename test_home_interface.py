import sys
import os
import time
import tempfile
import cv2
import numpy as np

# Thêm đường dẫn gốc dự án vào sys.path một cách động
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.desktop.ui.home_interface import HomeInterface
from src.desktop.ui.component.task_list_component import TaskItem, TaskOptions


def create_dummy_video(path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    for _ in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, 'TEST SUBTITLE', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def test_home_interface_workflow(tmp_path=None):
    """Kiểm thử tích hợp giao diện HomeInterface, TaskItem và vùng chọn trên môi trường sạch."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    home = HomeInterface()
    if tmp_path is None:
        temp_dir = tempfile.mkdtemp(prefix="sakai_home_test_")
        dummy_video_path = os.path.join(temp_dir, "test_video.mp4")
    else:
        dummy_video_path = str(tmp_path / "test_video.mp4")

    try:
        create_dummy_video(dummy_video_path)
        abs_video_path = os.path.abspath(dummy_video_path)

        # 1. Kiểm thử TaskItem và thêm nhiệm vụ vào danh sách
        home.task_list_component.tasks.clear()
        home.task_list_component.refresh_table()
        ok = home.task_list_component.add_tasks([abs_video_path])
        assert ok is True
        task_item = home.task_list_component.get_task(0)
        assert isinstance(task_item, TaskItem)
        QApplication.processEvents()

        home.task_list_component.select_task(0)
        QApplication.processEvents()
        assert home.video_path == abs_video_path

        # 2. Kiểm thử Thêm Vùng Chọn và chuyển đổi tọa độ không nhân đôi
        home.add_area_button_clicked()
        QApplication.processEvents()
        selections = home.video_display_component.selection_rects
        assert len(selections) > 0

        video_coords = home.video_display_component.preview_coordinates_to_video_coordinates(selections)
        video_coords_again = home.video_display_component.preview_coordinates_to_video_coordinates(video_coords)
        assert video_coords == video_coords_again

        # 3. Kiểm thử Xem Trước Mask kết hợp nét cọ và hộp chọn
        home._preview_original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        home.video_display_component.freehand_strokes.append({
            "points": [(0.2, 0.2), (0.4, 0.4)],
            "size": 0.05,
            "is_eraser": False,
        })
        home.selections_changed(selections)
        task_opts = task_item.options
        assert TaskOptions.SUB_AREAS in task_opts
        assert TaskOptions.FREEHAND_STROKES in task_opts
        assert len(task_opts[TaskOptions.FREEHAND_STROKES]) == 1
    finally:
        home.close()
        if os.path.exists(dummy_video_path):
            try:
                os.remove(dummy_video_path)
            except Exception:
                pass


def run_tests():
    test_home_interface_workflow()
    print("Hoàn tất kiểm thử HomeInterface thành công.")


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    run_tests()
