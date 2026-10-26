import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import unittest
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QUrl, QMimeData, QPointF
from PySide6.QtGui import QDragEnterEvent, QDropEvent

# Đảm bảo QApplication tồn tại
app = QApplication.instance() or QApplication([])


def _create_dummy_video(path: str, fps: float = 10.0, num_frames: int = 5, width: int = 160, height: int = 120):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _create_dummy_file(path, content=""):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _create_drop_event(file_paths):
    mime_data = QMimeData()
    urls = [QUrl.fromLocalFile(p) for p in file_paths]
    mime_data.setUrls(urls)
    enter_ev = QDragEnterEvent(
        QPointF(10.0, 10.0).toPoint(),
        Qt.CopyAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    drop_ev = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.CopyAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    enter_ev._mime_data = mime_data
    drop_ev._mime_data = mime_data
    return enter_ev, drop_ev


class TestUIDragDropAndFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setting_interface_instantiation_without_parent(self):
        """Xác minh SettingInterface có thể khởi tạo an toàn với parent=None."""
        from src.desktop.ui.setting_interface import SettingInterface
        setting_layout = SettingInterface(parent=None)
        self.assertIsNotNone(setting_layout)

    def test_main_window_language_file_path(self):
        """Xác minh translation_file trong main.py trỏ đúng vào thư mục src/core/interface/."""
        from src.desktop.main import MainWindow
        import src.core.config as cfg

        main_win = MainWindow()
        core_interface_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "core",
            "interface",
        )
        self.assertTrue(os.path.isdir(core_interface_dir))
        vi_ini = os.path.join(core_interface_dir, "vi.ini")
        self.assertTrue(os.path.isfile(vi_ini))

        # Kiểm tra gọi change_language không gặp exception
        main_win.change_language("vi")

    def test_home_interface_drag_and_drop_video(self):
        """Xác minh kéo thả video vào HomeInterface sẽ tự động nạp video và thêm vào danh sách tác vụ."""
        from src.desktop.ui.home_interface import HomeInterface

        home = HomeInterface()
        dummy_vid = os.path.join(self.temp_dir, "drag_test_vid.mp4")
        _create_dummy_video(dummy_vid, num_frames=5)

        enter_ev, drop_ev = _create_drop_event([dummy_vid])
        home.dragEnterEvent(enter_ev)
        self.assertTrue(enter_ev.isAccepted(), "dragEnterEvent phải accept file video hợp lệ")

        home.dropEvent(drop_ev)
        self.assertTrue(drop_ev.isAccepted(), "dropEvent phải accept file video hợp lệ")
        all_tasks = home.task_list_component.get_all_tasks()
        self.assertTrue(any(t.path == dummy_vid for t in all_tasks), "Tệp video phải xuất hiện trong danh sách tác vụ")

    def test_task_list_component_drag_and_drop_video(self):
        """Xác minh kéo thả video vào TaskListComponent sẽ thêm tác vụ vào bảng."""
        from src.desktop.ui.component.task_list_component import TaskListComponent

        task_list = TaskListComponent()
        dummy_vid = os.path.join(self.temp_dir, "tasklist_drag.mp4")
        _create_dummy_file(dummy_vid, "dummy")

        enter_ev, drop_ev = _create_drop_event([dummy_vid])
        task_list.dragEnterEvent(enter_ev)
        self.assertTrue(enter_ev.isAccepted())

        task_list.dropEvent(drop_ev)
        self.assertTrue(drop_ev.isAccepted())
        self.assertTrue(any(t.path == dummy_vid for t in task_list.get_all_tasks()))

    def test_extractor_interface_drag_and_drop_video(self):
        """Xác minh kéo thả video vào ExtractorInterface sẽ nạp video."""
        from src.desktop.ui.extractor_interface import ExtractorInterface

        extractor = ExtractorInterface()
        dummy_vid = os.path.join(self.temp_dir, "extractor_drag.mp4")
        _create_dummy_file(dummy_vid, "dummy")

        enter_ev, drop_ev = _create_drop_event([dummy_vid])
        extractor.dragEnterEvent(enter_ev)
        self.assertTrue(enter_ev.isAccepted())

        extractor.dropEvent(drop_ev)
        self.assertTrue(drop_ev.isAccepted())
        self.assertEqual(extractor.video_path, dummy_vid)

    def test_translation_interface_drag_and_drop_subtitle(self):
        """Xác minh kéo thả tệp phụ đề vào TranslationInterface sẽ nạp phụ đề."""
        from src.desktop.ui.translation_interface import TranslationInterface

        translator = TranslationInterface()
        dummy_srt = os.path.join(self.temp_dir, "drag_sub.srt")
        _create_dummy_file(dummy_srt, "1\n00:00:01,000 --> 00:00:03,000\nXin chào\n\n")

        enter_ev, drop_ev = _create_drop_event([dummy_srt])
        translator.dragEnterEvent(enter_ev)
        self.assertTrue(enter_ev.isAccepted())

        translator.dropEvent(drop_ev)
        self.assertTrue(drop_ev.isAccepted())
        self.assertEqual(len(translator.subtitle_blocks), 1)

    def test_auto_pipeline_drag_and_drop_video(self):
        """Xác minh kéo thả video vào AutoPipelineInterface sẽ nạp vào hàng đợi."""
        from src.desktop.ui.auto_pipeline_interface import AutoPipelineInterface

        pipeline = AutoPipelineInterface()
        dummy_vid = os.path.join(self.temp_dir, "pipeline_drag.mp4")
        _create_dummy_file(dummy_vid, "dummy")

        enter_ev, drop_ev = _create_drop_event([dummy_vid])
        pipeline.dragEnterEvent(enter_ev)
        self.assertTrue(enter_ev.isAccepted())

        pipeline.dropEvent(drop_ev)
        self.assertTrue(drop_ev.isAccepted())
        self.assertIn(dummy_vid, pipeline.video_queue)

    def test_omnivoice_drag_and_drop_subtitle_and_video(self):
        """Xác minh kéo thả tệp phụ đề và video vào OmniVoiceInterface sẽ nạp dữ liệu."""
        from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

        omni = OmniVoiceInterface()
        dummy_srt = os.path.join(self.temp_dir, "omni_drag.srt")
        _create_dummy_file(dummy_srt, "1\n00:00:01,000 --> 00:00:02,000\nXin chào\n\n")
        dummy_vid = os.path.join(self.temp_dir, "omni_drag.mp4")
        _create_dummy_file(dummy_vid, "dummy")

        enter_ev, drop_ev = _create_drop_event([dummy_srt, dummy_vid])
        omni.dragEnterEvent(enter_ev)
        self.assertTrue(enter_ev.isAccepted())

        omni.dropEvent(drop_ev)
        self.assertTrue(drop_ev.isAccepted())
        self.assertEqual(omni.current_subtitle_path, dummy_srt)
        self.assertEqual(omni.current_video_path, dummy_vid)


if __name__ == "__main__":
    unittest.main()
