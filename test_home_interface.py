import sys
import os
import time
import cv2
import numpy as np

# Thêm đường dẫn gốc dự án vào sys.path
sys.path.insert(0, r"e:\Sakai-Studio")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QCoreApplication
from src.desktop.ui.home_interface import HomeInterface
from src.desktop.ui.component.task_list_component import TaskItem

def create_dummy_video(path):
    print("Tạo video giả lập...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    for i in range(60):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Vẽ một đoạn text
        cv2.putText(frame, 'TEST SUBTITLE', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print("Đã tạo video:", path)

def run_tests():
    print("Bắt đầu kiểm thử...")
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    home = HomeInterface()
    home.show()
    
    dummy_video_path = "test_video.mp4"
    create_dummy_video(dummy_video_path)
    abs_video_path = os.path.abspath(dummy_video_path)
    
    # Thêm task vào danh sách
    print("1. Kiểm thử Mở Video")
    home.task_list_component.add_tasks([abs_video_path])
    QApplication.processEvents()
    time.sleep(1)
    home.task_list_component.select_task(0)
    QApplication.processEvents()
    time.sleep(1)
    
    if home.video_path != abs_video_path:
        print("LỖI: Mở video thất bại!")
    else:
        print("-> Mở video thành công.")
        
    print("2. Kiểm thử Thêm Vùng Chọn")
    home.add_area_button_clicked()
    QApplication.processEvents()
    time.sleep(0.5)
    selections = home.video_display_component.selection_rects
    if len(selections) > 0:
        print("-> Thêm vùng chọn thành công.")
    else:
        print("LỖI: Thêm vùng chọn thất bại.")
        
    print("3. Kiểm thử Tự động Nhận Diện Chữ OCR")
    # Gắn callback để check
    ocr_done = False
    def on_ocr_done(rects):
        nonlocal ocr_done
        ocr_done = True
        print(f"-> OCR trả về {len(rects) if rects else 0} vùng.")
    home.auto_detect_result_signal.connect(on_ocr_done)
    home._on_auto_detect_clicked()
    
    # Đợi OCR chạy (background thread)
    for _ in range(50):
        QApplication.processEvents()
        time.sleep(0.1)
        if ocr_done:
            break
            
    print("4. Kiểm thử Nhận Diện Chuyển Động")
    tracking_done = False
    def on_track_done(res):
        nonlocal tracking_done
        tracking_done = True
        print(f"-> Tracking trả về {len(res) if res else 0} frames.")
    home.track_motion_finished_signal.connect(on_track_done)
    home._on_track_motion_clicked()
    for _ in range(50):
        QApplication.processEvents()
        time.sleep(0.1)
        if tracking_done:
            break
            
    print("5. Kiểm thử Xem Trước Mask")
    mask_preview_done = False
    def on_mask_preview(frame, err):
        nonlocal mask_preview_done
        mask_preview_done = True
        print(f"-> Xem trước mask {'thành công' if frame is not None else 'thất bại'}. Lỗi: {err}")
    home.mask_preview_result_signal.connect(on_mask_preview)
    # Cần set _preview_original_frame để mask preview không báo lỗi null
    home._preview_original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    home.mask_preview_button_clicked()
    for _ in range(50):
        QApplication.processEvents()
        time.sleep(0.1)
        if mask_preview_done:
            break
            
    print("6. Kiểm thử Bắt đầu/Dừng (Chỉ start rồi stop liền)")
    home.run_button_clicked()
    QApplication.processEvents()
    time.sleep(2)
    home.stop_button_clicked()
    QApplication.processEvents()
    time.sleep(1)
    print("-> Đã gửi lệnh Dừng")

    print("Hoàn tất kiểm thử.")
    if os.path.exists(abs_video_path):
        try:
            os.remove(abs_video_path)
        except Exception:
            pass
    QTimer.singleShot(500, app.quit)
    app.exec()

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    run_tests()
