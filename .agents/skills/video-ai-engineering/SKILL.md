---
name: video-ai-engineering
description: Kích hoạt khi làm việc với các mô hình AI (LaMa, STTN, ProPainter, SAM2), xử lý OCR, FFmpeg, CUDA và kiến trúc video pipeline của Sakai Studio.
---

# Video AI Engineering (Sakai Studio)

Kỹ năng này cung cấp hướng dẫn và ngữ cảnh chuyên sâu để làm việc với pipeline xử lý video, quản lý luồng dữ liệu đồ hoạ và các mô hình AI trong Sakai Studio.

## 1. Kiến trúc Pipeline Xử Lý Video (Architecture)
- **Luồng dữ liệu chính:** `Video Đầu Vào` -> `Trích xuất Khung Hình/Audio (FFmpeg/OpenCV)` -> `Nhận diện (OCR/Tracking)` -> `Tạo Mask` -> `Phục hồi nền (Inpaint)` -> `Ghép Audio (FFmpeg)` -> `Video Đầu Ra`.
- **Thành phần chính:**
  - `backend/main.py`: Quản lý luồng thực thi chính của quá trình inpaint (`video_inpaint`, `sttn_auto_mode`, `propainter_mode`).
  - `backend/tools/object_tracker.py`: Quản lý bám đuổi chuyển động (CSRT Hybrid, SAM 2).
  - `backend/tools/subtitle_detect.py`: Quản lý nhận diện văn bản (OCR).
  - `backend/inpaint/`: Thư mục chứa các adapter cho mô hình phục hồi nền (LaMa, STTN, ProPainter, OpenCV).

## 2. Đặc Tính Các Mô Hình AI
### A. Inpainting (Phục hồi nền)
- **LaMa (Large Mask Inpainting):**
  - *Đặc điểm:* Tốc độ siêu nhanh, phù hợp cho xử lý theo từng frame (per-frame inpainting). Rất tốt khi vùng mask liên tục thay đổi hình dáng (như logo xoay, phóng to).
  - *Sử dụng:* Được ưu tiên tự động kích hoạt trong tính năng "Xóa logo di chuyển".
- **STTN (Spatio-Temporal Transformer Network):**
  - *Đặc điểm:* Xử lý theo không gian và thời gian (dựa trên chuỗi frame lân cận). Giữ được sự liền mạch của background khi video chuyển động.
  - *Phân loại:* `STTN_AUTO` (áp dụng mask theo vùng chữ OCR phát hiện) và `STTN_DET` (áp dụng mask tĩnh cho toàn bộ hộp khoanh).
  - *Giới hạn:* Phải cắt video thành các chunk (clip_gap) để tránh tràn bộ nhớ.
- **ProPainter:**
  - *Đặc điểm:* Chất lượng hoàn thiện cao nhất hiện tại, dùng Optical Flow (RAFT) để khôi phục vùng nền phức tạp.
  - *Rủi ro:* Rất ngốn VRAM. Nguy cơ cao gây Out of Memory (OOM) nếu cấu hình hệ thống VRAM < 6GB hoặc set `MaxLoadNum` quá cao.

### B. Tracking & Segmentation
- **SAM 2 (Segment Anything Model 2):**
  - *Đặc điểm:* Bám sát viền pixel chính xác thay vì hộp chữ nhật.
  - *Tối ưu hóa:* Dữ liệu mask được nén trên RAM bằng `np.packbits` thành dạng nhị phân, tiết kiệm GB I/O đọc/ghi.
  - *Yêu cầu:* Tối ưu cho NVIDIA GPU, tự động sử dụng `bfloat16`.
- **ObjectTracker (RobustWatermarkTracker):**
  - *Đặc điểm:* Hybrid tracking (Template Matching + CSRT) làm phương án dự phòng mặc định.
  - *Anti-drift:* Sử dụng đối chiếu mẫu gốc (original_template) liên tục để chống hiện tượng bám nhầm vào nền video.

### C. OCR
- Các kết quả nhận diện rời rạc trong cùng một vùng khoanh (sub_areas) sẽ được tự động gộp (merge) lại thành một hộp lớn (Bounding Box) duy nhất nếu dùng để tracking.

## 3. Quản Lý Tài Nguyên & CUDA
- **CUDA / Tensor Cores:** Tự động kích hoạt TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`) cho dòng card NVIDIA RTX 30/40 Series trở lên để tối đa hóa tốc độ.
- **Tràn Bộ Nhớ (OOM):** 
  - Các pipeline phân luồng xử lý cần dọn dẹp biến tạm liên tục (`del`, `gc.collect()`).
  - Khi đọc tiến/lùi (Backward tracking), xử lý theo mô hình Chunk-based, đưa frame ra khỏi list bằng `pop()` sau khi dùng xong để giải phóng RAM tức thì.

## 4. Xử Lý Đa Phương Tiện (FFmpeg & OpenCV)
- **OpenCV HW Acceleration:** 
  - Bắt buộc phải tắt HW Acceleration (`cv2.VIDEO_ACCELERATION_NONE`) khi gọi `VideoCapture` cho tác vụ nhảy frame liên tục (như Backward Tracking) để tránh lỗi crash FFmpeg HW Decoder Pool.
- **Xử lý Âm thanh:** Giao phó toàn bộ cho `FFmpeg`. Xử lý hình ảnh riêng biệt, rồi muxing (ghép) vào sau cùng để đảm bảo không lệch đồng bộ (A/V sync).

## 5. Bảng Kiểm Tra (Checklist)
Khi sửa lỗi hoặc thêm tính năng liên quan đến xử lý Video/AI, bắt buộc phải duyệt qua:
- [ ] Tính năng mới có cảnh báo người dùng về phần cứng yếu không (Ví dụ: Cảnh báo VRAM)?
- [ ] Nếu là tác vụ nền (Background Thread), nó có kiểm tra biến cờ (cancel flag) để dừng khẩn cấp không?
- [ ] Có đảm bảo dọn rác bộ nhớ (`gc.collect()`) sau khi tải hoặc hoàn thành chuỗi inference không?
- [ ] Các loại dữ liệu mask (4-tuple box tĩnh, SAM2 packed nhị phân) có được hàm `get_mask()` trong `main.py` hỗ trợ tương thích ngược không?
- [ ] Giao diện (UI) phản hồi trạng thái mô hình rõ ràng bằng tiếng Việt không có lỗi hiển thị?
