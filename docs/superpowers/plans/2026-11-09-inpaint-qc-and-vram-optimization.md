# Kế hoạch Triển khai: Nâng cấp Kiểm soát Chất lượng Inpaint và Tối ưu hóa Bộ nhớ Đồ họa ProPainter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện dứt điểm trạng thái kiểm thử tự động xanh, nâng cấp cổng kiểm soát chất lượng sau xuất video (4 trạng thái PASS, WARNING, FAIL, UNVERIFIED; lấy mẫu thích ứng theo thời gian, mốc phụ đề và biến thiên quỹ đạo chuyển động; cảnh báo nhấp nháy khung hình) và bổ sung cơ chế chống tràn bộ nhớ đồ họa ProPainter theo công thức thực dụng $N_{\text{safe}}$, tự động chia nhỏ đợt thử lại và chuyển đổi dự phòng có cảnh báo.

**Architecture:** Mở rộng hàm `verify_media_quality` trong `home_interface.py` để hỗ trợ tra cứu tọa độ theo từng khung hình từ quỹ đạo theo dõi chuyển động, tính toán tần suất lấy mẫu tỷ lệ với thời lượng video và mốc phụ đề, trả về 4 trạng thái độc lập và bắt lỗi bộ nhận diện chữ không cho phép mặc định sạch. Trong `main.py`, áp dụng công thức $N_{\text{safe}}$ dựa trên VRAM thực tế và độ phân giải/vùng crop, kèm khối bắt ngoại lệ tràn bộ nhớ đồ họa chia nửa đợt và chuyển đổi sang LaMa từng khung hình kèm cờ cảnh báo `vram_fallback_used`.

**Tech Stack:** Python 3.12, PySide6, OpenCV, NumPy, PyTorch (CUDA), pytest.

## Global Constraints
- Tất cả các văn bản hiển thị trên giao diện, thông báo và tài liệu hướng dẫn phải sử dụng tiếng Việt chuẩn mực.
- Tuyệt đối không sử dụng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Tuân thủ quy tắc kiểm chứng thực tế trước khi khẳng định hoàn thành.

---

### Task 1: Ghi nhận Mã nguồn Khắc phục 3 Lỗi Kiểm thử Tự động và Siết Chặt Điều Kiện Giả Lập

**Files:**
- Modify: `src/ai_engines/speaker_embedding_onnx.py:202-265`
- Modify: `src/ai_engines/tts_engine.py:1805-2035`
- Modify: `test/test_omnivoice_improvements.py:1895-1915`

**Interfaces:**
- Consumes: Cờ `allow_mock` từ các bài kiểm thử đơn vị.
- Produces: Phiên làm việc giả lập an toàn `MockSpeakerSession` khi môi trường thiếu ONNX Runtime; cô lập hạt số ngẫu nhiên không phụ thuộc thư viện PyTorch ngoài; không che giấu lỗi mô hình hỏng hoặc tensor sai lệch.

- [x] **Step 1: Xác minh lại các bài kiểm thử bị ảnh hưởng**
- [x] **Step 2: Commit mã nguồn sửa lỗi kiểm thử**
Commit: `e5a8da9` và `1dfd290`.

---

### Task 2: Nâng cấp Cổng Kiểm soát Chất lượng sau Xuất (`verify_media_quality`)

**Files:**
- Modify: `src/desktop/ui/home_interface.py:72-225, 1649-1660, 1773-1780`
- Test: `test/test_subtitle_and_logo_remover.py:670-850`

**Interfaces:**
- Consumes: `video_path: str`, `sub_areas: List[Tuple[int, int, int, int]]`, tham số bổ sung `tracking_trajectory: Optional[Dict[int, Any]] = None`, `subtitle_intervals: Optional[List[Tuple[int, int]]] = None`, `fps: Optional[float] = None`, `fallback_used: bool = False`.
- Produces: Từ điển kết quả gồm `status: str` (PASS, WARNING, FAIL, UNVERIFIED), `passed: bool`, `residual_text_found: bool`, `flicker_score: float`, `message: str`, `diagnostics: Dict[str, Any]`.

- [x] **Step 1: Viết bài kiểm thử thất bại cho các tính năng mới của cổng chất lượng**
- [x] **Step 2: Chạy kiểm thử để xác nhận thất bại**
- [x] **Step 3: Triển khai mã nguồn nâng cấp trong `verify_media_quality`**
- [x] **Step 4: Chạy lại kiểm thử xác minh thành công**
- [x] **Step 5: Commit mã nguồn cổng kiểm soát chất lượng**
Commit: `6192f93`.

---

### Task 3: Tối ưu hóa Bộ nhớ Đồ họa và Tự Động Giảm Tải Thử Lại cho ProPainter

**Files:**
- Modify: `src/core/main.py:212-255, 477-495, 1094-1185`
- Test: `test/test_subtitle_and_logo_remover.py:850-930`

**Interfaces:**
- Consumes: Độ phân giải `(frame_w, frame_h)`, mặt nạ xóa `mask`, dung lượng VRAM thực tế từ `torch.cuda.mem_get_info()`.
- Produces: $N_{\text{safe}}$ theo công thức thực dụng $\lfloor (V_{\text{available}} \times \alpha) / M_{\text{frame}} \rfloor$; khối tự động chia đôi đợt và chuyển đổi sang LaMa từng khung hình kèm cờ cảnh báo `vram_fallback_used`.

- [x] **Step 1: Viết bài kiểm thử thất bại cho cơ chế thích ứng tải theo độ phân giải và OOM recovery**
- [x] **Step 2: Chạy kiểm thử để xác nhận thất bại**
- [x] **Step 3: Triển khai công thức thích ứng và cơ chế thử lại trong `src/core/main.py`**
- [x] **Step 4: Chạy kiểm thử xác minh thành công (33/33 passed)**
- [x] **Step 5: Commit mã nguồn tối ưu hóa ProPainter**
Commit: `befdb86`.

---

### Task 4: Làm rõ Kiến trúc và Chuẩn hóa Tài liệu Kỹ thuật

**Files:**
- Modify: `src/ai_engines/speaker_diarization.py:299-337`

**Interfaces:**
- Consumes: Hiện trạng thực tế của mô hình nhận diện giọng nói.
- Produces: Chú thích kỹ thuật chuẩn mực, giải thích rõ cơ chế hoạt động của CAM++ ONNX và vai trò của giao diện dự phòng mở rộng trong tương lai.

- [x] **Step 1: Cập nhật tài liệu và chú thích trong `speaker_diarization.py`**
- [x] **Step 2: Kiểm tra cú pháp và kiểm thử toàn bộ hệ thống**
- [x] **Step 3: Commit chuẩn hóa kiến trúc**
Commit: `befdb86`.
