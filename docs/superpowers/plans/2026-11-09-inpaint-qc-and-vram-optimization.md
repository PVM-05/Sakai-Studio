# Kế hoạch Triển khai: Nâng cấp Kiểm soát Chất lượng Inpaint và Tối ưu hóa Bộ nhớ Đồ họa ProPainter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện dứt điểm trạng thái kiểm thử tự động xanh, nâng cấp cổng kiểm soát chất lượng sau xuất video (lấy mẫu thích ứng, bám theo quỹ đạo chuyển động, tích hợp độ nhấp nháy khung hình) và bổ sung cơ chế chống tràn bộ nhớ đồ họa với khả năng tự giảm tải thử lại cho mô hình ProPainter.

**Architecture:** Mở rộng hàm `verify_media_quality` trong `home_interface.py` để hỗ trợ tra cứu tọa độ theo từng khung hình từ quỹ đạo theo dõi chuyển động, tính toán tần suất lấy mẫu tỷ lệ với thời lượng video và chặn kết quả khi chỉ số nhấp nháy khung hình vượt ngưỡng. Trong `main.py` và `propainter_inpaint.py`, bổ sung hệ số điều chỉnh tải theo diện tích xử lý và khối bắt ngoại lệ tràn bộ nhớ đồ họa để giảm nửa tải thử lại hoặc chuyển đổi phương án dự phòng.

**Tech Stack:** Python 3.12, PySide6, OpenCV, NumPy, PyTorch (CUDA), pytest.

## Global Constraints
- Tất cả các văn bản hiển thị trên giao diện, thông báo và tài liệu hướng dẫn phải sử dụng tiếng Việt chuẩn mực.
- Tuyệt đối không sử dụng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Tuân thủ quy tắc kiểm chứng thực tế trước khi khẳng định hoàn thành.

---

### Task 1: Ghi nhận Mã nguồn Khắc phục 3 Lỗi Kiểm thử Tự động

**Files:**
- Modify: `src/ai_engines/speaker_embedding_onnx.py:202-265`
- Modify: `src/ai_engines/tts_engine.py:1805-2035`
- Modify: `test/test_omnivoice_improvements.py:1895-1915`

**Interfaces:**
- Consumes: Cờ `allow_mock` từ các bài kiểm thử đơn vị.
- Produces: Phiên làm việc giả lập an toàn `MockSpeakerSession` khi môi trường thiếu ONNX Runtime; cô lập hạt số ngẫu nhiên không phụ thuộc thư viện PyTorch ngoài.

- [ ] **Step 1: Xác minh lại các bài kiểm thử bị ảnh hưởng**
Chạy lại tập lệnh kiểm thử tự động cục bộ để chắc chắn cả 4 bài kiểm thử đều đạt kết quả tốt.

- [ ] **Step 2: Commit mã nguồn sửa lỗi kiểm thử**
```bash
git add src/ai_engines/speaker_embedding_onnx.py src/ai_engines/tts_engine.py test/test_omnivoice_improvements.py
git commit -m "fix(diarization): cho phep phien gia lap onnx va bo sung kiem thu hat so doc lap"
```

---

### Task 2: Nâng cấp Cổng Kiểm soát Chất lượng sau Xuất (`verify_media_quality`)

**Files:**
- Modify: `src/desktop/ui/home_interface.py:72-222`
- Test: `test/test_subtitle_and_logo_remover.py:670-740`

**Interfaces:**
- Consumes: `video_path: str`, `sub_areas: List[Tuple[int, int, int, int]]`, tham số bổ sung `tracking_trajectory: Optional[Dict[int, Any]] = None`, `fps: Optional[float] = None`.
- Produces: Từ điển kết quả gồm `passed: bool`, `residual_text_found: bool`, `flicker_score: float`, `message: str`, `diagnostics: Dict[str, Any]`.

- [ ] **Step 1: Viết bài kiểm thử thất bại cho các tính năng mới của cổng chất lượng**
Tạo các bài kiểm thử trong `test/test_subtitle_and_logo_remover.py`:
1. `test_verify_media_quality_fails_on_high_flicker`: kiểm tra khi điểm số nhấp nháy cao thì trả về `passed = False`.
2. `test_verify_media_quality_respects_motion_trajectory`: kiểm tra khi vùng chọn tĩnh sạch chữ nhưng tọa độ theo vết chuyển động có chữ thì phát hiện được chữ còn sót.
3. `test_verify_media_quality_adaptive_sampling_count`: kiểm tra số khung hình lấy mẫu tăng tương ứng với video dài.

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**
Run: `pytest test/test_subtitle_and_logo_remover.py -k "test_verify_media_quality" -v`
Expected: FAIL đối với các hành vi mới.

- [ ] **Step 3: Triển khai mã nguồn nâng cấp trong `verify_media_quality`**
1. Lấy mẫu khung hình thích ứng theo thời lượng video và tốc độ khung hình:
   $$\text{sample\_count} = \min(30, \max(5, \text{int}(\text{total\_frames} / \max(1, \text{fps} \times 1.5))))$$
2. Tra cứu tọa độ theo quỹ đạo chuyển động `tracking_trajectory`: nếu tại khung hình `idx` có tọa độ bám đuổi thì dùng tọa độ đó, nếu không thì dùng `sub_areas` ban đầu.
3. Chặn kết quả nếu chỉ số nhấp nháy `flicker_val > 0.40`:
   Đặt `passed = False` kèm thông báo tiếng Việt: "Vùng phục hồi nền có hiện tượng nhấp nháy khung hình vượt ngưỡng an toàn. Khuyến nghị: bật tính năng làm mượt thời gian hoặc chuyển đổi mô hình phục hồi nền."
4. Ghi nhận chi tiết ngoại lệ của bộ nhận diện chữ vào khóa `diagnostics` trong kết quả trả về.
5. Cập nhật nơi gọi `verify_media_quality` trong `home_interface.py` để truyền thêm dữ liệu quỹ đạo bám đuổi `tracking_trajectory` khi có sẵn.

- [ ] **Step 4: Chạy lại kiểm thử xác minh thành công**
Run: `pytest test/test_subtitle_and_logo_remover.py -k "test_verify_media_quality" -v`
Expected: PASS toàn bộ.

- [ ] **Step 5: Commit mã nguồn cổng kiểm soát chất lượng**
```bash
git add src/desktop/ui/home_interface.py test/test_subtitle_and_logo_remover.py
git commit -m "feat(quality): nang cap cong kiem soat chat luong lay mau thich ung va chan nhap nhay"
```

---

### Task 3: Tối ưu hóa Bộ nhớ Đồ họa và Tự Động Giảm Tải Thử Lại cho ProPainter

**Files:**
- Modify: `src/core/main.py:1094-1130, 477-495`
- Modify: `src/ai_engines/inpaint/propainter_inpaint.py:415-446`
- Test: `test/test_subtitle_and_logo_remover.py`

**Interfaces:**
- Consumes: Kích thước khung hình `(H, W)`, diện tích vùng xóa, dung lượng bộ nhớ đồ họa thực tế từ `torch.cuda.mem_get_info()`.
- Produces: `safe_load` được điều chỉnh giảm khi xử lý độ phân giải cao; cơ chế tự động chia đôi đợt khung hình và dọn dẹp bộ nhớ đệm khi phát sinh lỗi tràn bộ nhớ đồ họa.

- [ ] **Step 1: Viết bài kiểm thử thất bại cho cơ chế thích ứng tải theo độ phân giải**
Thêm bài kiểm thử trong `test/test_subtitle_and_logo_remover.py`:
1. `test_propainter_adaptive_load_scales_down_for_high_resolution`: xác thực với độ phân giải Full HD / 4K hoặc vùng xóa lớn, số khung hình cấu hình được tự động giảm để tránh nguy cơ tràn bộ nhớ đồ họa trên dòng thẻ 6GB.
2. `test_propainter_inpaint_oom_recovery`: xác thực khi gặp ngoại lệ bộ nhớ đồ họa, hàm gọi tự động dọn bộ nhớ đệm và thử lại với đợt khung hình nhỏ hơn.

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**
Run: `pytest test/test_subtitle_and_logo_remover.py -k "test_propainter_adaptive" -v`
Expected: FAIL.

- [ ] **Step 3: Triển khai công thức thích ứng và cơ chế thử lại**
1. Cải tiến `get_adaptive_propainter_max_load(self, frame_size=None, crop_area=None)` trong `src/core/main.py`:
   - Nếu chiều rộng video $\ge 1920$ hoặc chiều cao $\ge 1080$, nhân hệ số giảm tải $0.65$.
   - Nếu chiều rộng $\ge 3840$, nhân hệ số giảm tải $0.35$.
   - Giữ mức tối thiểu an toàn là 8 khung hình.
2. Bổ sung cơ chế bọc xử lý ngoại lệ trong vòng lặp batch tại `src/core/main.py:479-495`:
   - Bắt `torch.cuda.OutOfMemoryError` và `RuntimeError` có chứa thông điệp bộ nhớ.
   - Khi xảy ra, gọi `gc.collect()` và `torch.cuda.empty_cache()`, sau đó chia đôi đợt khung hình `batch` và thử lại từng phần.
   - Nếu vẫn thất bại ở mức tối thiểu, chuyển hướng xử lý sang mô hình LaMa để cứu dữ liệu khung hình thay vì để chương trình bị dừng đột ngột.

- [ ] **Step 4: Chạy kiểm thử xác minh thành công**
Run: `pytest test/test_subtitle_and_logo_remover.py -k "propainter" -v`
Expected: PASS.

- [ ] **Step 5: Commit mã nguồn tối ưu hóa ProPainter**
```bash
git add src/core/main.py src/ai_engines/inpaint/propainter_inpaint.py test/test_subtitle_and_logo_remover.py
git commit -m "feat(inpaint): toi uu hoa bo nho do hoa propainter thich ung do phan giai va tu dong thu lai"
```

---

### Task 4: Làm rõ Kiến trúc và Chuẩn hóa Tài liệu Kỹ thuật

**Files:**
- Modify: `src/ai_engines/speaker_diarization.py:299-337`

**Interfaces:**
- Consumes: Hiện trạng thực tế của mô hình nhận diện giọng nói.
- Produces: Chú thích kỹ thuật chuẩn mực, giải thích rõ cơ chế hoạt động của CAM++ ONNX và vai trò của giao diện dự phòng mở rộng trong tương lai.

- [ ] **Step 1: Cập nhật tài liệu và chú thích trong `speaker_diarization.py`**
- [ ] **Step 2: Kiểm tra cú pháp và kiểm thử toàn bộ hệ thống**
- [ ] **Step 3: Commit chuẩn hóa kiến trúc**
```bash
git add src/ai_engines/speaker_diarization.py
git commit -m "docs(diarization): lam ro co che nhan dien giong noi CAM++ ONNX va giao dien du phong"
```
