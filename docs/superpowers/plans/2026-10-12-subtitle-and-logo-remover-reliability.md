# Kế Hoạch Khắc Phục Toàn Diện Luồng Xóa Phụ Đề & Logo Di Chuyển

**Mục tiêu:** Giải quyết triệt để 5 nhóm lỗi P1/P2 trong luồng Xóa Phụ Đề Video và Xóa Logo Di Chuyển (`HomeInterface`, `GUISubtitleRemover`, `inpaint_tools.py`, `object_tracker.py`, `subtitle_detect.py`, `video_display_component.py`, `task_list_component.py`), đảm bảo hợp đồng dữ liệu mặt nạ thống nhất, quy đổi tọa độ chính xác một lần, cọ vẽ và tẩy hoạt động đúng thứ tự, bám đuổi logo di chuyển có nội suy khoảng đứt ngắn kèm báo cáo độ phủ, và bảo vệ an toàn tệp đầu ra khi dừng hoặc lỗi.

---

## Wave 01 (P1): Chuẩn Hóa Hợp Đồng Mặt Nạ, Hệ Tọa Độ & Cọ/Tẩy Tự Do
1. **`src/core/tools/inpaint_tools.py`**:
   - Tách bạch 3 định dạng dữ liệu đầu vào:
     1. Hộp tọa độ 4 phần tử `(xmin, xmax, ymin, ymax)`
     2. Mặt nạ nhị phân đóng gói theo khung hình 5 phần tử `(xmin, xmax, ymin, ymax, (packed_mask, mask_shape))` (sửa lỗi `np.unpackbits(packed_mask)[:h*w].reshape(mask_shape)` khi `h * w` không chia hết cho 8)
     3. Danh sách nét cọ và tẩy tự do `[{'radius': r, 'points': [...], 'is_eraser': bool}, ...]`
   - Tách hàm `create_freehand_stroke_mask()` và `create_text_stroke_mask()`, giữ `create_stroke_mask()` làm bộ điều phối an toàn hỗ trợ cả hộp 4 phần tử, 5 phần tử SAM2 và nét cọ mà không ném `ValueError`.
   - Trong `create_text_stroke_mask()`, nếu ảnh trong vùng chọn có độ tương phản thấp khiến Canny+Otsu không tìm thấy cạnh (`count_nonzero == 0`), tự động fallback về mặt nạ hộp vùng chọn để không trả về mặt nạ rỗng ngoài dự kiến.
   - Trong `create_combined_mask()`, hỗ trợ `mask_type` (`"stroke"` hoặc `"box"`) và xử lý tuần tự các nét vẽ trong `strokes_list`: nét cọ (`is_eraser=False`) cộng vào mặt nạ (`255`), nét tẩy (`is_eraser=True`) trừ/xóa (`0`) khỏi mặt nạ (xóa được cả nét cọ trước đó lẫn mặt nạ hộp).
2. **`src/desktop/ui/component/video_display_component.py`, `src/desktop/ui/component/task_list_component.py` & `src/desktop/ui/home_interface.py`**:
   - Chuẩn hóa lưu trữ `TaskOptions.SUB_AREAS` theo hệ tọa độ tỷ lệ preview `(0..1)` và bổ sung kiểm tra idempotent trong `preview_coordinates_to_video_coordinates()` / `video_coordinates_to_preview_coordinates()` để không bao giờ nhân đôi kích thước pixel khi chạy Quét OCR 1 khung hình hoặc Tự động ôm khít.
   - Lưu `freehand_strokes` vào `TaskOptions.FREEHAND_STROKES` và truyền `strokes` sang `GUISubtitleRemover` để `get_mask()` dùng chung `create_combined_mask()` với bản xem trước.

## Wave 02 (P1): An Toàn Tệp Đầu Ra, Mã Thoát Tiến Trình & Dừng An Toàn
1. **`src/core/main.py`, `src/core/tools/subtitle_remover_remote_call.py` & `src/desktop/ui/home_interface.py`**:
   - Bổ sung hàm `verify_media_output(output_path, is_picture=False)` kiểm tra tệp tồn tại, dung lượng `> 0` và đọc được khung hình hợp lệ.
   - Trong `HomeInterface.remover_process()`, ghi ra tệp đích tạm thời riêng biệt (`temp_output_path`), chỉ thay thế nguyên tử (`os.replace`) sang `output_path` sau khi `verify_media_output()` xác nhận thành công.
   - Truyền kết quả có cấu trúc (`Command.RESULT`: `success`, `no_target_found`, `failed`, `cancelled`) qua `SubtitleRemoverRemoteCall` và gọi `sys.exit(1)` khi tiến trình con gặp lỗi để `process.exitcode != 0`.
   - Trong `HomeInterface.stop_button_clicked()`, chỉ xóa tệp tạm của lượt chạy hiện tại (`_current_temp_output_path`), tuyệt đối không xóa `task_item.output_path` đã tồn tại từ trước.

## Wave 03 (P1): Bám Đuổi Logo Di Chuyển, Nội Suy Khoảng Đứt & Cô Lập Tác Vụ
1. **`src/core/tools/object_tracker.py`**:
   - Bổ sung `ObjectTracker.interpolate_and_analyze_tracks()` để nội suy tuyến tính các khoảng mất dấu ngắn (`<= max_interp_gap` khung hình khi độ dịch chuyển hợp lý) và tính toán báo cáo độ phủ (`tracked_frames`, `total_frames`, `coverage_ratio`, `interpolated_frames`, `gap_ranges`).
   - Ghi cảnh báo tiếng Việt rõ ràng khi độ phủ thấp hoặc có khoảng đứt dài.
2. **`src/desktop/ui/home_interface.py` & `src/core/main.py`**:
   - Lưu cấu hình `moving_tracking` và `tracked_sub_list` theo từng nhiệm vụ trong `TaskOptions` thay vì phụ thuộc biến toàn cục `config.movingSubtitleTracking`.
   - Sửa lỗi `TaskStatus.IDLE` không tồn tại thành `TaskStatus.FAILED` và tiếp tục hàng đợi thay vì dừng toàn bộ hàng đợi.
   - Giữ hộp tọa độ gần nhất cho các khung hình mở rộng biên (`expand_frame_ranges`) trong chế độ bám đuổi chuyển động.

## Wave 04 (P2): Bước Lấy Mẫu OCR & Chuẩn Hóa Mặt Nạ Làm Nét / Inpaint
1. **`src/core/tools/subtitle_detect.py`**:
   - Liên kết `config.ocrMode.value` (`"fast"`, `"auto"`, `"precise"`) với `SubtitleDetect._init_sample_step()` (`SAMPLE_STEP = 1` khi chọn chế độ chính xác).
2. **`src/core/main.py`, `src/ai_engines/inpaint/opencv_inpaint.py` & `src/ai_engines/inpaint/propainter_inpaint.py`**:
   - Sửa lỗi chia `255.0` hai lần trong `apply_sharpening_to_inpainted_frame()` khi `mask` đã ở thang `float32` `[0.0, 1.0]`.
   - Chuẩn hóa mặt nạ sang `CV_8UC1` (`uint8`) trong `OpenCVInpaint` và `PropainterInpaint.read_mask()`, hỗ trợ danh sách mặt nạ theo từng khung hình.

## Wave 05 (P2): Bộ Kiểmử Hồi Quy Tự Động & CI
1. **`src/desktop/ui/component/task_list_component.py` & `test_home_interface.py`**:
   - Bổ sung alias `TaskItem = Task` và phương thức `add_tasks()` để tương thích ngược với `test_home_interface.py`.
2. **`test/test_subtitle_and_logo_remover.py` & `.github/workflows/ci-tests.yml`**:
   - Xây dựng bộ kiểm thử tự động bao phủ toàn bộ 6 nhóm kiểm thử bắt buộc và tích hợp vào CI workflow.
