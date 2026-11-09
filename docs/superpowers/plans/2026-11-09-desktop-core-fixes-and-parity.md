# Kế hoạch Khắc phục Các Điểm Nghẽn Cốt Lõi Trên Ứng Dụng Máy Tính

## 1. Mục tiêu và Phạm vi
Kế hoạch này tập trung giải quyết dứt điểm các phát hiện nghiêm trọng và các khuyến nghị ưu tiên cao nhất trên ứng dụng máy tính Sakai Studio:
1. **Mức ưu tiên 0**: Khắc phục lỗi gãy luồng Tự Động ở nhánh xóa phụ đề (`_remover_process_worker`) và bổ sung bài kiểm thử hồi quy tự động.
2. **Mức ưu tiên 0**: Dọn dẹp các đường dây chưa kết nối và chuẩn hóa các thông tin công bố (bổ sung thư viện nhận diện giọng nói vào tệp phụ thuộc, xóa bỏ thông tin không chính xác về dịch vụ chuyển ngữ, tích hợp nhà cung cấp mô hình cục bộ vào danh sách lựa chọn, kết nối chức năng xuất phụ đề song ngữ vào giao diện người dùng, kết nối bộ ghi nhận sự cố an toàn).
3. **Mức ưu tiên 1**: Kích hoạt chế độ tính toán dấu phẩy động 16 bit cho mô hình phục hồi nền khi bộ nhớ đồ họa hạn chế và chuẩn hóa tài liệu kiến trúc giữa hai tầng xử lý xóa chữ.

---

## 2. Các Bước Thực Hiện Chi Tiết

### Bước 1: Khắc phục lỗi nhánh xóa phụ đề trong luồng Tự Động và tạo lớp tương thích
- **Tệp chỉnh sửa 1**: `src/core/main.py`
  - Khai báo tên đại diện tương thích ngược: `SubtitleRemover = GUISubtitleRemover`.
  - Cập nhật danh sách xuất khẩu `__all__` để chứa cả `GUISubtitleRemover` và `SubtitleRemover`.
- **Tệp chỉnh sửa 2**: `src/desktop/ui/auto_pipeline_interface.py`
  - Tại hàm `_remover_process_worker`:
    - Thay thế lệnh nhập mô-đun: `from src.core.main import GUISubtitleRemover, verify_media_output`.
    - Sử dụng lớp kế thừa `ProcessCallback` từ `src.core.interface.worker_interface` để thiết lập cơ chế phản hồi tiến độ chuẩn xác tương tự `home_interface.py`.
    - Loại bỏ việc gán thuộc tính không tồn tại trên đối tượng.
    - Tạo tệp tạm và kiểm tra tính toàn vẹn đầu ra qua `verify_media_output` trước khi thay thế nguyên tử.

### Bước 2: Bổ sung bài kiểm thử hồi quy cho tiến trình con của luồng Tự Động
- **Tệp chỉnh sửa**: `test/test_auto_pipeline_reliability.py`
  - Thêm bài kiểm thử `test_remover_process_worker_imports_and_dispatches_correct_class`:
    - Giả lập hàng đợi tiến trình và kiểm tra việc gọi `_remover_process_worker` không phát sinh lỗi nhập mô-đun.
    - Xác minh đối tượng được khởi tạo đúng kiểu và thực thi hàm chạy mà không bị sập tiến trình con.

### Bước 3: Dọn dẹp thông tin công bố và kết nối các chức năng chưa hoàn chỉnh
- **Tệp chỉnh sửa 1**: `src/ai_engines/translator.py`
  - Sửa đổi phần mô tả mở đầu tệp: Xóa bỏ dòng thông tin nhắc đến dịch vụ DeepL không có mã nguồn thực tế.
- **Tệp chỉnh sửa 2**: `requirements.txt`
  - Bổ sung thư viện nhận diện giọng nói `faster-whisper>=1.0.0` để tính năng chuyển giọng nói thành văn bản dự phòng hoạt động đầy đủ.
- **Tệp chỉnh sửa 3**: `src/desktop/ui/translation_interface.py`
  - Bổ sung nhà cung cấp `Ollama` vào danh mục `PROVIDERS_INFO` với địa chỉ cổng giao tiếp mặc định `http://localhost:11434/v1` và các mô hình cục bộ phổ biến.
  - Bổ sung nút thao tác "Xuất song ngữ" tại giao diện Dịch Phụ Đề, kết nối trực tiếp với phương thức `to_bilingual_srt` đã được kiểm thử.
- **Tệp chỉnh sửa 4**: `src/desktop/main.py`
  - Nhập mô-đun và kích hoạt `crash_reporter.install()` trong khối an toàn: Chỉ kích hoạt khi người dùng cung cấp biến môi trường địa chỉ tiếp nhận báo cáo, tránh phát sinh lỗi nếu thiếu thư viện liên quan.

### Bước 4: Tối ưu bộ nhớ đồ họa cho mô hình phục hồi nền và chuẩn hóa kiến trúc
- **Tệp chỉnh sửa**: `src/core/main.py`
  - Tại phương thức khởi tạo `PropainterInpaint`:
    - Kiểm tra lượng bộ nhớ đồ họa khả dụng. Khi bộ nhớ nhỏ hơn hoặc bằng 6 gigabyte, tự động kích hoạt tham số `use_fp16=True` thay vì mặc định chạy 32 bit tiêu tốn bộ nhớ.
  - Bổ sung chú thích giải thích rõ ràng sự khác biệt kiến trúc giữa mô hình mạng nơ-ron học sâu của giao diện máy tính và giải pháp phơi sáng nền theo thời gian của công cụ dòng lệnh.

### Bước 5: Chạy toàn bộ bộ kiểm thử tự động xác minh kết quả
- Thực thi toàn bộ các bài kiểm thử tự động của ứng dụng máy tính.
- Đảm bảo 100% các bài kiểm thử đạt kết quả thành công trước khi hoàn tất.
