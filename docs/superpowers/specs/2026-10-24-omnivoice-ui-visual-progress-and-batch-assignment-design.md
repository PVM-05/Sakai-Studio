# Tài Liệu Đặc Tả Thiết Kế: Trực Quan Hóa Trạng Thái Phân Vai Và Thao Tác Hàng Loạt Trên Bảng Phụ Đề Cho OmniVoice

## 1. Bối cảnh và Mục tiêu Kỹ thuật

Trong quy trình làm việc thực tế với OmniVoice, khi người dùng xử lý video dài có hàng chục đến hàng trăm câu phụ đề:
1. **Thiếu cái nhìn tổng quan về trạng thái phân vai**: Hiện tại chỉ có một nhãn văn bản nhỏ tóm tắt số câu phân vai. Người dùng khó hình dung được tỷ lệ câu thoại đã hoàn thiện so với câu cần rà soát và câu chưa xác định.
2. **Khó phát hiện các dòng có nghi vấn**: Tất cả các dòng trên bảng phụ đề đều có màu nền giống nhau. Người dùng phải nhìn kỹ vào từng dòng để tìm câu có độ tin cậy thấp.
3. **Thao tác thủ công lặp đi lặp lại tốn thời gian**: Khi cần gán nhiều câu thoại liên tiếp cho cùng một nhân vật (ví dụ đoạn hội thoại dài của một nhân vật phụ), người dùng phải nhấp vào từng danh sách thả xuống trên từng dòng, gây mất thời gian và mệt mỏi.

Đặc tả này thiết kế bộ tính năng trực quan hóa và công cụ biên tập hàng loạt trên giao diện Bước 2 của OmniVoice.

---

## 2. Tiêu chuẩn và Ràng buộc Bắt buộc

- Sử dụng 100% tiếng Việt chuẩn mực trong giao diện, mã nguồn, nhãn và thông báo.
- Tuyệt đối không dùng từ tiếng Anh đặt trong dấu ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình khối trang trí trong giao diện và tiêu đề.
- Sử dụng biểu tượng FluentIcon đồng nhất phong cách hiện đại.
- Tất cả các thẻ và nhãn hướng dẫn phải hỗ trợ tự động xuống dòng và tự động co giãn chiều cao theo nội dung.
- Tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD) cho toàn bộ các chức năng.

---

## 3. Kiến trúc và Chi tiết Giải pháp

### 3.1. Tiện Ích Thanh Trạng Thái Phân Vai Phân Đoạn (`CastingProgressBarWidget`)

- **Vị trí**: Nằm ngay đầu khu vực bảng phân vai và bảng phụ đề tại Bước 2 (`page_step_1`).
- **Thành phần giao diện**:
  - Khung thanh tiến độ phân đoạn vẽ tùy biến bằng `QPainter`:
    - Dải màu xanh lá cây (`#4CAF50`): Tỷ lệ câu thoại có độ tin cậy cao hoặc đã khóa.
    - Dải màu vàng cam (`#FFA726`): Tỷ lệ câu thoại cần rà soát (`review_needed`).
    - Dải màu xám nhạt (`#E0E0E0`): Tỷ lệ câu thoại chưa xác định (`unassigned`).
    - Các góc bo tròn 4px, chiều cao thanh 10px, phối màu mượt mà.
  - Nhãn chi tiết đi kèm phía dưới với định dạng rõ ràng:
    *"Tiến độ phân vai: {total_assigned}/{total} câu thoại ({pct_assigned}%) · {total_locked} câu đã khóa · {total_review} câu cần rà soát · {total_unassigned} câu chưa xác định"*
  - Tự động làm mới khi phân tích nhân vật hoàn tất (`_on_speakers_detected`), khi người dùng đổi nhân vật trên bảng, hoặc khi thực hiện thao tác hàng loạt.

### 3.2. Đánh Dấu Màu Nền Cảnh Báo Nhẹ Nhàng Trên Từng Dòng Phụ Đề

- **Vị trí**: Phương thức `_populate_subtitle_table()` và sự kiện thay đổi dữ liệu trong `OmniVoiceInterface`.
- **Quy chuẩn màu sắc nền**:
  - Dòng *Cần rà soát* (`confidence_level == "review_needed"`): Màu nền vàng nhạt `QColor(255, 248, 225, 180)`.
  - Dòng *Chưa xác định* (`confidence_level == "unassigned"` hoặc chưa có người nói): Màu nền đỏ nhạt `QColor(255, 235, 238, 180)`.
  - Dòng *Độ tin cậy cao* hoặc *Đã khóa* (`is_speaker_locked == True` hoặc `confidence_level == "high"`): Màu nền trong suốt mặc định.
- **Bảo đảm kỹ thuật**:
  - Áp dụng qua `item.setBackground(color)` trên toàn bộ các ô `QTableWidgetItem` của dòng đó.
  - Phối màu nhẹ nhàng, bảo đảm phông chữ Segoe UI hiển thị sắc nét và dễ đọc.

### 3.3. Thanh Công Cụ Thao Tác Hàng Loạt Trên Bảng Phụ Đề

- **Cấu hình bảng**:
  - Kích hoạt chế độ chọn nhiều dòng: `self.sub_table.setSelectionMode(QAbstractItemView.ExtendedSelection)`, cho phép giữ phím Shift hoặc Ctrl để chọn liên tiếp nhiều dòng.
- **Thanh công cụ hàng loạt**:
  - Thêm một hàng thao tác nhanh phía trên bảng phụ đề:
    1. Bộ chọn nhân vật hàng loạt: `combo_batch_speaker` (tự động đồng bộ danh sách nhân vật từ `_sync_speaker_selectors`).
    2. Nút *"Gán Cho Các Câu Đã Chọn"*: `btn_batch_assign` (biểu tượng `FluentIcon.PEOPLE`).
       - Khi bấm: duyệt qua tất cả các dòng đang được chọn trên bảng phụ đề.
       - Cập nhật thuộc tính `speaker = selected_speaker`, `is_speaker_locked = True`, `confidence_level = "high"` cho từng khối phụ đề tương ứng.
       - Cập nhật giá trị hiển thị trên ComboBox của từng dòng và cập nhật màu nền của dòng về bình thường.
       - Cập nhật thanh tiến độ phân vai.
       - Hiển thị thông báo nhanh: *"Đã gán nhân vật '{selected_speaker}' cho {count} câu thoại."*
    3. Nút *"Khóa Các Câu Đã Chọn"*: `btn_batch_lock` (biểu tượng `FluentIcon.LOCK`).
       - Đánh dấu `is_speaker_locked = True` cho tất cả các câu đang chọn.
    4. Nút *"Mở Khóa Các Câu Đã Chọn"*: `btn_batch_unlock` (biểu tượng `FluentIcon.SYNC`).
       - Đánh dấu `is_speaker_locked = False` cho tất cả các câu đang chọn.

---

## 4. Kế Hoạch Kiểm Thử TDD

1. **`test_casting_progress_bar_calculates_ratios_correctly`**:
   - Xác minh widget thanh tiến độ phân vai tính toán chính xác tỷ lệ phần trăm giữa câu đã gán, câu cần rà soát và câu chưa xác định.
2. **`test_subtitle_table_row_background_colors_reflect_confidence`**:
   - Xác minh các dòng trong bảng phụ đề có màu nền tương ứng với mức độ tin cậy: vàng nhạt cho câu cần rà soát, đỏ nhạt cho câu chưa xác định, và trong suốt cho câu đã khóa/độ tin cậy cao.
3. **`test_batch_assign_speaker_updates_selected_rows_and_locks`**:
   - Xác minh khi chọn nhiều dòng trên bảng và bấm gán hàng loạt:
     - Tất cả các dòng đã chọn được gán đúng nhân vật mục tiêu.
     - Tất cả các dòng đã chọn được tự động khóa `is_speaker_locked = True` và nâng mức tin cậy `confidence_level = "high"`.
     - Các dòng không được chọn được giữ nguyên.
4. **`test_batch_lock_and_unlock_toggles_selected_rows`**:
   - Xác minh nút khóa và mở khóa hàng loạt cập nhật đúng thuộc tính `is_speaker_locked` trên tất cả các dòng được chọn.
