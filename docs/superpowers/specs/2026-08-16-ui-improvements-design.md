# Design Spec: Cải tiến UI Toàn diện cho Sakai Studio
Date: 2026-08-16

## Mục tiêu
Nâng cấp trải nghiệm người dùng (UX) và giao diện (UI) tổng thể của Sakai Studio theo tiêu chuẩn hiện đại, áp dụng toàn bộ các quy tắc thiết kế nghiêm ngặt: 100% tiếng Việt chuẩn, không dùng emoji, giao diện co giãn thông minh và sử dụng Fluent Icons.

## Các thay đổi đề xuất

### 1. Tối ưu Không gian & Bố cục (Layout)
- **Tự động mở Toàn màn hình (Maximized)**: Cập nhật tệp `gui.py` (lớp `SubtitleExtractorGUI`) để gọi `self.showMaximized()` ngay sau khi khởi tạo nhằm tận dụng tối đa không gian làm việc.
- **Kích thước tối thiểu an toàn**: Đã có sẵn `self.setMinimumSize(960, 600)`, sẽ kiểm tra và tinh chỉnh lại padding/margin trong thẻ Home và Setting nếu cần để tránh bị chèn ép nội dung.

### 2. Nâng cấp Trải nghiệm Đọc (Word Wrap & Text)
- **Word Wrap cho tất cả SettingCard**: Cập nhật hàm duyệt đệ quy qua các `SettingCard` và gán `setWordWrap(True)` cho `titleLabel` và `contentLabel` trong các tệp:
  - `ui/setting_interface.py`
  - `ui/advanced_setting_interface.py`
  - `ui/translation_interface.py`
  - `ui/extractor_interface.py`
  - `ui/ytdlp_interface.py`
- Loại bỏ hoàn toàn các cấu trúc từ tiếng Anh trong ngoặc đơn ở toàn bộ file giao diện.
- Rà soát các tệp giao diện để đảm bảo sử dụng `FluentIcon` thay vì emoji trang trí.

### 3. Logic Tương tác Thông minh (Smart Interactions)
- **Khóa/Mở khóa theo ngữ cảnh**: Bổ sung logic kết nối tín hiệu (Signal/Slot) giữa các chế độ cấu hình:
  - Khi bật chế độ tự động hóa (Auto Pipeline / Auto Config), các thiết lập thủ công liên quan (như thanh trượt thông số) sẽ bị khóa (`setEnabled(False)`).
  - Tự động thay đổi trạng thái nút "Bắt đầu" hoặc các nút liên kết khi các dữ liệu/mô hình đầu ra đã được tải xong.

## Kế hoạch kiểm thử
- Khởi động ứng dụng, quan sát trạng thái cửa sổ (phải là Maximized).
- Cố tình kéo thu nhỏ cửa sổ xuống `960x600` và kiểm tra hiện tượng tràn/che khuất chữ ở các thẻ có văn bản dài.
- Chuyển đổi trạng thái "Tự động" trên thẻ cài đặt và xác minh các thanh trượt bị khóa (màu xám).
