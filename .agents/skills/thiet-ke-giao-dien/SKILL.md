---
name: thiet-ke-giao-dien
description: Kích hoạt khi phát triển, chỉnh sửa hoặc kiểm thử giao diện người dùng cho ứng dụng Sakai Studio.
---

# Quy trình Thiết kế và Phát triển Giao diện

## 1. Nguyên tắc Cốt lõi
- **Ngôn ngữ**: Tiếng Việt chuẩn mực. Không kèm từ tiếng Anh trong ngoặc đơn.
- **Biểu tượng**: Sử dụng biểu tượng Fluent Icons hoặc SVG. Không dùng biểu tượng cảm xúc.
- **Phông chữ**: Segoe UI sắc nét, phối màu hiện đại và đồng nhất.

## 2. Quy trình Thực hiện Workflow
1. **Phân chia Thẻ Chức năng**:
   - Thẻ Xóa phụ đề video
   - Thẻ Trích xuất phụ đề
   - Thẻ Dịch phụ đề
   - Thẻ Tải video
   - Thẻ Cài đặt nâng cao
2. **Cấu hình Trải nghiệm Người dùng**:
   - Cửa sổ tự động mở toàn màn hình khi khởi động, duy trì kích thước tối thiểu an toàn.
   - Các thẻ cài đặt tự động xuống dòng và co giãn chiều cao theo độ dài văn bản.
   - Khi bật chế độ tự động, lập tức vô hiệu hóa nút điều chỉnh thủ công tương ứng.
   - Tự động mở khóa nút liên kết giữa các thẻ khi dữ liệu đầu ra sẵn sàng.

## 3. Danh sách Kiểm tra trước khi Nghiệm thu
- [ ] Tất cả nhãn và văn bản là tiếng Việt chuẩn.
- [ ] Không có từ tiếng Anh trong ngoặc đơn.
- [ ] Không chứa biểu tượng cảm xúc.
- [ ] Giao diện tự co giãn không che chữ.
- [ ] Nút thao tác mở khóa đúng trạng thái dữ liệu.
