---
trigger: always_on
---

# Quy tắc Thiết kế Giao diện và Ngôn ngữ

Tài liệu này quy định các tiêu chuẩn bắt buộc về thiết kế giao diện, trải nghiệm người dùng và quy chuẩn trình bày văn bản trong toàn bộ hệ thống phần mềm.

## 1. Quy tắc Ngôn ngữ và Trình bày Văn bản
- Sử dụng hoàn toàn tiếng Việt chuẩn mực trong giao diện, thông báo và tài liệu.
- Tuyệt đối không dùng từ tiếng Anh đặt trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc biểu tượng hình ảnh trang trí trong tiêu đề, thẻ cài đặt và văn bản hướng dẫn.
- Câu từ phải ngắn gọn, rõ nghĩa, chuẩn xác ngữ pháp tiếng Việt.

## 2. Quy tắc Kiến trúc và Phân chia Chức năng
- Mỗi thẻ giao diện chỉ đảm nhận một nhóm nhiệm vụ chuyên biệt, tuyệt đối không chèn trùng lặp tính năng giữa các thẻ:
  - Thẻ Xóa phụ đề video: Chỉ thực hiện nhiệm vụ quét vùng chữ và phục hồi nền video.
  - Thẻ Trích xuất phụ đề: Chỉ thực hiện nhiệm vụ nhận diện chữ hình ảnh và chuyển giọng nói thành tệp phụ đề.
  - Thẻ Dịch phụ đề: Chỉ thực hiện nhiệm vụ chuyển ngữ và biên dịch tệp phụ đề.
  - Thẻ Tải video: Chỉ thực hiện nhiệm vụ tải video và âm thanh từ nguồn trực tuyến.
  - Thẻ Cài đặt nâng cao: Quản lý các tham số kỹ thuật, GPU và cấu hình chuyên sâu.

## 3. Quy tắc Giao diện và Trải nghiệm Người dùng
- Sử dụng thiết kế đồng nhất theo phong cách hiện đại, phối màu hài hòa, phông chữ Segoe UI sắc nét.
- Tất cả các thẻ cài đặt phải hỗ trợ tự động xuống dòng và tự động điều chỉnh chiều cao theo độ dài văn bản để không bị che khuất chữ.
- Cửa sổ ứng dụng luôn duy trì kích thước tối thiểu an toàn và tự động mở rộng toàn màn hình khi khởi động.
- Khi một tính năng tự động được bật, các công cụ điều chỉnh thủ công liên quan phải tự động vô hiệu hóa để tránh xung đột.
- Các nút thao tác liên kết giữa các thẻ phải tự động mở khóa khi dữ liệu đầu ra sẵn sàng.
- Luôn sử dụng icon của fluent hoặc icon khác, không bao giờ sử dụng emoji
