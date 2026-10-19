# Kế Hoạch Triển Khai Thiết Kế Giao Diện Sakai Studio

## 1. Mục Tiêu Dự Án
Tái cấu trúc và nâng cấp trải nghiệm người dùng toàn diện cho Sakai Studio trên cả hai nền tảng ứng dụng máy tính và ứng dụng web theo tài liệu đặc tả kiến trúc tổng thể, không làm gián đoạn hoặc mất mát bất kỳ chức năng đang vận hành nào.

## 2. Tiêu Chuẩn Kỹ Thuật và Quy Tắc Bắt Buộc
- **Ngôn ngữ hiển thị:** Sử dụng 100% tiếng Việt chuẩn mực, không sử dụng từ ngữ tiếng Anh trong dấu ngoặc đơn cạnh từ tiếng Việt.
- **Biểu tượng:** Sử dụng biểu tượng Fluent Icons hoặc SVG hệ thống, không sử dụng biểu tượng cảm xúc trong tiêu đề, nhãn thẻ và thông báo.
- **Phông chữ và bố cục:** Segoe UI sắc nét, màu chủ đạo xanh chàm (#6366F1), tất cả các thẻ cài đặt hỗ trợ tự động xuống dòng và co giãn chiều cao theo nội dung văn bản.
- **Bảo toàn lõi xử lý:** Tái sử dụng toàn bộ các dịch vụ và thuật toán nhận diện ký tự, tẩy xóa hình ảnh, tổng hợp giọng nói và biên dịch phụ đề hiện có.

## 3. Lộ Trình Triển Khai Từng Giai Đoạn

### Giai đoạn 0: Khảo Sát và Sửa Chữa Kiểm Thử Cơ Sở
- Triển khai hàm kiểm soát chất lượng sau xuất `verify_media_quality` trong `src/desktop/ui/home_interface.py` để vượt qua 100% bài kiểm thử trong `test/test_subtitle_and_logo_remover.py`.
- Kiểm tra danh mục tín hiệu, dữ liệu và trạng thái giữa các thẻ giao diện.

### Giai đoạn 1: Chuẩn Hóa Khung Ứng Dụng và Hệ Thống Thiết Kế
- Đồng bộ bảng màu xanh chàm, phông chữ Segoe UI, khoảng cách lưới 4/8/12/16/24/32 px trên toàn bộ giao diện máy tính.
- Chuẩn hóa các thẻ thông báo phản hồi, thanh trạng thái tiến trình và xử lý hiển thị an toàn khi thu nhỏ cửa sổ.

### Giai đoạn 2: Tối Ưu Hóa Khu Vực Xóa Phụ Đề Video
- Tách bạch rõ ràng lớp hiển thị vùng tìm kiếm (nét đứt) và lớp mặt nạ xóa thực tế (phủ màu trong suốt).
- Mở rộng vùng tìm kiếm tự động với thông số đệm thích ứng theo chiều cao ký tự.
- Đồng bộ trạng thái xóa cọ vẽ vào tác vụ đang chọn ngay cả khi danh sách rỗng.
- Sử dụng khung hình lân cận thực tế từ tệp video cho tính năng xem trước theo thời gian.
- Hiển thị kết quả kiểm soát chất lượng sau xuất tách biệt giữa tính toàn vẹn tệp và chất lượng xóa chữ.

### Giai đoạn 3: Hoàn Thiện Quy Trình Công Cụ và Luồng Tự Động
- Công cụ Tải Video: Tách bạch ba hành động phân tích liên kết, thêm vào hàng đợi và bắt đầu tải; hỗ trợ mở tệp trực tiếp trong công cụ Xóa Phụ Đề.
- Công cụ Trích Xuất Phụ Đề: Hỗ trợ nhiều vùng tìm kiếm, hiển thị số lượng câu phát hiện và nút chuyển sang Dịch Phụ Đề.
- Công cụ Dịch Phụ Đề: Bảng chỉnh sửa phụ đề trực quan, bảo toàn mốc thời gian, kiểm tra an toàn trước khi đổi tệp và nút gửi sang Lồng Tiếng.
- Luồng Tự Động: Tích hợp bảng duyệt và chỉnh sửa phụ đề trước khi chèn vào video, đồng bộ thiết lập từ các công cụ con.

### Giai đoạn 4: Tái Cấu Trúc Lồng Tiếng, Cấu Hình MMO và Cài Đặt Nâng Cao
- Lồng Tiếng OmniVoice: Tổ chức thành quy trình từng bước rõ ràng (Nguồn, Phân vai, Giọng nói, Phòng thu, Hòa trộn, Xuất bản).
- Cấu Hình MMO: Gom nhóm các thiết lập sản xuất, mẫu giọng và xem trước trực quan.
- Cài Đặt Nâng Cao: Bổ sung thanh tìm kiếm thiết lập, hiển thị mô tả rõ ràng cho các thông số kỹ thuật và cơ chế phục hồi mặc định an toàn.

### Giai đoạn 5: Đồng Bộ Giao Diện Ứng Dụng Web
- Trang giới thiệu, trang xác thực và bảng điều khiển xử lý video.
- Chuyển địa chỉ kết nối máy chủ sang biến môi trường, xử lý bắt lỗi mất kết nối và làm sạch luồng nhận sự kiện từ máy chủ.

### Giai đoạn 6: Kiểm Định Toàn Diện và Nghiệm Thu
- Chạy toàn bộ các bộ kiểm thử đơn vị, kiểm thử tích hợp và kiểm thử giao diện.
- Đối chiếu đầy đủ danh mục tiêu chí nghiệm thu tại mục 11 của tài liệu thiết kế.
