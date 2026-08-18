# Đặc tả Thiết kế: Cải tổ Giao diện Xóa Phụ Đề theo Luồng công việc

## 1. Mục tiêu
Thiết kế lại giao diện thẻ Xóa Phụ Đề từ dạng danh sách liệt kê phẳng sang cấu trúc luồng công việc (Workflow/Wizard). Mục tiêu nhằm giúp người dùng mới dễ dàng tiếp cận quá trình xử lý video mà không bị quá tải thông tin, đồng thời phân chia các nhóm chức năng một cách logic.

## 2. Phạm vi thay đổi
Tập trung vào giao diện `ui/advanced_setting_interface.py` và các thành phần liên quan trên `ui/home_interface.py` nếu có.

## 3. Cấu trúc Giao diện Đề xuất

Giao diện sẽ được chia thành 4 phân nhóm chính, sắp xếp theo đúng trình tự thao tác tự nhiên của người dùng:

### Bước 1: Khởi tạo và Khoanh vùng
- Nhóm các công cụ liên quan đến việc xác định vị trí phụ đề gốc.
- Tính năng: Kích hoạt nhận diện logo, Bật/Tắt tính năng Tự động ôm khít nét chữ.
- Giao diện: Sử dụng thanh mở rộng (Expander) hoặc nhóm thẻ (Card Group) đặt ở vị trí đầu tiên.

### Bước 2: Lựa chọn Mô hình Trí tuệ Nhân tạo
- Nhóm các cấu hình về lõi xử lý AI.
- Tính năng: Lựa chọn thuật toán bám đuổi (CSRT, KCF, MIL), Bật/Tắt tăng tốc phần cứng.
- Giao diện: Cung cấp mô tả rõ ràng về ưu nhược điểm của từng thuật toán ngay trên thẻ hiển thị.

### Bước 3: Cấu hình Chất lượng và Độ nét
- Nhóm các tùy chọn ảnh hưởng đến đầu ra hình ảnh sau khi xóa chữ.
- Tính năng: Bảo tồn chuẩn màu sắc video gốc, Poisson Blending (Hòa trộn mượt mà biên giao thoa).
- Giao diện: Đặt ở bước thứ ba, mặc định có thể thu gọn để tránh rối mắt nếu người dùng không có nhu cầu điều chỉnh sâu.

### Bước 4: Xử lý Phụ đề Nâng cao (Tùy chọn)
- Nhóm các cấu hình trích xuất và dịch thuật.
- Tính năng: Tự động xuất file SRT, Nhận diện âm thanh Whisper, Dịch phụ đề tự động, In đè phụ đề đã dịch.
- Giao diện: Có công tắc tổng để Bật/Tắt toàn bộ nhóm chức năng này. Khi bật, các tùy chọn con mới hiển thị.

## 4. Rủi ro và Biện pháp giảm thiểu
- **Rủi ro:** Người dùng chuyên nghiệp cảm thấy phiền phức khi phải mở rộng các nhóm cài đặt.
- **Biện pháp giảm thiểu:** Ghi nhớ trạng thái Đóng/Mở của các thẻ nhóm (Expander) vào tệp cấu hình. Nếu người dùng thường xuyên mở một nhóm, hệ thống sẽ tự động mở nhóm đó trong các lần khởi động tiếp theo.

## 5. Kế hoạch Triển khai
1. Cấu trúc lại mã nguồn các SettingCard trong `ui/advanced_setting_interface.py`.
2. Khởi tạo các nhóm SettingCardGroup hoặc ExpandSettingCard tương ứng với 4 bước.
3. Di chuyển các SettingCard hiện có vào các nhóm mới.
4. Kiểm tra đảm bảo không phá vỡ logic lưu trữ cấu hình hiện tại của phần mềm.
