# Tài liệu hướng dẫn hệ thống Sakai Studio

Sakai Studio là hệ thống phần mềm chuyên dụng được thiết kế nhằm xử lý các tác vụ phức tạp liên quan đến video và phụ đề. Hệ thống vận hành dựa trên các mô hình học máy và trí tuệ nhân tạo tiên tiến, kết hợp với các thuật toán tối ưu hóa phần cứng nhằm mang lại hiệu suất và chất lượng tốt nhất.

## Chi tiết các phân hệ tính năng và nguyên lý hoạt động

Hệ thống được chia thành bốn phân hệ độc lập, mỗi phân hệ thực hiện một nhóm nhiệm vụ chuyên biệt nhằm tối ưu hóa trải nghiệm người dùng theo đúng quy tắc thiết kế.

### 1. Phân hệ Xóa phụ đề video
- **Chức năng:** Tự động nhận diện và xóa bỏ phần phụ đề, biểu trưng hoặc hình mờ gắn cứng trên video, đồng thời phục hồi lại phần hình ảnh nền bị che khuất.
- **Nguyên lý hoạt động:**
  - **Nhận diện vùng chữ:** Hệ thống cho phép người dùng khoanh vùng khu vực có phụ đề. Mô hình nhận diện hình ảnh sẽ chỉ tập trung quét các ký tự trong vùng này thay vì toàn bộ khung hình, giúp tăng tốc độ nhận diện.
  - **Tái tạo hình ảnh:** Đối với các khung hình phát hiện có chữ, hệ thống tự động tính toán và tạo ra một lớp mặt nạ có độ giãn nở thích ứng theo đúng độ dày của nét chữ. Các thuật toán tái tạo hình ảnh chuyên sâu sẽ phân tích các điểm ảnh lân cận để lấp đầy và phục hồi lại vùng nền, loại bỏ hoàn toàn hiện tượng bóng mờ.
  - **Bỏ qua khung hình trống:** Hệ thống liên tục đánh giá khung hình. Nếu phát hiện khung hình không chứa phụ đề, quá trình tái tạo hình ảnh sẽ bị bỏ qua. Khung hình gốc được sao chép trực tiếp sang luồng xuất, giúp tốc độ xử lý video tăng lên từ hai đến năm lần.
  - **Làm nét cục bộ:** Bộ lọc làm nét thông minh được áp dụng riêng cho vùng vừa bị xóa chữ, giúp chi tiết được khôi phục hài hòa với tổng thể video gốc.

### 2. Phân hệ Trích xuất phụ đề
- **Chức năng:** Chuyển đổi dữ liệu hình ảnh chứa chữ viết hoặc luồng âm thanh giọng nói trong video thành các tệp văn bản phụ đề chuẩn hóa.
- **Nguyên lý hoạt động:** Hệ thống chia tách video thành các khung hình và áp dụng mô hình nhận diện ký tự quang học. Thời gian xuất hiện và biến mất của đoạn văn bản trên màn hình sẽ được ghi nhận và đồng bộ hóa thành tệp phụ đề định dạng chuẩn.

### 3. Phân hệ Dịch phụ đề
- **Chức năng:** Chuyển ngữ các tệp phụ đề gốc sang ngôn ngữ đích một cách tự động với độ chính xác cao.
- **Nguyên lý hoạt động:** Phân hệ sẽ đọc cấu trúc tệp phụ đề, phân tách nội dung cần dịch và các mốc thời gian. Khối văn bản sau đó được gửi đến các dịch vụ xử lý ngôn ngữ tự nhiên. Kết quả trả về được ghép nối lại vào đúng cấu trúc mốc thời gian ban đầu để cho ra tệp phụ đề hoàn chỉnh.

### 4. Phân hệ Tải video
- **Chức năng:** Tải xuống video và âm thanh từ các nguồn cung cấp trực tuyến với chất lượng nguyên bản.
- **Nguyên lý hoạt động:** Hệ thống phân tích liên kết đầu vào, bóc tách và liệt kê các luồng dữ liệu hình ảnh, âm thanh có sẵn. Người dùng có thể lựa chọn tải luồng chất lượng cao nhất, sau đó hệ thống sẽ tự động tải về và hợp nhất hình ảnh cùng âm thanh thành tệp tin duy nhất.

## Kiến trúc tối ưu hóa phần cứng

- **Tăng tốc mã hóa video:** Hệ thống tự động dò tìm bộ xử lý đồ họa chuyên dụng hiện có trên máy tính của người dùng để thực thi quá trình giải mã và mã hóa video, mang lại tốc độ xuất video nhanh gấp năm đến mười lần so với phương thức mã hóa bằng vi xử lý trung tâm.
- **Quản lý bộ nhớ thông minh:** Các mô hình học máy được nạp và lưu giữ sẵn trong bộ nhớ đệm, giúp thời gian khởi tạo từ video thứ hai trở đi bằng không. Hệ thống cũng tự động đo lường dung lượng bộ nhớ đồ họa trống để điều chỉnh số lượng khung hình tải lên luồng xử lý cùng lúc, ngăn ngừa hoàn toàn lỗi tràn bộ nhớ trên các hệ thống có cấu hình phần cứng hạn chế.

## Hướng dẫn cài đặt chi tiết

### Yêu cầu cấu hình hệ thống
- **Hệ điều hành:** Yêu cầu Windows 10 hoặc Windows 11 phiên bản 64 bit.
- **Môi trường thực thi:** Trình thông dịch Python từ phiên bản 3.8 đến 3.11.
- **Phần cứng đề xuất:** Khuyến nghị sử dụng các dòng card xử lý đồ họa rời để đạt tốc độ xử lý tốt nhất và kích hoạt được tính năng tối ưu hóa phần cứng.

### Trình tự cài đặt

1. **Tải mã nguồn hệ thống:**
   Tải toàn bộ thư mục dự án về máy tính và lưu trữ tại một thư mục cục bộ. Mở ứng dụng dòng lệnh tại thư mục gốc của dự án.

2. **Thiết lập môi trường ảo:**
   Tạo môi trường ảo để cài đặt gói phụ thuộc, giúp cách ly và tránh xung đột với các ứng dụng khác trên máy tính bằng lệnh:
   ```cmd
   python -m venv venv
   ```
   Kích hoạt môi trường ảo vừa tạo:
   ```cmd
   venv\Scripts\activate
   ```

3. **Cài đặt thư viện phụ thuộc:**
   Trong khi môi trường ảo đang được kích hoạt, tiến hành cài đặt toàn bộ các gói thư viện cần thiết thông qua tập tin cấu hình bằng lệnh:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Khởi động hệ thống:**
   Sau khi hoàn tất quá trình cài đặt thư viện, người dùng có thể khởi chạy giao diện điều khiển trung tâm bằng lệnh:
   ```cmd
   python gui.py
   ```

## Hướng dẫn vận hành hệ thống

Giao diện của hệ thống được thiết kế theo dạng thẻ, mỗi thẻ độc lập và quản lý một tính năng duy nhất. Trình tự vận hành cơ bản như sau:
1. Chọn thẻ chức năng tương ứng với công việc cần thực hiện.
2. Nạp dữ liệu đầu vào thông qua việc kéo thả tệp tin hoặc chọn thư mục bằng nút bấm.
3. Đối với thẻ Xóa phụ đề video, người dùng cần dùng chuột vẽ vùng chọn bao quanh khu vực chữ trên khung xem trước.
4. Truy cập thẻ Cài đặt nâng cao nếu muốn thay đổi thông số phần cứng.
5. Nhấn nút Bắt đầu để quá trình tự động diễn ra. Khi một tiến trình tự động đang chạy, mọi tính năng can thiệp thủ công sẽ tự động bị khóa để đảm bảo an toàn cho dữ liệu. Cập nhật tiến độ sẽ được hiển thị qua thanh trạng thái và bảng nhật ký hệ thống.
