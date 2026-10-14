# Thiết kế Đặc tả: Thẻ Hồ sơ Giọng Nhân vật và Cột Loại Giọng trong Tab Chỉnh sửa Phụ đề

## 1. Mục tiêu và Yêu cầu
- **Mục tiêu:** Nâng cấp tab Chỉnh sửa phụ đề (Bước 2) để người dùng có thể quan sát trực tiếp:
  1. Số lượng nhân vật thực tế nhận diện được từ video (không phải giới hạn cấu hình số người tối đa).
  2. Danh sách các thẻ hồ sơ giọng nhân vật trực quan đặt ngay phía trên bảng phụ đề, gồm: mã nhân vật, trạng thái (Sẵn sàng / Cần kiểm tra), loại giọng (Nhân bản giọng từ video / Giọng thiết kế), và nút nghe thử giọng riêng cho từng nhân vật.
  3. Nút nghe thử giọng riêng từng nhân vật: Tổng hợp giọng nói câu thoại đầu tiên của nhân vật trong video bằng đúng hồ sơ giọng đã gán (không chỉ phát lại lát cắt âm thanh thô).
  4. Bảng câu thoại phụ đề: Cột loại giọng hiển thị rõ ràng kiểu giọng đang áp dụng cho từng câu (Nhân bản giọng, Giọng thiết kế, Chưa có giọng).
- **Quy chuẩn ngôn ngữ và giao diện:**
  - 100% tiếng Việt chuẩn mực.
  - Tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
  - Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí.
  - Sử dụng biểu tượng FluentIcon cho các thành phần giao diện.

---

## 2. Thiết kế Giao diện Chi tiết trong Tab Chỉnh sửa Phụ đề

### 2.1. Khối Tổng quan và Danh sách Hồ sơ Giọng Nhân vật
Nằm ngay phía trên bảng câu thoại phụ đề:
- **Thẻ CardWidget Tổng quan:**
  - Tiêu đề phụ: "Video đã phân tích âm thanh · Đã nhận diện"
  - Nhãn thống kê: "Tổng số nhân vật nhận diện: {N} nhân vật" (N là số nhân vật thực tế tìm thấy).
  - Vùng chứa danh sách thẻ giọng nhân vật: Bố trí cuộn ngang hoặc hàng ngang (QScrollArea / QHBoxLayout).
- **Thẻ Hồ sơ Nhân vật con (Character Profile Card):**
  - **Dòng 1:** Tên nhân vật (ví dụ: "Nhân vật 1") và Huy hiệu trạng thái:
    - "Sẵn sàng" (màu sắc tích cực) nếu mẫu âm thanh hoặc preset đạt chuẩn.
    - "Cần kiểm tra" (màu sắc cảnh báo) nếu mẫu âm thanh quá ngắn hoặc chưa xác định.
  - **Dòng 2:** Loại giọng:
    - "Nhân bản giọng từ video" kèm nhãn "Mẫu giọng đã tạo".
    - "Giọng thiết kế" kèm thông tin mô tả âm sắc (ví dụ: "Nam · Trầm · Trưởng thành" hoặc tên giọng danh mục).
    - "Chưa có giọng" nếu chưa gán.
  - **Dòng 3:** Nút bấm "Nghe Thử Giọng":
    - Biểu tượng FluentIcon.VOLUME.
    - Tiến trình hiển thị văn bản nút: "Nghe Thử Giọng" → "Đang tạo giọng..." → "Đang phát..." → "Nghe Thử Giọng".

### 2.2. Bảng Câu thoại Phụ đề
- Cột 0 — STT: Số thứ tự dòng phụ đề.
- Cột 1 — Bắt Đầu: Mốc thời gian bắt đầu (cho phép chỉnh sửa).
- Cột 2 — Kết Thúc: Mốc thời gian kết thúc.
- Cột 3 — Nhân Vật: Tên nhân vật (khi đổi nhân vật, tự động kế thừa hồ sơ giọng của nhân vật mới).
- Cột 4 — Loại Giọng & Nguồn: Hiển thị ngắn gọn loại giọng đang gán cho câu:
  - "Nhân bản giọng từ video"
  - "Giọng thiết kế: {tên_giọng}"
  - "Chưa có giọng"
- Cột 5 — Nội Dung Câu Thoại: Lời thoại lồng tiếng.

---

## 3. Quy tắc Hoạt động và Cơ chế Tổng hợp Nghe Thử

1. **Lấy câu nói nghe thử của nhân vật:**
   - Tìm kiếm trong danh sách phụ đề câu thoại đầu tiên được gán cho nhân vật đó có nội dung văn bản hợp lệ.
   - Nếu chưa có câu thoại nào, dùng câu mẫu mặc định: "Xin chào, đây là giọng nói thử nghiệm của {tên_nhân_vật}."
2. **Tổng hợp nghe thử riêng biệt:**
   - Sử dụng động cơ TTSEngine tổng hợp câu mẫu bằng đúng `VoiceClonePrompt` hoặc `instruct` đã khóa của nhân vật đó.
   - Lưu tệp âm thanh tạm vào thư mục đệm nghe thử và phát lại qua `QMediaPlayer`.
   - Quá trình nghe thử không làm thay đổi các khối phụ đề hoặc tiến trình lồng tiếng chính của video.
3. **Đồng bộ hai chiều:**
   - Khi hồ sơ nhân vật thay đổi loại giọng hoặc tệp mẫu (ở Bước 1 hoặc do người dùng chỉnh sửa), danh sách thẻ ở Bước 2 và Cột Loại Giọng trong bảng phụ đề được cập nhật tức thì.

---

## 4. Kế hoạch Kiểm thử và Nghiệm thu
- Kiểm thử số lượng nhân vật thực tế hiển thị chính xác theo kết quả phân cụm.
- Kiểm thử nhãn Loại Giọng & Nguồn trong bảng phụ đề hiển thị đúng loại giọng (Nhân bản hoặc Giọng thiết kế).
- Kiểm thử nút nghe thử giọng của từng nhân vật gọi đúng câu thoại đầu tiên và đúng hồ sơ giọng của nhân vật đó.
- Toàn bộ các bài kiểm thử hiện có tiếp tục vượt qua 100%.
