# Đặc Tả Thiết Kế: Tinh Gọn Giao Diện OmniVoice Và Hiện Đại Hóa Quy Trình Lồng Tiếng Đa Nhân Vật

## 1. Mục Tiêu Và Bối Cảnh
Giao diện OmniVoice trong Sakai Studio hiện có đầy đủ năng lực mạnh mẽ (tách người nói từ video/SRT, gán giọng trí tuệ nhân tạo, điều chỉnh âm thanh, ghép video). Tuy nhiên, giao diện hiện tại gặp vấn đề hiển thị quá nhiều điều khiển cùng lúc, thiếu sự phân cấp ngữ cảnh, khiến người dùng mới dễ bị bối rối và thao tác nhầm lẫn.

Mục tiêu của đợt cải tiến này là đưa trải nghiệm sử dụng OmniVoice lên tầm tiêu chuẩn của các phần mềm làm việc với âm thanh/video hiện đại (tương tự như tham chiếu giao diện studio):
- **P0: Tinh gọn thị giác (Declutter)**: Đưa các thiết lập nâng cao vào vùng thu gọn, giảm số nút bấm thường trực, mỗi bước chỉ có một trọng tâm làm việc duy nhất.
- **P1: Tối ưu quy trình và trạng thái (Workflow & State)**: Thanh điều hướng 5 bước phản ánh rõ tiến độ (hoàn thành / đang làm / chưa sẵn sàng), gom thao tác hàng loạt vào menu chuyên biệt có hộp thoại xác nhận an toàn, thanh công cụ ngữ cảnh xuất hiện đúng lúc.
- **P2: Trải nghiệm mượt mà trong Sakai Studio (Integration & Responsive)**: Đảm bảo vùng cuộn mượt mà không bị lồng hai thanh cuộn, giao diện thích ứng tốt ở độ phân giải tối thiểu 960x600 và 1366x768, bảo toàn dữ liệu khi chuyển đổi tab trong ứng dụng.

## 2. Ràng Buộc Toàn Cục
- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, thẻ và nút giao diện. Chỉ sử dụng biểu tượng FluentIcon.
- Tuân thủ nghiêm ngặt phương pháp phát triển hướng kiểm thử (TDD): viết bài kiểm thử thất bại trước, viết mã nguồn tối thiểu sau.

---

## 3. Kiến Trúc Chi Tiết Từng Bước Trong OmniVoice

### 3.1. Bước 0: Nguồn Đầu Vào (Tinh Gọn Thành 3 Khối Rõ Ràng)
1. **Khối 1 — Tệp Nguồn Đầu Vào**:
   - Nút nạp tệp phụ đề (SRT/VTT/ASS) và video nguồn.
   - Hiển thị nhãn thông tin tệp gọn gàng (tên tệp, số câu thoại, độ dài video).
   - Nút gỡ video nếu chỉ muốn lồng tiếng từ tệp phụ đề.
2. **Khối 2 — Hành Động Chính**:
   - Nút bấm chính "Nhận Diện Nhân Vật Từ Video + SRT" (`PrimaryPushButton`, biểu tượng `FluentIcon.PEOPLE`).
   - Nhãn hướng dẫn tự động cập nhật theo trạng thái nạp tệp.
3. **Khối 3 — Thiết Lập Giọng Mặc Định (Thu Gọn / Dự Phòng)**:
   - Được đóng gói trong một thẻ thu gọn có thể mở/đóng (`is_expanded = False` theo mặc định).
   - Nút bấm chuyển đổi: "Tùy Chọn Giọng Mặc Định (Dự Phòng)" có mũi tên hiển thị trạng thái mở/đóng.
   - Khi mở ra, nội dung bên trong tự động thích ứng theo chế độ:
     - Chọn *Nhân Bản Giọng Nói Từ Mẫu*: Chỉ hiển thị trường chọn tệp âm thanh mẫu hoặc trích xuất từ video.
     - Chọn *Thiết Kế Giọng Nói Theo Thuộc Tính*: Chỉ hiển thị các bộ chọn giới tính, độ tuổi, cao độ và phong cách.
     - Chọn *Giọng Đọc Tự Động*: Chỉ hiển thị bộ chọn ngôn ngữ và danh mục giọng tiếng Việt.
   - Tuyệt đối không hiển thị đan xen các trường của các chế độ khác nhau gây rối mắt.

### 3.2. Bước 1: Quản Lý Nhân Vật (Bảng Phân Vai Làm Trung Tâm)
1. **Bảng Phân Vai Làm Trọng Tâm**:
   - Cột 0: Mã nhân vật (spk_0, spk_1...).
   - Cột 1: Tên nhân vật (cho phép đổi tên trực tiếp).
   - Cột 2: Số câu thoại.
   - Cột 3: Mẫu giọng gốc trích xuất từ video.
   - Cột 4: Phương pháp tạo giọng và giọng đọc đã gán.
   - Cột 5: Trạng thái (Đã khóa giọng / Cần kiểm tra / Chưa xác định).
   - Cột 6: Thao tác gồm 2 nút:
     - Nút nghe âm thanh (`Nghe Giọng Gốc` hoặc `Nghe Thử Giọng Trí Tuệ Nhân Tạo`).
     - Nút `Chỉnh Sửa Giọng` (mở bảng tinh chỉnh chi tiết của nhân vật đó).
2. **Menu Thao Tác Khác (DropdownButton / Menu Lệnh)**:
   - Chuyển toàn bộ các nút thao tác hàng loạt vào menu này để làm sạch thanh tiêu đề:
     - "Tự Động Nhận Diện Lại Nhân Vật"
     - "Nghe Thử Toàn Bộ Nhân Vật"
     - "Gán Nhân Vật Cho Các Câu Đang Chọn"
     - "Đặt Lại Toàn Bộ Giọng"
3. **Hộp Thoại Xác Nhận An Toàn**:
   - Khi chọn "Đặt Lại Toàn Bộ Giọng", hiển thị hộp thoại xác nhận bắt buộc:
     - Tiêu đề: "Xác nhận đặt lại giọng"
     - Nội dung: "Bạn có chắc chắn muốn đặt lại toàn bộ giọng đã gán cho tất cả nhân vật về mặc định? Thao tác này sẽ hủy cấu hình đã khóa."
     - Nút "Đặt Lại" và "Hủy Bỏ".

### 3.3. Bước 2: Biên Tập Phụ Đề (Thanh Công Cụ Ngữ Cảnh Nổi Bật)
1. **Bảng Danh Sách Câu Thoại Làm Trọng Tâm**:
   - Giữ các cột rõ ràng: Số thứ tự, Thời gian bắt đầu, Thời gian kết thúc, Nhân vật, Giọng đọc, Nội dung câu thoại.
   - Bộ lọc trạng thái câu thoại (Tất cả, Cần rà soát, Chưa xác định).
2. **Thanh Công Cụ Ngữ Cảnh Nổi Bật**:
   - Khi người dùng chưa chọn câu nào: thanh công cụ hiển thị gợi ý mờ: "Hãy chọn một hoặc nhiều câu thoại để phân vai, chèn biểu cảm hoặc nghe thử". Các nút chức năng bị vô hiệu hóa.
   - Khi người dùng click chọn 1 hoặc nhiều dòng:
     - Kích hoạt nhanh bộ chọn đổi nhân vật cho các câu đã chọn.
     - Kích hoạt bộ chọn biểu cảm và nút chèn thẻ biểu cảm.
     - Nút "Nghe Thử Câu Đang Chọn" được bật sáng sẵn sàng phát âm thanh.
3. **Tách Khu Vực Phòng Thu Văn Bản**:
   - Đóng gói khu vực Phòng thu văn bản trực tiếp thành một khối thu gọn hoặc chuyển sang sub-tab độc lập, không ép buộc nằm chung chiếm diện tích bảng phụ đề.

### 3.4. Bước 3 & Bước 4: Cân Chỉnh Âm Thanh Và Xuất Video
1. **Bước 3 — Cân Chỉnh Âm Thanh**:
   - Thẻ điều chỉnh cơ bản: Âm lượng gốc, khoảng đệm thời gian, tốc độ nói.
   - Thẻ tham số khuếch tán chuyên sâu (diffusion steps, guidance scale, GPU VRAM): gom vào nhóm thu gọn "Cấu Hình Chuyên Sâu".
2. **Bước 4 — Xuất Và Ghép Video**:
   - Gom gọn thành một màn hình xuất chuyên nghiệp:
     - Lựa chọn chế độ xuất: "Chỉ xuất tệp âm thanh lồng tiếng" hoặc "Xuất video hoàn chỉnh kèm âm thanh lồng tiếng".
     - Nút kiểm tra chất lượng trước khi xuất (`_run_pre_export_quality_check`).
     - Nút hành động chính "Bắt Đầu Xuất" (`PrimaryPushButton`).
     - Thanh tiến trình tổng thể và nhật ký tiến trình thời gian thực.

### 3.5. Thanh Điều Hướng 5 Bước (Stepper)
- Hiển thị trực quan:
  - Bước hiện tại: Viền xanh nổi bật, chữ đậm, nền sáng.
  - Bước đã hoàn thành: Có dấu tích nhỏ hoặc màu sắc hoàn thành.
  - Bước chưa đủ điều kiện: Màu mờ, khi bấm vào sẽ hiển thị thông báo hướng dẫn hoàn thành bước trước.

---

## 4. Kế Hoạch Kiểm Thử Tự Động (TDD)
1. `test_step0_default_voice_settings_collapsible_and_mode_switching`:
   - Xác minh khối thiết lập giọng mặc định ở Bước 0 hỗ trợ thu gọn/mở rộng.
   - Khi chọn chế độ khác nhau, chỉ các điều khiển của chế độ đó hiển thị, các điều khiển khác ẩn đi.
2. `test_step1_batch_actions_menu_and_reset_dialog_confirmation`:
   - Xác minh các thao tác hàng loạt trong Bước 1 nằm trong menu thao tác khác.
   - Xác minh thao tác đặt lại toàn bộ giọng yêu cầu xác nhận và không tự ý xóa khi hủy.
3. `test_step2_contextual_toolbar_enables_only_on_subtitle_selection`:
   - Xác minh thanh công cụ ngữ cảnh ở Bước 2 tự động bật khi có dòng được chọn và tắt khi không chọn dòng nào.
4. `test_stepper_step_state_sync_and_validation`:
   - Xác minh thanh điều hướng 5 bước đồng bộ trạng thái hoàn thành và ngăn chặn nhảy bước khi chưa nạp dữ liệu.
