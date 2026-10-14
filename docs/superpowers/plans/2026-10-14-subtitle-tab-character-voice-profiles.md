# Kế hoạch Triển khai: Thẻ Hồ sơ Giọng Nhân vật và Cột Loại Giọng trong Tab Chỉnh sửa Phụ đề

## 1. Tổng quan
Kế hoạch triển khai nâng cấp giao diện Tab Chỉnh sửa phụ đề (Bước 2) của `OmniVoiceInterface`:
- Bổ sung khối Thẻ Hồ sơ Giọng Nhân vật đặt ngay phía trên bảng câu thoại phụ đề (`sub_table`).
- Hiển thị chính xác số lượng nhân vật thực tế nhận diện được từ video (`Tổng số nhân vật nhận diện: {N} nhân vật`).
- Bổ sung nút "Nghe Thử Giọng" riêng cho từng nhân vật: tổng hợp câu thoại đầu tiên của nhân vật trong video bằng TTS và mô hình gán của nhân vật đó.
- Nâng cấp Cột 4 của bảng phụ đề thành "Loại Giọng & Nguồn" hiển thị đúng định dạng ("Nhân bản giọng từ video", "Giọng thiết kế: ...", "Chưa có giọng") và tự động đồng bộ khi thay đổi nhân vật.

---

## 2. Kế hoạch Từng Bước (Bite-sized TDD Tasks)

### Nhiệm vụ 1: Viết bộ kiểm thử tự động (TDD) cho Hồ sơ Nhân vật và Cột Loại Giọng
- **Tệp:** `test/test_omnivoice_pipeline.py`
- **Nội dung kiểm thử:**
  1. `test_character_voice_profiles_overview_layout_and_count`: Xác minh nhãn tiêu đề phụ `Video đã phân tích âm thanh · Đã nhận diện`, nhãn số lượng thực tế `Tổng số nhân vật nhận diện: {N} nhân vật`, và số lượng thẻ nhân vật tạo ra tương ứng chính xác với số cụm nhận diện.
  2. `test_sub_table_voice_source_column_display_and_sync`: Xác minh tiêu đề Cột 4 là `Loại Giọng & Nguồn`, giá trị hiển thị cho các trường hợp (Nhân bản giọng từ video, Giọng thiết kế, Chưa có giọng), và khi đổi cột Nhân vật thì Cột 4 tự động đồng bộ giá trị tương ứng.
  3. `test_character_audition_finds_first_dialogue_line`: Xác minh cơ chế nghe thử giọng lấy chính xác câu thoại đầu tiên của nhân vật trong danh sách phụ đề để gửi vào worker tổng hợp âm thanh.

### Nhiệm vụ 2: Xây dựng Giao diện Khối Tổng quan Hồ sơ Nhân vật trong `OmniVoiceInterface`
- **Tệp:** `src/desktop/ui/omnivoice_interface.py`
- **Nội dung thực hiện:**
  1. Trong `page_step_2`, tạo `self.char_overview_card = CardWidget(self.page_step_2)` và chèn ngay phía trên `sub_page_card`.
  2. Thiết kế hàng trên gồm:
     - Nhãn tiêu đề: `Video đã phân tích âm thanh · Đã nhận diện`
     - Nhãn số lượng: `self.lbl_char_count` hiển thị `Tổng số nhân vật nhận diện: {N} nhân vật`
  3. Thiết kế vùng cuộn ngang `QScrollArea` chứa `self.char_cards_layout = QHBoxLayout()`.
  4. Cài đặt phương thức `_update_character_profiles_overview()`:
     - Tổng hợp danh sách nhân vật duy nhất xuất hiện trong `subtitle_blocks` và `speaker_profiles`.
     - Cập nhật nhãn số lượng $N$ thực tế.
     - Dọn dẹp thẻ cũ và tạo các thẻ `CharacterProfileCard` mới với tên, huy hiệu trạng thái (`Sẵn sàng` / `Cần kiểm tra`), loại giọng, mô tả và nút `Nghe Thử Giọng`.

### Nhiệm vụ 3: Cài đặt Cơ chế Tổng hợp Nghe Thử Giọng Nhân vật
- **Tệp:** `src/desktop/ui/omnivoice_interface.py`
- **Nội dung thực hiện:**
  1. Thêm tác vụ `character_audition` vào `OmniVoiceTaskWorker.run()`.
  2. Viết phương thức `_audition_character_voice(spk_name, btn)`:
     - Tìm câu thoại đầu tiên có chữ của nhân vật trong `subtitle_blocks` (hoặc câu mặc định nếu chưa có).
     - Lấy thông số giọng (`override_ref_audio` hoặc `override_instruct`).
     - Chuyển trạng thái nút: `Nghe Thử Giọng` -> `Đang tạo giọng...` (vô hiệu hóa nút trong lúc tạo).
     - Khởi chạy `_launch_worker("character_audition", ...)`.
  3. Trong `_on_task_finished()`:
     - Khi `character_audition` thành công: cập nhật nút thành `Đang phát...` và phát âm thanh qua `_play_audio_file()`.
     - Khi hoàn tất phát âm thanh hoặc có lỗi: khôi phục nút về `Nghe Thử Giọng`.

### Nhiệm vụ 4: Cập nhật Cột 4 Bảng Phụ Đề và Đồng Bộ Hai Chiều
- **Tệp:** `src/desktop/ui/omnivoice_interface.py`
- **Nội dung thực hiện:**
  1. Cập nhật tiêu đề cột 4: `Loại Giọng & Nguồn`.
  2. Nâng cấp `_format_voice_source_display(blk)` chuẩn hóa chuỗi trả về:
     - Nếu có mẫu âm thanh: `Nhân bản giọng từ video`
     - Nếu có instruct: `Giọng thiết kế: {blk.voice_instruct}`
     - Nếu chưa có: `Chưa có giọng`
  3. Cập nhật `_on_subtitle_table_edited`: khi người dùng sửa cột Nhân vật (cột 3), tự động cập nhật text cột 4 và gọi `_update_character_profiles_overview()`.
  4. Gọi `_update_character_profiles_overview()` tại tất cả các điểm mấu chốt: nạp video/phụ đề, nhận diện xong, đặt lại giọng, chọn mẫu giọng.

### Nhiệm vụ 5: Chạy Kiểm Thử Toàn Diện và Xác Minh Hệ Thống
- Chạy toàn bộ các bài kiểm thử:
  - `test/test_omnivoice_pipeline.py`
  - `test/test_auto_pipeline_reliability.py`
  - `test/test_srt_translator.py`
  - `test/test_subtitle_and_logo_remover.py`
  - `test_home_interface.py`
- Đảm bảo 100% kiểm thử thành công, không có bất kỳ hồi quy nào.
- Commit và push thay đổi lên Git.
