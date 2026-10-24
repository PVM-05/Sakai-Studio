# Tài Liệu Đặc Tả Thiết Kế: Cải Thiện Công Cụ Rà Soát Phụ Đề Và Đồng Bộ Thời Lượng OmniVoice

## 1. Bối cảnh và Mục tiêu

Trong hệ thống OmniVoice của Sakai Studio, người dùng nạp tệp video và tệp phụ đề SRT để tự động nhận diện người nói và lồng tiếng. Hiện tại, quy trình rà soát gặp các hạn chế:
1. Khi video và phụ đề không khớp độ dài (ví dụ video bị cắt ngắn hoặc phụ đề từ bản dựng khác), hệ thống không cảnh báo trước khiến quá trình phân vai bị lệch mốc thời gian.
2. Bảng phụ đề tại Bước 3 chưa hỗ trợ lọc các câu đã được xác nhận hoặc đã khóa.
3. Người dùng không thể nghe thử âm thanh gốc của từng câu thoại ngay trên bảng phụ đề để đối chiếu giọng nói nhân vật.
4. Cột người nói trên bảng phụ đề hiển thị dưới dạng chuỗi ký tự thông thường, yêu cầu người dùng gõ tay hoặc phải dùng bộ gán hàng loạt, chưa hỗ trợ chọn trực tiếp nhân vật và tự động khóa câu thoại để tránh bị AI ghi đè khi phân tích lại.

Mục tiêu của đợt cải tiến này là giải quyết triệt để 4 hạn chế trên, tối ưu hóa trải nghiệm rà soát phụ đề theo phong cách giao diện trực quan và chuyên nghiệp.

---

## 2. Tiêu chuẩn và Ràng buộc Thiết kế

Tuân thủ nghiêm ngặt các quy định trong `.agents/rules/design-rules.md`:
- Sử dụng 100% tiếng Việt chuẩn mực, ngữ pháp chính xác.
- Tuyệt đối không sử dụng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc (emoji) hoặc hình khối trang trí trong giao diện và thông báo.
- Sử dụng biểu tượng từ thư viện FluentIcon.
- Phông chữ Segoe UI, hỗ trợ xuống dòng và co giãn linh hoạt.

---

## 3. Kiến trúc và Chi tiết Kỹ thuật

### 3.1. Kiểm tra chênh lệch thời lượng Video và Phụ đề tại Bước 1

- **Hàm xử lý nghiệp vụ**:
  Tạo hàm `check_media_and_srt_duration(video_path: str, subtitle_blocks: List[Any], threshold_sec: float = 5.0) -> Dict[str, Any]` đặt tại `src/desktop/ui/omnivoice_interface.py` (hoặc module tiện ích lõi).
  - Đầu vào:
    - `video_path`: Đường dẫn tệp video nguồn.
    - `subtitle_blocks`: Danh sách các câu phụ đề `SubtitleBlock`.
    - `threshold_sec`: Ngưỡng chênh lệch tính bằng giây (mặc định 5.0 giây).
  - Xử lý:
    - Đo thời lượng video: Sử dụng `cv2.VideoCapture` để đọc số lượng khung hình `CAP_PROP_FRAME_COUNT` và tốc độ khung hình `CAP_PROP_FPS`. Thời lượng = số khung hình / fps.
    - Đo thời lượng phụ đề: Thời điểm kết thúc lớn nhất trong danh sách phụ đề `max(parse_timestamp_to_seconds(blk.end_time) for blk in subtitle_blocks)`.
    - Tính toán chênh lệch `diff = abs(video_duration - srt_duration)`.
    - Xác định trạng thái:
      - `is_mismatch = (diff > threshold_sec) or (srt_duration > video_duration + 2.0)`
      - Trả về từ điển kết quả: `{"video_duration": float, "srt_duration": float, "diff": float, "is_mismatch": bool, "message": str}`.
- **Tích hợp giao diện**:
  - Khi người dùng nạp SRT (`_load_subtitle_file_internal`), nạp video (`_select_video_file`), hoặc xóa video (`_clear_selected_video`), hệ thống tự động gọi hàm kiểm tra nếu có đủ cả video và phụ đề.
  - Cập nhật thông điệp lên nhãn gợi ý `lbl_detect_source_hint`. Nếu có chênh lệch lớn, nhãn sẽ hiển thị cảnh báo màu vàng kèm thông điệp rõ ràng: `"Cảnh báo chênh lệch thời lượng: Video dài {video_duration:.1f} giây, phụ đề dài {srt_duration:.1f} giây (lệch {diff:.1f} giây). Vui lòng kiểm tra lại để đảm bảo phụ đề khớp âm thanh video."`

### 3.2. Mở rộng Bộ lọc Danh mục Câu phụ đề tại Bước 3

- Mở rộng hộp chọn `combo_sub_filter` với các mục:
  - "Tất cả câu thoại"
  - "Độ tin cậy cao"
  - "Cần rà soát"
  - "Chưa xác định"
  - "Đã khóa"
- Phương thức `_filter_subtitle_table`:
  - Khi chọn "Đã khóa": Chỉ hiển thị các dòng có `blk.is_speaker_locked == True`.
  - Giữ nguyên các nhánh xử lý độ tin cậy hiện có, đảm bảo đồng bộ hoàn hảo với các cập nhật trạng thái dòng.

### 3.3. Cột Nghe Gốc trên Bảng Phụ Đề

- **Mở rộng bảng phụ đề `sub_table` thành 7 cột**:
  - Cột 0: STT
  - Cột 1: Mốc Bắt Đầu
  - Cột 2: Mốc Kết Thúc
  - Cột 3: Người Nói
  - Cột 4: Nguồn Giọng
  - Cột 5: Nghe Gốc
  - Cột 6: Nội Dung Thoại
- **Nút Nghe Gốc trong ô Cột 5**:
  - Sử dụng nút bấm nhỏ gọn với biểu tượng `FluentIcon.PLAY`, chú thích: `"Phát âm thanh gốc từ video theo mốc thời gian của câu này"`.
  - Nếu chưa có video hợp lệ: Nút ở trạng thái vô hiệu hóa.
- **Cơ chế trích xuất và phát âm thanh**:
  - Phương thức `_play_subtitle_original_segment(self, blk: SubtitleBlock, btn: Optional[PushButton] = None)`:
    - Tính thời gian bắt đầu `start_s = parse_timestamp_to_seconds(blk.start_time)` và kết thúc `end_s = parse_timestamp_to_seconds(blk.end_time)`.
    - Trích xuất đoạn âm thanh ngắn từ `self.current_video_path` ra tệp tạm WAV trong thư mục bộ nhớ đệm (sử dụng FFmpeg hoặc audio extractor có sẵn).
    - Phát tệp âm thanh này thông qua `_media_player`.
    - Quản lý trạng thái nút: Đổi biểu tượng sang `FluentIcon.PAUSE` trong lúc phát và phục hồi về `FluentIcon.PLAY` khi kết thúc hoặc khi người dùng bấm dừng.

### 3.4. Bộ Chọn Nhân Vật Trực Tiếp trên Dòng và Tự Động Khóa

- **Giao diện Cột 3 (Người Nói)**:
  - Thay vì `QTableWidgetItem` dạng văn bản thuần, đặt widget `ComboBox` vào ô cột 3 cho từng dòng phụ đề.
  - Các mục trong ComboBox:
    - Danh sách tên các nhân vật có trong `self.speaker_profiles` (hoặc các nhân vật đã được nhận diện).
    - Mục mặc định `"Chưa xác định"`.
  - Giá trị hiển thị ban đầu: Khớp với `blk.speaker` hiện tại.
- **Xử lý sự kiện thay đổi nhân vật (`_on_row_speaker_changed`)**:
  - Khi người dùng chọn một nhân vật từ danh sách:
    - Gán `blk.speaker = selected_speaker`.
    - Tự động đánh dấu `blk.is_speaker_locked = True` (khóa câu này để AI không ghi đè khi nhận diện lại).
    - Cập nhật `blk.confidence_level = "high"`.
    - Tự động cập nhật thuộc tính giọng `blk.voice_ref_path` và `blk.voice_instruct` từ hồ sơ nhân vật được chọn.
    - Làm mới hiển thị tại Cột 4 (Nguồn Giọng) của dòng đó.
    - Cập nhật các bộ đếm thống kê và nhãn tóm tắt nhận diện (`_update_diarization_summary_label`).

---

## 4. Kế hoạch Kiểm Thử TDD

1. **`test_check_media_and_srt_duration`**:
   - Kiểm tra hàm `check_media_and_srt_duration` tính đúng thời lượng video từ tệp giả lập hoặc mock, đúng thời lượng phụ đề, phát hiện lệch thời lượng vượt ngưỡng.
2. **`test_sub_filter_locked_category`**:
   - Xác minh `combo_sub_filter` có tùy chọn `"Đã khóa"`.
   - Khi chọn `"Đã khóa"`, các dòng có `is_speaker_locked = False` bị ẩn, các dòng `is_speaker_locked = True` được hiển thị.
3. **`test_subtitle_table_has_original_audio_listen_column`**:
   - Xác minh bảng phụ đề có 7 cột với tiêu đề đúng quy chuẩn.
   - Hàng phụ đề có nút "Nghe Gốc" tại cột 5 với FluentIcon phù hợp.
4. **`test_row_speaker_combobox_locks_and_updates_block`**:
   - Xác minh cột Người nói chứa ComboBox đồng bộ danh sách nhân vật.
   - Khi thay đổi nhân vật trên ComboBox, `blk.speaker` cập nhật đúng, `blk.is_speaker_locked` đổi thành `True`, và độ tin cậy đổi thành `high`.
