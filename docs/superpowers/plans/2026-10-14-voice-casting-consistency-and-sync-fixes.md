# Kế hoạch Thực hiện: Khắc phục Triệt để Tính Nhất quán và Đồng bộ Phân vai Giọng nói

> **Quy định bắt buộc:** Tuân thủ quy tắc thiết kế giao diện (tiếng Việt chuẩn mực, không dùng từ tiếng Anh trong ngoặc đơn, không dùng biểu tượng cảm xúc, sử dụng FluentIcon). Mọi thay đổi phải đi kèm kiểm thử đơn vị và vượt qua 100% các bài kiểm thử.

**Mục tiêu:** Đảm bảo hệ sinh thái giọng nói OmniVoice và giao diện phân vai phụ đề tôn trọng tuyệt đối phương pháp giọng đã chọn (nhân bản hoặc giọng thiết kế), loại bỏ hoàn toàn việc dùng sót mẫu giọng cũ khi đổi giọng hoặc đổi người nói, đồng thời làm sạch văn bản phụ đề khi phân vai từ tệp SRT.

**Kiến trúc:** 
1. Tầng TTS (`tts_engine.py`): Điều hướng tổng hợp nghiêm ngặt theo phương pháp được chỉ định (`method="preset"` sẽ vô hiệu hóa hoàn toàn mẫu nhân bản và prompt cũ; làm sạch tiền tố người nói và bảo vệ dòng đã khóa trong `detect_speakers_from_srt_text`; đưa phương pháp và dấu vết tệp vào khóa bộ nhớ đệm).
2. Tầng Giao diện (`omnivoice_interface.py`): Đồng bộ hai chiều đầy đủ giữa hồ sơ nhân vật, bảng câu thoại phụ đề và thẻ tổng quan; xóa sạch các trường không liên quan khi đổi phương pháp hoặc gán lại nhân vật; phân biệt chính xác mức độ tin cậy và trạng thái phát âm thanh.

---

### Nhiệm vụ 1: Sửa cơ chế tổng hợp giọng trong TTSEngine để tôn trọng phương pháp đã chọn và nâng cấp khóa bộ nhớ đệm

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Nội dung công việc:**
1. Trong `synthesize_blocks()`:
   - Xác định rõ phương pháp của từng dòng thoại hoặc nhân vật (`prof_method` từ `speaker_profiles` hoặc cấu hình dòng).
   - Nếu nhân vật hoặc dòng thoại mang phương pháp `preset` (hoặc có `voice_instruct` mà không có `voice_ref_path`):
     - Tuyệt đối không tìm kiếm hoặc nạp `source_audio_path`, `ref_audio_path` hay `_speaker_prompt_cache`.
     - Đặt `override_mode = "design"`, `override_prompt = None`, `override_ref = None`.
   - Nếu nhân vật mang phương pháp `clone`:
     - Đặt `override_mode = "clone"`.
   - Truyền `override_mode=active_block_mode` vào `self.generate_single(...)`.
2. Trong `_compute_block_cache_path()`:
   - Bổ sung `active_mode` ("clone" hoặc "preset") và thời gian sửa đổi (mtime) của tệp mẫu âm thanh (nếu có) vào chuỗi băm `raw_key`, ngăn chặn việc tái sử dụng tệp âm thanh đệm sai lệch khi đổi phương pháp.

- [ ] **Bước 1: Viết bài kiểm thử xác minh TTSEngine tôn trọng phương pháp preset khi hồ sơ còn đường dẫn tệp mẫu**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại trước khi sửa**
- [ ] **Bước 3: Chỉnh sửa `synthesize_blocks` và `_compute_block_cache_path` trong `src/ai_engines/tts_engine.py`**
- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**

---

### Nhiệm vụ 2: Xử lý dứt điểm việc gán nhân vật và đổi phương pháp trong giao diện OmniVoice

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Nội dung công việc:**
1. Trong `_on_character_method_changed()`:
   - Khi chọn preset: Xóa `source_audio_path = ""`, `ref_audio_path = ""`, `source_audio_sample = ""`, `prompt = None` khỏi hồ sơ nhân vật.
   - Gọi đồng bộ cả `_populate_subtitle_table()`, `_update_casting_table()` và `_update_character_profiles_overview()`.
2. Trong `_apply_preset_to_target_speaker()`:
   - Chuyển `method = "preset"`, xóa `source_audio_path = ""`, `ref_audio_path = ""` và gán `preset_name`, `voice_instruct`.
   - Đồng bộ toàn diện danh sách phụ đề, bảng phân vai và thẻ tổng quan.
3. Trong `_assign_target_speaker_to_selected_rows()`:
   - Luôn cập nhật và làm sạch cả ba trường `speaker`, `voice_ref_path` và `voice_instruct`. Trường nào không thuộc hồ sơ nhân vật mới phải được xóa về rỗng thay vì giữ giá trị cũ.
   - Khóa dòng đã gán thủ công `is_speaker_locked = True`.
   - Cập nhật lại thẻ tổng quan nhân vật.
4. Trong `_on_player_state_changed()`:
   - Phân biệt trạng thái tạm dừng (`PausedState`) với trạng thái dừng hẳn (`StoppedState`), không tự ý trả nút nghe thử về ban đầu khi chỉ tạm dừng phát.

- [ ] **Bước 1: Viết bài kiểm thử cho `_assign_target_speaker_to_selected_rows` và `_apply_preset_to_target_speaker`**
- [ ] **Bước 2: Chạy kiểm thử để ghi nhận lỗi hiện tại**
- [ ] **Bước 3: Chỉnh sửa các hàm tương ứng trong `src/desktop/ui/omnivoice_interface.py`**
- [ ] **Bước 4: Chạy kiểm thử xác nhận thành công**

---

### Nhiệm vụ 3: Hoàn thiện nhận diện nhân vật từ tệp phụ đề SRT và phân định trạng thái thẻ tổng quan

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/tts_engine.py`
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Nội dung công việc:**
1. Trong `detect_speakers_from_srt_text()`:
   - Giữ nguyên các dòng thoại đã khóa (`is_speaker_locked == True`), không ghi đè người nói.
   - Xóa `voice_ref_path = ""` cho các dòng được phân vai từ văn bản phụ đề (vì nhận diện từ SRT chỉ sử dụng thiết kế giọng).
   - Tách và loại bỏ tiền tố nhãn người nói (ví dụ `[Nam]: `, `Người 1: `) khỏi văn bản câu thoại để TTS không phát âm tên người nói.
2. Trong `_update_character_profiles_overview()`:
   - Xét trạng thái rà soát: Nếu nhân vật có `status == "review_needed"` hoặc tỷ lệ tin cậy thấp, hiển thị nhãn huy hiệu "Cần kiểm tra" và mô tả "Mẫu giọng cần rà soát" thay vì luôn báo "Sẵn sàng".

- [ ] **Bước 1: Viết bài kiểm thử cho việc tách tiền tố SRT, bảo vệ dòng khóa và trạng thái rà soát**
- [ ] **Bước 2: Triển khai mã nguồn trong `tts_engine.py` và `omnivoice_interface.py`**
- [ ] **Bước 3: Chạy kiểm thử xác nhận thành công**

---

### Nhiệm vụ 4: Kiểm thử toàn diện và đồng bộ mã nguồn

- [ ] **Bước 1: Chạy toàn bộ bộ kiểm thử của dự án (81 bài kiểm thử)**
- [ ] **Bước 2: Tạo cam kết Git và đẩy lên cả nhánh `main` và `feature/ui-redesign-xoa-phu-de`**
