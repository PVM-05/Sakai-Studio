# Kế Hoạch Triển Khai: Tách Giọng Nói Khỏi Nhạc Nền Bằng Mô Hình Trí Tuệ Nhân Tạo Nhẹ Cho OmniVoice

> **Mục tiêu**: Tích hợp mô hình tách giọng nơ-ron AI siêu nhẹ (VRAM < 250 MB) vào chuỗi xử lý âm thanh của OmniVoice, loại bỏ triệt để nhạc nền, bài hát và tiếng ồn môi trường để việc nhận diện giọng nói đạt độ chính xác tối đa và mẫu giọng lồng tiếng đạt chất lượng phát thanh trong trẻo.

---

### Nhiệm Vụ 1: Khắc phục triệt để các lỗi runtime giao diện (Đã hoàn thành)
- Đã sửa lỗi `RuntimeError: libshiboken: Internal C++ object (PushButton) already deleted` bằng phương thức kiểm tra an toàn `_is_valid_widget`.
- Đã sửa lỗi `AttributeError: 'OmniVoiceInterface' object has no attribute 'voice_prompt_cache_dir'` bằng việc khởi tạo thư mục lưu đệm đầy đủ trong `__init__` và bổ sung phương án dự phòng an toàn.
- Đã kiểm tra 124/124 bài test và đồng bộ lên nhánh chính (`commit 18213d0`).

---

### Nhiệm Vụ 2: Xây dựng bộ tách giọng nơ-ron nhân tạo `AIVoiceSeparator` trong `src/ai_engines/voice_separator.py`
- **Tệp nguồn**: `src/ai_engines/voice_separator.py`
- **Nội dung**:
  - Xây dựng lớp xử lý tách giọng nơ-ron nhẹ `AIVoiceSeparator` sử dụng ONNX Runtime (hỗ trợ cả GPU CUDA và CPU).
  - Thuật toán Spectral Masking nơ-ron: Phân tích phổ STFT, áp dụng mặt nạ tỷ lệ giọng nói (Speech Ratio Mask) để triệt tiêu năng lượng nhạc cụ và tạp âm xung quanh mà không làm méo hài âm giọng người.
  - Xử lý theo từng khối thời gian ngắn (Chunk streaming) bảo đảm mức tiêu thụ VRAM luôn dưới 200 MB trên GPU RTX 4050.
  - Dự phòng mượt mà: nếu chưa tải mô hình AI, tự động chuyển về bộ lọc dải thông 80-7600 Hz chất lượng cao đã chuẩn hóa.
- **Kiểm thử**: Viết bài test trong `test/test_auto_pipeline_reliability.py` xác minh khả năng tách giọng nơ-ron và giảm thiểu năng lượng nhạc nền.

---

### Nhiệm Vụ 3: Tích hợp vào luồng xử lý `tts_engine.py` và `SpeakerDiarizationCoordinator`
- **Tệp nguồn**: `src/ai_engines/tts_engine.py`, `src/ai_engines/speaker_diarization.py`
- **Nội dung**:
  - Thêm cờ tham số `enable_ai_vocal_separation: bool = True` vào `detect_speakers_from_video` và `process_video_subtitles`.
  - Khi cờ này bật, âm thanh tách sạch nhạc nền được sử dụng cho việc trích xuất đặc trưng sinh trắc học và cắt lưu tệp mẫu giọng nhân vật (`voice_ref_path`).
- **Kiểm thử**: Viết test xác minh luồng nhận diện tiếp nhận đúng cờ tách giọng và sinh hồ sơ nhân vật sạch.

---

### Nhiệm Vụ 4: Tích hợp tùy chọn trực quan trên giao diện Bước 1 OmniVoice
- **Tệp nguồn**: `src/desktop/ui/omnivoice_interface.py`, `src/core/config.py`
- **Nội dung**:
  - Bổ sung ô kiểm `chk_ai_vocal_separation`: "Tách sạch nhạc nền bằng trí tuệ nhân tạo" trong thẻ Nhận diện người nói ở Bước 1.
  - Sử dụng biểu tượng `FluentIcon.MICROPHONE` hoặc `FluentIcon.MUSIC`.
  - Lưu và đồng bộ trạng thái vào cấu hình `mmoTTSAIVocalSeparation`.
  - Tự động hiển thị tiến trình trên thanh trạng thái: "Đang tách sạch nhạc nền bằng mô hình trí tuệ nhân tạo...".
- **Kiểm thử**: Viết test xác minh ô kiểm giao diện lưu và nạp đúng cấu hình.

---

### Nhiệm Vụ 5: Kiểm thử hồi quy toàn diện và đẩy mã nguồn
- Chạy toàn bộ các bộ kiểm thử `test_omnivoice_pipeline.py` và `test_auto_pipeline_reliability.py`.
- Bảo đảm 100% các bài kiểm thử đạt kết quả thành công.
- Đẩy mã nguồn lên GitHub.
