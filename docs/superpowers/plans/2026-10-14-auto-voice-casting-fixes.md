# Kế Hoạch Thực Thi: Khắc Phục Các Điểm Cốt Lõi Tính Năng Auto Voice Casting (P0 · P1 · P2)

> **Dành cho kỹ sư đại lý:** KỸ NĂNG BẮT BUỘC: Sử dụng subagent-driven-development (khuyến nghị) hoặc executing-plans để thực thi kế hoạch này theo từng tác vụ. Các bước sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Khắc phục triệt để các tồn đọng kỹ thuật P0, P1, P2 trong tính năng Tự động phân vai và chọn giọng (Auto Voice Casting) của Sakai Studio: đưa mô hình nhận diện người nói CAM++ ONNX vào hoạt động thực tế với tiền xử lý FBank 80-bin, xóa bỏ hoàn toàn fallback nhiễu ngẫu nhiên, tuần tự hóa bộ nhớ VRAM 6 GB trên NVIDIA GeForce RTX 4050 Laptop GPU, trích xuất mẫu giọng chuẩn cho từng nhân vật và hoàn thiện giao diện 5 bước cùng độ an toàn vòng đời tín hiệu luồng giao diện.

**Kiến trúc:**
- Tầng trích xuất đặc trưng `speaker_embedding_onnx.py`: Tiền xử lý âm thanh 16 kHz chuẩn FBank 80-bin theo kiến trúc CAM++ (Kaldi fbank + CMVN), suy luận qua ONNX Runtime GPU (CUDAExecutionProvider / giới hạn 768 MiB arena) hoặc CPU; chỉ cho phép MockSession khi bật cờ kiểm thử đơn vị.
- Tầng điều phối `speaker_diarization.py`: Không sinh nhiễu giả lập khi lỗi âm thanh; đánh dấu "Chưa xác định" và mức độ tin cậy `unassigned`; bảo toàn tuyệt đối các câu người dùng đã khóa (`is_speaker_locked`).
- Tầng tổng hợp giọng `tts_engine.py`: Chạy nhận diện người nói và giải phóng phiên ONNX trước khi nạp mô hình OmniVoice TTS vào VRAM GPU; trích xuất mẫu âm thanh giọng nói đại diện cho từng nhân vật.
- Tầng giao diện `omnivoice_interface.py` & `translation_interface.py`: Đồng bộ chỉnh sửa bảng phân vai, bộ lọc câu thoại theo độ tin cậy ở Bước 3, nút nghe thử giọng mẫu, cảnh báo thiếu mô hình CAM++ và bọc an toàn vòng đời luồng nền.

**Ngăn xếp công nghệ:** Python 3.12, PyTorch, Torchaudio, ONNX Runtime (CUDA / CPU), PySide6, QFluentWidgets, NumPy, SoundFile.

## Ràng Buộc Toàn Cục

- 100% tiếng Việt chuẩn mực trên giao diện, thông báo, ghi chú kiểm thử và tài liệu.
- Tuyệt đối không sử dụng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Giới hạn cấp phát bộ nhớ đồ họa ONNX Runtime ở mức 768 MiB trên NVIDIA GeForce RTX 4050 Laptop GPU.
- Luôn giải phóng phiên làm việc ONNX trước khi nạp mô hình tổng hợp giọng nói OmniVoice lên GPU.
- Không tự ý gán ép các câu thoại thiếu dữ liệu hoặc dưới 0.3 giây vào cụm trước; phải đánh dấu Chưa xác định.
- Cho phép người dùng chuyển qua lại giữa 5 bước giao diện mà không làm mất trạng thái dữ liệu hay kết quả đã gán.

---

### Tác Vụ 1: Tiền Xử Lý FBank 80-Bin Và Quản Lý Mô Hình CAM++ ONNX Thực Tế (P0-A)

**Các tệp tin:**
- Sửa đổi: `src/core/config.py`
- Sửa đổi: `src/ai_engines/speaker_embedding_onnx.py`
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp:
  - `extract_fbank_features(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray`: trích xuất đặc trưng 80-bin FBank kèm CMVN chuẩn hóa trừ giá trị trung bình theo thời gian.
  - `create_speaker_session(model_path: Union[str, Path] = "", device_id: int = 0, gpu_mem_limit_mb: int = 768, prefer_cuda: bool = True, strict_cuda: bool = False, allow_mock: bool = False) -> Tuple[Any, str]`: nếu không tìm thấy tệp và `allow_mock=False`, báo lỗi `FileNotFoundError` bằng tiếng Việt rõ ràng.
  - `extract_speaker_embedding`: nhận dạng đầu vào `(1, T, 80)` cho CAM++ ONNX, trả về vector 192 chiều chuẩn L2.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho tiền xử lý FBank và cơ chế từ chối mock trong chế độ thật**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**
- [ ] **Bước 3: Triển khai hàm `extract_fbank_features` và cơ chế nạp CAM++ ONNX trong `speaker_embedding_onnx.py`**
- [ ] **Bước 4: Bổ sung cấu hình `mmoSpeakerModelPath` vào `src/core/config.py`**
- [ ] **Bước 5: Chạy kiểm thử để xác nhận vượt qua**
- [ ] **Bước 6: Lưu bản ghi thay đổi git**

---

### Tác Vụ 2: Xử Lý Lỗi Âm Thanh Và Bảo Vệ Tuyệt Đối Câu Thoại Đã Khóa (P0-B)

**Các tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Sửa đổi: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp:
  - `process_video_subtitles()`: Khi âm thanh nguồn rỗng hoặc không đọc được, không dùng `np.random.randn()`; đánh dấu toàn bộ câu chưa khóa là `speaker="Chưa xác định"`, `confidence_level="unassigned"`.
  - Bảo toàn toàn diện các câu thoại có `is_speaker_locked=True`.
  - Các đoạn thoại có năng lượng âm quá nhỏ (im lặng) hoặc thời lượng < 0.3s được đánh dấu `unassigned`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại khi âm thanh không hợp lệ và kiểm tra giữ nguyên câu khóa**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**
- [ ] **Bước 3: Loại bỏ hoàn toàn fallback nhiễu ngẫu nhiên trong `process_video_subtitles`**
- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**
- [ ] **Bước 5: Lưu bản ghi thay đổi git**

---

### Tác Vụ 3: Tuần Tự Hóa VRAM GPU Và Trích Xuất Mẫu Giọng Từng Nhân Vật (P1)

**Các tệp tin:**
- Sửa đổi: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp:
  - `synthesize_blocks()`: Chạy nhận diện người nói `detect_speakers_from_video()` TRƯỚC `self.load_model()`. Đảm bảo giải phóng phiên ONNX xong mới nạp OmniVoice TTS.
  - `detect_speakers_from_video()`: Trích xuất đoạn âm thanh đại diện rõ nhất (thời lượng dài, độ tin cậy cao) cho từng nhân vật và lưu thành tệp WAV vào thư mục đệm; gán đường dẫn vào `ref_audio_path` và `source_audio_sample`.
  - Đồng bộ hóa các tên thuộc tính `ref_audio_path` và `voice_ref_path`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho quy trình nạp mô hình tuần tự và trích xuất mẫu giọng**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**
- [ ] **Bước 3: Tái cơ cấu thứ tự thực thi trong `synthesize_blocks()` và thêm cơ chế trích xuất mẫu giọng**
- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**
- [ ] **Bước 5: Lưu bản ghi thay đổi git**

---

### Tác Vụ 4: Hoàn Thiện Giao Diện 5 Bước Và An Toàn Vòng Đời Tín Hiệu (P2)

**Các tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Sửa đổi: `src/desktop/ui/translation_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp:
  - Đồng bộ bảng phân vai: Chỉnh sửa tên, chiến lược, preset trong `casting_table` tự động cập nhật vào `speaker_profiles` và các dòng phụ đề tương ứng.
  - Nút nghe thử mẫu giọng nhân vật tại cột Thao Tác trong `casting_table`.
  - Bộ lọc phụ đề theo người nói và độ tin cậy tại Bước 3.
  - Giới hạn thanh trượt tốc độ nói ở mức 0.8–1.5× và thiết lập thẻ cài đặt nâng cao dạng thu gọn tại Bước 4.
  - Báo lỗi rõ ràng trên giao diện khi chưa cài đặt mô hình CAM++ ONNX.
  - Khắc phục lỗi vòng đời tín hiệu `RuntimeError: Signal source has been deleted` trong `translation_interface.py`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho đồng bộ bảng phân vai, lọc phụ đề và an toàn tín hiệu**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**
- [ ] **Bước 3: Triển khai các tính năng trên giao diện `omnivoice_interface.py` và bảo vệ luồng `translation_interface.py`**
- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**
- [ ] **Bước 5: Chạy toàn bộ bộ kiểm thử để xác nhận không có hồi quy**
- [ ] **Bước 6: Lưu bản ghi thay đổi git**
