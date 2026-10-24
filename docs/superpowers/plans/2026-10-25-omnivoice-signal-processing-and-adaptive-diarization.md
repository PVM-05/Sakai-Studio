# Kế Hoạch Triển Khai: Chuẩn Hóa Chuỗi Xử Lý Tín Hiệu, Thuật Toán Phân Cụm Thích Ứng & Sửa Lỗi Giao Diện OmniVoice

> **Mục tiêu**: Nâng cấp vượt bậc độ chính xác nhận diện tiếng nói nhân vật trong video, loại bỏ hiện tượng méo tiếng do cắt ngưỡng biên độ thô sơ, tự động ước tính chính xác số lượng nhân vật bằng phân tích trị riêng (Eigengap), chống dính giọng nói ở biên câu thoại, và hoàn thiện giao diện người dùng theo đúng quy chuẩn thiết kế.

---

### Nhiệm Vụ 1: Chuẩn hóa bộ lọc âm thanh trong `VoiceSeparator`
- **Tệp nguồn**: `src/ai_engines/voice_separator.py`
- **Nội dung**:
  - Loại bỏ hoàn toàn cơ chế cắt ngưỡng biên độ cứng (`mask = np.abs(filtered) > noise_rms; filtered = filtered * mask`) gây méo mó và mất hài âm giọng nói.
  - Mở rộng dải tần số bảo toàn từ 100-3500 Hz lên 80-7600 Hz (đáp ứng trọn vẹn dải tần 16 kHz của mô hình CAM++).
  - Chuẩn hóa âm lượng mềm (Peak Normalization an toàn ở mức -1.0 dBFS) để bảo toàn tỷ lệ năng lượng và sắc thái giọng nói.
- **Kiểm thử**: Viết test trong `test/test_auto_pipeline_reliability.py` xác minh tín hiệu âm thanh sau khi lọc không bị méo mó, không bị giật cụt về 0 và bảo toàn hài âm.

---

### Nhiệm Vụ 2: Thuật toán ước lượng số lượng nhân vật thích ứng bằng phân tích trị riêng (Eigengap) & Hạ ngưỡng câu ngắn
- **Tệp nguồn**: `src/ai_engines/speaker_diarization.py`
- **Nội dung**:
  - Triển khai hàm `estimate_optimal_clusters_eigengap` phân tích trị riêng trên ma trận tương đồng cosin và ma trận Laplacian chuẩn hóa để tự động tìm số cụm tự nhiên tối ưu khi người dùng chọn chế độ tự động.
  - Hạ ngưỡng thời lượng câu nòng cốt từ 0.8 giây xuống 0.35 giây để nhân vật phụ có câu thoại ngắn vẫn có cơ hội tạo cụm nhận diện riêng.
  - Cơ chế gán câu ngắn thông minh: phân biệt rõ ràng giữa câu thuộc về cụm có sẵn và câu của nhân vật mới.
- **Kiểm thử**: Viết test xác minh khả năng tự động ước lượng chính xác số lượng người nói (2 người, 3 người) từ ma trận đặc trưng mà không cần người dùng chỉ định trước số lượng.

---

### Nhiệm Vụ 3: Gọt tỉa biên câu thoại thông minh và chống dính giọng nói liền kề
- **Tệp nguồn**: `src/ai_engines/speaker_diarization.py`
- **Nội dung**:
  - Tối ưu hóa hàm tính khoảng đệm âm thanh an toàn `_calculate_safe_audio_padding`: nếu khoảng cách giữa hai câu thoại liền kề nhỏ hơn 0.3 giây, giới hạn khoảng đệm không vượt quá một nửa khoảng cách này để ngăn hiện tượng liếm vào lời nói của người khác.
  - Tinh chỉnh `trim_silence_vad` với ngưỡng năng lượng cục bộ tự thích ứng theo nền âm thanh của riêng câu thoại đó.
- **Kiểm thử**: Viết test xác minh hai câu thoại sát nhau (cách nhau 0.1 giây) không bị dính chéo âm thanh của nhau sau khi cắt đoạn.

---

### Nhiệm Vụ 4: Mở rộng bộ chọn nhân vật và chuẩn hóa toàn diện giao diện người dùng
- **Tệp nguồn**: `src/desktop/ui/omnivoice_interface.py`
- **Nội dung**:
  - Mở rộng giới hạn chọn số lượng nhân vật tối đa từ 6 lên 12 (`spin_max_speakers.setRange(1, 12)`).
  - Chuẩn hóa ngôn ngữ 100% tiếng Việt chuẩn mực, loại bỏ toàn bộ dấu ngoặc đơn và cụm từ chưa tối ưu:
    + "Tự động ước lượng theo giới hạn tối đa"
    + "Cố định chính xác số lượng nhân vật"
    + "Độ nhạy cân bằng", "Độ nhạy cao — Phân tách chi tiết", "Độ nhạy thấp — Ưu tiên gộp nhóm".
  - Kiểm tra và làm sạch toàn bộ các nhãn thông báo, chú giải công cụ trên thẻ Nhận diện người nói.
- **Kiểm thử**: Viết test xác minh giao diện khởi tạo đúng dải giá trị 1-12 và nạp đúng các tham số nhận diện vào luồng xử lý.

---

### Nhiệm Vụ 5: Kiểm thử hồi quy toàn diện và đồng bộ mã nguồn
- **Nội dung**:
  - Chạy toàn bộ các bộ kiểm thử `test_omnivoice_pipeline.py` và `test_auto_pipeline_reliability.py`.
  - Bảo đảm 100% các bài kiểm thử đạt kết quả thành công.
  - Đẩy mã nguồn lên kho lưu trữ GitHub chính của dự án.
