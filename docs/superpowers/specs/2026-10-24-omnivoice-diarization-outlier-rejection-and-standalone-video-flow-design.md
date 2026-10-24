# Tài Liệu Đặc Tả Thiết Kế: Cơ Chế Từ Chối Ép Gán, Lọc Dải Tần Giọng Nói Và Luồng Nhận Diện Video Độc Lập Cho OmniVoice

## 1. Bối cảnh và Mục tiêu Kỹ thuật

Trong các phiên bản trước, hệ thống nhận diện giọng nói OmniVoice đã giải quyết được việc phân tách cụm cơ bản và phân đoạn âm thanh VAD sơ bộ. Tuy nhiên, qua quá trình rà soát mã nguồn chuyên sâu, còn các rủi ro kỹ thuật trọng yếu sau:

1. **Ép gán đoạn thoại ngắn vào cụm gần nhất (Rủi ro P0)**: Trong cơ chế phân cụm mẫu nòng cốt (`anchor-based clustering`), các đoạn thoại ngắn ngoài danh sách nòng cốt (`non-anchor`) hiện bị gán vô điều kiện vào tâm cụm gần nhất (`np.argmin(dists)`). Nếu đoạn âm thanh đó thực sự là của một nhân vật phụ mới hoặc tạp âm cách biệt, việc ép gán này làm méo mó danh tính nhân vật.
2. **Điểm nghẽn trên giao diện khi xử lý video chưa có phụ đề (Rủi ro P0)**: Mặc dù tầng điều phối nhận diện đã có khả năng sinh câu thoại tự động từ âm thanh, giao diện người dùng và bộ xử lý ngầm vẫn bị chặn bởi ba điều kiện kiểm tra tồn tại phụ đề cũ (tại nút nhận diện Bước 1, hàm khởi chạy tác vụ và hàm đồng bộ kết quả).
3. **Bộ phát hiện hoạt động giọng nói thuần năng lượng dễ bị đánh lừa bởi tiếng ồn và nhạc nền (Rủi ro P0)**: Việc chỉ sử dụng năng lượng bình phương trung bình gốc ngắn hạn 30 mili giây khiến tiếng súng, tiếng nổ, tiếng gõ đập hoặc nhạc nền tiết tấu mạnh bị nhận định sai thành lời nói con người.
4. **Thiếu công cụ lưu trữ mốc thời gian câu thoại tự sinh**: Khi người dùng nhận diện trực tiếp từ video, các câu thoại được sinh ra trên bảng phụ đề nhưng chưa có nút xuất nhanh ra tệp phụ đề định dạng SRT.

Đặc tả này thiết kế các giải pháp kỹ thuật để xử lý triệt để các vấn đề trên.

---

## 2. Tiêu chuẩn và Ràng buộc Bắt buộc

- Sử dụng 100% tiếng Việt chuẩn mực trong giao diện, mã nguồn, nhãn và thông báo.
- Tuyệt đối không dùng từ tiếng Anh đặt trong dấu ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình khối trang trí trong giao diện và tiêu đề.
- Tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD) cho toàn bộ các chức năng.
- Bảo đảm an toàn tài nguyên bộ nhớ dưới 800 MB VRAM trên card đồ họa NVIDIA GeForce RTX 4050 Laptop.

---

## 3. Kiến trúc và Chi tiết Giải pháp

### 3.1. Cơ Chế Từ Chối Ép Gán (Outlier Rejection) Trong Phân Cụm Cosin

- **Vị trí**: Hàm `cosine_cluster_embeddings` trong `src/ai_engines/speaker_diarization.py`.
- **Cơ chế**:
  - Đối với từng mẫu ngoài nhóm nòng cốt (`non_anchor`):
    - Tính khoảng cách cosin tới tất cả các tâm cụm nòng cốt hiện có: $d_{\min} = \min_{c} (1.0 - \vec{v} \cdot \vec{c})$.
    - Nếu $d_{\min} \le \text{distance\_threshold}$ (mặc định 0.40): gán vào cụm gần nhất như bình thường.
    - Nếu $d_{\min} > \text{distance\_threshold}$:
      - **Trường hợp A**: Nếu số lượng cụm hiện tại nhỏ hơn `max_speakers` và mẫu có độ dài hợp lý ($\ge 0.5$ giây): Khởi tạo một cụm nhân vật mới với tâm chính là vector đặc trưng này, tăng số lượng cụm của phiên làm việc.
      - **Trường hợp B**: Nếu đã đạt số cụm tối đa hoặc mẫu quá ngắn / quá xa ($d_{\min} > 0.60$): Không ép gán. Đặt nhãn của mẫu là `-1`, mức độ tin cậy là `ConfidenceLevel.UNASSIGNED` ("Chưa xác định"), điểm tin cậy bằng $0.0$, bảo vệ dữ liệu không bị gán sai lệch.

### 3.2. Bộ Lọc Dải Tần Giọng Nói Đa Đặc Trưng Cho VAD

- **Vị trí**: Hàm `detect_speech_segments_vad` trong `src/ai_engines/speaker_diarization.py`.
- **Nguyên lý âm học**:
  - Giọng nói con người tập trung năng lượng chủ yếu trong dải tần số 200 Hz đến 4000 Hz.
  - Tiếng ồn trắng, tiếng súng, tiếng nổ và gió rít có năng lượng trải đều ở tần số rất cao (> 5000 Hz) hoặc dưới 100 Hz.
- **Triển khai**:
  - Tính tỷ lệ năng lượng dải giọng nói (Voice Band Energy Ratio):
    $$\text{Tỷ lệ} = \frac{\sum_{f \in [200, 4000] \text{ Hz}} |X(f)|^2}{\sum_{f} |X(f)|^2}$$
  - Kết hợp với năng lượng khung RMS và tốc độ đổi dấu tín hiệu (Zero-Crossing Rate) nhẹ nhàng bằng biến đổi Fourier nhanh (FFT) trên từng khung 30 mili giây.
  - Khung âm thanh chỉ được công nhận là có giọng nói khi vừa vượt ngưỡng RMS vừa có tỷ lệ năng lượng dải giọng nói $\ge 0.40$.

### 3.3. Mở Thông Hoàn Toàn Luồng Giao Diện và Bộ Xử Lý Ngầm Cho Video Độc Lập

- **Cập nhật `_update_identify_button_state()` trong `OmniVoiceInterface`**:
  - Khi có video hợp lệ nhưng chưa có phụ đề (`has_video and not has_subs`):
    - Kích hoạt nút `btn_identify_characters`.
    - Hiển thị nhãn gợi ý: *"Sẵn sàng nhận diện trực tiếp từ video: hệ thống sẽ tự động phân đoạn giọng nói và phân vai nhân vật."*
- **Cập nhật `_run_auto_detect_speakers()` trong `OmniVoiceInterface`**:
  - Loại bỏ điều kiện chặn `if not self.subtitle_blocks: return`.
  - Nếu `has_video`: Chuyển sang chế độ tự động nhận diện từ video và khởi chạy bộ xử lý ngầm bình thường với danh sách phụ đề rỗng `[]`.
- **Cập nhật `_on_speakers_detected()` trong `OmniVoiceInterface`**:
  - Khi `not self.subtitle_blocks` và danh sách phụ đề của bộ xử lý ngầm có phần tử:
    - Tiếp nhận toàn bộ: `self.subtitle_blocks = list(self._worker.subtitle_blocks)`.
    - Hiển thị bảng phụ đề qua `_populate_subtitle_table()` và bảng phân vai qua `_update_casting_table()`.
    - Cập nhật nhãn trạng thái nguồn: *"Nguồn: Tự động phân đoạn và nhận diện từ âm thanh video."*
- **Bổ sung nút xuất phụ đề nhanh**:
  - Thêm nút *"Xuất Tệp Phụ Đề"* bên cạnh các thao tác bảng phụ đề để người dùng có thể lưu lại danh sách câu thoại đã phân đoạn ra tệp `.srt`.

---

## 4. Kế Hoạch Kiểm Thử TDD

1. **`test_cosine_clustering_outlier_rejection_marks_distant_samples_unassigned`**:
   - Kiểm tra các mẫu ngoài nòng cốt có khoảng cách cosin lớn (> 0.40) tới tất cả các tâm cụm không bị ép gán vào người khác mà được chuyển thành trạng thái Chưa xác định (`unassigned`).
2. **`test_vad_spectral_band_filters_out_high_frequency_noise`**:
   - Kiểm tra bộ phát hiện giọng nói loại bỏ tín hiệu tạp âm tần số cao (tiếng súng/nhiễu trắng) và chỉ giữ lại phân đoạn có giọng nói trong dải tần con người.
3. **`test_step0_identify_button_enabled_with_video_only`**:
   - Xác minh tại Bước 1, khi chỉ nạp tệp video mà chưa có tệp phụ đề, nút nhận diện nhân vật vẫn mở và nhãn thông báo hiển thị đúng chế độ phân đoạn từ video.
4. **`test_speakers_detected_populates_empty_subtitles_from_worker`**:
   - Xác minh khi danh sách phụ đề ban đầu là rỗng, hàm `_on_speakers_detected()` tiếp nhận toàn bộ các câu thoại do bộ xử lý ngầm sinh ra và nạp đầy đủ lên bảng phụ đề.
5. **`test_export_srt_from_detected_blocks`**:
   - Xác minh hàm xuất tệp phụ đề tạo đúng định dạng SRT từ danh sách câu thoại đã nhận diện trực tiếp từ video.
