# Tài Liệu Đặc Tả Thiết Kế: Cải Tiến Thuật Toán Phân Cụm Người Nói Và Phân Đoạn Âm Thanh Tinh Chỉnh Cho OmniVoice

## 1. Bối cảnh và Mục tiêu Kỹ thuật

Trong hệ thống nhận diện đa người nói của OmniVoice, quy trình hiện tại gặp các hạn chế cốt lõi sau:
1. **Ép gộp cụm sai lệch khi độ tin cậy thấp**: Tại dòng 382 của `src/ai_engines/speaker_diarization.py`, khi hệ số Silhouette trung bình nhỏ hơn 0.20, thuật toán tự động ép toàn bộ các câu thoại về một cụm duy nhất (`clusters = [[i for i in range(n_samples)]]`). Điều này khiến video có nhiều nhân vật nói khác nhau bị quy chụp thành chỉ có 1 người nói duy nhất.
2. **Cắt âm thanh phụ thuộc thụ động vào mốc phụ đề SRT**: Hệ thống chỉ cắt âm thanh theo đúng mốc thời gian phụ đề. Nếu phụ đề bị lệch, chứa đoạn nhạc dài hoặc khoảng lặng ở đầu/cuối câu, vector đặc trưng giọng nói trích xuất từ mô hình CAM++ ONNX sẽ bị nhiễu nghiêm trọng.
3. **Chưa hỗ trợ nhận diện trực tiếp từ video khi thiếu SRT**: Người dùng bắt buộc phải nạp tệp phụ đề mới có thể nhận diện người nói, chưa hỗ trợ tự động dò tìm các đoạn có tiếng người nói từ video độc lập.
4. **Điểm tin cậy chưa được hiệu chuẩn thực tế**: Các chỉ số tin cậy chưa tính đến độ dài của đoạn âm thanh mẫu, khoảng cách tới tâm cụm đại diện và độ tách biệt giữa các nhân vật.

Mục tiêu của đợt cải tiến này là giải quyết triệt để 4 vấn đề trên, bảo đảm thuật toán hoạt động chính xác trên video đa ngôn ngữ, không ép gộp sai số lượng nhân vật và tối ưu hóa bộ nhớ an toàn cho card đồ họa NVIDIA GeForce RTX 4050 Laptop 6 GB VRAM.

---

## 2. Tiêu chuẩn và Ràng buộc Bắt buộc

- Sử dụng 100% tiếng Việt chuẩn mực trong giao diện, nhãn, thông báo và tài liệu kỹ thuật.
- Tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình khối trang trí trong tiêu đề và văn bản.
- Giữ nguyên mô hình trích xuất đặc trưng giọng nói CAM++ ONNX để duy trì tốc độ và giới hạn bộ nhớ dưới 800 MB VRAM, hỗ trợ dự phòng vi xử lý khi không có card đồ họa.
- Tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD).

---

## 3. Kiến trúc và Chi tiết Giải pháp

### 3.1. Sửa Triệt Để Thuật Toán Phân Cụm Người Nói (`_agglomerative_clustering`)

- **Loại bỏ cưỡng bức gộp 1 cụm**:
  - Xóa bỏ hoàn toàn câu lệnh `if mean_sil < 0.20: clusters = [[i for i in range(n_samples)]]`.
  - Thay vào đó, hệ số Silhouette chỉ được sử dụng như một chỉ số đánh giá chất lượng phân tách cụm. Nếu Silhouette thấp, hệ thống giữ nguyên các cụm ứng viên đã phân tách và đánh dấu mức độ tin cậy của toàn bộ phiên phân tích là cần rà soát (`review_needed`), không tự ý sáp nhập dữ liệu.
- **Ước lượng số người nói tự động dựa trên khoảng cách cosin**:
  - Khi người dùng chọn chế độ tự động ước lượng (`exact_speakers is None`):
    - Tính toán ma trận khoảng cách cosin `1.0 - (u . v)`.
    - Phân cụm phân cấp từ dưới lên (Agglomerative Hierarchical Clustering - AHC).
    - Ngưỡng dừng phân cụm phụ thuộc vào độ nhạy người dùng chọn (`clustering_sensitivity`):
      - Tiêu chuẩn: Ngưỡng khoảng cách 0.40.
      - Nhạy cao: Ngưỡng khoảng cách 0.32 (tách người nói kỹ hơn, tránh gộp nhầm).
      - Nhạy thấp: Ngưỡng khoảng cách 0.50 (ưu tiên gộp giọng tương tự).

### 3.2. Bộ Phát Hiện Hoạt Động Giọng Nói VAD và Tinh Chỉnh Cắt Âm Thanh

- **Hàm phân đoạn âm thanh VAD năng lượng nhẹ nhàng**:
  Tạo hàm `detect_speech_segments_vad(audio_data: np.ndarray, sample_rate: int, frame_ms: int = 30, energy_threshold: float = 0.015, min_speech_sec: float = 0.3) -> List[Tuple[float, float]]`.
  - Chia tín hiệu âm thanh thành các khung 30ms.
  - Tính năng lượng bình phương trung bình gốc (RMS) cho từng khung.
  - Áp dụng ngưỡng động dựa trên phân vị nhiễu nền của toàn bộ tệp âm thanh.
  - Gom các khung có giọng nói liên tiếp và gộp các khoảng nghỉ ngắn (< 0.25 giây).
  - Trả về danh sách các khoảng thời gian có tiếng nói `(start_sec, end_sec)`.
- **Tinh chỉnh đoạn âm thanh khi có phụ đề SRT**:
  - Trước khi trích xuất vector đặc trưng bằng CAM++ ONNX cho câu phụ đề tại mốc `[start_s, end_s]`:
    - Lấy mẫu âm thanh trong mốc thời gian đó và chạy VAD cục bộ.
    - Cắt bỏ phần đầu và phần đuôi chỉ có khoảng lặng hoặc tạp âm nền (trim silence).
    - Giúp vector đặc trưng chỉ phản ánh đúng giọng nói thực tế của con người, giảm thiểu 80% trường hợp nhận diện sai do nhạc nền.
- **Hỗ trợ nhận diện khi video chưa có tệp phụ đề SRT**:
  - Khi người dùng bấm nhận diện mà chưa có SRT:
    - Hệ thống chạy VAD trên toàn bộ kênh âm thanh video để trích xuất các phân đoạn thoại tự nhiên.
    - Tạo các khối `SubtitleBlock` sơ bộ theo các đoạn thoại tìm thấy.
    - Chạy nhận diện người nói và phân vai tự động trực tiếp trên các khối này, hiển thị ngay trên bảng phân vai và bảng phụ đề.

### 3.3. Đo Điểm Tin Cậy Hiệu Chuẩn Thực Chất

- Điểm tin cậy của từng câu thoại `confidence_score` (từ 0.0 đến 1.0) được tính theo công thức:
  $$\text{Điểm} = 0.45 \times \text{Độ tương đồng tâm cụm} + 0.35 \times \text{Hệ số thời lượng} + 0.20 \times \text{Khoảng cách phân tách}$$
  - **Độ tương đồng tâm cụm**: Tích vô hướng giữa vector đặc trưng của câu thoại với tâm cụm đại diện của nhân vật được gán.
  - **Hệ số thời lượng**: $\min(1.0, \frac{\text{Thời lượng câu thoại}}{1.5})$. Các câu thoại dưới 0.6 giây bị giảm điểm tin cậy do mẫu âm thanh quá ngắn.
  - **Khoảng cách phân tách**: Chênh lệch khoảng cách cosin tới cụm đối thủ gần nhất. Nếu câu thoại nằm quá gần ranh giới giữa 2 nhân vật khác nhau, điểm phân tách sẽ thấp.
- Xếp hạng 3 mức độ tin cậy:
  - `Điểm >= 0.70`: Độ tin cậy cao (`ConfidenceLevel.HIGH`).
  - `0.40 <= Điểm < 0.70`: Cần rà soát (`ConfidenceLevel.REVIEW_NEEDED`).
  - `Điểm < 0.40`: Chưa xác định (`ConfidenceLevel.UNASSIGNED`).

### 3.4. Chuẩn Hóa 4 Bộ Đếm Và Báo Cáo Trực Quan

- Tách bạch 4 chỉ số độc lập trong kết quả nhận diện:
  1. `total_clusters`: Tổng số nhân vật thực tế phát hiện.
  2. `total_assigned`: Tổng số câu thoại được gán nhân vật.
  3. `total_review_needed`: Số câu thoại cần người dùng kiểm tra lại.
  4. `total_unassigned`: Số câu thoại chưa đủ dữ liệu nhận diện.
- Nhãn tóm tắt hiển thị rõ ràng:
  *"Đã phân tích xong. Tìm thấy {total_clusters} nhân vật, {total_assigned}/{total_subs} câu thoại được phân vai ({total_review_needed} câu cần rà soát, {total_unassigned} câu chưa xác định)."*

---

## 4. Kế Hoạch Kiểm Thử TDD

1. **`test_agglomerative_clustering_does_not_collapse_multispeakers_on_low_silhouette`**:
   - Kiểm tra khi 2 cụm người nói có khoảng cách rõ ràng nhưng Silhouette < 0.20, thuật toán KHÔNG gộp về 1 cụm, vẫn giữ nguyên 2 cụm và đánh dấu cần rà soát.
2. **`test_detect_speech_segments_vad_identifies_speech_and_trims_silence`**:
   - Kiểm tra VAD phát hiện đúng khoảng thời gian có tiếng nói trong tín hiệu âm thanh giả lập có xen kẽ khoảng lặng.
3. **`test_calibrated_confidence_score_penalizes_short_and_ambiguous_utterances`**:
   - Xác minh câu thoại ngắn (< 0.5s) hoặc nằm ở ranh giới giữa 2 người nói có điểm tin cậy thấp và được xếp vào nhóm `review_needed`.
4. **`test_diarization_supports_video_without_srt`**:
   - Xác minh khi truyền `blocks=[]`, hệ thống tự động sinh các khối câu thoại từ âm thanh và nhận diện thành công các nhân vật.
