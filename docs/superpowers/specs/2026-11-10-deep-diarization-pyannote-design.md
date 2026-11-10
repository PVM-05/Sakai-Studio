# Thiết kế Chi tiết Phân đoạn Người nói Chuyên sâu với pyannote và Dự phòng CAM++

Tài liệu thiết kế kỹ thuật cho Giai đoạn 0 và Giai đoạn 1 thuộc kế hoạch nâng cấp năng lực phân tích giọng nói và phân đoạn người nói trong Sakai Studio, tối ưu hóa cho cấu hình phần cứng có bộ nhớ đồ họa 6 GB (NVIDIA GeForce RTX 4050).

---

## 1. Bối cảnh và Mục tiêu

### 1.1. Hiện trạng
Hệ thống hiện tại phân tích người nói qua mô hình CAM++ ONNX (~27 MB) kết hợp thuật toán phân cụm AHC và đánh giá số lượng cụm bằng chỉ số silhouette. Lớp kiểm chứng đối chiếu `ExternalDiarizationVerifier` hiện chỉ là khung giữ chỗ (luôn trả về nguyên vẹn kết quả CAM++). 

Thực tế kiểm nghiệm trên tập dữ liệu video Cheems 6 nhân vật nam (589 giây, 185 câu phụ đề):
- Thuật toán phân cụm CAM++ đạt điểm silhouette cao nhất tại K=2 (0,328) và điểm thấp tại K=6 (0,216), dẫn đến việc đếm thiếu số lượng nhân vật thực tế.
- Việc phân đoạn dựa trên năng lượng âm thanh đơn giản không phát hiện được trường hợp hai người nói chồng nhau.
- Tùy chọn "Phân đoạn người nói chuyên sâu" trên giao diện người dùng đang ở trạng thái khóa và hiển thị cảnh báo chưa tích hợp.

### 1.2. Mục tiêu kỹ thuật
1. Hiện thực hóa động cơ phân đoạn người nói chuyên sâu sử dụng pyannote.audio 3.1 nối trực tiếp vào `ExternalDiarizationVerifier`.
2. Giữ nguyên 100% tính tương thích ngược: CAM++ ONNX tiếp tục đóng vai trò bộ phân cụm cơ sở nhanh và là phương án dự phòng mặc định khi tắt tính năng, khi thiếu môi trường card đồ họa hoặc khi tải mô hình thất bại.
3. Cung cấp cổng thử nghiệm NeMo Diarization / Nemotron 3 cho video có tối đa 8 người nói và xử lý âm thanh nói chồng.
4. Kiểm soát chặt chẽ ngân sách bộ nhớ đồ họa: Giải phóng bộ nhớ hoàn toàn sau khi phân đoạn xong để bảo toàn tài nguyên cho tác vụ tổng hợp giọng đọc OmniVoice.
5. Xây dựng bộ công cụ kiểm thử đường chuẩn (baseline) và đoạn trích video 30 giây để xác minh tự động.

---

## 2. Kiến trúc Hệ thống và Luồng Dữ liệu

```
[Video Đầu Vào + Tệp Phụ Đề SRT]
                │
                ▼
[1. Trích xuất Âm thanh & Tiền xử lý]
  - FFmpeg trích xuất tệp WAV 16 kHz chuẩn
  - Bộ phát hiện phân đoạn âm thanh
                │
                ▼
[2. Phân đoạn Người nói Cơ sở (CAM++ ONNX)]
  - Trích xuất đặc trưng âm học 192 chiều
  - Thuật toán phân cụm AHC cosin
  - Khởi tạo danh sách hồ sơ nhân vật và gán nhãn câu phụ đề
                │
                ▼
[3. Bộ Kiểm chứng Đối chiếu (ExternalDiarizationVerifier)]
  - Kiểm tra cờ bật: "Phân đoạn người nói chuyên sâu"
         │
         ├─ [Trường hợp TẮT hoặc Lỗi / Thiếu GPU / Thiếu mã khóa]:
         │      └─ Giữ nguyên kết quả CAM++ (Tự động dự phòng an toàn)
         │
         └─ [Trường hợp BẬT (pyannote.audio 3.1)]:
                ├─ Tải trọng số pipeline từ thư mục lưu trữ cục bộ
                ├─ Nạp lên card đồ họa (sử dụng 1,2 - 1,5 GB VRAM)
                ├─ Thực thi diarization với num_speakers hoặc min/max speakers
                ├─ Thu hồi hoàn toàn bộ nhớ đồ họa (empty_cache + gc.collect)
                └─ Tinh chỉnh nhãn câu thoại theo độ trùng thời gian IoU:
                     ├─ IoU > 0,5: Đồng thuận, giữ nhãn và tăng độ tin cậy
                     └─ Lệch nhãn hoặc phát hiện nói chồng: Cập nhật nhãn mới,
                        đánh dấu cờ review_needed
                │
                ▼
[4. Đồng bộ Nhãn vào Khối Phụ đề SRT & Hồ sơ Nhân vật]
                │
                ▼
[5. Giải phóng Tài nguyên & Chuyển sang Bước Tổng hợp Giọng đọc]
  - Thu hồi toàn bộ tài nguyên nhận diện
  - Sẵn sàng toàn bộ 6 GB bộ nhớ đồ họa cho động cơ OmniVoice TTS
```

---

## 3. Thiết kế Chi tiết Từng Thành phần

### 3.1. Nâng cấp Lớp ExternalDiarizationVerifier
- **Vị trí tệp:** `src/ai_engines/speaker_diarization.py`
- **Phương thức `is_available()`:**
  - Kiểm tra khả dụng của thư viện `pyannote.audio`.
  - Kiểm tra khả dụng của card đồ họa NVIDIA CUDA (`torch.cuda.is_available()`).
  - Trả về `True` nếu môi trường đáp ứng đầy đủ điều kiện thực thi tăng tốc phần cứng.
- **Phương thức `verify_or_fallback()`:**
  - Nhận vào: `audio_path`, `campp_speakers` (kết quả từ CAM++), `min_speakers`, `max_speakers`, `exact_speakers`.
  - Đọc cấu hình hệ thống: kiểm tra trạng thái cờ `deepDiarizationEnabled` và loại động cơ `deepDiarizationEngine`.
  - Nếu cờ tắt: ghi nhật ký thông tin và hoàn trả nguyên vẹn `campp_speakers`.
  - Khối bọc an toàn (`try...except`):
    - Khởi tạo đường dẫn lưu trữ mô hình từ cấu hình hoặc thư mục mặc định `%APPDATA%/Sakai-Studio/models`.
    - Nạp `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", cache_dir=model_dir, use_auth_token=hf_token)`.
    - Chuyển pipeline sang thiết bị `cuda`.
    - Chạy phân đoạn với các tham số khống chế số lượng người nói.
    - Gọi hàm ánh xạ và tinh chỉnh nhãn phụ đề giữa pyannote và CAM++.
    - Giải phóng đối tượng pipeline và dọn dẹp bộ nhớ đồ họa bằng `torch.cuda.empty_cache()` và `gc.collect()`.
  - Khối xử lý ngoại lệ: Bắt mọi lỗi phát sinh (thiếu mạng, mã token không hợp lệ, lỗi nạp trọng số), ghi nhật ký cảnh báo chi tiết và tự động hoàn trả kết quả CAM++ mà không ngắt luồng xử lý chính.

### 3.2. Thuật toán Đối chiếu và Tinh chỉnh Nhãn Phụ đề
1. Lấy danh sách các đoạn phát âm từ kết quả pyannote: Mỗi đoạn gồm thời gian bắt đầu, thời gian kết thúc và nhãn người nói định danh.
2. Với mỗi khối câu phụ đề trong danh sách SRT:
   - Tính toán độ trùng lặp thời gian giữa khoảng thời gian của câu phụ đề và các đoạn phát âm từ pyannote.
   - Xác định nhãn người nói có độ dài giao nhau lớn nhất.
   - Nếu tỷ lệ giao nhau lớn hơn 50% thời lượng câu:
     - Trường hợp trùng khớp với phân cụm của CAM++: Giữ nguyên nhãn, gán độ tin cậy cao.
     - Trường hợp pyannote nhận diện người nói khác: Cập nhật mã nhân vật theo pyannote, gắn cờ `review_needed = True` để người dùng kiểm tra trên bảng duyệt câu thoại.
   - Trường hợp phát hiện nhiều người nói trong cùng một câu phụ đề (nói chồng): Ghi nhận nhãn người nói chiếm ưu thế, đồng thời lưu thông tin cảnh báo nói chồng vào thuộc tính mở rộng của khối phụ đề.

### 3.3. Cổng Thử nghiệm Nemotron 3 Diarization
- **Vị trí tệp:** `src/ai_engines/speaker_diarization.py`
- **Hàm `verify_with_nemotron()`:**
  - Sử dụng mô hình NeMo Sortformer / Nemotron 3 Diarization với kiến trúc đầu ra xác suất tối đa 8 người nói.
  - Phù hợp với các video đối thoại phức tạp có số lượng nhân vật cố định dưới 8.
  - Được kích hoạt có điều kiện khi `deepDiarizationEngine == "nemotron"`.

### 3.4. Quản lý Cấu hình Hệ thống
- **Vị trí tệp:** `src/core/config.py`
- Bổ sung các trường dữ liệu cấu hình:
  - `deepDiarizationEnabled: bool = False`: Cờ kích hoạt phân đoạn chuyên sâu.
  - `deepDiarizationEngine: str = "pyannote"`: Tùy chọn động cơ (`"pyannote"` hoặc `"nemotron"`).
  - `deepDiarizationModelDir: Optional[str] = None`: Đường dẫn thư mục chứa trọng số mô hình đã tải về.
  - `huggingFaceToken: str = ""`: Mã khóa định danh Hugging Face dùng để xác thực quyền tải mô hình.

### 3.5. Tích hợp Giao diện Người dùng
- **Vị trí tệp:** `src/desktop/ui/omnivoice_interface.py`
- **Mục chọn phần cứng (dòng 2542):**
  - Đổi chuỗi hiển thị thành: "Phân đoạn người nói chuyên sâu — pyannote" (hoặc hiển thị rõ trạng thái khả dụng).
- **Xử lý khi người dùng chọn chế độ chuyên sâu (dòng 6961):**
  - Loại bỏ hộp thoại cảnh báo vô hiệu hóa cũ.
  - Kiểm tra xem thư viện và card đồ họa đã sẵn sàng chưa:
    - Nếu chưa có card đồ họa CUDA: Hiển thị thanh thông báo khuyến nghị chạy trên card đồ họa để đảm bảo tốc độ.
    - Nếu thiếu mã khóa định danh hoặc mô hình chưa tải: Hiển thị thanh thông báo hướng dẫn người dùng cung cấp mã khóa Hugging Face trong bảng Cài đặt nâng cao.
    - Nếu đủ điều kiện: Bật cờ `deepDiarizationEnabled = True` và tiến hành quy trình nhận diện.

---

## 4. Quản lý Bộ nhớ Đồ họa và An toàn Phần cứng

- **Phần cứng mục tiêu:** Card đồ họa NVIDIA GeForce RTX 4050 Laptop GPU (Tổng bộ nhớ đồ họa: 6144 MB).
- **Ngân sách phân bổ:**
  - Mức nền hệ thống và hiển thị giao diện: ~1,5 - 2,5 GB.
  - Bộ nhớ khả dụng cho tác vụ trí tuệ nhân tạo: ~3,5 - 4,0 GB.
  - Mô hình CAM++ ONNX: ~150 - 300 MB.
  - Mô hình pyannote 3.1: ~1,2 - 1,5 GB trong thời gian xử lý phân đoạn.
  - Mô hình OmniVoice TTS: ~2,5 - 3,5 GB trong thời gian tổng hợp giọng nói.
- **Quy tắc thực thi tuần tự bất biến:**
  - Tuyệt đối không nạp song song pyannote và động cơ TTS.
  - Ngay sau khi hàm `verify_or_fallback()` hoàn tất:
    ```python
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ```
  - Kiểm tra bảo đảm bộ nhớ đồ họa được giải phóng về trạng thái nghỉ trước khi nạp động cơ giọng đọc.

---

## 5. Kế hoạch Kiểm thử và Tiêu chuẩn Nghiệm thu

### 5.1. Tạo Dữ liệu Đo đếm Baseline (Giai đoạn 0)
- Trích xuất đoạn video chuẩn 30 giây: `test/fixtures/cheems_30s.mp4` (chứa các đoạn đối thoại giữa 3 nhân vật Sếp, Lão Ngũ, Bác tài).
- Tạo tệp phụ đề và nhãn chuẩn tương ứng: `test/fixtures/cheems_30s.srt` và `test/fixtures/cheems_30s_ground_truth.json`.
- Đo đếm và ghi nhận 4 chỉ số đường chuẩn của thuật toán CAM++ hiện tại:
  1. Số nhân vật nhận diện được.
  2. Tỷ lệ phần trăm câu phụ đề được gán đúng người nói so với nhãn chuẩn.
  3. Thời gian thực thi toàn bộ quy trình.
  4. Mức tiêu hao bộ nhớ đồ họa đỉnh đo bằng công cụ giám sát phần cứng.

### 5.2. Tiêu chuẩn Nghiệm thu Kỹ thuật (Giai đoạn 1)
- **Độ chính xác phân đoạn:** Nhận diện đúng 6 trên 6 nhân vật trên tập video Cheems đầy đủ; tỷ lệ câu phụ đề gán đúng người nói đạt tối thiểu 90%.
- **Thời gian xử lý:** Tổng thời gian thực thi trên card đồ họa không vượt quá 2 lần so với thuật toán CAM++ cũ trên cùng một video.
- **Ngưỡng an toàn bộ nhớ:** Mức sử dụng bộ nhớ đồ họa đỉnh không vượt quá 5,5 GB trong suốt quá trình chạy.
- **Kiểm thử hồi quy:** Khi tắt tùy chọn chuyên sâu, hệ thống hoạt động chính xác theo luồng CAM++ cũ; toàn bộ 91 bài kiểm thử tự động của OmniVoice duy trì trạng thái thành công.
- **Khả năng tự phục hồi:** Khi giả lập mất kết nối mạng hoặc mã khóa không hợp lệ, hệ thống tự động ghi nhật ký và quay về phân đoạn CAM++ mà không làm dừng ứng dụng.
