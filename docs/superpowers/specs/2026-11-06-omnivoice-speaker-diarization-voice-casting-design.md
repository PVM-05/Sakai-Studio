# Thiết kế Kiến trúc Nhận diện Người nói và Tự động Phân bổ Giọng Trí tuệ Nhân tạo cho OmniVoice

Tài liệu này xác định kiến trúc chi tiết, giải thuật và các quy chuẩn kỹ thuật cho hệ thống nhận diện người nói và tự động phân vai giọng đọc tiếng Việt trong ứng dụng Sakai Studio, tối ưu hóa cho cấu hình phần cứng có bộ nhớ đồ họa 6GB.

---

## 1. Mục tiêu và Nguyên tắc Cốt lõi

1. **Hai nhiệm vụ trung tâm:**
   - Nhận diện chính xác ai đang nói từ âm thanh video và văn bản phụ đề.
   - Tự động gán đúng giọng đọc trí tuệ nhân tạo phù hợp cho từng nhân vật, đảm bảo các nhân vật khác nhau không bị trùng giọng.
2. **Khóa giọng bất biến:**
   - Mỗi nhân vật khi đã được gán giọng sẽ giữ nguyên một giọng duy nhất cho toàn bộ các câu thoại xuyên suốt video.
   - Tuyệt đối không chọn lại giọng cho từng câu phụ đề riêng lẻ.
   - Tuyệt đối không thay đổi giọng khi một câu có cường độ, cảm xúc hoặc âm lượng khác biệt.
   - Chỉ thay đổi giọng khi người dùng chủ động điều chỉnh trên giao diện.
3. **An toàn bộ nhớ đồ họa (RTX 4050 6GB VRAM):**
   - Không đồng thời nạp nhiều mô hình nặng vào bộ nhớ đồ họa.
   - Động cơ nhận diện cốt lõi chạy hoàn toàn qua môi trường ONNX Runtime siêu nhẹ (chiếm dưới 300 MB bộ nhớ đồ họa hoặc vận hành mượt mà trên CPU).
   - Tách biệt hoàn toàn luồng nhận diện nhân vật và luồng tổng hợp giọng đọc OmniVoice.

---

## 2. Kiến trúc Luồng Dữ liệu Toàn diện

```
[Video gốc + Phụ đề SRT]
       │
       ▼
[1. Trích xuất & Tiền xử lý Âm thanh]
       ├─ FFmpeg: Trích xuất WAV 24 kHz (mẫu nhân bản) và 16 kHz (nhận diện)
       └─ Silero VAD (ONNX): Phân đoạn giọng nói & gọt khoảng lặng
       │
       ▼
[2. Nhận diện Người nói Đa tầng]
       ├─ Tầng 1 (Chính): CAM++ ONNX + Biểu diễn Đa Đoạn thoại + AHC Phân cấp
       └─ Tầng 2 (Đối chiếu): Cổng NeMo Sortformer / Pyannote (chạy có điều kiện)
       │
       ▼
[3. Đồng bộ Mốc thời gian Phụ đề SRT]
       ├─ Ghép nhãn nhân vật theo độ dài phủ lớn nhất giữa VAD và SRT
       ├─ Giữ nguyên 100% nội dung, thứ tự câu và mốc thời gian
       └─ Bảo vệ các câu thoại đã được người dùng khóa thủ công
       │
       ▼
[4. Phân tích Hồ sơ Âm học Nhân vật]
       ├─ Giới tính cảm nhận (25%)
       ├─ Âm sắc (25%)
       ├─ Cao độ F0 (20%)
       ├─ Phương ngữ tiếng Việt (15%)
       └─ Tốc độ nói (15%)
       │
       ▼
[5. Tự động Phân bổ Giọng AI Toàn cục]
       ├─ Lọc kho giọng riêng biệt tinh chọn (14 giọng chuẩn)
       ├─ Tách nhóm độc lập Nam / Nữ (chống hoàn toàn gán nhầm giới tính)
       ├─ Thuật toán Hungarian (linear_sum_assignment) tối ưu chi phí
       └─ Cảnh báo review_needed khi số nhân vật vượt số giọng khả dụng
       │
       ▼
[6. Khóa Mỏ neo Âm sắc & Xuất bản]
       ├─ Tạo mỏ neo riêng biệt có kiểm định khoảng cách âm học CAM++
       ├─ Cố định hạt ngẫu nhiên đồng nhất
       └─ Xuất bản video lồng tiếng nhất quán tuyệt đối
```

---

## 3. Chi tiết Giải thuật và Xử lý 3 Vấn đề Then chốt

### Vấn đề 1: Một người thay đổi cảm xúc (hét, thì thầm, cười)
- **Giải pháp:** Sử dụng cơ chế **Biểu diễn Đa Mẫu Nòng cốt (Multi-prototype Representation)**.
- Thay vì chỉ tính trung bình một vector đặc trưng duy nhất cho nhân vật (dễ bị kéo lệch khi có câu thoại bất thường), hệ thống lưu lại các vector đại diện cho từng dải năng lượng:
  - Mẫu giọng bình thường (âm lượng trung bình, độ dài chuẩn).
  - Mẫu giọng năng lượng cao (khi cười to, hét, cao độ tăng cao).
  - Mẫu giọng năng lượng thấp (khi thì thầm, nói nhỏ).
- Khi phân cụm và gán nhãn, độ tương đồng được tính theo khoảng cách ngắn nhất đến các mẫu nòng cốt của cụm, ngăn ngừa việc tách một người thành nhiều nhân vật chỉ vì biểu cảm thay đổi.

### Vấn đề 2: Hai người có chất giọng gần giống nhau
- **Giải pháp:** Đánh giá phối hợp đa chiều giữa vector đặc trưng CAM++, đường bao tần số F0 và phân bố năng lượng dải tần:
  - Nếu khoảng cách cosin nằm trong vùng nghi vấn ($0,28 \le \text{dist} \le 0,40$), hệ thống phân tích độ lệch tần số cơ bản trung vị $\Delta F_0$.
  - Nếu $\Delta F_0 > 75\text{ Hz}$, áp dụng hình phạt khoảng cách $+0,25$, buộc thuật toán gom cụm phải phân tách thành hai người nói riêng biệt.
  - Phân tích tương quan trên toàn bộ chuỗi các câu thoại của nhân vật thay vì chỉ so sánh từng cặp câu ngắn riêng lẻ.

### Vấn đề 3: Thiếu dữ liệu nhận diện (câu quá ngắn hoặc tạp âm)
- **Giải pháp:** Áp dụng hệ thống phân loại 3 mức độ tin cậy:
  - `high`: Độ dài câu $\ge 0,5\text{s}$, biên độ tín hiệu đạt chuẩn, khoảng cách tới tâm cụm nhỏ.
  - `review_needed`: Câu có độ dài từ $0,35\text{s}$ đến $0,5\text{s}$ hoặc độ tương đồng biên giới.
  - `unassigned`: Câu dưới $0,35\text{s}$ hoặc năng lượng tín hiệu dưới $1\times 10^{-4}$.
- Tuyệt đối không tự tiện sinh nhân vật mới cho câu thiếu dữ liệu. Hệ thống gán kế thừa từ nhân vật của câu nói liền kề trước hoặc sau (nếu khoảng cách thời gian $\le 1,5\text{s}$) và gắn nhãn cảnh báo để người dùng chủ động kiểm tra.

---

## 4. Công thức Chấm điểm 5 Tiêu chí Phân bổ Giọng

Hệ thống đánh giá độ phù hợp của từng giọng đọc trí tuệ nhân tạo đối với hồ sơ âm học của nhân vật theo công thức:

$$\text{Tổng điểm} = 0,25 \times S_{\text{giới tính}} + 0,25 \times S_{\text{âm sắc}} + 0,20 \times S_{\text{cao độ}} + 0,15 \times S_{\text{phương ngữ}} + 0,15 \times S_{\text{tốc độ}}$$

Trong đó:
1. **$S_{\text{giới tính}}$ (25%):** Khớp tuyệt đối theo giới tính cảm nhận. Cấm gán chéo Nam sang Nữ hoặc ngược lại.
2. **$S_{\text{âm sắc}}$ (25%):** Đo lường sự tương thích về trọng tâm phổ tần số và độ ấm/sáng của giọng nói.
3. **$S_{\text{cao độ}}$ (20%):** So khớp tần số cơ bản trung vị của nhân vật với mức cao độ của giọng mẫu (rất trầm, trầm, trung bình, cao, rất cao).
4. **$S_{\text{phương ngữ}}$ (15%):** Nhận diện phương ngữ Bắc, Trung, Nam từ từ vựng trong văn bản phụ đề của nhân vật để ưu tiên giọng đọc cùng vùng miền.
5. **$S_{\text{tốc độ}}$ (15%):** So sánh tốc độ phát âm thực tế của nhân vật (số âm tiết trên giây) với tốc độ đọc tự nhiên của giọng mẫu để đảm bảo không bị đọc quá nhanh hoặc quá chậm so với video gốc.

---

## 5. Kế hoạch Triển khai Mã nguồn

1. **Tệp `src/ai_engines/speaker_diarization.py`:**
   - Bổ sung lớp `MultiPrototypeSpeakerProfile` hỗ trợ biểu diễn đa đoạn thoại.
   - Nâng cấp hàm `extract_chunk_f0` và hàm đo tốc độ nói (âm tiết / giây).
   - Thiết lập giao diện trừu tượng `ExternalDiarizationVerifier` phục vụ kiểm chứng chéo khi có NeMo hoặc Pyannote.
2. **Tệp `src/ai_engines/tts_engine.py`:**
   - Cập nhật hàm `calculate_voice_match` theo chuẩn 5 tiêu chí với đúng trọng số (25%, 25%, 20%, 15%, 15%).
   - Cập nhật `GlobalVoiceAllocator` đảm bảo phân bổ Hungarian độc lập theo giới tính, khóa vĩnh viễn mỏ neo cho từng nhân vật.
3. **Tệp `src/desktop/ui/omnivoice_interface.py`:**
   - Hiển thị đầy đủ thông tin hồ sơ âm học (giới tính, cao độ, tốc độ nói, độ tin cậy) trên bảng phân vai.
   - Thêm bộ lọc và cảnh báo người dùng khi có câu thoại cần kiểm tra.
4. **Tệp kiểm thử `test/test_omnivoice_improvements.py` và `test/test_omnivoice_pipeline.py`:**
   - Bổ sung kịch bản kiểm thử cho biểu diễn đa mẫu nòng cốt, bộ chấm điểm 5 tiêu chí, kiểm định khóa giọng bất biến và cách ly chéo giới tính.
