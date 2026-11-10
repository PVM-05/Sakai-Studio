# Thiết kế Chi tiết Phân đoạn Người nói Chuyên sâu với pyannote và Dự phòng CAM++

Tài liệu thiết kế kỹ thuật cho Giai đoạn 0 và Giai đoạn 1 thuộc kế hoạch nâng cấp năng lực phân tích giọng nói và phân đoạn người nói trong Sakai Studio, tối ưu hóa cho cấu hình phần cứng có bộ nhớ đồ họa 6 GB (NVIDIA GeForce RTX 4050 Laptop GPU).

---

## 1. Bối cảnh và Mục tiêu

### 1.1. Hiện trạng
Hệ thống hiện tại phân tích người nói qua mô hình CAM++ ONNX với vector đặc trưng 192 chiều kết hợp thuật toán phân cụm AHC và đánh giá số lượng cụm bằng chỉ số silhouette. Lớp kiểm chứng đối chiếu `ExternalDiarizationVerifier` tại `src/ai_engines/speaker_diarization.py` hiện chỉ là khung giữ chỗ và luôn trả về nguyên vẹn kết quả từ CAM++.

Thực tế kiểm tra và đo đạc trên mã nguồn:
- Các số liệu phân cụm silhouette K=2 (0,328) và K=6 (0,216) từng được ghi nhận trong tài liệu phân tích là kết quả của thí nghiệm trích xuất MFCC kết hợp F0 bằng thư viện ngoài, không phải số liệu đo đạc trực tiếp từ mô hình CAM++ của Sakai Studio. Đường chuẩn hiệu năng và độ chính xác thực tế của CAM++ trên video Cheems 6 nhân vật chưa từng được đo lường chính thức.
- Bộ phân đoạn giọng nói dựa trên năng lượng âm thanh hiện tại không hỗ trợ phát hiện các phân đoạn có nhiều người cùng nói đồng thời.
- Tùy chọn "Phân đoạn người nói chuyên sâu" trên giao diện người dùng tại `src/desktop/ui/omnivoice_interface.py` chưa được kích hoạt và đang hiển thị thông báo yêu cầu chọn cấu hình phần cứng khác.

### 1.2. Mục tiêu kỹ thuật
1. Hiện thực hóa động cơ phân đoạn người nói chuyên sâu sử dụng thư viện `pyannote.audio==3.1.1` nối trực tiếp vào `ExternalDiarizationVerifier`.
2. Đảm bảo tính tương thích ngược tuyệt đối: CAM++ ONNX tiếp tục là động cơ cơ sở nhanh và là phương án dự phòng tự động khi tắt tùy chọn chuyên sâu, khi thiếu môi trường phần cứng tăng tốc hoặc khi quá trình nạp mô hình phát sinh lỗi.
3. Đo lường chính thức đường chuẩn CAM++ trên video Cheems 589 giây trong Giai đoạn 0 làm căn cứ so sánh kỹ thuật.
4. Xây dựng thuật toán ánh xạ danh tính tối ưu từ các nhãn ẩn danh của pyannote sang hồ sơ nhân vật của CAM++, tránh tình trạng mất dấu mẫu giọng và gán trùng lặp.
5. Kiểm soát chặt chẽ ngân sách bộ nhớ đồ họa: Đo đạc thực tế mức tiêu thụ bộ nhớ và giải phóng hoàn toàn bộ nhớ đồ họa sau khi hoàn tất phân đoạn để bảo toàn trọn vẹn tài nguyên cho động cơ tổng hợp giọng đọc OmniVoice.

---

## 2. Kiến trúc Hệ thống và Luồng Dữ liệu

```
[Video Đầu Vào + Tệp Phụ Đề SRT]
                │
                ▼
[1. Trích xuất Âm thanh & Tiền xử lý]
  - Trích xuất âm thanh WAV chất lượng cao (mặc định 24 kHz từ luồng hiện tại)
  - pyannote tự động lấy mẫu nội bộ về 16 kHz phục vụ phân đoạn
  - Bộ phát hiện phân đoạn âm thanh cơ sở
                │
                ▼
[2. Phân đoạn Người nói Cơ sở (CAM++ ONNX)]
  - Trích xuất đặc trưng âm học 192 chiều
  - Phân cụm AHC cosin và xác định các cụm sơ bộ
  - Tạo danh sách hồ sơ nhân vật kèm mẫu âm thanh cắt đoạn
                │
                ▼
[3. Bộ Kiểm chứng Đối chiếu (ExternalDiarizationVerifier)]
  - Đọc cấu hình deep_diarization_enabled
         │
         ├─ [Trường hợp TẮT hoặc Lỗi / Chưa chấp thuận bản quyền / Thiếu GPU]:
         │      └─ Hoàn trả nguyên vẹn kết quả CAM++ (Dự phòng an toàn)
         │
         └─ [Trường hợp BẬT (pyannote.audio 3.1.1)]:
                ├─ Kiểm tra khóa xác thực Hugging Face (biến môi trường HF_TOKEN)
                ├─ Nạp pipeline pyannote/speaker-diarization-3.1 lên card đồ họa
                ├─ Xử lý âm thanh dài qua cơ chế phân đoạn cửa sổ trượt nội bộ
                ├─ Thu hồi bộ nhớ đồ họa: synchronize + empty_cache + gc.collect
                ├─ Ánh xạ danh tính Hungarian giữa nhãn pyannote và hồ sơ CAM++
                ├─ Gán nhãn câu phụ đề theo tỷ lệ bao phủ thời gian
                └─ Cổng kiểm soát chất lượng: Cảnh báo và dự phòng nếu độ đồng thuận thấp
                │
                ▼
[4. Đồng bộ Cập nhật Khối Phụ đề SRT & Hồ sơ Nhân vật]
                │
                ▼
[5. Thu hồi Tài nguyên Toàn diện & Chuyển sang Bước Tổng hợp Giọng]
  - Sẵn sàng toàn bộ bộ nhớ đồ họa cho động cơ OmniVoice TTS
```

---

## 3. Thiết kế Chi tiết Từng Thành phần

### 3.1. Nâng cấp Lớp ExternalDiarizationVerifier
- **Vị trí tệp:** `src/ai_engines/speaker_diarization.py`
- **Chữ ký phương thức chuẩn hóa:**
  ```python
  @classmethod
  def verify_or_fallback(
      cls,
      audio_path: str,
      campp_speakers: Dict[str, Any],
      min_speakers: Optional[int] = None,
      max_speakers: Optional[int] = None,
      exact_speakers: Optional[int] = None,
  ) -> Dict[str, Any]:
  ```
  Điểm gọi tại dòng 1638 của `speaker_diarization.py` sẽ được cập nhật đồng bộ để truyền thêm tham số `exact_speakers`.

- **Phương thức `is_available()`:**
  - Kiểm tra khả năng import thư viện `pyannote.audio`.
  - Kiểm tra tính sẵn sàng của card đồ họa CUDA qua lệnh `torch.cuda.is_available()`.
  - Trả về giá trị đúng khi cả hai điều kiện trên được đáp ứng.

- **Quy tắc điều phối tham số số người nói:**
  - Khi người dùng ấn định số lượng người nói cụ thể (`exact_speakers` có giá trị): Truyền tham số `num_speakers = exact_speakers` trực tiếp vào pipeline pyannote.
  - Khi không ấn định số lượng cụ thể: Truyền khoảng dao động `min_speakers` và `max_speakers` tương ứng với thiết lập hệ thống.

- **Xử lý nạp mô hình và thu hồi bộ nhớ an toàn:**
  - Phiên bản thư viện quy định: Khóa cố định `pyannote.audio==3.1.1` trong tài liệu môi trường.
  - Kiểm tra kết quả trả về từ hàm nạp mô hình:
    ```python
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        cache_dir=model_dir,
        use_auth_token=hf_token,
    )
    if pipeline is None:
        logger.warning("Không thể nạp mô hình pyannote do thiếu quyền hoặc chưa chấp thuận điều khoản. Chuyển sang sử dụng CAM++.")
        return campp_speakers
    ```
  - Chuyển mô hình sang thiết bị card đồ họa CUDA: `pipeline.to(torch.device("cuda"))`.
  - Thực thi phân đoạn âm thanh trên tệp âm thanh đầu vào.
  - Thu hồi triệt để bộ nhớ đồ họa ngay sau khi hoàn thành suy luận:
    ```python
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    ```
  - Khối bọc ngoại lệ toàn diện: Bắt mọi ngoại lệ phát sinh trong quá trình thực thi, ghi nhật ký lỗi chi tiết và tự động hoàn trả kết quả CAM++ ban đầu mà không làm gián đoạn chương trình.

### 3.2. Thuật toán Ánh xạ Danh tính và Gán Nhãn Phụ đề

#### A. Thuật toán Ánh xạ Danh tính từ pyannote sang CAM++
Mô hình pyannote trả về các nhãn ẩn danh có dạng `SPEAKER_00`, `SPEAKER_01` theo thứ tự xuất hiện trong âm thanh. Để không phá vỡ liên kết với hồ sơ nhân vật `SpeakerProfile` của CAM++ (bao gồm đoạn mẫu âm thanh, tham số giọng đọc và các đánh giá chất lượng đã thiết lập), hệ thống áp dụng thuật toán ánh xạ tối ưu:
1. Trích xuất danh sách các phân đoạn phát âm từ kết quả pyannote thông qua phương thức lặp luồng nhãn.
2. Với mỗi cặp nhãn ẩn danh pyannote và hồ sơ CAM++, xây dựng ma trận trọng số tương đồng dựa trên:
   - Tổng thời lượng giao nhau giữa các câu phụ đề của cụm CAM++ và các phân đoạn phát âm của nhãn pyannote.
   - Độ tương đồng cosin trung bình giữa vector đặc trưng CAM++ của các câu phụ đề và tâm cụm tương ứng.
3. Áp dụng giải thuật Hungarian thông qua hàm `linear_sum_assignment` từ thư viện `scipy.optimize` để tìm cặp ánh xạ có độ tương đồng tổng thể lớn nhất.
4. Trường hợp pyannote nhận diện số lượng người nói nhiều hơn số cụm CAM++, các nhãn phát sinh mới sẽ được khởi tạo hồ sơ nhân vật bổ sung.

#### B. Công thức Gán Nhãn Câu Phụ đề
Để tránh mâu thuẫn giữa các định nghĩa hình học, hệ thống quy định phân định rành mạch hai công thức:
1. **Công thức gán nhãn câu phụ đề (Tỷ lệ bao phủ thời gian):**
   $$\text{Tỷ lệ bao phủ} = \frac{\text{Thời lượng giao nhau giữa phân đoạn pyannote và câu phụ đề}}{\text{Thời lượng của câu phụ đề SRT}}$$
   - Khi tỷ lệ bao phủ đạt từ 0,5 trở lên: Gán câu phụ đề theo nhãn người nói tương ứng từ pyannote.
   - Nếu nhãn sau khi ánh xạ trùng khớp với nhãn ban đầu của CAM++: Giữ nguyên trạng thái và đánh dấu câu thoại có độ tin cậy cao.
   - Nếu nhãn sau khi ánh xạ khác biệt với CAM++: Cập nhật nhãn nhân vật mới và đánh dấu cờ cần rà soát `review_needed = True`.
   - Nếu phát hiện nhiều người nói trong cùng một câu phụ đề với tỷ lệ xấp xỉ nhau: Gán nhãn cho người nói có thời lượng lớn nhất, đồng thời lưu cảnh báo có hiện tượng nói chồng vào dữ liệu mở rộng của câu phụ đề.
2. **Chỉ số đo lường mức độ đồng thuận (IoU):**
   $$\text{IoU} = \frac{\text{Thời lượng giao nhau}}{\text{Thời lượng hợp nhất}}$$
   Chỉ số này chỉ được sử dụng trong bước thống kê và lập báo cáo đánh giá chất lượng phân đoạn giữa hai mô hình, không dùng làm điều kiện trực tiếp để gán nhãn cho từng câu thoại.

#### C. Cổng Kiểm soát Chất lượng và Dự phòng Thích ứng
Nếu quá trình phân đoạn pyannote diễn ra thành công nhưng kết quả có dấu hiệu bất thường:
- Số lượng người nói pyannote phát hiện sai lệch quá 2 người so với ước lượng của CAM++.
- Hoặc chỉ số đồng thuận IoU trung bình giữa hai hệ thống thấp hơn ngưỡng 0,3.
Hệ thống sẽ ghi nhật ký cảnh báo và tự động giữ nguyên nhãn CAM++ cho các câu phụ đề có thời lượng phát âm ngắn hoặc chất lượng tín hiệu yếu, ngăn chặn việc làm sai lệch cấu trúc nhân vật đã phân bổ.

### 3.3. Định vị Mô hình NeMo và Giới hạn Môi trường Hệ thống
- **Làm rõ sự khác biệt công nghệ:**
  - Mô hình NeMo Sortformer 4 người nói cũ (`diar_streaming_sortformer_4spk-v2.1`) chỉ hỗ trợ tối đa 4 người nói đồng thời.
  - Mô hình NVIDIA Nemotron 3 Diarization phát hành tháng 9 năm 2026 sử dụng kiến trúc phân loại nhãn khung thời gian đa kênh và hỗ trợ tối đa 8 người nói.
- **Giới hạn môi trường thực thi:**
  - Bộ công cụ NeMo được thiết kế tối ưu trên hệ điều hành Linux và yêu cầu các bản dựng thư viện nhị phân chuyên biệt. Ứng dụng Sakai Studio vận hành chính thức trên môi trường Windows native.
  - Việc tích hợp NeMo trực tiếp vào bản cài đặt Windows tiềm ẩn rủi ro xung đột môi trường và khó phân phối cho người dùng cuối.
- **Quyết định kiến trúc:**
  - Tạm thời hoãn việc tích hợp mô hình NeMo trong Giai đoạn 1.
  - Toàn bộ năng lực phân đoạn chuyên sâu của Giai đoạn 1 tập trung hoàn thiện và tối ưu trên nền tảng `pyannote.audio 3.1.1`.
  - Khung giao diện thử nghiệm của NeMo sẽ được xem xét triển khai trong giai đoạn tiếp theo dưới dạng dịch vụ tách biệt hoặc môi trường phụ trợ tương thích.

### 3.4. Quản lý Cấu hình và Bảo mật Thông tin Xác thực
- **Cấu hình hệ thống tại `src/core/config.py`:**
  Định nghĩa các trường cấu hình mới theo quy chuẩn `snake_case` đồng nhất trong lớp cấu hình xử lý `ProcessingConfig`:
  - `deep_diarization_enabled: bool = False`: Cờ kích hoạt phân đoạn chuyên sâu.
  - `deep_diarization_engine: str = "pyannote"`: Tên động cơ phân đoạn chuyên sâu lựa chọn.
  - `deep_diarization_model_dir: Optional[str] = None`: Đường dẫn lưu trữ trọng số mô hình cục bộ.
- **Quy chuẩn bảo mật mã khóa Hugging Face:**
  - Tuyệt đối không lưu trữ mã khóa xác thực dưới dạng chuỗi văn bản thuần trong tệp cấu hình hoặc mã nguồn.
  - Quá trình nạp mô hình sẽ tự động tiếp nhận mã xác thực theo thứ tự ưu tiên:
    1. Biến môi trường hệ thống `HF_TOKEN`.
    2. Phiên làm việc đã xác thực cục bộ từ công cụ dòng lệnh `huggingface-cli login`.
    3. Kho lưu trữ thông tin xác thực an toàn của hệ điều hành.

### 3.5. Tích hợp Giao diện Người dùng
- **Vị trí tệp:** `src/desktop/ui/omnivoice_interface.py`
- **Mục chọn phần cứng (dòng 2542):**
  - Cập nhật hiển thị tùy chọn: "Phân đoạn người nói chuyên sâu — pyannote".
- **Xử lý sự kiện điều khiển (dòng 6961):**
  - Xóa bỏ thông báo cảnh báo từ chối thao tác cũ.
  - Kiểm tra điều kiện thực thi phần cứng và trạng thái nạp mô hình:
    - Nếu phát hiện hệ thống không có card đồ họa CUDA: Hiển thị thanh thông báo khuyến nghị chạy trên card đồ họa để tránh làm chậm tốc độ xử lý.
    - Nếu chưa cấu hình mã xác thực hoặc chưa chấp thuận điều khoản tải mô hình: Hiển thị thanh thông báo hướng dẫn người dùng thiết lập biến môi trường `HF_TOKEN` hoặc đăng nhập bằng công cụ dòng lệnh.
    - Khi các điều kiện hợp lệ: Lưu trạng thái cấu hình `deep_diarization_enabled = True` và kích hoạt quy trình phân đoạn nâng cao.

---

## 4. Quản lý Bộ nhớ Đồ họa và An toàn Phần cứng

- **Phần cứng mục tiêu:** Card đồ họa NVIDIA GeForce RTX 4050 Laptop GPU với tổng dung lượng bộ nhớ đồ họa 6144 MB.
- **Nguyên tắc đo đạc thực tế:**
  Mọi chỉ số tiêu hao bộ nhớ trước đây đều mang tính ước tính lý thuyết. Do đó, Giai đoạn 0 bắt buộc thực hiện đo đạc thực tế bằng các công cụ giám sát phần cứng (`nvidia-smi` và `torch.cuda.max_memory_allocated()`) để xác định chính xác:
  1. Mức sử dụng bộ nhớ đồ họa nền của hệ điều hành và giao diện hiển thị.
  2. Mức tiêu hao đỉnh của `pyannote.audio 3.1.1` khi nạp đầy đủ cả mô hình phân đoạn thời gian và mô hình trích xuất đặc trưng âm học.
  3. Mức tiêu hao đỉnh của động cơ tổng hợp giọng đọc OmniVoice.
- **Quy trình thu hồi bộ nhớ tuần tự bất biến:**
  - Không nạp đồng thời mô hình pyannote và mô hình tổng hợp giọng đọc vào bộ nhớ đồ họa trong cùng một thời điểm.
  - Ngay sau khi hoàn thành bước phân đoạn:
    ```python
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    ```
  - Đảm bảo toàn bộ dung lượng bộ nhớ đồ họa được thu hồi về mức an toàn trước khi chuyển giao quyền kiểm soát cho động cơ giọng đọc.

---

## 5. Kế hoạch Kiểm thử và Tiêu chuẩn Nghiệm thu

### 5.1. Thiết lập Dữ liệu Đo đếm và Chấp thuận Điều khoản Bản quyền (Giai đoạn 0)
1. **Chấp thuận điều khoản bản quyền trên Hugging Face:**
   - Thực hiện đăng nhập và bấm chấp thuận điều khoản người dùng đối với hai kho mô hình có bản quyền giới hạn:
     - `pyannote/speaker-diarization-3.1`
     - `pyannote/segmentation-3.0`
   - Xác thực việc tải trọng số thành công thông qua công cụ dòng lệnh trước khi chạy kiểm thử mã nguồn.
2. **Chuẩn bị dữ liệu kiểm thử rút gọn:**
   - Tạo đoạn âm thanh kiểm thử 30 giây: `test/fixtures/cheems_30s.wav` (lấy mẫu chuẩn 16 kHz, chỉ lưu trữ âm thanh để tối ưu dung lượng kho mã nguồn và tuân thủ bản quyền).
   - Xác định danh sách nhân vật thực tế xuất hiện trong đoạn trích 30 giây: Đoạn trích gồm 4 nhân vật (Sếp, Lão Ngũ, Bác tài và Cheems).
   - Thẩm định và chốt nhãn phân vai dứt khoát cho câu phụ đề số 124 trong tệp nhãn chuẩn `test/fixtures/cheems_30s_ground_truth.json` trước khi sử dụng làm mốc đo lường.
3. **Đo đạc chính thức đường chuẩn CAM++:**
   - Thực thi hàm `SpeakerDiarizationCoordinator.process_video_subtitles()` trên toàn bộ video Cheems 589 giây.
   - Ghi nhận đầy đủ 4 thông số đường chuẩn thực tế:
     - Số lượng nhân vật thuật toán tự động nhận diện.
     - Tỷ lệ phần trăm câu phụ đề được gán đúng nhân vật so với tệp thẩm định chuẩn `assignment.json`.
     - Tổng thời gian thực thi toàn bộ quy trình trên card đồ họa.
     - Mức tiêu hao bộ nhớ đồ họa đỉnh đo bằng công cụ giám sát.

### 5.2. Tiêu chuẩn Nghiệm thu Kỹ thuật (Giai đoạn 1)
- **Độ chính xác nhận diện nhân vật:** Nhận diện chính xác 6 trên 6 nhân vật trên toàn bộ video Cheems 589 giây; tỷ lệ câu phụ đề được gán đúng người nói đạt từ 90% trở lên.
- **Thời gian xử lý toàn trình:** Tổng thời gian thực thi của pyannote trên card đồ họa không vượt quá 2 lần so với thời gian đường chuẩn của CAM++ đã đo đạc trên cùng một video.
- **Ngưỡng an toàn bộ nhớ:** Mức sử dụng bộ nhớ đồ họa đỉnh không vượt quá 5,5 GB trong suốt quá trình chạy.
- **Độ tin cậy của cơ chế dự phòng:**
  - Khi tắt tùy chọn phân đoạn chuyên sâu, kết quả đầu ra đồng nhất hoàn toàn với thuật toán CAM++ ban đầu.
  - Khi giả lập môi trường thiếu mạng hoặc token không hợp lệ, hệ thống tự động ghi nhật ký cảnh báo và kích hoạt dự phòng CAM++ mà không phát sinh lỗi ngoại lệ làm dừng ứng dụng.
- **Kiểm thử tích hợp mới:**
  Xây dựng tệp kiểm thử tự động `test/test_diarization_verifier_integration.py` bao gồm:
  - Kiểm thử phương thức `verify_or_fallback` với đối tượng chú thích phân đoạn mô phỏng.
  - Kiểm thử trường hợp hàm nạp trả về giá trị rỗng (`pipeline is None`) kích hoạt dự phòng an toàn.
  - Kiểm thử tính chính xác của thuật toán ghép cặp danh tính Hungarian và công thức tỷ lệ bao phủ câu thoại.
  - Kiểm thử xử lý ngoại lệ khi mất kết nối mạng.
