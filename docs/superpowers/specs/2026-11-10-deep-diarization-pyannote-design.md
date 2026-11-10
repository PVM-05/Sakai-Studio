# Thiết kế Chi tiết Phân đoạn Người nói Chuyên sâu với pyannote và Dự phòng CAM++ (Bản Hoàn thiện v3)

Tài liệu thiết kế kỹ thuật cho Giai đoạn 0 và Giai đoạn 1 thuộc kế hoạch nâng cấp năng lực phân tích giọng nói và phân đoạn người nói trong Sakai Studio, tối ưu hóa cho cấu hình phần cứng có bộ nhớ đồ họa 6 GB (NVIDIA GeForce RTX 4050 Laptop GPU) và hỗ trợ vận hành dự phòng trên CPU.

---

## 1. Bối cảnh và Mục tiêu

### 1.1. Hiện trạng
Hệ thống hiện tại phân tích người nói qua mô hình CAM++ ONNX với vector đặc trưng 192 chiều kết hợp thuật toán phân cụm AHC và đánh giá số lượng cụm bằng chỉ số silhouette. Lớp kiểm chứng đối chiếu `ExternalDiarizationVerifier` tại `src/ai_engines/speaker_diarization.py` hiện chỉ là khung giữ chỗ và luôn trả về nguyên vẹn kết quả từ CAM++.

Thực tế kiểm tra và đo đạc trên mã nguồn:
- Các số liệu phân cụm silhouette K=2 (0,328) và K=6 (0,216) từng được ghi nhận trong tài liệu phân tích trước đây là kết quả của thí nghiệm trích xuất MFCC kết hợp F0 bằng thư viện ngoài, không phải số liệu đo đạc trực tiếp từ mô hình CAM++ của Sakai Studio. Đường chuẩn hiệu năng và độ chính xác thực tế của CAM++ trên video Cheems 6 nhân vật chưa từng được đo lường chính thức.
- Bộ phân đoạn giọng nói dựa trên năng lượng âm thanh hiện tại không hỗ trợ phát hiện các phân đoạn có nhiều người cùng nói đồng thời.
- Tùy chọn "Phân đoạn người nói chuyên sâu" trên giao diện người dùng tại `src/desktop/ui/omnivoice_interface.py` (dòng 2542 và 6959–6961) chưa được kích hoạt và đang hiển thị thông báo yêu cầu chọn cấu hình phần cứng khác.
- Thư viện `scipy` đang được sử dụng ở nhiều vị trí trong mã nguồn (`linear_sum_assignment`, `linkage`, `fcluster`) nhưng chưa được khai báo tường minh trong `requirements.txt`.

### 1.2. Mục tiêu kỹ thuật
1. Hiện thực hóa động cơ phân đoạn người nói chuyên sâu sử dụng thư viện `pyannote.audio==3.1.1` nối trực tiếp vào `ExternalDiarizationVerifier`.
2. Đảm bảo tính tương thích ngược tuyệt đối: CAM++ ONNX tiếp tục là động cơ cơ sở nhanh và là phương án dự phòng tự động khi tắt tùy chọn chuyên sâu, khi môi trường thiếu thư viện, thiếu mã xác thực, hoặc khi nạp mô hình thất bại.
3. Đo lường chính thức đường chuẩn CAM++ trên video Cheems 589 giây trong Giai đoạn 0 làm căn cứ so sánh kỹ thuật.
4. Xây dựng thuật toán ánh xạ danh tính Hungarian hoàn chỉnh kết hợp thời gian giao nhau và độ tương đồng cosin đặc trưng âm học, bảo toàn hồ sơ `SpeakerProfile` của CAM++.
5. Kiểm soát chất lượng nhị phân rõ ràng: Tự động hoàn trả toàn bộ kết quả CAM++ khi phát hiện độ sai lệch vượt ngưỡng an toàn.
6. Quản lý chặt chẽ ngân sách bộ nhớ đồ họa: Đo đạc thực tế mức tiêu thụ bộ nhớ và giải phóng hoàn toàn bộ nhớ đồ họa sau khi hoàn tất phân đoạn để bảo toàn trọn vẹn tài nguyên cho động cơ tổng hợp giọng đọc OmniVoice.

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
  - Trích xuất đặc trưng âm học 192 chiều cho từng câu phụ đề
  - Phân cụm AHC cosin và xác định các cụm sơ bộ
  - Tạo danh sách hồ sơ nhân vật kèm mẫu âm thanh cắt đoạn
                │
                ▼
[3. Bộ Kiểm chứng Đối chiếu (ExternalDiarizationVerifier)]
  - Đọc cấu hình app_config.mmoTTSDeepDiarization
         │
         ├─ [Trường hợp TẮT hoặc Lỗi / Chưa chấp thuận bản quyền / Thiếu token]:
         │      └─ Hoàn trả nguyên vẹn kết quả CAM++ (Dự phòng an toàn)
         │
         └─ [Trường hợp BẬT (pyannote.audio 3.1.1)]:
                ├─ Phân giải mã xác thực: HF_TOKEN -> huggingface-cli -> fallback
                ├─ Nạp pipeline pyannote/speaker-diarization-3.1 lên CUDA (hoặc CPU)
                ├─ Xử lý âm thanh dài qua cơ chế phân đoạn cửa sổ trượt nội bộ
                ├─ Thu hồi bộ nhớ: synchronize + empty_cache + gc.collect
                ├─ Thuật toán Hungarian ghép cặp danh tính (trọng số 0,6 thời gian + 0,4 cosin)
                ├─ Gán nhãn câu phụ đề theo tỷ lệ bao phủ (ngưỡng 0,5 kèm nhánh else rõ ràng)
                └─ Cổng kiểm soát chất lượng nhị phân (IoU khung thời gian < 0,35 -> giữ CAM++)
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
  - Trả về `True` nếu thư viện có sẵn trong môi trường Python, không bắt buộc cứng phần cứng CUDA (cho phép chạy trên CPU kèm cảnh báo tốc độ).

- **Quy tắc điều phối tham số số người nói:**
  - Khi người dùng ấn định số lượng người nói cụ thể (`exact_speakers` có giá trị hợp lệ): Truyền tham số `num_speakers = exact_speakers` trực tiếp vào pipeline pyannote.
  - Khi không ấn định số lượng cụ thể: Truyền khoảng dao động `min_speakers` và `max_speakers` tương ứng với thiết lập hệ thống.

- **Quản lý phiên bản thư viện và cấu hình nạp:**
  - Khai báo bổ sung trong `requirements.txt`:
    ```txt
    scipy>=1.11.0
    pyannote.audio==3.1.1
    ```
  - Phân giải mã khóa xác thực Hugging Face:
    1. Kiểm tra biến môi trường `HF_TOKEN`. Nếu tồn tại chuỗi ký tự hợp lệ, sử dụng mã này.
    2. Nếu không có biến môi trường, truyền `use_auth_token=True` để thư viện `huggingface_hub` tự động tiếp nhận phiên đăng nhập từ công cụ `huggingface-cli login`.
    3. Nếu cả hai phương thức đều không có thông tin xác thực, ghi nhật ký cảnh báo và tự động hoàn trả kết quả CAM++.
  - Xác định thư mục lưu trữ trọng số mô hình:
    - Nếu cấu hình `app_config.mmoTTSDeepDiarizationModelDir` có đường dẫn cụ thể, truyền đường dẫn đó vào tham số `cache_dir`.
    - Nếu giá trị là chuỗi rỗng `""` hoặc `None`, truyền `cache_dir=None` để thư viện lưu trữ vào thư mục mặc định của hệ thống (`~/.cache/torch/pyannote`).
  - Kiểm tra kết quả trả về từ hàm nạp mô hình:
    ```python
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        cache_dir=cache_dir,
        use_auth_token=auth_token,
    )
    if pipeline is None:
        logger.warning("Không thể nạp mô hình pyannote do thiếu quyền hoặc chưa chấp thuận điều khoản. Tự động chuyển sang sử dụng CAM++.")
        return campp_speakers
    ```
  - Lựa chọn thiết bị thực thi:
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    ```
    Nếu chạy trên thiết bị CPU, ghi nhật ký thông báo về tốc độ xử lý để người dùng nắm được thông tin.
  - Thu hồi triệt để bộ nhớ đồ họa ngay sau khi hoàn thành suy luận:
    ```python
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    ```
  - Khối bọc ngoại lệ toàn diện: Bắt mọi ngoại lệ phát sinh trong quá trình nạp và thực thi, ghi nhật ký lỗi chi tiết và tự động hoàn trả kết quả CAM++ ban đầu mà không làm gián đoạn chương trình.

### 3.2. Thuật toán Ánh xạ Danh tính và Gán Nhãn Phụ đề

#### A. Thuật toán Ánh xạ Danh tính Hungarian (pyannote sang CAM++)
Mô hình pyannote trả về tập hợp các nhãn ẩn danh $P = \{p_0, p_1, \dots, p_{M-1}\}$ (`SPEAKER_00`, `SPEAKER_01`, ...).
Hệ thống CAM++ đã thiết lập tập hợp các cụm hồ sơ $C = \{c_0, c_1, \dots, c_{K-1}\}$ kèm đoạn âm thanh mẫu, thông số giọng và các đánh giá chất lượng.

Để ghép cặp tối ưu giữa hai tập nhãn, hệ thống xây dựng ma trận độ tương đồng kết hợp kích thước $M \times K$:
1. **Thành phần tương đồng thời gian giao nhau ($S_{\text{overlap}}$):**
   Với mỗi cặp $(p_i, c_j)$, tính tổng thời lượng giao nhau giữa các phân đoạn phát âm của nhãn $p_i$ và các câu phụ đề thuộc cụm $c_j$:
   $$T_{\text{overlap}}(p_i, c_j) = \sum_{k \in \text{cues}(c_j)} \text{duration}(\text{segment}(k) \cap \text{tracks}(p_i))$$
   Chuẩn hóa theo tổng thời lượng câu phụ đề của cụm $c_j$:
   $$S_{\text{overlap}}(p_i, c_j) = \frac{T_{\text{overlap}}(p_i, c_j)}{\sum_{k \in \text{cues}(c_j)} \text{duration}(\text{segment}(k)) + 1e-6} \in [0, 1]$$

2. **Thành phần tương đồng âm học cosin ($S_{\text{cosine}}$):**
   Mỗi câu phụ đề $k$ đã có sẵn vector đặc trưng 192 chiều trích xuất từ CAM++ ($e_k$).
   Với mỗi nhãn pyannote $p_i$, vector biểu diễn âm học trung bình của $p_i$ được tính bằng trung bình các vector CAM++ của những câu phụ đề có phân đoạn của $p_i$ bao phủ từ 50% thời lượng trở lên:
   $$v(p_i) = \frac{1}{|K_i|} \sum_{k \in K_i} e_k$$
   (Trường hợp tập hợp $K_i$ rỗng, gán $S_{\text{cosine}} = 0,5$).
   Độ tương đồng cosin giữa vector $v(p_i)$ và tâm cụm CAM++ $u(c_j)$ được chuẩn hóa về khoảng $[0, 1]$:
   $$\text{cosine}(p_i, c_j) = \frac{v(p_i) \cdot u(c_j)}{\|v(p_i)\| \|u(c_j)\| + 1e-6} \in [-1, 1]$$
   $$S_{\text{cosine}}(p_i, c_j) = \frac{\text{cosine}(p_i, c_j) + 1,0}{2,0} \in [0, 1]$$

3. **Điểm tương đồng kết hợp và Ma trận chi phí:**
   $$S(p_i, c_j) = 0,6 \cdot S_{\text{overlap}}(p_i, c_j) + 0,4 \cdot S_{\text{cosine}}(p_i, c_j)$$
   Trọng số $0,6 / 0,4$ ưu tiên tính đồng bộ trực tiếp trên dòng thời gian phụ đề, đồng thời sử dụng đặc trưng âm học làm chốt kiểm soát chống ghép nhầm giữa các nhân vật có giọng điệu tương đồng.
   Chuyển đổi sang ma trận chi phí để hàm `scipy.optimize.linear_sum_assignment` tối thiểu hóa:
   $$\text{Cost}(p_i, c_j) = 1,0 - S(p_i, c_j)$$

4. **Xử lý số lượng người nói không bằng nhau:**
   - Trường hợp $M > K$ (pyannote tìm ra nhiều người nói hơn CAM++): $K$ nhãn có độ tương đồng cao nhất được ánh xạ vào $K$ hồ sơ CAM++ có sẵn. $M - K$ nhãn pyannote còn lại được tạo thành các đối tượng `SpeakerProfile` mới với mẫu âm thanh cắt từ câu phụ đề dài nhất tương ứng.
   - Trường hợp $M < K$ (pyannote gom cụm gọn hơn CAM++): $M$ nhãn pyannote được ánh xạ vào $M$ cụm CAM++ tương ứng. $K - M$ cụm CAM++ dôi dư không có câu thoại nào sẽ được chuyển trạng thái sang `status = "unassigned"`, không để lại hồ sơ rác trong bảng phân vai.

#### B. Công thức Gán Nhãn Câu Phụ đề và Nhánh Rẽ Tường Minh
Với từng câu phụ đề trong danh sách SRT:
1. Tính toán tỷ lệ bao phủ của từng nhãn người nói pyannote đối với thời lượng của câu phụ đề:
   $$r_m = \frac{\text{Thời lượng giao nhau giữa phân đoạn pyannote của nhãn } m \text{ và câu phụ đề}}{\text{Thời lượng của câu phụ đề}}$$
   Sắp xếp các nhãn theo tỷ lệ bao phủ giảm dần: $r_{\text{max}} = r_{(1)} \ge r_{(2)} \dots$

2. **Nhánh gán nhãn chính xác:**
   - **Trường hợp $r_{\text{max}} \ge 0,5$:**
     - Câu phụ đề được gán cho nhân vật tương ứng với nhãn $m_{(1)}$ (thông qua bảng ánh xạ Hungarian ở mục A).
     - Kiểm tra hiện tượng nói chồng: Nếu có người nói thứ hai xuất hiện với độ chênh lệch nhỏ ($r_{(1)} - r_{(2)} < 0,10$), lưu thông tin cảnh báo nói chồng `overlap_detected = True` vào dữ liệu mở rộng của khối phụ đề và đánh dấu cờ `review_needed = True`.
     - Nếu nhãn được gán trùng khớp với nhãn ban đầu của CAM++: Giữ nguyên trạng thái và đánh dấu câu thoại có độ tin cậy cao.
     - Nếu nhãn được gán khác biệt so với CAM++: Cập nhật nhãn nhân vật mới và đánh dấu cờ `review_needed = True`.
   - **Trường hợp $r_{\text{max}} < 0,5$ (Nhánh xử lý ngoại lệ):**
     - Xảy ra khi câu phụ đề ngắn rơi vào khoảng lặng, khoảng nhạc nền, hoặc nằm ở ranh giới giao nhau giữa hai phân đoạn phát âm.
     - Hệ thống giữ nguyên nhãn nhân vật đã được phân bổ ban đầu bởi CAM++, không ép đổi nhãn.
     - Đánh dấu cờ cần rà soát `review_needed = True` để người dùng dễ dàng nhận diện và kiểm tra trên bảng câu thoại.

#### C. Cổng Kiểm soát Chất lượng Nhị phân
Để ngăn chặn trường hợp mô hình pyannote phân đoạn sai lệch nghiêm trọng trên các đoạn âm thanh phức tạp, hệ thống thiết lập cổng kiểm soát chất lượng nhị phân trước khi áp dụng kết quả:
1. **Đo lường độ đồng thuận khung thời gian:**
   Chuyển đổi chuỗi phân đoạn của CAM++ (theo từng câu phụ đề) và kết quả của pyannote thành chuỗi nhãn thời gian theo khung 100 mili-giây.
   $$\text{Chỉ số đồng thuận} = \frac{\sum_{t} \mathbb{I}(\text{nhãn}_{\text{pyannote}}(t) == \text{nhãn}_{\text{CAM++\_mapped}}(t))}{\text{Tổng số khung thời gian có tiếng nói}}$$

2. **Điều kiện kích hoạt cơ chế bảo vệ an toàn:**
   Hệ thống từ chối toàn bộ kết quả của pyannote khi xảy ra một trong hai điều kiện:
   - Khi không ấn định số người nói (`exact_speakers` là `None`) và độ lệch số lượng nhân vật phát hiện vượt quá 2 người:
     $$|\text{Số người}_{\text{pyannote}} - \text{Số người}_{\text{CAM++}}| > 2$$
   - Hoặc chỉ số đồng thuận khung thời gian thấp hơn ngưỡng an toàn:
     $$\text{Chỉ số đồng thuận} < 0,35$$

3. **Hành vi xử lý:**
   Khi điều kiện an toàn bị vi phạm, hệ thống kích hoạt cơ chế dự phòng nhị phân: Hủy bỏ toàn bộ các thay đổi từ pyannote, hoàn trả nguyên vẹn 100% kết quả từ mô hình CAM++ ONNX, đồng thời ghi nhật ký cảnh báo chi tiết lý do từ chối để hỗ trợ phân tích kỹ thuật.

### 3.3. Định vị Mô hình NeMo và Giới hạn Môi trường Hệ thống
- Mô hình NVIDIA Nemotron 3 Diarization (phát hành tháng 9 năm 2026) sử dụng kiến trúc phân loại khung thời gian đa kênh và hỗ trợ tối đa 8 người nói. Tuy nhiên, bộ công cụ NeMo yêu cầu môi trường Linux chuyên biệt và tiềm ẩn xung đột nhị phân trên môi trường Windows native của Sakai Studio.
- Quyết định kiến trúc: Hoãn hoàn toàn việc cài đặt Nemotron 3 trong Giai đoạn 1. Toàn bộ tài nguyên và mã nguồn tập trung hoàn thiện pipeline `pyannote.audio==3.1.1`. Cổng NeMo sẽ được xem xét trong các giai đoạn sau dưới dạng dịch vụ tách biệt hoặc môi trường phụ trợ tương thích.

### 3.4. Quản lý Cấu hình Hệ thống và Tính Bền vững Dữ liệu
- **Cấu hình lớp `Config(QConfig)` tại `src/core/config.py`:**
  Khai báo các thuộc tính cấu hình mới thuộc nhóm `"MMO"` theo đúng mẫu thiết kế của `mmoTTSMaxSpeakers`:
  ```python
  mmoTTSDeepDiarization = ConfigItem("MMO", "TTSDeepDiarization", False, BoolValidator())
  mmoTTSDeepDiarizationEngine = OptionsConfigItem(
      "MMO",
      "TTSDeepDiarizationEngine",
      "pyannote",
      OptionsValidator(["pyannote", "nemotron"]),
  )
  mmoTTSDeepDiarizationModelDir = ConfigItem("MMO", "TTSDeepDiarizationModelDir", "")
  ```
  Cách tổ chức này đảm bảo tính tương thích với hệ thống lưu trữ cấu hình giao diện `QConfig`, hỗ trợ tự động lưu trữ và phục hồi cài đặt giữa các phiên làm việc của người dùng.

### 3.5. Tích hợp Giao diện Người dùng
- **Vị trí tệp:** `src/desktop/ui/omnivoice_interface.py`
- **Mục chọn phần cứng (dòng 2542):**
  - Cập nhật chuỗi hiển thị thành: "Phân đoạn người nói chuyên sâu — pyannote".
- **Xử lý sự kiện điều khiển (dòng 6959–6964):**
  - Loại bỏ hộp thoại cảnh báo từ chối thao tác cũ.
  - Kiểm tra trạng thái nạp và phân giải mã xác thực:
    - Nếu hệ thống không có card đồ họa CUDA: Hiển thị thanh thông báo trạng thái vận hành trên CPU với lưu ý về thời gian xử lý.
    - Nếu chưa cấu hình mã xác thực qua biến môi trường hoặc phiên đăng lệnh: Hiển thị thanh thông báo hướng dẫn thiết lập biến môi trường `HF_TOKEN` hoặc đăng nhập bằng công cụ dòng lệnh.
    - Khi các điều kiện hợp lệ: Cập nhật giá trị `app_config.mmoTTSDeepDiarization = True` và tiến hành quy trình phân đoạn nâng cao.

---

## 4. Quản lý Bộ nhớ Đồ họa và An toàn Phần cứng

- **Phần cứng mục tiêu:** Card đồ họa NVIDIA GeForce RTX 4050 Laptop GPU (Tổng dung lượng bộ nhớ đồ họa: 6144 MB).
- **Nguyên tắc đo đạc thực tế tại Giai đoạn 0:**
  Bắt buộc thực hiện đo đạc thực tế bằng các công cụ giám sát phần cứng (`nvidia-smi` và `torch.cuda.max_memory_allocated()`) để ghi nhận:
  1. Mức sử dụng bộ nhớ đồ họa nền của hệ điều hành và giao diện hiển thị.
  2. Mức tiêu hao đỉnh của `pyannote.audio 3.1.1` khi nạp đầy đủ cả mô hình phân đoạn thời gian và mô hình trích xuất đặc trưng âm học.
  3. Mức tiêu hao đỉnh của động cơ tổng hợp giọng đọc OmniVoice.
- **Quy trình thu hồi bộ nhớ tuần tự bất biến:**
  - Tuyệt đối không nạp đồng thời mô hình pyannote và mô hình tổng hợp giọng đọc trong cùng một thời điểm.
  - Ngay sau khi hoàn thành bước phân đoạn:
    ```python
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    ```
  - Đảm bảo toàn bộ dung lượng bộ nhớ đồ họa được thu hồi về trạng thái nghỉ trước khi nạp động cơ giọng đọc.

---

## 5. Kế hoạch Kiểm thử và Tiêu chuẩn Nghiệm thu

### 5.1. Thiết lập Dữ liệu Đo đếm và Chấp thuận Điều khoản Bản quyền (Giai đoạn 0)
1. **Chấp thuận điều khoản bản quyền trên Hugging Face:**
   - Thực hiện đăng nhập và bấm chấp thuận điều khoản người dùng đối với hai kho mô hình có bản quyền giới hạn:
     - `pyannote/speaker-diarization-3.1`
     - `pyannote/segmentation-3.0`
   - Xác thực việc tải trọng số thành công thông qua công cụ dòng lệnh `huggingface-cli` trước khi chạy kiểm thử mã nguồn.
2. **Chuẩn bị dữ liệu kiểm thử rút gọn:**
   - Chốt cửa sổ thời gian chuẩn 30 giây: Từ giây `378,2` đến giây `408,2` (bao gồm 12 câu phụ đề từ số 122 đến số 133).
   - Phân bố 4 nhân vật thực tế: Sếp (5 câu), Lão Ngũ (3 câu), Bác tài (2 câu) và Cheems (2 câu).
   - Thẩm định chốt nhãn phân vai dứt khoát cho câu phụ đề số 124 (thời gian 386,3 - 388,3 s) là Bác tài.
   - Quản lý tệp dữ liệu tuân thủ bản quyền kho mã nguồn:
     - Xây dựng kịch bản trích xuất tự động `scripts/prepare_cheems_fixture.py` để tạo tệp âm thanh kiểm thử cục bộ `test/fixtures/cheems_30s.wav` từ video nguồn.
     - Chỉ lưu trữ trong kho mã nguồn: Tệp kịch bản trích xuất, tệp phụ đề `test/fixtures/cheems_30s.srt` và tệp nhãn thẩm định chuẩn `test/fixtures/cheems_30s_ground_truth.json`.
     - Bổ sung quy tắc bỏ qua tệp âm thanh `test/fixtures/*.wav` trong tệp cấu hình `.gitignore`.
3. **Đo đạc chính thức đường chuẩn CAM++:**
   - Thực thi hàm `SpeakerDiarizationCoordinator.process_video_subtitles()` trên toàn bộ video Cheems 589 giây.
   - Ghi nhận đầy đủ 4 thông số đường chuẩn thực tế:
     - Số lượng nhân vật thuật toán tự động nhận diện.
     - Tỷ lệ phần trăm câu phụ đề được gán đúng nhân vật so với tệp thẩm định chuẩn `assignment.json`.
     - Tổng thời gian thực thi toàn bộ quy trình trên card đồ họa.
     - Mức tiêu hao bộ nhớ đồ họa đỉnh đo bằng công cụ giám sát phần cứng.

### 5.2. Tiêu chuẩn Nghiệm thu Kỹ thuật (Giai đoạn 1)
- **Độ chính xác nhận diện nhân vật:** Nhận diện chính xác 6 trên 6 nhân vật trên toàn bộ video Cheems 589 giây; tỷ lệ câu phụ đề được gán đúng người nói đạt từ 90% trở lên.
- **Tỷ lệ lỗi phân đoạn (DER):** Chỉ số DER đo đạc trên đoạn trích chuẩn 30 giây đạt dưới 15%.
- **Thời gian xử lý toàn trình:** Tổng thời gian thực thi của pyannote trên card đồ họa không vượt quá 2 lần so với thời gian đường chuẩn của CAM++ đã đo đạc trên cùng một video.
- **Ngưỡng an toàn bộ nhớ:** Mức sử dụng bộ nhớ đồ họa đỉnh không vượt quá 5,5 GB trong suốt quá trình chạy.
- **Độ tin cậy của cơ chế dự phòng và chất lượng:**
  - Số lần kích hoạt dự phòng về CAM++ trong một phiên chạy chuẩn có đủ môi trường và bản quyền: bằng 0.
  - Khi tắt tùy chọn phân đoạn chuyên sâu, kết quả đầu ra đồng nhất hoàn toàn với thuật toán CAM++ ban đầu.
  - Khi giả lập môi trường thiếu mạng hoặc token không hợp lệ, hệ thống tự động kích hoạt dự phòng CAM++ mà không phát sinh ngoại lệ làm dừng ứng dụng.
  - Các đoạn hội thoại có từ 2 người nói đồng thời trở lên được phát hiện và lưu cờ cảnh báo `overlap_detected = True`.
- **Kiểm thử tích hợp mới:**
  Xây dựng tệp kiểm thử tự động `test/test_diarization_verifier_integration.py` bao gồm:
  - Kiểm thử phương thức `verify_or_fallback` với đối tượng chú thích phân đoạn mô phỏng.
  - Kiểm thử trường hợp hàm nạp trả về giá trị rỗng (`pipeline is None`) kích hoạt dự phòng an toàn.
  - Kiểm thử thuật toán ghép cặp danh tính Hungarian với trọng số kết hợp 0,6 thời gian và 0,4 cosin.
  - Kiểm thử các nhánh điều kiện tỷ lệ bao phủ và nhánh rẽ ngoại lệ (dưới 0,5).
  - Kiểm thử cổng kiểm soát chất lượng nhị phân khi phát hiện độ lệch số lượng hoặc độ đồng thuận thấp.
