# Sakai Studio

Sakai Studio là phần mềm xử lý video, phụ đề và lồng tiếng chuyên sâu ứng dụng trí tuệ nhân tạo. Phần mềm cung cấp hệ thống công cụ tự động hóa toàn diện từ khâu tải video, nhận diện trích xuất chữ, dịch thuật đa ngôn ngữ, phân vai lồng tiếng đa nhân vật cho đến xóa phụ đề gắn cứng và phục hồi nền video chất lượng cao.

Hệ thống được thiết kế tối ưu hóa bộ nhớ đồ họa, vận hành mượt mà trên các dòng thẻ đồ họa thông dụng như NVIDIA GeForce RTX 4050 với dung lượng bộ nhớ đồ họa 6GB, đồng thời hỗ trợ đầy đủ môi trường DirectML và bộ xử lý trung tâm.

---

## 1. Các Tính Năng Cốt Lõi

Hệ thống được tổ chức thành các phân hệ chuyên biệt theo đúng quy tắc kiến trúc và trải nghiệm người dùng:

### Thẻ Tự Động
- Luồng xử lý liên hoàn khép kín: Tự động tải video từ liên kết trực tuyến, trích xuất phụ đề, dịch sang tiếng Việt, phân vai lồng tiếng và xóa sạch phụ đề gắn cứng cũ trên video.
- Giám sát tiến độ từng bước trực quan, tự động mở khóa các công cụ tương ứng khi dữ liệu sẵn sàng.
- Hỗ trợ chế độ xử lý hàng loạt nhiều video cùng lúc với cơ chế kiểm soát lỗi độc lập.

### Thẻ Công Cụ
Khu vực tích hợp bốn công cụ chuyên sâu thông qua thanh điều hướng tiện lợi:
- **Tải Video:** Tải video và âm thanh độ phân giải cao từ hơn 1000 nền tảng trực tuyến như YouTube, Bilibili, TikTok, Facebook. Hỗ trợ tải trọn bộ danh sách phát và tự động chuyển đổi định dạng âm thanh.
- **Trích Xuất Phụ Đề:** 
  - Nhận diện chữ hình ảnh từ khung hình video bằng các mô hình OCR và mô hình thị giác ngôn ngữ lớn.
  - Chuyển đổi giọng nói thành văn bản chuẩn xác theo mốc thời gian phụ đề bằng mô hình Whisper nhận diện đa ngôn ngữ.
- **Dịch Phụ Đề:** Dịch thuật tệp phụ đề giữ nguyên cấu trúc mốc thời gian. Hỗ trợ các mô hình trí tuệ nhân tạo chạy cục bộ ngoại tuyến thông qua Ollama, LM Studio, tệp mô hình GGUF và các dịch vụ trực tuyến như Google, Gemini, DeepSeek, OpenAI.
- **Xóa Phụ Đề Video:** Quét vùng chữ phụ đề, tự động theo dõi đối tượng chuyển động bằng mô hình SAM2, loại bỏ phụ đề gắn cứng và tái tạo khung hình video sắc nét bằng các thuật toán LaMa, STTN và ProPainter.

### Thẻ Lồng Tiếng
Phân hệ lồng tiếng đa nhân vật OmniVoice chất lượng cao:
- Tự động nhận diện số lượng người nói động bằng thuật toán phân tích bước nhảy khoảng cách trên cây phân cấp, không cố định số người và loại bỏ hoàn toàn tình trạng phình to cụm người nói.
- Trích xuất đặc trưng âm thanh và mỏ neo giọng nói bằng mô hình CAM++ qua môi trường tăng tốc ONNX Runtime.
- Đo lường đặc trưng âm học đa chiều: cao độ, trường độ, năng lượng vùng tần số lồng ngực, phân loại giới tính chuẩn xác.
- Tự động phân bổ giọng đọc tiếng Việt riêng biệt không trùng lặp theo tỷ lệ một - một từ kho 14 giọng đọc chuẩn gồm 7 giọng nam và 7 giọng nữ với ngữ điệu tự nhiên.
- Đồng bộ thời lượng câu thoại với phụ đề, hỗ trợ làm chậm hoặc giãn thời gian và tự động giảm âm lượng nhạc nền khi có lời thoại.

### Thẻ Cấu Hình MMO
- Quản lý cấu hình sản xuất nội dung video hàng loạt: thiết lập kiểu hiển thị phụ đề, phông chữ Segoe UI sắc nét, kích cỡ chữ, màu sắc, viền chữ và vị trí xuất hiện.
- Lưu trữ các mẫu giọng đọc ưa thích cho từng dự án sáng tạo nội dung.

### Thẻ Cài Đặt Nâng Cao
- Quản lý thiết bị phần cứng: chuyển đổi linh hoạt giữa tăng tốc phần cứng CUDA của NVIDIA, DirectML hoặc bộ xử lý trung tâm.
- Quản lý bộ nhớ đồ họa an toàn, thiết lập các tham số xử lý khung hình, ngưỡng làm mịn mặt nạ và đường dẫn lưu trữ tạm.

---

## 2. Bảng Yêu Cầu Phần Cứng và Môi Trường

| Phân loại | Môi trường đề xuất | Môi trường tối thiểu |
|---|---|---|
| Hệ điều hành | Windows 10, Windows 11 phiên bản 64-bit | Linux Ubuntu 20.04+, macOS 12+ |
| Bộ xử lý trung tâm | 6 nhân 12 luồng trở lên | 4 nhân 8 luồng |
| Bộ nhớ hệ thống | 16 GB trở lên | 8 GB |
| Thẻ đồ họa | NVIDIA GeForce RTX 3060, RTX 4050 6GB trở lên | Thẻ đồ họa tích hợp hỗ trợ DirectML hoặc chỉ dùng bộ xử lý trung tâm |
| Môi trường tính toán | CUDA 11.8 hoặc CUDA 12.6, cuDNN 8.6 trở lên | DirectML hoặc Bộ xử lý trung tâm |
| Phiên bản Python | Python 3.12, khuyến nghị phiên bản 3.12.9 | Python 3.10 trở lên |

---

## 3. Hướng Dẫn Cài Đặt và Khởi Chạy

### Cách 1: Khởi chạy nhanh trên Windows
Nếu bạn sử dụng gói cài đặt sẵn của Sakai Studio:
1. Tải về và giải nén thư mục dự án.
2. Nhấp đúp vào tệp tin `start.bat` để tự động kích hoạt môi trường và khởi chạy giao diện phần mềm.

### Cách 2: Cài đặt từ mã nguồn

#### Bước 1: Chuẩn bị môi trường Python
Đảm bảo máy tính đã cài đặt Python phiên bản 3.12 trở lên. Tạo và kích hoạt môi trường ảo:

Trên hệ điều hành Windows:
```shell
python -m venv venv
venv\Scripts\activate
```

Trên hệ điều hành Linux hoặc macOS:
```shell
python3 -m venv venv
source venv/bin/activate
```

#### Bước 2: Cài đặt thư viện tính toán tăng tốc thẻ đồ họa

Dành cho thẻ đồ họa NVIDIA hỗ trợ CUDA 11.8:
```shell
pip install torch==2.7.0 torchvision==0.22.0 torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Dành cho thẻ đồ họa NVIDIA hỗ trợ CUDA 12.4 hoặc 12.6:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu124/
```

Dành cho thẻ đồ họa AMD, Intel qua DirectML hoặc Bộ xử lý trung tâm:
```shell
pip install torch torchvision torchaudio
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

#### Bước 3: Cài đặt các gói phụ thuộc còn lại
```shell
pip install -r requirements.txt
```

#### Bước 4: Khởi chạy ứng dụng

Khởi chạy giao diện đồ họa:
```shell
python -m src.desktop.main
```

Khởi chạy công cụ dòng lệnh:
```shell
python -m src.core.main -i video_goc.mp4 -o video_da_xu_ly.mp4
```

---

## 4. Hướng Dẫn Vận Hành Quy Trình Chi Tiết

### Quy trình Lồng tiếng Đa nhân vật OmniVoice
1. **Bước 1 - Nạp dữ liệu:** Nạp tệp phụ đề SRT và tệp video nguồn tương ứng. Hệ thống sẽ tự động đồng bộ mốc thời gian và độ dài câu thoại.
2. **Bước 2 - Nhận diện nhân vật:** Nhấn nút "Tự Động Nhận Diện Nhân Vật". Thuật toán sẽ quét âm thanh, phân tích bước nhảy trên cây phân cấp để tìm ra đúng số lượng người nói thực tế và phân tích đặc trưng giới tính, cao độ.
3. **Bước 3 - Kiểm tra bảng phân vai:** Xem danh sách nhân vật đã được gán giọng đọc tương ứng từ kho 14 giọng tiếng Việt chuẩn. Bạn có thể nghe thử giọng từng người hoặc thay đổi giọng đọc mong muốn.
4. **Bước 4 - Kiểm tra câu thoại:** Bảng phụ đề hiển thị chi tiết từng câu cùng tên nhân vật phụ trách. Bạn có thể bấm nghe thử từng dòng thoại hoặc tinh chỉnh nội dung dịch.
5. **Bước 5 - Xuất kết quả:** Chọn xuất tệp âm thanh WAV tổng hợp hoặc xuất thẳng tệp video MP4 hoàn chỉnh kèm tính năng tự động hòa trộn giảm âm lượng nhạc nền.

### Quy trình Tự động Tất-cả-trong-một
1. Dán đường dẫn liên kết video trực tuyến vào ô nhập liệu hoặc chọn tệp video có sẵn trong máy tính.
2. Lựa chọn các công đoạn cần thực hiện: Tải video, Trích xuất lời thoại, Dịch thuật, Lồng tiếng đa nhân vật, Xóa phụ đề cũ.
3. Bấm "Bắt Đầu Xử Lý Tự Động". Toàn bộ chuỗi quy trình sẽ vận hành tự động và thông báo khi hoàn tất.

### Quy trình Xóa phụ đề và Tái tạo nền
1. Chọn tệp video cần xóa phụ đề tại thẻ Công Cụ, mục Xóa Phụ Đề Video.
2. Dùng chuột đóng khung vùng chữ phụ đề trên trình phát video xem trước.
3. Chọn mô hình tái tạo phù hợp:
   - Thuật toán STTN: Tốc độ xử lý cao, tối ưu cho video có chuyển động nhanh.
   - Thuật toán LAMA: Khả năng lấp đầy tĩnh hoàn hảo, tái tạo khung cảnh tĩnh cực sạch.
   - Thuật toán PROPAINTER: Tái tạo chất lượng cao nhất cho các vùng chuyển động phức tạp.
4. Nhấn nút "Bắt Đầu Xóa Phụ Đề" để xử lý và xuất video mới.

---

## 5. Các Tham Số Dòng Lệnh

Công cụ dòng lệnh hỗ trợ tự động hóa trong các tập lệnh và hệ thống máy chủ:
- `-i`, `--input`: Đường dẫn tệp tin video đầu vào cần xử lý.
- `-o`, `--output`: Đường dẫn tệp tin video đầu ra sau khi xử lý.
- `-c`, `--subtitle-area-coords`: Tọa độ vùng phụ đề cần xóa, định dạng ymin, ymax, xmin, xmax. Có thể khai báo nhiều lần cho nhiều khu vực.
- `--inpaint-mode`: Thuật toán phục hồi nền hình ảnh, mặc định là sttn-auto, hỗ trợ các chế độ sttn, lama, propainter.
- `--device`: Thiết bị tính toán sử dụng, bao gồm cuda hoặc cpu.

---

## 6. Giải Đáp Các Câu Hỏi Thường Gặp

**Làm sao để xử lý tình trạng tràn bộ nhớ đồ họa khi dùng thẻ 6GB?**
Hệ thống Sakai Studio đã được tích hợp cơ chế thu hồi bộ nhớ tự động sau mỗi giai đoạn. Nếu xử lý video độ phân giải cao 2K hoặc 4K, hãy vào thẻ Cài Đặt Nâng Cao, giảm kích thước phân đoạn âm thanh xuống mức 3.0 đến 4.0 giây và ưu tiên sử dụng thuật toán xóa chữ LaMa hoặc STTN.

**Làm thế nào khi video có nhiều nhân vật nói chuyện liên tục?**
Thuật toán phân tích bước nhảy trên cây phân cấp sẽ tự động gom các câu nói cùng âm sắc vào cùng một nhân vật và tự động cấp phát các giọng đọc khác nhau từ kho giọng đọc chuẩn. Mọi câu thoại của cùng một nhân vật sẽ luôn duy trì giọng đọc nhất quán xuyên suốt toàn bộ video.

**Phần mềm có hỗ trợ chạy khi không có mạng Internet không?**
Có. Toàn bộ các mô hình nhận diện chữ OCR, tách giọng, gom cụm người nói CAM++, mô hình dịch thuật GGUF/Ollama và mô hình lồng tiếng OmniVoice đều hoạt động hoàn toàn ngoại tuyến trên máy tính của bạn sau khi đã tải mô hình.

---

## 7. Giấy Phép và Quy Định Sử Dụng

Sakai Studio được phát triển nhằm mục đích phục vụ học tập, nghiên cứu và sản xuất nội dung sáng tạo hợp pháp. Người sử dụng có trách nhiệm tuân thủ bản quyền của video gốc và các quy định pháp luật hiện hành khi xuất bản sản phẩm.
