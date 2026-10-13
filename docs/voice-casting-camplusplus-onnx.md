# Cài đặt nhận diện người nói bằng CAM++ ONNX

Tính năng Auto Voice Casting cần một tệp trọng số ONNX được cài riêng. Sakai-Studio không tải mô hình tự động, vì vậy sau khi cài mô hình, bước nhận diện có thể chạy ngoại tuyến.

## 1. Tải mô hình

Mô hình tham chiếu: [CAM++ 3D-Speaker ONNX (Chinese + English)](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx).

- Tải tệp \`campplus_cn_en_common_200k.onnx\`.
- Không đưa tệp trọng số vào Git; đặt nó trong thư mục cục bộ của máy.
- Định dạng mô hình kỳ vọng: đầu vào FBank 80 chiều \`(batch, frames, 80)\`, đầu ra embedding 192 chiều, âm thanh 16 kHz mono.
- Mô hình được công bố theo Apache-2.0. Hãy đọc model card và các điều kiện liên quan trước khi phân phối lại.

Đường dẫn mặc định được tự dò:
\`models/speaker_embedding/campplus_cn_en_common_200k.onnx\`

Có thể dùng vị trí khác bằng biến môi trường:

\`\`\`powershell
$env:SAKAI_SPEAKER_EMBEDDING_MODEL = "D:\\AIModels\\campplus_cn_en_common_200k.onnx"
\`\`\`

Lệnh trên áp dụng cho cửa sổ PowerShell hiện tại. Khởi chạy Sakai-Studio từ cửa sổ đó.

## 2. Cài runtime GPU

Kích hoạt đúng môi trường Python đang chạy Sakai-Studio, sau đó cài frontend FBank và ONNX Runtime GPU tương thích với môi trường CUDA/cuDNN trên máy:

\`\`\`powershell
.\\venv\\Scripts\\python.exe -m pip install kaldi-native-fbank
.\\venv\\Scripts\\python.exe -m pip install onnxruntime-gpu
\`\`\`

ONNX Runtime CPU và GPU dùng cùng namespace Python; tránh để hai bản phân phối xung đột trong một môi trường. Nếu dự án hoặc một thư viện OCR đã cài runtime CPU, hãy kiểm tra dependency và cài lại theo hướng dẫn ONNX Runtime dành cho CUDA trước khi xác nhận tính năng.

Kiểm tra runtime:

\`\`\`powershell
.\\venv\\Scripts\\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
\`\`\`

Để dùng RTX 4050, kết quả cần có \`CUDAExecutionProvider\`. Nếu chỉ có \`CPUExecutionProvider\`, ứng dụng sẽ không thể thực thi embedding bằng CUDA. Kiểm tra lại driver NVIDIA, gói ONNX Runtime GPU và tính tương thích CUDA/cuDNN.

## 3. Chạy trong ứng dụng

1. Đặt mô hình vào thư mục mặc định hoặc cấu hình biến môi trường.
2. Mở trang Lồng Tiếng và nạp cả video lẫn SRT.
3. Chọn **Ưu tiên card đồ họa NVIDIA RTX 4050** ở phần cài đặt nhận diện.
4. Chạy nhận diện. Nhật ký phải nêu provider thực tế, không chỉ thiết bị được ưu tiên.
5. Sau khi nhận diện hoàn tất, phiên ONNX được đóng trước khi OmniVoice được nạp để tổng hợp giọng.

Nếu thiếu mô hình, không đọc được âm thanh hoặc embedding không hợp lệ, tính năng sẽ trả lỗi thay vì phân vai từ vector giả. Chế độ giả lập chỉ dành cho kiểm thử đơn vị.

## 4. Giới hạn hiện tại

- Bài kiểm thử CI kiểm tra cấu trúc FBank và logic phân cụm bằng dữ liệu tổng hợp. Nó không chứng minh độ chính xác của mô hình trên video thật và không kiểm tra RTX 4050 vật lý.
- Nhãn người nói được suy ra từ từng đoạn theo timestamp SRT. Một dòng phụ đề có nhiều người nói hoặc hội thoại chồng tiếng vẫn cần diarization theo thời gian chuyên sâu.
- Tùy chọn Pyannote chưa được tích hợp trong nhánh hiện tại; giao diện đánh dấu rõ là chưa tích hợp.
