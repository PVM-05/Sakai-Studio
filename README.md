# Giới thiệu dự án

Sakai Studio là phần mềm ứng dụng trí tuệ nhân tạo để xóa phụ đề được gắn cứng trên video. Phần mềm chủ yếu thực hiện các chức năng sau:
- Xóa phụ đề gắn cứng khỏi video và xuất ra tệp tin không chứa phụ đề mà không làm giảm chất lượng hình ảnh.
- Lấp đầy vùng chữ phụ đề bị xóa bằng các mô hình thuật toán trí tuệ nhân tạo mạnh mẽ.
- Hỗ trợ xóa phụ đề tại vị trí tùy chỉnh, hệ thống sẽ chỉ xóa phụ đề trong khu vực được người dùng định vị.
- Hỗ trợ tự động xóa toàn bộ văn bản xuất hiện trên toàn bộ video mà không cần người dùng định vị.
- Hỗ trợ chọn nhiều hình ảnh cùng lúc để xóa văn bản hoặc hình mờ hàng loạt.

## Hướng dẫn cơ bản

- Tải về gói lưu trữ nén, giải nén và chạy trực tiếp. Nếu không thể khởi chạy, hãy làm theo tài liệu hướng dẫn bên dưới để thử cài đặt trực tiếp từ mã nguồn.

## Bảng so sánh các gói đóng gói sẵn

| Tên gói lưu trữ | Phiên bản Python | Phiên bản Paddle | Phiên bản Torch | Môi trường hệ thống | Khả năng tính toán hỗ trợ |
|---|---|---|---|---|---|
| Gói xử lý trung tâm | 3.12 | 3.0.0 | 2.7.0 | Phổ thông | Phổ thông |
| Gói DirectML | 3.12 | 3.0.0 | 2.4.1 | Máy tính Windows không có thẻ đồ họa NVIDIA | Phổ thông |
| Gói CUDA 11.8 | 3.12 | 3.0.0 | 2.7.0 | Môi trường CUDA 11.8 | 3.5 đến 8.9 |
| Gói CUDA 12.6 | 3.12 | 3.0.0 | 2.7.0 | Môi trường CUDA 12.6 | 5.0 đến 8.9 |
| Gói CUDA 12.8 | 3.12 | 3.0.0 | 2.7.0 | Môi trường CUDA 12.8 | 5.0 đến 9.0 trở lên |

## Phiên bản Docker

```shell
# Thẻ đồ họa NVIDIA dòng 10, 20, 30
docker run -it --name sakai --gpus all eritpchy/video-subtitle-remover:1.4.0-cuda11.8 python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Thẻ đồ họa NVIDIA dòng 40
docker run -it --name sakai --gpus all eritpchy/video-subtitle-remover:1.4.0-cuda12.6 python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Thẻ đồ họa NVIDIA dòng 50
docker run -it --name sakai --gpus all eritpchy/video-subtitle-remover:1.4.0-cuda12.8 python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Thẻ đồ họa AMD hoặc Intel
docker run -it --name sakai --gpus all eritpchy/video-subtitle-remover:1.4.0-directml python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Bộ xử lý trung tâm
docker run -it --name sakai --gpus all eritpchy/video-subtitle-remover:1.4.0-cpu python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Xuất tệp tin video
docker cp sakai:/vsr/test/test_no_sub.mp4 ./
```

## Giao diện dòng lệnh

Công cụ dòng lệnh xóa phụ đề video:
- `-h`, `--help`: Hiển thị thông báo trợ giúp này và thoát.
- `--input`, `-i`: Đường dẫn tệp video đầu vào.
- `--output`, `-o`: Đường dẫn tệp video xuất ra.
- `--subtitle-area-coords`, `-c`: Tọa độ vùng phụ đề. Có thể được chỉ định nhiều lần cho nhiều khu vực.
- `--inpaint-mode`: Chế độ lấp đầy tái tạo, mặc định là sttn-auto.

## Hướng dẫn sử dụng mã nguồn

### 1. Cài đặt Python
Vui lòng đảm bảo rằng bạn đã cài đặt Python 3.12 trở lên.
- Người dùng Windows có thể truy cập trang web chính thức của Python để tải xuống và cài đặt.
- Người dùng macOS có thể cài đặt thông qua Homebrew:
  ```shell
  brew install python@3.12
  ```
- Người dùng Linux có thể cài đặt qua trình quản lý gói:
  ```shell
  sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
  ```

### 2. Cài đặt các gói phụ thuộc
Nên sử dụng môi trường ảo để quản lý các gói phụ thuộc của dự án nhằm tránh xung đột với môi trường hệ thống.

Tạo và kích hoạt môi trường ảo:
```shell
python -m venv videoEnv
```

Kích hoạt trên Windows:
```shell
videoEnv\Scripts\activate
```

Kích hoạt trên MacOS hoặc Linux:
```shell
source videoEnv/bin/activate
```

### 3. Khởi tạo và kích hoạt thư mục dự án
Di chuyển đến thư mục chứa mã nguồn của bạn:
```shell
cd duong_dan_thu_muc_ma_nguon
```

### 4. Cài đặt môi trường thực thi phù hợp
Dự án này hỗ trợ bốn chế độ chạy: môi trường CUDA, bộ xử lý trung tâm, môi trường DirectML và môi trường macOS.

#### Cài đặt cho môi trường CUDA
Đảm bảo trình điều khiển thẻ đồ họa NVIDIA của bạn hỗ trợ phiên bản CUDA đã chọn. Khuyến nghị sử dụng CUDA 11.8 cùng với cuDNN 8.6.0.

Cài đặt CUDA:
- Windows: Tải xuống CUDA 11.8.
- Linux:
  ```shell
  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  sudo sh cuda_11.8.0_520.61.05_linux.run
  ```
- Không hỗ trợ CUDA trên macOS.

Cài đặt cuDNN:
- Tải xuống cuDNN 8.6.0 cho nền tảng tương ứng và làm theo hướng dẫn cài đặt trong tài liệu chính thức của NVIDIA.

Cài đặt phiên bản PaddlePaddle cho thẻ đồ họa:
```shell
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Cài đặt phiên bản Torch cho thẻ đồ họa:
```shell
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
```

Cài đặt các gói phụ trợ khác:
```shell
pip install -r requirements.txt
```

Đối với hệ thống Linux, bạn cần cài đặt thêm:
```shell
# Dành cho môi trường CUDA 12
pip install onnxruntime-gpu==1.22.0
# Dành cho môi trường CUDA 11
pip install onnxruntime-gpu==1.20.1 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

#### Cài đặt cho môi trường DirectML
Phù hợp cho các thiết bị Windows có thẻ đồ họa AMD, NVIDIA hoặc Intel.
```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt
pip install torch_directml==0.2.5.dev240914
```

#### Cài đặt cho môi trường bộ xử lý trung tâm
Phù hợp cho các hệ thống không có thẻ đồ họa hoặc những hệ thống không muốn sử dụng khả năng tăng tốc phần cứng.
```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
```

#### Cài đặt cho môi trường macOS
Phù hợp cho các thiết bị macOS kiến trúc dòng chip Apple. Đối với kiến trúc chip Intel, vui lòng sử dụng chế độ bộ xử lý trung tâm.
```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
```

### 5. Khởi chạy chương trình

Chạy giao diện đồ họa người dùng:
```shell
python gui.py
```

Chạy phiên bản dòng lệnh:
```shell
python ./backend/main.py
```

## Các vấn đề thường gặp

**Làm thế nào để xử lý tốc độ xóa chậm?**
Bạn có thể tăng đáng kể tốc độ xóa bằng cách sửa đổi các tham số trong tệp cấu hình phụ trợ:
```python
MODE = InpaintMode.STTN
STTN_SKIP_DETECTION = True
```

**Phải làm gì nếu kết quả xóa video không đạt yêu cầu?**
Sửa đổi các giá trị trong tệp cấu hình phụ trợ và thử các thuật toán xóa khác nhau. Dưới đây là giới thiệu về các thuật toán:
- Thuật toán STTN: Phù hợp với các video hành động thực tế và có tốc độ nhanh, có khả năng bỏ qua việc phát hiện phụ đề.
- Thuật toán LAMA: Phù hợp nhất cho hình ảnh và hiệu quả với các video hoạt hình, tốc độ trung bình, không thể bỏ qua phát hiện phụ đề.
- Thuật toán PROPAINTER: Tiêu tốn một lượng lớn bộ nhớ đồ họa, tốc độ chậm hơn, hoạt động tốt hơn đối với các video có chuyển động rất mạnh.

Ví dụ cho thuật toán STTN:
```python
MODE = InpaintMode.STTN
STTN_NEIGHBOR_STRIDE = 10
STTN_REFERENCE_LENGTH = 10
STTN_MAX_LOAD_NUM = 30
```

Ví dụ cho thuật toán LAMA:
```python
MODE = InpaintMode.LAMA
LAMA_SUPER_FAST = False
```

**Lỗi giải nén tệp lưu trữ 7z**
Giải pháp: Nâng cấp chương trình giải nén lên phiên bản mới nhất.
