# Hướng dẫn Triển khai Hệ thống Sakai Studio lên Mạng Internet

Tài liệu này hướng dẫn chi tiết quy trình đưa toàn bộ hệ thống dịch vụ Sakai Studio từ môi trường thử nghiệm cục bộ lên mạng Internet phục vụ người dùng thực tế.

---

## 1. Kiến trúc Tổng thể của Hệ thống

Toàn bộ hệ thống Sakai Studio hoạt động theo mô hình phân tán đồng bộ gồm 5 thành phần chuyên biệt:

1. **Cổng điều phối đảo ngược Nginx:** Tiếp nhận lưu lượng truy cập từ Internet trên cổng 80 và 443, tự động phân luồng yêu cầu người dùng tới giao diện hoặc máy chủ xử lý dữ liệu.
2. **Giao diện người dùng Next.js:** Cung cấp trải nghiệm thao tác trên trình duyệt, kết nối thời gian thực với máy chủ qua luồng sự kiện.
3. **Máy chủ xử lý FastAPI:** Điều hành toàn bộ các dịch vụ nghiệp vụ, bóc tách liên kết đa phương tiện, quản trị cơ sở dữ liệu và giao việc cho hàng đợi.
4. **Hàng đợi thông điệp Redis:** Cầu nối phân phối công việc ổn định cho các tiến trình nền.
5. **Tiến trình xử lý ngầm Celery Worker:** Tiếp nhận công việc tải video độ phân giải cao, bóc tách phụ đề và tái tạo hình ảnh video.

---

## 2. Phương án 1: Triển khai qua Đường hầm Cloudflare (Khuyên dùng nhất)

Đây là phương thức tối ưu nhất dành cho người dùng muốn sử dụng ngay máy tính cá nhân hoặc máy trạm làm máy chủ phát hành ra Internet mà không cần chi phí thuê máy chủ riêng, không cần địa chỉ IP tĩnh và không cần mở cổng trên thiết bị mạng gia đình.

### Ưu điểm
- Hoàn toàn miễn phí.
- Tự động kích hoạt chứng chỉ bảo mật HTTPS toàn cầu.
- Bảo vệ máy tính cá nhân khỏi các cuộc tấn công mạng nguy hiểm.
- Không cần cấu hình phức tạp trên modem mạng.

### Các bước thực hiện

#### Bước 1: Khởi chạy toàn bộ hệ thống
Người dùng mở cửa sổ lệnh tại thư mục gốc của dự án và khởi chạy hệ thống bằng Docker:
```bash
docker compose up -d
```
Hoặc nếu chạy trực tiếp trên hệ điều hành Windows mà không dùng Docker, người dùng chạy tệp:
```cmd
start_saas.bat
```

#### Bước 2: Tải công cụ điều phối đường hầm Cloudflared
1. Truy cập trang phát hành chính thức của Cloudflare và tải tệp thực thi `cloudflared-windows-amd64.exe` (đối với Windows) hoặc phiên bản Linux tương ứng.
2. Đổi tên tệp thành `cloudflared.exe` và đặt trong thư mục hệ thống hoặc ngay tại thư mục dự án.

#### Bước 3: Mở đường hầm ra Internet
Mở cửa sổ dòng lệnh mới và thực thi lệnh:
```cmd
cloudflared.exe tunnel --url http://localhost:80
```
*(Nếu đang chạy bằng tệp bat cục bộ cổng giao diện là 3000, hãy trỏ đường hầm về `http://localhost:3000`)*.

Ngay sau khi lệnh chạy, Cloudflare sẽ cung cấp một liên kết công khai có định dạng:
```text
https://xxxx-xxxx-xxxx.trycloudflare.com
```
Bất kỳ ai trên Internet mở liên kết này đều có thể truy cập đầy đủ các chức năng của trang web Sakai Studio với kết nối bảo mật tuyệt đối.

---

## 3. Phương án 2: Triển khai lên Máy chủ Đám mây Linux Độc lập

Phương án này phù hợp cho việc vận hành lâu dài trên các nhà cung cấp hạ tầng máy chủ như Linode, Hetzner, DigitalOcean hoặc AWS.

### Yêu cầu cấu hình tối thiểu
- **Bộ vi xử lý:** Tối thiểu 2 nhân CPU (Khuyến nghị từ 4 nhân).
- **Bộ nhớ trong RAM:** Tối thiểu 4 GB RAM (Khuyến nghị 8 GB trở lên khi xử lý video dung lượng cao).
- **Dung lượng lưu trữ:** 50 GB SSD trở lên.
- **Hệ điều hành:** Ubuntu 22.04 LTS hoặc Ubuntu 24.04 LTS.

### Các bước thực hiện

#### Bước 1: Cài đặt môi trường Docker trên máy chủ
Đăng nhập vào máy chủ qua kết nối SSH và chạy các lệnh:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

#### Bước 2: Tải mã nguồn dự án lên máy chủ
```bash
git clone https://github.com/TênTàiKhoản/Sakai-Studio.git
cd Sakai-Studio
```

#### Bước 3: Cấu hình tệp xác thực âm thầm (Nếu cần)
Để hệ thống tự động tải được các video yêu cầu đăng nhập hoặc video bị giới hạn độ tuổi mà người dùng trên giao diện web không cần cung cấp thông tin, người quản trị chỉ cần đặt tệp `cookies.txt` vào thư mục gốc của dự án:
```bash
cp /duong-dan/cookies.txt ./cookies.txt
```
Hệ thống sẽ tự động nhận diện và sử dụng tệp này trong nền cho mọi tác vụ tải.

#### Bước 4: Xây dựng và khởi chạy các thùng chứa
```bash
docker compose up -d --build
```

#### Bước 5: Kiểm tra trạng thái hoạt động
```bash
docker compose ps
```
Cả 5 dịch vụ (`sakai_redis`, `sakai_backend`, `sakai_celery_worker`, `sakai_frontend`, `sakai_nginx`) phải hiển thị trạng thái hoạt động bình thường.

#### Bước 6: Cài đặt chứng chỉ bảo mật cho tên miền chính thức
Nếu người dùng có tên miền riêng (ví dụ `sakai.example.com`), hãy trỏ bản ghi A của tên miền về địa chỉ IP máy chủ, sau đó cài đặt chứng chỉ bảo mật bằng công cụ Certbot:
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d sakai.example.com
```

---

## 4. Quản lý và Vận hành Hệ thống

### Xem nhật ký hoạt động thời gian thực
- Xem nhật ký của máy chủ trung tâm:
  ```bash
  docker compose logs -f backend
  ```
- Xem nhật ký của tiến trình tải và xử lý video ngầm:
  ```bash
  docker compose logs -f celery_worker
  ```

### Cập nhật phiên bản mới
Khi có mã nguồn mới từ kho lưu trữ Git:
```bash
git pull
docker compose build
docker compose up -d
```

### Sao lưu dữ liệu
Dữ liệu lịch sử tác vụ và các tệp tải về được lưu trữ an toàn trong các phân vùng lưu trữ bền vững của Docker (`sakai_db`, `sakai_downloads`). Dữ liệu này được giữ nguyên vẹn kể cả khi khởi động lại hoặc xây dựng lại thùng chứa.
