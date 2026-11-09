# Kế hoạch Thực thi: Gỡ bỏ Hoàn toàn Phân hệ SaaS

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gỡ bỏ hoàn toàn phân hệ web SaaS (giao diện Next.js, API FastAPI, Celery, Redis, Docker, kịch bản đám mây và thư viện phụ thuộc liên quan) để khôi phục Sakai Studio về trạng thái phần mềm máy tính độc lập, tinh gọn và thuần khiết.

**Architecture:** Sakai Studio là ứng dụng máy tính cục bộ độc lập (PySide6 / PyQt). Phân hệ SaaS được phát triển tách biệt dưới dạng lớp giao diện web bổ sung trên `web_client/` và `src/web_api/`. Gỡ bỏ sạch sẽ toàn bộ các thư mục, tệp cấu hình, kịch bản khởi chạy và bài kiểm thử dành riêng cho SaaS mà không tác động tới tầng giao diện máy tính `src/desktop/` hay các động cơ xử lý âm thanh và video cốt lõi.

**Tech Stack:** Python 3.12, PySide6, OpenCV, ONNX Runtime, PyTorch, FFmpeg.

## Global Constraints

- Tuân thủ nghiêm ngặt quy tắc thiết kế: 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thông báo.
- Bảo đảm 100% bộ kiểm thử tự động cốt lõi của ứng dụng máy tính (xóa phụ đề, dịch thuật, lồng tiếng OmniVoice) tiếp tục vượt qua thành công sau khi gỡ bỏ.

---

### Task 1: Gỡ bỏ các tệp kiểm thử tự động dành riêng cho SaaS

**Files:**
- Delete: `test/test_auth_saas.py`
- Delete: `test/test_extract_saas.py`
- Delete: `test/test_saas_api.py`
- Delete: `test/test_ytdlp_saas.py`

**Interfaces:**
- Consumes: Không phụ thuộc.
- Produces: Bộ kiểm thử chỉ còn lại các bài test dành riêng cho ứng dụng máy tính.

- [ ] **Step 1: Xóa 4 tệp kiểm thử SaaS**
  Xóa `test/test_auth_saas.py`, `test/test_extract_saas.py`, `test/test_saas_api.py`, `test/test_ytdlp_saas.py`.
- [ ] **Step 2: Xác minh danh sách kiểm thử còn lại**
  Chạy `pytest --collect-only` kiểm tra không còn bài kiểm thử nào trỏ vào `src.web_api`.
- [ ] **Step 3: Tạo commit lưu trữ bước 1**
  Commit: `refactor(saas): go bo cac tep kiem thu tu dong saas`

---

### Task 2: Gỡ bỏ thư mục giao diện web Next.js (`web_client/`)

**Files:**
- Delete: Thư mục `web_client/`

**Interfaces:**
- Consumes: Không phụ thuộc.
- Produces: Loại bỏ hoàn toàn mã nguồn giao diện web.

- [ ] **Step 1: Xóa toàn bộ thư mục `web_client/`**
  Thực hiện xóa toàn bộ thư mục `web_client/` khỏi cây mã nguồn.
- [ ] **Step 2: Xác nhận thư mục đã được xóa sạch**
  Kiểm tra lệnh kiểm tra tệp xác nhận `web_client` không còn tồn tại.
- [ ] **Step 3: Tạo commit lưu trữ bước 2**
  Commit: `refactor(saas): go bo thu muc giao dien web nextjs`

---

### Task 3: Gỡ bỏ thư mục dịch vụ API và tác vụ nền (`src/web_api/`)

**Files:**
- Delete: Thư mục `src/web_api/`

**Interfaces:**
- Consumes: Không phụ thuộc.
- Produces: Loại bỏ hoàn toàn tầng backend FastAPI và Celery.

- [ ] **Step 1: Xóa toàn bộ thư mục `src/web_api/`**
  Thực hiện xóa toàn bộ thư mục `src/web_api/` khỏi cây mã nguồn.
- [ ] **Step 2: Xác minh không còn tệp nào sót lại trong `src/web_api`**
  Kiểm tra cây thư mục `src/` xác nhận chỉ còn: `core`, `ai_engines`, `desktop`, `video_ops`, `config`, `logs`.
- [ ] **Step 3: Tạo commit lưu trữ bước 3**
  Commit: `refactor(saas): go bo thu muc dich vu web api va celery worker`

---

### Task 4: Gỡ bỏ các tệp cấu hình triển khai, kịch bản khởi chạy và cơ sở dữ liệu SaaS

**Files:**
- Delete: `docker-compose.yml`
- Delete: `Dockerfile.backend`
- Delete: `Dockerfile.frontend`
- Delete: `nginx.conf`
- Delete: `requirements-docker.txt`
- Delete: `start_saas.bat`
- Delete: `chay_duong_ham_cloudflare.bat`
- Delete: `sakai_saas.db`
- Delete: `docs/superpowers/plans/2026-09-09-saas-core-features.md`

**Interfaces:**
- Consumes: Không phụ thuộc.
- Produces: Cây thư mục gốc sạch sẽ, chỉ giữ lại `start.bat` phục vụ ứng dụng máy tính.

- [ ] **Step 1: Xóa các tệp cấu hình đám mây, kịch bản khởi chạy và cơ sở dữ liệu**
  Xóa các tệp liệt kê ở trên.
- [ ] **Step 2: Xác nhận cây thư mục gốc chỉ còn các tệp thiết yếu của ứng dụng máy tính**
  Kiểm tra danh sách tệp tại thư mục gốc.
- [ ] **Step 3: Tạo commit lưu trữ bước 4**
  Commit: `refactor(saas): go bo cau hinh trien khai dam may, kich ban va co so du lieu saas`

---

### Task 5: Loại bỏ thư viện phụ thuộc SaaS và dọn dẹp quy tắc bỏ qua theo dõi

**Files:**
- Modify: `requirements.txt:99-106`
- Modify: `.gitignore:29-30`

**Interfaces:**
- Consumes: Tệp `requirements.txt` và `.gitignore`.
- Produces: Danh mục thư viện Python sạch không chứa `fastapi`, `uvicorn`, `celery`, `redis`, `sqlalchemy`, `psycopg2-binary`, `stripe`.

- [ ] **Step 1: Cắt bỏ khối SaaS Dependencies khỏi `requirements.txt`**
  Loại bỏ các dòng từ 99 đến 106 (`fastapi`, `uvicorn`, `celery`, `redis`, `sqlalchemy`, `psycopg2-binary`, `stripe`).
- [ ] **Step 2: Xóa bỏ các dòng trỏ tới `src/web_api/` trong `.gitignore`**
  Xóa bỏ `src/web_api/uploads/` và `src/web_api/downloads/`.
- [ ] **Step 3: Tạo commit lưu trữ bước 5**
  Commit: `refactor(saas): loai bo goi phu thuoc saas trong requirements va don dep gitignore`

---

### Task 6: Xác minh chất lượng và tính toàn vẹn của ứng dụng máy tính

**Files:**
- Test: Tất cả các bộ kiểm thử tự động của ứng dụng máy tính.

**Interfaces:**
- Consumes: Toàn bộ mã nguồn ứng dụng máy tính.
- Produces: Xác nhận trạng thái 263 bài kiểm thử tự động tiếp tục đạt 100%.

- [ ] **Step 1: Chạy toàn bộ bộ kiểm thử tự động máy tính**
  Chạy lệnh `pytest` với các tệp kiểm thử cốt lõi: `test_auto_pipeline_reliability.py`, `test_omnivoice_improvements.py`, `test_omnivoice_pipeline.py`, `test_srt_translator.py`, `test_subtitle_and_logo_remover.py`, `test_home_interface.py`.
- [ ] **Step 2: Đẩy các commit lên kho lưu trữ từ xa**
  Chạy `git push origin main`.
- [ ] **Step 3: Xác minh trạng thái kiểm thử tự động trên GitHub Actions**
  Theo dõi quy trình CI trên GitHub Actions đảm bảo chuyển màu xanh hoàn toàn.
