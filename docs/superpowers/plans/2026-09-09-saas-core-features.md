# Kế hoạch Kỹ thuật Triển khai Nền tảng SaaS Sakai Studio

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện 5 tính năng cốt lõi cho hệ thống dịch vụ phần mềm trực tuyến (SaaS) gồm chuẩn hóa tọa độ, nhận diện chữ tự động, luồng sự kiện thời gian thực qua Server-Sent Events, cơ sở dữ liệu quản lý tác vụ và chuẩn hóa giao diện tiếng Việt.

**Architecture:** Tách biệt độc lập dịch vụ mạng (FastAPI, Celery, SQLAlchemy) và giao diện web (Next.js, TailwindCSS) với ứng dụng máy tính, gọi các mô-đun AI làm thư viện dùng chung.

**Tech Stack:** Python 3.12, FastAPI, Celery, Redis, SQLAlchemy, SQLite, OpenCV, PaddleOCR, Next.js 16, React 19, TypeScript, TailwindCSS v4.

## Global Constraints
- Không chỉnh sửa các tệp trong `src/desktop/` hoặc làm ảnh hưởng cấu trúc ứng dụng máy tính.
- Toàn bộ giao diện và thông báo sử dụng 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn, không dùng biểu tượng cảm xúc.
- Mỗi tác vụ đều có bước kiểm thử xác nhận.

---

### Nhiệm vụ 1: Cơ sở dữ liệu và Quản lý Tác vụ (database.py)

**Files:**
- Modify: `src/web_api/api/database.py`
- Test: `test/test_saas_api.py`

**Interfaces:**
- Consumes: SQLAlchemy declarative_base, Column, Integer, String, DateTime, Text, Float
- Produces: `User`, `TaskJob`, `init_db()`, `create_task_job(db, job_id, filename, original_path, sub_areas)`, `update_task_job_progress(db, job_id, status, progress, output_path, error)`, `get_recent_jobs(db, limit=10)`

- [ ] **Bước 1: Viết kịch bản kiểm thử cơ sở dữ liệu**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận lỗi ban đầu**
- [ ] **Bước 3: Cập nhật mã nguồn database.py định nghĩa bảng và các hàm tiện ích**
- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**

---

### Nhiệm vụ 2: Chuẩn hóa Tọa độ, Nhận diện Tự động và Luồng Sự kiện (main.py)

**Files:**
- Modify: `src/web_api/api/main.py`
- Modify: `src/web_api/api/worker_tasks.py`
- Test: `test/test_saas_api.py`

**Interfaces:**
- Consumes: `database.py` (SessionLocal, TaskJob, helper functions), OpenCV (`cv2.VideoCapture`), `paddle_compat.extract_paddle_boxes`
- Produces:
  - `POST /detect-subtitles`: nhận `filename`, `timestamp`, trả về danh sách tọa độ hộp chữ nhật chuẩn hóa `[{x_pct, y_pct, w_pct, h_pct}]`.
  - `POST /process`: tự động quy đổi tọa độ chuẩn hóa sang tọa độ điểm ảnh thực tế của video gốc trước khi đẩy vào hàng đợi Celery.
  - `GET /events/{task_id}`: luồng sự kiện Server-Sent Events đẩy tiến trình thời gian thực.
  - `GET /projects`: danh sách các dự án gần đây.

- [ ] **Bước 1: Viết kịch bản kiểm thử các cổng dịch vụ mới**
- [ ] **Bước 2: Chạy kiểm thử để xác nhận lỗi ban đầu**
- [ ] **Bước 3: Triển khai các cổng dịch vụ mới và cập nhật tiến trình worker**
- [ ] **Bước 4: Chạy kiểm thử xác nhận tất cả vượt qua thành công**

---

### Nhiệm vụ 3: Chuẩn hóa Tọa độ và Thao tác Vùng chọn (VideoPlayer.tsx)

**Files:**
- Modify: `web_client/src/components/VideoPlayer.tsx`

**Interfaces:**
- Consumes: Props `videoSrc`, `onBoxesChange`, `externalBoxes`, `onClearBoxes`
- Produces: Tọa độ hộp được chuẩn hóa độc lập với tỷ lệ hiển thị trên màn hình, hỗ trợ hiển thị vùng nhận diện tự động, 100% nhãn tiếng Việt chuẩn.

- [ ] **Bước 1: Cập nhật hàm tính toán tọa độ theo tỷ lệ tương đối của video**
- [ ] **Bước 2: Bổ sung khả năng tiếp nhận hộp khoanh từ bên ngoài**
- [ ] **Bước 3: Chuẩn hóa ngôn ngữ nhãn hiển thị sang tiếng Việt**

---

### Nhiệm vụ 4: Hoàn thiện Bảng điều khiển và Kết nối Luồng Sự kiện (dashboard/page.tsx)

**Files:**
- Modify: `web_client/src/app/dashboard/page.tsx`

**Interfaces:**
- Consumes: `/upload`, `/detect-subtitles`, `/process`, `/events/{task_id}`, `/projects`
- Produces: Giao diện bảng điều khiển hoàn chỉnh bằng tiếng Việt, nút quét chữ tự động, thanh tiến độ Server-Sent Events thời gian thực, danh sách dự án thực tế.

- [ ] **Bước 1: Tích hợp nút Tự động phát hiện phụ đề**
- [ ] **Bước 2: Thay thế vòng lặp thăm dò bằng EventSource và cơ chế dự phòng**
- [ ] **Bước 3: Tải dữ liệu dự án từ cổng /projects**
- [ ] **Bước 4: Chuyển toàn bộ văn bản sang tiếng Việt chuẩn mực**

---

### Nhiệm vụ 5: Chuẩn hóa Ngôn ngữ Trang Đích (page.tsx)

**Files:**
- Modify: `web_client/src/app/page.tsx`

- [ ] **Bước 1: Chuyển ngữ toàn bộ tiêu đề, tính năng, thanh điều hướng và chân trang sang tiếng Việt chuẩn mực**
- [ ] **Bước 2: Rà soát không có từ tiếng Anh trong ngoặc đơn và không có biểu tượng cảm xúc**

---

### Nhiệm vụ 6: Kiểm thử Tích hợp Toàn diện và Nghiệm thu

- [ ] **Bước 1: Chạy kiểm thử tự động API bằng pytest**
- [ ] **Bước 2: Chạy kiểm tra tĩnh mã nguồn TypeScript bằng npm run lint**
- [ ] **Bước 3: Xác minh khởi động hệ thống qua start_saas.bat**
