# Xóa Phụ Đề UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tổ chức lại thẻ Xóa Phụ Đề và Cài Đặt Nâng Cao thành luồng công việc 4 bước mạch lạc (Khoanh vùng -> Mô hình AI -> Độ nét -> Dịch thuật) bằng cách sử dụng các `SettingCardGroup` được đánh số rõ ràng trên giao diện.

**Architecture:** Sử dụng trực tiếp `SettingCardGroup` có sẵn trong `qfluentwidgets` để đổi tên và gom nhóm lại các tính năng đang phân tán trong `ui/advanced_setting_interface.py`. Di chuyển các `SettingCard` vào đúng nhóm logic của chúng. Không thay đổi logic hoạt động cốt lõi của các chức năng AI và config.

**Tech Stack:** Python, PySide6, qfluentwidgets

## Global Constraints

- 100% tiếng Việt, không dùng từ tiếng Anh trong ngoặc đơn, không dùng biểu tượng cảm xúc.
- Giữ nguyên cấu trúc dữ liệu `config` cũ để tránh mất cấu hình của người dùng.

---

### Task 1: Cấu trúc lại thẻ Tab 2 (Chỉnh Sửa Video)

**Files:**
- Modify: `e:\Sakai-Studio\ui\advanced_setting_interface.py`

**Interfaces:**
- Consumes: Các thẻ `SettingCard` hiện tại.
- Produces: Nhóm thẻ mới được sắp xếp theo Bước 1, Bước 2, Bước 3.

- [ ] **Step 1: Đổi tên và tạo nhóm Bước 1 (Khởi tạo và Khoanh vùng)**

Sửa mã nguồn để gom các chức năng khoanh vùng vào chung một `SettingCardGroup` tên là "Bước 1: Khởi tạo và Khoanh vùng". Đưa `auto_tighten_card` vào đây.

- [ ] **Step 2: Tạo nhóm Bước 2 (Lựa chọn Mô hình Trí tuệ Nhân tạo)**

Tạo nhóm "Bước 2: Lựa chọn Mô hình Trí tuệ Nhân tạo". Di chuyển các thẻ cấu hình AI như `tracker_algorithm_combo`, `hardware_acceleration`, `auto_hardware_tuning`, `gpu_video_encoding` vào nhóm này.

- [ ] **Step 3: Tạo nhóm Bước 3 (Cấu hình Chất lượng và Độ nét)**

Tạo nhóm "Bước 3: Cấu hình Chất lượng và Độ nét". Di chuyển `preserve_color_card`, `poisson_blending`, `temporal_smoothing`, `sharpen_inpainted_area`, `temporal_smoothing_radius` vào nhóm này. Cập nhật `setup_layout` để đưa các group này vào giao diện.

- [ ] **Step 4: Chạy thử giao diện**

Chạy câu lệnh `venv\Scripts\python.exe gui.py` để kiểm tra. Đảm bảo giao diện Tab 2 hiển thị đúng thứ tự Bước 1, 2, 3 và không bị lỗi.

- [ ] **Step 5: Commit mã nguồn**

```bash
git add ui/advanced_setting_interface.py
git commit -m "refactor(ui): reorganize video tab into workflow steps 1-3"
```

### Task 2: Cấu trúc lại thẻ Tab 3 (Trích xuất SRT)

**Files:**
- Modify: `e:\Sakai-Studio\ui\advanced_setting_interface.py`

**Interfaces:**
- Consumes: Các thẻ `SettingCard` ở Tab 3.
- Produces: Nhóm thẻ "Bước 4: Xử lý Phụ đề Nâng cao".

- [ ] **Step 1: Đổi tên nhóm Bước 4**

Đổi tên tham số khởi tạo `srt_feature_group` thành "Bước 4: Trích xuất và Dịch thuật Phụ đề (Tùy chọn)".

- [ ] **Step 2: Đổi tên nhóm thông số kỹ thuật (Tùy chọn)**

Đổi tên tham số khởi tạo `subtitle_detection_group` thành "Thông số kỹ thuật nhận diện chữ (Nâng cao)" để tách biệt rõ với luồng chính.

- [ ] **Step 3: Khớp ngôn ngữ chuẩn**

Duyệt qua tất cả các label để đảm bảo sử dụng 100% tiếng Việt, không dùng emoji và không dùng tiếng Anh trong ngoặc.

- [ ] **Step 4: Chạy thử tổng thể**

Khởi động lại ứng dụng `venv\Scripts\python.exe gui.py`, kiểm tra toàn bộ luồng giao diện ở Tab 3. Đóng ứng dụng khi đã xác nhận hoàn hảo.

- [ ] **Step 5: Commit mã nguồn**

```bash
git add ui/advanced_setting_interface.py
git commit -m "refactor(ui): rename srt tab groups to match workflow step 4"
```
