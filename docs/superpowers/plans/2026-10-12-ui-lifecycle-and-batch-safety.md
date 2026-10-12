# UI Lifecycle, Batch Output Safety & Async Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khắc phục triệt để 5 nhóm vấn đề P1/P2 còn lại trên giao diện Sakai Studio (khóa toàn diện điều khiển khi OmniVoice chạy, chống ghi đè/trùng tên đầu ra khi chạy hàng loạt, chống race-condition khi nạp model bất đồng bộ, bảo mật khóa API bằng Windows DPAPI kèm trạng thái xác thực, hoàn thiện `retranslateUi()` và CI kiểm thử tự động).

**Architecture:** Tách thành 5 đợt (Waves) độc lập có kiểm thử TDD riêng cho từng đợt. Mọi luồng bất đồng bộ đều gắn định danh yêu cầu (`request_id`) hoặc quản lý vòng đời tiến trình con (`Popen`), đồng thời khóa/mở khóa toàn bộ điều khiển giao diện theo trạng thái tác vụ.

**Tech Stack:** Python 3.12, PySide6, PySide6-Fluent-Widgets, Windows DPAPI (`crypt32`), Pytest, GitHub Actions.

## Global Constraints

- Sử dụng 100% tiếng Việt chuẩn mực trên giao diện, không dùng từ tiếng Anh trong ngoặc đơn, không dùng emoji.
- Kích thước cửa sổ tối thiểu `960x600`, tự động phóng đại khi mở ứng dụng.
- Mọi thay đổi mã nguồn phải có kịch bản kiểm thử thất bại (RED) trước khi viết mã thực thi (GREEN).

---

### Task 1: Đợt 1 — Khóa toàn diện điều khiển OmniVoice khi chạy & Chuẩn hóa hủy FFmpeg

**Files:**
- Modify: `src/desktop/ui/omnivoice_interface.py`
- Modify: `src/desktop/ui/auto_pipeline_interface.py`
- Test: `test/test_auto_pipeline_reliability.py`

- [ ] **Step 1: Viết kiểm thử thất bại cho khóa toàn diện điều khiển OmniVoice và hủy tiến trình FFmpeg**
- [ ] **Step 2: Chạy pytest xác nhận trạng thái RED**
- [ ] **Step 3: Cập nhật `_set_task_running_state()` trong `omnivoice_interface.py` và `_overlay_subtitle_ffmpeg()` / `stop()` trong `auto_pipeline_interface.py`**
- [ ] **Step 4: Chạy pytest xác nhận trạng thái GREEN**

---

### Task 2: Đợt 2 — An toàn phiên nhân vật & Ngăn ghi đè / trùng tên đầu ra khi chạy hàng loạt

**Files:**
- Modify: `src/desktop/ui/omnivoice_interface.py`
- Modify: `src/desktop/ui/auto_pipeline_interface.py`
- Test: `test/test_auto_pipeline_reliability.py`

- [ ] **Step 1: Viết kiểm thử thất bại cho xóa `speaker_profiles` khi nạp SRT thủ công và giải quyết xung đột tên đầu ra hàng loạt**
- [ ] **Step 2: Chạy pytest xác nhận trạng thái RED**
- [ ] **Step 3: Cập nhật `_load_subtitle_file_internal()` và thêm `resolve_batch_output_plan()` + tùy chọn xử lý trùng tên đầu ra trong `auto_pipeline_interface.py`**
- [ ] **Step 4: Chạy pytest xác nhận trạng thái GREEN**

---

### Task 3: Đợt 3 — Chống trả kết quả sai khi nạp model bất đồng bộ & Bảo mật API key bằng DPAPI

**Files:**
- Modify: `src/desktop/ui/translation_interface.py`
- Modify: `src/desktop/ui/auto_pipeline_interface.py`
- Test: `test/test_auto_pipeline_reliability.py`

- [ ] **Step 1: Viết kiểm thử thất bại cho `request_id` khi nạp model liên tiếp và trạng thái xác thực khóa DPAPI**
- [ ] **Step 2: Chạy pytest xác nhận trạng thái RED**
- [ ] **Step 3: Thêm `request_id` vào `_start_async_fetch_models` / `_on_fetch_models_done`, mã hóa DPAPI và cờ `api_key_verified`**
- [ ] **Step 4: Chạy pytest xác nhận trạng thái GREEN**

---

### Task 4: Đợt 4 & 5 — Hoàn thiện `retranslateUi()`, kiểm thử 8 tiêu chí nghiệm thu & GitHub Actions CI

**Files:**
- Modify: `src/desktop/ui/auto_pipeline_interface.py`
- Modify: `src/desktop/ui/omnivoice_interface.py`
- Modify: `src/desktop/ui/translation_interface.py`
- Modify: `src/desktop/ui/tools_interface.py`
- Create: `.github/workflows/ci-tests.yml`
- Test: `test/test_auto_pipeline_reliability.py`

- [ ] **Step 1: Viết kiểm thử cho `retranslateUi()` và bộ 8 tiêu chí nghiệm thu**
- [ ] **Step 2: Triển khai `retranslateUi()` và tệp cấu hình GitHub Actions `.github/workflows/ci-tests.yml`**
- [ ] **Step 3: Chạy toàn bộ pytest xác nhận 100% GREEN, commit và push**
