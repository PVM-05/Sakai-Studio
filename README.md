# Sakai Studio - Video Subtitle Remover

**Sakai Studio - Video Subtitle Remover** là công cụ chuyên nghiệp, tự động phát hiện và xóa sạch phụ đề, logo hoặc hình mờ (watermark) khỏi video bằng các mô hình AI tiên tiến (LaMa, STTN, ProPainter) kết hợp tối ưu phần cứng.

---

## 🚀 Các Tính năng Nổi bật & Cải tiến

### 1. Công nghệ Xóa chữ Siêu sạch (Stroke-level Masking)
*   **Mặt nạ Nét chữ Thích ứng:** Tự động giãn nở mặt nạ thông minh theo độ dày nét chữ thay vì bôi đen cả vùng hộp. Giúp xóa chữ sạch nhất, loại bỏ hoàn toàn bóng mờ (ghosting) và giữ nguyên vẹn chi tiết nền xung quanh.

### 2. Tối ưu hóa Tốc độ Vượt trội (Skip Frame & Cropped OCR)
*   **Cropped OCR:** Chỉ quét chữ trên vùng phụ đề được chọn thay vì quét toàn bộ khung hình Full HD/4K, tăng tốc độ nhận diện lên gấp nhiều lần.
*   **Bỏ qua Khung hình Trống:** Tự động lọc các vùng nền tĩnh/dải đen. Nếu không có chữ, hệ thống tự động bỏ qua mô hình inpaint và chép trực tiếp khung hình gốc. Tốc độ xử lý video tăng từ **2 - 5 lần** (200% - 500%).

### 3. Tăng tốc Xuất Video bằng GPU (FFmpeg Hardware Encoding)
*   Tự động dò tìm và tận dụng chip mã hóa phần cứng trên card đồ họa (**NVIDIA NVENC, AMD AMF, Intel QSV**) để ghi và xuất video thành phẩm.
*   Tốc độ xuất video nhanh gấp **5 - 10 lần** so với sử dụng CPU mã hóa thông thường.

### 4. Bộ lọc Làm nét Vùng xóa (Sharpen Inpainted Area)
*   Áp dụng bộ lọc **Unsharp Mask (USM)** thông minh cục bộ trên vùng phụ đề đã xóa để bù đắp các chi tiết mờ nhẹ, giúp vùng vẽ lại có độ nét tiệp hoàn toàn vào chất lượng chung của video gốc.

### 5. Quản lý Bộ nhớ đệm & Tự động Tinh chỉnh GPU
*   **Model Caching:** Cache model trên VRAM/RAM giúp thời gian nạp model từ video thứ hai giảm xuống còn **0 giây**.
*   **Auto Hardware Tuning:** Tự động tính toán dung lượng VRAM hiện tại để điều chỉnh hàng đợi frame (`MaxLoadNum`), ngăn chặn hoàn toàn lỗi tràn bộ nhớ đồ họa (Out of Memory) trên các dòng máy cấu hình yếu.

---

## 🛠️ Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu Hệ thống
*   Hệ điều hành: Windows 10/11 64-bit.
*   Python 3.8 - 3.11.
*   GPU NVIDIA có hỗ trợ CUDA (Khuyên dùng để đạt tốc độ tốt nhất).

### Các bước cài đặt
1. **Tải mã nguồn:**
   ```bash
   git clone https://github.com/PVM-05/Sakai-Studio.git
   cd Sakai-Studio
   ```

2. **Tạo môi trường ảo (Khuyên dùng):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng:**
   ```bash
   python gui.py
   ```

---

## ⚙️ Hướng dẫn Sử dụng nhanh trên Giao diện
1. Kéo thả trực tiếp video hoặc danh sách video cần xử lý vào cửa sổ chính (Hỗ trợ kéo thả hàng loạt).
2. Vẽ hoặc khoanh vùng phụ đề cần xóa trên khung xem trước.
3. Nếu xử lý hàng loạt, click chuột phải vào danh sách nhiệm vụ và chọn **"Áp dụng vùng xóa sub cho tất cả nhiệm vụ"**.
4. Truy cập **Cài đặt nâng cao** để gạt bật **Tự động tối ưu hóa GPU** và **Tăng tốc xuất video bằng GPU**.
5. Nhấn **Bắt đầu (Start)** để chạy tiến trình.

---

## 📄 Giấy phép & Bản quyền
Dự án được phát triển và tối ưu hóa dựa trên mã nguồn mở. 
Mọi thông tin phản hồi hoặc báo lỗi vui lòng gửi trực tiếp lên [Sakai Studio Issues](https://github.com/PVM-05/Sakai-Studio/issues).
