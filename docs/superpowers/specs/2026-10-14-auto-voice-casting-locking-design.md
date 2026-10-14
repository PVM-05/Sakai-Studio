# Thiết kế Hệ thống: Tự động phân vai và khóa giọng nhân vật xuyên suốt video

## 1. Mục tiêu và Bối cảnh
- **Mục tiêu:** Đảm bảo khi người dùng đưa video và tệp phụ đề vào Sakai Studio, hệ thống tự động nhận diện các nhân vật bằng giọng nói (CAM++ ONNX), tự tạo mẫu giọng sạch cho từng nhân vật, cho phép chọn hoặc khóa giọng lồng tiếng, và giữ đúng một chất giọng duy nhất cho từng nhân vật từ đầu đến cuối video.
- **Phần cứng mục tiêu:** Card đồ họa rời NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), vi xử lý Intel Core i5-12500H, hệ điều hành Windows 11.
- **Yêu cầu giao diện và ngôn ngữ:** 100% tiếng Việt chuẩn mực, không sử dụng từ tiếng Anh trong ngoặc đơn, không sử dụng biểu tượng cảm xúc.

---

## 2. Kiến trúc Hệ thống

### 2.1. Cấu trúc dữ liệu Hồ sơ giọng nhân vật (`CharacterVoiceProfile`)
Hệ thống sử dụng đối tượng quản lý tập trung thay vì lưu rải rác trên từng dòng phụ đề:
```python
@dataclass
class CharacterVoiceProfile:
    speaker_id: str             # Mã nhân vật cố định, ví dụ: "speaker_01", "speaker_02"
    display_name: str           # Tên hiển thị người dùng có thể chỉnh sửa: "Nam chính", "Bác sĩ"
    utterance_count: int = 0    # Số câu thoại trong video
    total_duration: float = 0.0 # Tổng thời lượng nói (giây)
    source_audio_path: str = "" # Tệp âm thanh mẫu nguyên bản trích xuất từ video (ưu tiên câu đơn sạch >= 2.5s)
    method: str = "clone"       # Phương pháp: "clone" (Nhân bản từ video) hoặc "preset" (Danh mục tiếng Việt)
    preset_name: str = ""       # Tên giọng mẫu từ danh mục (ví dụ: "Nam miền Bắc trầm ấm")
    voice_instruct: str = ""    # Chỉ dẫn âm sắc bổ trợ
    status: str = "locked"      # Trạng thái: "locked" (Đã khóa), "review_needed" (Cần kiểm tra), "unassigned" (Chưa gán)
    custom_audio_path: str = "" # Đường dẫn tệp âm thanh bên ngoài do người dùng tự nạp (nếu có)
```

### 2.2. Vòng đời dữ liệu và Quy trình xử lý 5 bước
1. **Bước 1 — Nạp dữ liệu:**
   - Đầu vào bắt buộc gồm Tệp video và Tệp phụ đề (SRT, VTT, ASS) chứa mốc thời gian.
   - Trích xuất luồng âm thanh 16 kHz mono phục vụ nhận diện.
2. **Bước 2 — Nhận diện và Phân cụm người nói:**
   - Cắt các đoạn âm thanh theo từng mốc thời gian phụ đề.
   - Trích xuất vector đặc trưng 192 chiều bằng mô hình CAM++ ONNX trên GPU RTX 4050 (giới hạn bộ nhớ 512–768 MiB).
   - Gom cụm cosine tự động và gán nhãn cố định (`speaker_01`, `speaker_02`...).
   - Giải phóng phiên làm việc của mô hình CAM++ để trả lại toàn bộ VRAM cho mô hình OmniVoice.
3. **Bước 3 — Tạo hồ sơ giọng chuẩn cho từng nhân vật:**
   - Với mỗi nhân vật, tìm câu thoại sạch nhất có độ tin cậy cao và độ tương đồng lớn nhất với tâm cụm.
   - Ưu tiên chọn câu đơn nguyên bản từ 2,5 giây trở lên (single-take clean reference) để tránh hiện tượng cắt ghép lẫn tạp âm chuyển cảnh.
   - Tạo đối tượng `CharacterVoiceProfile`, khởi tạo phương pháp mặc định là nhân bản từ video và đánh dấu trạng thái đã khóa.
4. **Bước 4 — Bảng hồ sơ giọng và Đồng bộ giao diện:**
   - Bảng hồ sơ giọng hiển thị danh sách nhân vật kèm các điều khiển trực tiếp trên từng dòng:
     - Ô đổi tên nhân vật.
     - Hộp chọn phương pháp và giọng đọc: "Nhân bản giọng từ video" hoặc các giọng trong danh mục tiếng Việt chuẩn.
     - Nút chọn tệp âm thanh ngoài.
     - Nút nghe thử mẫu giọng.
     - Nhãn trạng thái đã khóa.
   - Khi người dùng thay đổi giọng của một nhân vật trên bảng hồ sơ, toàn bộ câu thoại của nhân vật đó được cập nhật đồng bộ tức thì.
   - Khi người dùng đổi nhân vật của một câu thoại trên bảng phụ đề, câu thoại đó nhận toàn bộ cấu hình giọng của nhân vật mới, xóa sạch dữ liệu tham chiếu cũ.
5. **Bước 5 — Khóa gợi ý âm sắc và Lồng tiếng nhất quán:**
   - Trước khi bắt đầu lồng tiếng câu đầu tiên, bộ điều khiển duyệt qua danh sách hồ sơ nhân vật và tạo trước đối tượng gợi ý âm sắc (`VoiceClonePrompt`) cho từng nhân vật, lưu vào bộ nhớ đệm `_speaker_prompt_cache`.
   - Nếu việc tạo gợi ý âm sắc từ mẫu video thất bại: Tự động chuyển hồ sơ nhân vật sang giọng mẫu tiếng Việt dự phòng, ghi nhận cảnh báo và khóa phương án dự phòng cho toàn bộ câu của nhân vật đó.
   - Mọi câu thoại thuộc cùng nhân vật khi tổng hợp đều dùng đúng đối tượng gợi ý âm sắc đã khóa, đảm bảo chất giọng đồng nhất tuyệt đối từ đầu đến cuối video.

---

## 3. Thiết kế Giao diện Bảng hồ sơ giọng nhân vật

Tại trang phân vai (Bước 1/Bước 2):
- **Cột 0 — Mã nhân vật:** Nhãn văn bản cố định (ví dụ: `speaker_01`).
- **Cột 1 — Tên nhân vật:** Ô nhập văn bản, chỉnh sửa tự động đồng bộ sang bảng phụ đề.
- **Cột 2 — Số câu thoại:** Thống kê số lượng câu thoại và thời lượng.
- **Cột 3 — Mẫu giọng nguồn:** Tên tệp âm thanh trích từ video kèm nút bấm chọn tệp âm thanh bên ngoài.
- **Cột 4 — Phương pháp tạo giọng:** Hộp chọn trực tiếp trên bảng:
  - Nhân bản giọng từ video
  - Nam miền Bắc trầm ấm
  - Nam miền Nam truyền cảm
  - Nữ miền Bắc ngọt ngào
  - Nữ miền Nam tự nhiên
  - Các giọng chuẩn khác trong danh mục tiếng Việt
- **Cột 5 — Trạng thái:** Hiển thị "Đã khóa", "Cần kiểm tra" hoặc "Chưa xác định".
- **Cột 6 — Thao tác:** Nút "Nghe thử" mẫu âm thanh của nhân vật.

---

## 4. Kế hoạch Kiểm thử và Tiêu chí Nghiệm thu

1. **Kiểm thử đơn vị và hồi quy:**
   - Kiểm tra khởi tạo và đồng bộ `CharacterVoiceProfile`.
   - Kiểm tra đổi phương pháp giọng từ "Nhân bản" sang "Danh mục tiếng Việt" tự động cập nhật mọi câu thoại của nhân vật đó.
   - Kiểm tra khóa gợi ý âm sắc và tái sử dụng bộ đệm xuyên suốt video.
   - Kiểm tra cơ chế tự động chuyển dự phòng sang danh mục tiếng Việt khi tệp mẫu lỗi, đảm bảo mô hình không bao giờ chạy không có điều kiện âm sắc.
2. **Tiêu chí nghiệm thu:**
   - 100% các bài kiểm thử tự động vượt qua (bao gồm toàn bộ bộ kiểm thử tích hợp 71 bài hiện có và các bài mới).
   - Giao diện và thông báo 100% tiếng Việt chuẩn mực, không có từ tiếng Anh trong ngoặc đơn, không có biểu tượng cảm xúc.
