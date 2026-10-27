# Tài Liệu Thiết Kế: Tối Ưu Hóa Chất Lượng Âm Thanh OmniVoice Bằng Kho Giọng Trí Tuệ Nhân Tạo Có Sẵn Và Loại Bỏ Nhân Bản Mẫu Âm Thanh

- **Ngày ban hành**: 27-10-2026
- **Trạng thái**: Đã được chấp thuận, chuẩn bị lập kế hoạch triển khai
- **Phạm vi tác động**: 
  - Giao diện người dùng: `src/desktop/ui/omnivoice_interface.py`
  - Động cơ xử lý âm thanh: `src/ai_engines/tts_engine.py`
  - Bộ kiểm thử tự động: `test/test_omnivoice_improvements.py`, `test/test_omnivoice_pipeline.py`

---

## 1. Mục Tiêu Thiết Kế Và Cơ Sở Thực Tiễn

### 1.1. Vấn đề giải quyết
Hiện tại, chất lượng âm thanh do OmniVoice tạo ra bị suy giảm do 3 nguyên nhân cốt lõi:
1. **Suy thoái nhân bản lũy kế**: Cơ chế tự động khóa giọng lấy âm thanh nhân tạo vừa sinh của câu đầu tiên làm mẫu nhân bản cho các câu tiếp theo, khiến các khuyết tật âm học và tạp âm bị khuếch đại qua từng câu thoại.
2. **Mẫu giọng trích xuất từ video bị đục ngầu**: Việc cắt đoạn 8 giây dựa trên năng lượng biên độ thô sơ thường xuyên thu phải nhạc nền, tiếng nổ hoặc tạp âm môi trường, phá hỏng vector đặc trưng giọng mẫu.
3. **Quy trình quá phức tạp**: Người dùng chỉ có nhu cầu đọc toàn bộ phụ đề bằng một giọng đọc duy nhất vẫn bị buộc phải đi qua các bước nhận diện đa nhân vật rườm rà.

### 1.2. Mục tiêu đạt được
- **Loại bỏ hoàn toàn tính năng nhân bản mẫu âm thanh**: Gỡ bỏ các nút trích xuất mẫu từ video, nút nạp tệp âm thanh ngoài và cơ chế tự khóa mẫu nhân tạo.
- **Lấy kho giọng tiếng Việt của OmniVoice làm trung tâm**: Sử dụng 22 cấu hình giọng đọc chọn lọc chuẩn phòng thu, đa dạng vùng miền (Bắc, Trung, Nam), giới tính và độ tuổi.
- **Hỗ trợ hai luồng sử dụng rành mạch**:
  - *Luồng đơn giản*: Nhập phụ đề, chọn một giọng đọc duy nhất cho toàn bộ tệp, nghe thử và xuất video ngay lập tức.
  - *Luồng phân vai*: Tự động chia nhân vật, tự động gán các giọng đọc khác nhau từ kho, cho phép người dùng tùy chỉnh thủ công từng nhân vật trong bảng phân vai và nghe thử câu thoại đầu tiên.
- **Khắc phục các lỗi âm học nền tảng**:
  - Bổ sung phiên âm số đếm tiếng Việt sang chữ trước khi đưa vào mô hình.
  - Loại bỏ hiện tượng co giãn thời lượng hai lần và gọt tỉa khoảng lặng thừa trước khi co giãn.
  - Cập nhật bộ nhớ đệm có dấu vân tay cấu hình để phản ánh ngay lập tức các thay đổi thiết lập.
  - Chuẩn hóa âm lượng đầu ra theo chuẩn phát thanh, loại bỏ hoàn toàn hiện tượng xé tiếng và nổ lách tách.

---

## 2. Kiến Trúc Hai Luồng Sử Dụng Trên Giao Diện

```
                                [Nạp Tệp Phụ Đề SRT / VTT]
                                             |
                                             v
                         +---------------------------------------+
                         | Lựa Chọn Chế Độ Lồng Tiếng Tại Bước 1 |
                         +---------------------------------------+
                                    /                 \
                                   /                   \
        [Chế Độ 1: Một Giọng Cho Toàn Bộ File]        [Chế Độ 2: Phân Vai Nhiều Nhân Vật]
                         |                                             |
                         v                                             v
        +----------------------------------+          +----------------------------------+
        | Chọn 1 giọng từ kho 22 giọng mẫu |          | Tự động nhận diện nhân vật       |
        | Bấm nút "Nghe thử giọng mẫu"     |          | Tự động gán giọng AI từ kho mẫu  |
        +----------------------------------+          +----------------------------------+
                         |                                             |
                         v                                             v
        +----------------------------------+          +----------------------------------+
        | Bỏ qua phân vai nhân vật         |          | Bước 2: Bảng Phân Vai Nhân Vật   |
        | Chuyển thẳng sang Bước 3 hoặc 5  |          | - Đổi giọng AI từ danh mục       |
        +----------------------------------+          | - Bấm "Nghe thử giọng" từng người|
                         |                            +----------------------------------+
                         |                                             |
                         \---------------------\ /---------------------/
                                                |
                                                v
                                  [Bước 5: Lồng Tiếng Và Xuất Video]
```

### 2.1. Chế Độ 1: Một Giọng Đọc Đồng Nhất Cho Toàn Bộ Phụ Đề
- Xuất hiện nổi bật ngay tại Bước 1 (Nạp Dữ Liệu).
- Cho phép người dùng đánh dấu chọn: `"Sử dụng một giọng đọc đồng nhất cho toàn bộ tệp phụ đề"`.
- Khi chế độ này được kích hoạt:
  - Hiển thị hộp danh sách chọn giọng từ kho 22 giọng mẫu tiếng Việt kèm nút `"Nghe Thử Giọng Này"`.
  - Nút `"Nhận Diện Nhân Vật Từ Video + SRT"` tự động chuyển sang trạng thái ẩn hoặc vô hiệu hóa.
  - Người dùng có thể bấm trực tiếp chuyển sang Bước 3 (Biên tập phụ đề) hoặc Bước 5 (Lồng tiếng và xuất bản) mà không cần đi qua Bước 2 (Quản lý nhân vật).
  - Toàn bộ các dòng phụ đề tự động áp dụng cấu hình giọng đọc này.

### 2.2. Chế Độ 2: Phân Vai Đa Nhân Vật Bằng Kho Giọng Trí Tuệ Nhân Tạo
- Kích hoạt khi người dùng bỏ dấu chọn chế độ đơn giọng.
- Bấm nút `"Nhận Diện Nhân Vật Từ Video + SRT"` hoặc `"Tự Động Phân Vai Theo Phụ Đề"`:
  - Hệ thống tự động phát hiện số lượng nhân vật thực tế.
  - Thay vì cắt mẫu âm thanh, hệ thống gán tuần tự mỗi nhân vật một giọng đọc khác nhau từ kho mẫu tiếng Việt (ví dụ: Người nói 1 nhận giọng Nam Thanh Niên Chuẩn, Người nói 2 nhận giọng Nữ Thanh Niên Chuẩn, Người nói 3 nhận giọng Nam Trầm Ấm...).
- Tại Bước 2 (Quản Lý Nhân Vật):
  - Bảng phân vai hiển thị rõ ràng từng nhân vật: Tên người nói, số lượng câu thoại, hộp danh sách chọn giọng AI từ kho mẫu, trạng thái phân vai.
  - Cột thao tác của mỗi nhân vật có nút `"Nghe Thử Giọng"`: Khi bấm, hệ thống tạo ra một câu đọc thử ngắn bằng chính giọng AI đã gán để người dùng thẩm định âm sắc trước khi xuất bản.
  - Loại bỏ hoàn toàn cột `"Mẫu Giọng Nguồn"` cũ vốn dùng cho tệp âm thanh nhân bản.

---

## 3. Cải Tiến Kỹ Thuật Trong Động Cơ Xử Lý Âm Thanh

### 3.1. Loại bỏ triệt để cơ chế suy thoái nhân bản
- Trong hàm `synthesize_blocks`:
  - Xóa bỏ logic tự động tạo `VoiceClonePrompt` từ câu số 1 và xóa bỏ biến lưu đệm `consistent_{spk_key}`.
  - Mỗi nhân vật khi sử dụng giọng chỉ dẫn hoặc giọng mẫu trong danh mục sẽ được giữ nguyên vẹn chuỗi thuộc tính chỉ dẫn văn bản từ đầu đến cuối tệp.
  - Đảm bảo câu số 100 hay câu số 1 đều được sinh ra từ cùng một điều kiện âm học tinh khiết, không bị thoái hóa chất lượng.

### 3.2. Chuẩn hóa số đếm và ký tự tiếng Việt trước khi tổng hợp
- Xây dựng bộ hàm `normalize_vietnamese_numbers_and_text` trong chuỗi tiền xử lý:
  - Tự động chuyển đổi các chữ số nguyên sang chữ viết tiếng Việt chuẩn xác (ví dụ: `2024` thành `hai nghìn không trăm hai mươi tư`, `15` thành `mười lăm`, `100%` thành `một trăm phần trăm`).
  - Chuyển đổi các số thập phân (ví dụ: `3.5` thành `ba phẩy năm`).
  - Chuyển đổi ngày tháng năm dạng số sang chữ viết.
  - Ngăn ngừa hiện tượng mô hình đọc chữ số bằng tiếng Anh hoặc phát ra âm thanh vô nghĩa.

### 3.3. Tối ưu hóa thuật toán co giãn thời lượng và loại bỏ co giãn kép
- Trước khi tính toán tỷ lệ co giãn thời lượng:
  - Gọt bỏ phần im lặng đầu đuôi nhân tạo của câu thoại.
  - Chỉ kích hoạt co giãn khi câu thoại thực sự dài hơn thời lượng khe phụ đề trên 5% và giới hạn tỷ lệ co giãn tối đa không vượt quá 1.15 lần để bảo toàn cấu trúc hài âm tiếng Việt.
  - Ngăn chặn triệt để hiện tượng co giãn hai lần: Nếu hàm tổng hợp đơn lẻ đã co giãn thì hàm quản lý chuỗi khối sẽ không lặp lại bước này.

### 3.4. Bổ sung dấu vân tay cấu hình vào bộ nhớ đệm
- Cập nhật hàm `_compute_block_cache_path`:
  - Đưa toàn bộ các tham số quyết định chất lượng vào khóa băm: `guidance_scale`, `num_step`, `denoise`, `voice_preset_slug`, `instruct_text`, `speed`.
  - Khi người dùng thay đổi bất kỳ tham số nào trong cài đặt hoặc đổi giọng đọc khác cho nhân vật, khóa băm sẽ thay đổi lập tức, bảo đảm hệ thống tạo âm thanh mới thay vì nạp lại tệp cũ bị méo tiếng.

### 3.5. Chuẩn hóa âm lượng đầu ra phát thanh
- Tại khâu cuối của hàm `synthesize_blocks`:
  - Quét toàn bộ mảng sóng âm tổng hợp `full_audio`.
  - Áp dụng bộ chuẩn hóa âm lượng theo mức đỉnh âm thanh an toàn `-1.0 dBFS` (biên độ đỉnh tối đa `0.891`).
  - Áp dụng bộ nắn giới hạn đỉnh mềm đối với các điểm chồng lấn âm thanh để loại bỏ hoàn toàn hiện tượng vỡ hạt sóng vuông khi ghi định dạng 16-bit.

---

## 4. Kế Hoạch Xác Minh Tự Động Và Nghiệm Thu

1. **Kiểm thử đơn vị**:
   - Xác minh chế độ một giọng đọc: Nạp tệp phụ đề, chọn giọng từ kho, xác nhận toàn bộ câu thoại được gán đúng giọng chỉ dẫn và không có bất kỳ thao tác trích xuất mẫu âm thanh nào.
   - Xác minh bảng quản lý nhân vật: Thay đổi giọng của Người nói 1 từ danh mục, bấm nghe thử và kiểm tra tệp âm thanh được tạo đúng với thuộc tính của giọng đó.
   - Xác minh bộ chuẩn hóa số tiếng Việt: Các chuỗi số đếm phức tạp được chuyển đổi thành ngữ lưu loát tiếng Việt.
   - Xác minh bộ nhớ đệm: Khi thay đổi giọng đọc hoặc hệ số hướng dẫn, khóa băm đổi mới và tệp âm thanh được tạo mới hoàn toàn.
2. **Kiểm thử tích hợp chuỗi xử lý**:
   - Chạy toàn bộ 91 kịch bản kiểm thử quy trình của hệ thống bảo đảm không có hồi quy mã nguồn.
   - Kiểm tra tệp WAV đầu ra đạt chuẩn: thời lượng khớp, không có giá trị bất thường, năng lượng đạt chuẩn và biên độ đỉnh an toàn dưới 0.90.
