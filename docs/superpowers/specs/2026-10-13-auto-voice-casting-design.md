# Tài Liệu Đặc Tả Thiết Kế: Hệ Thống Tự Động Phân Vai Và Chọn Giọng

- **Mã tính năng**: `auto-voice-casting`
- **Ngày lập**: 13/10/2026
- **Trạng thái**: Đã phê duyệt bản thiết kế, chuẩn bị lập kế hoạch thực thi

---

## 1. Mục Tiêu Và Bối Cảnh

Hệ thống lồng tiếng hiện tại trong Sakai Studio hỗ trợ chế độ đa người nói, tạo mẫu nhái giọng và tái sử dụng cấu hình giọng. Tuy nhiên, cơ chế nhận diện người nói hiện tại chỉ dựa trên ước tính cao độ tần số cơ bản F0 và đặc trưng phổ thô sơ. Điều này dẫn đến nguy cơ nhận diện nhầm lẫn khi người nói thay đổi cảm xúc, nói to nhỏ, hoặc khi hai người có cao độ gần nhau.

Tính năng **Tự động phân vai và chọn giọng** nâng cấp toàn diện khả năng phân biệt người nói bằng kiến trúc lai hai tầng, chuẩn hóa hồ sơ giọng của từng nhân vật, tự động gán giọng nhất quán qua mọi câu thoại và quy hoạch lại giao diện trang Lồng Tiếng theo quy trình 5 bước trực quan.

---

## 2. Kiến Trúc Kỹ Thuật Hai Tầng

### 2.1 Tầng 1: Trích Xuất Đặc Trưng Giọng Nói Ngoại Tuyến (Mặc Định)
- **Tệp nguồn mới**: `src/ai_engines/speaker_embedding_onnx.py`
- **Bộ máy thực thi**: ONNX Runtime chạy trên vi xử lý trung tâm, hoàn toàn ngoại tuyến, không yêu cầu kết nối mạng và không chiếm dụng bộ nhớ đồ họa trước khi lồng tiếng.
- **Mô hình mục tiêu**:
  - Mô hình chính: CAM++ ONNX (kích thước khoảng 28.3 megabyte, đầu ra vector đặc trưng 192 chiều).
  - Mô hình đối chứng: ECAPA-TDNN ONNX (kích thước khoảng 83.4 megabyte, 192 chiều).
- **Quy trình xử lý âm thanh**:
  1. Tách kênh âm thanh lời thoại 16 kHz dạng đơn kênh từ video hoặc tệp âm thanh nguồn.
  2. Cắt lát âm thanh theo mốc thời gian phụ đề, bổ sung đệm an toàn 0.15 giây ở hai đầu mốc thời gian để bảo toàn phụ âm đầu và âm đuôi.
  3. Lọc bỏ các đoạn im lặng và các đoạn có thời lượng thực tế dưới 0.3 giây trước khi đưa vào mô hình suy luận.
  4. Trích xuất vector đặc trưng và chuẩn hóa độ dài vector theo chuẩn L2.

### 2.2 Tầng 2: Phân Đoạn Người Nói Chuyên Sâu (Tùy Chọn Nâng Cao)
- **Tệp nguồn mới**: `src/ai_engines/speaker_diarization.py`
- Cung cấp giao diện chuẩn hóa `SpeakerDiarizationEngine` làm cầu nối chung.
- Cho phép nạp pipeline phân đoạn người nói Pyannote chạy ngoại tuyến từ thư mục mô hình cục bộ khi người dùng kích hoạt chế độ chuyên sâu trong phần cài đặt nâng cao.
- Hiển thị 4 trạng thái mô hình rõ ràng trên giao diện:
  1. Chưa cài đặt mô hình nâng cao.
  2. Đang kiểm tra cấu hình mô hình cục bộ.
  3. Sẵn sàng thực thi ngoại tuyến.
  4. Lỗi nạp mô hình hoặc không đủ tài nguyên.

### 2.3 Thuật Toán Phân Cụm Và Đối Chiếu Phụ Đề
- Tính toán ma trận khoảng cách dựa trên độ tương đồng cosin giữa các vector đặc trưng.
- Áp dụng thuật toán phân cụm phân cấp với ngưỡng khoảng cách tối ưu để tự động xác định số lượng người nói thực tế.
- Phân loại 3 mức độ tin cậy rõ ràng cho từng câu thoại:
  1. **Tin cậy cao**: Vector nằm gần trọng tâm cụm và tách biệt rõ ràng với cụm đứng thứ hai (chênh lệch khoảng cách lớn hơn 0.25).
  2. **Cần kiểm tra**: Khoảng cách giữa hai cụm gần nhau (chênh lệch dưới 0.15) hoặc âm thanh bị lẫn tạp âm.
  3. **Chưa xác định**: Đoạn thoại quá ngắn dưới 0.3 giây hoặc không trích xuất được vector. Các câu này được giữ nguyên nhãn Chưa xác định để người dùng tự xác nhận trên bảng, tuyệt đối không tự ý gán ép vào cụm của câu trước đó.

---

## 3. Cấu Trúc Dữ Liệu

### 3.1 Cấu Trúc Hồ Sơ Nhân Vật (`SpeakerVoiceProfile`)
```python
class SpeakerVoiceProfile:
    speaker_id: str                 # Định danh kỹ thuật: spk_0, spk_1...
    display_name: str               # Tên nhân vật do người dùng đặt (mặc định: Người nói 1, Người nói 2...)
    source_audio_sample: str        # Đường dẫn tệp âm thanh trích từ video có chất lượng tốt nhất
    assigned_strategy: str          # "clone" (nhái giọng nguồn) hoặc "preset" (giọng thiết kế)
    voice_preset_key: str           # Tên cấu hình giọng mẫu tiếng Việt được chọn
    voice_clone_prompt_path: str    # Đường dẫn tệp khóa mẫu nhái giọng nếu dùng chiến lược nhái giọng
    instruct: str                   # Chỉ dẫn phong cách đọc
    dialogue_count: int             # Tổng số câu thoại được phân cho nhân vật
    high_confidence_ratio: float    # Tỷ lệ số câu đạt độ tin cậy cao của nhân vật
```

### 3.2 Bổ Sung Vào `SubtitleBlock`
- `speaker`: Tên nhân vật (ví dụ: `Người nói 1`, `Người nói 2` hoặc `Chưa xác định`).
- `confidence_level`: Chuỗi trạng thái (`"high"`, `"review_needed"`, `"unassigned"`).
- `is_speaker_locked`: Cờ đánh dấu người dùng đã chủ động khóa nhân vật cho câu thoại này (khi khóa, thao tác tự động phân tích lại sẽ không ghi đè câu này).

---

## 4. Tái Thiết Kế Giao Diện Trang Lồng Tiếng

Quy hoạch lại `OmniVoiceInterface` tại `src/desktop/ui/omnivoice_interface.py` thành mô hình ngăn xếp 5 bước tuần tự:

```
[ Thanh Tiến Trình 5 Bước: 1. Nguồn Đầu Vào -> 2. Quản Lý Nhân Vật -> 3. Biên Tập Phụ Đề -> 4. Cân Chỉnh Âm Thanh -> 5. Kiểm Tra Và Xuất ]
```

### Bước 1: Nguồn Đầu Vào
- Hai thẻ nạp tệp độc lập hỗ trợ kéo thả tệp video và tệp phụ đề.
- Hiển thị trực quan thời lượng, kích thước khung hình, định dạng, số câu thoại và trạng thái khớp thời gian.
- Nếu có cả video và phụ đề: Bật nút "Tiến hành phân tích giọng nói".
- Nếu chỉ có phụ đề: Hiển thị thông báo chuyển sang chế độ phân vai theo văn bản hoặc phân vai thủ công.

### Bước 2: Quản Lý Nhân Vật (Bảng Voice Casting)
- Bảng danh sách nhân vật:
  - Cột 1: Mã và Tên nhân vật (nhấp đúp để đổi tên trực tiếp).
  - Cột 2: Số lượng câu thoại được phát hiện.
  - Cột 3: Nút nghe thử đoạn giọng gốc cắt từ video.
  - Cột 4: Hộp chọn giọng lồng tiếng (nhái giọng nguồn hoặc chọn từ danh sách giọng mẫu tiếng Việt).
  - Cột 5: Nút nghe thử giọng lồng tiếng (tổng hợp nhanh 1 câu mẫu ngắn).
  - Cột 6: Mức độ tin cậy tổng thể kèm cảnh báo nếu có câu chưa xác định.
- Nút tác vụ hàng loạt: "Tự động phân tích lại", "Đặt lại toàn bộ giọng", "Nghe thử toàn bộ nhân vật".

### Bước 3: Biên Tập Phụ Đề
- Bảng danh sách câu thoại tích hợp:
  - Cột Nhân vật hiển thị thẻ màu sắc riêng cho từng người nói.
  - Nhấp vào cột để chọn nhanh nhân vật khác hoặc bôi đen nhiều hàng để gán hàng loạt.
  - Ô tìm kiếm và bộ lọc nhanh: lọc theo nhân vật cụ thể, lọc các câu "Cần kiểm tra" hoặc "Chưa xác định".
- Trình xem trước video thu nhỏ tự động nhảy đến mốc thời gian của câu thoại đang chọn.

### Bước 4: Cân Chỉnh Âm Thanh
- Bảng điều khiển trực quan cho các thông số thường dùng:
  - Tốc độ đọc (0.8x đến 1.5x).
  - Tỷ lệ âm lượng video gốc và âm lượng giọng lồng tiếng.
  - Chế độ tự động hạ âm lượng video gốc khi có giọng đọc.
- Ngăn **Cài đặt nâng cao** có thể thu gọn: Chứa toàn bộ các tham số kỹ thuật khuếch tán, số bước khử nhiễu, nhiệt độ lấy mẫu và lựa chọn bộ xử lý đồ họa.

### Bước 5: Kiểm Tra Chất Lượng Và Xuất Video
- Bảng kiểm tra tự động trước khi xuất:
  - Báo cáo số câu chưa gán nhân vật.
  - Cảnh báo các câu có thời lượng đọc vượt quá giới hạn mốc thời gian phụ đề.
- Thanh tiến trình tổng hợp chi tiết theo từng câu thoại.
- Nút hủy an toàn giải phóng tiến trình nền.
- Nút mở tệp video đầu ra và mở thư mục chứa tệp sau khi hoàn tất.

*Quy tắc bảo toàn dữ liệu*: Người dùng có thể quay lại bất kỳ bước nào trước đó để chỉnh sửa mà không làm mất dữ liệu đã phân tích hoặc các chỉnh sửa đã thực hiện. Chỉ phân tích lại khi người dùng bấm nút yêu cầu phân tích lại hoặc thay đổi tệp đầu vào.

---

## 5. Quản Lý Tài Nguyên Và Tối Ưu Hiệu Năng

1. **Bộ nhớ đồ họa**:
   - Bước nhận diện và trích xuất vector đặc trưng chạy hoàn toàn trên vi xử lý trung tâm qua ONNX Runtime CPU.
   - Chỉ nạp mô hình tổng hợp giọng nói lên bộ xử lý đồ họa khi người dùng nhấn nghe thử hoặc bắt đầu xuất video ở Bước 5.
2. **Bộ nhớ đệm thông minh**:
   - Lưu trữ các vector đặc trưng và tệp âm thanh mẫu trong thư mục tạm có gắn mã băm của tệp video nguồn.
   - Khi người dùng đổi giọng của Người nói 2, hệ thống chỉ hủy bộ nhớ đệm âm thanh của Người nói 2, không phải tổng hợp lại âm thanh của các nhân vật khác.

---

## 6. Kế Hoạch Kiểm Thử Tự Động

Thực thi bộ kiểm thử trong `test/test_omnivoice_pipeline.py`:
1. `test_onnx_speaker_embedding_extraction`: Kiểm tra trích xuất vector đặc trưng với kích thước chuẩn 192 chiều.
2. `test_cosine_clustering_and_confidence_levels`: Kiểm tra phân cụm chính xác 2-3 người nói và gán đúng 3 mức độ tin cậy.
3. `test_short_and_silent_segments_marked_unassigned`: Đảm bảo các đoạn dưới 0.3 giây được đánh dấu Chưa xác định, không gán bừa vào cụm lân cận.
4. `test_ui_wizard_preserves_state_on_backward_navigation`: Kiểm tra việc điều hướng qua lại giữa các bước không làm mất nhãn nhân vật và danh sách phụ đề.
5. `test_pre_export_quality_checks_and_warnings`: Xác minh các cảnh báo trước khi xuất hoạt động chính xác khi còn câu chưa có nhân vật.
