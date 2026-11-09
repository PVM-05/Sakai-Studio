# Kế Hoạch Triển Khai Cải Tiến Thuật Toán Giai Đoạn 1

## 1. Mục Tiêu Và Phạm Vi
Giai đoạn 1 tập trung giải quyết triệt để các nguyên nhân trực tiếp làm giảm chất lượng đầu ra của hai tính năng cốt lõi:
1. **Xóa phụ đề video**:
   - **Mục tiêu ưu tiên 0-1**: Tạo mặt nạ bám sát nét chữ đa giác và mở rộng viền mặt nạ theo trục thời gian để không sót nét chữ khi phụ đề xuất hiện hoặc mờ dần.
   - **Mục tiêu ưu tiên 0-2**: Kích hoạt cổng kiểm định chất lượng tự động với cơ chế nhận diện lại nét chữ tồn dư và tự động thử lại đối với các đoạn khung hình chưa đạt yêu cầu.
2. **Lồng tiếng nhân vật**:
   - **Mục tiêu ưu tiên 0-1**: Chuẩn hóa toàn diện tệp âm thanh đối chiếu mẫu và xây dựng thang điểm đánh giá chất lượng âm thanh tham chiếu trước khi khóa giọng nhân vật.

---

## 2. Kế Hoạch Thực Hiện Chi Tiết

### Nhóm 1: Xóa Phụ Đề Video

#### Nhiệm vụ 1.1: Tạo mặt nạ theo nét chữ đa giác và mở rộng theo thời gian
- **Tệp chỉnh sửa**: `src/ai_engines/detection.py`, `src/video_ops/tracking.py`, `src/core/main.py`, `src/core/processor.py`.
- **Nội dung thực hiện**:
  1. Bổ sung phương thức `detect_polygons(frame)` trong `SubtitleDetector`: Trích xuất trực tiếp danh sách các tọa độ đa giác của từng từ hoặc dòng chữ từ kết quả nhận diện, thay vì chỉ ép về khung chữ nhật trục tọa độ.
  2. Bổ sung cơ chế đo độ dày nét chữ thích ứng: Sử dụng thuật toán biến đổi khoảng cách hình học để tự động điều chỉnh độ giãn nở mặt nạ phù hợp với kích thước font chữ thay vì dùng giá trị cố định.
  3. Mở rộng mặt nạ theo thời gian: Khi phát hiện chữ xuất hiện tại khung hình $N$, tự động kích hoạt mặt nạ đệm cho các khung hình lân cận $N \pm 2$ để triệt tiêu hiện tượng nhòe mờ khi chữ bắt đầu xuất hiện hoặc biến mất.
  4. Nội suy tọa độ theo dõi: Tận dụng bộ lọc dự đoán để điền khuyết vùng mặt nạ tại các khung hình mà bộ nhận diện bỏ sót ngắn hạn.

#### Nhiệm vụ 1.2: Cổng kiểm định chất lượng tự động và nhận diện chữ tồn dư
- **Tệp chỉnh sửa**: `src/core/quality.py`, `src/core/quality_gate.py`, `src/core/main.py`, `src/core/processor.py`.
- **Nội dung thực hiện**:
  1. Bổ sung hàm đo lường nét chữ tồn dư thực tế qua bộ nhận diện quang học trên các vùng mặt nạ cũ: Nếu phát hiện chuỗi ký tự còn đọc được, đánh dấu đoạn video chưa đạt chuẩn.
  2. Kết nối hàm sinh cấu hình thử lại vào luồng xử lý chính: Khi kiểm định chất lượng không đạt, tự động kích hoạt tối đa hai lần xử lý lại với thang điều chỉnh tăng cường độ giãn mặt nạ và làm mượt theo thời gian.
  3. Cơ chế xử lý cục bộ: Chỉ chạy lại đúng phạm vi các đoạn khung hình bị lỗi, không xử lý lại toàn bộ video để tối ưu hóa thời gian.

---

## 3. Kế Hoạch Thực Hiện Chi Tiết

### Nhóm 2: Lồng Tiếng Nhân Vật

#### Nhiệm vụ 2.1: Chuẩn hóa tự động và bộ chấm điểm âm thanh tham chiếu
- **Tệp tạo mới**: `src/ai_engines/ref_audio_qc.py`.
- **Tệp chỉnh sửa**: `src/ai_engines/tts_engine.py`, `src/desktop/ui/omnivoice_interface.py`.
- **Nội dung thực hiện**:
  1. Xây dựng quy trình chuẩn hóa âm thanh mẫu tự động:
     - Tự động chuyển đổi tần số lấy mẫu về 24000 héc đơn kênh.
     - Cân bằng năng lượng âm lượng chuẩn mực tương đương mức -16 LUFS.
     - Tự động cắt bỏ các đoạn im lặng ở đầu và cuối tệp âm thanh.
     - Khử nhiễu quang phổ nhẹ nhàng và lọc bỏ nhạc nền nếu có tạp âm.
  2. Xây dựng bộ chấm điểm âm thanh tham chiếu theo thang điểm 100:
     - Đo lường thời lượng hiệu dụng (tối ưu từ 8 đến 20 giây).
     - Đo tỷ lệ tín hiệu trên tạp âm.
     - Kiểm tra hiện tượng vỡ tiếng và tỷ lệ khoảng lặng.
     - Cảnh báo người dùng khi điểm số dưới 40 và từ chối khóa mẫu giọng không đạt chất lượng.
  3. Lưu trữ mẫu đã chuẩn hóa vào bộ nhớ đệm để tái sử dụng nhanh chóng.

---

## 4. Kịch Bản Kiểm Thử Tự Động Xác Minh
1. Viết bài kiểm thử tạo mặt nạ đa giác bám sát nét chữ so với mặt nạ hộp chữ nhật truyền thống.
2. Viết bài kiểm thử mở rộng mặt nạ theo trục thời gian cho các khung hình xuất hiện chữ.
3. Viết bài kiểm thử cơ chế tự động thử lại của cổng kiểm định chất lượng trên video mẫu có nét chữ tồn dư.
4. Viết bài kiểm thử chuẩn hóa âm thanh mẫu, đo tỷ lệ tín hiệu trên tạp âm và tính điểm chất lượng tham chiếu.
5. Chạy toàn bộ bộ kiểm thử tự động của ứng dụng máy tính bảo đảm đạt 100% thành công.
