# Tài Liệu Kiến Trúc Tái Tạo Nền Video Và Tối Ưu Hóa Bộ Nhớ Đồ Họa

Tài liệu này làm rõ kiến trúc phân tầng của các thuật toán phục hồi nền và cơ chế quản trị bộ nhớ đồ họa trên ứng dụng máy tính Sakai Studio.

---

## 1. Phân Tầng Kiến Trúc Phục Hồi Nền Video

Hệ thống cung cấp hai tầng xử lý chuyên biệt nhằm đáp ứng các kịch bản sử dụng khác nhau:

### 1.1. Tầng Giao Diện Máy Tính (Mạng Nơ-ron Học Sâu Toàn Phần)
- **Bộ điều phối chính**: Lớp `GUISubtitleRemover` tại tệp `src/core/main.py`.
- **Mô hình và trọng số thực thi**:
  - **LaMa**: Tải trực tiếp mô hình TorchScript tại `models/big-lama/big-lama.pt`, lấp đầy vùng chữ tĩnh dựa trên trường tiếp nhận diện rộng và biến đổi Fourier nhanh.
  - **STTN**: Tải mô hình PyTorch tại `models/sttn/infer_model.pth`, sử dụng cơ chế chú ý theo thời gian trên chuỗi khung hình lân cận.
  - **ProPainter**: Tích hợp chuỗi mạng học sâu bao gồm mạng luồng quang học RAFT (`raft-things.pth`), mạng hoàn thiện luồng quang học lặp (`recurrent_flow_completion.pth`) và mạng sinh ảnh không-thời gian (`ProPainter.pth`).
- **Tối ưu hóa bộ nhớ đồ họa**:
  - Tự động kích hoạt chế độ tính toán dấu phẩy động 16 bit trên các bộ xử lý đồ họa NVIDIA nhằm tiết kiệm 50% dung lượng bộ nhớ.
  - Tự động tính toán kích thước lô khung hình an toàn dựa trên dung lượng bộ nhớ khả dụng đo đạc thực tế trước khi nạp mô hình.
  - Bắt lỗi quá giới hạn bộ nhớ, thu gom bộ nhớ đệm và đệ quy giảm một nửa số lượng khung hình xử lý; nếu vẫn vượt ngưỡng sẽ chuyển đổi dự phòng sang mô hình LaMa từng khung hình kèm ghi nhận trạng thái cảnh báo kiểm định chất lượng.

### 1.2. Tầng Dòng Lệnh Và Xử Lý Siêu Nhẹ (Phơi Sáng Nền Theo Thời Gian)
- **Bộ điều phối chính**: Lớp `ProSubtitleRemover` tại tệp `src/core/processor.py`.
- **Nguyên lý hoạt động**:
  - Áp dụng kỹ thuật phơi sáng nền theo thời gian để tìm kiếm và sao chép các điểm ảnh nền không bị che khuất từ các khung hình lân cận có bù trừ chuyển động.
  - Kết hợp mô hình LaMa qua bộ suy luận ONNX Runtime hoặc OpenCV 5 DNN để xử lý các vùng chữ tồn dư xuất hiện liên tục trong toàn bộ đoạn video.
  - Không đòi hỏi bộ nhớ đồ họa lớn, vận hành ổn định trên cả máy chủ chỉ có bộ xử lý trung tâm hoặc card đồ họa dung lượng thấp.

---

## 2. Tiêu Chuẩn Kiểm Định Chất Lượng Sau Khi Xử Lý

Mọi tệp video sau khi xuất từ giao diện máy tính đều phải vượt qua cổng kiểm định chất lượng:
1. **Kiểm tra tính toàn vẹn tệp tin**: Xác thực dung lượng tệp lớn hơn 0, giải mã thành công khung hình đầu và khung hình cuối, kiểm tra tốc độ khung hình và số lượng khung hình đạt tối thiểu 95% video gốc.
2. **Bảo toàn luồng âm thanh**: Đảm bảo video đầu ra giữ nguyên vẹn luồng âm thanh gốc từ nguồn, ngăn chặn các trường hợp video xuất ra bị câm tiếng.
3. **Lấy mẫu thích ứng**: Kiểm tra các khung hình tại các mốc xuất hiện và biến mất của phụ đề, thời điểm đối tượng theo dõi đổi hướng di chuyển và các khung hình phân bố đều theo trục thời gian.
4. **Phát hiện nét chữ tồn dư và nhấp nháy**: Quét mật độ cạnh nét chữ chưa được xóa sạch và đo độ biến thiên chênh lệch cường độ sáng theo thời gian; tự động cảnh báo người dùng khi video cần rà soát lại thủ công.
