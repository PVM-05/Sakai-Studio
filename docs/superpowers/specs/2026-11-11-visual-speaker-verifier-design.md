# Đặc tả Kỹ thuật: Bộ Kiểm chứng Người nói qua Thị giác Hai Nhánh (Giai đoạn 2)

Tài liệu thiết kế kỹ thuật cho Giai đoạn 2 thuộc lộ trình nâng cấp năng lực phân tích giọng nói và phân đoạn người nói trong Sakai Studio. Hệ thống bổ sung mô-đun kiểm chứng thị giác hai nhánh nhằm hỗ trợ xác thực danh tính người nói qua hình ảnh video, tối ưu hóa cho cấu hình phần cứng 6 GB bộ nhớ đồ họa (NVIDIA GeForce RTX 4050 Laptop GPU) và hỗ trợ vận hành dự phòng trên CPU.

---

## 1. Mục tiêu và Bối cảnh Kỹ thuật

### 1.1. Hiện trạng sau Giai đoạn 1
- Giai đoạn 1 đã hoàn thiện việc tích hợp mô hình phân đoạn âm thanh chuyên sâu kết hợp cơ chế dự phòng an toàn.
- Hệ thống âm thanh gắn nhãn cho từng câu phụ đề và phân loại độ tin cậy thông qua các trường dữ liệu:
  - Cờ yêu cầu xem xét lại (`review_needed`): Bật khi có hiện tượng nói chồng hoặc khi có sự bất đồng nhãn giữa hai thuật toán âm thanh.
  - Tỷ lệ bao phủ âm thanh (`coverage_ratio`): Đo lường mức độ khớp thời gian giữa câu phụ đề và đoạn phát âm thực tế.
  - Trạng thái tin cậy (`status`): Nhận một trong ba giá trị định danh là mức cao, cần xem xét lại hoặc chưa gán.
- **Hạn chế:** Các trường hợp nhiều nhân vật có chất giọng tương đồng, môi trường âm thanh lẫn nhạc nền phức tạp hoặc câu thoại ngắn có thể dẫn đến gán sai nhãn nếu chỉ dựa vào tín hiệu âm học thuần túy.

### 1.2. Bằng chứng Thực nghiệm trên Dữ liệu Video Cheems
Qua phân tích thực nghiệm trên video kiểm chuẩn Cheems 6 nhân vật, các đặc điểm thị giác cốt lõi được xác định như sau:
1. **Chân dung người nói mang tính chất tĩnh:** Chân dung nhân vật xuất hiện ở góc dưới video (ví dụ chú chó thường, chú chó mặc vest đại diện cho Sếp), nhưng là ảnh đồ họa tĩnh hoàn toàn trong suốt thời gian hiển thị. Chênh lệch điểm ảnh giữa các khung hình liên tiếp chỉ là nhiễu nén video, không có hiện tượng cử động miệng hay nhấp nháy theo nhịp nói.
2. **Vị trí và sự xuất hiện không cố định:** Vị trí chân dung có thể dịch chuyển linh hoạt giữa các cảnh và hoàn toàn vắng mặt ở nhiều phân đoạn (cảnh góc quay cận, cảnh trong xe ô tô, cảnh ban đêm hoặc khi người dẫn chuyện nói). Giả định về vùng cố định ở bốn góc màn hình là không chính xác trên thực tế.
3. **Trùng lặp ngoại hình giữa các nhân vật khác nhau:** Nhân vật Lão Ngũ và Bác tài sử dụng chung một hình ảnh đồ họa (chú chó trắng cơ bắp mặc áo ba lỗ). Mọi phương pháp thị giác thuần túy đều không thể phân biệt hai nhân vật này; việc phân định bắt buộc phải bảo lưu cho tín hiệu âm thanh.
4. **Đặc tính tín hiệu thưa:** Bằng chứng thị giác chỉ xuất hiện ở một số câu thoại nhất định. Do đó, trạng thái không tìm thấy dữ liệu thị giác là một trạng thái nghiệp vụ bình thường, hoàn toàn không phải là lỗi.

---

## 2. Kiến trúc Hệ thống: Bộ Điều phối Kiểm chứng Thị giác Hai Nhánh

Mô-đun kiểm chứng thị giác được xây dựng thành một lớp điều phối độc lập `VisualSpeakerVerifier` đặt tại `src/ai_engines/visual_speaker_verifier.py`.

```
                    +------------------------------------+
                    |  Kết quả Phân đoạn Âm thanh GĐ 1  |
                    +-----------------+------------------+
                                      |
                         Kiểm tra điều kiện kích hoạt
                         (review_needed == True hoặc
                          coverage_ratio < 0.5 hoặc
                          chủ động bật toàn bộ)
                                      |
                     +----------------v-----------------+
                     |      VisualSpeakerVerifier       |
                     +----------------+-----------------+
                                      |
               +----------------------+----------------------+
               |                                             |
     [Chế độ Hoạt họa]                              [Chế độ Người thật]
+--------------v---------------+              +--------------v---------------+
|    CartoonAvatarVerifier     |              |   RealPersonVisualVerifier   |
| - Trích xuất khung hình      |              | - Phát hiện mặt (YuNet/ONNX) |
| - Phát hiện vùng chân dung   |              | - Khoảng cách khóe miệng     |
| - Gom cụm sai khác điểm ảnh  |              | - Năng lượng vùng miệng      |
| - Ánh xạ đa số phiếu         |              | - Chấm điểm hoạt động phát âm|
+--------------+---------------+              +--------------+---------------+
               |                                             |
               +----------------------+----------------------+
                                      |
                       +--------------v---------------+
                       |  Cổng Hợp nhất Quyết định    |
                       |  - Xác nhận nhãn âm thanh    |
                       |  - Điều chỉnh nhãn âm thanh  |
                       |  - Không có bằng chứng (giữ) |
                       |  - Không thể phân biệt (giữ) |
                       +------------------------------+
```

---

## 3. Thiết kế Chi tiết Từng Nhánh

### 3.1. Nhánh Hoạt họa (`CartoonAvatarVerifier`)

Nhánh Hoạt họa giải quyết bài toán nhận diện nhân vật qua ảnh đại diện tĩnh hoặc biểu tượng đồ họa.

#### A. Trích xuất Khung hình Đại diện
- Với mỗi câu phụ đề cần kiểm chứng, trích xuất từ 3 đến 5 khung hình đại diện nằm trong khoảng thời gian từ lúc bắt đầu đến lúc kết thúc câu.
- Tắt tính năng tăng tốc phần cứng khi đọc khung hình (`cv2.VIDEO_ACCELERATION_NONE`) để ngăn ngừa lỗi bộ giải mã video khi nhảy thời gian liên tục.

#### B. Định vị Vùng Chân dung Tự thích ứng
- Thay vì quét tọa độ cố định, thuật toán chia khung hình thành các vùng ứng viên tiềm năng (nửa dưới màn hình và các biên cạnh).
- Tìm kiếm các vùng đồ họa có độ tương phản cao, đường biên rõ nét hoặc có sự thay đổi nội dung giữa cảnh nền và vùng chân dung nhân vật.

#### C. Giải thuật Gom cụm Chân dung Tự xây dựng
- Do các ảnh đại diện là đồ họa tĩnh lặp lại, sai lệch mức xám trung bình giữa các lần xuất hiện của cùng một nhân vật rất nhỏ (sai số trung bình tuyệt đối khoảng 10 trên 255 đơn vị mức xám do nhiễu nén).
- Ngược lại, sai lệch mức xám giữa các nhân vật khác nhau dao động rõ rệt trong khoảng từ 25 đến 65 trên 255 đơn vị.
- Thuật toán gom cụm trực tiếp:
  - Khởi tạo danh sách các cụm chân dung trực quan.
  - So sánh ảnh ứng viên mới với tâm của từng cụm hiện có bằng sai số trung bình tuyệt đối đã chuẩn hóa kích thước.
  - Nếu khoảng cách nhỏ hơn ngưỡng dung sai (ngưỡng mặc định 15/255), đưa ảnh vào cụm tương ứng và cập nhật tâm cụm.
  - Nếu khoảng cách vượt ngưỡng đối với mọi cụm, khởi tạo một cụm chân dung mới.

#### D. Ánh xạ Danh tính bằng Giải thuật Đa số Phiếu
- Với mỗi cụm chân dung trực quan được hình thành:
  - Thống kê các nhãn người nói từ luồng âm thanh của tất cả các câu thoại có sự xuất hiện của cụm đó.
  - Gán danh tính nhân vật của cụm theo nhãn âm thanh chiếm tỷ lệ phiếu áp đảo (vượt ngưỡng 60% tổng số phiếu của cụm).
- **Quy tắc xử lý trùng lặp ngoại hình:** Nếu một cụm chân dung có hai nhãn âm thanh cùng chiếm tỷ lệ phiếu ngang ngửa nhau và được ghi nhận là cùng chia sẻ ngoại hình (như trường hợp Lão Ngũ và Bác tài), cụm này được đánh dấu trạng thái không thể phân biệt. Thuật toán từ chối sửa nhãn và giữ nguyên phân định của âm thanh.

---

### 3.2. Nhánh Người thật (`RealPersonVisualVerifier`)

Nhánh Người thật xử lý video quay cận cảnh hoặc bán thân của người thật (phim truyện, chương trình phỏng vấn, tin tức).

#### A. Thư viện và Mô hình Nhận diện Khuôn mặt
- Sử dụng mô hình nhận diện khuôn mặt YuNet định dạng ONNX tích hợp trực tiếp qua bộ thực thi OpenCV DNN hoặc ONNX Runtime (đã có sẵn trong hệ thống).
- Không thêm bất kỳ thư viện ngoài nặng nề nào vào danh mục phụ thuộc.

#### B. Thuật toán Phát hiện Hoạt động Miệng từ 5 Điểm Mốc
- Mô hình YuNet cung cấp 5 điểm mốc đặc trưng: mắt phải, mắt trái, đỉnh mũi, khóe miệng phải và khóe miệng trái (không cung cấp điểm môi trên và môi dưới).
- Để đo lường trạng thái nói mà không cần điểm môi trên/dưới, thuật toán kết hợp hai chỉ số:
  1. **Biến thiên độ rộng khóe miệng:** Tính khoảng cách Euclid giữa khóe miệng trái và khóe miệng phải, chuẩn hóa theo khoảng cách giữa hai đồng tử mắt để loại bỏ ảnh hưởng khi nhân vật tiến gần hoặc lùi xa máy quay:
     $$\text{Độ rộng chuẩn hóa} = \frac{\|\text{Khóe miệng trái} - \text{Khóe miệng phải}\|}{\|\text{Mắt trái} - \text{Mắt phải}\|}$$
  2. **Năng lượng chuyển động vùng miệng:** Trích xuất vùng hình chữ nhật nằm ngay phía dưới đỉnh mũi bao trùm hai khóe miệng. Tính mức độ biến thiên cường độ điểm ảnh giữa các khung hình liên tiếp trong suốt câu thoại:
     $$\text{Năng lượng chuyển động} = \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{1}{N} \sum_{p} |I_{t+1}(p) - I_t(p)|$$
- **Chỉ số hoạt động phát âm tổng hợp:** Kết hợp độ lệch chuẩn của độ rộng chuẩn hóa và năng lượng chuyển động vùng miệng. Người có chỉ số vượt trội và đồng pha với âm thanh được xác định là người nói tích cực trong khung hình.

---

## 4. Cơ chế Hợp nhất Tín hiệu và Xử lý Tình huống

### 4.1. Bốn Trạng thái Quyết định Thị giác
Kết quả phân tích thị giác cho từng câu phụ đề được trả về dưới dạng đối tượng `VisualVerificationResult` gồm 4 trạng thái định danh:

1. `CONFIRMED` (Xác nhận): Nhãn nhận diện qua thị giác trùng khớp với nhãn âm thanh. Câu thoại được nâng lên mức độ tin cậy cao và gỡ cờ cần xem xét lại.
2. `CORRECTED` (Điều chỉnh): Thị giác tìm thấy bằng chứng áp đảo thuộc về một nhân vật khác có trong hồ sơ âm thanh, đồng thời cụm chân dung không bị tranh chấp. Nhãn người nói của câu phụ đề được cập nhật theo thị giác, kèm ghi chú nguyên nhân điều chỉnh.
3. `VISUAL_ABSENT` (Không có bằng chứng): Khung hình không xuất hiện chân dung hoạt họa hoặc không phát hiện khuôn mặt nào hoạt động phát âm. Hệ thống giữ nguyên nhãn âm thanh gốc, không thay đổi cờ trạng thái.
4. `INDISTINGUISHABLE` (Không thể phân biệt): Chân dung phát hiện được thuộc về nhóm dùng chung ngoại hình (như Lão Ngũ và Bác tài) hoặc có nhiều người cùng nói nhưng không phân định được ai áp đảo. Quyền quyết định hoàn toàn thuộc về luồng âm thanh.

### 4.2. Ma trận Ra Quyết định Tổng hợp

| Trạng thái Âm thanh | Bằng chứng Thị giác | Quyết định Cuối cùng | Ghi chú Trạng thái |
|---|---|---|---|
| Mức tin cậy cao | Xác nhận (`CONFIRMED`) | Giữ nguyên nhãn âm thanh | Xác thực hai nguồn đồng thuận |
| Cần xem xét lại (`review_needed`) | Xác nhận (`CONFIRMED`) | Giữ nguyên nhãn âm thanh, gỡ cờ nghi ngờ | Thị giác đã giải tỏa tranh chấp âm thanh |
| Cần xem xét lại (`review_needed`) | Điều chỉnh (`CORRECTED`) | Thay đổi sang nhãn thị giác | Thị giác sửa nhãn lỗi của âm thanh |
| Mức tin cậy cao | Bất đồng nhưng thị giác thưa | Giữ nguyên nhãn âm thanh | Ưu tiên âm thanh khi độ tin cậy gốc cao |
| Bất kỳ | Không có bằng chứng (`VISUAL_ABSENT`) | Giữ nguyên nhãn âm thanh | Tình huống bình thường do tín hiệu thưa |
| Bất kỳ | Không thể phân biệt (`INDISTINGUISHABLE`)| Giữ nguyên nhãn âm thanh | Trường hợp nhân vật chung ngoại hình |

---

## 5. Tích hợp Giao diện và Tham số Cấu hình

### 5.1. Khóa Cấu hình Hệ thống (`src/core/config.py`)
Bổ sung các trường cấu hình vào nhóm `"MMO"` trong lớp `Config`:
- `mmoTTSVisualVerification`: Kiểu logic (`bool`), mặc định `False` (bật/tắt kiểm chứng thị giác).
- `mmoTTSVisualMode`: Kiểu chuỗi (`str`), nhận các giá trị `"cartoon"`, `"real_person"`, `"auto"`. Mặc định `"cartoon"`.
- `mmoTTSVisualVerifyAll`: Kiểu logic (`bool`), mặc định `False` (kiểm chứng thưa theo cờ nghi ngờ hoặc kiểm chứng toàn bộ).

### 5.2. Cập nhật Thẻ Điều khiển trên Giao diện
- Thêm hàng điều khiển "Kiểm chứng người nói qua thị giác" tại giao diện lồng tiếng phân vai.
- Cung cấp hộp chọn chế độ nội dung (Hoạt họa / Người thật / Tự động).
- Hiển thị biểu tượng trạng thái trực quan trên bảng danh sách câu thoại khi một câu được xác nhận hoặc điều chỉnh bởi thị giác.

---

## 6. Tiêu chuẩn Nghiệm thu Kỹ thuật (Giai đoạn 2)

1. **Bộ kiểm thử đơn vị (`test/test_visual_speaker_verifier.py`):**
   - Đạt 100% các ca kiểm thử cho giải thuật gom cụm chân dung tự xây dựng với độ lệch nội cụm nhỏ và liên cụm lớn.
   - Đạt 100% các ca kiểm thử cho logic đa số phiếu và xử lý nhân vật trùng ngoại hình.
   - Đạt 100% các ca kiểm thử cho thuật toán biến thiên độ rộng khóe miệng và năng lượng vùng miệng.
   - Đạt 100% kiểm thử trạng thái tín hiệu thưa không gây ra ngoại lệ.
2. **Bộ kiểm thử tích hợp (`test/test_visual_diarization_integration.py`):**
   - Tích hợp liền mạch với kết quả đầu ra của `SpeakerDiarizationCoordinator` và `diarization_matching.py`.
   - Thời gian xử lý trung bình dưới 0,15 giây cho mỗi câu phụ đề cần kiểm chứng trên CPU.
   - Mức chiếm dụng bộ nhớ đồ họa bổ sung bằng 0 đối với chế độ hoạt họa và dưới 200 MB đối với chế độ người thật.
