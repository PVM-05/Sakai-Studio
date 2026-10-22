# Đặc Tả Thiết Kế: Cải Tiến Thuật Toán Phân Cụm Người Nói Và Bổ Sung Chế Độ Số Người Chính Xác

**Ngày thiết kế:** 22/10/2026  
**Trạng thái:** Đã phê duyệt  
**Mục tiêu:** Nâng cao độ chính xác nhận diện số lượng người nói từ âm thanh video, khắc phục tình trạng dồn về một người hoặc chia nhỏ sai lệch nhân vật, đồng thời cung cấp tùy chọn chỉ định số lượng người nói chính xác trên giao diện Sakai Studio.

---

## 1. Bối cảnh và Thách thức Kỹ thuật

Hiện tại, thuật toán phân cụm người nói trong `src/ai_engines/speaker_diarization.py` gặp các hạn chế:
1. **Thuật toán K-Means và điểm Silhouette bị đánh lừa khi dữ liệu mất cân bằng:** Trong video thực tế, nhân vật chính chiếm đa số câu thoại trong khi nhân vật phụ chỉ nói một vài câu ngắn. Hệ số Silhouette phạt nặng các cụm không đồng đều kích thước, kéo điểm tổng thể xuống thấp (< 0.25) khiến hệ thống ép nhầm toàn bộ video về 1 người nói duy nhất.
2. **Nhiễu từ các câu thoại ngắn:** Các đoạn thoại dưới 0.8 giây ("vâng", "dạ", tiếng thở) có vector đặc trưng CAM++ biến động mạnh, làm sai lệch việc tìm tâm cụm.
3. **Cắt dính giọng giữa hai câu sát nhau:** Khoảng đệm cố định 0.15 giây có thể cắt lẹm vào câu của người nói tiếp theo nếu hai câu thoại liên tiếp diễn ra cách nhau dưới 0.3 giây.
4. **Thiếu khả năng chỉ định số người trên giao diện:** Người dùng không có cách nào yêu cầu hệ thống phân cụm thành đúng số lượng người nói mà họ đã biết chắc chắn (ví dụ: đúng 2 người).

---

## 2. Giải pháp Kiến trúc và Thuật toán

### 2.1. Phân Cụm Thứ Bậc Tích Lũy (Agglomerative Hierarchical Clustering)
- **Ma trận khoảng cách Cosine:** Tính toán khoảng cách giữa tất cả các vector đặc trưng của các câu thoại nòng cốt:
  $$D_{ij} = 1.0 - \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$$
- **Liên kết trung bình (Average Linkage):** Khoảng cách giữa hai cụm $A$ và $B$ là trung bình cộng khoảng cách giữa tất cả các cặp phần tử:
  $$d(A, B) = \frac{1}{|A||B|} \sum_{\mathbf{e}_a \in A} \sum_{\mathbf{e}_b \in B} D(\mathbf{e}_a, \mathbf{e}_b)$$
- **Ưu điểm:** Không giả định các cụm có hình cầu hay kích thước đồng đều; xử lý hoàn hảo trường hợp nhân vật chính nói nhiều câu và nhân vật phụ nói ít câu.

### 2.2. Cơ chế Mẫu Nòng Cốt (Anchor-based Diarization)
1. **Giai đoạn 1 - Xác định mẫu nòng cốt:**
   - Lọc các câu thoại có thời lượng $\ge 0.8\text{s}$ và năng lượng tín hiệu RMS tốt làm tập nòng cốt.
   - Nếu số câu $\ge 0.8\text{s}$ quá ít (< 3 câu), hạ ngưỡng thời lượng xuống $0.5\text{s}$ để đảm bảo có đủ dữ liệu.
2. **Giai đoạn 2 - Phân cụm tập nòng cốt:**
   - Thực hiện phân cụm trên tập nòng cốt để xác định số lượng người nói $k$ và tính toán vector tâm cụm chuẩn của từng nhân vật.
3. **Giai đoạn 3 - Gán các câu ngắn còn lại:**
   - Đối với các câu ngắn (< 0.8s), tính khoảng cách cosin tới từng tâm cụm đã xác định và gán về nhân vật có khoảng cách gần nhất.
   - Đánh dấu mức độ tin cậy "Cần rà soát" nếu câu quá ngắn hoặc khoảng cách tới tâm cụm vượt ngưỡng an toàn.

### 2.3. Chế Độ Xác Định Số Lượng Người Nói
Hệ thống hỗ trợ 2 chế độ:
- **Chế độ tự động ước lượng (Auto):** Dựa trên ngưỡng khoảng cách liên kết thứ bậc (khoảng cách cosin từ $0.35$ đến $0.48$ tùy theo độ nhạy), tự động dừng gom cụm khi khoảng cách giữa các nhóm vượt ngưỡng.
- **Chế độ cố định chính xác (Exact):** Khi người dùng kích hoạt, thuật toán buộc phải dừng đúng tại số lượng cụm $k = \text{exact\_count}$ (ví dụ $k=2$), bất kể điểm số tự động, giúp người dùng nắm quyền kiểm soát tuyệt đối.

### 2.4. Cắt Đệm Thời Gian An Toàn (Boundary-safe Audio Padding)
- Trước khi cắt âm thanh cho từng câu thoại, kiểm tra mốc thời gian của câu liền trước và câu liền sau:
  $$\text{pad\_start} = \min(0.15, \frac{\text{start}_i - \text{end}_{i-1}}{2})$$
  $$\text{pad\_end} = \min(0.15, \frac{\text{start}_{i+1} - \text{end}_i}{2})$$
- Tránh hoàn toàn việc cắt dính tiếng của người nói kế tiếp khi hai câu phụ đề nằm sát nhau.

---

## 3. Thiết Kế Giao Diện Người Dùng (Sakai Studio UI)

Tại Bước 2 (Quản Lý Nhân Vật) của `src/desktop/ui/omnivoice_interface.py`:
1. **Bộ chọn chế độ nhận diện số người (`combo_speaker_count_mode`):**
   - "Tự động ước lượng (Tối đa)"
   - "Cố định chính xác số người"
2. **Bộ điều khiển số người (`spin_max_speakers`):**
   - Khi ở chế độ "Tự động ước lượng", nhãn hiển thị là "Số người tối đa:"
   - Khi ở chế độ "Cố định chính xác số người", nhãn hiển thị là "Số người chính xác:"
3. **Thanh điều chỉnh độ nhạy phân tách giọng (`combo_clustering_sensitivity`):**
   - "Độ nhạy tiêu chuẩn" (ngưỡng khoảng cách 0.40)
   - "Độ nhạy cao — Dễ tách người nói" (ngưỡng khoảng cách 0.34, thích hợp khi các nhân vật có giọng tương đồng)
   - "Độ nhạy thấp — Ưu tiên gộp giọng" (ngưỡng khoảng cách 0.46, thích hợp khi một người nói nhiều cung bậc cảm xúc)

---

## 4. Kế Hoạch Kiểm Thử (TDD)
1. Kiểm thử phân cụm thứ bậc với ma trận cosin và mẫu mất cân bằng (nhân vật chính 20 câu, nhân vật phụ 2 câu).
2. Kiểm thử chế độ cố định chính xác số người (`exact_speakers=2`) đảm bảo luôn cho ra đúng 2 người nói.
3. Kiểm thử cơ chế mẫu nòng cốt: các câu ngắn < 0.8s không làm sai lệch số cụm nhân vật.
4. Kiểm thử cắt âm thanh có kẹp biên thời gian an toàn không tràn sang câu kế tiếp.
5. Kiểm thử đồng bộ điều khiển trên giao diện người dùng.
