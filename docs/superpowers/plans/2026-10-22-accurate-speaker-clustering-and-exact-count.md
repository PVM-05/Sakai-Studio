# Kế Hoạch Triển Khai: Nâng Cấp Thuật Toán Phân Cụm Người Nói Và Chế Độ Chỉ Định Số Người Chính Xác

> **Dành cho tác nhân thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:executing-plans` để triển khai từng nhiệm vụ bên dưới. Sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ. Thực thi trực tiếp tuần tự (Inline execution).

**Mục tiêu:** Cải tiến độ chính xác phân cụm nhận diện người nói bằng thuật toán Phân Cụm Thứ Bậc (Agglomerative Hierarchical Clustering), lọc mẫu nòng cốt (Anchor-based), cắt đệm thời gian an toàn, và bổ sung tùy chọn "Cố định chính xác số người" kèm "Độ nhạy phân tách" trên giao diện Sakai Studio.

**Ngăn xếp Công nghệ:** Python 3.12, NumPy, SciPy (nếu có) / Thuật toán thuần NumPy, PySide6, QFluentWidgets, Pytest.

## Ràng buộc Toàn cục
- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện. Chỉ dùng biểu tượng FluentIcon.
- Tuân thủ quy trình phát triển hướng kiểm thử (TDD): viết bài kiểm thử thất bại trước, viết mã nguồn tối thiểu sau.
- Không được để lại mã chờ như "TODO", "TBD" hoặc triển khai dở dang.

---

## Bản đồ Tệp tin

| Tệp | Vai trò |
|-----|---------|
| `src/ai_engines/speaker_diarization.py` | Thuật toán phân cụm thứ bậc AHC, lọc mẫu nòng cốt, và cắt đệm thời gian an toàn |
| `src/ai_engines/tts_engine.py` | Truyền các tham số `exact_speakers`, `clustering_sensitivity` từ giao diện xuống bộ điều phối |
| `src/desktop/ui/omnivoice_interface.py` | Thêm bộ chọn Chế độ số người và Độ nhạy phân tách trên giao diện người dùng |
| `test/test_omnivoice_pipeline.py` | Bộ kiểm thử chức năng phân cụm, chế độ cố định và giao diện |

---

### Nhiệm vụ 1: Nâng Cấp Thuật Toán Phân Cụm Thứ Bậc Tích Lũy (AHC) Và Lọc Mẫu Nòng Cốt

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Viết hàm `agglomerative_cluster_embeddings(embeddings, threshold=0.40, exact_k=None, min_k=1, max_k=6)`:
   - Sử dụng ma trận khoảng cách cosin và liên kết trung bình (Average Linkage).
   - Nếu `exact_k` được chỉ định: dừng gom cụm khi số cụm đạt đúng `exact_k`.
   - Nếu không có `exact_k`: gom cụm tự động cho đến khi khoảng cách liên kết nhỏ nhất vượt quá `threshold` (hoặc đạt `min_k`).
2. Cập nhật `cosine_cluster_embeddings` kết hợp cơ chế mẫu nòng cốt:
   - Đánh giá các câu dài $\ge 0.8\text{s}$ làm nòng cốt.
   - Phân cụm tập nòng cốt bằng `agglomerative_cluster_embeddings`.
   - Gán các câu ngắn $(< 0.8\text{s})$ về tâm cụm gần nhất.

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**
- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
- [ ] **Bước 3: Viết mã nguồn tối thiểu vào `src/ai_engines/speaker_diarization.py`**
- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
- [ ] **Bước 5: Cam kết thay đổi vào git**

---

### Nhiệm vụ 2: Cắt Đệm Thời Gian An Toàn Tránh Lẹm Giọng Câu Kề Nhau

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/speaker_diarization.py` (trong `process_video_subtitles`)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Thay thế khoảng đệm cố định 0.15 giây bằng khoảng đệm thích ứng:
   - Tính khoảng cách tới câu trước `gap_before = max(0.0, start_i - end_{i-1})`.
   - Tính khoảng cách tới câu sau `gap_after = max(0.0, start_{i+1} - end_i)`.
   - `pad_start = min(0.15, gap_before / 2.0)` nếu câu trước nằm liền kề.
   - `pad_end = min(0.15, gap_after / 2.0)` nếu câu sau nằm liền kề.
2. Tránh cắt dính giọng người khác khi hai câu thoại kề sát nhau trong phim hội thoại.

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**
- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
- [ ] **Bước 3: Viết mã nguồn tối thiểu vào `src/ai_engines/speaker_diarization.py`**
- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
- [ ] **Bước 5: Cam kết thay đổi vào git**

---

### Nhiệm vụ 3: Tích Hợp Tham Số Số Người Chính Xác Và Độ Nhạy Vào `TTSEngine`

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Mở rộng phương thức `TTSEngine.detect_speakers_from_video` nhận thêm:
   - `exact_speakers: Optional[int] = None`
   - `clustering_sensitivity: str = "standard"` (standard / high / low)
2. Chuyển đổi độ nhạy thành ngưỡng khoảng cách cosin tương ứng:
   - `standard` -> `threshold = 0.40`
   - `high` (dễ tách người nói) -> `threshold = 0.34`
   - `low` (ưu tiên gộp người nói) -> `threshold = 0.46`
3. Truyền xuống `SpeakerDiarizationCoordinator.process_video_subtitles`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**
- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
- [ ] **Bước 3: Viết mã nguồn tối thiểu vào `src/ai_engines/tts_engine.py`**
- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
- [ ] **Bước 5: Cam kết thay đổi vào git**

---

### Nhiệm vụ 4: Bổ Sung Điều Khiển Chế Độ Số Người Và Độ Nhạy Trên Giao Diện

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Tại khu vực cấu hình nhận diện Bước 2:
   - Thêm bộ chọn `self.combo_speaker_count_mode`:
     - "Tự Động Ước Lượng (Tối Đa)"
     - "Cố Định Chính Xác Số Người"
   - Khi chọn "Cố Định Chính Xác Số Người", nhãn `self.lbl_max_spk` đổi thành "Số người chính xác:", ngược lại là "Số người tối đa:".
   - Thêm bộ chọn `self.combo_clustering_sensitivity`:
     - "Độ Nhạy Tiêu Chuẩn"
     - "Độ Nhạy Cao — Dễ Tách Nhân Vật"
     - "Độ Nhạy Thấp — Ưu Tiên Gộp Giọng"
2. Truyền các tham số này vào worker khi bấm nhận diện người nói.

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**
- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
- [ ] **Bước 3: Viết mã nguồn tối thiểu vào `src/desktop/ui/omnivoice_interface.py`**
- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
- [ ] **Bước 5: Chạy toàn bộ kiểm thử hồi quy**
- [ ] **Bước 6: Cam kết git và đồng bộ nhánh**
