# Kế hoạch Thực thi Bộ Kiểm chứng Người nói qua Thị giác Hai Nhánh (Giai đoạn 2)

> **Dành cho kỹ sư thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:subagent-driven-development` hoặc `superpowers:executing-plans` để thực thi từng nhiệm vụ theo đúng quy trình kiểm thử trước khi viết mã nguồn. Các bước sử dụng cú pháp ô đánh dấu (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Xây dựng và tích hợp mô-đun kiểm chứng người nói qua thị giác hai nhánh (`VisualSpeakerVerifier`) hỗ trợ cả video hoạt họa chứa chân dung tĩnh ở vùng dưới khung hình và video người thật dựa trên biến thiên độ rộng khóe miệng kết hợp năng lượng chuyển động vùng miệng, xử lý tín hiệu thưa như trạng thái bình thường và tuân thủ nghiêm ngặt giới hạn bộ nhớ đồ họa 6 GB.

**Kiến trúc:** Tách biệt hoàn toàn làm hai nhánh độc lập trong `src/ai_engines/visual_speaker_verifier.py`: `CartoonAvatarVerifier` (phát hiện vùng chân dung thích ứng, gom cụm ảnh tĩnh bằng sai số mức xám tuyệt đối và ánh xạ danh tính qua đa số phiếu từ âm thanh kèm quy tắc từ chối khi trùng ngoại hình) và `RealPersonVisualVerifier` (định vị khuôn mặt qua YuNet ONNX hoặc OpenCV Cascade, tính chỉ số biến thiên độ rộng khóe miệng chuẩn hóa và năng lượng chuyển động vùng miệng). Bộ điều phối tổng hợp kết quả theo ma trận quyết định, ưu tiên giữ nguyên nhãn âm thanh khi không có bằng chứng thị giác hoặc khi ngoại hình trùng lặp, nối trực tiếp vào `SpeakerDiarizationCoordinator.process_video_subtitles` và hiển thị kiểm soát trên giao diện `OmniVoiceInterface`.

**Ngăn xếp Công nghệ:** Python 3.12, OpenCV (`opencv-python >= 4.12.0`), NumPy, SciPy, ONNX Runtime, PySide6, QFluentWidgets, pytest.

## Ràng buộc Toàn cục

- Ngôn ngữ giao diện, thông báo, bình luận mã nguồn và tài liệu: 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt, tuyệt đối không sử dụng biểu tượng cảm xúc.
- Nguyên lý tín hiệu thưa bất biến: Trạng thái không có bằng chứng thị giác (`VISUAL_ABSENT`) là một trạng thái nghiệp vụ bình thường, giữ nguyên 100% nhãn âm thanh gốc mà không coi là ngoại lệ hoặc lỗi.
- Xử lý trùng lặp ngoại hình (`INDISTINGUISHABLE`): Khi nhiều nhân vật chia sẻ cùng một ảnh đại diện khiến tỷ lệ phiếu âm thanh trong cụm không đạt đa số áp đảo (>= 60% và chênh lệch nhì >= 20%), từ chối ghi đè và bảo lưu quyền phân định cho âm thanh.
- An toàn OpenCV đa luồng: Luôn khởi tạo `cv2.VideoCapture` với cờ `cv2.VIDEO_ACCELERATION_NONE` và đóng tài nguyên trong khối `finally`.
- Không thêm gói thư viện nặng mới vào `requirements.txt`.
- Lưu trữ cấu hình trong lớp `Config(QConfig)` thuộc nhóm `"MMO"` tại `src/core/config.py`.
- Giới hạn tài nguyên: Mức tiêu hao bộ nhớ đồ họa bổ sung bằng 0 MB ở chế độ hoạt họa và dưới 200 MB ở chế độ người thật; giữ tương thích ngược hoàn toàn với các lời gọi hiện có tới `process_video_subtitles`.

---

### Nhiệm vụ 1: Cấu hình Hệ thống và Mở rộng Dữ liệu Khối Phụ đề

**Tệp thực hiện:**
- Chỉnh sửa: `src/core/config.py:845-855`
- Chỉnh sửa: `src/ai_engines/translator.py:60-78`
- Tạo mới: `test/test_visual_speaker_config.py`

**Giao diện:**
- Cung cấp: `app_config.mmoTTSVisualVerification` (`bool`), `app_config.mmoTTSVisualMode` (`"cartoon" | "real_person" | "auto"`), `app_config.mmoTTSVisualVerifyAll` (`bool`).
- Cung cấp trên `SubtitleBlock`: `visual_status: str = ""`, `visual_reason: str = ""`, `review_needed: bool = False`, `coverage_ratio: float = 1.0`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại trước**
  Tạo tệp `test/test_visual_speaker_config.py`:
  ```python
  """Kiểm thử các khóa cấu hình và trường dữ liệu thị giác trên SubtitleBlock."""

  from src.core.config import app_config
  from src.ai_engines.translator import SubtitleBlock


  def test_visual_verification_config_defaults():
      """Kiểm tra giá trị mặc định của các trường cấu hình kiểm chứng thị giác."""
      assert hasattr(app_config, "mmoTTSVisualVerification")
      assert hasattr(app_config, "mmoTTSVisualMode")
      assert hasattr(app_config, "mmoTTSVisualVerifyAll")

      assert app_config.mmoTTSVisualVerification.value is False
      assert app_config.mmoTTSVisualMode.value == "cartoon"
      assert app_config.mmoTTSVisualVerifyAll.value is False


  def test_visual_verification_config_options():
      """Kiểm tra chuyển đổi qua lại các chế độ kiểm chứng thị giác hợp lệ."""
      orig_enable = app_config.mmoTTSVisualVerification.value
      orig_mode = app_config.mmoTTSVisualMode.value
      orig_all = app_config.mmoTTSVisualVerifyAll.value
      try:
          app_config.mmoTTSVisualVerification.value = True
          assert app_config.mmoTTSVisualVerification.value is True

          for mode in ("cartoon", "real_person", "auto"):
              app_config.mmoTTSVisualMode.value = mode
              assert app_config.mmoTTSVisualMode.value == mode

          app_config.mmoTTSVisualVerifyAll.value = True
          assert app_config.mmoTTSVisualVerifyAll.value is True
      finally:
          app_config.mmoTTSVisualVerification.value = orig_enable
          app_config.mmoTTSVisualMode.value = orig_mode
          app_config.mmoTTSVisualVerifyAll.value = orig_all


  def test_subtitle_block_visual_fields_backward_compatible():
      """Kiểm tra các trường thị giác mới trên SubtitleBlock không phá vỡ khởi tạo cũ."""
      blk_old = SubtitleBlock(1, "00:00:00,000", "00:00:02,000", "Xin chào")
      assert blk_old.visual_status == ""
      assert blk_old.visual_reason == ""
      assert blk_old.review_needed is False
      assert blk_old.coverage_ratio == 1.0
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác minh thất bại**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_speaker_config.py -v`
  Kỳ vọng: Thất bại do thiếu thuộc tính `mmoTTSVisualVerification`.

- [ ] **Bước 3: Triển khai cấu hình và trường dữ liệu tối thiểu**
  Trong `src/core/config.py`, bổ sung ngay sau dòng `mmoTTSDeepDiarizationModelDir`:
  ```python
      mmoTTSVisualVerification = ConfigItem("MMO", "TTSVisualVerification", False, BoolValidator())
      mmoTTSVisualMode = OptionsConfigItem(
          "MMO",
          "TTSVisualMode",
          "cartoon",
          OptionsValidator(["cartoon", "real_person", "auto"]),
      )
      mmoTTSVisualVerifyAll = ConfigItem("MMO", "TTSVisualVerifyAll", False, BoolValidator())
  ```
  Trong `src/ai_engines/translator.py`, bổ sung vào cuối phần khai báo trường của lớp `SubtitleBlock`:
  ```python
      visual_status: str = ""
      visual_reason: str = ""
      review_needed: bool = False
      coverage_ratio: float = 1.0
  ```

- [ ] **Bước 4: Chạy kiểm thử để xác minh thành công**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_speaker_config.py -v`
  Kỳ vọng: 3 ca kiểm thử đạt 100%.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/core/config.py src/ai_engines/translator.py test/test_visual_speaker_config.py
  git commit -m "feat: bo sung cau hinh kiem chung thi giac va truong trang thai SubtitleBlock"
  ```

---

### Nhiệm vụ 2: Nhánh Hoạt họa — Định vị Chân dung Thích ứng, Gom cụm Ảnh tĩnh và Đa số Phiếu

**Tệp thực hiện:**
- Tạo mới: `src/ai_engines/visual_speaker_verifier.py`
- Tạo mới: `test/test_cartoon_avatar_verifier.py`

**Giao diện:**
- Định nghĩa `VisualVerificationStatus` (`Enum`: `CONFIRMED`, `CORRECTED`, `VISUAL_ABSENT`, `INDISTINGUISHABLE`).
- Định nghĩa `VisualVerificationResult` (`dataclass`: `cue_index`, `status`, `visual_speaker`, `confidence`, `cluster_id`, `reason`).
- Định nghĩa `CartoonAvatarVerifier`:
  - `extract_cue_portrait_patch(video_path: str, start_s: float, end_s: float, num_samples: int = 3) -> Optional[np.ndarray]`
  - `cluster_portrait_patches(patches: Dict[int, np.ndarray], distance_threshold: float = 15.0) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]`
  - `map_clusters_to_speakers(cue_to_cluster: Dict[int, int], cues: List[Dict[str, Any]], majority_threshold: float = 0.60, min_margin: float = 0.20) -> Tuple[Dict[int, Optional[str]], Dict[int, str]]`

- [ ] **Bước 1: Viết bài kiểm thử đơn vị thất bại cho Nhánh Hoạt họa**
  Tạo tệp `test/test_cartoon_avatar_verifier.py` kiểm chứng các tình huống thực nghiệm cốt lõi:
  1. Chân dung tĩnh dịch chuyển vị trí vẫn được trích xuất chuẩn; cảnh nền thuần nhất hoặc không có chân dung trả về `None` (`VISUAL_ABSENT`).
  2. Gom cụm chân dung với sai lệch nội cụm <= 10/255 và sai lệch liên cụm >= 25/255 tách đúng số cụm nhân vật.
  3. Đa số phiếu gán đúng cụm về nhân vật (Sếp áp đảo 4/5 phiếu -> Sếp).
  4. Nhân vật trùng ngoại hình (Lão Ngũ 2 phiếu + Bác tài 2 phiếu trong cùng một cụm) đánh dấu cụm không thể phân biệt (`None` kèm lý do trùng ngoại hình).

  ```python
  """Kiểm thử đơn vị cho nhánh nhận diện chân dung hoạt họa CartoonAvatarVerifier."""

  import os
  import numpy as np
  import cv2
  import pytest

  from src.ai_engines.visual_speaker_verifier import (
      CartoonAvatarVerifier,
      VisualVerificationStatus,
  )


  def _make_synthetic_cartoon_video(path: str, fps: int = 10, duration_s: int = 6):
      """Tạo video giả lập với 3 đoạn: [0-2s] Chó A góc trái, [2-4s] Cảnh nền không chân dung, [4-6s] Chó B dịch góc phải."""
      w, h = 320, 240
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
      total_frames = fps * duration_s
      for idx in range(total_frames):
          t = idx / float(fps)
          # Tạo nền chuyển động nhẹ mô phỏng video thực tế
          frame = np.full((h, w, 3), 25 + (idx % 5) * 4, dtype=np.uint8)
          cv2.line(frame, ((idx * 12) % w, 20), (((idx * 12) + 40) % w, 60), (60, 60, 60), 2)

          if t < 2.0:
              # Nhân vật A ở góc dưới trái
              cv2.rectangle(frame, (15, 130), (95, 225), (220, 190, 50), -1)
              cv2.circle(frame, (55, 165), 18, (40, 40, 200), -1)
          elif t >= 4.0:
              # Nhân vật B dịch sang vùng giữa-phải phía dưới
              cv2.rectangle(frame, (195, 135), (275, 225), (40, 120, 230), -1)
              cv2.rectangle(frame, (215, 155), (255, 195), (240, 240, 240), -1)
          writer.write(frame)
      writer.release()


  def test_extract_portrait_patch_sparse_and_shifted(tmp_path):
      """Chân dung dịch chuyển vẫn bắt được và cảnh vắng mặt trả về None."""
      vid_path = str(tmp_path / "cheems_mock.mp4")
      _make_synthetic_cartoon_video(vid_path)

      verifier = CartoonAvatarVerifier()
      patch_a = verifier.extract_cue_portrait_patch(vid_path, 0.2, 1.8)
      patch_absent = verifier.extract_cue_portrait_patch(vid_path, 2.2, 3.8)
      patch_b = verifier.extract_cue_portrait_patch(vid_path, 4.2, 5.8)

      assert patch_a is not None
      assert patch_absent is None
      assert patch_b is not None
      diff_ab = float(np.mean(np.abs(patch_a.astype(np.float32) - patch_b.astype(np.float32))))
      assert diff_ab >= 20.0


  def test_portrait_clustering_separates_distinct_and_merges_noisy():
      """Gom cụm phân tách các ảnh chênh lệch >= 25 và gộp ảnh nhiễu nội cụm <= 10."""
      rng = np.random.default_rng(42)
      base_a = np.full((64, 64), 40.0, dtype=np.float32)
      base_a[16:48, 16:48] = 180.0
      base_b = np.full((64, 64), 200.0, dtype=np.float32)
      base_b[16:48, 16:48] = 60.0

      a1 = np.clip(base_a + rng.normal(0, 4.0, base_a.shape), 0, 255).astype(np.float32)
      a2 = np.clip(base_a + rng.normal(0, 4.0, base_a.shape), 0, 255).astype(np.float32)
      b1 = np.clip(base_b + rng.normal(0, 4.0, base_b.shape), 0, 255).astype(np.float32)

      verifier = CartoonAvatarVerifier(cluster_threshold=15.0)
      cue_to_cluster, centroids = verifier.cluster_portrait_patches({0: a1, 1: a2, 2: b1})

      assert cue_to_cluster[0] == cue_to_cluster[1]
      assert cue_to_cluster[2] != cue_to_cluster[0]
      assert len(centroids) == 2


  def test_majority_voting_and_identical_avatar_ambiguity():
      """Cụm đạt đa số phiếu được gán nhãn; cụm chia phiếu 50-50 bị đánh dấu không thể phân biệt."""
      verifier = CartoonAvatarVerifier()
      cue_to_cluster = {
          0: 0, 1: 0, 2: 0, 3: 0,  # Cụm 0: 3 câu Sếp (tin cậy cao), 1 câu Cheems (cần xem xét)
          4: 1, 5: 1, 6: 1, 7: 1,  # Cụm 1: 2 câu Lão Ngũ, 2 câu Bác tài (trùng ngoại hình)
      }
      cues = [
          {"id": 0, "speaker": "Sếp", "confidence_level": "high"},
          {"id": 1, "speaker": "Sếp", "confidence_level": "high"},
          {"id": 2, "speaker": "Sếp", "confidence_level": "high"},
          {"id": 3, "speaker": "Cheems", "confidence_level": "review_needed"},
          {"id": 4, "speaker": "Lão Ngũ", "confidence_level": "high"},
          {"id": 5, "speaker": "Lão Ngũ", "confidence_level": "high"},
          {"id": 6, "speaker": "Bác tài", "confidence_level": "high"},
          {"id": 7, "speaker": "Bác tài", "confidence_level": "high"},
      ]

      cluster_map, cluster_reasons = verifier.map_clusters_to_speakers(cue_to_cluster, cues)
      assert cluster_map[0] == "Sếp"
      assert cluster_map[1] is None
      assert "trùng" in cluster_reasons[1].lower() or "không thể phân biệt" in cluster_reasons[1].lower()
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác minh thất bại**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_cartoon_avatar_verifier.py -v`
  Kỳ vọng: Thất bại với lỗi `ModuleNotFoundError: No module named 'src.ai_engines.visual_speaker_verifier'`.

- [ ] **Bước 3: Hiện thực hóa `CartoonAvatarVerifier` trong `src/ai_engines/visual_speaker_verifier.py`**
  Lưu ý kỹ thuật trọng tâm khi trích xuất chân dung thích ứng từ khung hình:
  - Dùng `cv2.VideoCapture(video_path, cv2.CAP_ANY, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])` để đảm bảo tương thích với quy chuẩn OpenCV đa luồng của dự án. Nếu tham số HW acceleration gây lỗi mở trên một số bản dựng OpenCV, tự động lùi về `cv2.VideoCapture(video_path)` rồi ép `cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)` trong khối `try...finally`.
  - Lấy `num_samples` khung hình (mặc định 3) phân bố đều bên trong `[start_s + 0.1 * dur, end_s - 0.1 * dur]`.
  - Định vị ứng viên chân dung trên nửa dưới khung hình (`y in [int(0.50 * H), H]`):
    - Tính ảnh độ lệch chuẩn hoặc bản đồ sai khác khung hình giữa các khung mẫu trong câu: vì chân dung nhân vật hoạt họa là lớp đồ họa tĩnh chồng lên nền (`intra_cue_diff` cực nhỏ), trong khi cảnh quay thực tế có chuyển động nền; tuy nhiên ở các khung hình nền tĩnh không có chân dung, toàn bộ nền không có đường viền nổi bật hoặc có phương sai không gian cực thấp.
    - Xét 3 vùng ứng viên trượt thích ứng ở nửa dưới (hoặc phát hiện contour có kích thước hợp lý `>= 10%` chiều cao khung hình trong vùng nửa dưới với mật độ biên Canny cao và phương sai điểm ảnh đủ lớn `std >= 12.0`).
    - Tính độ lệch tuyệt đối trung bình giữa các khung mẫu bên trong vùng ứng viên (`temporal_diff`): nếu vùng ứng viên thay đổi mạnh giữa các khung liên tiếp trong cùng một câu (`temporal_diff > 14.0`) thì đó là cảnh nền đang chuyển động (không phải chân dung tĩnh) -> loại bỏ.
    - Nếu không có vùng ứng viên nào thỏa mãn cả độ tương phản không gian (`spatial_std >= 12.0`), mật độ biên (`edge_density >= 0.02`) và tính tĩnh thời gian (`temporal_diff <= 14.0`), trả về `None` (`VISUAL_ABSENT`).
  - Hàm `cluster_portrait_patches` dùng trung bình sai khác tuyệt đối trên ảnh chuẩn hóa `(64, 64)` để gom cụm trực tiếp và cập nhật tâm cụm trung bình.
  - Hàm `map_clusters_to_speakers` tổng hợp trọng số phiếu từ các câu thoại thuộc cụm (ưu tiên trọng số `1.0` cho câu tin cậy cao, `0.35` cho câu cần rà soát, bỏ qua câu `"Chưa xác định"` hoặc rỗng). Áp dụng ngưỡng đa số `>= 0.60` và biên cách biệt với hạng nhì `>= 0.20` để ngăn ngừa gán sai khi 2 nhân vật dùng chung ảnh đại diện.

- [ ] **Bước 4: Chạy kiểm thử Nhánh Hoạt họa để xác minh đạt**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_cartoon_avatar_verifier.py -v`
  Kỳ vọng: 3/3 ca kiểm thử đạt 100%.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/ai_engines/visual_speaker_verifier.py test/test_cartoon_avatar_verifier.py
  git commit -m "feat: hien thuc hoa nhanh kiem chung chan dung hoat hoa va gom cum tu xau"
  ```

---

### Nhiệm vụ 3: Nhánh Người thật — Nhận diện Khuôn mặt và Tổ hợp Chuyển động Miệng

**Tệp thực hiện:**
- Chỉnh sửa: `src/ai_engines/visual_speaker_verifier.py` (bổ sung `FaceDetectorBackend` và `RealPersonVisualVerifier`)
- Tạo mới: `test/test_real_person_verifier.py`

**Giao diện:**
- Định nghĩa `FaceDetectionSample` (`bbox`, `landmarks_5`, `face_crop`, `mouth_crop`, `mouth_width_norm`).
- Định nghĩa `FaceDetectorBackend`: tự động dùng `cv2.FaceDetectorYN` khi có tệp trọng số YuNet ONNX, dự phòng ngoại tuyến tức thì bằng `cv2.CascadeClassifier(haarcascade_frontalface_default.xml)` đi kèm sẵn trong `opencv-python`.
- Định nghĩa `RealPersonVisualVerifier`:
  - `compute_mouth_metrics(face_sequence: List[FaceDetectionSample]) -> Tuple[float, float, float]` trả về `(mouth_width_std, mouth_motion_energy, combined_activity_score)`.
  - `analyze_cue_faces(video_path: str, start_s: float, end_s: float, num_samples: int = 6) -> Dict[str, Any]`
  - `verify_cues(...) -> Dict[int, VisualVerificationResult]`

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho Nhánh Người thật**
  Tạo tệp `test/test_real_person_verifier.py`:
  1. Kiểm chứng công thức tính `mouth_width_std` từ 5 điểm mốc YuNet (mắt trái, mắt phải, mũi, khóe miệng phải, khóe miệng trái) không phụ thuộc điểm môi trên/dưới.
  2. Kiểm chứng khi mặt tiến lại gần máy quay đồng đều (phóng đại tỷ lệ đồng thời cả mắt và miệng), `mouth_width_std` vẫn xấp xỉ 0 (không bị nhận diện nhầm là đang nói nhờ chuẩn hóa theo khoảng cách hai mắt).
  3. Kiểm chứng khi có chuyển động miệng thực tế (năng lượng chuyển động vùng miệng cao kết hợp biến thiên độ rộng khóe miệng), điểm hoạt động phát âm vượt ngưỡng và chọn đúng người đang nói trong khung hình có 2 khuôn mặt (1 người nói, 1 người im lặng).
  4. Kiểm chứng trạng thái không có khuôn mặt nào nói trả về `VISUAL_ABSENT`; trạng thái cả 2 khuôn mặt cùng có hoạt động miệng ngang nhau trả về `INDISTINGUISHABLE`.

  ```python
  """Kiểm thử đơn vị cho nhánh nhận diện người thật RealPersonVisualVerifier."""

  import numpy as np
  from src.ai_engines.visual_speaker_verifier import (
      FaceDetectionSample,
      RealPersonVisualVerifier,
      VisualVerificationStatus,
  )


  def _make_mouth_crop(open_height: int) -> np.ndarray:
      """Tạo vùng ảnh miệng giả lập 48x24 mức xám."""
      crop = np.full((24, 48), 160, dtype=np.uint8)
      y_center = 12
      half = max(1, open_height // 2)
      crop[y_center - half : y_center + half, 12:36] = 30
      return crop


  def test_mouth_width_scale_invariant_when_zooming():
      """Khi khuôn mặt phóng đại đồng đều theo tỷ lệ, độ rộng chuẩn hóa giữ nguyên bất biến."""
      verifier = RealPersonVisualVerifier()
      seq = []
      for scale in (1.0, 1.2, 1.5, 1.8):
          # Khoảng cách 2 mắt = 40 * scale; khoảng cách 2 khóe miệng = 20 * scale -> tỷ lệ luôn 0.5
          landmarks = np.array(
              [
                  [100.0, 100.0],
                  [100.0 + 40.0 * scale, 100.0],
                  [100.0 + 20.0 * scale, 120.0],
                  [100.0 + 10.0 * scale, 140.0],
                  [100.0 + 30.0 * scale, 140.0],
              ],
              dtype=np.float32,
          )
          crop = _make_mouth_crop(open_height=2)
          seq.append(
              FaceDetectionSample(
                  bbox=(0, 0, int(100 * scale), int(100 * scale)),
                  landmarks_5=landmarks,
                  face_crop=np.zeros((64, 64), dtype=np.uint8),
                  mouth_crop=crop,
                  mouth_width_norm=verifier.compute_normalized_mouth_width(landmarks),
              )
          )

      width_std, motion_energy, score = verifier.compute_mouth_metrics(seq)
      assert width_std < 1e-5
      assert motion_energy < 1e-5
      assert score < 0.01


  def test_detects_active_speaker_between_two_faces():
      """Phân biệt được người đang cử động miệng với người đứng im cạnh bên."""
      verifier = RealPersonVisualVerifier(activity_threshold=0.05)

      talking_seq = []
      silent_seq = []
      for idx in range(6):
          width_delta = 5.0 if idx % 2 == 0 else -5.0
          talk_lm = np.array(
              [
                  [100.0, 100.0],
                  [140.0, 100.0],
                  [120.0, 120.0],
                  [110.0 - width_delta, 140.0],
                  [130.0 + width_delta, 140.0],
              ],
              dtype=np.float32,
          )
          silent_lm = np.array(
              [
                  [300.0, 100.0],
                  [340.0, 100.0],
                  [320.0, 120.0],
                  [310.0, 140.0],
                  [330.0, 140.0],
              ],
              dtype=np.float32,
          )
          talking_seq.append(
              FaceDetectionSample(
                  bbox=(80, 70, 80, 90),
                  landmarks_5=talk_lm,
                  face_crop=np.full((64, 64), 100, dtype=np.uint8),
                  mouth_crop=_make_mouth_crop(12 if idx % 2 == 0 else 2),
                  mouth_width_norm=verifier.compute_normalized_mouth_width(talk_lm),
              )
          )
          silent_seq.append(
              FaceDetectionSample(
                  bbox=(280, 70, 80, 90),
                  landmarks_5=silent_lm,
                  face_crop=np.full((64, 64), 200, dtype=np.uint8),
                  mouth_crop=_make_mouth_crop(2),
                  mouth_width_norm=verifier.compute_normalized_mouth_width(silent_lm),
              )
          )

      _, _, talk_score = verifier.compute_mouth_metrics(talking_seq)
      _, _, silent_score = verifier.compute_mouth_metrics(silent_seq)

      assert talk_score > 0.08
      assert silent_score < 0.01

      selection = verifier.select_active_face_in_cue(
          [
              {"track_id": 0, "score": talk_score, "face_crop": talking_seq[0].face_crop},
              {"track_id": 1, "score": silent_score, "face_crop": silent_seq[0].face_crop},
          ]
      )
      assert selection["status"] == "single_active"
      assert selection["track_id"] == 0


  def test_absent_and_simultaneous_talking_states():
      """Không ai nói trả về absent; hai người nói ngang nhau trả về ambiguous."""
      verifier = RealPersonVisualVerifier(activity_threshold=0.05)

      res_silent = verifier.select_active_face_in_cue(
          [{"track_id": 0, "score": 0.01, "face_crop": np.zeros((64, 64), dtype=np.uint8)}]
      )
      assert res_silent["status"] == "absent"

      res_both = verifier.select_active_face_in_cue(
          [
              {"track_id": 0, "score": 0.18, "face_crop": np.zeros((64, 64), dtype=np.uint8)},
              {"track_id": 1, "score": 0.17, "face_crop": np.zeros((64, 64), dtype=np.uint8)},
          ]
      )
      assert res_both["status"] == "ambiguous"
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác minh thất bại**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_real_person_verifier.py -v`
  Kỳ vọng: Thất bại do chưa có `RealPersonVisualVerifier`.

- [ ] **Bước 3: Triển khai `FaceDetectorBackend` và `RealPersonVisualVerifier` trong `src/ai_engines/visual_speaker_verifier.py`**
  Bổ sung đầy đủ thuật toán:
  - `compute_normalized_mouth_width(landmarks_5)`: tính `||L_mouth - R_mouth|| / max(1e-5, ||L_eye - R_eye||)`.
  - `extract_mouth_roi(frame_gray, bbox, landmarks_5)`: nếu có 5 điểm mốc từ YuNet thì cắt vùng chữ nhật bên dưới đỉnh mũi (`y_nose + 0.12*(y_mouth - y_nose)`) mở rộng xuống dưới `1.8 * (y_mouth - y_nose)` và mở rộng ngang qua 2 khóe miệng thêm 25%; nếu chạy nhánh dự phòng Cascade không có 5 điểm mốc thì cắt vùng `[x + 0.25*w, y + 0.62*h, 0.50*w, 0.28*h]`, thu gọn chuẩn hóa về `(48, 24)`.
  - `compute_mouth_metrics`: tính độ lệch chuẩn của `mouth_width_norm` (`width_std`) và trung bình sai khác tuyệt đối liên khung của `mouth_crop` chuẩn hóa về `[0.0, 1.0]` (`motion_energy`), tổng hợp thành `0.4 * width_std + 0.6 * motion_energy` (nếu toàn bộ giá trị `mouth_width_norm` bằng 0 do chạy dự phòng Cascade thì lấy `motion_energy`).
  - `select_active_face_in_cue`:
    - Nếu không có khuôn mặt nào đạt ngưỡng `activity_threshold` -> `"absent"`.
    - Nếu mặt thứ hai đạt ngưỡng và đạt >= 80% điểm của mặt cao nhất -> `"ambiguous"` (`INDISTINGUISHABLE`).
    - Ngược lại trả về `"single_active"` kèm `track_id` và `face_crop` đại diện để gom cụm khuôn mặt + bỏ phiếu đa số từ nhãn âm thanh giống quy trình của nhánh hoạt họa.

- [ ] **Bước 4: Chạy kiểm thử Nhánh Người thật**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_real_person_verifier.py -v`
  Kỳ vọng: 3/3 ca kiểm thử đạt 100%.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/ai_engines/visual_speaker_verifier.py test/test_real_person_verifier.py
  git commit -m "feat: hien thuc hoa nhanh nguoi that voi do rong khoe mieng va nang luong vung mieng"
  ```

---

### Nhiệm vụ 4: Bộ Điều phối `VisualSpeakerVerifier` và Ma trận Hợp nhất Quyết định

**Tệp thực hiện:**
- Chỉnh sửa: `src/ai_engines/visual_speaker_verifier.py` (hoàn thiện lớp `VisualSpeakerVerifier` và quy tắc chọn câu mỏ neo + câu mục tiêu)
- Tạo mới: `test/test_visual_speaker_verifier.py`

**Giao diện:**
- `VisualSpeakerVerifier.should_verify_cue(blk: Any, verify_all: bool = False) -> bool`
- `VisualSpeakerVerifier.apply_decision_matrix(blk: Any, visual_res: VisualVerificationResult) -> bool`
- `VisualSpeakerVerifier.verify_video_speakers(cls, video_path: str, subtitle_blocks: List[Any], profiles: Dict[str, Any], mode: str = "cartoon", verify_all: bool = False) -> Dict[int, VisualVerificationResult]`

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho Bộ Điều phối và Ma trận Hợp nhất**
  Tạo tệp `test/test_visual_speaker_verifier.py` kiểm chứng đầy đủ 6 dòng trong Bảng Ma trận Quyết định của đặc tả kỹ thuật:
  1. Âm thanh tin cậy cao + thị giác xác nhận -> giữ nguyên nhãn và tin cậy cao.
  2. Âm thanh cần rà soát (`review_needed=True` hoặc `confidence_level == "review_needed"`) + thị giác xác nhận -> giữ nhãn, nâng lên `confidence_level = "high"`, xóa cờ `review_needed`.
  3. Âm thanh cần rà soát + thị giác điều chỉnh (`CORRECTED`) -> cập nhật sang nhãn thị giác, nâng lên `high`, xóa cờ `review_needed`.
  4. Âm thanh tin cậy cao + thị giác bất đồng -> giữ nguyên nhãn âm thanh gốc (ưu tiên âm thanh khi độ tin cậy gốc cao).
  5. Mọi trạng thái + `VISUAL_ABSENT` -> giữ nguyên nhãn và trạng thái âm thanh.
  6. Mọi trạng thái + `INDISTINGUISHABLE` -> giữ nguyên nhãn và trạng thái âm thanh.
  7. Câu thoại đã khóa bởi người dùng (`is_speaker_locked=True`) -> bất biến không bị chỉnh sửa.

  ```python
  """Kiểm thử bộ điều phối VisualSpeakerVerifier và ma trận hợp nhất quyết định."""

  from dataclasses import dataclass
  from src.ai_engines.visual_speaker_verifier import (
      VisualSpeakerVerifier,
      VisualVerificationResult,
      VisualVerificationStatus,
  )


  @dataclass
  class MockBlock:
      index: int
      start_time: str = "00:00:00,000"
      end_time: str = "00:00:02,000"
      speaker: str = "Người nói 1"
      confidence_level: str = "high"
      review_needed: bool = False
      coverage_ratio: float = 1.0
      is_speaker_locked: bool = False
      visual_status: str = ""
      visual_reason: str = ""


  def test_should_verify_cue_sparse_logic():
      """Kiểm chứng điều kiện kích hoạt thưa trên review_needed, coverage_ratio và verify_all."""
      blk_high = MockBlock(1, confidence_level="high", review_needed=False, coverage_ratio=0.9)
      blk_rev = MockBlock(2, confidence_level="review_needed", review_needed=True, coverage_ratio=0.8)
      blk_low_cov = MockBlock(3, confidence_level="high", review_needed=False, coverage_ratio=0.3)
      blk_unassigned = MockBlock(4, speaker="Chưa xác định", confidence_level="unassigned")

      assert VisualSpeakerVerifier.should_verify_cue(blk_high, verify_all=False) is False
      assert VisualSpeakerVerifier.should_verify_cue(blk_high, verify_all=True) is True
      assert VisualSpeakerVerifier.should_verify_cue(blk_rev, verify_all=False) is True
      assert VisualSpeakerVerifier.should_verify_cue(blk_low_cov, verify_all=False) is True
      assert VisualSpeakerVerifier.should_verify_cue(blk_unassigned, verify_all=False) is True


  def test_decision_matrix_all_six_cases_and_lock_protection():
      """Xác minh đầy đủ ma trận quyết định giữa âm thanh và thị giác."""
      # 1. Âm thanh review_needed + Thị giác xác nhận -> Nâng lên high
      b1 = MockBlock(1, speaker="Sếp", confidence_level="review_needed", review_needed=True)
      r1 = VisualVerificationResult(1, VisualVerificationStatus.CONFIRMED, "Sếp", 0.9, 0, "Khớp chân dung cụm 0")
      VisualSpeakerVerifier.apply_decision_matrix(b1, r1)
      assert b1.speaker == "Sếp"
      assert b1.confidence_level == "high"
      assert b1.review_needed is False
      assert b1.visual_status == VisualVerificationStatus.CONFIRMED.value

      # 2. Âm thanh review_needed + Thị giác điều chỉnh -> Sửa nhãn
      b2 = MockBlock(2, speaker="Cheems", confidence_level="review_needed", review_needed=True)
      r2 = VisualVerificationResult(2, VisualVerificationStatus.CORRECTED, "Sếp", 0.92, 0, "Thị giác khớp cụm Sếp")
      VisualSpeakerVerifier.apply_decision_matrix(b2, r2)
      assert b2.speaker == "Sếp"
      assert b2.confidence_level == "high"
      assert b2.review_needed is False
      assert b2.visual_status == VisualVerificationStatus.CORRECTED.value

      # 3. Âm thanh tin cậy cao + Thị giác bất đồng -> Giữ âm thanh gốc
      b3 = MockBlock(3, speaker="Cheems", confidence_level="high", review_needed=False)
      r3 = VisualVerificationResult(3, VisualVerificationStatus.CORRECTED, "Sếp", 0.85, 0, "Bất đồng nhãn tin cậy cao")
      VisualSpeakerVerifier.apply_decision_matrix(b3, r3)
      assert b3.speaker == "Cheems"
      assert b3.confidence_level == "high"

      # 4. Không có bằng chứng thị giác -> Giữ nguyên
      b4 = MockBlock(4, speaker="Cheems", confidence_level="review_needed", review_needed=True)
      r4 = VisualVerificationResult(4, VisualVerificationStatus.VISUAL_ABSENT, "", 0.0, -1, "Không có chân dung")
      VisualSpeakerVerifier.apply_decision_matrix(b4, r4)
      assert b4.speaker == "Cheems"
      assert b4.confidence_level == "review_needed"
      assert b4.review_needed is True

      # 5. Không thể phân biệt -> Giữ nguyên (trường hợp Lão Ngũ và Bác tài)
      b5 = MockBlock(5, speaker="Lão Ngũ", confidence_level="review_needed", review_needed=True)
      r5 = VisualVerificationResult(5, VisualVerificationStatus.INDISTINGUISHABLE, "", 0.5, 1, "Trùng ngoại hình")
      VisualSpeakerVerifier.apply_decision_matrix(b5, r5)
      assert b5.speaker == "Lão Ngũ"
      assert b5.confidence_level == "review_needed"

      # 6. Câu thoại đã khóa bởi người dùng -> Bảo toàn tuyệt đối
      b6 = MockBlock(6, speaker="Sếp", confidence_level="high", is_speaker_locked=True)
      r6 = VisualVerificationResult(6, VisualVerificationStatus.CORRECTED, "Cheems", 0.95, 2, "Cố sửa câu đã khóa")
      VisualSpeakerVerifier.apply_decision_matrix(b6, r6)
      assert b6.speaker == "Sếp"
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác minh thất bại**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_speaker_verifier.py -v`
  Kỳ vọng: Thất bại do thiếu `VisualSpeakerVerifier.should_verify_cue` / `apply_decision_matrix`.

- [ ] **Bước 3: Hoàn thiện `VisualSpeakerVerifier` trong `src/ai_engines/visual_speaker_verifier.py`**
  Triển khai:
  - Quy tắc thu thập mẫu cho `verify_video_speakers`:
    - Quét chọn tối đa 5 câu mỏ neo có độ tin cậy cao cho mỗi nhân vật âm thanh (tổng không quá 30 câu mỏ neo) + toàn bộ các câu mục tiêu thỏa `should_verify_cue(blk, verify_all=verify_all)`.
    - Với chế độ `"auto"`: trích xuất nhanh 3 câu mẫu; nếu phát hiện thấy khuôn mặt người rõ nét mà không thấy chân dung tĩnh ở vùng dưới thì chọn `RealPersonVisualVerifier`, ngược lại chọn `CartoonAvatarVerifier`.
    - Chạy trích xuất -> gom cụm -> ánh xạ đa số phiếu từ danh sách câu mẫu + câu mục tiêu -> tạo `VisualVerificationResult` cho từng câu mục tiêu -> áp dụng `apply_decision_matrix`.
    - Mọi ngoại lệ OpenCV hoặc đọc video đều được bắt an toàn, ghi log cảnh báo và trả về tập kết quả rỗng để không ảnh hưởng luồng âm thanh.

- [ ] **Bước 4: Chạy kiểm thử Bộ Điều phối**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_speaker_verifier.py -v`
  Kỳ vọng: Đạt 100%.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/ai_engines/visual_speaker_verifier.py test/test_visual_speaker_verifier.py
  git commit -m "feat: hoan thien bo dieu phoi VisualSpeakerVerifier va ma tran hop nhat"
  ```

---

### Nhiệm vụ 5: Nối Dây Đường Ống Phân đoạn và `TTSEngine`

**Tệp thực hiện:**
- Chỉnh sửa: `src/ai_engines/speaker_diarization.py:1299-1315,1755-1773`
- Chỉnh sửa: `src/ai_engines/tts_engine.py:2134-2145`
- Tạo mới: `test/test_visual_diarization_integration.py`

**Giao diện:**
- Mở rộng chữ ký `SpeakerDiarizationCoordinator.process_video_subtitles(..., source_video_path: Optional[str] = None)` với mặc định `None` giữ tương thích ngược hoàn toàn.
- Bổ sung hàm trợ giúp cập nhật lại thống kê hồ sơ (`utterance_count`, `total_duration`, `high_confidence_ratio`) và liên kết `voice_ref_path` sau khi thị giác điều chỉnh nhãn câu phụ đề.

- [ ] **Bước 1: Viết bài kiểm thử tích hợp thất bại**
  Tạo tệp `test/test_visual_diarization_integration.py` tạo một tệp WAV hợp lệ và tệp MP4 giả lập đồng bộ, chạy `SpeakerDiarizationCoordinator.process_video_subtitles(..., source_video_path=mp4_path)` trong trạng thái bật và tắt `app_config.mmoTTSVisualVerification`:
  1. Khi tắt cấu hình: không gọi `VisualSpeakerVerifier`, hành vi giữ nguyên 100%.
  2. Khi bật cấu hình và truyền `source_video_path`: `VisualSpeakerVerifier.verify_video_speakers` được gọi với đúng đường dẫn video thật (không phải tệp WAV trung gian), và thống kê hồ sơ được đồng bộ sạch.
  3. Khi chỉ truyền tệp WAV (`source_video_path=None`): bỏ qua kiểm chứng thị giác một cách an toàn không phát sinh ngoại lệ.

  ```python
  """Kiểm thử tích hợp luồng phân đoạn âm thanh với kiểm chứng thị giác Giai đoạn 2."""

  import numpy as np
  import soundfile as sf
  from dataclasses import dataclass
  from unittest.mock import patch

  from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator
  from src.ai_engines.visual_speaker_verifier import VisualVerificationResult, VisualVerificationStatus
  from src.core.config import app_config


  @dataclass
  class DummyBlock:
      index: int
      start_time: str
      end_time: str
      text: str
      speaker: str = ""
      voice_ref_path: str = ""
      voice_instruct: str = ""
      confidence_level: str = "high"
      is_speaker_locked: bool = False
      visual_status: str = ""
      visual_reason: str = ""
      review_needed: bool = False
      coverage_ratio: float = 1.0


  def _write_test_wav(path: str, duration_s: float = 4.0, sr: int = 16000):
      t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
      sig = 0.25 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
      sf.write(path, sig, sr)


  def test_coordinator_skips_visual_when_disabled_or_no_video(tmp_path):
      """Khi tắt kiểm chứng thị giác hoặc thiếu tệp video gốc, luồng âm thanh chạy bình thường."""
      wav_path = str(tmp_path / "audio.wav")
      _write_test_wav(wav_path)
      blocks = [
          DummyBlock(1, "00:00:00,000", "00:00:01,800", "Câu một"),
          DummyBlock(2, "00:00:02,000", "00:00:03,800", "Câu hai"),
      ]
      coord = SpeakerDiarizationCoordinator(prefer_cuda=False, allow_mock=True)

      orig_flag = app_config.mmoTTSVisualVerification.value
      try:
          app_config.mmoTTSVisualVerification.value = False
          with patch(
              "src.ai_engines.visual_speaker_verifier.VisualSpeakerVerifier.verify_video_speakers"
          ) as mock_vis:
              profiles = coord.process_video_subtitles(wav_path, blocks, temp_dir=str(tmp_path))
              assert len(profiles) >= 1
              mock_vis.assert_not_called()

          app_config.mmoTTSVisualVerification.value = True
          with patch(
              "src.ai_engines.visual_speaker_verifier.VisualSpeakerVerifier.verify_video_speakers"
          ) as mock_vis:
              profiles = coord.process_video_subtitles(wav_path, blocks, temp_dir=str(tmp_path))
              assert len(profiles) >= 1
              mock_vis.assert_not_called()
      finally:
          app_config.mmoTTSVisualVerification.value = orig_flag


  def test_coordinator_invokes_visual_with_source_video_path(tmp_path):
      """Khi bật kiểm chứng thị giác và có source_video_path, hệ thống truyền đúng tệp video gốc."""
      wav_path = str(tmp_path / "audio.wav")
      vid_path = str(tmp_path / "sample.mp4")
      _write_test_wav(wav_path)
      with open(vid_path, "wb") as f:
          f.write(b"\x00\x00\x00\x20ftypisom")

      blocks = [
          DummyBlock(1, "00:00:00,000", "00:00:01,800", "Câu một"),
          DummyBlock(2, "00:00:02,000", "00:00:03,800", "Câu hai"),
      ]
      coord = SpeakerDiarizationCoordinator(prefer_cuda=False, allow_mock=True)

      orig_flag = app_config.mmoTTSVisualVerification.value
      orig_mode = app_config.mmoTTSVisualMode.value
      try:
          app_config.mmoTTSVisualVerification.value = True
          app_config.mmoTTSVisualMode.value = "cartoon"

          with patch(
              "src.ai_engines.visual_speaker_verifier.VisualSpeakerVerifier.verify_video_speakers",
              return_value={
                  0: VisualVerificationResult(1, VisualVerificationStatus.CONFIRMED, "Người nói 1", 0.9, 0, "Khớp"),
              },
          ) as mock_vis:
              profiles = coord.process_video_subtitles(
                  wav_path,
                  blocks,
                  temp_dir=str(tmp_path),
                  source_video_path=vid_path,
              )
              assert len(profiles) >= 1
              mock_vis.assert_called_once()
              call_args = mock_vis.call_args
              assert call_args[0][0] == vid_path or call_args.kwargs.get("video_path") == vid_path
      finally:
          app_config.mmoTTSVisualVerification.value = orig_flag
          app_config.mmoTTSVisualMode.value = orig_mode
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác minh thất bại**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_diarization_integration.py -v`
  Kỳ vọng: Thất bại với `TypeError: unexpected keyword argument 'source_video_path'`.

- [ ] **Bước 3: Chỉnh sửa `speaker_diarization.py` và `tts_engine.py`**
  - Trong `src/ai_engines/speaker_diarization.py`:
    - Thêm `source_video_path: Optional[str] = None` vào `process_video_subtitles`.
    - Ngay sau dòng `ExternalDiarizationVerifier.verify_or_fallback(...)`, kiểm tra `app_config.mmoTTSVisualVerification.value` và `source_video_path` tồn tại; gọi `VisualSpeakerVerifier.verify_video_speakers(...)`, đồng bộ `voice_ref_path`, `utterance_count`, `total_duration`, `high_confidence_ratio` của các hồ sơ nếu có câu được điều chỉnh nhãn.
  - Trong `src/ai_engines/tts_engine.py:2134`:
    - Truyền `source_video_path=video_path` vào `coordinator.process_video_subtitles(...)`.

- [ ] **Bước 4: Chạy kiểm thử tích hợp**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_diarization_integration.py -v`
  Kỳ vọng: Đạt 100%.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/ai_engines/speaker_diarization.py src/ai_engines/tts_engine.py test/test_visual_diarization_integration.py
  git commit -m "feat: tich hop VisualSpeakerVerifier vao quy trinh phan doan va TTSEngine"
  ```

---

### Nhiệm vụ 6: Tích hợp Điều khiển và Trạng thái trên Giao diện `OmniVoiceInterface`

**Tệp thực hiện:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (tại các vùng `diarize_options_row`, `_run_auto_detect_speakers`, `_on_speakers_detected`, `_populate_subtitle_table`)
- Tạo mới: `test/test_visual_ui_controls.py`

**Giao diện:**
- Bổ sung trên `OmniVoiceInterface`:
  - `self.chk_visual_verification` (CheckBox `"Kiểm chứng người nói qua thị giác"`)
  - `self.combo_visual_mode` (ComboBox gồm `"Hoạt họa — Ảnh đại diện"`, `"Người thật — Cử động miệng"`, `"Tự động chọn chế độ"`)
  - `self.chk_visual_verify_all` (CheckBox `"Quét toàn bộ câu thoại"`)
- Quy tắc giao diện:
  - 100% tiếng Việt chuẩn mực, không tiếng Anh trong ngoặc đơn, không biểu tượng cảm xúc.
  - Khi tắt `chk_visual_verification`, tự động vô hiệu hóa `combo_visual_mode` và `chk_visual_verify_all`.
  - Đồng bộ `visual_status` và `visual_reason` từ worker về `ui_blk` trong `_on_speakers_detected`, hiển thị chú giải trực quan trong `_populate_subtitle_table`.

- [ ] **Bước 1: Viết bài kiểm thử giao diện thất bại**
  Tạo tệp `test/test_visual_ui_controls.py` kiểm tra khởi tạo `OmniVoiceInterface` (hoặc kiểm tra widget và hàm xử lý đồng bộ cấu hình), xác nhận các điều khiển tồn tại, trạng thái vô hiệu hóa tự động hoạt động đúng, và văn bản không chứa tiếng Anh trong ngoặc đơn hay biểu tượng cảm xúc.

  ```python
  """Kiểm thử các điều khiển kiểm chứng thị giác trên giao diện OmniVoiceInterface."""

  import re
  import sys
  from PySide6.QtWidgets import QApplication

  from src.core.config import app_config


  def _get_app():
      app = QApplication.instance()
      if app is None:
          app = QApplication(sys.argv)
      return app


  def test_visual_ui_controls_and_enable_disable_sync(tmp_path):
      """Xác minh widget tồn tại, đồng bộ tắt mở phụ thuộc và tuân thủ quy chuẩn tiếng Việt."""
      _get_app()
      from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

      orig_vis = app_config.mmoTTSVisualVerification.value
      orig_mode = app_config.mmoTTSVisualMode.value
      orig_all = app_config.mmoTTSVisualVerifyAll.value
      try:
          app_config.mmoTTSVisualVerification.value = False
          app_config.mmoTTSVisualMode.value = "cartoon"
          app_config.mmoTTSVisualVerifyAll.value = False

          win = OmniVoiceInterface()
          try:
              assert hasattr(win, "chk_visual_verification")
              assert hasattr(win, "combo_visual_mode")
              assert hasattr(win, "chk_visual_verify_all")

              # Khi checkbox chưa bật, các điều khiển con phải bị vô hiệu hóa
              assert win.chk_visual_verification.isChecked() is False
              assert win.combo_visual_mode.isEnabled() is False
              assert win.chk_visual_verify_all.isEnabled() is False

              # Kiểm tra không có tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt
              texts = [
                  win.chk_visual_verification.text(),
                  win.chk_visual_verify_all.text(),
                  *(win.combo_visual_mode.itemText(i) for i in range(win.combo_visual_mode.count())),
              ]
              for txt in texts:
                  assert not re.search(r"\([A-Za-z0-9\s\-_]+\)", txt), f"Vi phạm quy chuẩn văn bản: {txt}"

              # Khi bật kiểm chứng thị giác, điều khiển con mở khóa
              win.chk_visual_verification.setChecked(True)
              assert win.combo_visual_mode.isEnabled() is True
              assert win.chk_visual_verify_all.isEnabled() is True

              win._sync_visual_verification_config()
              assert app_config.mmoTTSVisualVerification.value is True
          finally:
              win.close()
              win.deleteLater()
      finally:
          app_config.mmoTTSVisualVerification.value = orig_vis
          app_config.mmoTTSVisualMode.value = orig_mode
          app_config.mmoTTSVisualVerifyAll.value = orig_all
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác minh thất bại**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_ui_controls.py -v`
  Kỳ vọng: Thất bại do thiếu thuộc tính `chk_visual_verification`.

- [ ] **Bước 3: Chỉnh sửa `src/desktop/ui/omnivoice_interface.py`**
  - Tại vùng `diarize_options_row` (~dòng 1892): thêm hàng hoặc các điều khiển `chk_visual_verification`, `combo_visual_mode`, `chk_visual_verify_all`, kết nối tín hiệu `toggled` để bật/tắt `combo_visual_mode` và `chk_visual_verify_all`.
  - Thêm phương thức `_sync_visual_verification_config(self)` lưu trạng thái UI vào `app_config.mmoTTSVisualVerification`, `app_config.mmoTTSVisualMode`, `app_config.mmoTTSVisualVerifyAll`.
  - Gọi `self._sync_visual_verification_config()` ngay trong `_run_auto_detect_speakers(self)`.
  - Trong `_on_speakers_detected(self)` (~dòng 7098): sao chép `visual_status` và `visual_reason` từ `w_blk` sang `ui_blk`.
  - Trong `_populate_subtitle_table(self)` (~dòng 6597): bổ sung chú giải tiếng Việt cho `visual_status` vào tooltip của `item_stt`.

- [ ] **Bước 4: Chạy kiểm thử giao diện và kiểm thử hồi quy toàn bộ Giai đoạn 1 & Giai đoạn 2**
  Lệnh: `E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_visual_speaker_config.py test/test_cartoon_avatar_verifier.py test/test_real_person_verifier.py test/test_visual_speaker_verifier.py test/test_visual_diarization_integration.py test/test_visual_ui_controls.py test/test_deep_diarization_config.py test/test_hungarian_diarization_matching.py test/test_external_diarization_verifier.py test/test_diarization_verifier_integration.py -v`
  Kỳ vọng: Toàn bộ các bài kiểm thử đều vượt qua 100%.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/desktop/ui/omnivoice_interface.py test/test_visual_ui_controls.py
  git commit -m "feat: bo sung dieu khien kiem chung thi giac va hien thi trang thai tren giao dien"
  ```
