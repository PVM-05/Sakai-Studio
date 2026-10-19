# Kế Hoạch Triển Khai: Khắc Phục Lõi Phân Cụm Người Nói Và Chuẩn Hóa Giao Diện OmniVoice

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Giải quyết triệt để lỗi thuật toán đoán sai số người nói (bổ sung giả thuyết 1 người nói, sửa Silhouette cho cụm đơn lẻ, kiểm tra khoảng cách tâm cụm), đồng bộ hóa giọng nghe thử với xuất video, không bỏ qua nhận diện khi còn câu chưa phân vai, và chuẩn hóa trạng thái cùng cảnh báo trước khi xuất.

**Architecture:** Bổ sung giả thuyết $k = 1$ dựa trên ma trận khoảng cách cosin và khoảng cách tâm cụm; sửa Silhouette của cụm đơn lẻ thành 0.0; mở rộng dải ứng viên $k$; sửa điều kiện kích hoạt `detect_speakers_from_video` trong `synthesize_blocks`; truyền `override_mode` vào `block_preview`; phân định rõ các nút nghe thử và hiển thị "Chưa phân vai"; thêm hộp thoại xác nhận trước khi xuất; cấu hình `pytest.ini` và `ci-tests.yml`.

**Tech Stack:** Python 3.12, PySide6, ONNX Runtime, NumPy, SciPy, soundfile, kaldi-native-fbank, pytest.

## Global Constraints

- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Sử dụng biểu tượng FluentIcon.
- Tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD).

---

### Task 1: Thiết lập cấu hình kiểm thử tự động (pytest.ini & ci-tests.yml)

**Files:**
- Create: `pytest.ini`
- Modify: `.github/workflows/ci-tests.yml:15-18`
- Test: Chạy `pytest` từ dòng lệnh không cần khai báo `$env:PYTHONPATH`

**Interfaces:**
- Consumes: Cấu hình pytest
- Produces: Môi trường kiểm thử tự động nạp đúng gói `src` cả ở máy cục bộ và máy chủ ảo CI.

- [ ] **Step 1: Tạo tệp pytest.ini**

Tạo `pytest.ini` tại thư mục gốc:
```ini
[pytest]
pythonpath = .
testpaths = test
```

- [ ] **Step 2: Cập nhật .github/workflows/ci-tests.yml**

Bổ sung `PYTHONPATH: .` vào `env:` của tác vụ kiểm thử:
```yaml
    env:
      QT_QPA_PLATFORM: offscreen
      PYTHONIOENCODING: utf-8
      PYTHONPATH: .
```

- [ ] **Step 3: Chạy thử pytest cục bộ không dùng tiền tố biến môi trường**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_srt_translator.py -v
```
Kỳ vọng: Đạt (PASSED) mà không gặp lỗi `ModuleNotFoundError: No module named 'src'`.

- [ ] **Step 4: Cam kết git**

```powershell
git add pytest.ini .github/workflows/ci-tests.yml
git commit -m "ci: configure pytest.ini and PYTHONPATH for automated workflows"
```

---

### Task 2: Khắc phục lõi phân cụm người nói: Giả thuyết 1 người nói, sửa Silhouette và kiểm tra khoảng cách tâm cụm

**Files:**
- Modify: `src/ai_engines/speaker_diarization.py:185-300`
- Test: `test/test_omnivoice_pipeline.py`

**Interfaces:**
- Consumes: `embeddings: np.ndarray`, `valid_indices: List[int]`, `max_speakers: int = 6`
- Produces: `cosine_cluster_embeddings(...) -> Tuple[np.ndarray, List[str], List[float]]`

- [ ] **Step 1: Viết các bài kiểm thử thất bại**

Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_cosine_clustering_single_speaker_hypothesis():
    """
    Xác minh cosine_cluster_embeddings nhận diện đúng 1 người nói (k = 1)
    khi tất cả các vector đặc trưng có khoảng cách cosin nhỏ (< 0.35, tương đồng > 0.65).
    Trước đây hàm luôn ép buộc k >= 2.
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    rng = np.random.default_rng(42)
    base_vec = rng.standard_normal(192).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    # Tạo 6 đoạn thoại của cùng một người nói (nhiễu nhỏ, khoảng cách cosin < 0.20)
    samples = []
    for _ in range(6):
        noise = rng.standard_normal(192).astype(np.float32) * 0.1
        v = base_vec + noise
        v /= np.linalg.norm(v)
        samples.append(v)
    embs = np.stack(samples, axis=0)

    labels, conf_levels, scores = cosine_cluster_embeddings(embs, list(range(6)), max_speakers=4)
    unique_speakers = np.unique(labels)
    # Phải nhận diện là 1 người nói duy nhất
    assert len(unique_speakers) == 1
    assert unique_speakers[0] == 0


def test_cosine_clustering_singleton_cluster_silhouette_not_inflated():
    """
    Xác minh cụm đơn lẻ không được nhận điểm Silhouette bằng 1.0 (tránh phân mảnh mỗi câu một cụm).
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    rng = np.random.default_rng(42)
    # 2 cụm người nói rõ ràng (mỗi cụm 3 mẫu, cách xa nhau)
    v1 = rng.standard_normal(192).astype(np.float32); v1 /= np.linalg.norm(v1)
    v2 = -v1  # vuông góc hoặc đối nghịch
    v2 /= np.linalg.norm(v2)

    samples = [v1 + rng.standard_normal(192)*0.05 for _ in range(3)] + [v2 + rng.standard_normal(192)*0.05 for _ in range(3)]
    embs = np.stack([s / np.linalg.norm(s) for s in samples], axis=0)

    labels, _, _ = cosine_cluster_embeddings(embs, list(range(6)), max_speakers=4)
    # Phải nhận diện đúng 2 cụm, không tách lẻ thành 3 hoặc 4 cụm đơn
    assert len(np.unique(labels)) == 2


def test_cosine_clustering_allows_full_k_candidates():
    """
    Xác minh thuật toán cho phép thử nghiệm đủ k ứng viên lên tới min(max_speakers, n_samples).
    Với N = 5 mẫu và max_speakers = 4, trước đây bị chặn ở N//2 = 2, nay phải phân biệt được 3 người nói.
    """
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings

    # Tạo 3 nhóm người nói trực giao
    e1 = np.zeros(192, dtype=np.float32); e1[0] = 1.0
    e2 = np.zeros(192, dtype=np.float32); e2[50] = 1.0
    e3 = np.zeros(192, dtype=np.float32); e3[100] = 1.0

    samples = [e1, e1.copy(), e2, e2.copy(), e3]
    embs = np.stack(samples, axis=0)

    labels, _, _ = cosine_cluster_embeddings(embs, list(range(5)), max_speakers=4)
    # Phải nhận diện đúng 3 người nói khác nhau
    assert len(np.unique(labels)) == 3
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_cosine_clustering_single_speaker_hypothesis or test_cosine_clustering_allows_full_k_candidates" -v
```
Kỳ vọng: Thất bại với `AssertionError: assert len(unique_speakers) == 1` (trả về 2 do ép buộc $k \ge 2$) hoặc `assert len(np.unique(labels)) == 3` (bị chặn ở 2).

- [ ] **Step 3: Viết mã nguồn tối thiểu**

Trong `src/ai_engines/speaker_diarization.py`:
1. Kiểm tra giả thuyết $k = 1$:
   - Tính `max_dist = float(np.max(dist_matrix))`.
   - Tính tâm cụm toàn cầu `c_global` và khoảng cách trung bình tới tâm.
   - Nếu `max_dist < 0.35` hoặc khoảng cách trung bình tới tâm $< 0.20$: Trả về nghiệm $k = 1$ với toàn bộ nhãn bằng `0`.
2. Mở rộng dải ứng viên:
   - `max_k = max(2, min(max_speakers, n_samples))`.
3. Sửa cách tính Silhouette cho cụm đơn lẻ:
   - Nếu `not np.any(same_mask)`: `sil_scores.append(0.0)`.
4. Kiểm tra khoảng cách giữa các tâm cụm:
   - Với mỗi nghiệm K-means++, nếu tồn tại 2 tâm cụm có khoảng cách cosin $< 0.30$: Bỏ qua hoặc phạt điểm Silhouette của nghiệm này.
5. So sánh với phương án $k = 1$:
   - Nếu `best_score < 0.25`: Kết luận chỉ có 1 người nói ($k = 1$).

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_cosine_clustering_single_speaker_hypothesis or test_cosine_clustering_singleton_cluster_silhouette_not_inflated or test_cosine_clustering_allows_full_k_candidates" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết git**

```powershell
git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py
git commit -m "fix(diarization): add single speaker hypothesis and fix silhouette calculation"
```

---

### Task 3: Kích hoạt nhận diện người nói cho các câu chưa phân vai trong luồng tổng hợp

**Files:**
- Modify: `src/ai_engines/tts_engine.py:1810-1825`
- Test: `test/test_omnivoice_pipeline.py`

**Interfaces:**
- Consumes: `blocks: List[SubtitleBlock]`, `video_path: str`
- Produces: Kích hoạt `detect_speakers_from_video` hoặc `detect_speakers_from_srt_text` khi có câu chưa khóa hoặc chưa phân vai.

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_synthesize_blocks_does_not_skip_diarization_if_some_blocks_assigned(tmp_path):
    """
    Xác minh trong synthesize_blocks():
    Khi có 1 câu đã có nhãn người nói nhưng còn các câu khác chưa khóa/chưa phân vai,
    chế độ auto_video_cluster vẫn chạy nhận diện người nói thay vì bỏ qua hoàn toàn.
    """
    from unittest.mock import MagicMock
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    engine = TTSEngine(mode="clone", multi_speaker_mode="auto_video_cluster")
    engine.detect_speakers_from_video = MagicMock(return_value={})
    engine.load_model = MagicMock()
    engine._prepare_and_lock_character_prompts = MagicMock()

    vid_file = str(tmp_path / "mock_vid.mp4")
    Path(vid_file).write_bytes(b"mock")

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu đã phân vai", speaker="Nhân vật Cũ", is_speaker_locked=True)
    b2 = SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Câu chưa phân vai", speaker="")

    engine.synthesize_blocks(vid_file, [b1, b2], output_audio_path=str(tmp_path / "out.wav"))

    # Phải gọi detect_speakers_from_video vì vẫn còn câu b2 chưa phân vai
    assert engine.detect_speakers_from_video.called is True
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_synthesize_blocks_does_not_skip_diarization_if_some_blocks_assigned" -v
```
Kỳ vọng: Thất bại với `AssertionError: assert False is True`.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

Trong `src/ai_engines/tts_engine.py`:
Sửa điều kiện kiểm tra trong `synthesize_blocks()`:
```python
        has_unassigned_or_unlocked = any(
            not getattr(b, "is_speaker_locked", False)
            or not getattr(b, "speaker", "").strip()
            or getattr(b, "speaker", "") in ("Chưa phân vai", "Chưa xác định")
            for b in blocks
        )

        if self.multi_speaker_mode == "auto_video_cluster" and video_path and os.path.exists(video_path):
            if has_unassigned_or_unlocked:
                self.detect_speakers_from_video(video_path, blocks, max_speakers=self.max_speakers)
        elif self.multi_speaker_mode == "auto_srt_text":
            if has_unassigned_or_unlocked:
                self.detect_speakers_from_srt_text(blocks, max_speakers=self.max_speakers)
```

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_synthesize_blocks_does_not_skip_diarization_if_some_blocks_assigned" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết git**

```powershell
git add src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py
git commit -m "fix(tts): do not skip diarization when some blocks remain unassigned or unlocked"
```

---

### Task 4: Đồng bộ hóa chế độ giọng cho nghe thử câu phụ đề (block_preview)

**Files:**
- Modify: `src/desktop/ui/omnivoice_interface.py:325-345, 3000-3025, 3180-3220`
- Test: `test/test_omnivoice_pipeline.py`

**Interfaces:**
- Consumes: `_launch_worker(task_type="block_preview", ..., override_mode=...)`
- Produces: `OmniVoiceTaskWorker` nhận và chuyển `override_mode` vào `engine.generate_single()`.

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_block_preview_uses_correct_override_mode_and_voice(tmp_path):
    """
    Xác minh nghe thử một câu phụ đề (block_preview):
    Khi câu được gán phương pháp preset, worker truyền override_mode='design'
    vào generate_single bất kể chế độ toàn cục của engine là clone.
    """
    from unittest.mock import MagicMock
    from src.desktop.ui.omnivoice_interface import OmniVoiceTaskWorker

    captured = {}
    def fake_generate_single(text, **kwargs):
        captured.update(kwargs)
        return str(tmp_path / "mock.wav")

    mock_engine = MagicMock()
    mock_engine.generate_single = MagicMock(side_effect=fake_generate_single)

    worker = OmniVoiceTaskWorker(
        task_type="block_preview",
        engine_params={"mode": "clone"},
        text="Xin chào",
        output_path=str(tmp_path / "preview.wav"),
        override_mode="design",
        override_instruct="female sweet voice",
    )
    worker._active_engine = mock_engine
    worker._run_engine_task(mock_engine)

    assert captured.get("override_mode") == "design"
    assert captured.get("override_instruct") == "female sweet voice"
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_block_preview_uses_correct_override_mode_and_voice" -v
```
Kỳ vọng: Thất bại với `assert captured.get("override_mode") == "design"`.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

1. Trong `OmniVoiceTaskWorker`:
- Bổ sung tham số `override_mode: Optional[str] = None` vào `__init__`.
- Trong `_run_engine_task`: Với `self.task_type in ("single_synth", "block_preview", "vn_catalog_preview", "character_audition")`:
  ```python
  override_mode = self.override_mode
  if not override_mode:
      if self.override_instruct and not self.override_ref_audio:
          override_mode = "design"
      elif self.override_ref_audio and not self.override_instruct:
          override_mode = "clone"
  ```
  Truyền `override_mode=override_mode` vào `engine.generate_single(...)`.

2. Trong `_preview_selected_subtitle_row`:
- Xác định phương pháp của nhân vật ở dòng đang chọn:
  Nếu là preset: `override_mode = "design"`, `row_ref = ""`, `row_ins = voice_ins`.
  Nếu là clone: `override_mode = "clone"`, `row_ref = ref_path`, `row_ins = ""`.
- Truyền `override_mode` vào `_launch_worker("block_preview", ..., override_mode=override_mode)`.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_block_preview_uses_correct_override_mode_and_voice" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết git**

```powershell
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "fix(ui): pass correct override_mode to block_preview to align with export"
```

---

### Task 5: Chuẩn hóa nhãn "Chưa phân vai", trạng thái hồ sơ và các nút nghe thử

**Files:**
- Modify: `src/desktop/ui/omnivoice_interface.py:1690-1760, 1930-1960, 2170-2210, 2670-2685`
- Test: `test/test_omnivoice_pipeline.py`

**Interfaces:**
- Consumes: `speaker_profiles`, `subtitle_blocks`
- Produces: Bảng phụ đề hiển thị "Chưa phân vai"; thẻ tổng quan không đếm "Chưa phân vai"; `_audition_all_speakers` duyệt lần lượt từng nhân vật.

- [ ] **Step 1: Viết các bài kiểm thử thất bại**

Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_character_overview_ignores_unassigned_labels():
    """
    Xác minh _update_character_profiles_overview không tính nhãn 'Chưa phân vai'
    hoặc 'Chưa xác định' vào tổng số nhân vật nhận diện được.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.is_speakers_detected = True
    ui.speaker_profiles = {"Nhân vật 1": {"method": "preset", "status": "locked", "voice_instruct": "male"}}
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một", speaker="Nhân vật 1"),
        SubtitleBlock(2, "00:00:04,000", "00:00:06,000", "Câu hai", speaker="Chưa phân vai"),
    ]

    ui._update_character_profiles_overview()
    # Chỉ có 1 nhân vật duy nhất
    assert len(ui._character_cards) == 1
    assert "1 nhân vật" in ui.lbl_char_count.text()


def test_clone_method_without_sample_marks_review_needed():
    """
    Xác minh khi người dùng đổi phương pháp sang clone nhưng chưa có tệp mẫu âm thanh,
    trạng thái hồ sơ phải là review_needed, không được hiển thị là locked.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    prof = CharacterVoiceProfile(speaker_id="spk_test", display_name="Nhân vật Mới", source_audio_path="")
    ui.speaker_profiles = {"Nhân vật Mới": prof}

    ui._on_character_method_changed("Nhân vật Mới", "Nhân bản giọng từ video")
    assert prof.status == "review_needed"
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_character_overview_ignores_unassigned_labels or test_clone_method_without_sample_marks_review_needed" -v
```
Kỳ vọng: Thất bại với `assert len(ui._character_cards) == 1` (hiện tại tính cả Chưa phân vai thành 2) và `assert prof.status == "review_needed"` (hiện tại gán cứng locked).

- [ ] **Step 3: Viết mã nguồn tối thiểu**

1. Trong `_populate_subtitle_table`:
   - Dòng 2677: `spk = getattr(blk, "speaker", "") or "Chưa phân vai"`.
2. Trong `_update_character_profiles_overview`:
   - Bỏ qua các chuỗi `spk in ("Chưa phân vai", "Chưa xác định", "")`.
3. Trong `_on_character_method_changed`:
   - Khi chọn "Nhân bản giọng từ video":
     `has_sample = bool(ref_path and os.path.exists(ref_path))`
     `status = "locked" if has_sample else "review_needed"`
4. Trong `_audition_all_speakers`:
   - Duyệt qua danh sách `self._character_cards` và phát lần lượt từng nhân vật kèm thông báo tiến trình.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_character_overview_ignores_unassigned_labels or test_clone_method_without_sample_marks_review_needed" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết git**

```powershell
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "fix(ui): standardize unassigned speaker label, profile review status, and audition controls"
```

---

### Task 6: Thêm hộp thoại xác nhận cảnh báo trước khi xuất video/âm thanh

**Files:**
- Modify: `src/desktop/ui/omnivoice_interface.py:3020-3075`
- Test: `test/test_omnivoice_pipeline.py`

**Interfaces:**
- Consumes: `issues = self._run_pre_export_quality_check()`
- Produces: Hộp thoại xác nhận `MessageBox` hiển thị cảnh báo phân vai; dừng tác vụ nếu người dùng bấm hủy.

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_export_shows_warning_confirmation_dialog(monkeypatch):
    """
    Xác minh khi danh sách phụ đề có cảnh báo chưa phân vai:
    Hàm xuất gọi hộp thoại xác nhận. Nếu người dùng chọn từ chối (False),
    hệ thống dừng lại mà không mở hộp thoại lưu tệp.
    """
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Câu một", speaker="Chưa phân vai")
    ]

    mock_confirm = MagicMock(return_value=False)
    monkeypatch.setattr(ui, "_confirm_pre_export_warnings", mock_confirm)
    ui._launch_worker = MagicMock()

    ui._export_subtitle_audio_track()

    assert mock_confirm.called is True
    assert ui._launch_worker.called is False
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_export_shows_warning_confirmation_dialog" -v
```
Kỳ vọng: Thất bại vì `_confirm_pre_export_warnings` chưa tồn tại.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

Trong `src/desktop/ui/omnivoice_interface.py`:
1. Thêm phương thức `_confirm_pre_export_warnings(self, warnings: List[Dict[str, Any]]) -> bool`:
   - Tạo hộp thoại xác nhận `MessageBox` thông báo danh sách câu chưa gán nhân vật hoặc cần kiểm tra.
   - Trả về `True` nếu người dùng chọn "Tiếp Tục Xuất", trả về `False` nếu chọn "Quay Lại Chỉnh Sửa".
2. Trong `_export_subtitle_audio_track` và `_dub_and_merge_video`:
   - Lọc các mục có `severity == "warning"`.
   - Nếu có cảnh báo: gọi `if not self._confirm_pre_export_warnings(warns): return`.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_export_shows_warning_confirmation_dialog" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết git**

```powershell
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(ui): add pre-export confirmation dialog for unassigned speakers"
```

---

### Task 7: Kiểm thử hồi quy toàn diện hệ thống và nghiệm thu

**Files:**
- Test: `test/test_auto_pipeline_reliability.py`, `test/test_omnivoice_pipeline.py`, `test/test_srt_translator.py`, `test/test_subtitle_and_logo_remover.py`, `test_home_interface.py`

- [ ] **Step 1: Chạy toàn bộ các bộ kiểm thử của dự án**

```powershell
.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py test/test_srt_translator.py test/test_subtitle_and_logo_remover.py test_home_interface.py -v
```
Kỳ vọng: Đạt 100% (toàn bộ 105+ bài kiểm thử đều PASSED).

- [ ] **Step 2: Đẩy toàn bộ thay đổi lên máy chủ Git**

```powershell
git push origin main
```
