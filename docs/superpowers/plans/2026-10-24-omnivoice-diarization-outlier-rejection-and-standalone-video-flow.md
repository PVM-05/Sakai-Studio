# Kế Hoạch Triển Khai: Cơ Chế Từ Chối Ép Gán, Lọc Dải Tần Giọng Nói Và Luồng Nhận Diện Video Độc Lập

> **Dành cho kỹ sư và tác nhân thực thi:** BẮT BUỘC tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD). Mỗi nhiệm vụ đều có các bước nhỏ được đánh dấu checkbox (`- [ ]`). Cam kết git sau mỗi nhiệm vụ hoàn thành.

**Mục tiêu:** Khắc phục triệt để việc ép gán đoạn thoại ngắn vào cụm gần nhất, nâng cấp VAD lọc dải tần giọng nói để chống nhầm tiếng ồn/nhạc nền, và mở thông toàn bộ luồng giao diện cho video độc lập khi không có tệp phụ đề.

**Kiến trúc:** Cập nhật `cosine_cluster_embeddings` và `detect_speech_segments_vad` trong `src/ai_engines/speaker_diarization.py`; cập nhật `_update_identify_button_state`, `_run_auto_detect_speakers`, `_on_speakers_detected` và bổ sung tính năng xuất phụ đề trong `src/desktop/ui/omnivoice_interface.py`.

**Ràng buộc bắt buộc toàn cục:**
- 100% tiếng Việt chuẩn mực trong toàn bộ mã nguồn, nhãn, thông báo và tài liệu.
- Tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình khối trang trí.
- An toàn bộ nhớ dưới 800 MB VRAM trên GPU NVIDIA GeForce RTX 4050 Laptop.
- Thực thi trực tiếp tuần tự (Inline execution), không tạo subagent.

---

### Nhiệm Vụ 1: Cơ Chế Từ Chối Ép Gán (Outlier Rejection) Cho Đoạn Thoại Ngoài Nhóm Nòng Cốt

**Tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_cosine_clustering_outlier_rejection_marks_distant_samples_unassigned():
    """
    Xác minh trong cơ chế mẫu nòng cốt (anchor-based):
    Nếu một đoạn thoại ngoài nòng cốt có khoảng cách cosin lớn (> 0.45) tới tất cả các tâm cụm nòng cốt,
    hệ thống KHÔNG được ép gán vào người gần nhất mà phải từ chối gán (unassigned).
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings, ConfidenceLevel

    # 2 mẫu nòng cốt của 2 nhân vật (dài 2.0s)
    spk1_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spk2_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # Mẫu 3: Đoạn ngắn 0.3s, có vector thuộc chiều thứ 4 [0, 0, 0, 1] (khoảng cách cosin = 1.0 tới cả 2 người)
    outlier_emb = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    embs = np.vstack([spk1_emb, spk2_emb, outlier_emb])
    durations = [2.0, 2.0, 0.3]

    labels, conf_levels, conf_scores = cosine_cluster_embeddings(
        embs,
        valid_indices=[0, 1, 2],
        max_speakers=2,
        exact_speakers=2,
        distance_threshold=0.40,
        sample_durations=durations,
    )

    # 2 mẫu đầu được gán vào 2 cụm khác nhau
    assert labels[0] != labels[1]
    assert conf_levels[0] == ConfidenceLevel.HIGH.value
    assert conf_levels[1] == ConfidenceLevel.HIGH.value

    # Mẫu thứ 3 bị từ chối gán, chuyển sang Chưa xác định
    assert conf_levels[2] == ConfidenceLevel.UNASSIGNED.value
    assert conf_scores[2] == 0.0
    assert labels[2] == -1
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_cosine_clustering_outlier_rejection_marks_distant_samples_unassigned" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/ai_engines/speaker_diarization.py` tại `cosine_cluster_embeddings`:
Khi gán `non_anchor`:
```python
        non_anchor = [idx for idx in range(n_samples) if idx not in anchor_indices]
        for idx in non_anchor:
            dists = [1.0 - float(np.dot(norm_embs[idx], cluster_centers[c_i])) for c_i in unique_k]
            min_dist_idx = int(np.argmin(dists))
            min_dist = dists[min_dist_idx]
            if min_dist > distance_threshold:
                # Từ chối gán khi khoảng cách quá xa so với mọi cụm nòng cốt
                best_labels[idx] = -1
            else:
                best_labels[idx] = unique_k[min_dist_idx]
```
Và trong vòng lặp tính độ tin cậy:
Nếu `best_labels[idx] == -1`:
`conf_levels.append(ConfidenceLevel.UNASSIGNED.value)`
`conf_scores.append(0.0)`
`continue`

Đồng thời trong `process_video_subtitles`:
Nếu `cluster_id == -1`:
`blk.speaker = "Chưa xác định"`
`blk.confidence_level = ConfidenceLevel.UNASSIGNED.value`

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_cosine_clustering_outlier_rejection_marks_distant_samples_unassigned" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py; git commit -m "feat(diarization): implement outlier rejection for non-anchor segments"`

---

### Nhiệm Vụ 2: Bộ Lọc Dải Tần Giọng Nói Cho VAD Nhằm Loại Trừ Tiếng Ồn Cao Tần Và Nhạc Nền

**Tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_vad_spectral_band_filters_out_high_frequency_noise():
    """
    Xác minh hàm detect_speech_segments_vad:
    1. Tín hiệu nhiễu cao tần (tần số 7000 Hz) có năng lượng RMS lớn nhưng bị loại bỏ do ngoài dải giọng nói.
    2. Tín hiệu giọng nói (tần số 400 Hz) có năng lượng tương đương được giữ lại chính xác.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import detect_speech_segments_vad

    sr = 16000
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    # Tín hiệu nhiễu cao tần (tiếng súng/rít kim loại) 7000 Hz
    hf_noise = (0.5 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    # Tín hiệu giọng nói con người 400 Hz
    voice = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    audio = np.concatenate([hf_noise, silence, voice])

    segs = detect_speech_segments_vad(audio, sample_rate=sr, min_speech_sec=0.3)
    assert len(segs) == 1
    # Chỉ có đoạn giọng nói (từ 1.5s đến 2.5s) được công nhận
    assert 1.4 <= segs[0][0] <= 1.6
    assert 2.4 <= segs[0][1] <= 2.6
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_vad_spectral_band_filters_out_high_frequency_noise" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/ai_engines/speaker_diarization.py` tại `detect_speech_segments_vad`:
Phân tích dải tần giọng nói trên từng khung:
Tính FFT thực `rfft`:
`fft_vals = np.abs(np.fft.rfft(frames, axis=1))`
`freqs = np.fft.rfftfreq(frame_len, d=1.0/sample_rate)`
`voice_mask = (freqs >= 200.0) & (freqs <= 4000.0)`
`voice_power = np.sum(fft_vals[:, voice_mask] ** 2, axis=1)`
`total_power = np.maximum(np.sum(fft_vals ** 2, axis=1), 1e-12)`
`voice_ratio = voice_power / total_power`
Điều kiện phát hiện hoạt động giọng nói:
`is_speech = (rms >= threshold) & (voice_ratio >= 0.40)`

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_vad_spectral_band_filters_out_high_frequency_noise" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py; git commit -m "feat(diarization): add voice frequency band energy filter to VAD"`

---

### Nhiệm Vụ 3: Mở Thông Hoàn Toàn Luồng Giao Diện Cho Video Độc Lập Khi Chưa Có SRT

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_auto_pipeline_reliability.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_auto_pipeline_reliability.py`:
```python
def test_step0_identify_button_enabled_with_video_only(self):
    """
    Xác minh nút nhận diện nhân vật tại Bước 1 mở khi chỉ có video (chưa có SRT):
    1. Nút btn_identify_characters được kích hoạt (enabled=True).
    2. Nhãn hiển thị thông báo sẵn sàng nhận diện trực tiếp từ video qua VAD.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    import tempfile, os

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Tạo tệp video giả lập hợp lệ
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy video content")
        temp_v = f.name

    try:
        ui.subtitle_blocks = []
        ui.current_video_path = temp_v
        ui._update_identify_button_state()

        self.assertTrue(ui.btn_identify_characters.isEnabled())
        hint = ui.lbl_detect_source_hint.text()
        self.assertIn("video", hint.lower())
        self.assertIn("tự động", hint.lower())
    finally:
        if os.path.exists(temp_v):
            os.remove(temp_v)


def test_speakers_detected_populates_empty_subtitles_from_worker(self):
    """
    Xác minh khi phụ đề ban đầu rỗng, _on_speakers_detected tiếp nhận
    toàn bộ các câu thoại do worker sinh ra và nạp vào bảng phụ đề.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from unittest.mock import MagicMock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()
    ui.subtitle_blocks = []

    # Giả lập worker đã sinh 2 câu thoại từ VAD
    worker = MagicMock()
    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Đoạn thoại 1")
    b1.speaker = "Người nói 1"
    b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Đoạn thoại 2")
    b2.speaker = "Người nói 2"
    worker.subtitle_blocks = [b1, b2]
    ui._worker = worker

    profiles = {
        "Người nói 1": CharacterVoiceProfile(speaker_id="spk_0", display_name="Người nói 1"),
        "Người nói 2": CharacterVoiceProfile(speaker_id="spk_1", display_name="Người nói 2"),
    }
    ui._on_speakers_detected(profiles)

    # UI phải tiếp nhận đủ 2 câu thoại
    self.assertEqual(len(ui.subtitle_blocks), 2)
    self.assertEqual(ui.sub_table.rowCount(), 2)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_step0_identify_button_enabled_with_video_only or test_speakers_detected_populates_empty_subtitles_from_worker" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/desktop/ui/omnivoice_interface.py`:
1. `_update_identify_button_state`:
   - Nếu `not has_subs and has_video`:
     `self.btn_identify_characters.setEnabled(True)`
     `self.lbl_detect_source_hint.setText("Sẵn sàng nhận diện trực tiếp từ video: hệ thống sẽ tự động phân đoạn giọng nói và phân vai nhân vật.")`
2. `_run_auto_detect_speakers`:
   - Thay `if not self.subtitle_blocks: return` bằng kiểm tra nếu cả video lẫn phụ đề đều thiếu thì mới cảnh báo.
3. `_on_speakers_detected`:
   - Nếu `not self.subtitle_blocks and self._worker and self._worker.subtitle_blocks`:
     `self.subtitle_blocks = list(self._worker.subtitle_blocks)`

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_step0_identify_button_enabled_with_video_only or test_speakers_detected_populates_empty_subtitles_from_worker" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_auto_pipeline_reliability.py; git commit -m "feat(omnivoice): enable standalone video character detection without SRT on UI"`

---

### Nhiệm Vụ 4: Bổ Sung Nút Xuất Phụ Đề SRT Từ Các Câu Thoại Tự Sinh

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_auto_pipeline_reliability.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_auto_pipeline_reliability.py`:
```python
def test_export_detected_subtitles_to_srt_file(self):
    """
    Xác minh phương thức _export_subtitles_to_srt tạo đúng tệp SRT
    với đầy đủ mốc thời gian và tên nhân vật.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    import tempfile, os

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,500", "Chào bạn")
    b1.speaker = "Nhân vật A"
    b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,500", "Hẹn gặp lại")
    b2.speaker = "Nhân vật B"
    ui.subtitle_blocks = [b1, b2]

    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
        out_srt = f.name

    try:
        ok = ui._export_subtitles_to_srt(out_srt)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(out_srt))
        with open(out_srt, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("00:00:01,000 --> 00:00:02,500", content)
        self.assertIn("Chào bạn", content)
        self.assertIn("00:00:03,000 --> 00:00:04,500", content)
    finally:
        if os.path.exists(out_srt):
            os.remove(out_srt)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_export_detected_subtitles_to_srt_file" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Thêm phương thức `_export_subtitles_to_srt(self, target_path: str = "") -> bool` vào `OmniVoiceInterface`:
- Duyệt qua `self.subtitle_blocks`, định dạng chuẩn SRT với chỉ số, mốc thời gian, nội dung câu thoại và ghi UTF-8.
- Bổ sung nút `btn_export_subtitles` vào hàng công cụ phụ đề trên giao diện kết nối tới `_export_subtitles_to_srt`.

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_export_detected_subtitles_to_srt_file" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_auto_pipeline_reliability.py; git commit -m "feat(omnivoice): add export subtitles to SRT functionality"`

---

### Nhiệm Vụ 5: Kiểm Thử Toàn Diện Và Đẩy Mã Nguồn Lên GitHub

- [ ] **Bước 1: Chạy toàn bộ 115+ bài kiểm thử**
`.\venv\Scripts\pytest test/test_omnivoice_pipeline.py test/test_auto_pipeline_reliability.py -q`
- [ ] **Bước 2: Rà soát ngôn ngữ chuẩn mực**
Đảm bảo 100% tiếng Việt chuẩn mực, không từ tiếng Anh trong ngoặc đơn, không biểu tượng cảm xúc.
- [ ] **Bước 3: Đẩy commit lên GitHub**
`git push origin main`
