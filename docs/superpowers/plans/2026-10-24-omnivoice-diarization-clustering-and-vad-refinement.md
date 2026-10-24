# Kế Hoạch Triển Khai: Cải Tiến Thuật Toán Phân Cụm Người Nói Và Phân Đoạn Âm Thanh Tinh Chỉnh Cho OmniVoice

> **Dành cho kỹ sư và tác nhân thực thi:** BẮT BUỘC tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD). Mỗi nhiệm vụ đều có các bước nhỏ được đánh dấu checkbox (`- [ ]`). Cam kết git sau mỗi nhiệm vụ hoàn thành.

**Mục tiêu:** Cải tiến thuật toán phân cụm người nói trong OmniVoice (loại bỏ ép gộp sai lệch về 1 người), tích hợp bộ phát hiện hoạt động giọng nói VAD để tinh chỉnh lát cắt âm thanh và hỗ trợ nhận diện video không có phụ đề, chuẩn hóa điểm tin cậy thực chất.

**Kiến trúc:** Cập nhật `_agglomerative_clustering` trong `src/ai_engines/speaker_diarization.py`; bổ sung `detect_speech_segments_vad` và `trim_silence_vad`; nâng cấp công thức tính độ tin cậy trong `cosine_cluster_embeddings`; mở rộng `process_video_subtitles` để tự động tạo khối thoại khi thiếu SRT.

**Ngăn xếp kỹ thuật:** Python 3.10+, NumPy, SoundFile, ONNX Runtime (CAM++ ONNX), Pytest.

## Ràng Buộc Bắt Buộc Toàn Cục

- 100% tiếng Việt chuẩn mực trong toàn bộ mã nguồn, thông báo, nhãn và tài liệu.
- Tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình khối trang trí.
- Duy trì an toàn tài nguyên bộ nhớ dưới 800 MB VRAM trên GPU NVIDIA GeForce RTX 4050 Laptop 6 GB VRAM.
- Tuân thủ quy trình TDD: viết bài kiểm thử thất bại trước, viết mã nguồn tối thiểu sau.
- Thực thi trực tiếp tuần tự (Inline execution), không tạo subagent.

---

### Nhiệm Vụ 1: Loại Bỏ Ép Gộp Về Một Người Khi Silhouette Thấp Trong Phân Cụm

**Tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
Trước đây dòng 382-384 ép gộp `clusters = [[i for i in range(n_samples)]]` khi `mean_sil < 0.20`. Thay vào đó, ta giữ nguyên các cụm đã phân tách và trả về cờ cảnh báo hoặc phân loại độ tin cậy, bảo vệ sự tách biệt của các nhân vật khác nhau.

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_agglomerative_clustering_does_not_collapse_multispeakers_on_low_silhouette():
    """
    Xác minh thuật toán _agglomerative_clustering không ép gộp toàn bộ về 1 người
    khi các mẫu có độ phân tách cosin rõ ràng giữa 2 người khác nhau nhưng hệ số Silhouette thấp.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import _agglomerative_clustering

    # Giả lập 2 nhóm vector đặc trưng của 2 người nói khác nhau
    # Người 1: vector quanh hướng [1, 0, 0]
    # Người 2: vector quanh hướng [0, 1, 0]
    # Khoảng cách cosin giữa 2 người là ~1.0 (rất xa nhau)
    spk1_embs = np.array([
        [0.98, 0.10, 0.05],
        [0.95, 0.15, 0.08],
        [0.97, 0.08, 0.12],
    ], dtype=np.float32)
    spk2_embs = np.array([
        [0.10, 0.98, 0.05],
        [0.15, 0.95, 0.08],
        [0.08, 0.97, 0.12],
    ], dtype=np.float32)

    embs = np.vstack([spk1_embs, spk2_embs])
    # Chuẩn hóa L2
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    labels = _agglomerative_clustering(embs, threshold=0.40, exact_k=None, max_k=4)
    unique_clusters = np.unique(labels)

    # Phải phát hiện đúng 2 người nói riêng biệt, KHÔNG bị gộp thành 1 cụm
    assert len(unique_clusters) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_agglomerative_clustering_does_not_collapse_multispeakers_on_low_silhouette" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/ai_engines/speaker_diarization.py`:
Tìm hàm `_agglomerative_clustering`:
Xóa bỏ đoạn cưỡng bức gộp `if mean_sil < 0.20: clusters = [[i for i in range(n_samples)]]`.
Bảo toàn danh sách các cụm đã gom được dựa trên `threshold` khoảng cách cosin.

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_agglomerative_clustering_does_not_collapse_multispeakers_on_low_silhouette" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py; git commit -m "fix(diarization): remove premature single-speaker collapse on low silhouette"`

---

### Nhiệm Vụ 2: Bộ Phát Hiện Hoạt Động Giọng Nói VAD Và Tinh Chỉnh Cắt Âm Thanh

**Tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
Bổ sung hàm `detect_speech_segments_vad` và `trim_silence_vad` để loại bỏ khoảng lặng/nhạc nền ở đầu và cuối câu trước khi đưa vào CAM++ ONNX, giúp vector đặc trưng chính xác hơn.

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_detect_speech_segments_vad_identifies_speech_and_trims_silence():
    """
    Xác minh hàm VAD và cắt khoảng lặng:
    1. Phát hiện đúng đoạn có tiếng nói trong tín hiệu âm thanh có chứa khoảng lặng.
    2. trim_silence_vad cắt bỏ khoảng lặng ở 2 đầu, giữ trọn vẹn phần giọng nói.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import detect_speech_segments_vad, trim_silence_vad

    sr = 16000
    # Tạo 1 giây im lặng + 1.5 giây tín hiệu có tiếng (sin) + 1 giây im lặng
    silence_start = np.zeros(int(1.0 * sr), dtype=np.float32)
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    speech = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    silence_end = np.zeros(int(1.0 * sr), dtype=np.float32)

    full_audio = np.concatenate([silence_start, speech, silence_end])

    segments = detect_speech_segments_vad(full_audio, sample_rate=sr, min_speech_sec=0.3)
    assert len(segments) >= 1
    # Đoạn tiếng nói phải nằm trong khoảng từ 0.9s đến 2.6s
    seg_start, seg_end = segments[0]
    assert 0.8 <= seg_start <= 1.1
    assert 2.4 <= seg_end <= 2.7

    # trim_silence_vad phải cắt bỏ 2 giây im lặng
    trimmed = trim_silence_vad(full_audio, sample_rate=sr)
    trimmed_duration = len(trimmed) / sr
    assert 1.3 <= trimmed_duration <= 1.7
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_detect_speech_segments_vad_identifies_speech_and_trims_silence" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Thêm `detect_speech_segments_vad` và `trim_silence_vad` vào `src/ai_engines/speaker_diarization.py`:
```python
def detect_speech_segments_vad(
    audio_data: np.ndarray,
    sample_rate: int,
    min_speech_sec: float = 0.3,
    frame_ms: int = 30,
    energy_threshold_ratio: float = 0.15,
) -> List[Tuple[float, float]]:
    """
    Phát hiện các đoạn có hoạt động giọng nói dựa trên năng lượng RMS khung ngắn hạn.
    Trả về danh sách các mốc thời gian (bắt đầu, kết thúc) tính bằng giây.
    """
    if audio_data is None or len(audio_data) == 0 or sample_rate <= 0:
        return []

    data = np.asarray(audio_data, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
    n_frames = len(data) // frame_len
    if n_frames == 0:
        return []

    frames = data[:n_frames * frame_len].reshape((n_frames, frame_len))
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    max_rms = float(np.max(rms)) if rms.size > 0 else 0.0
    if max_rms < 1e-4:
        return []

    # Ngưỡng năng lượng tương đối có sàn an toàn
    threshold = max(0.005, max_rms * energy_threshold_ratio)
    is_speech = rms >= threshold

    segments: List[Tuple[float, float]] = []
    in_speech = False
    start_frame = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_frame = i
        elif not active and in_speech:
            in_speech = False
            start_s = start_frame * (frame_ms / 1000.0)
            end_s = i * (frame_ms / 1000.0)
            if end_s - start_s >= min_speech_sec:
                segments.append((start_s, end_s))

    if in_speech:
        start_s = start_frame * (frame_ms / 1000.0)
        end_s = n_frames * (frame_ms / 1000.0)
        if end_s - start_s >= min_speech_sec:
            segments.append((start_s, end_s))

    # Gộp các đoạn cách nhau dưới 0.3 giây
    merged: List[Tuple[float, float]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev_s, prev_e = merged[-1]
            if seg[0] - prev_e <= 0.30:
                merged[-1] = (prev_s, max(prev_e, seg[1]))
            else:
                merged.append(seg)

    return merged


def trim_silence_vad(
    chunk: np.ndarray,
    sample_rate: int,
    padding_sec: float = 0.05,
) -> np.ndarray:
    """
    Cắt bỏ khoảng lặng ở hai đầu đoạn âm thanh, giữ lại khoảng đệm an toàn.
    """
    if chunk is None or len(chunk) == 0:
        return chunk
    segs = detect_speech_segments_vad(chunk, sample_rate, min_speech_sec=0.15)
    if not segs:
        return chunk
    s_sec = max(0.0, segs[0][0] - padding_sec)
    e_sec = min(len(chunk) / float(sample_rate), segs[-1][1] + padding_sec)
    start_idx = int(s_sec * sample_rate)
    end_idx = int(e_sec * sample_rate)
    if end_idx - start_idx >= int(0.20 * sample_rate):
        return chunk[start_idx:end_idx]
    return chunk
```
Và trong `process_video_subtitles`:
Trước khi append `chunk` vào `audio_chunks`, chạy `chunk = trim_silence_vad(chunk, sr)`.

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_detect_speech_segments_vad_identifies_speech_and_trims_silence" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py; git commit -m "feat(diarization): add VAD speech segment detection and audio slice trimming"`

---

### Nhiệm Vụ 3: Đo Điểm Tin Cậy Hiệu Chuẩn Thực Chất

**Tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
Tính điểm tin cậy `conf_scores` và `conf_levels` dựa trên độ tương đồng cosin tới tâm cụm, độ dài đoạn âm thanh (phạt câu thoại quá ngắn), và khoảng cách phân tách tới cụm gần nhất.

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_calibrated_confidence_score_penalizes_short_and_ambiguous_utterances():
    """
    Xác minh thuật toán cosine_cluster_embeddings:
    1. Câu thoại dài, nằm gần tâm cụm có độ tin cậy 'high'.
    2. Câu thoại quá ngắn (< 0.5s) hoặc nằm ở ranh giới giữa 2 người nói bị hạ mức 'review_needed'.
    """
    import numpy as np
    from src.ai_engines.speaker_diarization import cosine_cluster_embeddings, ConfidenceLevel

    # 2 cụm người nói rõ ràng
    spk1_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    spk2_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Mẫu 1: Rất chuẩn người 1, dài 2.0s -> HIGH
    m1 = spk1_emb.copy()
    # Mẫu 2: Nằm gần ranh giới giữa 2 người (45 độ) -> REVIEW_NEEDED
    m2 = np.array([0.707, 0.707, 0.0], dtype=np.float32)
    # Mẫu 3: Rất chuẩn người 2 nhưng siêu ngắn 0.3s -> REVIEW_NEEDED
    m3 = spk2_emb.copy()

    embs = np.vstack([m1, m2, m3])
    durations = [2.0, 2.0, 0.3]

    labels, conf_levels, conf_scores = cosine_cluster_embeddings(
        embs,
        valid_indices=[0, 1, 2],
        max_speakers=2,
        exact_speakers=2,
        sample_durations=durations,
    )

    assert conf_levels[0] == ConfidenceLevel.HIGH.value
    assert conf_levels[1] == ConfidenceLevel.REVIEW_NEEDED.value
    assert conf_levels[2] == ConfidenceLevel.REVIEW_NEEDED.value
    assert conf_scores[0] > conf_scores[1]
    assert conf_scores[0] > conf_scores[2]
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_calibrated_confidence_score_penalizes_short_and_ambiguous_utterances" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `cosine_cluster_embeddings`:
Cải tiến việc tính `conf_score`:
```python
        # 1. Tương đồng cosin với tâm cụm gán
        center = cluster_centers[c_i]
        sim_center = float(np.dot(vec, center))
        sim_score = max(0.0, min(1.0, sim_center))

        # 2. Hệ số thời lượng mẫu (phạt mẫu dưới 1.0 giây)
        dur = sample_durations[i] if sample_durations and i < len(sample_durations) else 1.5
        len_factor = min(1.0, max(0.3, dur / 1.5))

        # 3. Độ phân tách với các tâm cụm khác
        other_sims = [float(np.dot(vec, cluster_centers[other_c])) for other_c in unique_k if other_c != c_i]
        max_other_sim = max(other_sims) if other_sims else -1.0
        sep_margin = max(0.0, min(1.0, 1.0 - max(0.0, max_other_sim)))

        # Điểm tổng hợp hiệu chuẩn
        if len(unique_k) > 1:
            score = 0.45 * sim_score + 0.35 * len_factor + 0.20 * sep_margin
        else:
            score = 0.60 * sim_score + 0.40 * len_factor

        score = float(max(0.0, min(1.0, score)))
        if score >= 0.70 and dur >= 0.8 and (not other_sims or (sim_center - max_other_sim >= 0.20)):
            level = ConfidenceLevel.HIGH.value
        elif score >= 0.35:
            level = ConfidenceLevel.REVIEW_NEEDED.value
        else:
            level = ConfidenceLevel.UNASSIGNED.value
```

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_calibrated_confidence_score_penalizes_short_and_ambiguous_utterances" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py; git commit -m "feat(diarization): calibrate confidence scores with duration factor and separation margin"`

---

### Nhiệm Vụ 4: Hỗ Trợ Nhận Diện Người Nói Trực Tiếp Từ Video Khi Không Có Tệp Phụ Đề

**Tệp tin:**
- Sửa đổi: `src/ai_engines/speaker_diarization.py`
- Sửa đổi: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
Khi người dùng chỉ nạp video và bấm nhận diện (chưa nạp SRT), hệ thống tự động chạy VAD để trích xuất các phân đoạn thoại, tạo danh sách câu thoại và nhận diện nhân vật thành công.

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_diarization_supports_video_without_srt(tmp_path):
    """
    Xác minh khi truyền blocks=[] (chưa có phụ đề):
    1. process_video_subtitles tự động phân đoạn âm thanh qua VAD.
    2. Tự sinh các khối câu thoại và nhận diện đúng các nhân vật.
    3. Trả về hồ sơ nhân vật và điền đầy đủ danh sách blocks.
    """
    import numpy as np
    import soundfile as sf
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator

    sr = 16000
    # Tạo 2 phân đoạn âm thanh của 2 người nói
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    spk1 = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    spk2 = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)

    audio = np.concatenate([spk1, silence, spk2])
    wav_path = str(tmp_path / "multi_speech.wav")
    sf.write(wav_path, audio, sr)

    coord = SpeakerDiarizationCoordinator(allow_mock=True, prefer_cuda=False)
    empty_blocks = []
    profiles = coord.process_video_subtitles(
        video_path=wav_path,
        blocks=empty_blocks,
        max_speakers=2,
    )

    assert len(empty_blocks) >= 2
    assert len(profiles) >= 1
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_diarization_supports_video_without_srt" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `process_video_subtitles`:
Nếu `not blocks`:
1. Đọc tệp âm thanh và chạy `segs = detect_speech_segments_vad(data, sr, min_speech_sec=0.4)`.
2. Nếu `segs`:
   Tạo các khối `SubtitleBlock(index=i, start_time=format_time(s), end_time=format_time(e), text=f"Đoạn thoại {i}")` và thêm vào `blocks`.
3. Tiếp tục quy trình phân tích và phân cụm bình thường.

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest -o testpaths=test/test_omnivoice_pipeline.py -k "test_diarization_supports_video_without_srt" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/ai_engines/speaker_diarization.py src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py; git commit -m "feat(diarization): support standalone video speaker detection without prior SRT file"`

---

### Nhiệm Vụ 5: Kiểm Thử Toàn Diện Và Rà Soát Hồi Quy

- [ ] **Bước 1: Chạy toàn bộ bộ kiểm thử tự động**
`.\venv\Scripts\pytest test/test_omnivoice_pipeline.py test/test_auto_pipeline_reliability.py -q`
Kỳ vọng: 100% passed (tối thiểu 110 bài kiểm thử).
- [ ] **Bước 2: Rà soát ngôn ngữ chuẩn mực**
Đảm bảo 100% tiếng Việt, không từ tiếng Anh trong ngoặc đơn, không emoji.
- [ ] **Bước 3: Đẩy commit lên kho lưu trữ GitHub**
`git push origin main`
