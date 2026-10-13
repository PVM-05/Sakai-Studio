# Kế Hoạch Thực Thi: Hệ Thống Tự Động Phân Vai Và Chọn Giọng

> **Dành cho kỹ sư đại lý:** KỸ NĂNG BẮT BUỘC: Sử dụng subagent-driven-development (khuyến nghị) hoặc executing-plans để thực thi kế hoạch này theo từng tác vụ. Các bước sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Xây dựng tính năng Tự động phân vai và chọn giọng (Auto Voice Casting) cho Sakai Studio, tự động nhận diện người nói từ video bằng ONNX Runtime tăng tốc NVIDIA GeForce RTX 4050 Laptop GPU với cơ chế tuần tự hóa bộ nhớ đồ họa 6 GB, đối chiếu mốc thời gian phụ đề với 3 mức độ tin cậy và tái quy hoạch giao diện trang Lồng Tiếng theo quy trình 5 bước mạch lạc.

**Kiến trúc:** Tầng nhận diện tách riêng vào `speaker_embedding_onnx.py` (trích xuất vector đặc trưng 192 chiều qua CAM++ ONNX) và `speaker_diarization.py` (điều phối phân cụm cosin, tính toán 3 mức độ tin cậy, giao diện kết nối Pyannote). Mô hình lồng tiếng `tts_engine.py` nhận hồ sơ chuẩn hóa `SpeakerVoiceProfile` và chạy tuần tự sau khi giải phóng phiên làm việc nhận diện. Giao diện `omnivoice_interface.py` được cấu trúc lại thành ngăn xếp 5 bước bảo toàn trạng thái dữ liệu.

**Ngăn xếp công nghệ:** Python 3.12, PySide6, QFluentWidgets, ONNX Runtime (CUDA / CPU), NumPy, SciPy, SoundFile, FFmpeg.

## Ràng Buộc Toàn Cục

- 100% tiếng Việt chuẩn mực trên giao diện, thông báo, ghi chú kiểm thử và tài liệu.
- Tuyệt đối không sử dụng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Giới hạn cấp phát bộ nhớ đồ họa ONNX Runtime ở mức 768 MiB trên NVIDIA GeForce RTX 4050 Laptop GPU.
- Luôn giải phóng phiên làm việc ONNX trước khi nạp mô hình tổng hợp giọng nói OmniVoice lên GPU.
- Không tự ý gán ép các câu thoại thiếu dữ liệu hoặc dưới 0.3 giây vào cụm trước; phải đánh dấu Chưa xác định.
- Cho phép người dùng chuyển qua lại giữa 5 bước giao diện mà không làm mất trạng thái dữ liệu hay kết quả đã gán.

---

### Tác Vụ 1: Mô Hình Dữ Liệu Và Bộ Trích Xuất Vector Đặc Trưng ONNX

**Các tệp tin:**
- Tạo mới: `src/ai_engines/speaker_embedding_onnx.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp:
  - `create_speaker_session(model_path: str, device_id: int = 0, gpu_mem_limit_mb: int = 768, prefer_cuda: bool = True) -> Tuple[Any, str]`
  - `extract_speaker_embedding(session: Any, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray`
  - `batch_extract_speaker_embeddings(session: Any, audio_chunks: List[np.ndarray], sample_rate: int = 16000, batch_size: int = 8) -> np.ndarray`
  - `close_speaker_session(session: Any) -> None`

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho bộ trích xuất vector đặc trưng ONNX**

```python
# Thêm vào test/test_omnivoice_pipeline.py
def test_onnx_speaker_embedding_session_and_extraction():
    from src.ai_engines.speaker_embedding_onnx import (
        create_speaker_session,
        extract_speaker_embedding,
        batch_extract_speaker_embeddings,
        close_speaker_session,
    )
    import numpy as np

    # Giả lập phiên làm việc ONNX với đầu ra 192 chiều
    mock_audio = np.random.randn(16000 * 2).astype(np.float32)
    session, provider = create_speaker_session(model_path="", prefer_cuda=False)
    assert provider in ("CPUExecutionProvider", "MockExecutionProvider")

    emb = extract_speaker_embedding(session, mock_audio, sample_rate=16000)
    assert emb.shape == (192,)
    # Kiểm tra chuẩn L2 xấp xỉ 1.0
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    chunks = [mock_audio, mock_audio]
    batch_embs = batch_extract_speaker_embeddings(session, chunks, sample_rate=16000, batch_size=2)
    assert batch_embs.shape == (2, 192)

    close_speaker_session(session)
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_onnx_speaker_embedding_session_and_extraction -v`
Kết quả kỳ vọng: THẤT BẠI do chưa tồn tại module `src.ai_engines.speaker_embedding_onnx`.

- [ ] **Bước 3: Triển khai mã nguồn tối thiểu cho `src/ai_engines/speaker_embedding_onnx.py`**

Triển khai cấu trúc module `speaker_embedding_onnx.py` hỗ trợ nạp mô hình CAM++, thiết lập arena memory 768 MiB, kiểm tra tính khả dụng của CUDAExecutionProvider, tiền xử lý âm thanh chuẩn 16 kHz và chuẩn hóa L2 vector 192 chiều.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_onnx_speaker_embedding_session_and_extraction -v`
Kết quả kỳ vọng: ĐẠT (PASS).

- [ ] **Bước 5: Lưu bản ghi thay đổi**

```bash
git add src/ai_engines/speaker_embedding_onnx.py test/test_omnivoice_pipeline.py
git commit -m "feat(voice-casting): add onnx speaker embedding module"
```

---

### Tác Vụ 2: Bộ Điều Phối Phân Cụm Cosin Và Xếp Hạng 3 Mức Độ Tin Cậy

**Các tệp tin:**
- Tạo mới: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Tiếp nhận: `extract_speaker_embedding` và `batch_extract_speaker_embeddings` từ Tác vụ 1.
- Cung cấp:
  - `class ConfidenceLevel(Enum)`: `HIGH = "high"`, `REVIEW_NEEDED = "review_needed"`, `UNASSIGNED = "unassigned"`
  - `class SpeakerVoiceProfile`: chứa `speaker_id`, `display_name`, `source_audio_sample`, `assigned_strategy`, `voice_preset_key`, `instruct`, `dialogue_count`, `high_confidence_ratio`
  - `cosine_cluster_embeddings(embeddings: np.ndarray, valid_indices: List[int], max_speakers: int = 6) -> Tuple[np.ndarray, List[str], List[float]]`
  - `SpeakerDiarizationCoordinator.process_video_subtitles(video_path: str, blocks: List[SubtitleBlock], max_speakers: int = 4, prefer_cuda: bool = True) -> Dict[str, SpeakerVoiceProfile]`

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho phân cụm cosin và 3 mức độ tin cậy**

```python
# Thêm vào test/test_omnivoice_pipeline.py
def test_cosine_clustering_and_three_confidence_levels():
    from src.ai_engines.speaker_diarization import (
        SpeakerDiarizationCoordinator,
        SpeakerVoiceProfile,
        ConfidenceLevel,
        cosine_cluster_embeddings,
    )
    import numpy as np

    # Tạo dữ liệu giả lập gồm 2 cụm vector đặc trưng tách biệt rõ ràng
    spk1_base = np.zeros(192, dtype=np.float32)
    spk1_base[0] = 1.0
    spk2_base = np.zeros(192, dtype=np.float32)
    spk2_base[1] = 1.0

    cluster_1 = spk1_base + np.random.normal(0, 0.02, (5, 192)).astype(np.float32)
    cluster_2 = spk2_base + np.random.normal(0, 0.02, (5, 192)).astype(np.float32)
    ambiguous = (spk1_base + spk2_base) / np.sqrt(2.0)
    ambiguous_noisy = ambiguous + np.random.normal(0, 0.05, (1, 192)).astype(np.float32)

    all_embs = np.vstack([cluster_1, cluster_2, ambiguous_noisy])
    # Chuẩn hóa chuẩn L2
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_embs = all_embs / norms

    labels, conf_levels, conf_scores = cosine_cluster_embeddings(all_embs, valid_indices=list(range(11)))
    assert len(labels) == 11
    # Cụm 1 và Cụm 2 phải có nhãn khác nhau
    assert labels[0] != labels[5]
    # Các câu rõ ràng phải đạt tin cậy cao
    assert conf_levels[0] == ConfidenceLevel.HIGH.value
    assert conf_levels[5] == ConfidenceLevel.HIGH.value
    # Câu lưỡng lự phải rơi vào mức cần kiểm tra
    assert conf_levels[10] in (ConfidenceLevel.REVIEW_NEEDED.value, ConfidenceLevel.UNASSIGNED.value)
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_cosine_clustering_and_three_confidence_levels -v`
Kết quả kỳ vọng: THẤT BẠI do chưa tồn tại module `src.ai_engines.speaker_diarization`.

- [ ] **Bước 3: Triển khai mã nguồn cho `src/ai_engines/speaker_diarization.py`**

Triển khai cấu trúc lớp `SpeakerVoiceProfile`, thuật toán tính khoảng cách cosin, phân cụm phân cấp tự động xác định số người nói và phân loại 3 mức độ tin cậy dựa trên khoảng cách tới tâm cụm và chênh lệch tâm cụm lân cận.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_cosine_clustering_and_three_confidence_levels -v`
Kết quả kỳ vọng: ĐẠT (PASS).

- [ ] **Bước 5: Lưu bản ghi thay đổi**

```bash
git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py
git commit -m "feat(voice-casting): add speaker diarization and confidence coordinator"
```

---

### Tác Vụ 3: Tích Hợp Tuần Tự Hóa Bộ Nhớ Đồ Họa Và Cập Nhật TTSEngine

**Các tệp tin:**
- Sửa đổi: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Tiếp nhận: `SpeakerDiarizationCoordinator` từ Tác vụ 2.
- Sửa đổi:
  - `TTSEngine.detect_speakers_from_video(...)`: Gọi điều phối nhận diện bằng ONNX GPU/CPU tuần tự, lưu trữ `SpeakerVoiceProfile`, gán `speaker` và `confidence_level` cho từng `SubtitleBlock`.
  - Bảo vệ cờ `is_speaker_locked`: Không ghi đè các câu thoại đã được người dùng khóa thủ công.
  - Xóa bỏ logic phân cụm F0/trọng tâm phổ cũ trong `_cluster_speaker_features`.
  - Đảm bảo giải phóng hoàn toàn phiên làm việc ONNX trước khi gọi `synthesize_blocks`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho bảo toàn khóa câu thoại và thu hồi bộ nhớ đồ họa trước khi lồng tiếng**

```python
# Thêm vào test/test_omnivoice_pipeline.py
def test_tts_engine_respects_locked_speaker_and_releases_diarizer_session():
    from src.ai_engines.tts_engine import TTSEngine, SubtitleBlock

    engine = TTSEngine()
    b1 = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:03,000", text="Xin chào")
    b2 = SubtitleBlock(index=2, start_time="00:00:04,000", end_time="00:00:06,000", text="Chào bạn")
    b2.speaker = "Nhân vật đã khóa"
    b2.is_speaker_locked = True

    # Giả lập quá trình nhận diện
    mock_profiles = {
        "Người nói 1": {"speaker_id": "spk_0", "count": 1}
    }
    # Sau khi chạy detect_speakers_from_video, câu 2 vẫn giữ nguyên "Nhân vật đã khóa"
    # và phiên làm việc nhận diện phải là None
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_tts_engine_respects_locked_speaker_and_releases_diarizer_session -v`
Kết quả kỳ vọng: THẤT BẠI.

- [ ] **Bước 3: Cập nhật `src/ai_engines/tts_engine.py`**

Thay thế `_cluster_speaker_features` và cập nhật `detect_speakers_from_video` để kết nối trực tiếp với `SpeakerDiarizationCoordinator`, áp dụng cờ `is_speaker_locked`, đồng thời giải phóng phiên làm việc và gọi dọn dẹp bộ nhớ trước khi bước vào tổng hợp giọng đọc.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_tts_engine_respects_locked_speaker_and_releases_diarizer_session -v`
Kết quả kỳ vọng: ĐẠT (PASS).

- [ ] **Bước 5: Lưu bản ghi thay đổi**

```bash
git add src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py
git commit -m "refactor(tts-engine): wire speaker diarization coordinator and enforce vram sequencing"
```

---

### Tác Vụ 4: Tái Quy Hoạch Giao Diện Trang Lồng Tiếng Thành 5 Bước Trực Quan

**Các tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Tái cấu trúc lớp `OmniVoiceInterface`:
  - Khởi tạo thanh điều hướng 5 bước (Bước 1: Nguồn đầu vào, Bước 2: Quản lý nhân vật, Bước 3: Biên tập phụ đề, Bước 4: Cân chỉnh âm thanh, Bước 5: Kiểm tra và xuất).
  - Tích hợp `QStackedWidget` quản lý 5 trang giao diện tương ứng.
  - Thêm thẻ cài đặt phần cứng trong Cài đặt nâng cao: Ưu tiên NVIDIA RTX 4050 Laptop GPU, Chế độ vi xử lý trung tâm, Phân đoạn chuyên sâu.
  - Bảng Quản lý nhân vật ở Bước 2 với đầy đủ tính năng: đổi tên, nghe thử giọng nguồn, chọn preset giọng, nghe thử giọng tổng hợp, xem mức độ tin cậy.
  - Cơ chế bảo toàn trạng thái dữ liệu khi điều hướng lùi/tiến giữa 5 bước.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho quy trình giao diện 5 bước và bảo toàn dữ liệu**

```python
# Thêm vào test/test_omnivoice_pipeline.py
def test_ui_wizard_preserves_state_on_step_navigation():
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    ui = OmniVoiceInterface()
    assert ui.step_stack.count() == 5
    assert ui.current_step_index == 0

    # Chuyển sang bước 2 và bước 3
    ui._switch_to_step(1)
    assert ui.current_step_index == 1
    ui._switch_to_step(2)
    assert ui.current_step_index == 2

    # Giả lập nạp phụ đề ở bước 3 rồi quay lại bước 1
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Thử nghiệm", speaker="Người nói 1")
    ]
    ui._switch_to_step(0)
    assert len(ui.subtitle_blocks) == 1
    assert ui.subtitle_blocks[0].speaker == "Người nói 1"
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_ui_wizard_preserves_state_on_step_navigation -v`
Kết quả kỳ vọng: THẤT BẠI do chưa có `step_stack`.

- [ ] **Bước 3: Tái cấu trúc giao diện `src/desktop/ui/omnivoice_interface.py`**

Tái cấu trúc bố cục theo `QStackedWidget` 5 bước, tích hợp bảng nhân vật trực quan, bộ chọn phần cứng GPU/CPU và đảm bảo tuân thủ nghiêm ngặt quy tắc thiết kế giao diện (tiếng Việt chuẩn, không có từ tiếng Anh trong ngoặc đơn, không có biểu tượng cảm xúc, tự động co giãn thẻ điều khiển).

- [ ] **Bước 4: Chạy kiểm thử để xác nhận vượt qua**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k test_ui_wizard_preserves_state_on_step_navigation -v`
Kết quả kỳ vọng: ĐẠT (PASS).

- [ ] **Bước 5: Lưu bản ghi thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(ui): implement 5-step wizard and character voice casting table"
```

---

### Tác Vụ 5: Kiểm Tra Chất Lượng Trước Xuất Và Xác Minh Toàn Diện

**Các tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Sửa đổi: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp:
  - Báo cáo kiểm tra chất lượng trước xuất: cảnh báo các câu chưa có nhân vật (`unassigned`), cảnh báo câu thoại quá dài có nguy cơ tràn mốc thời gian phụ đề.
  - Xác minh toàn bộ 51+ bài kiểm thử đơn vị và kiểm thử tích hợp trên toàn hệ thống.

- [ ] **Bước 1: Viết bài kiểm thử cho kiểm tra chất lượng trước khi xuất video**

```python
# Thêm vào test/test_omnivoice_pipeline.py
def test_pre_export_quality_checks_and_warnings():
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.tts_engine import SubtitleBlock

    ui = OmniVoiceInterface()
    ui.subtitle_blocks = [
        SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Đoạn 1", speaker="Người nói 1"),
        SubtitleBlock(index=2, start_time="00:00:02,000", end_time="00:00:03,000", text="Đoạn 2", speaker="Chưa xác định"),
    ]
    issues = ui._run_pre_export_quality_check()
    assert len(issues) >= 1
    assert any("Chưa xác định" in item["message"] or "chưa gán" in item["message"] for item in issues)
```

- [ ] **Bước 2: Triển khai hàm kiểm tra chất lượng trước khi xuất trong `omnivoice_interface.py`**

Bổ sung phương thức `_run_pre_export_quality_check()`, hiển thị hộp thoại cảnh báo cho người dùng xác nhận nếu phát hiện các câu thoại chưa gán nhân vật hoặc mốc thời gian bất thường.

- [ ] **Bước 3: Chạy toàn bộ bộ kiểm thử tự động của hệ thống**

Chạy lệnh: `.\venv\Scripts\python.exe -m pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py test/test_srt_translator.py test/test_subtitle_and_logo_remover.py test_home_interface.py -v`
Kết quả kỳ vọng: 100% bài kiểm thử đạt kết quả tốt (PASS).

- [ ] **Bước 4: Lưu bản ghi thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(qc): add pre-export voice casting quality checks and comprehensive test suite"
```
