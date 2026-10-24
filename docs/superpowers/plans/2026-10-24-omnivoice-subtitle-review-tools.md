# Kế Hoạch Triển Khai: Cải Thiện Công Cụ Rà Soát Phụ Đề Và Đồng Bộ Thời Lượng OmniVoice

> **Dành cho kỹ sư và tác nhân thực thi:** BẮT BUỘC tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD). Mỗi nhiệm vụ đều có các bước nhỏ được đánh dấu checkbox (`- [ ]`). Cam kết git sau mỗi nhiệm vụ hoàn thành.

**Mục tiêu:** Cải thiện công cụ rà soát phụ đề trong OmniVoice bằng cách kiểm tra chênh lệch thời lượng Video-SRT, bổ sung bộ lọc "Đã khóa", nâng cấp cột Người nói thành danh sách chọn trực tiếp kèm tự động khóa, và thêm cột nút nghe âm thanh gốc từng dòng.

**Kiến trúc:** Bổ sung hàm đo kiểm thời lượng phương tiện độc lập `check_media_and_srt_duration`; mở rộng bộ lọc `combo_sub_filter` và logic ẩn/hiện hàng; mở rộng bảng phụ đề `sub_table` lên 7 cột với ComboBox chọn nhân vật và nút nghe âm thanh gốc cắt đoạn tạm thời qua FFmpeg/Pydub.

**Ngăn xếp kỹ thuật:** Python 3.10+, PySide6, QFluentWidgets (FluentIcon), OpenCV (cv2), PyDub / FFmpeg, Pytest.

## Ràng Buộc Bắt Buộc Toàn Cục

- 100% tiếng Việt chuẩn mực trong toàn bộ giao diện, thông báo, nhãn và tài liệu.
- Tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc (emoji) hoặc hình khối trang trí trong giao diện và tiêu đề.
- Sử dụng biểu tượng từ thư viện FluentIcon.
- Tuân thủ nghiêm ngặt quy trình TDD: viết bài kiểm thử thất bại trước, viết mã nguồn tối thiểu sau.
- Thực thi trực tiếp tuần tự (Inline execution), không tạo subagent để tránh cạn kiệt tài nguyên.

---

### Nhiệm Vụ 1: Kiểm Tra Chênh Lệch Thời Lượng Video Và Phụ Đề Tại Bước 1

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_auto_pipeline_reliability.py` hoặc `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cung cấp: `check_media_and_srt_duration(video_path: str, subtitle_blocks: List[Any], threshold_sec: float = 5.0) -> Dict[str, Any]`
- Cập nhật: `_update_identify_button_state(self)` hiển thị cảnh báo khi `is_mismatch` là `True`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_check_media_and_srt_duration_detects_discrepancy(tmp_path):
    """
    Xác minh hàm check_media_and_srt_duration:
    1. Phát hiện chênh lệch khi thời lượng video và phụ đề lệch vượt ngưỡng 5 giây.
    2. Trả về thông tin đầy đủ: video_duration, srt_duration, diff, is_mismatch.
    3. Cảnh báo khi phụ đề dài hơn video.
    """
    import cv2
    import numpy as np
    from src.desktop.ui.omnivoice_interface import check_media_and_srt_duration
    from src.ai_engines.translator import SubtitleBlock

    # Tạo video ngắn 2 giây (50 frames, 25 fps)
    video_file = str(tmp_path / "short_video.mp4")
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (100, 100))
    for _ in range(50):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()

    # Phụ đề dài 15 giây -> Lệch 13 giây > ngưỡng 5 giây
    blocks = [
        SubtitleBlock(1, "00:00:01,000", "00:00:05,000", "Câu một"),
        SubtitleBlock(2, "00:00:06,000", "00:00:15,000", "Câu hai"),
    ]

    res = check_media_and_srt_duration(video_file, blocks, threshold_sec=5.0)
    assert res["is_mismatch"] is True
    assert round(res["video_duration"], 1) == 2.0
    assert round(res["srt_duration"], 1) == 15.0
    assert res["diff"] >= 12.9
    assert "chênh lệch" in res["message"].lower() or "cảnh báo" in res["message"].lower()

    # Trường hợp đồng bộ (lệch dưới 1 giây)
    synced_blocks = [
        SubtitleBlock(1, "00:00:00,500", "00:00:02,000", "Câu khớp"),
    ]
    res_sync = check_media_and_srt_duration(video_file, synced_blocks, threshold_sec=5.0)
    assert res_sync["is_mismatch"] is False
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_check_media_and_srt_duration_detects_discrepancy" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Thêm hàm `check_media_and_srt_duration` vào `src/desktop/ui/omnivoice_interface.py`:
```python
def check_media_and_srt_duration(
    video_path: str,
    subtitle_blocks: List[Any],
    threshold_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Kiểm tra tính đồng bộ thời lượng giữa video nguồn và tệp phụ đề.
    Trả về thông số thời lượng và cờ cảnh báo nếu chênh lệch đáng kể.
    """
    vid_dur = 0.0
    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            vid_dur = total_frames / fps if fps > 0 else 0.0
            cap.release()

    sub_dur = 0.0
    if subtitle_blocks:
        max_end = 0.0
        for blk in subtitle_blocks:
            end_val = getattr(blk, "end_time", "")
            if end_val:
                try:
                    s = parse_timestamp_to_seconds(str(end_val))
                    if s > max_end:
                        max_end = s
                except Exception:
                    pass
        sub_dur = max_end

    diff = abs(vid_dur - sub_dur)
    is_mismatch = False
    msg = ""

    if vid_dur > 0 and sub_dur > 0:
        if sub_dur > vid_dur + 1.0 or diff > threshold_sec:
            is_mismatch = True
            msg = (
                f"Cảnh báo chênh lệch thời lượng: Video dài {vid_dur:.1f} giây, "
                f"phụ đề dài {sub_dur:.1f} giây (chênh lệch {diff:.1f} giây). "
                f"Vui lòng kiểm tra lại để đảm bảo phụ đề khớp âm thanh video."
            )
        else:
            msg = f"Thời lượng video ({vid_dur:.1f}s) và phụ đề ({sub_dur:.1f}s) đồng bộ tốt."

    return {
        "video_duration": vid_dur,
        "srt_duration": sub_dur,
        "diff": diff,
        "is_mismatch": is_mismatch,
        "message": msg,
    }
```
Và trong `_update_identify_button_state(self)`: gọi `check_media_and_srt_duration` khi có cả video lẫn phụ đề. Nếu `res["is_mismatch"]`, cập nhật nhãn `self.lbl_detect_source_hint.setText(res["message"])`.

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_check_media_and_srt_duration_detects_discrepancy" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py; git commit -m "feat(omnivoice): add video and subtitle duration discrepancy verification"`

---

### Nhiệm Vụ 2: Bổ Sung Tùy Chọn Lọc "Đã Khóa" Vào Bảng Phụ Đề

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_filter_subtitle_table_supports_locked_category():
    """
    Xác minh bộ lọc phụ đề hỗ trợ tùy chọn 'Đã khóa':
    1. combo_sub_filter chứa mục 'Đã khóa'.
    2. Khi chọn 'Đã khóa', chỉ hiện câu có is_speaker_locked = True, ẩn câu is_speaker_locked = False.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
    blk1.is_speaker_locked = True

    blk2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
    blk2.is_speaker_locked = False

    ui.subtitle_blocks = [blk1, blk2]
    ui._populate_subtitle_table()

    # Kiểm tra combo_sub_filter có mục 'Đã khóa'
    items = [ui.combo_sub_filter.itemText(i) for i in range(ui.combo_sub_filter.count())]
    assert "Đã khóa" in items

    # Lọc theo 'Đã khóa'
    ui.combo_sub_filter.setCurrentText("Đã khóa")
    assert not ui.sub_table.isRowHidden(0)
    assert ui.sub_table.isRowHidden(1)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_filter_subtitle_table_supports_locked_category" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/desktop/ui/omnivoice_interface.py`:
1. Tại khởi tạo `self.combo_sub_filter` (khoảng dòng 1053): thêm `"Đã khóa"` vào danh sách mục (`addItems`).
2. Trong `_filter_subtitle_table`:
```python
            elif selected == "Đã khóa":
                is_locked = getattr(blk, "is_speaker_locked", False)
                self.sub_table.setRowHidden(row, not is_locked)
```

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_filter_subtitle_table_supports_locked_category" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py; git commit -m "feat(omnivoice): add locked category filter to subtitle table"`

---

### Nhiệm Vụ 3: Nâng Cấp Cột Người Nói Thành ComboBox Trực Tiếp Kèm Tự Động Khóa

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_subtitle_table_speaker_combobox_and_auto_locking():
    """
    Xác minh cột Người nói trên bảng phụ đề:
    1. Sử dụng ComboBox thay vì chuỗi văn bản tĩnh.
    2. Chứa danh sách nhân vật đã nhận diện.
    3. Khi đổi người nói qua ComboBox:
       - blk.speaker được cập nhật.
       - blk.is_speaker_locked tự động chuyển thành True.
       - blk.confidence_level chuyển thành 'high'.
       - Nguồn giọng ở cột 4 tự động làm mới.
    """
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import ComboBox
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ui.speaker_profiles = {
        "Thám tử Lừng Danh": CharacterVoiceProfile(speaker_id="spk_0", display_name="Thám tử Lừng Danh"),
        "Kẻ Giấu Mặt": CharacterVoiceProfile(speaker_id="spk_1", display_name="Kẻ Giấu Mặt"),
    }

    blk = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào")
    blk.speaker = "Chưa xác định"
    blk.is_speaker_locked = False
    blk.confidence_level = "unassigned"

    ui.subtitle_blocks = [blk]
    ui._populate_subtitle_table()

    combo_spk = ui.sub_table.cellWidget(0, 3)
    assert isinstance(combo_spk, ComboBox)
    assert combo_spk.findText("Thám tử Lừng Danh") >= 0

    # Người dùng chọn nhân vật
    combo_spk.setCurrentText("Thám tử Lừng Danh")

    assert blk.speaker == "Thám tử Lừng Danh"
    assert blk.is_speaker_locked is True
    assert blk.confidence_level == "high"
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_subtitle_table_speaker_combobox_and_auto_locking" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `_populate_subtitle_table`:
- Thay vì `self.sub_table.setItem(row, 3, QTableWidgetItem(spk))`:
  - Tạo `combo_row_spk = ComboBox(self.sub_table)`.
  - Nạp danh sách các nhân vật từ `self.speaker_profiles` kèm `"Chưa xác định"`.
  - Đặt giá trị hiện tại theo `blk.speaker`.
  - Kết nối tín hiệu `combo_row_spk.currentTextChanged.connect(lambda txt, r=row: self._on_row_speaker_changed(r, txt))`.
  - Gán vào ô: `self.sub_table.setCellWidget(row, 3, combo_row_spk)`.
- Viết phương thức `_on_row_speaker_changed(self, row: int, new_spk: str)`:
  - Cập nhật `blk.speaker = new_spk`.
  - Nếu `new_spk` khác `"Chưa xác định"`:
    - Đặt `blk.is_speaker_locked = True`.
    - Đặt `blk.confidence_level = "high"`.
    - Đồng bộ `voice_ref_path` và `voice_instruct` từ hồ sơ nhân vật `self.speaker_profiles.get(new_spk)`.
  - Cập nhật hiển thị ô cột 4 (Nguồn giọng): `self.sub_table.setItem(row, 4, QTableWidgetItem(self._format_voice_source_display(blk)))`.
  - Gọi `self._update_casting_table()` và `self._update_character_profiles_overview()`.

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_subtitle_table_speaker_combobox_and_auto_locking" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py; git commit -m "feat(omnivoice): enhance subtitle table speaker column with interactive combobox and auto-lock"`

---

### Nhiệm Vụ 4: Thêm Cột Nghe Gốc Và Trích Xuất Phát Âm Thanh Mốc Thời Gian

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_omnivoice_pipeline.py`:
```python
def test_subtitle_table_has_7_columns_with_listen_button():
    """
    Xác minh bảng phụ đề có 7 cột:
    0: STT, 1: Bắt đầu, 2: Kết thúc, 3: Người nói, 4: Nguồn giọng, 5: Nghe gốc, 6: Nội dung thoại.
    Cột 5 có nút nghe với FluentIcon.
    """
    from PySide6.QtWidgets import QApplication, QPushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk = SubtitleBlock(1, "00:00:01,000", "00:00:03,000", "Xin chào")
    ui.subtitle_blocks = [blk]
    ui._populate_subtitle_table()

    assert ui.sub_table.columnCount() == 7
    header_texts = [ui.sub_table.horizontalHeaderItem(c).text() for c in range(7)]
    assert "Nghe Gốc" in header_texts[5] or "Nghe gốc" in header_texts[5]

    btn_listen = ui.sub_table.cellWidget(0, 5)
    assert isinstance(btn_listen, QPushButton)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_subtitle_table_has_7_columns_with_listen_button" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/desktop/ui/omnivoice_interface.py`:
1. Tại khởi tạo `sub_table` (khoảng dòng 1079):
   - Đổi `setColumnCount(7)`.
   - Cập nhật tiêu đề cột:
     ```python
     self.sub_table.setHorizontalHeaderLabels([
         "STT",
         "Bắt Đầu",
         "Kết Thúc",
         "Người Nói",
         "Nguồn Giọng",
         "Nghe Gốc",
         "Nội Dung Thoại",
     ])
     ```
2. Trong `_populate_subtitle_table`:
   - Cột 5: Tạo `btn_row_listen = PushButton("Nghe", self.sub_table)` (hoặc icon `FluentIcon.PLAY`).
   - Kết nối: `btn_row_listen.clicked.connect(lambda chk=False, b=blk, btn=btn_row_listen: self._play_subtitle_original_segment(b, btn))`.
   - Nếu `not is_valid_media_file(self.current_video_path)`: `btn_row_listen.setEnabled(False)`.
   - Cột 6: Đặt `QTableWidgetItem(txt)` cho nội dung thoại.
3. Cập nhật `_on_subtitle_table_edited`: Cột nội dung thoại chuyển sang kiểm tra `col == 6`.
4. Viết phương thức `_play_subtitle_original_segment(self, blk: SubtitleBlock, btn: Optional[PushButton] = None)`:
   - Nếu đang phát, bấm lại thì dừng và phục hồi biểu tượng.
   - Trích xuất đoạn âm thanh ngắn từ `self.current_video_path` theo `blk.start_time` và `blk.end_time` lưu vào tệp tạm bằng subprocess FFmpeg hoặc audio separator.
   - Phát tệp tạm thông qua `self._play_audio_file(...)`.
   - Cập nhật biểu tượng nút thành `FluentIcon.PAUSE` trong lúc phát.

- [ ] **Bước 4: Chạy lại kiểm thử xác nhận thành công**
Chạy: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_subtitle_table_has_7_columns_with_listen_button" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py; git commit -m "feat(omnivoice): add original audio segment preview button to subtitle table"`

---

### Nhiệm Vụ 5: Kiểm Thử Toàn Diện Và Rà Soát Hồi Quy

- [ ] **Bước 1: Chạy toàn bộ bộ kiểm thử**
`.\venv\Scripts\pytest test/test_omnivoice_pipeline.py test/test_auto_pipeline_reliability.py -q`
Kỳ vọng: 100% passed (tối thiểu 106+ bài kiểm thử).
- [ ] **Bước 2: Rà soát ngôn ngữ và quy chuẩn thiết kế**
Kiểm tra không có từ tiếng Anh trong ngoặc đơn, không có emoji, 100% tiếng Việt chuẩn mực.
- [ ] **Bước 3: Đẩy commit lên GitHub**
`git push origin main`
