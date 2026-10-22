# Kế hoạch Triển khai: Hoàn thiện Luồng Nhận diện và Gợi ý Giọng Tiếng Việt Trong OmniVoice

> **Dành cho tác nhân thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để triển khai từng nhiệm vụ bên dưới. Các bước sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Cải tiến luồng người dùng OmniVoice theo 3 mũi nhọn: (1) Đưa nút nhận diện nhân vật ra ngay Bước 1 ngay sau khi nạp tệp; (2) Hiển thị kết quả nhận diện trực quan và tách biệt rõ hai nút nghe giọng gốc / nghe thử giọng trí tuệ nhân tạo trong bảng phân vai; (3) Kết nối `VoiceRecommendationEngine` vào giao diện để hiển thị Top 3 gợi ý giọng Việt Nam cho từng nhân vật.

**Kiến trúc:** Chỉnh sửa `src/desktop/ui/omnivoice_interface.py` theo bốn nhóm thay đổi độc lập: thêm nút thao tác chính ở Bước 1 (Nguồn Đầu Vào), làm rõ nhãn và chức năng của hai nút nghe thử trong bảng phân vai ở Bước 2, hiển thị danh sách Top 3 gợi ý giọng Việt Nam trong thẻ hồ sơ nhân vật, và thêm nhãn tóm tắt kết quả nhận diện trực quan.

**Ngăn xếp Công nghệ:** Python 3.12, PySide6 6.x, QFluentWidgets, Pytest, `VoiceRecommendationEngine` (đã có tại `src/ai_engines/tts_engine.py`).

## Ràng buộc Toàn cục
- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện. Chỉ dùng biểu tượng Fluent.
- Tuân thủ quy trình phát triển hướng kiểm thử (TDD): viết bài kiểm thử thất bại trước, viết mã nguồn tối thiểu sau.
- Không được để lại các đoạn mã chờ như "TODO", "TBD" hoặc triển khai dở dang.

---

## Bản đồ Tệp tin

| Tệp | Vai trò |
|-----|---------|
| `src/desktop/ui/omnivoice_interface.py` | Tệp giao diện duy nhất cần chỉnh sửa |
| `test/test_auto_pipeline_reliability.py` | Kiểm thử hồi quy giao diện |
| `test/test_omnivoice_pipeline.py` | Kiểm thử đơn vị các tính năng OmniVoice |

---

### Nhiệm vụ 1: Nút Nhận Diện Nhân Vật Chính Ở Bước 1 Và Nhãn Theo Nguồn Thực Tế

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực `page_step_0`, dòng 557-582 và `_update_sub_video_status_label`, `_update_subtitle_action_buttons`, `_run_auto_detect_speakers`)
- Kiểm thử: `test/test_auto_pipeline_reliability.py`

**Giao diện cung cấp:**
- Widget mới: `self.btn_identify_characters` — nút thao tác chính "Nhận Diện Nhân Vật Từ Video + SRT".
- Widget mới: `self.lbl_detect_source_hint` — nhãn trạng thái nguồn nhận diện.
- Phương thức mới: `_update_identify_button_state(self)` — bật/tắt nút và cập nhật nhãn theo trạng thái tệp đã nạp.

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_auto_pipeline_reliability.py`**

```python
def test_step0_has_identify_button_enabled_only_when_srt_and_video_present(self):
    """
    Xác minh Bước 1 (Nguồn Đầu Vào) có nút 'Nhận Diện Nhân Vật Từ Video + SRT':
    1. Nút bị vô hiệu hóa khi chưa nạp cả SRT lẫn video.
    2. Nút bị vô hiệu hóa khi chỉ có SRT mà chưa có video.
    3. Nút được kích hoạt khi cả SRT hợp lệ và video đều đã được nạp.
    4. Nhãn trạng thái thay đổi theo nguồn thực tế.
    """
    import os
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Trạng thái ban đầu: chưa có tệp nào
    ui._update_identify_button_state()
    self.assertFalse(ui.btn_identify_characters.isEnabled())

    # Chỉ có SRT: nút phụ phải có, nút chính vẫn tắt
    ui.subtitle_blocks = [object()]  # Giả lập có câu phụ đề
    ui.current_video_path = ""
    ui._update_identify_button_state()
    self.assertFalse(ui.btn_identify_characters.isEnabled())
    hint_text = ui.lbl_detect_source_hint.text()
    self.assertIn("SRT", hint_text)

    # Có cả SRT lẫn video: nút chính bật
    ui.current_video_path = self.temp_dir  # Dùng thư mục có sẵn thay cho video thật
    ui._update_identify_button_state()
    self.assertTrue(ui.btn_identify_characters.isEnabled())
    hint_text = ui.lbl_detect_source_hint.text()
    self.assertIn("video", hint_text.lower())
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py -k "test_step0_has_identify_button_enabled_only_when_srt_and_video_present" -v`
Kỳ vọng: FAIL vì thiếu `btn_identify_characters`, `lbl_detect_source_hint`, `_update_identify_button_state`.

- [ ] **Bước 3: Viết mã nguồn tối thiểu vào `src/desktop/ui/omnivoice_interface.py`**

**3a. Thêm hai widget vào `page_step_0` ngay SAU dòng `sub_file_layout.addLayout(sub_file_btn_row)`** (khoảng dòng 581):

```python
        # Hàng nút nhận diện chính ở Bước 1
        detect_action_row = QHBoxLayout()
        detect_action_row.setSpacing(10)

        self.btn_identify_characters = PrimaryPushButton("Nhận Diện Nhân Vật Từ Video + SRT", sub_file_card)
        self.btn_identify_characters.setIcon(FluentIcon.PEOPLE)
        self.btn_identify_characters.setEnabled(False)
        self.btn_identify_characters.setMinimumHeight(36)

        self.lbl_detect_source_hint = CaptionLabel("Nạp tệp SRT và video để bắt đầu nhận diện nhân vật.", sub_file_card)
        self.lbl_detect_source_hint.setWordWrap(True)

        detect_action_row.addWidget(self.btn_identify_characters)
        detect_action_row.addWidget(self.lbl_detect_source_hint, 1)
        sub_file_layout.addLayout(detect_action_row)
```

**3b. Thêm phương thức `_update_identify_button_state(self)` vào lớp `OmniVoiceInterface`** (đặt gần `_update_sub_video_status_label`):

```python
    def _update_identify_button_state(self):
        """Cập nhật trạng thái nút nhận diện nhân vật và nhãn gợi ý theo nguồn đã nạp."""
        if not hasattr(self, "btn_identify_characters"):
            return
        has_subs = len(self.subtitle_blocks) > 0
        has_video = bool(self.current_video_path and os.path.exists(self.current_video_path))

        if has_subs and has_video:
            self.btn_identify_characters.setEnabled(True)
            self.lbl_detect_source_hint.setText(
                "Sẵn sàng nhận diện từ âm thanh video và mốc thời gian SRT."
            )
        elif has_subs and not has_video:
            self.btn_identify_characters.setEnabled(False)
            self.lbl_detect_source_hint.setText(
                "Chỉ có tệp SRT — hãy thêm video để nhận diện giọng thực, hoặc dùng 'Tự Động Phân Tích Lại' ở Bước 2 để phân vai theo văn bản."
            )
        else:
            self.btn_identify_characters.setEnabled(False)
            self.lbl_detect_source_hint.setText(
                "Nạp tệp SRT và video để bắt đầu nhận diện nhân vật."
            )
```

**3c. Kết nối tín hiệu** — thêm vào phần kết nối tín hiệu của `_connect_signals` (sau dòng `self.btn_load_video.clicked.connect`):

```python
        self.btn_identify_characters.clicked.connect(self._run_auto_detect_speakers)
```

**3d. Gọi `_update_identify_button_state()`** tại cuối `_load_subtitle_file_internal()`, cuối `_select_video_file()` và cuối `_clear_selected_video()`. Tìm hàm `_load_subtitle_file_internal`, sau dòng `self._update_sub_video_status_label()` thêm:
```python
            self._update_identify_button_state()
```
Tìm hàm `_select_video_file`, sau khi gán `self.current_video_path` thêm:
```python
        self._update_identify_button_state()
```
Tìm hàm `_clear_selected_video`, sau khi xóa `self.current_video_path` thêm:
```python
        self._update_identify_button_state()
```

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py -k "test_step0_has_identify_button_enabled_only_when_srt_and_video_present" -v`
Kỳ vọng: PASS

- [ ] **Bước 5: Chạy toàn bộ kiểm thử hồi quy**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py -q`
Kỳ vọng: 100% passed, 0 failed.

- [ ] **Bước 6: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_auto_pipeline_reliability.py
git commit -m "feat(omnivoice): add primary detect characters button to step 0 with source-aware label"
```

---

### Nhiệm vụ 2: Tách Biệt Nút Nghe Giọng Gốc Và Nghe Thử Giọng Trí Tuệ Nhân Tạo Trong Bảng Phân Vai

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực `_update_casting_table`, dòng ~1750-1765)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện cung cấp:**
- Bảng phân vai thay nút "Nghe Thử" thành hai nút riêng biệt trong cột 6:
  - `"Nghe Giọng Gốc"` — gọi `_preview_character_voice(spk_name)` (phát tệp mẫu âm thanh, chỉ hiển thị khi có `source_audio_path`).
  - `"Nghe Thử Giọng Trí Tuệ Nhân Tạo"` — gọi `_audition_character_voice(spk_name, btn)` (tổng hợp giọng mới).

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_casting_table_has_two_separate_listen_buttons():
    """
    Xác minh bảng phân vai có đúng 2 nút nghe riêng biệt cho mỗi nhân vật:
    - Nút 'Nghe Giọng Gốc' (phát mẫu âm thanh gốc).
    - Nút 'Nghe Thử Giọng Trí Tuệ Nhân Tạo' (tổng hợp câu thử).
    Không có nút chỉ mang tên 'Nghe Thử' không rõ nguồn gốc.
    """
    from PySide6.QtWidgets import QApplication, QPushButton
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    ui.speaker_profiles = {
        "Nhân vật A": CharacterVoiceProfile(
            speaker_id="spk_0",
            display_name="Nhân vật A",
            source_audio_path="/fake/audio.wav",
        )
    }
    ui._update_casting_table()

    cell_widget = ui.casting_table.cellWidget(0, 6)
    assert cell_widget is not None, "Ô cột 6 phải có widget chứa các nút nghe"

    buttons = cell_widget.findChildren(QPushButton)
    btn_texts = [b.text() for b in buttons]

    assert any("Giọng Gốc" in t for t in btn_texts), f"Thiếu nút Nghe Giọng Gốc, chỉ có: {btn_texts}"
    assert any("Trí Tuệ Nhân Tạo" in t for t in btn_texts), f"Thiếu nút Nghe Thử Giọng TTNNT, chỉ có: {btn_texts}"
    assert not any(t.strip() == "Nghe Thử" for t in btn_texts), f"Nút không rõ nguồn 'Nghe Thử' vẫn còn: {btn_texts}"
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_casting_table_has_two_separate_listen_buttons" -v`
Kỳ vọng: FAIL.

- [ ] **Bước 3: Sửa mã nguồn trong `_update_casting_table` tại `src/desktop/ui/omnivoice_interface.py`**

Tìm đoạn xây dựng cột 6 (tìm `btn_act.clicked.connect(lambda checked=False, s=spk_name: self._preview_character_voice(s))`). Thay toàn bộ khối đó bằng:

```python
            # Cột 6: Hai nút nghe tách biệt
            cell_listen = QWidget(self.casting_table)
            layout_listen = QHBoxLayout(cell_listen)
            layout_listen.setContentsMargins(2, 1, 2, 1)
            layout_listen.setSpacing(4)

            prof_item = self.speaker_profiles.get(spk_name)
            has_orig_audio = bool(
                getattr(prof_item, "source_audio_path", "")
                or (prof_item.get("source_audio_path", "") if isinstance(prof_item, dict) else "")
            )
            btn_orig = PushButton("Nghe Giọng Gốc", cell_listen)
            btn_orig.setIcon(FluentIcon.MUSIC)
            btn_orig.setToolTip("Phát đoạn âm thanh gốc đã trích xuất từ video cho nhân vật này")
            btn_orig.setEnabled(has_orig_audio)
            btn_orig.clicked.connect(lambda checked=False, s=spk_name: self._preview_character_voice(s))
            layout_listen.addWidget(btn_orig)

            btn_ai = PushButton("Nghe Thử Giọng Trí Tuệ Nhân Tạo", cell_listen)
            btn_ai.setIcon(FluentIcon.VOLUME)
            btn_ai.setToolTip("Tổng hợp câu thoại đầu tiên bằng giọng trí tuệ nhân tạo đang được gán cho nhân vật này")
            btn_ai.clicked.connect(lambda checked=False, s=spk_name, b=btn_ai: self._audition_character_voice(s, b))
            layout_listen.addWidget(btn_ai)

            self.casting_table.setCellWidget(row, 6, cell_listen)
```

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_casting_table_has_two_separate_listen_buttons" -v`
Kỳ vọng: PASS

- [ ] **Bước 5: Chạy toàn bộ kiểm thử hồi quy**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py -q`
Kỳ vọng: 100% passed, 0 failed.

- [ ] **Bước 6: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): split casting table listen button into source audio and AI synthesis"
```

---

### Nhiệm vụ 3: Kết Nối VoiceRecommendationEngine Vào Thẻ Hồ Sơ Nhân Vật — Hiển Thị Top 3 Gợi Ý Giọng Việt Nam

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực xây dựng thẻ hồ sơ nhân vật, khoảng dòng 1978-2157)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện tiêu thụ:**
- `VoiceRecommendationEngine.recommend_voices(profile, top_k=3)` từ `src.ai_engines.tts_engine`.
- Trả về `List[Dict]` với các khóa: `preset_name`, `score`, `reason`, `gender`, `accent`.

**Giao diện cung cấp:**
- Phương thức: `_build_voice_recommendation_widget(spk_name: str) -> QWidget`.
- Phương thức: `_apply_recommended_voice(spk_name: str, preset_name: str)`.

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_voice_recommendation_widget_shows_top3_for_character():
    """
    Xác minh _build_voice_recommendation_widget() tạo widget hiển thị tối đa 3 gợi ý giọng,
    mỗi gợi ý có điểm phù hợp và nút Áp Dụng.
    """
    from PySide6.QtWidgets import QApplication, QPushButton, QLabel
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Nhân vật thử nghiệm",
        gender="female",
        vietnamese_accent="bac",
        pitch_level="high",
    )
    ui.speaker_profiles = {"Nhân vật thử nghiệm": prof}

    widget = ui._build_voice_recommendation_widget("Nhân vật thử nghiệm")
    assert widget is not None

    apply_buttons = [b for b in widget.findChildren(QPushButton) if "Áp Dụng" in b.text()]
    assert 1 <= len(apply_buttons) <= 3, f"Số nút Áp Dụng phải từ 1 đến 3, hiện có: {len(apply_buttons)}"

    labels = widget.findChildren(QLabel)
    label_texts = [l.text() for l in labels]
    assert any("%" in t for t in label_texts), f"Phải có nhãn điểm (%), hiện có: {label_texts}"
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_voice_recommendation_widget_shows_top3_for_character" -v`
Kỳ vọng: FAIL vì thiếu `_build_voice_recommendation_widget`.

- [ ] **Bước 3: Viết mã nguồn vào `src/desktop/ui/omnivoice_interface.py`**

Đảm bảo import ở đầu file (bổ sung nếu chưa có):
```python
from src.ai_engines.tts_engine import VoiceRecommendationEngine
```

Thêm hai phương thức vào lớp `OmniVoiceInterface`:

```python
    def _build_voice_recommendation_widget(self, spk_name: str) -> QWidget:
        """Tạo widget hiển thị Top 3 gợi ý giọng Việt Nam cho nhân vật."""
        from PySide6.QtWidgets import QFrame
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        prof = self.speaker_profiles.get(spk_name)
        if prof is None:
            layout.addWidget(CaptionLabel("Chưa có hồ sơ nhân vật để đề xuất giọng.", container))
            return container

        try:
            recommendations = VoiceRecommendationEngine.recommend_voices(prof, top_k=3)
        except Exception:
            recommendations = []

        if not recommendations:
            layout.addWidget(CaptionLabel("Không tìm được giọng phù hợp trong danh mục.", container))
            return container

        layout.addWidget(StrongBodyLabel("Giọng đọc được gợi ý:", container))

        for idx, rec in enumerate(recommendations, start=1):
            rec_card = QFrame(container)
            rec_card.setFrameShape(QFrame.StyledPanel)
            rec_card_layout = QHBoxLayout(rec_card)
            rec_card_layout.setContentsMargins(8, 4, 8, 4)
            rec_card_layout.setSpacing(8)

            preset_label = rec.get("preset_name", "")
            score = rec.get("score", 0)
            reason = rec.get("reason", "")

            lbl_rank = BodyLabel(f"{idx}.", rec_card)
            lbl_rank.setFixedWidth(18)
            rec_card_layout.addWidget(lbl_rank)

            lbl_name = BodyLabel(preset_label, rec_card)
            lbl_name.setWordWrap(True)
            rec_card_layout.addWidget(lbl_name, 1)

            lbl_score = CaptionLabel(f"{score}%", rec_card)
            lbl_score.setToolTip(reason)
            rec_card_layout.addWidget(lbl_score)

            btn_apply = PushButton("Áp Dụng", rec_card)
            btn_apply.setIcon(FluentIcon.ACCEPT)
            btn_apply.clicked.connect(
                lambda checked=False, s=spk_name, p=preset_label: self._apply_recommended_voice(s, p)
            )
            rec_card_layout.addWidget(btn_apply)

            layout.addWidget(rec_card)

        return container

    def _apply_recommended_voice(self, spk_name: str, preset_name: str):
        """Áp dụng giọng được gợi ý cho nhân vật và làm mới thẻ hồ sơ."""
        from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG
        if preset_name not in VIETNAMESE_VOICE_CATALOG or spk_name not in self.speaker_profiles:
            return
        instruct_val = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
        prof = self.speaker_profiles[spk_name]
        prof["method"] = "preset"
        prof["preset_name"] = preset_name
        prof["voice_instruct"] = instruct_val
        prof["source_audio_path"] = ""
        prof["status"] = "locked"
        self._update_casting_table()
        InfoBar.success(
            title="Đã áp dụng giọng đọc",
            content=f"Nhân vật '{spk_name}' đã được gán giọng '{preset_name}'.",
            parent=self,
            duration=3000,
        )
```

Tìm vị trí xây dựng từng thẻ nhân vật trong `_build_character_cards` hoặc hàm xây dựng phần cấu hình nhân vật (khoảng dòng 1978-2157). Sau phần xây dựng các điều khiển chọn phương thức và mẫu giọng, thêm:

```python
            rec_widget = self._build_voice_recommendation_widget(spk_name)
            card_layout.addWidget(rec_widget)
```

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_voice_recommendation_widget_shows_top3_for_character" -v`
Kỳ vọng: PASS

- [ ] **Bước 5: Chạy toàn bộ kiểm thử hồi quy**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py -q`
Kỳ vọng: 115+ passed, 0 failed.

- [ ] **Bước 6: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): integrate VoiceRecommendationEngine into character profile cards with top 3 UI"
```

---

### Nhiệm vụ 4: Tóm Tắt Kết Quả Nhận Diện Trực Quan Và Nhãn Nguồn Phân Tích

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (hàm `_on_speakers_detected`, dòng ~2918-2937 và khu vực đầu bảng phân vai `page_step_1`, dòng ~840-860)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện cung cấp:**
- Widget mới: `self.lbl_diarization_summary` — tóm tắt kết quả nhận diện (số nhân vật, số câu, số cần rà soát).
- Widget mới: `self.lbl_diarization_method` — nguồn phân tích ("Nhận diện từ âm thanh video", "Phân vai từ văn bản SRT").

- [ ] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_diarization_summary_label_updates_after_detection():
    """
    Xác minh sau khi _on_speakers_detected() được gọi,
    nhãn lbl_diarization_summary hiển thị đủ thông tin:
    số nhân vật, số câu đã phân vai.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
    blk1.speaker = "Nhân vật A"
    blk1.confidence_level = "high"
    blk1.is_speaker_locked = True

    blk2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
    blk2.speaker = "Nhân vật B"
    blk2.confidence_level = "review_needed"
    blk2.is_speaker_locked = True

    blk3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu 3")
    blk3.speaker = "Nhân vật A"
    blk3.confidence_level = "high"
    blk3.is_speaker_locked = True

    ui.subtitle_blocks = [blk1, blk2, blk3]
    profiles = {
        "Nhân vật A": CharacterVoiceProfile(speaker_id="spk_0", display_name="Nhân vật A"),
        "Nhân vật B": CharacterVoiceProfile(speaker_id="spk_1", display_name="Nhân vật B"),
    }
    ui._on_speakers_detected(profiles)

    summary_text = ui.lbl_diarization_summary.text()
    assert "2" in summary_text, f"Phải đề cập 2 nhân vật: {summary_text}"
    assert "câu" in summary_text or "3" in summary_text, f"Phải đề cập số câu: {summary_text}"
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_diarization_summary_label_updates_after_detection" -v`
Kỳ vọng: FAIL vì thiếu `lbl_diarization_summary`.

- [ ] **Bước 3: Viết mã nguồn vào `src/desktop/ui/omnivoice_interface.py`**

**3a.** Tìm khu vực đầu bảng phân vai tại `page_step_1` (sau `casting_header_row`). Thêm hai nhãn trước bảng:

```python
        self.lbl_diarization_summary = CaptionLabel("Chưa phân tích — bấm nhận diện để bắt đầu.", casting_card)
        self.lbl_diarization_summary.setWordWrap(True)
        self.lbl_diarization_method = CaptionLabel("", casting_card)
        self.lbl_diarization_method.setWordWrap(True)
        casting_layout.addWidget(self.lbl_diarization_summary)
        casting_layout.addWidget(self.lbl_diarization_method)
```

**3b.** Trong `_on_speakers_detected`, sau khi chuẩn hóa `self.speaker_profiles` và trước khi gọi `_populate_subtitle_table()`, thêm:

```python
        # Cập nhật nhãn tóm tắt kết quả nhận diện
        total_chars = len(self.speaker_profiles)
        total_assigned = sum(1 for blk in self.subtitle_blocks if getattr(blk, "is_speaker_locked", False))
        review_count = sum(1 for blk in self.subtitle_blocks if getattr(blk, "confidence_level", "") == "review_needed")
        unassigned_count = sum(1 for blk in self.subtitle_blocks if getattr(blk, "confidence_level", "") == "unassigned")
        total_subs = len(self.subtitle_blocks)

        summary = (
            f"Đã phân tích xong. Tìm thấy {total_chars} nhân vật, "
            f"{total_assigned}/{total_subs} câu thoại được phân vai"
        )
        if review_count > 0:
            summary += f", {review_count} câu cần rà soát"
        if unassigned_count > 0:
            summary += f", {unassigned_count} câu chưa xác định"
        summary += "."
        if hasattr(self, "lbl_diarization_summary"):
            self.lbl_diarization_summary.setText(summary)

        has_video = bool(self.current_video_path and os.path.exists(self.current_video_path))
        method_text = "Nguồn: Nhận diện từ âm thanh video." if has_video else "Nguồn: Phân vai từ văn bản SRT."
        if hasattr(self, "lbl_diarization_method"):
            self.lbl_diarization_method.setText(method_text)
```

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_diarization_summary_label_updates_after_detection" -v`
Kỳ vọng: PASS

- [ ] **Bước 5: Chạy toàn bộ kiểm thử hồi quy**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py test/test_srt_translator.py test/test_subtitle_and_logo_remover.py test_home_interface.py -q`
Kỳ vọng: 119+ passed, 0 failed.

- [ ] **Bước 6: Cam kết và đẩy lên máy chủ**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): show detection result summary and source method label in casting step"
git push origin main
git checkout feature/ui-redesign-xoa-phu-de
git merge main
git push origin feature/ui-redesign-xoa-phu-de
git checkout main
```
