# Kế Hoạch Triển Khai: Tối Ưu Hóa Chất Lượng Âm Thanh OmniVoice Bằng Kho Giọng Trí Tuệ Nhân Tạo Có Sẵn Và Loại Bỏ Nhân Bản Mẫu Âm Thanh

> **Dành cho kỹ sư / tác tử thực thi:** KỸ NĂNG BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi từng nhiệm vụ trong kế hoạch. Mỗi bước sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Loại bỏ hoàn toàn tính năng nhân bản mẫu âm thanh và vòng lặp suy thoái giọng, đưa kho 22 giọng đọc tiếng Việt của OmniVoice làm trung tâm, cung cấp chế độ một giọng đọc toàn tệp và bảng phân vai tùy chỉnh kèm nút nghe thử cho từng nhân vật.

**Kiến trúc:** 
Chuỗi tiền xử lý chuẩn hóa số đếm tiếng Việt sang chữ viết, động cơ `TTSEngine` tổng hợp trực tiếp từ chuỗi chỉ dẫn văn bản của kho giọng mẫu không qua tệp âm thanh trung gian, khóa bộ nhớ đệm mang đầy đủ dấu vân tay tham số cấu hình, và giao diện 5 bước hỗ trợ cả chế độ một giọng đọc nhanh lẫn chế độ phân vai đa nhân vật trực quan.

**Ngăn xếp công nghệ:** PyTorch, OmniVoice, NumPy, SoundFile, PySide6, QFluentWidgets, Pytest.

## Ràng Buộc Toàn Cục

- Ngôn ngữ giao diện, thông báo và tài liệu phải sử dụng 100% tiếng Việt chuẩn mực.
- Tuyệt đối không dùng từ tiếng Anh đặt trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không sử dụng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Tất cả các thẻ cài đặt phải hỗ trợ tự động xuống dòng và co giãn chiều cao theo nội dung (`setWordWrap(True)`).
- Tần số lấy mẫu mặc định của OmniVoice là 24.000 Hz; mức đỉnh âm thanh đầu ra của tệp hoàn chỉnh được chuẩn hóa an toàn ở mức -1.0 dBFS (biên độ đỉnh <= 0.891).

---

### Nhiệm Vụ 1: Chuẩn Hóa Số Đếm Và Ký Tự Tiếng Việt Trong Tiền Xử Lý Văn Bản

**Tệp thực hiện:**
- Sửa đổi: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_improvements.py`

**Giao diện:**
- Cung cấp: `def normalize_vietnamese_numbers_and_text(text: str) -> str`
- Cung cấp: Tích hợp vào `preprocess_vietnamese_tts_text(text: str) -> str`

- [ ] **Bước 1: Viết kịch bản kiểm thử thất bại**

Thêm hàm kiểm thử vào `test/test_omnivoice_improvements.py`:

```python
def test_vietnamese_number_normalization():
    from src.ai_engines.tts_engine import normalize_vietnamese_numbers_and_text

    # Số nguyên đơn giản và số năm
    assert normalize_vietnamese_numbers_and_text("Năm 2024") == "Năm hai nghìn không trăm hai mươi tư"
    assert normalize_vietnamese_numbers_and_text("Có 15 người") == "Có mười lăm người"
    assert normalize_vietnamese_numbers_and_text("Số 0") == "Số không"
    assert normalize_vietnamese_numbers_and_text("Tầng 21") == "Tầng hai mươi mốt"

    # Số thập phân và phần trăm
    assert normalize_vietnamese_numbers_and_text("Tăng 3.5%") == "Tăng ba phẩy năm phần trăm"
    assert normalize_vietnamese_numbers_and_text("Đạt 100%") == "Đạt một trăm phần trăm"

    # Giữ nguyên thẻ biểu cảm
    assert normalize_vietnamese_numbers_and_text("Chào bạn [laughter] năm 2025") == "Chào bạn [laughter] năm hai nghìn không trăm hai mươi lăm"
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k test_vietnamese_number_normalization`
Kết quả mong đợi: FAIL với lỗi `ImportError: cannot import name 'normalize_vietnamese_numbers_and_text'`

- [ ] **Bước 3: Viết mã nguồn triển khai tối thiểu**

Trong `src/ai_engines/tts_engine.py`, triển khai hàm đọc số tiếng Việt tự nhiên và tích hợp vào `preprocess_vietnamese_tts_text`:

```python
def _number_to_vietnamese_words(n: int) -> str:
    """Chuyển đổi số nguyên không âm (0 đến 999.999.999) sang chữ viết tiếng Việt chuẩn xác."""
    if n == 0:
        return "không"

    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    def read_three_digits(num: int, is_highest: bool = False) -> str:
        hundreds = num // 100
        tens = (num % 100) // 10
        units = num % 10
        res = []

        if hundreds > 0 or not is_highest:
            res.append(f"{digits[hundreds]} trăm")

        if tens > 1:
            res.append(f"{digits[tens]} mươi")
            if units == 1:
                res.append("mốt")
            elif units == 4:
                res.append("tư")
            elif units == 5:
                res.append("lăm")
            elif units > 0:
                res.append(digits[units])
        elif tens == 1:
            res.append("mười")
            if units == 5:
                res.append("lăm")
            elif units > 0:
                res.append(digits[units])
        elif tens == 0:
            if units > 0:
                if hundreds > 0 or not is_highest:
                    res.append("lẻ")
                res.append(digits[units])

        return " ".join(res)

    groups = []
    scale = ["", "nghìn", "triệu", "tỷ"]
    g_idx = 0
    temp = n
    while temp > 0:
        part = temp % 1000
        if part > 0 or g_idx == 0:
            groups.append((part, g_idx))
        temp //= 1000
        g_idx += 1

    words = []
    for part, idx in reversed(groups):
        if part == 0:
            continue
        is_highest = (idx == groups[-1][1])
        r = read_three_digits(part, is_highest=is_highest)
        unit = scale[idx]
        if unit:
            words.append(f"{r} {unit}".strip())
        else:
            words.append(r)

    return " ".join(words).strip()


def normalize_vietnamese_numbers_and_text(text: str) -> str:
    """
    Chuẩn hóa số đếm, số thập phân, phần trăm sang chữ viết tiếng Việt lưu loát.
    Bảo vệ nguyên vẹn các thẻ biểu cảm phi ngôn từ trong ngoặc vuông.
    """
    if not text or not text.strip():
        return ""

    # Tạm thời trích xuất các thẻ biểu cảm để không bị ảnh hưởng
    tags = []
    def save_tag(match):
        tags.append(match.group(0))
        return f"__TAG_{len(tags) - 1}__"

    s = re.sub(r"\[[a-zA-Z0-9_\-]+\]", save_tag, text.strip())

    # 1. Phần trăm
    s = re.sub(r"(?<=\d)\s*%", " phần trăm", s)
    s = re.sub(r"%", " phần trăm ", s)

    # 2. Đơn vị đo lường phổ biến
    units = [
        (r"\bkm/h\b", "ki lô mét trên giờ"),
        (r"\bm/s\b", "mét trên giây"),
        (r"\bkm2\b|\bkm²\b", "ki lô mét vuông"),
        (r"\bm2\b|\bm²\b", "mét vuông"),
        (r"°C\b", " độ C"),
        (r"\bVNĐ\b|\bVND\b", "đồng"),
        (r"\bUSD\b", "đô la Mỹ"),
    ]
    for pattern, repl in units:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # 3. Số thập phân (ví dụ 3.5 hoặc 3,5 đi kèm số)
    def repl_decimal(m):
        integer_part = int(m.group(1))
        decimal_part = "".join(digits[int(d)] for d in m.group(2))
        return f"{_number_to_vietnamese_words(integer_part)} phẩy {decimal_part}"

    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    s = re.sub(r"\b(\d+)[.,](\d+)\b", repl_decimal, s)

    # 4. Số nguyên đứng độc lập (giới hạn dưới 1 tỷ)
    def repl_integer(m):
        val = int(m.group(0))
        if val <= 999_999_999:
            return _number_to_vietnamese_words(val)
        return m.group(0)

    s = re.sub(r"\b\d+\b", repl_integer, s)

    # Khôi phục các thẻ biểu cảm
    for idx, tag in enumerate(tags):
        s = s.replace(f"__TAG_{idx}__", tag)

    s = re.sub(r"\s{2,}", " ", s).strip()
    return s
```

- [ ] **Bước 4: Chạy lại kiểm thử để xác nhận vượt qua**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k test_vietnamese_number_normalization`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Cam kết mã nguồn**

Chạy lệnh:
```bash
git add src/ai_engines/tts_engine.py test/test_omnivoice_improvements.py
git commit -m "feat(tts): chuan hoa so dem va ky hieu tieng viet truoc khi tong hop am thanh"
```

---

### Nhiệm Vụ 2: Loại Bỏ Vòng Lặp Suy Thoái Nhân Bản, Khắc Phục Co Giãn Thời Lượng Và Bổ Sung Dấu Vân Tay Bộ Nhớ Đệm

**Tệp thực hiện:**
- Sửa đổi: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_improvements.py`

**Giao diện:**
- Loại bỏ hoàn toàn khối `self._speaker_prompt_cache[f"consistent_{spk_key}"] = locked_prompt`
- Cập nhật `_compute_block_cache_path` nhận thêm `guidance_scale`, `num_step`, `denoise`
- Thêm bước chuẩn hóa âm lượng đỉnh `-1.0 dBFS` cuối hàm `synthesize_blocks`

- [ ] **Bước 1: Viết kịch bản kiểm thử thất bại**

Thêm hàm kiểm thử vào `test/test_omnivoice_improvements.py`:

```python
def test_synthesize_blocks_never_creates_consistent_prompt_for_design():
    from src.ai_engines.tts_engine import TTSEngine
    from src.ai_engines.translator import SubtitleBlock
    from unittest.mock import MagicMock

    engine = TTSEngine(auto_extract_ref=False, mode="design")
    engine.load_model = MagicMock()
    mock_model = MagicMock()
    sr = 24000
    mock_model.generate.return_value = [np.ones(sr, dtype=np.float32) * 0.3]
    engine.model = mock_model

    blocks = [
        SubtitleBlock(index=1, start_time=0.0, end_time=2.0, text="Câu thứ nhất", speaker="Người nói 1"),
        SubtitleBlock(index=2, start_time=2.5, end_time=4.5, text="Câu thứ hai", speaker="Người nói 1"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_out = f.name

    try:
        report = engine.synthesize_blocks(blocks, total_duration=5.0, output_path=tmp_out)
        # Tuyệt đối không sinh khóa consistent_Người nói 1 để ép các câu sau phải clone câu 1
        assert "consistent_Người nói 1" not in engine._speaker_prompt_cache
        assert report.status in ("success", "partial")
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def test_cache_key_includes_quality_parameters():
    from src.ai_engines.tts_engine import TTSEngine

    engine = TTSEngine(auto_extract_ref=False)
    p1 = engine._compute_block_cache_path(0, "Chào bạn", "spk_1", None, "instruct_1", 2.0)

    # Khi đổi guidance_scale, đường dẫn cache phải đổi khác hoàn toàn
    engine.guidance_scale = 3.5
    p2 = engine._compute_block_cache_path(0, "Chào bạn", "spk_1", None, "instruct_1", 2.0)
    assert p1 != p2

    # Khi đổi num_step, đường dẫn cache phải đổi khác hoàn toàn
    engine.num_step = 16
    p3 = engine._compute_block_cache_path(0, "Chào bạn", "spk_1", None, "instruct_1", 2.0)
    assert p2 != p3
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k "test_synthesize_blocks_never_creates_consistent_prompt_for_design or test_cache_key_includes_quality_parameters"`
Kết quả mong đợi: FAIL

- [ ] **Bước 3: Viết mã nguồn triển khai tối thiểu**

1. Cập nhật `_compute_block_cache_path` trong `src/ai_engines/tts_engine.py`:
```python
    def _compute_block_cache_path(
        self,
        index: int,
        text: str,
        speaker: str,
        ref_path: Optional[str],
        instruct: Optional[str],
        target_dur: Optional[float],
        method_or_mode: Optional[str] = None,
    ) -> Optional[str]:
        """Tính đường dẫn tệp WAV lưu tạm cho từng câu kèm dấu vân tay cấu hình suy luận âm thanh."""
        if not self.resume_cache_dir:
            return None
        mtime_str = ""
        if ref_path and os.path.exists(ref_path):
            try:
                mtime_str = str(os.path.getmtime(ref_path))
            except Exception:
                mtime_str = ""
        raw_key = (
            f"{index}|{text}|{speaker}|{method_or_mode or ''}|{ref_path or ''}|{mtime_str}|"
            f"{instruct or ''}|{target_dur or 0.0:.2f}|{self.mode}|{self.speed:.2f}|"
            f"{self.guidance_scale:.2f}|{self.num_step}|{self.denoise}|{self.t_shift:.2f}"
        )
        digest = hashlib.sha1(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return os.path.join(self.resume_cache_dir, f"blk_{index + 1:04d}_{digest}.wav")
```

2. Trong `synthesize_blocks`, loại bỏ hoàn toàn khối khóa mẫu giọng tự động lấy âm thanh câu trước làm mẫu clone:
Xóa bỏ khối mã từ dòng 2380 đến dòng 2407 trong `tts_engine.py`. Thay vào đó, nếu nhân vật dùng chỉ dẫn, giữ nguyên vẹn chỉ dẫn đó qua tất cả các câu.

3. Trong `synthesize_blocks`, loại bỏ việc co giãn thời lượng hai lần và gọt tỉa khoảng lặng thừa:
```python
                # Gọt tỉa khoảng lặng đệm nhẹ trước khi kiểm tra chồng lấn
                cur_dur_sec = len(audio_arr) / float(self.sample_rate)
                if i + 1 < len(blocks):
                    next_start_sec = parse_timestamp_to_seconds(blocks[i + 1].start_time)
                    avail_before_next = next_start_sec - start_sec
                    if avail_before_next > 0.4 and cur_dur_sec > avail_before_next * 1.05 and target_dur is None:
                        needed_speed = cur_dur_sec / avail_before_next
                        if needed_speed <= 1.15:
                            audio_arr = self._safe_time_stretch(audio_arr, needed_speed, sample_rate=self.sample_rate)
                            report.stretched_count += 1
```

4. Chuẩn hóa mức đỉnh âm lượng an toàn phát thanh (-1.0 dBFS) trước khi xuất tệp:
```python
        # Chuẩn hóa âm lượng đầu ra an toàn theo chuẩn phát thanh, loại bỏ méo đỉnh
        peak_amp = float(np.max(np.abs(full_audio))) if full_audio.size > 0 else 0.0
        if peak_amp > 0.891:
            full_audio = full_audio * (0.891 / peak_amp)
        elif 0.0 < peak_amp < 0.35:
            full_audio = full_audio * min(2.5, 0.70 / peak_amp)

        self.log(f"Đang lưu track lồng tiếng hoàn chỉnh vào: {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, full_audio, self.sample_rate)
```

- [ ] **Bước 4: Chạy lại kiểm thử để xác nhận vượt qua**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k "test_synthesize_blocks_never_creates_consistent_prompt_for_design or test_cache_key_includes_quality_parameters"`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Cam kết mã nguồn**

Chạy lệnh:
```bash
git add src/ai_engines/tts_engine.py test/test_omnivoice_improvements.py
git commit -m "fix(tts): loai bo suy thoai nhan ban giong va chuan hoa am luong dau ra an toan"
```

---

### Nhiệm Vụ 3: Tái Cấu Trúc Bước 1 — Bổ Sung Chế Độ Một Giọng Đọc Toàn Bộ Phụ Đề Và Loại Bỏ Nhân Bản Mẫu Ngoài

**Tệp thực hiện:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_improvements.py`

**Giao diện:**
- Thêm `self.chk_single_unified_voice`: Hộp kiểm `"Sử dụng một giọng đọc đồng nhất cho toàn bộ tệp phụ đề"`
- Thêm `self.combo_single_unified_preset`: Hộp chọn 22 giọng mẫu tiếng Việt
- Thêm `self.btn_preview_single_unified`: Nút `"Nghe Thử Giọng Này"`
- Gỡ bỏ thẻ nhân bản giọng mẫu tệp ngoài (`btn_pick_ref_audio`, `btn_extract_ref_video`, `btn_load_prompt_pt`, `btn_save_prompt_pt`)

- [ ] **Bước 1: Viết kịch bản kiểm thử thất bại**

Thêm hàm kiểm thử vào `test/test_omnivoice_improvements.py`:

```python
def test_single_unified_voice_mode_ui():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    interface = OmniVoiceInterface()
    assert hasattr(interface, "chk_single_unified_voice")
    assert hasattr(interface, "combo_single_unified_preset")
    assert hasattr(interface, "btn_preview_single_unified")

    # Mặc định bật chế độ một giọng đọc đồng nhất để đơn giản hóa thao tác
    assert interface.chk_single_unified_voice.isChecked() is True

    # Khi bật chế độ 1 giọng, cho phép chuyển thẳng sang Bước 3 (Biên tập phụ đề) nếu có phụ đề
    from src.ai_engines.translator import SubtitleBlock
    interface.subtitle_blocks = [SubtitleBlock(index=1, start_time=0.0, end_time=1.0, text="Chào bạn")]
    assert interface._can_switch_to_step(2) is True
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k test_single_unified_voice_mode_ui`
Kết quả mong đợi: FAIL với lỗi `AttributeError`

- [ ] **Bước 3: Viết mã nguồn triển khai tối thiểu**

Trong `src/desktop/ui/omnivoice_interface.py`:
1. Tại khu vực Bước 0 (`page_step_0`), thêm thẻ cấu hình lựa chọn:
```python
        # Khối chế độ một giọng đọc đồng nhất hoặc phân vai đa nhân vật
        voice_mode_card = CardWidget(self.page_step_0)
        voice_mode_layout = QVBoxLayout(voice_mode_card)
        voice_mode_layout.setContentsMargins(18, 14, 18, 14)
        voice_mode_layout.setSpacing(10)

        self.chk_single_unified_voice = CheckBox("Sử dụng một giọng đọc đồng nhất cho toàn bộ tệp phụ đề", voice_mode_card)
        self.chk_single_unified_voice.setChecked(True)
        self.chk_single_unified_voice.setStyleSheet("font-weight: bold; font-size: 14px;")
        voice_mode_layout.addWidget(self.chk_single_unified_voice)

        single_voice_row = QHBoxLayout()
        single_voice_row.setSpacing(10)
        lbl_unified = BodyLabel("Giọng đọc tiếng Việt:", voice_mode_card)
        lbl_unified.setWordWrap(True)
        self.combo_single_unified_preset = ComboBox(voice_mode_card)
        self.combo_single_unified_preset.addItems(list(VIETNAMESE_VOICE_CATALOG.keys()))
        self.combo_single_unified_preset.setMinimumWidth(320)

        self.btn_preview_single_unified = PushButton("Nghe Thử Giọng Này", voice_mode_card)
        self.btn_preview_single_unified.setIcon(FluentIcon.VOLUME)
        self.btn_preview_single_unified.clicked.connect(self._preview_unified_voice)

        single_voice_row.addWidget(lbl_unified)
        single_voice_row.addWidget(self.combo_single_unified_preset, 1)
        single_voice_row.addWidget(self.btn_preview_single_unified)
        voice_mode_layout.addLayout(single_voice_row)

        p0_layout.addWidget(voice_mode_card)
```

2. Gắn kết tín hiệu thay đổi `self.chk_single_unified_voice`:
Khi được chọn, tự động khóa toàn bộ câu thoại phụ đề vào preset này, cập nhật trạng thái nút nhận diện nhân vật.
Loại bỏ thẻ nhân bản mẫu giọng cũ (`self.clone_card`) khỏi giao diện.

- [ ] **Bước 4: Chạy lại kiểm thử để xác nhận vượt qua**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k test_single_unified_voice_mode_ui`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Cam kết mã nguồn**

Chạy lệnh:
```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_improvements.py
git commit -m "feat(ui): them che do mot giong doc dong nhat va loai bo the nhan ban cu"
```

---

### Nhiệm Vụ 4: Tái Cấu Trúc Bước 2 — Bảng Phân Vai Đa Nhân Vật Thuần Trí Tuệ Nhân Tạo Có Chọn Giọng Và Nghe Thử

**Tệp thực hiện:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_improvements.py`

**Giao diện:**
- Cập nhật `self.casting_table`:
  - Cột 0: Mã nhân vật
  - Cột 1: Tên nhân vật
  - Cột 2: Số câu thoại
  - Cột 3: Giọng đọc trí tuệ nhân tạo (ComboBox 22 giọng tiếng Việt)
  - Cột 4: Trạng thái phân vai
  - Cột 5: Thao tác (Nút `"Nghe Thử Giọng"`)
- Khi người dùng đổi giọng trong ComboBox của hàng, tự động cập nhật cấu hình nhân vật và câu thoại tương ứng.

- [ ] **Bước 1: Viết kịch bản kiểm thử thất bại**

Thêm hàm kiểm thử vào `test/test_omnivoice_improvements.py`:

```python
def test_casting_table_pure_ai_presets_and_audition():
    _ensure_qapp()
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    interface = OmniVoiceInterface()
    prof = CharacterVoiceProfile(
        speaker_id="spk_0",
        display_name="Người nói 1",
        method="preset",
        preset_name=list(VIETNAMESE_VOICE_CATALOG.keys())[0],
    )
    interface.speaker_profiles = {"Người nói 1": prof}
    interface._update_casting_table()

    # Bảng phân vai phải có 6 cột chuẩn mực, loại bỏ cột mẫu giọng tệp ngoài
    assert interface.casting_table.columnCount() == 6
    header_labels = [interface.casting_table.horizontalHeaderItem(c).text() for c in range(6)]
    assert "Mẫu Giọng Nguồn" not in header_labels
    assert "Giọng Đọc Trí Tuệ Nhân Tạo" in header_labels
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k test_casting_table_pure_ai_presets_and_audition`
Kết quả mong đợi: FAIL

- [ ] **Bước 3: Viết mã nguồn triển khai tối thiểu**

Trong `src/desktop/ui/omnivoice_interface.py`:
1. Sửa định nghĩa tiêu đề cột trong `_init_ui`:
```python
        self.casting_table.setColumnCount(6)
        self.casting_table.setHorizontalHeaderLabels([
            "Mã Nhân Vật",
            "Tên Nhân Vật",
            "Số Câu Thoại",
            "Giọng Đọc Trí Tuệ Nhân Tạo",
            "Trạng Thái",
            "Thao Tác",
        ])
```

2. Cập nhật hàm `_update_casting_table`:
Tại cột 3, tạo `ComboBox` chứa danh mục 22 giọng đọc tiếng Việt của OmniVoice. Khi người dùng thay đổi lựa chọn, tự động gọi `_on_character_preset_changed(spk_name, new_preset)`.
Tại cột 5, đặt nút `"Nghe Thử Giọng"` kích hoạt phát âm câu đầu tiên của nhân vật bằng giọng đọc đó.

- [ ] **Bước 4: Chạy lại kiểm thử để xác nhận vượt qua**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py -k test_casting_table_pure_ai_presets_and_audition`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Cam kết mã nguồn**

Chạy lệnh:
```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_improvements.py
git commit -m "feat(ui): nang cap bang phan vai nhan vat thuan giong ai kem nghe thu"
```

---

### Nhiệm Vụ 5: Kiểm Thử Toàn Diện Chuỗi Xử Lý Âm Thanh Và Xác Minh Không Hồi Quy

**Tệp thực hiện:**
- Kiểm thử: `test/test_omnivoice_improvements.py`, `test/test_omnivoice_pipeline.py`

- [ ] **Bước 1: Chạy toàn bộ kiểm thử cải tiến OmniVoice**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_improvements.py`
Xác nhận tất cả các kịch bản kiểm thử cải tiến đều vượt qua (100% PASS).

- [ ] **Bước 2: Chạy toàn bộ 91 kịch bản kiểm thử đường ống xử lý**

Chạy lệnh: `venv\Scripts\pytest -q test/test_omnivoice_pipeline.py`
Xác nhận tất cả 91 kịch bản kiểm thử quy trình âm thanh đều vượt qua (100% PASS).

- [ ] **Bước 3: Kiểm tra trạng thái Git và đẩy lên kho lưu trữ**

Chạy lệnh:
```bash
git status -s
git push origin main
```
Xác nhận toàn bộ các nhánh sạch sẽ và đã được đồng bộ với máy chủ kho lưu trữ.
