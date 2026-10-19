# Kế hoạch Triển khai: Nền tảng Dữ liệu và Động cơ Đề xuất Giọng Trí tuệ Nhân tạo Cho OmniVoice

> **Dành cho tác nhân thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để triển khai từng nhiệm vụ bên dưới. Các bước sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Xây dựng nền tảng dữ liệu đồng nhất hai chiều cho hồ sơ nhân vật, hoàn thiện siêu dữ liệu danh mục giọng Việt hỗ trợ ba miền Bắc/Trung/Nam, thiết lập động cơ xếp hạng top 3 giọng thông minh và sửa triệt để các lỗi đồng bộ bộ chọn cũng như cơ chế nghe thử tuần tự chống gián đoạn.

**Kiến trúc:** Mở rộng lớp thực thể `CharacterVoiceProfile` để tương thích hoàn toàn giữa truy cập thuộc tính và truy cập từ điển; nâng cấp `VIETNAMESE_VOICE_CATALOG` với các thuộc tính vùng miền, cao độ, phong cách; bổ sung lớp `VoiceRecommendationEngine` tính điểm tương đồng đa tiêu chí; tích hợp hàm đồng bộ bộ chọn `_sync_speaker_selectors` và cơ chế tự phục hồi hàng đợi nghe thử vào `OmniVoiceInterface`.

**Ngăn xếp Công nghệ:** Python 3.12, PySide6, QFluentWidgets, Pytest, NumPy, SoundFile.

## Ràng buộc Toàn cục
- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện. Chỉ dùng biểu tượng Fluent.
- Tuân thủ quy trình kiểm thử trước mã nguồn sau: viết bài kiểm thử xác nhận thất bại trước, sau đó mới viết mã nguồn tối thiểu để bài kiểm thử chuyển sang màu xanh.
- Không được để lại các đoạn mã chờ như "TODO", "TBD" hoặc triển khai dở dang.

---

### Nhiệm vụ 1: Chuẩn hóa Mô hình Dữ liệu `CharacterVoiceProfile` và Hàm `normalize_character_profile`

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/speaker_diarization.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Đầu vào: Dữ liệu hồ sơ nhân vật dạng từ điển hoặc đối tượng `CharacterVoiceProfile`.
- Đầu ra: Đối tượng `CharacterVoiceProfile` hỗ trợ đầy đủ `__getitem__`, `__setitem__`, `__contains__`, `get()`, `to_dict()`, `from_dict()`, và hàm `normalize_character_profile(raw, speaker_id)`.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_character_voice_profile_dict_and_attribute_interoperability():
    """
    Xác minh CharacterVoiceProfile hỗ trợ truy cập hai chiều:
    1. Đọc và ghi thuộc tính qua cú pháp từ điển (profile['field'] = val) không gây lỗi TypeError.
    2. Đọc và ghi qua cú pháp đối tượng (profile.field = val).
    3. Hỗ trợ to_dict(), from_dict(), normalize_character_profile() với các trường mới:
       vietnamese_accent, gender, pitch_level, style_tag, recommended_voices, error_message.
    """
    from src.ai_engines.speaker_diarization import (
        CharacterVoiceProfile,
        normalize_character_profile,
    )

    prof = CharacterVoiceProfile(
        speaker_id="spk_test",
        display_name="Người nói thử nghiệm",
        vietnamese_accent="nam",
        gender="male",
        pitch_level="low",
        style_tag="story",
    )

    # 1. Truy cập và gán dạng từ điển
    assert prof["speaker_id"] == "spk_test"
    assert prof["vietnamese_accent"] == "nam"
    assert prof["gender"] == "male"
    assert prof.get("pitch_level") == "low"

    # Thao tác gán dạng từ điển phải thành công không bị ném TypeError
    prof["method"] = "preset"
    prof["preset_name"] = "nam_mien_nam_tram_am"
    prof["error_message"] = "Lỗi thử nghiệm"
    assert prof.method == "preset"
    assert prof.preset_name == "nam_mien_nam_tram_am"
    assert prof.error_message == "Lỗi thử nghiệm"

    # 2. Chuyển đổi sang từ điển và phục hồi lại
    d = prof.to_dict()
    assert d["speaker_id"] == "spk_test"
    assert d["vietnamese_accent"] == "nam"
    assert d["error_message"] == "Lỗi thử nghiệm"

    reconstructed = CharacterVoiceProfile.from_dict(d)
    assert reconstructed.speaker_id == "spk_test"
    assert reconstructed.vietnamese_accent == "nam"
    assert reconstructed.preset_name == "nam_mien_nam_tram_am"

    # 3. Hàm chuẩn hóa an toàn
    norm1 = normalize_character_profile(prof)
    assert norm1 is prof

    norm2 = normalize_character_profile({"speaker_id": "spk_dict", "display_name": "Nhân vật từ điển", "vietnamese_accent": "trung"})
    assert isinstance(norm2, CharacterVoiceProfile)
    assert norm2.speaker_id == "spk_dict"
    assert norm2.vietnamese_accent == "trung"

    norm3 = normalize_character_profile(None, speaker_id="spk_fallback")
    assert isinstance(norm3, CharacterVoiceProfile)
    assert norm3.speaker_id == "spk_fallback"
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_character_voice_profile_dict_and_attribute_interoperability" -v`
Kỳ vọng: Thất bại do thiếu `__setitem__` hoặc thiếu các trường mới.

- [x] **Bước 3: Viết mã nguồn tối thiểu vào `src/ai_engines/speaker_diarization.py`**

Mở rộng lớp `CharacterVoiceProfile`:
```python
@dataclass
class CharacterVoiceProfile:
    """Hồ sơ cấu hình giọng nói và danh tính nhân vật thống nhất theo đặc tả."""

    speaker_id: str
    display_name: str
    utterance_count: int = 0
    total_duration: float = 0.0
    source_audio_path: str = ""
    method: str = "clone"  # "clone" (nhân bản từ video) hoặc "preset" (giọng danh mục)
    preset_name: str = ""
    voice_instruct: str = "male, young adult, moderate pitch"
    status: str = "locked"  # "locked", "review_needed", "unassigned"
    custom_audio_path: str = ""
    high_confidence_ratio: float = 1.0
    voice_clone_prompt_path: str = ""
    vietnamese_accent: str = "bac"  # "bac", "trung", "nam"
    gender: str = "auto"  # "male", "female", "auto"
    pitch_level: str = "moderate"  # "very_low", "low", "moderate", "high", "very_high"
    style_tag: str = "natural"
    recommended_voices: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""

    # Các thuộc tính bí danh hỗ trợ tương thích ngược 2 chiều
    ...

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển tương thích hoàn toàn với các tầng xử lý hiện có."""
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "count": self.utterance_count,
            "utterance_count": self.utterance_count,
            "dialogue_count": self.utterance_count,
            "total_duration": self.total_duration,
            "ref_audio": self.source_audio_path,
            "ref_audio_path": self.source_audio_path,
            "source_audio_path": self.source_audio_path,
            "source_audio_sample": self.source_audio_path,
            "method": self.method,
            "assigned_strategy": self.method,
            "preset_name": self.preset_name,
            "voice_preset": self.preset_name,
            "voice_instruct": self.voice_instruct,
            "instruct": self.voice_instruct,
            "status": self.status,
            "confidence": "high" if self.high_confidence_ratio >= 0.7 else ("review_needed" if self.high_confidence_ratio >= 0.3 else "unassigned"),
            "confidence_level": "high" if self.high_confidence_ratio >= 0.7 else ("review_needed" if self.high_confidence_ratio >= 0.3 else "unassigned"),
            "custom_audio_path": self.custom_audio_path,
            "high_confidence_ratio": self.high_confidence_ratio,
            "vietnamese_accent": self.vietnamese_accent,
            "gender": self.gender,
            "pitch_level": self.pitch_level,
            "style_tag": self.style_tag,
            "recommended_voices": list(self.recommended_voices),
            "error_message": self.error_message,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        key_alias_map = {
            "count": "utterance_count",
            "dialogue_count": "utterance_count",
            "ref_audio": "source_audio_path",
            "ref_audio_path": "source_audio_path",
            "source_audio_sample": "source_audio_path",
            "assigned_strategy": "method",
            "voice_preset": "preset_name",
            "instruct": "voice_instruct",
        }
        target_key = key_alias_map.get(key, key)
        if hasattr(self, target_key):
            setattr(self, target_key, value)
        else:
            self.__dict__[target_key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterVoiceProfile":
        if not isinstance(data, dict):
            return cls(speaker_id="", display_name="")
        spk_id = str(data.get("speaker_id") or "")
        disp = str(data.get("display_name") or spk_id or "Người nói 1")
        cnt = int(data.get("utterance_count") or data.get("count") or data.get("dialogue_count") or 0)
        dur = float(data.get("total_duration") or 0.0)
        src_path = str(data.get("source_audio_path") or data.get("ref_audio") or data.get("ref_audio_path") or "")
        meth = str(data.get("method") or data.get("assigned_strategy") or "clone")
        preset = str(data.get("preset_name") or data.get("voice_preset") or "")
        ins = str(data.get("voice_instruct") or data.get("instruct") or "male, young adult, moderate pitch")
        stat = str(data.get("status") or "locked")
        custom_path = str(data.get("custom_audio_path") or "")
        ratio = float(data.get("high_confidence_ratio") or 1.0)
        prompt_path = str(data.get("voice_clone_prompt_path") or "")
        accent = str(data.get("vietnamese_accent") or "bac")
        gender = str(data.get("gender") or "auto")
        pitch = str(data.get("pitch_level") or "moderate")
        style = str(data.get("style_tag") or "natural")
        recs = list(data.get("recommended_voices") or [])
        err = str(data.get("error_message") or "")

        return cls(
            speaker_id=spk_id,
            display_name=disp,
            utterance_count=cnt,
            total_duration=dur,
            source_audio_path=src_path,
            method=meth,
            preset_name=preset,
            voice_instruct=ins,
            status=stat,
            custom_audio_path=custom_path,
            high_confidence_ratio=ratio,
            voice_clone_prompt_path=prompt_path,
            vietnamese_accent=accent,
            gender=gender,
            pitch_level=pitch,
            style_tag=style,
            recommended_voices=recs,
            error_message=err,
        )


def normalize_character_profile(raw: Any, speaker_id: str = "") -> CharacterVoiceProfile:
    """Chuẩn hóa mọi dạng dữ liệu hồ sơ nhân vật về một đối tượng CharacterVoiceProfile duy nhất."""
    if isinstance(raw, CharacterVoiceProfile):
        return raw
    if isinstance(raw, dict):
        prof = CharacterVoiceProfile.from_dict(raw)
        if not prof.speaker_id and speaker_id:
            prof.speaker_id = speaker_id
        return prof
    return CharacterVoiceProfile(speaker_id=speaker_id or "spk_unknown", display_name=speaker_id or "Người nói")
```

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_character_voice_profile_dict_and_attribute_interoperability" -v`
Kỳ vọng: PASS

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/ai_engines/speaker_diarization.py test/test_omnivoice_pipeline.py
git commit -m "feat(diarization): standardize CharacterVoiceProfile bidirectional dictionary interoperability"
```

---

### Nhiệm vụ 2: Nâng cấp Siêu Dữ liệu Danh mục Giọng Tiếng Việt `VIETNAMESE_VOICE_CATALOG`

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Đầu ra: Biến `VIETNAMESE_VOICE_CATALOG` chứa đầy đủ các trường `name`, `gender`, `accent`, `age_group`, `pitch`, `style`, `instruct`, `sample_vi`, `slug`.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_voice_catalog_vietnamese_metadata_completeness():
    """
    Xác minh tất cả các mục giọng trong VIETNAMESE_VOICE_CATALOG:
    1. Đều có đủ các trường: name, gender, accent, age_group, pitch, style, instruct, sample_vi, slug.
    2. Giá trị accent nằm trong {'bac', 'trung', 'nam'}.
    3. Giá trị gender nằm trong {'male', 'female'}.
    4. Giá trị pitch nằm trong {'very_low', 'low', 'moderate', 'high', 'very_high'}.
    5. Bảo đảm tồn tại giọng đại diện cho cả 3 miền Bắc, Trung, Nam.
    """
    from src.ai_engines.tts_engine import VIETNAMESE_VOICE_CATALOG

    assert len(VIETNAMESE_VOICE_CATALOG) >= 22
    valid_accents = {"bac", "trung", "nam"}
    valid_genders = {"male", "female"}
    valid_pitches = {"very_low", "low", "moderate", "high", "very_high"}

    accents_found = set()
    for preset_label, info in VIETNAMESE_VOICE_CATALOG.items():
        assert "name" in info, f"Thiếu 'name' trong {preset_label}"
        assert "gender" in info, f"Thiếu 'gender' trong {preset_label}"
        assert "accent" in info, f"Thiếu 'accent' trong {preset_label}"
        assert "age_group" in info, f"Thiếu 'age_group' trong {preset_label}"
        assert "pitch" in info, f"Thiếu 'pitch' trong {preset_label}"
        assert "style" in info, f"Thiếu 'style' trong {preset_label}"
        assert "instruct" in info, f"Thiếu 'instruct' trong {preset_label}"
        assert "sample_vi" in info, f"Thiếu 'sample_vi' trong {preset_label}"
        assert "slug" in info, f"Thiếu 'slug' trong {preset_label}"

        assert info["accent"] in valid_accents, f"Accent không hợp lệ: {info['accent']} tại {preset_label}"
        assert info["gender"] in valid_genders, f"Gender không hợp lệ: {info['gender']} tại {preset_label}"
        assert info["pitch"] in valid_pitches, f"Pitch không hợp lệ: {info['pitch']} tại {preset_label}"
        accents_found.add(info["accent"])

    assert accents_found == valid_accents, f"Chưa đủ cả 3 miền: {accents_found}"
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_voice_catalog_vietnamese_metadata_completeness" -v`
Kỳ vọng: Thất bại vì `VIETNAMESE_VOICE_CATALOG` hiện tại thiếu các trường `accent`, `gender`, `pitch`, v.v.

- [x] **Bước 3: Cập nhật siêu dữ liệu trong `src/ai_engines/tts_engine.py`**

Cập nhật `VIETNAMESE_VOICE_CATALOG` với cấu trúc đầy đủ chuẩn hóa cho 22 giọng, bảo đảm có đầy đủ giọng miền Bắc, miền Trung và miền Nam.

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_voice_catalog_vietnamese_metadata_completeness" -v`
Kỳ vọng: PASS

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py
git commit -m "feat(tts): enrich VIETNAMESE_VOICE_CATALOG metadata with regional accents, gender, pitch and style"
```

---

### Nhiệm vụ 3: Xây dựng Động cơ Gợi ý Giọng Trí tuệ Nhân tạo `VoiceRecommendationEngine`

**Tệp tin:**
- Chỉnh sửa: `src/ai_engines/tts_engine.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Lớp: `VoiceRecommendationEngine`
- Phương thức:
  - `calculate_voice_match(profile: Any, voice_entry: Dict[str, Any]) -> Tuple[int, str]`
  - `recommend_voices(profile: Any, catalog: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]`

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_voice_recommendation_engine_matches_and_ranks_top_k():
    """
    Xác minh VoiceRecommendationEngine:
    1. Đề xuất đúng top 3 giọng đọc có điểm tương đồng cao nhất.
    2. Khớp chính xác giới tính và khẩu âm vùng miền (Ví dụ: Nam miền Nam đề xuất giọng miền Nam lên đầu).
    3. Trả về lý do đề xuất tiếng Việt chuẩn mực và điểm số từ 0 đến 100.
    """
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile
    from src.ai_engines.tts_engine import VoiceRecommendationEngine

    engine = VoiceRecommendationEngine()

    # Trường hợp 1: Nam miền Nam, giọng trầm
    prof_nam = CharacterVoiceProfile(
        speaker_id="spk_nam",
        display_name="Chú Ba",
        gender="male",
        vietnamese_accent="nam",
        pitch_level="low",
        style_tag="ke_chuyen",
    )

    recs_nam = engine.recommend_voices(prof_nam, top_k=3)
    assert len(recs_nam) == 3
    # Vị trí thứ nhất phải là giọng nam miền Nam
    top1_nam = recs_nam[0]
    assert top1_nam["gender"] == "male"
    assert top1_nam["accent"] == "nam"
    assert top1_nam["score"] >= 80
    assert "miền Nam" in top1_nam["reason"]
    assert "Phù hợp" in top1_nam["reason"]

    # Trường hợp 2: Nữ miền Bắc, cao độ cao
    prof_nu = CharacterVoiceProfile(
        speaker_id="spk_nu",
        display_name="Cô Lan",
        gender="female",
        vietnamese_accent="bac",
        pitch_level="high",
        style_tag="tam_su",
    )

    recs_nu = engine.recommend_voices(prof_nu, top_k=3)
    assert len(recs_nu) == 3
    top1_nu = recs_nu[0]
    assert top1_nu["gender"] == "female"
    assert top1_nu["accent"] == "bac"
    assert top1_nu["score"] >= 80
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_voice_recommendation_engine_matches_and_ranks_top_k" -v`
Kỳ vọng: Thất bại do chưa định nghĩa `VoiceRecommendationEngine`.

- [x] **Bước 3: Viết mã nguồn lớp `VoiceRecommendationEngine` vào `src/ai_engines/tts_engine.py`**

Thực hiện:
```python
class VoiceRecommendationEngine:
    """Động cơ tính toán và xếp hạng giọng đọc phù hợp cho từng nhân vật dựa trên đặc tính âm thanh và ngữ cảnh."""

    PITCH_LEVELS = ["very_low", "low", "moderate", "high", "very_high"]

    @classmethod
    def calculate_voice_match(cls, profile: Any, voice_entry: Dict[str, Any]) -> Tuple[int, str]:
        score = 0
        reasons: List[str] = []

        # 1. Giới tính (Trọng số 40 điểm)
        prof_gender = getattr(profile, "gender", None) or (profile.get("gender") if isinstance(profile, dict) else "auto")
        voice_gender = voice_entry.get("gender", "")
        if prof_gender in ("auto", "", None):
            score += 35
        elif prof_gender == voice_gender:
            score += 40
            reasons.append("giới tính phù hợp")
        else:
            return 0, "Khác giới tính"

        # 2. Khẩu âm vùng miền Việt Nam (Trọng số 30 điểm)
        prof_accent = getattr(profile, "vietnamese_accent", None) or (profile.get("vietnamese_accent") if isinstance(profile, dict) else "bac")
        voice_accent = voice_entry.get("accent", "bac")
        accent_names = {"bac": "miền Bắc", "trung": "miền Trung", "nam": "miền Nam"}
        if prof_accent == voice_accent:
            score += 30
            reasons.append(f"giọng {accent_names.get(voice_accent, voice_accent)}")
        else:
            score += 5

        # 3. Cao độ âm thanh (Trọng số 15 điểm)
        prof_pitch = getattr(profile, "pitch_level", None) or (profile.get("pitch_level") if isinstance(profile, dict) else "moderate")
        voice_pitch = voice_entry.get("pitch", "moderate")
        if prof_pitch == voice_pitch:
            score += 15
            reasons.append("cao độ phù hợp")
        elif prof_pitch in cls.PITCH_LEVELS and voice_pitch in cls.PITCH_LEVELS:
            diff = abs(cls.PITCH_LEVELS.index(prof_pitch) - cls.PITCH_LEVELS.index(voice_pitch))
            if diff == 1:
                score += 10
            elif diff == 2:
                score += 5

        # 4. Phong cách và độ tuổi (Trọng số 15 điểm)
        prof_style = getattr(profile, "style_tag", None) or (profile.get("style_tag") if isinstance(profile, dict) else "natural")
        voice_style = voice_entry.get("style", "")
        if prof_style and voice_style and prof_style == voice_style:
            score += 10
            reasons.append("phong cách tương đồng")
        else:
            score += 5

        score = min(100, max(0, score))
        desc = f"Phù hợp {score}% ({', '.join(reasons)})" if reasons else f"Phù hợp {score}%"
        return score, desc

    @classmethod
    def recommend_voices(
        cls,
        profile: Any,
        catalog: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        target_catalog = catalog if catalog is not None else VIETNAMESE_VOICE_CATALOG
        candidates: List[Dict[str, Any]] = []

        for preset_name, info in target_catalog.items():
            sc, reason = cls.calculate_voice_match(profile, info)
            if sc > 0:
                item = dict(info)
                item["preset_name"] = preset_name
                item["score"] = sc
                item["reason"] = reason
                candidates.append(item)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
```

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_voice_recommendation_engine_matches_and_ranks_top_k" -v`
Kỳ vọng: PASS

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py
git commit -m "feat(tts): implement VoiceRecommendationEngine with multi-criteria scoring"
```

---

### Nhiệm vụ 4: Đồng bộ hóa Danh sách Nhân vật và Bộ chọn Gán Giọng Trong Giao diện

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Phương thức: `_sync_speaker_selectors(self)` trong `OmniVoiceInterface`.
- Hành vi: Cập nhật `combo_target_speaker` với danh sách tên nhân vật thực tế mà không làm mất lựa chọn hiện tại.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_speaker_selectors_sync_on_character_rename():
    """
    Xác minh khi danh sách nhân vật thay đổi hoặc đổi tên trên bảng:
    1. _sync_speaker_selectors() cập nhật combo_target_speaker với danh sách tên mới.
    2. Không còn giữ danh sách mặc định cứng 'Người nói 1'..'Người nói 6'.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Giả lập hồ sơ 2 nhân vật đã được đổi tên
    ui.speaker_profiles = {
        "Thám tử Lừng Danh": CharacterVoiceProfile(speaker_id="spk_0", display_name="Thám tử Lừng Danh"),
        "Kẻ Giấu Mặt": CharacterVoiceProfile(speaker_id="spk_1", display_name="Kẻ Giấu Mặt"),
    }

    ui._sync_speaker_selectors()

    items = [ui.combo_target_speaker.itemText(i) for i in range(ui.combo_target_speaker.count())]
    assert "Thám tử Lừng Danh" in items
    assert "Kẻ Giấu Mặt" in items
    assert "Người nói 6" not in items
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_speaker_selectors_sync_on_character_rename" -v`
Kỳ vọng: Thất bại do chưa có phương thức `_sync_speaker_selectors()`.

- [x] **Bước 3: Viết mã nguồn vào `src/desktop/ui/omnivoice_interface.py`**

Triển khai phương thức:
```python
    def _sync_speaker_selectors(self):
        """Đồng bộ danh sách tên nhân vật thực tế vào toàn bộ các bộ chọn trên giao diện."""
        if not hasattr(self, "combo_target_speaker"):
            return
        current_choice = self.combo_target_speaker.currentText().strip()
        names: List[str] = []
        if self.speaker_profiles:
            for spk_name, prof in self.speaker_profiles.items():
                disp = getattr(prof, "display_name", "") or (prof.get("display_name") if isinstance(prof, dict) else "") or spk_name
                if disp and disp not in names:
                    names.append(disp)
        elif self.subtitle_blocks:
            for blk in self.subtitle_blocks:
                spk = getattr(blk, "speaker", "")
                if spk and spk not in names:
                    names.append(spk)

        if not names:
            names = [f"Người nói {i}" for i in range(1, 7)]

        self.combo_target_speaker.blockSignals(True)
        self.combo_target_speaker.clear()
        self.combo_target_speaker.addItems(names)
        if current_choice in names:
            self.combo_target_speaker.setCurrentText(current_choice)
        elif names:
            self.combo_target_speaker.setCurrentIndex(0)
        self.combo_target_speaker.blockSignals(False)
```
Và gọi `self._sync_speaker_selectors()` trong `_on_speakers_detected()`, `_update_casting_table()`, và khi ô tên nhân vật được chỉnh sửa.

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_speaker_selectors_sync_on_character_rename" -v`
Kỳ vọng: PASS

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(ui): add centralized speaker selector synchronization on character updates"
```

---

### Nhiệm vụ 5: Cơ chế Nghe Thử Tuần Tự Tự Phục Hồi Khi Gặp Sự Cố Nhân Vật

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Giao diện:**
- Cập nhật xử lý trong `_on_worker_finished(self, success, out_path, detail, log_msg)`.
- Khi `self._audition_all_active` đang kích hoạt: Nếu một nhân vật lỗi, lưu `error_message`, hiển thị cảnh báo và tiếp tục chuyển sang nhân vật kế tiếp trong `self._audition_queue`.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_audition_all_queue_continues_on_individual_failure():
    """
    Xác minh trong luồng nghe thử tuần tự toàn bộ nhân vật:
    Nếu một nhân vật tổng hợp thất bại (success=False), hệ thống không dừng im lặng
    mà ghi nhận lỗi vào hồ sơ và tự động tiếp tục phát nhân vật kế tiếp trong hàng đợi.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    prof1 = CharacterVoiceProfile(speaker_id="spk_fail", display_name="Nhân vật lỗi")
    prof2 = CharacterVoiceProfile(speaker_id="spk_ok", display_name="Nhân vật tốt")
    ui.speaker_profiles = {"spk_fail": prof1, "spk_ok": prof2}

    ui._audition_queue = ["spk_ok"]
    ui._audition_all_active = True
    ui._audition_all_total = 2
    ui._audition_all_current_idx = 0

    called_next = []
    ui._audition_next_in_queue = lambda: called_next.append(True)

    # Giả lập worker báo thất bại cho nhân vật đầu tiên
    ui._on_worker_finished(False, "", "Lỗi nạp mô hình", "Chi tiết lỗi")

    # Xác minh lỗi được ghi nhận và hàng đợi vẫn tiếp tục gọi nhân vật kế tiếp
    assert prof1.error_message == "Lỗi nạp mô hình"
    assert len(called_next) == 1
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_audition_all_queue_continues_on_individual_failure" -v`
Kỳ vọng: Thất bại do hiện tại `_on_worker_finished` chỉ reset nút khi gặp lỗi mà không gọi `_audition_next_in_queue()`.

- [x] **Bước 3: Viết mã nguồn vào `src/desktop/ui/omnivoice_interface.py`**

Cập nhật hàm `_on_worker_finished()` trong `OmniVoiceInterface`:
```python
        if not success:
            # Ghi nhận lỗi vào hồ sơ nhân vật nếu đang nghe thử
            target_spk = getattr(self, "_current_audition_speaker", "")
            if target_spk and target_spk in self.speaker_profiles:
                prof = self.speaker_profiles[target_spk]
                if hasattr(prof, "error_message"):
                    prof.error_message = detail or "Lỗi tổng hợp giọng nói"
                elif isinstance(prof, dict):
                    prof["error_message"] = detail or "Lỗi tổng hợp giọng nói"

            if self._audition_all_active:
                self.log(f"Nhân vật '{target_spk}' gặp sự cố ({detail}). Tự động chuyển sang nhân vật kế tiếp...")
                self._audition_next_in_queue()
                return

            self._reset_audition_buttons()
            InfoBar.error(
                title="Thất bại",
                content=detail or "Không thể hoàn thành tác vụ OmniVoice.",
                position=InfoBarPosition.TOP,
                parent=self,
                duration=5000,
            )
            return
```

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Chạy lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_audition_all_queue_continues_on_individual_failure" -v`
Kỳ vọng: PASS

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "fix(omnivoice): enable automatic audition queue continuation on individual speaker failure"
```

---

### Nhiệm vụ 6: Chạy Toàn Bộ Kiểm Thử Hồi Quy và Kiểm Tra Hoàn Thành

**Tệp tin:**
- Chạy: Toàn bộ bộ kiểm thử của dự án.

- [x] **Bước 1: Chạy toàn bộ các bài kiểm thử**

```bash
.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py test/test_srt_translator.py test/test_subtitle_and_logo_remover.py test_home_interface.py -q
```
Kỳ vọng: 100% bài kiểm thử đạt kết quả tốt (115+ passed, 0 failed).

- [x] **Bước 2: Đẩy toàn bộ thay đổi lên máy chủ git**

```bash
git push origin main
git checkout feature/ui-redesign-xoa-phu-de
git merge main
git push origin feature/ui-redesign-xoa-phu-de
git checkout main
```
