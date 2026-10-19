# Tài liệu Thiết kế Kỹ thuật: Nền tảng Dữ liệu và Động cơ Đề xuất Giọng Trí tuệ Nhân tạo Cho OmniVoice

## 1. Tổng quan Dự án
OmniVoice là phân hệ lồng tiếng đa nhân vật thông minh của Sakai Studio. Tài liệu này đặc tả kiến trúc kỹ thuật của Đợt 1, tập trung vào:
- Chuẩn hóa mô hình dữ liệu hồ sơ nhân vật đa phương thức nhằm loại bỏ xung đột kiểu dữ liệu giữa luồng nền và giao diện người dùng.
- Chuẩn hóa siêu dữ liệu danh mục giọng đọc tiếng Việt hỗ trợ vùng miền Bắc, Trung, Nam.
- Thiết lập động cơ đề xuất thông minh xếp hạng 3 giọng đọc phù hợp nhất cho từng người nói.
- Đồng bộ hóa danh sách nhân vật trên toàn bộ các bảng điều khiển giao diện.
- Nâng cấp cơ chế nghe thử tuần tự đa nhân vật có khả năng tự phục hồi và tiếp tục khi gặp sự cố tại một nhân vật riêng lẻ.

---

## 2. Ràng buộc Toàn cục
- **Ngôn ngữ**: 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- **Biểu tượng**: Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện. Chỉ dùng biểu tượng Fluent.
- **Quy trình phát triển**: Tuân thủ phương pháp phát triển hướng kiểm thử: viết kiểm thử trước, viết mã nguồn tối thiểu sau, và xác minh trước khi kết thúc tác vụ.

---

## 3. Kiến trúc Mô hình Dữ liệu Hồ sơ Nhân vật

### 3.1. Lớp thực thể `CharacterVoiceProfile`
Lớp `CharacterVoiceProfile` tại tệp `src/ai_engines/speaker_diarization.py` được mở rộng để cung cấp giao diện truy cập kép: vừa là lớp đối tượng hướng thuộc tính vừa là từ điển tương thích hoàn toàn.

#### Các trường thuộc tính:
- `speaker_id: str`: Định danh duy nhất không đổi của nhân vật (ví dụ: `spk_0`, `spk_1`).
- `display_name: str`: Tên hiển thị của nhân vật do người dùng đặt hoặc hệ thống tạo tự động (ví dụ: `Người nói 1`, `Thám tử`).
- `utterance_count: int`: Số lượng câu thoại được gán cho nhân vật.
- `total_duration: float`: Tổng thời lượng thoại của nhân vật tính theo giây.
- `source_audio_path: str`: Đường dẫn tệp âm thanh trích xuất từ video đại diện cho giọng gốc.
- `method: str`: Phương thức tạo giọng, nhận một trong hai giá trị: `clone` (nhân bản mẫu) hoặc `preset` (giọng danh mục).
- `preset_name: str`: Khóa nhận diện giọng đọc trong danh mục tiếng Việt nếu phương thức là `preset`.
- `voice_instruct: str`: Chuỗi mô tả thuộc tính giọng nói truyền vào mô hình tổng hợp.
- `status: str`: Trạng thái khóa của nhân vật (`locked`, `review_needed`, `unassigned`).
- `custom_audio_path: str`: Đường dẫn tệp âm thanh bên ngoài do người dùng tự tải lên.
- `high_confidence_ratio: float`: Tỷ lệ phần trăm các câu thoại đạt độ tin cậy cao của nhân vật.
- `voice_clone_prompt_path: str`: Đường dẫn lưu tệp mẫu mã hóa giọng nhân bản.
- `vietnamese_accent: str`: Khẩu âm vùng miền Việt Nam (`bac`, `trung`, `nam`). Mặc định là `bac`.
- `gender: str`: Giới tính giọng nói (`male`, `female`, hoặc `auto`). Mặc định là `auto`.
- `pitch_level: str`: Mức cao độ (`very_low`, `low`, `moderate`, `high`, `very_high`). Mặc định là `moderate`.
- `style_tag: str`: Phong cách thể hiện mong muốn (`natural`, `narrative`, `story`, `whisper`, `energetic`). Mặc định là `natural`.
- `recommended_voices: List[Dict[str, Any]]`: Danh sách 3 giọng đọc phù hợp nhất kèm điểm số và lý do do động cơ đề xuất tính toán.
- `error_message: str`: Thông báo lỗi phát sinh khi tổng hợp nghe thử không thành công.

#### Phương thức hỗ trợ tương thích từ điển:
- `__getitem__(self, key: str) -> Any`: Cho phép đọc theo cú pháp `profile["field"]`.
- `__setitem__(self, key: str, value: Any) -> None`: Cho phép ghi theo cú pháp `profile["field"] = val`.
- `__contains__(self, key: str) -> bool`: Cho phép kiểm tra `"field" in profile`.
- `get(self, key: str, default: Any = None) -> Any`: Truy xuất an toàn với giá trị dự phòng.
- `to_dict(self) -> Dict[str, Any]`: Trích xuất từ điển chứa toàn bộ các khóa và bí danh tương thích ngược.
- `from_dict(cls, data: Dict[str, Any]) -> CharacterVoiceProfile`: Phương thức tạo đối tượng từ từ điển bất kỳ.

### 3.2. Hàm Chuẩn hóa Tập trung
Thêm hàm `normalize_character_profile(raw: Any, speaker_id: str = "") -> CharacterVoiceProfile` tại tệp `src/ai_engines/speaker_diarization.py`:
- Nếu `raw` là đối tượng `CharacterVoiceProfile`, trả về chính đối tượng đó.
- Nếu `raw` là từ điển, gọi `CharacterVoiceProfile.from_dict(raw)`.
- Nếu `raw` không hợp lệ hoặc rỗng, tạo một hồ sơ mặc định với `speaker_id` được chỉ định.

---

## 4. Chuẩn hóa Danh mục Giọng Tiếng Việt `VIETNAMESE_VOICE_CATALOG`

Tại tệp `src/ai_engines/tts_engine.py`, nâng cấp danh mục 22 giọng đọc mẫu tiếng Việt để mỗi mục là một cấu trúc hoàn chỉnh:

```python
VIETNAMESE_VOICE_CATALOG = {
    "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức": {
        "slug": "nam_thanh_nien_chuan_bac",
        "name": "Nam Thanh Niên Chuẩn",
        "gender": "male",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "thuyet_minh",
        "instruct": "male, young adult, moderate pitch",
        "sample_vi": "Xin chào các bạn, đây là giọng nam thanh niên chuẩn dùng cho thuyết minh và bản tin thời sự.",
    },
    "Nam Miền Nam Trầm Ấm - Kể Chuyện Và Phim": {
        "slug": "nam_mien_nam_tram_am",
        "name": "Nam Miền Nam Trầm Ấm",
        "gender": "male",
        "accent": "nam",
        "age_group": "middle_aged",
        "pitch": "low",
        "style": "ke_chuyen",
        "instruct": "male, middle-aged, low pitch",
        "sample_vi": "Đằng sau cánh cửa bí ẩn kia là một sự thật mà chưa ai từng dám tiết lộ.",
    },
    "Nữ Miền Nam Trong Trẻo - Tâm Sự Và Tình Cảm": {
        "slug": "nu_mien_nam_trong_treo",
        "name": "Nữ Miền Nam Trong Trẻo",
        "gender": "female",
        "accent": "nam",
        "age_group": "young_adult",
        "pitch": "high",
        "style": "tam_su",
        "instruct": "female, young adult, high pitch",
        "sample_vi": "Có những ký ức dù năm tháng trôi qua bao lâu vẫn luôn vẹn nguyên trong trái tim.",
    },
    "Nữ Miền Trung Dịu Dàng - Thuyết Minh Văn Hóa": {
        "slug": "nu_mien_trung_diu_dang",
        "name": "Nữ Miền Trung Dịu Dàng",
        "gender": "female",
        "accent": "trung",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "thuyet_minh",
        "instruct": "female, young adult, moderate pitch",
        "sample_vi": "Chào mừng bạn đến với vùng đất di sản văn hóa miền Trung thân thương.",
    },
    ...
}
```

Mỗi giọng trong 22 mục đều được gán đúng nhãn `gender` (`male`/`female`), `accent` (`bac`/`trung`/`nam`), `age_group` (`child`/`teenager`/`young_adult`/`middle_aged`/`elderly`), `pitch` (`very_low`/`low`/`moderate`/`high`/`very_high`) và `style`.

---

## 5. Động cơ Gợi ý Giọng Trí tuệ Nhân tạo Thông minh (`VoiceRecommendationEngine`)

Thêm lớp `VoiceRecommendationEngine` vào tệp `src/ai_engines/tts_engine.py`:

### 5.1. Thuật toán Tính điểm Tương đồng (Match Score)
Hàm `calculate_voice_match(profile: CharacterVoiceProfile, voice_entry: Dict[str, Any]) -> Tuple[int, List[str]]`:
- **Giới tính (Tối đa 40 điểm)**:
  - Nếu `profile.gender` là `auto`: tự động khớp 35 điểm.
  - Nếu `profile.gender == voice_entry["gender"]`: được cộng 40 điểm.
  - Nếu `profile.gender != voice_entry["gender"]`: nhận 0 điểm và bị loại khỏi nhóm ưu tiên.
- **Khẩu âm vùng miền Việt Nam (Tối đa 30 điểm)**:
  - Nếu `profile.vietnamese_accent == voice_entry["accent"]`: được cộng 30 điểm.
  - Nếu không chỉ định: cộng 15 điểm.
  - Khác vùng miền: cộng 0 điểm.
- **Cao độ âm thanh (Tối đa 15 điểm)**:
  - Trùng khớp cao độ: 15 điểm.
  - Lệch 1 bậc cao độ (ví dụ `low` so với `very_low` hoặc `moderate`): 10 điểm.
  - Lệch 2 bậc trở lên: 0 điểm.
- **Phong cách và độ tuổi (Tối đa 15 điểm)**:
  - Khớp phong cách: 10 điểm.
  - Khớp nhóm độ tuổi: 5 điểm.

Tổng điểm dao động từ 0 đến 100 điểm.

### 5.2. Hàm Đề xuất Top 3
Hàm `recommend_voices(profile: CharacterVoiceProfile, catalog: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]`:
- Tính điểm cho toàn bộ các mục trong danh mục giọng Việt.
- Sắp xếp theo thứ tự điểm số giảm dần.
- Chọn lọc `top_k` giọng tốt nhất.
- Mỗi phần tử trả về bao gồm:
  - `preset_name`: Tên giọng trong danh mục.
  - `score`: Điểm số phù hợp (0-100).
  - `reason`: Chuỗi lý do đề xuất tiếng Việt chuẩn mực (Ví dụ: "Khớp giọng nam miền Nam, cao độ trầm, phong cách kể chuyện (Phù hợp 95%)").
  - `slug`: Khóa định danh của giọng.
  - `instruct`: Câu lệnh cấu hình giọng.

---

## 6. Đồng bộ hóa và Xử lý Lỗi trên Giao diện

### 6.1. Đồng bộ hóa Danh sách Nhân vật Duy nhất
Trong tệp `src/desktop/ui/omnivoice_interface.py`:
- Viết phương thức `_sync_speaker_selectors()`:
  - Thu thập toàn bộ tên nhân vật thực tế từ `self.speaker_profiles.keys()` hoặc từ các câu thoại trong `self.subtitle_blocks`.
  - Cập nhật đồng bộ các hộp chọn nhân vật mục tiêu (`combo_target_speaker`, v.v.).
  - Giữ nguyên lựa chọn hiện tại nếu tên nhân vật vẫn tồn tại, tránh làm mất trạng thái người dùng đang thao tác.
- Gọi `_sync_speaker_selectors()` tự động khi:
  - Nhận diện người nói hoàn tất trong `_on_speakers_detected()`.
  - Người dùng sửa tên nhân vật trên cột tên của bảng phân vai `_on_casting_cell_changed()`.
  - Thêm hoặc xóa nhân vật thủ công.

### 6.2. Cơ chế Nghe Thử Tuần tự Chống Gián đoạn
Trong `_on_worker_finished()`:
- Nếu cờ `self._audition_all_active` đang bật:
  - Kiểm tra kết quả tổng hợp của nhân vật hiện tại:
    - Nếu thất bại (`success == False`): Cập nhật thông báo lỗi vào hồ sơ nhân vật và giao diện, nhưng **không dừng luồng nghe thử**.
    - Nếu thành công: Phát âm thanh bình thường.
  - Sau khi câu hiện tại kết thúc (hoặc khi phát hiện lỗi không có âm thanh để phát): Tự động lấy nhân vật tiếp theo trong `self._audition_queue` và tiếp tục kích hoạt tác vụ nghe thử cho đến khi hết hàng đợi.
  - Nếu người dùng bấm nút dừng: Hủy toàn bộ hàng đợi còn lại và trả giao diện về trạng thái sẵn sàng.

---

## 7. Kế hoạch Kiểm thử (TDD)
- `test/test_omnivoice_pipeline.py`:
  - `test_character_voice_profile_dict_and_attribute_interoperability`: Kiểm tra việc truy cập và gán thuộc tính qua cả hai kiểu cú pháp, không gây lỗi gán giá trị.
  - `test_voice_catalog_vietnamese_metadata_completeness`: Kiểm tra toàn bộ 22 giọng trong danh mục đều có đủ thông tin giới tính, khẩu âm Bắc/Trung/Nam, cao độ, nhóm tuổi và phong cách.
  - `test_voice_recommendation_engine_top_k`: Kiểm tra thuật toán xếp hạng top 3 giọng phù hợp nhất với các thuộc tính của nhân vật.
  - `test_speaker_selectors_sync_on_character_rename`: Kiểm tra việc đổi tên nhân vật trong bảng phân vai tự động cập nhật hộp chọn gán giọng.
  - `test_audition_all_queue_continues_on_individual_failure`: Giả lập lỗi tại một nhân vật và xác nhận hàng đợi tự động chuyển sang nhân vật kế tiếp.
