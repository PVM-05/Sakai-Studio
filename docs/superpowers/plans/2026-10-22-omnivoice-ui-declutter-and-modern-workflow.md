# Kế Hoạch Triển Khai: Tinh Gọn Giao Diện OmniVoice Và Hiện Đại Hóa Quy Trình Lồng Tiếng Đa Nhân Vật

> **Dành cho tác nhân thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:executing-plans` để triển khai từng nhiệm vụ bên dưới. Sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ. Thực thi trực tiếp tuần tự (Inline execution).

**Mục tiêu:** Tinh gọn thị giác, giảm tải số lượng điều khiển hiển thị cùng lúc, đưa thao tác hàng loạt vào menu an toàn có xác nhận, tạo thanh công cụ ngữ cảnh theo lựa chọn dòng câu thoại, và hoàn thiện thanh điều hướng 5 bước chuẩn mực cho OmniVoice trong Sakai Studio.

**Ngăn xếp Công nghệ:** Python 3.12, PySide6 6.x, QFluentWidgets, Pytest.

## Ràng buộc Toàn cục
- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện. Chỉ dùng biểu tượng FluentIcon.
- Tuân thủ quy trình phát triển hướng kiểm thử (TDD): viết bài kiểm thử thất bại trước, viết mã nguồn tối thiểu sau.
- Không được để lại mã chờ như "TODO", "TBD" hoặc triển khai dở dang.

---

## Bản đồ Tệp tin

| Tệp | Vai trò |
|-----|---------|
| `src/desktop/ui/omnivoice_interface.py` | Giao diện OmniVoice chính cần tối ưu hóa |
| `test/test_omnivoice_pipeline.py` | Bộ kiểm thử chức năng và luồng giao diện OmniVoice |
| `test/test_auto_pipeline_reliability.py` | Bộ kiểm thử hồi quy luồng tích hợp hệ thống |

---

### Nhiệm vụ 1: Thu Gọn Thiết Lập Giọng Mặc Định Ở Bước 0 Và Hiển Thị Thích Ứng Theo Chế Độ

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực `page_step_0`, dòng ~603-775)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Thẻ `general_voice_card` và `voice_source_stack` được gom vào một nhóm có thể thu gọn/mở rộng `self.card_default_voice_settings`.
2. Mặc định ở trạng thái thu gọn (`is_expanded = False`) để người dùng tập trung vào Bước nạp tệp và bấm Nhận diện nhân vật. Có nút bấm "Tùy Chọn Giọng Mặc Định (Dự Phòng)" có icon mở/đóng.
3. Khi mở ra, tùy theo `combo_mode` đang chọn, chỉ hiển thị đúng các điều khiển tương ứng:
   - "Nhân Bản Giọng Nói Từ Mẫu": Hiển thị `self.clone_card`, ẩn `self.design_card`.
   - "Thiết Kế Giọng Nói Theo Thuộc Tính": Hiển thị `self.design_card`, ẩn `self.clone_card`.
   - "Giọng Đọc Tự Động": Ẩn cả hai khối phức tạp trên, chỉ giữ bộ chọn ngôn ngữ và danh mục giọng tiếng Việt.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_step0_default_voice_settings_collapsible_and_adaptive_mode():
    """
    Xác minh Bước 0 (Nguồn Đầu Vào):
    1. Có nút thu gọn/mở rộng thiết lập giọng mặc định (mặc định đóng để giảm rối mắt).
    2. Khi chuyển chế độ tạo giọng, các khối điều khiển không liên quan tự động ẩn.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # 1. Kiểm tra trạng thái thu gọn ban đầu
    assert hasattr(ui, "btn_toggle_default_voice_settings"), "Thiếu nút chuyển đổi thu gọn thiết lập giọng mặc định"
    assert hasattr(ui, "default_voice_settings_container"), "Thiếu khung chứa thiết lập giọng mặc định"
    assert not ui.default_voice_settings_container.isVisible(), "Mặc định khung thiết lập giọng mặc định phải thu gọn (ẩn)"

    # Bấm mở rộng
    ui.btn_toggle_default_voice_settings.click()
    assert ui.default_voice_settings_container.isVisible(), "Sau khi bấm mở, khung thiết lập giọng phải hiển thị"

    # 2. Kiểm tra hiển thị thích ứng theo chế độ tạo giọng
    # Chế độ 0: Nhân bản
    ui.combo_mode.setCurrentIndex(0)
    assert ui.voice_source_stack.currentWidget() == ui.clone_card

    # Chế độ 1: Thiết kế
    ui.combo_mode.setCurrentIndex(1)
    assert ui.voice_source_stack.currentWidget() == ui.design_card

    # Chế độ 2: Tự động
    ui.combo_mode.setCurrentIndex(2)
    assert not ui.voice_source_stack.isVisible()
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_step0_default_voice_settings_collapsible_and_adaptive_mode" -v`
Kỳ vọng: FAIL vì thiếu `btn_toggle_default_voice_settings` và `default_voice_settings_container`.

- [x] **Bước 3: Viết mã nguồn tối thiểu vào `src/desktop/ui/omnivoice_interface.py`**
- Đóng gói `general_voice_card` và `voice_source_stack` vào `default_voice_settings_container`.
- Thêm nút `self.btn_toggle_default_voice_settings` với biểu tượng `FluentIcon.CHEVRON_RIGHT` và nhãn "Tùy Chọn Giọng Mặc Định (Dự Phòng)".
- Khi bấm nút, đảo trạng thái `isVisible()` của `default_voice_settings_container` và đổi biểu tượng (`CHEVRON_DOWN` khi mở, `CHEVRON_RIGHT` khi đóng).
- Trong `_on_mode_changed`, nếu chọn "Giọng Đọc Tự Động", ẩn `self.voice_source_stack`.

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_step0_default_voice_settings_collapsible_and_adaptive_mode" -v`
Kỳ vọng: PASS.

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): collapse default voice settings at step 0 with adaptive mode switching"
```

---

### Nhiệm vụ 2: Tinh Gọn Quản Lý Nhân Vật — Menu Thao Tác Khác Và Xác Nhận Đặt Lại Giọng

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực `page_step_1`, dòng ~780-915)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Giảm thiểu các nút trên thẻ `diarize_card` và `casting_card`:
   - Đưa cấu hình nhận diện (chế độ và số người tối đa) thành thanh điều khiển gọn gàng ngay trên đầu bảng phân vai.
   - Các thao tác hàng loạt được chuyển vào một nút bấm menu "Thao Tác Khác" (`DropDownPushButton` hoặc `PushButton` kết hợp menu lệnh `RoundMenu`).
2. Khi người dùng chọn "Đặt Lại Toàn Bộ Giọng", hiển thị hộp thoại xác nhận `MessageBoxBase` hoặc `MessageBox` chuẩn tiếng Việt. Nếu người dùng chọn "Hủy Bỏ", không làm thay đổi hồ sơ nhân vật.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_step1_batch_actions_menu_and_safe_reset_dialog():
    """
    Xác minh Bước 1 (Quản Lý Nhân Vật):
    1. Có nút menu 'Thao Tác Khác' chứa các tác vụ hàng loạt thay vì phơi ra nhiều nút bấm.
    2. Thao tác đặt lại toàn bộ giọng có cơ chế xác nhận an toàn.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.speaker_diarization import CharacterVoiceProfile

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # 1. Có nút menu thao tác khác
    assert hasattr(ui, "btn_more_actions"), "Thiếu nút menu Thao Tác Khác"
    assert ui.btn_more_actions.text() == "Thao Tác Khác"

    # 2. Kiểm tra thao tác đặt lại toàn bộ có xác nhận
    ui.speaker_profiles = {
        "Nhân vật 1": CharacterVoiceProfile(speaker_id="spk_0", display_name="Nhân vật 1", status="locked")
    }

    # Giả lập người dùng từ chối xác nhận (confirm=False)
    ui._handle_reset_all_voices_with_confirmation(confirmed=False)
    assert "Nhân vật 1" in ui.speaker_profiles, "Khi hủy bỏ, hồ sơ nhân vật không được bị xóa"
    assert ui.speaker_profiles["Nhân vật 1"].status == "locked"

    # Giả lập người dùng đồng ý xác nhận (confirmed=True)
    ui._handle_reset_all_voices_with_confirmation(confirmed=True)
    assert len(ui.speaker_profiles) == 0 or all(p.status != "locked" for p in ui.speaker_profiles.values())
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_step1_batch_actions_menu_and_safe_reset_dialog" -v`
Kỳ vọng: FAIL.

- [x] **Bước 3: Viết mã nguồn tối thiểu vào `src/desktop/ui/omnivoice_interface.py`**
- Tạo nút `self.btn_more_actions` kiểu `DropDownPushButton` (từ `qfluentwidgets`) hoặc gắn `RoundMenu` với các mục:
  - "Tự Động Nhận Diện Lại Nhân Vật"
  - "Nghe Thử Toàn Bộ Nhân Vật"
  - "Gán Nhân Vật Cho Các Câu Đang Chọn"
  - "Đặt Lại Toàn Bộ Giọng"
- Thêm phương thức `_handle_reset_all_voices_with_confirmation(self, confirmed=None)`. Nếu `confirmed is None`, hiển thị hộp thoại `MessageBox("Xác nhận đặt lại giọng", "Bạn có chắc chắn muốn đặt lại toàn bộ giọng đã gán về mặc định?", self)`. Nếu người dùng chấp nhận mới gọi logic đặt lại.

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_step1_batch_actions_menu_and_safe_reset_dialog" -v`
Kỳ vọng: PASS.

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): consolidate batch actions into dropdown menu with safe reset confirmation"
```

---

### Nhiệm vụ 3: Thanh Công Cụ Ngữ Cảnh Nổi Bật Ở Bước 2 (Biên Tập Phụ Đề)

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực `page_step_2`, dòng ~957-1090)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Tạo một thanh công cụ ngữ cảnh `self.subtitle_context_bar` phía trên bảng phụ đề:
   - Khi không có câu thoại nào được chọn: hiển thị nhãn gợi ý "Hãy chọn một hoặc nhiều câu thoại bên dưới để phân vai, chèn biểu cảm hoặc nghe thử". Các nút điều khiển bị vô hiệu hóa.
   - Khi có ít nhất 1 câu thoại được chọn: thanh công cụ bật sáng, cho phép:
     - Đổi nhân vật cho câu thoại đang chọn.
     - Chọn biểu cảm và chèn thẻ biểu cảm vào câu.
     - Nghe thử câu đang chọn bằng giọng đã gán.
2. Thu gọn khu vực Phòng thu văn bản trực tiếp (`studio_card`) thành một nhóm có thể mở ra khi cần, giúp bảng phụ đề có toàn bộ không gian hiển thị rộng rãi, không bị chen chúc.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_step2_contextual_toolbar_enables_only_on_subtitle_selection():
    """
    Xác minh Bước 2 (Biên Tập Phụ Đề):
    1. Thanh công cụ ngữ cảnh tự động vô hiệu hóa các thao tác khi không có câu nào được chọn.
    2. Thanh công cụ kích hoạt đầy đủ các chức năng khi có ít nhất 1 dòng được chọn.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    blk1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Xin chào bạn")
    blk2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Tôi là trợ lý ảo")
    ui.subtitle_blocks = [blk1, blk2]
    ui._populate_subtitle_table()

    # Chưa chọn dòng nào
    ui.sub_table.clearSelection()
    ui._update_subtitle_action_buttons()

    assert not ui.btn_preview_selected_row.isEnabled(), "Chưa chọn dòng thì nút nghe thử phải tắt"
    assert not ui.btn_insert_tag_to_sub.isEnabled(), "Chưa chọn dòng thì nút chèn biểu cảm phải tắt"

    # Chọn dòng 0
    ui.sub_table.selectRow(0)
    ui._update_subtitle_action_buttons()

    assert ui.btn_preview_selected_row.isEnabled(), "Đã chọn dòng thì nút nghe thử phải bật"
    assert ui.btn_insert_tag_to_sub.isEnabled(), "Đã chọn dòng thì nút chèn biểu cảm phải bật"
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_step2_contextual_toolbar_enables_only_on_subtitle_selection" -v`

- [x] **Bước 3: Viết mã nguồn tối thiểu vào `src/desktop/ui/omnivoice_interface.py`**
- Cập nhật `_update_subtitle_action_buttons` kiểm tra `len(selected_rows) > 0`:
  - Cập nhật nhãn trạng thái ngữ cảnh `self.lbl_sub_context_hint` ("Đang chọn X câu thoại" hoặc "Chưa chọn câu thoại nào").
  - Bật/tắt `btn_preview_selected_row`, `btn_insert_tag_to_sub`, `combo_sub_tag`, và bộ chọn đổi nhân vật cho dòng đã chọn.
- Thêm cơ chế thu gọn/mở rộng cho `studio_card` ("Phòng Thu Tổng Hợp Trực Tiếp") với nút bấm mở/đóng, mặc định thu gọn để nhường chỗ cho bảng phụ đề.

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_step2_contextual_toolbar_enables_only_on_subtitle_selection" -v`
Kỳ vọng: PASS.

- [x] **Bước 5: Cam kết thay đổi**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): enhance contextual subtitle toolbar and collapse standalone studio"
```

---

### Nhiệm vụ 4: Hoàn Thiện Điều Hướng 5 Bước — Đồng Bộ Trạng Thái Và Điều Kiện Chuyển Bước

**Tệp tin:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py` (khu vực điều hướng `_switch_to_step`, `_update_step_navigation_states`)
- Kiểm thử: `test/test_omnivoice_pipeline.py`

**Mô tả:**
1. Mỗi nút trong 5 bước điều hướng hiển thị trạng thái chuẩn xác:
   - **Bước hiện tại (Active)**: Viền và nền xanh Fluent nổi bật (`#0078D4`, chữ trắng, đậm).
   - **Bước hoàn thành (Completed)**: Có biểu tượng hoàn thành hoặc hiển thị trạng thái đã nạp đủ dữ liệu.
   - **Bước chưa đủ điều kiện (Pending/Disabled)**: Nếu chưa nạp tệp phụ đề ở Bước 0, các bước 1, 2, 3, 4 sẽ có cảnh báo khi bấm vào hoặc ngăn chặn nhảy bước vô lý.
2. Kiểm tra tính mượt mà khi cuộn trong Sakai Studio:
   - Đảm bảo `self.setWidgetResizable(True)`.
   - Các bảng `sub_table`, `casting_table` tự động điều chỉnh chiều cao theo màn hình, tránh tình trạng hai thanh cuộn lồng nhau.

- [x] **Bước 1: Viết bài kiểm thử thất bại vào `test/test_omnivoice_pipeline.py`**

```python
def test_stepper_step_state_sync_and_validation():
    """
    Xác minh thanh điều hướng 5 bước:
    1. Cập nhật kiểu dáng cho bước hiện tại.
    2. Ngăn chặn chuyển sang Bước 1, 2 nếu chưa có tệp phụ đề nào được nạp.
    3. Cho phép chuyển bước bình thường khi đã có phụ đề hợp lệ.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    # Chưa có phụ đề: thử chuyển sang Bước 1
    ui.subtitle_blocks = []
    can_switch = ui._can_switch_to_step(1)
    assert not can_switch, "Chưa nạp phụ đề thì không được phép chuyển sang Bước 1"

    # Thử gọi chuyển bước: vẫn phải ở Bước 0
    ui._switch_to_step(1)
    assert ui.current_step_index == 0, "Không đủ điều kiện phải được giữ lại ở Bước 0"

    # Đã nạp phụ đề: cho phép chuyển sang Bước 1
    ui.subtitle_blocks = [SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")]
    assert ui._can_switch_to_step(1) is True
    ui._switch_to_step(1)
    assert ui.current_step_index == 1
```

- [x] **Bước 2: Chạy kiểm thử xác nhận thất bại**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_stepper_step_state_sync_and_validation" -v`
Kỳ vọng: FAIL vì thiếu `_can_switch_to_step`.

- [x] **Bước 3: Viết mã nguồn tối thiểu vào `src/desktop/ui/omnivoice_interface.py`**
- Thêm phương thức `_can_switch_to_step(self, target_idx: int) -> bool`:
  - Bước 0: Luôn cho phép.
  - Bước 1, 2, 3, 4: Yêu cầu `len(self.subtitle_blocks) > 0`. Nếu chưa có, hiển thị `InfoBar.warning(title="Chưa nạp phụ đề", content="Vui lòng nạp tệp phụ đề ở Bước 1 trước khi tiếp tục.", parent=self)`.
- Cập nhật `_switch_to_step` gọi `_can_switch_to_step`.
- Thêm phương thức `_update_step_navigation_states(self)` để cập nhật màu sắc và trạng thái của các nút bước:
  - Nút của bước có phụ đề/nhận diện hoàn tất hiển thị biểu tượng `FluentIcon.ACCEPT` hoặc dấu tích hoàn thành.

- [x] **Bước 4: Chạy kiểm thử xác nhận đạt**

Lệnh: `.\venv\Scripts\pytest test/test_omnivoice_pipeline.py -k "test_stepper_step_state_sync_and_validation" -v`
Kỳ vọng: PASS.

- [x] **Bước 5: Chạy toàn bộ bộ kiểm thử hồi quy**

Lệnh: `.\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py -q`
Kỳ vọng: 100% passed, 0 failed.

- [x] **Bước 6: Cam kết và đồng bộ git**

```bash
git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py
git commit -m "feat(omnivoice): implement step transition validation and stepper state synchronization"
git checkout feature/ui-redesign-xoa-phu-de
git merge main
git checkout main
```
