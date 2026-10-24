# Kế Hoạch Triển Khai: Trực Quan Hóa Trạng Thái Phân Vai Và Thao Tác Hàng Loạt Trên Bảng Phụ Đề

> **Dành cho kỹ sư và tác nhân thực thi:** BẮT BUỘC tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD). Mỗi nhiệm vụ đều có các bước nhỏ được đánh dấu checkbox (`- [ ]`). Cam kết git sau mỗi nhiệm vụ hoàn thành.

**Mục tiêu:** Xây dựng tiện ích thanh tiến độ phân vai trực quan ba màu, đánh dấu màu nền dòng bảng phụ đề theo độ tin cậy và bổ sung thanh công cụ thao tác hàng loạt (gán nhân vật, khóa, mở khóa) cho OmniVoice.

**Kiến trúc:** Bổ sung lớp `CastingProgressBarWidget` và tích hợp các phương thức thao tác hàng loạt trong `src/desktop/ui/omnivoice_interface.py`.

**Ràng buộc bắt buộc toàn cục:**
- 100% tiếng Việt chuẩn mực trong toàn bộ mã nguồn, nhãn, thông báo và tài liệu.
- Tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình khối trang trí.
- Biểu tượng FluentIcon đồng nhất.
- Thực thi trực tiếp tuần tự (Inline execution), không tạo subagent.

---

### Nhiệm Vụ 1: Tiện Ích Thanh Tiến Độ Phân Vai Trực Quan Ba Màu

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_auto_pipeline_reliability.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_auto_pipeline_reliability.py`:
```python
def test_casting_progress_bar_calculates_ratios_correctly(self):
    """
    Xác minh tiện ích CastingProgressBarWidget:
    1. Tính đúng tỷ lệ phần trăm phân đoạn: đã gán/khóa, cần rà soát, chưa xác định.
    2. Cập nhật nhãn chi tiết tiếng Việt chuẩn xác theo dữ liệu câu thoại.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import CastingProgressBarWidget
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    widget = CastingProgressBarWidget()

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
    b1.speaker = "Người nói 1"
    b1.confidence_level = "high"
    b1.is_speaker_locked = True

    b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
    b2.speaker = "Người nói 2"
    b2.confidence_level = "review_needed"
    b2.is_speaker_locked = False

    b3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu 3")
    b3.speaker = ""
    b3.confidence_level = "unassigned"
    b3.is_speaker_locked = False

    b4 = SubtitleBlock(4, "00:00:07,000", "00:00:08,000", "Câu 4")
    b4.speaker = "Người nói 1"
    b4.confidence_level = "high"
    b4.is_speaker_locked = True

    widget.update_progress([b1, b2, b3, b4])

    # 4 câu: 2 câu high/locked (50%), 1 câu review (25%), 1 câu unassigned (25%)
    self.assertAlmostEqual(widget.ratio_high, 0.50, places=2)
    self.assertAlmostEqual(widget.ratio_review, 0.25, places=2)
    self.assertAlmostEqual(widget.ratio_unassigned, 0.25, places=2)

    text = widget.lbl_details.text()
    self.assertIn("3/4", text)  # 3 câu đã gán
    self.assertIn("2 câu đã khóa", text)
    self.assertIn("1 câu cần rà soát", text)
    self.assertIn("1 câu chưa xác định", text)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_casting_progress_bar_calculates_ratios_correctly" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Thêm lớp `CastingProgressBarWidget(QWidget)` vào `src/desktop/ui/omnivoice_interface.py`:
- Vẽ thanh bo góc bằng `QPainter`:
  - Đoạn 1: Xanh lá `#4CAF50` (chiều rộng = `width * ratio_high`)
  - Đoạn 2: Vàng cam `#FFA726` (chiều rộng = `width * ratio_review`)
  - Đoạn 3: Xám nhạt `#E0E0E0` (phần còn lại)
- Nhãn `lbl_details` hiển thị thống kê.
- Tích hợp vào khu vực đầu bảng phân vai trong `page_step_1`.

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_casting_progress_bar_calculates_ratios_correctly" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_auto_pipeline_reliability.py; git commit -m "feat(ui): add CastingProgressBarWidget with visual segmented progress"`

---

### Nhiệm Vụ 2: Đánh Dấu Màu Nền Cảnh Báo Nhẹ Nhàng Trên Từng Dòng Phụ Đề

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_auto_pipeline_reliability.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_auto_pipeline_reliability.py`:
```python
def test_subtitle_table_row_background_colors_reflect_confidence(self):
    """
    Xác minh màu nền các dòng trên bảng phụ đề:
    - Dòng 'review_needed' có màu nền vàng nhạt.
    - Dòng 'unassigned' có màu nền đỏ nhạt.
    - Dòng 'high' hoặc đã khóa giữ màu nền mặc định.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu tin cậy cao")
    b1.speaker = "Người nói 1"
    b1.confidence_level = "high"
    b1.is_speaker_locked = True

    b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu cần rà soát")
    b2.speaker = "Người nói 1"
    b2.confidence_level = "review_needed"
    b2.is_speaker_locked = False

    b3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu chưa xác định")
    b3.speaker = ""
    b3.confidence_level = "unassigned"
    b3.is_speaker_locked = False

    ui.subtitle_blocks = [b1, b2, b3]
    ui._populate_subtitle_table()

    # Dòng 1 (review_needed): màu nền vàng nhạt (kênh red > 240, green > 230, blue < 240)
    bg2 = ui.sub_table.item(1, 0).background().color()
    self.assertGreater(bg2.red(), 240)
    self.assertGreater(bg2.green(), 230)

    # Dòng 2 (unassigned): màu nền đỏ nhạt (kênh red > 240, green < 240)
    bg3 = ui.sub_table.item(2, 0).background().color()
    self.assertGreater(bg3.red(), 240)
    self.assertLess(bg3.green(), 240)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_subtitle_table_row_background_colors_reflect_confidence" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `_populate_subtitle_table` của `OmniVoiceInterface`:
Xác định màu nền dòng:
```python
            if conf == "review_needed":
                row_bg = QColor(255, 248, 225, 180)  # Vàng nhạt
            elif conf == "unassigned" or not spk or spk in ("Chưa xác định", "unassigned"):
                row_bg = QColor(255, 235, 238, 180)  # Đỏ nhạt
            else:
                row_bg = QColor(0, 0, 0, 0)  # Mặc định trong suốt
```
Và áp dụng `item.setBackground(row_bg)` cho tất cả các cột `QTableWidgetItem`.

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_subtitle_table_row_background_colors_reflect_confidence" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_auto_pipeline_reliability.py; git commit -m "feat(ui): apply subtle warning row background colors based on confidence level"`

---

### Nhiệm Vụ 3: Thanh Công Cụ Thao Tác Hàng Loạt Trên Bảng Phụ Đề

**Tệp tin:**
- Sửa đổi: `src/desktop/ui/omnivoice_interface.py`
- Kiểm thử: `test/test_auto_pipeline_reliability.py`

- [ ] **Bước 1: Viết bài kiểm thử thất bại**
Viết vào `test/test_auto_pipeline_reliability.py`:
```python
def test_batch_assign_speaker_updates_selected_rows_and_locks(self):
    """
    Xác minh chức năng gán nhân vật hàng loạt:
    1. Khi chọn dòng 0 và dòng 2 và bấm gán cho 'Nhân vật Đặc Biệt':
       - Dòng 0 và 2 được cập nhật speaker='Nhân vật Đặc Biệt', is_speaker_locked=True.
       - Dòng 1 giữ nguyên.
    2. Chức năng khóa và mở khóa hàng loạt cập nhật đúng trạng thái.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.omnivoice_interface import OmniVoiceInterface
    from src.ai_engines.translator import SubtitleBlock

    app = QApplication.instance() or QApplication([])
    ui = OmniVoiceInterface()

    b1 = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "Câu 1")
    b2 = SubtitleBlock(2, "00:00:03,000", "00:00:04,000", "Câu 2")
    b2.speaker = "Người cũ"
    b3 = SubtitleBlock(3, "00:00:05,000", "00:00:06,000", "Câu 3")

    ui.subtitle_blocks = [b1, b2, b3]
    ui._populate_subtitle_table()

    # Chọn dòng 0 và dòng 2
    ui.sub_table.selectRow(0)
    ui.sub_table.item(2, 0).setSelected(True)

    # Đặt nhân vật mục tiêu trên bộ chọn hàng loạt
    ui.combo_batch_speaker.clear()
    ui.combo_batch_speaker.addItem("Nhân vật Đặc Biệt")
    ui.combo_batch_speaker.setCurrentText("Nhân vật Đặc Biệt")

    # Kích hoạt gán hàng loạt
    ui._batch_assign_selected_speaker()

    self.assertEqual(ui.subtitle_blocks[0].speaker, "Nhân vật Đặc Biệt")
    self.assertTrue(ui.subtitle_blocks[0].is_speaker_locked)
    self.assertEqual(ui.subtitle_blocks[2].speaker, "Nhân vật Đặc Biệt")
    self.assertTrue(ui.subtitle_blocks[2].is_speaker_locked)
    self.assertEqual(ui.subtitle_blocks[1].speaker, "Người cũ")

    # Kích hoạt mở khóa hàng loạt cho dòng 0
    ui.sub_table.clearSelection()
    ui.sub_table.selectRow(0)
    ui._batch_set_locked_state(False)
    self.assertFalse(ui.subtitle_blocks[0].is_speaker_locked)
```

- [ ] **Bước 2: Chạy kiểm thử xác nhận thất bại**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_batch_assign_speaker_updates_selected_rows_and_locks" -v`

- [ ] **Bước 3: Viết mã nguồn tối thiểu**
Trong `src/desktop/ui/omnivoice_interface.py`:
1. Cho phép chọn nhiều dòng: `self.sub_table.setSelectionMode(QAbstractItemView.ExtendedSelection)`.
2. Thêm thanh công cụ hàng loạt `batch_row = QHBoxLayout()`:
   - `self.combo_batch_speaker = ComboBox()`
   - `self.btn_batch_assign = PushButton("Gán Cho Các Câu Đã Chọn")`
   - `self.btn_batch_lock = PushButton("Khóa Câu Đã Chọn")`
   - `self.btn_batch_unlock = PushButton("Mở Khóa Câu Đã Chọn")`
3. Thêm phương thức `_get_selected_subtitle_rows(self) -> List[int]`
4. Thêm phương thức `_batch_assign_selected_speaker(self)`
5. Thêm phương thức `_batch_set_locked_state(self, locked: bool)`
6. Kết nối `_sync_speaker_selectors` để cập nhật cả `combo_batch_speaker`.

- [ ] **Bước 4: Chạy kiểm thử xác nhận đạt**
`.\venv\Scripts\pytest -o testpaths=test/test_auto_pipeline_reliability.py -k "test_batch_assign_speaker_updates_selected_rows_and_locks" -v`

- [ ] **Bước 5: Cam kết git**
`git add src/desktop/ui/omnivoice_interface.py test/test_auto_pipeline_reliability.py; git commit -m "feat(ui): add batch speaker assignment and bulk locking tools to subtitle table"`

---

### Nhiệm Vụ 4: Kiểm Thử Toàn Diện Và Đẩy Mã Nguồn Lên GitHub

- [ ] **Bước 1: Chạy toàn bộ 118+ bài kiểm thử**
`.\venv\Scripts\pytest test/test_omnivoice_pipeline.py test/test_auto_pipeline_reliability.py -q`
- [ ] **Bước 2: Rà soát ngôn ngữ chuẩn mực**
Đảm bảo 100% tiếng Việt chuẩn mực, không từ tiếng Anh trong ngoặc đơn, không biểu tượng cảm xúc.
- [ ] **Bước 3: Đẩy commit lên GitHub**
`git push origin main`
