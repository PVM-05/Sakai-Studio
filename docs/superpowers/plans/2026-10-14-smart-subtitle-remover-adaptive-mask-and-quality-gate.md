# Kế Hoạch Triển Khai: Xóa Phụ Đề Thích Ứng, Đồng Bộ Mặt Nạ Và Kiểm Soát Chất Lượng Đầu Ra

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng cơ chế mặt nạ tự co giãn theo kích thước chữ thực tế, khắc phục triệt để 3 lỗi ưu tiên P1 (đồng bộ cọ, đồng bộ xem trước, cổng chất lượng sau xuất) và ngăn chặn tình trạng xóa lan sang nền hoặc bỏ sót chữ.

**Architecture:** Tách biệt vùng tìm kiếm mở rộng với mặt nạ thích ứng theo chiều cao chữ; thống nhất quy trình nhận diện và tạo mặt nạ cho xem trước và xuất video; nạp 5 khung hình lân cận thực tế cho các mô hình thời gian (STTN, ProPainter); tích hợp cổng kiểm soát chất lượng tự động quét chữ còn sót sau khi xuất video.

**Tech Stack:** Python 3.12, PySide6, OpenCV, RapidOCR (PaddleX), NumPy, PyTorch, pytest.

## Global Constraints

- 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt.
- Tuyệt đối không dùng biểu tượng cảm xúc hoặc hình trang trí trong tiêu đề, văn bản và thẻ giao diện.
- Sử dụng biểu tượng FluentIcon.
- Tuân thủ quy trình kiểm thử trước mã nguồn sau (TDD) cho mọi tác vụ.

---

### Task 1: Đồng bộ hóa trạng thái nét cọ khi xóa nét cọ (Khắc phục lỗi P1)

**Files:**
- Modify: `src/desktop/ui/component/video_display_component.py:784-795`
- Modify: `src/desktop/ui/home_interface.py:180-185, 1090-1100, 1200-1205`
- Test: `test/test_subtitle_and_logo_remover.py`

**Interfaces:**
- Consumes: `TaskOptions.FREEHAND_STROKES`, `video_display_component.clear_freehand_strokes()`
- Produces: `video_display_component.strokes_cleared` signal hoặc hàm dọn dẹp ghi đè cấu hình tác vụ `FREEHAND_STROKES = []`.

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_subtitle_and_logo_remover.py`:
```python
def test_clear_freehand_strokes_persists_empty_list_to_task_options():
    """
    Xác minh khi bấm nút 'Xóa nét cọ':
    1. Cấu hình TaskOptions.FREEHAND_STROKES của tác vụ đang chọn được cập nhật thành [].
    2. Khi khởi chạy hàng đợi, danh sách nét cọ truyền vào options['strokes'] là rỗng,
       không bị khôi phục lại các nét cọ cũ đã lưu trước đó.
    """
    from PySide6.QtWidgets import QApplication
    from src.desktop.ui.home_interface import HomeInterface
    from src.desktop.ui.component.task_list_component import TaskItem, TaskOptions

    app = QApplication.instance() or QApplication([])
    ui = HomeInterface()
    
    # Tạo tác vụ giả lập có chứa nét cọ cũ
    t_item = TaskItem("dummy.mp4", 1)
    ui.task_list_component.add_task(t_item)
    ui.task_list_component.select_task(0)
    old_strokes = [{"radius": 0.02, "points": [(10, 10), (20, 20)], "is_eraser": False}]
    ui.task_list_component.update_task_option(0, TaskOptions.FREEHAND_STROKES, old_strokes)
    ui.video_display_component.freehand_strokes = list(old_strokes)

    # Kích hoạt hành động xóa nét cọ
    ui._handle_clear_brush_clicked()

    # Xác minh nét cọ trong cả giao diện và cấu hình tác vụ đều đã thành rỗng
    assert len(ui.video_display_component.freehand_strokes) == 0
    saved_strokes = ui.task_list_component.get_task_option(0, TaskOptions.FREEHAND_STROKES, None)
    assert saved_strokes == []
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_clear_freehand_strokes_persists_empty_list_to_task_options" -v
```
Kỳ vọng: Thất bại với lỗi `AttributeError: 'HomeInterface' object has no attribute '_handle_clear_brush_clicked'` hoặc `AssertionError: assert [...] == []`.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

1. Trong `src/desktop/ui/component/video_display_component.py`:
```python
    def clear_freehand_strokes(self):
        """Xóa tất cả các nét cọ vẽ tự do và thông báo thay đổi"""
        self.freehand_strokes.clear()
        self.current_stroke = None
        self.update_preview_with_rect()
```

2. Trong `src/desktop/ui/home_interface.py`:
Định nghĩa hàm xử lý dọn dẹp nét cọ:
```python
    def _handle_clear_brush_clicked(self):
        """Xóa sạch nét cọ trên màn hình và lập tức đồng bộ danh sách rỗng vào tác vụ hiện tại"""
        if hasattr(self, 'video_display_component'):
            self.video_display_component.clear_freehand_strokes()
        cur_idx = self.task_list_component.get_current_task_index()
        if cur_idx >= 0:
            self.task_list_component.update_task_option(cur_idx, TaskOptions.FREEHAND_STROKES, [])
```
Kết nối nút `clear_brush_btn`:
```python
        if hasattr(self.setting_interface, 'clear_brush_btn'):
            self.setting_interface.clear_brush_btn.clicked.connect(self._handle_clear_brush_clicked)
```
Trong `run_button_clicked()`:
```python
            selected_idx = self.task_list_component.get_current_task_index()
            if selected_idx >= 0 and hasattr(self, 'video_display_component'):
                cur_strokes = list(getattr(self.video_display_component, 'freehand_strokes', []))
                self.task_list_component.update_task_option(selected_idx, TaskOptions.FREEHAND_STROKES, cur_strokes)
```

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_clear_freehand_strokes_persists_empty_list_to_task_options" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết mã nguồn**

```powershell
git add src/desktop/ui/component/video_display_component.py src/desktop/ui/home_interface.py test/test_subtitle_and_logo_remover.py
git commit -m "fix(remover): sync empty stroke list to task options on brush clear"
```

---

### Task 2: Mở rộng vùng tìm kiếm và bộ lọc hộp chữ linh hoạt (Giai đoạn A)

**Files:**
- Modify: `src/core/tools/subtitle_detect.py:200-265`
- Test: `test/test_subtitle_and_logo_remover.py`

**Interfaces:**
- Consumes: `area = (ymin, ymax, xmin, xmax)`, `frame_w`, `frame_h`
- Produces: `expand_search_roi(area, frame_w, frame_h, scale_w=0.35, scale_h=0.60, min_pad_x=24, min_pad_y=16)` -> `(ymin_crop, ymax_crop, xmin_crop, xmax_crop)`
- Produces: `overlaps_selected_region(box, roi, min_ratio=0.15)` -> `bool`

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_subtitle_and_logo_remover.py`:
```python
def test_expand_search_roi_adds_margins_and_clips_boundary():
    """Kiểm tra mở rộng lề tìm kiếm theo tỷ lệ cạnh và kẹp biên an toàn trong khung hình."""
    from src.core.tools.subtitle_detect import expand_search_roi

    # Vùng chọn 300x40 tại vị trí (ymin=100, ymax=140, xmin=200, xmax=500), khung hình 1920x1080
    area = (100, 140, 200, 500)
    # pad_x = max(24, round(300 * 0.35)) = 105
    # pad_y = max(16, round(40 * 0.60)) = 24
    exp_area = expand_search_roi(area, frame_w=1920, frame_h=1080)
    assert exp_area == (100 - 24, 140 + 24, 200 - 105, 500 + 105)

    # Trường hợp sát biên trái và trên (phải được kẹp ở 0)
    edge_area = (10, 50, 5, 200)
    clipped_area = expand_search_roi(edge_area, frame_w=1920, frame_h=1080)
    assert clipped_area[0] == 0  # ymin kẹp về 0
    assert clipped_area[2] == 0  # xmin kẹp về 0


def test_detect_subtitle_accepts_partially_overlapping_box():
    """
    Xác minh detect_subtitle chấp nhận hộp chữ tràn nhẹ ra ngoài vùng khoanh
    khi có tâm nằm trong vùng khoanh hoặc tỷ lệ giao thoa >= 15%.
    """
    from unittest.mock import MagicMock
    from src.core.tools.subtitle_detect import SubtitleDetect

    detector = SubtitleDetect("dummy.mp4", sub_areas=[(100, 150, 100, 300)])
    # Giả lập hộp chữ phát hiện được (xmin=90, xmax=310, ymin=95, ymax=155) tràn nhẹ cả 4 phía nhưng tâm nằm trong
    dummy_img = np.full((300, 500, 3), 128, dtype=np.uint8)
    # Vẽ chữ giả để std() > 3.0
    dummy_img[100:150, 100:300] = 255
    
    detector._get_coordinates_from_paddle = MagicMock(return_value=[(90 - 0, 310 - 0, 95 - 0, 155 - 0)])
    
    results = detector.detect_subtitle(dummy_img)
    assert len(results) == 1
    # Hộp chữ tràn biên không bị loại bỏ
    assert results[0] == (90, 310, 95, 155)
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_expand_search_roi_adds_margins_and_clips_boundary or test_detect_subtitle_accepts_partially_overlapping_box" -v
```
Kỳ vọng: Thất bại với `ImportError: cannot import name 'expand_search_roi'` hoặc `AssertionError: assert len(results) == 1` thất bại vì hộp tràn biên bị lọc mất.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

Trong `src/core/tools/subtitle_detect.py`:
1. Thêm hàm `expand_search_roi`:
```python
def expand_search_roi(area, frame_w, frame_h, scale_w=0.35, scale_h=0.60, min_pad_x=24, min_pad_y=16):
    """Mở rộng vùng tìm kiếm chữ có lề đệm và kẹp biên an toàn trong khung hình."""
    ymin, ymax, xmin, xmax = map(int, area)
    roi_w = max(1, xmax - xmin)
    roi_h = max(1, ymax - ymin)
    pad_x = max(int(min_pad_x), int(round(roi_w * scale_w)))
    pad_y = max(int(min_pad_y), int(round(roi_h * scale_h)))
    return (
        max(0, ymin - pad_y),
        min(int(frame_h), ymax + pad_y),
        max(0, xmin - pad_x),
        min(int(frame_w), xmax + pad_x),
    )


def overlaps_selected_region(box, roi, min_ratio=0.15):
    """Kiểm tra hộp chữ có tâm nằm trong ROI hoặc tỷ lệ diện tích giao nhau >= min_ratio."""
    x1, x2, y1, y2 = box
    ry1, ry2, rx1, rx2 = roi
    inter_w = max(0, min(x2, rx2) - max(x1, rx1))
    inter_h = max(0, min(y2, ry2) - max(y1, ry1))
    box_area = max(1, (x2 - x1) * (y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
    overlap_ratio = (inter_w * inter_h) / float(box_area)
    return center_inside or (overlap_ratio >= min_ratio)
```
2. Cập nhật `detect_subtitle()`:
Sử dụng `expand_search_roi()` để tạo ảnh cắt có lề đệm. Dịch tọa độ chính xác. Khi lọc kết quả theo `sub_areas`, sử dụng `overlaps_selected_region()` thay vì ép buộc hộp chữ phải nằm 100% bên trong.
Kiểm tra nếu có hộp chữ nào chạm sát mép crop (cách biên <= 3 điểm ảnh), tự động mở rộng vùng crop thêm 1 lần và gộp kết quả.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_expand_search_roi_adds_margins_and_clips_boundary or test_detect_subtitle_accepts_partially_overlapping_box" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết mã nguồn**

```powershell
git add src/core/tools/subtitle_detect.py test/test_subtitle_and_logo_remover.py
git commit -m "feat(detect): implement adaptive search ROI and soft overlap filtering"
```

---

### Task 3: Mặt nạ thích ứng theo chiều cao chữ (Giai đoạn B)

**Files:**
- Modify: `src/core/tools/inpaint_tools.py:70-100, 175-225, 320-345`
- Test: `test/test_subtitle_and_logo_remover.py`

**Interfaces:**
- Consumes: `box = (xmin, xmax, ymin, ymax)`, `k=0.15`, `p_min=3`, `p_max=20`
- Produces: `calc_adaptive_padding(text_height)` -> `int`

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_subtitle_and_logo_remover.py`:
```python
def test_adaptive_mask_dilation_scales_with_text_height():
    """
    Xác minh mặt nạ hộp thích ứng co giãn lề theo chiều cao chữ:
    Chữ nhỏ (chiều cao 20px) có khoảng đệm 3px;
    Chữ lớn (chiều cao 80px) có khoảng đệm 12px;
    Không bao giờ vượt quá trần p_max = 20px.
    """
    from src.core.tools.inpaint_tools import calc_adaptive_padding, _build_box_uint8_mask

    assert calc_adaptive_padding(20) == 3
    assert calc_adaptive_padding(80) == 12
    assert calc_adaptive_padding(250) == 20

    # Kiểm tra mặt nạ thực tế
    mask_small = _build_box_uint8_mask((200, 200), [(50, 150, 80, 100)], dilation=0)
    # Chiều cao hộp là 20 -> pad = 3 -> vùng mặt nạ từ y = 77 đến 103 (chiều cao 26)
    y_indices, _ = np.where(mask_small > 0)
    assert y_indices.max() - y_indices.min() + 1 == 26
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_adaptive_mask_dilation_scales_with_text_height" -v
```
Kỳ vọng: Thất bại với `ImportError: cannot import name 'calc_adaptive_padding'`.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

Trong `src/core/tools/inpaint_tools.py`:
1. Thêm hàm `calc_adaptive_padding`:
```python
def calc_adaptive_padding(text_height: int, k: float = 0.15, p_min: int = 3, p_max: int = 20) -> int:
    """Tính toán khoảng đệm thích ứng theo chiều cao chữ để bao trọn viền và bóng đổ."""
    h = max(1, int(text_height))
    pad = int(round(h * k))
    return max(int(p_min), min(int(p_max), pad))
```
2. Trong `_build_box_uint8_mask` và `_build_text_stroke_uint8_mask`:
Tự động áp dụng `calc_adaptive_padding(abs(ymax - ymin))` làm khoảng đệm tối thiểu cho hộp chữ.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_adaptive_mask_dilation_scales_with_text_height" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết mã nguồn**

```powershell
git add src/core/tools/inpaint_tools.py test/test_subtitle_and_logo_remover.py
git commit -m "feat(inpaint): add adaptive mask dilation based on detected text height"
```

---

### Task 4: Đồng bộ hóa Xem Trước với Xuất Video và Nạp Khung Hình Thực Tế (Khắc phục lỗi P1 & Giai đoạn C)

**Files:**
- Modify: `src/desktop/ui/home_interface.py:1780-1935`
- Test: `test/test_subtitle_and_logo_remover.py`

**Interfaces:**
- Consumes: `self.current_frame_no`, `self.current_video_path`, `self.sub_areas`, `inpaint_mode`
- Produces: Chuỗi 5 khung hình lân cận thực tế nạp vào STTN/ProPainter trong luồng xem trước.

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_subtitle_and_logo_remover.py`:
```python
def test_preview_loads_actual_adjacent_video_frames_with_boundary_clamp(tmp_path):
    """
    Xác minh hàm tải khung hình lân cận cho xem trước:
    1. Đọc đúng 5 khung hình thực tế quanh vị trí hiện tại (current_frame_no).
    2. Khi ở đầu video (frame 0 hoặc 1) hoặc cuối video, kẹp chỉ số an toàn không gây crash.
    """
    from src.desktop.ui.home_interface import _load_adjacent_video_frames

    # Tạo tệp video ngắn 10 frame thử nghiệm với màu sắc thay đổi
    video_file = str(tmp_path / "test_frames.mp4")
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (100, 100))
    for f in range(10):
        frame = np.full((100, 100, 3), f * 20, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Vị trí giữa video: frame 5 (0-indexed) -> lấy frames [3, 4, 5, 6, 7]
    frames = _load_adjacent_video_frames(video_file, current_frame_idx=5, count=5)
    assert len(frames) == 5
    # Các khung hình phải có màu sắc khác nhau thực tế
    assert int(frames[0][0, 0, 0]) == 3 * 20
    assert int(frames[2][0, 0, 0]) == 5 * 20
    assert int(frames[4][0, 0, 0]) == 7 * 20

    # Vị trí đầu video: frame 0 -> kẹp chỉ số [0, 0, 0, 1, 2]
    first_frames = _load_adjacent_video_frames(video_file, current_frame_idx=0, count=5)
    assert len(first_frames) == 5
    assert int(first_frames[2][0, 0, 0]) == 0
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_preview_loads_actual_adjacent_video_frames_with_boundary_clamp" -v
```
Kỳ vọng: Thất bại với `ImportError: cannot import name '_load_adjacent_video_frames'`.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

1. Thêm hàm `_load_adjacent_video_frames` vào `src/desktop/ui/home_interface.py`:
```python
def _load_adjacent_video_frames(video_path: str, current_frame_idx: int, count: int = 5) -> List[np.ndarray]:
    """Nạp chuỗi khung hình lân cận thực tế quanh vị trí xem trước có kẹp biên an toàn."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half = count // 2
    indices = [max(0, min(total - 1, current_frame_idx + offset)) for offset in range(-half, half + 1)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
        else:
            # Fallback nếu đọc lỗi
            frames.append(frames[-1].copy() if frames else np.zeros((100, 100, 3), dtype=np.uint8))
    cap.release()
    return frames
```
2. Trong `mask_preview_button_clicked`:
- Nếu `config.subtitleDetectMode` khác `NONE`: Chạy `detect_subtitle()` trên `self.current_frame` trước để trích xuất hộp chữ thực tế; nếu không thấy chữ, phát thông báo cảnh báo và không xóa lan.
- Đối với `STTN_AUTO`, `STTN_DET`, `PROPAINTER`: Sử dụng `_load_adjacent_video_frames(self.current_video_path, self.current_frame_no, 5)` để đưa khung hình thực tế vào mô hình.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_preview_loads_actual_adjacent_video_frames_with_boundary_clamp" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết mã nguồn**

```powershell
git add src/desktop/ui/home_interface.py test/test_subtitle_and_logo_remover.py
git commit -m "feat(preview): load real adjacent video frames and align preview mask with export"
```

---

### Task 5: Cổng kiểm soát chất lượng sau xuất video (Khắc phục lỗi P1 & Giai đoạn D)

**Files:**
- Modify: `src/desktop/ui/home_interface.py:1340-1410`
- Test: `test/test_subtitle_and_logo_remover.py`

**Interfaces:**
- Consumes: `output_path`, `sub_areas`, `detect_subtitle()`
- Produces: `verify_media_quality(output_path, sub_areas)` -> `{"passed": bool, "residual_text_found": bool, "flicker_score": float, "message": str}`

- [ ] **Step 1: Viết bài kiểm thử thất bại**

Viết vào `test/test_subtitle_and_logo_remover.py`:
```python
def test_verify_media_quality_detects_residual_text(tmp_path):
    """
    Xác minh hàm kiểm soát chất lượng sau xuất:
    Nếu video đầu ra vẫn còn chữ rõ ràng trong vùng đã xử lý,
    hệ thống trả về passed=False, residual_text_found=True kèm khuyến nghị xử lý.
    """
    from src.desktop.ui.home_interface import verify_media_quality

    # Tạo video đầu ra giả lập vẫn còn chữ 'RESIDUAL' trong vùng (ymin=80, ymax=120, xmin=50, xmax=250)
    video_path = str(tmp_path / "bad_cleaned.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (300, 200))
    for _ in range(10):
        frame = np.full((200, 300, 3), 128, dtype=np.uint8)
        cv2.putText(frame, "RESIDUAL TEXT", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    res = verify_media_quality(video_path, sub_areas=[(80, 120, 50, 250)])
    assert res["passed"] is False
    assert res["residual_text_found"] is True
    assert "chưa sạch" in res["message"] or "Cần kiểm tra" in res["message"]
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_verify_media_quality_detects_residual_text" -v
```
Kỳ vọng: Thất bại với `ImportError: cannot import name 'verify_media_quality'`.

- [ ] **Step 3: Viết mã nguồn tối thiểu**

1. Định nghĩa `verify_media_quality` trong `src/desktop/ui/home_interface.py`:
Lấy mẫu từ 5 đến 15 khung hình trên video đầu ra, dùng `SubtitleDetect` quét lại vùng `sub_areas`. Nếu phát hiện chữ, đánh dấu `passed = False`, `residual_text_found = True`.
2. Trong luồng hoàn thành tác vụ của `HomeInterface`:
Sau khi `verify_media_output` thành công về kỹ thuật, gọi `verify_media_quality()`. Nếu `res["passed"] is False`:
- Gán trạng thái tác vụ là `TaskStatus.WARNING` hoặc `"needs_review"`.
- Hiển thị `InfoBar.warning("Cần kiểm tra chất lượng", res["message"], parent=self, duration=5000)`.

- [ ] **Step 4: Chạy kiểm thử để xác nhận thành công**

Chạy lệnh:
```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_subtitle_and_logo_remover.py -k "test_verify_media_quality_detects_residual_text" -v
```
Kỳ vọng: Đạt (PASSED).

- [ ] **Step 5: Cam kết mã nguồn**

```powershell
git add src/desktop/ui/home_interface.py test/test_subtitle_and_logo_remover.py
git commit -m "feat(quality): integrate post-export residual text quality gate"
```

---

### Task 6: Kiểm thử hồi quy toàn diện hệ thống và nghiệm thu (Giai đoạn E)

**Files:**
- Test: `test/test_auto_pipeline_reliability.py`, `test/test_omnivoice_pipeline.py`, `test/test_srt_translator.py`, `test/test_subtitle_and_logo_remover.py`, `test_home_interface.py`

- [ ] **Step 1: Chạy toàn bộ các bộ kiểm thử của dự án**

```powershell
$env:PYTHONPATH="."; .\venv\Scripts\pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py test/test_srt_translator.py test/test_subtitle_and_logo_remover.py test_home_interface.py -v
```
Kỳ vọng: Đạt 100% (toàn bộ 95+ bài kiểm thử đều PASSED).

- [ ] **Step 2: Đẩy toàn bộ thay đổi lên máy chủ Git**

```powershell
git push origin main
```
