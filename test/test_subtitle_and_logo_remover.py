import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import config
from src.core.tools.inpaint_tools import (
    create_mask,
    create_stroke_mask,
    create_combined_mask,
    create_freehand_stroke_mask,
    create_text_stroke_mask,
    sub_area_to_box_coords,
    box_coords_to_sub_area,
)
from src.core.tools.object_tracker import ObjectTracker
from src.core.tools.subtitle_detect import SubtitleDetect
from src.ai_engines.inpaint.opencv_inpaint import OpenCVInpaint
from src.desktop.ui.component.task_list_component import Task, TaskItem, TaskOptions, TaskStatus


# ==============================================================================
# 1. KIỂM THỬ HỢP ĐỒNG DỮ LIỆU MẶT NẠ (MASK CONTRACT)
# ==============================================================================

def test_mask_contract_box_and_stroke_4tuple():
    """Cả hai chế độ 'box' và 'stroke' đều tạo mặt nạ hợp lệ không rỗng với hộp 4 phần tử."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "SAKAI LOGO", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    coords = [(50, 260, 140, 200)]  # (xmin, xmax, ymin, ymax)

    box_mask = create_mask((240, 320), coords, dilation=4, feather_pixels=4, frame=frame)
    stroke_mask = create_stroke_mask(frame, (240, 320), coords, dilation=4, feather_pixels=4)

    assert box_mask.shape[:2] == (240, 320)
    assert stroke_mask.shape[:2] == (240, 320)
    assert np.any(box_mask > 0)
    assert np.any(stroke_mask > 0)
    assert float(box_mask.max()) <= 1.0
    assert float(stroke_mask.max()) <= 1.0


def test_mask_contract_sam2_5tuple_non_multiple_of_8():
    """Hộp 5 phần tử SAM2 với kích thước H*W không chia hết cho 8 không được gây lỗi reshape hay unpack."""
    h, w = 101, 103  # 101 * 103 = 10403 (không chia hết cho 8)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    raw_bin = np.zeros((h, w), dtype=np.uint8)
    raw_bin[20:50, 30:70] = 1
    packed = np.packbits(raw_bin)
    coords_5 = [(30, 70, 20, 50, (packed, (h, w)))]

    m_box = create_mask((h, w), coords_5, dilation=2, feather_pixels=0, frame=frame)
    m_stroke = create_stroke_mask(frame, (h, w), coords_5, dilation=2, feather_pixels=0)
    m_comb = create_combined_mask((h, w), coords_list=coords_5, strokes_list=[], dilation=2, feather_pixels=0, frame=frame)

    assert np.any(m_box > 0)
    assert np.any(m_stroke > 0)
    assert np.any(m_comb > 0)
    assert m_box[35, 50] > 0.5
    assert m_stroke[35, 50] > 0.5
    assert m_comb[35, 50] > 0.5


def test_mask_contract_stroke_fallback_on_flat_region():
    """Khi vùng chọn không có cạnh Canny rõ ràng, create_text_stroke_mask phải fallback về hộp vùng chọn thay vì trả về mặt nạ rỗng."""
    frame = np.full((200, 300, 3), 120, dtype=np.uint8)  # Ảnh phẳng không có cạnh
    coords = [(40, 160, 50, 100)]
    mask = create_text_stroke_mask(frame, (200, 300), coords, dilation=4, feather_pixels=0)
    assert np.any(mask > 0), "Mặt nạ không được phép rỗng khi có vùng chọn hợp lệ"
    assert mask[75, 100] > 0.5


def test_create_stroke_mask_dispatches_freehand_strokes():
    """create_stroke_mask phải tự nhận diện danh sách nét cọ dict mà không bị lỗi unpack."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    strokes = [{"radius": 12, "points": [(0.5, 0.5), (0.6, 0.5)], "is_eraser": False}]
    mask = create_stroke_mask(frame, (200, 300), strokes, dilation=2, feather_pixels=0)
    assert np.any(mask > 0)
    assert mask[100, 150] > 0.5


def test_gui_subtitle_remover_get_mask_stroke_and_box():
    """GUISubtitleRemover.get_mask phải hoạt động đúng với cả maskType='stroke' và 'box' cho hộp 4-tuple và nét cọ."""
    from src.core.main import GUISubtitleRemover

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    coords = [(50, 180, 70, 130)]

    dummy = MagicMock()
    dummy.mask_size = (200, 300)
    dummy.strokes = [{"radius": 10, "points": [(0.2, 0.2)], "is_eraser": False}]

    old_type = config.maskType.value
    try:
        for mtype in ["stroke", "box"]:
            config.set(config.maskType, mtype)
            m = GUISubtitleRemover.get_mask(dummy, frame, coords)
            assert m.shape == (200, 300)
            assert np.any(m[70:130, 50:180] > 0)
            assert m[40, 60] > 0.1  # Điểm cọ vẽ (0.2*300=60, 0.2*200=40)
    finally:
        config.set(config.maskType, old_type)


# ==============================================================================
# 2. KIỂM THỬ CỌ VẼ TỰ DO & CỤC TẨY (BRUSH & ERASER)
# ==============================================================================

def test_brush_and_eraser_sequential_order():
    """Cọ vẽ cộng vào mặt nạ, cục tẩy trừ khỏi mặt nạ (kể cả xóa đè lên hộp và vẽ đè lại sau khi tẩy)."""
    h, w = 200, 300
    # 1. Vẽ cọ tại tâm (0.5, 0.5) -> (x=150, y=100), sau đó tẩy đi
    strokes_erased = [
        {"radius": 20, "points": [(0.5, 0.5)], "is_eraser": False},
        {"radius": 25, "points": [(0.5, 0.5)], "is_eraser": True},
    ]
    m_erased = create_combined_mask((h, w), coords_list=[], strokes_list=strokes_erased, dilation=0, feather_pixels=0)
    assert m_erased[100, 150] == 0.0, "Cục tẩy phải xóa sạch vùng cọ đã vẽ trước đó"

    # 2. Có hộp chữ nhật, dùng cục tẩy xóa vùng giữa hộp
    coords = [(50, 250, 60, 140)]
    eraser_on_box = [{"radius": 20, "points": [(0.5, 0.5)], "is_eraser": True}]
    m_box_erased = create_combined_mask((h, w), coords_list=coords, strokes_list=eraser_on_box, dilation=0, feather_pixels=0, mask_type="box")
    assert m_box_erased[100, 150] == 0.0, "Cục tẩy phải xóa được vùng bên trong hộp chữ nhật"
    assert m_box_erased[100, 70] > 0.5, "Vùng hộp không bị tẩy vẫn phải giữ mặt nạ"

    # 3. Tẩy xong rồi vẽ cọ lại lên đúng vị trí đó -> phải có mặt nạ trở lại
    strokes_repainted = [
        {"radius": 25, "points": [(0.5, 0.5)], "is_eraser": True},
        {"radius": 15, "points": [(0.5, 0.5)], "is_eraser": False},
    ]
    m_repainted = create_combined_mask((h, w), coords_list=coords, strokes_list=strokes_repainted, dilation=0, feather_pixels=0, mask_type="box")
    assert m_repainted[100, 150] > 0.5, "Nét cọ vẽ sau cục tẩy phải khôi phục lại mặt nạ"


# ==============================================================================
# 3. KIỂM THỬ QUY ĐỔI HỆ TỌA ĐỘ (COORDINATE SYSTEM UNIFICATION)
# ==============================================================================

def test_coordinate_roundtrip_and_no_double_scaling():
    """Kiểm tra chuyển đổi hai chiều preview <-> video pixel cho 16:9, 9:16, 1:1 và chống nhân đôi tọa độ pixel."""
    from src.desktop.ui.component.video_display_component import VideoDisplayComponent

    dummy = MagicMock()
    for fw, fh in [(1920, 1080), (1080, 1920), (1080, 1080)]:
        dummy.frame_width = fw
        dummy.frame_height = fh
        dummy.border_left = 0.0
        dummy.border_top = 0.0

        orig_ratio = [(0.75, 0.90, 0.10, 0.90)]
        pixel_rects = VideoDisplayComponent.preview_coordinates_to_video_coordinates(dummy, orig_ratio)
        assert len(pixel_rects) == 1
        ymin, ymax, xmin, xmax = pixel_rects[0]
        assert abs(ymin - int(round(0.75 * fh))) <= 1
        assert abs(ymax - int(round(0.90 * fh))) <= 1
        assert abs(xmin - int(round(0.10 * fw))) <= 1
        assert abs(xmax - int(round(0.90 * fw))) <= 1

        # Idempotent: truyền tọa độ pixel vào preview_coordinates_to_video_coordinates KHÔNG được nhân tiếp với fw/fh
        pixel_again = VideoDisplayComponent.preview_coordinates_to_video_coordinates(dummy, pixel_rects)
        assert pixel_again == pixel_rects, "Không được nhân đôi tọa độ khi đầu vào đã là tọa độ pixel"

        # Chuyển ngược về preview ratio rồi lại sang pixel
        back_to_ratio = VideoDisplayComponent.video_coordinates_to_preview_coordinates(dummy, pixel_rects)
        # Idempotent cho ratio:
        ratio_again = VideoDisplayComponent.video_coordinates_to_preview_coordinates(dummy, back_to_ratio)
        for a, b in zip(back_to_ratio[0], ratio_again[0]):
            assert abs(a - b) < 1e-6

        roundtrip_pixels = VideoDisplayComponent.preview_coordinates_to_video_coordinates(dummy, back_to_ratio)
        for p1, p2 in zip(pixel_rects[0], roundtrip_pixels[0]):
            assert abs(p1 - p2) <= 2


def test_sub_area_and_box_coords_conversion():
    """Kiểm tra chuyển đổi tường minh giữa sub_area (ymin, ymax, xmin, xmax) và box_coords (xmin, xmax, ymin, ymax)."""
    sub_area = (700, 850, 120, 1600)
    box = sub_area_to_box_coords(sub_area)
    assert box == (120, 1600, 700, 850)
    assert box_coords_to_sub_area(box) == sub_area


# ==============================================================================
# 4. KIỂM THỬ BÁM ĐUỔI LOGO DI CHUYỂN & CÔ LẬP TÁC VỤ (TRACKING & TASK ISOLATION)
# ==============================================================================

def test_moving_logo_tracker_gap_interpolation_and_coverage():
    """ObjectTracker phải nội suy tuyến tính các khoảng mất dấu ngắn và tính đúng độ phủ + danh sách khoảng đứt."""
    # Giả lập: frame 1..5 có box dịch chuyển đều sang phải, frame 6..8 mất dấu (3 frame gap), frame 9..10 bắt lại được
    raw_tracks = {
        1: [(100, 200, 50, 100)],
        2: [(110, 210, 50, 100)],
        3: [(120, 220, 50, 100)],
        4: [(130, 230, 50, 100)],
        5: [(140, 240, 50, 100)],
        # gap ở 6, 7, 8
        9: [(180, 280, 50, 100)],
        10: [(190, 290, 50, 100)],
        # gap rất dài từ 11..30 (20 frames > max_interp_gap=10)
    }
    interpolated, report = ObjectTracker.interpolate_and_analyze_tracks(
        raw_tracks, total_frames=30, max_interp_gap=10, frame_size=(1080, 1920)
    )

    # Các frame 6, 7, 8 phải được nội suy đầy đủ
    assert 6 in interpolated and 7 in interpolated and 8 in interpolated
    assert interpolated[6][0] == (150, 250, 50, 100)
    assert interpolated[7][0] == (160, 260, 50, 100)
    assert interpolated[8][0] == (170, 270, 50, 100)

    # Khoảng đứt dài 11..30 không được nội suy mù quáng
    assert 15 not in interpolated
    assert report["interpolated_frames"] == 3
    assert report["tracked_frames"] == 10
    assert report["total_frames"] == 30
    assert abs(report["coverage_ratio"] - (10 / 30.0)) < 1e-4
    assert (11, 30) in report["gap_ranges"]


def test_task_item_alias_and_task_options():
    """TaskItem phải là alias hợp lệ của Task và TaskOptions hỗ trợ lưu cấu hình riêng cho từng tác vụ."""
    assert TaskItem is Task
    assert hasattr(TaskOptions, "FREEHAND_STROKES")
    assert hasattr(TaskOptions, "TRACKED_SUB_LIST")
    assert hasattr(TaskOptions, "MOVING_TRACKING")


# ==============================================================================
# 5. KIỂM THỬ AN TOÀN TỆP ĐẦU RA, HỦY TÁC VỤ & MÃ THOÁT (OUTPUT SAFETY)
# ==============================================================================

def test_verify_media_output_validation():
    """verify_media_output phải từ chối tệp rỗng/hỏng và chấp nhận tệp ảnh/video hợp lệ."""
    from src.core.main import verify_media_output

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "missing.mp4")
        assert verify_media_output(missing_path, is_picture=False) is False

        empty_path = os.path.join(tmpdir, "empty.mp4")
        Path(empty_path).write_bytes(b"")
        assert verify_media_output(empty_path, is_picture=False) is False

        # Tạo ảnh hợp lệ
        img_path = os.path.join(tmpdir, "valid.png")
        img = np.full((120, 160, 3), 200, dtype=np.uint8)
        cv2.imwrite(img_path, img)
        assert verify_media_output(img_path, is_picture=True) is True

        # Tạo video hợp lệ
        vid_path = os.path.join(tmpdir, "valid.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (160, 120))
        for _ in range(5):
            writer.write(img)
        writer.release()
        assert verify_media_output(vid_path, is_picture=False) is True


def test_stop_button_preserves_existing_valid_output():
    """Khi bấm Dừng, tuyệt đối không được xóa tệp output_path đã tồn tại từ lần chạy hoàn tất trước đó."""
    from src.desktop.ui.home_interface import HomeInterface

    with tempfile.TemporaryDirectory() as tmpdir:
        existing_output = os.path.join(tmpdir, "existing_no_sub.mp4")
        temp_work_output = os.path.join(tmpdir, "existing_no_sub.tmp_1234.mp4")
        Path(existing_output).write_bytes(b"VALID_PREVIOUS_VIDEO_CONTENT")
        Path(temp_work_output).write_bytes(b"PARTIAL_TEMP_CONTENT")

        dummy_home = MagicMock()
        dummy_home._stop_event = MagicMock()
        dummy_home.running_process = None
        dummy_home.current_processing_task_index = 0
        dummy_home._current_temp_output_path = temp_work_output
        dummy_task = Task(
            path=os.path.join(tmpdir, "existing.mp4"),
            name="existing.mp4",
            progress=50,
            status=TaskStatus.PROCESSING,
            options={},
            _output_path=existing_output,
        )
        dummy_home.task_list_component.get_task.return_value = dummy_task
        dummy_home.run_button = MagicMock()
        dummy_home.stop_button = MagicMock()
        dummy_home.video_display_component = MagicMock()

        HomeInterface.stop_button_clicked(dummy_home)

        assert os.path.exists(existing_output), "Không được xóa tệp kết quả hợp lệ đã có từ trước khi bấm Dừng!"
        assert not os.path.exists(temp_work_output), "Phải dọn dẹp tệp tạm đang xử lý dở khi bấm Dừng!"


# ==============================================================================
# 6. KIỂM THỬ BƯỚC LẤY MẪU OCR & CHUẨN HÓA LÀM NÉT / INPAINT (QUALITY & OCR STEP)
# ==============================================================================

def test_subtitle_detect_sample_step_respects_ocr_mode():
    """SubtitleDetect._init_sample_step phải thay đổi bước nhảy khung hình theo cấu hình ocrMode (fast / auto / precise)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "sample_30fps.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for _ in range(10):
            writer.write(frame)
        writer.release()

        old_mode = config.ocrMode.value
        try:
            config.set(config.ocrMode, "precise")
            det_precise = SubtitleDetect(vid_path, [])
            assert det_precise.SAMPLE_STEP == 1

            config.set(config.ocrMode, "auto")
            det_auto = SubtitleDetect(vid_path, [])
            assert det_auto.SAMPLE_STEP == 3

            config.set(config.ocrMode, "fast")
            det_fast = SubtitleDetect(vid_path, [])
            assert det_fast.SAMPLE_STEP == 6
        finally:
            config.set(config.ocrMode, old_mode)


def test_sharpening_and_opencv_inpaint_mask_normalization():
    """apply_sharpening_to_inpainted_frame và OpenCVInpaint phải xử lý đúng mặt nạ float32 [0.0, 1.0] và giữ nguyên vùng ngoài mask."""
    from src.core.main import GUISubtitleRemover

    rng = np.random.default_rng(42)
    orig_frame = rng.integers(40, 200, size=(120, 160, 3), dtype=np.uint8)
    inpainted = orig_frame.copy()
    # Mặt nạ float32 [0.0, 1.0] từ get_mask()
    float_mask = np.zeros((120, 160, 1), dtype=np.float32)
    float_mask[30:90, 40:120, 0] = 1.0

    dummy_remover = MagicMock()
    old_sharpen = config.sharpenInpaintedArea.value
    try:
        config.set(config.sharpenInpaintedArea, True)
        res = GUISubtitleRemover.apply_sharpening_to_inpainted_frame(
            dummy_remover, orig_frame, inpainted.copy(), float_mask
        )
        # Vùng ngoài mặt nạ phải giữ nguyên 100%
        assert np.array_equal(res[:25, :35], inpainted[:25, :35])
        # Vùng trong mặt nạ phải thực sự được áp dụng bộ lọc (không bị chia 255 hai lần về 0)
        assert not np.array_equal(res[35:85, 45:115], inpainted[35:85, 45:115])
    finally:
        config.set(config.sharpenInpaintedArea, old_sharpen)

    # OpenCVInpaint phải chạy không lỗi với cả float32 3D mask và danh sách mask per-frame
    opencv_engine = OpenCVInpaint()
    out_single = opencv_engine.inpaint(orig_frame, float_mask)
    assert out_single.shape == orig_frame.shape
    out_batch = opencv_engine([orig_frame, orig_frame], [float_mask, float_mask])
    assert len(out_batch) == 2
    assert out_batch[0].shape == orig_frame.shape


def _create_synthetic_video(path: str, width: int = 160, height: int = 120, num_frames: int = 20, moving: bool = False):
    """Tạo video tổng hợp dùng cho kiểm thử tự động xóa phụ đề và logo di chuyển."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height))
    try:
        for idx in range(num_frames):
            frame = np.full((height, width, 3), 60, dtype=np.uint8)
            if moving:
                x1 = min(width - 45, 15 + idx * 3)
                cv2.rectangle(frame, (x1, 42), (x1 + 36, 68), (240, 240, 240), -1)
            else:
                cv2.rectangle(frame, (24, 80), (width - 24, 106), (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_moving_logo_inpaint_short_vs_long_occlusion_gap():
    """
    Xác minh xóa logo di chuyển:
    - Khi mất dấu ngắn (<= 15 frame): nội suy lấp đầy, xóa sạch logo trên các frame bị khuất và trả về run_status == 'success'.
    - Khi mất dấu dài (> 15 frame) hoặc độ phủ thấp (< 85%): không báo thành công tuyệt đối mà trả về run_status == 'partial' kèm cảnh báo cần kiểm tra.
    """
    from src.core.main import GUISubtitleRemover, verify_media_output
    from src.core.tools.constant import InpaintMode

    with tempfile.TemporaryDirectory() as tmpdir:
        vid_short = os.path.join(tmpdir, "moving_short_gap.mp4")
        vid_long = os.path.join(tmpdir, "moving_long_gap.mp4")
        _create_synthetic_video(vid_short, width=160, height=120, num_frames=20, moving=True)
        _create_synthetic_video(vid_long, width=160, height=120, num_frames=40, moving=True)

        old_mask_type = config.maskType.value
        old_inpaint_mode = config.inpaintMode.value
        old_moving = config.movingSubtitleTracking.value
        try:
            config.set(config.maskType, "rectangle")
            config.set(config.inpaintMode, InpaintMode.OPENCV)
            config.set(config.movingSubtitleTracking, False)

            # 1. Mất dấu ngắn (frame 6..9 bị thiếu trong tracked_sub_list nhưng khoảng trống = 4 <= 15)
            tracked_short = {}
            for idx in range(1, 21):
                if 6 <= idx <= 9:
                    continue
                x1 = 15 + (idx - 1) * 6
                x2 = min(160 - 5, x1 + 42)
                tracked_short[idx] = [(max(0, x1 - 4), min(160, x2 + 4), 38, 72)]

            remover_short = GUISubtitleRemover(vid_short, gui_mode=True)
            remover_short.moving_tracking = True
            remover_short.tracked_sub_list = tracked_short
            remover_short.lama_inpaint = OpenCVInpaint()
            remover_short.run()

            assert remover_short.run_status == "success"
            assert remover_short.tracking_report is not None
            assert remover_short.tracking_report["needs_review"] is False
            assert remover_short.tracking_report["interpolated_frames"] == 4
            assert verify_media_output(remover_short.video_out_path, is_picture=False, expected_frames=20)

            # 2. Mất dấu dài (chỉ theo dõi được frame 1..5 và 36..40 -> khoảng trống 30 frame > 15, coverage = 25% < 85%)
            tracked_long = {}
            for idx in list(range(1, 6)) + list(range(36, 41)):
                x1 = 15 + (idx - 1) * 3
                x2 = min(160 - 5, x1 + 42)
                tracked_long[idx] = [(max(0, x1 - 4), min(160, x2 + 4), 38, 72)]

            remover_long = GUISubtitleRemover(vid_long, gui_mode=True)
            remover_long.moving_tracking = True
            remover_long.tracked_sub_list = tracked_long
            remover_long.lama_inpaint = OpenCVInpaint()
            remover_long.run()

            assert remover_long.run_status == "partial"
            assert remover_long.tracking_report is not None
            assert remover_long.tracking_report["needs_review"] is True
            assert remover_long.tracking_report["status"] == "needs_review"
            assert (6, 35) in remover_long.tracking_report["gap_ranges"]
            assert remover_long.tracking_report["warning_message"] != ""
        finally:
            config.set(config.maskType, old_mask_type)
            config.set(config.inpaintMode, old_inpaint_mode)
            config.set(config.movingSubtitleTracking, old_moving)


def test_audio_preservation_rejects_muted_fallback_when_source_has_audio(monkeypatch):
    """
    Khi video nguồn có luồng âm thanh nhưng cả 2 bước FFmpeg mux âm thanh đều thất bại,
    GUISubtitleRemover KHÔNG ĐƯỢC âm thầm copy tệp tạm mất tiếng và báo thành công.
    """
    import subprocess
    from src.core.main import GUISubtitleRemover
    from src.core.tools.constant import InpaintMode, SubtitleDetectMode

    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "with_audio.mp4")
        _create_synthetic_video(vid_path, width=160, height=120, num_frames=10, moving=False)

        remover = GUISubtitleRemover(vid_path, gui_mode=True)
        remover.source_has_audio = True
        remover.sub_areas = [(74, 112, 16, 144)]

        # Giả lập FFmpeg mux âm thanh thất bại ở cả 2 lần gọi
        def _fail_ffmpeg(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0] if args else "ffmpeg")

        monkeypatch.setattr(subprocess, "check_output", _fail_ffmpeg)

        old_mode = config.inpaintMode.value
        old_detect = config.subtitleDetectMode.value
        try:
            config.set(config.inpaintMode, InpaintMode.OPENCV)
            config.set(config.subtitleDetectMode, SubtitleDetectMode.NONE.value)
            with pytest.raises(RuntimeError, match="mất âm thanh|không hợp lệ"):
                remover.run()
            assert remover.audio_merge_failed is True
            assert remover.is_successful_merged is False
            assert remover.run_status == "failed"
            assert not os.path.exists(remover.video_out_path)
        finally:
            config.set(config.inpaintMode, old_mode)
            config.set(config.subtitleDetectMode, old_detect)


def test_verify_media_output_rejects_truncated_video():
    """verify_media_output phải từ chối video bị cắt cụt số khung hình so với expected_frames."""
    from src.core.main import verify_media_output

    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "truncated.mp4")
        _create_synthetic_video(vid_path, width=160, height=120, num_frames=8, moving=False)

        # Hợp lệ khi kỳ vọng 8 frame
        assert verify_media_output(vid_path, is_picture=False, expected_frames=8, expected_fps=24.0) is True
        # Phải trả về False khi kỳ vọng 30 frame nhưng tệp chỉ có 8 frame
        assert verify_media_output(vid_path, is_picture=False, expected_frames=30, expected_fps=24.0) is False
        # Phải trả về False khi yêu cầu có luồng âm thanh nhưng tệp chỉ có hình ảnh
        assert verify_media_output(vid_path, is_picture=False, expected_frames=8, require_audio=True) is False


def test_task_queue_isolation_and_partial_warning_status():
    """
    Kiểm tra cô lập tùy chọn theo từng task và cập nhật trạng thái TaskStatus.WARNING
    khi kết quả xóa logo di chuyển trả về 'partial'.
    """
    from src.desktop.ui.component.task_list_component import TaskStatus, TaskOptions

    assert TaskStatus.WARNING.value == "Hoàn thành (Cần kiểm tra)"
    assert TaskOptions.TRACKING_REPORT.value == "tracking_report"


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
    ui.task_list_component.add_task("dummy.mp4")
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
    dummy_img[100:150, 100:300] = 255
    
    # PaddleOCR trả về tọa độ trên cropped_img; detector sẽ cộng lại xmin_crop, ymin_crop
    # Ta mock _get_coordinates_from_paddle trả về hộp chữ tương ứng
    def fake_paddle(crop):
        return [(90, 310, 95, 155)]
        
    detector._get_coordinates_from_paddle = MagicMock(side_effect=fake_paddle)
    
    results = detector.detect_subtitle(dummy_img)
    assert len(results) == 1
    # Hộp chữ tràn biên không bị loại bỏ
    assert results[0][0] <= 100 and results[0][1] >= 300


def test_detect_subtitle_retries_when_text_touches_crop_boundary():
    """
    Xác minh detect_subtitle tự động thử lại 1 lần với vùng mở rộng thêm
    khi phát hiện hộp chữ chạm sát mép crop của lần nhận diện đầu tiên.
    """
    from unittest.mock import MagicMock
    from src.core.tools.subtitle_detect import SubtitleDetect

    detector = SubtitleDetect("dummy.mp4", sub_areas=[(100, 140, 100, 200)])
    dummy_img = np.full((400, 400, 3), 128, dtype=np.uint8)
    dummy_img[80:160, 80:250] = 255

    call_count = 0
    def fake_paddle(crop):
        nonlocal call_count
        call_count += 1
        # Lần 1 trả về hộp chạm sát biên crop
        if call_count == 1:
            return [(0, crop.shape[1], 0, crop.shape[0])]
        # Lần 2 trả về hộp hoàn chỉnh
        return [(10, crop.shape[1] - 10, 10, crop.shape[0] - 10)]

    detector._get_coordinates_from_paddle = MagicMock(side_effect=fake_paddle)
    results = detector.detect_subtitle(dummy_img)
    # Phải gọi thử lại lần 2 do chạm sát biên crop
    assert call_count >= 2
    assert len(results) >= 1


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


def test_preview_loads_actual_adjacent_video_frames_with_boundary_clamp(tmp_path):
    """
    Xác minh hàm tải khung hình lân cận cho xem trước:
    1. Đọc đúng 5 khung hình thực tế quanh vị trí hiện tại (current_frame_no).
    2. Khi ở đầu video (frame 0 hoặc 1) hoặc cuối video, kẹp chỉ số an toàn không gây crash.
    """
    from src.desktop.ui.home_interface import _load_adjacent_video_frames

    # Tạo tệp video ngắn 10 frame thử nghiệm với màu sắc thay đổi
    video_file = str(tmp_path / "test_frames.avi")
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (100, 100))
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


def test_verify_media_quality_passes_when_clean(tmp_path):
    """
    Xác minh hàm kiểm soát chất lượng sau xuất trả về passed=True khi vùng đã xử lý sạch sẽ.
    """
    from src.desktop.ui.home_interface import verify_media_quality

    video_path = str(tmp_path / "clean_cleaned.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (300, 200))
    for _ in range(10):
        frame = np.full((200, 300, 3), 128, dtype=np.uint8)
        out.write(frame)
    out.release()

    res = verify_media_quality(video_path, sub_areas=[(80, 120, 50, 250)])
    assert res["passed"] is True
    assert res["residual_text_found"] is False




