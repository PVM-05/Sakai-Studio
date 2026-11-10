import numpy as np
import cv2
import pytest
from src.ai_engines.detection import (
    measure_text_stroke_width,
    calculate_adaptive_stroke_dilation,
)
from src.core.tools.inpaint_tools import (
    create_polygon_mask,
    create_combined_mask,
)
from src.core.quality import residual_ocr_verification
from src.core.quality_gate import (
    retry_config_patch_for_gate,
    STATUS_REVIEW,
    LADDER_INCREASE_DILATION,
    LADDER_ALTERNATE_INPAINTER,
)
from src.core.config import config


def test_measure_text_stroke_width_and_adaptive_dilation():
    """Kiểm tra thuật toán đo độ dày nét chữ và tính toán độ giãn nở thích ứng."""
    thin_img = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.putText(thin_img, "TEST", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), thickness=2)
    thin_stroke = measure_text_stroke_width(thin_img)
    thin_dilation = calculate_adaptive_stroke_dilation(thin_img, base_dilation=8)

    thick_img = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.putText(thick_img, "TEST", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), thickness=8)
    thick_stroke = measure_text_stroke_width(thick_img)
    thick_dilation = calculate_adaptive_stroke_dilation(thick_img, base_dilation=8)

    assert thick_stroke > thin_stroke
    assert thick_dilation >= thin_dilation
    assert 3 <= thin_dilation <= 16
    assert 3 <= thick_dilation <= 16


def test_create_polygon_mask_geometry():
    """Kiểm tra tạo mặt nạ đa giác không làm phình diện tích như hộp chữ nhật."""
    h, w = 100, 100
    poly = np.array([[20, 20], [80, 20], [60, 80], [20, 80]], dtype=np.int32)
    mask = create_polygon_mask((h, w), [poly], dilation=2, feather_pixels=0)

    assert mask.shape == (h, w)
    assert mask.dtype == np.float32
    assert mask[40, 40] == 1.0
    assert mask[0, 0] == 0.0
    assert mask[90, 90] == 0.0


def test_create_polygon_mask_accepts_boxes_and_polygons():
    """Kiểm tra create_polygon_mask chấp nhận cả hộp 4 phần tử lẫn đa giác điểm."""
    h, w = 120, 120
    box_coord = (10, 60, 20, 80)  # xmin, xmax, ymin, ymax
    mask = create_polygon_mask((h, w), [box_coord], dilation=0, feather_pixels=0)
    assert mask.shape == (h, w)
    assert mask[40, 30] == 1.0
    assert mask[0, 0] == 0.0


def test_create_combined_mask_with_polygon_mode():
    """Kiểm tra create_combined_mask hỗ trợ kiểu mặt nạ đa giác polygon."""
    h, w = 100, 100
    poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
    mask = create_combined_mask((h, w), coords_list=[poly], mask_type="polygon", feather_pixels=2)

    assert mask.shape == (h, w)
    assert np.any(mask > 0)
    assert np.all((mask >= 0.0) & (mask <= 1.0))


def test_residual_ocr_verification_blank_frame():
    """Kiểm tra residual_ocr_verification xử lý an toàn khung hình sạch và khung hình rỗng (ca âm tính)."""
    res_none = residual_ocr_verification(None)
    assert res_none["has_residual"] is False
    assert res_none["char_count"] == 0

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = residual_ocr_verification(empty_frame)
    assert res_empty["has_residual"] is False

    blank_frame = np.zeros((100, 300, 3), dtype=np.uint8)
    res_blank = residual_ocr_verification(blank_frame)
    assert res_blank["has_residual"] is False
    assert res_blank["char_count"] == 0


def test_residual_ocr_verification_positive_case():
    """Kiểm tra residual_ocr_verification phát hiện chữ tồn dư trên khung hình (ca dương tính)."""
    # Tạo ảnh trắng có dòng chữ đen sắc nét rõ ràng
    frame_with_text = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(
        frame_with_text,
        "PHU DE SOT",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        thickness=3,
    )
    res = residual_ocr_verification(frame_with_text, threshold_chars=3)
    # Xác nhận RapidOCR phát hiện ra chữ sót
    assert res["has_residual"] is True
    assert res["char_count"] >= 3
    assert res["confidence"] >= 0.40


def test_retry_config_patch_reads_sakai_config_mask_dilation():
    """Kiểm tra retry_config_patch_for_gate đọc đúng thuộc tính maskDilation của Sakai config thay vì tính từ 8."""
    # Giả định người dùng đã đặt maskDilation là 14 trên giao diện
    orig_dilation = config.maskDilation.value
    try:
        config.maskDilation.value = 14
        gate_dilation = {
            "status": STATUS_REVIEW,
            "ladderStep": LADDER_INCREASE_DILATION,
            "reasons": [{"metric": "residual_text", "ladder": LADDER_INCREASE_DILATION}],
        }
        patch = retry_config_patch_for_gate(gate_dilation, config)
        # 14 + 4 = 18 (không bị reset về 8 + 4 = 12)
        assert patch.get("mask_dilate_px") == 18
    finally:
        config.maskDilation.value = orig_dilation


def test_retry_config_patch_for_gate_steps():
    """Kiểm tra thang nâng cấp tự động thử lại khi chất lượng không đạt."""
    class FakeConfig:
        def __init__(self):
            self.mask_dilate_px = 8
            self.mask_feather_px = 4
            self.mode = "sttn"

    fake_cfg = FakeConfig()

    gate_dilation = {
        "status": STATUS_REVIEW,
        "ladderStep": LADDER_INCREASE_DILATION,
        "reasons": [{"metric": "residual_text", "ladder": LADDER_INCREASE_DILATION}],
    }
    patch1 = retry_config_patch_for_gate(gate_dilation, fake_cfg)
    assert patch1.get("mask_dilate_px") == 12
    assert patch1.get("quality_report") is True

    gate_inpaint = {
        "status": STATUS_REVIEW,
        "ladderStep": LADDER_ALTERNATE_INPAINTER,
        "reasons": [{"metric": "residual_text", "ladder": LADDER_ALTERNATE_INPAINTER}],
    }
    patch2 = retry_config_patch_for_gate(gate_inpaint, fake_cfg)
    assert patch2.get("mode") == "lama"


def test_are_similar_supports_polygons():
    """Kiểm tra hàm are_similar so khớp độ trùng khớp cho cả mảng đa giác (N, 2) và hộp 4 phần tử."""
    from src.core.tools.subtitle_detect import SubtitleDetect

    # 2 đa giác gần trùng khớp
    poly1 = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.int32)
    poly2 = np.array([[12, 10], [52, 10], [52, 30], [12, 30]], dtype=np.int32)
    assert SubtitleDetect.are_similar(poly1, poly2) is True

    # 2 đa giác cách xa nhau
    poly3 = np.array([[100, 100], [150, 100], [150, 130], [100, 130]], dtype=np.int32)
    assert SubtitleDetect.are_similar(poly1, poly3) is False

    # So khớp chéo giữa mảng đa giác và hộp (xmin, xmax, ymin, ymax)
    box1 = (10, 50, 10, 30)
    assert SubtitleDetect.are_similar(poly1, box1) is True


def test_gui_subtitle_remover_reuses_cached_sub_list():
    """Kiểm tra GUISubtitleRemover tái sử dụng bản đồ phụ đề cached_sub_list khi retry mà không chạy lại OCR."""
    from src.core.main import GUISubtitleRemover
    from unittest.mock import MagicMock

    cached_data = [
        {"frame_no": 1, "box_list": [np.array([[10, 10], [50, 10], [50, 30], [10, 30]])]},
        {"frame_no": 2, "box_list": [np.array([[10, 10], [50, 10], [50, 30], [10, 30]])]},
    ]
    remover = GUISubtitleRemover("dummy.mp4", cached_sub_list=cached_data)
    mock_detector = MagicMock()
    mock_detector.find_subtitle_frame_no.side_effect = RuntimeError("Không được phép gọi dò phụ đề khi đã có cache")

    # Gọi _get_subtitle_map phải dùng cache và không đụng vào find_subtitle_frame_no
    sub_map = remover._get_subtitle_map(mock_detector)
    assert len(sub_map) == 2
    assert remover.last_sub_list == cached_data
    mock_detector.find_subtitle_frame_no.assert_not_called()
