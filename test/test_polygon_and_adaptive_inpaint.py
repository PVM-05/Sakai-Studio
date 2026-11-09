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


def test_measure_text_stroke_width_and_adaptive_dilation():
    """Kiểm tra thuật toán đo độ dày nét chữ và tính toán độ giãn nở thích ứng."""
    # Tạo ảnh trắng nền đen có nét chữ mảnh (độ dày 2px)
    thin_img = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.putText(thin_img, "TEST", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), thickness=2)
    thin_stroke = measure_text_stroke_width(thin_img)
    thin_dilation = calculate_adaptive_stroke_dilation(thin_img, base_dilation=8)

    # Tạo ảnh có nét chữ dày (độ dày 8px)
    thick_img = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.putText(thick_img, "TEST", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), thickness=8)
    thick_stroke = measure_text_stroke_width(thick_img)
    thick_dilation = calculate_adaptive_stroke_dilation(thick_img, base_dilation=8)

    # Nét dày phải có stroke width lớn hơn nét mảnh
    assert thick_stroke > thin_stroke
    # Độ giãn nở thích ứng của nét dày phải lớn hơn nét mảnh
    assert thick_dilation >= thin_dilation
    assert 3 <= thin_dilation <= 16
    assert 3 <= thick_dilation <= 16


def test_create_polygon_mask_geometry():
    """Kiểm tra tạo mặt nạ đa giác không làm phình diện tích như hộp chữ nhật."""
    h, w = 100, 100
    # Đa giác tam giác hoặc tứ giác nghiêng
    poly = np.array([[20, 20], [80, 20], [60, 80], [20, 80]], dtype=np.int32)
    mask = create_polygon_mask((h, w), [poly], dilation=2, feather_pixels=0)

    assert mask.shape == (h, w)
    assert mask.dtype == np.float32
    # Điểm bên trong đa giác phải bằng 1.0
    assert mask[40, 40] == 1.0
    # Điểm ở góc ngoài (0, 0) phải bằng 0.0
    assert mask[0, 0] == 0.0
    # Điểm góc ngoài (90, 90) phải bằng 0.0
    assert mask[90, 90] == 0.0


def test_create_combined_mask_with_polygon_mode():
    """Kiểm tra create_combined_mask hỗ trợ kiểu mặt nạ đa giác polygon."""
    h, w = 100, 100
    poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
    mask = create_combined_mask((h, w), coords_list=[poly], mask_type="polygon", feather_pixels=2)

    assert mask.shape == (h, w)
    assert np.any(mask > 0)
    assert np.all((mask >= 0.0) & (mask <= 1.0))


def test_residual_ocr_verification_blank_frame():
    """Kiểm tra residual_ocr_verification xử lý an toàn khung hình sạch và khung hình rỗng."""
    # Khung hình rỗng / None
    res_none = residual_ocr_verification(None)
    assert res_none["has_residual"] is False
    assert res_none["char_count"] == 0

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = residual_ocr_verification(empty_frame)
    assert res_empty["has_residual"] is False

    # Khung hình nền đen không có chữ
    blank_frame = np.zeros((100, 300, 3), dtype=np.uint8)
    res_blank = residual_ocr_verification(blank_frame)
    assert res_blank["has_residual"] is False
    assert res_blank["char_count"] == 0


def test_retry_config_patch_for_gate_steps():
    """Kiểm tra thang nâng cấp tự động thử lại khi chất lượng không đạt."""
    class FakeConfig:
        def __init__(self):
            self.mask_dilate_px = 8
            self.mask_feather_px = 4
            self.mode = "sttn"

    config = FakeConfig()

    # Bước 1: Tăng độ giãn nở mặt nạ
    gate_dilation = {
        "status": STATUS_REVIEW,
        "ladderStep": LADDER_INCREASE_DILATION,
        "reasons": [{"metric": "residual_text", "ladder": LADDER_INCREASE_DILATION}],
    }
    patch1 = retry_config_patch_for_gate(gate_dilation, config)
    assert patch1.get("mask_dilate_px") == 12
    assert patch1.get("quality_report") is True

    # Bước 2: Chuyển đổi mô hình phục hồi nền thay thế
    gate_inpaint = {
        "status": STATUS_REVIEW,
        "ladderStep": LADDER_ALTERNATE_INPAINTER,
        "reasons": [{"metric": "residual_text", "ladder": LADDER_ALTERNATE_INPAINTER}],
    }
    patch2 = retry_config_patch_for_gate(gate_inpaint, config)
    assert patch2.get("mode") == "lama"
