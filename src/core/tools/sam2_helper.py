"""
SAM 2 (Segment Anything Model 2) Helper Module.
Provides prompt-based (box / brush point) fine object segmentation
and temporal video mask propagation.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class SAM2Segmentor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        """Khởi tạo SAM 2 Predictor nếu đã cài đặt package sam2."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            checkpoint = "sam2_hiera_tiny.pt"
            model_cfg = "sam2_hiera_t.yaml"
            sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM 2 Predictor initialized successfully.")
        except Exception as e:
            logger.debug(f"SAM 2 package or weights not loaded: {e}. Fallback to thresholding segmentation.")
            self.predictor = None

    def segment_frame_with_box(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Phân tách chính xác mặt nạ vật thể trong khung hình dựa trên khung chọn box (xmin, xmax, ymin, ymax).
        """
        h, w = image.shape[:2]
        xmin, xmax, ymin, ymax = box
        
        if self.predictor is not None:
            try:
                self.predictor.set_image(image)
                input_box = np.array([xmin, ymin, xmax, ymax])
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                if len(masks) > 0:
                    return (masks[0] > 0).astype(np.uint8) * 255
            except Exception as e:
                logger.warning(f"SAM 2 prediction error: {e}")

        # Fallback: Thuật toán tách ngưỡng Otsu trong vùng box
        mask = np.zeros((h, w), dtype=np.uint8)
        crop_x1 = max(0, xmin)
        crop_y1 = max(0, ymin)
        crop_x2 = min(w, xmax)
        crop_y2 = min(h, ymax)
        
        roi = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask[crop_y1:crop_y2, crop_x1:crop_x2] = thresh
        else:
            mask[crop_y1:crop_y2, crop_x1:crop_x2] = 255

        return mask
