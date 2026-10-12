import cv2
import numpy as np


class OpenCVInpaint:

    def __init__(self):
        pass

    @staticmethod
    def _normalize_mask(mask, target_shape=None):
        if mask is None:
            if target_shape is not None:
                return np.zeros(target_shape[:2], dtype=np.uint8)
            return np.zeros((1, 1), dtype=np.uint8)
        m = np.asarray(mask)
        if m.ndim == 3:
            if m.shape[2] == 1:
                m = m[:, :, 0]
            else:
                m = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if m.dtype != np.uint8 or m.max() <= 1.0:
            m = ((m > 0) * 255).astype(np.uint8)
        else:
            m = np.where(m > 0, 255, 0).astype(np.uint8)
        return m

    def inpaint(self, frame, mask):
        norm_mask = self._normalize_mask(mask, target_shape=frame.shape)
        if not np.any(norm_mask > 0):
            return frame.copy()
        return cv2.inpaint(frame, norm_mask, 3, cv2.INPAINT_TELEA)

    def __call__(self, frames, mask):
        comp = []
        if isinstance(mask, (list, tuple)):
            for idx, frame in enumerate(frames):
                m = mask[idx] if idx < len(mask) else mask[-1]
                comp.append(self.inpaint(frame, m))
        else:
            for frame in frames:
                comp.append(self.inpaint(frame, mask))
        return comp