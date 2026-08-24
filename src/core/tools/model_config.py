import os
from pathlib import Path
from src.core.config import config, BASE_DIR
from src.core.tools.common_tools import merge_big_file_if_not_exists
from src.core.tools.constant import SubtitleDetectMode

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE.value: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER.value: "PP-OCRv5_server_det",
    SubtitleDetectMode.RAPID_OCR.value: "RapidOCR",
    SubtitleDetectMode.PADDLE_OCR.value: "PaddleOCR",
}

class ModelConfig:
    def __init__(self):
        models_dir = os.path.join(str(Path(BASE_DIR).parent), 'ai_engines', 'models')
        self.LAMA_MODEL_DIR = os.path.join(models_dir, 'big-lama')
        self.STTN_AUTO_MODEL_PATH = os.path.join(models_dir, 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(models_dir, 'sttn-det', 'sttn.pth')
        self.PROPAINTER_MODEL_DIR = os.path.join(models_dir, 'propainter')
        paddlex_official = os.path.join(models_dir, 'paddlex', 'official_models')
        mode = config.subtitleDetectMode.value
        mode_val = mode.value if hasattr(mode, 'value') else str(mode)
        
        if mode_val == SubtitleDetectMode.PP_OCRv5_MOBILE.value:
            self.DET_MODEL_DIR = os.path.join(paddlex_official, 'PP-OCRv5_mobile_det')
            self.REC_MODEL_DIR = os.path.join(paddlex_official, 'PP-OCRv5_mobile_rec')
        elif mode_val == SubtitleDetectMode.PP_OCRv5_SERVER.value:
            self.DET_MODEL_DIR = os.path.join(paddlex_official, 'PP-OCRv5_server_det')
            self.REC_MODEL_DIR = os.path.join(paddlex_official, 'PP-OCRv5_server_rec')
        elif mode_val == SubtitleDetectMode.PADDLE_OCR.value:
            # PaddleOCR cổ điển: để None, PaddleOCR tự tải model mặc định
            self.DET_MODEL_DIR = None
            self.REC_MODEL_DIR = None
        elif mode_val == SubtitleDetectMode.RAPID_OCR.value:
            self.DET_MODEL_DIR = None
            self.REC_MODEL_DIR = None
        else:
            raise ValueError(f"Invalid subtitle detect mode: {mode_val}")
        self.DET_MODEL_NAME = _MODEL_NAME_MAP.get(mode_val, "")

        merge_big_file_if_not_exists(self.LAMA_MODEL_DIR, 'bit-lama.pt')
        merge_big_file_if_not_exists(self.PROPAINTER_MODEL_DIR, 'ProPainter.pth')
