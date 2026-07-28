# -*- coding: utf-8 -*-
"""
Unified Video OCR Engine cho Sakai Studio.

Module này gom toàn bộ logic OCR đang bị lặp 4 lần ở 4 file UI
vào 1 class duy nhất `VideoOcrEngine`, đảm bảo:
- RapidOCR frame extraction + SequenceMatcher clustering
- Whisper AI speech-to-text fallback (khi không phát hiện chữ trên màn hình)
- TypoMap filter (lọc quảng cáo, sửa lỗi chính tả OCR)
- Safe sub_areas normalization (xử lý mọi format)
- Progress callback support
- Voice separation preprocessing (tách giọng nói cho Whisper chính xác hơn)
"""

from __future__ import annotations

import os
import re
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """Một đoạn phụ đề được trích xuất từ video."""
    index: int
    start_frame: int
    end_frame: int
    start_time: str  # SRT timestamp format: HH:MM:SS,mmm
    end_time: str
    text: str
    source: str = "ocr"  # "ocr" hoặc "whisper"


def frame_to_srt_timestamp(frame: int, fps: float) -> str:
    """Chuyển đổi số khung hình thành timestamp SRT."""
    if fps <= 0:
        fps = 30.0
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int((total_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def apply_typo_map(text: str) -> str:
    """Lọc và thay thế văn bản theo quy tắc trong tệp config/typoMap.json."""
    if not text:
        return ""
    _base_dir = Path(__file__).parent.parent
    typo_file = str(_base_dir / "config" / "typoMap.json")
    if os.path.exists(typo_file):
        try:
            with open(typo_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for k, v in mapping.items():
                if k:
                    text = text.replace(k, v)
        except Exception:
            pass
    return text.strip()


class VideoOcrEngine:
    """
    Unified OCR Engine dùng chung cho tất cả các tab UI.
    
    Tính năng:
    - RapidOCR frame-by-frame subtitle detection
    - SequenceMatcher clustering để gộp các khung hình có cùng text
    - Whisper AI speech-to-text fallback
    - TypoMap post-processing filter
    - Voice separation preprocessing cho Whisper
    - Safe sub_areas normalization
    
    Usage:
        engine = VideoOcrEngine(ocr_mode="auto", use_whisper_fallback=True, use_typo_map=True)
        segments = engine.extract_subtitles(
            video_path="video.mp4",
            sub_areas=[(0.7, 0.95, 0.05, 0.95)],
            progress_callback=lambda pct, msg: print(f"{pct}% - {msg}")
        )
    """

    def __init__(
        self,
        ocr_mode: str = "auto",
        ocr_lang: str = "auto",
        use_typo_map: bool = True,
        use_whisper_fallback: bool = True,
        use_voice_separation: bool = False,
        similarity_threshold: float = 0.7,
        default_crop_ratio: float = 0.65,
    ):
        self.ocr_mode = ocr_mode
        self.ocr_lang = ocr_lang
        self.use_typo_map = use_typo_map
        self.use_whisper_fallback = use_whisper_fallback
        self.use_voice_separation = use_voice_separation
        self.similarity_threshold = similarity_threshold
        self.default_crop_ratio = default_crop_ratio
        self._is_stopped = False

    def stop(self):
        """Dừng quá trình trích xuất."""
        self._is_stopped = True

    def reset(self):
        """Reset trạng thái để chạy lại."""
        self._is_stopped = False

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def extract_subtitles(
        self,
        video_path: str,
        sub_areas: Optional[Union[list, tuple]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[SubtitleSegment]:
        """
        Trích xuất phụ đề từ video bằng OCR + optional Whisper fallback.
        
        Args:
            video_path: Đường dẫn tới tệp video.
            sub_areas: Vùng quét phụ đề. Hỗ trợ mọi format:
                - None / [] → Quét vùng mặc định (bottom 35%)
                - (ymin, ymax, xmin, xmax) → Single tuple (normalized 0..1 hoặc pixel)
                - [(ymin, ymax, xmin, xmax)] → List of tuples
            progress_callback: Callback(percentage: float, message: str)
            
        Returns:
            List[SubtitleSegment]: Danh sách các đoạn phụ đề phát hiện được.
        """
        self.reset()
        
        if not video_path or not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        # Normalize sub_areas
        ymin, ymax, xmin, xmax = self._normalize_sub_areas(sub_areas, width, height)

        # Load RapidOCR
        ocr = self._load_ocr_engine()

        # Determine sample step based on mode
        sample_step = self._get_sample_step(fps)

        # Phase 1: OCR Frame Extraction
        if progress_callback:
            progress_callback(0.0, "Đang khởi động nhận diện phụ đề OCR...")

        items = self._ocr_frame_loop(cap, ocr, fps, total_frames, width, height,
                                      ymin, ymax, xmin, xmax, sample_step, progress_callback)
        cap.release()

        if self._is_stopped:
            return []

        # Phase 2: Whisper AI Fallback (nếu OCR không phát hiện được gì)
        if not items and self.use_whisper_fallback:
            if progress_callback:
                progress_callback(50.0, "Không phát hiện chữ trên màn hình. Đang dùng Whisper AI trích xuất từ giọng nói...")
            whisper_items = self._run_whisper_fallback(video_path, fps)
            if whisper_items:
                items = whisper_items

        if self._is_stopped:
            return []

        # Phase 3: Build SubtitleSegment list
        segments: List[SubtitleSegment] = []
        for idx, item in enumerate(items, 1):
            sf = item["start_frame"]
            ef = item["end_frame"]
            segments.append(SubtitleSegment(
                index=idx,
                start_frame=sf,
                end_frame=ef,
                start_time=frame_to_srt_timestamp(sf, fps),
                end_time=frame_to_srt_timestamp(ef, fps),
                text=item["text"],
                source=item.get("source", "ocr"),
            ))

        if progress_callback:
            progress_callback(100.0, f"Hoàn thành! Phát hiện {len(segments)} câu phụ đề.")

        return segments

    # =========================================================================
    # PRIVATE: sub_areas normalization
    # =========================================================================

    def _normalize_sub_areas(self, sub_areas, width: int, height: int) -> tuple:
        """
        Chuẩn hóa mọi format sub_areas thành (ymin, ymax, xmin, xmax) pixel.
        
        Xử lý an toàn:
        - None / [] → Fallback vùng quét mặc định
        - (y1, y2, x1, x2) → Single tuple
        - [(y1, y2, x1, x2)] → List of tuples
        - Normalized 0..1 → Convert sang pixel
        - Pixel values → Dùng trực tiếp
        """
        if not sub_areas or (isinstance(sub_areas, (list, tuple)) and len(sub_areas) == 0):
            return int(height * self.default_crop_ratio), height, 0, width

        # Detect format
        target = None
        if isinstance(sub_areas, (list, tuple)):
            if len(sub_areas) == 4 and all(isinstance(v, (int, float)) for v in sub_areas):
                # Single flat tuple/list: (y1, y2, x1, x2)
                target = sub_areas
            elif len(sub_areas) > 0 and isinstance(sub_areas[0], (list, tuple)) and len(sub_areas[0]) == 4:
                # List of tuples: [(y1, y2, x1, x2)]
                target = sub_areas[0]

        if target is None:
            return int(height * self.default_crop_ratio), height, 0, width

        y1, y2, x1, x2 = target
        ymin = int(y1 * height) if isinstance(y1, float) and 0.0 <= y1 <= 1.0 else int(y1)
        ymax = int(y2 * height) if isinstance(y2, float) and 0.0 <= y2 <= 1.0 else int(y2)
        xmin = int(x1 * width) if isinstance(x1, float) and 0.0 <= x1 <= 1.0 else int(x1)
        xmax = int(x2 * width) if isinstance(x2, float) and 0.0 <= x2 <= 1.0 else int(x2)

        # Normalize: ensure min < max
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)

        # Clamp to frame bounds
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))

        if ymax <= ymin or xmax <= xmin:
            return int(height * self.default_crop_ratio), height, 0, width

        return ymin, ymax, xmin, xmax

    # =========================================================================
    # PRIVATE: OCR Engine Loading
    # =========================================================================

    def _load_ocr_engine(self):
        """Load RapidOCR engine với fallback."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            return RapidOCR()
        except ImportError:
            try:
                from rapidocr import RapidOCR
                return RapidOCR()
            except ImportError:
                logger.warning("RapidOCR không khả dụng. OCR sẽ bị bỏ qua.")
                return None

    def _get_sample_step(self, fps: float) -> int:
        """Tính bước nhảy mẫu theo chế độ OCR."""
        if self.ocr_mode == "fast":
            return max(6, int(fps / 2))
        elif self.ocr_mode == "precise":
            return max(2, int(fps / 5))
        else:  # auto
            return max(4, int(fps / 3))

    # =========================================================================
    # PRIVATE: OCR Frame Loop
    # =========================================================================

    def _ocr_frame_loop(
        self, cap, ocr, fps, total_frames, width, height,
        ymin, ymax, xmin, xmax, sample_step,
        progress_callback
    ) -> list:
        """Vòng lặp OCR chính trên các khung hình video."""
        current_frame = 0
        active_text = ""
        start_frame = 0
        end_frame = 0
        items = []

        while cap.isOpened() and not self._is_stopped:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

            if current_frame % sample_step != 0:
                continue

            if total_frames > 0 and progress_callback:
                pct = (current_frame / total_frames) * 95  # Reserve 5% for post-processing
                progress_callback(pct, f"Đang nhận diện OCR khung hình {current_frame}/{total_frames}...")

            # Crop subtitle region
            crop = frame[max(0, ymin):min(height, ymax), max(0, xmin):min(width, xmax)]
            detected_text = self._ocr_single_frame(ocr, crop)

            # Apply typo map if enabled
            if detected_text and self.use_typo_map:
                detected_text = apply_typo_map(detected_text)

            # Clustering logic with SequenceMatcher
            if detected_text:
                if not active_text:
                    active_text = detected_text
                    start_frame = current_frame
                    end_frame = current_frame + sample_step
                elif self._text_similarity(active_text, detected_text) >= self.similarity_threshold:
                    end_frame = current_frame + sample_step
                    # Keep the longer/better text
                    if len(detected_text) > len(active_text):
                        active_text = detected_text
                else:
                    items.append({
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "text": active_text,
                        "source": "ocr"
                    })
                    active_text = detected_text
                    start_frame = current_frame
                    end_frame = current_frame + sample_step
            else:
                if active_text:
                    items.append({
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "text": active_text,
                        "source": "ocr"
                    })
                    active_text = ""

        # Flush remaining
        if active_text:
            items.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "text": active_text,
                "source": "ocr"
            })

        return items

    def _ocr_single_frame(self, ocr, crop: np.ndarray) -> str:
        """OCR trên một crop image duy nhất."""
        if ocr is None or crop.size == 0:
            return ""
        if crop.std() < 1.0:
            return ""
        try:
            res = ocr(crop)
            if res and isinstance(res, (list, tuple)) and len(res) > 0 and res[0]:
                lines = []
                for item in res[0]:
                    if item and len(item) > 1 and item[1]:
                        lines.append(str(item[1]).strip())
                return " ".join(lines).strip()
        except Exception as e:
            logger.debug(f"OCR frame error: {e}")
        return ""

    def _text_similarity(self, s1: str, s2: str) -> float:
        """Tính độ tương đồng giữa 2 chuỗi text."""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    # =========================================================================
    # PRIVATE: Whisper AI Fallback
    # =========================================================================

    def _run_whisper_fallback(self, video_path: str, fps: float) -> Optional[list]:
        """Chạy Whisper AI speech-to-text khi OCR không phát hiện được chữ."""
        try:
            from backend import whisper_fallback as _wf
        except ImportError:
            logger.info("whisper_fallback module không khả dụng.")
            return None

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="vsr_ocr_engine_whisper_")
            
            # Tách âm thanh từ video
            audio_path = _wf.extract_audio_to_temp(video_path, temp_dir)
            if not audio_path:
                return None

            # Optional: Tách giọng nói trước khi chạy Whisper (cải thiện chất lượng)
            if self.use_voice_separation:
                separated_path = self._separate_vocals(audio_path, temp_dir)
                if separated_path:
                    audio_path = separated_path

            # Chạy Whisper
            lang = self.ocr_lang if self.ocr_lang != "auto" else None
            segments = _wf.run_whisper_segments(audio_path, model_size="tiny", language=lang)

            if not segments:
                return None

            items = []
            for s_sec, e_sec, txt in segments:
                clean_txt = txt.strip()
                if self.use_typo_map:
                    clean_txt = apply_typo_map(clean_txt)
                if clean_txt:
                    items.append({
                        "start_frame": int(s_sec * fps),
                        "end_frame": int(e_sec * fps),
                        "text": clean_txt,
                        "source": "whisper"
                    })
            return items if items else None

        except Exception as e:
            logger.warning(f"Whisper fallback error: {e}")
            return None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    # =========================================================================
    # PRIVATE: Voice Separation
    # =========================================================================

    def _separate_vocals(self, audio_path: str, temp_dir: str) -> Optional[str]:
        """
        Tách giọng nói (vocals) từ file âm thanh.
        Sử dụng FFmpeg hoặc backend/voice_separator.py nếu có.
        """
        try:
            from backend.voice_separator import VoiceSeparator
            separator = VoiceSeparator()
            output_path = os.path.join(temp_dir, "vocals.wav")
            success = separator.separate(audio_path, output_path)
            if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Voice separation thành công: {output_path}")
                return output_path
        except ImportError:
            logger.info("voice_separator module không khả dụng. Bỏ qua tách giọng nói.")
        except Exception as e:
            logger.warning(f"Voice separation error: {e}")
        return None
