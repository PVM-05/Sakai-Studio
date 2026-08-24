import cv2
from PySide6.QtCore import Signal, QThread
from src.ai_engines.translator import SubtitleBlock
from src.ai_engines.ocr_engine import VideoOcrEngine
from src.core.config import config

class VideoOcrThread(QThread):
    """Worker Thread chạy trích xuất OCR phụ đề từ tệp Video ngầm."""
    progress_signal = Signal(float, str)
    finished_signal = Signal(list)  # Emits list[SubtitleBlock]

    def __init__(self, video_path: str, sub_areas: list, parent=None):
        super().__init__(parent=parent)
        self.video_path = video_path
        self.sub_areas = sub_areas
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        blocks: list[SubtitleBlock] = []
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit(blocks)
            return

        cap.release()

        engine = VideoOcrEngine(
            use_typo_map=True,
            use_whisper_fallback=config.whisperFallback.value,
            use_voice_separation=config.voiceSeparation.value
        )

        def progress_callback(pct, msg):
            if self._is_stopped:
                engine.stop()
            self.progress_signal.emit(pct, msg)

        try:
            segments = engine.extract_subtitles(self.video_path, self.sub_areas, progress_callback)
        except Exception:
            self.finished_signal.emit(blocks)
            return
            
        if self._is_stopped:
            self.finished_signal.emit([])
            return
            
        if segments:
            for idx, seg in enumerate(segments, start=1):
                blocks.append(SubtitleBlock(idx, seg.start_time, seg.end_time, seg.text))

        self.finished_signal.emit(blocks)
