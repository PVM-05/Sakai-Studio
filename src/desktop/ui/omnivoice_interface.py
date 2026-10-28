# -*- coding: utf-8 -*-

"""

Giao diện chuyên biệt Lồng Tiếng Trí Tuệ Nhân Tạo OmniVoice.

Cung cấp toàn bộ tính năng của mã nguồn OmniVoice chạy trực tiếp trên card đồ họa NVIDIA:

- 3 chế độ tạo giọng: Nhân bản giọng nói từ mẫu, Thiết kế giọng nói theo thuộc tính, Giọng đọc tự động.

- Hỗ trợ 646 ngôn ngữ có tìm kiếm nhanh.

- Không gian 1: Phòng thu tổng hợp văn bản trực tiếp kèm thanh chèn nhanh 13 thẻ biểu cảm phi ngôn từ và ghi đè phiên âm.

- Không gian 2: Lồng tiếng đa nhân vật từ danh sách phụ đề, tự động nhận diện nhiều người nói từ video hoặc văn bản phụ đề,

  nhái giọng trực tiếp theo từng câu thoại, gán mẫu giọng riêng cho từng nhân vật, nghe thử từng câu và ghép video.

- Toàn bộ tham số giải mã khuếch tán chuyên sâu, quản lý mẫu giọng đã mã hóa và giải phóng bộ nhớ card đồ họa.

"""

from __future__ import annotations

import os

import re

import copy

import time

import shutil

import tempfile

from pathlib import Path

from typing import List, Optional, Dict, Any

from PySide6.QtWidgets import (

    QWidget,

    QVBoxLayout,

    QHBoxLayout,

    QGridLayout,

    QStackedWidget,

    QTableWidgetItem,

    QHeaderView,

    QFileDialog,

    QAbstractItemView,

    QFrame,

)

from PySide6.QtCore import Qt, Signal, QThread, QUrl, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from qfluentwidgets import (

    ScrollArea,

    CardWidget,

    PushButton,

    PrimaryPushButton,

    ComboBox,

    EditableComboBox,

    TableWidget,

    TitleLabel,

    SubtitleLabel,

    StrongBodyLabel,

    BodyLabel,

    CaptionLabel,

    InfoBar,

    FluentIcon,

    ProgressBar,

    LineEdit,

    TextEdit,

    DoubleSpinBox,

    SpinBox,

    CheckBox,

    Pivot,

    Slider,

    ToolButton,

    DropDownPushButton,

    RoundMenu,

    Action,

    MessageBox,

)

from src.core.config import config, BASE_DIR

from src.core.tools.folder_memory import FolderMemoryDialog
from src.core.tools.common_tools import is_video_file

from src.ai_engines.translator import SubtitleBlock, parse_srt

from src.ai_engines.speaker_diarization import CharacterVoiceProfile, normalize_character_profile

from src.ai_engines.tts_engine import (

    TTSEngine,

    VoiceRecommendationEngine,

    VIETNAMESE_VOICE_CATALOG,

    DEFAULT_SPEAKER_PRESETS,

    parse_timestamp_to_seconds,

    validate_audio_file,

)

# Ánh xạ ngôn ngữ phổ biến sang mã chuẩn của OmniVoice

PRIORITY_LANGUAGE_MAP: Dict[str, Optional[str]] = {

    "Tự động phát hiện ngôn ngữ": None,

    "Tiếng Việt": "vi",

    "Tiếng Anh": "en",

    "Tiếng Trung": "zh",

    "Tiếng Nhật": "ja",

    "Tiếng Hàn": "ko",

    "Tiếng Pháp": "fr",

    "Tiếng Đức": "de",

    "Tiếng Nga": "ru",

    "Tiếng Tây Ban Nha": "es",

    "Tiếng Bồ Đào Nha": "pt",

    "Tiếng Ý": "it",

    "Tiếng Thái": "th",

    "Tiếng Indonesia": "id",

    "Tiếng Mã Lai": "ms",

    "Tiếng Ả Rập": "ar",

    "Tiếng Ấn Độ": "hi",

}

# Ánh xạ các thuộc tính Thiết kế giọng nói từ nhãn tiếng Việt sang mã chuẩn của OmniVoice

GENDER_MAP = {

    "Tự động": "",

    "Giọng Nam": "male",

    "Giọng Nữ": "female",

}

AGE_MAP = {

    "Tự động": "",

    "Trẻ em": "child",

    "Thiếu niên": "teenager",

    "Thanh niên": "young adult",

    "Trung niên": "middle-aged",

    "Cao tuổi": "elderly",

}

PITCH_MAP = {

    "Tự động": "",

    "Rất trầm": "very low pitch",

    "Trầm": "low pitch",

    "Trung bình": "moderate pitch",

    "Cao": "high pitch",

    "Rất cao": "very high pitch",

}

STYLE_MAP = {

    "Bình thường": "",

    "Giọng thì thầm": "whisper",

}

ENGLISH_ACCENT_MAP = {

    "Không chỉ định": "",

    "Khẩu âm Mỹ": "american accent",

    "Khẩu âm Anh": "british accent",

    "Khẩu âm Úc": "australian accent",

    "Khẩu âm Canada": "canadian accent",

    "Khẩu âm Ấn Độ": "indian accent",

    "Khẩu âm Hàn Quốc": "korean accent",

    "Khẩu âm Nhật Bản": "japanese accent",

    "Khẩu âm Nga": "russian accent",

    "Khẩu âm Bồ Đào Nha": "portuguese accent",

    "Khẩu âm Trung Quốc": "chinese accent",

}

CHINESE_DIALECT_MAP = {

    "Không chỉ định": "",

    "Phương ngữ Tứ Xuyên": "四川话",

    "Phương ngữ Đông Bắc": "东北话",

    "Phương ngữ Thiểm Tây": "陕西话",

    "Phương ngữ Hà Nam": "河南话",

    "Phương ngữ Quý Châu": "贵州话",

    "Phương ngữ Vân Nam": "云南话",

    "Phương ngữ Quế Lâm": "桂林话",

    "Phương ngữ Tế Nam": "济南话",

    "Phương ngữ Thạch Gia Trang": "石家庄话",

    "Phương ngữ Cam Túc": "甘肃话",

    "Phương ngữ Ninh Hạ": "宁夏话",

    "Phương ngữ Thanh Đảo": "青岛话",

}

# Danh sách 13 thẻ biểu cảm phi ngôn từ với tên hiển thị tiếng Việt

EXPRESSIVE_TAG_BUTTONS = [

    ("Tiếng cười", "[laughter]"),

    ("Tiếng thở dài", "[sigh]"),

    ("Âm xác nhận", "[confirmation-en]"),

    ("Hỏi nhấn giọng", "[question-en]"),

    ("Hỏi ngạc nhiên A", "[question-ah]"),

    ("Hỏi ngạc nhiên O", "[question-oh]"),

    ("Hỏi nghi vấn Ê", "[question-ei]"),

    ("Hỏi nghi vấn Y", "[question-yi]"),

    ("Ngạc nhiên A", "[surprise-ah]"),

    ("Ngạc nhiên Ồ", "[surprise-oh]"),

    ("Ngạc nhiên Oa", "[surprise-wa]"),

    ("Ngạc nhiên Yo", "[surprise-yo]"),

    ("Không hài lòng", "[dissatisfaction-hnn]"),

]

# Danh sách 22 cấu hình giọng mẫu tiếng Việt để chọn nhanh và gán cho nhân vật

SPEAKER_VOICE_PRESETS: Dict[str, str] = {

    preset_name: info["instruct"]

    for preset_name, info in VIETNAMESE_VOICE_CATALOG.items()

}

MULTI_SPEAKER_MODE_ITEMS = [

    ("Một Giọng Đọc Đồng Nhất Cho Toàn Bộ Phụ Đề", "single"),

    ("Tự Động Nhận Diện Nhiều Người Nói Từ Video Gốc", "auto_video_cluster"),

    ("Nhái Giọng Trực Tiếp Từng Câu Theo Video Gốc", "per_line_video_clone"),

    ("Tự Động Phân Vai Đa Nhân Vật Từ Văn Bản Phụ Đề", "auto_srt_text"),

    ("Phân Vai Tùy Chỉnh Theo Bảng Nhân Vật", "custom_table"),

]

def is_valid_media_file(path: str) -> bool:

    """

    Kiểm tra một đường dẫn có phải là tệp âm thanh hoặc video hợp lệ hay không:

    - Đường dẫn tồn tại và là tệp thông thường (không phải thư mục).

    - Dung lượng tệp lớn hơn 0 byte.

    - Phần mở rộng thuộc danh sách định dạng video hoặc âm thanh được hỗ trợ.

    """

    if not path or not isinstance(path, str):

        return False

    if not os.path.isfile(path):

        return False

    try:

        if os.path.getsize(path) <= 0:

            return False

    except OSError:

        return False

    valid_exts = {

        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v", ".wmv",

        ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"

    }

    ext = os.path.splitext(path)[1].lower()

    return ext in valid_exts

def check_media_and_srt_duration(

    video_path: str,

    subtitle_blocks: List[Any],

    threshold_sec: float = 5.0,

) -> Dict[str, Any]:

    """

    Kiểm tra tính đồng bộ thời lượng giữa video nguồn và tệp phụ đề.

    Trả về thông số thời lượng và cờ cảnh báo nếu chênh lệch đáng kể.

    """

    vid_dur = 0.0

    if video_path and os.path.isfile(video_path):

        try:

            import cv2

            cap = cv2.VideoCapture(video_path)

            if cap.isOpened():

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

                vid_dur = total_frames / fps if fps > 0 else 0.0

                cap.release()

        except Exception:

            vid_dur = 0.0

    sub_dur = 0.0

    if subtitle_blocks:

        max_end = 0.0

        for blk in subtitle_blocks:

            end_val = getattr(blk, "end_time", "")

            if end_val:

                try:

                    s = parse_timestamp_to_seconds(str(end_val))

                    if s > max_end:

                        max_end = s

                except Exception:

                    pass

        sub_dur = max_end

    diff = abs(vid_dur - sub_dur)

    is_mismatch = False

    msg = ""

    if vid_dur > 0 and sub_dur > 0:

        if sub_dur > vid_dur + 1.0 or diff > threshold_sec:

            is_mismatch = True

            msg = (

                f"Cảnh báo chênh lệch thời lượng: Video dài {vid_dur:.1f} giây, "

                f"phụ đề dài {sub_dur:.1f} giây — chênh lệch {diff:.1f} giây. "

                f"Vui lòng kiểm tra lại để đảm bảo phụ đề khớp âm thanh video."

            )

        else:

            msg = f"Thời lượng video {vid_dur:.1f} giây và phụ đề {sub_dur:.1f} giây đồng bộ tốt."

    return {

        "video_duration": vid_dur,

        "srt_duration": sub_dur,

        "diff": diff,

        "is_mismatch": is_mismatch,

        "message": msg,

    }

class OmniVoiceTaskWorker(QThread):

    """Luồng nền thực thi các tác vụ tổng hợp giọng nói, nhận diện đa người nói và lồng tiếng video trên GPU."""

    progressSignal = Signal(int, str)

    logSignal = Signal(str)

    speakersDetectedSignal = Signal(dict)

    finishedSignal = Signal(bool, str, str, str)

    def __init__(

        self,

        task_type: str,

        engine_params: Dict[str, Any],

        text: str = "",

        output_path: str = "",

        video_path: str = "",

        subtitle_blocks: Optional[List[SubtitleBlock]] = None,

        original_volume: float = 0.15,

        fixed_duration: float = 0.0,

        override_ref_audio: str = "",

        override_instruct: str = "",

        prefer_cuda: bool = True,

        override_mode: Optional[str] = None,

        override_ref_text: str = "",

        parent=None,

    ):

        super().__init__(parent=parent)

        self.task_type = task_type

        self.engine_params = engine_params

        self.text = text

        self.output_path = output_path

        self.video_path = video_path

        self.subtitle_blocks = subtitle_blocks or []

        self.original_volume = original_volume

        self.fixed_duration = fixed_duration

        self.override_ref_audio = override_ref_audio

        self.override_instruct = override_instruct

        self.prefer_cuda = prefer_cuda

        self.override_mode = override_mode

        self.override_ref_text = override_ref_text

        self._engine: Optional[TTSEngine] = None

        self._did_emit_result: bool = False

    def _emit_log(self, msg: str):

        self.logSignal.emit(msg)

    def _emit_result(self, ok: bool, task_type: str, result_path: str, message: str):

        if not self._did_emit_result:

            self._did_emit_result = True

            self.finishedSignal.emit(ok, task_type, result_path, message)

    def request_cancel(self):

        """Yêu cầu hủy tác vụ đang chạy trên luồng nền một cách an toàn."""

        self.requestInterruption()

        if self._engine is not None:

            self._engine.request_cancel()

    def run(self):

        try:

            engine = TTSEngine(parent_logger=self._emit_log, **self.engine_params)

            self._engine = engine

            self._run_engine_task(engine)

        except Exception as e:

            self.logSignal.emit(f"Lỗi thực thi: {str(e)}")

            self._emit_result(False, self.task_type, "", str(e))

        finally:

            if not self._did_emit_result:

                self._emit_result(False, self.task_type, "", "Tác vụ kết thúc bất thường hoặc đã bị hủy.")

            self._engine = None

    def _run_engine_task(self, engine: TTSEngine):

            if self.task_type == "detect_speakers":

                max_spk = int(self.engine_params.get("max_speakers", 4))

                if self.video_path and os.path.exists(self.video_path):

                    from src.ai_engines.speaker_embedding_onnx import (

                        resolve_speaker_model_path,

                        download_speaker_embedding_model,

                    )

                    model_path = resolve_speaker_model_path()

                    if not model_path:

                        self.progressSignal.emit(10, "Đang tải mô hình nhận diện người nói CAM++ ONNX...")

                        self.logSignal.emit("Chưa tìm thấy mô hình CAM++ ONNX cục bộ. Đang tự động tải từ kho lưu trữ...")

                        try:

                            download_speaker_embedding_model(

                                progress_callback=lambda p, msg: self.progressSignal.emit(int(p * 20), msg)

                            )

                        except Exception as e:

                            self.logSignal.emit(f"Không thể tự động tải mô hình CAM++: {e}")

                    device_str = "ưu tiên CUDA trên GPU NVIDIA" if self.prefer_cuda else "CPU"

                    self.progressSignal.emit(25, f"Đang nhận diện người nói · {device_str}...")

                    exact_spk = self.engine_params.get("exact_speakers", None)

                    sensitivity = self.engine_params.get("clustering_sensitivity", "standard")

                    ai_vocal_sep = bool(self.engine_params.get("ai_vocal_separation", True))
                    if ai_vocal_sep:
                        self.progressSignal.emit(18, "Đang tách sạch nhạc nền và tiếng ồn bằng trí tuệ nhân tạo...")
                    profiles = engine.detect_speakers_from_video(

                        self.video_path,

                        self.subtitle_blocks,

                        max_speakers=max_spk,

                        exact_speakers=exact_spk,

                        clustering_sensitivity=sensitivity,

                        prefer_cuda=self.prefer_cuda,

                        enable_ai_vocal_separation=ai_vocal_sep,

                    )

                    source_desc = "từ âm thanh video gốc"

                else:

                    exact_spk = self.engine_params.get("exact_speakers", None)

                    self.progressSignal.emit(35, "Đang phân tích hội thoại và đại từ trong phụ đề để phân vai...")

                    profiles = engine.detect_speakers_from_srt_text(

                        self.subtitle_blocks,

                        max_speakers=exact_spk if exact_spk is not None else max_spk,

                    )

                    source_desc = "từ văn bản phụ đề"

                if profiles:

                    self.speakersDetectedSignal.emit(profiles)

                    self.progressSignal.emit(100, f"Đã nhận diện {len(profiles)} nhân vật {source_desc}.")

                    self._emit_result(

                        True,

                        self.task_type,

                        "",

                        f"Đã tự động phân vai {len(profiles)} người nói {source_desc}.",

                    )

                else:

                    self._emit_result(

                        False,

                        self.task_type,

                        "",

                        getattr(engine, "last_diarization_error", "")

                        or "Không thể phân tích danh sách người nói từ dữ liệu hiện tại.",

                    )

                return

            if self.task_type == "extract_ref_from_video":

                self.progressSignal.emit(20, "Đang tách giọng nói khỏi nhạc nền từ video...")

                ai_vocal_sep = bool(self.engine_params.get("ai_vocal_separation", True))

                ref_wav = engine.extract_reference_from_video(self.video_path, enable_ai_vocal_separation=ai_vocal_sep)

                if ref_wav and os.path.exists(ref_wav):

                    self.progressSignal.emit(100, "Đã trích xuất giọng mẫu từ video thành công.")

                    self._emit_result(

                        True,

                        self.task_type,

                        ref_wav,

                        f"Đã trích xuất tệp giọng mẫu: {os.path.basename(ref_wav)}",

                    )

                else:

                    self._emit_result(False, self.task_type, "", "Không thể tách giọng nói từ tệp video này.")

                return

            if self.task_type == "transcribe_ref":

                self.progressSignal.emit(30, "Đang nhận diện lời thoại của tệp giọng mẫu...")

                ref_path = self.engine_params.get("ref_audio_path") or ""

                transcript = engine._transcribe_ref_audio_fallback(ref_path)

                if not transcript:

                    engine.load_model()

                    if engine.model._asr_pipe is None:

                        engine.model.load_asr_model()

                    transcript = engine.model.transcribe(ref_path)

                self.progressSignal.emit(100, "Đã nhận diện lời thoại giọng mẫu.")

                self._emit_result(

                    bool(transcript),

                    self.task_type,

                    transcript or "",

                    "Đã nhận diện lời thoại giọng mẫu thành công." if transcript else "Không nhận diện được lời thoại rõ ràng.",

                )

                return

            if self.task_type == "save_prompt":

                self.progressSignal.emit(30, "Đang mã hóa đặc trưng giọng mẫu trên GPU...")

                ok = engine.save_voice_prompt(self.output_path)

                self.progressSignal.emit(100, "Hoàn tất lưu đặc trưng giọng.")

                self._emit_result(

                    ok,

                    self.task_type,

                    self.output_path if ok else "",

                    f"Đã lưu đặc trưng giọng mẫu tại: {self.output_path}" if ok else "Không thể lưu đặc trưng giọng mẫu.",

                )

                return

            if self.task_type in ("single_synth", "block_preview", "vn_catalog_preview", "character_audition"):

                self.progressSignal.emit(25, "Đang nạp mô hình và tổng hợp giọng nói tiếng Việt trên GPU...")

                if self.task_type == "vn_catalog_preview":

                    engine.language = "vi"

                target_dur = self.fixed_duration if self.fixed_duration > 0.4 else None

                override_mode = self.override_mode

                if not override_mode:

                    if self.task_type == "character_audition":

                        override_mode = "clone" if (self.override_ref_audio and not self.override_instruct) else ("design" if self.override_instruct else None)

                    elif self.override_instruct and not self.override_ref_audio:

                        override_mode = "design"

                    elif self.override_ref_audio and not self.override_instruct:

                        override_mode = "clone"

                engine.generate_single(

                    self.text,

                    output_path=self.output_path,

                    target_duration=target_dur,

                    override_mode=override_mode,

                    override_ref_audio=self.override_ref_audio if self.override_ref_audio else None,

                    override_instruct=self.override_instruct if self.override_instruct else None,

                    override_ref_text=self.override_ref_text if self.override_ref_text else None,

                )

                is_valid, wav_dur, _, _ = validate_audio_file(self.output_path)

                if is_valid:

                    self.progressSignal.emit(100, "Đã tổng hợp giọng nói thành công.")

                    self._emit_result(

                        True,

                        self.task_type,

                        self.output_path,

                        f"Đã tạo tệp âm thanh {wav_dur:.1f} giây: {os.path.basename(self.output_path)}",

                    )

                else:

                    self._emit_result(False, self.task_type, "", "Tệp âm thanh đầu ra trống hoặc không đạt chuẩn âm lượng.")

                return

            if self.task_type == "subtitle_audio_only":

                self.progressSignal.emit(15, "Đang chuẩn bị tổng hợp toàn bộ danh sách phụ đề...")

                max_end = 0.0

                for blk in self.subtitle_blocks:

                    end_sec = parse_timestamp_to_seconds(getattr(blk, "end_time", blk.start_time))

                    if end_sec > max_end:

                        max_end = end_sec

                total_dur = max(max_end, 5.0)

                report = engine.synthesize_blocks(

                    self.subtitle_blocks,

                    total_dur,

                    self.output_path,

                    video_path=self.video_path if self.video_path else None,

                )

                self.speaker_profiles = engine.speaker_profiles

                if engine.speaker_profiles:

                    self.speakersDetectedSignal.emit(engine.speaker_profiles)

                if report.status == "success" and report.valid_audio_file:

                    self.progressSignal.emit(100, report.message)

                    self._emit_result(True, self.task_type, self.output_path, report.message)

                elif report.status == "partial" and report.valid_audio_file:

                    self.progressSignal.emit(100, report.message)

                    self._emit_result(True, self.task_type, self.output_path, f"[PARTIAL] {report.message}")

                else:

                    self.progressSignal.emit(0, report.message)

                    self._emit_result(False, self.task_type, "", report.message)

                return

            if self.task_type == "dub_full_video":

                self.progressSignal.emit(15, "Đang tổng hợp giọng lồng tiếng và hòa trộn vào video...")

                ok = engine.dub_video(

                    video_path=self.video_path,

                    subtitle_blocks=self.subtitle_blocks,

                    output_video_path=self.output_path,

                    original_volume=self.original_volume,

                )

                if engine.speaker_profiles:

                    self.speakersDetectedSignal.emit(engine.speaker_profiles)

                rep = engine.last_report

                if ok and rep and rep.status == "partial":

                    msg = f"Đã xuất video một phần — {rep.message}: {self.output_path}"

                    self.progressSignal.emit(100, rep.message)

                    self._emit_result(True, self.task_type, self.output_path, msg)

                elif ok:

                    msg = f"Đã xuất video lồng tiếng thành công tại: {self.output_path}"

                    self.progressSignal.emit(100, "Hoàn tất xuất video lồng tiếng.")

                    self._emit_result(True, self.task_type, self.output_path, msg)

                else:

                    err_msg = rep.message if (rep and rep.message) else "Quá trình ghép âm thanh vào video thất bại."

                    self.progressSignal.emit(0, err_msg)

                    self._emit_result(False, self.task_type, "", err_msg)

                return

    _run_task = _run_engine_task


class _ProgressBarCanvas(QWidget):
    """Vùng vẽ thanh tiến độ phân đoạn ba màu bằng QPainter."""

    def __init__(self, owner: "CastingProgressBarWidget"):
        super().__init__(owner)
        self.owner = owner

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor, QPainterPath
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 4, 4)
        painter.setClipPath(path)

        painter.fillRect(0, 0, w, h, QColor("#E0E0E0"))

        r_high = getattr(self.owner, "ratio_high", 0.0)
        r_rev = getattr(self.owner, "ratio_review", 0.0)

        w_high = int(round(w * r_high))
        w_rev = int(round(w * r_rev))

        if w_high > 0:
            painter.fillRect(0, 0, w_high, h, QColor("#4CAF50"))
        if w_rev > 0:
            painter.fillRect(w_high, 0, w_rev, h, QColor("#FFA726"))

        painter.end()


class CastingProgressBarWidget(QWidget):
    """Tiện ích hiển thị thanh tiến độ phân vai ba màu và thông số chi tiết."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ratio_high: float = 0.0
        self.ratio_review: float = 0.0
        self.ratio_unassigned: float = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        self.bar_canvas = _ProgressBarCanvas(self)
        self.bar_canvas.setFixedHeight(10)
        layout.addWidget(self.bar_canvas)

        self.lbl_details = CaptionLabel("Chưa có dữ liệu câu thoại.", self)
        self.lbl_details.setWordWrap(True)
        layout.addWidget(self.lbl_details)

    def update_progress(self, blocks: List[Any]):
        """Cập nhật các tỷ lệ và nhãn chi tiết theo danh sách câu thoại."""
        total = len(blocks) if blocks else 0
        if total == 0:
            self.ratio_high = 0.0
            self.ratio_review = 0.0
            self.ratio_unassigned = 0.0
            self.lbl_details.setText("Chưa có dữ liệu câu thoại.")
            self.bar_canvas.update()
            return

        high_count = sum(
            1 for b in blocks
            if getattr(b, "is_speaker_locked", False) or getattr(b, "confidence_level", "") == "high"
        )
        review_count = sum(
            1 for b in blocks
            if not getattr(b, "is_speaker_locked", False) and getattr(b, "confidence_level", "") == "review_needed"
        )
        unassigned_count = sum(
            1 for b in blocks
            if not getattr(b, "is_speaker_locked", False) and getattr(b, "confidence_level", "") in ("unassigned", "")
        )
        assigned_count = sum(
            1 for b in blocks
            if getattr(b, "speaker", "") and getattr(b, "speaker", "") not in ("Chưa xác định", "unassigned", "")
        )
        locked_count = sum(1 for b in blocks if getattr(b, "is_speaker_locked", False))

        self.ratio_high = high_count / float(total)
        self.ratio_review = review_count / float(total)
        self.ratio_unassigned = unassigned_count / float(total)

        pct = int(round((assigned_count / float(total)) * 100))
        self.lbl_details.setText(
            f"Tiến độ phân vai: {assigned_count}/{total} câu thoại — Đạt {pct}% · "
            f"{locked_count} câu đã khóa · {review_count} câu cần rà soát · {unassigned_count} câu chưa xác định"
        )
        self.bar_canvas.update()


class OmniVoiceInterface(ScrollArea):

    """

    Thẻ chức năng chuyên biệt Lồng Tiếng Trí Tuệ Nhân Tạo OmniVoice.

    """

    def __init__(self, parent=None):

        super().__init__(parent=parent)

        self.setObjectName("OmniVoiceInterface")

        self.parent = parent

        self.current_step_index: int = 0

        self.step_buttons: List[PushButton] = []

        self.current_ref_audio_path: str = config.mmoTTSRefAudioPath.value or ""

        self.current_subtitle_path: str = ""

        self.current_video_path: str = ""

        self.last_generated_audio_path: str = ""

        self.voice_prompt_cache_dir: str = os.path.join(tempfile.gettempdir(), "sakai_omnivoice_cache")
        os.makedirs(self.voice_prompt_cache_dir, exist_ok=True)

        self.subtitle_blocks: List[SubtitleBlock] = []

        self.speaker_profiles: Dict[str, Dict[str, Any]] = {}

        self.is_speakers_detected: bool = False

        self._active_audition_button: Optional[PushButton] = None

        self._audition_queue: List[str] = []

        self._audition_all_active: bool = False

        self._current_audition_speaker: Optional[str] = None

        self._audition_all_total: int = 0

        self._audition_all_current_idx: int = 0

        self._worker: Optional[OmniVoiceTaskWorker] = None

        self._audio_output = QAudioOutput(self)

        self._audio_output.setVolume(1.0)

        self._media_player = QMediaPlayer(self)

        self._media_player.setAudioOutput(self._audio_output)

        self._init_ui()

        self._connect_signals()

        self._sync_ui_from_config()

        self._switch_to_step(0)

        self._update_identify_button_state()

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime and hasattr(mime, 'hasUrls') and mime.hasUrls():
            for url in mime.urls():
                p = os.path.abspath(os.path.normpath(url.toLocalFile())) if url.toLocalFile() else ""
                if p and (is_video_file(p) or p.lower().endswith(('.srt', '.vtt', '.ass', '.txt', '.wav', '.mp3', '.pt'))):
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        mime = event.mimeData()
        if mime and hasattr(mime, 'hasUrls') and mime.hasUrls():
            for url in mime.urls():
                p = os.path.abspath(os.path.normpath(url.toLocalFile())) if url.toLocalFile() else ""
                if p and (is_video_file(p) or p.lower().endswith(('.srt', '.vtt', '.ass', '.txt', '.wav', '.mp3', '.pt'))):
                    event.acceptProposedAction()
                    return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime and hasattr(mime, 'hasUrls') and mime.hasUrls():
            sub_file = None
            vid_file = None
            audio_file = None
            for url in mime.urls():
                p = os.path.abspath(os.path.normpath(url.toLocalFile())) if url.toLocalFile() else ""
                if not p or not os.path.exists(p):
                    continue
                if p.lower().endswith(('.srt', '.vtt', '.ass', '.txt')) and not sub_file:
                    sub_file = p
                elif is_video_file(p) and not vid_file:
                    vid_file = p
                elif p.lower().endswith(('.wav', '.mp3', '.pt')) and not audio_file:
                    audio_file = p

            handled = False
            if sub_file or vid_file:
                if sub_file:
                    self._load_subtitle_file_internal(sub_file)
                if vid_file:
                    self.current_video_path = vid_file
                    if hasattr(self, '_update_sub_video_status_label'):
                        self._update_sub_video_status_label()
                    if hasattr(self, '_update_subtitle_action_buttons'):
                        self._update_subtitle_action_buttons()
                    InfoBar.success("Đã nạp video nguồn", os.path.basename(vid_file), parent=self, duration=2500)
                self._update_step_navigation_states()
                handled = True
            elif audio_file:
                self.current_ref_audio_path = audio_file
                if hasattr(self, 'lbl_ref_status'):
                    self.lbl_ref_status.setText(f"Tệp âm thanh mẫu đang nạp: {os.path.basename(audio_file)}")
                if hasattr(self, '_update_control_states'):
                    self._update_control_states()
                InfoBar.success("Đã nạp mẫu giọng", os.path.basename(audio_file), parent=self, duration=2500)
                handled = True

            if handled:
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def _init_ui(self):

        self.container = QWidget(self)

        self.setWidget(self.container)

        self.setWidgetResizable(True)

        self.container.setObjectName("omnivoiceContainer")

        root_layout = QVBoxLayout(self.container)

        root_layout.setContentsMargins(24, 18, 24, 18)

        root_layout.setSpacing(14)

        # =====================================================================

        # 1. TIÊU ĐỀ GIAO DIỆN & TÁC VỤ NHANH

        # =====================================================================

        header_card = CardWidget(self.container)

        header_layout = QVBoxLayout(header_card)

        header_layout.setContentsMargins(18, 14, 18, 14)

        header_layout.setSpacing(10)

        top_row = QHBoxLayout()

        title_box = QVBoxLayout()

        title_lbl = TitleLabel("Lồng Tiếng Trí Tuệ Nhân Tạo", header_card)

        title_lbl.setWordWrap(True)

        desc_lbl = CaptionLabel(

            "Tự động nhận diện người nói, phân vai nhân vật và lồng tiếng đa ngôn ngữ trên card đồ họa NVIDIA.",

            header_card,

        )

        desc_lbl.setWordWrap(True)

        title_box.addWidget(title_lbl)

        title_box.addWidget(desc_lbl)

        top_row.addLayout(title_box, 1)

        self.btn_cancel_task = PushButton("Hủy Tác Vụ", header_card)

        self.btn_cancel_task.setIcon(FluentIcon.CLOSE)

        self.btn_cancel_task.setEnabled(False)

        top_row.addWidget(self.btn_cancel_task, 0, Qt.AlignRight | Qt.AlignTop)

        self.btn_free_vram = PushButton("Giải Phóng Bộ Nhớ Card Đồ Họa", header_card)

        self.btn_free_vram.setIcon(FluentIcon.BROOM)

        top_row.addWidget(self.btn_free_vram, 0, Qt.AlignRight | Qt.AlignTop)

        header_layout.addLayout(top_row)

        root_layout.addWidget(header_card)

        # =====================================================================

        # 2. THANH ĐIỀU HƯỚNG QUY TRÌNH 5 BƯỚC

        # =====================================================================

        nav_card = CardWidget(self.container)

        nav_layout = QHBoxLayout(nav_card)

        nav_layout.setContentsMargins(14, 10, 14, 10)

        nav_layout.setSpacing(8)

        self.btn_prev_step = PushButton("Bước Trước", nav_card)

        self.btn_prev_step.setIcon(FluentIcon.LEFT_ARROW)

        self.btn_prev_step.setEnabled(False)

        self.step_buttons = []

        step_labels = [

            "1. Nguồn Đầu Vào",

            "2. Quản Lý Nhân Vật",

            "3. Biên Tập Phụ Đề",

            "4. Cân Chỉnh Âm Thanh",

            "5. Kiểm Tra Và Xuất",

        ]

        for idx, lbl in enumerate(step_labels):

            btn = PushButton(lbl, nav_card)

            btn.clicked.connect(lambda checked=False, i=idx: self._on_step_button_clicked(i))

            self.step_buttons.append(btn)

        self.btn_next_step = PushButton("Bước Tiếp Theo", nav_card)

        self.btn_next_step.setIcon(FluentIcon.RIGHT_ARROW)

        nav_layout.addWidget(self.btn_prev_step)

        for btn in self.step_buttons:

            nav_layout.addWidget(btn, 1)

        nav_layout.addWidget(self.btn_next_step)

        root_layout.addWidget(nav_card)

        # =====================================================================

        # 3. NỘI DUNG 5 BƯỚC TUẦN TỰ

        # =====================================================================

        self.step_stack = QStackedWidget(self.container)

        # ---------------------------------------------------------------------

        # BƯỚC 0: NGUỒN ĐẦU VÀO

        # ---------------------------------------------------------------------

        self.page_step_0 = QWidget(self.step_stack)

        p0_layout = QVBoxLayout(self.page_step_0)

        p0_layout.setContentsMargins(0, 0, 0, 0)

        p0_layout.setSpacing(12)

        sub_file_card = CardWidget(self.page_step_0)

        sub_file_layout = QVBoxLayout(sub_file_card)

        sub_file_layout.setContentsMargins(18, 14, 18, 14)

        sub_file_layout.setSpacing(10)

        sub_file_layout.addWidget(SubtitleLabel("Tệp Phụ Đề Và Video Nguồn", sub_file_card))

        sub_file_btn_row = QHBoxLayout()

        self.btn_load_srt = PushButton("Chọn Tệp Phụ Đề", sub_file_card)

        self.btn_load_srt.setIcon(FluentIcon.DOCUMENT)

        self.btn_load_video = PushButton("Chọn Video Nguồn Để Ghép", sub_file_card)

        self.btn_load_video.setIcon(FluentIcon.VIDEO)

        self.btn_clear_video = PushButton("Bỏ Chọn Video", sub_file_card)

        self.btn_clear_video.setIcon(FluentIcon.DELETE)

        self.btn_clear_video.setEnabled(False)

        self.lbl_sub_video_info = CaptionLabel("Chưa nạp tệp phụ đề và video nguồn.", sub_file_card)

        self.lbl_sub_video_info.setWordWrap(True)

        sub_file_btn_row.addWidget(self.btn_load_srt)

        sub_file_btn_row.addWidget(self.btn_load_video)

        sub_file_btn_row.addWidget(self.btn_clear_video)

        sub_file_btn_row.addWidget(self.lbl_sub_video_info, 1)

        sub_file_layout.addLayout(sub_file_btn_row)

        # Hàng nút nhận diện chính ở Bước 1

        detect_action_row = QHBoxLayout()

        detect_action_row.setSpacing(10)

        self.btn_identify_characters = PrimaryPushButton("Nhận Diện Nhân Vật Từ Video + SRT", sub_file_card)

        self.btn_identify_characters.setIcon(FluentIcon.PEOPLE)

        self.btn_identify_characters.setEnabled(False)

        self.btn_identify_characters.setMinimumHeight(36)

        self.lbl_detect_source_hint = CaptionLabel("Nạp tệp SRT và video để bắt đầu nhận diện nhân vật.", sub_file_card)

        self.lbl_detect_source_hint.setWordWrap(True)

        detect_action_row.addWidget(self.btn_identify_characters)

        detect_action_row.addWidget(self.lbl_detect_source_hint, 1)

        sub_file_layout.addLayout(detect_action_row)

        p0_layout.addWidget(sub_file_card)

        # Khối chế độ một giọng đọc đồng nhất hoặc phân vai đa nhân vật
        voice_mode_card = CardWidget(self.page_step_0)
        voice_mode_layout = QVBoxLayout(voice_mode_card)
        voice_mode_layout.setContentsMargins(18, 14, 18, 14)
        voice_mode_layout.setSpacing(10)

        voice_mode_layout.addWidget(SubtitleLabel("Chế Độ Giọng Đọc Phụ Đề", voice_mode_card))

        self.chk_single_unified_voice = CheckBox("Sử dụng một giọng đọc đồng nhất cho toàn bộ tệp phụ đề", voice_mode_card)
        self.chk_single_unified_voice.setChecked(True)
        self.chk_single_unified_voice.setStyleSheet("font-weight: bold; font-size: 14px;")
        voice_mode_layout.addWidget(self.chk_single_unified_voice)

        single_voice_row = QHBoxLayout()
        single_voice_row.setSpacing(10)
        lbl_unified = BodyLabel("Giọng đọc tiếng Việt:", voice_mode_card)
        lbl_unified.setWordWrap(True)
        self.combo_single_unified_preset = ComboBox(voice_mode_card)
        self.combo_single_unified_preset.addItems(list(VIETNAMESE_VOICE_CATALOG.keys()))
        self.combo_single_unified_preset.setMinimumWidth(320)

        self.btn_preview_single_unified = PushButton("Nghe Thử Giọng Này", voice_mode_card)
        self.btn_preview_single_unified.setIcon(FluentIcon.VOLUME)

        single_voice_row.addWidget(lbl_unified)
        single_voice_row.addWidget(self.combo_single_unified_preset, 1)
        single_voice_row.addWidget(self.btn_preview_single_unified)
        voice_mode_layout.addLayout(single_voice_row)

        p0_layout.addWidget(voice_mode_card)

        # Nút chuyển đổi thu gọn thiết lập giọng mặc định (mặc định đóng để giảm rối mắt)

        self.btn_toggle_default_voice_settings = PushButton("Tùy Chọn Giọng Mặc Định Dự Phòng", self.page_step_0)

        self.btn_toggle_default_voice_settings.setIcon(FluentIcon.CHEVRON_RIGHT_MED)

        self.btn_toggle_default_voice_settings.clicked.connect(self._toggle_default_voice_settings)

        p0_layout.addWidget(self.btn_toggle_default_voice_settings)

        self.default_voice_settings_container = QWidget(self.page_step_0)

        default_voice_layout = QVBoxLayout(self.default_voice_settings_container)

        default_voice_layout.setContentsMargins(0, 0, 0, 0)

        default_voice_layout.setSpacing(10)

        general_voice_card = CardWidget(self.default_voice_settings_container)

        general_voice_layout = QVBoxLayout(general_voice_card)

        general_voice_layout.setContentsMargins(18, 14, 18, 14)

        general_voice_layout.setSpacing(10)

        general_voice_layout.addWidget(SubtitleLabel("Thiết Lập Giọng Nói Mặc Định", general_voice_card))

        mode_row = QHBoxLayout()

        mode_row.setSpacing(12)

        lbl_mode = BodyLabel("Chế độ tạo giọng:", general_voice_card)

        lbl_mode.setStyleSheet("font-weight: bold;")

        lbl_mode.setWordWrap(True)

        self.combo_mode = ComboBox(general_voice_card)

        self.combo_mode.addItems([

            "Nhân Bản Giọng Nói Từ Mẫu",

            "Thiết Kế Giọng Nói Theo Thuộc Tính",

            "Giọng Đọc Tự Động",

        ])

        self.combo_mode.setMinimumWidth(260)
        self.combo_mode.setCurrentIndex(1)

        lbl_lang = BodyLabel("Ngôn ngữ đọc:", general_voice_card)

        lbl_lang.setStyleSheet("font-weight: bold;")

        lbl_lang.setWordWrap(True)

        self.combo_lang = EditableComboBox(general_voice_card)

        self.combo_lang.setMinimumWidth(250)

        self._populate_languages()

        mode_row.addWidget(lbl_mode)

        mode_row.addWidget(self.combo_mode)

        mode_row.addSpacing(16)

        mode_row.addWidget(lbl_lang)

        mode_row.addWidget(self.combo_lang, 1)

        general_voice_layout.addLayout(mode_row)

        vn_catalog_row = QHBoxLayout()

        vn_catalog_row.setSpacing(10)

        lbl_vn_cat = BodyLabel("Giọng đọc mẫu tiếng Việt:", general_voice_card)

        lbl_vn_cat.setStyleSheet("font-weight: bold;")

        lbl_vn_cat.setWordWrap(True)

        self.combo_vn_catalog = ComboBox(general_voice_card)

        self.combo_vn_catalog.addItems(list(VIETNAMESE_VOICE_CATALOG.keys()))

        self.combo_vn_catalog.setMinimumWidth(290)

        self.edit_vn_sample_text = LineEdit(general_voice_card)

        first_preset = next(iter(VIETNAMESE_VOICE_CATALOG.values()))

        self.edit_vn_sample_text.setText(first_preset["sample_vi"])

        self.edit_vn_sample_text.setPlaceholderText("Nhập câu tiếng Việt bất kỳ để nghe thử giọng đọc mẫu đang chọn...")

        self.edit_vn_sample_text.setClearButtonEnabled(True)

        self.btn_preview_vn_voice = PrimaryPushButton("Nghe Thử Tiếng Việt", general_voice_card)

        self.btn_preview_vn_voice.setIcon(FluentIcon.VOLUME)

        self.btn_use_vn_voice_as_ref = PushButton("Chốt Làm Giọng Mẫu Chính", general_voice_card)

        self.btn_use_vn_voice_as_ref.setIcon(FluentIcon.ACCEPT)

        self.btn_use_vn_voice_as_ref.setEnabled(False)

        vn_catalog_row.addWidget(lbl_vn_cat)

        vn_catalog_row.addWidget(self.combo_vn_catalog)

        vn_catalog_row.addWidget(self.edit_vn_sample_text, 1)

        vn_catalog_row.addWidget(self.btn_preview_vn_voice)

        vn_catalog_row.addWidget(self.btn_use_vn_voice_as_ref)

        general_voice_layout.addLayout(vn_catalog_row)

        default_voice_layout.addWidget(general_voice_card)

        self.voice_source_stack = QStackedWidget(self.default_voice_settings_container)

        self.clone_card = CardWidget(self.voice_source_stack)

        clone_layout = QGridLayout(self.clone_card)

        clone_layout.setContentsMargins(18, 14, 18, 14)

        clone_layout.setSpacing(10)

        clone_title = SubtitleLabel("Thiết Lập Nhân Bản Giọng Nói Từ Mẫu", self.clone_card)

        clone_title.setWordWrap(True)

        clone_layout.addWidget(clone_title, 0, 0, 1, 3)

        self.btn_pick_ref_audio = PushButton("Chọn Tệp Âm Thanh Mẫu", self.clone_card)

        self.btn_pick_ref_audio.setIcon(FluentIcon.MUSIC)

        clone_layout.addWidget(self.btn_pick_ref_audio, 1, 0)

        self.btn_extract_ref_video = PushButton("Tách Giọng Mẫu Từ Video", self.clone_card)

        self.btn_extract_ref_video.setIcon(FluentIcon.VIDEO)

        clone_layout.addWidget(self.btn_extract_ref_video, 1, 1)

        self.btn_play_ref_audio = PushButton("Nghe Giọng Mẫu", self.clone_card)

        self.btn_play_ref_audio.setIcon(FluentIcon.VOLUME)

        clone_layout.addWidget(self.btn_play_ref_audio, 1, 2)

        self.btn_load_prompt_pt = PushButton("Nạp Mẫu Giọng Đã Mã Hóa", self.clone_card)

        self.btn_load_prompt_pt.setIcon(FluentIcon.FOLDER)

        clone_layout.addWidget(self.btn_load_prompt_pt, 2, 0)

        self.btn_save_prompt_pt = PushButton("Lưu Mẫu Giọng Đã Mã Hóa", self.clone_card)

        self.btn_save_prompt_pt.setIcon(FluentIcon.SAVE)

        clone_layout.addWidget(self.btn_save_prompt_pt, 2, 1)

        self.chk_auto_extract_video = CheckBox("Tự động lấy giọng từ video khi lồng tiếng phụ đề", self.clone_card)

        clone_layout.addWidget(self.chk_auto_extract_video, 2, 2)

        self.lbl_ref_status = CaptionLabel(

            "Tệp giọng mẫu hiện tại: Chưa chọn tệp âm thanh mẫu, khuyến nghị đoạn thu âm từ 3 đến 10 giây.",

            self.clone_card,

        )

        self.lbl_ref_status.setWordWrap(True)

        clone_layout.addWidget(self.lbl_ref_status, 3, 0, 1, 3)

        lbl_ref_text = BodyLabel("Lời thoại của giọng mẫu:", self.clone_card)

        lbl_ref_text.setWordWrap(True)

        clone_layout.addWidget(lbl_ref_text, 4, 0)

        self.edit_ref_text = LineEdit(self.clone_card)

        self.edit_ref_text.setPlaceholderText("Nhập lời thoại của tệp giọng mẫu hoặc bấm nút bên cạnh để nhận diện tự động...")

        self.edit_ref_text.setClearButtonEnabled(True)

        clone_layout.addWidget(self.edit_ref_text, 4, 1)

        self.btn_transcribe_ref = PushButton("Nhận Diện Lời Thoại Mẫu", self.clone_card)

        self.btn_transcribe_ref.setIcon(FluentIcon.MICROPHONE)

        clone_layout.addWidget(self.btn_transcribe_ref, 4, 2)

        self.voice_source_stack.addWidget(self.clone_card)
        self.clone_card.setVisible(False)

        self.design_card = CardWidget(self.voice_source_stack)

        design_layout = QGridLayout(self.design_card)

        design_layout.setContentsMargins(18, 14, 18, 14)

        design_layout.setSpacing(10)

        design_title = SubtitleLabel("Thiết Kế Giọng Nói Theo Thuộc Tính Nhân Vật", self.design_card)

        design_title.setWordWrap(True)

        design_layout.addWidget(design_title, 0, 0, 1, 6)

        design_layout.addWidget(BodyLabel("Giới tính:", self.design_card), 1, 0)

        self.combo_gender = ComboBox(self.design_card)

        self.combo_gender.addItems(list(GENDER_MAP.keys()))

        design_layout.addWidget(self.combo_gender, 1, 1)

        design_layout.addWidget(BodyLabel("Độ tuổi:", self.design_card), 1, 2)

        self.combo_age = ComboBox(self.design_card)

        self.combo_age.addItems(list(AGE_MAP.keys()))

        design_layout.addWidget(self.combo_age, 1, 3)

        design_layout.addWidget(BodyLabel("Cao độ:", self.design_card), 1, 4)

        self.combo_pitch = ComboBox(self.design_card)

        self.combo_pitch.addItems(list(PITCH_MAP.keys()))

        design_layout.addWidget(self.combo_pitch, 1, 5)

        design_layout.addWidget(BodyLabel("Phong cách:", self.design_card), 2, 0)

        self.combo_style = ComboBox(self.design_card)

        self.combo_style.addItems(list(STYLE_MAP.keys()))

        design_layout.addWidget(self.combo_style, 2, 1)

        design_layout.addWidget(BodyLabel("Khẩu âm tiếng Anh:", self.design_card), 2, 2)

        self.combo_accent = ComboBox(self.design_card)

        self.combo_accent.addItems(list(ENGLISH_ACCENT_MAP.keys()))

        design_layout.addWidget(self.combo_accent, 2, 3)

        design_layout.addWidget(BodyLabel("Phương ngữ tiếng Trung:", self.design_card), 2, 4)

        self.combo_dialect = ComboBox(self.design_card)

        self.combo_dialect.addItems(list(CHINESE_DIALECT_MAP.keys()))

        design_layout.addWidget(self.combo_dialect, 2, 5)

        design_layout.addWidget(BodyLabel("Chuỗi thuộc tính tổng hợp:", self.design_card), 3, 0)

        self.edit_instruct_preview = LineEdit(self.design_card)

        self.edit_instruct_preview.setPlaceholderText("Các thuộc tính được chọn ở trên sẽ tự động kết hợp tại đây...")

        self.edit_instruct_preview.setClearButtonEnabled(True)

        design_layout.addWidget(self.edit_instruct_preview, 3, 1, 1, 5)

        self.voice_source_stack.addWidget(self.design_card)

        self.auto_voice_card = CardWidget(self.voice_source_stack)

        auto_voice_layout = QVBoxLayout(self.auto_voice_card)

        auto_voice_layout.setContentsMargins(18, 14, 18, 14)

        auto_voice_layout.addWidget(SubtitleLabel("Chế Độ Giọng Đọc Tự Động", self.auto_voice_card))

        lbl_auto_desc = BodyLabel(

            "Ở chế độ này, mô hình OmniVoice tự động lựa chọn âm sắc giọng đọc tự nhiên phù hợp nhất với ngôn ngữ và văn bản đầu vào mà không cần tệp âm thanh mẫu.",

            self.auto_voice_card,

        )

        lbl_auto_desc.setWordWrap(True)

        auto_voice_layout.addWidget(lbl_auto_desc)

        self.voice_source_stack.addWidget(self.auto_voice_card)

        default_voice_layout.addWidget(self.voice_source_stack)

        self.default_voice_settings_container.setVisible(False)

        p0_layout.addWidget(self.default_voice_settings_container)

        self.step_stack.addWidget(self.page_step_0)

        # ---------------------------------------------------------------------

        # BƯỚC 1: QUẢN LÝ NHÂN VẬT (BẢNG VOICE CASTING)

        # ---------------------------------------------------------------------

        self.page_step_1 = QWidget(self.step_stack)

        p1_layout = QVBoxLayout(self.page_step_1)

        p1_layout.setContentsMargins(0, 0, 0, 0)

        p1_layout.setSpacing(12)

        diarize_card = CardWidget(self.page_step_1)

        diarize_layout = QVBoxLayout(diarize_card)

        diarize_layout.setContentsMargins(14, 12, 14, 12)

        diarize_layout.setSpacing(10)

        diarize_top_row = QHBoxLayout()

        diarize_top_row.setSpacing(8)

        lbl_multi_mode = BodyLabel("Chế độ giọng phụ đề:", diarize_card)

        lbl_multi_mode.setStyleSheet("font-weight: bold;")

        lbl_multi_mode.setWordWrap(True)

        self.combo_multi_speaker_mode = ComboBox(diarize_card)

        for label_vn, _ in MULTI_SPEAKER_MODE_ITEMS:

            self.combo_multi_speaker_mode.addItem(label_vn)

        self.combo_multi_speaker_mode.setMinimumWidth(260)

        self.btn_auto_detect_speakers = PrimaryPushButton("Tự Động Nhận Diện Nhiều Người Nói", diarize_card)

        self.btn_auto_detect_speakers.setIcon(FluentIcon.PEOPLE)

        self.btn_auto_detect_speakers.setEnabled(False)

        diarize_top_row.addWidget(lbl_multi_mode)

        diarize_top_row.addWidget(self.combo_multi_speaker_mode, 1)

        diarize_top_row.addWidget(self.btn_auto_detect_speakers)

        diarize_layout.addLayout(diarize_top_row)

        diarize_clustering_row = QHBoxLayout()

        diarize_clustering_row.setSpacing(8)

        lbl_count_mode = BodyLabel("Quy tắc phân vai:", diarize_card)

        lbl_count_mode.setWordWrap(True)

        self.combo_speaker_count_mode = ComboBox(diarize_card)

        self.combo_speaker_count_mode.addItem("Tự động ước lượng theo giới hạn tối đa")

        self.combo_speaker_count_mode.addItem("Cố định chính xác số lượng nhân vật")

        self.lbl_max_spk = BodyLabel("Số lượng nhân vật tối đa:", diarize_card)

        self.lbl_max_spk.setWordWrap(True)

        self.spin_max_speakers = SpinBox(diarize_card)

        self.spin_max_speakers.setRange(1, 12)

        self.spin_max_speakers.setValue(4)

        lbl_sensitivity = BodyLabel("Độ nhạy phân tách:", diarize_card)

        lbl_sensitivity.setWordWrap(True)

        self.combo_clustering_sensitivity = ComboBox(diarize_card)

        self.combo_clustering_sensitivity.addItem("Độ nhạy cân bằng")

        self.combo_clustering_sensitivity.addItem("Độ nhạy cao — Phân tách chi tiết")

        self.combo_clustering_sensitivity.addItem("Độ nhạy thấp — Ưu tiên gộp nhóm")

        diarize_clustering_row.addWidget(lbl_count_mode)

        diarize_clustering_row.addWidget(self.combo_speaker_count_mode, 1)

        diarize_clustering_row.addWidget(self.lbl_max_spk)

        diarize_clustering_row.addWidget(self.spin_max_speakers)

        diarize_clustering_row.addWidget(lbl_sensitivity)

        diarize_clustering_row.addWidget(self.combo_clustering_sensitivity, 1)

        diarize_layout.addLayout(diarize_clustering_row)

        diarize_options_row = QHBoxLayout()
        diarize_options_row.setSpacing(8)
        self.chk_ai_vocal_separation = CheckBox("Tách sạch nhạc nền bằng trí tuệ nhân tạo", diarize_card)
        self.chk_ai_vocal_separation.setChecked(True)
        self.chk_ai_vocal_separation.setToolTip("Triệt tiêu bài hát, nhạc nền và tạp âm môi trường trước khi trích xuất giọng nói")
        diarize_options_row.addWidget(self.chk_ai_vocal_separation)
        diarize_options_row.addStretch(1)
        diarize_layout.addLayout(diarize_options_row)

        spk_assign_row1 = QHBoxLayout()

        spk_assign_row1.setSpacing(8)

        spk_assign_row1.addWidget(BodyLabel("Nhân vật:", diarize_card))

        self.combo_target_speaker = ComboBox(diarize_card)

        self.combo_target_speaker.addItems([f"Người nói {i}" for i in range(1, 7)])

        self.combo_speaker_voice_preset = ComboBox(diarize_card)

        self.combo_speaker_voice_preset.addItems(list(SPEAKER_VOICE_PRESETS.keys()))

        self.btn_preview_speaker_preset_vi = PushButton("Nghe Thử Giọng Nhân Vật", diarize_card)

        self.btn_preview_speaker_preset_vi.setIcon(FluentIcon.VOLUME)

        spk_assign_row1.addWidget(self.combo_target_speaker)

        spk_assign_row1.addWidget(self.combo_speaker_voice_preset, 1)

        spk_assign_row1.addWidget(self.btn_preview_speaker_preset_vi)

        diarize_layout.addLayout(spk_assign_row1)

        spk_assign_row2 = QHBoxLayout()

        spk_assign_row2.setSpacing(8)

        self.btn_apply_preset_to_speaker = PushButton("Áp Dụng Thiết Kế Giọng Cho Nhân Vật", diarize_card)

        self.btn_apply_preset_to_speaker.setIcon(FluentIcon.ACCEPT)

        self.btn_pick_audio_for_speaker = PushButton("Chọn Tệp Giọng Mẫu Cho Nhân Vật", diarize_card)

        self.btn_pick_audio_for_speaker.setIcon(FluentIcon.MUSIC)

        self.btn_assign_speaker_to_rows = PushButton("Gán Nhân Vật Vào Câu Đang Chọn", diarize_card)

        self.btn_assign_speaker_to_rows.setIcon(FluentIcon.EDIT)

        self.btn_reset_speaker_voice = PushButton("Tạo Lại Mẫu Giọng", diarize_card)

        self.btn_reset_speaker_voice.setIcon(FluentIcon.SYNC)

        spk_assign_row2.addWidget(self.btn_apply_preset_to_speaker)

        spk_assign_row2.addWidget(self.btn_pick_audio_for_speaker)

        spk_assign_row2.addWidget(self.btn_assign_speaker_to_rows)

        spk_assign_row2.addWidget(self.btn_reset_speaker_voice)

        diarize_layout.addLayout(spk_assign_row2)

        self.lbl_speaker_summary = CaptionLabel(

            "Mẹo: Bấm Tự Động Nhận Diện Nhiều Người Nói để hệ thống tự tách giọng từng nhân vật từ video hoặc phân vai hội thoại từ tệp phụ đề.",

            diarize_card,

        )

        self.lbl_speaker_summary.setWordWrap(True)

        diarize_layout.addWidget(self.lbl_speaker_summary)

        p1_layout.addWidget(diarize_card)

        casting_card = CardWidget(self.page_step_1)

        casting_layout = QVBoxLayout(casting_card)

        casting_layout.setContentsMargins(14, 12, 14, 12)

        casting_layout.setSpacing(8)

        casting_header_row = QHBoxLayout()

        casting_header_row.addWidget(SubtitleLabel("Bảng Phân Vai Và Giọng Đọc Nhân Vật", casting_card))

        casting_header_row.addStretch(1)

        self.btn_reanalyze_speakers = PrimaryPushButton("Tự Động Phân Tích Lại", casting_card)

        self.btn_reanalyze_speakers.setIcon(FluentIcon.SYNC)

        self.btn_more_actions = DropDownPushButton("Thao Tác Khác", casting_card)

        self.btn_more_actions.setIcon(FluentIcon.MORE)

        self._init_more_actions_menu()

        self.btn_reset_all_voices = PushButton("Đặt Lại Toàn Bộ Giọng", casting_card)

        self.btn_reset_all_voices.setIcon(FluentIcon.DELETE)

        self.btn_audition_all_speakers = PushButton("Nghe Thử Toàn Bộ Nhân Vật", casting_card)

        self.btn_audition_all_speakers.setIcon(FluentIcon.VOLUME)

        casting_header_row.addWidget(self.btn_reanalyze_speakers)

        casting_header_row.addWidget(self.btn_more_actions)

        casting_layout.addLayout(casting_header_row)

        self.lbl_diarization_summary = CaptionLabel("Chưa phân tích — bấm nhận diện để bắt đầu.", casting_card)

        self.lbl_diarization_summary.setWordWrap(True)

        self.lbl_diarization_method = CaptionLabel("", casting_card)

        self.lbl_diarization_method.setWordWrap(True)

        casting_layout.addWidget(self.lbl_diarization_summary)

        casting_layout.addWidget(self.lbl_diarization_method)
        self.casting_progress_bar = CastingProgressBarWidget(casting_card)
        casting_layout.addWidget(self.casting_progress_bar)

        self.casting_table = TableWidget(casting_card)
        self.casting_table.setColumnCount(6)
        self.casting_table.setHorizontalHeaderLabels([
            "Mã Nhân Vật",
            "Tên Nhân Vật",
            "Số Câu Thoại",
            "Giọng Đọc Trí Tuệ Nhân Tạo",
            "Trạng Thái",
            "Thao Tác",
        ])
        self.casting_table.setColumnWidth(0, 100)
        self.casting_table.setColumnWidth(1, 140)
        self.casting_table.setColumnWidth(2, 110)
        self.casting_table.setColumnWidth(3, 260)
        self.casting_table.setColumnWidth(4, 120)
        self.casting_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

        self.casting_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.casting_table.setMinimumHeight(200)

        self.casting_table.itemChanged.connect(self._on_casting_table_item_changed)

        self.casting_table.cellChanged.connect(self._on_casting_cell_changed)

        casting_layout.addWidget(self.casting_table, 1)

        p1_layout.addWidget(casting_card, 1)

        self.step_stack.addWidget(self.page_step_1)

        # ---------------------------------------------------------------------

        # BƯỚC 2: BIÊN TẬP PHỤ ĐỀ

        # ---------------------------------------------------------------------

        self.page_step_2 = QWidget(self.step_stack)

        p2_layout = QVBoxLayout(self.page_step_2)

        p2_layout.setContentsMargins(0, 0, 0, 0)

        p2_layout.setSpacing(12)

        # Khối Tổng Quan Hồ Sơ Giọng Nhân Vật

        self.char_overview_card = CardWidget(self.page_step_2)

        char_card_layout = QVBoxLayout(self.char_overview_card)

        char_card_layout.setContentsMargins(14, 12, 14, 12)

        char_card_layout.setSpacing(8)

        char_top_row = QHBoxLayout()

        self.lbl_char_overview_title = SubtitleLabel("Video đã phân tích âm thanh · Đã nhận diện", self.char_overview_card)

        self.lbl_char_overview_title.setWordWrap(True)

        self.lbl_char_count = StrongBodyLabel("Tổng số nhân vật nhận diện: 0 nhân vật", self.char_overview_card)

        self.lbl_char_count.setWordWrap(True)

        char_top_row.addWidget(self.lbl_char_overview_title)

        char_top_row.addStretch(1)

        char_top_row.addWidget(self.lbl_char_count)

        char_card_layout.addLayout(char_top_row)

        self.char_scroll = ScrollArea(self.char_overview_card)

        self.char_scroll.setWidgetResizable(True)

        self.char_scroll.setFrameShape(QFrame.NoFrame)

        self.char_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.char_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.char_scroll.setStyleSheet("background: transparent; border: none;")

        self.char_cards_container = QWidget()

        self.char_cards_container.setStyleSheet("background: transparent;")

        self.char_cards_layout = QHBoxLayout(self.char_cards_container)

        self.char_cards_layout.setContentsMargins(0, 4, 0, 4)

        self.char_cards_layout.setSpacing(10)

        self.char_cards_layout.setAlignment(Qt.AlignLeft)

        self.char_scroll.setWidget(self.char_cards_container)

        self.char_scroll.setFixedHeight(230)

        char_card_layout.addWidget(self.char_scroll)

        p2_layout.addWidget(self.char_overview_card, 0)

        self._character_cards: List[Dict[str, Any]] = []

        sub_page_card = CardWidget(self.page_step_2)

        sub_layout = QVBoxLayout(sub_page_card)

        sub_layout.setContentsMargins(14, 12, 14, 12)

        sub_layout.setSpacing(8)

        sub_top_row = QHBoxLayout()

        sub_top_row.addWidget(SubtitleLabel("Danh Sách Câu Thoại Phụ Đề", sub_page_card))

        sub_top_row.addStretch(1)

        sub_top_row.addWidget(BodyLabel("Bộ lọc:", sub_page_card))

        self.combo_sub_filter = ComboBox(sub_page_card)

        self.combo_sub_filter.addItems([

            "Tất cả câu thoại",

            "Độ tin cậy cao",

            "Cần rà soát",

            "Chưa xác định",
            "Đã khóa",
        ])

        self.combo_sub_filter.currentTextChanged.connect(self._filter_subtitle_table)

        sub_top_row.addWidget(self.combo_sub_filter)

        sub_top_row.addWidget(BodyLabel("Biểu cảm:", sub_page_card))

        self.combo_sub_tag = ComboBox(sub_page_card)

        for vn_label, tag_code in EXPRESSIVE_TAG_BUTTONS:

            self.combo_sub_tag.addItem(f"{vn_label} {tag_code}")

        self.btn_insert_tag_to_sub = PushButton("Chèn Thẻ Biểu Cảm Vào Câu Đang Chọn", sub_page_card)

        self.btn_insert_tag_to_sub.setIcon(FluentIcon.ADD)

        self.btn_insert_tag_to_sub.setEnabled(False)

        self.btn_preview_selected_row = PushButton("Nghe Thử Câu Đang Chọn", sub_page_card)
        self.btn_preview_selected_row.setIcon(FluentIcon.VOLUME)
        self.btn_preview_selected_row.setEnabled(False)

        self.btn_export_subtitles = PushButton("Xuất Tệp Phụ Đề SRT", sub_page_card)
        self.btn_export_subtitles.setIcon(FluentIcon.SAVE)
        self.btn_export_subtitles.setToolTip("Lưu danh sách câu thoại và mốc thời gian hiện tại ra tệp phụ đề định dạng SRT")
        self.btn_export_subtitles.clicked.connect(self._handle_export_subtitles_clicked)

        sub_top_row.addWidget(self.combo_sub_tag)
        sub_top_row.addWidget(self.btn_insert_tag_to_sub)
        sub_top_row.addWidget(self.btn_preview_selected_row)
        sub_top_row.addWidget(self.btn_export_subtitles)

        sub_layout.addLayout(sub_top_row)

        batch_row = QHBoxLayout()
        batch_row.setSpacing(8)
        batch_row.addWidget(BodyLabel("Thao tác hàng loạt:", sub_page_card))
        self.combo_batch_speaker = ComboBox(sub_page_card)
        self.combo_batch_speaker.setMinimumWidth(150)
        batch_row.addWidget(self.combo_batch_speaker)

        self.btn_batch_assign = PushButton("Gán Cho Các Câu Đang Chọn", sub_page_card)
        self.btn_batch_assign.setIcon(FluentIcon.PEOPLE)
        self.btn_batch_assign.setToolTip("Gán nhân vật đã chọn cho tất cả các dòng phụ đề đang được bôi đen và tự động khóa")
        self.btn_batch_assign.clicked.connect(self._batch_assign_selected_speaker)
        batch_row.addWidget(self.btn_batch_assign)

        self.btn_batch_lock = PushButton("Khóa Câu Đang Chọn", sub_page_card)
        self.btn_batch_lock.setIcon(FluentIcon.PIN)
        self.btn_batch_lock.setToolTip("Đánh dấu khóa cho tất cả các câu đang chọn để bảo vệ không bị ghi đè")
        self.btn_batch_lock.clicked.connect(lambda: self._batch_set_locked_state(True))
        batch_row.addWidget(self.btn_batch_lock)

        self.btn_batch_unlock = PushButton("Mở Khóa Câu Đang Chọn", sub_page_card)
        self.btn_batch_unlock.setIcon(FluentIcon.SYNC)
        self.btn_batch_unlock.setToolTip("Mở khóa các câu đang chọn để cho phép nhận diện lại tự động")
        self.btn_batch_unlock.clicked.connect(lambda: self._batch_set_locked_state(False))
        batch_row.addWidget(self.btn_batch_unlock)

        batch_row.addStretch(1)
        sub_layout.addLayout(batch_row)

        self.sub_table = TableWidget(sub_page_card)
        self.sub_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sub_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sub_table.setColumnCount(7)
        self.sub_table.setHorizontalHeaderLabels([
            "STT",
            "Bắt Đầu",
            "Kết Thúc",
            "Người Nói",
            "Loại Giọng & Nguồn",
            "Nghe Gốc",
            "Nội Dung Thoại",
        ])
        self.sub_table.setColumnWidth(0, 50)
        self.sub_table.setColumnWidth(1, 95)
        self.sub_table.setColumnWidth(2, 95)
        self.sub_table.setColumnWidth(3, 140)
        self.sub_table.setColumnWidth(4, 150)
        self.sub_table.setColumnWidth(5, 80)
        self.sub_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)

        self.sub_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.sub_table.setMinimumHeight(220)

        sub_layout.addWidget(self.sub_table, 1)

        p2_layout.addWidget(sub_page_card, 1)

        # Nút chuyển đổi thu gọn phòng thu văn bản trực tiếp

        self.btn_toggle_studio = PushButton("Mở Rộng Phòng Thu Văn Bản Trực Tiếp", self.page_step_2)

        self.btn_toggle_studio.setIcon(FluentIcon.CHEVRON_RIGHT_MED)

        self.btn_toggle_studio.clicked.connect(self._toggle_studio_card)

        p2_layout.addWidget(self.btn_toggle_studio)

        self.studio_card = CardWidget(self.page_step_2)

        self.studio_card.setVisible(False)

        studio_card_layout = QVBoxLayout(self.studio_card)

        studio_card_layout.setContentsMargins(14, 10, 14, 10)

        studio_card_layout.setSpacing(8)

        self.workspace_pivot = Pivot(self.studio_card)

        self.workspace_stack = QStackedWidget(self.studio_card)

        self.studio_page = QWidget(self.studio_card)

        studio_layout = QVBoxLayout(self.studio_page)

        studio_layout.setContentsMargins(0, 4, 0, 0)

        studio_layout.setSpacing(8)

        lbl_tags = BodyLabel("Chèn nhanh thẻ biểu cảm phi ngôn từ vào vị trí con trỏ:", self.studio_page)

        lbl_tags.setStyleSheet("font-weight: bold;")

        lbl_tags.setWordWrap(True)

        studio_layout.addWidget(lbl_tags)

        tags_grid = QGridLayout()

        tags_grid.setSpacing(6)

        self._studio_tag_buttons: List[PushButton] = []

        for idx, (vn_label, tag_code) in enumerate(EXPRESSIVE_TAG_BUTTONS):

            btn_tag = PushButton(f"{vn_label} {tag_code}", self.studio_page)

            btn_tag.clicked.connect(lambda checked=False, t=tag_code: self._insert_tag_into_studio(t))

            tags_grid.addWidget(btn_tag, idx // 4, idx % 4)

            self._studio_tag_buttons.append(btn_tag)

        studio_layout.addLayout(tags_grid)

        pronun_row = QHBoxLayout()

        pronun_row.setSpacing(8)

        pronun_row.addWidget(BodyLabel("Ghi đè cách phát âm nội tuyến:", self.studio_page))

        self.edit_pronun_token = LineEdit(self.studio_page)

        self.edit_pronun_token.setPlaceholderText("Nhập âm vị tiếng Anh như [B EY1 S] hoặc phiên âm tiếng Trung như CHONG2 HANG2...")

        self.btn_insert_pronun = PushButton("Chèn Phiên Âm Vào Văn Bản", self.studio_page)

        self.btn_insert_pronun.setIcon(FluentIcon.EDIT)

        pronun_row.addWidget(self.edit_pronun_token, 1)

        pronun_row.addWidget(self.btn_insert_pronun)

        studio_layout.addLayout(pronun_row)

        self.studio_text_edit = TextEdit(self.studio_page)

        self.studio_text_edit.setMinimumHeight(120)

        self.studio_text_edit.setPlaceholderText(

            "Nhập văn bản cần tổng hợp giọng nói tại đây...\n"

            "Mẹo: Bạn có thể bấm các nút phía trên để chèn tiếng cười [laughter], tiếng thở dài [sigh] hoặc ghi đè cách phát âm."

        )

        self.studio_text_edit.setPlainText(

            "Xin chào các bạn! Hôm nay chúng ta cùng trải nghiệm công nghệ tổng hợp giọng nói đa ngôn ngữ chạy trực tiếp trên card đồ họa."

        )

        studio_layout.addWidget(self.studio_text_edit, 1)

        studio_action_row = QHBoxLayout()

        self.btn_studio_synth = PrimaryPushButton("Tổng Hợp Giọng Nói Ngay", self.studio_page)

        self.btn_studio_synth.setIcon(FluentIcon.PLAY)

        self.btn_studio_save_wav = PushButton("Lưu Tệp Âm Thanh", self.studio_page)

        self.btn_studio_save_wav.setIcon(FluentIcon.SAVE)

        self.btn_studio_save_wav.setEnabled(False)

        studio_action_row.addWidget(self.btn_studio_synth)

        studio_action_row.addWidget(self.btn_studio_save_wav)

        studio_action_row.addStretch(1)

        studio_layout.addLayout(studio_action_row)

        self.workspace_stack.addWidget(self.studio_page)

        self.sub_dub_page = QWidget(self.studio_card)

        sub_dub_inner = QVBoxLayout(self.sub_dub_page)

        sub_dub_inner.addWidget(CaptionLabel("Chế độ lồng tiếng phụ đề đa nhân vật đã sẵn sàng ở bảng phía trên.", self.sub_dub_page))

        self.workspace_stack.addWidget(self.sub_dub_page)

        self.workspace_pivot.addItem(

            routeKey="studio_page",

            text="Phòng Thu Tổng Hợp Văn Bản",

            onClick=lambda: self.workspace_stack.setCurrentIndex(0),

            icon=FluentIcon.MICROPHONE,

        )

        self.workspace_pivot.addItem(

            routeKey="sub_dub_page",

            text="Lồng Tiếng Phụ Đề Đa Nhân Vật Và Ghép Video",

            onClick=lambda: self.workspace_stack.setCurrentIndex(1),

            icon=FluentIcon.VIDEO,

        )

        self.workspace_pivot.setCurrentItem("studio_page")

        studio_card_layout.addWidget(self.workspace_pivot, 0, Qt.AlignLeft)

        studio_card_layout.addWidget(self.workspace_stack, 1)

        p2_layout.addWidget(self.studio_card)

        self.step_stack.addWidget(self.page_step_2)

        # ---------------------------------------------------------------------

        # BƯỚC 3: CÂN CHỈNH ÂM THANH

        # ---------------------------------------------------------------------

        self.page_step_3 = QWidget(self.step_stack)

        p3_layout = QVBoxLayout(self.page_step_3)

        p3_layout.setContentsMargins(0, 0, 0, 0)

        p3_layout.setSpacing(12)

        audio_card = CardWidget(self.page_step_3)

        audio_layout = QVBoxLayout(audio_card)

        audio_layout.setContentsMargins(18, 14, 18, 14)

        audio_layout.setSpacing(10)

        audio_layout.addWidget(SubtitleLabel("Tham Số Âm Thanh Và Hòa Trộn", audio_card))

        param_grid = QGridLayout()

        param_grid.setSpacing(8)

        param_grid.addWidget(BodyLabel("Tốc độ phát ngôn:", audio_card), 0, 0)

        self.spin_speed = DoubleSpinBox(audio_card)

        self.spin_speed.setRange(0.8, 1.5)

        self.spin_speed.setSingleStep(0.05)

        self.spin_speed.setValue(1.0)

        param_grid.addWidget(self.spin_speed, 0, 1)

        param_grid.addWidget(BodyLabel("Thời lượng cố định theo giây:", audio_card), 0, 2)

        self.spin_duration = DoubleSpinBox(audio_card)

        self.spin_duration.setRange(0.0, 120.0)

        self.spin_duration.setSingleStep(0.5)

        self.spin_duration.setValue(0.0)

        param_grid.addWidget(self.spin_duration, 0, 3)

        param_grid.addWidget(BodyLabel("Âm lượng video gốc:", audio_card), 1, 0)

        self.spin_duck_vol = DoubleSpinBox(audio_card)

        self.spin_duck_vol.setRange(0.0, 1.0)

        self.spin_duck_vol.setSingleStep(0.05)

        self.spin_duck_vol.setValue(0.15)

        param_grid.addWidget(self.spin_duck_vol, 1, 1)

        param_grid.addWidget(BodyLabel("Chế độ hòa trộn nhạc nền:", audio_card), 1, 2)

        self.combo_ducking_mode = ComboBox(audio_card)

        self.combo_ducking_mode.addItems([

            "Tự động giảm nhạc nền khi có thoại",

            "Giảm âm lượng nhạc nền cố định",

        ])

        param_grid.addWidget(self.combo_ducking_mode, 1, 3)

        audio_layout.addLayout(param_grid)

        chk_grid = QGridLayout()

        chk_grid.setSpacing(8)

        self.chk_sync_sub_duration = CheckBox("Tự động khớp thời lượng theo từng câu phụ đề", audio_card)

        self.chk_sync_sub_duration.setChecked(True)

        chk_grid.addWidget(self.chk_sync_sub_duration, 0, 0)

        self.chk_denoise = CheckBox("Khử nhiễu giọng đọc đầu ra", audio_card)

        self.chk_denoise.setChecked(True)

        chk_grid.addWidget(self.chk_denoise, 0, 1)

        self.chk_preprocess_prompt = CheckBox("Tiền xử lý cắt khoảng lặng giọng mẫu", audio_card)

        self.chk_preprocess_prompt.setChecked(True)

        chk_grid.addWidget(self.chk_preprocess_prompt, 1, 0)

        self.chk_postprocess_output = CheckBox("Hậu xử lý làm mượt âm thanh đầu ra", audio_card)

        self.chk_postprocess_output.setChecked(True)

        chk_grid.addWidget(self.chk_postprocess_output, 1, 1)

        self.chk_normalize_text = CheckBox("Chuẩn hóa chữ số và ngày tháng thành chữ", audio_card)

        self.chk_normalize_text.setChecked(True)

        chk_grid.addWidget(self.chk_normalize_text, 2, 0)

        self.chk_flashinfer = CheckBox("Tăng tốc phần cứng FlashInfer trên card đồ họa", audio_card)

        self.chk_flashinfer.setChecked(False)

        chk_grid.addWidget(self.chk_flashinfer, 2, 1)

        audio_layout.addLayout(chk_grid)

        p3_layout.addWidget(audio_card)

        adv_card = CardWidget(self.page_step_3)

        adv_layout = QVBoxLayout(adv_card)

        adv_layout.setContentsMargins(18, 14, 18, 14)

        adv_layout.setSpacing(8)

        adv_header_row = QHBoxLayout()

        adv_header_row.addWidget(SubtitleLabel("Phần Cứng Nhận Diện Và Tham Số Nâng Cao", adv_card))

        adv_header_row.addStretch(1)

        self.btn_toggle_adv = PushButton("Thu Gọn / Mở Rộng", adv_card)

        self.btn_toggle_adv.setIcon(FluentIcon.CHEVRON_RIGHT)

        adv_header_row.addWidget(self.btn_toggle_adv)

        adv_layout.addLayout(adv_header_row)

        self.adv_content_widget = QWidget(adv_card)

        adv_content_layout = QVBoxLayout(self.adv_content_widget)

        adv_content_layout.setContentsMargins(0, 4, 0, 0)

        adv_content_layout.setSpacing(8)

        hw_row = QHBoxLayout()

        hw_row.addWidget(BodyLabel("Bộ xử lý nhận diện người nói:", self.adv_content_widget))

        self.combo_hardware_selector = ComboBox(self.adv_content_widget)

        self.combo_hardware_selector.addItems([

            "Ưu tiên card đồ họa NVIDIA RTX 4050",

            "Chế độ vi xử lý trung tâm",

            "Phân đoạn người nói chuyên sâu — Chưa kích hoạt",

        ])

        self.combo_hardware_selector.setCurrentIndex(0)

        self.combo_hardware_selector.setMinimumWidth(320)

        hw_row.addWidget(self.combo_hardware_selector, 1)

        adv_content_layout.addLayout(hw_row)

        diff_grid = QGridLayout()

        diff_grid.setSpacing(6)

        diff_grid.addWidget(BodyLabel("Số bước giải mã khuếch tán:", self.adv_content_widget), 0, 0)

        self.spin_num_step = SpinBox(self.adv_content_widget)

        self.spin_num_step.setRange(4, 64)

        self.spin_num_step.setValue(32)

        diff_grid.addWidget(self.spin_num_step, 0, 1)

        diff_grid.addWidget(BodyLabel("Hệ số dẫn hướng:", self.adv_content_widget), 0, 2)

        self.spin_guidance = DoubleSpinBox(self.adv_content_widget)

        self.spin_guidance.setRange(0.0, 4.0)

        self.spin_guidance.setSingleStep(0.1)

        self.spin_guidance.setValue(2.0)

        diff_grid.addWidget(self.spin_guidance, 0, 3)

        diff_grid.addWidget(BodyLabel("Độ dịch bước thời gian:", self.adv_content_widget), 1, 0)

        self.spin_t_shift = DoubleSpinBox(self.adv_content_widget)

        self.spin_t_shift.setRange(0.01, 1.0)

        self.spin_t_shift.setSingleStep(0.05)

        self.spin_t_shift.setValue(0.10)

        diff_grid.addWidget(self.spin_t_shift, 1, 1)

        diff_grid.addWidget(BodyLabel("Hệ số phạt tầng mã hóa:", self.adv_content_widget), 1, 2)

        self.spin_layer_penalty = DoubleSpinBox(self.adv_content_widget)

        self.spin_layer_penalty.setRange(0.0, 20.0)

        self.spin_layer_penalty.setSingleStep(0.5)

        self.spin_layer_penalty.setValue(5.0)

        diff_grid.addWidget(self.spin_layer_penalty, 1, 3)

        diff_grid.addWidget(BodyLabel("Nhiệt độ chọn vị trí:", self.adv_content_widget), 2, 0)

        self.spin_pos_temp = DoubleSpinBox(self.adv_content_widget)

        self.spin_pos_temp.setRange(0.0, 10.0)

        self.spin_pos_temp.setSingleStep(0.5)

        self.spin_pos_temp.setValue(5.0)

        diff_grid.addWidget(self.spin_pos_temp, 2, 1)

        diff_grid.addWidget(BodyLabel("Nhiệt độ lấy mẫu lớp:", self.adv_content_widget), 2, 2)

        self.spin_cls_temp = DoubleSpinBox(self.adv_content_widget)

        self.spin_cls_temp.setRange(0.0, 2.0)

        self.spin_cls_temp.setSingleStep(0.1)

        self.spin_cls_temp.setValue(0.0)

        diff_grid.addWidget(self.spin_cls_temp, 2, 3)

        diff_grid.addWidget(BodyLabel("Thời lượng chia đoạn dài:", self.adv_content_widget), 3, 0)

        self.spin_chunk_dur = DoubleSpinBox(self.adv_content_widget)

        self.spin_chunk_dur.setRange(5.0, 60.0)

        self.spin_chunk_dur.setSingleStep(1.0)

        self.spin_chunk_dur.setValue(15.0)

        diff_grid.addWidget(self.spin_chunk_dur, 3, 1)

        diff_grid.addWidget(BodyLabel("Ngưỡng tự chia đoạn dài:", self.adv_content_widget), 3, 2)

        self.spin_chunk_thresh = DoubleSpinBox(self.adv_content_widget)

        self.spin_chunk_thresh.setRange(10.0, 120.0)

        self.spin_chunk_thresh.setSingleStep(5.0)

        self.spin_chunk_thresh.setValue(30.0)

        diff_grid.addWidget(self.spin_chunk_thresh, 3, 3)

        diff_grid.addWidget(BodyLabel("Khoảng lặng đệm hai đầu:", self.adv_content_widget), 4, 0)

        self.spin_pad_dur = DoubleSpinBox(self.adv_content_widget)

        self.spin_pad_dur.setRange(0.0, 2.0)

        self.spin_pad_dur.setSingleStep(0.05)

        self.spin_pad_dur.setValue(0.10)

        diff_grid.addWidget(self.spin_pad_dur, 4, 1)

        diff_grid.addWidget(BodyLabel("Thời gian làm mượt đầu cuối:", self.adv_content_widget), 4, 2)

        self.spin_fade_dur = DoubleSpinBox(self.adv_content_widget)

        self.spin_fade_dur.setRange(0.0, 1.0)

        self.spin_fade_dur.setSingleStep(0.05)

        self.spin_fade_dur.setValue(0.10)

        diff_grid.addWidget(self.spin_fade_dur, 4, 3)

        adv_content_layout.addLayout(diff_grid)

        adv_layout.addWidget(self.adv_content_widget)

        self.btn_toggle_adv.clicked.connect(self._toggle_advanced_settings)

        p3_layout.addWidget(adv_card)

        self.step_stack.addWidget(self.page_step_3)

        # ---------------------------------------------------------------------

        # BƯỚC 4: KIỂM TRA VÀ XUẤT

        # ---------------------------------------------------------------------

        self.page_step_4 = QWidget(self.step_stack)

        p4_layout = QVBoxLayout(self.page_step_4)

        p4_layout.setContentsMargins(0, 0, 0, 0)

        p4_layout.setSpacing(12)

        qc_card = CardWidget(self.page_step_4)

        qc_layout = QVBoxLayout(qc_card)

        qc_layout.setContentsMargins(18, 14, 18, 14)

        qc_layout.setSpacing(10)

        qc_layout.addWidget(SubtitleLabel("Kiểm Tra Chất Lượng Trước Khi Xuất", qc_card))

        qc_row = QHBoxLayout()

        self.btn_run_qc = PrimaryPushButton("Kiểm Tra Chất Lượng Trước Xuất", qc_card)

        self.btn_run_qc.setIcon(FluentIcon.ACCEPT)

        qc_row.addWidget(self.btn_run_qc)

        qc_row.addStretch(1)

        qc_layout.addLayout(qc_row)

        self.lbl_qc_result = BodyLabel("Bấm nút kiểm tra phía trên để rà soát toàn bộ câu thoại trước khi xuất.", qc_card)

        self.lbl_qc_result.setWordWrap(True)

        qc_layout.addWidget(self.lbl_qc_result)

        p4_layout.addWidget(qc_card)

        export_card = CardWidget(self.page_step_4)

        export_layout = QVBoxLayout(export_card)

        export_layout.setContentsMargins(18, 14, 18, 14)

        export_layout.setSpacing(10)

        export_layout.addWidget(SubtitleLabel("Xuất Bản Âm Thanh Và Video Lồng Tiếng", export_card))

        export_action_row = QHBoxLayout()

        self.btn_export_dub_wav = PushButton("Xuất Tệp Âm Thanh Lồng Tiếng", export_card)

        self.btn_export_dub_wav.setIcon(FluentIcon.MUSIC)

        self.btn_export_dub_wav.setEnabled(False)

        self.btn_dub_merge_video = PrimaryPushButton("Lồng Tiếng Và Xuất Video", export_card)

        self.btn_dub_merge_video.setIcon(FluentIcon.VIDEO)

        self.btn_dub_merge_video.setEnabled(False)

        export_action_row.addWidget(self.btn_export_dub_wav)

        export_action_row.addWidget(self.btn_dub_merge_video)

        export_action_row.addStretch(1)

        export_layout.addLayout(export_action_row)

        p4_layout.addWidget(export_card)

        player_card = CardWidget(self.page_step_4)

        player_layout = QHBoxLayout(player_card)

        player_layout.setContentsMargins(14, 10, 14, 10)

        player_layout.setSpacing(10)

        self.btn_play_pause_output = PushButton("Phát Nghe Thử", player_card)

        self.btn_play_pause_output.setIcon(FluentIcon.PLAY)

        self.btn_play_pause_output.setEnabled(False)

        self.slider_audio_pos = Slider(Qt.Horizontal, player_card)

        self.slider_audio_pos.setRange(0, 1000)

        self.slider_audio_pos.setValue(0)

        self.lbl_audio_time = CaptionLabel("00:00 / 00:00", player_card)

        player_layout.addWidget(BodyLabel("Trình phát kết quả:", player_card))

        player_layout.addWidget(self.btn_play_pause_output)

        player_layout.addWidget(self.slider_audio_pos, 1)

        player_layout.addWidget(self.lbl_audio_time)

        p4_layout.addWidget(player_card)

        log_card = CardWidget(self.page_step_4)

        log_layout = QVBoxLayout(log_card)

        log_layout.setContentsMargins(14, 10, 14, 10)

        log_layout.setSpacing(6)

        log_layout.addWidget(BodyLabel("Nhật ký xử lý trên card đồ họa:", log_card))

        self.log_edit = TextEdit(log_card)

        self.log_edit.setReadOnly(True)

        self.log_edit.setMinimumHeight(140)

        self.log_edit.setPlaceholderText("Nhật ký vận hành OmniVoice trên card đồ họa sẽ hiển thị tại đây...")

        log_layout.addWidget(self.log_edit, 1)

        p4_layout.addWidget(log_card, 1)

        self.step_stack.addWidget(self.page_step_4)

        root_layout.addWidget(self.step_stack, 1)

        # =====================================================================

        # 4. THANH TIẾN TRÌNH PHÍA DƯỚI

        # =====================================================================

        bottom_card = CardWidget(self.container)

        bottom_layout = QHBoxLayout(bottom_card)

        bottom_layout.setContentsMargins(18, 10, 18, 10)

        bottom_layout.setSpacing(12)

        self.progress_bar = ProgressBar(bottom_card)

        self.progress_bar.setValue(0)

        self.lbl_status = BodyLabel("Sẵn sàng tổng hợp giọng nói trên GPU.", bottom_card)

        self.lbl_status.setWordWrap(True)

        bottom_layout.addWidget(self.progress_bar, 1)

        bottom_layout.addWidget(self.lbl_status)

        root_layout.addWidget(bottom_card)

    def _populate_languages(self):

        """Nạp danh sách ngôn ngữ ưu tiên kèm toàn bộ 646 ngôn ngữ của OmniVoice."""

        self.combo_lang.clear()

        for label in PRIORITY_LANGUAGE_MAP.keys():

            self.combo_lang.addItem(label)

        all_langs = TTSEngine.get_supported_languages()

        for lang_item in all_langs:

            if lang_item != "Auto":

                self.combo_lang.addItem(lang_item)

        self.combo_lang.setCurrentIndex(1)

    def _connect_signals(self):

        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.btn_cancel_task.clicked.connect(self._cancel_running_task)

        self.btn_free_vram.clicked.connect(self._free_gpu_vram)

        # Tín hiệu Kho 22 Giọng Đọc Mẫu Tiếng Việt & Nghe Thử

        self.combo_vn_catalog.currentIndexChanged.connect(self._on_vn_catalog_changed)

        self.btn_preview_vn_voice.clicked.connect(self._preview_vn_catalog_voice)

        self.btn_use_vn_voice_as_ref.clicked.connect(self._lock_vn_catalog_voice_as_ref)

        # Tín hiệu nhóm Nhân bản giọng nói

        self.btn_pick_ref_audio.clicked.connect(self._pick_ref_audio)

        self.btn_extract_ref_video.clicked.connect(self._extract_ref_from_video_dialog)

        self.btn_play_ref_audio.clicked.connect(self._play_ref_audio)

        self.btn_load_prompt_pt.clicked.connect(self._load_prompt_pt_dialog)

        self.btn_save_prompt_pt.clicked.connect(self._save_prompt_pt_dialog)

        self.btn_transcribe_ref.clicked.connect(self._transcribe_ref_audio)

        self.chk_auto_extract_video.stateChanged.connect(self._update_control_states)

        # Tín hiệu nhóm Thiết kế giọng nói

        for cb in [

            self.combo_gender,

            self.combo_age,

            self.combo_pitch,

            self.combo_style,

            self.combo_accent,

            self.combo_dialect,

        ]:

            cb.currentIndexChanged.connect(self._rebuild_instruct_string)

        # Tín hiệu chế độ một giọng đọc đồng nhất
        self.chk_single_unified_voice.stateChanged.connect(self._on_single_unified_voice_changed)
        self.combo_single_unified_preset.currentTextChanged.connect(self._on_unified_preset_changed)
        self.btn_preview_single_unified.clicked.connect(self._preview_unified_voice)

        # Tín hiệu Phòng thu văn bản

        self.btn_insert_pronun.clicked.connect(self._insert_pronunciation_into_studio)

        self.btn_studio_synth.clicked.connect(self._run_studio_synthesis)

        self.btn_studio_save_wav.clicked.connect(self._save_studio_wav)

        # Tín hiệu Lồng tiếng phụ đề đa nhân vật & video

        self.btn_load_srt.clicked.connect(self._select_subtitle_file)

        self.btn_load_video.clicked.connect(self._select_video_file)

        self.btn_clear_video.clicked.connect(self._clear_selected_video)

        self.btn_identify_characters.clicked.connect(self._run_auto_detect_speakers)

        self.combo_multi_speaker_mode.currentIndexChanged.connect(self._update_control_states)

        self.combo_speaker_count_mode.currentIndexChanged.connect(self._on_speaker_count_mode_changed)

        self.btn_auto_detect_speakers.clicked.connect(self._run_auto_detect_speakers)

        self.btn_preview_speaker_preset_vi.clicked.connect(self._preview_speaker_preset_vi)

        self.btn_apply_preset_to_speaker.clicked.connect(self._apply_preset_to_target_speaker)

        self.btn_pick_audio_for_speaker.clicked.connect(self._pick_audio_for_target_speaker)

        self.btn_assign_speaker_to_rows.clicked.connect(self._assign_target_speaker_to_selected_rows)

        self.btn_reset_speaker_voice.clicked.connect(self._reset_target_speaker_locked_voice)

        self.btn_insert_tag_to_sub.clicked.connect(self._insert_expressive_tag_to_selected_sub)

        self.sub_table.itemSelectionChanged.connect(self._update_subtitle_action_buttons)

        self.sub_table.itemChanged.connect(self._on_subtitle_table_edited)

        self.btn_preview_selected_row.clicked.connect(self._preview_selected_subtitle_row)

        self.btn_export_dub_wav.clicked.connect(self._export_subtitle_audio_track)

        self.btn_dub_merge_video.clicked.connect(self._dub_and_merge_video)

        # Tín hiệu khóa/mở khóa tốc độ khi bật Thời lượng cố định

        self.spin_duration.valueChanged.connect(self._update_control_states)

        self.chk_sync_sub_duration.stateChanged.connect(self._update_control_states)

        # Tín hiệu thanh điều hướng 5 bước

        self.btn_prev_step.clicked.connect(self._go_prev_step)

        self.btn_next_step.clicked.connect(self._go_next_step)

        # Tín hiệu bảng phân vai nhân vật (Bước 1)

        self.btn_reanalyze_speakers.clicked.connect(self._run_auto_detect_speakers)

        self.btn_reset_all_voices.clicked.connect(lambda: self._handle_reset_all_voices_with_confirmation(None))

        self.btn_audition_all_speakers.clicked.connect(self._audition_all_speakers)

        # Tín hiệu kiểm tra chất lượng trước xuất (Bước 4)

        self.btn_run_qc.clicked.connect(self._on_run_pre_export_qc)

        # Tín hiệu Trình phát kết quả

        self.btn_play_pause_output.clicked.connect(self._toggle_play_output)

        self._media_player.positionChanged.connect(self._on_player_position_changed)

        self._media_player.durationChanged.connect(self._on_player_duration_changed)

        self._media_player.playbackStateChanged.connect(self._on_player_state_changed)

        self._media_player.mediaStatusChanged.connect(self._on_player_media_status_changed)

        self.slider_audio_pos.sliderMoved.connect(self._on_slider_moved)

    def _sync_ui_from_config(self):

        """Đồng bộ các giá trị ban đầu từ cấu hình hệ thống."""

        mode_map = {"clone": 0, "design": 1, "auto": 2}

        self.combo_mode.setCurrentIndex(mode_map.get(config.mmoTTSMode.value, 0))

        self.voice_source_stack.setCurrentIndex(self.combo_mode.currentIndex())

        self.chk_auto_extract_video.setChecked(bool(config.mmoTTSAutoExtractRef.value))

        if self.current_ref_audio_path and os.path.exists(self.current_ref_audio_path):

            self.lbl_ref_status.setText(f"Tệp giọng mẫu hiện tại: {self.current_ref_audio_path}")

        self.edit_ref_text.setText(str(config.mmoTTSRefText.value or ""))

        self.spin_speed.setValue(float(config.mmoTTSSpeed.value))

        self.spin_num_step.setValue(int(config.mmoTTSNumStep.value))

        self.spin_guidance.setValue(float(config.mmoTTSGuidanceScale.value))

        self.spin_duck_vol.setValue(float(config.mmoAudioDuckingVolume.value))

        self.chk_sync_sub_duration.setChecked(bool(config.mmoTTSSyncDuration.value))

        self.chk_denoise.setChecked(bool(config.mmoTTSDenoise.value))

        self.chk_preprocess_prompt.setChecked(bool(config.mmoTTSPreprocessPrompt.value))

        self.chk_postprocess_output.setChecked(bool(config.mmoTTSPostprocessOutput.value))

        self.chk_normalize_text.setChecked(bool(config.mmoTTSNormalizeText.value))

        self.chk_flashinfer.setChecked(bool(config.mmoTTSFlashInfer.value))

        if hasattr(config, "mmoTTSTShift"):

            self.spin_t_shift.setValue(float(config.mmoTTSTShift.value))

        if hasattr(config, "mmoTTSLayerPenalty"):

            self.spin_layer_penalty.setValue(float(config.mmoTTSLayerPenalty.value))

        if hasattr(config, "mmoTTSPosTemp"):

            self.spin_pos_temp.setValue(float(config.mmoTTSPosTemp.value))

        if hasattr(config, "mmoTTSClsTemp"):

            self.spin_cls_temp.setValue(float(config.mmoTTSClsTemp.value))

        if hasattr(config, "mmoTTSChunkDuration"):

            self.spin_chunk_dur.setValue(float(config.mmoTTSChunkDuration.value))

        if hasattr(config, "mmoTTSChunkThreshold"):

            self.spin_chunk_thresh.setValue(float(config.mmoTTSChunkThreshold.value))

        if hasattr(config, "mmoTTSPadDuration"):

            self.spin_pad_dur.setValue(float(config.mmoTTSPadDuration.value))

        if hasattr(config, "mmoTTSFadeDuration"):

            self.spin_fade_dur.setValue(float(config.mmoTTSFadeDuration.value))

        if hasattr(config, "mmoTTSMaxSpeakers"):

            self.spin_max_speakers.setValue(int(config.mmoTTSMaxSpeakers.value))

        if hasattr(config, "mmoTTSDuckingMode"):

            self.combo_ducking_mode.setCurrentIndex(0 if config.mmoTTSDuckingMode.value == "dynamic" else 1)

        if hasattr(config, "mmoTTSMultiSpeakerMode"):

            saved_ms = str(config.mmoTTSMultiSpeakerMode.value)

            for idx, (_, code) in enumerate(MULTI_SPEAKER_MODE_ITEMS):

                if code == saved_ms:

                    self.combo_multi_speaker_mode.setCurrentIndex(idx)

                    break

        self._update_control_states()

    def _save_ui_to_config(self):

        """Lưu các tham số từ giao diện vào cấu hình chung."""

        idx_to_mode = {0: "clone", 1: "design", 2: "auto"}

        config.set(config.mmoTTSMode, idx_to_mode.get(self.combo_mode.currentIndex(), "clone"))

        config.set(config.mmoTTSAutoExtractRef, self.chk_auto_extract_video.isChecked())

        config.set(config.mmoTTSRefAudioPath, self.current_ref_audio_path)

        config.set(config.mmoTTSRefText, self.edit_ref_text.text().strip())

        config.set(config.mmoTTSSpeed, float(self.spin_speed.value()))

        config.set(config.mmoTTSNumStep, int(self.spin_num_step.value()))

        config.set(config.mmoTTSGuidanceScale, float(self.spin_guidance.value()))

        config.set(config.mmoAudioDuckingVolume, float(self.spin_duck_vol.value()))

        config.set(config.mmoTTSSyncDuration, self.chk_sync_sub_duration.isChecked())

        config.set(config.mmoTTSDenoise, self.chk_denoise.isChecked())
        if hasattr(self, "chk_ai_vocal_separation") and hasattr(config, "mmoTTSAIVocalSeparation"):
            config.set(config.mmoTTSAIVocalSeparation, self.chk_ai_vocal_separation.isChecked())

        config.set(config.mmoTTSPreprocessPrompt, self.chk_preprocess_prompt.isChecked())

        config.set(config.mmoTTSPostprocessOutput, self.chk_postprocess_output.isChecked())

        config.set(config.mmoTTSNormalizeText, self.chk_normalize_text.isChecked())

        config.set(config.mmoTTSFlashInfer, self.chk_flashinfer.isChecked())

        if hasattr(config, "mmoTTSMultiSpeakerMode"):

            config.set(config.mmoTTSMultiSpeakerMode, self._get_selected_multi_speaker_mode())

        if hasattr(config, "mmoTTSMaxSpeakers"):

            config.set(config.mmoTTSMaxSpeakers, int(self.spin_max_speakers.value()))

        if hasattr(config, "mmoTTSDuckingMode"):

            config.set(config.mmoTTSDuckingMode, "dynamic" if self.combo_ducking_mode.currentIndex() == 0 else "fixed")

        if hasattr(config, "mmoTTSTShift"):

            config.set(config.mmoTTSTShift, float(self.spin_t_shift.value()))

        if hasattr(config, "mmoTTSLayerPenalty"):

            config.set(config.mmoTTSLayerPenalty, float(self.spin_layer_penalty.value()))

        if hasattr(config, "mmoTTSPosTemp"):

            config.set(config.mmoTTSPosTemp, float(self.spin_pos_temp.value()))

        if hasattr(config, "mmoTTSClsTemp"):

            config.set(config.mmoTTSClsTemp, float(self.spin_cls_temp.value()))

        if hasattr(config, "mmoTTSChunkDuration"):

            config.set(config.mmoTTSChunkDuration, float(self.spin_chunk_dur.value()))

        if hasattr(config, "mmoTTSChunkThreshold"):

            config.set(config.mmoTTSChunkThreshold, float(self.spin_chunk_thresh.value()))

        if hasattr(config, "mmoTTSPadDuration"):

            config.set(config.mmoTTSPadDuration, float(self.spin_pad_dur.value()))

        if hasattr(config, "mmoTTSFadeDuration"):

            config.set(config.mmoTTSFadeDuration, float(self.spin_fade_dur.value()))

        if hasattr(config, "mmoTTSInstructCustom"):

            config.set(config.mmoTTSInstructCustom, self.edit_instruct_preview.text().strip())

    def _toggle_default_voice_settings(self):

        """Chuyển đổi trạng thái thu gọn / mở rộng của khối thiết lập giọng mặc định ở Bước 0."""

        if not hasattr(self, "default_voice_settings_container"):

            return

        is_vis = not self.default_voice_settings_container.isVisible()

        self.default_voice_settings_container.setVisible(is_vis)

        if hasattr(self, "btn_toggle_default_voice_settings"):

            self.btn_toggle_default_voice_settings.setIcon(

                FluentIcon.CHEVRON_DOWN_MED if is_vis else FluentIcon.CHEVRON_RIGHT_MED

            )

    def _toggle_studio_card(self):

        """Chuyển đổi trạng thái thu gọn / mở rộng của khối phòng thu trực tiếp ở Bước 2."""

        if not hasattr(self, "studio_card"):

            return

        is_vis = not self.studio_card.isVisible()

        self.studio_card.setVisible(is_vis)

        if hasattr(self, "btn_toggle_studio"):

            self.btn_toggle_studio.setIcon(

                FluentIcon.CHEVRON_DOWN_MED if is_vis else FluentIcon.CHEVRON_RIGHT_MED

            )

    def _on_mode_changed(self, index: int):

        self.voice_source_stack.setCurrentIndex(index)

        if index == 2:

            self.voice_source_stack.setVisible(False)

        else:

            self.voice_source_stack.setVisible(True)

        self._save_ui_to_config()

        self._update_control_states()

    def _get_selected_multi_speaker_mode(self) -> str:

        idx = self.combo_multi_speaker_mode.currentIndex()

        if 0 <= idx < len(MULTI_SPEAKER_MODE_ITEMS):

            return MULTI_SPEAKER_MODE_ITEMS[idx][1]

        return "single"

    def _update_control_states(self):

        """Tự động khóa các điều khiển thủ công khi tính năng tự động tương ứng được bật."""

        has_ref = bool(self.current_ref_audio_path and os.path.exists(self.current_ref_audio_path))

        self.btn_play_ref_audio.setEnabled(has_ref and not self.current_ref_audio_path.lower().endswith(".pt"))

        self.btn_transcribe_ref.setEnabled(has_ref and not self.current_ref_audio_path.lower().endswith(".pt"))

        self.btn_save_prompt_pt.setEnabled(has_ref)

        fixed_dur_active = self.spin_duration.value() > 0.0

        self.spin_speed.setEnabled(not fixed_dur_active)

        ms_mode = self._get_selected_multi_speaker_mode()

        is_per_line_clone = ms_mode == "per_line_video_clone"

        self.spin_max_speakers.setEnabled(ms_mode not in ("single", "per_line_video_clone"))

    # =========================================================================

    # QUY TRÌNH ĐIỀU HƯỚNG 5 BƯỚC & QUẢN LÝ BẢNG NHÂN VẬT

    # =========================================================================

    def _can_switch_to_step(self, target_idx: int) -> bool:

        """Kiểm tra điều kiện chuyển sang bước mục tiêu."""

        if target_idx == 0:

            return True

        if target_idx == 1:

            # Cho phép chuyển sang Bước 2 nếu có phụ đề, hoặc có video, hoặc đã có nhân vật

            if not self.subtitle_blocks and not self.current_video_path and not self.speaker_profiles:

                InfoBar.warning(

                    title="Chưa có dữ liệu nguồn",

                    content="Vui lòng nạp tệp phụ đề hoặc tệp video ở Bước 1 trước khi quản lý nhân vật.",

                    parent=self,

                    duration=3000,

                )

                return False

            return True

        if target_idx in (2, 4):

            if not self.subtitle_blocks:

                InfoBar.warning(

                    title="Chưa nạp phụ đề",

                    content="Vui lòng nạp tệp phụ đề ở Bước 1 trước khi tiếp tục.",

                    parent=self,

                    duration=3000,

                )

                return False

            return True

        if target_idx == 3:

            # Bước 4: Phòng thu và cân chỉnh âm thanh luôn sẵn sàng

            return True

        return True

    def _update_step_navigation_states(self):

        """Cập nhật hiển thị màu sắc và trạng thái của các nút bước trong thanh điều hướng."""

        has_subs = len(self.subtitle_blocks) > 0

        has_video = bool(self.current_video_path and os.path.exists(self.current_video_path))

        has_cast = len(self.speaker_profiles) > 0

        for idx, btn in enumerate(self.step_buttons):

            if not self._is_valid_widget(btn):

                continue

            if idx == self.current_step_index:

                btn.setStyleSheet("font-weight: bold; background-color: #0078D4; color: white; border-radius: 6px;")

            else:

                is_completed = False

                if idx == 0 and (has_subs or has_video):

                    is_completed = True

                elif idx == 1 and has_cast:

                    is_completed = True

                elif idx == 2 and has_subs:

                    is_completed = True

                elif idx == 3 and (has_subs or getattr(self, "last_generated_audio_path", None)):

                    is_completed = True

                elif idx == 4 and getattr(self, "last_exported_video_path", None):

                    is_completed = True

                if is_completed:

                    btn.setStyleSheet("color: #107C41; font-weight: 500;")

                else:

                    btn.setStyleSheet("")

    def _on_step_button_clicked(self, step_idx: int):

        """Xử lý sự kiện người dùng bấm vào nút bước trong thanh điều hướng."""

        if self._can_switch_to_step(step_idx):

            self._switch_to_step(step_idx)

    def _switch_to_step(self, step_idx: int):

        """Chuyển đổi giữa 5 bước quy trình giao diện và bảo toàn trạng thái dữ liệu."""

        if not (0 <= step_idx < self.step_stack.count()):

            return

        self.current_step_index = step_idx

        self.step_stack.setCurrentIndex(step_idx)

        self.btn_prev_step.setEnabled(step_idx > 0)

        self.btn_next_step.setEnabled(step_idx < self.step_stack.count() - 1)

        self._update_step_navigation_states()

        if step_idx == 1:

            self._update_casting_table()

        elif step_idx == 2:

            self._populate_subtitle_table()

        elif step_idx == 4:

            self._update_export_card_state()

    def _go_prev_step(self):

        if self.current_step_index > 0:

            self._switch_to_step(self.current_step_index - 1)

    def _go_next_step(self):

        if self.current_step_index < self.step_stack.count() - 1:

            target = self.current_step_index + 1

            if self._can_switch_to_step(target):

                self._switch_to_step(target)

    def _update_export_card_state(self):

        has_subs = len(self.subtitle_blocks) > 0

        has_video = bool(self.current_video_path and os.path.exists(self.current_video_path))

        self.btn_export_dub_wav.setEnabled(has_subs)

        self.btn_dub_merge_video.setEnabled(has_subs and has_video)

    def _sync_speaker_selectors(self):

        """Đồng bộ danh sách tên nhân vật thực tế vào toàn bộ các bộ chọn trên giao diện."""

        if not hasattr(self, "combo_target_speaker"):

            return

        current_choice = self.combo_target_speaker.currentText().strip()

        names: List[str] = []

        if self.speaker_profiles:

            for spk_name, prof in self.speaker_profiles.items():

                disp = getattr(prof, "display_name", "") or (prof.get("display_name") if isinstance(prof, dict) else "") or spk_name

                if disp and disp not in names:

                    names.append(disp)

        elif self.subtitle_blocks:

            for blk in self.subtitle_blocks:

                spk = getattr(blk, "speaker", "")

                if spk and spk not in names:

                    names.append(spk)

        if not names:

            names = [f"Người nói {i}" for i in range(1, 7)]

        self.combo_target_speaker.blockSignals(True)

        self.combo_target_speaker.clear()

        self.combo_target_speaker.addItems(names)

        if current_choice in names:

            self.combo_target_speaker.setCurrentText(current_choice)

        elif names:

            self.combo_target_speaker.setCurrentIndex(0)

        self.combo_target_speaker.blockSignals(False)

        if hasattr(self, "combo_batch_speaker"):
            current_batch = self.combo_batch_speaker.currentText().strip()
            self.combo_batch_speaker.blockSignals(True)
            self.combo_batch_speaker.clear()
            self.combo_batch_speaker.addItems(names)
            if current_batch in names:
                self.combo_batch_speaker.setCurrentText(current_batch)
            elif names:
                self.combo_batch_speaker.setCurrentIndex(0)
            self.combo_batch_speaker.blockSignals(False)

    def _update_casting_table(self):

        """Đồng bộ danh sách nhân vật vào bảng Voice Casting ở Bước 2."""

        if not hasattr(self, "casting_table"):

            return

        self.casting_table.blockSignals(True)

        self.casting_table.setRowCount(0)

        if self.speaker_profiles:

            normalized: Dict[str, CharacterVoiceProfile] = {}

            for idx, (spk, prof) in enumerate(self.speaker_profiles.items()):

                sid = getattr(prof, "speaker_id", "") or (prof.get("speaker_id") if isinstance(prof, dict) else "") or f"spk_{idx}"

                norm = normalize_character_profile(prof, speaker_id=sid)

                if not getattr(norm, "display_name", ""):

                    norm.display_name = str(spk)

                normalized[spk] = norm

            self.speaker_profiles = normalized

        speaker_map: Dict[str, Dict[str, Any]] = {}

        for blk in self.subtitle_blocks:

            spk = getattr(blk, "speaker", "") or "Người nói 1"

            if spk not in speaker_map:

                speaker_map[spk] = {

                    "count": 0,

                    "confidence": getattr(blk, "confidence_level", "high"),

                    "ref_audio": getattr(blk, "voice_ref_path", ""),

                    "voice_instruct": getattr(blk, "voice_instruct", ""),

                }

            speaker_map[spk]["count"] += 1

            conf = getattr(blk, "confidence_level", "high")

            if conf == "unassigned":

                speaker_map[spk]["confidence"] = "unassigned"

            elif conf == "review_needed" and speaker_map[spk]["confidence"] != "unassigned":

                speaker_map[spk]["confidence"] = "review_needed"

        for spk, prof in self.speaker_profiles.items():

            if spk not in speaker_map:

                speaker_map[spk] = {

                    "count": prof.get("count", 0),

                    "confidence": prof.get("confidence", "high"),

                    "ref_audio": prof.get("ref_audio", ""),

                    "voice_instruct": prof.get("voice_instruct", ""),

                }

            else:

                if "ref_audio" in prof and prof["ref_audio"]:

                    speaker_map[spk]["ref_audio"] = prof["ref_audio"]

        if not speaker_map and not self.subtitle_blocks:

            self._sync_speaker_selectors()

            self.casting_table.blockSignals(False)

            return

        self.casting_table.setRowCount(len(speaker_map))

        for row, (spk_name, data) in enumerate(speaker_map.items()):

            prof_info = self.speaker_profiles.get(spk_name)

            if prof_info is None:

                prof_info = normalize_character_profile({

                    "speaker_id": f"spk_{row}",

                    "display_name": spk_name,

                    "count": data.get("count", 0),

                    "ref_audio": data.get("ref_audio", ""),

                    "voice_instruct": data.get("voice_instruct", ""),

                }, speaker_id=f"spk_{row}")

            spk_id = prof_info.get("speaker_id", f"spk_{row}")

            item_id = QTableWidgetItem(spk_id)

            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)

            self.casting_table.setItem(row, 0, item_id)

            item_name = QTableWidgetItem(spk_name)

            self.casting_table.setItem(row, 1, item_name)

            dur_val = data.get("total_duration") or prof_info.get("total_duration", 0.0)

            dur_str = f" — {dur_val:.1f} giây" if dur_val > 0 else ""

            item_cnt = QTableWidgetItem(f"{data['count']} câu{dur_str}")

            item_cnt.setFlags(item_cnt.flags() & ~Qt.ItemIsEditable)

            self.casting_table.setItem(row, 2, item_cnt)

            # Cột 3: Giọng Đọc Trí Tuệ Nhân Tạo (ComboBox 22 giọng tiếng Việt)
            combo_voice = ComboBox(self.casting_table)
            catalog_list = list(VIETNAMESE_VOICE_CATALOG.keys())
            combo_voice.addItems(catalog_list)
            curr_preset = prof_info.get("preset_name") or prof_info.get("voice_preset") or ""
            curr_method = prof_info.get("method") or prof_info.get("assigned_strategy") or "clone"
            if curr_method == "preset" and curr_preset in VIETNAMESE_VOICE_CATALOG:
                combo_voice.setCurrentText(curr_preset)
            else:
                default_preset = curr_preset if curr_preset in VIETNAMESE_VOICE_CATALOG else catalog_list[row % len(catalog_list)]
                combo_voice.setCurrentText(default_preset)

            combo_voice.currentTextChanged.connect(
                lambda txt, s=spk_name: self._on_character_preset_changed(s, txt)
            )
            self.casting_table.setCellWidget(row, 3, combo_voice)

            # Cột 4: Trạng Thái
            status_val = prof_info.get("status", "locked")
            conf_code = data.get("confidence", "high")
            if status_val == "locked":
                status_str = "Đã khóa"
            elif conf_code == "review_needed" or status_val == "review_needed":
                status_str = "Cần kiểm tra"
            else:
                status_str = "Chưa xác định"

            item_status = QTableWidgetItem(status_str)
            item_status.setFlags(item_status.flags() & ~Qt.ItemIsEditable)
            self.casting_table.setItem(row, 4, item_status)

            # Cột 5: Thao Tác (Nút Nghe Giọng Gốc và Nghe Thử Giọng Trí Tuệ Nhân Tạo)
            cell_action = QWidget(self.casting_table)
            layout_action = QHBoxLayout(cell_action)
            layout_action.setContentsMargins(2, 1, 2, 1)
            layout_action.setSpacing(4)

            prof_item = self.speaker_profiles.get(spk_name)
            has_orig_audio = bool(
                getattr(prof_item, "source_audio_path", "")
                or (prof_item.get("source_audio_path", "") if isinstance(prof_item, dict) else "")
            )
            btn_orig = PushButton("Nghe Giọng Gốc", cell_action)
            btn_orig.setIcon(FluentIcon.MUSIC)
            btn_orig.setToolTip("Phát đoạn âm thanh gốc đã trích xuất từ video cho nhân vật này")
            btn_orig.setEnabled(has_orig_audio)
            btn_orig.clicked.connect(lambda checked=False, s=spk_name: self._preview_character_voice(s))
            layout_action.addWidget(btn_orig)

            btn_ai = PushButton("Nghe Thử Giọng Trí Tuệ Nhân Tạo", cell_action)
            btn_ai.setIcon(FluentIcon.VOLUME)
            btn_ai.setToolTip("Tổng hợp câu thoại đầu tiên bằng giọng trí tuệ nhân tạo đang được gán cho nhân vật này")
            btn_ai.clicked.connect(lambda checked=False, s=spk_name, b=btn_ai: self._audition_character_voice(s, b))
            layout_action.addWidget(btn_ai)

            self.casting_table.setCellWidget(row, 5, cell_action)

        self._sync_speaker_selectors()
        self.casting_table.blockSignals(False)

    def _on_character_preset_changed(self, spk_name: str, new_preset: str):
        """Đồng bộ khi người dùng đổi giọng đọc trí tuệ nhân tạo trên Bảng phân vai."""
        if not new_preset or new_preset not in VIETNAMESE_VOICE_CATALOG:
            return
        prof = self.speaker_profiles.get(spk_name)
        ins = VIETNAMESE_VOICE_CATALOG[new_preset]["instruct"]
        if isinstance(prof, dict):
            if prof.get("source_audio_path"):
                prof["_backup_source_audio"] = prof["source_audio_path"]
            prof["method"] = "preset"
            prof["preset_name"] = new_preset
            prof["voice_preset"] = new_preset
            prof["voice_instruct"] = ins
            prof["instruct"] = ins
            prof["source_audio_path"] = ""
            prof["ref_audio_path"] = ""
            prof["status"] = "locked"
        elif hasattr(prof, "method"):
            if getattr(prof, "source_audio_path", ""):
                prof._backup_source_audio = prof.source_audio_path
            prof.method = "preset"
            prof.preset_name = new_preset
            prof.voice_instruct = ins
            prof.source_audio_path = ""
            prof.status = "locked"

        for blk in self.subtitle_blocks:
            if getattr(blk, "speaker", "") == spk_name:
                blk.voice_instruct = ins
                blk.voice_ref_path = ""
                blk.is_speaker_locked = True

        if hasattr(self, "tts_engine"):
            self.tts_engine.set_character_profiles(self.speaker_profiles)
        if hasattr(self, "sub_table"):
            self._populate_subtitle_table()

    def _on_character_method_changed(self, spk_name: str, chosen_method_text: str):

        """Đồng bộ khi người dùng đổi phương pháp tạo giọng hoặc chọn preset trên Bảng hồ sơ giọng."""

        prof = self.speaker_profiles.get(spk_name)

        if not prof:

            return

        matching_preset = None

        if chosen_method_text in VIETNAMESE_VOICE_CATALOG:

            matching_preset = chosen_method_text

        elif chosen_method_text != "Nhân bản giọng từ video":

            for k in VIETNAMESE_VOICE_CATALOG.keys():

                if chosen_method_text.lower() in k.lower() or k.lower() in chosen_method_text.lower():

                    matching_preset = k

                    break

        if chosen_method_text == "Nhân bản giọng từ video":

            if hasattr(prof, "method"):

                prof.method = "clone"

                ref_path = getattr(prof, "source_audio_path", "") or getattr(prof, "_backup_source_audio", "")

                if not getattr(prof, "source_audio_path", "") and ref_path:

                    prof.source_audio_path = ref_path

                has_sample = bool(ref_path and os.path.exists(ref_path))

                prof.status = "locked" if has_sample else "review_needed"

            else:

                prof["method"] = "clone"

                ref_path = (

                    prof.get("ref_audio_path")

                    or prof.get("source_audio_sample")

                    or prof.get("source_audio_path", "")

                    or prof.get("_backup_source_audio", "")

                )

                if not prof.get("source_audio_path") and ref_path:

                    prof["source_audio_path"] = ref_path

                has_sample = bool(ref_path and os.path.exists(ref_path))

                prof["status"] = "locked" if has_sample else "review_needed"

            for blk in self.subtitle_blocks:

                if getattr(blk, "speaker", "") == spk_name:

                    blk.voice_ref_path = ref_path

                    blk.voice_instruct = ""

                    blk.is_speaker_locked = True

        elif matching_preset:

            ins = VIETNAMESE_VOICE_CATALOG[matching_preset]["instruct"]

            if hasattr(prof, "method"):

                prof.method = "preset"

                prof.preset_name = matching_preset

                prof.voice_instruct = ins

                prof.status = "locked"

                prof.custom_audio_path = ""

                prof.voice_clone_prompt_path = ""

                if getattr(prof, "source_audio_path", ""):

                    prof._backup_source_audio = prof.source_audio_path

                prof.source_audio_path = ""

                if hasattr(prof, "ref_audio_path"):

                    setattr(prof, "ref_audio_path", "")

            else:

                prof["method"] = "preset"

                prof["preset_name"] = matching_preset

                prof["voice_preset"] = matching_preset

                prof["voice_instruct"] = ins

                prof["instruct"] = ins

                prof["status"] = "locked"

                if prof.get("source_audio_path"):

                    prof["_backup_source_audio"] = prof["source_audio_path"]

                prof["source_audio_path"] = ""

                prof["ref_audio_path"] = ""

                prof["prompt"] = None

            for blk in self.subtitle_blocks:

                if getattr(blk, "speaker", "") == spk_name:

                    blk.voice_ref_path = ""

                    blk.voice_instruct = ins

                    blk.is_speaker_locked = True

        self._populate_subtitle_table()

        self._update_casting_table()

        self._update_character_profiles_overview()

    def _choose_custom_reference_audio(self, spk_name: str):

        """Cho phép người dùng chọn tệp âm thanh mẫu từ máy tính cho nhân vật."""

        file_path, _ = QFileDialog.getOpenFileName(

            self,

            f"Chọn tệp âm thanh mẫu cho {spk_name}",

            "",

            "Tệp âm thanh (*.wav *.mp3 *.m4a *.aac *.flac);;Tất cả tệp (*.*)",

        )

        if not file_path or not os.path.isfile(file_path):

            return

        prof = self.speaker_profiles.get(spk_name)

        if not prof:

            prof = {

                "speaker_id": spk_name,

                "display_name": spk_name,

                "method": "clone",

                "status": "locked",

                "source_audio_path": file_path,

            }

            self.speaker_profiles[spk_name] = prof

        if hasattr(prof, "source_audio_path"):

            prof.source_audio_path = file_path

            prof.custom_audio_path = file_path

            prof.method = "clone"

            prof.status = "locked"

        else:

            prof["source_audio_path"] = file_path

            prof["source_audio_sample"] = file_path

            prof["ref_audio_path"] = file_path

            prof["ref_audio"] = file_path

            prof["custom_audio_path"] = file_path

            prof["method"] = "clone"

            prof["status"] = "locked"

        for blk in self.subtitle_blocks:

            if getattr(blk, "speaker", "") == spk_name:

                blk.voice_ref_path = file_path

                blk.voice_instruct = ""

                blk.is_speaker_locked = True

        self._update_casting_table()

        self._populate_subtitle_table()

        self._update_character_profiles_overview()

        InfoBar.success("Đã nạp mẫu giọng", f"Đã gán tệp âm thanh mẫu cho {spk_name}.", parent=self, duration=2500)

    def _on_casting_table_item_changed(self, item: QTableWidgetItem):

        """Đồng bộ khi người dùng chỉnh sửa trực tiếp tên nhân vật trên bảng phân vai."""

        if item.column() != 1:

            return

        row = item.row()

        id_item = self.casting_table.item(row, 0)

        if not id_item:

            return

        spk_id = id_item.text().strip()

        new_name = item.text().strip()

        if not new_name:

            return

        old_name = None

        for name, prof in list(self.speaker_profiles.items()):

            sid = getattr(prof, "speaker_id", "") or (prof.get("speaker_id") if isinstance(prof, dict) else "")

            if sid == spk_id:

                old_name = name

                break

        if old_name and old_name != new_name:

            prof = self.speaker_profiles.pop(old_name)

            if hasattr(prof, "speaker_id"):

                prof.speaker_id = spk_id

                prof.display_name = new_name

            else:

                prof["speaker_id"] = spk_id

                prof["display_name"] = new_name

            self.speaker_profiles[new_name] = normalize_character_profile(prof, speaker_id=spk_id)

            for blk in self.subtitle_blocks:

                if getattr(blk, "speaker", "") == old_name:

                    blk.speaker = new_name

            self._populate_subtitle_table()

        elif not old_name:

            self.speaker_profiles[new_name] = normalize_character_profile({

                "speaker_id": spk_id,

                "display_name": new_name,

                "count": 0,

            }, speaker_id=spk_id)

            for blk in self.subtitle_blocks:

                if getattr(blk, "speaker", "") == spk_id:

                    blk.speaker = new_name

            self._populate_subtitle_table()

        self._sync_speaker_selectors()
        self._update_character_profiles_overview()

    def _on_casting_cell_changed(self, row: int, column: int):

        """Xử lý khi một ô trong bảng phân vai thay đổi."""

        if column == 1:

            self._sync_speaker_selectors()


    def _export_subtitles_to_srt(self, target_path: str = "") -> bool:
        """Xuất danh sách khối phụ đề hiện tại ra tệp định dạng SRT chuẩn UTF-8."""
        if not self.subtitle_blocks:
            return False

        if not target_path:
            from PySide6.QtWidgets import QFileDialog
            from pathlib import Path
            default_dir = os.path.dirname(self.current_video_path) if self.current_video_path else os.getcwd()
            suggested_name = "omnivoice_subtitles.srt"
            if self.current_video_path:
                suggested_name = f"{Path(self.current_video_path).stem}_phu_de.srt"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu tệp phụ đề SRT",
                os.path.join(default_dir, suggested_name),
                "Tệp phụ đề (*.srt);;Tất cả tệp (*.*)",
            )
            if not file_path:
                return False
            target_path = file_path

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                for idx, blk in enumerate(self.subtitle_blocks, start=1):
                    start_t = getattr(blk, "start_time", "00:00:00,000")
                    end_t = getattr(blk, "end_time", "00:00:01,000")
                    text = getattr(blk, "translated_text", "") or getattr(blk, "text", "")
                    spk = getattr(blk, "speaker", "")
                    if spk and spk not in ("Chưa xác định", "unassigned", "") and not text.startswith(f"[{spk}]"):
                        content = f"[{spk}] {text}"
                    else:
                        content = text
                    f.write(f"{idx}\n{start_t} --> {end_t}\n{content}\n\n")
            return True
        except Exception as e:
            self.log(f"Lỗi khi xuất tệp phụ đề SRT: {e}")
            return False

    def _handle_export_subtitles_clicked(self):
        """Xử lý sự kiện khi người dùng bấm nút Xuất Tệp Phụ Đề SRT."""
        if not self.subtitle_blocks:
            InfoBar.warning("Chưa có phụ đề", "Không có câu thoại nào để xuất tệp.", parent=self, duration=3000)
            return
        ok = self._export_subtitles_to_srt()
        if ok:
            InfoBar.success("Đã xuất phụ đề", "Tệp phụ đề SRT đã được lưu thành công.", parent=self, duration=3000)


    def _get_selected_subtitle_rows(self) -> List[int]:
        """Lấy danh sách các chỉ số dòng đang được người dùng chọn trên bảng phụ đề."""
        if not hasattr(self, "sub_table"):
            return []
        rows = set()
        for item in self.sub_table.selectedItems():
            rows.add(item.row())
        for idx in self.sub_table.selectedIndexes():
            rows.add(idx.row())
        return sorted(list(rows))

    def _batch_assign_selected_speaker(self):
        """Gán nhân vật được chọn cho toàn bộ các dòng phụ đề đang được bôi đen."""
        rows = self._get_selected_subtitle_rows()
        if not rows:
            InfoBar.warning("Chưa chọn dòng", "Vui lòng chọn ít nhất một câu thoại trên bảng trước khi gán hàng loạt.", parent=self, duration=3000)
            return
        target_spk = self.combo_batch_speaker.currentText().strip()
        if not target_spk:
            return

        if target_spk != "Chưa xác định" and target_spk not in self.speaker_profiles:
            self.speaker_profiles[target_spk] = CharacterVoiceProfile(
                speaker_id=f"spk_{len(self.speaker_profiles)}",
                display_name=target_spk,
            )

        self.sub_table.blockSignals(True)
        try:
            for row in rows:
                if row >= len(self.subtitle_blocks):
                    continue
                blk = self.subtitle_blocks[row]
                blk.speaker = target_spk
                if target_spk != "Chưa xác định":
                    blk.is_speaker_locked = True
                    blk.confidence_level = "high"
                else:
                    blk.is_speaker_locked = False
                    blk.confidence_level = "unassigned"

                prof = self.speaker_profiles.get(target_spk)
                if prof:
                    is_dict = isinstance(prof, dict)
                    method = (prof.get("method") if is_dict else getattr(prof, "method", "clone")) or "clone"
                    if method == "preset":
                        blk.voice_ref_path = ""
                        blk.voice_instruct = (prof.get("voice_instruct") or prof.get("instruct") if is_dict else getattr(prof, "voice_instruct", "")) or ""
                    else:
                        blk.voice_ref_path = (prof.get("source_audio_path") or prof.get("ref_audio_path") or prof.get("source_audio_sample") if is_dict else getattr(prof, "source_audio_path", "")) or ""
                        blk.voice_instruct = ""
                else:
                    blk.voice_ref_path = ""
                    blk.voice_instruct = ""

                combo = self.sub_table.cellWidget(row, 3)
                if isinstance(combo, ComboBox):
                    combo.blockSignals(True)
                    if target_spk not in [combo.itemText(i) for i in range(combo.count())]:
                        combo.addItem(target_spk)
                    combo.setCurrentText(target_spk)
                    combo.blockSignals(False)

                item_spk = self.sub_table.item(row, 3)
                if item_spk:
                    item_spk.setText(target_spk)

                item_src = self.sub_table.item(row, 4)
                if item_src:
                    item_src.setText(self._format_voice_source_display(blk))

                for c in (0, 1, 2, 3, 4, 6):
                    it = self.sub_table.item(row, c)
                    if it:
                        it.setBackground(QColor(0, 0, 0, 0))
        finally:
            self.sub_table.blockSignals(False)

        if hasattr(self, "casting_progress_bar"):
            self.casting_progress_bar.update_progress(self.subtitle_blocks)
        self._update_casting_table()
        self._update_character_profiles_overview()
        self._filter_subtitle_table()
        InfoBar.success("Đã gán hàng loạt", f"Đã gán nhân vật '{target_spk}' cho {len(rows)} câu thoại thành công.", parent=self, duration=3000)

    def _batch_set_locked_state(self, locked: bool):
        """Khóa hoặc mở khóa hàng loạt cho các câu thoại đang được bôi đen."""
        rows = self._get_selected_subtitle_rows()
        if not rows:
            InfoBar.warning("Chưa chọn dòng", "Vui lòng chọn ít nhất một câu thoại trên bảng.", parent=self, duration=3000)
            return

        self.sub_table.blockSignals(True)
        try:
            for row in rows:
                if row >= len(self.subtitle_blocks):
                    continue
                blk = self.subtitle_blocks[row]
                blk.is_speaker_locked = locked

                conf = getattr(blk, "confidence_level", "high")
                spk = getattr(blk, "speaker", "")
                if not locked and conf == "review_needed":
                    row_bg = QColor(255, 248, 225, 180)
                elif not locked and (conf in ("unassigned", "") or not spk or spk in ("Chưa xác định", "unassigned")):
                    row_bg = QColor(255, 235, 238, 180)
                else:
                    row_bg = QColor(0, 0, 0, 0)

                for c in (0, 1, 2, 3, 4, 6):
                    it = self.sub_table.item(row, c)
                    if it:
                        it.setBackground(row_bg)
        finally:
            self.sub_table.blockSignals(False)

        if hasattr(self, "casting_progress_bar"):
            self.casting_progress_bar.update_progress(self.subtitle_blocks)
        self._filter_subtitle_table()
        msg = f"Đã khóa {len(rows)} câu thoại." if locked else f"Đã mở khóa {len(rows)} câu thoại."
        InfoBar.success("Thành công", msg, parent=self, duration=3000)

    def _filter_subtitle_table(self, filter_text: str = ""):

        """Lọc danh sách câu thoại theo độ tin cậy nhận diện người nói tại Bước 3."""

        if not hasattr(self, "sub_table") or not hasattr(self, "combo_sub_filter"):

            return

        selected = filter_text or self.combo_sub_filter.currentText()

        for row in range(self.sub_table.rowCount()):

            if row >= len(self.subtitle_blocks):

                continue

            blk = self.subtitle_blocks[row]

            conf = getattr(blk, "confidence_level", "high")

            if selected == "Độ tin cậy cao":

                self.sub_table.setRowHidden(row, conf != "high")

            elif selected == "Cần rà soát":

                self.sub_table.setRowHidden(row, conf != "review_needed")

            elif selected == "Chưa xác định":

                self.sub_table.setRowHidden(row, conf != "unassigned")

            elif selected == "Đã khóa":
                is_locked = getattr(blk, "is_speaker_locked", False)
                self.sub_table.setRowHidden(row, not is_locked)
            else:

                self.sub_table.setRowHidden(row, False)

    def _is_valid_widget(self, widget: Any) -> bool:
        """Kiểm tra an toàn đối tượng widget C++ của PySide6 còn tồn tại hay đã bị xóa."""
        if widget is None:
            return False
        try:
            from shiboken6 import isValid
            return bool(isValid(widget))
        except Exception:
            try:
                _ = widget.objectName()
                return True
            except RuntimeError:
                return False

    def _reset_audition_button(self):

        """Khôi phục trạng thái nút nghe thử giọng của nhân vật an toàn."""

        btn = getattr(self, "_active_audition_button", None)
        self._active_audition_button = None

        if self._is_valid_widget(btn):
            try:
                default_t = btn.property("default_text") or "Nghe Thử Giọng"
                btn.setText(default_t)
                default_icon = btn.property("default_icon") or (FluentIcon.PLAY if default_t == "Nghe" else FluentIcon.VOLUME)
                btn.setIcon(default_icon)
                btn.setEnabled(True)
            except RuntimeError:
                pass

    def _on_player_media_status_changed(self, status):

        """Phục hồi nút nghe thử khi bài phát âm thanh kết thúc hoặc lỗi."""

        if status in (QMediaPlayer.MediaStatus.EndOfMedia, QMediaPlayer.MediaStatus.InvalidMedia):

            self._reset_audition_button()

            if getattr(self, "_audition_all_active", False):

                QTimer.singleShot(400, self._play_next_audition_in_queue)

        elif status == QMediaPlayer.MediaStatus.NoMedia:

            self._reset_audition_button()

    def _update_character_profiles_overview(self):

        """Cập nhật khối tổng quan hồ sơ giọng nhân vật ở Bước 2."""

        if not hasattr(self, "char_cards_layout"):

            return

        is_analyzed = bool(getattr(self, "is_speakers_detected", False) or self.speaker_profiles)

        # Xóa các thẻ cũ

        while self.char_cards_layout.count():

            item = self.char_cards_layout.takeAt(0)

            w = item.widget()

            if w:

                w.deleteLater()

        self._character_cards = []

        if not is_analyzed:

            if hasattr(self, "lbl_char_overview_title"):

                self.lbl_char_overview_title.setText("Âm thanh video chưa phân tích nhân vật")

            if hasattr(self, "lbl_char_count"):

                self.lbl_char_count.setText("Trạng thái: Chưa phân tích")

            lbl_hint = BodyLabel(

                "Chưa phân tích âm thanh video. Hãy bấm 'Tự Động Phân Tích Lại' hoặc 'Tự Động Nhận Diện Nhiều Người Nói' ở Bước 1 để tự động nhận diện và phân vai các nhân vật.",

                self.char_cards_container,

            )

            lbl_hint.setWordWrap(True)

            self.char_cards_layout.addWidget(lbl_hint)

            return

        if hasattr(self, "lbl_char_overview_title"):

            self.lbl_char_overview_title.setText("Video đã phân tích âm thanh · Đã nhận diện")

        speaker_names: List[str] = []

        seen = set()

        for spk in self.speaker_profiles.keys():

            if spk and spk not in ("Chưa phân vai", "Chưa xác định") and spk not in seen:

                seen.add(spk)

                speaker_names.append(spk)

        for blk in self.subtitle_blocks:

            spk = (getattr(blk, "speaker", "") or "").strip()

            if spk and spk not in ("Chưa phân vai", "Chưa xác định") and spk not in seen:

                seen.add(spk)

                speaker_names.append(spk)

        # Cập nhật nhãn đếm số lượng nhân vật thực tế nhận diện

        count = len(speaker_names)

        if hasattr(self, "lbl_char_count"):

            self.lbl_char_count.setText(f"Tổng số nhân vật nhận diện: {count} nhân vật")

        if not speaker_names:

            lbl_empty = BodyLabel("Không tìm thấy nhân vật nào trong video hoặc tệp phụ đề.", self.char_cards_container)

            lbl_empty.setWordWrap(True)

            self.char_cards_layout.addWidget(lbl_empty)

            return

        for spk_name in speaker_names:

            prof = self.speaker_profiles.get(spk_name)

            is_dict = isinstance(prof, dict)

            method = (prof.get("method") if is_dict else getattr(prof, "method", "clone")) if prof else "clone"

            status_val = (prof.get("status") if is_dict else getattr(prof, "status", "unassigned")) if prof else "unassigned"

            ref_audio = (prof.get("source_audio_path") or prof.get("ref_audio_path") if is_dict else getattr(prof, "source_audio_path", "")) if prof else ""

            voice_ins = (prof.get("voice_instruct") or prof.get("instruct") if is_dict else getattr(prof, "voice_instruct", "")) if prof else ""

            preset_name = (prof.get("preset_name") if is_dict else getattr(prof, "preset_name", "")) if prof else ""

            desc = (prof.get("description") if is_dict else getattr(prof, "preset_name", "")) if prof else ""

            # Kiểm tra thêm từ các khối phụ đề nếu profile chưa ghi

            for blk in self.subtitle_blocks:

                if getattr(blk, "speaker", "").strip() == spk_name:

                    if not ref_audio and getattr(blk, "voice_ref_path", ""):

                        ref_audio = blk.voice_ref_path

                    if not voice_ins and getattr(blk, "voice_instruct", ""):

                        voice_ins = blk.voice_instruct

                    break

            has_valid_ref = bool(ref_audio and os.path.exists(ref_audio))

            has_valid_ins = bool(voice_ins)

            # Phân biệt rõ các trạng thái theo độ tin cậy và sự sẵn sàng của mẫu giọng

            conf_level = prof.get("confidence_level", "") if is_dict else getattr(prof, "confidence_level", "")

            raw_ratio = prof.get("high_confidence_ratio", 1.0) if is_dict else getattr(prof, "high_confidence_ratio", 1.0)

            try:

                conf_ratio = float(raw_ratio) if raw_ratio is not None else 1.0

            except (ValueError, TypeError):

                conf_ratio = 1.0

            is_review_needed = (

                status_val in ("review_needed", "unassigned")

                or conf_level == "review_needed"

                or conf_ratio < 0.7

            )

            if method == "clone":

                if has_valid_ref and not is_review_needed:

                    type_text = "Nhân bản giọng từ video"

                    detail_text = "Mẫu giọng đã tạo"

                    badge_text = "Sẵn sàng"

                    badge_bg = "#107c41"

                elif has_valid_ref and is_review_needed:

                    type_text = "Nhân bản giọng từ video"

                    detail_text = "Mẫu giọng cần rà soát"

                    badge_text = "Cần kiểm tra"

                    badge_bg = "#d83b01"

                else:

                    type_text = "Nhân bản giọng chưa có mẫu"

                    detail_text = "Thiếu tệp âm thanh mẫu từ video"

                    badge_text = "Cần kiểm tra"

                    badge_bg = "#d83b01"

            elif method == "preset":

                if has_valid_ins:

                    type_text = "Giọng thiết kế"

                    detail_text = preset_name or desc or voice_ins or "Giọng danh mục"

                    badge_text = "Sẵn sàng"

                    badge_bg = "#107c41"

                else:

                    type_text = "Giọng thiết kế chưa chọn"

                    detail_text = "Chưa chọn giọng danh mục"

                    badge_text = "Cần kiểm tra"

                    badge_bg = "#d83b01"

            else:

                type_text = "Chưa có giọng"

                detail_text = "Chưa gán phương pháp giọng"

                badge_text = "Cần kiểm tra"

                badge_bg = "#d83b01"

            card = CardWidget(self.char_cards_container)

            card.setMinimumWidth(260)

            card.setMaximumWidth(320)

            card_layout = QVBoxLayout(card)

            card_layout.setContentsMargins(12, 10, 12, 10)

            card_layout.setSpacing(6)

            # Dòng 1: Tên nhân vật + Huy hiệu

            row1 = QHBoxLayout()

            lbl_name = StrongBodyLabel(spk_name, card)

            badge = CaptionLabel(badge_text, card)

            badge.setStyleSheet(

                f"color: white; background-color: {badge_bg}; border-radius: 4px; padding: 2px 6px; font-weight: bold;"

            )

            row1.addWidget(lbl_name)

            row1.addStretch(1)

            row1.addWidget(badge)

            card_layout.addLayout(row1)

            # Dòng 2: Loại giọng & Mô tả chi tiết

            lbl_type = CaptionLabel(f"{type_text} · {detail_text}", card)

            lbl_type.setWordWrap(True)

            card_layout.addWidget(lbl_type)

            # Dòng 3: Nút Nghe Thử Giọng Lồng Tiếng & Nút Nghe Mẫu Gốc (nếu có tệp mẫu)

            btn_row = QHBoxLayout()

            btn_row.setSpacing(6)

            has_sample = bool(

                (prof.get("source_audio_path") or prof.get("ref_audio_path") if is_dict else getattr(prof, "source_audio_path", "")) if prof else ""

            )

            btn_audition = PushButton("Nghe Thử Giọng Lồng Tiếng" if has_sample else "Nghe Thử Giọng", card)

            btn_audition.setProperty("default_text", btn_audition.text())

            btn_audition.setIcon(FluentIcon.VOLUME)

            btn_audition.clicked.connect(lambda checked=False, s=spk_name, b=btn_audition: self._audition_character_voice(s, b))

            btn_row.addWidget(btn_audition)

            if has_sample:

                btn_orig = PushButton("Nghe Mẫu Giọng Gốc", card)

                btn_orig.setIcon(FluentIcon.HEADPHONE)

                btn_orig.clicked.connect(lambda checked=False, s=spk_name: self._preview_character_voice(s))

                btn_row.addWidget(btn_orig)

            card_layout.addLayout(btn_row)

            rec_widget = self._build_voice_recommendation_widget(spk_name)

            card_layout.addWidget(rec_widget)

            self.char_cards_layout.addWidget(card)

            self._character_cards.append({

                "name": spk_name,

                "speaker": spk_name,

                "status": badge_text,

                "type": type_text,

                "detail": detail_text,

                "widget": card,

                "button": btn_audition,

            })

    def _build_voice_recommendation_widget(self, spk_name: str) -> QWidget:

        """Tạo widget hiển thị Top 3 gợi ý giọng Việt Nam cho nhân vật."""

        container = QWidget()

        layout = QVBoxLayout(container)

        layout.setContentsMargins(0, 4, 0, 4)

        layout.setSpacing(6)

        prof = self.speaker_profiles.get(spk_name)

        if prof is None:

            layout.addWidget(CaptionLabel("Chưa có hồ sơ nhân vật để đề xuất giọng.", container))

            return container

        try:

            recommendations = VoiceRecommendationEngine.recommend_voices(prof, top_k=3)

        except Exception:

            recommendations = []

        if not recommendations:

            layout.addWidget(CaptionLabel("Không tìm được giọng phù hợp trong danh mục.", container))

            return container

        layout.addWidget(StrongBodyLabel("Giọng đọc được gợi ý:", container))

        for idx, rec in enumerate(recommendations, start=1):

            rec_card = QFrame(container)

            rec_card.setFrameShape(QFrame.StyledPanel)

            rec_card_layout = QHBoxLayout(rec_card)

            rec_card_layout.setContentsMargins(8, 4, 8, 4)

            rec_card_layout.setSpacing(8)

            preset_label = rec.get("preset_name", "")

            score = rec.get("score", 0)

            reason = rec.get("reason", "")

            lbl_rank = BodyLabel(f"{idx}.", rec_card)

            lbl_rank.setFixedWidth(18)

            rec_card_layout.addWidget(lbl_rank)

            lbl_name = BodyLabel(preset_label, rec_card)

            lbl_name.setWordWrap(True)

            rec_card_layout.addWidget(lbl_name, 1)

            lbl_score = CaptionLabel(f"{score}%", rec_card)

            lbl_score.setToolTip(reason)

            rec_card_layout.addWidget(lbl_score)

            btn_apply = PushButton("Áp Dụng", rec_card)

            btn_apply.setIcon(FluentIcon.ACCEPT)

            btn_apply.clicked.connect(

                lambda checked=False, s=spk_name, p=preset_label: self._apply_recommended_voice(s, p)

            )

            rec_card_layout.addWidget(btn_apply)

            layout.addWidget(rec_card)

        return container

    def _apply_recommended_voice(self, spk_name: str, preset_name: str):

        """Áp dụng giọng được gợi ý cho nhân vật và làm mới thẻ hồ sơ."""

        if preset_name not in VIETNAMESE_VOICE_CATALOG or spk_name not in self.speaker_profiles:

            return

        instruct_val = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

        prof = self.speaker_profiles[spk_name]

        if hasattr(prof, "method"):
            prof.method = "preset"
            prof.preset_name = preset_name
            prof.voice_instruct = instruct_val
            prof.source_audio_path = ""
            prof.status = "locked"
        else:
            prof["method"] = "preset"
            prof["preset_name"] = preset_name
            prof["voice_instruct"] = instruct_val
            prof["source_audio_path"] = ""
            prof["status"] = "locked"

        for blk in self.subtitle_blocks:
            if getattr(blk, "speaker", "") == spk_name:
                blk.voice_ref_path = ""
                blk.voice_instruct = instruct_val
                blk.is_speaker_locked = True

        self._populate_subtitle_table()
        self._update_casting_table()
        self._update_character_profiles_overview()

        InfoBar.success(

            title="Đã áp dụng giọng đọc",

            content=f"Nhân vật '{spk_name}' đã được gán giọng '{preset_name}'.",

            parent=self,

            duration=3000,

        )

    def _audition_character_voice(self, spk_name: str, btn: Optional[PushButton] = None):

        """Nghe thử giọng nhân vật bằng cách tổng hợp câu thoại đầu tiên của nhân vật trong video."""

        # 1. Nếu hệ thống đang bận chạy tác vụ ngầm, từ chối ngay mà không khóa nút

        if self._worker and self._worker.isRunning():

            InfoBar.warning(

                "Đang bận",

                "Hệ thống đang xử lý tác vụ trước đó, vui lòng đợi...",

                parent=self,

                duration=2500,

            )

            return

        # 2. Nếu đang phát chính nhân vật này, bấm lại để dừng phát

        if (

            btn is not None

            and getattr(self, "_active_audition_button", None) == btn

            and self._media_player.playbackState() == QMediaPlayer.PlayingState

        ):

            self._media_player.stop()

            self._reset_audition_button()

            return

        # 3. Tìm câu thoại đầu tiên có văn bản của nhân vật

        first_line = ""

        target_blk = None

        for blk in self.subtitle_blocks:

            spk = (getattr(blk, "speaker", "") or "").strip()

            if spk == spk_name:

                txt = (blk.translated_text if blk.translated_text else blk.text or "").strip()

                if txt:

                    first_line = txt

                    target_blk = blk

                    break

        if not first_line:
            has_multiple_speakers = len(getattr(self, "speaker_profiles", {})) > 1
            if not has_multiple_speakers and hasattr(self, "subtitle_blocks") and self.subtitle_blocks:
                for blk in self.subtitle_blocks:
                    txt = (blk.translated_text if blk.translated_text else blk.text or "").strip()
                    if txt:
                        first_line = txt
                        target_blk = blk
                        break

        if not first_line:

            first_line = f"Xin chào, đây là giọng nói thử nghiệm của {spk_name}."

        prof = self.speaker_profiles.get(spk_name)

        is_dict = isinstance(prof, dict)

        method = (prof.get("method") if is_dict else getattr(prof, "method", "clone")) if prof else "clone"

        ref_audio = (prof.get("source_audio_path") or prof.get("ref_audio_path") if is_dict else getattr(prof, "source_audio_path", "")) if prof else ""

        voice_ins = (prof.get("voice_instruct") or prof.get("instruct") if is_dict else getattr(prof, "voice_instruct", "")) if prof else ""

        ref_text = (prof.get("ref_text") if is_dict else getattr(prof, "ref_text", "")) if prof else ""

        if target_blk:

            if getattr(target_blk, "voice_ref_path", ""):

                ref_audio = target_blk.voice_ref_path

            if getattr(target_blk, "voice_instruct", ""):

                voice_ins = target_blk.voice_instruct

            if not ref_text and getattr(target_blk, "text", ""):

                ref_text = getattr(target_blk, "text", "")

        if method == "preset":

            ref_audio = ""

        elif method == "clone":

            voice_ins = ""

        # 4. Kiểm tra tính hợp lệ của giọng nhân vật, tuyệt đối không âm thầm đổi sang giọng mặc định

        if method == "clone":

            if not (ref_audio and os.path.exists(ref_audio)):

                InfoBar.warning(

                    "Thiếu mẫu giọng nhân bản",

                    f"Nhân vật {spk_name} chưa có tệp âm thanh mẫu từ video. Hãy phân tích lại video hoặc chọn mẫu giọng từ máy tính trước khi nghe thử.",

                    parent=self,

                    duration=4500,

                )

                self._reset_audition_button()

                if getattr(self, "_audition_all_active", False):

                    QTimer.singleShot(400, self._play_next_audition_in_queue)

                return

        elif method == "preset":

            if not voice_ins:

                preset_name = (prof.get("preset_name") if is_dict else getattr(prof, "preset_name", "")) if prof else ""

                if preset_name in VIETNAMESE_VOICE_CATALOG:

                    voice_ins = VIETNAMESE_VOICE_CATALOG[preset_name].get("instruct", "")

            if not voice_ins:

                InfoBar.warning(

                    "Chưa cấu hình giọng thiết kế",

                    f"Nhân vật {spk_name} chưa có mô tả giọng nói thiết kế. Vui lòng chọn giọng trong danh mục trước khi nghe thử.",

                    parent=self,

                    duration=4500,

                )

                self._reset_audition_button()

                if getattr(self, "_audition_all_active", False):

                    QTimer.singleShot(400, self._play_next_audition_in_queue)

                return

        else:

            if not (ref_audio and os.path.exists(ref_audio)) and not voice_ins:

                InfoBar.warning(

                    "Chưa gán giọng nhân vật",

                    f"Nhân vật {spk_name} chưa có mẫu giọng hoặc cấu hình thiết kế giọng. Vui lòng cấu hình giọng trước khi nghe thử.",

                    parent=self,

                    duration=4500,

                )

                self._reset_audition_button()

                if getattr(self, "_audition_all_active", False):

                    QTimer.singleShot(400, self._play_next_audition_in_queue)

                return

        # 5. Chuyển trạng thái nút sang Đang tạo giọng...

        if btn is not None and self._is_valid_widget(btn):
            try:
                self._active_audition_button = btn
                btn.setText("Đang tạo giọng...")
                btn.setIcon(FluentIcon.SYNC)
                btn.setEnabled(False)
            except RuntimeError:
                self._active_audition_button = None

        out_dir = os.path.join(BASE_DIR, "output", "omnivoice_studio", "auditions")

        os.makedirs(out_dir, exist_ok=True)

        safe_spk = re.sub(r'[^\w\-_\.]', '_', spk_name)

        import hashlib
        line_hash = hashlib.md5(first_line.encode("utf-8")).hexdigest()[:8]
        out_wav = os.path.join(out_dir, f"audition_{safe_spk}_{line_hash}.wav")

        self._launch_worker(

            "character_audition",

            text=first_line,

            output_path=out_wav,

            override_ref_audio=ref_audio if method == "clone" else "",

            override_instruct=voice_ins if method == "preset" else "",

            override_ref_text=ref_text if method == "clone" else "",

        )

    def _preview_character_voice(self, spk_name: str):

        """Nghe mẫu âm thanh gốc từ video hoặc tệp âm thanh của nhân vật."""

        prof = self.speaker_profiles.get(spk_name, {})

        is_dict = isinstance(prof, dict)

        ref_path = (

            prof.get("source_audio_path") or prof.get("ref_audio_path")

            if is_dict

            else getattr(prof, "source_audio_path", "") or getattr(prof, "ref_audio_path", "")

        ) if prof else ""

        if ref_path and os.path.exists(ref_path):

            self._play_audio_file(ref_path)

            InfoBar.info("Đang phát mẫu giọng", f"Đang nghe thử giọng mẫu gốc của {spk_name}.", parent=self, duration=2500)

        else:

            InfoBar.warning(

                "Không có mẫu giọng gốc",

                f"Nhân vật {spk_name} không có tệp âm thanh mẫu từ video. Hãy sử dụng nút 'Nghe Thử Giọng' để tạo giọng thử nghiệm.",

                parent=self,

                duration=3500,

            )

    def _toggle_advanced_settings(self):

        """Đóng mở nhóm thiết lập phần cứng và tham số nâng cao."""

        if not hasattr(self, "adv_content_widget"):

            return

        vis = not self.adv_content_widget.isVisible()

        self.adv_content_widget.setVisible(vis)

        if hasattr(self, "btn_toggle_adv"):

            self.btn_toggle_adv.setIcon(FluentIcon.CHEVRON_DOWN if vis else FluentIcon.CHEVRON_RIGHT)

    def _init_more_actions_menu(self):

        """Khởi tạo menu lệnh gom nhóm các thao tác hàng loạt cho Quản lý nhân vật."""

        if not hasattr(self, "btn_more_actions"):

            return

        menu = RoundMenu(parent=self.btn_more_actions)

        menu.addAction(Action(FluentIcon.SYNC, "Tự Động Phân Tích Lại", triggered=self._run_auto_detect_speakers))

        menu.addAction(Action(FluentIcon.VOLUME, "Nghe Thử Toàn Bộ Nhân Vật", triggered=self._audition_all_speakers))

        menu.addAction(Action(FluentIcon.EDIT, "Gán Nhân Vật Vào Câu Đang Chọn", triggered=self._assign_target_speaker_to_selected_rows))

        menu.addAction(Action(FluentIcon.DELETE, "Đặt Lại Toàn Bộ Giọng", triggered=lambda: self._handle_reset_all_voices_with_confirmation(None)))

        self.btn_more_actions.setMenu(menu)

    def _handle_reset_all_voices_with_confirmation(self, confirmed: Optional[bool] = None) -> bool:

        """Đặt lại toàn bộ phân vai và giọng nhân vật có xác nhận an toàn."""

        if confirmed is False:

            return False

        if confirmed is True:

            self._reset_all_character_voices()

            return True

        msg = MessageBox(

            "Xác nhận đặt lại giọng",

            "Bạn có chắc chắn muốn đặt lại toàn bộ giọng đã gán cho tất cả nhân vật về mặc định? Thao tác này sẽ hủy toàn bộ cấu hình đã khóa.",

            self,

        )

        msg.yesButton.setText("Đặt Lại")

        msg.cancelButton.setText("Hủy Bỏ")

        if msg.exec():

            self._reset_all_character_voices()

            return True

        return False

    def _reset_all_character_voices(self):

        """Đặt lại toàn bộ phân vai và cấu hình giọng nhân vật."""

        self.speaker_profiles.clear()

        self.is_speakers_detected = False

        for blk in self.subtitle_blocks:

            blk.voice_ref_path = ""

            blk.voice_instruct = ""

            blk.is_speaker_locked = False

        self._update_casting_table()

        self._populate_subtitle_table()

        InfoBar.success("Đã đặt lại", "Đã xóa toàn bộ cấu hình giọng gán cho nhân vật.", parent=self, duration=2500)

    def _audition_all_speakers(self):

        """Nghe thử tuần tự từng nhân vật đã được phân vai."""

        if getattr(self, "_audition_all_active", False):

            self._stop_audition_all()

            return

        chars: List[str] = []

        for c in getattr(self, "_character_cards", []):

            spk = c.get("name") or c.get("speaker")

            if spk and spk not in ("Chưa phân vai", "Chưa xác định") and spk not in chars:

                chars.append(spk)

        if not chars:

            for s in self.speaker_profiles.keys():

                if s and s not in ("Chưa phân vai", "Chưa xác định") and s not in chars:

                    chars.append(s)

        if not chars:

            InfoBar.warning("Chưa có nhân vật", "Vui lòng nạp phụ đề hoặc nhận diện nhân vật trước khi nghe thử.", parent=self, duration=2500)

            return

        self._audition_queue = list(chars)

        self._audition_all_total = len(chars)

        self._audition_all_current_idx = 0

        self._audition_all_active = True

        if hasattr(self, "btn_audition_all_speakers"):

            self.btn_audition_all_speakers.setText("Dừng Nghe Thử")

            self.btn_audition_all_speakers.setIcon(FluentIcon.PAUSE)

        self._audition_next_in_queue()

    def _audition_next_in_queue(self):

        """Phát nhân vật kế tiếp trong hàng đợi nghe thử toàn bộ nhân vật."""

        self._play_next_audition_in_queue()

    def _play_next_audition_in_queue(self):

        """Phát nhân vật tiếp theo trong hàng đợi nghe thử."""

        if not getattr(self, "_audition_all_active", False):

            return

        if not self._audition_queue:

            self._stop_audition_all()

            InfoBar.success("Hoàn thành", "Đã nghe thử toàn bộ nhân vật.", parent=self, duration=3000)

            return

        spk = self._audition_queue.pop(0)

        self._current_audition_speaker = spk

        curr_idx = self._audition_all_total - len(self._audition_queue)

        self._audition_all_current_idx = curr_idx

        if hasattr(self, "btn_audition_all_speakers"):

            self.btn_audition_all_speakers.setText(f"Dừng [{curr_idx}/{self._audition_all_total}]")

            self.btn_audition_all_speakers.setIcon(FluentIcon.PAUSE)

        InfoBar.info("Nghe thử nhân vật", f"Đang nghe thử giọng của {spk} — Nhân vật {curr_idx}/{self._audition_all_total}.", parent=self, duration=2500)

        card_btn = None

        for c in getattr(self, "_character_cards", []):

            if (c.get("name") or c.get("speaker")) == spk:

                card_btn = c.get("button")

                break

        self._audition_character_voice(spk, card_btn)

    def _stop_audition_all(self):

        """Dừng hàng đợi nghe thử toàn bộ nhân vật."""

        self._audition_all_active = False

        self._audition_queue = []

        self._current_audition_speaker = None

        self._audition_all_current_idx = 0

        if hasattr(self, "btn_audition_all_speakers"):

            self.btn_audition_all_speakers.setText("Nghe Thử Toàn Bộ Nhân Vật")

            self.btn_audition_all_speakers.setIcon(FluentIcon.VOLUME)

        self._media_player.stop()

        self._reset_audition_button()

    def _run_pre_export_quality_check(self) -> List[Dict[str, Any]]:

        """Kiểm tra chất lượng trước khi xuất âm thanh hoặc video lồng tiếng."""

        issues: List[Dict[str, Any]] = []

        if not self.subtitle_blocks:

            issues.append({

                "severity": "error",

                "index": 0,

                "message": "Danh sách phụ đề đang trống. Vui lòng nạp tệp phụ đề trước khi xuất.",

            })

            return issues

        is_single_unified = hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked()
        if is_single_unified:
            preset_name = self.combo_single_unified_preset.currentText().strip() if hasattr(self, "combo_single_unified_preset") else ""
            if not preset_name or preset_name not in VIETNAMESE_VOICE_CATALOG:
                issues.append({
                    "severity": "error",
                    "index": 0,
                    "message": "Chưa chọn giọng đọc đồng nhất hợp lệ từ kho mẫu tiếng Việt.",
                })
            else:
                expected_ins = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
                for blk in self.subtitle_blocks:
                    spk_name = (getattr(blk, "speaker", "") or "").strip()
                    if spk_name not in ("Chưa phân vai", "Chưa xác định"):
                        blk.voice_instruct = expected_ins
                        blk.voice_ref_path = ""
                        blk.speaker = preset_name
                        blk.is_speaker_locked = True

        for blk in self.subtitle_blocks:

            spk = (getattr(blk, "speaker", "") or "").strip()

            conf = getattr(blk, "confidence_level", "high")

            text = (blk.translated_text if blk.translated_text else blk.text or "").strip()

            if not text:

                issues.append({

                    "severity": "warning",

                    "index": blk.index,

                    "message": f"Câu {blk.index}: Nội dung phụ đề đang trống.",

                })

            if not spk or spk in ("Chưa phân vai", "Chưa xác định") or conf in ("unassigned", "review_needed"):

                issues.append({

                    "severity": "warning",

                    "index": blk.index,

                    "message": f"Câu {blk.index}: Chưa xác định danh tính người nói hoặc chưa gán nhân vật.",

                })

            s_sec = parse_timestamp_to_seconds(blk.start_time)

            e_sec = parse_timestamp_to_seconds(blk.end_time)

            dur = max(0.0, e_sec - s_sec)

            if len(text) > 40 and dur > 0 and (len(text) / dur) > 25.0:

                issues.append({

                    "severity": "warning",

                    "index": blk.index,

                    "message": f"Câu {blk.index}: Thời lượng quá ngắn so với độ dài câu thoại, có nguy cơ đọc tràn mốc thời gian.",

                })

        return issues

    def _confirm_pre_export_warnings(self, warnings: List[Dict[str, Any]]) -> bool:

        """Hiển thị hộp thoại xác nhận khi phát hiện các câu phụ đề chưa phân vai hoặc cần kiểm tra."""

        from qfluentwidgets import MessageBox

        warn_lines = []

        for w in warnings[:5]:

            warn_lines.append(f"• {w.get('message', '')}")

        if len(warnings) > 5:

            warn_lines.append(f"• Và còn {len(warnings) - 5} cảnh báo khác...")

        warn_content = "\n".join(warn_lines)

        content = (

            f"Phát hiện {len(warnings)} câu phụ đề có thể chưa hoàn thiện trước khi xuất:\n\n"

            f"{warn_content}\n\n"

            "Bạn có muốn tiếp tục tiến trình xuất hay quay lại rà soát?"

        )

        msg_box = MessageBox("Cảnh Báo Trước Khi Xuất", content, self)

        msg_box.yesButton.setText("Tiếp Tục Xuất")

        msg_box.cancelButton.setText("Quay Lại Rà Soát")

        return bool(msg_box.exec())

    def _on_run_pre_export_qc(self):

        """Hiển thị kết quả kiểm tra chất lượng trước xuất."""

        issues = self._run_pre_export_quality_check()

        if not issues:

            self.lbl_qc_result.setText("Kết quả kiểm tra: Toàn bộ phụ đề và giọng nhân vật đã sẵn sàng để xuất.")

            InfoBar.success("Kiểm tra hoàn tất", "Không phát hiện lỗi hoặc cảnh báo nào.", parent=self, duration=3000)

        else:

            warn_cnt = len(issues)

            lines = [f"Phát hiện {warn_cnt} điểm cần lưu ý trước khi xuất:"]

            for itm in issues[:5]:

                lines.append(f"• {itm['message']}")

            if warn_cnt > 5:

                lines.append(f"... và {warn_cnt - 5} điểm khác.")

            summary_txt = "\n".join(lines)

            self.lbl_qc_result.setText(summary_txt)

            InfoBar.warning("Cảnh báo chất lượng", f"Có {warn_cnt} câu cần kiểm tra lại trước khi xuất.", parent=self, duration=4000)

    def _rebuild_instruct_string(self):

        """Tự động ghép chuỗi thuộc tính thiết kế giọng nói từ các hộp chọn."""

        parts = []

        for val in [

            GENDER_MAP.get(self.combo_gender.currentText(), ""),

            AGE_MAP.get(self.combo_age.currentText(), ""),

            PITCH_MAP.get(self.combo_pitch.currentText(), ""),

            STYLE_MAP.get(self.combo_style.currentText(), ""),

            ENGLISH_ACCENT_MAP.get(self.combo_accent.currentText(), ""),

            CHINESE_DIALECT_MAP.get(self.combo_dialect.currentText(), ""),

        ]:

            if val:

                parts.append(val)

        self.edit_instruct_preview.setText(", ".join(parts))

    def _insert_tag_into_studio(self, tag: str):

        """Chèn thẻ biểu cảm phi ngôn từ vào vị trí con trỏ trong trình soạn thảo."""

        cursor = self.studio_text_edit.textCursor()

        cursor.insertText(f" {tag} ")

        self.studio_text_edit.setFocus()

    def _insert_pronunciation_into_studio(self):

        """Chèn chuỗi ghi đè phiên âm nội tuyến vào trình soạn thảo phòng thu."""

        token = self.edit_pronun_token.text().strip()

        if not token:

            InfoBar.warning("Chưa nhập phiên âm", "Vui lòng nhập chuỗi phiên âm cần chèn.", parent=self, duration=2500)

            return

        cursor = self.studio_text_edit.textCursor()

        cursor.insertText(f" {token} ")

        self.studio_text_edit.setFocus()

    def _resolve_selected_language(self) -> Optional[str]:

        txt = self.combo_lang.currentText().strip()

        if txt in PRIORITY_LANGUAGE_MAP:

            return PRIORITY_LANGUAGE_MAP[txt]

        if "(" in txt and txt.endswith(")"):

            return txt.rsplit("(", 1)[-1].rstrip(")").strip()

        return txt if txt else None

    def _collect_engine_params(self) -> Dict[str, Any]:

        """Thu thập đầy đủ các tham số cấu hình cho TTSEngine."""

        self._save_ui_to_config()

        idx_to_mode = {0: "clone", 1: "design", 2: "auto"}

        mode = idx_to_mode.get(self.combo_mode.currentIndex(), "clone")

        params = {

            "mode": mode,

            "language": self._resolve_selected_language(),

            "speed": float(self.spin_speed.value()),

            "sync_to_subtitle_duration": self.chk_sync_sub_duration.isChecked(),

            "ref_audio_path": self.current_ref_audio_path if self.current_ref_audio_path else None,

            "ref_text": self.edit_ref_text.text().strip() or None,

            "auto_extract_ref": self.chk_auto_extract_video.isChecked(),

            "instruct": self.edit_instruct_preview.text().strip() or None,

            "num_step": int(self.spin_num_step.value()),

            "guidance_scale": float(self.spin_guidance.value()),

            "t_shift": float(self.spin_t_shift.value()),

            "layer_penalty_factor": float(self.spin_layer_penalty.value()),

            "position_temperature": float(self.spin_pos_temp.value()),

            "class_temperature": float(self.spin_cls_temp.value()),

            "audio_chunk_duration": float(self.spin_chunk_dur.value()),

            "audio_chunk_threshold": float(self.spin_chunk_thresh.value()),

            "pad_duration": float(self.spin_pad_dur.value()),

            "fade_duration": float(self.spin_fade_dur.value()),

            "denoise": self.chk_denoise.isChecked(),

            "preprocess_prompt": self.chk_preprocess_prompt.isChecked(),

            "postprocess_output": self.chk_postprocess_output.isChecked(),

            "normalize_text": self.chk_normalize_text.isChecked(),

            "enable_flashinfer": self.chk_flashinfer.isChecked(),

            "multi_speaker_mode": self._get_selected_multi_speaker_mode(),

            "max_speakers": int(self.spin_max_speakers.value()),

            "speaker_count_mode": "exact" if (hasattr(self, "combo_speaker_count_mode") and self.combo_speaker_count_mode.currentIndex() == 1) else "auto",

            "exact_speakers": int(self.spin_max_speakers.value()) if (hasattr(self, "combo_speaker_count_mode") and self.combo_speaker_count_mode.currentIndex() == 1) else None,

            "clustering_sensitivity": self._get_clustering_sensitivity_code(),

            "ai_vocal_separation": self.chk_ai_vocal_separation.isChecked() if hasattr(self, "chk_ai_vocal_separation") else True,

            "speaker_profiles": dict(self.speaker_profiles),

            "ducking_mode": "dynamic" if self.combo_ducking_mode.currentIndex() == 0 else "fixed",

            "resume_cache_dir": os.path.join(BASE_DIR, "output", "omnivoice_studio", "resume_cache"),

        }

        if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
            params["multi_speaker_mode"] = "single"
            params["mode"] = "design"
            params["ref_audio_path"] = None
            params["ref_text"] = None
            params["auto_extract_ref"] = False
            preset_name = self.combo_single_unified_preset.currentText().strip() if hasattr(self, "combo_single_unified_preset") else ""
            if preset_name in VIETNAMESE_VOICE_CATALOG:
                params["instruct"] = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
            if params.get("speaker_profiles"):
                for spk_key, prof in params["speaker_profiles"].items():
                    if isinstance(prof, dict):
                        prof["method"] = "preset"
                        prof["preset_name"] = preset_name
                        prof["voice_instruct"] = params.get("instruct", "")
                        prof["instruct"] = params.get("instruct", "")
                        prof["source_audio_path"] = ""
                        prof["ref_audio_path"] = ""
                        prof["status"] = "locked"
                    elif hasattr(prof, "method"):
                        prof.method = "preset"
                        prof.preset_name = preset_name
                        prof.voice_instruct = params.get("instruct", "")
                        prof.source_audio_path = ""
                        prof.status = "locked"

        return params

    def _on_speaker_count_mode_changed(self):

        """Cập nhật nhãn và trạng thái khi thay đổi quy tắc số lượng người nói."""

        if not hasattr(self, "combo_speaker_count_mode") or not hasattr(self, "lbl_max_spk"):

            return

        if self.combo_speaker_count_mode.currentIndex() == 1:

            self.lbl_max_spk.setText("Số lượng nhân vật chính xác:")

        else:

            self.lbl_max_spk.setText("Số lượng nhân vật tối đa:")

    def _get_clustering_sensitivity_code(self) -> str:

        """Lấy mã định danh độ nhạy phân cụm tương ứng."""

        if not hasattr(self, "combo_clustering_sensitivity"):

            return "standard"

        idx = self.combo_clustering_sensitivity.currentIndex()

        if idx == 1:

            return "high"

        elif idx == 2:

            return "low"

        return "standard"

    def _append_log(self, msg: str):

        if hasattr(self, "log_edit") and self.log_edit is not None:

            self.log_edit.append(msg)

    def log(self, msg: str):

        """Ghi nhật ký hệ thống lên giao diện người dùng."""

        self._append_log(msg)

    # =========================================================================

    # KHO 22 GIỌNG ĐỌC MẪU TIẾNG VIỆT & NGHE THỬ NHANH

    # =========================================================================

    def _on_vn_catalog_changed(self, index: int):

        preset_name = self.combo_vn_catalog.currentText().strip()

        info = VIETNAMESE_VOICE_CATALOG.get(preset_name)

        if info:

            self.edit_vn_sample_text.setText(info["sample_vi"])

            self.edit_instruct_preview.setText(info["instruct"])

    def _preview_vn_catalog_voice(self):

        preset_name = self.combo_vn_catalog.currentText().strip()

        info = VIETNAMESE_VOICE_CATALOG.get(preset_name, {})

        first_sub_text = ""
        if hasattr(self, "subtitle_blocks") and self.subtitle_blocks:
            for blk in self.subtitle_blocks:
                txt = (blk.translated_text if blk.translated_text else blk.text or "").strip()
                if txt:
                    first_sub_text = txt
                    break

        sample_text = (
            self.edit_vn_sample_text.text().strip()
            or first_sub_text
            or info.get(
                "sample_vi",
                "Xin chào các bạn, đây là giọng đọc mẫu tiếng Việt chạy trực tiếp trên card đồ họa.",
            )
        )

        instruct_str = info.get("instruct", "male, young adult, moderate pitch")

        slug = info.get("slug", "vn_sample")

        out_dir = os.path.join(BASE_DIR, "output", "voice_prompts")

        os.makedirs(out_dir, exist_ok=True)

        import hashlib
        text_hash = hashlib.md5(sample_text.encode("utf-8")).hexdigest()[:8]
        out_wav = os.path.join(out_dir, f"vn_sample_{slug}_{text_hash}.wav")

        self.edit_instruct_preview.setText(instruct_str)

        self._launch_worker(

            "vn_catalog_preview",

            text=sample_text,

            output_path=out_wav,

            override_instruct=instruct_str,

        )

    def _lock_vn_catalog_voice_as_ref(self):

        if not self.last_generated_audio_path or not os.path.exists(self.last_generated_audio_path):

            InfoBar.warning("Chưa có tệp nghe thử", "Vui lòng bấm Nghe Thử Tiếng Việt trước khi chốt giọng mẫu.", parent=self, duration=2500)

            return

        self.current_ref_audio_path = self.last_generated_audio_path

        sample_txt = self.edit_vn_sample_text.text().strip()

        if sample_txt:

            self.edit_ref_text.setText(sample_txt)

        self.combo_mode.setCurrentIndex(0)

        self.lbl_ref_status.setText(f"Tệp giọng mẫu tiếng Việt đã chốt: {self.current_ref_audio_path}")

        self._save_ui_to_config()

        self._update_control_states()

        InfoBar.success(

            "Đã chốt giọng mẫu tiếng Việt",

            f"Đã khóa mẫu giọng '{self.combo_vn_catalog.currentText()}' để sử dụng cố định cho toàn bộ dự án.",

            parent=self,

            duration=3500,

        )

    def _on_single_unified_voice_changed(self, state=None):
        is_single = self.chk_single_unified_voice.isChecked()
        self.combo_single_unified_preset.setEnabled(is_single)
        self.btn_preview_single_unified.setEnabled(is_single)
        self._update_identify_button_state()
        if is_single:
            self._apply_unified_voice_to_all_subtitles()

    def _on_unified_preset_changed(self, preset_name: str):
        if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
            self._apply_unified_voice_to_all_subtitles()

    def _apply_unified_voice_to_all_subtitles(self):
        if not hasattr(self, "combo_single_unified_preset"):
            return
        preset_name = self.combo_single_unified_preset.currentText().strip()
        if not preset_name or preset_name not in VIETNAMESE_VOICE_CATALOG:
            return
        instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
        for blk in self.subtitle_blocks:
            blk.voice_instruct = instruct
            blk.voice_ref_path = ""
            blk.speaker = preset_name
            blk.is_speaker_locked = True
        if self.speaker_profiles:
            for spk_key, prof in self.speaker_profiles.items():
                if isinstance(prof, dict):
                    prof["method"] = "preset"
                    prof["preset_name"] = preset_name
                    prof["voice_instruct"] = instruct
                    prof["instruct"] = instruct
                    prof["source_audio_path"] = ""
                    prof["ref_audio_path"] = ""
                    prof["status"] = "locked"
                elif hasattr(prof, "method"):
                    prof.method = "preset"
                    prof.preset_name = preset_name
                    prof.voice_instruct = instruct
                    prof.source_audio_path = ""
                    prof.status = "locked"
        if hasattr(self, "sub_table"):
            self._populate_subtitle_table()
        if hasattr(self, "casting_table"):
            self._update_casting_table()

    def _preview_unified_voice(self):
        preset_name = self.combo_single_unified_preset.currentText().strip()
        info = VIETNAMESE_VOICE_CATALOG.get(preset_name, {})
        first_sub_text = ""
        if hasattr(self, "subtitle_blocks") and self.subtitle_blocks:
            for blk in self.subtitle_blocks:
                t = (blk.translated_text if blk.translated_text else blk.text or "").strip()
                if t:
                    first_sub_text = t
                    break
        sample_text = first_sub_text or info.get(
            "sample_vi",
            "Xin chào các bạn, đây là giọng đọc mẫu tiếng Việt chạy trực tiếp trên card đồ họa.",
        )
        instruct_str = info.get("instruct", "male, young adult, moderate pitch")
        slug = info.get("slug", "unified_sample")
        out_dir = os.path.join(BASE_DIR, "output", "voice_prompts")
        os.makedirs(out_dir, exist_ok=True)
        import hashlib
        text_hash = hashlib.md5(sample_text.encode("utf-8")).hexdigest()[:8]
        out_wav = os.path.join(out_dir, f"unified_preview_{slug}_{text_hash}.wav")
        self._launch_worker(
            "vn_catalog_preview",
            text=sample_text,
            output_path=out_wav,
            override_instruct=instruct_str,
        )


    def _preview_speaker_preset_vi(self):

        preset_name = self.combo_speaker_voice_preset.currentText().strip()

        info = VIETNAMESE_VOICE_CATALOG.get(preset_name, {})

        target_spk = self.combo_target_speaker.currentText().strip() if hasattr(self, "combo_target_speaker") else ""
        first_sub_text = ""
        if hasattr(self, "subtitle_blocks") and self.subtitle_blocks:
            for blk in self.subtitle_blocks:
                spk = (getattr(blk, "speaker", "") or "").strip()
                if spk == target_spk:
                    txt = (blk.translated_text if blk.translated_text else blk.text or "").strip()
                    if txt:
                        first_sub_text = txt
                        break
            if not first_sub_text:
                for blk in self.subtitle_blocks:
                    txt = (blk.translated_text if blk.translated_text else blk.text or "").strip()
                    if txt:
                        first_sub_text = txt
                        break

        sample_text = first_sub_text or info.get(

            "sample_vi",

            f"Xin chào, đây là giọng thử nghiệm tiếng Việt của {target_spk}." if target_spk else "Xin chào các bạn, đây là giọng đọc mẫu tiếng Việt chạy trực tiếp trên card đồ họa.",

        )

        instruct_str = SPEAKER_VOICE_PRESETS.get(preset_name, "male, young adult, moderate pitch")

        slug = info.get("slug", "speaker_sample")

        out_dir = os.path.join(BASE_DIR, "output", "voice_prompts")

        os.makedirs(out_dir, exist_ok=True)

        import hashlib
        text_hash = hashlib.md5(sample_text.encode("utf-8")).hexdigest()[:8]
        out_wav = os.path.join(out_dir, f"vn_speaker_{slug}_{text_hash}.wav")

        self._launch_worker(

            "vn_catalog_preview",

            text=sample_text,

            output_path=out_wav,

            override_instruct=instruct_str,

        )

    # =========================================================================

    # XỬ LÝ NHÂN BẢN GIỌNG NÓI & MẪU GIỌNG ĐÃ MÃ HÓA

    # =========================================================================

    def _pick_ref_audio(self):

        filepath, _ = FolderMemoryDialog.getOpenFileName(

            self,

            "Chọn Tệp Âm Thanh Mẫu",

            filter_str="Tệp Âm Thanh (*.wav *.mp3 *.m4a *.flac *.ogg);;Tất cả tệp (*.*)",

            category="audio",

        )

        if filepath:

            self.current_ref_audio_path = filepath

            self.lbl_ref_status.setText(f"Tệp giọng mẫu hiện tại: {filepath}")

            self._save_ui_to_config()

            self._update_control_states()

            InfoBar.success("Đã nạp tệp giọng mẫu", os.path.basename(filepath), parent=self, duration=2500)

    def _extract_ref_from_video_dialog(self):

        filepath, _ = FolderMemoryDialog.getOpenFileName(

            self,

            "Chọn Video Để Tách Giọng Mẫu",

            filter_str="Tệp Video (*.mp4 *.mkv *.avi *.mov *.webm);;Tất cả tệp (*.*)",

            category="video",

        )

        if not filepath:

            return

        self._launch_worker("extract_ref_from_video", video_path=filepath)

    def _play_ref_audio(self):

        if self.current_ref_audio_path and os.path.exists(self.current_ref_audio_path):

            self._play_audio_file(self.current_ref_audio_path)

    def _load_prompt_pt_dialog(self):

        filepath, _ = QFileDialog.getOpenFileName(

            self,

            "Nạp Tệp Đặc Trưng Giọng Đã Mã Hóa",

            os.path.join(BASE_DIR, "output", "voice_prompts"),

            "Tệp Đặc Trưng Giọng (*.pt)",

        )

        if filepath:

            self.current_ref_audio_path = filepath

            self.lbl_ref_status.setText(f"Tệp đặc trưng giọng đang nạp: {filepath}")

            self._save_ui_to_config()

            self._update_control_states()

            InfoBar.success("Đã nạp mẫu giọng đã mã hóa", os.path.basename(filepath), parent=self, duration=2500)

    def _save_prompt_pt_dialog(self):

        if not self.current_ref_audio_path or not os.path.exists(self.current_ref_audio_path):

            InfoBar.warning("Chưa có giọng mẫu", "Vui lòng chọn tệp âm thanh giọng mẫu trước.", parent=self, duration=3000)

            return

        default_dir = os.path.join(BASE_DIR, "output", "voice_prompts")

        os.makedirs(default_dir, exist_ok=True)

        default_name = os.path.join(default_dir, Path(self.current_ref_audio_path).stem + ".pt")

        save_path, _ = QFileDialog.getSaveFileName(

            self,

            "Lưu Tệp Đặc Trưng Giọng Đã Mã Hóa",

            default_name,

            "Tệp Đặc Trưng Giọng (*.pt)",

        )

        if save_path:

            self._launch_worker("save_prompt", output_path=save_path)

    def _transcribe_ref_audio(self):

        if not self.current_ref_audio_path or not os.path.exists(self.current_ref_audio_path):

            return

        self._launch_worker("transcribe_ref")

    # =========================================================================

    # XỬ LÝ PHÒNG THU VĂN BẢN TRỰC TIẾP

    # =========================================================================

    def _run_studio_synthesis(self):

        text = self.studio_text_edit.toPlainText().strip()

        if not text:

            InfoBar.warning("Thiếu nội dung", "Vui lòng nhập văn bản cần tổng hợp giọng nói.", parent=self, duration=3000)

            return

        out_dir = os.path.join(BASE_DIR, "output", "omnivoice_studio")

        os.makedirs(out_dir, exist_ok=True)

        out_wav = os.path.join(out_dir, "omnivoice_studio_output.wav")

        self._launch_worker(

            "single_synth",

            text=text,

            output_path=out_wav,

            fixed_duration=float(self.spin_duration.value()),

        )

    def _save_studio_wav(self):

        if not self.last_generated_audio_path or not os.path.exists(self.last_generated_audio_path):

            return

        save_path, _ = QFileDialog.getSaveFileName(

            self,

            "Lưu Tệp Âm Thanh Đã Tổng Hợp",

            "giong_doc_omnivoice.wav",

            "Tệp Âm Thanh WAV (*.wav)",

        )

        if save_path:

            shutil.copy2(self.last_generated_audio_path, save_path)

            InfoBar.success("Đã lưu tệp âm thanh", save_path, parent=self, duration=3000)

    # =========================================================================

    # XỬ LÝ LỒNG TIẾNG TỆP PHỤ ĐỀ ĐA NHÂN VẬT & GHÉP VIDEO

    # =========================================================================

    def load_subtitle_and_video(

        self,

        srt_path: Optional[str] = None,

        video_path: Optional[str] = None,

        blocks: Optional[List[SubtitleBlock]] = None,

    ):

        """

        Khởi tạo một phiên làm việc lồng tiếng phụ đề mới khi nhận dữ liệu từ các thẻ khác.

        Đồng bộ trạng thái phụ đề và video nguồn, tránh giữ lại tệp cũ từ phiên trước.

        """

        self.workspace_pivot.setCurrentItem("sub_dub_page")

        self.workspace_stack.setCurrentIndex(1)

        # Đặt lại trạng thái phiên để không âm thầm giữ tệp video hoặc phụ đề cũ

        self.current_subtitle_path = srt_path if (srt_path and os.path.exists(srt_path)) else ""

        self.current_video_path = video_path if (video_path and os.path.exists(video_path)) else ""

        self.speaker_profiles = {}

        self.is_speakers_detected = False

        if blocks is not None:

            self.subtitle_blocks = [

                SubtitleBlock(

                    index=b.index,

                    start_time=b.start_time,

                    end_time=b.end_time,

                    text=b.text,

                    translated_text=b.translated_text,

                    speaker=getattr(b, "speaker", ""),

                    voice_ref_path=getattr(b, "voice_ref_path", ""),

                    voice_instruct=getattr(b, "voice_instruct", ""),

                    confidence_level=getattr(b, "confidence_level", "high"),

                    is_speaker_locked=getattr(b, "is_speaker_locked", False),

                )

                for b in blocks

            ]

            self._populate_subtitle_table()

            if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
                self._apply_unified_voice_to_all_subtitles()

            self._update_casting_table()

        elif self.current_subtitle_path:

            self._load_subtitle_file_internal(self.current_subtitle_path)

        else:

            self.subtitle_blocks = []

            self._populate_subtitle_table()

            if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
                self._apply_unified_voice_to_all_subtitles()

            self._update_casting_table()

        self._update_sub_video_status_label()
        if hasattr(self, 'casting_progress_bar'):
            self.casting_progress_bar.update_progress(self.subtitle_blocks)

        self._update_subtitle_action_buttons()

    def _clear_selected_video(self):

        """Gỡ video nguồn hiện tại để chuyển về chế độ chỉ xuất tệp âm thanh lồng tiếng."""

        self.current_video_path = ""

        self._update_sub_video_status_label()

        self._update_subtitle_action_buttons()

        InfoBar.success("Đã bỏ chọn video", "Đã gỡ video nguồn khỏi phiên lồng tiếng hiện tại.", parent=self, duration=2500)

    def _select_subtitle_file(self):

        filepath, _ = FolderMemoryDialog.getOpenFileName(

            self,

            "Chọn Tệp Phụ Đề",

            filter_str="Tệp Phụ Đề (*.srt *.ass *.vtt);;Tất cả tệp (*.*)",

            category="subtitle",

        )

        if filepath:

            self._load_subtitle_file_internal(filepath)

    def _load_subtitle_file_internal(self, filepath: str):

        try:

            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")

            blocks = parse_srt(content)

            if not blocks and filepath.lower().endswith(".ass"):

                blocks = self._parse_ass_fallback(content)

            self.subtitle_blocks = blocks

            self.current_subtitle_path = filepath

            self.speaker_profiles = {}

            self.is_speakers_detected = False

            self.lbl_speaker_summary.setText(

                "Mẹo: Bấm Tự Động Nhận Diện Nhiều Người Nói để hệ thống tự tách giọng từng nhân vật từ video hoặc phân vai hội thoại từ tệp phụ đề."

            )

            self._populate_subtitle_table()

            if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
                self._apply_unified_voice_to_all_subtitles()

            self._update_casting_table()

            self._update_sub_video_status_label()

            self._update_subtitle_action_buttons()

            if not blocks:

                InfoBar.error(

                    "Không tìm thấy phụ đề hợp lệ",

                    f"Tệp {os.path.basename(filepath)} không chứa câu thoại nào đúng định dạng SRT/VTT/ASS.",

                    parent=self,

                    duration=4000,

                )

            else:

                InfoBar.success("Đã nạp phụ đề", f"{len(blocks)} câu thoại từ {os.path.basename(filepath)}", parent=self, duration=2500)

        except Exception as e:

            InfoBar.error("Lỗi đọc phụ đề", str(e), parent=self, duration=4000)

    def _parse_ass_fallback(self, content: str) -> List[SubtitleBlock]:

        blocks: List[SubtitleBlock] = []

        idx = 1

        for line in content.splitlines():

            if line.startswith("Dialogue:"):

                parts = line.split(",", 9)

                if len(parts) >= 10:

                    start_t = parts[1].strip()

                    end_t = parts[2].strip()

                    speaker_col = parts[4].strip() if len(parts) > 4 else ""

                    txt = parts[9].replace("\\N", " ").strip()

                    blocks.append(

                        SubtitleBlock(

                            index=idx,

                            start_time=start_t,

                            end_time=end_t,

                            text=txt,

                            speaker=speaker_col,

                        )

                    )

                    idx += 1

        return blocks

    def _select_video_file(self):

        filepath, _ = FolderMemoryDialog.getOpenFileName(

            self,

            "Chọn Video Nguồn",

            filter_str="Tệp Video (*.mp4 *.mkv *.avi *.mov *.webm);;Tất cả tệp (*.*)",

            category="video",

        )

        if filepath:

            self.current_video_path = filepath

            self._update_sub_video_status_label()

            self._update_subtitle_action_buttons()

            InfoBar.success("Đã chọn video nguồn", os.path.basename(filepath), parent=self, duration=2500)

    def _update_sub_video_status_label(self):
        sub_name = os.path.basename(self.current_subtitle_path) if self.current_subtitle_path else "Chưa chọn"
        vid_name = os.path.basename(self.current_video_path) if self.current_video_path else "Chưa chọn"
        self.lbl_sub_video_info.setText(f"Phụ đề: {sub_name} — {len(self.subtitle_blocks)} câu | Video: {vid_name}")
        if hasattr(self, "sub_table"):
            has_vid = is_valid_media_file(self.current_video_path)
            for r in range(self.sub_table.rowCount()):
                cell_w = self.sub_table.cellWidget(r, 5)
                if cell_w:
                    cell_w.setEnabled(has_vid)
        self._update_identify_button_state()

    def _update_identify_button_state(self):
        """Cập nhật trạng thái nút nhận diện nhân vật và nhãn gợi ý theo nguồn đã nạp."""
        if not hasattr(self, "btn_identify_characters"):
            return

        if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
            self.btn_identify_characters.setEnabled(False)
            self.lbl_detect_source_hint.setText(
                "Đang chọn chế độ một giọng đọc đồng nhất cho toàn bộ tệp phụ đề. Bạn có thể chuyển thẳng sang Bước 3 hoặc Bước 5."
            )
            return

        has_subs = len(self.subtitle_blocks) > 0
        has_video = is_valid_media_file(self.current_video_path)

        if has_subs and has_video:
            self.btn_identify_characters.setEnabled(True)
            dur_info = check_media_and_srt_duration(self.current_video_path, self.subtitle_blocks)
            if dur_info.get("is_mismatch"):
                self.lbl_detect_source_hint.setText(dur_info["message"])
            else:
                self.lbl_detect_source_hint.setText(
                    "Sẵn sàng nhận diện người nói từ âm thanh video gốc và mốc thời gian phụ đề."
                )
        elif not has_subs and has_video:
            self.btn_identify_characters.setEnabled(True)
            self.lbl_detect_source_hint.setText(
                "Sẵn sàng nhận diện trực tiếp từ video: hệ thống sẽ tự động phát hiện các đoạn giọng nói và phân vai nhân vật."
            )
        elif has_subs and not has_video:
            self.btn_identify_characters.setEnabled(False)
            self.lbl_detect_source_hint.setText(
                "Chỉ có tệp phụ đề SRT — hãy thêm tệp video hợp lệ để nhận diện giọng nói thực, hoặc dùng 'Tự Động Phân Tích Lại' ở Bước 2 để phân vai theo văn bản."
            )
        else:
            self.btn_identify_characters.setEnabled(False)
            self.lbl_detect_source_hint.setText(
                "Nạp tệp video hoặc phụ đề để bắt đầu nhận diện nhân vật."
            )

    @staticmethod

    def _format_voice_source_display(blk: SubtitleBlock) -> str:

        ref_path = getattr(blk, "voice_ref_path", "")

        if ref_path and os.path.exists(ref_path):

            return "Nhân bản giọng từ video"

        if ref_path and not os.path.exists(ref_path):

            return "Nhân bản giọng · Thiếu tệp mẫu"

        instruct = getattr(blk, "voice_instruct", "")

        if instruct:

            return f"Giọng thiết kế: {instruct}"

        return "Chưa có giọng"

    def _populate_subtitle_table(self):
        self._reset_audition_button()
        self.sub_table.blockSignals(True)
        self.sub_table.setRowCount(len(self.subtitle_blocks))

        # Thu thập danh sách người nói có sẵn
        speaker_names = []
        if self.speaker_profiles:
            for k, p in self.speaker_profiles.items():
                disp = getattr(p, "display_name", "") or (p.get("display_name") if isinstance(p, dict) else "") or k
                if disp and disp not in speaker_names:
                    speaker_names.append(disp)
        if not speaker_names:
            for b in self.subtitle_blocks:
                s_name = getattr(b, "speaker", "")
                if s_name and s_name not in ("Chưa phân vai", "Chưa xác định") and s_name not in speaker_names:
                    speaker_names.append(s_name)
        if "Chưa xác định" not in speaker_names:
            speaker_names.append("Chưa xác định")

        has_valid_video = is_valid_media_file(self.current_video_path)

        for row, blk in enumerate(self.subtitle_blocks):
            conf = getattr(blk, "confidence_level", "high")
            is_locked = getattr(blk, "is_speaker_locked", False)
            spk_val = getattr(blk, "speaker", "")

            # Phối màu nền nhẹ nhàng trực quan phân biệt mức độ tin cậy
            if not is_locked and conf == "review_needed":
                row_bg = QColor(255, 248, 225, 180)  # Vàng nhạt cảnh báo cần rà soát
            elif not is_locked and (conf in ("unassigned", "") or not spk_val or spk_val in ("Chưa xác định", "unassigned")):
                row_bg = QColor(255, 235, 238, 180)  # Đỏ nhạt chưa xác định
            else:
                row_bg = QColor(0, 0, 0, 0)  # Trong suốt mặc định

            item_stt = QTableWidgetItem(str(blk.index))
            item_stt.setFlags(item_stt.flags() & ~Qt.ItemIsEditable)
            item_stt.setBackground(row_bg)
            self.sub_table.setItem(row, 0, item_stt)

            item_start = QTableWidgetItem(str(blk.start_time))
            item_start.setBackground(row_bg)
            self.sub_table.setItem(row, 1, item_start)

            item_end = QTableWidgetItem(str(blk.end_time))
            item_end.setBackground(row_bg)
            self.sub_table.setItem(row, 2, item_end)

            # Cột 3: Đặt QTableWidgetItem tương thích ngược và ComboBox trực quan
            spk = getattr(blk, "speaker", "") or "Chưa xác định"
            item_spk = QTableWidgetItem(spk)
            item_spk.setBackground(row_bg)
            self.sub_table.setItem(row, 3, item_spk)

            combo_row_spk = ComboBox(self.sub_table)
            combo_row_spk.addItems(speaker_names)
            if spk in speaker_names:
                combo_row_spk.setCurrentText(spk)
            else:
                combo_row_spk.addItem(spk)
                combo_row_spk.setCurrentText(spk)
            combo_row_spk.currentTextChanged.connect(
                lambda text, r=row: self._on_row_speaker_changed(r, text)
            )
            self.sub_table.setCellWidget(row, 3, combo_row_spk)

            # Cột 4: Nguồn giọng
            item_src = QTableWidgetItem(self._format_voice_source_display(blk))
            item_src.setFlags(item_src.flags() & ~Qt.ItemIsEditable)
            item_src.setBackground(row_bg)
            self.sub_table.setItem(row, 4, item_src)

            # Cột 5: Nút Nghe Gốc
            btn_listen = PushButton("Nghe", self.sub_table)
            btn_listen.setIcon(FluentIcon.PLAY)
            btn_listen.setToolTip("Phát âm thanh gốc từ video theo mốc thời gian của câu này")
            btn_listen.setEnabled(has_valid_video)
            btn_listen.clicked.connect(
                lambda chk=False, b=blk, btn=btn_listen: self._play_subtitle_original_segment(b, btn)
            )
            self.sub_table.setCellWidget(row, 5, btn_listen)

            # Cột 6: Nội Dung Thoại
            txt = blk.translated_text if blk.translated_text else blk.text
            item_txt = QTableWidgetItem(txt)
            item_txt.setBackground(row_bg)
            self.sub_table.setItem(row, 6, item_txt)

        self.sub_table.blockSignals(False)
        self._filter_subtitle_table()
        self._update_character_profiles_overview()

    
    def _on_row_speaker_changed(self, row: int, new_spk: str):
        """Xử lý khi người dùng chọn nhân vật trực tiếp trên dòng phụ đề, tự động khóa câu thoại."""
        if row < 0 or row >= len(self.subtitle_blocks):
            return
        blk = self.subtitle_blocks[row]
        blk.speaker = new_spk
        item_spk = self.sub_table.item(row, 3)
        if item_spk and item_spk.text() != new_spk:
            self.sub_table.blockSignals(True)
            item_spk.setText(new_spk)
            self.sub_table.blockSignals(False)
        if new_spk and new_spk != "Chưa xác định":
            blk.is_speaker_locked = True
            blk.confidence_level = "high"
            prof = self.speaker_profiles.get(new_spk)
            if prof:
                is_dict = isinstance(prof, dict)
                method = (prof.get("method") if is_dict else getattr(prof, "method", "clone")) or "clone"
                if method == "preset":
                    blk.voice_ref_path = ""
                    blk.voice_instruct = (prof.get("voice_instruct") or prof.get("instruct") if is_dict else getattr(prof, "voice_instruct", "")) or ""
                else:
                    blk.voice_ref_path = (prof.get("source_audio_path") or prof.get("ref_audio_path") or prof.get("source_audio_sample") if is_dict else getattr(prof, "source_audio_path", "")) or ""
                    blk.voice_instruct = ""
        else:
            blk.is_speaker_locked = False
            blk.confidence_level = "unassigned"
            blk.voice_ref_path = ""
            blk.voice_instruct = ""

        # Cập nhật ô nguồn giọng ở cột 4
        item_src = QTableWidgetItem(self._format_voice_source_display(blk))
        item_src.setFlags(item_src.flags() & ~Qt.ItemIsEditable)
        self.sub_table.blockSignals(True)
        self.sub_table.setItem(row, 4, item_src)
        self.sub_table.blockSignals(False)

        # Cập nhật lại màu nền dòng sau khi người dùng đổi nhân vật
        conf_now = getattr(blk, "confidence_level", "high")
        is_locked_now = getattr(blk, "is_speaker_locked", False)
        spk_now = getattr(blk, "speaker", "")
        if not is_locked_now and conf_now == "review_needed":
            new_bg = QColor(255, 248, 225, 180)
        elif not is_locked_now and (conf_now in ("unassigned", "") or not spk_now or spk_now in ("Chưa xác định", "unassigned")):
            new_bg = QColor(255, 235, 238, 180)
        else:
            new_bg = QColor(0, 0, 0, 0)

        for col_idx in (0, 1, 2, 3, 4, 6):
            cell_item = self.sub_table.item(row, col_idx)
            if cell_item:
                cell_item.setBackground(new_bg)

        if hasattr(self, "casting_progress_bar"):
            self.casting_progress_bar.update_progress(self.subtitle_blocks)

        self._update_casting_table()
        self._update_character_profiles_overview()

    def _play_subtitle_original_segment(self, blk: SubtitleBlock, btn: Optional[PushButton] = None):
        """Phát đoạn âm thanh gốc từ video theo mốc thời gian của câu phụ đề."""
        if (
            btn is not None
            and getattr(self, "_active_audition_button", None) == btn
            and self._media_player.playbackState() == QMediaPlayer.PlayingState
        ):
            self._media_player.stop()
            self._reset_audition_button()
            return

        if not is_valid_media_file(self.current_video_path):
            InfoBar.warning(
                "Không có tệp video",
                "Vui lòng nạp tệp video nguồn để nghe đoạn âm thanh gốc theo câu thoại.",
                parent=self,
                duration=3000,
            )
            return

        start_s = parse_timestamp_to_seconds(blk.start_time)
        end_s = parse_timestamp_to_seconds(blk.end_time)
        dur = max(0.2, end_s - start_s)

        cache_dir = getattr(self, "voice_prompt_cache_dir", "") or os.path.join(tempfile.gettempdir(), "sakai_omnivoice_cache")
        self.voice_prompt_cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        out_wav = os.path.join(
            cache_dir,
            f"seg_{blk.index}_{int(start_s * 1000)}_{int(end_s * 1000)}.wav",
        )

        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            import subprocess
            from src.core.tools.ffmpeg_cli import FFmpegCLI
            ffmpeg_bin = getattr(FFmpegCLI.instance(), "ffmpeg_path", "ffmpeg") or "ffmpeg"
            cmd = [
                ffmpeg_bin,
                "-y",
                "-ss", str(max(0.0, start_s)),
                "-t", str(dur),
                "-i", self.current_video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                out_wav,
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception as e:
                InfoBar.error(
                    "Lỗi trích xuất âm thanh",
                    f"Không thể cắt đoạn âm thanh từ video: {e}",
                    parent=self,
                    duration=3500,
                )
                return

        if os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0:
            self._reset_audition_button()
            if btn is not None and self._is_valid_widget(btn):
                try:
                    self._active_audition_button = btn
                    btn.setProperty("default_text", btn.text())
                    btn.setIcon(FluentIcon.PAUSE)
                except RuntimeError:
                    self._active_audition_button = None
            self._play_audio_file(out_wav)

    def _on_subtitle_table_edited(self, item: QTableWidgetItem):

        row = item.row()

        col = item.column()

        if 0 <= row < len(self.subtitle_blocks):

            blk = self.subtitle_blocks[row]

            if col in (1, 2):

                new_val = item.text().strip()

                cand_start = new_val if col == 1 else blk.start_time

                cand_end = new_val if col == 2 else blk.end_time

                s_sec = parse_timestamp_to_seconds(cand_start)

                e_sec = parse_timestamp_to_seconds(cand_end)

                if not new_val or ":" not in new_val or s_sec < 0 or e_sec <= s_sec:

                    self.sub_table.blockSignals(True)

                    item.setText(str(blk.start_time if col == 1 else blk.end_time))

                    self.sub_table.blockSignals(False)

                    InfoBar.warning(

                        "Mốc thời gian không hợp lệ",

                        "Mốc kết thúc phải lớn hơn mốc bắt đầu và đúng định dạng HH:MM:SS,mmm.",

                        parent=self,

                        duration=3000,

                    )

                    return

                if col == 1:

                    blk.start_time = new_val

                else:

                    blk.end_time = new_val

            elif col == 3:
                new_spk = item.text().strip()
                w = self.sub_table.cellWidget(row, 3)
                if isinstance(w, ComboBox):
                    w.blockSignals(True)
                    if w.findText(new_spk) < 0:
                        w.addItem(new_spk)
                    w.setCurrentText(new_spk)
                    w.blockSignals(False)
                self._on_row_speaker_changed(row, new_spk)
                blk.speaker = new_spk

                prof = self.speaker_profiles.get(new_spk)

                if prof:

                    is_dict = isinstance(prof, dict)

                    method = (prof.get("method") if is_dict else getattr(prof, "method", "clone")) or "clone"

                    if method == "preset":

                        blk.voice_ref_path = ""

                        blk.voice_instruct = (prof.get("voice_instruct") or prof.get("instruct") if is_dict else getattr(prof, "voice_instruct", "")) or ""

                    else:

                        blk.voice_ref_path = (prof.get("source_audio_path") or prof.get("ref_audio_path") or prof.get("source_audio_sample") if is_dict else getattr(prof, "source_audio_path", "")) or ""

                        blk.voice_instruct = ""

                    blk.is_speaker_locked = True

                elif new_spk and new_spk != "Chưa xác định":

                    blk.voice_ref_path = ""

                    blk.voice_instruct = ""

                    blk.is_speaker_locked = True

                    self.speaker_profiles[new_spk] = CharacterVoiceProfile(

                        speaker_id=f"spk_{len(self.speaker_profiles)}",

                        display_name=new_spk,

                    )

                else:

                    blk.voice_ref_path = ""

                    blk.voice_instruct = ""

                    blk.is_speaker_locked = False

                item_src = QTableWidgetItem(self._format_voice_source_display(blk))

                item_src.setFlags(item_src.flags() & ~Qt.ItemIsEditable)

                self.sub_table.blockSignals(True)

                self.sub_table.setItem(row, 4, item_src)

                self.sub_table.blockSignals(False)

                self._update_casting_table()

                self._update_character_profiles_overview()

            elif col in (5, 6):

                blk.translated_text = item.text()

                if not blk.text:

                    blk.text = item.text()

    def _update_subtitle_action_buttons(self):

        has_subs = len(self.subtitle_blocks) > 0

        has_video = bool(self.current_video_path and os.path.exists(self.current_video_path))

        has_selection = len(self.sub_table.selectedItems()) > 0

        self.btn_clear_video.setEnabled(has_video)

        self.btn_auto_detect_speakers.setEnabled(has_subs)

        self.btn_preview_selected_row.setEnabled(has_subs and has_selection)

        if hasattr(self, "btn_insert_tag_to_sub"):

            self.btn_insert_tag_to_sub.setEnabled(has_subs and has_selection)

        self.btn_export_dub_wav.setEnabled(has_subs)

        self.btn_dub_merge_video.setEnabled(has_subs and has_video)

    # =========================================================================

    # TỰ ĐỘNG NHẬN DIỆN NHIỀU NGƯỜI NÓI & GÁN GIỌNG NHÂN VẬT

    # =========================================================================

    def _run_auto_detect_speakers(self):

        if self.combo_hardware_selector.currentIndex() == 2:

            InfoBar.warning(

                "Chế độ chưa sẵn sàng",

                "Phân đoạn người nói chuyên sâu chưa được tích hợp. Hãy chọn CUDA hoặc CPU để tiếp tục.",

                parent=self,

                duration=4000,

            )

            return

        has_subs = len(self.subtitle_blocks) > 0
        has_video = is_valid_media_file(self.current_video_path)
        if not has_subs and not has_video:
            InfoBar.warning("Chưa có dữ liệu", "Vui lòng nạp tệp video hoặc tệp phụ đề trước khi phân tích.", parent=self, duration=3000)
            return

        if has_video:
            self.combo_multi_speaker_mode.setCurrentIndex(1)  # Tự động nhận diện từ video
        else:
            self.combo_multi_speaker_mode.setCurrentIndex(3)  # Tự động phân vai từ văn bản phụ đề

        self._launch_worker(

            "detect_speakers",

            video_path=self.current_video_path,

            subtitle_blocks=self.subtitle_blocks,

        )

    def _on_speakers_detected(self, profiles: dict):

        if not profiles:

            self.is_speakers_detected = False

            self.speaker_profiles = {}

            if hasattr(self, "lbl_diarization_summary"):

                self.lbl_diarization_summary.setText(

                    "Không tìm thấy nhân vật nào từ nguồn dữ liệu. "

                    "Vui lòng kiểm tra lại tệp âm thanh hoặc thử phân vai từ văn bản phụ đề."

                )

            if hasattr(self, "lbl_diarization_method"):

                self.lbl_diarization_method.setText("Trạng thái: Không đủ dữ liệu âm thanh.")

            InfoBar.warning(

                "Không tìm thấy nhân vật",

                "Không phát hiện được đoạn thoại hoặc đặc trưng giọng nói đủ điều kiện. Bạn có thể thử phân vai theo văn bản phụ đề.",

                parent=self,

                duration=4000,

            )

            return

        self.is_speakers_detected = True

        normalized: Dict[str, CharacterVoiceProfile] = {}

        catalog_keys = list(VIETNAMESE_VOICE_CATALOG.keys())
        for idx, (spk, prof) in enumerate((profiles or {}).items()):
            sid = getattr(prof, "speaker_id", "") or (prof.get("speaker_id") if isinstance(prof, dict) else "") or f"spk_{idx}"
            norm = normalize_character_profile(prof, speaker_id=sid)
            if not getattr(norm, "display_name", ""):
                norm.display_name = str(spk)
            # Luôn gán giọng trí tuệ nhân tạo từ kho mẫu tiếng Việt
            if not getattr(norm, "preset_name", "") or getattr(norm, "method", "") != "preset":
                assigned_preset = catalog_keys[idx % len(catalog_keys)]
                norm.method = "preset"
                norm.preset_name = assigned_preset
                norm.source_audio_path = ""
                norm.voice_instruct = VIETNAMESE_VOICE_CATALOG[assigned_preset]["instruct"]
                norm.status = "locked"
            normalized[spk] = norm

        self.speaker_profiles = normalized
        self._sync_speaker_selectors()

        # Đồng bộ lại thông tin phân vai từ bản sao phụ đề của worker về danh sách UI
        if self._worker and self._worker.subtitle_blocks:
            if not self.subtitle_blocks:
                self.subtitle_blocks = list(self._worker.subtitle_blocks)
            elif len(self._worker.subtitle_blocks) == len(self.subtitle_blocks):
                for ui_blk, w_blk in zip(self.subtitle_blocks, self._worker.subtitle_blocks):
                    ui_blk.speaker = getattr(w_blk, "speaker", ui_blk.speaker)
                    ui_blk.voice_ref_path = ""
                    spk_name = getattr(ui_blk, "speaker", "")
                    if spk_name in normalized:
                        ui_blk.voice_instruct = getattr(normalized[spk_name], "voice_instruct", "")
                    else:
                        ui_blk.voice_instruct = getattr(w_blk, "voice_instruct", ui_blk.voice_instruct)
                    ui_blk.confidence_level = getattr(w_blk, "confidence_level", getattr(ui_blk, "confidence_level", "high"))
                    ui_blk.is_speaker_locked = getattr(w_blk, "is_speaker_locked", getattr(ui_blk, "is_speaker_locked", False))
        else:
            for blk in self.subtitle_blocks:
                spk_name = getattr(blk, "speaker", "")
                if spk_name in normalized:
                    blk.voice_instruct = getattr(normalized[spk_name], "voice_instruct", "")
                    blk.voice_ref_path = ""

        self._populate_subtitle_table()

        self._update_casting_table()

        # Cập nhật nhãn tóm tắt kết quả nhận diện

        total_chars = len(self.speaker_profiles)

        total_assigned = sum(

            1 for blk in self.subtitle_blocks

            if getattr(blk, "speaker", "") and getattr(blk, "speaker", "") not in ("Chưa xác định", "unassigned", "")

        )

        total_locked = sum(1 for blk in self.subtitle_blocks if getattr(blk, "is_speaker_locked", False))

        review_count = sum(1 for blk in self.subtitle_blocks if getattr(blk, "confidence_level", "") == "review_needed")

        unassigned_count = sum(

            1 for blk in self.subtitle_blocks

            if getattr(blk, "confidence_level", "") == "unassigned"

            or not getattr(blk, "speaker", "")

            or getattr(blk, "speaker", "") in ("Chưa xác định", "unassigned")

        )

        total_subs = len(self.subtitle_blocks)

        summary = (

            f"Đã phân tích xong. Tìm thấy {total_chars} nhân vật, "

            f"{total_assigned}/{total_subs} câu thoại được phân vai"

        )

        if total_locked > 0:

            summary += f", {total_locked} câu đã khóa"

        if review_count > 0:

            summary += f", {review_count} câu cần rà soát"

        if unassigned_count > 0:

            summary += f", {unassigned_count} câu chưa xác định"

        summary += "."

        if hasattr(self, "lbl_diarization_summary"):

            self.lbl_diarization_summary.setText(summary)
        if hasattr(self, 'casting_progress_bar'):
            self.casting_progress_bar.update_progress(self.subtitle_blocks)

        has_video = is_valid_media_file(self.current_video_path)

        method_text = "Nguồn: Nhận diện từ âm thanh video gốc." if has_video else "Nguồn: Phân vai từ văn bản phụ đề SRT."

        if hasattr(self, "lbl_diarization_method"):

            self.lbl_diarization_method.setText(method_text)

        summary_parts = [

            f"{spk}: {info.get('description', '')} ({info.get('count', 0)} câu)"

            for spk, info in self.speaker_profiles.items()

        ]

        if summary_parts:

            self.lbl_speaker_summary.setText("Kết quả phân vai: " + " | ".join(summary_parts))

        self._switch_to_step(1)

    def _apply_preset_to_target_speaker(self):

        target_spk = self.combo_target_speaker.currentText().strip()

        preset_label = self.combo_speaker_voice_preset.currentText().strip()

        instruct_val = SPEAKER_VOICE_PRESETS.get(preset_label, "male, young adult, moderate pitch")

        if target_spk not in self.speaker_profiles:

            self.speaker_profiles[target_spk] = {"speaker_id": target_spk, "count": 0}

        prof = self.speaker_profiles[target_spk]

        if hasattr(prof, "method"):

            prof.method = "preset"

            prof.preset_name = preset_label

            prof.voice_instruct = instruct_val

            prof.source_audio_path = ""

            prof.custom_audio_path = ""

            prof.voice_clone_prompt_path = ""

            prof.status = "locked"

            if hasattr(prof, "ref_audio_path"):

                setattr(prof, "ref_audio_path", "")

        else:

            prof["method"] = "preset"

            prof["preset_name"] = preset_label

            prof["voice_preset"] = preset_label

            prof["instruct"] = instruct_val

            prof["voice_instruct"] = instruct_val

            prof["ref_audio_path"] = ""

            prof["source_audio_path"] = ""

            prof["source_audio_sample"] = ""

            prof["prompt"] = None

            prof["description"] = preset_label

            prof["status"] = "locked"

        updated = 0

        for blk in self.subtitle_blocks:

            if (getattr(blk, "speaker", "") or "Người nói 1") == target_spk:

                blk.speaker = target_spk

                blk.voice_instruct = instruct_val

                blk.voice_ref_path = ""

                blk.is_speaker_locked = True

                updated += 1

        if self.combo_multi_speaker_mode.currentIndex() == 0:

            self.combo_multi_speaker_mode.setCurrentIndex(4)  # Phân vai tùy chỉnh theo bảng

        self._populate_subtitle_table()

        self._update_casting_table()

        self._update_character_profiles_overview()

        InfoBar.success(

            "Đã cập nhật thiết kế giọng nhân vật",

            f"Đã gán '{preset_label}' cho {target_spk} — Cập nhật {updated} câu thoại.",

            parent=self,

            duration=2500,

        )

    def _pick_audio_for_target_speaker(self):

        target_spk = self.combo_target_speaker.currentText().strip()

        filepath, _ = FolderMemoryDialog.getOpenFileName(

            self,

            f"Chọn Tệp Giọng Mẫu Cho {target_spk}",

            filter_str="Tệp Giọng Mẫu (*.wav *.mp3 *.m4a *.flac *.pt);;Tất cả tệp (*.*)",

            category="audio",

        )

        if not filepath:

            return

        if target_spk not in self.speaker_profiles:

            self.speaker_profiles[target_spk] = {"speaker_id": target_spk, "count": 0}

        prof = self.speaker_profiles[target_spk]

        if hasattr(prof, "method"):

            prof.method = "clone"

            prof.source_audio_path = filepath

            prof.custom_audio_path = filepath

            prof.voice_instruct = ""

            prof.status = "locked"

            if hasattr(prof, "ref_audio_path"):

                setattr(prof, "ref_audio_path", filepath)

        else:

            prof["method"] = "clone"

            prof["source_audio_path"] = filepath

            prof["ref_audio_path"] = filepath

            prof["source_audio_sample"] = filepath

            prof["custom_audio_path"] = filepath

            prof["voice_instruct"] = ""

            prof["description"] = os.path.basename(filepath)

            prof["status"] = "locked"

        updated = 0

        for blk in self.subtitle_blocks:

            if (getattr(blk, "speaker", "") or "Người nói 1") == target_spk:

                blk.speaker = target_spk

                blk.voice_ref_path = filepath

                blk.voice_instruct = ""

                blk.is_speaker_locked = True

                updated += 1

        if self.combo_multi_speaker_mode.currentIndex() == 0:

            self.combo_multi_speaker_mode.setCurrentIndex(4)

        self._populate_subtitle_table()

        self._update_casting_table()

        self._update_character_profiles_overview()

        InfoBar.success(

            "Đã gán tệp giọng mẫu cho nhân vật",

            f"{target_spk} sử dụng mẫu: {os.path.basename(filepath)} — Cập nhật {updated} câu thoại.",

            parent=self,

            duration=2500,

        )

    def _assign_target_speaker_to_selected_rows(self):

        selected_rows = sorted({item.row() for item in self.sub_table.selectedItems()})

        if not selected_rows:

            InfoBar.warning("Chưa chọn dòng", "Vui lòng bôi đen các dòng phụ đề cần gán nhân vật.", parent=self, duration=2500)

            return

        target_spk = self.combo_target_speaker.currentText().strip()

        prof = self.speaker_profiles.get(target_spk, {})

        is_dict = isinstance(prof, dict)

        method = (prof.get("method") if is_dict else getattr(prof, "method", "preset")) if prof else "preset"

        if method == "clone":

            ref_path = (prof.get("ref_audio_path") or prof.get("source_audio_path") if is_dict else (getattr(prof, "source_audio_path", "") or getattr(prof, "ref_audio_path", ""))) if prof else ""

            ins_val = ""

        else:

            ref_path = ""

            ins_val = (prof.get("voice_instruct") or prof.get("instruct") if is_dict else getattr(prof, "voice_instruct", "")) if prof else ""

            if not ins_val:

                preset_name = (prof.get("preset_name") if is_dict else getattr(prof, "preset_name", "")) if prof else ""

                if preset_name in VIETNAMESE_VOICE_CATALOG:

                    ins_val = VIETNAMESE_VOICE_CATALOG[preset_name].get("instruct", "")

                elif target_spk in DEFAULT_SPEAKER_PRESETS:

                    ins_val = DEFAULT_SPEAKER_PRESETS[target_spk].get("instruct", "")

        for r in selected_rows:

            if 0 <= r < len(self.subtitle_blocks):

                blk = self.subtitle_blocks[r]

                blk.speaker = target_spk

                blk.voice_ref_path = ref_path

                blk.voice_instruct = ins_val

                blk.is_speaker_locked = True

        if self.combo_multi_speaker_mode.currentIndex() == 0:

            self.combo_multi_speaker_mode.setCurrentIndex(4)

        self._populate_subtitle_table()

        self._update_character_profiles_overview()

        InfoBar.success(

            "Đã gán nhân vật",

            f"Đã gán {len(selected_rows)} câu thoại đang chọn cho {target_spk}.",

            parent=self,

            duration=2500,

        )

    def _reset_target_speaker_locked_voice(self):

        """Xóa mẫu giọng tự động khóa của nhân vật đang chọn để hệ thống tạo lại mẫu mới sạch hơn."""

        target_spk = self.combo_target_speaker.currentText().strip()

        prof = self.speaker_profiles.get(target_spk)

        if prof and "locked_" in str(prof.get("ref_audio_path", "")):

            locked_path = prof.get("ref_audio_path", "")

            prof["ref_audio_path"] = ""

            if locked_path and os.path.exists(locked_path):

                try:

                    os.remove(locked_path)

                except Exception:

                    pass

        for blk in self.subtitle_blocks:

            if getattr(blk, "speaker", "") == target_spk and "locked_" in str(getattr(blk, "voice_ref_path", "")):

                blk.voice_ref_path = ""

        self._populate_subtitle_table()

        InfoBar.success(

            "Đã đặt lại mẫu giọng nhân vật",

            f"Đã xóa mẫu giọng khóa tự động của {target_spk}. Lần tổng hợp tới sẽ tạo lại mẫu mới.",

            parent=self,

            duration=3000,

        )

    def _insert_expressive_tag_to_selected_sub(self):

        row = self.sub_table.currentRow()

        if row < 0 or row >= len(self.subtitle_blocks):

            InfoBar.warning("Chưa chọn câu phụ đề", "Vui lòng chọn một dòng trong bảng phụ đề để chèn biểu cảm.", parent=self, duration=2500)

            return

        idx = self.combo_sub_tag.currentIndex()

        if 0 <= idx < len(EXPRESSIVE_TAG_BUTTONS):

            tag_code = EXPRESSIVE_TAG_BUTTONS[idx][1]

            blk = self.subtitle_blocks[row]

            cur_text = blk.translated_text if blk.translated_text else blk.text

            new_text = f"{tag_code} {cur_text}".strip()

            blk.translated_text = new_text

            if not blk.text:

                blk.text = cur_text

            self._populate_subtitle_table()

            self.sub_table.selectRow(row)

    def _preview_selected_subtitle_row(self):

        row = self.sub_table.currentRow()

        if row < 0 or row >= len(self.subtitle_blocks):

            return

        blk = self.subtitle_blocks[row]

        text = blk.translated_text if blk.translated_text else blk.text

        start_s = parse_timestamp_to_seconds(blk.start_time)

        end_s = parse_timestamp_to_seconds(blk.end_time)

        slot_dur = max(0.0, end_s - start_s) if self.chk_sync_sub_duration.isChecked() else 0.0

        # Lấy giọng riêng của nhân vật ở dòng đang chọn nếu có

        spk_key = getattr(blk, "speaker", "")

        prof = self.speaker_profiles.get(spk_key, {}) if spk_key else {}

        prof_method = getattr(prof, "method", None) or (prof.get("method") if isinstance(prof, dict) else "")

        if prof_method == "preset":

            row_mode = "design"

            row_ref = ""

            row_ins = (

                getattr(blk, "voice_instruct", "")

                or getattr(prof, "voice_instruct", "")

                or (prof.get("voice_instruct", "") if isinstance(prof, dict) else "")

                or getattr(prof, "instruct", "")

                or (prof.get("instruct", "") if isinstance(prof, dict) else "")

            )

        elif prof_method == "clone":

            row_mode = "clone"

            row_ref = (

                getattr(blk, "voice_ref_path", "")

                or getattr(prof, "source_audio_path", "")

                or (prof.get("source_audio_path", "") if isinstance(prof, dict) else "")

                or getattr(prof, "ref_audio_path", "")

                or (prof.get("ref_audio_path", "") if isinstance(prof, dict) else "")

            )

            row_ins = ""

        else:

            row_ref = getattr(blk, "voice_ref_path", "") or (prof.get("ref_audio_path", "") if isinstance(prof, dict) else getattr(prof, "ref_audio_path", ""))

            row_ins = getattr(blk, "voice_instruct", "") or (prof.get("instruct", "") if isinstance(prof, dict) else getattr(prof, "instruct", ""))

            if row_ins and not row_ref:

                row_mode = "design"

            elif row_ref and not row_ins:

                row_mode = "clone"

            else:

                row_mode = None

        out_dir = os.path.join(BASE_DIR, "output", "omnivoice_studio")

        os.makedirs(out_dir, exist_ok=True)

        out_wav = os.path.join(out_dir, f"preview_line_{row + 1}.wav")

        self._launch_worker(

            "block_preview",

            text=text,

            output_path=out_wav,

            fixed_duration=slot_dur,

            override_ref_audio=row_ref,

            override_instruct=row_ins,

            override_mode=row_mode,

        )

    def _export_subtitle_audio_track(self):

        if not self.subtitle_blocks:

            InfoBar.error("Chưa có phụ đề", "Danh sách phụ đề đang trống. Vui lòng nạp phụ đề trước khi xuất.", parent=self, duration=3500)

            return

        issues = self._run_pre_export_quality_check()

        if any(item.get("severity") == "error" for item in issues):

            InfoBar.error("Không thể xuất", "Danh sách phụ đề đang trống hoặc không hợp lệ.", parent=self, duration=3500)

            return

        warnings = [item for item in issues if item.get("severity") == "warning"]

        if warnings and not self._confirm_pre_export_warnings(warnings):

            return

        default_name = (

            str(Path(self.current_subtitle_path).with_name(Path(self.current_subtitle_path).stem + "_dubbed.wav"))

            if self.current_subtitle_path

            else "track_long_tieng.wav"

        )

        save_path, _ = QFileDialog.getSaveFileName(

            self,

            "Lưu Tệp Âm Thanh Lồng Tiếng",

            default_name,

            "Tệp Âm Thanh WAV (*.wav)",

        )

        if save_path:

            self._launch_worker(

                "subtitle_audio_only",

                output_path=save_path,

                video_path=self.current_video_path,

                subtitle_blocks=self.subtitle_blocks,

            )

    def _dub_and_merge_video(self):

        if not self.subtitle_blocks or not self.current_video_path:

            InfoBar.warning("Thiếu dữ liệu", "Cần có danh sách phụ đề và video nguồn để xuất video lồng tiếng.", parent=self, duration=3500)

            return

        issues = self._run_pre_export_quality_check()

        if any(item.get("severity") == "error" for item in issues):

            InfoBar.error("Không thể xuất", "Danh sách phụ đề đang trống hoặc không hợp lệ.", parent=self, duration=3500)

            return

        warnings = [item for item in issues if item.get("severity") == "warning"]

        if warnings and not self._confirm_pre_export_warnings(warnings):

            return

        default_name = str(Path(self.current_video_path).with_name(Path(self.current_video_path).stem + "_long_tieng.mp4"))

        save_path, _ = QFileDialog.getSaveFileName(

            self,

            "Lưu Video Đã Lồng Tiếng",

            default_name,

            "Tệp Video MP4 (*.mp4)",

        )

        if save_path:

            self._launch_worker(

                "dub_full_video",

                output_path=save_path,

                video_path=self.current_video_path,

                subtitle_blocks=self.subtitle_blocks,

                original_volume=float(self.spin_duck_vol.value()),

            )

    # =========================================================================

    # ĐIỀU PHỐI LUỒNG NỀN & TRÌNH PHÁT ÂM THANH

    # =========================================================================

    def _set_task_running_state(self, running: bool):

        """Quản lý tập trung trạng thái khóa/mở khóa bảng phụ đề và toàn bộ điều khiển khi luồng nền hoạt động."""

        self.btn_studio_synth.setEnabled(not running)

        self.btn_preview_vn_voice.setEnabled(not running)

        self.btn_preview_speaker_preset_vi.setEnabled(not running)

        self.btn_cancel_task.setEnabled(running)

        # Khóa chỉnh sửa bảng phụ đề và toàn bộ điều khiển ảnh hưởng đến tác vụ trong lúc worker đang chạy

        self.sub_table.setEditTriggers(

            QAbstractItemView.NoEditTriggers

            if running

            else (QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)

        )

        all_task_controls = [

            self.combo_mode,

            self.combo_lang,

            self.combo_vn_catalog,

            self.edit_vn_sample_text,

            self.btn_free_vram,

            self.btn_pick_ref_audio,

            self.btn_extract_ref_video,

            self.btn_load_prompt_pt,

            self.chk_auto_extract_video,

            self.edit_ref_text,

            self.combo_gender,

            self.combo_age,

            self.combo_pitch,

            self.combo_style,

            self.combo_accent,

            self.combo_dialect,

            self.edit_instruct_preview,

            self.studio_text_edit,

            self.edit_pronun_token,

            self.btn_insert_pronun,

            self.btn_load_srt,

            self.btn_load_video,

            self.combo_multi_speaker_mode,

            self.spin_max_speakers,

            self.combo_target_speaker,

            self.combo_speaker_voice_preset,

            self.btn_apply_preset_to_speaker,

            self.btn_pick_audio_for_speaker,

            self.btn_assign_speaker_to_rows,

            self.btn_reset_speaker_voice,

            self.combo_sub_tag,

            self.btn_insert_tag_to_sub,

            self.spin_speed,

            self.spin_duration,

            self.spin_num_step,

            self.spin_guidance,

            self.spin_t_shift,

            self.spin_layer_penalty,

            self.spin_pos_temp,

            self.spin_cls_temp,

            self.spin_chunk_dur,

            self.spin_chunk_thresh,

            self.spin_pad_dur,

            self.spin_fade_dur,

            self.spin_duck_vol,

            self.combo_ducking_mode,

            self.chk_sync_sub_duration,

            self.chk_denoise,

            self.chk_preprocess_prompt,

            self.chk_postprocess_output,

            self.chk_normalize_text,

            self.chk_flashinfer,

            *getattr(self, "_studio_tag_buttons", []),

        ]

        extra_task_controls = [

            getattr(self, "btn_prev_step", None),

            getattr(self, "btn_next_step", None),

            getattr(self, "btn_reanalyze_speakers", None),

            getattr(self, "btn_reset_all_voices", None),

            getattr(self, "btn_audition_all_speakers", None),

            getattr(self, "combo_hardware_selector", None),

            getattr(self, "btn_run_qc", None),

            *getattr(self, "step_buttons", []),

        ]

        for ctrl in [*all_task_controls, *extra_task_controls]:

            if ctrl is not None:

                ctrl.setEnabled(not running)

        if not running:

            has_subs = len(self.subtitle_blocks) > 0

            self.btn_export_dub_wav.setEnabled(has_subs)

            self.btn_dub_merge_video.setEnabled(has_subs and bool(self.current_video_path))

            self.btn_auto_detect_speakers.setEnabled(has_subs or bool(self.current_video_path))

            self.btn_use_vn_voice_as_ref.setEnabled(

                bool(self.last_generated_audio_path and os.path.exists(self.last_generated_audio_path))

            )

            if hasattr(self, "btn_prev_step"):

                self.btn_prev_step.setEnabled(self.current_step_index > 0)

            if hasattr(self, "btn_next_step"):

                self.btn_next_step.setEnabled(self.current_step_index < self.step_stack.count() - 1)

            self._update_control_states()

            self._update_subtitle_action_buttons()

            self._update_step_navigation_states()

        else:

            self.btn_use_vn_voice_as_ref.setEnabled(False)

            self.btn_clear_video.setEnabled(False)

            self.btn_save_prompt_pt.setEnabled(False)

            self.btn_transcribe_ref.setEnabled(False)

            self.btn_export_dub_wav.setEnabled(False)

            self.btn_dub_merge_video.setEnabled(False)

            self.btn_auto_detect_speakers.setEnabled(False)

            self.btn_preview_selected_row.setEnabled(False)

    def _cancel_running_task(self):

        """Gửi tín hiệu yêu cầu dừng tác vụ đang chạy trên GPU."""

        if self._worker and self._worker.isRunning():

            self._worker.request_cancel()

            self.lbl_status.setText("Đang dừng tác vụ theo yêu cầu...")

            InfoBar.warning("Đang hủy tác vụ", "Hệ thống sẽ dừng sau khi hoàn tất câu hiện tại.", parent=self, duration=2500)

    def _launch_worker(

        self,

        task_type: str,

        text: str = "",

        output_path: str = "",

        video_path: str = "",

        subtitle_blocks: Optional[List[SubtitleBlock]] = None,

        original_volume: float = 0.15,

        fixed_duration: float = 0.0,

        override_ref_audio: str = "",

        override_instruct: str = "",

        override_mode: Optional[str] = None,

        override_ref_text: str = "",

    ):

        if self._worker and self._worker.isRunning():

            self._reset_audition_button()

            InfoBar.warning("Đang bận", "Hệ thống đang xử lý tác vụ trước đó, vui lòng đợi...", parent=self, duration=2500)

            return

        params = copy.deepcopy(self._collect_engine_params())

        ms_mode = params.get("multi_speaker_mode", "single")

        # Kiểm tra giọng mẫu khi ở chế độ nhân bản 1 giọng đơn lẻ

        if (

            task_type not in ("extract_ref_from_video", "detect_speakers")

            and params["mode"] == "clone"

            and ms_mode == "single"

            and not override_ref_audio

            and not override_instruct

            and not params["ref_audio_path"]

            and not (params["auto_extract_ref"] and (video_path or self.current_video_path))

        ):

            InfoBar.warning(

                "Thiếu tệp giọng mẫu",

                "Chế độ Nhân bản giọng nói đơn lẻ cần chọn tệp âm thanh mẫu, tách giọng từ video hoặc bật chế độ Phân vai đa nhân vật.",

                parent=self,

                duration=3500,

            )

            self._reset_audition_button()

            return

        self.progress_bar.setValue(10)

        self.lbl_status.setText("Đang thực thi trên GPU NVIDIA...")

        self._set_task_running_state(True)

        if hasattr(self, "chk_single_unified_voice") and self.chk_single_unified_voice.isChecked():
            self._apply_unified_voice_to_all_subtitles()

        isolated_blocks = copy.deepcopy(subtitle_blocks) if subtitle_blocks is not None else None

        prefer_cuda = True

        if hasattr(self, "combo_hardware_selector"):

            prefer_cuda = (self.combo_hardware_selector.currentIndex() == 0)

        self._worker = OmniVoiceTaskWorker(

            task_type=task_type,

            engine_params=params,

            text=text,

            output_path=output_path,

            video_path=video_path or self.current_video_path,

            subtitle_blocks=isolated_blocks,

            original_volume=original_volume,

            fixed_duration=fixed_duration,

            override_ref_audio=override_ref_audio,

            override_instruct=override_instruct,

            prefer_cuda=prefer_cuda,

            override_mode=override_mode,

            override_ref_text=override_ref_text,

            parent=self,

        )

        self._worker.progressSignal.connect(self._on_worker_progress)

        self._worker.logSignal.connect(self._append_log)

        self._worker.speakersDetectedSignal.connect(self._on_speakers_detected)

        self._worker.finishedSignal.connect(self._on_worker_finished)

        self._worker.finished.connect(self._on_worker_thread_terminated)

        self._worker.start()

    def _on_worker_progress(self, pct: int, msg: str):

        self.progress_bar.setValue(pct)

        self.lbl_status.setText(msg)

    def _on_worker_thread_terminated(self):

        """Bảo đảm mở khóa giao diện ngay cả khi luồng kết thúc bất thường."""

        self._set_task_running_state(False)

    def _on_worker_finished(self, success: bool, out_path: str, detail: str, log_msg: str):

        self._set_task_running_state(False)

        msg_str = log_msg if isinstance(log_msg, str) else str(log_msg or "")

        is_partial = msg_str.startswith("[PARTIAL] ")

        clean_msg = msg_str[len("[PARTIAL] ") :].strip() if is_partial else (msg_str or detail or "")

        self.lbl_status.setText(clean_msg)

        if not success:

            self.progress_bar.setValue(0)

            target_spk = getattr(self, "_current_audition_speaker", "")

            err_text = detail or log_msg or "Lỗi không xác định"

            if target_spk:

                prof = self.speaker_profiles.get(target_spk)

                if not prof:

                    for p in self.speaker_profiles.values():

                        if getattr(p, "speaker_id", None) == target_spk or getattr(p, "display_name", None) == target_spk:

                            prof = p

                            break

                if prof is not None:

                    if isinstance(prof, dict):

                        prof["error_message"] = err_text

                    else:

                        setattr(prof, "error_message", err_text)

            if getattr(self, "_audition_all_active", False):

                self._reset_audition_button()

                detail_log = detail if detail else log_msg

                self.log(f"Nhân vật '{target_spk}' gặp sự cố: {detail_log}. Tự động chuyển sang nhân vật kế tiếp...")

                self._audition_next_in_queue()

                return

            if out_path == "character_audition" or getattr(self, "_active_audition_button", None):

                self._reset_audition_button()

            InfoBar.error("Không thành công", clean_msg or detail or "Lỗi thực thi", parent=self, duration=4500)

            return

        self.progress_bar.setValue(100)

        if is_partial:

            InfoBar.warning("Hoàn thành một phần", clean_msg, parent=self, duration=5500)

        else:

            InfoBar.success("Hoàn thành", clean_msg, parent=self, duration=3500)

        task_type = out_path

        result_path = detail

        if task_type == "extract_ref_from_video" and result_path:

            self.current_ref_audio_path = result_path

            self.lbl_ref_status.setText(f"Tệp giọng mẫu vừa tách từ video: {result_path}")

            self._save_ui_to_config()

            self._update_control_states()

            self._play_audio_file(result_path)

        elif task_type == "transcribe_ref":

            self.edit_ref_text.setText(result_path)

            self._save_ui_to_config()

        elif task_type in ("single_synth", "block_preview", "subtitle_audio_only", "vn_catalog_preview") and os.path.exists(result_path):

            self.last_generated_audio_path = result_path

            self.btn_studio_save_wav.setEnabled(True)

            if task_type == "vn_catalog_preview":

                self.btn_use_vn_voice_as_ref.setEnabled(True)

            self._play_audio_file(result_path)

        elif task_type == "character_audition":

            if success and result_path and os.path.exists(result_path):

                self.last_generated_audio_path = result_path

                if self._is_valid_widget(getattr(self, "_active_audition_button", None)):
                    try:
                        self._active_audition_button.setEnabled(True)
                        self._active_audition_button.setText("Dừng Phát")
                        self._active_audition_button.setIcon(FluentIcon.PAUSE)
                    except RuntimeError:
                        self._active_audition_button = None
                else:
                    self._active_audition_button = None

                self._play_audio_file(result_path)

            else:

                self._reset_audition_button()

    def closeEvent(self, event):

        """Dừng phát âm thanh và dọn dẹp luồng nền an toàn khi đóng cửa sổ."""

        try:

            self._media_player.stop()

            if self._worker and self._worker.isRunning():

                self.lbl_status.setText("Đang giải phóng tài nguyên GPU trước khi đóng...")

                self._worker.request_cancel()

                for sig in (

                    self._worker.progressSignal,

                    self._worker.logSignal,

                    self._worker.speakersDetectedSignal,

                    self._worker.finishedSignal,

                ):

                    try:

                        sig.disconnect()

                    except Exception:

                        pass

                self._worker.wait(5000)

        except Exception:

            pass

        super().closeEvent(event)

    def _play_audio_file(self, filepath: str):

        if not os.path.exists(filepath):

            return

        self._media_player.stop()

        self._media_player.setSource(QUrl.fromLocalFile(filepath))

        self.btn_play_pause_output.setEnabled(True)

        self._media_player.play()

    def _toggle_play_output(self):

        if self._media_player.playbackState() == QMediaPlayer.PlayingState:

            self._media_player.pause()

        else:

            self._media_player.play()

    def _on_player_state_changed(self, state):

        if state == QMediaPlayer.PlayingState:

            self.btn_play_pause_output.setText("Tạm Dừng")

            self.btn_play_pause_output.setIcon(FluentIcon.PAUSE)

            if self._is_valid_widget(getattr(self, "_active_audition_button", None)):
                try:
                    self._active_audition_button.setEnabled(True)
                    self._active_audition_button.setText("Dừng Phát")
                    self._active_audition_button.setIcon(FluentIcon.PAUSE)
                except RuntimeError:
                    self._active_audition_button = None
            else:
                self._active_audition_button = None

        elif state == QMediaPlayer.PausedState:

            self.btn_play_pause_output.setText("Tiếp Tục")

            self.btn_play_pause_output.setIcon(FluentIcon.PLAY)

            if self._is_valid_widget(getattr(self, "_active_audition_button", None)):
                try:
                    self._active_audition_button.setEnabled(True)
                    self._active_audition_button.setText("Tiếp Tục")
                    self._active_audition_button.setIcon(FluentIcon.PLAY)
                except RuntimeError:
                    self._active_audition_button = None
            else:
                self._active_audition_button = None

        else:

            self.btn_play_pause_output.setText("Phát Nghe Thử")

            self.btn_play_pause_output.setIcon(FluentIcon.PLAY)

            self._reset_audition_button()

    def _on_player_position_changed(self, pos_ms: int):

        dur_ms = self._media_player.duration()

        if dur_ms > 0:

            self.slider_audio_pos.setValue(int(pos_ms * 1000 / dur_ms))

        self.lbl_audio_time.setText(f"{self._fmt_ms(pos_ms)} / {self._fmt_ms(dur_ms)}")

    def _on_player_duration_changed(self, dur_ms: int):

        self.lbl_audio_time.setText(f"00:00 / {self._fmt_ms(dur_ms)}")

    def _on_slider_moved(self, val: int):

        dur_ms = self._media_player.duration()

        if dur_ms > 0:

            self._media_player.setPosition(int(val * dur_ms / 1000))

    @staticmethod

    def _fmt_ms(ms: int) -> str:

        s = max(0, int(ms // 1000))

        return f"{s // 60:02d}:{s % 60:02d}"

    def _free_gpu_vram(self):

        import gc

        import torch

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        self._append_log("Đã giải phóng bộ nhớ đệm VRAM trên card đồ họa.")

        InfoBar.success("Đã giải phóng VRAM", "Bộ nhớ đệm trên card đồ họa đã được làm sạch.", parent=self, duration=2500)

    def retranslateUi(self):

        self.btn_cancel_task.setText("Hủy Tác Vụ")

        self.btn_free_vram.setText("Giải Phóng Bộ Nhớ Card Đồ Họa")

        if hasattr(self, "btn_prev_step"):

            self.btn_prev_step.setText("Bước Trước")

        if hasattr(self, "btn_next_step"):

            self.btn_next_step.setText("Bước Tiếp Theo")

        self.btn_preview_vn_voice.setText("Nghe Thử Tiếng Việt")

        if hasattr(self, "chk_single_unified_voice"):
            self.chk_single_unified_voice.setText("Sử dụng một giọng đọc đồng nhất cho toàn bộ tệp phụ đề")

        if hasattr(self, "btn_preview_single_unified"):
            self.btn_preview_single_unified.setText("Nghe Thử Giọng Này")

        self.btn_use_vn_voice_as_ref.setText("Chốt Làm Giọng Mẫu Chính")

        self.btn_studio_synth.setText("Tổng Hợp Giọng Nói Ngay")

        self.btn_studio_save_wav.setText("Lưu Tệp Âm Thanh")

        self.btn_load_srt.setText("Chọn Tệp Phụ Đề")

        self.btn_load_video.setText("Chọn Video Nguồn Để Ghép")

        self.btn_clear_video.setText("Bỏ Chọn Video")

        self.btn_auto_detect_speakers.setText("Tự Động Nhận Diện Nhiều Người Nói")

        self.btn_preview_selected_row.setText("Nghe Thử Câu Đang Chọn")

        self.btn_export_dub_wav.setText("Xuất Tệp Âm Thanh Lồng Tiếng")

        self.btn_dub_merge_video.setText("Lồng Tiếng Và Xuất Video")

        if hasattr(self, "btn_reanalyze_speakers"):

            self.btn_reanalyze_speakers.setText("Tự Động Phân Tích Lại")

        if hasattr(self, "btn_reset_all_voices"):

            self.btn_reset_all_voices.setText("Đặt Lại Toàn Bộ Giọng")

        if hasattr(self, "btn_audition_all_speakers"):

            self.btn_audition_all_speakers.setText("Nghe Thử Toàn Bộ Nhân Vật")

        if hasattr(self, "btn_run_qc"):

            self.btn_run_qc.setText("Kiểm Tra Chất Lượng Trước Xuất")

        self._update_sub_video_status_label()

