# -*- coding: utf-8 -*-
"""
Mô-đun tổng hợp giọng nói (TTS) tích hợp trực tiếp mã nguồn OmniVoice chạy trên GPU NVIDIA.
Hỗ trợ đầy đủ tất cả tính năng của mã nguồn gốc OmniVoice:
1. Nhân bản giọng nói (Voice Cloning): Từ tệp âm thanh mẫu, tự động trích xuất từ video nguồn, hoặc tệp mẫu .pt đã lưu.
2. Thiết kế giọng nói (Voice Design): Điều khiển giới tính, độ tuổi, cao độ, phong cách thì thầm, khẩu âm và phương ngữ qua thuộc tính.
3. Giọng nói tự động (Auto Voice): Tự động sinh giọng đọc tự nhiên không cần mẫu.
4. Hỗ trợ hơn 600 ngôn ngữ, chuẩn hóa chữ số, biểu cảm phi ngôn từ ([laughter], [sigh]...), khớp thời lượng phụ đề,
   cấu hình giải mã khuếch tán (num_step, guidance_scale, denoise...) và tăng tốc FlashInfer / CUDA Graph.
"""

from __future__ import annotations

import os
import sys
import gc
import re
import uuid
import shutil
import logging
import subprocess
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict, Union, Tuple

import numpy as np
import soundfile as sf
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Đảm bảo nạp trực tiếp gói mã nguồn omnivoice tích hợp trong src/ai_engines/omnivoice
_AI_ENGINES_DIR = str(Path(__file__).resolve().parent)
if _AI_ENGINES_DIR not in sys.path:
    sys.path.insert(0, _AI_ENGINES_DIR)

from omnivoice import (
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES, lang_display_name
from omnivoice.utils.voice_design import _INSTRUCT_ALL_VALID, _INSTRUCT_VALID_EN, _INSTRUCT_VALID_ZH

from src.ai_engines.translator import SubtitleBlock
from src.ai_engines.voice_separator import VoiceSeparator
from src.core.tools.ffmpeg_cli import FFmpegCLI
from src.core.config import BASE_DIR, config as app_config

logger = logging.getLogger(__name__)


@dataclass
class DubbingReport:
    """Báo cáo nghiệm thu chi tiết kết quả tổng hợp phụ đề và lồng tiếng OmniVoice."""

    status: str = "failed"  # "success" | "partial" | "failed" | "cancelled"
    output_path: str = ""
    total_blocks: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    skipped_empty_count: int = 0
    truncated_count: int = 0
    stretched_count: int = 0
    resumed_from_cache_count: int = 0
    failed_lines: List[int] = field(default_factory=list)
    truncated_lines: List[int] = field(default_factory=list)
    timestamp_issues: List[Dict[str, Any]] = field(default_factory=list)
    audio_duration_sec: float = 0.0
    rms_energy: float = 0.0
    peak_amplitude: float = 0.0
    valid_audio_file: bool = False
    oom_detected: bool = False
    message: str = ""


def get_readable_path(path_str: str) -> str:
    """Chuẩn hóa đường dẫn tệp tin đọc được trên mọi hệ điều hành."""
    return os.path.abspath(str(path_str)) if path_str else ""

# Danh sách ký hiệu biểu cảm phi ngôn từ được OmniVoice hỗ trợ
NON_VERBAL_TAGS = [
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
]


def parse_timestamp_to_seconds(time_val: Any) -> float:
    """
    Chuyển đổi mốc thời gian phụ đề sang giây.
    Hỗ trợ cả dạng số (float/int) và chuỗi định dạng (HH:MM:SS,mmm hoặc HH:MM:SS.mmm).
    """
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if isinstance(time_val, str):
        cleaned = time_val.strip().replace(",", ".")
        try:
            neg = cleaned.startswith("-")
            if neg:
                cleaned = cleaned[1:].strip()
            parts = cleaned.split(":")
            if len(parts) == 3:
                val = float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            elif len(parts) == 2:
                val = float(parts[0]) * 60.0 + float(parts[1])
            else:
                val = float(parts[0])
            return -val if neg else val
        except Exception:
            return 0.0
    return 0.0


def validate_and_normalize_subtitles(
    blocks: List[SubtitleBlock],
    video_duration: float = 0.0,
    auto_fix: bool = True,
) -> List[Dict[str, Any]]:
    """
    Giai đoạn 1: Kiểm tra và chuẩn hóa danh sách phụ đề trước khi tổng hợp giọng nói.
    Phát hiện mốc thời gian âm, kết thúc <= bắt đầu, sai thứ tự thời gian, câu trống và vượt quá độ dài video.
    """
    issues: List[Dict[str, Any]] = []
    prev_start = -1e9

    for idx, blk in enumerate(blocks, start=1):
        start_s = parse_timestamp_to_seconds(getattr(blk, "start_time", 0.0))
        end_s = parse_timestamp_to_seconds(getattr(blk, "end_time", start_s))
        raw_text = (getattr(blk, "translated_text", "") or getattr(blk, "text", "") or "").strip()

        if not raw_text:
            issues.append({
                "line": idx,
                "type": "empty_text",
                "message": f"Dòng phụ đề {idx} không có nội dung văn bản.",
            })

        if start_s < 0.0 or end_s < 0.0:
            issues.append({
                "line": idx,
                "type": "negative_timestamp",
                "message": f"Dòng phụ đề {idx} có mốc thời gian âm ({start_s:.2f}s -> {end_s:.2f}s).",
            })
            if auto_fix:
                start_s = max(0.0, start_s)
                end_s = max(start_s + 1.0, end_s)
                blk.start_time = start_s
                blk.end_time = end_s

        if end_s <= start_s:
            issues.append({
                "line": idx,
                "type": "inverted_duration",
                "message": f"Dòng phụ đề {idx} có thời gian kết thúc ({end_s:.2f}s) nhỏ hơn hoặc bằng thời gian bắt đầu ({start_s:.2f}s).",
            })
            if auto_fix:
                end_s = start_s + 1.5
                blk.end_time = end_s

        if start_s < prev_start - 0.05:
            issues.append({
                "line": idx,
                "type": "out_of_order",
                "message": f"Dòng phụ đề {idx} bị sai thứ tự thời gian (bắt đầu ở {start_s:.2f}s, trước câu liền kề ở {prev_start:.2f}s).",
            })
        prev_start = max(prev_start, start_s)

        if video_duration > 0.0 and (start_s > video_duration + 0.5 or end_s > video_duration + 2.0):
            issues.append({
                "line": idx,
                "type": "exceeds_video_duration",
                "message": f"Dòng phụ đề {idx} ({start_s:.2f}s -> {end_s:.2f}s) vượt ngoài thời lượng video ({video_duration:.2f}s).",
            })

    return issues


def preprocess_vietnamese_tts_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt trước khi đưa vào OmniVoice:
    - Mở rộng các đơn vị đo lường, ký hiệu phần trăm, nhiệt độ, tiền tệ phổ biến theo ngữ cảnh.
    - Giữ nguyên các thẻ biểu cảm phi ngôn từ ([laughter], [sigh]...).
    """
    if not text or not text.strip():
        return ""
    s = text.strip()
    # Mở rộng đơn vị đo phổ biến khi đi kèm số hoặc độc lập
    replacements = [
        (r"(?<=\d)\s*%", " phần trăm"),
        (r"%", " phần trăm "),
        (r"\bkm/h\b", "ki lô mét trên giờ"),
        (r"\bm/s\b", "mét trên giây"),
        (r"\bkm2\b|\bkm²\b", "ki lô mét vuông"),
        (r"\bm2\b|\bm²\b", "mét vuông"),
        (r"°C\b", " độ C"),
        (r"\bVNĐ\b|\bVND\b", "đồng"),
        (r"\bUSD\b", "đô la Mỹ"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def split_long_vietnamese_clauses(text: str, max_chars: int = 220) -> List[str]:
    """
    Chia câu tiếng Việt quá dài theo dấu câu hoặc cụm ngữ nghĩa để tránh hụt hơi khi tổng hợp,
    nhưng vẫn giữ nguyên thứ tự để ghép lại trong cùng mốc thời gian phụ đề.
    """
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    raw_parts = re.split(r"(?<=[.!?;:…])\s+|(?<=,)\s+", cleaned)
    clauses: List[str] = []
    current = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        else:
            clauses.append(current)
            current = part
    if current:
        clauses.append(current)
    return clauses if clauses else [cleaned]


def validate_audio_file(
    filepath: str,
    min_duration_sec: float = 0.05,
    min_rms: float = 1e-4,
) -> Tuple[bool, float, float, float]:
    """
    Giai đoạn 4: Kiểm định tệp âm thanh WAV đầu ra (tồn tại, đọc được, độ dài, năng lượng RMS, biên độ đỉnh).
    Trả về: (is_valid, duration_sec, rms_energy, peak_amplitude).
    """
    if not filepath or not os.path.exists(filepath):
        return False, 0.0, 0.0, 0.0
    try:
        if os.path.getsize(filepath) <= 44:
            return False, 0.0, 0.0, 0.0
        data, sr = sf.read(filepath, dtype="float32")
        if data.size == 0 or sr <= 0:
            return False, 0.0, 0.0, 0.0
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if np.isnan(data).any() or np.isinf(data).any():
            return False, 0.0, 0.0, 0.0
        duration = float(len(data)) / float(sr)
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        is_valid = (duration >= min_duration_sec) and (rms >= min_rms)
        return is_valid, duration, rms, peak
    except Exception:
        return False, 0.0, 0.0, 0.0


def is_audio_quality_acceptable_for_voice_lock(
    audio_arr: np.ndarray,
    sr: int = 24000,
    min_duration_sec: float = 1.4,
) -> Tuple[bool, str]:
    """
    Kiểm tra chất lượng âm thanh câu đầu tiên trước khi khóa làm VoiceClonePrompt cho nhân vật:
    - Thời lượng >= 1.4 giây
    - Không bị im lặng (RMS >= 0.012)
    - Không bị vỡ tiếng / clipping nặng (peak <= 0.995)
    - Không chứa NaN/Inf
    """
    if audio_arr is None or len(audio_arr) < int(min_duration_sec * sr):
        return False, "Thời lượng câu quá ngắn để khóa làm mẫu giọng chuẩn."
    if np.isnan(audio_arr).any() or np.isinf(audio_arr).any():
        return False, "Dữ liệu âm thanh chứa giá trị không hợp lệ."
    rms = float(np.sqrt(np.mean(audio_arr ** 2)))
    peak = float(np.max(np.abs(audio_arr)))
    if rms < 0.012:
        return False, f"Năng lượng âm thanh quá thấp (RMS={rms:.4f})."
    if peak > 0.995:
        return False, f"Âm thanh có dấu hiệu vượt ngưỡng đỉnh (Peak={peak:.3f})."
    return True, "Đạt chuẩn chất lượng giọng mẫu."


def apply_dynamic_ducking(
    bg_audio: np.ndarray,
    tts_audio: np.ndarray,
    sr: int = 24000,
    duck_gain: float = 0.15,
    speech_threshold: float = 0.018,
    attack_ms: float = 35.0,
    release_ms: float = 280.0,
) -> np.ndarray:
    """
    Giai đoạn 5: Hòa trộn âm thanh với cơ chế Ducking động theo đường bao lời thoại (Envelope Ducking)
    kết hợp giới hạn đỉnh âm thanh (Soft Limiter) chống vỡ tiếng (clipping).
    - Khi không có tiếng thoại TTS: âm thanh nền giữ nguyên mức 1.0.
    - Khi xuất hiện tiếng thoại TTS: âm thanh nền tự động hạ mượt xuống mức duck_gain.
    """
    bg = np.asarray(bg_audio, dtype=np.float32)
    tts = np.asarray(tts_audio, dtype=np.float32)

    if bg.ndim > 1:
        bg = np.mean(bg, axis=1)
    if tts.ndim > 1:
        tts = np.mean(tts, axis=1)

    max_len = max(len(bg), len(tts))
    if max_len == 0:
        return np.zeros(0, dtype=np.float32)

    if len(bg) < max_len:
        bg = np.pad(bg, (0, max_len - len(bg)))
    if len(tts) < max_len:
        tts = np.pad(tts, (0, max_len - len(tts)))

    # Tính đường bao năng lượng theo từng khung 20ms để tốc độ xử lý cực nhanh bằng NumPy
    hop = max(1, int(0.02 * sr))
    num_frames = (max_len + hop - 1) // hop
    pad_len = num_frames * hop - max_len
    tts_padded = np.pad(tts, (0, pad_len)) if pad_len > 0 else tts
    frames = tts_padded.reshape(num_frames, hop)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

    target_gains = np.where(frame_rms >= speech_threshold, float(duck_gain), 1.0).astype(np.float32)

    # Làm mượt đường bao gain với hệ số attack nhanh và release mượt
    attack_coeff = float(np.exp(-hop / max(1.0, (attack_ms / 1000.0) * sr)))
    release_coeff = float(np.exp(-hop / max(1.0, (release_ms / 1000.0) * sr)))
    smoothed_gains = np.empty_like(target_gains)
    cur_g = 1.0
    for idx in range(num_frames):
        tg = float(target_gains[idx])
        if tg < cur_g:
            cur_g = attack_coeff * cur_g + (1.0 - attack_coeff) * tg
        else:
            cur_g = release_coeff * cur_g + (1.0 - release_coeff) * tg
        smoothed_gains[idx] = cur_g

    sample_gains = np.repeat(smoothed_gains, hop)[:max_len]
    mixed = bg * sample_gains + tts

    # Giới hạn đỉnh mềm (Soft knee limiter) bảo đảm không bao giờ vượt 0.98
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed = 0.96 * np.tanh(mixed / 0.96)
    return np.clip(mixed, -0.98, 0.98).astype(np.float32)


def build_voice_design_instruct(
    gender: str = "auto",
    age: str = "auto",
    pitch: str = "auto",
    style: str = "normal",
    custom_instruct: str = "",
) -> Optional[str]:
    """
    Tạo chuỗi thuộc tính thiết kế giọng nói chuẩn xác cho OmniVoice từ các tùy chọn giao diện.
    """
    parts: List[str] = []
    if gender and gender not in ("auto", "Auto", ""):
        parts.append(gender.strip())
    if age and age not in ("auto", "Auto", ""):
        parts.append(age.strip())
    if pitch and pitch not in ("auto", "Auto", ""):
        parts.append(pitch.strip())
    if style and style not in ("normal", "auto", "Auto", ""):
        parts.append(style.strip())
    if custom_instruct and custom_instruct.strip():
        for token in custom_instruct.split(","):
            t = token.strip()
            if t and t not in parts:
                parts.append(t)
    return ", ".join(parts) if parts else None


# Danh mục 22 giọng đọc mẫu tiếng Việt đa phong cách kèm câu thoại nghe thử
VIETNAMESE_VOICE_CATALOG: Dict[str, Dict[str, str]] = {
    "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức": {
        "slug": "nam_thanh_nien_chuan",
        "instruct": "male, young adult, moderate pitch",
        "sample_vi": "Xin chào các bạn, đây là giọng nam thanh niên chuẩn dùng cho thuyết minh và bản tin thời sự.",
    },
    "Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện": {
        "slug": "nu_thanh_nien_chuan",
        "instruct": "female, young adult, moderate pitch",
        "sample_vi": "Xin chào mọi người, hôm nay chúng ta sẽ cùng khám phá một câu chuyện vô cùng hấp dẫn.",
    },
    "Nam Trầm Ấm - Tóm Tắt Phim Và Tài Liệu": {
        "slug": "nam_tram_am_review",
        "instruct": "male, middle-aged, low pitch",
        "sample_vi": "Đằng sau cánh cửa bí ẩn kia là một sự thật mà chưa ai từng dám tiết lộ.",
    },
    "Nam Rất Trầm Uy Lực - Phim Hành Động Và Kiếm Hiệp": {
        "slug": "nam_rat_tram_uy_luc",
        "instruct": "male, middle-aged, very low pitch",
        "sample_vi": "Giữa chốn giang hồ hiểm ác này, chỉ có bản lĩnh mới quyết định được thắng bại.",
    },
    "Nữ Trầm Truyền Cảm - Đọc Truyện Đêm Khuya": {
        "slug": "nu_tram_truyen_cam",
        "instruct": "female, middle-aged, low pitch",
        "sample_vi": "Đêm đã khuya, mời quý vị cùng lắng lòng nghe lại những dòng hồi ức sâu lắng nhất.",
    },
    "Nữ Trẻ Năng Động - Video Ngắn Và Giải Trí": {
        "slug": "nu_tre_nang_dong",
        "instruct": "female, teenager, high pitch",
        "sample_vi": "Mọi người ơi, hôm nay mình vừa phát hiện ra một bí quyết cực kỳ thú vị luôn nhé!",
    },
    "Nam Trẻ Sôi Nổi - Bình Luận Và Sáng Tạo Nội Dung": {
        "slug": "nam_tre_soi_noi",
        "instruct": "male, teenager, high pitch",
        "sample_vi": "Chào mừng anh em đã quay trở lại với video đặc biệt của chúng ta ngày hôm nay!",
    },
    "Nữ Ngọt Ngào Trong Trẻo - Tâm Sự Và Tình Cảm": {
        "slug": "nu_ngot_ngao_trong_treo",
        "instruct": "female, young adult, high pitch",
        "sample_vi": "Có những ký ức dù năm tháng trôi qua bao lâu vẫn luôn vẹn nguyên trong trái tim.",
    },
    "Nam Cao Sáng Rõ Ràng - Dẫn Chương Trình Sự Kiện": {
        "slug": "nam_cao_sang_mc",
        "instruct": "male, young adult, high pitch",
        "sample_vi": "Xin nồng nhiệt chào đón toàn thể quý vị khán giả đã có mặt trong chương trình tối nay.",
    },
    "Nam Trung Niên Điềm Đạm - Sách Nói Và Triết Lý": {
        "slug": "nam_trung_nien_diem_dam",
        "instruct": "male, middle-aged, moderate pitch",
        "sample_vi": "Thành công không phải là đích đến cuối cùng, mà là hành trình vượt qua từng thử thách.",
    },
    "Nữ Trung Niên Chững Chạc - Giảng Giải Và Giáo Dục": {
        "slug": "nu_trung_nien_chung_chac",
        "instruct": "female, middle-aged, moderate pitch",
        "sample_vi": "Trong nội dung bài học hôm nay, chúng ta sẽ cùng phân tích những nguyên lý cốt lõi nhất.",
    },
    "Ông Lão Trầm Mặc - Cổ Tích Và Lịch Sử": {
        "slug": "ong_lao_tram_mac",
        "instruct": "male, elderly, very low pitch",
        "sample_vi": "Ngày xửa ngày xưa, ở vùng đất phương nam có một truyền thuyết được lưu truyền qua nhiều thế hệ.",
    },
    "Bà Lão Hiền Từ - Kể Chuyện Dân Gian": {
        "slug": "ba_lao_hien_tu",
        "instruct": "female, elderly, low pitch",
        "sample_vi": "Các cháu lại đây ngồi bên bếp lửa ấm, bà sẽ kể cho nghe chuyện cổ tích ngày xưa.",
    },
    "Thiếu Nhi Hồn Nhiên - Phim Hoạt Hình": {
        "slug": "thieu_nhi_hon_nhien",
        "instruct": "child, high pitch",
        "sample_vi": "Mẹ ơi, hôm nay ở lớp con được cô giáo khen vì giúp đỡ bạn bè đấy ạ!",
    },
    "Em Bé Nhí Nhảnh - Giọng Cao Đáng Yêu": {
        "slug": "em_be_nhi_nhanh",
        "instruct": "child, very high pitch",
        "sample_vi": "[surprise-wa] Chú cún con này biết bắt tay dễ thương quá đi mất thôi!",
    },
    "Nam Thì Thầm Bí Ẩn - Phim Kinh Dị Và Trinh Thám": {
        "slug": "nam_thi_tham_bi_an",
        "instruct": "male, young adult, low pitch, whisper",
        "sample_vi": "Suỵt, đừng phát ra tiếng động nào cả, hình như có tiếng bước chân ngoài hành lang.",
    },
    "Nữ Thì Thầm Nhẹ Nhàng - Thư Giãn Và Ru Ngủ": {
        "slug": "nu_thi_tham_nhe_nhang",
        "instruct": "female, young adult, moderate pitch, whisper",
        "sample_vi": "Hãy nhắm mắt lại, thả lỏng tâm trí và tận hưởng bầu không khí tĩnh lặng đêm nay.",
    },
    "Nam Vui Vẻ Tự Nhiên - Kèm Tiếng Cười Biểu Cảm": {
        "slug": "nam_vui_ve_bieu_cam",
        "instruct": "male, young adult, moderate pitch",
        "sample_vi": "[laughter] Thật sự không thể ngờ tình huống bất ngờ đó lại diễn ra ngay trước mắt chúng tôi!",
    },
    "Nữ Xúc Động Sâu Lắng - Kèm Tiếng Thở Dài Biểu Cảm": {
        "slug": "nu_xuc_dong_bieu_cam",
        "instruct": "female, young adult, moderate pitch",
        "sample_vi": "[sigh] Giá như ngày hôm ấy chúng ta kịp nói với nhau một lời chân thành.",
    },
    "Nam Phong Cách Âu Mỹ Đọc Tiếng Việt": {
        "slug": "nam_phong_cach_au_my",
        "instruct": "male, young adult, moderate pitch, american accent",
        "sample_vi": "Xin chào Việt Nam, tôi rất ấn tượng với cảnh đẹp và ẩm thực tuyệt vời của đất nước các bạn.",
    },
    "Nữ Phong Cách Anh Quốc Đọc Tiếng Việt": {
        "slug": "nu_phong_cach_anh_quoc",
        "instruct": "female, young adult, moderate pitch, british accent",
        "sample_vi": "Chào buổi sáng, hôm nay thời tiết thật dễ chịu để cùng nhau dạo quanh phố cổ.",
    },
    "Nữ Phong Cách Đông Á Đọc Tiếng Việt": {
        "slug": "nu_phong_cach_dong_a",
        "instruct": "female, teenager, high pitch, japanese accent",
        "sample_vi": "Rất vui được làm quen với tất cả các bạn, mong nhận được sự ủng hộ của mọi người.",
    },
}

# Danh sách cấu hình thiết kế giọng mặc định cho chế độ phân vai đa nhân vật
DEFAULT_SPEAKER_PRESETS: Dict[str, Dict[str, str]] = {
    "Người nói 1": {
        "label": "Người nói 1 - Nam Thanh Niên",
        "instruct": "male, young adult, moderate pitch",
    },
    "Người nói 2": {
        "label": "Người nói 2 - Nữ Thanh Niên",
        "instruct": "female, young adult, moderate pitch",
    },
    "Người nói 3": {
        "label": "Người nói 3 - Nam Trầm Trung Niên",
        "instruct": "male, middle-aged, low pitch",
    },
    "Người nói 4": {
        "label": "Người nói 4 - Nữ Cao Thiếu Niên",
        "instruct": "female, teenager, high pitch",
    },
    "Người nói 5": {
        "label": "Người nói 5 - Nam Cao Tuổi",
        "instruct": "male, elderly, very low pitch",
    },
    "Người nói 6": {
        "label": "Người nói 6 - Trẻ Em",
        "instruct": "child, high pitch",
    },
}



class TTSEngine:
    """
    Công cụ tổng hợp giọng nói sử dụng mã nguồn OmniVoice tích hợp, chạy trực tiếp trên GPU NVIDIA.
    Hỗ trợ đơn giọng, đa nhân vật (tự nhận diện nhiều người nói từ video hoặc văn bản phụ đề) và nhái giọng theo từng câu thoại.
    """

    VIETNAMESE_VOICE_CATALOG = VIETNAMESE_VOICE_CATALOG
    DEFAULT_SPEAKER_PRESETS = DEFAULT_SPEAKER_PRESETS
    NON_VERBAL_TAGS = NON_VERBAL_TAGS

    def __init__(
        self,
        parent_logger=None,
        model_name_or_path: str = "k2-fsa/OmniVoice",
        mode: str = "clone",
        language: Optional[str] = None,
        speed: float = 1.0,
        sync_to_subtitle_duration: bool = True,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        auto_extract_ref: bool = True,
        instruct: Optional[str] = None,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
        audio_chunk_duration: float = 15.0,
        audio_chunk_threshold: float = 30.0,
        pad_duration: float = 0.1,
        fade_duration: float = 0.1,
        denoise: bool = True,
        preprocess_prompt: bool = True,
        postprocess_output: bool = True,
        normalize_text: bool = True,
        enable_flashinfer: bool = False,
        load_asr: bool = False,
        asr_model_name: str = "openai/whisper-large-v3-turbo",
        voice_prompt_cache_dir: Optional[str] = None,
        multi_speaker_mode: str = "single",
        max_speakers: int = 4,
        speaker_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        resume_cache_dir: Optional[str] = None,
        ducking_mode: str = "dynamic",
        speaker_model_path: Optional[str] = None,
        allow_mock: bool = False,
    ):
        self.model: Optional[OmniVoice] = None
        self.parent_logger = parent_logger
        self.model_name_or_path = model_name_or_path
        self.mode = mode if mode in ("clone", "design", "auto") else "clone"
        self.language = language if (language and language not in ("auto", "Auto")) else None
        self.speed = max(0.5, min(2.0, float(speed)))
        self.sync_to_subtitle_duration = bool(sync_to_subtitle_duration)
        self.ref_audio_path = ref_audio_path if (ref_audio_path and str(ref_audio_path).strip()) else None
        self.ref_text = ref_text if (ref_text and str(ref_text).strip()) else None
        self.auto_extract_ref = bool(auto_extract_ref)
        self.instruct = instruct if (instruct and str(instruct).strip()) else None

        # Tham số OmniVoiceGenerationConfig đầy đủ
        self.num_step = max(4, min(64, int(num_step)))
        self.guidance_scale = max(0.0, min(4.0, float(guidance_scale)))
        self.t_shift = float(t_shift)
        self.layer_penalty_factor = float(layer_penalty_factor)
        self.position_temperature = float(position_temperature)
        self.class_temperature = float(class_temperature)
        self.audio_chunk_duration = max(3.0, float(audio_chunk_duration))
        self.audio_chunk_threshold = max(5.0, float(audio_chunk_threshold))
        self.pad_duration = max(0.0, float(pad_duration))
        self.fade_duration = max(0.0, float(fade_duration))
        self.denoise = bool(denoise)
        self.preprocess_prompt = bool(preprocess_prompt)
        self.postprocess_output = bool(postprocess_output)
        self.normalize_text = bool(normalize_text)
        self.enable_flashinfer = bool(enable_flashinfer)
        self.load_asr = bool(load_asr)
        self.asr_model_name = asr_model_name

        # Cấu hình đa giọng đọc / phân vai người nói
        # multi_speaker_mode: "single" | "auto_video_cluster" | "per_line_video_clone" | "auto_srt_text" | "custom_table"
        self.multi_speaker_mode = multi_speaker_mode or "single"
        self.max_speakers = max(1, min(8, int(max_speakers)))
        self.speaker_profiles: Dict[str, Dict[str, Any]] = speaker_profiles or {}
        if speaker_model_path:
            self.speaker_model_path = str(speaker_model_path).strip()
        else:
            cfg_item = getattr(app_config, "mmoSpeakerModelPath", None)
            self.speaker_model_path = str(cfg_item.value).strip() if (cfg_item and hasattr(cfg_item, "value")) else ""
        self._speaker_prompt_cache: Dict[str, VoiceClonePrompt] = {}
        self._speaker_instruct_cache: Dict[str, str] = {}
        self.resume_cache_dir = str(resume_cache_dir).strip() if resume_cache_dir else None
        self.ducking_mode = ducking_mode if ducking_mode in ("dynamic", "fixed") else "dynamic"
        self.allow_mock = bool(allow_mock)
        self._cancel_requested: bool = False
        self.last_report: Optional[DubbingReport] = None

        self.sample_rate = 24000
        self.current_prompt: Optional[VoiceClonePrompt] = None

        if voice_prompt_cache_dir and str(voice_prompt_cache_dir).strip():
            self.voice_prompt_cache_dir = str(voice_prompt_cache_dir).strip()
        else:
            self.voice_prompt_cache_dir = os.path.join(BASE_DIR, "output", "voice_prompts")
        os.makedirs(self.voice_prompt_cache_dir, exist_ok=True)
        if self.resume_cache_dir:
            os.makedirs(self.resume_cache_dir, exist_ok=True)

    def request_cancel(self):
        """Yêu cầu dừng sớm quá trình tổng hợp phụ đề đang chạy."""
        self._cancel_requested = True

    def reset_cancel(self):
        """Đặt lại cờ hủy tác vụ trước khi chạy lượt tổng hợp mới."""
        self._cancel_requested = False

    def reset_speaker_locked_prompt(self, speaker_id: Optional[str] = None):
        """
        Xóa mẫu giọng tự động khóa (consistent_{speaker_id}) của một nhân vật hoặc toàn bộ nhân vật
        để tạo lại mẫu giọng mới sạch hơn ở lần tổng hợp tiếp theo.
        """
        if speaker_id:
            self._speaker_prompt_cache.pop(f"consistent_{speaker_id}", None)
            prof = self.speaker_profiles.get(speaker_id)
            if prof and str(prof.get("ref_audio_path", "")).find("locked_") != -1:
                prof["ref_audio_path"] = ""
            self.log(f"Đã đặt lại mẫu giọng khóa tự động của {speaker_id}.")
        else:
            keys_to_remove = [k for k in self._speaker_prompt_cache if k.startswith("consistent_")]
            for k in keys_to_remove:
                self._speaker_prompt_cache.pop(k, None)
            self.log("Đã đặt lại toàn bộ mẫu giọng khóa tự động của các nhân vật.")

    @classmethod
    def from_app_config(cls, parent_logger=None, language: Optional[str] = None) -> "TTSEngine":
        """
        Khởi tạo TTSEngine trực tiếp từ cấu hình toàn cục của ứng dụng Sakai Studio.
        """
        mode = getattr(app_config.mmoTTSMode, "value", "clone") if hasattr(app_config, "mmoTTSMode") else "clone"
        auto_extract = getattr(app_config.mmoTTSAutoExtractRef, "value", True) if hasattr(app_config, "mmoTTSAutoExtractRef") else True
        ref_audio = getattr(app_config.mmoTTSRefAudioPath, "value", "") if hasattr(app_config, "mmoTTSRefAudioPath") else ""
        ref_text = getattr(app_config.mmoTTSRefText, "value", "") if hasattr(app_config, "mmoTTSRefText") else ""
        prompt_dir = getattr(app_config.mmoTTSVoicePromptDir, "value", "") if hasattr(app_config, "mmoTTSVoicePromptDir") else ""

        gender = getattr(app_config.mmoTTSGender, "value", "auto") if hasattr(app_config, "mmoTTSGender") else "auto"
        age = getattr(app_config.mmoTTSAge, "value", "auto") if hasattr(app_config, "mmoTTSAge") else "auto"
        pitch = getattr(app_config.mmoTTSPitch, "value", "auto") if hasattr(app_config, "mmoTTSPitch") else "auto"
        style = getattr(app_config.mmoTTSStyle, "value", "normal") if hasattr(app_config, "mmoTTSStyle") else "normal"
        custom_instruct = getattr(app_config.mmoTTSInstructCustom, "value", "") if hasattr(app_config, "mmoTTSInstructCustom") else ""
        instruct = build_voice_design_instruct(gender, age, pitch, style, custom_instruct)

        speed = getattr(app_config.mmoTTSSpeed, "value", 1.0) if hasattr(app_config, "mmoTTSSpeed") else 1.0
        sync_dur = getattr(app_config.mmoTTSSyncDuration, "value", True) if hasattr(app_config, "mmoTTSSyncDuration") else True
        num_step = getattr(app_config.mmoTTSNumStep, "value", 32) if hasattr(app_config, "mmoTTSNumStep") else 32
        guidance_scale = getattr(app_config.mmoTTSGuidanceScale, "value", 2.0) if hasattr(app_config, "mmoTTSGuidanceScale") else 2.0
        denoise = getattr(app_config.mmoTTSDenoise, "value", True) if hasattr(app_config, "mmoTTSDenoise") else True
        preprocess_prompt = getattr(app_config.mmoTTSPreprocessPrompt, "value", True) if hasattr(app_config, "mmoTTSPreprocessPrompt") else True
        postprocess_output = getattr(app_config.mmoTTSPostprocessOutput, "value", True) if hasattr(app_config, "mmoTTSPostprocessOutput") else True
        normalize_text = getattr(app_config.mmoTTSNormalizeText, "value", True) if hasattr(app_config, "mmoTTSNormalizeText") else True
        enable_flashinfer = getattr(app_config.mmoTTSFlashInfer, "value", False) if hasattr(app_config, "mmoTTSFlashInfer") else False
        multi_speaker_mode = getattr(app_config.mmoTTSMultiSpeakerMode, "value", "single") if hasattr(app_config, "mmoTTSMultiSpeakerMode") else "single"
        max_speakers = getattr(app_config.mmoTTSMaxSpeakers, "value", 4) if hasattr(app_config, "mmoTTSMaxSpeakers") else 4
        ducking_mode = getattr(app_config.mmoTTSDuckingMode, "value", "dynamic") if hasattr(app_config, "mmoTTSDuckingMode") else "dynamic"
        t_shift = getattr(app_config.mmoTTSTShift, "value", 0.1) if hasattr(app_config, "mmoTTSTShift") else 0.1
        layer_penalty = getattr(app_config.mmoTTSLayerPenalty, "value", 5.0) if hasattr(app_config, "mmoTTSLayerPenalty") else 5.0
        pos_temp = getattr(app_config.mmoTTSPosTemp, "value", 5.0) if hasattr(app_config, "mmoTTSPosTemp") else 5.0
        cls_temp = getattr(app_config.mmoTTSClsTemp, "value", 0.0) if hasattr(app_config, "mmoTTSClsTemp") else 0.0
        chunk_dur = getattr(app_config.mmoTTSChunkDuration, "value", 8.0) if hasattr(app_config, "mmoTTSChunkDuration") else 8.0
        chunk_thresh = getattr(app_config.mmoTTSChunkThreshold, "value", 12.0) if hasattr(app_config, "mmoTTSChunkThreshold") else 12.0
        pad_dur = getattr(app_config.mmoTTSPadDuration, "value", 0.08) if hasattr(app_config, "mmoTTSPadDuration") else 0.08
        fade_dur = getattr(app_config.mmoTTSFadeDuration, "value", 0.03) if hasattr(app_config, "mmoTTSFadeDuration") else 0.03

        target_lang = language or (getattr(app_config.targetLanguage, "value", "vi") if hasattr(app_config, "targetLanguage") else "vi")
        if target_lang == "zh-CN":
            target_lang = "zh"

        return cls(
            parent_logger=parent_logger,
            mode=mode,
            language=target_lang,
            speed=speed,
            sync_to_subtitle_duration=sync_dur,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            auto_extract_ref=auto_extract,
            instruct=instruct,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            layer_penalty_factor=layer_penalty,
            position_temperature=pos_temp,
            class_temperature=cls_temp,
            audio_chunk_duration=chunk_dur,
            audio_chunk_threshold=chunk_thresh,
            pad_duration=pad_dur,
            fade_duration=fade_dur,
            denoise=denoise,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            normalize_text=normalize_text,
            enable_flashinfer=enable_flashinfer,
            voice_prompt_cache_dir=prompt_dir,
            multi_speaker_mode=multi_speaker_mode,
            max_speakers=max_speakers,
            ducking_mode=ducking_mode,
        )

    def log(self, msg: str):
        """Ghi nhật ký thông tin ra hệ thống và giao diện người dùng."""
        if self.parent_logger:
            self.parent_logger(msg)
        else:
            logger.info(f"[OmniVoice] {msg}")
            try:
                if hasattr(sys.stdout, "reconfigure"):
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                print(f"[OmniVoice] {msg}")
            except Exception:
                pass

    @staticmethod
    def get_supported_languages() -> List[str]:
        """Trả về danh sách hơn 600 ngôn ngữ được OmniVoice hỗ trợ."""
        return ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

    def build_generation_config(self) -> OmniVoiceGenerationConfig:
        """Tạo đối tượng OmniVoiceGenerationConfig đầy đủ từ các thiết lập hiện tại."""
        return OmniVoiceGenerationConfig(
            num_step=self.num_step,
            guidance_scale=self.guidance_scale,
            t_shift=self.t_shift,
            layer_penalty_factor=self.layer_penalty_factor,
            position_temperature=self.position_temperature,
            class_temperature=self.class_temperature,
            denoise=self.denoise,
            preprocess_prompt=self.preprocess_prompt,
            postprocess_output=self.postprocess_output,
            audio_chunk_duration=self.audio_chunk_duration,
            audio_chunk_threshold=self.audio_chunk_threshold,
            pad_duration=self.pad_duration,
            fade_duration=self.fade_duration,
        )

    def load_model(self):
        """
        Tải mô hình OmniVoice trực tiếp lên GPU NVIDIA với dtype=torch.float16.
        Tự động phát hiện thiết bị phần cứng phù hợp (ưu tiên CUDA trước CPU).
        """
        if self.model is not None:
            return

        if torch is None or OmniVoice is None:
            raise RuntimeError("Thư viện PyTorch hoặc OmniVoice chưa được cài đặt trên môi trường hiện tại.")

        self.log("Đang kiểm tra thiết bị phần cứng để tải mô hình OmniVoice...")
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.float16
            gpu_name = torch.cuda.get_device_name(0)
            self.log(f"Phát hiện GPU NVIDIA: {gpu_name}. Đang nạp OmniVoice vào VRAM (float16)...")
        else:
            device = "cpu"
            dtype = torch.float32
            self.log("Không phát hiện GPU NVIDIA. Đang nạp OmniVoice trên CPU (float32)...")

        try:
            self.model = OmniVoice.from_pretrained(
                self.model_name_or_path,
                device_map=device,
                dtype=dtype,
                load_asr=self.load_asr,
                asr_model_name=self.asr_model_name,
            )
            if self.model.sampling_rate:
                self.sample_rate = int(self.model.sampling_rate)

            # Kích hoạt tăng tốc FlashInfer + CUDA Graph nếu được bật trên GPU NVIDIA
            if self.enable_flashinfer and torch.cuda.is_available():
                try:
                    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
                    apply_flashinfer(self.model, enable_cuda_graph=True)
                    self.log("Đã kích hoạt tăng tốc phần cứng FlashInfer + CUDA Graph trên GPU NVIDIA.")
                except Exception as fi_err:
                    self.log(f"Không thể kích hoạt FlashInfer (sử dụng nhân CUDA tiêu chuẩn): {fi_err}")

            self.log("Đã nạp mô hình OmniVoice thành công và sẵn sàng tổng hợp giọng nói.")
        except Exception as e:
            self.log(f"Lỗi khi tải mô hình OmniVoice: {str(e)}")
            raise e

    def unload_model(self):
        """Giải phóng hoàn toàn bộ nhớ VRAM của mô hình OmniVoice."""
        if self.model is not None:
            self.log("Đang giải phóng bộ nhớ VRAM của mô hình OmniVoice...")
            del self.model
            self.model = None
            self.current_prompt = None
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log("Đã giải phóng hoàn toàn bộ nhớ VRAM.")

    def _transcribe_ref_audio_fallback(self, ref_audio_path: str) -> str:
        """
        Tự động nhận diện văn bản của tệp âm thanh giọng mẫu nếu người dùng không nhập ref_text.
        Ưu tiên dùng faster-whisper nội bộ của Sakai Studio để tiết kiệm VRAM trước khi dùng ASR của OmniVoice.
        """
        if self.ref_text and self.ref_text.strip():
            return self.ref_text.strip()

        try:
            from src.ai_engines.whisper_fallback import WhisperFallback
            wf = WhisperFallback(model_size="base")
            if wf.is_available():
                self.log("Đang tự động nhận diện nội dung giọng mẫu bằng Whisper...")
                segs = wf.transcribe_audio(ref_audio_path)
                if segs:
                    transcript = " ".join(s.get("text", "").strip() for s in segs if s.get("text")).strip()
                    if transcript:
                        self.log(f"Nội dung giọng mẫu nhận diện được: {transcript}")
                        return transcript
        except Exception as e:
            logger.debug(f"WhisperFallback không khả dụng cho giọng mẫu: {e}")

        # Nếu WhisperFallback không trả về kết quả, để OmniVoice tự xử lý hoặc dùng chuỗi mặc định
        return ""

    def set_reference_voice(self, ref_audio_path: str, ref_text: Optional[str] = None) -> bool:
        """
        Đặt giọng mẫu từ tệp âm thanh (.wav, .mp3, .m4a) hoặc tệp mẫu đã lưu (.pt).
        Tạo VoiceClonePrompt và lưu vào bộ nhớ đệm để tái sử dụng cho toàn bộ video.
        """
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            self.log(f"Cảnh báo: Không tìm thấy tệp giọng mẫu tại: {ref_audio_path}")
            return False

        # Nếu tệp đầu vào là mẫu .pt đã lưu từ trước
        if ref_audio_path.lower().endswith(".pt"):
            return self.load_voice_prompt(ref_audio_path)

        self.ref_audio_path = ref_audio_path
        if ref_text is not None:
            self.ref_text = ref_text

        self.load_model()
        self.log(f"Đang mã hóa đặc trưng giọng mẫu từ tệp: {os.path.basename(ref_audio_path)}...")

        try:
            resolved_ref_text = self.ref_text
            if not resolved_ref_text:
                resolved_ref_text = self._transcribe_ref_audio_fallback(ref_audio_path)

            # Nếu vẫn không có ref_text và chưa nạp ASR lớn của OmniVoice, truyền chuỗi phiên âm ngắn để tránh phải tải thêm 3GB Whisper-large-v3-turbo
            if not resolved_ref_text and self.model._asr_pipe is None and not self.load_asr:
                resolved_ref_text = "."

            self.current_prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=resolved_ref_text if resolved_ref_text else None,
                preprocess_prompt=self.preprocess_prompt,
            )
            self.log("Đã tạo mẫu nhân bản giọng nói trên GPU thành công.")
            return True
        except Exception as e:
            self.log(f"Lỗi khi tạo mẫu nhân bản giọng nói: {str(e)}")
            return False

    def extract_reference_from_video(self, video_path: str) -> Optional[str]:
        """
        Trích xuất giọng mẫu tự động từ video đầu vào bằng VoiceSeparator.
        Tự động chọn đoạn 3-10 giây đầu tiên có giọng nói rõ ràng nhất.
        """
        if not os.path.exists(video_path):
            self.log(f"Tệp video không tồn tại: {video_path}")
            return None

        self.log(f"Đang tách giọng nói gốc từ video để làm mẫu nhân bản: {os.path.basename(video_path)}")
        separator = VoiceSeparator()

        temp_audio_path = os.path.join(
            self.voice_prompt_cache_dir,
            f"temp_vocals_{uuid.uuid4().hex[:8]}.wav"
        )
        success = separator.separate_from_video(video_path, temp_audio_path)
        if not success or not os.path.exists(temp_audio_path):
            self.log("Không thể tách giọng nói từ video nguồn.")
            return None

        try:
            data, sr = sf.read(temp_audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            frame_length = int(sr * 0.05)
            if len(data) > frame_length:
                energy = np.array([
                    np.mean(data[i:i + frame_length] ** 2)
                    for i in range(0, len(data) - frame_length, frame_length)
                ])
                threshold = np.max(energy) * 0.05 if len(energy) > 0 else 0.001

                start_sample = 0
                for idx, e in enumerate(energy):
                    if e > threshold:
                        start_sample = idx * frame_length
                        break
            else:
                start_sample = 0

            # Cắt đoạn 8 giây (nằm trong khoảng khuyến nghị 3-10 giây của OmniVoice)
            duration_samples = int(8.0 * sr)
            end_sample = min(start_sample + duration_samples, len(data))
            clipped_audio = data[start_sample:end_sample]

            # Nếu đoạn cắt quá ngắn (< 1.5s), lấy từ đầu tệp
            if len(clipped_audio) < int(1.5 * sr) and len(data) >= int(1.5 * sr):
                clipped_audio = data[:min(duration_samples, len(data))]

            ref_output_path = os.path.join(
                self.voice_prompt_cache_dir,
                f"ref_voice_{Path(video_path).stem}_{uuid.uuid4().hex[:6]}.wav"
            )
            sf.write(ref_output_path, clipped_audio, sr)

            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass

            self.log(f"Đã trích xuất giọng mẫu (8 giây) thành công: {os.path.basename(ref_output_path)}")
            self.set_reference_voice(ref_output_path)
            return ref_output_path
        except Exception as e:
            self.log(f"Lỗi khi xử lý đoạn giọng mẫu trích xuất: {str(e)}")
            return None

    def save_voice_prompt(self, path: str) -> bool:
        """Lưu giọng mẫu đã mã hóa ra tệp .pt để tái sử dụng tức thì ở các phiên sau."""
        if self.current_prompt is None:
            if self.ref_audio_path and os.path.exists(self.ref_audio_path):
                self.set_reference_voice(self.ref_audio_path, self.ref_text)
            if self.current_prompt is None:
                self.log("Chưa có mẫu giọng nói nào trong bộ nhớ để lưu.")
                return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.current_prompt.save(path)
            self.log(f"Đã lưu mẫu giọng nói (.pt) thành công vào: {path}")
            return True
        except Exception as e:
            self.log(f"Lỗi khi lưu mẫu giọng nói: {str(e)}")
            return False

    def load_voice_prompt(self, path: str) -> bool:
        """Tải lại mẫu giọng nói đã mã hóa từ tệp .pt."""
        if not os.path.exists(path):
            self.log(f"Không tìm thấy tệp mẫu giọng nói: {path}")
            return False

        try:
            self.current_prompt = VoiceClonePrompt.load(path)
            self.log(f"Đã nạp mẫu giọng nói (.pt) thành công từ: {os.path.basename(path)}")
            return True
        except Exception as e:
            self.log(f"Lỗi khi nạp mẫu giọng nói từ tệp: {str(e)}")
            return False

    def _get_or_create_prompt_for_audio(
        self,
        audio_source: Union[str, tuple],
        cache_key: str,
        ref_text: Optional[str] = None,
    ) -> Optional[VoiceClonePrompt]:
        """Lấy hoặc mã hóa mới VoiceClonePrompt có lưu bộ nhớ đệm theo khóa nhân vật."""
        if cache_key and cache_key in self._speaker_prompt_cache:
            return self._speaker_prompt_cache[cache_key]

        self.load_model()
        try:
            if isinstance(audio_source, str) and audio_source.lower().endswith(".pt"):
                prompt = VoiceClonePrompt.load(audio_source)
                if cache_key:
                    self._speaker_prompt_cache[cache_key] = prompt
                return prompt

            resolved_text = ref_text
            if not resolved_text and isinstance(audio_source, str) and os.path.exists(audio_source):
                resolved_text = self._transcribe_ref_audio_fallback(audio_source)
            if not resolved_text and (self.model is None or getattr(self.model, "_asr_pipe", None) is None) and not self.load_asr:
                resolved_text = "."

            if self.model is None:
                return None

            prompt = self.model.create_voice_clone_prompt(
                ref_audio=audio_source,
                ref_text=resolved_text if resolved_text else None,
                preprocess_prompt=self.preprocess_prompt,
            )
            if cache_key:
                self._speaker_prompt_cache[cache_key] = prompt
            return prompt
        except Exception as e:
            self.log(f"Không thể tạo mẫu giọng cho {cache_key}: {e}")
            return None

    @staticmethod
    def _extract_acoustic_features(seg: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Trích xuất vector đặc trưng âm học của người nói từ đoạn sóng âm:
        - Cao độ cơ bản F0 (Hz) và độ biến thiên F0 qua tự tương quan
        - Trọng tâm phổ (Spectral Centroid) và 8 dải năng lượng formant âm sắc
        """
        if len(seg) < int(0.25 * sr):
            return None
        seg = seg.astype(np.float32)
        peak = float(np.max(np.abs(seg)))
        if peak < 1e-4:
            return None
        seg = seg / peak

        frame_len = int(0.04 * sr)
        hop_len = int(0.02 * sr)
        if len(seg) < frame_len:
            return None

        min_lag = max(1, int(sr / 360.0))
        max_lag = min(frame_len - 1, int(sr / 75.0))

        f0_list: List[float] = []
        spectra: List[np.ndarray] = []
        window = np.hanning(frame_len).astype(np.float32)

        for start in range(0, len(seg) - frame_len + 1, hop_len):
            frame = seg[start:start + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < 0.04:
                continue

            windowed = frame * window
            spec = np.abs(np.fft.rfft(windowed))
            spectra.append(spec)

            # Tính tự tương quan chuẩn hóa để tìm F0
            ac = np.correlate(windowed, windowed, mode="full")[frame_len - 1:]
            if ac[0] > 1e-6 and max_lag > min_lag:
                sub_ac = ac[min_lag:max_lag] / ac[0]
                best_idx = int(np.argmax(sub_ac))
                if sub_ac[best_idx] > 0.28:
                    lag = min_lag + best_idx
                    f0_list.append(float(sr) / float(lag))

        if not spectra:
            return None

        mean_spec = np.mean(np.stack(spectra, axis=0), axis=0)
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
        spec_sum = float(np.sum(mean_spec)) + 1e-8
        centroid = float(np.sum(freqs * mean_spec) / spec_sum)

        # 8 dải tần số đặc trưng âm sắc giọng nói (100Hz -> 3800Hz)
        band_edges = [100, 250, 450, 700, 1000, 1500, 2200, 3000, 3800]
        band_energies: List[float] = []
        for b_idx in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[b_idx]) & (freqs < band_edges[b_idx + 1])
            e = float(np.log1p(np.mean(mean_spec[mask]))) if np.any(mask) else 0.0
            band_energies.append(e)
        band_arr = np.array(band_energies, dtype=np.float32)
        band_arr = band_arr / (np.linalg.norm(band_arr) + 1e-6)

        median_f0 = float(np.median(f0_list)) if f0_list else 155.0
        iqr_f0 = float(np.percentile(f0_list, 75) - np.percentile(f0_list, 25)) if len(f0_list) >= 4 else 15.0

        feat = np.concatenate([
            np.array([median_f0 / 200.0, iqr_f0 / 100.0, centroid / 2000.0], dtype=np.float32),
            band_arr,
        ])
        return feat

    @staticmethod
    def _cluster_speaker_features(features: np.ndarray, max_speakers: int = 4) -> np.ndarray:
        """
        Phân cụm tự động các vector đặc trưng giọng nói thành các nhóm nhân vật.
        Tự động chọn số lượng người nói tối ưu từ 1 đến max_speakers dựa trên độ tách biệt âm sắc và cao độ.
        """
        n_samples = features.shape[0]
        if n_samples <= 1 or max_speakers <= 1:
            return np.zeros(n_samples, dtype=np.int32)

        # Kiểm tra độ lệch chuẩn của cao độ F0 (cột 0 * 200Hz): nếu biến thiên quá nhỏ và phổ đồng nhất -> 1 người nói
        f0_hz = features[:, 0] * 200.0
        f0_spread = float(np.percentile(f0_hz, 90) - np.percentile(f0_hz, 10))

        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-6
        norm_feat = (features - mean) / std
        # Tăng trọng số cho cao độ F0 và trọng tâm phổ để phân biệt rõ Nam / Nữ / Trầm / Cao
        norm_feat[:, 0] *= 2.2
        norm_feat[:, 2] *= 1.5

        if f0_spread < 18.0 and float(np.mean(np.std(features[:, 3:], axis=0))) < 0.06:
            return np.zeros(n_samples, dtype=np.int32)

        max_k = min(max_speakers, n_samples // 2 if n_samples >= 4 else n_samples)
        max_k = max(2, min(max_k, 6))

        best_labels = np.zeros(n_samples, dtype=np.int32)
        best_score = -1.0

        rng = np.random.default_rng(42)
        for k in range(2, max_k + 1):
            # Khởi tạo K-Means++ theo phân vị cao độ F0 để hội tụ ổn định
            quantiles = np.linspace(10, 90, k)
            init_indices = [int(np.argmin(np.abs(f0_hz - np.percentile(f0_hz, q)))) for q in quantiles]
            centers = norm_feat[init_indices].copy()

            labels = np.zeros(n_samples, dtype=np.int32)
            for _ in range(25):
                dists = np.linalg.norm(norm_feat[:, None, :] - centers[None, :, :], axis=2)
                new_labels = np.argmin(dists, axis=1).astype(np.int32)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for c_idx in range(k):
                    members = norm_feat[labels == c_idx]
                    if len(members) > 0:
                        centers[c_idx] = np.mean(members, axis=0)
                    else:
                        centers[c_idx] = norm_feat[rng.integers(0, n_samples)]

            # Tính hệ số Silhouette để chọn số người nói K phù hợp nhất
            unique_clusters = np.unique(labels)
            if len(unique_clusters) < 2:
                continue

            sil_scores: List[float] = []
            for idx in range(n_samples):
                own_c = labels[idx]
                same_mask = (labels == own_c)
                same_mask[idx] = False
                a_val = float(np.mean(np.linalg.norm(norm_feat[same_mask] - norm_feat[idx], axis=1))) if np.any(same_mask) else 0.0
                b_val = float("inf")
                for other_c in unique_clusters:
                    if other_c == own_c:
                        continue
                    other_mask = (labels == other_c)
                    if np.any(other_mask):
                        dist_other = float(np.mean(np.linalg.norm(norm_feat[other_mask] - norm_feat[idx], axis=1)))
                        if dist_other < b_val:
                            b_val = dist_other
                denom = max(a_val, b_val, 1e-6)
                sil_scores.append((b_val - a_val) / denom)

            mean_sil = float(np.mean(sil_scores)) if sil_scores else -1.0
            if mean_sil > best_score:
                best_score = mean_sil
                best_labels = labels.copy()

        # Chuẩn hóa thứ tự nhãn theo lần xuất hiện đầu tiên trong kịch bản (0, 1, 2...)
        ordered_labels = np.zeros_like(best_labels)
        mapping: Dict[int, int] = {}
        next_id = 0
        for lbl in best_labels:
            val = int(lbl)
            if val not in mapping:
                mapping[val] = next_id
                next_id += 1
            ordered_labels[len(mapping) and 0] = ordered_labels[0]  # no-op
        for idx, lbl in enumerate(best_labels):
            ordered_labels[idx] = mapping[int(lbl)]
        return ordered_labels

    def detect_speakers_from_video(
        self,
        video_path: str,
        blocks: List[SubtitleBlock],
        max_speakers: int = 4,
        prefer_cuda: Optional[bool] = None,
        allow_mock: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tự động nhận diện nhiều người nói trực tiếp từ âm thanh gốc của video theo mốc thời gian phụ đề (.srt).
        Sử dụng bộ điều phối SpeakerDiarizationCoordinator (ONNX Runtime tăng tốc GPU/CPU) và bảo vệ các câu đã khóa.
        """
        if not video_path or not os.path.exists(video_path):
            self.log("Không tìm thấy tệp video để phân tích giọng người nói.")
            return {}
        if not blocks:
            return {}

        self._active_diarizer_session = None

        from .speaker_diarization import SpeakerDiarizationCoordinator, ConfidenceLevel

        cuda_avail = bool(torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
        use_cuda = cuda_avail if prefer_cuda is None else bool(prefer_cuda)
        effective_allow_mock = self.allow_mock if allow_mock is None else bool(allow_mock)
        coordinator = SpeakerDiarizationCoordinator(
            model_path=self.speaker_model_path,
            prefer_cuda=use_cuda,
            allow_mock=effective_allow_mock,
        )

        self.last_diarization_error = ""
        self.log(f"Đang tách kênh giọng nói từ video để nhận diện đa nhân vật: {os.path.basename(video_path)}...")
        separator = VoiceSeparator()
        temp_vocals = os.path.join(self.voice_prompt_cache_dir, f"diarize_vocals_{uuid.uuid4().hex[:8]}.wav")
        temp_raw_audio = os.path.join(self.voice_prompt_cache_dir, f"diarize_audio_{uuid.uuid4().hex[:8]}.wav")
        source_audio_for_diarization = ""

        try:
            ok = separator.separate_from_video(video_path, temp_vocals)
            if ok and os.path.isfile(temp_vocals) and os.path.getsize(temp_vocals) > 0:
                try:
                    _voc_info = sf.info(temp_vocals)
                    if _voc_info.frames > 0:
                        _sample_frames = min(_voc_info.frames, 32000)
                        _check_data, _ = sf.read(temp_vocals, stop=_sample_frames)
                        if np.max(np.abs(_check_data)) >= 1e-4:
                            source_audio_for_diarization = temp_vocals
                except Exception:
                    pass

            if not source_audio_for_diarization:
                # Nếu tách giọng thất bại hoặc bị lọc thành im lặng, dùng âm thanh gốc từ video/audio
                self.log("Không tách được giọng sạch; đang thử trích xuất âm thanh gốc để phân tích.")
                if separator._extract_audio(video_path, temp_raw_audio) and os.path.isfile(temp_raw_audio):
                    source_audio_for_diarization = temp_raw_audio
                elif os.path.isfile(video_path):
                    try:
                        import soundfile as sf
                        _info = sf.info(video_path)
                        if _info.frames > 0:
                            source_audio_for_diarization = video_path
                    except Exception:
                        pass

            if not source_audio_for_diarization:
                raise RuntimeError(
                    "Không thể tách giọng hoặc trích xuất âm thanh từ video. "
                    "Kiểm tra tệp nguồn và FFmpeg rồi thử lại."
                )

            # VoiceSeparator có thể đã sử dụng PyTorch/CUDA (ví dụ Demucs).
            # Thu hồi cache của PyTorch trước khi khởi tạo phiên ONNX để giảm tranh chấp VRAM.
            gc.collect()
            if torch is not None and getattr(torch, "cuda", None) is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    self.log("Không thể dọn cache CUDA trước khi chạy ONNX; tiếp tục với kiểm soát lỗi.")

            diar_profiles = coordinator.process_video_subtitles(
                video_path=source_audio_for_diarization,
                blocks=blocks,
                max_speakers=max_speakers,
                prefer_cuda=use_cuda,
                temp_dir=self.voice_prompt_cache_dir,
            )
            self.log(f"Nhận diện người nói hoàn tất bằng {coordinator.last_provider}.")

            # Đảm bảo mọi câu chưa khóa đều có confidence_level
            for blk in blocks:
                if not hasattr(blk, "confidence_level"):
                    blk.confidence_level = ConfidenceLevel.HIGH.value if getattr(blk, "speaker", "") else ConfidenceLevel.UNASSIGNED.value

            detected_profiles: Dict[str, Dict[str, Any]] = {}
            for spk_name, prof in diar_profiles.items():
                if hasattr(prof, "to_dict"):
                    prof_data = prof.to_dict()
                else:
                    prof_data = dict(prof)
                prof_data["pitch_hz"] = 150.0
                ratio = getattr(prof, "high_confidence_ratio", 1.0)
                prof_data["high_confidence_ratio"] = float(ratio)
                prof_data["description"] = f"Độ tin cậy: {int(ratio * 100)}%"
                prof_data["confidence_level"] = "high" if ratio >= 0.7 else "review_needed"
                if ratio < 0.7:
                    prof_data["status"] = "review_needed"
                detected_profiles[spk_name] = prof_data

            self.speaker_profiles = detected_profiles
            self.log(
                f"Đã tự động nhận diện {len(detected_profiles)} người nói từ video: "
                + ", ".join(f"{k} ({v['description']} - {v['count']} câu)" for k, v in detected_profiles.items())
            )
            return detected_profiles
        except Exception as e:
            self.last_diarization_error = str(e)
            self.log(f"Lỗi khi phân tích đa người nói từ video: {e}")
            return {}
        finally:
            self._active_diarizer_session = None
            for temp_path in (temp_vocals, temp_raw_audio):
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            gc.collect()

    def detect_speakers_from_srt_text(
        self,
        blocks: List[SubtitleBlock],
        max_speakers: int = 4,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tự động nhận diện và phân vai nhiều người nói khi chỉ cung cấp tệp phụ đề (.srt) mà không có video:
        - Nhận diện tiền tố tên nhân vật ([Nam]:, [Nữ]:, Người 1:, Nhân vật A:, SPEAKER_01:...)
        - Nhận diện hội thoại gạch đầu dòng (- ) và đại từ nhân xưng tiếng Việt / Anh / Trung
        - Tự động gán thiết kế giọng nói riêng biệt cho từng nhân vật
        """
        if not blocks:
            return {}

        prefix_re = re.compile(
            r"^\s*(?:\[([^\]]{1,25})\]\s*[:：\-–—]?|\(([^\)]{1,25})\)\s*[:：\-–—]?|([A-Za-zÀ-ỹ0-9_ ]{1,20})\s*[:：\-–—])\s*(.+)$"
        )
        female_markers = {"nữ", "nu", "female", "woman", "girl", "cô", "chị", "bà", "mẹ", "em", "nàng", "thiếp", "tiểu thư"}
        male_markers = {"nam", "male", "man", "boy", "anh", "ông", "bố", "cha", "chú", "bác", "chàng", "huynh", "trẫm"}
        elder_markers = {"ông", "bà", "lão", "cụ", "elder", "old"}
        child_markers = {"bé", "con", "cháu", "trẻ", "child", "kid"}

        named_speaker_map: Dict[str, str] = {}
        detected_profiles: Dict[str, Dict[str, Any]] = {}

        def _register_speaker(spk_id: str, instruct_str: str, desc_str: str, method: str = "preset", ref_path: str = ""):
            if spk_id not in detected_profiles:
                detected_profiles[spk_id] = {
                    "speaker_id": spk_id,
                    "description": desc_str,
                    "pitch_hz": 0.0,
                    "ref_audio_path": ref_path,
                    "source_audio_path": ref_path,
                    "method": method,
                    "status": "locked",
                    "instruct": instruct_str,
                    "voice_instruct": instruct_str,
                    "count": 0,
                }
            detected_profiles[spk_id]["count"] += 1

        current_turn = 0
        for blk in blocks:
            raw_text = (blk.translated_text if blk.translated_text else blk.text).strip()
            lower_text = raw_text.lower()

            # 1. Tách tiền tố người nói khỏi nội dung thoại nếu có để TTS không đọc tên nhân vật
            m = prefix_re.match(raw_text)
            has_valid_prefix = False
            tag_name = ""
            tag_key = ""
            if m:
                tag_name = (m.group(1) or m.group(2) or m.group(3) or "").strip()
                tag_key = tag_name.lower()
                if f"[{tag_key}]" not in NON_VERBAL_TAGS:
                    has_valid_prefix = True
                    clean_text = m.group(4).lstrip(":：-–— ").strip()
                    if blk.translated_text:
                        m_tr = prefix_re.match(blk.translated_text.strip())
                        if m_tr and (m_tr.group(1) or m_tr.group(2) or m_tr.group(3)):
                            blk.translated_text = m_tr.group(4).lstrip(":：-–— ").strip()
                    if blk.text:
                        m_tx = prefix_re.match(blk.text.strip())
                        if m_tx and (m_tx.group(1) or m_tx.group(2) or m_tx.group(3)):
                            blk.text = m_tx.group(4).lstrip(":：-–— ").strip()
                    raw_text = clean_text
                    lower_text = raw_text.lower()

            # 2. Nếu dòng thoại đã được người dùng khóa nhân vật thủ công, giữ nguyên hoàn toàn cả tên và phương pháp giọng
            if getattr(blk, "is_speaker_locked", False) and getattr(blk, "speaker", "").strip():
                assigned_spk = blk.speaker
                assigned_instruct = getattr(blk, "voice_instruct", "") or DEFAULT_SPEAKER_PRESETS.get(assigned_spk, {}).get("instruct", "")
                assigned_desc = getattr(blk, "speaker", "")

                existing_prof = self.speaker_profiles.get(assigned_spk, {})
                is_ex_dict = isinstance(existing_prof, dict)
                ex_method = (
                    existing_prof.get("method")
                    if is_ex_dict
                    else getattr(existing_prof, "method", "")
                )
                ex_ref = (
                    getattr(blk, "voice_ref_path", "")
                    or (
                        (existing_prof.get("ref_audio_path") or existing_prof.get("source_audio_path", ""))
                        if is_ex_dict
                        else (getattr(existing_prof, "source_audio_path", "") or getattr(existing_prof, "ref_audio_path", ""))
                    )
                )
                if not ex_method:
                    ex_method = "clone" if ex_ref else "preset"

                _register_speaker(assigned_spk, assigned_instruct, assigned_desc, method=ex_method, ref_path=ex_ref)
                continue

            assigned_spk = ""
            assigned_instruct = ""
            assigned_desc = ""

            if has_valid_prefix:
                if tag_key not in named_speaker_map:
                    spk_num = min(len(named_speaker_map) + 1, max_speakers)
                    spk_id = f"Người nói {spk_num}"
                    named_speaker_map[tag_key] = spk_id
                assigned_spk = named_speaker_map[tag_key]

                if any(w in tag_key for w in female_markers):
                    assigned_instruct = "female, young adult, moderate pitch"
                    assigned_desc = f"Giọng Nữ ({tag_name})"
                elif any(w in tag_key for w in male_markers):
                    assigned_instruct = "male, young adult, moderate pitch"
                    assigned_desc = f"Giọng Nam ({tag_name})"
                elif any(w in tag_key for w in child_markers):
                    assigned_instruct = "child, high pitch"
                    assigned_desc = f"Giọng Trẻ Em ({tag_name})"
                elif any(w in tag_key for w in elder_markers):
                    assigned_instruct = "male, elderly, very low pitch"
                    assigned_desc = f"Giọng Cao Tuổi ({tag_name})"

            if not assigned_spk:
                # Phân tích dấu gạch đầu dòng hội thoại hoặc đại từ xưng hô trong câu
                words = set(re.findall(r"[a-zA-ZÀ-ỹ]+", lower_text))
                if raw_text.startswith("-") or raw_text.startswith("—") or "?" in raw_text:
                    current_turn = (current_turn + 1) % min(max(2, max_speakers), 4)

                if words & child_markers and max_speakers >= 4:
                    assigned_spk = f"Người nói {min(4, max_speakers)}"
                    assigned_instruct = "child, high pitch"
                    assigned_desc = "Giọng Trẻ Em / Cao"
                elif words & elder_markers and max_speakers >= 3:
                    assigned_spk = f"Người nói {min(3, max_speakers)}"
                    assigned_instruct = "male, middle-aged, low pitch"
                    assigned_desc = "Giọng Trầm / Trung Niên"
                elif (words & female_markers) and not (words & male_markers):
                    assigned_spk = "Người nói 2" if max_speakers >= 2 else "Người nói 1"
                    assigned_instruct = "female, young adult, moderate pitch"
                    assigned_desc = "Giọng Nữ Thanh Niên"
                elif (words & male_markers) and not (words & female_markers):
                    assigned_spk = "Người nói 1"
                    assigned_instruct = "male, young adult, moderate pitch"
                    assigned_desc = "Giọng Nam Thanh Niên"
                else:
                    spk_idx = (current_turn % min(max(2, max_speakers), len(DEFAULT_SPEAKER_PRESETS))) + 1
                    assigned_spk = f"Người nói {spk_idx}"

            preset = DEFAULT_SPEAKER_PRESETS.get(assigned_spk, DEFAULT_SPEAKER_PRESETS["Người nói 1"])
            if not assigned_instruct:
                assigned_instruct = preset["instruct"]
            if not assigned_desc:
                assigned_desc = preset["label"].split(" - ")[-1]

            blk.speaker = assigned_spk
            blk.voice_instruct = assigned_instruct
            blk.voice_ref_path = ""
            _register_speaker(assigned_spk, assigned_instruct, assigned_desc)

        self.speaker_profiles = detected_profiles
        self.log(
            f"Đã tự động phân vai {len(detected_profiles)} nhân vật từ văn bản phụ đề: "
            + ", ".join(f"{k} ({v['description']} - {v['count']} câu)" for k, v in detected_profiles.items())
        )
        return detected_profiles

    def generate_single(
        self,
        text: str,
        output_path: Optional[str] = None,
        target_duration: Optional[float] = None,
        override_mode: Optional[str] = None,
        override_prompt: Optional[VoiceClonePrompt] = None,
        override_ref_audio: Optional[str] = None,
        override_instruct: Optional[str] = None,
    ) -> np.ndarray:
        """
        Tổng hợp giọng nói cho một câu đơn lẻ bằng OmniVoice trên GPU.
        Hỗ trợ ghi đè mẫu giọng hoặc thuộc tính thiết kế giọng trên từng câu cho chế độ Đa Nhân Vật.
        """
        self.load_model()
        text_clean = text.strip()
        if not text_clean:
            return np.zeros(0, dtype=np.float32)

        gen_config = self.build_generation_config()

        gen_kwargs: Dict[str, Any] = {
            "text": text_clean,
            "language": self.language,
            "generation_config": gen_config,
            "normalize_text": self.normalize_text,
        }

        if target_duration is not None and target_duration > 0.4:
            gen_kwargs["duration"] = float(target_duration)
        elif self.speed != 1.0:
            gen_kwargs["speed"] = float(self.speed)

        active_mode = override_mode or self.mode

        if active_mode == "design":
            # Chế độ thiết kế giọng nói: Tuyệt đối không dùng mẫu âm thanh nhân bản
            if override_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = override_prompt
            elif override_instruct:
                gen_kwargs["instruct"] = override_instruct
            elif self.instruct:
                gen_kwargs["instruct"] = self.instruct
            else:
                gen_kwargs["instruct"] = DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]
        elif override_mode == "clone":
            if override_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = override_prompt
            elif override_ref_audio and os.path.exists(override_ref_audio):
                prompt = self._get_or_create_prompt_for_audio(override_ref_audio, cache_key=override_ref_audio)
                if prompt is not None:
                    gen_kwargs["voice_clone_prompt"] = prompt
                else:
                    raise RuntimeError(f"Không thể tạo đặc trưng giọng từ tệp mẫu: {override_ref_audio}")
            else:
                raise FileNotFoundError(f"Tệp âm thanh mẫu nhân bản giọng không tồn tại hoặc chưa được chỉ định: {override_ref_audio or 'chưa có'}")
        elif override_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = override_prompt
        elif override_ref_audio and os.path.exists(override_ref_audio):
            prompt = self._get_or_create_prompt_for_audio(override_ref_audio, cache_key=override_ref_audio)
            if prompt is not None:
                gen_kwargs["voice_clone_prompt"] = prompt
            elif override_instruct:
                gen_kwargs["instruct"] = override_instruct
            elif self.instruct:
                gen_kwargs["instruct"] = self.instruct
        elif override_instruct:
            gen_kwargs["instruct"] = override_instruct
        elif active_mode == "clone":
            if self.current_prompt is None and self.ref_audio_path and os.path.exists(self.ref_audio_path):
                self.set_reference_voice(self.ref_audio_path, self.ref_text)

            if self.current_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = self.current_prompt
            elif self.instruct:
                gen_kwargs["instruct"] = self.instruct

        # Luôn bảo đảm có điều kiện giọng nói để không bị lệch chất giọng ngoài ý muốn
        if "voice_clone_prompt" not in gen_kwargs and "instruct" not in gen_kwargs:
            if override_instruct:
                gen_kwargs["instruct"] = override_instruct
            elif self.instruct:
                gen_kwargs["instruct"] = self.instruct
            else:
                gen_kwargs["instruct"] = "male, young adult, moderate pitch"

        result = self.model.generate(**gen_kwargs)

        if isinstance(result, (list, tuple)) and len(result) > 0:
            first_item = result[0]
            if hasattr(first_item, "detach"):
                audio_arr = first_item.detach().cpu().numpy()
            else:
                audio_arr = np.asarray(first_item, dtype=np.float32)
        elif hasattr(result, "detach"):
            audio_arr = result.detach().cpu().numpy()
        else:
            audio_arr = np.asarray(result, dtype=np.float32)

        audio_arr = np.squeeze(audio_arr).astype(np.float32)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, audio_arr, self.sample_rate)

        return audio_arr

    def generate_batch(
        self,
        texts: List[str],
        languages: Optional[List[Optional[str]]] = None,
        prompts: Optional[List[Optional[VoiceClonePrompt]]] = None,
        instructs: Optional[List[Optional[str]]] = None,
        durations: Optional[List[Optional[float]]] = None,
    ) -> List[np.ndarray]:
        """Tổng hợp đồng thời một lô nhiều câu thoại trên GPU bằng cơ chế Batch của OmniVoice."""
        self.load_model()
        if not texts:
            return []
        gen_config = self.build_generation_config()
        kwargs: Dict[str, Any] = {
            "text": texts,
            "language": languages if languages is not None else self.language,
            "generation_config": gen_config,
            "normalize_text": self.normalize_text,
        }
        if prompts is not None and any(p is not None for p in prompts):
            kwargs["voice_clone_prompt"] = prompts
        elif instructs is not None and any(ins is not None for ins in instructs):
            kwargs["instruct"] = instructs
        if durations is not None and any(d is not None for d in durations):
            kwargs["duration"] = durations
        elif self.speed != 1.0:
            kwargs["speed"] = self.speed

        results = self.model.generate(**kwargs)
        return [np.squeeze(np.asarray(r, dtype=np.float32)) for r in results]

    def synthesize_preview(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Tổng hợp nhanh một câu nghe thử từ giao diện Cài đặt MMO và trả về đường dẫn tệp WAV.
        """
        if not text or not text.strip():
            text = "Xin chào, đây là giọng đọc trí tuệ nhân tạo OmniVoice chạy trực tiếp trên card đồ họa của bạn."
        if not output_path:
            output_path = os.path.join(self.voice_prompt_cache_dir, "omnivoice_preview.wav")
        self.generate_single(text, output_path=output_path)
        return output_path if os.path.exists(output_path) else None

    def generate_audio_for_block(self, text: str, output_path: str) -> bool:
        """Tạo tệp WAV từ chuỗi văn bản cho một khối phụ đề."""
        try:
            self.generate_single(text, output_path=output_path)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            self.log(f"Lỗi khi sinh âm thanh cho câu: {str(e)}")
            return False

    @staticmethod
    def _safe_time_stretch(audio_arr: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Co giãn thời lượng âm thanh trong ngưỡng an toàn (0.80x - 1.35x) bằng nội suy tuyến tính
        kèm làm mượt biên độ hai đầu (fade-in/fade-out) để tránh chồng lấn sang câu tiếp theo mà không cắt cụt từ.
        """
        if audio_arr.size == 0 or abs(speed_factor - 1.0) < 0.02:
            return audio_arr
        clamped_factor = max(0.80, min(1.35, float(speed_factor)))
        new_len = max(1, int(round(len(audio_arr) / clamped_factor)))
        old_indices = np.linspace(0.0, len(audio_arr) - 1, num=new_len, dtype=np.float32)
        base_idx = np.arange(len(audio_arr), dtype=np.float32)
        stretched = np.interp(old_indices, base_idx, audio_arr).astype(np.float32)
        fade_len = min(240, len(stretched) // 10)
        if fade_len > 2:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            stretched[:fade_len] *= ramp
            stretched[-fade_len:] *= ramp[::-1]
        return stretched

    def _compute_block_cache_path(
        self,
        index: int,
        text: str,
        speaker: str,
        ref_path: Optional[str],
        instruct: Optional[str],
        target_dur: Optional[float],
        method_or_mode: Optional[str] = None,
    ) -> Optional[str]:
        """Tính đường dẫn tệp WAV lưu tạm (checkpoint) cho từng câu để hỗ trợ tiếp tục khi gián đoạn."""
        if not self.resume_cache_dir:
            return None
        mtime_str = ""
        if ref_path and os.path.exists(ref_path):
            try:
                mtime_str = str(os.path.getmtime(ref_path))
            except Exception:
                mtime_str = ""
        raw_key = f"{index}|{text}|{speaker}|{method_or_mode or ''}|{ref_path or ''}|{mtime_str}|{instruct or ''}|{target_dur or 0.0:.2f}|{self.mode}|{self.speed:.2f}"
        digest = hashlib.sha1(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return os.path.join(self.resume_cache_dir, f"blk_{index + 1:04d}_{digest}.wav")

    def _prepare_and_lock_character_prompts(self, blocks: List[SubtitleBlock]):
        """
        Khởi tạo trước và khóa vĩnh viễn prompt âm sắc hoặc preset tiếng Việt cho từng nhân vật
        trước khi bắt đầu tổng hợp câu thoại, đảm bảo tính nhất quán tuyệt đối xuyên suốt video.
        """
        speaker_keys = set()
        for blk in blocks:
            spk = getattr(blk, "speaker", "").strip()
            if spk and spk != "Chưa xác định":
                speaker_keys.add(spk)
        for spk in self.speaker_profiles.keys():
            if spk and spk != "Chưa xác định":
                speaker_keys.add(spk)

        preset_keys = list(VIETNAMESE_VOICE_CATALOG.keys())

        for idx, spk in enumerate(sorted(speaker_keys)):
            prof = self.speaker_profiles.get(spk, {})
            method = prof.get("method") or prof.get("assigned_strategy") or "clone"
            preset_name = prof.get("preset_name") or prof.get("voice_preset") or ""
            ref_path = (
                prof.get("source_audio_path")
                or prof.get("ref_audio_path")
                or prof.get("source_audio_sample")
                or ""
            )
            if not ref_path:
                for blk in blocks:
                    if getattr(blk, "speaker", "") == spk and getattr(blk, "voice_ref_path", ""):
                        ref_path = blk.voice_ref_path
                        break

            char_prompt_key = f"character_{spk}"

            # 1. Nếu nhân vật chọn phương pháp preset hoặc đã có preset chỉ định rõ
            if method == "preset" and preset_name in VIETNAMESE_VOICE_CATALOG:
                self._speaker_instruct_cache[spk] = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
                continue

            # 2. Nếu nhân vật chọn phương pháp clone
            if method == "clone" and ref_path and os.path.exists(ref_path):
                if char_prompt_key not in self._speaker_prompt_cache:
                    prompt = self._get_or_create_prompt_for_audio(ref_path, cache_key=char_prompt_key)
                    if prompt is not None:
                        self._speaker_prompt_cache[char_prompt_key] = prompt
                    else:
                        # Tạo prompt thất bại: Chuyển đổi dự phòng an toàn sang preset tiếng Việt và khóa vĩnh viễn
                        chosen_preset = preset_name if preset_name in VIETNAMESE_VOICE_CATALOG else preset_keys[idx % len(preset_keys)]
                        fallback_ins = VIETNAMESE_VOICE_CATALOG[chosen_preset]["instruct"]
                        self._speaker_instruct_cache[spk] = fallback_ins
                        if hasattr(prof, "method"):
                            prof.method = "preset"
                            prof.preset_name = chosen_preset
                            prof.voice_instruct = fallback_ins
                            prof.status = "locked"
                        elif isinstance(prof, dict):
                            prof["method"] = "preset"
                            prof["preset_name"] = chosen_preset
                            prof["voice_instruct"] = fallback_ins
                            prof["instruct"] = fallback_ins
                            prof["status"] = "locked"
                        self.log(
                            f"Không thể tạo mẫu nhân bản cho '{spk}' từ tệp âm thanh; "
                            f"đã tự động khóa phương án dự phòng ({chosen_preset}) cho toàn bộ video."
                        )
                continue

            # 3. Nếu không có tệp mẫu hoặc method khác, khóa vào instruct/preset
            if spk not in self._speaker_instruct_cache:
                chosen_preset = preset_name if preset_name in VIETNAMESE_VOICE_CATALOG else preset_keys[idx % len(preset_keys)]
                ins_cand = (
                    prof.get("voice_instruct")
                    or prof.get("instruct")
                    or VIETNAMESE_VOICE_CATALOG[chosen_preset]["instruct"]
                )
                self._speaker_instruct_cache[spk] = ins_cand

    def synthesize_blocks(
        self,
        blocks: List[SubtitleBlock],
        total_duration: float,
        output_path: str,
        video_path: Optional[str] = None,
    ) -> DubbingReport:
        """
        Tổng hợp giọng nói cho toàn bộ danh sách khối phụ đề theo kiến trúc 5 giai đoạn:
        - Giai đoạn 1: Kiểm tra và chuẩn hóa mốc thời gian phụ đề.
        - Giai đoạn 2: Chuẩn bị hồ sơ giọng nhân vật (đơn giọng, phân cụm từ video, phân vai từ phụ đề, hoặc nhái từng câu).
        - Giai đoạn 3: Tổng hợp trên GPU kèm khôi phục checkpoint, chia câu dài tiếng Việt, thử lại khi lỗi và xử lý thiếu VRAM.
        - Giai đoạn 4: Căn chỉnh chống chồng lấn, tự động mở rộng bộ đệm không cắt câu cuối và nghiệm thu tệp WAV đầu ra.
        """
        self.reset_cancel()
        report = DubbingReport(output_path=output_path, total_blocks=len(blocks))
        self.last_report = report

        # GIAI ĐOẠN 1: Chuẩn hóa và kiểm tra phụ đề
        ts_issues = validate_and_normalize_subtitles(blocks, video_duration=total_duration, auto_fix=True)
        report.timestamp_issues = ts_issues
        for issue in ts_issues:
            if issue["type"] != "empty_text":
                self.log(f"Cảnh báo phụ đề: {issue['message']}")

        # GIAI ĐOẠN 2: Chuẩn bị hồ sơ giọng nhân vật trước khi nạp OmniVoice
        vocals_waveform: Optional[np.ndarray] = None
        vocals_sr: int = 24000

        if self.multi_speaker_mode == "auto_video_cluster" and video_path and os.path.exists(video_path):
            if not any(getattr(b, "speaker", "") for b in blocks):
                self.detect_speakers_from_video(video_path, blocks, max_speakers=self.max_speakers)
        elif self.multi_speaker_mode == "auto_srt_text":
            if not any(getattr(b, "speaker", "") for b in blocks):
                self.detect_speakers_from_srt_text(blocks, max_speakers=self.max_speakers)
        elif self.multi_speaker_mode == "per_line_video_clone" and video_path and os.path.exists(video_path):
            self.log("Đang tách kênh giọng nói gốc khỏi nhạc nền để nhái giọng theo từng câu thoại...")
            sep = VoiceSeparator()
            temp_voc = os.path.join(self.voice_prompt_cache_dir, f"perline_voc_{uuid.uuid4().hex[:8]}.wav")
            if sep.separate_from_video(video_path, temp_voc) and os.path.exists(temp_voc):
                try:
                    v_data, vocals_sr = sf.read(temp_voc)
                    if v_data.ndim > 1:
                        v_data = np.mean(v_data, axis=1)
                    vocals_waveform = v_data.astype(np.float32)
                finally:
                    try:
                        os.remove(temp_voc)
                    except Exception:
                        pass

        # Nạp mô hình TTS sau khi hoàn tất nhận diện và giải phóng phiên ONNX.
        self.load_model()
        self._prepare_and_lock_character_prompts(blocks)

        if self.mode == "clone" and self.current_prompt is None and self.multi_speaker_mode == "single":
            if not self.auto_extract_ref and self.ref_audio_path and os.path.exists(self.ref_audio_path):
                self.set_reference_voice(self.ref_audio_path, self.ref_text)
            elif self.auto_extract_ref and video_path and os.path.exists(video_path):
                self.extract_reference_from_video(video_path)
            elif self.ref_audio_path and os.path.exists(self.ref_audio_path):
                self.set_reference_voice(self.ref_audio_path, self.ref_text)

        total_samples = int((max(total_duration, 1.0) + 5.0) * self.sample_rate)
        full_audio = np.zeros(total_samples, dtype=np.float32)

        # GIAI ĐOẠN 3 & 4: Tổng hợp từng câu, kiểm tra chất lượng và căn chỉnh âm thanh
        for i, blk in enumerate(blocks):
            if self._cancel_requested:
                self.log(f"Đã dừng tác vụ theo yêu cầu người dùng tại dòng phụ đề {i + 1}.")
                report.status = "cancelled"
                report.message = f"Đã hủy tác vụ sau khi xử lý {report.succeeded_count}/{len(blocks)} câu."
                self.last_report = report
                return report

            raw_text = (blk.translated_text if blk.translated_text else blk.text) or ""
            if not raw_text.strip():
                report.skipped_empty_count += 1
                continue

            text_to_speak = (
                preprocess_vietnamese_tts_text(raw_text)
                if (self.language in (None, "vi") and self.normalize_text)
                else raw_text.strip()
            )

            start_sec = max(0.0, parse_timestamp_to_seconds(blk.start_time))
            end_sec = max(start_sec + 0.2, parse_timestamp_to_seconds(getattr(blk, "end_time", start_sec)))
            slot_duration = max(0.0, end_sec - start_sec)

            target_dur: Optional[float] = None
            if self.sync_to_subtitle_duration and slot_duration >= 0.8:
                target_dur = slot_duration

            override_prompt: Optional[VoiceClonePrompt] = None
            override_ref: Optional[str] = None
            override_instruct: Optional[str] = None
            spk_key = getattr(blk, "speaker", "")
            spk_label = spk_key or "Mặc định"

            # 1. Chế độ nhái giọng trực tiếp theo từng câu từ video gốc (có kiểm định chất lượng đoạn mẫu)
            if self.multi_speaker_mode == "per_line_video_clone" and vocals_waveform is not None:
                s_idx = max(0, int(start_sec * vocals_sr))
                e_idx = min(len(vocals_waveform), int(end_sec * vocals_sr))
                if e_idx - s_idx < int(1.8 * vocals_sr):
                    pad_half = int(0.9 * vocals_sr)
                    s_idx = max(0, s_idx - pad_half)
                    e_idx = min(len(vocals_waveform), e_idx + pad_half)
                seg_wav = vocals_waveform[s_idx:e_idx]
                ok_seg, _ = is_audio_quality_acceptable_for_voice_lock(seg_wav, vocals_sr, min_duration_sec=0.9)
                if ok_seg:
                    # Ưu tiên lấy câu gốc của phụ đề làm phiên âm nếu không rỗng, tránh dùng dấu chấm vô nghĩa
                    line_ref_text = (getattr(blk, "text", "") or "").strip() or None
                    seg_tensor = torch.from_numpy(seg_wav).unsqueeze(0) if torch is not None else seg_wav
                    override_prompt = self._get_or_create_prompt_for_audio(
                        (seg_tensor, vocals_sr),
                        cache_key=f"line_{i}",
                        ref_text=line_ref_text,
                    )
                    spk_label = f"Giọng gốc câu {i + 1}"
                else:
                    self.log(f"Câu {i + 1}: Đoạn giọng gốc quá ngắn hoặc nhiễu, chuyển sang dùng mẫu giọng nhân vật dự phòng.")

            # 2. Chế độ Đa nhân vật hoặc dự phòng khi nhái giọng từng câu không đủ chất lượng
            active_block_mode: Optional[str] = None
            if override_prompt is None and (self.multi_speaker_mode != "single" or spk_key):
                prof = self.speaker_profiles.get(spk_key, {}) if spk_key else {}
                is_dict = isinstance(prof, dict)
                prof_method = prof.get("method") if is_dict else getattr(prof, "method", "")

                if prof_method == "preset":
                    active_block_mode = "design"
                    # Xóa prompt nhân bản từ video cũ nếu còn sót
                    character_prompt_key = f"character_{spk_key}" if spk_key else ""
                    self._speaker_prompt_cache.pop(character_prompt_key, None)
                    override_ref = None

                    # Giữ prompt nhất quán đã tạo từ câu trước của chính nhân vật này (nếu có)
                    if spk_key and f"consistent_{spk_key}" in self._speaker_prompt_cache:
                        override_prompt = self._speaker_prompt_cache[f"consistent_{spk_key}"]
                    else:
                        override_prompt = None

                    ins_cand = (
                        (self._speaker_instruct_cache.get(spk_key, "") if spk_key else "")
                        or getattr(blk, "voice_instruct", "")
                        or (prof.get("instruct") or prof.get("voice_instruct") if is_dict else getattr(prof, "voice_instruct", ""))
                    )
                    preset_name = (prof.get("preset_name") if is_dict else getattr(prof, "preset_name", "")) if prof else ""
                    if not ins_cand and preset_name in VIETNAMESE_VOICE_CATALOG:
                        ins_cand = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
                    if not ins_cand and spk_key in DEFAULT_SPEAKER_PRESETS:
                        ins_cand = DEFAULT_SPEAKER_PRESETS[spk_key]["instruct"]
                    if not ins_cand and self.instruct:
                        ins_cand = self.instruct
                    override_instruct = ins_cand
                else:
                    ref_cand = getattr(blk, "voice_ref_path", "") or (prof.get("ref_audio_path", "") or prof.get("source_audio_path", "") if is_dict else getattr(prof, "source_audio_path", ""))
                    character_prompt_key = f"character_{spk_key}" if spk_key else ""

                    if character_prompt_key and character_prompt_key in self._speaker_prompt_cache:
                        override_prompt = self._speaker_prompt_cache[character_prompt_key]
                        override_ref = ref_cand
                        active_block_mode = "clone"
                    elif spk_key and f"consistent_{spk_key}" in self._speaker_prompt_cache:
                        override_prompt = self._speaker_prompt_cache[f"consistent_{spk_key}"]
                        override_ref = ref_cand
                        active_block_mode = "clone" if ref_cand else "design"
                    elif ref_cand and os.path.exists(ref_cand):
                        prompt_key = character_prompt_key or f"spk_{spk_key}_{ref_cand}"
                        override_prompt = self._get_or_create_prompt_for_audio(ref_cand, cache_key=prompt_key)
                        if override_prompt is not None and character_prompt_key:
                            self._speaker_prompt_cache[character_prompt_key] = override_prompt
                        override_ref = ref_cand
                        active_block_mode = "clone"
                    elif spk_key and spk_key in self._speaker_instruct_cache:
                        override_instruct = self._speaker_instruct_cache[spk_key]
                        active_block_mode = "design"

                    # Nếu không có prompt nhân bản (hoặc tạo thất bại), luôn chuyển sang hướng dẫn giọng đã khóa
                    if override_prompt is None and not override_instruct:
                        if spk_key and spk_key in self._speaker_instruct_cache:
                            override_instruct = self._speaker_instruct_cache[spk_key]
                            active_block_mode = "design"
                        else:
                            ins_cand = (
                                getattr(blk, "voice_instruct", "")
                                or (self._speaker_instruct_cache.get(spk_key, "") if spk_key else "")
                                or (prof.get("instruct", "") or prof.get("voice_instruct", "") if is_dict else getattr(prof, "voice_instruct", ""))
                            )
                            if not ins_cand and spk_key in DEFAULT_SPEAKER_PRESETS:
                                ins_cand = DEFAULT_SPEAKER_PRESETS[spk_key]["instruct"]

                            if ins_cand:
                                override_instruct = ins_cand
                                active_block_mode = "design"
                            elif self.instruct:
                                override_instruct = self.instruct
                                active_block_mode = "design"
                            elif spk_key and spk_key != "Chưa xác định":
                                preset_keys = list(VIETNAMESE_VOICE_CATALOG.keys())
                                cluster_num = 0
                                import re
                                match = re.search(r"\d+", spk_key)
                                if match:
                                    cluster_num = int(match.group()) - 1
                                preset_name = preset_keys[cluster_num % len(preset_keys)]
                                override_instruct = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]
                                active_block_mode = "design"

            # Kiểm tra bộ nhớ đệm khôi phục tác vụ (Checkpoint Resume)
            cache_wav_path = self._compute_block_cache_path(
                i, text_to_speak, spk_key, override_ref, override_instruct, target_dur, method_or_mode=active_block_mode
            )
            if cache_wav_path and os.path.exists(cache_wav_path):
                ok_cache, _, _, _ = validate_audio_file(cache_wav_path)
                if ok_cache:
                    cached_arr, c_sr = sf.read(cache_wav_path, dtype="float32")
                    if cached_arr.ndim > 1:
                        cached_arr = np.mean(cached_arr, axis=1)
                    start_sample = int(start_sec * self.sample_rate)
                    end_sample = start_sample + len(cached_arr)
                    if end_sample > len(full_audio):
                        full_audio = np.pad(full_audio, (0, end_sample - len(full_audio) + int(0.5 * self.sample_rate)))
                    full_audio[start_sample:end_sample] = np.clip(
                        full_audio[start_sample:end_sample] + cached_arr, -1.0, 1.0
                    )
                    report.succeeded_count += 1
                    report.resumed_from_cache_count += 1
                    self.log(f"Khôi phục từ bộ nhớ đệm [{i + 1}/{len(blocks)}] ({spk_label}): {text_to_speak}")
                    continue

            self.log(f"Đang tổng hợp [{i + 1}/{len(blocks)}] ({spk_label}): {text_to_speak}")
            try:
                clauses = split_long_vietnamese_clauses(text_to_speak, max_chars=220)
                if len(clauses) == 1:
                    audio_arr = self.generate_single(
                        clauses[0],
                        target_duration=target_dur,
                        override_mode=active_block_mode,
                        override_prompt=override_prompt,
                        override_ref_audio=override_ref,
                        override_instruct=override_instruct,
                    )
                else:
                    total_chars = max(1, sum(len(c) for c in clauses))
                    clause_waves: List[np.ndarray] = []
                    pause_samples = np.zeros(int(0.06 * self.sample_rate), dtype=np.float32)
                    for c_idx, clause in enumerate(clauses):
                        c_dur = (target_dur * (len(clause) / total_chars)) if target_dur else None
                        c_wav = self.generate_single(
                            clause,
                            target_duration=c_dur,
                            override_mode=active_block_mode,
                            override_prompt=override_prompt,
                            override_ref_audio=override_ref,
                            override_instruct=override_instruct,
                        )
                        if len(c_wav) > 0:
                            clause_waves.append(c_wav)
                            if c_idx < len(clauses) - 1:
                                clause_waves.append(pause_samples)
                    audio_arr = np.concatenate(clause_waves) if clause_waves else np.zeros(0, dtype=np.float32)

                audio_arr = np.squeeze(np.asarray(audio_arr, dtype=np.float32))
                if audio_arr.size == 0 or float(np.max(np.abs(audio_arr))) < 1e-5:
                    raise RuntimeError("Âm thanh sinh ra trống hoặc hoàn toàn im lặng.")

                # Khóa giọng nhân vật tự động có kiểm định chất lượng (tránh khóa nhầm câu lỗi/ngắn)
                if (
                    spk_key
                    and override_prompt is None
                    and f"consistent_{spk_key}" not in self._speaker_prompt_cache
                ):
                    ok_lock, lock_reason = is_audio_quality_acceptable_for_voice_lock(audio_arr, self.sample_rate)
                    if ok_lock and self.model is not None and hasattr(self.model, "create_voice_clone_prompt"):
                        try:
                            wav_t = torch.from_numpy(audio_arr).unsqueeze(0) if torch is not None else audio_arr
                            locked_prompt = self.model.create_voice_clone_prompt(
                                ref_audio=(wav_t, self.sample_rate),
                                ref_text=text_to_speak,
                                preprocess_prompt=False,
                            )
                            self._speaker_prompt_cache[f"consistent_{spk_key}"] = locked_prompt
                            if spk_key in self.speaker_profiles and not self.speaker_profiles[spk_key].get("ref_audio_path"):
                                auto_ref_file = os.path.join(self.voice_prompt_cache_dir, f"locked_{spk_key.replace(' ', '_')}.wav")
                                sf.write(auto_ref_file, audio_arr, self.sample_rate)
                                self.speaker_profiles[spk_key]["ref_audio_path"] = auto_ref_file
                                for b_other in blocks:
                                    if getattr(b_other, "speaker", "") == spk_key and not getattr(b_other, "voice_ref_path", ""):
                                        b_other.voice_ref_path = auto_ref_file
                        except Exception as lock_err:
                            self.log(f"Cảnh báo: Không thể khóa mẫu giọng tự động cho {spk_key} tại câu {i + 1}: {lock_err}")
                    else:
                        self.log(f"Chưa khóa mẫu giọng cho {spk_key} ở câu {i + 1} ({lock_reason}), sẽ thử ở câu tiếp theo.")

                # Kiểm tra chồng lấn với câu phụ đề kế tiếp và co giãn thời lượng trong ngưỡng an toàn
                if i + 1 < len(blocks):
                    next_start_sec = parse_timestamp_to_seconds(blocks[i + 1].start_time)
                    avail_before_next = next_start_sec - start_sec
                    cur_dur_sec = len(audio_arr) / float(self.sample_rate)
                    if avail_before_next > 0.4 and cur_dur_sec > avail_before_next * 1.03:
                        needed_speed = cur_dur_sec / avail_before_next
                        if needed_speed <= 1.35:
                            audio_arr = self._safe_time_stretch(audio_arr, needed_speed)
                            report.stretched_count += 1

                # Lưu vào bộ nhớ đệm khôi phục tác vụ nếu được bật
                if cache_wav_path:
                    try:
                        sf.write(cache_wav_path, audio_arr, self.sample_rate)
                    except Exception:
                        pass

                start_sample = int(start_sec * self.sample_rate)
                end_sample = start_sample + len(audio_arr)

                # Tự động mở rộng bộ đệm âm thanh nếu câu cuối (hoặc câu dài) vượt quá độ dài dự kiến ban đầu
                if end_sample > len(full_audio):
                    pad_extra = (end_sample - len(full_audio)) + int(0.5 * self.sample_rate)
                    full_audio = np.pad(full_audio, (0, pad_extra))

                existing = full_audio[start_sample:end_sample]
                full_audio[start_sample:end_sample] = np.clip(existing + audio_arr, -1.0, 1.0)
                report.succeeded_count += 1

            except Exception as e:
                err_str = str(e)
                report.failed_count += 1
                report.failed_lines.append(i + 1)
                cuda_mod = getattr(torch, "cuda", None) if torch is not None else None
                oom_cls = getattr(cuda_mod, "OutOfMemoryError", RuntimeError)
                if "out of memory" in err_str.lower() or (isinstance(e, oom_cls) and "cuda" in err_str.lower()):
                    report.oom_detected = True
                    gc.collect()
                    if cuda_mod is not None and cuda_mod.is_available():
                        cuda_mod.empty_cache()
                    self.log(
                        f"Lỗi thiếu bộ nhớ VRAM trên GPU tại câu {i + 1}: Đã tự động giải phóng bộ nhớ đệm. "
                        "Khuyến nghị giảm Số bước giải mã xuống 16, tắt Tăng tốc đồ họa chuyên sâu hoặc nhấn Giải phóng VRAM."
                    )
                else:
                    self.log(f"Lỗi khi tổng hợp khối phụ đề {i + 1}: {err_str}")

        self.log(f"Đang lưu track lồng tiếng hoàn chỉnh vào: {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, full_audio, self.sample_rate)

        is_valid, wav_dur, wav_rms, wav_peak = validate_audio_file(output_path)
        report.valid_audio_file = is_valid
        report.audio_duration_sec = wav_dur
        report.rms_energy = wav_rms
        report.peak_amplitude = wav_peak

        if report.succeeded_count > 0 and report.failed_count == 0 and is_valid:
            report.status = "success"
            report.message = f"Đã tổng hợp thành công trọn vẹn {report.succeeded_count}/{report.total_blocks} câu thoại."
            self.log(report.message)
        elif report.succeeded_count > 0 and report.failed_count > 0 and is_valid:
            report.status = "partial"
            report.message = (
                f"Hoàn thành một phần: thành công {report.succeeded_count} câu, "
                f"lỗi {report.failed_count} câu (dòng: {', '.join(map(str, report.failed_lines))})."
            )
            self.log(report.message)
        else:
            report.status = "failed"
            report.message = (
                f"Tổng hợp thất bại: không tạo được tệp âm thanh hợp lệ "
                f"(thành công {report.succeeded_count}, lỗi {report.failed_count} câu)."
            )
            self.log(report.message)

        self.last_report = report
        return report

    def _get_video_duration(self, video_path: str) -> float:
        """Lấy độ dài thời gian của video bằng FFmpeg."""
        try:
            ffmpeg_bin = FFmpegCLI.instance().ffmpeg_path
            cmd = [ffmpeg_bin, "-i", video_path]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
            if match:
                h, m, s = match.groups()
                return int(h) * 3600.0 + int(m) * 60.0 + float(s)
        except Exception as e:
            self.log(f"Không thể đo độ dài video bằng FFmpeg: {e}")
        return 0.0

    @staticmethod
    def _has_audio_stream(video_path: str) -> bool:
        """Kiểm tra xem tệp video đầu vào có chứa luồng âm thanh (Audio stream) hay không."""
        if not video_path or not os.path.exists(video_path):
            return False
        try:
            ffmpeg_bin = FFmpegCLI.instance().ffmpeg_path
            res = subprocess.run(
                [ffmpeg_bin, "-i", video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return bool(re.search(r"Stream\s+#\d+:\d+.*:\s+Audio:", res.stderr))
        except Exception:
            return False

    @staticmethod
    def mix_audio_ducking(
        video_in: str,
        tts_audio_in: str,
        video_out: str,
        original_volume: float = 0.15,
        ducking_mode: str = "dynamic",
    ) -> bool:
        """
        Giai đoạn 5: Hòa trộn âm thanh lồng tiếng TTS với âm thanh gốc của video bằng FFmpeg.
        - Hỗ trợ chế độ Ducking động theo đường bao lời thoại (dynamic): chỉ giảm nhạc nền khi có tiếng nói,
          giữ nguyên nhạc nền ở đoạn không thoại và chống clipping.
        - Hỗ trợ chế độ âm lượng cố định (fixed) truyền thống.
        - Nếu video gốc không có kênh âm thanh, tự động ghép trực tiếp kênh âm thanh TTS vào video.
        """
        ffmpeg_bin = FFmpegCLI.instance().ffmpeg_path
        has_bg_audio = TTSEngine._has_audio_stream(video_in)

        if not has_bg_audio:
            fallback_cmd = [
                ffmpeg_bin, "-y",
                "-i", video_in,
                "-i", tts_audio_in,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                video_out,
            ]
            subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return os.path.exists(video_out) and os.path.getsize(video_out) > 0

        if ducking_mode == "dynamic":
            temp_bg_wav = f"{video_out}.temp_bg_{uuid.uuid4().hex[:6]}.wav"
            temp_mix_wav = f"{video_out}.temp_mix_{uuid.uuid4().hex[:6]}.wav"
            try:
                tts_data, tts_sr = sf.read(tts_audio_in, dtype="float32")
                if tts_data.ndim > 1:
                    tts_data = np.mean(tts_data, axis=1)
                extract_cmd = [
                    ffmpeg_bin, "-y",
                    "-i", video_in,
                    "-vn", "-ac", "1", "-ar", str(tts_sr),
                    temp_bg_wav,
                ]
                ext_res = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if ext_res.returncode == 0 and os.path.exists(temp_bg_wav):
                    bg_data, _ = sf.read(temp_bg_wav, dtype="float32")
                    mixed_data = apply_dynamic_ducking(
                        bg_audio=bg_data,
                        tts_audio=tts_data,
                        sr=tts_sr,
                        duck_gain=original_volume,
                    )
                    sf.write(temp_mix_wav, mixed_data, tts_sr)
                    mux_cmd = [
                        ffmpeg_bin, "-y",
                        "-i", video_in,
                        "-i", temp_mix_wav,
                        "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest",
                        video_out,
                    ]
                    mux_res = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if mux_res.returncode == 0 and os.path.exists(video_out) and os.path.getsize(video_out) > 0:
                        return True
            except Exception:
                pass
            finally:
                for p in (temp_bg_wav, temp_mix_wav):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

        # Chế độ âm lượng cố định hoặc dự phòng FFmpeg filter
        cmd = [
            ffmpeg_bin, "-y",
            "-i", video_in,
            "-i", tts_audio_in,
            "-filter_complex",
            f"[0:a]volume={original_volume}[a0]; [1:a]volume=1.0[a1]; [a0][a1]amix=inputs=2:duration=first:dropout_transition=2,alimiter=limit=0.96[outa]",
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            video_out,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            fallback_cmd = [
                ffmpeg_bin, "-y",
                "-i", video_in,
                "-i", tts_audio_in,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                video_out,
            ]
            subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(video_out) and os.path.getsize(video_out) > 0

    def dub_video(
        self,
        video_path: str,
        subtitle_blocks: List[SubtitleBlock],
        output_video_path: str,
        fps: float = 30.0,
        original_volume: float = 0.15,
    ) -> bool:
        """
        Lồng tiếng hoàn chỉnh cho video chạy trực tiếp trên GPU NVIDIA với kiểm định nghiêm ngặt từng giai đoạn.
        Không báo thành công giả nếu toàn bộ câu thoại bị lỗi hoặc tệp đầu ra không hợp lệ.
        """
        video_path = get_readable_path(video_path)
        output_video_path = get_readable_path(output_video_path)

        if not os.path.exists(video_path):
            self.log(f"Tệp video nguồn không tồn tại: {video_path}")
            return False

        if not subtitle_blocks:
            self.log("Danh sách phụ đề trống, tiến hành sao chép video gốc.")
            shutil.copy(video_path, output_video_path)
            return True

        video_duration = self._get_video_duration(video_path)
        max_sub_time = 0.0
        for block in subtitle_blocks:
            end_sec = parse_timestamp_to_seconds(getattr(block, "end_time", block.start_time))
            if end_sec > max_sub_time:
                max_sub_time = end_sec

        total_duration = max(video_duration, max_sub_time)
        if total_duration <= 0.0:
            total_duration = 60.0

        temp_dir = Path(output_video_path).parent / f"temp_tts_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_audio_path = str(temp_dir / "tts_dub_track.wav")

        try:
            self.log("Bắt đầu quy trình tổng hợp giọng lồng tiếng OmniVoice trên GPU...")
            report = self.synthesize_blocks(
                subtitle_blocks,
                total_duration,
                temp_audio_path,
                video_path=video_path,
            )
            if report.status in ("failed", "cancelled") or not report.valid_audio_file:
                self.log(f"Không thể ghép video do bước tổng hợp âm thanh không đạt yêu cầu: {report.message}")
                return False

            self.log(
                f"Đang hòa trộn âm thanh vào video (chế độ: {self.ducking_mode}, mức nhạc nền khi có thoại: {original_volume})..."
            )
            mux_ok = self.mix_audio_ducking(
                video_in=video_path,
                tts_audio_in=temp_audio_path,
                video_out=output_video_path,
                original_volume=original_volume,
                ducking_mode=self.ducking_mode,
            )
            if not mux_ok or not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
                self.log("Lỗi: Tệp video xuất ra không tồn tại hoặc có dung lượng bằng 0.")
                return False

            self.log(f"Đã xuất video lồng tiếng thành công: {output_video_path}")
            return True
        except Exception as e:
            self.log(f"Lỗi trong quá trình lồng tiếng cho video: {str(e)}")
            return False
        finally:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass


# Tương thích ngược với các lệnh gọi OmniVoiceEngine
OmniVoiceEngine = TTSEngine
