# -*- coding: utf-8 -*-
"""
Mô-đun tổng hợp giọng nói tích hợp trực tiếp mã nguồn OmniVoice chạy trên GPU NVIDIA.
Hỗ trợ đầy đủ tất cả tính năng của mã nguồn gốc OmniVoice:
1. Nhân bản giọng nói: Từ tệp âm thanh mẫu, tự động trích xuất từ video nguồn, hoặc tệp mẫu .pt đã lưu.
2. Thiết kế giọng nói: Điều khiển giới tính, độ tuổi, cao độ, phong cách thì thầm, khẩu âm và phương ngữ qua thuộc tính.
3. Giọng nói tự động: Tự động sinh giọng đọc tự nhiên không cần mẫu.
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
import json
import zlib
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
    overflow_lines: List[Dict[str, Any]] = field(default_factory=list)
    timestamp_issues: List[Dict[str, Any]] = field(default_factory=list)
    inconsistent_retried_count: int = 0
    speaker_consistency_score: float = 1.0  # Thang 0.0 - 1.0 dựa trên CAM++ cosine similarity
    compressed_ratio: float = 0.0  # Tỷ lệ câu bị co nén thời lượng (0.0 - 1.0)
    fallback_count: int = 0  # Số câu phải dùng giọng hoặc mỏ neo dự phòng
    quality_score: float = 100.0  # Điểm chất lượng tổng thể 0 - 100
    audio_duration_sec: float = 0.0
    rms_energy: float = 0.0
    peak_amplitude: float = 0.0
    valid_audio_file: bool = False
    oom_detected: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi báo cáo thành từ điển dữ liệu để hiển thị hoặc lưu trữ."""
        return {
            "status": self.status,
            "output_path": self.output_path,
            "total_blocks": self.total_blocks,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "skipped_empty_count": self.skipped_empty_count,
            "truncated_count": self.truncated_count,
            "stretched_count": self.stretched_count,
            "resumed_from_cache_count": self.resumed_from_cache_count,
            "failed_lines": list(self.failed_lines),
            "truncated_lines": list(self.truncated_lines),
            "overflow_lines": list(self.overflow_lines),
            "timestamp_issues": list(self.timestamp_issues),
            "inconsistent_retried_count": self.inconsistent_retried_count,
            "speaker_consistency_score": self.speaker_consistency_score,
            "compressed_ratio": self.compressed_ratio,
            "fallback_count": self.fallback_count,
            "quality_score": self.quality_score,
            "audio_duration_sec": self.audio_duration_sec,
            "rms_energy": self.rms_energy,
            "peak_amplitude": self.peak_amplitude,
            "valid_audio_file": self.valid_audio_file,
            "oom_detected": self.oom_detected,
            "message": self.message,
        }


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


def _number_to_vietnamese_words(n: int) -> str:
    """Chuyển đổi số nguyên không âm (0 đến 999.999.999) sang chữ viết tiếng Việt chuẩn xác."""
    if n == 0:
        return "không"

    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    def read_three_digits(num: int, is_highest: bool = False) -> str:
        hundreds = num // 100
        tens = (num % 100) // 10
        units = num % 10
        res = []

        if hundreds > 0 or not is_highest:
            res.append(f"{digits[hundreds]} trăm")

        if tens > 1:
            res.append(f"{digits[tens]} mươi")
            if units == 1:
                res.append("mốt")
            elif units == 4:
                res.append("tư")
            elif units == 5:
                res.append("lăm")
            elif units > 0:
                res.append(digits[units])
        elif tens == 1:
            res.append("mười")
            if units == 5:
                res.append("lăm")
            elif units > 0:
                res.append(digits[units])
        elif tens == 0:
            if units > 0:
                if hundreds > 0 or not is_highest:
                    res.append("lẻ")
                res.append(digits[units])

        return " ".join(res)

    groups = []
    scale = ["", "nghìn", "triệu", "tỷ"]
    g_idx = 0
    temp = n
    while temp > 0:
        part = temp % 1000
        groups.append((part, g_idx))
        temp //= 1000
        g_idx += 1

    words = []
    for part, idx in reversed(groups):
        if part == 0:
            continue
        is_highest = (idx == groups[-1][1])
        r = read_three_digits(part, is_highest=is_highest)
        unit = scale[idx]
        if unit:
            words.append(f"{r} {unit}".strip())
        else:
            words.append(r)

    return " ".join(words).strip()


def normalize_vietnamese_numbers_and_text(text: str) -> str:
    """
    Chuẩn hóa số đếm, số thập phân, phần trăm sang chữ viết tiếng Việt lưu loát.
    Hỗ trợ quy tắc riêng cho số điện thoại/mã bắt đầu bằng 0, định dạng phiên bản và ngày tháng.
    Bảo vệ nguyên vẹn các thẻ biểu cảm phi ngôn từ trong ngoặc vuông được OmniVoice hỗ trợ,
    lọc bỏ các thẻ phụ đề âm thanh không thuộc danh mục hỗ trợ (ví dụ: [Music], (cười)...).
    """
    if not text or not text.strip():
        return ""

    # Chuyển đổi ký tự xuống dòng thành khoảng trắng và xóa gạch đầu dòng phân cách câu thoại
    cleaned_input = re.sub(r"\s*[\r\n]+\s*", " ", text)
    cleaned_input = re.sub(r"(^|\s)[-–—]\s+", r"\1", cleaned_input).strip()
    if not cleaned_input:
        return ""

    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    # 1. Lọc bỏ các thẻ âm thanh trong ngoặc đơn hoặc ngoặc vuông không được OmniVoice hỗ trợ
    # Chỉ bảo vệ các thẻ phi ngôn từ chuẩn của OmniVoice
    valid_tags_set = set(NON_VERBAL_TAGS)
    tags = []

    def handle_bracket_tag(match):
        raw_tag = match.group(0)
        if raw_tag in valid_tags_set:
            tags.append(raw_tag)
            return f"__TAG_{len(tags) - 1}__"
        # Thẻ lạ không được hỗ trợ -> lọc bỏ để tránh TTS đọc thành lời
        return " "

    s = re.sub(r"\[[^\]]+\]", handle_bracket_tag, cleaned_input)
    # Lọc bỏ các chú thích biểu cảm và hiệu ứng âm thanh trong ngoặc đơn thông dụng của phụ đề
    s = re.sub(r"\([^)]*(?:cười|khóc|thở dài|hét|nhạc|vỗ tay|music|applause|laughter|crying|sigh)[^)]*\)", " ", s, flags=re.IGNORECASE)

    # 2. Xử lý định dạng ngày tháng: 29/10/2026 hoặc 29-10-2026
    def repl_date(m):
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
            day_str = _number_to_vietnamese_words(day)
            month_str = _number_to_vietnamese_words(month)
            year_str = _number_to_vietnamese_words(year)
            return f"ngày {day_str} tháng {month_str} năm {year_str}"
        return m.group(0)

    s = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", repl_date, s)

    # 3. Xử lý định dạng giờ: 14h30 hoặc 14h
    def repl_hour_h(m):
        h = int(m.group(1))
        m_val = int(m.group(2)) if m.group(2) else None
        h_str = _number_to_vietnamese_words(h)
        if m_val is not None:
            m_str = _number_to_vietnamese_words(m_val)
            return f"{h_str} giờ {m_str} phút"
        return f"{h_str} giờ"

    s = re.sub(r"\b(\d{1,2})h(\d{1,2})\b", repl_hour_h, s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d{1,2})h\b", repl_hour_h, s, flags=re.IGNORECASE)

    # 4. Số điện thoại và chuỗi mã liên lạc có dấu gạch ngang hoặc dấu chấm (ví dụ: 0987.123.456, 098-123-4567)
    def repl_phone(m):
        digits_only = re.sub(r"\D", "", m.group(0))
        return " ".join(digits[int(d)] for d in digits_only)

    s = re.sub(r"\b0\d{2,3}[-.\s]\d{3,4}[-.\s]\d{3,4}\b", repl_phone, s)

    # 5. Phân tách hàng nghìn có dấu chấm trong tiếng Việt:
    # Trong tiếng Việt, dấu chấm theo sau đúng 3 chữ số là phân tách hàng nghìn (ví dụ: 2.500 người, 1.000.000 đồng).
    # Cần chuẩn hóa hàng nghìn trước khi phân tích chuỗi phiên bản để tránh biến '1.000.000' thành 'một chấm không không không...'.
    s = re.sub(r"\b\d{1,3}(\.\d{3}){2,}\b", lambda m: m.group(0).replace(".", ""), s)
    s = re.sub(r"\b\d{1,3}\.000\b", lambda m: m.group(0).replace(".", ""), s)
    s = re.sub(r"\b\d{1,3}\.\d{3}(?=\s*(?:đồng|đ|vnd|usd|k|triệu|tỷ))\b", lambda m: m.group(0).replace(".", ""), s, flags=re.IGNORECASE)
    # Hàng nghìn đơn lẻ: 2.500 người -> 2500 người (trừ trường hợp số 0 ở đầu)
    s = re.sub(r"\b([1-9]\d{0,2})\.(\d{3})\b", r"\1\2", s)

    # 6. Xử lý chuỗi phiên bản có nhiều dấu chấm (ví dụ: v1.2.3, 1.0.4)
    def repl_version(m):
        raw = m.group(0)
        prefix = "v " if raw.lower().startswith("v") else ""
        nums = re.findall(r"\d+", raw)
        word_parts = [" ".join(digits[int(char)] for char in d) if (len(d) > 1 and d.startswith("0")) else _number_to_vietnamese_words(int(d)) for d in nums]
        return prefix + " chấm ".join(word_parts)

    s = re.sub(r"\b[vV]?\d+(\.\d+){2,}\b", repl_version, s)
    s = re.sub(r"\b[vV]\d+\.\d+\b", repl_version, s)

    # 7. Phần trăm và đơn vị đo lường
    s = re.sub(r"(?<=\d)\s*%", " phần trăm", s)
    s = re.sub(r"%", " phần trăm ", s)

    units = [
        (r"\bkm/h\b", "ki lô mét trên giờ"),
        (r"\bm/s\b", "mét trên giây"),
        (r"\bkm2\b|\bkm²\b", "ki lô mét vuông"),
        (r"\bm2\b|\bm²\b", "mét vuông"),
        (r"°C\b", " độ C"),
        (r"\bVNĐ\b|\bVND\b", "đồng"),
        (r"\bUSD\b", "đô la Mỹ"),
    ]
    for pattern, repl in units:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # 8. Số thập phân: 1,5 hoặc 1.5 (sau khi đã bóc tách hàng nghìn \d{1,3}\.\d{3} ở bước 6)
    def repl_decimal(m):
        integer_part = int(m.group(1))
        decimal_part = " ".join(digits[int(d)] for d in m.group(2))
        return f"{_number_to_vietnamese_words(integer_part)} phẩy {decimal_part}"

    s = re.sub(r"\b(\d+)[.,](\d+)\b", repl_decimal, s)

    # 9. Số nguyên và chuỗi số đặc biệt (bắt đầu bằng 0 hoặc mã dài)
    def repl_integer(m):
        raw = m.group(0)
        # Chuỗi số bắt đầu bằng chữ số 0 (ví dụ: 0123, 0987...): đọc từng chữ số để giữ trọn ý nghĩa số điện thoại/mã số
        if len(raw) > 1 and raw.startswith("0"):
            return " ".join(digits[int(d)] for d in raw)
        val = int(raw)
        if val <= 999_999_999:
            return _number_to_vietnamese_words(val)
        return " ".join(digits[int(d)] for d in raw)

    s = re.sub(r"\b\d+\b", repl_integer, s)

    for idx, tag in enumerate(tags):
        s = s.replace(f"__TAG_{idx}__", tag)

    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def preprocess_vietnamese_tts_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt trước khi đưa vào OmniVoice:
    - Chuyển đổi xuống dòng thành khoảng trắng và xóa gạch đầu dòng thoại.
    - Mở rộng các đơn vị đo lường, ký hiệu phần trăm, nhiệt độ, tiền tệ phổ biến theo ngữ cảnh.
    - Phiên âm số đếm, số thập phân, số điện thoại sang chữ tiếng Việt tự nhiên.
    - Bổ sung dấu ngắt câu cuối nếu thiếu để hạ giọng trọn vẹn và không bị cụt câu.
    - Giữ nguyên các thẻ biểu cảm phi ngôn từ trong ngoặc vuông.
    """
    if not text or not text.strip():
        return ""
    cleaned_text = re.sub(r"\s*[\r\n]+\s*", " ", text)
    cleaned_text = re.sub(r"(^|\s)[-–—]\s+", r"\1", cleaned_text).strip()
    normalized = normalize_vietnamese_numbers_and_text(cleaned_text)
    if not normalized:
        return ""
    if normalized[-1] not in ".!?;:…":
        normalized += "."
    return normalized


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
    Kiểm tra chất lượng âm thanh câu đầu tiên trước khi khóa làm VoiceClonePrompt cho nhân vật
    thông qua bộ kiểm định toàn diện evaluate_reference_audio_quality theo thang điểm 100.
    """
    try:
        from src.ai_engines.ref_audio_qc import evaluate_reference_audio_quality
        report = evaluate_reference_audio_quality(audio_arr, sr=sr)
        if not report.is_acceptable:
            msg = report.rejection_reason or f"Chất lượng âm thanh không đạt chuẩn ({report.score}/100 điểm)."
            return False, msg
        return True, f"Đạt chuẩn chất lượng giọng mẫu ({report.score}/100 điểm - {report.grade})."
    except Exception:
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


# Câu thoại mẫu chuẩn tiếng Việt dùng để tạo mẫu mỏ neo khóa âm sắc nhất quán
ANCHOR_SAMPLE_VI = "Xin chào các bạn, đây là giọng đọc mẫu tiếng Việt chuẩn để đồng bộ âm sắc xuyên suốt toàn bộ video."
ANCHOR_SAMPLE_EN = "Hello everyone, this is a standard reference voice sample used to calibrate timbre, pitch, and speech rhythm."
ANCHOR_SAMPLE_ZH = "大家好，这是一段用于校准音色、音高与语速节奏的基准声音样本。"
ANCHOR_SAMPLE_JA = "皆さんこんにちは、これは声の音色やピッチ、話速を正確に補正するための基準音声サンプルです。"
ANCHOR_SAMPLE_KO = "안녕하세요 여러분, 이것은 음색과 음높이, 말하기 리듬을 일관되게 보정하기 위한 기준 음성 샘플입니다."

# Danh mục 22 giọng đọc mẫu tiếng Việt đa phong cách kèm câu thoại nghe thử
VIETNAMESE_VOICE_CATALOG: Dict[str, Dict[str, str]] = {
    "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức": {
        "slug": "nam_thanh_nien_chuan",
        "name": "Nam Thanh Niên Chuẩn",
        "gender": "male",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "thuyet_minh",
        "instruct": "male, young adult, moderate pitch",
        "sample_vi": "Xin chào các bạn, đây là giọng nam thanh niên chuẩn dùng cho thuyết minh và bản tin thời sự.",
    },
    "Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện": {
        "slug": "nu_thanh_nien_chuan",
        "name": "Nữ Thanh Niên Chuẩn",
        "gender": "female",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "ke_chuyen",
        "instruct": "female, young adult, moderate pitch",
        "sample_vi": "Xin chào mọi người, hôm nay chúng ta sẽ cùng khám phá một câu chuyện vô cùng hấp dẫn.",
    },
    "Nam Trầm Ấm - Tóm Tắt Phim Và Tài Liệu": {
        "slug": "nam_tram_am_review",
        "name": "Nam Trầm Ấm",
        "gender": "male",
        "accent": "nam",
        "age_group": "middle_aged",
        "pitch": "low",
        "style": "ke_chuyen",
        "instruct": "male, middle-aged, low pitch",
        "sample_vi": "Đằng sau cánh cửa bí ẩn kia là một sự thật mà chưa ai từng dám tiết lộ.",
    },
    "Nam Rất Trầm Uy Lực - Phim Hành Động Và Kiếm Hiệp": {
        "slug": "nam_rat_tram_uy_luc",
        "name": "Nam Rất Trầm Uy Lực",
        "gender": "male",
        "accent": "bac",
        "age_group": "middle_aged",
        "pitch": "very_low",
        "style": "ke_chuyen",
        "instruct": "male, middle-aged, very low pitch",
        "sample_vi": "Giữa chốn giang hồ hiểm ác này, chỉ có bản lĩnh mới quyết định được thắng bại.",
    },
    "Nữ Trầm Truyền Cảm - Đọc Truyện Đêm Khuya": {
        "slug": "nu_tram_truyen_cam",
        "name": "Nữ Trầm Truyền Cảm",
        "gender": "female",
        "accent": "trung",
        "age_group": "middle_aged",
        "pitch": "low",
        "style": "ke_chuyen",
        "instruct": "female, middle-aged, low pitch",
        "sample_vi": "Đêm đã khuya, mời quý vị cùng lắng lòng nghe lại những dòng hồi ức sâu lắng nhất.",
    },
    "Nữ Trẻ Năng Động - Video Ngắn Và Giải Trí": {
        "slug": "nu_tre_nang_dong",
        "name": "Nữ Trẻ Năng Động",
        "gender": "female",
        "accent": "nam",
        "age_group": "teenager",
        "pitch": "high",
        "style": "soi_noi",
        "instruct": "female, teenager, high pitch",
        "sample_vi": "Mọi người ơi, hôm nay mình vừa phát hiện ra một bí quyết cực kỳ thú vị luôn nhé!",
    },
    "Nam Trẻ Sôi Nổi - Bình Luận Và Sáng Tạo Nội Dung": {
        "slug": "nam_tre_soi_noi",
        "name": "Nam Trẻ Sôi Nổi",
        "gender": "male",
        "accent": "bac",
        "age_group": "teenager",
        "pitch": "high",
        "style": "soi_noi",
        "instruct": "male, teenager, high pitch",
        "sample_vi": "Chào mừng anh em đã quay trở lại với video đặc biệt của chúng ta ngày hôm nay!",
    },
    "Nữ Ngọt Ngào Trong Trẻo - Tâm Sự Và Tình Cảm": {
        "slug": "nu_ngot_ngao_trong_treo",
        "name": "Nữ Ngọt Ngào Trong Trẻo",
        "gender": "female",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "high",
        "style": "tam_su",
        "instruct": "female, young adult, high pitch",
        "sample_vi": "Có những ký ức dù năm tháng trôi qua bao lâu vẫn luôn vẹn nguyên trong trái tim.",
    },
    "Nam Cao Sáng Rõ Ràng - Dẫn Chương Trình Sự Kiện": {
        "slug": "nam_cao_sang_mc",
        "name": "Nam Cao Sáng Rõ Ràng",
        "gender": "male",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "high",
        "style": "thuyet_minh",
        "instruct": "male, young adult, high pitch",
        "sample_vi": "Xin nồng nhiệt chào đón toàn thể quý vị khán giả đã có mặt trong chương trình tối nay.",
    },
    "Nam Trung Niên Điềm Đạm - Sách Nói Và Triết Lý": {
        "slug": "nam_trung_nien_diem_dam",
        "name": "Nam Trung Niên Điềm Đạm",
        "gender": "male",
        "accent": "trung",
        "age_group": "middle_aged",
        "pitch": "moderate",
        "style": "ke_chuyen",
        "instruct": "male, middle-aged, moderate pitch",
        "sample_vi": "Thành công không phải là đích đến cuối cùng, mà là hành trình vượt qua từng thử thách.",
    },
    "Nữ Trung Niên Chững Chạc - Giảng Giải Và Giáo Dục": {
        "slug": "nu_trung_nien_chung_chac",
        "name": "Nữ Trung Niên Chững Chạc",
        "gender": "female",
        "accent": "bac",
        "age_group": "middle_aged",
        "pitch": "moderate",
        "style": "thuyet_minh",
        "instruct": "female, middle-aged, moderate pitch",
        "sample_vi": "Trong nội dung bài học hôm nay, chúng ta sẽ cùng phân tích những nguyên lý cốt lõi nhất.",
    },
    "Ông Lão Trầm Mặc - Cổ Tích Và Lịch Sử": {
        "slug": "ong_lao_tram_mac",
        "name": "Ông Lão Trầm Mặc",
        "gender": "male",
        "accent": "nam",
        "age_group": "elderly",
        "pitch": "very_low",
        "style": "ke_chuyen",
        "instruct": "male, elderly, very low pitch",
        "sample_vi": "Ngày xửa ngày xưa, ở vùng đất phương nam có một truyền thuyết được lưu truyền qua nhiều thế hệ.",
    },
    "Bà Lão Hiền Từ - Kể Chuyện Dân Gian": {
        "slug": "ba_lao_hien_tu",
        "name": "Bà Lão Hiền Từ",
        "gender": "female",
        "accent": "bac",
        "age_group": "elderly",
        "pitch": "low",
        "style": "ke_chuyen",
        "instruct": "female, elderly, low pitch",
        "sample_vi": "Các cháu lại đây ngồi bên bếp lửa ấm, bà sẽ kể cho nghe chuyện cổ tích ngày xưa.",
    },
    "Thiếu Nhi Hồn Nhiên - Phim Hoạt Hình": {
        "slug": "thieu_nhi_hon_nhien",
        "name": "Thiếu Nhi Hồn Nhiên",
        "gender": "female",
        "accent": "bac",
        "age_group": "child",
        "pitch": "high",
        "style": "hoat_hinh",
        "instruct": "child, high pitch",
        "sample_vi": "Mẹ ơi, hôm nay ở lớp con được cô giáo khen vì giúp đỡ bạn bè đấy ạ!",
    },
    "Em Bé Nhí Nhảnh - Giọng Cao Đáng Yêu": {
        "slug": "em_be_nhi_nhanh",
        "name": "Em Bé Nhí Nhảnh",
        "gender": "male",
        "accent": "nam",
        "age_group": "child",
        "pitch": "very_high",
        "style": "hoat_hinh",
        "instruct": "child, very high pitch",
        "sample_vi": "[surprise-wa] Chú cún con này biết bắt tay dễ thương quá đi mất thôi!",
    },
    "Nam Thì Thầm Bí Ẩn - Phim Kinh Dị Và Trinh Thám": {
        "slug": "nam_thi_tham_bi_an",
        "name": "Nam Thì Thầm Bí Ẩn",
        "gender": "male",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "low",
        "style": "thi_tham",
        "instruct": "male, young adult, low pitch, whisper",
        "sample_vi": "Suỵt, đừng phát ra tiếng động nào cả, hình như có tiếng bước chân ngoài hành lang.",
    },
    "Nữ Thì Thầm Nhẹ Nhàng - Thư Giãn Và Ru Ngủ": {
        "slug": "nu_thi_tham_nhe_nhang",
        "name": "Nữ Thì Thầm Nhẹ Nhàng",
        "gender": "female",
        "accent": "nam",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "thi_tham",
        "instruct": "female, young adult, moderate pitch, whisper",
        "sample_vi": "Hãy nhắm mắt lại, thả lỏng tâm trí và tận hưởng bầu không khí tĩnh lặng đêm nay.",
    },
    "Nam Vui Vẻ Tự Nhiên - Kèm Tiếng Cười Biểu Cảm": {
        "slug": "nam_vui_ve_bieu_cam",
        "name": "Nam Vui Vẻ Tự Nhiên",
        "gender": "male",
        "accent": "nam",
        "age_group": "young_adult",
        "pitch": "high",
        "style": "soi_noi",
        "instruct": "male, young adult, high pitch",
        "sample_vi": "[laughter] Thật sự không thể ngờ tình huống bất ngờ đó lại diễn ra ngay trước mắt chúng tôi!",
    },
    "Nữ Xúc Động Sâu Lắng - Kèm Tiếng Thở Dài Biểu Cảm": {
        "slug": "nu_xuc_dong_bieu_cam",
        "name": "Nữ Xúc Động Sâu Lắng",
        "gender": "female",
        "accent": "trung",
        "age_group": "young_adult",
        "pitch": "low",
        "style": "tam_su",
        "instruct": "female, young adult, low pitch",
        "sample_vi": "[sigh] Giá như ngày hôm ấy chúng ta kịp nói với nhau một lời chân thành.",
    },
    "Nam Phong Cách Âu Mỹ Đọc Tiếng Việt": {
        "slug": "nam_phong_cach_au_my",
        "name": "Nam Phong Cách Âu Mỹ",
        "gender": "male",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "thuyet_minh",
        "instruct": "male, young adult, moderate pitch, american accent",
        "sample_vi": "Xin chào Việt Nam, tôi rất ấn tượng với cảnh đẹp và ẩm thực tuyệt vời của đất nước các bạn.",
    },
    "Nữ Phong Cách Anh Quốc Đọc Tiếng Việt": {
        "slug": "nu_phong_cach_anh_quoc",
        "name": "Nữ Phong Cách Anh Quốc",
        "gender": "female",
        "accent": "bac",
        "age_group": "young_adult",
        "pitch": "moderate",
        "style": "thuyet_minh",
        "instruct": "female, young adult, moderate pitch, british accent",
        "sample_vi": "Chào buổi sáng, hôm nay thời tiết thật dễ chịu để cùng nhau dạo quanh phố cổ.",
    },
    "Nữ Phong Cách Đông Á Đọc Tiếng Việt": {
        "slug": "nu_phong_cach_dong_a",
        "name": "Nữ Phong Cách Đông Á",
        "gender": "female",
        "accent": "nam",
        "age_group": "teenager",
        "pitch": "high",
        "style": "soi_noi",
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
        max_speakers: Optional[int] = 4,
        speaker_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        resume_cache_dir: Optional[str] = None,
        ducking_mode: str = "dynamic",
        speaker_model_path: Optional[str] = None,
        allow_mock: bool = False,
        speaker_count_mode: str = "auto",
        exact_speakers: Optional[int] = None,
        clustering_sensitivity: str = "standard",
        enable_ai_vocal_separation: bool = True,
        overflow_strategy: str = "preserve_speech",
        preset_name: Optional[str] = None,
        **kwargs,
    ):
        self.model: Optional[OmniVoice] = None
        self.parent_logger = parent_logger
        self.model_name_or_path = model_name_or_path
        self.mode = mode if mode in ("clone", "design", "auto") else "clone"
        self.preset_name = preset_name or kwargs.get("preset_name")
        self.language = language if (language and language not in ("auto", "Auto")) else None
        self.speed = max(0.5, min(2.0, float(speed))) if speed is not None else 1.0
        self.sync_to_subtitle_duration = bool(sync_to_subtitle_duration)
        self.ref_audio_path = ref_audio_path if (ref_audio_path and str(ref_audio_path).strip()) else None
        self.ref_text = ref_text if (ref_text and str(ref_text).strip()) else None
        self.auto_extract_ref = bool(auto_extract_ref)
        if "ai_vocal_separation" in kwargs:
            enable_ai_vocal_separation = kwargs["ai_vocal_separation"]
        self.enable_ai_vocal_separation = bool(enable_ai_vocal_separation)
        self.instruct = instruct if (instruct and str(instruct).strip()) else None

        # Tham số OmniVoiceGenerationConfig đầy đủ
        try:
            self.num_step = max(4, min(64, int(num_step))) if num_step is not None else 32
        except (TypeError, ValueError):
            self.num_step = 32
        self.guidance_scale = max(0.0, min(4.0, float(guidance_scale))) if guidance_scale is not None else 2.0
        self.t_shift = float(t_shift) if t_shift is not None else 0.1
        self.layer_penalty_factor = float(layer_penalty_factor) if layer_penalty_factor is not None else 5.0
        self.position_temperature = float(position_temperature) if position_temperature is not None else 5.0
        self.class_temperature = float(class_temperature) if class_temperature is not None else 0.0
        self.audio_chunk_duration = max(3.0, float(audio_chunk_duration)) if audio_chunk_duration is not None else 15.0
        self.audio_chunk_threshold = max(5.0, float(audio_chunk_threshold)) if audio_chunk_threshold is not None else 30.0
        self.pad_duration = max(0.0, float(pad_duration)) if pad_duration is not None else 0.1
        self.fade_duration = max(0.0, float(fade_duration)) if fade_duration is not None else 0.1
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
        try:
            self.max_speakers = max(1, min(16, int(max_speakers))) if max_speakers is not None else None
        except (TypeError, ValueError):
            self.max_speakers = None
        self.speaker_profiles: Dict[str, Dict[str, Any]] = speaker_profiles or {}
        if speaker_model_path:
            self.speaker_model_path = str(speaker_model_path).strip()
        else:
            cfg_item = getattr(app_config, "mmoSpeakerModelPath", None)
            self.speaker_model_path = str(cfg_item.value).strip() if (cfg_item and hasattr(cfg_item, "value")) else ""
        self._speaker_prompt_cache: Dict[str, VoiceClonePrompt] = {}
        self._speaker_prompt_wav_cache: Dict[str, str] = {}
        self._speaker_instruct_cache: Dict[str, str] = {}
        self._speaker_anchor_files: Dict[str, Set[str]] = {}
        self.resume_cache_dir = str(resume_cache_dir).strip() if resume_cache_dir else None
        self.ducking_mode = ducking_mode if ducking_mode in ("dynamic", "fixed") else "dynamic"
        self.allow_mock = bool(allow_mock)
        self.speaker_count_mode = str(speaker_count_mode or "auto")
        try:
            self.exact_speakers = max(1, min(16, int(exact_speakers))) if exact_speakers is not None else None
        except (TypeError, ValueError):
            self.exact_speakers = None
        self.clustering_sensitivity = str(clustering_sensitivity or "standard")
        self.overflow_strategy = "strict_sync" if str(overflow_strategy).strip() == "strict_sync" else "preserve_speech"
        self._cancel_requested: bool = False
        self.last_report: Optional[DubbingReport] = None
        self.last_diarization_error: str = ""

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
        Xóa mẫu giọng tự động khóa và bộ nhớ đệm prompt của một nhân vật hoặc toàn bộ nhân vật
        để tạo lại mẫu giọng mới sạch hơn ở lần tổng hợp tiếp theo.
        """
        prefixes = ("character_", "char_", "anchor_", "unified_", "spk_", "line_", "consistent_")
        if speaker_id:
            keys_to_remove = [
                k for k in list(self._speaker_prompt_cache.keys())
                if any(k.startswith(f"{pfx}{speaker_id}") for pfx in prefixes) or k == f"character_{speaker_id}"
            ]
            for k in keys_to_remove:
                self._speaker_prompt_cache.pop(k, None)
                if hasattr(self, "_speaker_prompt_wav_cache"):
                    self._speaker_prompt_wav_cache.pop(k, None)

            # Xóa các tệp đĩa mỏ neo của nhân vật cụ thể nếu đã lưu
            anchor_files_to_remove = set()
            if hasattr(self, "_speaker_anchor_files"):
                for a_key in list(self._speaker_anchor_files.keys()):
                    if speaker_id in a_key or any(f"{pfx}{speaker_id}" in a_key for pfx in prefixes):
                        anchor_files_to_remove.update(self._speaker_anchor_files.pop(a_key, set()))
            for f_path in anchor_files_to_remove:
                try:
                    if os.path.exists(f_path):
                        os.remove(f_path)
                except Exception:
                    pass

            prof = self.speaker_profiles.get(speaker_id)
            if prof:
                if isinstance(prof, dict):
                    if any(marker in str(prof.get("ref_audio_path", "")) for marker in ("locked_", "anchor_", "temp_")):
                        prof["ref_audio_path"] = ""
                else:
                    if any(marker in str(getattr(prof, "source_audio_path", "")) for marker in ("locked_", "anchor_", "temp_")):
                        prof.source_audio_path = ""
            self.log(f"Đã đặt lại mẫu giọng khóa tự động của {speaker_id}.")
        else:
            self._speaker_prompt_cache.clear()
            self._speaker_instruct_cache.clear()
            if hasattr(self, "_speaker_prompt_wav_cache"):
                self._speaker_prompt_wav_cache.clear()
            if hasattr(self, "_speaker_anchor_files"):
                self._speaker_anchor_files.clear()
            # Dọn dẹp các tệp mẫu mỏ neo tạm thời trên đĩa nếu tồn tại
            cache_dir = getattr(self, "voice_prompt_cache_dir", None)
            if cache_dir and os.path.exists(cache_dir):
                try:
                    for f_name in os.listdir(cache_dir):
                        if any(f_name.startswith(pfx) for pfx in ("anchor_", "perline_voc_", "diarize_", "raw24k_", "ref_voice_")):
                            try:
                                os.remove(os.path.join(cache_dir, f_name))
                            except Exception:
                                pass
                except Exception:
                    pass
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
        if target_lang not in LANG_IDS and target_lang not in LANG_NAMES and target_lang != "Auto":
            target_lang = "vi"

        resume_cache_dir = getattr(app_config.mmoTTSResumeCacheDir, "value", "") if hasattr(app_config, "mmoTTSResumeCacheDir") else ""
        speaker_model_path = getattr(app_config.mmoSpeakerModelPath, "value", "") if hasattr(app_config, "mmoSpeakerModelPath") else ""
        enable_ai_vocal_sep = (
            getattr(app_config.mmoTTSEnableAiVocalSeparation, "value", True)
            if hasattr(app_config, "mmoTTSEnableAiVocalSeparation")
            else (getattr(app_config.mmoTTSAIVocalSeparation, "value", True) if hasattr(app_config, "mmoTTSAIVocalSeparation") else True)
        )
        overflow_strategy = getattr(app_config.mmoTTSOverflowStrategy, "value", "extend") if hasattr(app_config, "mmoTTSOverflowStrategy") else "extend"

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
            resume_cache_dir=resume_cache_dir,
            speaker_model_path=speaker_model_path,
            enable_ai_vocal_separation=enable_ai_vocal_sep,
            overflow_strategy=overflow_strategy,
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
                low_cpu_mem_usage=True,
                load_asr=self.load_asr,
                asr_model_name=self.asr_model_name,
            )
            if self.model.sampling_rate:
                self.sample_rate = int(self.model.sampling_rate)

            # Kích hoạt tăng tốc FlashInfer + CUDA Graph nếu được bật trên GPU NVIDIA
            if self.enable_flashinfer and torch.cuda.is_available():
                try:
                    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
                    try:
                        apply_flashinfer(self.model, enable_cuda_graph=True, cuda_graph_buckets=[4, 8, 12, 16, 20, 28, 36, 48])
                    except TypeError:
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

    def _transcribe_ref_audio_fallback(self, ref_audio_path: str, use_global: bool = False) -> str:
        """
        Tự động nhận diện văn bản của tệp âm thanh giọng mẫu nếu người dùng không nhập ref_text.
        Ưu tiên dùng faster-whisper nội bộ của Sakai Studio để tiết kiệm VRAM trước khi dùng ASR của OmniVoice.
        """
        if use_global and self.ref_text and self.ref_text.strip():
            return self.ref_text.strip()

        try:
            from src.ai_engines.whisper_fallback import run_whisper_segments
            segs = run_whisper_segments(ref_audio_path, model_size="small", language=None)
            if segs:
                transcript = " ".join(t.strip() for _, _, t in segs if t and t.strip()).strip()
                if transcript:
                    self.log(f"Nội dung giọng mẫu nhận diện được từ Whisper: {transcript}")
                    return transcript
        except Exception as e:
            logger.debug(f"Whisper fallback không khả dụng cho giọng mẫu: {e}")

        # Nếu Whisper fallback không trả về kết quả, để OmniVoice tự xử lý hoặc dùng chuỗi mặc định
        return ""

    def set_reference_voice(self, ref_audio_path: str, ref_text: Optional[str] = None, force: bool = False) -> bool:
        """
        Đặt giọng mẫu từ tệp âm thanh (.wav, .mp3, .m4a) hoặc tệp mẫu đã lưu (.pt).
        Tạo VoiceClonePrompt và lưu vào bộ nhớ đệm để tái sử dụng cho toàn bộ video.
        """
        if not ref_audio_path or not os.path.exists(ref_audio_path) or os.path.getsize(ref_audio_path) <= 0:
            self.log(f"Cảnh báo: Tệp giọng mẫu không tồn tại hoặc bị rỗng: {ref_audio_path}")
            return False

        # Nếu tệp đầu vào là mẫu .pt đã lưu từ trước
        if ref_audio_path.lower().endswith(".pt"):
            return self.load_voice_prompt(ref_audio_path)

        self.ref_audio_path = ref_audio_path
        if ref_text is not None:
            self.ref_text = ref_text

        # Tự động chuẩn hóa LUFS, khử nhiễu quang phổ và chấm điểm chất lượng âm thanh mẫu
        effective_audio_path = ref_audio_path
        try:
            from src.ai_engines.ref_audio_qc import normalize_reference_audio, evaluate_reference_audio_quality
            import soundfile as sf
            raw_audio, in_sr = sf.read(ref_audio_path, dtype="float32")
            qc_report = evaluate_reference_audio_quality(raw_audio, sr=in_sr)
            self.last_qc_report = qc_report
            self.log(f"Đánh giá chất lượng âm thanh mẫu: {qc_report.score}/100 điểm ({qc_report.grade}).")
            if not qc_report.is_acceptable:
                if not force:
                    self.log(f"Từ chối tệp âm thanh mẫu: {qc_report.rejection_reason} (Điểm chất lượng: {qc_report.score}/100). Vui lòng chọn mẫu âm thanh khác rõ ràng hơn.")
                    return False
                self.log(f"Cảnh báo: Âm thanh mẫu không đạt chuẩn nhưng được cưỡng bức sử dụng: {qc_report.rejection_reason}")
            norm_audio = normalize_reference_audio(raw_audio, source_sr=in_sr, target_sr=24000, target_lufs=-16.0)
            if len(norm_audio) > 0:
                base_stem, _ = os.path.splitext(ref_audio_path)
                norm_path = f"{base_stem}_norm24k.wav"
                sf.write(norm_path, norm_audio, 24000)
                effective_audio_path = norm_path
        except Exception as exc:
            self.log(f"Lỗi khi phân tích và kiểm định chất lượng âm thanh mẫu: {exc}")
            if not force:
                self.log("Từ chối nạp tệp âm thanh mẫu do không thể kiểm định chất lượng hoặc định dạng không hợp lệ.")
                return False
            self.log("Tiếp tục nạp tệp âm thanh mẫu theo cờ cưỡng bức.")

        self.load_model()
        self.log(f"Đang mã hóa đặc trưng giọng mẫu từ tệp: {os.path.basename(ref_audio_path)}...")

        try:
            resolved_ref_text = self.ref_text
            if not resolved_ref_text:
                resolved_ref_text = self._transcribe_ref_audio_fallback(ref_audio_path, use_global=True)

            self.current_prompt = self.model.create_voice_clone_prompt(
                ref_audio=effective_audio_path,
                ref_text=resolved_ref_text if resolved_ref_text else None,
                preprocess_prompt=self.preprocess_prompt,
            )
            self.log("Đã tạo mẫu nhân bản giọng nói trên GPU thành công.")
            return True
        except Exception as e:
            self.log(f"Lỗi khi tạo mẫu nhân bản giọng nói: {str(e)}")
            return False

    def extract_reference_from_video(
        self,
        video_path: str,
        enable_ai_vocal_separation: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Trích xuất giọng mẫu tự động từ video đầu vào bằng VoiceSeparator.
        Tự động chọn đoạn 3-10 giây đầu tiên có giọng nói rõ ràng nhất.
        """
        if not os.path.exists(video_path):
            self.log(f"Tệp video không tồn tại: {video_path}")
            return None

        self.log(f"Đang tách giọng nói gốc từ video để làm mẫu nhân bản: {os.path.basename(video_path)}")
        use_ai_sep = (
            self.enable_ai_vocal_separation
            if enable_ai_vocal_separation is None
            else bool(enable_ai_vocal_separation)
        )
        separator = VoiceSeparator(backend="ai" if use_ai_sep else "scipy")

        temp_audio_path = os.path.join(
            self.voice_prompt_cache_dir,
            f"temp_vocals_{uuid.uuid4().hex[:8]}.wav"
        )
        success = False
        if use_ai_sep:
            try:
                success = separator.separate_from_video(video_path, temp_audio_path, sample_rate=24000)
            except Exception:
                success = False

        if not success or not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            # Trích xuất âm thanh gốc chất lượng cao 24 kHz trực tiếp không qua nén dải tần
            success = separator._extract_audio(video_path, temp_audio_path, sample_rate=24000)

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
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
            self.log(f"Tệp mẫu giọng nói không tồn tại hoặc bị rỗng: {path}")
            return False

        try:
            self.current_prompt = VoiceClonePrompt.load(path)
            self.log(f"Đã nạp mẫu giọng nói (.pt) thành công từ: {os.path.basename(path)}")
            return True
        except Exception as e:
            self.log(f"Lỗi khi nạp mẫu giọng nói từ tệp: {str(e)}")
            return False

    def save_speaker_profiles(self, path: str) -> bool:
        """Lưu toàn bộ hồ sơ phân vai nhân vật và giọng đọc ra tệp JSON dự án."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            serializable_profiles: Dict[str, Any] = {}
            for spk_name, prof in self.speaker_profiles.items():
                if isinstance(prof, dict):
                    p_dict = dict(prof)
                else:
                    p_dict = {
                        "name": getattr(prof, "name", spk_name),
                        "method": getattr(prof, "method", "clone"),
                        "preset_name": getattr(prof, "preset_name", ""),
                        "voice_instruct": getattr(prof, "voice_instruct", ""),
                        "source_audio_path": getattr(prof, "source_audio_path", ""),
                        "ref_text": getattr(prof, "ref_text", ""),
                        "user_locked": getattr(prof, "user_locked", False),
                        "description": getattr(prof, "description", ""),
                    }
                serializable_profiles[spk_name] = p_dict

            payload = {
                "version": 1,
                "multi_speaker_mode": self.multi_speaker_mode,
                "speaker_profiles": serializable_profiles,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.log(f"Đã lưu hồ sơ nhân vật ({len(serializable_profiles)} nhân vật) thành công vào: {path}")
            return True
        except Exception as e:
            self.log(f"Lỗi khi lưu hồ sơ nhân vật ra tệp: {str(e)}")
            return False

    def load_speaker_profiles(self, path: str, merge: bool = True) -> Dict[str, Any]:
        """Tải hồ sơ phân vai nhân vật từ tệp JSON dự án."""
        if not os.path.exists(path):
            self.log(f"Không tìm thấy tệp hồ sơ nhân vật: {path}")
            return self.speaker_profiles

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded_profiles = payload.get("speaker_profiles", payload) if isinstance(payload, dict) else {}
            if not isinstance(loaded_profiles, dict):
                return self.speaker_profiles

            if merge and self.speaker_profiles:
                merged = dict(self.speaker_profiles)
                for k, v in loaded_profiles.items():
                    cur = merged.get(k)
                    if cur and (cur.get("user_locked") if isinstance(cur, dict) else getattr(cur, "user_locked", False)):
                        continue
                    merged[k] = v
                self.speaker_profiles = merged
            else:
                self.speaker_profiles = loaded_profiles

            self.log(f"Đã nạp thành công {len(self.speaker_profiles)} hồ sơ nhân vật từ: {os.path.basename(path)}")
            return self.speaker_profiles
        except Exception as e:
            self.log(f"Lỗi khi nạp hồ sơ nhân vật từ tệp: {str(e)}")
            return self.speaker_profiles

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
                resolved_text = self._transcribe_ref_audio_fallback(audio_source, use_global=False)

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

    def _get_anchor_sample_text(self, preset_name: Optional[str] = None) -> str:
        """Lấy câu thoại mẫu chuẩn hóa phù hợp với ngôn ngữ và preset được chọn."""
        if self.language in ("vi", None, ""):
            if preset_name and preset_name in VIETNAMESE_VOICE_CATALOG:
                s_vi = VIETNAMESE_VOICE_CATALOG[preset_name].get("sample_vi", "")
                if len(s_vi.split()) >= 12:
                    return s_vi
            return ANCHOR_SAMPLE_VI
        elif self.language in ("en", "english"):
            return ANCHOR_SAMPLE_EN
        elif self.language in ("zh", "chinese"):
            return ANCHOR_SAMPLE_ZH
        elif self.language in ("ja", "japanese"):
            return ANCHOR_SAMPLE_JA
        elif self.language in ("ko", "korean"):
            return ANCHOR_SAMPLE_KO
        return ANCHOR_SAMPLE_VI

    def _build_anchor_prompt(
        self,
        key: str,
        instruct: Optional[str] = None,
        sample_text: Optional[str] = None,
        tries: int = 6,
        seed_offset: int = 0,
    ) -> Tuple[Optional[VoiceClonePrompt], Optional[str]]:
        """
        Tạo hoặc nạp mẫu giọng mỏ neo (Anchor Voice Prompt) bằng Voice Design với nhiều hạt số ngẫu nhiên,
        chọn mẫu sạch nhất đạt chuẩn chất lượng âm thanh rồi tạo VoiceClonePrompt để dùng lại cho mọi câu.
        Bảo đảm âm sắc, cao độ và ngữ điệu đồng nhất 100% giữa tất cả các câu trong tệp phụ đề.
        """
        target_ins = (instruct or self.instruct or DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]).strip()
        target_sample = sample_text or self._get_anchor_sample_text()

        tag_content = f"{key}|{target_ins}|{self.model_name_or_path}|{self.language}"
        if seed_offset != 0:
            tag_content += f"|seed_{seed_offset}"
        tag = hashlib.sha1(tag_content.encode("utf-8")).hexdigest()[:10]

        anchor_dir = getattr(self, "voice_prompt_cache_dir", None) or os.path.join(BASE_DIR, "output", "voice_prompts")
        os.makedirs(anchor_dir, exist_ok=True)
        wav_path = os.path.join(anchor_dir, f"anchor_{tag}.wav")
        pt_path = os.path.join(anchor_dir, f"anchor_{tag}.pt")
        ck = f"anchor_{tag}"

        if not hasattr(self, "_speaker_anchor_files"):
            self._speaker_anchor_files = {}
        self._speaker_anchor_files.setdefault(key, set()).update([wav_path, pt_path])

        if ck in self._speaker_prompt_cache:
            return self._speaker_prompt_cache[ck], wav_path

        try:
            if os.path.exists(pt_path) and os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                if hasattr(VoiceClonePrompt, "load"):
                    p = VoiceClonePrompt.load(pt_path)
                    if p is not None:
                        self._speaker_prompt_cache[ck] = p
                        return p, wav_path
        except Exception:
            pass

        self.load_model()
        if self.model is None:
            return None, None

        best, best_score = None, -1.0
        best_fallback, best_fallback_score = None, -1.0
        base = (zlib.crc32(tag.encode("utf-8")) + seed_offset) % 100000

        lowered_ins = target_ins.lower()
        is_female_target = any(w in lowered_ins for w in ("female", "nữ", "woman", "girl"))
        is_child_target = any(w in lowered_ins for w in ("child", "trẻ em", "kid", "boy"))
        is_male_target = any(w in lowered_ins for w in ("male", "nam", "man")) and not is_female_target
        has_low_pitch = "low pitch" in lowered_ins or "very low pitch" in lowered_ins
        has_high_pitch = "high pitch" in lowered_ins or "very high pitch" in lowered_ins

        for t in range(tries):
            seed = base + t * 7919
            if torch is not None:
                try:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass

            try:
                gen_config = self.build_generation_config()
                res = self.model.generate(
                    text=target_sample,
                    language=self.language or "vi",
                    instruct=target_ins,
                    generation_config=gen_config,
                    normalize_text=False,
                )
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    first = res[0]
                    arr = first.detach().cpu().numpy() if hasattr(first, "detach") else np.asarray(first)
                else:
                    arr = np.asarray(res, dtype=np.float32)
                arr = np.squeeze(arr).astype(np.float32)
                ok, _ = is_audio_quality_acceptable_for_voice_lock(arr, self.sample_rate, min_duration_sec=2.5)
                if not ok:
                    continue
                score = float(np.sqrt(np.mean(arr ** 2))) * min(len(arr) / float(self.sample_rate) / 5.0, 1.0)
                if score > best_fallback_score:
                    best_fallback, best_fallback_score = arr, score

                # Kiểm tra F0 đối chiếu với giới tính yêu cầu trong instruct để chống lẫn giọng
                c_feat = self._extract_acoustic_features(arr, self.sample_rate)
                if c_feat is not None:
                    sample_f0 = float(c_feat[0] * 200.0)
                    if is_female_target:
                        min_female_f0 = 140.0 if has_low_pitch else 155.0
                        if sample_f0 < min_female_f0:
                            continue
                        elif sample_f0 >= 180.0:
                            score *= 1.30
                    elif is_child_target:
                        if sample_f0 < 210.0:
                            continue
                        elif sample_f0 >= 235.0:
                            score *= 1.40
                    elif is_male_target:
                        max_male_f0 = 195.0 if has_high_pitch else 185.0
                        if sample_f0 > max_male_f0:
                            continue
                        elif sample_f0 <= 165.0:
                            score *= 1.25
                if score > best_score:
                    best, best_score = arr, score
            except Exception as e:
                self.log(f"Lỗi khi thử tạo mẫu mỏ neo lần {t + 1}: {e}")

        if best is None:
            if best_fallback is not None:
                self.log(f"Cảnh báo: Mẫu mỏ neo cho {key} sử dụng bản dự phòng chất lượng tốt nhất và đánh dấu cần kiểm tra.")
                best = best_fallback
                spk_clean = key.replace("char_", "")
                if spk_clean in self.speaker_profiles:
                    sp_p = self.speaker_profiles[spk_clean]
                    if isinstance(sp_p, dict):
                        sp_p["status"] = "review_needed"
                    elif hasattr(sp_p, "status"):
                        sp_p.status = "review_needed"
            else:
                self.log(f"Không thể tạo được mẫu mỏ neo đạt chuẩn chất lượng cho {key}.")
                return None, None

        sf.write(wav_path, best, self.sample_rate)
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=wav_path,
            ref_text=target_sample,
            preprocess_prompt=False,
        )
        if prompt is not None:
            if hasattr(prompt, "save"):
                try:
                    prompt.save(pt_path)
                except Exception:
                    pass
            self._speaker_prompt_cache[ck] = prompt
            return prompt, wav_path

        return None, None

    def _get_or_create_preset_anchor_prompt(
        self,
        instruct: Optional[str] = None,
        preset_name: Optional[str] = None,
    ) -> Tuple[Optional[VoiceClonePrompt], Optional[str]]:
        """Hàm tương thích ngược gọi _build_anchor_prompt."""
        resolved_preset = preset_name or getattr(self, "preset_name", "unified")
        target_ins = instruct or self.instruct
        sample_text = self._get_anchor_sample_text(resolved_preset)
        return self._build_anchor_prompt(
            key=f"preset_{resolved_preset}",
            instruct=target_ins,
            sample_text=sample_text,
        )

    def _get_or_create_fallback_anchor(self) -> Tuple[Optional[VoiceClonePrompt], Optional[str]]:
        """Lấy hoặc tạo mẫu giọng mỏ neo dự phòng khi câu thoại bị khuyết nhân vật."""
        ck = "fallback_anchor_global"
        if ck in self._speaker_prompt_cache:
            return self._speaker_prompt_cache[ck], getattr(self, "_fallback_anchor_wav_path", None)

        fb_instruct = ""
        if self.speaker_profiles:
            for p in self.speaker_profiles.values():
                ins = p.get("voice_instruct") or p.get("instruct") if isinstance(p, dict) else getattr(p, "voice_instruct", "")
                if ins:
                    fb_instruct = ins
                    break
        if not fb_instruct:
            fb_instruct = self.instruct or DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]

        sample_text = self._get_anchor_sample_text()
        prompt, wav_path = self._build_anchor_prompt(
            key="fallback_anchor_global",
            instruct=fb_instruct,
            sample_text=sample_text,
        )
        self._fallback_anchor_wav_path = wav_path
        if prompt is not None:
            self._speaker_prompt_cache[ck] = prompt
        return prompt, wav_path

    def build_distinct_anchors(
        self,
        speaker_keys: List[str],
        speaker_instructs: Optional[Dict[str, str]] = None,
        max_sim: float = 0.80,
        tries_per_speaker: int = 4,
    ) -> Dict[str, Tuple[Optional[VoiceClonePrompt], Optional[str]]]:
        """
        Tạo và kiểm định mỏ neo (Anchor Voice Prompt) cho tất cả các nhân vật bằng mô hình CAM++ ONNX.
        Bảo đảm không có 2 nhân vật nào có độ tương đồng cosin >= max_sim (mặc định 0.80).
        Nếu phát hiện 2 nhân vật quá giống nhau, tự động gieo lại hạt số ngẫu nhiên (seed_offset) để tạo giọng khác biệt.
        """
        results: Dict[str, Tuple[Optional[VoiceClonePrompt], Optional[str]]] = {}
        if not speaker_keys:
            return results

        ins_map = dict(speaker_instructs or {})
        spk_embs: Dict[str, np.ndarray] = {}

        # Khởi tạo phiên CAM++ nếu khả dụng
        session = None
        try:
            from .speaker_embedding_onnx import create_speaker_session, extract_speaker_embedding, close_speaker_session
            sess_res = create_speaker_session(self.speaker_model_path, prefer_cuda=False, allow_mock=self.allow_mock)
            session = sess_res[0] if isinstance(sess_res, tuple) else sess_res
        except Exception:
            session = None

        try:
            for spk in speaker_keys:
                ins = (
                    ins_map.get(spk)
                    or self._speaker_instruct_cache.get(spk)
                    or (self.speaker_profiles.get(spk, {}).get("voice_instruct") if isinstance(self.speaker_profiles.get(spk), dict) else getattr(self.speaker_profiles.get(spk, {}), "voice_instruct", ""))
                    or DEFAULT_SPEAKER_PRESETS.get(spk, {}).get("instruct", "")
                    or self.instruct
                )

                best_p, best_w = None, None
                best_emb = None

                for attempt in range(tries_per_speaker):
                    offset = attempt * 12345
                    p, w = self._build_anchor_prompt(
                        key=f"char_{spk}",
                        instruct=ins,
                        seed_offset=offset,
                    )
                    if p is None or not w or not os.path.exists(w):
                        continue

                    best_p, best_w = p, w

                    if session is not None:
                        try:
                            audio_data, sr = sf.read(w, dtype="float32")
                            if audio_data.ndim > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            emb = extract_speaker_embedding(session, audio_data, sample_rate=sr)
                            norm = float(np.linalg.norm(emb))
                            if norm > 1e-6:
                                emb = emb / norm

                            # Kiểm tra độ tương đồng cosin với các nhân vật đã tạo trước đó
                            is_too_similar = False
                            for other_spk, other_emb in spk_embs.items():
                                sim = float(np.dot(emb, other_emb))
                                if sim >= max_sim:
                                    is_too_similar = True
                                    self.log(
                                        f"Nhân vật {spk} có độ tương đồng âm sắc cao với {other_spk} "
                                        f"({round(sim, 2)} >= {max_sim}). Đang thử hạt số khác..."
                                    )
                                    break

                            if not is_too_similar:
                                best_emb = emb
                                break
                        except Exception as emb_err:
                            self.log(f"Không thể đo embedding cho {spk}: {emb_err}")
                            break
                    else:
                        break

                if best_p is not None:
                    results[spk] = (best_p, best_w)
                    if best_emb is not None:
                        spk_embs[spk] = best_emb
                    elif "emb" in locals() and emb is not None:
                        spk_embs[spk] = emb
                        self.log(f"Cảnh báo: Nhân vật {spk} chưa đạt độ tách biệt âm học tối ưu sau {tries_per_speaker} lần thử hạt số.")
                        spk_prof = self.speaker_profiles.get(spk)
                        if isinstance(spk_prof, dict):
                            spk_prof["status"] = "review_needed"
                        elif spk_prof is not None and hasattr(spk_prof, "status"):
                            spk_prof.status = "review_needed"

                    # Lưu vào cache để synthesize_blocks sử dụng lại
                    p_key = self._compute_character_prompt_key(spk)
                    self._speaker_prompt_cache[p_key] = best_p
                    self._speaker_prompt_cache[f"character_{spk}"] = best_p
                    if best_w:
                        if not hasattr(self, "_speaker_prompt_wav_cache"):
                            self._speaker_prompt_wav_cache = {}
                        self._speaker_prompt_wav_cache[p_key] = best_w
                        self._speaker_prompt_wav_cache[f"character_{spk}"] = best_w
        finally:
            if session is not None:
                try:
                    close_speaker_session(session)
                except Exception:
                    pass

        return results

    @staticmethod
    def _extract_acoustic_features(seg: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Trích xuất vector đặc trưng âm học của người nói từ đoạn sóng âm:
        - Cao độ cơ bản F0 (Hz) và độ biến thiên F0 qua tự tương quan
        - Trọng tâm phổ tần số và 8 dải năng lượng hình thái âm sắc
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

            # Tính tự tương quan chuẩn hóa để tìm F0 (yêu cầu đỉnh nội bộ để loại bỏ tín hiệu phẳng DC/nhiễu)
            ac = np.correlate(windowed, windowed, mode="full")[frame_len - 1:]
            if ac[0] > 1e-6 and max_lag > min_lag:
                sub_ac = ac[min_lag:max_lag] / ac[0]
                best_idx = int(np.argmax(sub_ac))
                if (
                    0 < best_idx < len(sub_ac) - 1
                    and sub_ac[best_idx] > sub_ac[best_idx - 1]
                    and sub_ac[best_idx] > sub_ac[best_idx + 1]
                    and sub_ac[best_idx] > 0.28
                ):
                    lag = min_lag + best_idx
                    f0_list.append(float(sr) / float(lag))

        if not spectra or not f0_list:
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

        median_f0 = float(np.median(f0_list))
        iqr_f0 = float(np.percentile(f0_list, 75) - np.percentile(f0_list, 25)) if len(f0_list) >= 4 else 15.0

        feat = np.concatenate([
            np.array([median_f0 / 200.0, iqr_f0 / 100.0, centroid / 2000.0], dtype=np.float32),
            band_arr,
        ])
        return feat

    def detect_speakers_from_video(
        self,
        video_path: str,
        blocks: List[SubtitleBlock],
        max_speakers: Optional[int] = None,
        exact_speakers: Optional[int] = None,
        clustering_sensitivity: str = "standard",
        prefer_cuda: Optional[bool] = None,
        allow_mock: Optional[bool] = None,
        enable_ai_vocal_separation: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tự động nhận diện nhiều người nói trực tiếp từ âm thanh gốc của video theo mốc thời gian phụ đề (.srt).
        Sử dụng bộ điều phối SpeakerDiarizationCoordinator (ONNX Runtime tăng tốc GPU/CPU) và bảo vệ các câu đã khóa.
        """
        if not video_path or not os.path.exists(video_path):
            self.log("Không tìm thấy tệp video để phân tích giọng người nói.")
            return {}

        self._active_diarizer_session = None

        # Xóa các mẫu giọng khóa tự động cũ và cache prompt nhân vật từ lần nhận diện trước
        self.reset_speaker_locked_prompt()
        keys_to_clean = [k for k in self._speaker_prompt_cache if k.startswith("character_") or k.startswith("char_")]
        for k in keys_to_clean:
            self._speaker_prompt_cache.pop(k, None)

        from .speaker_diarization import SpeakerDiarizationCoordinator, ConfidenceLevel

        cuda_avail = bool(torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
        use_cuda = cuda_avail if prefer_cuda is None else bool(prefer_cuda)
        effective_allow_mock = self.allow_mock if allow_mock is None else bool(allow_mock)

        # Tự động tải mô hình CAM++ nếu chưa có hoặc dung lượng không hợp lệ (< 1MB)
        if not effective_allow_mock and (not self.speaker_model_path or not os.path.exists(self.speaker_model_path) or os.path.getsize(self.speaker_model_path) < 1024 * 1024):
            try:
                from .speaker_embedding_onnx import download_speaker_embedding_model, resolve_speaker_model_path
                resolved = resolve_speaker_model_path(self.speaker_model_path)
                if resolved and os.path.exists(resolved) and os.path.getsize(resolved) >= 1024 * 1024:
                    self.speaker_model_path = resolved
                else:
                    target_dir = os.path.dirname(self.speaker_model_path) if self.speaker_model_path else self.voice_prompt_cache_dir
                    self.log(f"Đang kiểm tra và tải mô hình nhận diện giọng CAM++ vào: {target_dir}...")
                    dl_path = download_speaker_embedding_model(target_dir)
                    if dl_path and os.path.exists(dl_path) and os.path.getsize(dl_path) >= 1024 * 1024:
                        self.speaker_model_path = dl_path
                        self.log("Tải mô hình nhận diện giọng CAM++ thành công.")
                    else:
                        self.log("Cảnh báo: Tệp mô hình CAM++ không đủ dung lượng tiêu chuẩn sau khi tải.")
            except Exception as dl_err:
                self.log(f"Cảnh báo: Không thể tự động tải mô hình CAM++: {dl_err}")

        coordinator = SpeakerDiarizationCoordinator(
            model_path=self.speaker_model_path,
            prefer_cuda=use_cuda,
            allow_mock=effective_allow_mock,
        )

        self.last_diarization_error = ""
        self.log(f"Đang chuẩn bị âm thanh từ video để nhận diện đa nhân vật: {os.path.basename(video_path)}...")
        separator = VoiceSeparator()
        temp_vocals = os.path.join(self.voice_prompt_cache_dir, f"diarize_vocals_{uuid.uuid4().hex[:8]}.wav")
        temp_raw_audio = os.path.join(self.voice_prompt_cache_dir, f"diarize_audio_{uuid.uuid4().hex[:8]}.wav")
        temp_raw_audio_24k = os.path.join(self.voice_prompt_cache_dir, f"raw24k_audio_{uuid.uuid4().hex[:8]}.wav")
        source_audio_for_diarization = ""
        clone_audio_for_samples = ""

        try:
            # Trích xuất âm thanh gốc 24 kHz chất lượng cao để làm mẫu clone nhân vật
            try:
                if separator._extract_audio(video_path, temp_raw_audio_24k, sample_rate=24000) and os.path.isfile(temp_raw_audio_24k):
                    clone_audio_for_samples = temp_raw_audio_24k
            except Exception:
                clone_audio_for_samples = ""

            if enable_ai_vocal_separation:
                try:
                    ok = separator.separate_from_video(video_path, temp_vocals)
                    if ok and os.path.isfile(temp_vocals) and os.path.getsize(temp_vocals) > 0:
                        _voc_info = sf.info(temp_vocals)
                        if _voc_info.frames > 0:
                            _sample_frames = min(_voc_info.frames, 32000)
                            _check_data, _ = sf.read(temp_vocals, stop=_sample_frames)
                            if np.max(np.abs(_check_data)) >= 1e-4:
                                source_audio_for_diarization = temp_vocals
                except Exception as e:
                    self.log(f"Tách giọng AI không khả dụng hoặc lỗi: {e}. Sử dụng âm thanh gốc.")

            if not source_audio_for_diarization:
                # Trích xuất âm thanh gốc chất lượng cao từ video
                if separator._extract_audio(video_path, temp_raw_audio) and os.path.isfile(temp_raw_audio):
                    source_audio_for_diarization = temp_raw_audio
                elif os.path.isfile(video_path):
                    try:
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

            sensitivity_map = {
                "standard": 0.40,
                "high": 0.34,
                "low": 0.46,
            }
            distance_threshold = sensitivity_map.get(clustering_sensitivity, 0.40)

            diar_profiles = coordinator.process_video_subtitles(
                video_path=source_audio_for_diarization,
                blocks=blocks,
                max_speakers=max_speakers,
                exact_speakers=exact_speakers,
                distance_threshold=distance_threshold,
                prefer_cuda=use_cuda,
                temp_dir=self.voice_prompt_cache_dir,
                clone_audio_path=clone_audio_for_samples or None,
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

                # Đo đặc trưng âm học thực tế (F0 trung vị, giới tính, nhóm tuổi, cao độ) trên toàn bộ các câu của từng nhân vật
                src_path = prof_data.get("source_audio_path") or getattr(prof, "source_audio_path", "")
                speaker_f0_list: List[float] = []
                speaker_feat_list: List[np.ndarray] = []

                if source_audio_for_diarization and os.path.exists(source_audio_for_diarization):
                    try:
                        diar_info = sf.info(source_audio_for_diarization)
                        d_sr = diar_info.samplerate
                        for blk in blocks:
                            if getattr(blk, "speaker", "") == spk_name:
                                # Chỉ quét các câu có độ tin cậy cao để tránh lẫn giọng của nhân vật khác
                                blk_conf = getattr(blk, "confidence_level", "")
                                if blk_conf and blk_conf != "high" and blk_conf != ConfidenceLevel.HIGH.value:
                                    continue
                                b_start = parse_timestamp_to_seconds(blk.start_time)
                                b_end = parse_timestamp_to_seconds(getattr(blk, "end_time", blk.start_time))
                                if b_end - b_start >= 0.8:
                                    s_f = max(0, int(b_start * d_sr))
                                    e_f = min(diar_info.frames, int(b_end * d_sr))
                                    if e_f - s_f >= int(0.4 * d_sr):
                                        seg_audio, _ = sf.read(source_audio_for_diarization, start=s_f, stop=e_f, dtype="float32")
                                        if seg_audio.ndim > 1:
                                            seg_audio = np.mean(seg_audio, axis=1)
                                        c_feat = self._extract_acoustic_features(seg_audio, d_sr)
                                        if c_feat is not None:
                                            speaker_feat_list.append(c_feat)
                                            speaker_f0_list.append(float(c_feat[0] * 200.0))
                    except Exception as loop_err:
                        self.log(f"Không thể quét đặc trưng âm học chuỗi câu thoại của {spk_name}: {loop_err}")

                if not speaker_f0_list and src_path and os.path.exists(src_path):
                    try:
                        audio_d, a_sr = sf.read(src_path, dtype="float32")
                        if audio_d.ndim > 1:
                            audio_d = np.mean(audio_d, axis=1)
                        feat = self._extract_acoustic_features(audio_d, a_sr)
                        if feat is not None:
                            speaker_feat_list.append(feat)
                            speaker_f0_list.append(float(feat[0] * 200.0))
                    except Exception as feat_err:
                        self.log(f"Không thể đo đặc trưng âm học của {spk_name}: {feat_err}")

                # Phân loại âm học đa chiều (tách biệt giới tính và độ tuổi)
                centroid_val = float(np.median([cf[2] * 2000.0 for cf in speaker_feat_list])) if speaker_feat_list else 1500.0
                iqr_val = float(np.median([cf[1] * 100.0 for cf in speaker_feat_list])) if speaker_feat_list else 15.0
                band_arr = np.mean([cf[3:11] for cf in speaker_feat_list], axis=0) if speaker_feat_list else None

                acoustic_res = AcousticClassifier.classify(
                    f0_list=speaker_f0_list,
                    centroid=centroid_val,
                    iqr_f0=iqr_val,
                    band_energies=band_arr,
                    allow_child=False,
                )

                # Nhận diện phương ngữ tiếng Việt từ phụ đề của nhân vật
                spk_sub_texts = [getattr(b, "text", "") for b in blocks if getattr(b, "speaker", "") == spk_name]
                accent, accent_conf = VietnameseAccentClassifier.classify(spk_sub_texts)

                gender = acoustic_res["gender"]
                gender_conf = acoustic_res["gender_confidence"]
                age_group = acoustic_res["age_group"]
                pitch_lvl = acoustic_res["pitch_level"]
                pitch_hz = acoustic_res["pitch_hz"]
                style_tag = acoustic_res["style_tag"]

                voice_ins = build_voice_design_instruct(gender=gender, age=age_group, pitch=pitch_lvl, style=style_tag) or f"{gender}, young adult, moderate pitch"

                prof_data["pitch_hz"] = pitch_hz
                prof_data["gender"] = gender
                prof_data["gender_confidence"] = gender_conf
                prof_data["age_group"] = age_group
                prof_data["pitch_level"] = pitch_lvl
                prof_data["style_tag"] = style_tag
                prof_data["vietnamese_accent"] = accent
                prof_data["accent_confidence"] = accent_conf
                prof_data["method"] = "preset"
                prof_data["user_locked"] = False
                prof_data["voice_instruct"] = voice_ins
                prof_data["instruct"] = voice_ins
                if src_path:
                    prof_data["source_audio_path"] = src_path
                    prof_data["ref_audio_path"] = src_path

                if hasattr(prof, "gender"):
                    prof.gender = gender
                if hasattr(prof, "gender_confidence"):
                    prof.gender_confidence = gender_conf
                if hasattr(prof, "age_group"):
                    prof.age_group = age_group
                if hasattr(prof, "pitch_level"):
                    prof.pitch_level = pitch_lvl
                if hasattr(prof, "vietnamese_accent"):
                    prof.vietnamese_accent = accent
                if hasattr(prof, "accent_confidence"):
                    prof.accent_confidence = accent_conf
                if hasattr(prof, "style_tag"):
                    prof.style_tag = style_tag
                if hasattr(prof, "method"):
                    prof.method = "preset"
                if hasattr(prof, "user_locked"):
                    prof.user_locked = False
                if hasattr(prof, "voice_instruct"):
                    prof.voice_instruct = voice_ins
                if voice_ins:
                    for b in blocks:
                        if getattr(b, "speaker", "") == spk_name and not getattr(b, "is_speaker_locked", False):
                            b.voice_instruct = voice_ins

                ratio = getattr(prof, "high_confidence_ratio", 1.0)
                prof_data["high_confidence_ratio"] = float(ratio if ratio is not None else 1.0)
                prof_data["description"] = f"Độ tin cậy: {int((ratio or 1.0) * 100)}%"
                needs_review = ratio < 0.7 or gender_conf < 0.80 or acoustic_res.get("review_needed", False)
                prof_data["confidence_level"] = "review_needed" if needs_review else "high"
                if needs_review:
                    prof_data["status"] = "review_needed"
                else:
                    prof_data["status"] = "locked"
                detected_profiles[spk_name] = prof_data

            # Phân bổ giọng đọc AI tiếng Việt không trùng lặp (1-đối-1) cho toàn bộ các nhân vật
            GlobalVoiceAllocator.allocate_distinct_voices(list(detected_profiles.values()), subtitle_blocks=blocks)
            for spk_name, p_data in detected_profiles.items():
                assigned_preset = p_data.get("preset_name", "")
                assigned_ins = p_data.get("voice_instruct", "")
                p_data["recommended_voices"] = VoiceRecommendationEngine.recommend_voices(p_data, top_k=3)
                if spk_name in diar_profiles:
                    d_prof = diar_profiles[spk_name]
                    if hasattr(d_prof, "preset_name"):
                        d_prof.preset_name = assigned_preset
                    if hasattr(d_prof, "voice_instruct"):
                        d_prof.voice_instruct = assigned_ins
                    if hasattr(d_prof, "method"):
                        d_prof.method = "preset"
                    if hasattr(d_prof, "recommended_voices"):
                        d_prof.recommended_voices = p_data["recommended_voices"]
                if assigned_ins:
                    self._speaker_instruct_cache[spk_name] = assigned_ins
                    for b in blocks:
                        if getattr(b, "speaker", "") == spk_name and not getattr(b, "is_speaker_locked", False):
                            b.voice_instruct = assigned_ins
                            b.voice_ref_path = ""

            # Hợp nhất (merge) detected_profiles với self.speaker_profiles hiện có
            # Bảo toàn nguyên vẹn hồ sơ của các nhân vật đã được người dùng tùy chỉnh hoặc đã khóa
            locked_names = {
                getattr(b, "speaker", "")
                for b in (blocks or [])
                if getattr(b, "is_speaker_locked", False) and getattr(b, "speaker", "").strip()
            }
            def _user_locked(k, p):
                if k in locked_names:
                    return True
                if isinstance(p, dict):
                    return bool(p.get("user_locked", False))
                return bool(getattr(p, "user_locked", False))

            merged_profiles = {k: v for k, v in (self.speaker_profiles or {}).items() if _user_locked(k, v)}
            for spk_name, p_data in detected_profiles.items():
                if spk_name not in merged_profiles:
                    merged_profiles[spk_name] = p_data

            self.speaker_profiles = merged_profiles
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
            for temp_path in (temp_vocals, temp_raw_audio, temp_raw_audio_24k):
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            gc.collect()

    def detect_speakers_from_srt_text(
        self,
        blocks: List[SubtitleBlock],
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tự động nhận diện và phân vai nhiều người nói khi chỉ cung cấp tệp phụ đề (.srt) mà không có video:
        - Nhận diện tiền tố tên nhân vật ([Nam]:, [Nữ]:, Người 1:, Nhân vật A:, SPEAKER_01:...)
        - Nhận diện hội thoại gạch đầu dòng (- ) và đại từ nhân xưng tiếng Việt / Anh / Trung
        - Tự động gán thiết kế giọng nói riêng biệt cho từng nhân vật
        """
        if not blocks:
            return {}

        effective_max_spk = int(max_speakers) if max_speakers is not None else 16

        prefix_re = re.compile(
            r"^\s*(?:\[([^\]\r\n]{1,25})\]\s*[:：]?|\(([^)\r\n]{1,25})\)\s*[:：]|([A-Za-zÀ-ỹ0-9_ ]{1,25})\s*[:：])\s*(.+)$"
        )
        ignored_speaker_prefixes = {
            "lưu ý", "chú ý", "ghi chú", "hướng dẫn", "lời dẫn", "lời thoại",
            "note", "warning", "tip", "info", "tập", "phần", "bài", "trích dẫn",
            "chương", "cảnh", "hồi", "đoạn", "nhạc", "tiếng động",
            "hôm nay", "ngày mai", "hôm qua", "lúc này", "khi này", "tại đây",
            "sau đó", "tiếp theo", "ban đầu", "kết quả", "nguyên nhân", "lý do", "ví dụ", "chẳng hạn",
        }
        female_markers = {"nữ", "nu", "female", "woman", "girl", "cô", "chị", "bà", "mẹ", "em", "nàng", "thiếp", "tiểu thư"}
        male_markers = {"nam", "male", "man", "boy", "anh", "ông", "bố", "cha", "chú", "bác", "chàng", "huynh", "trẫm"}
        elder_markers = {"ông", "bà", "lão", "cụ", "elder", "old"}
        child_markers = {"bé", "con", "cháu", "trẻ", "child", "kid"}

        named_speaker_map: Dict[str, str] = {}
        detected_profiles: Dict[str, Dict[str, Any]] = {}

        # Kiểm tra xem toàn bộ tệp phụ đề có chứa bất kỳ tiền tố người nói tường minh hoặc câu đã khóa hay không
        has_any_explicit_prefix = any(
            getattr(blk, "is_speaker_locked", False) and getattr(blk, "speaker", "").strip()
            for blk in blocks
        )
        if not has_any_explicit_prefix:
            for blk in blocks:
                r_txt = (blk.translated_text if blk.translated_text else blk.text).strip()
                m_chk = prefix_re.match(r_txt)
                if m_chk:
                    c_tag = (m_chk.group(1) or m_chk.group(2) or m_chk.group(3) or "").strip()
                    c_key = c_tag.lower()
                    is_pure_num_or_ts = bool(re.match(r"^\d+([:.]\d+)?$", c_tag))
                    is_ignored_word = c_key in ignored_speaker_prefixes
                    is_non_verbal = f"[{c_key}]" in NON_VERBAL_TAGS
                    is_valid_plain = True
                    if m_chk.group(3):
                        words_in_tag = c_tag.split()
                        is_title_or_upper = c_tag.isupper() or all(w[0].isupper() for w in words_in_tag if w)
                        if len(words_in_tag) > 3 or not any(c.isalpha() for c in c_tag) or not is_title_or_upper:
                            is_valid_plain = False
                    if not is_pure_num_or_ts and not is_ignored_word and not is_non_verbal and is_valid_plain:
                        has_any_explicit_prefix = True
                        break

        # Nếu không có tiền tố tên nhân vật trong toàn bộ phụ đề và không có số người nói cố định do người dùng chỉ định,
        # tránh đoán mò 2+ người nói theo đại từ hoặc dấu chấm hỏi; mặc định tất cả thuộc về 1 giọng đọc người dẫn chuyện
        user_fixed_speakers = (getattr(self, "exact_speakers", None) is not None and getattr(self, "exact_speakers", None) > 1)
        allow_multi_inference = has_any_explicit_prefix or user_fixed_speakers

        def _register_speaker(
            spk_id: str,
            instruct_str: str,
            desc_str: str,
            method: str = "preset",
            ref_path: str = "",
            user_locked: bool = False,
            marker_matched: bool = False,
            display_name: str = "",
        ):
            if spk_id not in detected_profiles:
                ins_low = (instruct_str or "").lower()
                g = "female" if "female" in ins_low else ("male" if "male" in ins_low else "auto")
                g_conf = 0.90 if marker_matched else 0.40
                detected_profiles[spk_id] = {
                    "speaker_id": spk_id,
                    "display_name": display_name or spk_id,
                    "description": desc_str,
                    "gender": g,
                    "gender_confidence": g_conf,
                    "pitch_hz": 0.0,
                    "ref_audio_path": ref_path,
                    "source_audio_path": ref_path,
                    "method": method,
                    "status": "locked" if marker_matched else "review_needed",
                    "user_locked": user_locked,
                    "instruct": instruct_str,
                    "voice_instruct": instruct_str,
                    "count": 0,
                }
            detected_profiles[spk_id]["count"] += 1

        current_turn = 0
        last_assigned_spk = ""
        last_assigned_ins = ""
        last_assigned_desc = ""
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
                is_pure_num_or_ts = bool(re.match(r"^\d+([:.]\d+)?$", tag_name))
                is_ignored_word = tag_key in ignored_speaker_prefixes
                is_non_verbal = f"[{tag_key}]" in NON_VERBAL_TAGS
                
                is_valid_plain_name = True
                if m.group(3):
                    words_in_tag = tag_name.split()
                    is_title_or_upper = tag_name.isupper() or all(w[0].isupper() for w in words_in_tag if w)
                    if len(words_in_tag) > 3 or not any(c.isalpha() for c in tag_name) or not is_title_or_upper:
                        is_valid_plain_name = False

                if not is_pure_num_or_ts and not is_ignored_word and not is_non_verbal and is_valid_plain_name:
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

                _register_speaker(assigned_spk, assigned_instruct, assigned_desc, method=ex_method, ref_path=ex_ref, user_locked=True)
                continue

            assigned_spk = ""
            assigned_instruct = ""
            assigned_desc = ""

            if has_valid_prefix:
                if tag_key not in named_speaker_map:
                    spk_num = min(len(named_speaker_map) + 1, effective_max_spk)
                    spk_id = f"Người nói {spk_num}"
                    named_speaker_map[tag_key] = spk_id
                assigned_spk = named_speaker_map[tag_key]

                marker_matched = False
                if any(w in tag_key for w in female_markers):
                    assigned_instruct = "female, young adult, moderate pitch"
                    assigned_desc = f"Giọng Nữ - {tag_name}" if tag_name else "Giọng Nữ"
                    marker_matched = True
                elif any(w in tag_key for w in male_markers):
                    assigned_instruct = "male, young adult, moderate pitch"
                    assigned_desc = f"Giọng Nam - {tag_name}" if tag_name else "Giọng Nam"
                    marker_matched = True
                elif any(w in tag_key for w in child_markers):
                    assigned_instruct = "child, high pitch"
                    assigned_desc = f"Giọng Trẻ Em - {tag_name}" if tag_name else "Giọng Trẻ Em"
                    marker_matched = True
                elif any(w in tag_key for w in elder_markers):
                    assigned_instruct = "male, elderly, very low pitch"
                    assigned_desc = f"Giọng Cao Tuổi - {tag_name}" if tag_name else "Giọng Cao Tuổi"
                    marker_matched = True

            marker_matched = False if "marker_matched" not in locals() else marker_matched
            if not assigned_spk:
                if not allow_multi_inference:
                    assigned_spk = "Người nói 1"
                    assigned_instruct = "male, young adult, moderate pitch"
                    assigned_desc = "Giọng Người Dẫn Chuyện"
                    marker_matched = True
                else:
                    is_dialogue_dash = raw_text.startswith("-") or raw_text.startswith("—")
                    if not is_dialogue_dash and last_assigned_spk:
                        assigned_spk = last_assigned_spk
                        assigned_instruct = last_assigned_ins
                        assigned_desc = last_assigned_desc
                    else:
                        words = set(re.findall(r"[a-zA-ZÀ-ỹ]+", lower_text))
                        if is_dialogue_dash or "?" in raw_text:
                            current_turn = (current_turn + 1) % min(max(2, effective_max_spk), 4)

                        if words & child_markers and effective_max_spk >= 4:
                            assigned_spk = f"Người nói {min(4, effective_max_spk)}"
                            assigned_instruct = "child, high pitch"
                            assigned_desc = "Giọng Trẻ Em / Cao"
                            marker_matched = True
                        elif words & elder_markers and effective_max_spk >= 3:
                            assigned_spk = f"Người nói {min(3, effective_max_spk)}"
                            assigned_instruct = "male, middle-aged, low pitch"
                            assigned_desc = "Giọng Trầm / Trung Niên"
                            marker_matched = True
                        elif (words & female_markers) and not (words & male_markers):
                            assigned_spk = "Người nói 2" if effective_max_spk >= 2 else "Người nói 1"
                            assigned_instruct = "female, young adult, moderate pitch"
                            assigned_desc = "Giọng Nữ Thanh Niên"
                            marker_matched = True
                        elif (words & male_markers) and not (words & female_markers):
                            assigned_spk = "Người nói 1"
                            assigned_instruct = "male, young adult, moderate pitch"
                            assigned_desc = "Giọng Nam Thanh Niên"
                            marker_matched = True
                        else:
                            spk_idx = (current_turn % min(max(2, effective_max_spk), len(DEFAULT_SPEAKER_PRESETS))) + 1
                            assigned_spk = f"Người nói {spk_idx}"

            preset = DEFAULT_SPEAKER_PRESETS.get(assigned_spk, DEFAULT_SPEAKER_PRESETS["Người nói 1"])
            if not assigned_instruct:
                assigned_instruct = preset["instruct"]
            if not assigned_desc:
                assigned_desc = preset["label"].split(" - ")[-1]

            blk.speaker = assigned_spk
            blk.voice_instruct = assigned_instruct
            blk.voice_ref_path = ""
            disp_name = tag_name if has_valid_prefix and tag_name else assigned_spk
            _register_speaker(assigned_spk, assigned_instruct, assigned_desc, marker_matched=marker_matched, display_name=disp_name)
            last_assigned_spk = assigned_spk
            last_assigned_ins = assigned_instruct
            last_assigned_desc = assigned_desc

        # Phân bổ giọng đọc AI tiếng Việt không trùng lặp cho toàn bộ nhân vật
        GlobalVoiceAllocator.allocate_distinct_voices(list(detected_profiles.values()), subtitle_blocks=blocks)
        for spk_name, p_data in detected_profiles.items():
            assigned_preset = p_data.get("preset_name", "")
            assigned_ins = p_data.get("voice_instruct", "")
            p_data["recommended_voices"] = VoiceRecommendationEngine.recommend_voices(p_data, top_k=3)
            if assigned_ins:
                self._speaker_instruct_cache[spk_name] = assigned_ins
                for b in blocks:
                    if getattr(b, "speaker", "") == spk_name and not getattr(b, "is_speaker_locked", False):
                        b.voice_instruct = assigned_ins
                        b.voice_ref_path = ""

        # Hợp nhất với self.speaker_profiles hiện có, bảo toàn các hồ sơ người dùng đã khóa
        locked_names = {
            getattr(b, "speaker", "")
            for b in (blocks or [])
            if getattr(b, "is_speaker_locked", False) and getattr(b, "speaker", "").strip()
        }
        def _user_locked(k, p):
            if k in locked_names:
                return True
            if isinstance(p, dict):
                return bool(p.get("user_locked", False))
            return bool(getattr(p, "user_locked", False))

        merged_profiles = {k: v for k, v in (self.speaker_profiles or {}).items() if _user_locked(k, v)}
        for spk_name, p_data in detected_profiles.items():
            if spk_name not in merged_profiles:
                merged_profiles[spk_name] = p_data

        self.speaker_profiles = merged_profiles
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
        override_ref_text: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Tổng hợp giọng nói cho một câu đơn lẻ bằng OmniVoice trên GPU.
        Hỗ trợ ghi đè mẫu giọng hoặc thuộc tính thiết kế giọng trên từng câu cho chế độ Đa Nhân Vật.
        """
        self.load_model()
        text_clean = re.sub(r"\s*[\r\n]+\s*", " ", text)
        text_clean = re.sub(r"(^|\s)[-–—]\s+", r"\1", text_clean).strip()
        if not text_clean:
            return np.zeros(0, dtype=np.float32)

        # Cố định hạt ngẫu nhiên theo chỉ thị thiết kế giọng để bảo đảm âm sắc, cao độ và giọng đọc
        # của cùng một nhân vật luôn nhất quán tuyệt đối giữa các câu, không bị đổi giọng mỗi câu
        if torch is not None:
            if seed is not None:
                target_seed = int(seed)
            else:
                active_instruct = (override_instruct or self.instruct or "").strip()
                target_seed = (zlib.crc32(active_instruct.encode("utf-8")) % 100000) + 42 if active_instruct else 42
            torch.manual_seed(target_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(target_seed)

        gen_config = self.build_generation_config()

        gen_kwargs: Dict[str, Any] = {
            "text": text_clean,
            "language": self.language,
            "generation_config": gen_config,
            "normalize_text": bool(self.normalize_text and self.language not in ("vi", None, "")),
        }

        if self.speed != 1.0:
            gen_kwargs["speed"] = float(self.speed)

        # Nếu target_duration được truyền và nằm trong tỷ lệ co giãn an toàn (0.75x - 1.30x),
        # sử dụng trực tiếp tham số duration gốc của OmniVoice để luồng nén âm thanh sinh ra tự nhiên
        if target_duration is not None and target_duration > 0.4:
            words = max(1, len(text_clean.split()))
            natural_dur = max(0.6, words / 3.8)
            dur_ratio = natural_dur / float(target_duration)
            if 0.75 <= dur_ratio <= 1.30:
                gen_kwargs["duration"] = float(target_duration)

        active_mode = override_mode or self.mode

        if active_mode == "design":
            # Chế độ thiết kế giọng nói: Tuyệt đối chỉ sử dụng instruct, không bao giờ dùng prompt âm thanh nhân bản
            if override_instruct:
                gen_kwargs["instruct"] = override_instruct
            elif self.instruct:
                gen_kwargs["instruct"] = self.instruct
            else:
                gen_kwargs["instruct"] = DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]
        elif override_mode == "clone":
            if override_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = override_prompt
            elif override_ref_audio and os.path.exists(override_ref_audio):
                prompt = self._get_or_create_prompt_for_audio(
                    override_ref_audio,
                    cache_key=override_ref_audio,
                    ref_text=override_ref_text,
                )
                if prompt is not None:
                    gen_kwargs["voice_clone_prompt"] = prompt
                else:
                    raise RuntimeError(f"Không thể tạo đặc trưng giọng từ tệp mẫu: {override_ref_audio}")
            else:
                raise FileNotFoundError(f"Tệp âm thanh mẫu nhân bản giọng không tồn tại hoặc chưa được chỉ định: {override_ref_audio or 'chưa có'}")
        elif override_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = override_prompt
        elif override_ref_audio and os.path.exists(override_ref_audio):
            prompt = self._get_or_create_prompt_for_audio(
                override_ref_audio,
                cache_key=override_ref_audio,
                ref_text=override_ref_text,
            )
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

        trimmed = self._trim_silence_padding(audio_arr, threshold_db=-45.0, sample_rate=self.sample_rate)
        if len(trimmed) > 0:
            audio_arr = trimmed

        # Nếu cần khớp thời lượng phụ đề và âm thanh sinh ra dài hơn khung cho phép,
        # co giãn thời lượng an toàn bằng WSOLA để bảo toàn 100% cao độ và âm sắc
        if target_duration is not None and target_duration > 0.4 and audio_arr.size > 0:
            cur_dur = len(audio_arr) / float(self.sample_rate)
            if cur_dur > float(target_duration) * 1.05:
                needed_speed = cur_dur / float(target_duration)
                audio_arr = self._safe_time_stretch(audio_arr, needed_speed, sample_rate=self.sample_rate)

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
            "normalize_text": False,
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
    def _safe_time_stretch(
        audio_arr: np.ndarray,
        speed_factor: float,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        Co giãn thời lượng âm thanh trong ngưỡng an toàn (0.75x - 1.35x)
        bảo toàn 100% cao độ và đặc trưng formant bằng thuật toán WSOLA thuần NumPy,
        loại bỏ hoàn toàn hiện tượng méo tiếng hoặc biến thành giọng chuột sóc the thé.
        """
        if audio_arr.size == 0 or abs(speed_factor - 1.0) < 0.02:
            return audio_arr

        clamped_factor = max(0.75, min(1.35, float(speed_factor)))
        alpha = 1.0 / clamped_factor

        win_size = int(0.028 * sample_rate)
        if win_size % 2 != 0:
            win_size += 1

        synth_hop = win_size // 2
        ana_hop = int(round(synth_hop / alpha))
        search_range = int(0.014 * sample_rate)

        if len(audio_arr) < win_size + search_range + 10:
            return audio_arr

        window = np.hanning(win_size).astype(np.float32)
        out_len = int(round(len(audio_arr) * alpha))
        output = np.zeros(out_len + win_size, dtype=np.float32)
        weights = np.zeros(out_len + win_size, dtype=np.float32)

        output[:win_size] += audio_arr[:win_size] * window
        weights[:win_size] += window

        synth_pos = synth_hop
        ana_nom = ana_hop
        step_cand = max(1, sample_rate // 4000)

        while synth_pos + win_size <= out_len and ana_nom < len(audio_arr):
            ref_seg = output[synth_pos : synth_pos + win_size]
            start_search = max(0, ana_nom - search_range)
            end_search = min(len(audio_arr) - win_size, ana_nom + search_range)

            if end_search <= start_search:
                break

            ref_norm = float(np.linalg.norm(ref_seg)) + 1e-6
            best_pos = start_search
            best_corr = -2.0
            for cand in range(start_search, end_search, step_cand):
                cand_seg = audio_arr[cand : cand + win_size]
                cand_norm = float(np.linalg.norm(cand_seg)) + 1e-6
                corr = float(np.dot(ref_seg, cand_seg)) / (ref_norm * cand_norm)
                if corr > best_corr:
                    best_corr = corr
                    best_pos = cand

            grain = audio_arr[best_pos : best_pos + win_size] * window
            output[synth_pos : synth_pos + win_size] += grain
            weights[synth_pos : synth_pos + win_size] += window

            synth_pos += synth_hop
            ana_nom += ana_hop

        nz = weights > 1e-4
        output[nz] /= weights[nz]
        res = output[:out_len]

        fade_len = min(240, len(res) // 10)
        if fade_len > 2:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            res[:fade_len] *= ramp
            res[-fade_len:] *= ramp[::-1]
        return res

    @staticmethod
    def _overlay_audio_segment(
        full_audio: np.ndarray,
        audio_arr: np.ndarray,
        start_sample: int,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        Ghép nối mượt mà đoạn âm thanh vào luồng tổng bằng kỹ thuật Crossfade công suất tương đương,
        loại bỏ hoàn toàn hiện tượng Hard Clipping và nổ lách tách do cộng dồn tín hiệu vượt ngưỡng.
        """
        if audio_arr.size == 0:
            return full_audio

        end_sample = start_sample + len(audio_arr)
        if end_sample > len(full_audio):
            pad_extra = (end_sample - len(full_audio)) + int(0.5 * sample_rate)
            full_audio = np.pad(full_audio, (0, pad_extra))

        existing = full_audio[start_sample:end_sample]
        active_indices = np.where(np.abs(existing) > 1e-4)[0]
        if len(active_indices) > 0:
            overlap_end = int(active_indices[-1]) + 1
            max_fade = min(int(0.12 * sample_rate), len(audio_arr), len(existing), overlap_end)
            fade_len = max(max_fade, 4) if max_fade > 4 else 0
            if fade_len > 4:
                t = np.linspace(0.0, np.pi / 2.0, fade_len, dtype=np.float32)
                fade_out = np.cos(t) ** 2
                fade_in = np.sin(t) ** 2
                blended = existing[:fade_len] * fade_out + audio_arr[:fade_len] * fade_in
                full_audio[start_sample : start_sample + fade_len] = blended

                # Nếu đoạn cũ còn kéo dài sau fade_len (do đuôi câu trước bị lấn), hòa trộn nhẹ thay vì ghi đè mất tiếng
                if overlap_end > fade_len:
                    mid_existing = existing[fade_len:overlap_end]
                    mid_audio = audio_arr[fade_len:overlap_end]
                    full_audio[start_sample + fade_len : start_sample + overlap_end] = np.maximum(-1.0, np.minimum(1.0, mid_existing + mid_audio))

                if end_sample > start_sample + overlap_end:
                    full_audio[start_sample + overlap_end : end_sample] = audio_arr[overlap_end:]
            else:
                full_audio[start_sample:end_sample] = np.maximum(-1.0, np.minimum(1.0, existing + audio_arr))
        else:
            full_audio[start_sample:end_sample] = audio_arr

        # Bộ giới hạn biên độ mềm bảo vệ chống nổ âm lượng
        seg = full_audio[start_sample:end_sample]
        peak = float(np.max(np.abs(seg))) if len(seg) > 0 else 0.0
        if peak > 0.98:
            full_audio[start_sample:end_sample] = np.tanh(seg / 0.98) * 0.98

        return full_audio

    @staticmethod
    def _calculate_safe_target_duration(text: str, slot_duration: float) -> Optional[float]:
        """
        Tính toán thời lượng tổng hợp an toàn dựa trên tốc độ nói tự nhiên của tiếng Việt:
        - Nếu khung phụ đề quá dài so với câu ngắn (ví dụ 2 từ trong 4s): Không ép kéo dài dị dạng, trả về None.
        - Nếu khung phụ đề quá ngắn so với câu dài (ví dụ 30 từ trong 1.5s): Chặn lại ở ngưỡng co nén tối đa 0.75x.
        - Nếu nằm trong dải tốc độ bình thường: Trả về chính xác slot_duration để đồng bộ khớp khung hình.
        """
        clean = text.strip()
        if not clean or slot_duration <= 0.0:
            return None

        words = max(1, len(clean.split()))
        natural_dur = max(0.6, words / 3.8)

        # Khung phụ đề quá dài: để nói tự nhiên và để khoảng lặng tự nhiên phía sau
        if slot_duration > natural_dur * 1.35:
            return None

        # Khung phụ đề quá ngắn: chặn ở ngưỡng co nén tối đa cho phép
        if slot_duration < natural_dur * 0.70:
            return round(natural_dur * 0.75, 2)

        return round(slot_duration, 2)

    @staticmethod
    def _trim_silence_padding(
        audio_arr: np.ndarray,
        threshold_db: float = -45.0,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        Cắt bớt khoảng lặng vô thanh ở đầu và đuôi đoạn âm thanh (dưới ngưỡng -45 dB)
        nhưng giữ lại khoảng 60ms đệm mượt để tránh làm cụt âm bật và phụ âm đầu hoặc cuối.
        """
        if audio_arr.size == 0:
            return audio_arr
        thresh = 10.0 ** (threshold_db / 20.0)
        active = np.where(np.abs(audio_arr) > thresh)[0]
        if len(active) == 0:
            return audio_arr
        margin = max(1, int(0.060 * sample_rate))
        start_idx = max(0, active[0] - margin)
        end_idx = min(len(audio_arr), active[-1] + margin + 1)
        return audio_arr[start_idx:end_idx]

    def _compute_block_cache_path(
        self,
        index: int,
        text: str,
        speaker: str,
        ref_path: Optional[str],
        instruct: Optional[str],
        target_dur: Optional[float],
        method_or_mode: Optional[str] = None,
        avail_before_next: Optional[float] = None,
    ) -> Optional[str]:
        """Tính đường dẫn tệp WAV lưu tạm (checkpoint) cho từng câu để hỗ trợ tiếp tục khi gián đoạn."""
        if not self.resume_cache_dir:
            return None
        effective_ref = ref_path or (self.ref_audio_path if self.mode == "clone" else "")
        mtime_str = ""
        if effective_ref and os.path.exists(effective_ref):
            try:
                mtime_str = str(os.path.getmtime(effective_ref))
            except Exception:
                mtime_str = ""
        raw_key = (
            f"{index}|{text}|{speaker}|{method_or_mode or ''}|{effective_ref or ''}|{mtime_str}|"
            f"{getattr(self, 'ref_text', '') or ''}|{self.language or 'vi'}|{self.model_name_or_path or ''}|"
            f"{instruct or ''}|{target_dur or 0.0:.2f}|{avail_before_next or 0.0:.2f}|{self.mode}|{self.speed:.2f}|"
            f"{getattr(self, 'guidance_scale', 2.0):.2f}|{getattr(self, 'num_step', 32)}|"
            f"{int(bool(getattr(self, 'denoise', True)))}|{getattr(self, 't_shift', 0.5):.2f}|{getattr(self, 'overflow_strategy', 'extend')}"
        )
        digest = hashlib.sha1(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return os.path.join(self.resume_cache_dir, f"blk_{index + 1:04d}_{digest}.wav")

    @staticmethod
    def _compute_character_prompt_key(spk: str, ref_path: Optional[str] = None) -> str:
        """Tạo khóa bộ nhớ đệm prompt nhân vật bao gồm định danh tệp mẫu và mtime để tránh dùng lại bản cũ khi đổi tệp."""
        if not spk:
            return ""
        mtime_str = ""
        path_hash = ""
        if ref_path:
            path_hash = hashlib.sha1(str(ref_path).encode("utf-8", errors="ignore")).hexdigest()[:6]
            if os.path.exists(ref_path):
                try:
                    mtime_str = f"_{int(os.path.getmtime(ref_path))}"
                except Exception:
                    mtime_str = ""
        return f"character_{spk}_{path_hash}{mtime_str}" if path_hash else f"character_{spk}"

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

        # Đảm bảo các nhân vật phân vai preset có mỏ neo phân biệt âm học qua CAM++
        def _get_prof_method(s_key):
            p = self.speaker_profiles.get(s_key, {})
            return (p.get("method") if isinstance(p, dict) else getattr(p, "method", "")) or "preset"

        preset_spks = [
            spk for spk in sorted(speaker_keys)
            if _get_prof_method(spk) in ("preset", "design", None, "")
        ]
        if len(preset_spks) > 1:
            try:
                self.build_distinct_anchors(preset_spks)
            except Exception as e_dist:
                self.log(f"Không thể khởi tạo mỏ neo phân biệt CAM++ cho các nhân vật: {e_dist}")

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

            char_prompt_key = self._compute_character_prompt_key(spk, ref_path)

            # 1. Nếu nhân vật chọn phương pháp preset hoặc đã có preset chỉ định rõ
            if method == "preset":
                ins_val = (
                    (prof.get("instruct") or prof.get("voice_instruct") if isinstance(prof, dict) else getattr(prof, "voice_instruct", ""))
                    or (VIETNAMESE_VOICE_CATALOG.get(preset_name, {}).get("instruct") if preset_name in VIETNAMESE_VOICE_CATALOG else "")
                    or DEFAULT_SPEAKER_PRESETS.get(spk, {}).get("instruct", "")
                )
                if ins_val:
                    self._speaker_instruct_cache[spk] = ins_val
                    p_key = self._compute_character_prompt_key(spk)
                    if p_key not in self._speaker_prompt_cache and f"character_{spk}" not in self._speaker_prompt_cache:
                        p, w = self._build_anchor_prompt(
                            key=f"char_{spk}",
                            instruct=ins_val,
                            sample_text=self._get_anchor_sample_text(preset_name),
                        )
                        if p is not None:
                            self._speaker_prompt_cache[p_key] = p
                            self._speaker_prompt_cache[f"character_{spk}"] = p
                            if w:
                                self._speaker_prompt_wav_cache[p_key] = w
                                self._speaker_prompt_wav_cache[f"character_{spk}"] = w
                continue

            # 2. Nếu nhân vật chọn phương pháp clone
            if method == "clone" and ref_path and os.path.exists(ref_path):
                if char_prompt_key not in self._speaker_prompt_cache:
                    prof_ref_text = prof.get("ref_text", "") if isinstance(prof, dict) else getattr(prof, "ref_text", "")
                    prompt = self._get_or_create_prompt_for_audio(
                        ref_path,
                        cache_key=char_prompt_key,
                        ref_text=prof_ref_text or None,
                    )
                    if prompt is not None:
                        self._speaker_prompt_cache[char_prompt_key] = prompt
                        self._speaker_prompt_cache[f"character_{spk}"] = prompt
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

        has_unassigned_or_unlocked = any(
            not getattr(b, "is_speaker_locked", False)
            or not getattr(b, "speaker", "").strip()
            or getattr(b, "speaker", "") in ("Chưa phân vai", "Chưa xác định")
            for b in blocks
        )

        if self.multi_speaker_mode == "auto_video_cluster" and video_path and os.path.exists(video_path):
            if has_unassigned_or_unlocked:
                try:
                    self.detect_speakers_from_video(
                        video_path,
                        blocks,
                        max_speakers=self.max_speakers,
                        exact_speakers=getattr(self, "exact_speakers", None),
                        clustering_sensitivity=getattr(self, "clustering_sensitivity", "standard"),
                        enable_ai_vocal_separation=getattr(self, "enable_ai_vocal_separation", True),
                    )
                except Exception as exc:
                    self.last_diarization_error = str(exc)
                    self.log(f"Cảnh báo: Nhận diện người nói từ video thất bại ({exc}), tiếp tục tổng hợp với cấu hình mặc định.")
            if self.last_diarization_error:
                report.timestamp_issues.append({
                    "type": "diarization_warning",
                    "message": f"Nhận diện người nói gặp cảnh báo: {self.last_diarization_error}",
                })
        elif self.multi_speaker_mode == "auto_srt_text":
            if has_unassigned_or_unlocked:
                try:
                    self.detect_speakers_from_srt_text(blocks, max_speakers=self.max_speakers)
                except Exception as exc:
                    self.last_diarization_error = str(exc)
                    self.log(f"Cảnh báo: Phân vai từ văn bản phụ đề thất bại ({exc}), tiếp tục tổng hợp với cấu hình mặc định.")
        elif self.multi_speaker_mode == "per_line_video_clone" and video_path and os.path.exists(video_path):
            self.log("Đang tách kênh giọng nói gốc khỏi nhạc nền để nhái giọng theo từng câu thoại...")
            sep = VoiceSeparator()
            temp_voc = os.path.join(self.voice_prompt_cache_dir, f"perline_voc_{uuid.uuid4().hex[:8]}.wav")
            if sep.separate_from_video(video_path, temp_voc, sample_rate=24000) and os.path.exists(temp_voc):
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
        if not (self.multi_speaker_mode == "single" and self.mode in ("design", "auto")):
            self._prepare_and_lock_character_prompts(blocks)

        unified_anchor_prompt: Optional[VoiceClonePrompt] = None
        unified_anchor_wav: Optional[str] = None
        if self.multi_speaker_mode == "single" and self.mode in ("design", "auto"):
            target_ins = self.instruct
            if not target_ins and blocks:
                target_ins = getattr(blocks[0], "voice_instruct", "")
            if not target_ins:
                target_ins = DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]

            resolved_preset = getattr(self, "preset_name", "unified")
            sample_text = self._get_anchor_sample_text(resolved_preset)

            unified_anchor_prompt, unified_anchor_wav = self._build_anchor_prompt(
                key=f"unified_{resolved_preset}",
                instruct=target_ins,
                sample_text=sample_text,
            )
            if unified_anchor_prompt is not None:
                self.log(
                    f"Đã tạo và khóa mẫu giọng mỏ neo đồng nhất cho toàn bộ tệp phụ đề: {os.path.basename(unified_anchor_wav)}"
                )
            else:
                self.log(
                    "Cảnh báo: Không thể tạo mẫu giọng mỏ neo đồng nhất. Chuyển sang chế độ thiết kế giọng trực tiếp từng câu."
                )

        has_character_profiles = bool(self.speaker_profiles and any(getattr(b, "speaker", "") in self.speaker_profiles for b in blocks))
        if self.mode == "clone" and self.current_prompt is None and self.multi_speaker_mode == "single" and not has_character_profiles:
            if not self.auto_extract_ref and self.ref_audio_path and os.path.exists(self.ref_audio_path):
                self.set_reference_voice(self.ref_audio_path, self.ref_text)
            elif self.auto_extract_ref and video_path and os.path.exists(video_path):
                self.extract_reference_from_video(video_path, enable_ai_vocal_separation=self.enable_ai_vocal_separation)
            elif self.ref_audio_path and os.path.exists(self.ref_audio_path):
                self.set_reference_voice(self.ref_audio_path, self.ref_text)

            if self.current_prompt is None:
                report.status = "failed"
                report.message = "Không tìm thấy tệp âm thanh mẫu hoặc không thể trích xuất mẫu giọng cho chế độ nhân bản giọng đơn lẻ."
                self.log(report.message)
                self.last_report = report
                return report

        fallback_anchor_prompt: Optional[VoiceClonePrompt] = None
        fallback_anchor_wav: Optional[str] = None
        if self.multi_speaker_mode != "single":
            has_unassigned = any(not getattr(b, "speaker", "") or getattr(b, "speaker", "") in ("Chưa xác định", "unassigned", "") for b in blocks)
            if has_unassigned:
                fallback_anchor_prompt, fallback_anchor_wav = self._get_or_create_fallback_anchor()

        total_samples = int((max(total_duration, 1.0) + 5.0) * self.sample_rate)
        full_audio = np.zeros(total_samples, dtype=np.float32)
        speech_cursor_sample: int = 0
        generated_blocks_audio: List[Dict[str, Any]] = []

        # GIAI ĐOẠN 3 & 4: Tổng hợp từng câu, kiểm tra chất lượng và căn chỉnh âm thanh
        for i, blk in enumerate(blocks):
            if self._cancel_requested:
                self.log(f"Đã dừng tác vụ theo yêu cầu người dùng tại dòng phụ đề {i + 1}.")
                report.status = "cancelled"
                report.message = f"Đã hủy tác vụ sau khi xử lý {report.succeeded_count}/{len(blocks)} câu."
                self.last_report = report
                return report

            trans_text = (getattr(blk, "translated_text", "") or "").strip()
            orig_text = (getattr(blk, "text", "") or "").strip()
            raw_text = trans_text if trans_text else orig_text
            if not raw_text:
                report.skipped_empty_count += 1
                continue

            if self.language in (None, "vi") and self.normalize_text:
                text_to_speak = preprocess_vietnamese_tts_text(raw_text)
            else:
                cleaned_tx = re.sub(r"\s*[\r\n]+\s*", " ", raw_text)
                text_to_speak = re.sub(r"(^|\s)[-–—]\s+", r"\1", cleaned_tx).strip()

            start_sec = max(0.0, parse_timestamp_to_seconds(blk.start_time))
            end_sec = max(start_sec + 0.2, parse_timestamp_to_seconds(getattr(blk, "end_time", start_sec)))
            slot_duration = max(0.0, end_sec - start_sec)

            target_dur: Optional[float] = None
            if self.sync_to_subtitle_duration and slot_duration >= 0.8:
                target_dur = self._calculate_safe_target_duration(text_to_speak, slot_duration)

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
            if self.multi_speaker_mode == "single" and (self.mode in ("design", "auto") or not (self.speaker_profiles and spk_key in self.speaker_profiles)):
                # Chế độ 1 giọng đọc đồng nhất: Áp dụng một mẫu giọng duy nhất cho mọi câu thoại trong tệp phụ đề
                spk_label = "Giọng đồng nhất"
                if self.mode in ("design", "auto"):
                    if unified_anchor_prompt is not None:
                        active_block_mode = "clone"
                        override_prompt = unified_anchor_prompt
                        override_ref = unified_anchor_wav
                        override_instruct = None
                    else:
                        active_block_mode = "design"
                        override_instruct = (
                            self.instruct
                            or getattr(blk, "voice_instruct", "")
                            or DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]
                        )
                        override_prompt = None
                        override_ref = None
                elif self.mode == "clone":
                    active_block_mode = "clone"
                    if self.current_prompt is not None:
                        override_prompt = self.current_prompt
                    elif self.ref_audio_path and os.path.exists(self.ref_audio_path):
                        override_prompt = self._get_or_create_prompt_for_audio(
                            self.ref_audio_path,
                            cache_key=f"unified_{self.ref_audio_path}",
                            ref_text=self.ref_text,
                        )
                    if override_prompt is None:
                        self.log("Cảnh báo: Không thể nạp đặc trưng giọng mẫu nhân bản trong chế độ một giọng.")
                        report.status = "failed"
                        report.message = f"Không thể khởi tạo mẫu giọng nhân bản cho chế độ một giọng từ tệp: {self.ref_audio_path or 'chưa có'}"
                        self.last_report = report
                        return report
            elif override_prompt is None:
                # 2b. Xử lý câu chưa xác định người nói hoặc đa nhân vật
                if not spk_key or spk_key == "Chưa xác định":
                    neighbor_spk = None
                    if i > 0 and getattr(blocks[i - 1], "speaker", "") not in ("", "Chưa xác định"):
                        neighbor_spk = getattr(blocks[i - 1], "speaker", "")
                    elif i + 1 < len(blocks) and getattr(blocks[i + 1], "speaker", "") not in ("", "Chưa xác định"):
                        neighbor_spk = getattr(blocks[i + 1], "speaker", "")

                    if neighbor_spk:
                        n_prof = self.speaker_profiles.get(neighbor_spk, {})
                        n_dict = isinstance(n_prof, dict)
                        n_ref = n_prof.get("ref_audio_path", "") or n_prof.get("source_audio_path", "") if n_dict else getattr(n_prof, "source_audio_path", "")
                        n_char_key = self._compute_character_prompt_key(neighbor_spk, n_ref)
                        n_prompt = self._speaker_prompt_cache.get(n_char_key) or self._speaker_prompt_cache.get(f"character_{neighbor_spk}")
                        n_wav = (
                            (self._speaker_prompt_wav_cache.get(n_char_key) or self._speaker_prompt_wav_cache.get(f"character_{neighbor_spk}"))
                            if hasattr(self, "_speaker_prompt_wav_cache")
                            else None
                        ) or n_ref
                        if n_prompt is not None:
                            override_prompt = n_prompt
                            override_ref = n_wav
                            active_block_mode = "clone"
                            spk_label = f"{neighbor_spk} - kế thừa"
                        else:
                            n_method = n_prof.get("method") if n_dict else getattr(n_prof, "method", "")
                            n_preset = n_prof.get("preset_name") if n_dict else getattr(n_prof, "preset_name", "")
                            n_ins = (
                                self._speaker_instruct_cache.get(neighbor_spk, "")
                                or (n_prof.get("instruct") or n_prof.get("voice_instruct") if n_dict else getattr(n_prof, "voice_instruct", ""))
                                or (VIETNAMESE_VOICE_CATALOG.get(n_preset, {}).get("instruct", "") if n_preset else "")
                                or DEFAULT_SPEAKER_PRESETS.get(neighbor_spk, {}).get("instruct", "")
                            )
                            if n_method == "preset" and n_ins:
                                n_prompt, n_wav = self._build_anchor_prompt(
                                    key=f"char_{neighbor_spk}",
                                    instruct=n_ins,
                                    sample_text=self._get_anchor_sample_text(n_preset),
                                )
                                if n_prompt is not None:
                                    self._speaker_prompt_cache[n_char_key] = n_prompt
                                    self._speaker_prompt_cache[f"character_{neighbor_spk}"] = n_prompt
                                    if n_wav and hasattr(self, "_speaker_prompt_wav_cache"):
                                        self._speaker_prompt_wav_cache[n_char_key] = n_wav
                                        self._speaker_prompt_wav_cache[f"character_{neighbor_spk}"] = n_wav
                                    override_prompt = n_prompt
                                    override_ref = n_wav
                                    active_block_mode = "clone"
                                    spk_label = f"{neighbor_spk} - kế thừa"
                            if override_prompt is None and n_ins:
                                override_instruct = n_ins
                                active_block_mode = "design"
                                spk_label = f"{neighbor_spk} - kế thừa"

                    if override_prompt is None and not override_instruct:
                        report.fallback_count += 1
                        fb_p, fb_w = fallback_anchor_prompt, fallback_anchor_wav
                        if fb_p is None:
                            fb_p, fb_w = self._get_or_create_fallback_anchor()
                        if fb_p is not None:
                            override_prompt = fb_p
                            override_ref = fb_w
                            active_block_mode = "clone"
                            spk_label = "Giọng mỏ neo dự phòng"
                        else:
                            override_instruct = self.instruct or DEFAULT_SPEAKER_PRESETS["Người nói 1"]["instruct"]
                            active_block_mode = "design"
                            spk_label = "Thiết kế giọng dự phòng"
                else:
                    prof = self.speaker_profiles.get(spk_key, {})
                    is_dict = isinstance(prof, dict)
                    prof_method = prof.get("method") if is_dict else getattr(prof, "method", "")

                    if prof_method == "preset":
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

                        anchor_key = self._compute_character_prompt_key(spk_key)
                        prompt = self._speaker_prompt_cache.get(anchor_key) or self._speaker_prompt_cache.get(f"character_{spk_key}")
                        wav = (
                            self._speaker_prompt_wav_cache.get(anchor_key)
                            or self._speaker_prompt_wav_cache.get(f"character_{spk_key}")
                            if hasattr(self, "_speaker_prompt_wav_cache")
                            else None
                        )
                        if prompt is None and ins_cand:
                            sample_text = self._get_anchor_sample_text(preset_name)
                            prompt, wav = self._build_anchor_prompt(
                                key=f"char_{spk_key}",
                                instruct=ins_cand,
                                sample_text=sample_text,
                            )
                            if prompt is not None:
                                self._speaker_prompt_cache[anchor_key] = prompt
                                self._speaker_prompt_cache[f"character_{spk_key}"] = prompt
                                if wav and hasattr(self, "_speaker_prompt_wav_cache"):
                                    self._speaker_prompt_wav_cache[anchor_key] = wav
                                    self._speaker_prompt_wav_cache[f"character_{spk_key}"] = wav

                        if prompt is not None:
                            active_block_mode = "clone"
                            override_prompt = prompt
                            override_ref = wav
                            override_instruct = None
                        else:
                            active_block_mode = "design"
                            override_ref = None
                            override_prompt = None
                            override_instruct = ins_cand
                    else:
                        ref_cand = getattr(blk, "voice_ref_path", "") or (prof.get("ref_audio_path", "") or prof.get("source_audio_path", "") if is_dict else getattr(prof, "source_audio_path", ""))
                        character_prompt_key = self._compute_character_prompt_key(spk_key, ref_cand)

                        if character_prompt_key and character_prompt_key in self._speaker_prompt_cache:
                            override_prompt = self._speaker_prompt_cache[character_prompt_key]
                            override_ref = ref_cand
                            active_block_mode = "clone"
                        elif f"character_{spk_key}" in self._speaker_prompt_cache:
                            override_prompt = self._speaker_prompt_cache[f"character_{spk_key}"]
                            override_ref = ref_cand
                            active_block_mode = "clone"
                        elif ref_cand and os.path.exists(ref_cand):
                            prompt_key = character_prompt_key or f"spk_{spk_key}_{ref_cand}"
                            prof_ref_text = prof.get("ref_text", "") if is_dict else getattr(prof, "ref_text", "")
                            override_prompt = self._get_or_create_prompt_for_audio(ref_cand, cache_key=prompt_key, ref_text=prof_ref_text or None)
                            if override_prompt is not None:
                                self._speaker_prompt_cache[character_prompt_key] = override_prompt
                                self._speaker_prompt_cache[f"character_{spk_key}"] = override_prompt
                            override_ref = ref_cand
                            active_block_mode = "clone"
                        elif spk_key and spk_key in self._speaker_instruct_cache:
                            override_instruct = self._speaker_instruct_cache[spk_key]
                            active_block_mode = "design"

                        # Nếu không có prompt nhân bản (hoặc tạo thất bại), luôn tạo và khóa mỏ neo cho nhân vật
                        if override_prompt is None:
                            ins_cand = (
                                override_instruct
                                or getattr(blk, "voice_instruct", "")
                                or (self._speaker_instruct_cache.get(spk_key, "") if spk_key else "")
                                or (prof.get("instruct", "") or prof.get("voice_instruct", "") if is_dict else getattr(prof, "voice_instruct", ""))
                            )
                            if not ins_cand and spk_key in DEFAULT_SPEAKER_PRESETS:
                                ins_cand = DEFAULT_SPEAKER_PRESETS[spk_key]["instruct"]

                            if not ins_cand and self.instruct:
                                ins_cand = self.instruct
                            elif not ins_cand:
                                preset_keys = list(VIETNAMESE_VOICE_CATALOG.keys())
                                cluster_num = 0
                                match = re.search(r"\d+", spk_key)
                                if match:
                                    cluster_num = int(match.group()) - 1
                                preset_name = preset_keys[cluster_num % len(preset_keys)]
                                ins_cand = VIETNAMESE_VOICE_CATALOG[preset_name]["instruct"]

                            anchor_key = self._compute_character_prompt_key(spk_key)
                            prompt = self._speaker_prompt_cache.get(anchor_key) or self._speaker_prompt_cache.get(f"character_{spk_key}")
                            wav = (
                                self._speaker_prompt_wav_cache.get(anchor_key)
                                or self._speaker_prompt_wav_cache.get(f"character_{spk_key}")
                                if hasattr(self, "_speaker_prompt_wav_cache")
                                else None
                            )
                            if prompt is None and ins_cand:
                                sample_text = self._get_anchor_sample_text(spk_key)
                                prompt, wav = self._build_anchor_prompt(
                                    key=f"char_{spk_key}",
                                    instruct=ins_cand,
                                    sample_text=sample_text,
                                )
                                if prompt is not None:
                                    self._speaker_prompt_cache[anchor_key] = prompt
                                    self._speaker_prompt_cache[f"character_{spk_key}"] = prompt
                                    if wav and hasattr(self, "_speaker_prompt_wav_cache"):
                                        self._speaker_prompt_wav_cache[anchor_key] = wav
                                        self._speaker_prompt_wav_cache[f"character_{spk_key}"] = wav

                            if prompt is not None:
                                active_block_mode = "clone"
                                override_prompt = prompt
                                override_ref = wav
                                override_instruct = None
                            else:
                                override_instruct = ins_cand
                                active_block_mode = "design"

            if active_block_mode == "design":
                method_desc = f"Thiết kế giọng: {override_instruct[:30]}..." if override_instruct else "Thiết kế giọng"
            elif override_ref and "anchor_" in os.path.basename(override_ref):
                method_desc = f"Mẫu mỏ neo đồng nhất: {os.path.basename(override_ref)}"
            else:
                method_desc = "Nhân bản giọng mẫu"
            # Kiểm tra bộ nhớ đệm khôi phục tác vụ từ điểm lưu trước
            cache_spk = "unified" if self.multi_speaker_mode == "single" else spk_key
            next_start_sec = parse_timestamp_to_seconds(blocks[i + 1].start_time) if i + 1 < len(blocks) else (start_sec + (target_dur or slot_duration))
            avail_before_next = max(0.0, next_start_sec - start_sec)
            cache_wav_path = self._compute_block_cache_path(
                i, text_to_speak, cache_spk, override_ref, override_instruct, target_dur, method_or_mode=active_block_mode, avail_before_next=avail_before_next
            )
            if cache_wav_path and os.path.exists(cache_wav_path):
                ok_cache, _, _, _ = validate_audio_file(cache_wav_path)
                if ok_cache:
                    cached_arr, c_sr = sf.read(cache_wav_path, dtype="float32")
                    if cached_arr.ndim > 1:
                        cached_arr = np.mean(cached_arr, axis=1)
                    nominal_sample = int(start_sec * self.sample_rate)
                    if self.overflow_strategy == "preserve_speech":
                        min_gap = int(0.05 * self.sample_rate)
                        max_drift = int(1.5 * self.sample_rate)
                        start_sample = min(max(nominal_sample, speech_cursor_sample + min_gap), nominal_sample + max_drift)
                    else:
                        start_sample = nominal_sample

                    full_audio = self._overlay_audio_segment(full_audio, cached_arr, start_sample, self.sample_rate)
                    speech_cursor_sample = start_sample + len(cached_arr)
                    report.succeeded_count += 1
                    report.resumed_from_cache_count += 1
                    generated_blocks_audio.append({
                        "index": i, "audio": cached_arr, "start_sample": start_sample, "spk_key": spk_key or spk_label,
                        "text": text_to_speak, "target_dur": target_dur, "mode": active_block_mode,
                        "prompt": override_prompt, "ref": override_ref, "instruct": override_instruct,
                    })
                    self.log(f"Khôi phục từ bộ nhớ đệm [{i + 1}/{len(blocks)}] ({spk_label} · {method_desc}): {text_to_speak}")
                    continue

            self.log(f"Đang tổng hợp [{i + 1}/{len(blocks)}] ({spk_label} · {method_desc}): {text_to_speak}")
            try:
                clauses = split_long_vietnamese_clauses(text_to_speak, max_chars=220)
                eff_target_dur = target_dur if (self.sync_to_subtitle_duration and target_dur and target_dur > 0.4) else None
                if len(clauses) == 1:
                    audio_arr = self.generate_single(
                        clauses[0],
                        target_duration=eff_target_dur,
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
                        c_target = (
                            round(eff_target_dur * (len(clause) / total_chars), 2)
                            if eff_target_dur
                            else None
                        )
                        c_wav = self.generate_single(
                            clause,
                            target_duration=c_target,
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

                # Tỉa khoảng lặng đệm vô thanh ở đầu/đuôi trước khi kiểm tra thời lượng
                trimmed_arr = self._trim_silence_padding(audio_arr, threshold_db=-45.0, sample_rate=self.sample_rate)
                if len(trimmed_arr) > 0:
                    audio_arr = trimmed_arr

                # Kiểm tra chồng lấn với câu phụ đề kế tiếp và co giãn thời lượng trong ngưỡng an toàn
                avail_before_next = None
                if i + 1 < len(blocks):
                    next_start_sec = parse_timestamp_to_seconds(blocks[i + 1].start_time)
                    avail_before_next = next_start_sec - start_sec
                elif target_dur and self.sync_to_subtitle_duration:
                    avail_before_next = target_dur

                if avail_before_next is not None and avail_before_next > 0.4:
                    cur_dur_sec = len(audio_arr) / float(self.sample_rate)
                    if cur_dur_sec > avail_before_next * 1.05:
                        needed_speed = cur_dur_sec / avail_before_next
                        max_stretch = 1.35 if self.overflow_strategy == "strict_sync" else 1.25
                        if needed_speed <= max_stretch:
                            audio_arr = self._safe_time_stretch(audio_arr, needed_speed, sample_rate=self.sample_rate)
                            report.stretched_count += 1
                        else:
                            # Co giãn thời lượng an toàn bằng WSOLA giữ nguyên cao độ và ngữ điệu tự nhiên
                            audio_arr = self._safe_time_stretch(audio_arr, max_stretch, sample_rate=self.sample_rate)
                            report.stretched_count += 1
                            stretched_dur = len(audio_arr) / float(self.sample_rate)
                            overflow_sec = max(0.0, stretched_dur - avail_before_next)
                            if overflow_sec > 0.05:
                                overflow_info = {
                                    "line": i + 1,
                                    "available_sec": round(avail_before_next, 2),
                                    "original_dur_sec": round(cur_dur_sec, 2),
                                    "stretched_dur_sec": round(stretched_dur, 2),
                                    "overflow_sec": round(overflow_sec, 2),
                                    "text": text_to_speak,
                                }
                                report.overflow_lines.append(overflow_info)
                                self.log(
                                    f"Câu {i + 1}: Đoạn đọc vượt mốc thời gian {round(overflow_sec, 2)} giây "
                                    f"(khả dụng: {round(avail_before_next, 2)}s, sau co giãn: {round(stretched_dur, 2)}s)."
                                )
                                if self.overflow_strategy == "strict_sync":
                                    # Chế độ ưu tiên khớp timeline: cắt nhẹ phần dư và làm mờ đuôi âm mượt mà
                                    target_max_samples = int(avail_before_next * self.sample_rate)
                                    fade_len = int(min(0.08 * self.sample_rate, target_max_samples * 0.1))
                                    if len(audio_arr) > target_max_samples:
                                        audio_arr = audio_arr[:target_max_samples]
                                        if fade_len > 0:
                                            fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                                            audio_arr[-fade_len:] *= fade_curve

                # Lưu vào bộ nhớ đệm khôi phục tác vụ nếu được bật
                if cache_wav_path:
                    try:
                        sf.write(cache_wav_path, audio_arr, self.sample_rate)
                    except Exception:
                        pass

                nominal_sample = int(start_sec * self.sample_rate)
                if self.overflow_strategy == "preserve_speech":
                    min_gap = int(0.05 * self.sample_rate)
                    max_drift = int(1.5 * self.sample_rate)
                    start_sample = min(max(nominal_sample, speech_cursor_sample + min_gap), nominal_sample + max_drift)
                else:
                    start_sample = nominal_sample

                full_audio = self._overlay_audio_segment(full_audio, audio_arr, start_sample, self.sample_rate)
                speech_cursor_sample = start_sample + len(audio_arr)
                report.succeeded_count += 1
                generated_blocks_audio.append({
                    "index": i, "audio": audio_arr, "start_sample": start_sample, "spk_key": spk_key or spk_label,
                    "text": text_to_speak, "target_dur": target_dur, "mode": active_block_mode,
                    "prompt": override_prompt, "ref": override_ref, "instruct": override_instruct,
                })

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

        # GIAI ĐOẠN 3.5: Kiểm định tính nhất quán âm sắc bằng mô hình CAM++ và tự động tái tạo câu lệch chuẩn
        if (
            not self._cancel_requested
            and getattr(self, "speaker_model_path", "")
            and os.path.exists(self.speaker_model_path)
            and len(generated_blocks_audio) >= 2
        ):
            try:
                full_audio = self._verify_and_refine_voice_consistency(
                    generated_blocks_audio=generated_blocks_audio,
                    full_audio=full_audio,
                    report=report,
                )
            except Exception as verif_err:
                self.log(f"Kiểm tra tính nhất quán CAM++ bỏ qua: {verif_err}")

        # Chuẩn hóa âm lượng đầu ra an toàn (-1.0 dBFS / đỉnh tối đa 0.891) tránh vỡ tiếng khi ghép video
        max_peak = float(np.max(np.abs(full_audio))) if full_audio.size > 0 else 0.0
        if max_peak > 0.891:
            # Áp dụng bộ nén soft-knee limiter mượt mà thay vì thu nhỏ tuyến tính toàn dải
            thresh = 0.85
            over = np.abs(full_audio) > thresh
            if np.any(over):
                scale = np.ones_like(full_audio)
                diff = np.abs(full_audio[over]) - thresh
                scale[over] = (thresh + 0.041 * np.tanh(diff / max(1e-4, max_peak - thresh))) / np.abs(full_audio[over])
                full_audio = full_audio * scale
            max_p2 = float(np.max(np.abs(full_audio)))
            if max_p2 > 0.891:
                full_audio = full_audio * (0.891 / max_p2)

        # Cắt bớt khoảng lặng đệm thừa ở đuôi toàn bộ bản thu
        active_indices = np.where(np.abs(full_audio) > 1e-4)[0]
        last_speech_sample = int(active_indices[-1]) + int(0.5 * self.sample_rate) if len(active_indices) > 0 else 0

        if video_path and total_duration > 0.0:
            target_samples = int(total_duration * self.sample_rate)
            if len(full_audio) > target_samples:
                full_audio = full_audio[:target_samples]
            elif len(full_audio) < target_samples:
                full_audio = np.pad(full_audio, (0, target_samples - len(full_audio)))
        elif last_speech_sample > 0 and last_speech_sample < len(full_audio):
            clean_end = max(int(total_duration * self.sample_rate), last_speech_sample)
            if clean_end < len(full_audio):
                full_audio = full_audio[:clean_end]

        self.log(f"Đang lưu track lồng tiếng hoàn chỉnh vào: {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, full_audio, self.sample_rate)

        is_valid, wav_dur, wav_rms, wav_peak = validate_audio_file(output_path)
        report.valid_audio_file = is_valid
        report.audio_duration_sec = wav_dur
        report.rms_energy = wav_rms
        report.peak_amplitude = wav_peak

        total_ok = max(1, report.succeeded_count)
        report.compressed_ratio = round(float(report.stretched_count) / float(total_ok), 4)

        # Tính điểm chất lượng tổng thể theo thang điểm 0 - 100
        calc_score = 100.0
        if report.total_blocks > 0:
            calc_score -= (float(report.failed_count) / float(report.total_blocks)) * 50.0
        calc_score -= min(30.0, report.compressed_ratio * 30.0)
        if report.speaker_consistency_score < 1.0:
            calc_score -= min(20.0, max(0.0, (1.0 - report.speaker_consistency_score) * 40.0))
        report.quality_score = round(max(0.0, min(100.0, calc_score)), 1)

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

    def _verify_and_refine_voice_consistency(
        self,
        generated_blocks_audio: List[Dict[str, Any]],
        full_audio: np.ndarray,
        report: DubbingReport,
        min_sim: float = 0.65,
    ) -> np.ndarray:
        """
        Kiểm định tính nhất quán âm sắc bằng mô hình CAM++ và tự động tái tạo câu lệch chuẩn:
        - Gom nhóm các câu thoại theo từng nhân vật (hoặc giọng đồng nhất)
        - Trích xuất embedding cho các câu có thời lượng >= 0.8 giây
        - Tính vector trọng tâm (centroid) âm sắc chuẩn hóa L2 của nhân vật
        - Đo độ tương đồng cosin của từng câu với centroid
        - Tự động tổng hợp lại tối đa 2 lần với seed khác đối với các câu có độ tương đồng < min_sim
        """
        from .speaker_embedding_onnx import create_speaker_session, extract_speaker_embedding, close_speaker_session

        session = None
        try:
            session_obj = create_speaker_session(self.speaker_model_path, prefer_cuda=False, allow_mock=self.allow_mock)
            if session_obj is None:
                return full_audio
            session = session_obj[0] if isinstance(session_obj, tuple) else session_obj
            if session is None:
                return full_audio

            is_single_mode = getattr(self, "multi_speaker_mode", "single") == "single"
            spk_groups: Dict[str, List[Dict[str, Any]]] = {}
            for rec in generated_blocks_audio:
                s_key = "single_unified_speaker" if is_single_mode else (rec.get("spk_key") or "default")
                spk_groups.setdefault(s_key, []).append(rec)

            all_sims: List[float] = []

            for s_key, items in spk_groups.items():
                if len(items) < 2:
                    continue

                valid_items = [it for it in items if len(it["audio"]) >= int(0.8 * self.sample_rate)]
                if len(valid_items) < 2:
                    continue

                embs = []
                for it in valid_items:
                    emb = extract_speaker_embedding(session, it["audio"], sample_rate=self.sample_rate)
                    embs.append(emb)

                centroid = np.mean(np.stack(embs, axis=0), axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm < 1e-6:
                    continue
                centroid = centroid / norm

                for idx_it, it in enumerate(valid_items):
                    emb = embs[idx_it]
                    sim = float(np.dot(emb, centroid))
                    self.log(f"Độ tương đồng âm sắc câu {it['index'] + 1} của {s_key}: {round(sim, 3)}")
                    effective_sim = sim
                    if sim < min_sim:
                        self.log(
                            f"Phát hiện câu {it['index'] + 1} có âm sắc lệch chuẩn ({round(sim, 2)} < {min_sim}). "
                            "Đang tự động tái tạo với hạt số mới..."
                        )
                        best_audio = it["audio"]
                        best_sim = sim
                        for retry in range(2):
                            if self._cancel_requested:
                                break
                            r_seed = 10007 + retry * 7919 + it["index"] * 101

                            slot_dur = it.get("slot_dur", 0.0)
                            new_arr = self.generate_single(
                                it["text"],
                                target_duration=slot_dur if slot_dur > 0.4 else None,
                                override_mode=it["mode"],
                                override_prompt=it["prompt"],
                                override_ref_audio=it["ref"],
                                override_instruct=it["instruct"],
                                seed=r_seed,
                            )
                            if len(new_arr) > 0:
                                new_arr = np.squeeze(np.asarray(new_arr, dtype=np.float32))
                                trimmed = self._trim_silence_padding(new_arr, sample_rate=self.sample_rate)
                                if len(trimmed) > 0:
                                    new_arr = trimmed
                                if slot_dur > 0.4 and (len(new_arr) / float(self.sample_rate)) > slot_dur and getattr(self, "overflow_strategy", "") == "strict_sync":
                                    new_arr = self._stretch_audio_wsola(new_arr, slot_dur, sample_rate=self.sample_rate)
                                if len(new_arr) >= int(0.5 * self.sample_rate):
                                    new_emb = extract_speaker_embedding(session, new_arr, sample_rate=self.sample_rate)
                                    new_sim = float(np.dot(new_emb, centroid))
                                    if new_sim > best_sim:
                                        best_sim = new_sim
                                        best_audio = new_arr
                                        if new_sim >= min_sim:
                                            break

                        if best_sim > sim:
                            old_len = len(it["audio"])
                            old_start = it["start_sample"]
                            full_audio[old_start : min(len(full_audio), old_start + old_len)] = 0.0
                            full_audio = self._overlay_audio_segment(full_audio, best_audio, old_start, self.sample_rate)
                            it["audio"] = best_audio
                            report.inconsistent_retried_count += 1
                            effective_sim = best_sim
                            self.log(
                                f"Tái tạo câu {it['index'] + 1} thành công: "
                                f"Độ tương đồng âm sắc nâng từ {round(sim, 2)} lên {round(best_sim, 2)}."
                            )
                    all_sims.append(effective_sim)

            if all_sims:
                report.speaker_consistency_score = round(float(np.mean(all_sims)), 3)
        finally:
            if session is not None:
                close_speaker_session(session)
        return full_audio

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
        mode: Optional[str] = None,
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
        effective_ducking_mode = mode or ducking_mode or "dynamic"

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
            try:
                subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            return os.path.exists(video_out) and os.path.getsize(video_out) > 0

        if effective_ducking_mode == "dynamic":
            # Thử nghiệm hòa trộn ducking động chất lượng cao giữ nguyên stereo 48 kHz bằng sidechaincompress
            sidechain_ratio = max(2.0, min(10.0, 1.0 / max(0.05, original_volume)))
            cmd_sidechain = [
                ffmpeg_bin, "-y",
                "-i", video_in,
                "-i", tts_audio_in,
                "-filter_complex",
                f"[1:a]asplit=2[sc][tts_main];"
                f"[0:a][sc]sidechaincompress=threshold=0.08:ratio={sidechain_ratio:.1f}:attack=80:release=350[ducked_bg];"
                f"[ducked_bg][tts_main]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.96[outa]",
                "-map", "0:v", "-map", "[outa]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                video_out,
            ]
            try:
                res_sc = subprocess.run(cmd_sidechain, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res_sc.returncode == 0 and os.path.exists(video_out) and os.path.getsize(video_out) > 0:
                    return True
            except Exception:
                pass

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
            f"[0:a]volume={original_volume}[a0]; [1:a]volume=1.0[a1]; [a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.96[outa]",
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
            try:
                subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
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


_ACCENT_PATTERN = re.compile(r",\s*\w+\s+accent", re.IGNORECASE)


def get_effective_instruct(instruct: str) -> str:
    """Loại bỏ các thành phần không có tác dụng âm học đối với tiếng Việt (như accent)."""
    cleaned = _ACCENT_PATTERN.sub("", instruct or "")
    parts = [p.strip().lower() for p in cleaned.split(",") if p.strip()]
    return ", ".join(parts)


def build_distinct_voice_pool(
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    allow_child: bool = False,
    allow_whisper: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Lọc danh mục giọng đọc để bảo đảm mọi preset đều có instruct hiệu dụng (effective instruct)
    hoàn toàn khác biệt, loại bỏ các preset trùng instruct hoặc các biến thể không cần thiết.
    """
    target = catalog if catalog is not None else VIETNAMESE_VOICE_CATALOG
    filtered_pool: Dict[str, Dict[str, Any]] = {}
    seen_effective_instructs = set()

    for preset_name, info in target.items():
        ins = info.get("instruct", "")
        style = info.get("style", "")
        age = info.get("age_group", "")

        # 1. Lọc giọng thì thầm nếu không cho phép
        if not allow_whisper:
            if "whisper" in ins.lower() or style == "thi_tham":
                continue

        # 2. Lọc giọng trẻ em nếu không cho phép
        if not allow_child:
            if age == "child" or "child" in ins.lower():
                continue

        eff_ins = get_effective_instruct(ins)
        if eff_ins not in seen_effective_instructs:
            seen_effective_instructs.add(eff_ins)
            filtered_pool[preset_name] = info

    return filtered_pool


class VoiceRecommendationEngine:
    """Động cơ tính toán và xếp hạng giọng đọc phù hợp cho từng nhân vật dựa trên đặc tính âm thanh và ngữ cảnh."""

    PITCH_LEVELS = ["very_low", "low", "moderate", "high", "very_high"]

    @classmethod
    def calculate_voice_match(cls, profile: Any, voice_entry: Dict[str, Any]) -> Tuple[int, str]:
        score = 0
        reasons: List[str] = []

        # 1. Giới tính cảm nhận (Trọng số 25% / 25 điểm)
        prof_gender = getattr(profile, "gender", None) or (profile.get("gender") if isinstance(profile, dict) else "auto")
        voice_gender = voice_entry.get("gender", "")
        gender_conf = getattr(profile, "gender_confidence", 0.5) if not isinstance(profile, dict) else profile.get("gender_confidence", 0.5)

        if prof_gender in ("auto", "", None, "unknown"):
            score += 20
        elif prof_gender == voice_gender:
            score += 25
            reasons.append("giới tính phù hợp")
        else:
            if gender_conf >= 0.75:
                return 0, "Khác giới tính"
            score += 5

        # 2. Âm sắc (Trọng số 25% / 25 điểm) - Tương thích về nhóm tuổi và phong cách thể hiện
        prof_age = getattr(profile, "age_group", None) or (profile.get("age_group") if isinstance(profile, dict) else "young_adult")
        voice_age = voice_entry.get("age_group", "young_adult")
        prof_style = getattr(profile, "style_tag", None) or (profile.get("style_tag") if isinstance(profile, dict) else "natural")
        voice_style = voice_entry.get("style", "")

        # Điểm nhóm tuổi (tối đa 15 điểm)
        if prof_age == voice_age:
            score += 15
            reasons.append("nhóm tuổi phù hợp")
        elif {prof_age, voice_age}.issubset({"teenager", "young_adult"}) or {prof_age, voice_age}.issubset({"young_adult", "middle_aged"}):
            score += 10
        elif {prof_age, voice_age}.issubset({"middle_aged", "elderly"}):
            score += 8
        elif prof_age == "child" or voice_age == "child":
            score += 0
        else:
            score += 4

        # Điểm phong cách thể hiện (tối đa 10 điểm)
        if prof_style and voice_style and prof_style == voice_style:
            score += 10
            reasons.append("phong cách tương đồng")
        else:
            score += 7

        # Phạt nặng preset thì thầm hoặc trẻ em nếu nhân vật không yêu cầu
        if voice_style == "thi_tham" and prof_style != "thi_tham":
            score -= 30
        if voice_age == "child" and prof_age != "child":
            score -= 50

        # 3. Cao độ âm thanh (Trọng số 20% / 20 điểm)
        prof_pitch = getattr(profile, "pitch_level", None) or (profile.get("pitch_level") if isinstance(profile, dict) else "moderate")
        voice_pitch = voice_entry.get("pitch", "moderate")
        if prof_pitch == voice_pitch:
            score += 20
            reasons.append("cao độ phù hợp")
        elif prof_pitch in cls.PITCH_LEVELS and voice_pitch in cls.PITCH_LEVELS:
            diff = abs(cls.PITCH_LEVELS.index(prof_pitch) - cls.PITCH_LEVELS.index(voice_pitch))
            if diff == 1:
                score += 14
            elif diff == 2:
                score += 7

        # 4. Accent / Phương ngữ tiếng Việt (Trọng số 15% / 15 điểm)
        prof_accent = getattr(profile, "vietnamese_accent", None) or (profile.get("vietnamese_accent") if isinstance(profile, dict) else "")
        voice_accent = voice_entry.get("accent", "")
        if prof_accent and voice_accent and prof_accent == voice_accent:
            score += 15
            accent_names = {"bac": "miền Bắc", "trung": "miền Trung", "nam": "miền Nam"}
            reasons.append(f"giọng {accent_names.get(voice_accent, voice_accent)}")
        elif not prof_accent or prof_accent in ("auto", "unknown"):
            score += 10
        else:
            score += 5

        # 5. Tốc độ nói (Trọng số 15% / 15 điểm)
        speech_rate = getattr(profile, "speech_rate_sps", 0.0) or (profile.get("speech_rate_sps", 0.0) if isinstance(profile, dict) else 0.0)
        if not speech_rate:
            ac_f = getattr(profile, "acoustic_features", {}) or (profile.get("acoustic_features", {}) if isinstance(profile, dict) else {})
            speech_rate = ac_f.get("speech_rate_sps", 0.0)

        if speech_rate > 5.5:
            if voice_style in ("soi_noi",) or voice_age == "teenager":
                score += 15
                reasons.append("tốc độ phát âm nhanh phù hợp")
            else:
                score += 8
        elif 0 < speech_rate < 3.5:
            if voice_style in ("ke_chuyen", "tam_su") or voice_age in ("middle_aged", "elderly"):
                score += 15
                reasons.append("tốc độ nói thong thả phù hợp")
            else:
                score += 8
        else:
            score += 12

        score = min(100, max(0, score))
        desc = f"Phù hợp {score}% ({', '.join(reasons)})" if reasons else f"Phù hợp {score}%"
        return score, desc

    @classmethod
    def recommend_voices(
        cls,
        profile: Any,
        catalog: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        target_catalog = catalog if catalog is not None else VIETNAMESE_VOICE_CATALOG
        candidates: List[Dict[str, Any]] = []

        for preset_name, info in target_catalog.items():
            sc, reason = cls.calculate_voice_match(profile, info)
            if sc > 0:
                item = dict(info)
                item["preset_name"] = preset_name
                item["score"] = sc
                item["reason"] = reason
                candidates.append(item)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


class AcousticClassifier:
    """
    Bộ phân loại âm học đa chiều (Multi-Feature Acoustic Classifier).
    Tách biệt độc lập hoàn toàn giữa Giới tính (gender) và Nhóm tuổi (age_group).
    Phân tích kết hợp F0 median, F0 IQR/variance, Trọng tâm phổ (Spectral Centroid)
    và 8 dải năng lượng âm sắc để phân biệt chính xác:
    - Giọng nữ trầm vs Giọng nam cao ở vùng giao thoa (155 - 190 Hz)
    - Giọng trẻ em (F0 > 250 Hz + centroid cao) tách bạch giới tính
    - Giọng người già / trung niên / thanh niên / thiếu niên
    """

    @classmethod
    def classify(
        cls,
        f0_list: List[float],
        centroid: float = 1500.0,
        iqr_f0: float = 15.0,
        band_energies: Optional[np.ndarray] = None,
        allow_child: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not f0_list:
            return {
                "gender": "male",
                "gender_confidence": 0.5,
                "age_group": "young_adult",
                "pitch_level": "moderate",
                "pitch_hz": 150.0,
                "style_tag": "natural",
                "review_needed": False,
            }

        median_f0 = float(np.median(f0_list))
        pitch_hz = round(median_f0, 1)

        # Tính tỷ lệ năng lượng dải thấp (100-450Hz: trầm) so với dải cao (1500-3800Hz: thanh/sáng)
        low_high_ratio = 1.0
        if band_energies is not None and len(band_energies) >= 8:
            low_energy = float(np.sum(band_energies[0:2]))
            high_energy = float(np.sum(band_energies[5:8])) + 1e-6
            low_high_ratio = low_energy / high_energy

        # 1. Phát hiện Giọng Trẻ Em (Child)
        if allow_child is None:
            is_child = (median_f0 >= 250.0 and centroid > 2350.0) or (median_f0 >= 285.0)
        elif allow_child is True:
            is_child = (median_f0 >= 240.0 and centroid > 2000.0) or (median_f0 >= 265.0)
        else:
            is_child = False

        if is_child:
            age_group = "child"
            pitch_level = "very_high" if median_f0 > 275.0 else "high"
            style_tag = "hoat_hinh"
            if low_high_ratio < 0.9 or centroid > 2300.0:
                gender = "female"
            else:
                gender = "male"
            gender_confidence = 0.70
            return {
                "gender": gender,
                "gender_confidence": gender_confidence,
                "age_group": age_group,
                "pitch_level": pitch_level,
                "pitch_hz": pitch_hz,
                "style_tag": style_tag,
                "review_needed": True,
            }

        # 2. Phân loại Giới tính cho Người lớn (Adult Gender Classification)
        # Giọng nam tiếng Việt có thể đạt 165 Hz khi nói câu hỏi hoặc thanh điệu cao (dấu sắc/ngã)
        if median_f0 < 165.0:
            gender = "male"
            gender_confidence = min(0.98, 0.82 + (165.0 - median_f0) / 100.0)
        elif median_f0 > 185.0:
            gender = "female"
            gender_confidence = min(0.98, 0.82 + (median_f0 - 185.0) / 100.0)
        else:
            # Vùng giao thoa F0 165 - 185 Hz:
            # Chỉ coi là nữ khi thiếu hụt năng lượng lồng ngực (low_high_ratio <= 1.15) VÀ trọng tâm phổ thanh sáng (> 1650 Hz)
            if centroid >= 1650.0 and low_high_ratio <= 1.15:
                gender = "female"
            else:
                gender = "male"
            gender_confidence = 0.70

        # 3. Phân loại Nhóm tuổi cho Người lớn (Adult Age Group Classification)
        if gender == "male":
            if (median_f0 < 105.0 and centroid < 1000.0) or (median_f0 < 95.0 and iqr_f0 > 25.0):
                age_group = "elderly"
                pitch_level = "very_low"
                style_tag = "ke_chuyen"
            elif median_f0 < 120.0 and (centroid < 1300.0 or iqr_f0 > 22.0):
                age_group = "middle_aged"
                pitch_level = "low"
                style_tag = "ke_chuyen"
            elif centroid > 1900.0 and iqr_f0 > 22.0 and median_f0 > 155.0:
                age_group = "teenager"
                pitch_level = "high"
                style_tag = "soi_noi"
            else:
                age_group = "young_adult"
                pitch_level = "low" if median_f0 < 120.0 else ("high" if median_f0 > 155.0 else "moderate")
                style_tag = "thuyet_minh"
        else:  # female
            if (median_f0 < 165.0 and centroid < 1300.0) or (median_f0 < 155.0 and iqr_f0 > 25.0):
                age_group = "elderly"
                pitch_level = "low"
                style_tag = "ke_chuyen"
            elif median_f0 < 185.0 and (centroid < 1600.0 or iqr_f0 > 20.0):
                age_group = "middle_aged"
                pitch_level = "low"
                style_tag = "ke_chuyen"
            elif centroid > 2100.0 and iqr_f0 > 22.0 and median_f0 > 225.0:
                age_group = "teenager"
                pitch_level = "high"
                style_tag = "soi_noi"
            else:
                age_group = "young_adult"
                pitch_level = "high" if median_f0 > 215.0 else "moderate"
                style_tag = "tam_su"

        return {
            "gender": gender,
            "gender_confidence": round(gender_confidence, 2),
            "age_group": age_group,
            "pitch_level": pitch_level,
            "pitch_hz": pitch_hz,
            "style_tag": style_tag,
            "review_needed": gender_confidence < 0.80,
        }


class VietnameseAccentClassifier:
    """
    Bộ nhận diện phương ngữ tiếng Việt (Bắc, Trung, Nam).
    Phân tích từ vựng khẩu ngữ vùng miền trong câu thoại phụ đề.
    """

    NAM_MARKERS = re.compile(
        r"\b(hổng|nghen|nè|bển|bắp|ghe|xỉn|bự|ráng|nhậu|tía|má|quẹo|chánh|heo|xíu|dzui|hén|trển|ba má|bận áo|muỗng|thiệt|mắc cười|hổng chừng|dữ dằn|coi chừng|rồi nè|nghe nè|vầy nè|trỏng|miệt|uổng|nhóc|chèn ơi|trời đất ơi|hông|dạ thưa|bữa nay|ngon lành)\b",
        re.IGNORECASE,
    )

    TRUNG_MARKERS = re.compile(
        r"\b(mô|tê|răng|rứa|chi|nớ|chừ|trôông|mi|tau|hỉ|hè|đọi|chộ|bầy tui|mần chi|mô tê|rứa hè|ưng|mần răng|răng rứa|chi rứa|mô rồi|nỏ|o|mệ|trôông đợi|lận)\b",
        re.IGNORECASE,
    )

    BAC_MARKERS = re.compile(
        r"\b(thế à|thế ư|nhé|nhỉ|giùm|ngô|thìa|bố mẹ|vâng|bảo|nhớn|lợn|cốc|đỗ|chè|dạ vâng|chứ lị|đấy chứ|nhá|chăng|thế phỏng|chứ nị|giời|u|bu|bảo sao|đấy thây)\b",
        re.IGNORECASE,
    )

    @classmethod
    def classify(cls, texts: List[str]) -> Tuple[str, float]:
        if not texts:
            return "bac", 0.50

        combined = " ".join(texts)
        nam_count = len(cls.NAM_MARKERS.findall(combined))
        trung_count = len(cls.TRUNG_MARKERS.findall(combined))
        bac_count = len(cls.BAC_MARKERS.findall(combined))

        if trung_count > 0 and trung_count >= nam_count and trung_count >= bac_count:
            conf = min(0.98, 0.65 + 0.1 * trung_count)
            return "trung", round(conf, 2)
        elif nam_count > 0 and nam_count >= bac_count:
            conf = min(0.98, 0.65 + 0.1 * nam_count)
            return "nam", round(conf, 2)
        elif bac_count > 0:
            conf = min(0.98, 0.65 + 0.1 * bac_count)
            return "bac", round(conf, 2)

        return "bac", 0.50


class GlobalVoiceAllocator:
    """
    Thuật toán phân bổ giọng nói toàn cục không trùng lặp (Non-Colliding Global Voice Allocator).
    Đảm bảo 100% mỗi nhân vật được gán một giọng đọc AI tiếng Việt riêng biệt từ danh mục giọng độc nhất.
    """

    @classmethod
    def allocate_distinct_voices(
        cls,
        profiles: List[Any],
        subtitle_blocks: Optional[List[Any]] = None,
        catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        if not profiles:
            return {}

        cat = catalog if catalog is not None else VIETNAMESE_VOICE_CATALOG

        # Kiểm tra xem có nhân vật nào là trẻ em hoặc thì thầm không
        has_child = any(
            (getattr(p, "age_group", "") or (p.get("age_group") if isinstance(p, dict) else "")) == "child"
            for p in profiles
        )
        has_whisper = any(
            (getattr(p, "style_tag", "") or (p.get("style_tag") if isinstance(p, dict) else "")) == "thi_tham"
            for p in profiles
        )

        distinct_pool = build_distinct_voice_pool(cat, allow_child=has_child, allow_whisper=has_whisper)
        catalog_names = list(distinct_pool.keys()) if distinct_pool else list(cat.keys())

        # Thu thập câu thoại theo nhân vật để nhận diện phương ngữ
        speaker_texts: Dict[str, List[str]] = {}
        if subtitle_blocks:
            for b in subtitle_blocks:
                spk = getattr(b, "speaker", "") or (b.get("speaker") if isinstance(b, dict) else "")
                txt = getattr(b, "text", "") or (b.get("text") if isinstance(b, dict) else "")
                if spk and txt:
                    speaker_texts.setdefault(spk, []).append(txt)

        # 1. Bảo vệ các nhân vật đã được người dùng khóa thủ công (user_locked)
        assigned: Dict[str, str] = {}
        used_presets = set()

        for prof in profiles:
            spk_name = getattr(prof, "display_name", "") or getattr(prof, "speaker_id", "") or (prof.get("display_name") if isinstance(prof, dict) else "") or (prof.get("speaker_id") if isinstance(prof, dict) else "")
            is_locked = bool(getattr(prof, "user_locked", False) if hasattr(prof, "user_locked") else (prof.get("user_locked", False) if isinstance(prof, dict) else False))
            curr_preset = getattr(prof, "preset_name", "") or (prof.get("preset_name") if isinstance(prof, dict) else "")
            curr_method = getattr(prof, "method", "") or (prof.get("method") if isinstance(prof, dict) else "")
            if curr_method == "clone" and (getattr(prof, "source_audio_path", "") or (isinstance(prof, dict) and prof.get("source_audio_path"))):
                assigned[spk_name] = ""
                continue
            if is_locked and curr_preset in cat:
                assigned[spk_name] = curr_preset
                used_presets.add(curr_preset)

        unassigned_profs = [
            p for p in profiles
            if (getattr(p, "display_name", "") or getattr(p, "speaker_id", "") or (p.get("display_name") if isinstance(p, dict) else "") or (p.get("speaker_id") if isinstance(p, dict) else "")) not in assigned
        ]

        if not unassigned_profs:
            return assigned

        available_presets = [k for k in catalog_names if k not in used_presets]
        if not available_presets:
            available_presets = [k for k in list(cat.keys()) if k not in used_presets]
        if not available_presets:
            available_presets = list(cat.keys())

        # Cập nhật phương ngữ cho các nhân vật chưa khóa
        for prof in unassigned_profs:
            spk_name = getattr(prof, "display_name", "") or getattr(prof, "speaker_id", "") or (prof.get("display_name") if isinstance(prof, dict) else "") or (prof.get("speaker_id") if isinstance(prof, dict) else "")
            if spk_name in speaker_texts and (not getattr(prof, "vietnamese_accent", "") or getattr(prof, "accent_confidence", 0.0) <= 0.5):
                acc, acc_conf = VietnameseAccentClassifier.classify(speaker_texts[spk_name])
                if hasattr(prof, "vietnamese_accent"):
                    prof.vietnamese_accent = acc
                    prof.accent_confidence = acc_conf
                elif isinstance(prof, dict):
                    prof["vietnamese_accent"] = acc
                    prof["accent_confidence"] = acc_conf

        def _allocate_group(group_profs: List[Any], candidate_presets: List[str]):
            if not group_profs:
                return
            n_grp = len(group_profs)
            n_cand = len(candidate_presets)

            if n_grp <= n_cand:
                cost_mat = np.zeros((n_grp, n_cand), dtype=np.float32)
                for i, prof in enumerate(group_profs):
                    for j, preset_name in enumerate(candidate_presets):
                        info = cat[preset_name]
                        sc, _ = VoiceRecommendationEngine.calculate_voice_match(prof, info)
                        cost_mat[i, j] = 100.0 - float(sc)
                try:
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(cost_mat)
                    for r, c in zip(row_ind, col_ind):
                        spk_name = getattr(group_profs[r], "display_name", "") or getattr(group_profs[r], "speaker_id", "") or (group_profs[r].get("display_name") if isinstance(group_profs[r], dict) else "")
                        assigned_preset = candidate_presets[c]
                        assigned[spk_name] = assigned_preset
                        used_presets.add(assigned_preset)
                        cls._apply_preset_to_profile(group_profs[r], assigned_preset, cat[assigned_preset])
                except Exception:
                    assigned_col = set()
                    for i, prof in enumerate(group_profs):
                        best_j = -1
                        best_cost = 999999.0
                        for j in range(n_cand):
                            if j not in assigned_col and cost_mat[i, j] < best_cost:
                                best_cost = cost_mat[i, j]
                                best_j = j
                        if best_j != -1:
                            assigned_col.add(best_j)
                            spk_name = getattr(prof, "display_name", "") or getattr(prof, "speaker_id", "") or (prof.get("display_name") if isinstance(prof, dict) else "")
                            p_name = candidate_presets[best_j]
                            assigned[spk_name] = p_name
                            used_presets.add(p_name)
                            cls._apply_preset_to_profile(prof, p_name, cat[p_name])
            else:
                # n_grp > n_cand: Phân bổ n_cand nhân vật đầu tiên tối ưu
                cost_mat = np.zeros((n_cand, n_cand), dtype=np.float32)
                for i in range(n_cand):
                    for j in range(n_cand):
                        info = cat[candidate_presets[j]]
                        sc, _ = VoiceRecommendationEngine.calculate_voice_match(group_profs[i], info)
                        cost_mat[i, j] = 100.0 - float(sc)
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                for r, c in zip(row_ind, col_ind):
                    spk_name = getattr(group_profs[r], "display_name", "") or getattr(group_profs[r], "speaker_id", "") or (group_profs[r].get("display_name") if isinstance(group_profs[r], dict) else "")
                    assigned_preset = candidate_presets[c]
                    assigned[spk_name] = assigned_preset
                    used_presets.add(assigned_preset)
                    cls._apply_preset_to_profile(group_profs[r], assigned_preset, cat[assigned_preset])

                # Với các nhân vật vượt quá số preset cùng giới: tái sử dụng theo thứ tự vòng lặp và đánh dấu review_needed
                preset_use = {k: 1 for k in candidate_presets}
                for prof in group_profs[n_cand:]:
                    spk_name = getattr(prof, "display_name", "") or getattr(prof, "speaker_id", "") or (prof.get("display_name") if isinstance(prof, dict) else "")
                    best_preset = min(candidate_presets, key=lambda k: preset_use.get(k, 0))
                    preset_use[best_preset] = preset_use.get(best_preset, 0) + 1
                    assigned[spk_name] = best_preset
                    if hasattr(prof, "status"):
                        prof.status = "review_needed"
                    elif isinstance(prof, dict):
                        prof["status"] = "review_needed"
                    cls._apply_preset_to_profile(prof, best_preset, cat[best_preset])

        # Phân bổ độc lập theo giới tính để đảm bảo tuyệt đối không có sự gán nhầm chéo giới tính
        male_profs = [
            p for p in unassigned_profs
            if (getattr(p, "gender", "") or (p.get("gender") if isinstance(p, dict) else "")) == "male"
        ]
        female_profs = [
            p for p in unassigned_profs
            if (getattr(p, "gender", "") or (p.get("gender") if isinstance(p, dict) else "")) == "female"
        ]
        other_profs = [
            p for p in unassigned_profs
            if p not in male_profs and p not in female_profs
        ]

        male_presets = [k for k in available_presets if cat.get(k, {}).get("gender") == "male"]
        female_presets = [k for k in available_presets if cat.get(k, {}).get("gender") == "female"]

        if not male_presets:
            male_presets = [k for k in catalog_names if cat.get(k, {}).get("gender") == "male"]
        if not female_presets:
            female_presets = [k for k in catalog_names if cat.get(k, {}).get("gender") == "female"]

        _allocate_group(male_profs, male_presets)
        _allocate_group(female_profs, female_presets)

        remaining_presets = [k for k in available_presets if k not in used_presets]
        if not remaining_presets:
            remaining_presets = list(catalog_names)
        _allocate_group(other_profs, remaining_presets)

        return assigned

    @staticmethod
    def _apply_preset_to_profile(prof: Any, preset_name: str, preset_info: Dict[str, Any]):
        ins = preset_info.get("instruct", "")
        if hasattr(prof, "method"):
            prof.method = "preset"
            prof.preset_name = preset_name
            prof.voice_instruct = ins
            if getattr(prof, "status", "") != "review_needed":
                prof.status = "locked"
        elif isinstance(prof, dict):
            prof["method"] = "preset"
            prof["preset_name"] = preset_name
            prof["voice_instruct"] = ins
            prof["instruct"] = ins
            if prof.get("status") != "review_needed":
                prof["status"] = "locked"


def verify_speaker_acoustic_distinction(
    profiles: List[Any],
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Xác minh độ tách biệt âm học giữa các giọng được phân bổ cho các nhân vật.
    Đảm bảo không có hai nhân vật nào bị trùng lặp preset hoặc trùng lặp instruct hiệu dụng.
    """
    cat = catalog if catalog is not None else VIETNAMESE_VOICE_CATALOG
    assigned_presets = []
    assigned_instructs = []

    for prof in profiles:
        p_name = getattr(prof, "preset_name", "") or (prof.get("preset_name") if isinstance(prof, dict) else "")
        if p_name:
            assigned_presets.append(p_name)
        ins = getattr(prof, "voice_instruct", "") or (prof.get("voice_instruct") if isinstance(prof, dict) else "")
        if not ins and p_name in cat:
            ins = cat[p_name].get("instruct", "")
        if ins:
            assigned_instructs.append(get_effective_instruct(ins))

    if len(assigned_presets) <= 1:
        return {"all_distinct": True, "duplicate_count": 0, "unique_ratio": 1.0}

    unique_presets = len(set(assigned_presets))
    dup_presets = len(assigned_presets) - unique_presets

    unique_instructs = len(set(assigned_instructs))
    dup_instructs = len(assigned_instructs) - unique_instructs

    total_dups = max(dup_presets, dup_instructs)
    return {
        "all_distinct": total_dups == 0,
        "duplicate_count": total_dups,
        "unique_ratio": round(unique_presets / len(assigned_presets), 4),
        "total_assigned": len(assigned_presets),
    }


# Gán phương thức tiện ích lên TTSEngine
TTSEngine.calculate_voice_match = VoiceRecommendationEngine.calculate_voice_match
TTSEngine.recommend_voices = VoiceRecommendationEngine.recommend_voices
TTSEngine.allocate_distinct_voices = GlobalVoiceAllocator.allocate_distinct_voices
TTSEngine.AcousticClassifier = AcousticClassifier
TTSEngine.VietnameseAccentClassifier = VietnameseAccentClassifier
TTSEngine.build_distinct_voice_pool = staticmethod(build_distinct_voice_pool)
TTSEngine.GlobalVoiceAllocator = GlobalVoiceAllocator

