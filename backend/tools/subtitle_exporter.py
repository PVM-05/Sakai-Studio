"""
Mô-đun xuất file phụ đề chuẩn định dạng SRT và ASS.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any


def frame_to_timestamp_srt(frame_no: int, fps: float) -> str:
    """Chuyển đổi số khung hình thành chuỗi mốc thời gian dạng SRT (HH:MM:SS,mmm)."""
    seconds = frame_no / max(fps, 1.0)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def frame_to_timestamp_ass(frame_no: int, fps: float) -> str:
    """Chuyển đổi số khung hình thành chuỗi mốc thời gian dạng ASS (H:MM:SS.cc)."""
    seconds = frame_no / max(fps, 1.0)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds - int(seconds)) * 100)
    return f"{hrs:01d}:{mins:02d}:{secs:02d}.{centis:02d}"


class SubtitleExporter:
    """
    Bộ xuất file phụ đề chuẩn định dạng SRT và ASS.
    """

    @staticmethod
    def export_srt(output_path: str, subtitle_items: List[Dict[str, Any]], fps: float) -> str:
        """Xuất file phụ đề dạng chuẩn SRT."""
        lines = []
        for idx, item in enumerate(subtitle_items, 1):
            start_str = frame_to_timestamp_srt(item['start_frame'], fps)
            end_str = frame_to_timestamp_srt(item['end_frame'], fps)
            text = item.get('translated_text') or item.get('text', '')
            if not text.strip():
                continue
            lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    @staticmethod
    def export_ass(output_path: str, subtitle_items: List[Dict[str, Any]], fps: float, style_type: str = "tiktok_yellow") -> str:
        """Xuất file phụ đề dạng chuẩn ASS với phong cách TikTok / Shorts viền đen nổi bật."""
        primary_color = "&H0000FFFF" if "yellow" in style_type else "&H00FFFFFF"
        outline_size = 3
        fontsize = 28

        header = f"""[Script Info]
Title: MMO Translated Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{fontsize},{primary_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_size},1,2,10,10,24,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogue_lines = []
        for item in subtitle_items:
            start_str = frame_to_timestamp_ass(item['start_frame'], fps)
            end_str = frame_to_timestamp_ass(item['end_frame'], fps)
            text = item.get('translated_text') or item.get('text', '')
            if not text.strip():
                continue
            text_cleaned = text.replace("\n", "\\N")
            dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_cleaned}")

        full_content = header + "\n".join(dialogue_lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        return output_path
