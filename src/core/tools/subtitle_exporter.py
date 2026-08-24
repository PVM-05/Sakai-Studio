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
    def export_ass(output_path: str, subtitle_items: List[Dict[str, Any]], fps: float, style_type: str = "tiktok_yellow", sub_areas: list = None) -> str:
        """Xuất file phụ đề dạng chuẩn ASS với phong cách TikTok / Shorts viền đen nổi bật."""
        import json
        from pathlib import Path
        mmo_cfg = {}
        cfg_path = Path(output_path).parent.parent / "config" / "mmo_settings.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    mmo_cfg = json.load(f)
            except Exception:
                pass
                
        font_name = mmo_cfg.get('sub_font_name', 'Arial')
        outline_size = int(mmo_cfg.get('sub_outline_size', '3'))
        fontsize = int(mmo_cfg.get('sub_base_size', '28'))
        
        primary_color = "&H0000FFFF" if "yellow" in style_type else "&H00FFFFFF"

        header = f"""[Script Info]
Title: MMO Translated Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{fontsize},{primary_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_size},1,2,10,10,24,1
Style: SmartOverride,{font_name},{fontsize},{primary_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_size},1,5,10,10,0,1

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
            
            # Smart Subtitle Replacement: Dùng vùng sub_areas do người dùng khoanh (ymin, ymax, xmin, xmax)
            has_box = False
            if sub_areas and len(sub_areas) > 0:
                # Format of sub_areas is usually [(ymin, ymax, xmin, xmax)] or [ymin, ymax, xmin, xmax]
                area = sub_areas[0] if isinstance(sub_areas[0], (list, tuple)) else sub_areas
                if len(area) == 4:
                    ymin, ymax, xmin, xmax = area
                    has_box = True
                    
            if has_box:
                # Calculate center position
                pos_x = int((xmin + xmax) / 2)
                pos_y = int((ymin + ymax) / 2)
                # Calculate approximate font size based on box height (tinh chỉnh hệ số 0.6 để font không quá to)
                box_height = ymax - ymin
                smart_fs = int(box_height * 0.6)
                # Inject ASS override tags {\an5\pos(X,Y)\fsSize}
                override_tags = f"{{\\an5\\pos({pos_x},{pos_y})\\fs{smart_fs}}}"
                dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},SmartOverride,,0,0,0,,{override_tags}{text_cleaned}")
            else:
                dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_cleaned}")

        full_content = header + "\n".join(dialogue_lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        return output_path
