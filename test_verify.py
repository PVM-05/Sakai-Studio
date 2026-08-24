import sys
import os

def check_voice_separator():
    print("[1] Kiểm tra VoiceSeparator Backend...")
    try:
        from src.ai_engines.voice_separator import VoiceSeparator
        sep = VoiceSeparator("ffmpeg")
        print(" => VoiceSeparator (ffmpeg): Khởi tạo thành công")
    except Exception as e:
        print(f" => VoiceSeparator (ffmpeg): FAILED - {e}")

def check_subtitle_exporter():
    print("[2] Kiểm tra SubtitleExporter với styles mới...")
    try:
        from src.core.tools.subtitle_exporter import SubtitleExporter
        SubtitleExporter.export_ass("test.ass", [{"start_frame":0, "end_frame":30, "text":"Hello"}], 30.0, "tiktok_yellow")
        with open("test.ass", "r", encoding="utf-8") as f:
            content = f.read()
        if "0000FFFF" in content:
            print(" => TikTok Yellow (Màu Hex 0000FFFF): Thành công")
        else:
            print(" => TikTok Yellow: LỖI không tìm thấy mã màu")
    except Exception as e:
        print(f" => SubtitleExporter: LỖI - {e}")

if __name__ == "__main__":
    check_voice_separator()
    check_subtitle_exporter()
