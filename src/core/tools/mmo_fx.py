import os
import subprocess
from pathlib import Path
from src.core.config import get_readable_path

class MMOFXEngine:
    """
    Engine chuyên xử lý các hiệu ứng Lách Bản Quyền (Copyright Bypass) cho dân MMO.
    Sử dụng FFmpeg để can thiệp sâu vào cấu trúc video.
    """
    
    @staticmethod
    def apply_copyright_bypass(video_path: str, output_path: str, 
                               horizontal_flip: bool = False,
                               speed_multiplier: float = 1.0,
                               crop_percentage: int = 0,
                               pitch_shift: float = 1.0) -> bool:
        """
        Áp dụng các bộ lọc lách gậy bản quyền.
        - horizontal_flip: Lật ngang video
        - speed_multiplier: Đổi tốc độ (vd: 1.05)
        - crop_percentage: Zoom nhẹ cắt viền (vd: 2%)
        """
        video_path = get_readable_path(video_path)
        output_path = get_readable_path(output_path)
        
        # Xây dựng chuỗi Video Filter (vf)
        vf_filters = []
        if horizontal_flip:
            vf_filters.append("hflip")
            
        if crop_percentage > 0:
            # Crop x% from all sides
            # format: crop=w=iw*(1-2*p):h=ih*(1-2*p)
            p = crop_percentage / 100.0
            vf_filters.append(f"crop=w=iw*(1-{2*p}):h=ih*(1-{2*p})")
            
        if speed_multiplier != 1.0:
            # setpts cho video
            vf_filters.append(f"setpts={1/speed_multiplier:.4f}*PTS")
            
        # Xây dựng chuỗi Audio Filter (af)
        af_filters = []
        
        # 1. Pitch Shifting (Bóp méo âm thanh chống bản quyền)
        # Bằng cách thay đổi sample rate, âm thanh bị méo đi, sau đó ta dùng atempo 
        # để bù trừ lại tốc độ, kết quả là âm thanh méo nhưng độ dài vẫn giữ nguyên.
        pitch_speed_compensation = 1.0
        if pitch_shift != 1.0:
            af_filters.append(f"asetrate=44100*{pitch_shift}")
            af_filters.append("aresample=44100")
            pitch_speed_compensation = 1.0 / pitch_shift
            
        # Tổng hợp tốc độ cần bù cho âm thanh (Gồm tua video + bù pitch shift)
        total_audio_speed = speed_multiplier * pitch_speed_compensation
        
        # 2. Atempo Chaining (Phá vỡ giới hạn 0.5 - 2.0 của FFmpeg cũ)
        if total_audio_speed != 1.0:
            while total_audio_speed < 0.5:
                af_filters.append("atempo=0.5")
                total_audio_speed /= 0.5
            while total_audio_speed > 2.0:
                af_filters.append("atempo=2.0")
                total_audio_speed /= 2.0
            if total_audio_speed != 1.0:
                af_filters.append(f"atempo={total_audio_speed:.4f}")
            
        # Command build
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        if vf_filters:
            cmd.extend(['-vf', ','.join(vf_filters)])
            
        if af_filters:
            cmd.extend(['-af', ','.join(af_filters)])
            
        # Giữ chất lượng cao nhất
        cmd.extend(['-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-c:a', 'aac', '-b:a', '192k'])
        cmd.append(output_path)
        
        try:
            print(f"[MMO FX] Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"[MMO FX] Error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"[MMO FX] Exception: {e}")
            return False
