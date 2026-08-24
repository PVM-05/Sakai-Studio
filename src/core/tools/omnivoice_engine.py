import os
import time
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from src.ai_engines.translator import SubtitleBlock
from src.core.config import get_readable_path

class OmniVoiceEngine:
    """
    Engine lồng tiếng AI (Auto Dubbing) sử dụng mã nguồn mở OmniVoice.
    Giả định OmniVoice đang chạy ở một Local API Server tương thích định dạng chuẩn.
    """
    def __init__(self, api_url: str = "http://localhost:8000/v1/audio/speech", voice: str = "vi-VN-Standard-A"):
        self.api_url = api_url
        self.voice = voice

    def generate_audio_for_block(self, text: str, output_path: str) -> bool:
        """Gọi API OmniVoice để tạo file WAV từ text."""
        try:
            payload = {
                "input": text,
                "voice": self.voice,
                "model": "omnivoice-1",
                "response_format": "wav"
            }
            # Timeout ngắn vì có thể Server chưa bật
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                print(f"[OmniVoice] Lỗi API: {response.text}")
                return False
        except Exception as e:
            print(f"[OmniVoice] Lỗi kết nối đến {self.api_url}: {e}")
            return False

    def dub_video(self, video_path: str, subtitle_blocks: List[SubtitleBlock], output_video_path: str, fps: float) -> bool:
        """
        Sinh âm thanh cho tất cả các câu phụ đề và ghép vào video gốc bằng FFmpeg.
        """
        video_path = get_readable_path(video_path)
        output_video_path = get_readable_path(output_video_path)
        temp_dir = Path(output_video_path).parent / "temp_omnivoice"
        os.makedirs(temp_dir, exist_ok=True)
        
        audio_files = []
        for i, block in enumerate(subtitle_blocks):
            text = block.translated_text if block.translated_text else block.text
            if not text.strip():
                continue
                
            wav_path = str(temp_dir / f"dub_{i}.wav")
            # Generate Audio
            success = self.generate_audio_for_block(text, wav_path)
            
            # Fallback for demonstration if API is offline
            if not success:
                print(f"[OmniVoice] Server Offline. Tạo file audio trống/fallback cho câu {i}")
                # Create a silent 1-second wav just to avoid ffmpeg crash
                subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '1', '-c:a', 'pcm_s16le', wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            if os.path.exists(wav_path):
                # Calculate start time in milliseconds from start_time string (HH:MM:SS,mmm)
                parts = block.start_time.replace('.', ',').split(',')
                time_parts = parts[0].split(':')
                h, m, s = int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
                ms = int(parts[1]) if len(parts) > 1 else 0
                total_ms = (h * 3600 + m * 60 + s) * 1000 + ms
                
                audio_files.append((wav_path, total_ms))

        if not audio_files:
            import shutil
            shutil.copy(video_path, output_video_path)
            return True

        # Build FFmpeg complex filter
        # [1]adelay=1000|1000[a1]; [2]adelay=3000|3000[a2]; [a1][a2]amix=inputs=2[dub]; [0:a][dub]amix=inputs=2[outa]
        filter_parts = []
        mix_inputs = []
        for i, (wav_path, delay_ms) in enumerate(audio_files):
            input_idx = i + 1  # 0 is video
            delay_filter = f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[a{i}]"
            filter_parts.append(delay_filter)
            mix_inputs.append(f"[a{i}]")
            
        # Mix all dubs together
        dub_mix = "".join(mix_inputs) + f"amix=inputs={len(audio_files)}:duration=longest[dub]"
        filter_parts.append(dub_mix)
        
        # Mix dubs with original audio
        # Note: volume=0.3 on [0:a] to lower original background audio (ducking)
        final_mix = "[0:a]volume=0.2[bg];[bg][dub]amix=inputs=2:duration=first[outa]"
        filter_parts.append(final_mix)
        
        complex_filter = ";".join(filter_parts)
        
        cmd = ['ffmpeg', '-y', '-i', video_path]
        for wav_path, _ in audio_files:
            cmd.extend(['-i', wav_path])
            
        cmd.extend([
            '-filter_complex', complex_filter,
            '-map', '0:v',
            '-map', '[outa]',
            '-c:v', 'copy',  # Copy video directly, don't re-encode
            '-c:a', 'aac',
            '-b:a', '192k',
            output_video_path
        ])
        
        print("[OmniVoice] Đang trộn âm thanh vào video...")
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            success = result.returncode == 0
            if not success:
                print(f"[OmniVoice] FFmpeg lỗi: {result.stderr}")
        except Exception as e:
            print(f"[OmniVoice] Lỗi thực thi FFmpeg: {e}")
            success = False
            
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
            
        return success
