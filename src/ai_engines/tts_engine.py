# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
import soundfile as sf
import subprocess
from typing import List
from src.ai_engines.translator import SubtitleBlock

class TTSEngine:
    def __init__(self, parent_logger=None):
        self.model = None
        self.logger = parent_logger

    def log(self, msg: str):
        if self.logger:
            self.logger(msg)
        else:
            print(f"[TTS] {msg}")

    def load_model(self):
        if self.model is None:
            self.log("Đang tải mô hình OmniVoice (Local) vào GPU... Vui lòng đợi (có thể mất vài phút).")
            from omnivoice import OmniVoice
            # Force CUDA if available
            self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")
            self.log("Tải mô hình OmniVoice thành công!")

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            self.log("Đã giải phóng VRAM của mô hình OmniVoice.")

    def synthesize_blocks(self, blocks: List[SubtitleBlock], total_duration: float, output_path: str):
        """
        Tạo ra một track âm thanh hoàn chỉnh chứa toàn bộ giọng đọc từ các blocks.
        """
        self.load_model()
        self.log(f"Bắt đầu tổng hợp giọng nói cho {len(blocks)} khối phụ đề...")

        # Sample rate cố định của OmniVoice thường là 24000 hoặc 22050.
        # Chúng ta sẽ dùng 24000 làm chuẩn.
        target_sr = 24000
        
        # Tạo mảng numpy rỗng tương ứng với tổng thời lượng video
        total_samples = int((total_duration + 5.0) * target_sr) # dư 5s
        full_audio = np.zeros(total_samples, dtype=np.float32)

        for i, blk in enumerate(blocks):
            text_to_speak = blk.translated_text if blk.translated_text else blk.text
            if not text_to_speak or len(text_to_speak.strip()) == 0:
                continue
                
            self.log(f"Đang tổng hợp [{i+1}/{len(blocks)}]: {text_to_speak}")
            try:
                # Trích xuất numpy array từ kết quả synthesize
                # OmniVoice API thường trả về dict hoặc trực tiếp là numpy array
                result = self.model.synthesize(text_to_speak)
                
                # Xử lý kết quả trả về an toàn
                if isinstance(result, tuple) and len(result) >= 2:
                    audio_arr, _ = result[0], result[1] # (audio, sr)
                elif isinstance(result, dict) and 'audio' in result:
                    audio_arr = result['audio']
                else:
                    audio_arr = result

                # Flatten nếu có multi-channel
                if hasattr(audio_arr, 'squeeze'):
                    audio_arr = audio_arr.squeeze()

                # Xác định vị trí chèn vào track
                start_sample = int(blk.start_time * target_sr)
                end_sample = start_sample + len(audio_arr)

                # Chèn audio vào full track
                if end_sample <= total_samples:
                    # Tránh chồng chéo nếu 2 câu dính nhau quá sát
                    existing = full_audio[start_sample:end_sample]
                    # Mix đơn giản: cộng dồn và kẹp giá trị
                    mixed = existing + audio_arr
                    full_audio[start_sample:end_sample] = np.clip(mixed, -1.0, 1.0)
                
            except Exception as e:
                self.log(f"Lỗi khi tổng hợp block {i+1}: {str(e)}")
        
        self.log(f"Đang lưu track lồng tiếng ra file tạm: {output_path}")
        sf.write(output_path, full_audio, target_sr)
        self.log("Đã lưu track lồng tiếng thành công.")

    @staticmethod
    def mix_audio_ducking(video_in: str, tts_audio_in: str, video_out: str, original_volume: float = 0.15):
        """
        Dùng FFmpeg mix tiếng TTS với tiếng video gốc (đã được làm nhỏ).
        """
        # amix filter: mix 2 inputs. 
        # Cấu hình để tiếng video gốc nhỏ đi, tiếng TTS giữ nguyên.
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", tts_audio_in,
            "-filter_complex", 
            f"[0:a]volume={original_volume}[a0]; [1:a]volume=1.0[a1]; [a0][a1]amix=inputs=2:duration=first:dropout_transition=2[outa]",
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            video_out
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
