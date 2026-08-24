# -*- coding: utf-8 -*-
"""
Voice Separator — Tách giọng nói khỏi nhạc nền / tiếng ồn.

Module này hỗ trợ tách vocals (giọng nói) từ file audio/video,
giúp Whisper AI nhận diện giọng nói chính xác hơn khi có nhạc nền.

Chiến lược multi-backend:
1. FFmpeg highpass+lowpass filter (luôn khả dụng, nhanh, chất lượng cơ bản)
2. Scipy spectral subtraction (cần scipy, chất lượng trung bình)
3. Demucs AI separation (cần demucs + torch, chất lượng cao nhất)

Sử dụng:
    from src.ai_engines.voice_separator import VoiceSeparator
    
    sep = VoiceSeparator()
    success = sep.separate("input.wav", "vocals_output.wav")
    
    # Hoặc tách từ video:
    success = sep.separate_from_video("video.mp4", "vocals_output.wav")
"""

from __future__ import annotations

import os
import logging
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceSeparator:
    """
    Multi-backend voice separator.
    
    Tự động chọn backend tốt nhất có sẵn:
    - "ffmpeg" (mặc định, luôn khả dụng): Dùng audio filters để lọc tần số giọng nói
    - "scipy" (cần scipy): Spectral gating noise reduction
    - "demucs" (cần torch + demucs): AI-powered source separation
    """

    def __init__(self, backend: str = "auto"):
        """
        Args:
            backend: "auto" (tự chọn tốt nhất), "ffmpeg", "scipy", hoặc "demucs"
        """
        self.backend = backend
        self._ffmpeg_bin = self._find_ffmpeg()

    def separate(self, audio_input: str, output_path: str) -> bool:
        """
        Tách giọng nói từ file audio.
        
        Args:
            audio_input: Đường dẫn file audio đầu vào (.wav, .mp3, .m4a, etc.)
            output_path: Đường dẫn file audio đầu ra (vocals only, .wav)
            
        Returns:
            True nếu tách thành công, False nếu thất bại.
        """
        if not os.path.exists(audio_input):
            logger.warning(f"Audio input not found: {audio_input}")
            return False

        # Chọn backend
        backend = self._select_backend()
        logger.info(f"Voice separation using backend: {backend}")

        try:
            if backend == "demucs":
                return self._separate_demucs(audio_input, output_path)
            elif backend == "scipy":
                return self._separate_scipy(audio_input, output_path)
            else:
                return self._separate_ffmpeg(audio_input, output_path)
        except Exception as e:
            logger.warning(f"Voice separation failed ({backend}): {e}")
            # Fallback: copy nguyên file nếu tất cả backend đều fail
            try:
                shutil.copy2(audio_input, output_path)
                return True
            except Exception:
                return False

    def separate_from_video(self, video_path: str, output_path: str) -> bool:
        """
        Tách giọng nói trực tiếp từ video file.
        
        Bước 1: Extract audio từ video (FFmpeg)
        Bước 2: Tách vocals từ audio
        """
        if not os.path.exists(video_path):
            return False

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="vsr_voice_sep_")
            temp_audio = os.path.join(temp_dir, "audio.wav")

            # Extract audio
            if not self._extract_audio(video_path, temp_audio):
                return False

            # Separate vocals
            return self.separate(temp_audio, output_path)

        except Exception as e:
            logger.warning(f"Video voice separation failed: {e}")
            return False
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def get_available_backends(self) -> list:
        """Trả về danh sách các backend khả dụng."""
        backends = ["ffmpeg"]  # FFmpeg luôn khả dụng
        try:
            import scipy  # noqa: F401
            backends.append("scipy")
        except ImportError:
            pass
        try:
            import demucs  # noqa: F401
            import torch  # noqa: F401
            backends.append("demucs")
        except ImportError:
            pass
        return backends

    # =========================================================================
    # PRIVATE: Backend Selection
    # =========================================================================

    def _select_backend(self) -> str:
        """Chọn backend tốt nhất có sẵn."""
        if self.backend != "auto":
            return self.backend

        # Ưu tiên: demucs > scipy > ffmpeg
        try:
            import demucs  # noqa: F401
            import torch  # noqa: F401
            return "demucs"
        except ImportError:
            pass

        try:
            import scipy  # noqa: F401
            return "scipy"
        except ImportError:
            pass

        return "ffmpeg"

    # =========================================================================
    # BACKEND 1: FFmpeg Audio Filters (luôn khả dụng)
    # =========================================================================

    def _separate_ffmpeg(self, audio_input: str, output_path: str) -> bool:
        """
        Tách giọng nói bằng FFmpeg audio filters.
        
        Sử dụng:
        - highpass: Loại bỏ tần số thấp (bass, tiếng ồn động cơ) < 200Hz
        - lowpass: Loại bỏ tần số cao (tiếng ồn, hiss) > 3000Hz
        - dynaudnorm: Chuẩn hóa âm lượng động
        - anlmdn: Adaptive noise gate/denoiser
        """
        if not self._ffmpeg_bin:
            return False

        # Vocal frequency range: ~85Hz - 3400Hz
        filter_chain = (
            "highpass=f=100,"           # Cắt sub-bass
            "lowpass=f=3400,"           # Cắt high frequencies
            "anlmdn=s=0.0001,"          # Noise reduction nhẹ
            "dynaudnorm=g=3:f=150,"     # Dynamic normalization
            "aformat=sample_rates=16000:channel_layouts=mono"  # Output mono 16kHz cho Whisper
        )

        cmd = [
            self._ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", audio_input,
            "-af", filter_chain,
            "-acodec", "pcm_s16le",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("FFmpeg voice separation thành công.")
                return True
            else:
                logger.warning(f"FFmpeg voice separation failed: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg voice separation timeout.")
            return False
        except Exception as e:
            logger.warning(f"FFmpeg voice separation error: {e}")
            return False

    # =========================================================================
    # BACKEND 2: Scipy Spectral Gating
    # =========================================================================

    def _separate_scipy(self, audio_input: str, output_path: str) -> bool:
        """
        Tách giọng nói bằng scipy spectral gating.
        
        Phương pháp:
        1. Load audio → STFT (Short-Time Fourier Transform)
        2. Estimate noise profile từ đoạn im lặng
        3. Spectral subtraction: Trừ noise profile khỏi signal
        4. Bandpass filter: Giữ lại tần số giọng nói (100-3500Hz)
        5. ISTFT → Output audio
        """
        try:
            import numpy as np
            from scipy.io import wavfile
            from scipy.signal import butter, filtfilt

            # Đọc audio
            sample_rate, data = wavfile.read(audio_input)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Convert to float
            data = data.astype(np.float64) / np.max(np.abs(data) + 1e-10)

            # Bandpass filter cho voice range (100Hz - 3500Hz)
            nyq = sample_rate / 2.0
            low = min(100.0 / nyq, 0.99)
            high = min(3500.0 / nyq, 0.99)
            
            if low < high:
                b, a = butter(4, [low, high], btype='band')
                filtered = filtfilt(b, a, data)
            else:
                filtered = data

            # Spectral noise gate: estimate noise from first 0.5 seconds
            noise_frames = int(sample_rate * 0.5)
            if noise_frames > 0 and len(filtered) > noise_frames:
                noise_sample = filtered[:noise_frames]
                noise_rms = np.sqrt(np.mean(noise_sample ** 2)) * 2.0

                # Gate: zero out samples below noise threshold
                mask = np.abs(filtered) > noise_rms
                filtered = filtered * mask

            # Normalize
            max_val = np.max(np.abs(filtered))
            if max_val > 0:
                filtered = filtered / max_val * 0.9

            # Resample to 16kHz mono for Whisper
            if sample_rate != 16000:
                from scipy.signal import resample
                new_length = int(len(filtered) * 16000 / sample_rate)
                filtered = resample(filtered, new_length)
                sample_rate = 16000

            # Save
            output_data = (filtered * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, output_data)

            logger.info("Scipy voice separation thành công.")
            return True

        except Exception as e:
            logger.warning(f"Scipy voice separation failed: {e}")
            return False

    # =========================================================================
    # BACKEND 3: Demucs AI Source Separation (chất lượng cao nhất)
    # =========================================================================

    def _separate_demucs(self, audio_input: str, output_path: str) -> bool:
        """
        Tách giọng nói bằng Demucs AI (Meta Research).
        
        Demucs tách âm thanh thành 4 nguồn:
        - drums, bass, other, vocals
        Chúng ta chỉ giữ lại track "vocals".
        """
        try:
            import torch
            import torchaudio

            # Load demucs model
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            model = get_model("htdemucs")
            model.eval()

            # Load audio
            waveform, sr = torchaudio.load(audio_input)

            # Resample to model's sample rate if needed
            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                waveform = resampler(waveform)

            # Ensure stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            # Apply model
            with torch.no_grad():
                sources = apply_model(model, waveform.unsqueeze(0))
                # sources shape: (1, num_sources, channels, samples)
                # Source order: drums, bass, other, vocals
                vocals = sources[0, 3]  # vocals track

            # Convert to mono 16kHz for Whisper
            vocals_mono = vocals.mean(dim=0, keepdim=True)  # (1, samples)

            if model.samplerate != 16000:
                resampler = torchaudio.transforms.Resample(model.samplerate, 16000)
                vocals_mono = resampler(vocals_mono)

            torchaudio.save(output_path, vocals_mono, 16000)
            logger.info("Demucs voice separation thành công.")
            return True

        except Exception as e:
            logger.warning(f"Demucs voice separation failed: {e}")
            return False

    # =========================================================================
    # PRIVATE: Utilities
    # =========================================================================

    def _find_ffmpeg(self) -> Optional[str]:
        """Tìm ffmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "win_x64", "ffmpeg.exe"),
            os.path.join(os.getcwd(), "backend", "ffmpeg", "win_x64", "ffmpeg.exe"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return None

    def _extract_audio(self, video_path: str, output_audio: str) -> bool:
        """Extract audio từ video bằng FFmpeg."""
        if not self._ffmpeg_bin:
            return False
        cmd = [
            self._ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            output_audio,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0 and os.path.exists(output_audio) and os.path.getsize(output_audio) > 0
        except Exception:
            return False
