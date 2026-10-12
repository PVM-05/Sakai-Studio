# -*- coding: utf-8 -*-
"""
Mô-đun tương thích chuyển tiếp sang TTSEngine chạy trực tiếp mã nguồn OmniVoice trên GPU NVIDIA.
Đã loại bỏ hoàn toàn cơ chế gọi HTTP API cũ để chạy trực tiếp bằng PyTorch CUDA.
"""

from src.ai_engines.tts_engine import (
    TTSEngine,
    OmniVoiceEngine,
    build_voice_design_instruct,
    NON_VERBAL_TAGS,
)

__all__ = [
    "TTSEngine",
    "OmniVoiceEngine",
    "build_voice_design_instruct",
    "NON_VERBAL_TAGS",
]
