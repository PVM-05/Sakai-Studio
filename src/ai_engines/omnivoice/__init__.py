import warnings
from importlib.metadata import PackageNotFoundError, version

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)

try:
    __version__ = version("omnivoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

try:
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceConfig,
        OmniVoiceGenerationConfig,
        VoiceClonePrompt,
    )
except ImportError:
    from dataclasses import dataclass
    from typing import Any, Optional

    OmniVoice = None
    OmniVoiceConfig = None

    @dataclass
    class OmniVoiceGenerationConfig:  # type: ignore[no-redef]
        num_step: int = 32
        guidance_scale: float = 2.0
        t_shift: float = 0.1
        layer_penalty_factor: float = 5.0
        position_temperature: float = 5.0
        class_temperature: float = 0.0
        denoise: bool = True
        preprocess_prompt: bool = True
        postprocess_output: bool = True
        audio_chunk_duration: float = 8.0
        audio_chunk_threshold: float = 12.0
        pad_duration: float = 0.08
        fade_duration: float = 0.03

    class VoiceClonePrompt:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def save(self, path: str) -> None:
            pass

        @classmethod
        def load(cls, path: str) -> "VoiceClonePrompt":
            return cls(path=path)

__all__ = [
    "OmniVoice",
    "OmniVoiceConfig",
    "OmniVoiceGenerationConfig",
    "VoiceClonePrompt",
]
