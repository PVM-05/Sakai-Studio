"""
Video Subtitle Remover Pro - Backend Module
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import GUISubtitleRemover
    from .processor import (
        ProSubtitleRemover,
        SubtitleDetector,
        InpaintMode,
        ProcessingConfig,
        STTNInpainter,
        LAMAInpainter,
        ProPainterInpainter,
    )

__all__ = [
    'GUISubtitleRemover',
    'ProSubtitleRemover',
    'SubtitleDetector',
    'InpaintMode',
    'ProcessingConfig',
    'STTNInpainter',
    'LAMAInpainter',
    'ProPainterInpainter',
]


def __getattr__(name):
    """Lazy-load processor exports so `python -m backend.processor` stays clean."""
    if name in __all__:
        module = import_module(".processor", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
