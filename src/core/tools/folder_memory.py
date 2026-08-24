# -*- coding: utf-8 -*-
"""
Folder Memory Manager for Sakai Studio.
Ghi nhớ vị trí thư mục mở gần nhất cho toàn bộ ứng dụng (QFileDialog Memory).
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from PySide6.QtWidgets import QFileDialog, QWidget

logger = logging.getLogger(__name__)

MEMORY_FILE = Path(__file__).parent.parent.parent / "config" / "folder_memory.json"

# In-memory cache
_MEMORY_CACHE: dict[str, str] = {}


def _load_memory():
    global _MEMORY_CACHE
    if not _MEMORY_CACHE and MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _MEMORY_CACHE = data
        except Exception as e:
            logger.warning(f"Failed to load folder memory: {e}")


def _save_memory():
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(_MEMORY_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to save folder memory: {e}")


def get_last_directory(category: str = "default") -> str:
    """Lấy đường dẫn thư mục đã mở gần nhất cho category chỉ định."""
    _load_memory()
    saved_path = _MEMORY_CACHE.get(category) or _MEMORY_CACHE.get("default")
    if saved_path and os.path.exists(saved_path):
        if os.path.isfile(saved_path):
            return str(Path(saved_path).parent)
        return saved_path
    
    # Fallback to User Documents / Home
    user_home = os.path.expanduser("~")
    return user_home


def set_last_directory(path: str | Path, category: str = "default"):
    """Ghi nhớ đường dẫn thư mục mở gần nhất."""
    if not path:
        return
    p = Path(path)
    dir_path = str(p.parent) if p.is_file() or p.suffix else str(p)
    if os.path.exists(dir_path):
        _MEMORY_CACHE[category] = dir_path
        _MEMORY_CACHE["default"] = dir_path
        _save_memory()


class FolderMemoryDialog:
    """Wrapper thông minh cho QFileDialog tự động đọc & lưu lịch sử mở thư mục."""

    @staticmethod
    def getOpenFileName(
        parent: Optional[QWidget],
        caption: str,
        dir_path: Optional[str] = None,
        filter_str: str = "",
        category: str = "default",
    ) -> Tuple[str, str]:
        start_dir = dir_path or get_last_directory(category)
        res_file, selected_filter = QFileDialog.getOpenFileName(parent, caption, start_dir, filter_str)
        if res_file:
            set_last_directory(res_file, category)
        return res_file, selected_filter

    @staticmethod
    def getOpenFileNames(
        parent: Optional[QWidget],
        caption: str,
        dir_path: Optional[str] = None,
        filter_str: str = "",
        category: str = "default",
    ) -> Tuple[List[str], str]:
        start_dir = dir_path or get_last_directory(category)
        res_files, selected_filter = QFileDialog.getOpenFileNames(parent, caption, start_dir, filter_str)
        if res_files:
            set_last_directory(res_files[0], category)
        return res_files, selected_filter

    @staticmethod
    def getExistingDirectory(
        parent: Optional[QWidget],
        caption: str,
        dir_path: Optional[str] = None,
        category: str = "default",
    ) -> str:
        start_dir = dir_path or get_last_directory(category)
        res_dir = QFileDialog.getExistingDirectory(parent, caption, start_dir)
        if res_dir:
            set_last_directory(res_dir, category)
        return res_dir

    @staticmethod
    def getSaveFileName(
        parent: Optional[QWidget],
        caption: str,
        default_filename: str = "",
        filter_str: str = "",
        category: str = "default",
    ) -> Tuple[str, str]:
        start_dir = get_last_directory(category)
        initial_path = os.path.join(start_dir, default_filename) if default_filename else start_dir
        res_file, selected_filter = QFileDialog.getSaveFileName(parent, caption, initial_path, filter_str)
        if res_file:
            set_last_directory(res_file, category)
        return res_file, selected_filter
