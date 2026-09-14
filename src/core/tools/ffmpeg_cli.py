import os
import stat

import platform
from .common_tools import merge_big_file_if_not_exists
from src.core.config import BASE_DIR

class FFmpegCLI:
    
    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None
    
    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance
    
    def __init__(self):
        ffmpeg_exe = self.ffmpeg_path
        if os.path.exists(ffmpeg_exe):
            try:
                os.chmod(ffmpeg_exe, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
            except Exception:
                pass
        
    @property
    def ffmpeg_path(self):
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg', 'win_x64')
            ffmpeg_exe = 'ffmpeg.exe'
        elif system == "Linux":
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg',  'linux_x64')
            ffmpeg_exe = 'ffmpeg'
        else:
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg', 'macos')
            ffmpeg_exe = 'ffmpeg'
            
        if os.path.exists(ffmpeg_dir):
            try:
                merge_big_file_if_not_exists(ffmpeg_dir, ffmpeg_exe)
                return os.path.join(ffmpeg_dir, ffmpeg_exe)
            except Exception:
                pass
        
        return "ffmpeg"