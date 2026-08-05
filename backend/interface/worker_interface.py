from abc import ABC, abstractmethod

class ProcessCallback(ABC):
    """
    Abstract base class for callbacks from backend processes.
    This decouples the core logic from the UI.
    """

    @abstractmethod
    def log_message(self, *args):
        """Called to append output log messages."""
        pass

    @abstractmethod
    def update_progress(self, progress: int, is_finished: bool, frame_no: int = 0):
        """Called to update progress (0-100), finished status, and current frame."""
        pass

    @abstractmethod
    def on_error(self, err: Exception):
        """Called when a fatal error occurs."""
        pass

    @abstractmethod
    def manage_process(self, pid: int):
        """Called to report a background process PID (e.g. ffmpeg) for process management."""
        pass

    @abstractmethod
    def update_preview(self, *args):
        """Called to update preview frames during processing."""
        pass
