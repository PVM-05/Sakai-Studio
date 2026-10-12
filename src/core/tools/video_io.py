import os
import queue
import subprocess
import threading

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self._buffer.put((ret, frame))
            if not ret:
                break

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。"""
        return self._buffer.get()

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """
    Thông qua FFmpeg để ghi các khung hình, hỗ trợ GPU Hardware Encoding và fallback CPU libx264.
    """

    _cached_gpu_encoder = {}

    @classmethod
    def detect_gpu_encoder(cls):
        ffmpeg_path = FFmpegCLI.instance().ffmpeg_path
        if ffmpeg_path in cls._cached_gpu_encoder:
            return cls._cached_gpu_encoder[ffmpeg_path]

        detected = 'libx264'
        try:
            # Chạy lệnh kiểm tra các encoder hỗ trợ của FFmpeg không hiển thị cửa sổ console
            startupinfo = None
            creationflags = 0
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = 0x08000000  # CREATE_NO_WINDOW

            res = subprocess.check_output(
                [ffmpeg_path, '-hide_banner', '-encoders'],
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                creationflags=creationflags,
                timeout=10,
            ).decode('utf-8', errors='ignore')

            for candidate in ('h264_nvenc', 'h264_amf', 'h264_qsv'):
                if candidate not in res:
                    continue
                # Kiểm tra thực tế phần cứng GPU có khởi tạo được bộ mã hóa hay không
                probe = subprocess.run(
                    [
                        ffmpeg_path,
                        '-hide_banner',
                        '-loglevel', 'error',
                        '-f', 'lavfi',
                        '-i', 'nullsrc=s=256x256:d=0.04',
                        '-frames:v', '1',
                        '-pix_fmt', 'yuv420p',
                        '-c:v', candidate,
                        '-f', 'null',
                        '-',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    startupinfo=startupinfo,
                    creationflags=creationflags,
                    timeout=5,
                )
                if probe.returncode == 0:
                    detected = candidate
                    break
        except Exception:
            detected = 'libx264'

        cls._cached_gpu_encoder[ffmpeg_path] = detected
        return detected

    def _build_extra_args(self, codec):
        gop_size = str(max(1, int(self.fps)))
        if codec == 'h264_nvenc':
            return [
                '-c:v', 'h264_nvenc',
                '-pix_fmt', 'yuv420p',
                '-g', gop_size,
                '-cq', '19',
                '-preset', 'p1',
            ]
        if codec == 'h264_amf':
            return [
                '-c:v', 'h264_amf',
                '-pix_fmt', 'yuv420p',
                '-g', gop_size,
                '-rc', 'cqp',
                '-qp_i', '19',
                '-qp_p', '19',
            ]
        if codec == 'h264_qsv':
            return [
                '-c:v', 'h264_qsv',
                '-pix_fmt', 'yuv420p',
                '-g', gop_size,
                '-global_quality', '19',
            ]
        return [
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-g', gop_size,
            '-crf', '17',
            '-preset', 'ultrafast',
        ]

    def _start_process(self, codec):
        w, h = self.size
        self._codec = codec
        extra_args = self._build_extra_args(codec)
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',
            *extra_args,
            '-loglevel', 'error',
            self.output_path,
        ]
        startupinfo = None
        creationflags = 0
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = 0x08000000  # CREATE_NO_WINDOW

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def __init__(self, output_path, fps, size):
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self._fallback_buffer = []
        w, h = size

        # Mặc định sử dụng CPU encoding libx264; chỉ bật GPU khi khung hình đủ lớn (>=256x256)
        codec = 'libx264'
        use_gpu_enc = True
        try:
            from src.core.config import config
            if getattr(config, 'gpuVideoEncoding', None) is not None:
                use_gpu_enc = config.gpuVideoEncoding.value
        except Exception:
            pass

        if use_gpu_enc and w >= 256 and h >= 256:
            detected_codec = self.detect_gpu_encoder()
            if detected_codec != 'libx264':
                codec = detected_codec

        self._start_process(codec)

    def _fallback_to_cpu(self):
        try:
            if self._process and self._process.stdin:
                self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            if self._process:
                self._process.wait(timeout=2)
        except Exception:
            pass
        self._start_process('libx264')
        for raw_frame in self._fallback_buffer:
            try:
                self._process.stdin.write(raw_frame)
            except (BrokenPipeError, OSError):
                break
        self._fallback_buffer.clear()

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        raw = frame.tobytes()
        if self._codec != 'libx264' and len(self._fallback_buffer) < 16:
            self._fallback_buffer.append(raw)
        try:
            if self._process.poll() is not None and self._codec != 'libx264':
                self._fallback_to_cpu()
                return
            self._process.stdin.write(raw)
        except (BrokenPipeError, OSError):
            if self._codec != 'libx264':
                self._fallback_to_cpu()

    def release(self):
        """关闭管道并等待编码完成。"""
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)
        if self._process.returncode != 0 and self._codec != 'libx264' and self._fallback_buffer:
            self._fallback_to_cpu()
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=5)
        self._fallback_buffer.clear()
