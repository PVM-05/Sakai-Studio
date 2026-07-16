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

    @staticmethod
    def detect_gpu_encoder():
        ffmpeg_path = FFmpegCLI.instance().ffmpeg_path
        try:
            # Chạy lệnh kiểm tra các encoder hỗ trợ của FFmpeg không hiển thị cửa sổ console
            startupinfo = None
            creationflags = 0
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = 0x08000000 # CREATE_NO_WINDOW
                
            res = subprocess.check_output(
                [ffmpeg_path, '-encoders'],
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                creationflags=creationflags
            ).decode('utf-8', errors='ignore')
            
            if 'h264_nvenc' in res:
                return 'h264_nvenc'
            elif 'h264_amf' in res:
                return 'h264_amf'
            elif 'h264_qsv' in res:
                return 'h264_qsv'
        except Exception:
            pass
        return 'libx264'

    def __init__(self, output_path, fps, size):
        w, h = size
        
        # Mặc định sử dụng CPU encoding libx264
        codec = 'libx264'
        extra_args = [
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',
            '-preset', 'ultrafast'
        ]
        
        # Nếu được bật tối ưu hóa mã hóa GPU, tự động dò quét và cấu hình tham số thích hợp
        from backend.config import config
        if getattr(config, 'gpuVideoEncoding', None) and config.gpuVideoEncoding.value:
            detected_codec = self.detect_gpu_encoder()
            if detected_codec == 'h264_nvenc':
                codec = 'h264_nvenc'
                # CQ 19 = visual lossless, preset p1 = performance 1 (fastest NVENC)
                extra_args = [
                    '-c:v', 'h264_nvenc',
                    '-pix_fmt', 'yuv420p',
                    '-cq', '19',
                    '-preset', 'p1'
                ]
            elif detected_codec == 'h264_amf':
                codec = 'h264_amf'
                extra_args = [
                    '-c:v', 'h264_amf',
                    '-pix_fmt', 'yuv420p',
                    '-rc', 'cqp',
                    '-qp_i', '19',
                    '-qp_p', '19'
                ]
            elif detected_codec == 'h264_qsv':
                codec = 'h264_qsv'
                extra_args = [
                    '-c:v', 'h264_qsv',
                    '-pix_fmt', 'yuv420p',
                    '-global_quality', '19'
                ]
                
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            *extra_args,
            '-loglevel', 'error',
            output_path
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        """关闭管道并等待编码完成。"""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)
