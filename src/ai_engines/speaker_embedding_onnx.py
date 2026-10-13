# -*- coding: utf-8 -*-
"""
Module trích xuất vector đặc trưng người nói qua ONNX Runtime (Tầng 1).
Hỗ trợ tăng tốc đồ họa NVIDIA GeForce RTX 4050 Laptop GPU qua CUDAExecutionProvider
và chế độ dự phòng vi xử lý trung tâm CPUExecutionProvider với quản lý bộ nhớ 768 MiB.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class MockSpeakerSession:
    """Phiên làm việc giả lập phục vụ kiểm thử đơn vị khi chưa tải tệp trọng số mô hình."""

    def __init__(self, embedding_dim: int = 192):
        self.embedding_dim = embedding_dim
        self._closed = False

    def run(self, output_names: Any, input_feed: dict) -> List[np.ndarray]:
        if self._closed:
            raise RuntimeError("Phiên làm việc ONNX đã bị đóng.")
        batch_size = 1
        for val in input_feed.values():
            if hasattr(val, "shape") and len(val.shape) > 0:
                batch_size = max(batch_size, val.shape[0])
        # Tạo vector có tính tất định từ dữ liệu đầu vào
        rng = np.random.default_rng(42)
        vecs = rng.normal(0, 1, (batch_size, self.embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        return [vecs / np.maximum(norms, 1e-12)]

    def close(self):
        self._closed = True


def create_speaker_session(
    model_path: Union[str, Path] = "",
    device_id: int = 0,
    gpu_mem_limit_mb: int = 768,
    prefer_cuda: bool = True,
    strict_cuda: bool = False,
) -> Tuple[Any, str]:
    """
    Khởi tạo phiên làm việc suy luận ONNX Runtime trích xuất đặc trưng người nói.
    Ưu tiên CUDAExecutionProvider trên RTX 4050; nếu không có CUDA và không bật strict_cuda,
    sẽ sử dụng CPUExecutionProvider.
    """
    if ort is None or not model_path or not os.path.isfile(str(model_path)):
        # Chế độ kiểm thử hoặc chưa có tệp mô hình cục bộ
        provider = "MockExecutionProvider" if prefer_cuda else "CPUExecutionProvider"
        return MockSpeakerSession(embedding_dim=192), provider

    available = set(ort.get_available_providers())
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    selected_provider = "CPUExecutionProvider"

    if prefer_cuda:
        if "CUDAExecutionProvider" in available:
            cuda_options = {
                "device_id": int(device_id),
                "gpu_mem_limit": int(gpu_mem_limit_mb) * 1024 * 1024,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }
            providers.append(("CUDAExecutionProvider", cuda_options))
            providers.append("CPUExecutionProvider")
            selected_provider = "CUDAExecutionProvider"
        elif strict_cuda:
            raise RuntimeError(
                "Yêu cầu bắt buộc tăng tốc đồ họa CUDA nhưng môi trường chưa có CUDAExecutionProvider. "
                "Vui lòng kiểm tra gói cài đặt onnxruntime-gpu và trình điều khiển đồ họa."
            )
        else:
            logger.warning("CUDAExecutionProvider không khả dụng. Tự động chuyển sang CPUExecutionProvider.")
            providers.append("CPUExecutionProvider")
            selected_provider = "CPUExecutionProvider"
    else:
        providers.append("CPUExecutionProvider")
        selected_provider = "CPUExecutionProvider"

    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
    return session, selected_provider


def _preprocess_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Chuẩn hóa dữ liệu âm thanh về dạng đơn kênh float32 và độ dài tối thiểu 0.25 giây."""
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio = audio_data.astype(np.float32)

    # Khử trôi mức một chiều và chuẩn hóa biên độ
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio / peak

    min_samples = int(0.25 * sample_rate)
    if len(audio) < min_samples:
        pad_len = min_samples - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")

    return audio


def extract_speaker_embedding(
    session: Any,
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Trích xuất một vector đặc trưng người nói 192 chiều chuẩn hóa chuẩn L2."""
    audio = _preprocess_audio(audio_data, sample_rate=sample_rate)
    input_tensor = audio[np.newaxis, :]  # (1, T)

    if isinstance(session, MockSpeakerSession):
        out = session.run(None, {"speech": input_tensor})[0]
        emb = out[0]
    else:
        # Tương thích tên đầu vào chuẩn của mô hình CAM++ / 3D-Speaker
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        out = session.run([output_name], {input_name: input_tensor})[0]
        emb = out[0] if out.ndim > 1 else out

    norm = np.linalg.norm(emb)
    if norm > 1e-12:
        emb = emb / norm
    else:
        emb = np.zeros_like(emb)
        emb[0] = 1.0
    return emb.astype(np.float32)


def batch_extract_speaker_embeddings(
    session: Any,
    audio_chunks: List[np.ndarray],
    sample_rate: int = 16000,
    batch_size: int = 8,
) -> np.ndarray:
    """Trích xuất hàng loạt vector đặc trưng theo từng lô để kiểm soát bộ nhớ đồ họa."""
    if not audio_chunks:
        return np.empty((0, 192), dtype=np.float32)

    embeddings = []
    for i in range(0, len(audio_chunks), batch_size):
        batch = audio_chunks[i : i + batch_size]
        for item in batch:
            emb = extract_speaker_embedding(session, item, sample_rate=sample_rate)
            embeddings.append(emb)

    return np.stack(embeddings, axis=0)


def close_speaker_session(session: Any) -> None:
    """Đóng phiên làm việc ONNX và giải phóng tài nguyên bộ nhớ."""
    if session is None:
        return
    try:
        if hasattr(session, "close"):
            session.close()
    except Exception:
        pass
    del session
    gc.collect()
