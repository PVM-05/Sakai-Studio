# -*- coding: utf-8 -*-
"""
Module trích xuất vector đặc trưng người nói qua ONNX Runtime (Tầng 1).
Hỗ trợ tăng tốc đồ họa NVIDIA GeForce RTX 4050 Laptop GPU qua CUDAExecutionProvider
và chế độ dự phòng vi xử lý trung tâm CPUExecutionProvider với quản lý bộ nhớ 768 MiB.
Đặc trưng đầu vào tuân thủ chuẩn 80-bin Mel-FilterBank (FBank) kèm CMVN theo kiến trúc CAM++.
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
        seed = 42
        for val in input_feed.values():
            if hasattr(val, "shape") and len(val.shape) > 0:
                batch_size = max(batch_size, val.shape[0])
            if isinstance(val, np.ndarray) and val.size > 0:
                seed = int(abs(float(np.mean(val.flat[:min(200, val.size)])) * 10000)) % 100000 + 1
        rng = np.random.default_rng(seed)
        vecs = rng.normal(0, 1, (batch_size, self.embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        return [vecs / np.maximum(norms, 1e-12)]

    def close(self):
        self._closed = True


def extract_fbank_features(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Trích xuất đặc trưng 80-chiều Log Mel-FilterBank (FBank) kèm CMVN chuẩn hóa giá trị trung bình.
    Khớp chuẩn đầu vào của mô hình nhận diện giọng nói CAM++ / 3D-Speaker.
    """
    audio = _preprocess_audio(audio_data, sample_rate=sample_rate)

    # Ưu tiên sử dụng Torchaudio Kaldi FBank chuẩn xác
    feats: Optional[np.ndarray] = None
    try:
        import torch
        import torchaudio
        waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
        feats_tensor = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            sample_frequency=sample_rate,
            dither=0.0,
            window_type="hamming",
        )
        feats = feats_tensor.cpu().numpy()
    except Exception as e:
        logger.debug(f"Không thể tính FBank bằng Torchaudio: {e}. Chuyển sang tính bằng NumPy.")
        feats = None

    if feats is None:
        # Triển khai dự phòng thuần NumPy khi môi trường không nạp được Torchaudio Kaldi
        frame_len = int(0.025 * sample_rate)  # 400
        hop_len = int(0.010 * sample_rate)    # 160
        n_fft = 512
        n_mels = 80

        # Phân khung và áp dụng cửa sổ Hamming
        num_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
        window = np.hamming(frame_len).astype(np.float32)

        frames = np.zeros((num_frames, frame_len), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_len
            frames[i] = audio[start : start + frame_len] * window

        # Biến đổi Fourier thực và tính phổ năng lượng
        mag = np.abs(np.fft.rfft(frames, n=n_fft))
        power = (mag ** 2) / float(n_fft)

        # Ngân hàng bộ lọc Mel tam giác
        low_freq_mel = 0.0
        high_freq_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        fbank_matrix = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                fbank_matrix[m - 1, k] = (k - bin_points[m - 1]) / max(1, f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fbank_matrix[m - 1, k] = (bin_points[m + 1] - k) / max(1, f_m_plus - f_m)

        mel_energies = np.dot(power, fbank_matrix.T)
        feats = np.log(np.maximum(mel_energies, 1e-10))

    # CMVN: Chuẩn hóa trừ giá trị trung bình theo trục thời gian trên từng đoạn thoại
    feats = feats - np.mean(feats, axis=0, keepdims=True)
    return feats.astype(np.float32)


def create_speaker_session(
    model_path: Union[str, Path] = "",
    device_id: int = 0,
    gpu_mem_limit_mb: int = 768,
    prefer_cuda: bool = True,
    strict_cuda: bool = False,
    allow_mock: bool = False,
) -> Tuple[Any, str]:
    """
    Khởi tạo phiên làm việc suy luận ONNX Runtime trích xuất đặc trưng người nói.
    Ưu tiên CUDAExecutionProvider trên RTX 4050; nếu không có CUDA và không bật strict_cuda,
    sẽ sử dụng CPUExecutionProvider.
    Khi chưa cài đặt tệp mô hình và không bật allow_mock, sẽ ném lỗi FileNotFoundError rõ ràng.
    """
    model_str = str(model_path).strip() if model_path else ""
    if not model_str or not os.path.isfile(model_str):
        if allow_mock:
            provider = "MockExecutionProvider" if prefer_cuda else "CPUExecutionProvider"
            return MockSpeakerSession(embedding_dim=192), provider
        raise FileNotFoundError(
            f"Không tìm thấy tệp trọng số mô hình CAM++ ONNX tại '{model_str}'. "
            f"Vui lòng tải tệp mô hình hoặc cấu hình đường dẫn chính xác trong mục Cài đặt."
        )

    if ort is None:
        if allow_mock:
            return MockSpeakerSession(embedding_dim=192), "MockExecutionProvider"
        raise RuntimeError(
            "Thư viện onnxruntime chưa được cài đặt trong môi trường hiện tại. "
            "Vui lòng cài đặt onnxruntime-gpu để sử dụng tính năng nhận diện người nói."
        )

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

    session = ort.InferenceSession(model_str, sess_options=session_options, providers=providers)
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
    """
    Trích xuất một vector đặc trưng người nói 192 chiều chuẩn hóa chuẩn L2
    dựa trên đặc trưng Log Mel-FilterBank 80-bin.
    """
    audio = _preprocess_audio(audio_data, sample_rate=sample_rate)
    feats = extract_fbank_features(audio, sample_rate=sample_rate)  # (T, 80)
    input_tensor = feats[np.newaxis, :, :]  # (1, T, 80)

    if isinstance(session, MockSpeakerSession):
        out = session.run(None, {"speech": input_tensor})[0]
        emb = out[0] if out.ndim > 1 else out
    else:
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        output_name = session.get_outputs()[0].name

        # Nếu mô hình yêu cầu dạng (1, 80, T) thì chuyển vị ma trận
        feed_tensor = input_tensor
        expected_shape = getattr(input_meta, "shape", None)
        if expected_shape and len(expected_shape) >= 3 and expected_shape[1] == 80:
            feed_tensor = np.transpose(input_tensor, (0, 2, 1))

        out = session.run([output_name], {input_name: feed_tensor})[0]
        emb = out[0] if out.ndim > 1 else out

    emb = np.squeeze(emb).astype(np.float32)
    if emb.shape[0] != 192:
        if emb.shape[0] > 192:
            emb = emb[:192]
        else:
            emb = np.pad(emb, (0, 192 - emb.shape[0]))

    norm = np.linalg.norm(emb)
    if norm > 1e-12:
        emb = emb / norm
    else:
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
    return emb


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
