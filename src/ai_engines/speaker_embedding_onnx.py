# -*- coding: utf-8 -*-
"""
Trích xuất vector đặc trưng người nói từ mô hình ONNX thật.

Mô hình được tìm theo biến môi trường SAKAI_SPEAKER_EMBEDDING_MODEL hoặc
thư mục models/speaker_embedding trong dự án. Chế độ giả lập chỉ được phép bật
tường minh trong kiểm thử; ứng dụng thật không được trả kết quả nhận diện giả.
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
    """Phiên giả lập chỉ dành cho kiểm thử đơn vị; không dùng trong ứng dụng thật."""

    def __init__(self, embedding_dim: int = 192):
        self.embedding_dim = embedding_dim
        self._closed = False

    def run(self, output_names: Any, input_feed: dict) -> List[np.ndarray]:
        if self._closed:
            raise RuntimeError("Phiên làm việc giả lập đã bị đóng.")
        value = next(iter(input_feed.values()))
        signal = np.asarray(value, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError("Không thể tạo embedding từ âm thanh rỗng.")
        # Vector kiểm thử xác định từ tín hiệu đầu vào; không đại diện mô hình đã huấn luyện.
        rng_seed = int(abs(float(np.sum(signal[: min(signal.size, 4096)] * 1e5)))) % (2**32)
        rng = np.random.default_rng(rng_seed)
        vec = rng.normal(0, 1, (1, self.embedding_dim)).astype(np.float32)
        vec /= np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)
        return [vec]

    def close(self):
        self._closed = True


def resolve_speaker_model_path(model_path: Union[str, Path, None] = None) -> str:
    """Tìm tệp ONNX CAM++ tại đường dẫn chỉ định, biến môi trường hoặc thư mục mô hình cục bộ."""
    if model_path:
        candidate = Path(model_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(f"Đường dẫn mô hình người nói không tồn tại: {candidate}")

    env_path = os.environ.get("SAKAI_SPEAKER_EMBEDDING_MODEL", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(
            "Biến môi trường SAKAI_SPEAKER_EMBEDDING_MODEL đã được đặt nhưng tệp không tồn tại: "
            f"{candidate}"
        )

    project_root = Path(__file__).resolve().parents[2]
    appdata = Path(os.environ.get("APPDATA", str(Path.home())))
    candidates = [
        project_root / "models" / "speaker_embedding" / "campplus.onnx",
        project_root / "models" / "speaker_embedding" / "campplus_cn_en_common_200k.onnx",
        project_root / "models" / "speaker_embedding" / "campplus_cn_common_200k.onnx",
        project_root / "models" / "speaker_embedding" / "cam++.onnx",
        project_root / "models" / "speaker_embedding" / "cam_plus.onnx",
        appdata / "Sakai-Studio" / "models" / "speaker_embedding" / "campplus.onnx",
        appdata / "Sakai-Studio" / "models" / "speaker_embedding" / "campplus_cn_en_common_200k.onnx",
        appdata / "Sakai-Studio" / "models" / "speaker_embedding" / "cam++.onnx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def download_speaker_embedding_model(
    target_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Any] = None,
) -> str:
    """Tự động tải mô hình nhận diện người nói CAM++ ONNX (~27 MB) từ HuggingFace."""
    if target_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        target_dir = project_root / "models" / "speaker_embedding"
    else:
        target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "campplus.onnx"
    if target_file.is_file() and target_file.stat().st_size > 1000:
        return str(target_file.resolve())

    alt_file = target_dir / "campplus_cn_en_common_200k.onnx"
    if alt_file.is_file() and alt_file.stat().st_size > 1000:
        return str(alt_file.resolve())

    try:
        from huggingface_hub import hf_hub_download

        if progress_callback:
            progress_callback(0.1, "Đang tải mô hình nhận diện người nói CAM++ ONNX từ kho lưu trữ...")
        downloaded = hf_hub_download(
            repo_id="welcomyou/campplus-3dspeaker-200k-onnx",
            filename="campplus_cn_en_common_200k.onnx",
            local_dir=str(target_dir),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.is_file() and not target_file.is_file():
            import shutil

            shutil.copyfile(str(downloaded_path), str(target_file))
        return str(target_file.resolve()) if target_file.is_file() else str(downloaded_path.resolve())
    except Exception as e:
        logger.error(f"Không thể tự động tải mô hình CAM++ ONNX: {e}")
        raise RuntimeError(
            f"Không thể tự động tải mô hình nhận diện người nói CAM++ ONNX: {e}. "
            "Bạn có thể tải thủ công từ https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx "
            "và đặt vào thư mục 'models/speaker_embedding/campplus.onnx'."
        )


def create_speaker_session(
    model_path: Union[str, Path, None] = None,
    device_id: int = 0,
    gpu_mem_limit_mb: int = 768,
    prefer_cuda: bool = True,
    strict_cuda: bool = False,
    allow_mock: bool = False,
) -> Tuple[Any, str]:
    """
    Nạp mô hình ONNX thật và ưu tiên CUDAExecutionProvider.
    allow_mock chỉ dành cho test và phải được bật tường minh.
    """
    resolved_path = ""
    try:
        resolved_path = resolve_speaker_model_path(model_path)
    except FileNotFoundError:
        if not allow_mock:
            raise

    if not resolved_path:
        if allow_mock:
            return MockSpeakerSession(192), "MockExecutionProvider"
        raise FileNotFoundError(
            "Chưa cài mô hình nhận diện người nói CAM++ ONNX. Đặt tệp tại "
            "'models/speaker_embedding/campplus.onnx' hoặc cấu hình biến "
            "SAKAI_SPEAKER_EMBEDDING_MODEL. Ứng dụng không chạy nhận diện giả lập."
        )
    if ort is None:
        raise ImportError(
            "Chưa cài ONNX Runtime. Hãy cài onnxruntime-gpu tương thích CUDA để dùng RTX 4050 "
            "hoặc onnxruntime cho chế độ CPU."
        )

    available = set(ort.get_available_providers())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": int(device_id),
                "gpu_mem_limit": int(gpu_mem_limit_mb) * 1024 * 1024,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            },
        ))
        providers.append("CPUExecutionProvider")
    elif prefer_cuda and strict_cuda:
        raise RuntimeError(
            "Đã yêu cầu GPU nhưng ONNX Runtime không có CUDAExecutionProvider. "
            "Kiểm tra onnxruntime-gpu, CUDA/cuDNN và trình điều khiển NVIDIA."
        )
    else:
        providers.append("CPUExecutionProvider")
        if prefer_cuda:
            logger.warning("CUDAExecutionProvider không khả dụng; dùng CPU theo cấu hình dự phòng.")

    session = ort.InferenceSession(
        resolved_path,
        sess_options=options,
        providers=providers,
    )
    active_providers = session.get_providers()
    if prefer_cuda and strict_cuda and "CUDAExecutionProvider" not in active_providers:
        del session
        raise RuntimeError("Không khởi tạo được phiên ONNX Runtime trên CUDA.")
    actual_provider = (
        "CUDAExecutionProvider"
        if prefer_cuda and "CUDAExecutionProvider" in active_providers
        else "CPUExecutionProvider"
    )

    output_shape = session.get_outputs()[0].shape
    final_dim = output_shape[-1] if output_shape else None
    if isinstance(final_dim, int) and final_dim != 192:
        del session
        raise ValueError(
            f"Mô hình speaker embedding xuất {final_dim} chiều; cấu hình hiện yêu cầu 192 chiều."
        )
    return session, actual_provider


def _preprocess_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Chuẩn hóa âm thanh đầu vào thành mono float32 và loại bỏ DC."""
    audio = np.asarray(audio_data)
    if audio.size == 0:
        raise ValueError("Đoạn âm thanh rỗng.")
    if sample_rate != 16000:
        raise ValueError(f"Mô hình speaker embedding yêu cầu 16 kHz; nhận được {sample_rate} Hz.")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    if not np.all(np.isfinite(audio)):
        raise ValueError("Âm thanh có chứa NaN hoặc vô cực.")
    audio = audio - float(np.mean(audio))
    if len(audio) < int(0.25 * sample_rate):
        audio = np.pad(audio, (0, int(0.25 * sample_rate) - len(audio)))
    return audio


def extract_fbank_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """
    Trích xuất FBank 80 kênh theo chuẩn Kaldi kết hợp chuẩn hóa CMVN per-utterance.
    Hỗ trợ kaldi-native-fbank, torchaudio Kaldi FBank và thuật toán dự phòng thuần NumPy.
    """
    cleaned_audio = _preprocess_audio(audio, sample_rate=sample_rate)

    # 1. Ưu tiên kaldi-native-fbank nếu có sẵn trong môi trường
    try:
        import kaldi_native_fbank as knf
        options = knf.FbankOptions()
        options.frame_opts.samp_freq = sample_rate
        options.frame_opts.dither = 0.0
        options.mel_opts.num_bins = num_mel_bins
        extractor = knf.OnlineFbank(options)
        extractor.accept_waveform(sample_rate, cleaned_audio.astype(np.float32, copy=False) * 32768.0)
        extractor.input_finished()
        count = int(extractor.num_frames_ready)
        if count > 0:
            features = np.stack(
                [extractor.get_frame(i) for i in range(count)],
                axis=0,
            ).astype(np.float32, copy=False)
            features -= features.mean(axis=0, keepdims=True)
            return features
    except (ImportError, Exception):
        pass

    # 2. Thử nghiệm torchaudio.compliance.kaldi.fbank
    try:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        t_waveform = torch.from_numpy(cleaned_audio).unsqueeze(0).float() * 32768.0
        fbank = kaldi.fbank(
            t_waveform,
            sample_frequency=float(sample_rate),
            num_mel_bins=num_mel_bins,
            frame_length=float(frame_length_ms),
            frame_shift=float(frame_shift_ms),
            dither=0.0,
            energy_floor=0.0,
            window_type="hamming",
            use_energy=False,
        )
        features = fbank.numpy().astype(np.float32)
        features -= features.mean(axis=0, keepdims=True)
        return features
    except (ImportError, Exception):
        pass

    # 3. Thuật toán dự phòng thuần NumPy
    win_len = int(frame_length_ms * sample_rate / 1000.0)
    win_shift = int(frame_shift_ms * sample_rate / 1000.0)
    n_fft = 1
    while n_fft < win_len:
        n_fft <<= 1

    pad_audio = cleaned_audio
    if len(pad_audio) < win_len:
        pad_audio = np.pad(pad_audio, (0, win_len - len(pad_audio)))

    num_frames = max(1, 1 + (len(pad_audio) - win_len) // win_shift)
    frames = np.lib.stride_tricks.sliding_window_view(pad_audio[: (num_frames - 1) * win_shift + win_len], win_len)[::win_shift]
    window = np.hamming(win_len).astype(np.float32)
    windowed = frames * window

    spec = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    hz_to_mel = lambda hz: 1127.0 * np.log(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (np.exp(mel / 1127.0) - 1.0)
    low_mel = hz_to_mel(20.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, num_mel_bins + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank_filters = np.zeros((num_mel_bins, n_fft // 2 + 1), dtype=np.float32)
    for m in range(num_mel_bins):
        f_m_minus = bin_points[m]
        f_m = bin_points[m + 1]
        f_m_plus = bin_points[m + 2]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fbank_filters[m, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fbank_filters[m, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = np.dot(spec, fbank_filters.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec).astype(np.float32)
    log_mel -= np.mean(log_mel, axis=0, keepdims=True)
    return log_mel


def _extract_fbank_features(audio: np.ndarray) -> np.ndarray:
    """Trích xuất FBank 80 chiều theo hợp đồng CAM++/3D-Speaker."""
    return extract_fbank_features(audio, sample_rate=16000, num_mel_bins=80)


def _make_model_input(session: Any, audio: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Tạo tensor theo metadata ONNX. CAM++ 3D-Speaker nhận FBank
    (batch, frames, 80); chỉ dùng sóng âm thô khi metadata xác định rõ kiểu đầu vào.
    """
    meta = session.get_inputs()[0]
    name = (meta.name or "").lower()
    shape = list(meta.shape or [])
    rank = len(shape)

    # Với CAM++ chuẩn, tên input thường là "input" và tensor có 3 chiều.
    # Chỉ coi là waveform nếu tên hoặc shape xác định rõ kênh đơn ở chiều giữa.
    wave_name = any(token in name for token in ("waveform", "pcm", "raw_audio"))
    wave_shape = rank == 3 and shape[1] == 1 and shape[-1] != 80
    wave_input = wave_name or wave_shape

    if rank >= 3 and not wave_input:
        features = _extract_fbank_features(audio)
        if shape[1] == 80 and shape[2] != 80:
            return features.T[np.newaxis, :, :], 2
        return features[np.newaxis, :, :], 1

    if rank == 2 and not wave_input and (
        "feat" in name or "feature" in name or shape[-1] == 80 or shape[0] == 80
    ):
        features = _extract_fbank_features(audio)
        if shape[0] == 80 and shape[1] != 80:
            return features.T, 1
        return features, 0

    if rank >= 3:
        return audio[np.newaxis, np.newaxis, :], 2
    if rank == 2:
        return audio[np.newaxis, :], 1
    return audio, 0

def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if emb.size != 192:
        raise ValueError(f"Mô hình trả vector {emb.size} chiều, cần đúng 192 chiều.")
    norm = float(np.linalg.norm(emb))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Mô hình trả vector embedding rỗng hoặc không hợp lệ.")
    return (emb / norm).astype(np.float32)


def extract_speaker_embedding(
    session: Any,
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Trích xuất một vector embedding người nói chuẩn hóa L2."""
    audio = _preprocess_audio(audio_data, sample_rate)
    if isinstance(session, MockSpeakerSession):
        output = session.run(None, {"speech": audio[np.newaxis, :]})[0]
        return _normalize_embedding(output[0])

    tensor, _ = _make_model_input(session, audio)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = np.asarray(session.run([output_name], {input_name: tensor})[0])
    if output.ndim == 0:
        raise ValueError("Mô hình không trả tensor embedding hợp lệ.")
    if output.ndim > 1:
        output = output[0]
    return _normalize_embedding(output)


def batch_extract_speaker_embeddings(
    session: Any,
    audio_chunks: List[np.ndarray],
    sample_rate: int = 16000,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Trích xuất theo từng nhóm nhỏ để giới hạn VRAM.
    Các đoạn có độ dài khác nhau được suy luận riêng để tránh padding làm lệch FBank.
    """
    if not audio_chunks:
        return np.empty((0, 192), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn 0.")

    embeddings: List[np.ndarray] = []
    for start in range(0, len(audio_chunks), batch_size):
        for chunk in audio_chunks[start : start + batch_size]:
            embeddings.append(extract_speaker_embedding(session, chunk, sample_rate))
    return np.stack(embeddings, axis=0)


def close_speaker_session(session: Any) -> None:
    """Giải phóng tham chiếu phiên làm việc ONNX."""
    if session is None:
        return
    try:
        close = getattr(session, "close", None)
        if callable(close):
            close()
    except Exception:
        logger.debug("Phiên ONNX không hỗ trợ close() tường minh.", exc_info=True)
    del session
    gc.collect()
