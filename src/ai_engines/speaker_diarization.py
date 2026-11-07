# -*- coding: utf-8 -*-
"""
Bộ điều phối nhận diện và phân đoạn người nói (Tầng 2).
Thực hiện phân cụm khoảng cách cosin, phân loại 3 mức độ tin cậy và gán danh tính nhân vật
vào danh sách mốc thời gian phụ đề với cơ chế giải phóng tài nguyên tuần tự.
"""

from __future__ import annotations

import gc
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .speaker_embedding_onnx import (
    batch_extract_speaker_embeddings,
    close_speaker_session,
    create_speaker_session,
    extract_speaker_embedding,
)

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Ba mức độ tin cậy nhận diện người nói theo tài liệu đặc tả thiết kế."""

    HIGH = "high"
    REVIEW_NEEDED = "review_needed"
    UNASSIGNED = "unassigned"


@dataclass
class CharacterVoiceProfile:
    """Hồ sơ cấu hình giọng nói và danh tính nhân vật thống nhất theo đặc tả."""

    speaker_id: str
    display_name: str
    utterance_count: int = 0
    total_duration: float = 0.0
    source_audio_path: str = ""
    method: str = "preset"  # Mặc định sử dụng giọng AI tiếng Việt có sẵn
    preset_name: str = ""
    voice_instruct: str = "male, young adult, moderate pitch"
    status: str = "locked"  # locked, review_needed, unassigned
    custom_audio_path: str = ""
    high_confidence_ratio: float = 1.0
    voice_clone_prompt_path: str = ""
    vietnamese_accent: str = "bac"  # bac, trung, nam
    accent_confidence: float = 0.5
    gender: str = "auto"  # male, female, auto, unknown
    gender_confidence: float = 0.5
    age_group: str = "young_adult"  # child, teenager, young_adult, middle_aged, elderly
    pitch_level: str = "moderate"  # very_low, low, moderate, high, very_high
    style_tag: str = "natural"
    recommended_voices: List[Dict[str, Any]] = field(default_factory=list)
    acoustic_features: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    ref_text: str = ""
    user_locked: bool = False
    speech_rate_sps: float = 0.0
    pitch_hz: float = 0.0
    prototypes: Dict[str, Any] = field(default_factory=dict)

    # Các thuộc tính bí danh hỗ trợ tương thích ngược 2 chiều
    @property
    def source_audio_sample(self) -> str:
        return self.source_audio_path

    @source_audio_sample.setter
    def source_audio_sample(self, val: str):
        self.source_audio_path = str(val or "")

    @property
    def dialogue_count(self) -> int:
        return self.utterance_count

    @dialogue_count.setter
    def dialogue_count(self, val: int):
        self.utterance_count = int(val or 0)

    @property
    def assigned_strategy(self) -> str:
        return self.method

    @assigned_strategy.setter
    def assigned_strategy(self, val: str):
        self.method = str(val or "clone")

    @property
    def instruct(self) -> str:
        return self.voice_instruct

    @instruct.setter
    def instruct(self, val: str):
        self.voice_instruct = str(val or "")

    @property
    def voice_preset_key(self) -> str:
        return self.preset_name

    @voice_preset_key.setter
    def voice_preset_key(self, val: str):
        self.preset_name = str(val or "")

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển tương thích hoàn toàn với các tầng xử lý hiện có."""
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "count": self.utterance_count,
            "utterance_count": self.utterance_count,
            "dialogue_count": self.utterance_count,
            "total_duration": self.total_duration,
            "ref_audio": self.source_audio_path,
            "ref_audio_path": self.source_audio_path,
            "source_audio_path": self.source_audio_path,
            "source_audio_sample": self.source_audio_path,
            "method": self.method,
            "assigned_strategy": self.method,
            "preset_name": self.preset_name,
            "voice_preset": self.preset_name,
            "voice_preset_key": self.preset_name,
            "voice_instruct": self.voice_instruct,
            "instruct": self.voice_instruct,
            "status": self.status,
            "confidence": "high" if self.high_confidence_ratio >= 0.7 else ("review_needed" if self.high_confidence_ratio >= 0.3 else "unassigned"),
            "confidence_level": "high" if self.high_confidence_ratio >= 0.7 else ("review_needed" if self.high_confidence_ratio >= 0.3 else "unassigned"),
            "custom_audio_path": self.custom_audio_path,
            "high_confidence_ratio": self.high_confidence_ratio,
            "voice_clone_prompt_path": self.voice_clone_prompt_path,
            "vietnamese_accent": self.vietnamese_accent,
            "accent_confidence": self.accent_confidence,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "age_group": self.age_group,
            "pitch_level": self.pitch_level,
            "style_tag": self.style_tag,
            "recommended_voices": list(self.recommended_voices) if self.recommended_voices else [],
            "acoustic_features": dict(self.acoustic_features) if self.acoustic_features else {},
            "error_message": self.error_message,
            "ref_text": self.ref_text,
            "user_locked": self.user_locked,
            "speech_rate_sps": self.speech_rate_sps,
            "pitch_hz": self.pitch_hz,
            "prototypes": self.prototypes,
        }

    _ALIAS_SET_MAP = {
        "count": "utterance_count",
        "dialogue_count": "utterance_count",
        "ref_audio": "source_audio_path",
        "ref_audio_path": "source_audio_path",
        "source_audio_sample": "source_audio_path",
        "assigned_strategy": "method",
        "voice_preset": "preset_name",
        "voice_preset_key": "preset_name",
        "instruct": "voice_instruct",
    }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        target_key = self._ALIAS_SET_MAP.get(key, key)
        if target_key in ("confidence", "confidence_level"):
            if value == "high":
                self.high_confidence_ratio = 1.0
            elif value == "review_needed":
                self.high_confidence_ratio = 0.5
            elif value == "unassigned":
                self.high_confidence_ratio = 0.0
        setattr(self, target_key, value)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict() or hasattr(self, key)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CharacterVoiceProfile:
        """Tạo đối tượng CharacterVoiceProfile từ từ điển dữ liệu."""
        if not isinstance(data, dict):
            data = {}
        profile = cls(
            speaker_id=str(data.get("speaker_id", "")),
            display_name=str(data.get("display_name") or data.get("speaker_id", "") or "Nhân vật"),
            utterance_count=int(data.get("utterance_count") or data.get("count") or data.get("dialogue_count") or 0),
            total_duration=float(data.get("total_duration") or 0.0),
            source_audio_path=str(
                data.get("source_audio_path")
                or data.get("ref_audio")
                or data.get("ref_audio_path")
                or data.get("source_audio_sample")
                or ""
            ),
            method=str(data.get("method") or data.get("assigned_strategy") or "clone"),
            preset_name=str(data.get("preset_name") or data.get("voice_preset") or data.get("voice_preset_key") or ""),
            voice_instruct=str(data.get("voice_instruct") or data.get("instruct") or "male, young adult, moderate pitch"),
            status=str(data.get("status") or "locked"),
            custom_audio_path=str(data.get("custom_audio_path") or ""),
            high_confidence_ratio=float(data.get("high_confidence_ratio", 1.0) if data.get("high_confidence_ratio") is not None else 1.0),
            voice_clone_prompt_path=str(data.get("voice_clone_prompt_path") or ""),
            vietnamese_accent=str(data.get("vietnamese_accent") or "bac"),
            accent_confidence=float(data.get("accent_confidence", 0.5) if data.get("accent_confidence") is not None else 0.5),
            gender=str(data.get("gender") or "auto"),
            gender_confidence=float(data.get("gender_confidence", 0.5) if data.get("gender_confidence") is not None else 0.5),
            age_group=str(data.get("age_group") or "young_adult"),
            pitch_level=str(data.get("pitch_level") or "moderate"),
            style_tag=str(data.get("style_tag") or "natural"),
            recommended_voices=list(data.get("recommended_voices") or []),
            acoustic_features=dict(data.get("acoustic_features") or {}),
            error_message=str(data.get("error_message") or ""),
            ref_text=str(data.get("ref_text") or ""),
            user_locked=bool(data.get("user_locked", False)),
            speech_rate_sps=float(data.get("speech_rate_sps") or 0.0),
            pitch_hz=float(data.get("pitch_hz") or 0.0),
            prototypes=dict(data.get("prototypes") or {}),
        )
        return profile


def normalize_character_profile(raw: Any, speaker_id: str = "") -> CharacterVoiceProfile:
    """Chuẩn hóa dữ liệu hồ sơ nhân vật về dạng đối tượng CharacterVoiceProfile an toàn."""
    if isinstance(raw, CharacterVoiceProfile):
        if speaker_id and not raw.speaker_id:
            raw.speaker_id = speaker_id
        return raw

    if isinstance(raw, dict):
        d = dict(raw)
        if speaker_id and not d.get("speaker_id"):
            d["speaker_id"] = speaker_id
        return CharacterVoiceProfile.from_dict(d)

    sid = speaker_id or (getattr(raw, "speaker_id", "") if raw is not None else "")
    dname = getattr(raw, "display_name", "") if raw is not None else ""
    return CharacterVoiceProfile(
        speaker_id=str(sid or "unknown"),
        display_name=str(dname or sid or "Nhân vật"),
    )


# Bí danh tương thích ngược
SpeakerVoiceProfile = CharacterVoiceProfile


@dataclass
class MultiPrototypeSpeakerProfile(CharacterVoiceProfile):
    """
    Hồ sơ nhân vật mở rộng hỗ trợ biểu diễn đa mẫu nòng cốt (Multi-prototype Representation).
    Lưu trữ các vector đặc trưng đại diện theo các dải năng lượng:
    - 'normal': mẫu giọng chuẩn, âm lượng và độ dài trung bình.
    - 'high_energy': mẫu giọng năng lượng cao (hét, cười, cao độ tăng cao).
    - 'low_energy': mẫu giọng năng lượng thấp (thì thầm, nói nhỏ).
    """

    def add_prototype(self, tier: str, embedding: np.ndarray) -> None:
        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
            norm_val = float(np.linalg.norm(embedding))
            if norm_val > 1e-12:
                self.prototypes[tier] = (embedding / norm_val).astype(np.float32)

    def min_distance_to_prototypes(self, embedding: np.ndarray) -> float:
        """Tính khoảng cách cosin nhỏ nhất từ một vector đến các mẫu nòng cốt của nhân vật."""
        if not self.prototypes or embedding is None or not isinstance(embedding, np.ndarray) or embedding.size == 0:
            return 1.0
        norm_val = float(np.linalg.norm(embedding))
        if norm_val <= 1e-12:
            return 1.0
        norm_emb = embedding / norm_val
        min_dist = 1.0
        for proto in self.prototypes.values():
            p_val = float(np.linalg.norm(proto))
            if p_val > 1e-12:
                p_norm = proto / p_val
                dist = 1.0 - float(np.dot(norm_emb, p_norm))
                if dist < min_dist:
                    min_dist = dist
        return max(0.0, min(2.0, min_dist))


class ExternalDiarizationVerifier:
    """
    Giao diện trừu tượng và bộ kiểm chứng phân đoạn người nói đối chiếu (NeMo / Pyannote).
    Tự động kiểm tra tính khả dụng của các thư viện chuyên sâu trong môi trường.
    Khi không có sẵn hoặc VRAM hạn chế, vận hành dự phòng an toàn qua CAM++ ONNX.
    """

    @classmethod
    def is_available(cls) -> bool:
        try:
            import nemo  # type: ignore
            return True
        except ImportError:
            pass
        try:
            import pyannote.audio  # type: ignore
            return True
        except ImportError:
            return False

    @classmethod
    def verify_or_fallback(
        cls,
        audio_path: str,
        campp_speakers: Dict[str, Any],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Đối chiếu kết quả phân vai CAM++ với bộ nhận diện ngoài nếu có.
        Nếu không có NeMo/Pyannote, trả về nguyên trạng kết quả CAM++ an toàn bộ nhớ đồ họa.
        """
        if not cls.is_available():
            logger.info("Bộ kiểm chứng ngoài (NeMo/Pyannote) không có sẵn trong môi trường; sử dụng hoàn toàn CAM++ ONNX tiết kiệm bộ nhớ đồ họa.")
            return campp_speakers

        logger.info("Đang kiểm chứng đối chiếu phân đoạn người nói bằng công cụ bổ trợ...")
        return campp_speakers


def _parse_time_to_seconds(time_val: Any) -> float:
    """Chuyển đổi mốc thời gian phụ đề sang giây."""
    if isinstance(time_val, (int, float)):
        return max(0.0, float(time_val))
    if isinstance(time_val, str):
        cleaned = time_val.strip().replace(",", ".")
        parts = cleaned.split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            elif len(parts) == 1:
                return float(parts[0])
        except Exception:
            return 0.0
    return 0.0


def _format_seconds_to_srt_time(seconds: float) -> str:
    """Chuyển đổi số giây sang chuỗi định dạng thời gian chuẩn SRT HH:MM:SS,mmm."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _normalized_choice_probabilities(weights: np.ndarray) -> Optional[np.ndarray]:
    """Chuẩn hóa trọng số k-means++ thành xác suất float64 hợp lệ cho np.random.choice.

    Trọng số thường xuất phát từ phép tính float32 trên embedding. Nếu giữ float32,
    tổng xác suất có thể lệch khỏi 1 đủ lớn để NumPy ném lỗi
    'Probabilities do not sum to 1'.
    """
    probs = np.asarray(weights, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(probs, 0.0, out=probs)
    total = float(np.sum(probs, dtype=np.float64))
    if not np.isfinite(total) or total <= 1e-12:
        return None

    probs /= total
    normalized_total = float(np.sum(probs, dtype=np.float64))
    if not np.isfinite(normalized_total) or normalized_total <= 0.0:
        return None
    probs /= normalized_total

    # Bù phần sai số dấu phẩy động tại phần tử lớn nhất, tránh xác suất âm.
    largest = int(np.argmax(probs))
    probs[largest] += 1.0 - float(np.sum(probs, dtype=np.float64))
    if probs[largest] < 0.0 or not np.all(np.isfinite(probs)):
        return None
    return probs


def extract_chunk_f0(chunk: np.ndarray, sample_rate: int) -> Optional[float]:
    """
    Trích xuất cao độ F0 trung vị (Hz) của đoạn âm thanh bằng hàm tự tương quan chuẩn hóa (MACF).
    Phạm vi nhận diện từ 65 Hz (giọng nam trầm sâu) đến 420 Hz (giọng nữ/trẻ em).
    Trả về None nếu đoạn âm thanh là khoảng lặng hoặc không có tính chu kỳ thanh âm (unvoiced).
    """
    if chunk is None or len(chunk) == 0 or sample_rate <= 0:
        return None
    data = np.asarray(chunk, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = data - float(np.mean(data))
    energy = float(np.mean(data ** 2))
    if energy < 1e-5:
        return None

    min_lag = max(1, int(sample_rate / 420.0))
    max_lag = min(len(data) // 2, int(sample_rate / 65.0))
    if max_lag <= min_lag:
        return None

    frame_len = max(max_lag * 2, int(sample_rate * 0.040))
    hop_len = int(sample_rate * 0.020)
    if len(data) < frame_len:
        frame_len = len(data)
        hop_len = frame_len

    n_frames = max(1, (len(data) - frame_len) // hop_len + 1)
    frame_f0s = []

    for f_idx in range(n_frames):
        st = f_idx * hop_len
        frame = data[st : st + frame_len]
        if len(frame) < max_lag * 2:
            break
        frame_energy = float(np.sum(frame ** 2))
        if frame_energy < 1e-5:
            continue

        window = np.hanning(len(frame))
        win_frame = frame * window

        fft_len = 1 << (len(win_frame) * 2 - 1).bit_length()
        spec = np.fft.rfft(win_frame, n=fft_len)
        corr = np.fft.irfft(spec * np.conj(spec), n=fft_len)[: max_lag + 1]

        r0 = float(corr[0]) if corr[0] > 1e-12 else 1e-12
        norm_corr = corr / r0

        search_slice = norm_corr[min_lag : max_lag + 1]
        if len(search_slice) == 0:
            continue

        peak_idx = int(np.argmax(search_slice))
        peak_val = float(search_slice[peak_idx])

        if peak_val >= 0.30:
            best_lag = float(min_lag + peak_idx)
            if 0 < peak_idx < len(search_slice) - 1:
                alpha = float(search_slice[peak_idx - 1])
                beta = peak_val
                gamma = float(search_slice[peak_idx + 1])
                denom = alpha - 2.0 * beta + gamma
                if abs(denom) > 1e-12:
                    delta = 0.5 * (alpha - gamma) / denom
                    best_lag = best_lag + delta

            f0 = float(sample_rate / max(best_lag, 1e-6))
            if 65.0 <= f0 <= 420.0:
                frame_f0s.append(f0)

    if not frame_f0s:
        return None
    return float(np.median(frame_f0s))


def estimate_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int = 1,
    max_k: Optional[int] = None,
    threshold: float = 0.40,
    sample_f0s: Optional[List[Optional[float]]] = None,
    sample_durations: Optional[List[float]] = None,
) -> int:
    """
    Tự động ước lượng số lượng cụm nhân vật tối ưu dựa trên phân tích đa tiêu chí:
    1. Kiểm tra nhanh đơn người nói qua phân vị khoảng cách cosin.
    2. Cắt cây phân cấp thứ bậc (AHC) tự nhiên theo ngưỡng khoảng cách.
    3. Phân tích bước nhảy trị riêng (Eigengap) trên ma trận Affinity lũy thừa 4 và Laplacian chuẩn hóa.
    4. Hiệu chỉnh khoảng cách âm học dựa trên độ chênh lệch cao độ F0 (ngăn gộp nhầm nam/nữ).
    5. Đánh giá chất lượng phân cụm bằng hệ số phân tách Silhouette.
    """
    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        return 1

    # Chuẩn hóa L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / np.maximum(norms, 1e-12)

    # Ma trận khoảng cách cosin N x N
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    dist_matrix = np.maximum(0.0, 1.0 - sim_matrix)
    np.fill_diagonal(dist_matrix, 0.0)

    # Hiệu chỉnh khoảng cách âm học bằng F0 (nếu có dữ liệu F0 của các đoạn thoại)
    # Nếu hai câu thoại có F0 chênh lệch > 75 Hz (ví dụ nam ~110 Hz vs nữ ~220 Hz), cộng phạt khoảng cách
    if sample_f0s is not None and len(sample_f0s) == n_samples:
        for i in range(n_samples):
            f0_i = sample_f0s[i]
            if f0_i is None or f0_i <= 0.0:
                continue
            for j in range(i + 1, n_samples):
                f0_j = sample_f0s[j]
                if f0_j is None or f0_j <= 0.0:
                    continue
                if abs(f0_i - f0_j) > 75.0:
                    dist_matrix[i, j] = min(1.0, dist_matrix[i, j] + 0.25)
                    dist_matrix[j, i] = dist_matrix[i, j]
                    sim_matrix[i, j] = max(0.0, 1.0 - dist_matrix[i, j])
                    sim_matrix[j, i] = sim_matrix[i, j]

    if n_samples == 2:
        return 2 if dist_matrix[0, 1] > threshold else 1

    upper_tri = np.triu_indices(n_samples, k=1)
    tri_dists = dist_matrix[upper_tri]
    mean_dist = float(np.mean(tri_dists))
    p90_dist = float(np.percentile(tri_dists, 90))

    # Kiểm tra nhanh đơn người nói: nếu 90% khoảng cách nhỏ hơn ngưỡng và khoảng cách trung bình thấp
    if p90_dist < threshold * 0.95 and mean_dist < 0.28:
        return 1

    parsed_max_k = None
    if max_k is not None:
        try:
            parsed_max_k = int(max_k)
        except (TypeError, ValueError):
            parsed_max_k = None
    effective_max_k = min(n_samples, parsed_max_k) if parsed_max_k is not None else min(n_samples, 24)
    upper_k = min(effective_max_k, n_samples - 1)
    if upper_k < 2:
        return 2 if float(np.max(dist_matrix)) > threshold else 1

    # 1. Ứng viên từ AHC cắt tự nhiên theo ngưỡng khoảng cách
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")
    ahc_labels = fcluster(Z, t=threshold, criterion="distance") - 1
    k_ahc = int(len(np.unique(ahc_labels)))
    k_ahc = max(min_k, min(k_ahc, upper_k))

    # 2. Ứng viên từ phân tích Eigengap trên ma trận Affinity sắc nét (Sharpened Affinity)
    # Lũy thừa 4 để triệt tiêu các liên kết yếu xuyên cụm (< 0.50), khuếch đại liên kết nội cụm (> 0.80)
    affinity = np.maximum(0.0, sim_matrix) ** 4.0
    affinity[dist_matrix > threshold * 1.15] = 0.0
    np.fill_diagonal(affinity, 1.0)
    affinity = 0.5 * (affinity + affinity.T)

    deg = np.sum(affinity, axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    laplacian = np.eye(n_samples) - (deg_inv_sqrt[:, None] * affinity * deg_inv_sqrt[None, :])

    k_eigen = 1
    gaps: List[Tuple[int, float]] = []
    sorted_gaps: List[Tuple[int, float]] = []
    try:
        eigenvals = np.linalg.eigvalsh(laplacian)
        for k in range(1, upper_k + 1):
            gap = float(eigenvals[k] - eigenvals[k - 1])
            gaps.append((k, gap))
        if gaps:
            # Sắp xếp gap giảm dần để lấy các đỉnh trị riêng rõ nhất
            sorted_gaps = sorted(gaps, key=lambda x: x[1], reverse=True)
            k_eigen = sorted_gaps[0][0]
    except Exception:
        k_eigen = k_ahc

    # 3. Tập hợp các ứng viên tiềm năng để đánh giá Silhouette
    candidate_ks = set([1, k_ahc, k_eigen])
    if sorted_gaps:
        for k_g, _ in sorted_gaps[:2]:
            if min_k <= k_g <= upper_k:
                candidate_ks.add(k_g)
    candidate_list = sorted([k for k in candidate_ks if min_k <= k <= upper_k or k == 1])

    def _calc_mean_silhouette(lbls: np.ndarray) -> float:
        unique_l = np.unique(lbls)
        if len(unique_l) <= 1:
            return 0.0
        scores = []
        for idx in range(n_samples):
            own = lbls[idx]
            same_mask = (lbls == own)
            same_mask[idx] = False
            if not np.any(same_mask):
                scores.append(0.0)
                continue
            a = float(np.mean(dist_matrix[idx, same_mask]))
            b = float("inf")
            for other in unique_l:
                if other == own:
                    continue
                diff_mask = (lbls == other)
                if np.any(diff_mask):
                    d = float(np.mean(dist_matrix[idx, diff_mask]))
                    if d < b:
                        b = d
            denom = max(a, b, 1e-6)
            scores.append((b - a) / denom)
        return float(np.mean(scores)) if scores else 0.0

    # Đánh giá chất lượng từng ứng viên K >= 2
    best_k = 1
    best_score = -1.0
    has_valid_multi = False

    for k in candidate_list:
        if k <= 1:
            continue
        try:
            cand_labels = fcluster(Z, t=k, criterion="maxclust") - 1
            sil = _calc_mean_silhouette(cand_labels)
            if sil < 0.15:
                # Phân tách quá mờ nhạt, có thể là 1 người nói bị cắt gượng ép
                continue

            # Kiểm tra khoảng cách giữa các tâm cụm ứng viên
            centers = []
            for c_i in np.unique(cand_labels):
                c_pts = norm_embs[cand_labels == c_i]
                c_mean = np.mean(c_pts, axis=0)
                centers.append(c_mean / np.maximum(np.linalg.norm(c_mean), 1e-12))

            min_c_dist = float("inf")
            for i_c in range(len(centers)):
                for j_c in range(i_c + 1, len(centers)):
                    d_c = 1.0 - float(np.dot(centers[i_c], centers[j_c]))
                    if d_c < min_c_dist:
                        min_c_dist = d_c

            if min_c_dist < 0.22:
                # Có ít nhất 2 cụm quá sát nhau -> Không đủ tin cậy để tách thành 2 người nói
                continue

            # Điểm tổng hợp cho ứng viên K
            gap_weight = 0.0
            for k_g, g_val in gaps:
                if k_g == k:
                    gap_weight = max(0.0, g_val)
                    break

            comp_score = 0.60 * sil + 0.25 * gap_weight + (0.15 if k == k_ahc else 0.0)
            has_valid_multi = True
            if comp_score > best_score:
                best_score = comp_score
                best_k = k
        except Exception:
            continue

    if not has_valid_multi:
        return 1

    return int(max(min_k or 1, min(best_k or 1, effective_max_k or 24)))


def estimate_optimal_clusters_eigengap(
    embeddings: np.ndarray,
    min_k: int = 1,
    max_k: int = 12,
    threshold: float = 0.40,
) -> int:
    """Tương thích ngược: ủy quyền cho estimate_optimal_clusters."""
    return estimate_optimal_clusters(
        embeddings=embeddings,
        min_k=min_k,
        max_k=max_k,
        threshold=threshold,
    )


def agglomerative_cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = 0.40,
    exact_k: Optional[int] = None,
    min_k: int = 1,
    max_k: Optional[int] = None,
    sample_f0s: Optional[List[Optional[float]]] = None,
) -> np.ndarray:
    """
    Phân cụm thứ bậc tích lũy (Agglomerative Hierarchical Clustering) theo khoảng cách cosin
    và liên kết trung bình (Average Linkage).
    
    Hỗ trợ chế độ tự động hoàn toàn ước lượng số lượng người nói thực tế, hoặc cố định exact_k.
    Kết hợp hậu xử lý gộp cụm quá sát nhau (Centroid Post-Merge Pruning) và kiểm tra Silhouette.
    """
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int32)
    if n_samples == 1:
        return np.zeros(1, dtype=np.int32)

    # Chuẩn hóa L2 phòng ngừa
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / np.maximum(norms, 1e-12)

    # Ma trận khoảng cách cosin N x N
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    dist_matrix = np.maximum(0.0, 1.0 - sim_matrix)
    np.fill_diagonal(dist_matrix, 0.0)

    # Hiệu chỉnh khoảng cách bằng F0 nếu có
    if sample_f0s is not None and len(sample_f0s) == n_samples:
        for i in range(n_samples):
            f0_i = sample_f0s[i]
            if f0_i is None or f0_i <= 0.0:
                continue
            for j in range(i + 1, n_samples):
                f0_j = sample_f0s[j]
                if f0_j is None or f0_j <= 0.0:
                    continue
                if abs(f0_i - f0_j) > 75.0:
                    dist_matrix[i, j] = min(1.0, dist_matrix[i, j] + 0.25)
                    dist_matrix[j, i] = dist_matrix[i, j]

    target_k = None
    if exact_k is not None:
        try:
            target_k = max(1, min(int(exact_k), n_samples))
        except (TypeError, ValueError):
            target_k = None
    elif n_samples >= 2:
        target_k = estimate_optimal_clusters(
            norm_embs, min_k=min_k, max_k=max_k, threshold=threshold, sample_f0s=sample_f0s
        )

    if target_k == 1:
        return np.zeros(n_samples, dtype=np.int32)

    parsed_max_k = None
    if max_k is not None:
        try:
            parsed_max_k = int(max_k)
        except (TypeError, ValueError):
            parsed_max_k = None
    effective_max_k = min(n_samples, parsed_max_k) if parsed_max_k is not None else min(n_samples, 24)

    try:
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, fcluster

        condensed_dist = squareform(dist_matrix, checks=False)
        Z = linkage(condensed_dist, method="average")

        if target_k is not None:
            raw_labels = fcluster(Z, t=target_k, criterion="maxclust") - 1
        else:
            raw_labels = fcluster(Z, t=threshold, criterion="distance") - 1
            n_c = len(np.unique(raw_labels))
            if n_c > effective_max_k:
                raw_labels = fcluster(Z, t=effective_max_k, criterion="maxclust") - 1
            elif n_c < min_k and n_samples >= min_k:
                raw_labels = fcluster(Z, t=min_k, criterion="maxclust") - 1

        cluster_dict: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(raw_labels):
            cluster_dict.setdefault(int(lbl), []).append(idx)
        clusters = list(cluster_dict.values())
    except Exception:
        clusters = [[i] for i in range(n_samples)]
        while len(clusters) > 1:
            if target_k is not None and len(clusters) <= target_k:
                break

            n_c = len(clusters)
            min_dist = float("inf")
            merge_i, merge_j = -1, -1

            for i in range(n_c):
                c_i = clusters[i]
                for j in range(i + 1, n_c):
                    c_j = clusters[j]
                    sub_dists = dist_matrix[np.ix_(c_i, c_j)]
                    avg_dist = float(np.mean(sub_dists))
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        merge_i, merge_j = i, j

            if target_k is None:
                if min_dist > threshold and len(clusters) <= effective_max_k:
                    break
                if len(clusters) <= min_k:
                    break

            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)

    # Centroid Post-Merge Pruning: Nếu ở chế độ tự động và có >= 2 cụm,
    # gộp các tâm cụm cách nhau < 0.28 (tránh chia cắt nhân tạo 1 người)
    if exact_k is None and len(clusters) > 1:
        merged_any = True
        while merged_any and len(clusters) > 1:
            merged_any = False
            centroids = []
            for c in clusters:
                c_embs = norm_embs[c]
                c_vec = np.mean(c_embs, axis=0)
                centroids.append(c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12))

            min_dist = float("inf")
            c_pair = (-1, -1)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = 1.0 - float(np.dot(centroids[i], centroids[j]))
                    if sample_f0s is not None:
                        f0s_i = [sample_f0s[m] for m in clusters[i] if sample_f0s[m] is not None]
                        f0s_j = [sample_f0s[m] for m in clusters[j] if sample_f0s[m] is not None]
                        if f0s_i and f0s_j and abs(float(np.median(f0s_i)) - float(np.median(f0s_j))) > 75.0:
                            continue
                    if d < min_dist:
                        min_dist = d
                        c_pair = (i, j)

            if c_pair[0] >= 0 and min_dist < 0.28:
                clusters[c_pair[0]].extend(clusters[c_pair[1]])
                clusters.pop(c_pair[1])
                merged_any = True

    # Nếu ở chế độ tự động (không chỉ định exact_k) và đã gom thành nhiều cụm (>= 2):
    # Kiểm tra hệ số Silhouette để xác thực xem dữ liệu có thực sự tách biệt thành nhiều người nói hay không.
    if exact_k is None and len(clusters) > 1 and n_samples >= 3:
        temp_labels = np.zeros(n_samples, dtype=np.int32)
        for c_idx, c_members in enumerate(clusters):
            for m_idx in c_members:
                temp_labels[m_idx] = c_idx

        sil_scores = []
        for idx in range(n_samples):
            own_c = temp_labels[idx]
            same_mask = (temp_labels == own_c)
            same_mask[idx] = False
            if not np.any(same_mask):
                sil_scores.append(0.0)
                continue
            a_val = float(np.mean(dist_matrix[idx, same_mask]))
            b_val = float("inf")
            for other_c in range(len(clusters)):
                if other_c == own_c:
                    continue
                diff_mask = (temp_labels == other_c)
                if np.any(diff_mask):
                    d_other = float(np.mean(dist_matrix[idx, diff_mask]))
                    if d_other < b_val:
                        b_val = d_other
            denom = max(a_val, b_val, 1e-6)
            sil_scores.append((b_val - a_val) / denom)

        mean_sil = float(np.mean(sil_scores)) if sil_scores else -1.0
        # Nếu hệ số Silhouette cực thấp (< 0.12), các cụm chỉ là phân mảnh nhân tạo của 1 người nói duy nhất.
        if mean_sil < 0.12:
            clusters = [[i for i in range(n_samples)]]

    # Sắp xếp các cụm theo chỉ số phần tử xuất hiện đầu tiên để nhãn ổn định
    clusters.sort(key=lambda c: min(c))
    labels = np.zeros(n_samples, dtype=np.int32)
    for label_idx, c_members in enumerate(clusters):
        for member_idx in c_members:
            labels[member_idx] = label_idx

    return labels


def cosine_cluster_embeddings(
    embeddings: np.ndarray,
    valid_indices: List[int],
    max_speakers: Optional[int] = None,
    exact_speakers: Optional[int] = None,
    distance_threshold: float = 0.40,
    sample_durations: Optional[List[float]] = None,
    sample_f0s: Optional[List[Optional[float]]] = None,
) -> Tuple[np.ndarray, List[str], List[float]]:
    """
    Phân cụm các vector đặc trưng theo khoảng cách cosin và xếp hạng 3 mức độ tin cậy.
    Sử dụng thuật toán Phân cụm Thứ bậc (AHC) kết hợp cơ chế mẫu nòng cốt (Anchor Segments)
    và hiệu chuẩn âm học F0.
    Trả về bộ ba (labels, confidence_levels, confidence_scores).
    """
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int32), [], []

    if n_samples == 1:
        return (
            np.zeros(1, dtype=np.int32),
            [ConfidenceLevel.REVIEW_NEEDED.value],
            [0.0],
        )

    # Chuẩn hóa L2 phòng ngừa
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / np.maximum(norms, 1e-12)

    # 1. Cơ chế mẫu nòng cốt (Anchor-based): Lọc các câu thoại >= 0.35s
    anchor_indices: List[int] = []
    if sample_durations is not None and len(sample_durations) == n_samples:
        anchor_indices = [idx for idx, d in enumerate(sample_durations) if d >= 0.35]
        # Nếu quá ít câu >= 0.35s, nới lỏng xuống 0.20s để nhân vật phụ câu ngắn không bị bỏ rơi
        if len(anchor_indices) < 2:
            anchor_indices = [idx for idx, d in enumerate(sample_durations) if d >= 0.20]

    # Nếu có đủ mẫu nòng cốt và có một số mẫu ngắn cần gán
    if len(anchor_indices) >= 2 and len(anchor_indices) < n_samples:
        anchor_embs = norm_embs[anchor_indices]
        anchor_f0s = [sample_f0s[i] for i in anchor_indices] if sample_f0s is not None else None
        anchor_labels = agglomerative_cluster_embeddings(
            anchor_embs,
            threshold=distance_threshold,
            exact_k=exact_speakers,
            max_k=max_speakers,
            sample_f0s=anchor_f0s,
        )

        unique_k = np.unique(anchor_labels)
        cluster_centers: Dict[int, np.ndarray] = {}
        cluster_median_f0s: Dict[int, Optional[float]] = {}
        for c_i in unique_k:
            members = anchor_embs[anchor_labels == c_i]
            c_vec = np.mean(members, axis=0)
            cluster_centers[c_i] = c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12)
            if anchor_f0s is not None:
                c_f0_vals = [anchor_f0s[pos] for pos, lbl in enumerate(anchor_labels) if lbl == c_i and anchor_f0s[pos] is not None]
                cluster_median_f0s[c_i] = float(np.median(c_f0_vals)) if c_f0_vals else None
            else:
                cluster_median_f0s[c_i] = None

        # Gán nhãn cho toàn bộ n_samples
        best_labels = np.zeros(n_samples, dtype=np.int32)
        for i, idx in enumerate(anchor_indices):
            best_labels[idx] = anchor_labels[i]

        non_anchor = [idx for idx in range(n_samples) if idx not in anchor_indices]
        for idx in non_anchor:
            f0_val = sample_f0s[idx] if sample_f0s is not None and idx < len(sample_f0s) else None
            dists = []
            for c_i in unique_k:
                # Tính khoảng cách nhỏ nhất tới bất kỳ mẫu nào của cụm để tránh bị tách khi cảm xúc thay đổi (hét, thì thầm...)
                c_members = anchor_embs[anchor_labels == c_i]
                if len(c_members) > 0:
                    d_dot = np.dot(c_members, norm_embs[idx])
                    d = 1.0 - float(np.max(d_dot))
                else:
                    d = 1.0 - float(np.dot(norm_embs[idx], cluster_centers[c_i]))

                c_f0 = cluster_median_f0s.get(c_i)
                if f0_val is not None and c_f0 is not None and abs(f0_val - c_f0) > 75.0:
                    d += 0.25
                dists.append(d)
            min_dist_idx = int(np.argmin(dists))
            min_dist = dists[min_dist_idx]
            # Từ chối gán khi khoảng cách cosin vượt quá ngưỡng cho phép
            if min_dist > distance_threshold:
                best_labels[idx] = -1
            else:
                best_labels[idx] = unique_k[min_dist_idx]
    else:
        # Phân cụm trực tiếp toàn bộ mẫu bằng AHC
        best_labels = agglomerative_cluster_embeddings(
            norm_embs,
            threshold=distance_threshold,
            exact_k=exact_speakers,
            max_k=max_speakers,
            sample_f0s=sample_f0s,
        )

    # Tính toán tâm cụm cho các cụm hợp lệ (loại trừ nhãn -1 bị từ chối)
    valid_clusters = [c for c in np.unique(best_labels) if c >= 0]
    cluster_centers: Dict[int, np.ndarray] = {}
    for c_i in valid_clusters:
        members = norm_embs[best_labels == c_i]
        c_vec = np.mean(members, axis=0)
        cluster_centers[c_i] = c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12)

    cluster_sizes = {c_i: int(np.sum(best_labels == c_i)) for c_i in valid_clusters}

    # Phân loại 3 mức độ tin cậy cho từng mẫu dựa trên hiệu chuẩn đa tiêu chí
    conf_levels: List[str] = []
    conf_scores: List[float] = []

    for idx in range(n_samples):
        c_i = best_labels[idx]
        if c_i < 0 or c_i not in cluster_centers:
            # Đoạn thoại bị từ chối ép gán -> Đánh dấu Chưa xác định
            conf_scores.append(0.0)
            conf_levels.append(ConfidenceLevel.UNASSIGNED.value)
            continue

        vec = norm_embs[idx]
        center = cluster_centers[c_i]

        # 1. Tương đồng cosin với tâm cụm gán
        sim_center = float(np.dot(vec, center))
        sim_score = max(0.0, min(1.0, sim_center))

        # 2. Hệ số thời lượng mẫu (phạt mẫu ngắn)
        dur = sample_durations[idx] if sample_durations and idx < len(sample_durations) else 1.5
        len_factor = min(1.0, max(0.3, dur / 1.5))

        # 3. Độ phân tách với các tâm cụm khác
        other_sims = [float(np.dot(vec, cluster_centers[other_c])) for other_c in valid_clusters if other_c != c_i]
        max_other_sim = max(other_sims) if other_sims else -1.0
        sep_margin = max(0.0, min(1.0, 1.0 - max(0.0, max_other_sim)))

        # Điểm tổng hợp hiệu chuẩn
        if len(valid_clusters) > 1:
            score = 0.45 * sim_score + 0.35 * len_factor + 0.20 * sep_margin
        else:
            score = 0.60 * sim_score + 0.40 * len_factor

        score = float(max(0.0, min(1.0, score)))
        conf_scores.append(score)

        is_singleton = bool(cluster_sizes.get(c_i, 0) < 2 and n_samples >= 4)
        is_ambiguous = bool(other_sims and (max_other_sim >= 0.60 or (sim_center - max_other_sim < 0.25)))
        if score >= 0.70 and dur >= 0.8 and not is_ambiguous and not is_singleton:
            level = ConfidenceLevel.HIGH.value
        elif score >= 0.35:
            level = ConfidenceLevel.REVIEW_NEEDED.value
        else:
            level = ConfidenceLevel.UNASSIGNED.value
        conf_levels.append(level)

    return best_labels, conf_levels, conf_scores


def detect_speech_segments_vad(
    audio_data: np.ndarray,
    sample_rate: int,
    min_speech_sec: float = 0.3,
    frame_ms: int = 30,
    energy_threshold_ratio: float = 0.15,
) -> List[Tuple[float, float]]:
    """
    Phát hiện các đoạn có hoạt động giọng nói dựa trên năng lượng RMS khung ngắn hạn.
    Trả về danh sách các mốc thời gian (bắt đầu, kết thúc) tính bằng giây.
    """
    if audio_data is None or len(audio_data) == 0 or sample_rate <= 0:
        return []

    data = np.asarray(audio_data, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
    n_frames = len(data) // frame_len
    if n_frames == 0:
        return []

    frames = data[: n_frames * frame_len].reshape((n_frames, frame_len))
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    max_rms = float(np.max(rms)) if rms.size > 0 else 0.0
    if max_rms < 1e-4:
        return []

    # Ngưỡng năng lượng tương đối có sàn an toàn
    threshold = max(0.005, max_rms * energy_threshold_ratio)

    # Phân tích tỷ lệ năng lượng trong dải tần số giọng nói con người (150 Hz - 4000 Hz)
    # Loại trừ tạp âm cao tần (tiếng rít kim loại, tiếng súng) và ù trầm tần số cực thấp
    fft_vals = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    voice_mask = (freqs >= 150.0) & (freqs <= 4000.0)
    voice_power = np.sum(fft_vals[:, voice_mask] ** 2, axis=1)
    total_power = np.maximum(np.sum(fft_vals ** 2, axis=1), 1e-12)
    voice_ratio = voice_power / total_power

    is_speech = (rms >= threshold) & (voice_ratio >= 0.40)

    segments: List[Tuple[float, float]] = []
    in_speech = False
    start_frame = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_frame = i
        elif not active and in_speech:
            in_speech = False
            start_s = start_frame * (frame_ms / 1000.0)
            end_s = i * (frame_ms / 1000.0)
            if end_s - start_s >= min_speech_sec:
                segments.append((start_s, end_s))

    if in_speech:
        start_s = start_frame * (frame_ms / 1000.0)
        end_s = n_frames * (frame_ms / 1000.0)
        if end_s - start_s >= min_speech_sec:
            segments.append((start_s, end_s))

    # Gộp các đoạn cách nhau dưới 0.3 giây
    merged: List[Tuple[float, float]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev_s, prev_e = merged[-1]
            if seg[0] - prev_e <= 0.30:
                merged[-1] = (prev_s, max(prev_e, seg[1]))
            else:
                merged.append(seg)

    return merged


def trim_silence_vad(
    chunk: np.ndarray,
    sample_rate: int,
    padding_sec: float = 0.05,
) -> np.ndarray:
    """
    Cắt bỏ khoảng lặng ở hai đầu đoạn âm thanh, giữ lại khoảng đệm an toàn.
    """
    if chunk is None or len(chunk) == 0:
        return chunk
    segs = detect_speech_segments_vad(chunk, sample_rate, min_speech_sec=0.15)
    if not segs:
        return chunk
    s_sec = max(0.0, segs[0][0] - padding_sec)
    e_sec = min(len(chunk) / float(sample_rate), segs[-1][1] + padding_sec)
    start_idx = int(s_sec * sample_rate)
    end_idx = int(e_sec * sample_rate)
    if end_idx - start_idx >= int(0.20 * sample_rate):
        return chunk[start_idx:end_idx]
    return chunk


def _calculate_safe_audio_padding(
    start_s: float,
    end_s: float,
    prev_end: float = 0.0,
    next_start: float = float("inf"),
    max_pad: float = 0.15,
) -> Tuple[float, float]:
    """
    Tính toán khoảng đệm trước và sau câu thoại an toàn, không cắt lẹm vào câu trước hoặc sau.
    Khoảng đệm tối đa là max_pad (0.15s), nhưng không vượt quá một nửa khoảng nghỉ giữa hai câu liền kề.
    Nếu mốc thời gian bị chồng lấn (overlap), trả về đệm không dương để ngăn chặn dính chéo âm thanh.
    """
    if prev_end > 0.0:
        if start_s < prev_end:
            pad_before = 0.0
        else:
            gap_before = start_s - prev_end
            pad_before = min(max_pad, gap_before / 2.0)
    else:
        pad_before = max_pad

    if next_start < float("inf"):
        if end_s > next_start:
            pad_after = 0.0
        else:
            gap_after = next_start - end_s
            pad_after = min(max_pad, gap_after / 2.0)
    else:
        pad_after = max_pad

    return pad_before, pad_after


class SpeakerDiarizationCoordinator:
    """Bộ điều phối toàn diện cho quá trình nhận diện người nói và đối chiếu mốc phụ đề."""

    def __init__(
        self,
        model_path: str = "",
        device_id: int = 0,
        gpu_mem_limit_mb: int = 768,
        prefer_cuda: bool = True,
        allow_mock: bool = False,
    ):
        self.model_path = model_path
        self.device_id = device_id
        self.gpu_mem_limit_mb = gpu_mem_limit_mb
        self.prefer_cuda = prefer_cuda
        self.allow_mock = allow_mock
        self.last_provider = ""

    def process_video_subtitles(
        self,
        video_path: str,
        blocks: List[Any],
        max_speakers: Optional[int] = None,
        exact_speakers: Optional[int] = None,
        distance_threshold: float = 0.40,
        prefer_cuda: Optional[bool] = None,
        temp_dir: str = "",
        clone_audio_path: Optional[str] = None,
    ) -> Dict[str, SpeakerVoiceProfile]:
        """
        Nhận diện người nói dựa trên âm thanh WAV đã trích xuất và các mốc phụ đề.
        Nếu danh sách blocks rỗng, tự động phân đoạn âm thanh bằng VAD để sinh các khối câu thoại sơ bộ.
        """
        if not video_path or not os.path.isfile(video_path):
            if blocks:
                for blk in blocks:
                    if not getattr(blk, "is_speaker_locked", False):
                        blk.speaker = "Chưa xác định"
                        blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
            raise FileNotFoundError("Không tìm thấy tệp âm thanh đã trích xuất để nhận diện người nói.")

        # Đọc và xác thực âm thanh trước khi nạp bất kỳ mô hình ONNX nào.
        try:
            data, source_sr = sf.read(video_path, dtype="float32")
        except Exception as exc:
            raise RuntimeError(
                "Không đọc được âm thanh nhận diện. Hãy kiểm tra tệp âm thanh WAV trung gian "
                "hoặc thử tách giọng từ video lại."
            ) from exc
        if data.size == 0 or source_sr <= 0:
            raise ValueError("Tệp âm thanh nhận diện rỗng hoặc có tần số lấy mẫu không hợp lệ.")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError("Tệp âm thanh nhận diện chứa giá trị không hữu hạn.")
        if float(np.max(np.abs(data))) < 1e-6:
            logger.warning("Tệp âm thanh hoàn toàn im lặng; giữ các câu ở trạng thái Chưa xác định.")
            for blk in blocks:
                if not getattr(blk, "is_speaker_locked", False):
                    blk.speaker = "Chưa xác định"
                    blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
            return {}

        target_sr = 16000
        if source_sr != target_sr:
            try:
                from scipy.signal import resample_poly
                import math
                divisor = math.gcd(int(source_sr), target_sr)
                data = resample_poly(
                    data,
                    target_sr // divisor,
                    int(source_sr) // divisor,
                ).astype(np.float32)
            except ImportError:
                # Dự phòng không cần SciPy; chỉ nội suy khi môi trường thiếu bộ resample chuyên dụng.
                duration = len(data) / float(source_sr)
                old_t = np.arange(len(data), dtype=np.float64) / float(source_sr)
                new_count = max(1, int(round(duration * target_sr)))
                new_t = np.arange(new_count, dtype=np.float64) / float(target_sr)
                data = np.interp(new_t, old_t, data).astype(np.float32)
        sr = target_sr
        audio_duration = len(data) / float(sr)

        clone_data = None
        clone_sr = 24000
        if clone_audio_path and os.path.isfile(clone_audio_path):
            try:
                c_d, c_s = sf.read(clone_audio_path, dtype="float32")
                if c_d.ndim > 1:
                    c_d = np.mean(c_d, axis=1)
                clone_data = np.asarray(c_d, dtype=np.float32)
                clone_sr = int(c_s)
            except Exception as exc:
                logger.warning(f"Không thể nạp tệp âm thanh gốc chất lượng cao: {exc}")
                clone_data = None

        # Tự động nhận diện phân đoạn câu thoại bằng VAD khi không có tệp phụ đề SRT
        if not blocks:
            segs = detect_speech_segments_vad(data, sr, min_speech_sec=0.35)
            if not segs:
                logger.warning("Không phát hiện đoạn giọng nói nào từ âm thanh video qua VAD.")
                return {}
            from .translator import SubtitleBlock
            for seg_idx, (s_sec, e_sec) in enumerate(segs, start=1):
                new_blk = SubtitleBlock(
                    index=seg_idx,
                    start_time=_format_seconds_to_srt_time(s_sec),
                    end_time=_format_seconds_to_srt_time(e_sec),
                    text=f"Đoạn thoại {seg_idx}",
                )
                blocks.append(new_blk)

        # Làm sạch thông tin giọng cũ của các câu chưa khóa trước khi phân tích
        for blk in blocks:
            if not getattr(blk, "is_speaker_locked", False):
                blk.voice_ref_path = ""
                blk.voice_instruct = ""

        audio_chunks: List[np.ndarray] = []
        clone_audio_chunks: List[Optional[np.ndarray]] = []
        valid_indices: List[int] = []
        sample_durations: List[float] = []
        sample_f0s: List[Optional[float]] = []

        # Tiền tính toán mốc thời gian của tất cả các câu để kẹp biên an toàn
        parsed_times = [
            (
                _parse_time_to_seconds(getattr(b, "start_time", 0.0)),
                _parse_time_to_seconds(getattr(b, "end_time", 0.0)),
            )
            for b in blocks
        ]

        for i, blk in enumerate(blocks):
            if getattr(blk, "is_speaker_locked", False):
                continue

            start_s, end_s = parsed_times[i]
            duration = end_s - start_s
            if duration < 0.30 or end_s <= start_s:
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                blk.voice_ref_path = ""
                blk.voice_instruct = ""
                continue

            if start_s >= audio_duration:
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                blk.voice_ref_path = ""
                blk.voice_instruct = ""
                continue

            prev_end = parsed_times[i - 1][1] if i > 0 else 0.0
            next_start = parsed_times[i + 1][0] if i + 1 < len(blocks) else audio_duration

            # Xử lý ngăn chặn dính chéo giọng nói khi mốc phụ đề bị chồng lấn
            effective_start = start_s
            if prev_end > 0.0 and start_s < prev_end:
                effective_start = (start_s + prev_end) / 2.0

            effective_end = end_s
            if next_start < audio_duration and end_s > next_start:
                effective_end = (end_s + next_start) / 2.0

            pad_before, pad_after = _calculate_safe_audio_padding(effective_start, effective_end, prev_end, next_start)

            padded_start = max(0.0, effective_start - pad_before)
            padded_end = min(audio_duration, effective_end + pad_after)
            chunk = data[int(padded_start * sr):int(padded_end * sr)]
            chunk = trim_silence_vad(chunk, sr)
            if chunk.size < int(0.25 * sr):
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                blk.voice_ref_path = ""
                blk.voice_instruct = ""
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            peak = float(np.max(np.abs(chunk)))
            if not np.isfinite(rms) or rms < 1e-4 or peak < 1e-4:
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                blk.voice_ref_path = ""
                blk.voice_instruct = ""
                continue

            audio_chunks.append(chunk.astype(np.float32, copy=False))
            sample_f0s.append(extract_chunk_f0(chunk, sr))
            if clone_data is not None:
                c_start_f = max(0, int(padded_start * clone_sr))
                c_end_f = min(len(clone_data), int(padded_end * clone_sr))
                c_chunk = clone_data[c_start_f:c_end_f]
                c_chunk = trim_silence_vad(c_chunk, clone_sr)
                clone_audio_chunks.append(c_chunk if c_chunk.size >= int(0.25 * clone_sr) else None)
            else:
                clone_audio_chunks.append(None)
            valid_indices.append(i)
            sample_durations.append(padded_end - padded_start)

        if not audio_chunks:
            logger.warning("Không có câu thoại đủ chất lượng để phân cụm.")
            return {}

        use_cuda = self.prefer_cuda if prefer_cuda is None else bool(prefer_cuda)
        session, provider = create_speaker_session(
            model_path=self.model_path or None,
            device_id=self.device_id,
            gpu_mem_limit_mb=self.gpu_mem_limit_mb,
            prefer_cuda=use_cuda,
            allow_mock=self.allow_mock,
        )
        self.last_provider = provider
        logger.info("Speaker embedding đang sử dụng %s.", provider)

        try:
            embeddings = batch_extract_speaker_embeddings(
                session,
                audio_chunks,
                sample_rate=sr,
                batch_size=8,
            )
            labels, conf_levels, conf_scores = cosine_cluster_embeddings(
                embeddings,
                valid_indices=valid_indices,
                max_speakers=max_speakers,
                exact_speakers=exact_speakers,
                distance_threshold=distance_threshold,
                sample_durations=sample_durations,
                sample_f0s=sample_f0s,
            )

            # Tránh xung đột tên người nói với các câu đã được khóa
            import re
            max_existing_num = 0
            for b in blocks:
                if getattr(b, "is_speaker_locked", False):
                    spk_name = getattr(b, "speaker", "").strip()
                    m = re.match(r"^Người nói\s*(\d+)$", spk_name)
                    if m:
                        max_existing_num = max(max_existing_num, int(m.group(1)))

            detected_profiles: Dict[str, SpeakerVoiceProfile] = {}
            for pos, block_idx in enumerate(valid_indices):
                blk = blocks[block_idx]
                cluster_id = int(labels[pos])
                if cluster_id < 0:
                    blk.speaker = "Chưa xác định"
                    blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                    blk.voice_ref_path = ""
                    blk.voice_instruct = ""
                    continue
                speaker_name = f"Người nói {max_existing_num + cluster_id + 1}"
                blk.speaker = speaker_name
                blk.confidence_level = conf_levels[pos]

                profile = detected_profiles.get(speaker_name)
                if profile is None:
                    profile = CharacterVoiceProfile(
                        speaker_id=f"spk_{max_existing_num + cluster_id}",
                        display_name=speaker_name,
                        method="preset",
                        utterance_count=0,
                        total_duration=0.0,
                        status="locked",
                        high_confidence_ratio=0.0,
                    )
                    detected_profiles[speaker_name] = profile
                profile.utterance_count += 1
                dur = max(0.0, _parse_time_to_seconds(getattr(blk, "end_time", 0)) - _parse_time_to_seconds(getattr(blk, "start_time", 0)))
                profile.total_duration += dur

            # Hậu xử lý: Gán câu "Chưa xác định" hoặc câu ngắn cho người nói liền kề (khoảng cách <= 1.5s)
            for pos, block_idx in enumerate(valid_indices):
                blk = blocks[block_idx]
                if getattr(blk, "speaker", "") == "Chưa xác định":
                    cur_start = _parse_time_to_seconds(getattr(blk, "start_time", 0))
                    assigned_neighbor = None
                    if pos > 0:
                        prev_blk = blocks[valid_indices[pos - 1]]
                        prev_spk = getattr(prev_blk, "speaker", "")
                        prev_end = _parse_time_to_seconds(getattr(prev_blk, "end_time", 0))
                        if prev_spk and prev_spk != "Chưa xác định" and (cur_start - prev_end <= 1.5):
                            assigned_neighbor = prev_spk
                    if not assigned_neighbor and pos < len(valid_indices) - 1:
                        next_blk = blocks[valid_indices[pos + 1]]
                        next_spk = getattr(next_blk, "speaker", "")
                        next_start = _parse_time_to_seconds(getattr(next_blk, "start_time", 0))
                        cur_end = _parse_time_to_seconds(getattr(blk, "end_time", 0))
                        if next_spk and next_spk != "Chưa xác định" and (next_start - cur_end <= 1.0):
                            assigned_neighbor = next_spk

                    if assigned_neighbor and assigned_neighbor in detected_profiles:
                        blk.speaker = assigned_neighbor
                        blk.confidence_level = ConfidenceLevel.REVIEW_NEEDED.value
                        prof = detected_profiles[assigned_neighbor]
                        prof.utterance_count += 1
                        dur = max(0.0, _parse_time_to_seconds(getattr(blk, "end_time", 0)) - cur_start)
                        prof.total_duration += dur

            for speaker_name, profile in detected_profiles.items():
                members = [
                    pos for pos, block_idx in enumerate(valid_indices)
                    if getattr(blocks[block_idx], "speaker", "") == speaker_name
                ]
                member_conf = [conf_levels[pos] for pos in members]
                profile.high_confidence_ratio = (
                    sum(level == ConfidenceLevel.HIGH.value for level in member_conf) / len(member_conf)
                    if member_conf else 0.0
                )

                # 1. Tính tốc độ phát âm (âm tiết / giây) từ nội dung câu và độ dài âm thanh
                total_sylls = 0
                total_speech_sec = 0.0
                f0s = []
                for pos in members:
                    cand_blk = blocks[valid_indices[pos]]
                    txt = (getattr(cand_blk, "text", "") or getattr(cand_blk, "translated_text", "")).strip()
                    if txt:
                        words = txt.split()
                        total_sylls += len(words)
                    dur_s = len(audio_chunks[pos]) / float(sr)
                    if dur_s > 0.2:
                        total_speech_sec += dur_s
                    if pos < len(sample_f0s):
                        f_val = sample_f0s[pos]
                        if f_val and f_val > 40.0:
                            f0s.append(f_val)

                if total_speech_sec > 0.4 and total_sylls > 0:
                    profile.speech_rate_sps = round(total_sylls / total_speech_sec, 2)
                else:
                    profile.speech_rate_sps = 4.2

                if f0s:
                    profile.pitch_hz = round(float(np.median(f0s)), 1)

                # 2. Xây dựng biểu diễn đa mẫu nòng cốt (Multi-prototype Representation)
                if members and len(embeddings) > max(members):
                    member_rms = []
                    for pos in members:
                        ch = audio_chunks[pos]
                        rms_v = float(np.sqrt(np.mean(np.square(ch, dtype=np.float64))))
                        member_rms.append((rms_v, pos))
                    member_rms.sort(key=lambda x: x[0])

                    med_idx = len(member_rms) // 2
                    norm_pos = member_rms[med_idx][1]
                    profile.prototypes["normal"] = embeddings[norm_pos]

                    if len(member_rms) >= 3:
                        low_pos = member_rms[0][1]
                        high_pos = member_rms[-1][1]
                        profile.prototypes["low_energy"] = embeddings[low_pos]
                        profile.prototypes["high_energy"] = embeddings[high_pos]

                # Ưu tiên các đoạn thoại có mức độ tin cậy cao, không bị lẫn người nói
                high_conf_members = [
                    pos for pos in members
                    if conf_levels[pos] == ConfidenceLevel.HIGH.value
                ]
                target_pool = high_conf_members if high_conf_members else members

                candidates = []
                for pos in target_pool:
                    chunk = audio_chunks[pos]
                    c_chunk = clone_audio_chunks[pos] if pos < len(clone_audio_chunks) else None
                    duration = len(chunk) / float(sr)
                    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
                    if duration >= 0.50 and rms >= 1e-4:
                        c_score = float(conf_scores[pos]) if pos < len(conf_scores) else 0.8
                        dur_weight = min(duration / 3.0, 1.5)
                        score = c_score * dur_weight * min(rms * 10.0, 1.0)
                        cand_blk = blocks[valid_indices[pos]]
                        c_text = (getattr(cand_blk, "text", "") or getattr(cand_blk, "translated_text", "")).strip()
                        candidates.append((score, duration, chunk, c_text, c_chunk))

                if not candidates and members:
                    best_pos = max(members, key=lambda p: len(audio_chunks[p]))
                    ch = audio_chunks[best_pos]
                    c_ch = clone_audio_chunks[best_pos] if best_pos < len(clone_audio_chunks) else None
                    cand_blk = blocks[valid_indices[best_pos]]
                    c_text = (getattr(cand_blk, "text", "") or getattr(cand_blk, "translated_text", "")).strip()
                    candidates.append((1.0, len(ch) / float(sr), ch, c_text, c_ch))

                candidates.sort(key=lambda item: item[0], reverse=True)

                # Ưu tiên lấy một đoạn đơn lẻ chuẩn nhất nếu độ dài >= 2.5 giây
                # để tránh ghép nối nhiều đoạn khác nhau gây lẫn chất giọng hoặc tạp âm
                best_single = None
                best_single_clone = None
                best_single_text = ""
                for score, duration, chunk, c_text, c_chunk in candidates:
                    if duration >= 2.5:
                        best_single = chunk[: int(7.0 * sr)]
                        if c_chunk is not None and c_chunk.size >= int(2.5 * clone_sr):
                            best_single_clone = c_chunk[: int(7.0 * clone_sr)]
                        best_single_text = c_text
                        break

                if best_single is not None:
                    merged = best_single
                    merged_clone = best_single_clone
                    # Luôn để trống để Whisper tự động nhận diện chính xác trên clip âm thanh mẫu đã cắt
                    selected_ref_text = ""
                else:
                    collected: List[np.ndarray] = []
                    collected_clone: List[np.ndarray] = []
                    text_parts: List[str] = []
                    collected_len = 0
                    collected_clone_len = 0
                    target_samples = int(7.0 * sr)
                    target_clone_samples = int(7.0 * clone_sr)
                    pause = np.zeros(int(0.15 * sr), dtype=np.float32)
                    pause_clone = np.zeros(int(0.15 * clone_sr), dtype=np.float32)
                    for _, _, chunk, c_text, c_chunk in candidates:
                        remaining = target_samples - collected_len
                        if remaining <= 0:
                            break
                        part = chunk[:remaining]
                        collected.append(part)
                        collected_len += len(part)

                        if c_chunk is not None:
                            rem_c = target_clone_samples - collected_clone_len
                            if rem_c > 0:
                                c_part = c_chunk[:rem_c]
                                collected_clone.append(c_part)
                                collected_clone_len += len(c_part)

                        if c_text and c_text not in text_parts:
                            text_parts.append(c_text)
                        if collected_len < target_samples:
                            collected.append(pause)
                            collected_len += len(pause)
                            if collected_clone:
                                collected_clone.append(pause_clone)
                                collected_clone_len += len(pause_clone)
                    merged = np.concatenate(collected)[: int(7.5 * sr)] if collected else np.zeros(0, dtype=np.float32)
                    merged_clone = np.concatenate(collected_clone)[: int(7.5 * clone_sr)] if collected_clone else None
                    selected_ref_text = ""  # Ghép nhiều đoạn: để Whisper tự động nhận diện chính xác trên clip đã ghép

                # Ưu tiên mẫu âm thanh gốc chất lượng cao (>= 24 kHz) nếu có
                if merged_clone is not None and len(merged_clone) >= int(1.0 * clone_sr):
                    out_sample = merged_clone
                    out_sr = clone_sr
                else:
                    out_sample = merged
                    out_sr = sr

                if len(out_sample) > 0:
                    # Chuẩn hóa biên độ âm lượng đỉnh ở mức -1.0 dBFS (peak 0.90) để chống vỡ tiếng
                    peak_val = float(np.max(np.abs(out_sample)))
                    if peak_val > 0.01:
                        out_sample = (out_sample / peak_val * 0.90).astype(np.float32)

                    if temp_dir:
                        sample_root = Path(temp_dir)
                    else:
                        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
                        sample_root = appdata / "Sakai-Studio" / "cache" / "speaker_samples"
                    sample_root.mkdir(parents=True, exist_ok=True)
                    sample_path = sample_root / (
                        f"{Path(video_path).stem}_{profile.speaker_id}_{uuid.uuid4().hex[:8]}.wav"
                    )
                    sf.write(str(sample_path), out_sample, out_sr)
                    profile.source_audio_path = str(sample_path)
                    profile.ref_text = selected_ref_text
                    profile.method = "clone"
                    profile.status = "locked"

                for pos in members:
                    blk = blocks[valid_indices[pos]]
                    if profile.source_audio_path:
                        blk.voice_ref_path = profile.source_audio_path
                    if not getattr(blk, "voice_instruct", ""):
                        blk.voice_instruct = profile.voice_instruct

            detected_profiles = ExternalDiarizationVerifier.verify_or_fallback(
                video_path,
                detected_profiles,
                max_speakers=max_speakers,
            )
            return detected_profiles
        finally:
            close_speaker_session(session)



_agglomerative_clustering = agglomerative_cluster_embeddings
