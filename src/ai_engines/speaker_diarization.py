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
    method: str = "clone"  # Phương thức clone hoặc preset
    preset_name: str = ""
    voice_instruct: str = "male, young adult, moderate pitch"
    status: str = "locked"  # locked, review_needed, unassigned
    custom_audio_path: str = ""
    high_confidence_ratio: float = 1.0
    voice_clone_prompt_path: str = ""
    vietnamese_accent: str = "bac"
    gender: str = "auto"
    pitch_level: str = "moderate"
    style_tag: str = "natural"
    recommended_voices: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""

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
            "gender": self.gender,
            "pitch_level": self.pitch_level,
            "style_tag": self.style_tag,
            "recommended_voices": list(self.recommended_voices) if self.recommended_voices else [],
            "error_message": self.error_message,
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
            gender=str(data.get("gender") or "auto"),
            pitch_level=str(data.get("pitch_level") or "moderate"),
            style_tag=str(data.get("style_tag") or "natural"),
            recommended_voices=list(data.get("recommended_voices") or []),
            error_message=str(data.get("error_message") or ""),
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


def agglomerative_cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = 0.40,
    exact_k: Optional[int] = None,
    min_k: int = 1,
    max_k: int = 6,
) -> np.ndarray:
    """
    Phân cụm thứ bậc tích lũy (Agglomerative Hierarchical Clustering) theo khoảng cách cosin
    và liên kết trung bình (Average Linkage).
    
    Hỗ trợ chế độ chỉ định chính xác số cụm (exact_k) hoặc tự động dừng theo ngưỡng khoảng cách (threshold).
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

    # Khởi tạo mỗi mẫu là một cụm riêng biệt
    clusters: List[List[int]] = [[i] for i in range(n_samples)]

    target_k = None
    if exact_k is not None:
        target_k = max(1, min(int(exact_k), n_samples))

    effective_max_k = max(1, min(int(max_k), n_samples))

    while len(clusters) > 1:
        if target_k is not None and len(clusters) <= target_k:
            break

        n_c = len(clusters)
        min_dist = float("inf")
        merge_i, merge_j = -1, -1

        # Tìm 2 cụm có khoảng cách liên kết trung bình nhỏ nhất
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
            # Chế độ tự động theo ngưỡng: nếu khoảng cách nhỏ nhất đã vượt threshold và số cụm đã <= max_k
            if min_dist > threshold and len(clusters) <= effective_max_k:
                break
            if len(clusters) <= min_k:
                break

        # Gộp cụm merge_j vào cụm merge_i và xóa cụm merge_j
        clusters[merge_i].extend(clusters[merge_j])
        clusters.pop(merge_j)

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
    max_speakers: int = 6,
    exact_speakers: Optional[int] = None,
    distance_threshold: float = 0.40,
    sample_durations: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[str], List[float]]:
    """
    Phân cụm các vector đặc trưng theo khoảng cách cosin và xếp hạng 3 mức độ tin cậy.
    Sử dụng thuật toán Phân cụm Thứ bậc (AHC) kết hợp cơ chế mẫu nòng cốt (Anchor Segments).
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

    # 1. Cơ chế mẫu nòng cốt (Anchor-based): Lọc các câu thoại >= 0.8s
    anchor_indices: List[int] = []
    if sample_durations is not None and len(sample_durations) == n_samples:
        anchor_indices = [idx for idx, d in enumerate(sample_durations) if d >= 0.80]
        # Nếu quá ít câu >= 0.8s, nới lỏng xuống 0.5s
        if len(anchor_indices) < 2:
            anchor_indices = [idx for idx, d in enumerate(sample_durations) if d >= 0.50]

    # Nếu có đủ mẫu nòng cốt và có một số mẫu ngắn cần gán
    if len(anchor_indices) >= 2 and len(anchor_indices) < n_samples:
        anchor_embs = norm_embs[anchor_indices]
        anchor_labels = agglomerative_cluster_embeddings(
            anchor_embs,
            threshold=distance_threshold,
            exact_k=exact_speakers,
            max_k=max_speakers,
        )

        unique_k = np.unique(anchor_labels)
        cluster_centers: Dict[int, np.ndarray] = {}
        for c_i in unique_k:
            members = anchor_embs[anchor_labels == c_i]
            c_vec = np.mean(members, axis=0)
            cluster_centers[c_i] = c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12)

        # Gán nhãn cho toàn bộ n_samples
        best_labels = np.zeros(n_samples, dtype=np.int32)
        for i, idx in enumerate(anchor_indices):
            best_labels[idx] = anchor_labels[i]

        non_anchor = [idx for idx in range(n_samples) if idx not in anchor_indices]
        for idx in non_anchor:
            # Gán vào tâm cụm gần nhất
            dists = [1.0 - float(np.dot(norm_embs[idx], cluster_centers[c_i])) for c_i in unique_k]
            best_labels[idx] = unique_k[int(np.argmin(dists))]
    else:
        # Phân cụm trực tiếp toàn bộ mẫu bằng AHC
        best_labels = agglomerative_cluster_embeddings(
            norm_embs,
            threshold=distance_threshold,
            exact_k=exact_speakers,
            max_k=max_speakers,
        )

    # Tính toán tâm cụm cuối cùng
    unique_k = np.unique(best_labels)
    cluster_centers: Dict[int, np.ndarray] = {}
    for c_i in unique_k:
        members = norm_embs[best_labels == c_i]
        c_vec = np.mean(members, axis=0)
        cluster_centers[c_i] = c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12)

    cluster_sizes = {c_i: int(np.sum(best_labels == c_i)) for c_i in unique_k}

    # Phân loại 3 mức độ tin cậy cho từng mẫu
    conf_levels: List[str] = []
    conf_scores: List[float] = []

    for idx in range(n_samples):
        own_c = best_labels[idx]
        own_center = cluster_centers[own_c]
        d_own = float(1.0 - np.dot(norm_embs[idx], own_center))

        d_second = float("inf")
        for other_c, other_center in cluster_centers.items():
            if other_c == own_c:
                continue
            d_cand = float(1.0 - np.dot(norm_embs[idx], other_center))
            if d_cand < d_second:
                d_second = d_cand

        gap = (d_second - d_own) if d_second != float("inf") else (1.0 - d_own)
        score = float(np.clip(1.0 - d_own, 0.0, 1.0))
        conf_scores.append(score)

        is_short = bool(sample_durations and idx < len(sample_durations) and sample_durations[idx] < 0.60)
        if (cluster_sizes.get(own_c, 0) < 2 and n_samples >= 4) or is_short:
            conf_levels.append(ConfidenceLevel.REVIEW_NEEDED.value)
        elif gap >= 0.18 and d_own <= 0.22:
            conf_levels.append(ConfidenceLevel.HIGH.value)
        elif gap < 0.10 or d_own > 0.28:
            conf_levels.append(ConfidenceLevel.REVIEW_NEEDED.value)
        else:
            conf_levels.append(ConfidenceLevel.HIGH.value)

    return best_labels, conf_levels, conf_scores


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
        max_speakers: int = 4,
        prefer_cuda: Optional[bool] = None,
        temp_dir: str = "",
    ) -> Dict[str, SpeakerVoiceProfile]:
        """
        Nhận diện người nói dựa trên âm thanh WAV đã trích xuất và các mốc phụ đề.
        Không sử dụng dữ liệu ngẫu nhiên khi thiếu/không đọc được âm thanh.
        """
        if not blocks:
            return {}
        if not video_path or not os.path.isfile(video_path):
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

        # Làm sạch thông tin giọng cũ của các câu chưa khóa trước khi phân tích
        for blk in blocks:
            if not getattr(blk, "is_speaker_locked", False):
                blk.voice_ref_path = ""
                blk.voice_instruct = ""

        audio_chunks: List[np.ndarray] = []
        valid_indices: List[int] = []
        for i, blk in enumerate(blocks):
            if getattr(blk, "is_speaker_locked", False):
                continue

            start_s = _parse_time_to_seconds(getattr(blk, "start_time", 0.0))
            end_s = _parse_time_to_seconds(getattr(blk, "end_time", start_s))
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

            padded_start = max(0.0, start_s - 0.15)
            padded_end = min(audio_duration, end_s + 0.15)
            chunk = data[int(padded_start * sr):int(padded_end * sr)]
            if chunk.size < int(0.30 * sr):
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
            valid_indices.append(i)

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
                speaker_name = f"Người nói {max_existing_num + cluster_id + 1}"
                blk.speaker = speaker_name
                blk.confidence_level = conf_levels[pos]

                profile = detected_profiles.get(speaker_name)
                if profile is None:
                    profile = CharacterVoiceProfile(
                        speaker_id=f"spk_{max_existing_num + cluster_id}",
                        display_name=speaker_name,
                        method="clone",
                        utterance_count=0,
                        total_duration=0.0,
                        status="locked",
                        high_confidence_ratio=0.0,
                    )
                    detected_profiles[speaker_name] = profile
                profile.utterance_count += 1
                dur = max(0.0, _parse_time_to_seconds(getattr(blk, "end_time", 0)) - _parse_time_to_seconds(getattr(blk, "start_time", 0)))
                profile.total_duration += dur

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

                # Ưu tiên các đoạn thoại có mức độ tin cậy cao, không bị lẫn người nói
                high_conf_members = [
                    pos for pos in members
                    if conf_levels[pos] == ConfidenceLevel.HIGH.value
                ]
                target_pool = high_conf_members if high_conf_members else members

                candidates = []
                for pos in target_pool:
                    chunk = audio_chunks[pos]
                    duration = len(chunk) / float(sr)
                    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
                    if duration >= 0.50 and rms >= 1e-4:
                        c_score = float(conf_scores[pos]) if pos < len(conf_scores) else 0.8
                        dur_weight = min(duration / 3.0, 1.5)
                        score = c_score * dur_weight * min(rms * 10.0, 1.0)
                        candidates.append((score, duration, chunk))

                if not candidates and members:
                    best_pos = max(members, key=lambda p: len(audio_chunks[p]))
                    ch = audio_chunks[best_pos]
                    candidates.append((1.0, len(ch) / float(sr), ch))

                candidates.sort(key=lambda item: item[0], reverse=True)

                # Ưu tiên lấy một đoạn đơn lẻ chuẩn nhất nếu độ dài >= 2.5 giây
                # để tránh ghép nối nhiều đoạn khác nhau gây lẫn chất giọng hoặc tạp âm
                best_single = None
                for score, duration, chunk in candidates:
                    if duration >= 2.5:
                        best_single = chunk[: int(7.0 * sr)]
                        break

                if best_single is not None:
                    merged = best_single
                else:
                    collected: List[np.ndarray] = []
                    collected_len = 0
                    target_samples = int(7.0 * sr)
                    pause = np.zeros(int(0.15 * sr), dtype=np.float32)
                    for _, _, chunk in candidates:
                        remaining = target_samples - collected_len
                        if remaining <= 0:
                            break
                        part = chunk[:remaining]
                        collected.append(part)
                        collected_len += len(part)
                        if collected_len < target_samples:
                            collected.append(pause)
                            collected_len += len(pause)
                    merged = np.concatenate(collected)[: int(7.5 * sr)] if collected else np.zeros(0, dtype=np.float32)

                if len(merged) > 0:
                    if temp_dir:
                        sample_root = Path(temp_dir)
                    else:
                        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
                        sample_root = appdata / "Sakai-Studio" / "cache" / "speaker_samples"
                    sample_root.mkdir(parents=True, exist_ok=True)
                    sample_path = sample_root / (
                        f"{Path(video_path).stem}_{profile.speaker_id}_{uuid.uuid4().hex[:8]}.wav"
                    )
                    sf.write(str(sample_path), merged, sr)
                    profile.source_audio_path = str(sample_path)
                    profile.method = "clone"
                    profile.status = "locked"

                for pos in members:
                    blk = blocks[valid_indices[pos]]
                    if profile.source_audio_path:
                        blk.voice_ref_path = profile.source_audio_path
                    if not getattr(blk, "voice_instruct", ""):
                        blk.voice_instruct = profile.voice_instruct

            return detected_profiles
        finally:
            close_speaker_session(session)

