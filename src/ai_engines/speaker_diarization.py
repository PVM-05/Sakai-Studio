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
class SpeakerVoiceProfile:
    """Hồ sơ cấu hình giọng nói và danh tính của một nhân vật."""

    speaker_id: str
    display_name: str
    source_audio_sample: str = ""
    assigned_strategy: str = "preset"  # "clone" (nhái giọng nguồn) hoặc "preset" (giọng thiết kế)
    voice_preset_key: str = ""
    voice_clone_prompt_path: str = ""
    instruct: str = "male, young adult, moderate pitch"
    dialogue_count: int = 0
    high_confidence_ratio: float = 1.0


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


def cosine_cluster_embeddings(
    embeddings: np.ndarray,
    valid_indices: List[int],
    max_speakers: int = 6,
) -> Tuple[np.ndarray, List[str], List[float]]:
    """
    Phân cụm các vector đặc trưng theo khoảng cách cosin và xếp hạng 3 mức độ tin cậy.
    Trả về bộ ba (labels, confidence_levels, confidence_scores).
    """
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int32), [], []

    if n_samples == 1:
        # Một câu không đủ dữ liệu để đánh giá mức tách biệt với người nói khác.
        return (
            np.zeros(1, dtype=np.int32),
            [ConfidenceLevel.REVIEW_NEEDED.value],
            [0.0],
        )

    # Chuẩn hóa L2 phòng ngừa
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / np.maximum(norms, 1e-12)

    # Tính ma trận độ tương đồng cosin và khoảng cách cosin
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    dist_matrix = np.maximum(0.0, 1.0 - sim_matrix)

    # Tránh để nghiệm phân cụm có quá nhiều cụm đơn lẻ thắng do silhouette bằng 1.
    if n_samples >= 4:
        max_k = max(2, min(max_speakers, n_samples // 2, 6))
    else:
        max_k = max(2, min(max_speakers, n_samples, 6))

    best_labels = np.zeros(n_samples, dtype=np.int32)
    best_score = -1.0
    rng = np.random.default_rng(42)

    for k in range(2, max_k + 1):
        # Khởi tạo tâm cụm k-means++ theo khoảng cách cosin
        centers = np.zeros((k, norm_embs.shape[1]), dtype=np.float32)
        centers[0] = norm_embs[rng.integers(0, n_samples)]
        for c_i in range(1, k):
            dists = np.min([np.sum((norm_embs - centers[j]) ** 2, axis=1) for j in range(c_i)], axis=0)
            total_dist = float(np.sum(dists))
            if not np.isfinite(total_dist) or total_dist <= 1e-12:
                # Các embedding trùng nhau: tránh xác suất toàn số 0 gây lỗi khởi tạo cụm.
                chosen_idx = int(rng.integers(0, n_samples))
            else:
                probs = _normalized_choice_probabilities(dists)
                if probs is None:
                    # Dự phòng an toàn nếu trọng số bị suy biến hoặc không hữu hạn.
                    chosen_idx = int(rng.integers(0, n_samples))
                else:
                    chosen_idx = int(rng.choice(n_samples, p=probs))
            centers[c_i] = norm_embs[chosen_idx]

        labels = np.zeros(n_samples, dtype=np.int32)
        for _ in range(25):
            # Tính khoảng cách tới các tâm cụm
            sims = np.dot(norm_embs, centers.T)
            new_labels = np.argmax(sims, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c_i in range(k):
                members = norm_embs[labels == c_i]
                if len(members) > 0:
                    c_vec = np.mean(members, axis=0)
                    centers[c_i] = c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12)
                else:
                    centers[c_i] = norm_embs[rng.integers(0, n_samples)]

        unique_clusters = np.unique(labels)
        if len(unique_clusters) < 2:
            continue

        # Đánh giá hệ số phân tách Silhouette theo khoảng cách cosin
        sil_scores = []
        for idx in range(n_samples):
            own_c = labels[idx]
            same_mask = (labels == own_c)
            same_mask[idx] = False
            a_val = float(np.mean(dist_matrix[idx, same_mask])) if np.any(same_mask) else 0.0

            b_val = float("inf")
            for other_c in unique_clusters:
                if other_c == own_c:
                    continue
                other_mask = (labels == other_c)
                if np.any(other_mask):
                    dist_other = float(np.mean(dist_matrix[idx, other_mask]))
                    if dist_other < b_val:
                        b_val = dist_other

            denom = max(a_val, b_val, 1e-6)
            sil_scores.append((b_val - a_val) / denom)

        mean_sil = float(np.mean(sil_scores)) if sil_scores else -1.0
        if mean_sil > best_score:
            best_score = mean_sil
            best_labels = labels.copy()

    # Tính toán tâm cụm tối ưu cuối cùng
    unique_k = np.unique(best_labels)
    cluster_centers: Dict[int, np.ndarray] = {}
    for c_i in unique_k:
        members = norm_embs[best_labels == c_i]
        c_vec = np.mean(members, axis=0)
        cluster_centers[c_i] = c_vec / np.maximum(np.linalg.norm(c_vec), 1e-12)

    # Tính toán kích thước của từng cụm
    cluster_sizes = {c_i: int(np.sum(best_labels == c_i)) for c_i in unique_k}

    # Phân loại 3 mức độ tin cậy cho từng mẫu
    conf_levels: List[str] = []
    conf_scores: List[float] = []

    for idx in range(n_samples):
        own_c = best_labels[idx]
        own_center = cluster_centers[own_c]
        d_own = float(1.0 - np.dot(norm_embs[idx], own_center))

        # Tính khoảng cách tới tâm cụm thứ hai gần nhất
        d_second = float("inf")
        for other_c, other_center in cluster_centers.items():
            if other_c == own_c:
                continue
            d_cand = float(1.0 - np.dot(norm_embs[idx], other_center))
            if d_cand < d_second:
                d_second = d_cand

        gap = d_second - d_own
        score = float(np.clip(1.0 - d_own, 0.0, 1.0))
        conf_scores.append(score)

        # Cụm đơn lẻ (chỉ có 1-2 câu trong bộ dữ liệu lớn) cần được rà soát
        if cluster_sizes.get(own_c, 0) < 2 and n_samples >= 4:
            conf_levels.append(ConfidenceLevel.REVIEW_NEEDED.value)
        elif gap >= 0.15 and d_own <= 0.35:
            conf_levels.append(ConfidenceLevel.HIGH.value)
        elif gap < 0.08 or d_own > 0.50:
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
                continue

            if start_s >= audio_duration:
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                continue

            padded_start = max(0.0, start_s - 0.15)
            padded_end = min(audio_duration, end_s + 0.15)
            chunk = data[int(padded_start * sr):int(padded_end * sr)]
            if chunk.size < int(0.30 * sr):
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            peak = float(np.max(np.abs(chunk)))
            if not np.isfinite(rms) or rms < 1e-4 or peak < 1e-4:
                blk.speaker = "Chưa xác định"
                blk.confidence_level = ConfidenceLevel.UNASSIGNED.value
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

            detected_profiles: Dict[str, SpeakerVoiceProfile] = {}
            for pos, block_idx in enumerate(valid_indices):
                blk = blocks[block_idx]
                cluster_id = int(labels[pos])
                speaker_name = f"Người nói {cluster_id + 1}"
                blk.speaker = speaker_name
                blk.confidence_level = conf_levels[pos]

                profile = detected_profiles.get(speaker_name)
                if profile is None:
                    profile = SpeakerVoiceProfile(
                        speaker_id=f"spk_{cluster_id}",
                        display_name=speaker_name,
                        assigned_strategy="clone",
                        dialogue_count=0,
                        high_confidence_ratio=0.0,
                    )
                    detected_profiles[speaker_name] = profile
                profile.dialogue_count += 1

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

                # Chọn các đoạn mẫu rõ, đủ dài, cùng thuộc một cụm; giới hạn tổng khoảng 7 giây.
                candidates = []
                for pos in members:
                    chunk = audio_chunks[pos]
                    duration = len(chunk) / float(sr)
                    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
                    if duration >= 0.60 and rms >= 1e-4:
                        candidates.append((rms * min(duration, 4.0), chunk))
                candidates.sort(key=lambda item: item[0], reverse=True)

                collected: List[np.ndarray] = []
                collected_len = 0
                target_samples = int(7.0 * sr)
                pause = np.zeros(int(0.15 * sr), dtype=np.float32)
                for _, chunk in candidates:
                    remaining = target_samples - collected_len
                    if remaining <= 0:
                        break
                    part = chunk[:remaining]
                    collected.append(part)
                    collected_len += len(part)
                    if collected_len < target_samples:
                        collected.append(pause)
                        collected_len += len(pause)

                if collected:
                    merged = np.concatenate(collected)[: int(7.5 * sr)]
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
                    profile.source_audio_sample = str(sample_path)
                    profile.assigned_strategy = "clone"

                for pos in members:
                    blk = blocks[valid_indices[pos]]
                    if profile.source_audio_sample:
                        blk.voice_ref_path = profile.source_audio_sample
                    if not getattr(blk, "voice_instruct", ""):
                        blk.voice_instruct = profile.instruct

            return detected_profiles
        finally:
            close_speaker_session(session)
