# -*- coding: utf-8 -*-
"""
Mô-đun kiểm định chất lượng và chuẩn hóa âm thanh mẫu tham chiếu cho OmniVoice.
Cung cấp thuật toán chuẩn hóa năng lượng âm thanh chuẩn LUFS, khử nhiễu quang phổ,
cắt tỉa khoảng lặng và chấm điểm chất lượng giọng mẫu theo thang điểm 100.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferenceAudioReport:
    """Báo cáo đánh giá chất lượng tệp âm thanh tham chiếu đối chiếu."""
    score: int
    grade: str
    effective_duration_sec: float
    estimated_snr_db: float
    clipping_ratio: float
    silence_ratio: float
    is_acceptable: bool
    rejection_reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "effective_duration_sec": self.effective_duration_sec,
            "estimated_snr_db": self.estimated_snr_db,
            "clipping_ratio": self.clipping_ratio,
            "silence_ratio": self.silence_ratio,
            "is_acceptable": self.is_acceptable,
            "rejection_reason": self.rejection_reason,
            "details": self.details,
        }


def calculate_integrated_lufs(audio_data: np.ndarray, sr: int = 24000) -> float:
    """
    Đo độ to âm thanh tích hợp theo tiêu chuẩn ITU-R BS.1770 và EBU R128.
    Nếu có SciPy, sử dụng chuỗi lọc K-weighting; nếu không, sử dụng phương pháp
    xấp xỉ năng lượng hiệu dụng tích hợp để ước tính độ to chuẩn xác.
    """
    if audio_data is None or len(audio_data) == 0:
        return -70.0
    data = np.asarray(audio_data, dtype=np.float64)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    filtered = data
    try:
        from scipy.signal import butter, sosfilt
        # Bộ lọc thông cao bậc 1 (High-pass filter tại 38 Hz mô phỏng bộ lọc K)
        sos_hp = butter(1, 38.0 / (sr / 2.0), btype='highpass', output='sos')
        filtered = sosfilt(sos_hp, filtered)
        # Bộ lọc kệ cao bậc 2 mô phỏng hiệu ứng khuếch đại đầu người (High-shelf filter tại 1680 Hz)
        sos_hs = butter(2, 1680.0 / (sr / 2.0), btype='lowpass', output='sos')
        # Xấp xỉ đơn giản: giữ độ nhạy dải trung - cao
        filtered = 0.8 * filtered + 0.2 * sosfilt(sos_hs, filtered)
    except Exception:
        # Dự phòng tính toán năng lượng trực tiếp khi không có SciPy
        filtered = data

    # Tính độ to theo từng cửa sổ 400 ms với độ chồng lấn 75%
    win_len = max(1, int(0.400 * sr))
    hop_len = max(1, int(0.100 * sr))
    if len(filtered) < win_len:
        mean_sq = np.mean(np.square(filtered))
        if mean_sq <= 1e-12:
            return -70.0
        return float(-0.691 + 10.0 * np.log10(mean_sq))

    powers = []
    for start in range(0, len(filtered) - win_len + 1, hop_len):
        chunk = filtered[start:start + win_len]
        ms = np.mean(np.square(chunk))
        powers.append(ms)

    powers = np.array(powers)
    valid_powers = powers[powers > 1e-7]
    if len(valid_powers) == 0:
        return -70.0

    loudnesses = -0.691 + 10.0 * np.log10(valid_powers)
    # Áp dụng cổng tuyệt đối tại -70 LUFS
    gated_abs = valid_powers[loudnesses > -70.0]
    if len(gated_abs) == 0:
        return -70.0

    # Áp dụng cổng tương đối tại -10 LU bên dưới mức trung bình cổng tuyệt đối
    gamma_a = -0.691 + 10.0 * np.log10(np.mean(gated_abs))
    gated_rel = gated_abs[(-0.691 + 10.0 * np.log10(gated_abs)) > (gamma_a - 10.0)]
    if len(gated_rel) == 0:
        return float(gamma_a)

    integrated = -0.691 + 10.0 * np.log10(np.mean(gated_rel))
    return float(np.clip(integrated, -70.0, 0.0))


def trim_silence_padding(
    audio_data: np.ndarray,
    sr: int = 24000,
    threshold_db: float = -45.0,
    min_silence_samples: int = 1200,
) -> np.ndarray:
    """
    Cắt bỏ phần khoảng lặng im ắng ở đầu và cuối tệp âm thanh,
    giữ lại khoảng đệm 50 mili giây để tránh cắt cụt nguyên âm.
    """
    if audio_data is None or len(audio_data) == 0:
        return np.zeros(0, dtype=np.float32)
    data = np.asarray(audio_data, dtype=np.float32)
    amp_thresh = float(10.0 ** (threshold_db / 20.0))
    non_silent = np.where(np.abs(data) >= amp_thresh)[0]
    if len(non_silent) == 0:
        return data

    pad = int(0.05 * sr)
    start_idx = max(0, int(non_silent[0]) - pad)
    end_idx = min(len(data), int(non_silent[-1]) + 1 + pad)
    return data[start_idx:end_idx]


def apply_spectral_gating_denoise(
    audio_data: np.ndarray,
    sr: int = 24000,
    noise_reduction_db: float = 8.0,
) -> np.ndarray:
    """
    Khử nhiễu quang phổ nhẹ nhàng bằng kỹ thuật trừ phổ thích ứng (Spectral Gating/Subtraction).
    Chỉ triệt tiêu tạp âm nền tĩnh ở dải năng lượng thấp mà không làm mỏng giọng hay méo tiếng.
    """
    if audio_data is None or len(audio_data) < 512:
        return audio_data
    data = np.asarray(audio_data, dtype=np.float32)

    try:
        n_fft = 512
        hop_length = 256
        # Phân rã biến đổi Fourier thời gian ngắn
        frames = []
        for i in range(0, len(data) - n_fft + 1, hop_length):
            frames.append(data[i:i + n_fft])
        if not frames:
            return data

        window = np.hanning(n_fft).astype(np.float32)
        spec = np.fft.rfft(np.array(frames) * window, axis=1)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Ước tính ngưỡng nhiễu từ 15% khung hình có năng lượng nhỏ nhất
        frame_energies = np.mean(mag, axis=1)
        k_noise = max(1, int(len(frame_energies) * 0.15))
        noise_indices = np.argsort(frame_energies)[:k_noise]
        noise_profile = np.mean(mag[noise_indices], axis=0, keepdims=True)

        # Áp dụng tỷ lệ trừ phổ an toàn
        over_subtraction = 1.2
        gain_factor = 10.0 ** (-noise_reduction_db / 20.0)
        subtracted = mag - over_subtraction * noise_profile
        clean_mag = np.maximum(subtracted, gain_factor * mag)

        # Tái tạo lại tín hiệu thời gian bằng biến đổi Fourier ngược
        clean_spec = clean_mag * np.exp(1j * phase)
        clean_frames = np.fft.irfft(clean_spec, n_fft, axis=1) * window

        out_len = (len(frames) - 1) * hop_length + n_fft
        output = np.zeros(out_len, dtype=np.float32)
        norm_win = np.zeros(out_len, dtype=np.float32)
        for idx, cf in enumerate(clean_frames):
            pos = idx * hop_length
            output[pos:pos + n_fft] += cf
            norm_win[pos:pos + n_fft] += window ** 2

        norm_win[norm_win < 1e-4] = 1.0
        cleaned = output / norm_win
        # Kẹp theo chiều dài tệp gốc
        return cleaned[:len(data)].astype(np.float32)
    except Exception as exc:
        logger.debug(f"Bỏ qua khử nhiễu quang phổ do ngoại lệ: {exc}")
        return data


def normalize_reference_audio(
    audio_data: np.ndarray,
    source_sr: int = 24000,
    target_sr: int = 24000,
    target_lufs: float = -16.0,
    enable_denoise: bool = True,
) -> np.ndarray:
    """
    Quy trình chuẩn hóa toàn diện cho âm thanh mẫu đối chiếu:
    1. Chuyển đổi tần số lấy mẫu về 24000 héc đơn kênh.
    2. Cắt tỉa khoảng lặng thừa ở đầu và cuối tệp.
    3. Khử nhiễu quang phổ nền nhẹ nhàng.
    4. Cân bằng năng lượng âm lượng chuẩn mực về mức LUFS mục tiêu.
    5. Giới hạn biên độ đỉnh ở mức -1.0 dBFS để chống vỡ tiếng.
    """
    if audio_data is None or len(audio_data) == 0:
        return np.zeros(0, dtype=np.float32)
    data = np.asarray(audio_data, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 1. Chuyển đổi tần số lấy mẫu nếu khác mục tiêu
    if source_sr != target_sr:
        try:
            from scipy.signal import resample_poly
            import math
            gcd = math.gcd(int(source_sr), int(target_sr))
            data = resample_poly(data, target_sr // gcd, int(source_sr) // gcd).astype(np.float32)
        except Exception:
            dur = len(data) / float(source_sr)
            old_t = np.arange(len(data)) / float(source_sr)
            new_t = np.arange(int(round(dur * target_sr))) / float(target_sr)
            data = np.interp(new_t, old_t, data).astype(np.float32)
    sr = target_sr

    # 2. Cắt khoảng lặng đầu và cuối
    data = trim_silence_padding(data, sr=sr, threshold_db=-45.0)
    if len(data) == 0:
        return np.zeros(0, dtype=np.float32)

    # 3. Khử nhiễu quang phổ nhẹ nhàng
    if enable_denoise:
        data = apply_spectral_gating_denoise(data, sr=sr, noise_reduction_db=6.0)

    # 4. Chuẩn hóa độ to theo tiêu chuẩn LUFS
    current_lufs = calculate_integrated_lufs(data, sr=sr)
    if current_lufs > -65.0:
        gain_db = float(target_lufs - current_lufs)
        # Giới hạn mức tăng giảm trong khoảng an toàn [-15 dB, +15 dB]
        gain_db = float(np.clip(gain_db, -15.0, 15.0))
        gain_linear = float(10.0 ** (gain_db / 20.0))
        data = data * gain_linear

    # 5. Giới hạn đỉnh âm lượng chống vỡ tiếng (Peak Limit tại 0.891 tương đương -1.0 dBFS)
    peak = float(np.max(np.abs(data)))
    if peak > 0.891:
        data = (data / peak) * 0.891

    return data.astype(np.float32)


def evaluate_reference_audio_quality(
    audio_data: np.ndarray,
    sr: int = 24000,
) -> ReferenceAudioReport:
    """
    Chấm điểm chất lượng âm thanh tham chiếu theo thang điểm 100:
    - Điểm thời lượng hiệu dụng (Tối đa 25 điểm)
    - Điểm tỷ lệ tín hiệu trên tạp âm (Tối đa 25 điểm)
    - Điểm không bị vỡ tiếng (Tối đa 25 điểm)
    - Điểm tỷ lệ khoảng lặng và độ ổn định (Tối đa 25 điểm)

    Quy định ngưỡng:
    - Điểm < 40: Từ chối khóa mẫu giọng tham chiếu.
    """
    if audio_data is None or len(audio_data) == 0:
        return ReferenceAudioReport(
            score=0,
            grade="KHÔNG ĐẠT",
            effective_duration_sec=0.0,
            estimated_snr_db=0.0,
            clipping_ratio=0.0,
            silence_ratio=1.0,
            is_acceptable=False,
            rejection_reason="Tệp âm thanh mẫu rỗng hoặc không đọc được dữ liệu.",
            details={},
        )

    data = np.asarray(audio_data, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    duration_sec = len(data) / float(sr)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))

    # 1. Chấm điểm thời lượng (25 điểm)
    duration_score = 0
    if duration_sec >= 8.0 and duration_sec <= 20.0:
        duration_score = 25
    elif duration_sec > 20.0:
        duration_score = 20
    elif duration_sec >= 4.0:
        duration_score = 18
    elif duration_sec >= 2.5:
        duration_score = 12
    elif duration_sec >= 1.4:
        duration_score = 6
    else:
        duration_score = 0

    # 2. Chấm điểm vỡ tiếng (25 điểm)
    clip_thresh = 0.992
    clipped_samples = int(np.sum(np.abs(data) >= clip_thresh))
    clipping_ratio = float(clipped_samples / len(data))
    if clipping_ratio == 0.0 and peak < 0.98:
        clip_score = 25
    elif clipping_ratio < 0.001:
        clip_score = 20
    elif clipping_ratio < 0.01:
        clip_score = 10
    else:
        clip_score = 0

    # 3. Chấm điểm tỷ lệ tín hiệu trên tạp âm (25 điểm)
    win_len = max(1, int(0.05 * sr))
    energies = []
    for i in range(0, len(data) - win_len + 1, win_len):
        chunk = data[i:i + win_len]
        energies.append(np.mean(np.square(chunk)))
    energies = np.array(energies)

    if len(energies) > 4:
        k_noise = max(1, int(len(energies) * 0.15))
        k_speech = max(1, int(len(energies) * 0.30))
        sorted_e = np.sort(energies)
        noise_floor = float(np.mean(sorted_e[:k_noise]))
        speech_floor = float(np.mean(sorted_e[-k_speech:]))
        if noise_floor > 1e-12:
            snr_db = float(10.0 * np.log10(max(1.0, speech_floor / noise_floor)))
        else:
            snr_db = 35.0
    else:
        snr_db = 15.0

    if snr_db >= 22.0:
        snr_score = 25
    elif snr_db >= 16.0:
        snr_score = 20
    elif snr_db >= 10.0:
        snr_score = 14
    elif snr_db >= 5.0:
        snr_score = 8
    else:
        snr_score = 0

    # 4. Chấm điểm tỷ lệ khoảng lặng và phân bổ năng lượng (25 điểm)
    silence_thresh = 0.015
    silent_chunks = np.sum(energies < (silence_thresh ** 2))
    silence_ratio = float(silent_chunks / len(energies)) if len(energies) > 0 else 0.0

    if silence_ratio <= 0.35 and rms >= 0.025:
        silence_score = 25
    elif silence_ratio <= 0.50 and rms >= 0.015:
        silence_score = 20
    elif silence_ratio <= 0.70 and rms >= 0.008:
        silence_score = 12
    else:
        silence_score = 4

    total_score = int(np.clip(duration_score + clip_score + snr_score + silence_score, 0, 100))

    # Kiểm tra các điều kiện loại trừ trực tiếp (Hard gates)
    is_hard_rejected = False
    rejection_reason = ""
    if duration_sec < 1.4:
        is_hard_rejected = True
        rejection_reason = "Thời lượng âm thanh quá ngắn (dưới 1.4 giây)."
        total_score = min(total_score, 25)
    elif rms < 0.008:
        is_hard_rejected = True
        rejection_reason = "Âm thanh hầu như hoàn toàn im lặng."
        total_score = min(total_score, 20)
    elif clipping_ratio >= 0.02:
        is_hard_rejected = True
        rejection_reason = "Âm thanh bị vỡ tiếng và vượt ngưỡng biên độ nghiêm trọng."
        total_score = min(total_score, 30)

    if total_score >= 80 and not is_hard_rejected:
        grade = "XUẤT SẮC"
    elif total_score >= 65 and not is_hard_rejected:
        grade = "ĐẠT CHUẨN"
    elif total_score >= 40 and not is_hard_rejected:
        grade = "CẦN CHÚ Ý"
    else:
        grade = "KHÔNG ĐẠT"

    is_acceptable = (not is_hard_rejected) and (total_score >= 40)
    if not is_acceptable and not rejection_reason:
        rejection_reason = f"Chất lượng âm thanh tổng thể không đạt chuẩn ({total_score}/100 điểm)." 

    details = {
        "duration_score": duration_score,
        "clip_score": clip_score,
        "snr_score": snr_score,
        "silence_score": silence_score,
        "rms": round(rms, 4),
        "peak": round(peak, 4),
    }

    return ReferenceAudioReport(
        score=total_score,
        grade=grade,
        effective_duration_sec=round(duration_sec, 2),
        estimated_snr_db=round(snr_db, 1),
        clipping_ratio=round(clipping_ratio, 4),
        silence_ratio=round(silence_ratio, 2),
        is_acceptable=is_acceptable,
        rejection_reason=rejection_reason,
        details=details,
    )
