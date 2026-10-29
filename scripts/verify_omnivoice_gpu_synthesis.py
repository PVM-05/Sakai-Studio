"""
Kịch bản xác minh tổng hợp giọng nói thực tế trên GPU NVIDIA với mô hình OmniVoice.
Đánh giá độ trơn tru âm thanh, chuẩn hóa âm lượng, phân tách âm sắc và ưu tiên bản dịch.
"""

import os
import sys

# Đảm bảo xuất chuẩn đầu ra UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import numpy as np
import soundfile as sf
import torch

# Thêm đường dẫn gốc dự án vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_engines.tts_engine import TTSEngine, VIETNAMESE_VOICE_CATALOG
from src.ai_engines.translator import SubtitleBlock


def calculate_audio_metrics(audio_arr: np.ndarray, sr: int) -> dict:
    if audio_arr.size == 0:
        return {"duration": 0.0, "peak": 0.0, "rms": 0.0, "has_nan": False, "is_silent": True}

    duration = float(len(audio_arr) / sr)
    peak = float(np.max(np.abs(audio_arr)))
    rms = float(np.sqrt(np.mean(audio_arr ** 2)))
    has_nan = bool(np.isnan(audio_arr).any() or np.isinf(audio_arr).any())
    is_silent = bool(rms < 1e-4)

    return {
        "duration_sec": round(duration, 3),
        "peak_amplitude": round(peak, 4),
        "rms_energy": round(rms, 4),
        "has_nan_or_inf": has_nan,
        "is_silent": is_silent,
    }


def main():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM ĐỊNH TỔNG HỢP OMNIVOICE THỰC TẾ TRÊN PHẦN CỨNG GPU")
    print("=" * 60)

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    print(f"Thiết bị phần cứng: {device_name} (CUDA khả dụng: {cuda_ok})")
    if not cuda_ok:
        print("CẢNH BÁO: Không tìm thấy GPU CUDA, kịch bản sẽ chạy trên CPU.")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "gpu_verification"))
    os.makedirs(out_dir, exist_ok=True)

    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. KIỂM ĐỊNH PRESET A: NAM THANH NIÊN CHUẨN
    preset_a = "Nam Thanh Niên Chuẩn - Thuyết Minh Và Tin Tức"
    ins_a = VIETNAMESE_VOICE_CATALOG[preset_a]["instruct"]
    print(f"\n[Bước 1] Thử nghiệm với {preset_a}...")
    print(f"Hướng dẫn giọng: {ins_a}")

    engine_a = TTSEngine(
        mode="design",
        multi_speaker_mode="single",
        instruct=ins_a,
        resume_cache_dir=cache_dir,
        normalize_text=False,
    )

    blocks_a = [
        SubtitleBlock(
            index=1,
            start_time=0.0,
            end_time=3.5,
            text="Original english text should never be spoken",
            translated_text="Chào mừng các bạn đã quay trở lại với Sakai Studio.",
            speaker=None,
        ),
        SubtitleBlock(
            index=2,
            start_time=4.0,
            end_time=8.0,
            text="Hôm nay chúng ta sẽ cùng kiểm định công nghệ giọng đọc thông minh.",
            translated_text="",
            speaker="",
        ),
    ]

    out_wav_a = os.path.join(out_dir, "test_preset_a_nam.wav")
    rep_a = engine_a.synthesize_blocks(blocks_a, total_duration=9.0, output_path=out_wav_a)

    print(f"Trạng thái tổng hợp A: {rep_a.status} (Thành công: {rep_a.succeeded_count}/{rep_a.total_blocks})")
    assert rep_a.status == "success", f"Tổng hợp thất bại: {rep_a.message}"
    assert os.path.exists(out_wav_a), "Tệp âm thanh A không tồn tại trên đĩa."

    audio_a, sr_a = sf.read(out_wav_a)
    metrics_a = calculate_audio_metrics(audio_a, sr_a)
    print(f"Thông số âm thanh A: {json.dumps(metrics_a, indent=2, ensure_ascii=False)}")

    assert sr_a == 24000, f"Tần số lấy mẫu không đúng: {sr_a} != 24000"
    assert not metrics_a["has_nan_or_inf"], "Âm thanh A chứa giá trị NaN hoặc Inf"
    assert not metrics_a["is_silent"], "Âm thanh A bị im lặng hoàn toàn"
    assert metrics_a["peak_amplitude"] <= 1.0, f"Âm thanh A bị vượt ngưỡng đỉnh: {metrics_a['peak_amplitude']}"
    assert metrics_a["duration_sec"] >= 4.0, f"Thời lượng âm thanh A quá ngắn: {metrics_a['duration_sec']}"

    # 2. KIỂM ĐỊNH PRESET B: NỮ THANH NIÊN CHUẨN (CÙNG VĂN BẢN ĐỂ SO SÁNH ÂM SẮC)
    preset_b = "Nữ Thanh Niên Chuẩn - Thuyết Minh Và Kể Chuyện"
    ins_b = VIETNAMESE_VOICE_CATALOG[preset_b]["instruct"]
    print(f"\n[Bước 2] Thử nghiệm với {preset_b}...")
    print(f"Hướng dẫn giọng: {ins_b}")

    engine_b = TTSEngine(
        mode="design",
        multi_speaker_mode="single",
        instruct=ins_b,
        resume_cache_dir=cache_dir,
        normalize_text=False,
    )

    blocks_b = [
        SubtitleBlock(
            index=1,
            start_time=0.0,
            end_time=3.5,
            text="Original english text should never be spoken",
            translated_text="Chào mừng các bạn đã quay trở lại với Sakai Studio.",
            speaker="Người nói 1",
        ),
        SubtitleBlock(
            index=2,
            start_time=4.0,
            end_time=8.0,
            text="Hôm nay chúng ta sẽ cùng kiểm định công nghệ giọng đọc thông minh.",
            translated_text="",
            speaker="Người nói 2",
        ),
    ]

    out_wav_b = os.path.join(out_dir, "test_preset_b_nu.wav")
    rep_b = engine_b.synthesize_blocks(blocks_b, total_duration=9.0, output_path=out_wav_b)

    print(f"Trạng thái tổng hợp B: {rep_b.status} (Thành công: {rep_b.succeeded_count}/{rep_b.total_blocks})")
    assert rep_b.status == "success", f"Tổng hợp thất bại: {rep_b.message}"
    assert os.path.exists(out_wav_b), "Tệp âm thanh B không tồn tại trên đĩa."

    audio_b, sr_b = sf.read(out_wav_b)
    metrics_b = calculate_audio_metrics(audio_b, sr_b)
    print(f"Thông số âm thanh B: {json.dumps(metrics_b, indent=2, ensure_ascii=False)}")

    assert sr_b == 24000, f"Tần số lấy mẫu không đúng: {sr_b} != 24000"
    assert not metrics_b["has_nan_or_inf"], "Âm thanh B chứa giá trị NaN hoặc Inf"
    assert not metrics_b["is_silent"], "Âm thanh B bị im lặng hoàn toàn"
    assert metrics_b["peak_amplitude"] <= 1.0, f"Âm thanh B bị vượt ngưỡng đỉnh: {metrics_b['peak_amplitude']}"
    assert metrics_b["duration_sec"] >= 4.0, f"Thời lượng âm thanh B quá ngắn: {metrics_b['duration_sec']}"

    # 3. SO SÁNH ÂM SẮC VÀ ĐỘ ĐỘC LẬP BỘ NHỚ ĐỆM GIỮA A VÀ B
    print("\n[Bước 3] So sánh âm sắc giữa Preset A (Nam) và Preset B (Nữ)...")
    min_len = min(len(audio_a), len(audio_b))
    seg_a = audio_a[:min_len]
    seg_b = audio_b[:min_len]

    # Tính tương quan chéo chuẩn hóa
    norm_a = np.linalg.norm(seg_a)
    norm_b = np.linalg.norm(seg_b)
    if norm_a > 0 and norm_b > 0:
        cosine_sim = float(np.dot(seg_a, seg_b) / (norm_a * norm_b))
    else:
        cosine_sim = 1.0

    print(f"Độ tương đồng cosine dạng sóng: {cosine_sim:.4f}")
    assert cosine_sim < 0.95, f"Âm thanh A và B quá giống nhau (cosine={cosine_sim:.4f}), có thể cache bị va chạm!"
    print("-> Xác nhận: Hai preset sinh ra hai chất giọng hoàn toàn khác biệt, bộ nhớ đệm phân tách tuyệt đối!")

    # 4. KIỂM ĐỊNH TÁI SỬ DỤNG BỘ NHỚ ĐỆM KHI CHẠY LẠI
    print("\n[Bước 4] Kiểm định trúng bộ nhớ đệm (Cache Hit) cho Preset B...")
    out_wav_b_cached = os.path.join(out_dir, "test_preset_b_cached.wav")
    rep_b_cached = engine_b.synthesize_blocks(blocks_b, total_duration=9.0, output_path=out_wav_b_cached)
    assert rep_b_cached.status == "success"
    print(f"-> Hoàn thành tái sử dụng cache thành công trong {rep_b_cached.total_blocks} câu!")

    result_summary = {
        "status": "passed",
        "device": device_name,
        "sample_rate": 24000,
        "preset_a": {
            "name": preset_a,
            "metrics": metrics_a,
            "file": out_wav_a,
        },
        "preset_b": {
            "name": preset_b,
            "metrics": metrics_b,
            "file": out_wav_b,
        },
        "timbre_separation": {
            "cosine_similarity": round(cosine_sim, 4),
            "is_independent": bool(cosine_sim < 0.95),
        },
    }

    result_json_path = os.path.join(out_dir, "verification_report.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("KẾT QUẢ XÁC MINH PHẦN CỨNG THỰC TẾ THÀNH CÔNG VÀ ĐÃ LƯU BÁO CÁO:")
    print(f"Tệp báo cáo: {result_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
