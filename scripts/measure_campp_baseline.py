import time
import json
import os
import sys
import torch

def run_baseline(audio_path: str, ground_truth_json: str, srt_path: str):
    from src.ai_engines.speaker_diarization import SpeakerDiarizationCoordinator
    from src.ai_engines.tts_engine import SubtitleBlock

    if not os.path.exists(audio_path):
        print(f"Không tìm thấy tệp âm thanh: {audio_path}")
        return

    # Đọc tệp nhãn chuẩn
    ground_truth = {}
    if os.path.exists(ground_truth_json):
        with open(ground_truth_json, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.time()
    coordinator = SpeakerDiarizationCoordinator()
    profiles = coordinator.process_video_subtitles(
        video_path=audio_path,
        subtitle_blocks=[],
        exact_speakers=None,
        language="vi",
    )
    elapsed = time.time() - start_time

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    detected_count = len(profiles) if profiles else 0
    print(f"Kết quả đường chuẩn CAM++:")
    print(f"- Số nhân vật phát hiện: {detected_count}")
    print(f"- Thời gian thực thi: {elapsed:.2f} giây")
    print(f"- Bộ nhớ đồ họa đỉnh: {peak_vram_mb:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_baseline(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "", sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        print("Cách dùng: python scripts/measure_campp_baseline.py <audio_path> [ground_truth_json] [srt_path]")
