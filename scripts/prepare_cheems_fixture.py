import os
import subprocess
import sys


def extract_fixture_audio(source_video_or_audio: str, output_wav: str = "test/fixtures/cheems_30s.wav"):
    if not os.path.exists(source_video_or_audio):
        print(f"Lỗi: Không tìm thấy tệp nguồn {source_video_or_audio}")
        return False
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "00:06:18.200",
        "-t",
        "30.0",
        "-i",
        source_video_or_audio,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_wav,
    ]
    print("Đang trích xuất đoạn âm thanh kiểm thử 30 giây...")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode == 0 and os.path.exists(output_wav):
        print(f"Trích xuất thành công: {output_wav} ({os.path.getsize(output_wav)} bytes)")
        return True
    print(f"Lỗi khi chạy FFmpeg:\n{ret.stderr}")
    return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_fixture_audio(sys.argv[1])
    else:
        print("Cách dùng: python scripts/prepare_cheems_fixture.py <duong_dan_tep_cheems_goc>")
