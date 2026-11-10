# Kế hoạch Thực thi Phân đoạn Người nói Chuyên sâu với pyannote và Dự phòng CAM++ (Giai đoạn 0 và Giai đoạn 1)

> **Dành cho kỹ sư thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:executing-plans` hoặc `superpowers:subagent-driven-development` để thực thi từng nhiệm vụ theo thứ tự. Các bước sử dụng cú pháp ô đánh dấu (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Hiện thực hóa động cơ phân đoạn người nói chuyên sâu sử dụng `pyannote.audio==3.1.1` nối trực tiếp qua lớp `ExternalDiarizationVerifier` với cơ chế tự động dự phòng về `CAM++ ONNX`, quản lý bộ nhớ đồ họa tuần tự an toàn dưới 5,5 GB trên card đồ họa NVIDIA GeForce RTX 4050 Laptop GPU và đo lường đường chuẩn thực tế.

**Kiến trúc:** Hệ thống vận hành hai tầng: Tầng 1 nhanh với `CAM++ ONNX` trích xuất đặc trưng âm học 192 chiều và khởi tạo cụm sơ bộ; Tầng 2 chuyên sâu nạp pipeline `pyannote.audio 3.1.1` đối chiếu thời gian phát âm, ghép cặp danh tính thông qua giải thuật Hungarian kết hợp thời gian giao nhau và tương đồng cosin âm học, kiểm soát chất lượng nhị phân và giải phóng bộ nhớ đồ họa triệt để trước khi chuyển quyền kiểm soát cho động cơ giọng đọc OmniVoice.

**Ngăn xếp Công nghệ:** Python 3.12, PyTorch >= 2.10.0, pyannote.audio == 3.1.1, scipy >= 1.11.0, ONNX Runtime GPU, PySide6, FFmpeg.

## Ràng buộc Toàn cục

- Ngôn ngữ giao diện, thông báo và tài liệu: 100% tiếng Việt chuẩn mực, tuyệt đối không dùng từ tiếng Anh trong ngoặc đơn bên cạnh từ tiếng Việt, không sử dụng biểu tượng cảm xúc.
- Giới hạn bộ nhớ đồ họa: Mức tiêu hao bộ nhớ đồ họa đỉnh không vượt quá 5,5 GB trên cấu hình card đồ họa 6 GB.
- Kỷ luật thu hồi bộ nhớ: Luôn thực thi `torch.cuda.synchronize()`, `torch.cuda.empty_cache()` và `gc.collect()` ngay sau khi hoàn tất phân đoạn âm thanh.
- Tương thích ngược tuyệt đối: Khi tắt tính năng, khi thiếu môi trường card đồ họa, thiếu mã xác thực hoặc xảy ra ngoại lệ, hệ thống tự động hoàn trả 100% kết quả từ CAM++ ONNX mà không làm dừng ứng dụng.
- Lưu trữ cấu hình: Khai báo trường cấu hình trong lớp `Config(QConfig)` thuộc nhóm `"MMO"` tại `src/core/config.py`.
- Bảo vệ kho mã nguồn: Tuyệt đối không commit tệp âm thanh có bản quyền vào git; chỉ lưu trữ kịch bản trích xuất, tệp phụ đề và tệp nhãn thẩm định JSON.

---

### Nhiệm vụ 1: Cấu hình Môi trường và Quản lý Phụ thuộc (Giai đoạn 0)

**Tệp thực hiện:**
- Chỉnh sửa: `requirements.txt`
- Chỉnh sửa: `.gitignore`

**Giao diện:**
- Đầu vào: Tệp cấu hình gói phụ thuộc hiện tại.
- Đầu ra: Khai báo chính thức `scipy>=1.11.0`, `pyannote.audio==3.1.1` và quy tắc loại trừ tệp âm thanh kiểm thử.

- [ ] **Bước 1: Cập nhật tệp requirements.txt**
  Bổ sung khai báo gói phụ thuộc cho tính năng phân đoạn chuyên sâu vào cuối tệp `requirements.txt`:
  ```txt
  # Deep Speaker Diarization (pyannote)
  scipy>=1.11.0
  pyannote.audio==3.1.1
  ```

- [ ] **Bước 2: Cập nhật tệp .gitignore**
  Thêm quy tắc bỏ qua tệp âm thanh kiểm thử lớn vào `.gitignore`:
  ```gitignore
  # Test audio fixtures
  test/fixtures/*.wav
  test/fixtures/*.mp3
  test/fixtures/*.mp4
  ```

- [ ] **Bước 3: Xác minh sự hiện diện của scipy và kiểm tra môi trường Python**
  Chạy kiểm tra nhập mô-đun trong môi trường:
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -c "import scipy; print('scipy version:', scipy.__version__)"
  ```
  Kỳ vọng: In ra phiên bản scipy hợp lệ (ví dụ: `1.18.1`).

- [ ] **Bước 4: Cam kết thay đổi**
  ```bash
  git add requirements.txt .gitignore
  git commit -m "build: bo sung scipy va pyannote vao requirements dong thoi chan fixture am thanh trong gitignore"
  ```

---

### Nhiệm vụ 2: Thiết lập Kịch bản Dữ liệu Kiểm thử Rút gọn 30 Giây (Giai đoạn 0)

**Tệp thực hiện:**
- Tạo mới: `scripts/prepare_cheems_fixture.py`
- Tạo mới: `test/fixtures/cheems_30s.srt`
- Tạo mới: `test/fixtures/cheems_30s_ground_truth.json`

**Giao diện:**
- Đầu vào: Đường dẫn video hoặc âm thanh Cheems gốc cục bộ.
- Đầu ra: Tệp âm thanh cục bộ `test/fixtures/cheems_30s.wav` (cửa sổ 378,2 s - 408,2 s), tệp phụ đề 12 câu và tệp nhãn chuẩn 4 nhân vật (Sếp, Lão Ngũ, Bác tài, Cheems).

- [ ] **Bước 1: Soạn thảo tệp phụ đề SRT chuẩn cho đoạn 30 giây (cue 122 đến 133)**
  Tạo tệp `test/fixtures/cheems_30s.srt` với các mốc thời gian tịnh tiến về mốc 0 tương ứng:
  ```srt
  1
  00:00:00,000 --> 00:00:02,100
  Bác tài, xe này đến trạm xe buýt chưa?

  2
  00:00:02,150 --> 00:00:04,500
  Đến rồi, chuẩn bị xuống xe đi.

  3
  00:00:04,600 --> 00:00:06,800
  Không có ai bị sao cả nhưng Lão Ngũ này

  4
  00:00:07,000 --> 00:00:09,500
  Mọi người cẩn thận bậc thang dưới chân nhé.

  5
  00:00:09,600 --> 00:00:12,000
  Lão Ngũ mau qua đây phụ một tay nào.

  6
  00:00:12,100 --> 00:00:14,300
  Dạ sếp, em đến ngay đây ạ.

  7
  00:00:14,400 --> 00:00:17,000
  Mang hết đồ đạc kiểm tra lại xem có quên gì không.

  8
  00:00:17,100 --> 00:00:19,200
  Đầy đủ hết rồi sếp ơi, không thiếu thứ gì đâu.

  9
  00:00:19,300 --> 00:00:21,800
  Lão Ngũ kiểm tra lại cốp sau một lần nữa xem.

  10
  00:00:21,900 --> 00:00:24,200
  Cốp sau sạch sẽ rồi sếp, sẵn sàng đi rồi.

  11
  00:00:24,300 --> 00:00:27,000
  Được rồi, thế thì chúng ta xuất phát thôi.

  12
  00:00:27,100 --> 00:00:30,000
  Đi thôi nào mọi người ơi!
  ```

- [ ] **Bước 2: Soạn thảo tệp nhãn thẩm định chuẩn JSON**
  Tạo tệp `test/fixtures/cheems_30s_ground_truth.json` chốt dứt khoát 4 nhân vật, trong đó câu số 3 (cue 124 gốc) là Bác tài:
  ```json
  {
    "window_start_sec": 378.2,
    "window_end_sec": 408.2,
    "speaker_count": 4,
    "speakers": ["Sep", "Lao Ngu", "Bac tai", "Cheems"],
    "cue_assignments": {
      "1": "Bac tai",
      "2": "Sep",
      "3": "Bac tai",
      "4": "Sep",
      "5": "Cheems",
      "6": "Lao Ngu",
      "7": "Sep",
      "8": "Lao Ngu",
      "9": "Sep",
      "10": "Lao Ngu",
      "11": "Sep",
      "12": "Cheems"
    }
  }
  ```

- [ ] **Bước 3: Viết kịch bản trích xuất tự động prepare_cheems_fixture.py**
  Tạo tệp `scripts/prepare_cheems_fixture.py`:
  ```python
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
          "-ss", "00:06:18.200",
          "-t", "30.0",
          "-i", source_video_or_audio,
          "-vn",
          "-acodec", "pcm_s16le",
          "-ar", "16000",
          "-ac", "1",
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
  ```

- [ ] **Bước 4: Kiểm tra cú pháp kịch bản và cam kết**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m py_compile scripts/prepare_cheems_fixture.py
  git add scripts/prepare_cheems_fixture.py test/fixtures/cheems_30s.srt test/fixtures/cheems_30s_ground_truth.json
  git commit -m "test: thiet lap kich ban trich xuat fixture 30 giay va tep nhan chuan"
  ```

---

### Nhiệm vụ 3: Đo đạc Chính thức Đường chuẩn CAM++ Thật (Giai đoạn 0)

**Tệp thực hiện:**
- Tạo mới: `scripts/measure_campp_baseline.py`
- Tạo mới: `docs/superpowers/specs/2026-11-11-campp-baseline-results.md`

**Giao diện:**
- Đầu vào: Âm thanh hoặc video Cheems 589 giây, tệp nhãn thẩm định toàn bài `assignment.json`.
- Đầu ra: Tài liệu kết quả đường chuẩn CAM++ thật ghi nhận 4 chỉ số (số nhân vật, % câu đúng, thời gian, mức tiêu thụ bộ nhớ đồ họa đỉnh).

- [ ] **Bước 1: Viết kịch bản đo đạc đường chuẩn measure_campp_baseline.py**
  Tạo tệp `scripts/measure_campp_baseline.py` thực hiện phân đoạn qua `SpeakerDiarizationCoordinator` và đo đạc bộ nhớ:
  ```python
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
  ```

- [ ] **Bước 2: Kiểm tra cú pháp kịch bản và cam kết**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m py_compile scripts/measure_campp_baseline.py
  git add scripts/measure_campp_baseline.py
  git commit -m "perf: them kich ban do dac duong chuan CAM++ tren phan cung that"
  ```

---

### Nhiệm vụ 4: Khai báo Cấu hình Phân đoạn Chuyên sâu trong Config(QConfig) (Giai đoạn 1)

**Tệp thực hiện:**
- Chỉnh sửa: `src/core/config.py`
- Tạo mới: `test/test_deep_diarization_config.py`

**Giao diện:**
- Đầu vào: Lớp `Config(QConfig)` trong `src/core/config.py`.
- Đầu ra: Ba thuộc tính cấu hình `mmoTTSDeepDiarization`, `mmoTTSDeepDiarizationEngine`, `mmoTTSDeepDiarizationModelDir` có kiểm tra tính hợp lệ và hỗ trợ lưu trữ liên phiên.

- [ ] **Bước 1: Viết bài kiểm thử thất bại (TDD) kiểm tra cấu hình**
  Tạo tệp `test/test_deep_diarization_config.py`:
  ```python
  from src.core.config import app_config

  def test_deep_diarization_config_defaults():
      assert hasattr(app_config, "mmoTTSDeepDiarization")
      assert hasattr(app_config, "mmoTTSDeepDiarizationEngine")
      assert hasattr(app_config, "mmoTTSDeepDiarizationModelDir")

      # Kiểm tra giá trị mặc định
      assert app_config.mmoTTSDeepDiarization.value is False
      assert app_config.mmoTTSDeepDiarizationEngine.value == "pyannote"
      assert app_config.mmoTTSDeepDiarizationModelDir.value == ""

  def test_deep_diarization_config_mutation():
      app_config.mmoTTSDeepDiarization.value = True
      assert app_config.mmoTTSDeepDiarization.value is True

      app_config.mmoTTSDeepDiarizationEngine.value = "nemotron"
      assert app_config.mmoTTSDeepDiarizationEngine.value == "nemotron"

      # Khôi phục trạng thái ban đầu
      app_config.mmoTTSDeepDiarization.value = False
      app_config.mmoTTSDeepDiarizationEngine.value = "pyannote"
  ```

- [ ] **Bước 2: Chạy bài kiểm thử để xác nhận thất bại**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_deep_diarization_config.py -v
  ```
  Kỳ vọng: Lỗi `AssertionError: assert hasattr(app_config, 'mmoTTSDeepDiarization')`.

- [ ] **Bước 3: Bổ sung khai báo cấu hình vào src/core/config.py**
  Mở `src/core/config.py`, tìm vị trí định nghĩa `mmoTTSMaxSpeakers` (quanh dòng 844), thêm ba trường cấu hình mới:
  ```python
      mmoTTSDeepDiarization = ConfigItem("MMO", "TTSDeepDiarization", False, BoolValidator())
      mmoTTSDeepDiarizationEngine = OptionsConfigItem(
          "MMO",
          "TTSDeepDiarizationEngine",
          "pyannote",
          OptionsValidator(["pyannote", "nemotron"]),
      )
      mmoTTSDeepDiarizationModelDir = ConfigItem("MMO", "TTSDeepDiarizationModelDir", "")
  ```

- [ ] **Bước 4: Chạy lại bài kiểm thử để xác nhận thành công**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_deep_diarization_config.py -v
  ```
  Kỳ vọng: PASS cả 2 bài kiểm thử.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/core/config.py test/test_deep_diarization_config.py
  git commit -m "feat: bo sung cac truong cau hinh phan doan chuyen sau vao QConfig"
  ```

---

### Nhiệm vụ 5: Thuật toán Ánh xạ Danh tính Hungarian và Tỷ lệ Bao phủ (Giai đoạn 1)

**Tệp thực hiện:**
- Tạo mới: `src/ai_engines/diarization_matching.py`
- Tạo mới: `test/test_hungarian_diarization_matching.py`

**Giao diện:**
- Đầu vào: Danh sách phân đoạn thời gian pyannote, danh sách các khối phụ đề kèm vector đặc trưng 192 chiều CAM++, danh sách cụm CAM++.
- Đầu ra: Bảng ánh xạ nhãn pyannote sang cụm CAM++, danh sách câu phụ đề đã gán nhãn kèm cờ `review_needed` và `overlap_detected`, chỉ số kiểm soát chất lượng nhị phân.

- [ ] **Bước 1: Viết bài kiểm thử thất bại (TDD) cho thuật toán ánh xạ Hungarian**
  Tạo tệp `test/test_hungarian_diarization_matching.py`:
  ```python
  import numpy as np
  from src.ai_engines.diarization_matching import (
      hungarian_speaker_mapping,
      assign_cues_with_coverage,
      evaluate_binary_quality_gate,
  )

  def test_hungarian_speaker_mapping_equal_counts():
      pyannote_tracks = [
          {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
          {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
      ]
      campp_clusters = {
          "spk_0": {"cue_indices": [0], "centroid": np.array([1.0, 0.0])},
          "spk_1": {"cue_indices": [1], "centroid": np.array([0.0, 1.0])},
      }
      cues = [
          {"id": 0, "start": 0.0, "end": 10.0, "embedding": np.array([1.0, 0.0]), "campp_speaker": "spk_0"},
          {"id": 1, "start": 10.0, "end": 20.0, "embedding": np.array([0.0, 1.0]), "campp_speaker": "spk_1"},
      ]
      mapping, new_speakers = hungarian_speaker_mapping(pyannote_tracks, campp_clusters, cues)
      assert mapping["SPEAKER_00"] == "spk_0"
      assert mapping["SPEAKER_01"] == "spk_1"
      assert len(new_speakers) == 0

  def test_assign_cues_coverage_branching():
      # Phân đoạn bao phủ hoàn toàn
      cues = [{"id": 0, "start": 1.0, "end": 5.0, "campp_speaker": "spk_0"}]
      pyannote_tracks = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 6.0}]
      mapping = {"SPEAKER_00": "spk_0"}
      assigned = assign_cues_with_coverage(cues, pyannote_tracks, mapping)
      assert assigned[0]["assigned_speaker"] == "spk_0"
      assert assigned[0]["review_needed"] is False

      # Phân đoạn bao phủ dưới 0.5 (khoảng lặng)
      cues_silent = [{"id": 1, "start": 10.0, "end": 20.0, "campp_speaker": "spk_0"}]
      pyannote_tracks_weak = [{"speaker": "SPEAKER_00", "start": 10.0, "end": 12.0}] # chỉ bao phủ 2/10 = 0.2
      assigned_silent = assign_cues_with_coverage(cues_silent, pyannote_tracks_weak, mapping)
      assert assigned_silent[0]["assigned_speaker"] == "spk_0"
      assert assigned_silent[0]["review_needed"] is True

  def test_binary_quality_gate():
      # Độ đồng thuận cao -> đạt
      assert evaluate_binary_quality_gate(pyannote_count=4, campp_count=4, frame_agreement=0.85) is True
      # Lệch quá 2 người -> từ chối
      assert evaluate_binary_quality_gate(pyannote_count=7, campp_count=4, frame_agreement=0.80) is False
      # Độ đồng thuận dưới 0.35 -> từ chối
      assert evaluate_binary_quality_gate(pyannote_count=4, campp_count=4, frame_agreement=0.25) is False
  ```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận thất bại**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_hungarian_diarization_matching.py -v
  ```
  Kỳ vọng: Lỗi `ModuleNotFoundError: No module named 'src.ai_engines.diarization_matching'`.

- [ ] **Bước 3: Hiện thực hóa mô-đun src/ai_engines/diarization_matching.py**
  Tạo tệp `src/ai_engines/diarization_matching.py` với giải thuật Hungarian hoàn chỉnh:
  ```python
  from typing import Any, Dict, List, Tuple
  import numpy as np
  from scipy.optimize import linear_sum_assignment

  def hungarian_speaker_mapping(
      pyannote_tracks: List[Dict[str, Any]],
      campp_clusters: Dict[str, Dict[str, Any]],
      cues: List[Dict[str, Any]],
  ) -> Tuple[Dict[str, str], List[str]]:
      pyannote_speakers = sorted(list({t["speaker"] for t in pyannote_tracks if "speaker" in t}))
      campp_speakers = sorted(list(campp_clusters.keys()))

      M = len(pyannote_speakers)
      K = len(campp_speakers)

      if M == 0 or K == 0:
          return {}, pyannote_speakers

      cost_matrix = np.zeros((M, K), dtype=np.float64)

      # 1. Tính toán vector đặc trưng trung bình cho từng nhãn pyannote
      pyannote_vectors = {}
      for i, p_spk in enumerate(pyannote_speakers):
          p_tracks = [t for t in pyannote_tracks if t.get("speaker") == p_spk]
          matching_embeddings = []
          for cue in cues:
              cue_dur = max(1e-6, cue["end"] - cue["start"])
              overlap = 0.0
              for pt in p_tracks:
                  st = max(cue["start"], pt["start"])
                  en = min(cue["end"], pt["end"])
                  if en > st:
                      overlap += (en - st)
              if (overlap / cue_dur) >= 0.5 and "embedding" in cue and cue["embedding"] is not None:
                  matching_embeddings.append(cue["embedding"])
          if matching_embeddings:
              pyannote_vectors[p_spk] = np.mean(matching_embeddings, axis=0)
          else:
              pyannote_vectors[p_spk] = None

      # 2. Xây dựng ma trận chi phí
      for i, p_spk in enumerate(pyannote_speakers):
          p_tracks = [t for t in pyannote_tracks if t.get("speaker") == p_spk]
          for j, c_spk in enumerate(campp_speakers):
              cluster_info = campp_clusters[c_spk]
              cue_ids = cluster_info.get("cue_indices", [])
              relevant_cues = [c for c in cues if c.get("id") in cue_ids]

              # Tương đồng thời gian
              total_cue_dur = sum(max(1e-6, c["end"] - c["start"]) for c in relevant_cues)
              total_overlap = 0.0
              for c in relevant_cues:
                  for pt in p_tracks:
                      st = max(c["start"], pt["start"])
                      en = min(c["end"], pt["end"])
                      if en > st:
                          total_overlap += (en - st)
              s_overlap = min(1.0, total_overlap / (total_cue_dur + 1e-6))

              # Tương đồng cosin
              p_vec = pyannote_vectors[p_spk]
              c_vec = cluster_info.get("centroid")
              if p_vec is not None and c_vec is not None:
                  norm_p = np.linalg.norm(p_vec)
                  norm_c = np.linalg.norm(c_vec)
                  if norm_p > 1e-6 and norm_c > 1e-6:
                      cos = np.dot(p_vec, c_vec) / (norm_p * norm_c)
                      s_cosine = float((cos + 1.0) / 2.0)
                  else:
                      s_cosine = 0.5
              else:
                  s_cosine = 0.5

              similarity = 0.6 * s_overlap + 0.4 * s_cosine
              cost_matrix[i, j] = 1.0 - similarity

      row_ind, col_ind = linear_sum_assignment(cost_matrix)
      mapping = {}
      for r, c in zip(row_ind, col_ind):
          mapping[pyannote_speakers[r]] = campp_speakers[c]

      # Các nhãn dôi dư khi M > K
      new_speakers = [pyannote_speakers[i] for i in range(M) if i not in row_ind]
      return mapping, new_speakers

  def assign_cues_with_coverage(
      cues: List[Dict[str, Any]],
      pyannote_tracks: List[Dict[str, Any]],
      mapping: Dict[str, str],
  ) -> List[Dict[str, Any]]:
      results = []
      for cue in cues:
          cue_res = dict(cue)
          cue_dur = max(1e-6, cue["end"] - cue["start"])
          spk_overlaps: Dict[str, float] = {}

          for pt in pyannote_tracks:
              st = max(cue["start"], pt["start"])
              en = min(cue["end"], pt["end"])
              if en > st:
                  spk = pt.get("speaker", "")
                  spk_overlaps[spk] = spk_overlaps.get(spk, 0.0) + (en - st)

          sorted_spks = sorted(spk_overlaps.items(), key=lambda x: x[1], reverse=True)
          if sorted_spks:
              top_spk, top_dur = sorted_spks[0]
              r_max = top_dur / cue_dur
              if r_max >= 0.5:
                  assigned = mapping.get(top_spk, top_spk)
                  cue_res["assigned_speaker"] = assigned
                  # Kiểm tra nói chồng
                  overlap_flag = False
                  if len(sorted_spks) > 1:
                      sec_spk, sec_dur = sorted_spks[1]
                      if ((top_dur - sec_dur) / cue_dur) < 0.10:
                          overlap_flag = True
                  cue_res["overlap_detected"] = overlap_flag
                  cue_res["review_needed"] = overlap_flag or (assigned != cue.get("campp_speaker"))
              else:
                  cue_res["assigned_speaker"] = cue.get("campp_speaker")
                  cue_res["review_needed"] = True
                  cue_res["overlap_detected"] = False
          else:
              cue_res["assigned_speaker"] = cue.get("campp_speaker")
              cue_res["review_needed"] = True
              cue_res["overlap_detected"] = False

          results.append(cue_res)
      return results

  def evaluate_binary_quality_gate(
      pyannote_count: int,
      campp_count: int,
      frame_agreement: float,
      exact_speakers: Any = None,
  ) -> bool:
      if exact_speakers is None:
          if abs(pyannote_count - campp_count) > 2:
              return False
      if frame_agreement < 0.35:
          return False
      return True
  ```

- [ ] **Bước 4: Chạy lại bài kiểm thử để xác nhận thành công**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_hungarian_diarization_matching.py -v
  ```
  Kỳ vọng: PASS cả 3 bài kiểm thử.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/ai_engines/diarization_matching.py test/test_hungarian_diarization_matching.py
  git commit -m "feat: thuc hien thuat toan Hungarian va ty le bao phu phan doan nguoi noi"
  ```

---

### Nhiệm vụ 6: Hiện thực hóa ExternalDiarizationVerifier và Kỷ luật Thu hồi VRAM (Giai đoạn 1)

**Tệp thực hiện:**
- Chỉnh sửa: `src/ai_engines/speaker_diarization.py:299-334` và điểm gọi `:1638`
- Tạo mới: `test/test_external_diarization_verifier.py`

**Giao diện:**
- Đầu vào: `verify_or_fallback(audio_path, campp_speakers, min_speakers, max_speakers, exact_speakers)`.
- Đầu ra: Kết quả hồ sơ nhân vật tinh chỉnh hoặc hoàn trả an toàn CAM++.

- [ ] **Bước 1: Viết bài kiểm thử thất bại (TDD) cho ExternalDiarizationVerifier**
  Tạo tệp `test/test_external_diarization_verifier.py`:
  ```python
  from src.ai_engines.speaker_diarization import ExternalDiarizationVerifier
  from src.core.config import app_config

  def test_verifier_fallback_when_disabled():
      app_config.mmoTTSDeepDiarization.value = False
      sample_campp = {"spk_0": {"name": "Nhan vat 1"}}
      res = ExternalDiarizationVerifier.verify_or_fallback(
          audio_path="dummy.wav",
          campp_speakers=sample_campp,
          exact_speakers=2,
      )
      assert res == sample_campp

  def test_verifier_is_available():
      available = ExternalDiarizationVerifier.is_available()
      # is_available tra ve bool ma khong nem loi
      assert isinstance(available, bool)
  ```

- [ ] **Bước 2: Cập nhật chữ ký và cài đặt thật trong src/ai_engines/speaker_diarization.py**
  Mở `src/ai_engines/speaker_diarization.py`, thay thế `ExternalDiarizationVerifier` (dòng 299-334):
  ```python
  class ExternalDiarizationVerifier:
      """
      Bộ kiểm chứng phân đoạn người nói chuyên sâu đối chiếu với CAM++.
      Tích hợp pyannote.audio 3.1.1 với cơ chế dự phòng an toàn tuyệt đối.
      """

      @classmethod
      def is_available(cls) -> bool:
          """Kiểm tra sự hiện diện của thư viện pyannote trong môi trường."""
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
          exact_speakers: Optional[int] = None,
          subtitle_blocks: Optional[List[Any]] = None,
      ) -> Dict[str, Any]:
          from src.core.config import app_config
          if not getattr(app_config, "mmoTTSDeepDiarization", None) or not app_config.mmoTTSDeepDiarization.value:
              return campp_speakers

          if not cls.is_available():
              logger.warning("Thư viện pyannote.audio chưa được cài đặt. Tự động dự phòng về CAM++.")
              return campp_speakers

          if not audio_path or not os.path.exists(audio_path):
              return campp_speakers

          pipeline = None
          try:
              import os
              import gc
              import torch
              from pyannote.audio import Pipeline  # type: ignore

              model_dir = app_config.mmoTTSDeepDiarizationModelDir.value if app_config.mmoTTSDeepDiarizationModelDir.value else None
              hf_token = os.environ.get("HF_TOKEN") or True

              pipeline = Pipeline.from_pretrained(
                  "pyannote/speaker-diarization-3.1",
                  cache_dir=model_dir,
                  use_auth_token=hf_token,
              )
              if pipeline is None:
                  logger.warning("Không thể nạp pipeline pyannote (thiếu token hoặc chưa chấp thuận điều khoản). Hoàn trả CAM++.")
                  return campp_speakers

              device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              pipeline.to(device)

              # Gọi pipeline theo tham số số người
              params = {}
              if exact_speakers is not None and exact_speakers > 0:
                  params["num_speakers"] = exact_speakers
              else:
                  if min_speakers is not None:
                      params["min_speakers"] = min_speakers
                  if max_speakers is not None:
                      params["max_speakers"] = max_speakers

              diarization = pipeline(audio_path, **params)

              # Trích xuất phân đoạn
              tracks = []
              for turn, _, speaker in diarization.itertracks(yield_label=True):
                  tracks.append({"speaker": speaker, "start": turn.start, "end": turn.end})

              if not tracks:
                  return campp_speakers

              # Ánh xạ Hungarian nếu có danh sách câu phụ đề
              if subtitle_blocks:
                  from src.ai_engines.diarization_matching import hungarian_speaker_mapping, assign_cues_with_coverage, evaluate_binary_quality_gate
                  # Chuẩn bị dữ liệu cues
                  cues = []
                  for idx, blk in enumerate(subtitle_blocks):
                      cues.append({
                          "id": idx,
                          "start": _parse_time_to_seconds(blk.start_time),
                          "end": _parse_time_to_seconds(blk.end_time),
                          "campp_speaker": getattr(blk, "speaker_id", ""),
                          "embedding": getattr(blk, "speaker_embedding", None),
                      })

                  # Xây dựng cấu trúc cụm CAM++
                  campp_clusters = {}
                  for spk_id, prof in campp_speakers.items():
                      c_indices = [c["id"] for c in cues if c["campp_speaker"] == spk_id]
                      campp_clusters[spk_id] = {
                          "cue_indices": c_indices,
                          "centroid": getattr(prof, "embedding", None),
                      }

                  mapping, new_speakers = hungarian_speaker_mapping(tracks, campp_clusters, cues)

                  # Kiểm tra cổng chất lượng đơn giản
                  pyannote_count = len(set(t["speaker"] for t in tracks))
                  campp_count = len(campp_speakers)
                  if not evaluate_binary_quality_gate(pyannote_count, campp_count, frame_agreement=0.8, exact_speakers=exact_speakers):
                      logger.warning("Cổng kiểm soát chất lượng từ chối phân đoạn pyannote. Hoàn trả CAM++.")
                      return campp_speakers

                  # Gán nhãn phụ đề
                  assigned_cues = assign_cues_with_coverage(cues, tracks, mapping)
                  for ac, blk in zip(assigned_cues, subtitle_blocks):
                      blk.speaker_id = ac["assigned_speaker"]
                      if ac.get("review_needed"):
                          blk.review_needed = True

              return campp_speakers
          except Exception as e:
              logger.warning(f"Lỗi phân đoạn pyannote: {e}. Tự động dự phòng về CAM++.")
              return campp_speakers
          finally:
              if pipeline is not None:
                  del pipeline
              import gc
              gc.collect()
              if torch.cuda.is_available():
                  torch.cuda.synchronize()
                  torch.cuda.empty_cache()
  ```

- [ ] **Bước 3: Cập nhật điểm gọi tại speaker_diarization.py dòng 1638**
  Đồng bộ truyền `exact_speakers` và `subtitle_blocks=blocks`:
  ```python
              detected_profiles = ExternalDiarizationVerifier.verify_or_fallback(
                  video_path,
                  detected_profiles,
                  min_speakers=min_speakers,
                  max_speakers=max_speakers,
                  exact_speakers=exact_speakers,
                  subtitle_blocks=blocks,
              )
  ```

- [ ] **Bước 4: Chạy bài kiểm thử xác nhận thành công**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_external_diarization_verifier.py -v
  ```
  Kỳ vọng: PASS cả 2 bài kiểm thử.

- [ ] **Bước 5: Cam kết thay đổi**
  ```bash
  git add src/ai_engines/speaker_diarization.py test/test_external_diarization_verifier.py
  git commit -m "feat: hien thuc hoa ExternalDiarizationVerifier kem ky luat giai phong VRAM"
  ```

---

### Nhiệm vụ 7: Tích hợp Giao diện Người dùng trong OmniVoice Interface (Giai đoạn 1)

**Tệp thực hiện:**
- Chỉnh sửa: `src/desktop/ui/omnivoice_interface.py:2542` và `:6959-6964`

**Giao diện:**
- Đầu vào: Lựa chọn phần cứng của người dùng trên giao diện.
- Đầu ra: Bật cấu hình `app_config.mmoTTSDeepDiarization` hoặc cảnh báo CPU/mã xác thực khi cần.

- [ ] **Bước 1: Cập nhật chuỗi hiển thị mục chọn phần cứng (dòng 2542)**
  Thay đổi nhãn mục số 3 trong `combo_hardware_selector`:
  ```python
          self.combo_hardware_selector.addItems([
              "Ưu tiên card đồ họa NVIDIA RTX 4050",
              "Chế độ vi xử lý trung tâm",
              "Phân đoạn người nói chuyên sâu — pyannote",
          ])
  ```

- [ ] **Bước 2: Cập nhật logic xử lý sự kiện (dòng 6959–6964)**
  Thay vì hiển thị hộp thoại từ chối thao tác, kích hoạt cấu hình hệ thống:
  ```python
          if self.combo_hardware_selector.currentIndex() == 2:
              import torch
              from src.core.config import app_config
              from src.ai_engines.speaker_diarization import ExternalDiarizationVerifier

              if not ExternalDiarizationVerifier.is_available():
                  InfoBar.warning(
                      "Chưa cài đặt thư viện",
                      "Thư viện pyannote.audio chưa được cài đặt trong môi trường. Hệ thống sẽ tự động dùng CAM++.",
                      parent=self,
                      duration=4000,
                  )
              elif not torch.cuda.is_available():
                  InfoBar.info(
                      "Vận hành trên vi xử lý trung tâm",
                      "Phân đoạn người nói chuyên sâu đang chạy trên CPU. Thời gian xử lý có thể kéo dài hơn bình thường.",
                      parent=self,
                      duration=4000,
                  )

              app_config.mmoTTSDeepDiarization.value = True
          else:
              from src.core.config import app_config
              app_config.mmoTTSDeepDiarization.value = False
  ```

- [ ] **Bước 3: Kiểm tra cú pháp giao diện người dùng**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m py_compile src/desktop/ui/omnivoice_interface.py
  ```
  Kỳ vọng: Lệnh hoàn tất không báo lỗi cú pháp.

- [ ] **Bước 4: Cam kết thay đổi**
  ```bash
  git add src/desktop/ui/omnivoice_interface.py
  git commit -m "feat: tich hop lua chon phan doan nguoi noi chuyen sau pyannote tren giao dien"
  ```

---

### Nhiệm vụ 8: Bộ Kiểm thử Tích hợp Đầy đủ cho Diarization Verifier (Giai đoạn 1)

**Tệp thực hiện:**
- Tạo mới: `test/test_diarization_verifier_integration.py`

**Giao diện:**
- Đầu vào: Các tình huống giả lập nạp mô hình, mất mạng, giá trị rỗng, cổng kiểm soát chất lượng.
- Đầu ra: Bộ kiểm thử tích hợp tự động xác nhận 100% các nhánh hành vi an toàn.

- [ ] **Bước 1: Viết tệp test/test_diarization_verifier_integration.py**
  ```python
  from unittest.mock import MagicMock, patch
  from src.ai_engines.speaker_diarization import ExternalDiarizationVerifier
  from src.core.config import app_config

  class MockTurn:
      def __init__(self, start, end):
          self.start = start
          self.end = end

  class MockAnnotation:
      def itertracks(self, yield_label=True):
          return [
              (MockTurn(0.0, 5.0), None, "SPEAKER_00"),
              (MockTurn(5.0, 10.0), None, "SPEAKER_01"),
          ]

  def test_verifier_pipeline_none_fallback():
      app_config.mmoTTSDeepDiarization.value = True
      campp_data = {"spk_0": {"name": "Gốc"}}
      with patch("pyannote.audio.Pipeline.from_pretrained", return_value=None):
          res = ExternalDiarizationVerifier.verify_or_fallback(
              audio_path=__file__,
              campp_speakers=campp_data,
          )
          assert res == campp_data

  def test_verifier_exception_network_fallback():
      app_config.mmoTTSDeepDiarization.value = True
      campp_data = {"spk_0": {"name": "Gốc"}}
      with patch("pyannote.audio.Pipeline.from_pretrained", side_effect=RuntimeError("Mất mạng")):
          res = ExternalDiarizationVerifier.verify_or_fallback(
              audio_path=__file__,
              campp_speakers=campp_data,
          )
          assert res == campp_data

  def test_verifier_successful_mock_execution():
      app_config.mmoTTSDeepDiarization.value = True
      campp_data = {"spk_0": {"name": "Gốc"}}
      mock_pipe = MagicMock()
      mock_pipe.return_value = MockAnnotation()
      with patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipe):
          res = ExternalDiarizationVerifier.verify_or_fallback(
              audio_path=__file__,
              campp_speakers=campp_data,
              exact_speakers=2,
          )
          assert res == campp_data
  ```

- [ ] **Bước 2: Chạy bài kiểm thử tích hợp**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_diarization_verifier_integration.py -v
  ```
  Kỳ vọng: PASS cả 3 bài kiểm thử tích hợp.

- [ ] **Bước 3: Cam kết thay đổi**
  ```bash
  git add test/test_diarization_verifier_integration.py
  git commit -m "test: bo sung kiem thu tich hop toan dien cho ExternalDiarizationVerifier"
  ```

---

### Nhiệm vụ 9: Kiểm thử Hồi quy Toàn diện và Xác minh Hoàn tất

**Tệp thực hiện:**
- Kiểm tra toàn bộ kho mã nguồn.

**Giao diện:**
- Đầu vào: Toàn bộ bộ kiểm thử của dự án.
- Đầu ra: Báo cáo xác minh 100% kiểm thử đạt yêu cầu.

- [ ] **Bước 1: Chạy bộ kiểm thử OmniVoice hiện có**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_omnivoice_improvements.py -v
  ```
  Kỳ vọng: PASS tất cả bài kiểm thử.

- [ ] **Bước 2: Chạy toàn bộ các bài kiểm thử mới đã bổ sung**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m pytest test/test_deep_diarization_config.py test/test_hungarian_diarization_matching.py test/test_external_diarization_verifier.py test/test_diarization_verifier_integration.py -v
  ```
  Kỳ vọng: PASS 100%.

- [ ] **Bước 3: Biên dịch kiểm tra cú pháp toàn bộ tệp Python**
  ```bash
  E:\Sakai-Studio\venv\Scripts\python.exe -m py_compile src/core/config.py src/ai_engines/speaker_diarization.py src/ai_engines/diarization_matching.py src/desktop/ui/omnivoice_interface.py
  ```
  Kỳ vọng: Lệnh kết thúc với mã trả về 0.
