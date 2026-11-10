# -*- coding: utf-8 -*-
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_engines.ref_audio_qc import (
    calculate_integrated_lufs,
    trim_silence_padding,
    apply_spectral_gating_denoise,
    normalize_reference_audio,
    evaluate_reference_audio_quality,
)
from src.ai_engines.tts_engine import is_audio_quality_acceptable_for_voice_lock


class TestRefAudioQC(unittest.TestCase):
    def test_evaluate_quality_rejects_empty_and_short_audio(self):
        """Xác minh bộ kiểm định từ chối âm thanh rỗng hoặc dưới 1.4 giây."""
        empty_rep = evaluate_reference_audio_quality(np.zeros(0, dtype=np.float32), sr=24000)
        self.assertEqual(empty_rep.score, 0)
        self.assertFalse(empty_rep.is_acceptable)

        # Âm thanh ngắn 0.5s
        short_data = np.random.uniform(-0.1, 0.1, int(0.5 * 24000)).astype(np.float32)
        short_rep = evaluate_reference_audio_quality(short_data, sr=24000)
        self.assertFalse(short_rep.is_acceptable)
        self.assertLess(short_rep.score, 40)

    def test_evaluate_quality_detects_clipping(self):
        """Xác minh phát hiện và trừ điểm nặng âm thanh bị vỡ tiếng (clipping)."""
        sr = 24000
        t = np.linspace(0, 5.0, int(5.0 * sr))
        # Tạo sóng sin bị cắt ngọn cực mạnh (clipping 100%)
        clipped_data = np.clip(np.sin(2 * np.pi * 440 * t) * 5.0, -1.0, 1.0).astype(np.float32)
        rep = evaluate_reference_audio_quality(clipped_data, sr=sr)
        self.assertGreater(rep.clipping_ratio, 0.05)
        self.assertEqual(rep.details["clip_score"], 0)

    def test_evaluate_quality_accepts_good_sample(self):
        """Xác minh âm thanh mẫu 8 giây có năng lượng và giọng nói đạt điểm cao."""
        sr = 24000
        dur = 8.0
        t = np.linspace(0, 8.0, int(dur * sr))
        # Mô phỏng giọng nói: sóng sin điều biên với nền sạch
        voice = (np.sin(2 * np.pi * 200 * t) * 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)
        rep = evaluate_reference_audio_quality(voice, sr=sr)
        self.assertTrue(rep.is_acceptable)
        self.assertGreaterEqual(rep.score, 50)
        self.assertIn(rep.grade, ["XUẤT SẮC", "ĐẠT CHUẨN"])

    def test_normalize_reference_audio_pipeline(self):
        """Xác minh quy trình chuẩn hóa: resample 24k, cắt silence, chuẩn hóa năng lượng."""
        in_sr = 16000
        # 1s im lặng ở đầu + 3s giọng + 1s im lặng ở cuối
        silence_start = np.zeros(in_sr, dtype=np.float32)
        voice = (np.random.normal(0, 0.05, in_sr * 3)).astype(np.float32)
        silence_end = np.zeros(in_sr, dtype=np.float32)
        raw_audio = np.concatenate([silence_start, voice, silence_end])

        norm_audio = normalize_reference_audio(raw_audio, source_sr=in_sr, target_sr=24000, target_lufs=-16.0)
        self.assertGreater(len(norm_audio), 0)
        # Thời lượng sau cắt silence phải ngắn hơn 5s ban đầu
        out_dur = len(norm_audio) / 24000.0
        self.assertLess(out_dur, 4.5)
        # Đỉnh âm lượng không vượt quá 0.90
        peak = np.max(np.abs(norm_audio))
        self.assertLessEqual(peak, 0.90)

    def test_is_audio_quality_acceptable_for_voice_lock_hook(self):
        """Xác minh hàm kiểm tra khóa giọng của TTSEngine liên kết đúng với bộ kiểm định mới."""
        sr = 24000
        # Mẫu ngắn dưới 1.4s bị từ chối
        short_data = np.zeros(int(0.8 * sr), dtype=np.float32)
        acc, msg = is_audio_quality_acceptable_for_voice_lock(short_data, sr=sr)
        self.assertFalse(acc)

        # Mẫu chuẩn 8s được chấp nhận kèm điểm số
        t = np.linspace(0, 8.0, int(8.0 * sr))
        voice = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
        acc, msg = is_audio_quality_acceptable_for_voice_lock(voice, sr=sr)
        self.assertTrue(acc)
        self.assertIn("điểm", msg)

    def test_set_reference_voice_rejects_poor_audio(self):
        """Xác minh set_reference_voice chặn hoàn toàn mẫu âm thanh kém chất lượng khi force=False."""
        import tempfile
        import soundfile as sf
        from src.ai_engines.tts_engine import TTSEngine

        engine = TTSEngine()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sr = 24000
            short_audio = np.zeros(int(0.5 * sr), dtype=np.float32)
            sf.write(tmp_path, short_audio, sr)

            res = engine.set_reference_voice(tmp_path, force=False)
            self.assertFalse(res)
            self.assertIsNotNone(engine.last_qc_report)
            self.assertFalse(engine.last_qc_report.is_acceptable)
            self.assertLess(engine.last_qc_report.score, 40)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_set_reference_voice_rejects_corrupted_audio_file(self):
        """Xác minh set_reference_voice từ chối tệp âm thanh hỏng khi force=False."""
        import tempfile
        from src.ai_engines.tts_engine import TTSEngine

        engine = TTSEngine()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"not a valid audio content header or body")
            tmp_path = tmp.name
        try:
            res = engine.set_reference_voice(tmp_path, force=False)
            self.assertFalse(res, "Phải từ chối tệp âm thanh lỗi định dạng khi force=False")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_set_reference_voice_rejects_empty_or_missing_pt_file(self):
        """Xác minh set_reference_voice và load_voice_prompt từ chối tệp mẫu .pt bị rỗng hoặc không tồn tại."""
        import tempfile
        from src.ai_engines.tts_engine import TTSEngine

        engine = TTSEngine()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.assertFalse(engine.set_reference_voice(tmp_path))
            self.assertFalse(engine.load_voice_prompt(tmp_path))
            self.assertFalse(engine.set_reference_voice("nonexistent_model_prompt.pt"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
