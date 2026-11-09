import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_engines.translator import (
    SubtitleBlock,
    parse_srt,
    blocks_to_srt,
    blocks_to_bilingual_srt,
    FreeGoogleTranslator,
)


class TestSrtTranslator(unittest.TestCase):
    def test_parse_and_bilingual_srt(self):
        sample_srt = """1
00:00:01,000 --> 00:00:03,500
Hello World!

2
00:00:04,000 --> 00:00:07,000
This is a test subtitle.
"""
        blocks = parse_srt(sample_srt)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, "Hello World!")
        blocks[0].translated_text = "Xin chào thế giới!"

        reconstructed = blocks_to_srt(blocks, use_translated=True)
        self.assertIn("Xin chào thế giới!", reconstructed)
        bilingual = blocks_to_bilingual_srt(blocks)
        self.assertIn("Hello World!\nXin chào thế giới!", bilingual)

    def test_free_google_translator_mocked_offline(self):
        translator = FreeGoogleTranslator()
        block = SubtitleBlock(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:02,000",
            text="Thank you",
        )
        fake_payload = [[["Cảm ơn bạn", "Thank you", None, None, 10]], None, "en"]
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            res = translator.translate_blocks([block], source_lang="en", target_lang="vi")

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].translated_text, "Cảm ơn bạn")

    def test_translation_interface_ollama_provider_and_bilingual_export(self):
        """
        Xác minh nhà cung cấp Ollama có sẵn trong danh mục cấu hình và nút xuất song ngữ
        tạo ra tệp phụ đề song ngữ hợp lệ.
        """
        from src.desktop.ui.translation_interface import PROVIDERS_INFO, TranslationInterface
        from PySide6.QtWidgets import QApplication
        import tempfile
        from pathlib import Path

        self.assertIn("Ollama", PROVIDERS_INFO)
        self.assertEqual(PROVIDERS_INFO["Ollama"]["base_url"], "http://localhost:11434/v1")
        self.assertTrue(PROVIDERS_INFO["Ollama"]["is_local"])

        app = QApplication.instance() or QApplication([])
        ui = TranslationInterface()
        self.assertTrue(hasattr(ui, "export_bilingual_srt_button"))

        block = SubtitleBlock(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Hello")
        block.translated_text = "Xin chào"
        ui.subtitle_blocks = [block]

        with tempfile.TemporaryDirectory() as td:
            out_file = os.path.join(td, "test_bilingual.srt")
            with mock.patch("src.desktop.ui.translation_interface.FolderMemoryDialog.getSaveFileName", return_value=(out_file, "")):
                ui._export_bilingual_srt()

            self.assertTrue(os.path.exists(out_file))
            saved_content = Path(out_file).read_text(encoding="utf-8")
            self.assertIn("Hello", saved_content)
            self.assertIn("Xin chào", saved_content)


if __name__ == "__main__":
    unittest.main()
