import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.translator import (
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

    def test_free_google_translator(self):
        translator = FreeGoogleTranslator()
        block = SubtitleBlock(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:02,000",
            text="Thank you",
        )
        res = translator.translate_blocks([block], source_lang="en", target_lang="vi")
        self.assertEqual(len(res), 1)
        self.assertTrue(len(res[0].translated_text) > 0)


if __name__ == "__main__":
    unittest.main()
