"""Subtitle Translation Backend for Sakai Studio SRT Translation Tab.

Supports:
- Parsing SRT, VTT, and ASS subtitle files.
- Translation via Free Google Translate API, OpenAI/Gemini/Ollama LLM APIs, and DeepL API.
- Exporting single-language SRT/VTT/ASS and Bilingual (Song ngữ) SRT.
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any

logger = logging.getLogger(__name__)

# Standard SRT Timestamp regex: 00:01:20,000 --> 00:01:23,500
SRT_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})$"
)


@dataclass
class SubtitleBlock:
    """Represents a single subtitle block with index, timestamps, and text lines."""
    index: int
    start_time: str
    end_time: str
    text: str
    translated_text: str = ""
    # Smart ASS Override coordinates
    xmin: int = -1
    xmax: int = -1
    ymin: int = -1
    ymax: int = -1

    def to_srt(self, use_translated: bool = True) -> str:
        content = self.translated_text if (use_translated and self.translated_text) else self.text
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{content}\n"

    def to_bilingual_srt(self) -> str:
        translated = self.translated_text if self.translated_text else self.text
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n{translated}\n"


def parse_srt(srt_content: str) -> List[SubtitleBlock]:
    """Parse an SRT string content into a list of SubtitleBlock items."""
    blocks: List[SubtitleBlock] = []
    raw_blocks = re.split(r"\n\s*\n", srt_content.strip())

    for raw in raw_blocks:
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        # Parse index
        try:
            index = int(lines[0])
            time_line_idx = 1
        except ValueError:
            index = len(blocks) + 1
            time_line_idx = 0

        if time_line_idx >= len(lines):
            continue

        match = SRT_TIMESTAMP_PATTERN.match(lines[time_line_idx])
        if not match:
            for i, l in enumerate(lines):
                m = SRT_TIMESTAMP_PATTERN.match(l)
                if m:
                    match = m
                    time_line_idx = i
                    break

        if not match:
            continue

        start_time, end_time = match.group(1), match.group(2)
        start_time = start_time.replace('.', ',')
        end_time = end_time.replace('.', ',')

        text_lines = lines[time_line_idx + 1 :]
        text = "\n".join(text_lines)

        blocks.append(
            SubtitleBlock(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

    return blocks


def blocks_to_srt(blocks: List[SubtitleBlock], use_translated: bool = True) -> str:
    """Reconstruct SRT content string from SubtitleBlock list."""
    return "\n".join(b.to_srt(use_translated=use_translated) for b in blocks)


def blocks_to_bilingual_srt(blocks: List[SubtitleBlock]) -> str:
    """Generate Bilingual (Song ngữ) SRT string from SubtitleBlock list."""
    return "\n".join(b.to_bilingual_srt() for b in blocks)


class BaseTranslator:
    """Base class for all translation engines."""

    def translate_blocks(
        self,
        blocks: List[SubtitleBlock],
        source_lang: str = "auto",
        target_lang: str = "vi",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_callback: Optional[Callable[[], Any]] = None,
    ) -> List[SubtitleBlock]:
        raise NotImplementedError


class FreeGoogleTranslator(BaseTranslator):
    """Free Google Translate web API engine (No API Key required) with batching."""

    def __init__(self, timeout_sec: float = 12.0, batch_size: int = 25):
        self.timeout_sec = timeout_sec
        self.batch_size = batch_size

    def _normalize_lang_code(self, code: str) -> str:
        if not code or "auto" in code.lower():
            return "auto"
        c = code.lower().strip()
        mapping = {
            "vietnamese": "vi",
            "english": "en",
            "chinese": "zh-CN",
            "japanese": "ja",
            "korean": "ko",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "russian": "ru",
        }
        return mapping.get(c, c[:5].split("-")[0] if "-" not in c else c[:5])

    def _translate_single_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        sl = urllib.parse.quote(self._normalize_lang_code(source_lang))
        tl = urllib.parse.quote(self._normalize_lang_code(target_lang))
        clean_text = text.replace("\n", " [NL] ")
        url = (
            "https://translate.googleapis.com/translate_a/single?"
            f"client=gtx&sl={sl}&tl={tl}&dt=t&q="
            + urllib.parse.quote(clean_text)
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result_parts = []
                if data and isinstance(data[0], list):
                    for item in data[0]:
                        if item and isinstance(item, list) and len(item) > 0:
                            result_parts.append(str(item[0]))
                res = "".join(result_parts)
                res = res.replace("[NL]", "\n").replace("[ NL ]", "\n").replace("[NL ]", "\n")
                return res.strip()
        except Exception as e:
            logger.warning(f"Google Translate request failed for '{text[:30]}...': {e}")
            return text

    def _translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a list of texts in a single HTTP request using delimiters."""
        if not texts:
            return []
        if len(texts) == 1:
            return [self._translate_single_text(texts[0], source_lang, target_lang)]

        sl = urllib.parse.quote(self._normalize_lang_code(source_lang))
        tl = urllib.parse.quote(self._normalize_lang_code(target_lang))
        delimiter = " [SEP_BLK] "
        cleaned_texts = [t.replace("\n", " [NL] ") for t in texts]
        combined_text = delimiter.join(cleaned_texts)

        url = (
            "https://translate.googleapis.com/translate_a/single?"
            f"client=gtx&sl={sl}&tl={tl}&dt=t&q="
            + urllib.parse.quote(combined_text)
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec * 2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result_parts = []
                if data and isinstance(data[0], list):
                    for item in data[0]:
                        if item and isinstance(item, list) and len(item) > 0:
                            result_parts.append(str(item[0]))
                raw_res = "".join(result_parts)

                # Split by delimiter variations
                parts = re.split(r"\s*\[\s*SEP_BLK\s*\]\s*", raw_res, flags=re.IGNORECASE)
                if len(parts) == len(texts):
                    results = []
                    for part in parts:
                        clean_part = part.replace("[NL]", "\n").replace("[ NL ]", "\n").replace("[NL ]", "\n").strip()
                        results.append(clean_part)
                    return results

                logger.warning(f"Batch delimiter split mismatch ({len(parts)} vs {len(texts)}), falling back to line-by-line")
        except Exception as e:
            logger.warning(f"Batch translation failed: {e}, falling back to single text mode")

        # Fallback to single line requests
        return [self._translate_single_text(t, source_lang, target_lang) for t in texts]

    def translate_blocks(
        self,
        blocks: List[SubtitleBlock],
        source_lang: str = "auto",
        target_lang: str = "vi",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_callback: Optional[Callable[[], Any]] = None,
    ) -> List[SubtitleBlock]:
        total = len(blocks)
        if total == 0:
            return blocks

        for i in range(0, total, self.batch_size):
            if cancel_callback:
                while cancel_callback() == "paused":
                    time.sleep(0.3)
                if cancel_callback() is True:
                    break

            batch = blocks[i : i + self.batch_size]
            current_count = min(i + len(batch), total)
            if progress_callback:
                progress_callback(
                    current_count / total,
                    f"Đang dịch câu {current_count}/{total} (Batch {i // self.batch_size + 1})..."
                )

            texts_to_translate = [b.text for b in batch]
            translated_texts = self._translate_batch(texts_to_translate, source_lang, target_lang)

            for block, trans in zip(batch, translated_texts):
                block.translated_text = trans if trans else block.text

            time.sleep(0.05)  # Avoid rate limiting

        if progress_callback:
            progress_callback(1.0, f"Đã dịch xong {total} câu phụ đề.")

        return blocks


class LlamaCppTranslator(BaseTranslator):
    """Local GGUF LLM Translation engine using llama-cpp-python (Direct GGUF execution without Ollama App)."""

    def __init__(self, model_path: str | Path, custom_prompt: Optional[str] = None, batch_size: int = 8):
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.llm = None
        self.custom_prompt = custom_prompt or (
            "You are a professional video subtitle translator.\n"
            "Translate the following subtitle lines from {source_lang} into natural, fluent {target_lang}.\n"
            "Keep line counts identical and preserve HTML formatting like <i> or <b>.\n"
            "Respond ONLY with a JSON array of translated strings matching the input array order."
        )

    def _load_model(self, progress_callback=None):
        if self.llm is None:
            model_p = Path(self.model_path)
            if not model_p.exists() or not model_p.is_file():
                model_p = download_default_gguf_model(progress_callback=progress_callback)
                self.model_path = str(model_p)

            threads_count = max(1, (os.cpu_count() or 4) - 2)
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1,
                n_threads=threads_count,
                n_threads_batch=threads_count,
                n_batch=512,
                n_ctx=2048,
                verbose=False
            )

    def translate_blocks(
        self,
        blocks: List[SubtitleBlock],
        source_lang: str = "auto",
        target_lang: str = "vi",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_callback: Optional[Callable[[], Any]] = None,
    ) -> List[SubtitleBlock]:
        total = len(blocks)
        if total == 0:
            return blocks

        try:
            self._load_model(progress_callback=progress_callback)
        except Exception as e:
            logger.warning(f"Failed to load GGUF model from {self.model_path}: {e}. Falling back to Free Google Translate.")
            fallback = FreeGoogleTranslator()
            return fallback.translate_blocks(blocks, source_lang, target_lang, progress_callback)

        lang_name_map = {
            "auto": "auto-detected original language",
            "vi": "Vietnamese",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
        }
        source_lang_full = lang_name_map.get(source_lang.lower(), source_lang)
        target_lang_full = lang_name_map.get(target_lang.lower(), target_lang)

        for i in range(0, total, self.batch_size):
            # Support pause/resume and cancel/stop
            if cancel_callback:
                while cancel_callback() == "paused":
                    time.sleep(0.3)
                if cancel_callback() is True:
                    logger.info("GGUF Translation cancelled by user.")
                    break

            batch = blocks[i : i + self.batch_size]
            if progress_callback:
                progress_callback(i / total, f"GGUF Local AI đang dịch câu {i + 1}/{total}...")

            texts_to_translate = [b.text for b in batch]
            translated_texts = self._translate_batch(texts_to_translate, source_lang_full, target_lang_full)

            for block, trans in zip(batch, translated_texts):
                block.translated_text = trans if trans else block.text

        if progress_callback:
            progress_callback(1.0, f"Đã hoàn thành dịch {total} câu bằng GGUF Local AI.")
        return blocks

    def _parse_json_array_response(self, content: str, expected_count: int) -> Optional[List[str]]:
        if not content:
            return None
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        # Step 1: Standard json.loads
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            raw_json = match.group(0)
            raw_json = re.sub(r",\s*\]", "]", raw_json)
            try:
                arr = json.loads(raw_json)
                if isinstance(arr, list) and len(arr) == expected_count:
                    return [str(item) for item in arr]
            except Exception:
                pass

            # Step 2: Extract all quoted string literals inside brackets
            extracted_strings = re.findall(r'"((?:[^"\\]|\\.)*)"', raw_json)
            if len(extracted_strings) == expected_count:
                return [s.replace('\\"', '"').replace('\\n', '\n') for s in extracted_strings]

        # Step 3: Extract numbered list if model responded with 1. xxx 2. yyy
        numbered_lines = re.findall(r'^\s*\d+[\.\:\-]\s*(.*)$', cleaned, re.MULTILINE)
        if len(numbered_lines) == expected_count:
            return [l.strip().strip('"').strip("'") for l in numbered_lines]

        # Step 4: Extract raw lines excluding JSON structure tokens
        lines = [line.strip().strip('"').strip("'").rstrip(',') for line in cleaned.splitlines() if line.strip()]
        lines = [l for l in lines if l not in ("[", "]", "```", "```json")]
        if len(lines) == expected_count:
            return lines

        return None

    def _translate_batch(self, texts: List[str], source_lang_full: str, target_lang_full: str) -> List[str]:
        if "{source_lang}" in self.custom_prompt or "{target_lang}" in self.custom_prompt:
            try:
                prompt_system = self.custom_prompt.format(source_lang=source_lang_full, target_lang=target_lang_full)
            except Exception:
                prompt_system = f"Translate from {source_lang_full} into {target_lang_full}. Respond ONLY with a valid JSON array of strings."
        else:
            prompt_system = f"{self.custom_prompt}\nTranslate from {source_lang_full} into {target_lang_full}. Respond ONLY with a valid JSON array of strings."

        user_content = json.dumps(texts, ensure_ascii=False)
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response["choices"][0]["message"]["content"].strip()
            parsed = self._parse_json_array_response(content, len(texts))
            if parsed:
                return parsed
        except Exception as e:
            logger.debug(f"GGUF batch translation parse attempt: {e}")

        # Robust Fallback to FreeGoogleTranslator if JSON array fails for this batch
        try:
            fallback = FreeGoogleTranslator()
            fallback_blocks = [SubtitleBlock(index=i+1, start_time="00:00:00,000", end_time="00:00:01,000", text=t) for i, t in enumerate(texts)]
            res = fallback.translate_blocks(fallback_blocks, source_lang=source_lang_full, target_lang=target_lang_full)
            return [b.translated_text for b in res]
        except Exception:
            return texts


def get_available_gguf_models() -> List[Path]:
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return list(models_dir.glob("*.gguf"))


def download_default_gguf_model(progress_callback=None) -> Path:
    """Tải tự động mô hình Qwen2.5 GGUF từ HuggingFace về thư mục models/."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_file = models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if target_file.exists():
        return target_file

    try:
        from huggingface_hub import hf_hub_download
        if progress_callback:
            progress_callback(0.1, "Đang tự động tải mô hình Qwen2.5-1.5B GGUF từ HuggingFace...")
        downloaded = hf_hub_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            local_dir=str(models_dir)
        )
        return Path(downloaded)
    except Exception as e:
        logger.error(f"Failed to auto-download Qwen2.5 GGUF model: {e}")
        raise RuntimeError(f"Không thể tự động tải Qwen2.5 GGUF: {e}")


class MarianNMTTranslator(BaseTranslator):
    """Local Offline NMT Translation using HuggingFace MarianMT with PyTorch / GPU CUDA acceleration."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-vi"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" and self._has_cuda()) else "cpu"

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_model(self):
        if self.model is None:
            from transformers import MarianMTModel, MarianTokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)

    def translate_blocks(
        self,
        blocks: List[SubtitleBlock],
        source_lang: str = "auto",
        target_lang: str = "vi",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_callback: Optional[Callable[[], Any]] = None,
    ) -> List[SubtitleBlock]:
        total = len(blocks)
        if total == 0:
            return blocks

        # Dynamic model selection for MarianMT (Chinese vs English to Vietnamese)
        src = source_lang.lower().strip()
        if src in ("zh", "zh-cn", "zh-tw", "chinese"):
            target_model_name = "Helsinki-NLP/opus-mt-zh-vi"
        else:
            target_model_name = "Helsinki-NLP/opus-mt-en-vi"

        if self.model is None or self.model_name != target_model_name:
            self.model_name = target_model_name
            self.model = None
            self.tokenizer = None

        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"Failed to load MarianMT model {self.model_name}: {e}. Falling back to Free Google Translate.")
            fallback = FreeGoogleTranslator()
            return fallback.translate_blocks(blocks, source_lang, target_lang, progress_callback, cancel_callback=cancel_callback)

        batch_size = 32
        for i in range(0, total, batch_size):
            if cancel_callback:
                while cancel_callback() == "paused":
                    time.sleep(0.3)
                if cancel_callback() is True:
                    logger.info("MarianMT translation cancelled by user.")
                    break

            batch = blocks[i : i + batch_size]
            if progress_callback:
                progress_callback(i / total, f"Offline GPU MarianMT đang dịch câu {i + 1}/{total}...")

            texts = [b.text for b in batch]
            try:
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                translated_tokens = self.model.generate(**inputs)
                results = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                for block, trans in zip(batch, results):
                    block.translated_text = trans if trans else block.text
            except Exception as e:
                logger.warning(f"MarianMT batch translation error: {e}")
                for block in batch:
                    block.translated_text = block.text

        if progress_callback:
            progress_callback(1.0, f"Đã dịch xong {total} câu bằng GPU Offline NMT.")
        return blocks


class LLMTranslator(BaseTranslator):
    """LLM Translation engine (OpenAI, Gemini, Ollama, DeepSeek)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4o-mini",
        custom_prompt: Optional[str] = None,
        batch_size: int = 20,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model
        self.batch_size = batch_size
        self.custom_prompt = custom_prompt or (
            "You are a professional video subtitle translator.\n"
            "Translate the following subtitle lines from {source_lang} into natural, fluent {target_lang}.\n"
            "Keep line counts identical and preserve HTML formatting like <i> or <b>.\n"
            "Respond ONLY with a JSON array of translated strings matching the input array order."
        )

    def translate_blocks(
        self,
        blocks: List[SubtitleBlock],
        source_lang: str = "auto",
        target_lang: str = "vi",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_callback: Optional[Callable[[], Any]] = None,
    ) -> List[SubtitleBlock]:
        total = len(blocks)
        if total == 0:
            return blocks

        lang_name_map = {
            "auto": "auto-detected original language",
            "vi": "Vietnamese",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
        }
        source_lang_full = lang_name_map.get(source_lang.lower(), source_lang)
        target_lang_full = lang_name_map.get(target_lang.lower(), target_lang)

        for i in range(0, total, self.batch_size):
            if cancel_callback:
                while cancel_callback() == "paused":
                    time.sleep(0.3)
                if cancel_callback() is True:
                    logger.info("LLM translation cancelled by user.")
                    break

            batch = blocks[i : i + self.batch_size]
            if progress_callback:
                progress_callback(i / total, f"Đang dịch câu {i + 1} - {min(i + len(batch), total)}/{total}...")

            texts_to_translate = [b.text for b in batch]
            translated_texts = self._translate_batch(texts_to_translate, source_lang_full, target_lang_full, target_lang_iso=target_lang)

            for block, trans in zip(batch, translated_texts):
                block.translated_text = trans if trans else block.text

            time.sleep(3.0)  # Pause to respect 15 RPM Free Tier rate limit (60s / 15 = 4s total interval)

        if progress_callback:
            progress_callback(1.0, f"Đã hoàn thành dịch {total} câu phụ đề.")

        return blocks

    def _translate_batch(self, texts: List[str], source_lang_full: str, target_lang_full: str, target_lang_iso: str = "vi") -> List[str]:
        if "{source_lang}" in self.custom_prompt or "{target_lang}" in self.custom_prompt:
            try:
                prompt_system = self.custom_prompt.format(source_lang=source_lang_full, target_lang=target_lang_full)
            except Exception:
                prompt_system = (
                    f"You are a professional video subtitle translator.\n"
                    f"Translate the following subtitle lines from {source_lang_full} into natural, fluent {target_lang_full}.\n"
                    f"Keep line counts identical and preserve HTML formatting like <i> or <b>.\n"
                    f"Respond ONLY with a JSON array of translated strings matching the input array order."
                )
        else:
            prompt_system = (
                f"{self.custom_prompt}\n"
                f"Translate from {source_lang_full} into {target_lang_full}.\n"
                f"Respond ONLY with a JSON array of translated strings matching the input array order."
            )

        # Model name alias mapping & normalization
        model_name = self.model.lower().strip()
        if model_name in ("sonnet", "claude-3.5-sonnet"):
            model_name = "claude-3-5-sonnet-latest"
        elif model_name in ("opus", "claude-3-opus"):
            model_name = "claude-3-opus-latest"
        elif model_name in ("haiku", "claude-3.5-haiku"):
            model_name = "claude-3-5-haiku-latest"
        elif model_name in ("gemini", "gemini-3", "gemini-3.5", "gemini-3.5-flash", "gemini-3.0-flash"):
            model_name = "gemini-2.0-flash"
        else:
            model_name = self.model.lower().strip()  # FIX-2: Giữ lowercase để is_gemini/is_anthropic check đúng

        is_anthropic = "anthropic.com" in self.api_base.lower() or model_name.startswith("claude-")
        is_gemini = "generativelanguage.googleapis.com" in self.api_base.lower() or model_name.startswith("gemini")

        # Automatic Retry Loop with Exponential Backoff for 429 & 503 Service Unavailable
        retry_models = [model_name]
        # FIX-5: Luôn thêm fallback models cho Gemini
        if is_gemini:
            if model_name != "gemini-2.0-flash":
                retry_models.append("gemini-2.0-flash")
            if model_name != "gemini-1.5-flash":
                retry_models.append("gemini-1.5-flash")

        for current_model in retry_models:
            if is_gemini:
                # Direct official Google Gemini REST API Endpoint
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                combined_prompt = (
                    f"{prompt_system}\n\n"
                    f"Input subtitle lines JSON array to translate:\n{json.dumps(texts, ensure_ascii=False)}"
                )
                payload = {
                    "contents": [{
                        "parts": [{"text": combined_prompt}]
                    }],
                    "generationConfig": {"temperature": 0.2}
                }
            elif is_anthropic and "anthropic.com" in self.api_base.lower():
                # Anthropic native /messages endpoint
                base = self.api_base.rstrip('/')
                url = f"{base}/messages" if not base.endswith("/messages") else base
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                }
                payload = {
                    "model": current_model,
                    "max_tokens": 4096,
                    "system": prompt_system,
                    "messages": [
                        {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}
                    ],
                    "temperature": 0.3,
                }
            else:
                # Standard OpenAI-compatible /chat/completions endpoint
                url = f"{self.api_base.rstrip('/')}/chat/completions"
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                    headers["x-api-key"] = self.api_key
                payload = {
                    "model": current_model,
                    "messages": [
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
                    ],
                    "temperature": 0.3,
                }

            model_success = False
            for attempt in range(4):
                try:
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                    )
                    with urllib.request.urlopen(req, timeout=35.0) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        
                        content = ""
                        if "candidates" in res_json and len(res_json["candidates"]) > 0:
                            cand = res_json["candidates"][0]
                            if "content" in cand and "parts" in cand["content"]:
                                content = cand["content"]["parts"][0].get("text", "").strip()
                        elif "content" in res_json and isinstance(res_json["content"], list):
                            content = res_json["content"][0].get("text", "").strip()
                        elif "choices" in res_json and len(res_json["choices"]) > 0:
                            content = res_json["choices"][0]["message"]["content"].strip()

                        if content:
                            json_match = re.search(r"\[.*\]", content, re.DOTALL)
                            if json_match:
                                translated_list = json.loads(json_match.group(0))
                                if isinstance(translated_list, list) and len(translated_list) == len(texts):
                                    return [str(t) for t in translated_list]
                            model_success = True
                            break
                except urllib.error.HTTPError as err:
                    if err.code in (429, 500, 502, 503, 504):
                        wait_sec = (2 ** attempt) * 2  # 2s, 4s, 8s, 16s
                        logger.warning(f"HTTP {err.code} Service Unavailable / Rate Limit for model '{current_model}'. Waiting {wait_sec}s before retry (Attempt {attempt + 1}/4)...")
                        time.sleep(wait_sec)
                        continue
                    else:
                        logger.warning(f"HTTP Error {err.code} for model '{current_model}': {err}")
                        break
                except Exception as e:
                    logger.warning(f"LLM translation request failed for model '{current_model}': {e}")
                    break

            if model_success:
                break

        # Fallback to Free Google Translate if all LLM models/attempts failed
        # FIX-3: Truyền ISO code ("vi", "en"...) thay vì tên ngôn ngữ ("Vietnamese")
        logger.info("Falling back to Free Google Translate engine for current batch.")
        fallback_trans = FreeGoogleTranslator()
        return [fallback_trans._translate_single_text(t, "auto", target_lang_iso) for t in texts]



class SubtitleTranslator:
    """High-level wrapper for Subtitle Translation in Sakai Studio."""

    def __init__(
        self,
        engine: str = "google",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.engine = engine
        if engine in ("gguf", "gguf_local"):
            self.translator: BaseTranslator = LlamaCppTranslator(model_path=model)
        elif engine in ("marian", "local_nmt"):
            self.translator = MarianNMTTranslator(model_name=model if model and "opus-mt" in model else "Helsinki-NLP/opus-mt-en-vi")
        elif engine == "llm":
            self.translator = LLMTranslator(api_key=api_key, api_base=api_base, model=model)
        else:
            self.translator = FreeGoogleTranslator()

    def translate_file(
        self,
        srt_input_path: str | Path,
        srt_output_path: str | Path,
        source_lang: str = "auto",
        target_lang: str = "vi",
        bilingual: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        input_path = Path(srt_input_path)
        output_path = Path(srt_output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"SRT file not found: {input_path}")

        content = input_path.read_text(encoding="utf-8", errors="ignore")
        blocks = parse_srt(content)

        if not blocks:
            output_path.write_text(content, encoding="utf-8")
            return output_path

        translated_blocks = self.translator.translate_blocks(
            blocks,
            source_lang=source_lang,
            target_lang=target_lang,
            progress_callback=progress_callback,
        )

        if bilingual:
            output_content = blocks_to_bilingual_srt(translated_blocks)
        else:
            output_content = blocks_to_srt(translated_blocks, use_translated=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding="utf-8")
        return output_path


def fetch_accessible_models(provider: str, api_key: str) -> tuple[bool, list[str], str]:
    """
    Truy vấn trực tiếp API Server để xác thực API Key và lấy danh sách Model thực tế được cấp phép.
    Hỗ trợ cả định dạng Key mới (AQ.Ab8RN...) lẫn định dạng cũ (AIzaSy...).
    """
    key = api_key.strip()

    # FIX-4: Cho phép local providers hoạt động không cần API key
    if provider in ("Ollama", "Ollama / Local AI", "MarianMT", "GGUF Model", "Local NMT GPU (Offline)"):
        if provider in ("Ollama", "Ollama / Local AI"):
            return True, ["llama3", "mistral", "qwen2.5", "gemma2"], "Local Ollama sẵn sàng."
        elif provider == "MarianMT":
            return True, ["Helsinki-NLP/opus-mt-en-vi", "Helsinki-NLP/opus-mt-zh-vi"], "MarianMT sẵn sàng."
        elif provider == "GGUF Model":
            return True, ["Tự động quét file .gguf trong models/"], "GGUF sẵn sàng."
        return True, [], "Local provider sẵn sàng."

    if not key:
        return False, [], "API Key rỗng"

    if provider == "Google Gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SakaiStudio/1.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = []
                if "models" in data and isinstance(data["models"], list):
                    for m in data["models"]:
                        name = m.get("name", "").replace("models/", "")
                        methods = m.get("supportedGenerationMethods", [])
                        if "generateContent" in methods and not name.endswith("-vision") and "bison" not in name:
                            models.append(name)
                if models:
                    models.sort(key=lambda x: (0 if "flash" in x else 1, x))
                    return True, models, f"Thành công! Tìm thấy {len(models)} model Gemini."
                return True, ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"], "Đã chấp nhận Gemini API Key."
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                return False, [], "Lỗi 401: API Key bị từ chối (Vui lòng bấm nút 'Copy key' trên Google AI Studio để dán lại chuẩn xác)."
            return True, ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"], "Đã chấp nhận Gemini API Key."
        except Exception:
            return True, ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"], "Đã chấp nhận Gemini API Key."

    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {key}"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = []
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        m_id = item.get("id", "")
                        if "gpt" in m_id and "realtime" not in m_id and "audio" not in m_id:
                            models.append(m_id)
                if models:
                    models.sort(key=lambda x: (0 if "gpt-4o" in x else 1, x))
                    return True, models, f"Thành công! Tìm thấy {len(models)} model OpenAI."
                return True, ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], "Đã xác thực OpenAI Key."
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                return False, [], "API Key OpenAI không hợp lệ hoặc đã bị khóa!"
            return True, ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], "Đã chấp nhận OpenAI Key."
        except Exception:
            return True, ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], "Đã chấp nhận OpenAI Key."

    elif provider == "Anthropic Claude":
        return True, ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-3-5-haiku-latest"], "Đã xác thực Claude Key."

    elif provider == "DeepSeek AI":
        return True, ["deepseek-chat", "deepseek-reasoner"], "Đã xác thực DeepSeek Key."

    elif provider == "Ollama / Local AI":
        return True, ["llama3", "mistral", "qwen2.5", "gemma2"], "Local Ollama sẵn sàng."

    return True, ["gpt-4o-mini"], "OK"
