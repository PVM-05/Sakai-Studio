# Auto Voice Casting and Character Voice Locking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified character voice profiling and locking pipeline that guarantees a single, consistent voice per character from start to finish across the entire video.

**Architecture:** Use `CharacterVoiceProfile` as the single source of truth for character voices. Diarization clusters video audio by SRT timestamps using CAM++ ONNX, extracts a clean single-take reference sample for each character, creates and locks a `VoiceClonePrompt` (or fail-safe Vietnamese preset) on GPU RTX 4050, and exposes an interactive character profile table in the UI that synchronizes bidirectionally with the subtitle table.

**Tech Stack:** Python 3.12, PySide6 (Fluent UI widgets), ONNX Runtime GPU (CUDAExecutionProvider), PyTorch / Torchaudio, OmniVoice, FFmpeg.

## Global Constraints
- 100% standard Vietnamese for all UI copy, log messages, docstrings, and errors.
- Absolutely no English words in parentheses beside Vietnamese words.
- Absolutely no emojis or decorative icons anywhere in UI or console output.
- FluentIcon must be used for UI icons.
- Strict VRAM sequencing on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM): CAM++ session must be closed before OmniVoice TTS loads.

---

### Task 1: Dataclass `CharacterVoiceProfile` and Diarization Contract

**Files to modify:**
- `src/ai_engines/speaker_diarization.py`
- `src/ai_engines/tts_engine.py`
- `test/test_omnivoice_pipeline.py`

- [x] **Step 1: Write failing tests in `test/test_omnivoice_pipeline.py`**
  - Verify `CharacterVoiceProfile` structure and helper methods (`to_dict`, backward compatibility for dict access).
  - Verify `process_video_subtitles` returns profile objects containing `speaker_id`, `display_name`, `utterance_count`, `source_audio_path`, `method`, `preset_name`, and `status`.
- [x] **Step 2: Run test to confirm failure**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k "test_character_voice_profile" -v`
- [x] **Step 3: Implement `CharacterVoiceProfile` in `src/ai_engines/speaker_diarization.py`**
  - Define `@dataclass class CharacterVoiceProfile` with dict compatibility.
  - Update `process_video_subtitles` to populate and return `CharacterVoiceProfile` instances.
- [x] **Step 4: Update `TTSEngine.detect_speakers_from_video` in `src/ai_engines/tts_engine.py`**
  - Ensure returned profile dictionary properly contains `CharacterVoiceProfile` representations.
- [x] **Step 5: Run tests and ensure they pass**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -v`
- [x] **Step 6: Commit changes**
  - Command: `git add src/ai_engines/speaker_diarization.py src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py; git commit -m "feat(voice-casting): define CharacterVoiceProfile and update diarization contract"`

---

### Task 2: Character Voice Profile & Preset Fallback Locking in `tts_engine.py`

**Files to modify:**
- `src/ai_engines/tts_engine.py`
- `test/test_omnivoice_pipeline.py`

- [x] **Step 1: Write failing tests in `test/test_omnivoice_pipeline.py`**
  - Test pre-initialization and locking of prompts for each `CharacterVoiceProfile`.
  - Test automatic permanent fallback to locked Vietnamese preset if clone prompt creation fails.
- [x] **Step 2: Run test to confirm failure**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k "test_profile_prompt_pre_initialization_and_failover" -v`
- [x] **Step 3: Implement character prompt pre-caching and locking in `TTSEngine.synthesize_blocks`**
  - Before looping through subtitle blocks, iterate over unique characters / profiles, build the prompt, and store in `_speaker_prompt_cache`.
  - If prompt creation fails, lock the character's fallback preset and instruct for all remaining blocks.
- [x] **Step 4: Run tests and ensure they pass**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -v`
- [x] **Step 5: Commit changes**
  - Command: `git add src/ai_engines/tts_engine.py test/test_omnivoice_pipeline.py; git commit -m "feat(voice-casting): pre-initialize and lock character voice prompts with fallback preset"`

---

### Task 3: Interactive Character Voice Profile Table in `omnivoice_interface.py`

**Files to modify:**
- `src/desktop/ui/omnivoice_interface.py`
- `test/test_omnivoice_pipeline.py`

- [x] **Step 1: Write failing tests in `test/test_omnivoice_pipeline.py`**
  - Test changing character voice method/preset in casting table immediately updates all corresponding subtitle blocks.
  - Test selecting custom audio file updates `source_audio_path` and clears stale prompt.
  - Test renaming character in casting table updates subtitle blocks without corrupting voice profiles.
- [x] **Step 2: Run test to confirm failure**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -k "test_casting_table_profile_sync" -v`
- [x] **Step 3: Update `_populate_casting_table` and table column setup in `omnivoice_interface.py`**
  - Reconstruct columns:
    - 0: Mã nhân vật
    - 1: Tên nhân vật (editable)
    - 2: Số câu thoại (e.g. "12 câu (34.5s)")
    - 3: Mẫu giọng nguồn (with button "Chọn tệp âm thanh...")
    - 4: Phương pháp tạo giọng (ComboBox: "Nhân bản giọng từ video" + Vietnamese voice presets)
    - 5: Trạng thái ("Đã khóa", "Cần kiểm tra", "Chưa xác định")
    - 6: Thao tác (Button "Nghe thử")
- [x] **Step 4: Connect interactive handlers**
  - Connect ComboBox signal `currentIndexChanged` to update the character profile and sync all subtitle blocks assigned to this character.
  - Connect "Chọn tệp âm thanh..." button to open QFileDialog and update profile.
  - Connect "Nghe thử" button to preview either the reference audio or the preset sample.
  - Update `_on_subtitle_table_item_changed` to inherit new character profile on character reassignment.
- [x] **Step 5: Run tests and ensure they pass**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_omnivoice_pipeline.py -v`
- [x] **Step 6: Commit changes**
  - Command: `git add src/desktop/ui/omnivoice_interface.py test/test_omnivoice_pipeline.py; git commit -m "feat(ui): interactive character voice profile table with instant subtitle synchronization"`

---

### Task 4: Full Test Suite Verification and Git Push

**Files to verify:**
- All test files across the project

- [x] **Step 1: Run full test suite**
  - Command: `.\venv\Scripts\python.exe -m pytest test/test_auto_pipeline_reliability.py test/test_omnivoice_pipeline.py test/test_srt_translator.py test/test_subtitle_and_logo_remover.py test_home_interface.py -v`
- [x] **Step 2: Verify zero lint/syntax errors and design compliance**
  - Check no English in parentheses, no emojis, 100% standard Vietnamese.
- [x] **Step 3: Push commits to `origin main` and `origin feature/ui-redesign-xoa-phu-de`**
  - Command: `git push origin main`
  - Merge into `feature/ui-redesign-xoa-phu-de` and push.
