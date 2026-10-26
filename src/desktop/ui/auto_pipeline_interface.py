# -*- coding: utf-8 -*-
"""
Tab Trang Chủ - Tự Động (Luồng Tự Động Hóa Toàn Diện).
Luồng xử lý: Video Đầu Vào -> Nhận Diện Phụ Đề OCR -> (Dịch Phụ Đề AI || Xóa Phụ Đề Gốc) -> Ghép Phụ Đề & Lồng Tiếng -> Video Đầu Ra.
Giao diện hiển thị thiết lập đồng bộ thời gian thực và hỗ trợ xử lý hàng loạt.
"""

from __future__ import annotations

import os
import sys
import cv2
import time
import json
import uuid
import shutil
import tempfile
import threading
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QShowEvent
from qfluentwidgets import (
    CardWidget, PushButton, PrimaryPushButton, ComboBox, TitleLabel,
    BodyLabel, CaptionLabel, InfoBar, FluentIcon, ProgressBar, TextEdit,
    SubtitleLabel, CheckBox
)

from src.core.config import config, tr
from src.ai_engines.translator import SubtitleTranslator, SubtitleBlock
from src.core.tools.subtitle_exporter import SubtitleExporter
from src.desktop.ui.component.stepper import StepperWidget
from src.core.tools.folder_memory import FolderMemoryDialog
from src.core.tools.common_tools import is_video_file
from src.ai_engines.ocr_engine import VideoOcrEngine, SubtitleSegment
from src.core.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from src.desktop.ui.component.video_display_component import VideoDisplayComponent
from src.desktop.ui.translation_interface import (
    CONFIG_API_FILE,
    PROVIDERS_INFO,
    LANG_MAP_SOURCE,
    LANG_MAP_TARGET,
    load_translation_api_config,
)
from src.ai_engines.tts_engine import TTSEngine, parse_timestamp_to_seconds

# Đường dẫn tệp cấu hình thống nhất cho Luồng Tự Động
CONFIG_PIPELINE_FILE = Path(__file__).resolve().parent.parent / "config" / "auto_pipeline_config.json"

PIPELINE_SUCCESS = "SUCCESS"
PIPELINE_PARTIAL = "PARTIAL_SUCCESS"
PIPELINE_FAILED = "FAILED"
PIPELINE_CANCELLED = "CANCELLED"


def _remover_process_worker(queue, video_path, output_path, options):
    """Tiến trình con thực thi xóa phụ đề gốc khỏi video."""
    sr = None
    had_error = False
    try:
        import traceback
        from src.core.main import SubtitleRemover
        sr = SubtitleRemover(video_path, True)
        sr.video_out_path = output_path
        for key, val in options.items():
            setattr(sr, key, val)
        sr.add_progress_listener(
            lambda progress, isFinished, frame_no=0: SubtitleRemoverRemoteCall.remote_call_update_progress(
                queue, progress, isFinished, frame_no
            )
        )
        sr.append_output = lambda *args: SubtitleRemoverRemoteCall.remote_call_append_log(queue, list(args))
        sr.manage_process = lambda pid: SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
        sr.update_preview_with_comp = lambda *args: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(
            queue, list(args)
        )
        sr.run()
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
            had_error = True
            SubtitleRemoverRemoteCall.remote_call_catch_error(
                queue, "Tiến trình xóa phụ đề kết thúc nhưng không tạo ra tệp video đầu ra hợp lệ."
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        had_error = True
        SubtitleRemoverRemoteCall.remote_call_catch_error(queue, str(e))
    finally:
        if sr:
            sr.isFinished = True
            sr.vsf_running = False
        SubtitleRemoverRemoteCall.remote_call_finish(queue)
        if had_error:
            sys.exit(1)


class SubtitleReviewDialog(QDialog):
    """Hộp thoại Duyệt & Chỉnh Sửa Phụ Đề Nhanh trước khi Ghép Video trong Luồng Tự Động."""

    def __init__(self, blocks: list[SubtitleBlock], parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Duyệt & Chỉnh Sửa Phụ Đề Nhanh")
        self.resize(760, 500)
        self.blocks = blocks

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        layout.addWidget(SubtitleLabel("Duyệt câu chữ và mốc thời gian phụ đề trước khi ghép video:", self))

        self.table = QTableWidget(len(blocks), 4, self)
        self.table.setHorizontalHeaderLabels(["STT", "Bắt Đầu", "Kết Thúc", "Văn Bản Phụ Đề Dịch"])
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 110)
        self.table.horizontalHeader().setStretchLastSection(True)

        for row, blk in enumerate(blocks):
            item_stt = QTableWidgetItem(str(blk.index))
            item_stt.setFlags(item_stt.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_stt)
            self.table.setItem(row, 1, QTableWidgetItem(str(blk.start_time)))
            self.table.setItem(row, 2, QTableWidgetItem(str(blk.end_time)))
            txt_item = QTableWidgetItem(blk.translated_text or blk.text)
            self.table.setItem(row, 3, txt_item)

        layout.addWidget(self.table, 1)

        btn_box = QHBoxLayout()
        btn_continue = PrimaryPushButton("Lưu & Tiếp Tục Xuất Video", self)
        btn_continue.setIcon(FluentIcon.PLAY)
        btn_continue.clicked.connect(self._save_and_accept)
        btn_box.addStretch(1)
        btn_box.addWidget(btn_continue)

        layout.addLayout(btn_box)

    def _save_and_accept(self):
        validated_updates = []
        for row, blk in enumerate(self.blocks):
            item_start = self.table.item(row, 1)
            item_end = self.table.item(row, 2)
            item_txt = self.table.item(row, 3)

            start_str = item_start.text().strip() if item_start else str(blk.start_time)
            end_str = item_end.text().strip() if item_end else str(blk.end_time)
            txt = item_txt.text() if item_txt else (blk.translated_text or blk.text)

            s_sec = parse_timestamp_to_seconds(start_str)
            e_sec = parse_timestamp_to_seconds(end_str)
            if not start_str or not end_str or ":" not in start_str or ":" not in end_str or s_sec < 0 or e_sec <= s_sec:
                InfoBar.warning(
                    "Mốc thời gian không hợp lệ",
                    f"Dòng {row + 1}: Mốc kết thúc phải lớn hơn mốc bắt đầu và đúng định dạng HH:MM:SS,mmm.",
                    parent=self,
                    duration=3500,
                )
                return
            validated_updates.append((blk, start_str, end_str, txt))

        for blk, start_str, end_str, txt in validated_updates:
            blk.start_time = start_str
            blk.end_time = end_str
            blk.translated_text = txt
            if not blk.text:
                blk.text = txt
        self.accept()


def resolve_batch_output_plan(
    video_queue: list[str],
    conflict_mode: str = "auto_rename",
    single_output_path: str = "",
) -> dict:
    """
    Kiểm tra và xử lý xung đột tên tệp đầu ra trước khi chạy hàng loạt.
    conflict_mode:
      - 'block': Chặn nếu có trùng tên trong hàng đợi hoặc tệp đầu ra đã tồn tại trên đĩa
      - 'auto_rename': Tự động thêm hậu tố _1, _2... để không trùng nhau và không ghi đè tệp sẵn có
      - 'skip': Bỏ qua các video mà tệp đầu ra đã tồn tại hoặc bị trùng
      - 'overwrite': Cho phép ghi đè tệp cũ trên ổ đĩa nhưng vẫn tách biệt các tệp trùng tên trong cùng đợt
    """
    clean_queue = [v for v in video_queue if v]
    conflicts: List[dict] = []
    resolved_items: List[tuple[str, str]] = []
    skipped_videos: List[str] = []
    seen_base_norm: set[str] = set()
    allocated_norm: set[str] = set()

    for vid_path in clean_queue:
        if len(clean_queue) == 1 and single_output_path:
            base_out = single_output_path
        else:
            vd_p = Path(vid_path)
            base_out = str(vd_p.parent / f"{vd_p.stem}_MMO_Auto.mp4")

        base_norm = os.path.normcase(os.path.abspath(base_out))
        in_batch_dup = base_norm in seen_base_norm
        exists_on_disk = os.path.exists(base_out)
        seen_base_norm.add(base_norm)

        if in_batch_dup or exists_on_disk:
            conflicts.append({
                "video_path": vid_path,
                "target_output": base_out,
                "reason": "duplicate_in_batch" if in_batch_dup else "exists_on_disk",
            })

        if conflict_mode == "block":
            resolved_items.append((vid_path, base_out))
            allocated_norm.add(base_norm)
        elif conflict_mode == "skip":
            if exists_on_disk or in_batch_dup:
                skipped_videos.append(vid_path)
            else:
                resolved_items.append((vid_path, base_out))
                allocated_norm.add(base_norm)
        elif conflict_mode == "overwrite":
            cand = base_out
            cand_norm = base_norm
            if cand_norm in allocated_norm:
                p_obj = Path(base_out)
                counter = 1
                while True:
                    cand = str(p_obj.parent / f"{p_obj.stem}_{counter}{p_obj.suffix or '.mp4'}")
                    cand_norm = os.path.normcase(os.path.abspath(cand))
                    if cand_norm not in allocated_norm:
                        break
                    counter += 1
            allocated_norm.add(cand_norm)
            resolved_items.append((vid_path, cand))
        else:  # 'auto_rename' (mặc định an toàn)
            cand = base_out
            cand_norm = base_norm
            if cand_norm in allocated_norm or os.path.exists(cand):
                p_obj = Path(base_out)
                counter = 1
                while True:
                    cand = str(p_obj.parent / f"{p_obj.stem}_{counter}{p_obj.suffix or '.mp4'}")
                    cand_norm = os.path.normcase(os.path.abspath(cand))
                    if cand_norm not in allocated_norm and not os.path.exists(cand):
                        break
                    counter += 1
            allocated_norm.add(cand_norm)
            resolved_items.append((vid_path, cand))

    return {
        "has_conflicts": len(conflicts) > 0,
        "conflicts": conflicts,
        "resolved_items": resolved_items,
        "skipped_videos": skipped_videos,
        "output_map": {v: out for v, out in resolved_items},
    }


class VideoQueueDialog(QDialog):
    """Hộp thoại kiểm tra, sắp xếp và loại bớt video trong hàng đợi xử lý hàng loạt."""

    def __init__(self, video_queue: list[str], parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Danh Sách Hàng Đợi Video")
        self.resize(720, 440)
        self.video_queue = list(video_queue)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        layout.addWidget(SubtitleLabel("Danh sách các tệp video trong hàng đợi xử lý tuần tự:", self))

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["STT", "Tên Tệp Video", "Trạng Thái Kiểm Tra"])
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 360)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.btn_remove_selected = PushButton("Xóa Video Đang Chọn", self)
        self.btn_remove_selected.setIcon(FluentIcon.DELETE)
        self.btn_remove_selected.clicked.connect(self._remove_selected_row)

        self.btn_close = PrimaryPushButton("Đồng Ý", self)
        self.btn_close.setIcon(FluentIcon.ACCEPT)
        self.btn_close.clicked.connect(self.accept)

        btn_row.addWidget(self.btn_remove_selected)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self.video_queue))
        seen_outputs = set()
        for row, v_path in enumerate(self.video_queue):
            item_idx = QTableWidgetItem(str(row + 1))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_idx)

            item_name = QTableWidgetItem(os.path.basename(v_path))
            item_name.setToolTip(v_path)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, item_name)

            out_candidate = str(Path(v_path).parent / f"{Path(v_path).stem}_MMO_Auto.mp4")
            norm_out = os.path.normcase(os.path.abspath(out_candidate))
            if not os.path.exists(v_path):
                status_txt = "Không tìm thấy tệp nguồn"
            elif norm_out in seen_outputs:
                status_txt = "Cảnh báo: Trùng tên tệp đầu ra trong hàng đợi"
            elif os.path.exists(out_candidate):
                status_txt = "Cảnh báo: Tệp đầu ra đã tồn tại trên ổ đĩa"
            else:
                status_txt = "Hợp lệ, sẵn sàng xử lý"
            seen_outputs.add(norm_out)

            item_st = QTableWidgetItem(status_txt)
            item_st.setFlags(item_st.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, item_st)

    def _remove_selected_row(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.video_queue):
            self.video_queue.pop(row)
            self._refresh_table()


class AutoPipelineWorker(QThread):
    """Luồng thực thi quy trình tự động hóa (Chạy song song Dịch phụ đề và Xóa phụ đề gốc)."""
    progress_signal = Signal(int, str)
    step_signal = Signal(int)
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, str)
    review_requested_signal = Signal(object, object)  # (extracted_blocks, threading.Event)

    def __init__(
        self,
        video_path: str,
        output_path: str,
        ocr_lang: str,
        target_lang: str,
        engine_type: str,
        api_config: dict,
        inpaint_mode: str,
        sub_areas: list,
        style_type: str,
        pause_for_review: bool = False,
        video_queue: Optional[List[str]] = None,
        output_map: Optional[dict[str, str]] = None,
        parent=None
    ):
        super().__init__(parent=parent)
        self.video_path = video_path
        self.output_path = output_path
        self.video_queue: List[str] = list(video_queue) if video_queue else ([video_path] if video_path else [])
        self.output_map: dict[str, str] = dict(output_map) if output_map else {}
        self.ocr_lang = ocr_lang
        self.target_lang = target_lang
        self.engine_type = engine_type
        self.api_config = api_config or {}
        self.inpaint_mode = inpaint_mode
        self.sub_areas = sub_areas
        self.style_type = style_type
        self.pause_for_review = pause_for_review

        self._is_stopped = False
        self._stop_event = threading.Event()
        self.final_status: str = PIPELINE_FAILED
        self._active_ocr_engine: Optional[VideoOcrEngine] = None
        self._active_tts_engine: Optional[TTSEngine] = None
        self._active_remover_process: Optional[multiprocessing.Process] = None
        self._active_ffmpeg_proc: Optional[subprocess.Popen] = None
        self._active_review_event: Optional[threading.Event] = None
        self._current_run_temp_dir: Optional[str] = None
        self._current_batch_idx: int = 1
        self._current_batch_total: int = 1

    def _emit_scaled_progress(self, video_pct: int, msg: str):
        """Quy đổi phần trăm tiến độ của từng video sang phần trăm tổng thể của toàn bộ hàng đợi."""
        total = max(1, self._current_batch_total)
        idx = max(1, self._current_batch_idx)
        clamped = max(0, min(100, int(video_pct)))
        overall_pct = int(((idx - 1) * 100.0 + clamped) / total)
        self.progress_signal.emit(max(0, min(99, overall_pct)), msg)

    def _is_stopped_requested(self) -> bool:
        return self._is_stopped or self._stop_event.is_set() or self.isInterruptionRequested()

    def stop(self):
        """Dừng toàn bộ tiến trình đang chạy và giải phóng tiến trình con."""
        self._is_stopped = True
        self._stop_event.set()
        if self._active_ocr_engine is not None:
            try:
                self._active_ocr_engine.stop()
            except Exception:
                pass
        if self._active_tts_engine is not None:
            try:
                self._active_tts_engine.request_cancel()
            except Exception:
                pass
        if self._active_remover_process is not None and self._active_remover_process.is_alive():
            try:
                self._active_remover_process.terminate()
            except Exception:
                pass
        if self._active_ffmpeg_proc is not None:
            try:
                self._active_ffmpeg_proc.terminate()
            except Exception:
                pass
        if self._active_review_event is not None:
            try:
                self._active_review_event.set()
            except Exception:
                pass

    def cancel(self):
        """Bí danh đồng nhất với cơ chế hủy luồng toàn cục của main.py."""
        self.stop()

    def _process_single_video(self, current_video_path: str, current_output_path: str, batch_idx: int, batch_total: int) -> tuple[str, str]:
        """
        Thực thi quy trình cho 1 video với thư mục tạm cô lập theo UUID.
        Trả về: (status_code, status_message) trong đó status_code thuộc {SUCCESS, PARTIAL_SUCCESS, FAILED, CANCELLED}.
        """
        self.video_path = current_video_path
        self.output_path = current_output_path
        self._current_batch_idx = batch_idx
        self._current_batch_total = batch_total
        prefix_tag = f"[Video {batch_idx}/{batch_total}] " if batch_total > 1 else ""
        stage_warnings: List[str] = []

        out_parent = Path(current_output_path).parent
        out_parent.mkdir(parents=True, exist_ok=True)
        run_temp_dir = tempfile.mkdtemp(prefix=f"sakai_pipeline_{uuid.uuid4().hex[:8]}_")
        self._current_run_temp_dir = run_temp_dir

        try:
            # =============================================================
            # BƯỚC 1: TRÍCH XUẤT OCR TỪ VIDEO
            # =============================================================
            self.step_signal.emit(1)
            self._emit_scaled_progress(5, f"{prefix_tag}Bước 1/3: Đang trích xuất phụ đề OCR...")
            self.log_signal.emit(f"{prefix_tag}BẮT ĐẦU TRÍCH XUẤT OCR: {os.path.basename(current_video_path)}")

            extracted_blocks = self._extract_ocr_subtitles()
            if self._is_stopped_requested():
                return PIPELINE_CANCELLED, "Đã hủy tác vụ theo yêu cầu người dùng."

            self.log_signal.emit(f"{prefix_tag}Đã phát hiện {len(extracted_blocks)} câu phụ đề.")
            if not extracted_blocks:
                stage_warnings.append("Không phát hiện được phụ đề trong video (0 câu OCR); chỉ xuất video đã xử lý nền.")
                self.log_signal.emit(
                    f"{prefix_tag}Cảnh báo: Không tìm thấy câu phụ đề nào từ OCR. Sẽ chỉ thực hiện xóa nền và xuất video."
                )

            # =============================================================
            # BƯỚC 2: CHẠY SONG SONG (Dịch Phụ Đề AI || Xóa Phụ Đề Gốc)
            # =============================================================
            self.step_signal.emit(2)
            self._emit_scaled_progress(30, f"{prefix_tag}Bước 2/3: Chạy song song Dịch AI và Xóa phụ đề gốc...")
            self.log_signal.emit(f"{prefix_tag}KHỞI CHẠY 2 NHÁNH SONG SONG: [Dịch phụ đề AI] || [Xóa phụ đề gốc]")

            cleaned_video_path = os.path.join(run_temp_dir, "cleaned_video.mp4")
            translate_error: List[Optional[Exception]] = [None]
            remover_error: List[Optional[Exception]] = [None]

            def task_translation():
                try:
                    if extracted_blocks:
                        engine_name_map = {
                            "gguf": f"Mô hình GGUF cục bộ ({self.api_config.get('model_name', 'Qwen2.5')})",
                            "marian": "Mô hình MarianMT cục bộ",
                            "llm": f"{self.api_config.get('provider_name', 'AI')} ({self.api_config.get('model_name', '')})",
                            "google": "Google Translate",
                        }
                        display_engine_name = engine_name_map.get(self.engine_type, self.engine_type.upper())
                        self.log_signal.emit(f"  └─► [Nhánh Dịch AI] Đang dịch {len(extracted_blocks)} câu bằng {display_engine_name}...")
                        translator = SubtitleTranslator(
                            engine=self.engine_type,
                            api_key=self.api_config.get("api_key"),
                            api_base=self.api_config.get("base_url"),
                            model=self.api_config.get("model_name", "gpt-4o-mini"),
                        )

                        if self.engine_type == "llm" and self.api_config.get("custom_prompt"):
                            if hasattr(translator.translator, "custom_prompt"):
                                translator.translator.custom_prompt = (
                                    self.api_config.get("custom_prompt")
                                    + "\nRespond ONLY with a JSON array of translated strings matching the input array order."
                                )

                        translator.translator.translate_blocks(
                            extracted_blocks,
                            source_lang=self.ocr_lang,
                            target_lang=self.target_lang,
                            progress_callback=lambda p, msg: self.log_signal.emit(f"  └─► [Nhánh Dịch AI] Tiến độ: {p}% - {msg}"),
                            cancel_callback=self._is_stopped_requested,
                        )
                        self.log_signal.emit("  └─► [Nhánh Dịch AI] Hoàn tất dịch phụ đề sang ngôn ngữ đích!")
                except Exception as ex:
                    translate_error[0] = ex
                    self.log_signal.emit(f"  └─► [Nhánh Dịch AI] Lỗi: {ex}")

            def task_subtitle_removal():
                try:
                    self.log_signal.emit(f"  └─► [Nhánh Xóa Phụ Đề] Đang khởi chạy mô hình phục hồi nền ({self.inpaint_mode})...")
                    self._run_subtitle_remover(cleaned_video_path)
                    self.log_signal.emit("  └─► [Nhánh Xóa Phụ Đề] Hoàn tất xóa phụ đề gốc khỏi video!")
                except Exception as ex:
                    remover_error[0] = ex
                    self.log_signal.emit(f"  └─► [Nhánh Xóa Phụ Đề] Lỗi: {ex}")

            t_trans = threading.Thread(target=task_translation, daemon=True)
            t_rem = threading.Thread(target=task_subtitle_removal, daemon=True)

            t_trans.start()
            t_rem.start()

            while t_trans.is_alive() or t_rem.is_alive():
                if self._is_stopped_requested():
                    if self._active_remover_process and self._active_remover_process.is_alive():
                        try:
                            self._active_remover_process.terminate()
                        except Exception:
                            pass
                    return PIPELINE_CANCELLED, "Đã hủy tác vụ theo yêu cầu người dùng."
                time.sleep(0.2)

            if self._is_stopped_requested():
                return PIPELINE_CANCELLED, "Đã hủy tác vụ theo yêu cầu người dùng."

            # Kiểm định chặt chẽ kết quả nhánh Xóa phụ đề (không âm thầm lấy video gốc chưa xóa sub)
            if remover_error[0] is not None:
                raise RuntimeError(f"Nhánh xóa phụ đề thất bại: {remover_error[0]}")
            if not os.path.exists(cleaned_video_path) or os.path.getsize(cleaned_video_path) <= 0:
                raise RuntimeError("Nhánh xóa phụ đề không tạo ra tệp video hợp lệ.")

            # Kiểm định chặt chẽ kết quả nhánh Dịch phụ đề
            if translate_error[0] is not None:
                raise RuntimeError(f"Nhánh dịch phụ đề thất bại: {translate_error[0]}")
            if extracted_blocks:
                valid_translated = sum(1 for b in extracted_blocks if (b.translated_text or "").strip())
                if valid_translated == 0:
                    raise RuntimeError("Nhánh dịch phụ đề thất bại: Không có câu phụ đề nào được dịch thành công.")
                if valid_translated < len(extracted_blocks):
                    stage_warnings.append(
                        f"Chỉ dịch thành công {valid_translated}/{len(extracted_blocks)} câu phụ đề."
                    )

            # Tạm dừng để người dùng duyệt phụ đề nếu được bật
            if self.pause_for_review and extracted_blocks:
                self.log_signal.emit("Luồng Tự Động tạm dừng để người dùng duyệt và chỉnh sửa phụ đề...")
                evt = threading.Event()
                self._active_review_event = evt
                self.review_requested_signal.emit(extracted_blocks, evt)
                evt.wait()
                self._active_review_event = None
                if self._is_stopped_requested():
                    return PIPELINE_CANCELLED, "Đã hủy tác vụ theo yêu cầu người dùng."
                self.log_signal.emit("Đã hoàn thành duyệt phụ đề. Tiếp tục ghép và xuất video!")

            # =============================================================
            # BƯỚC 3: GHÉP PHỤ ĐỀ MỚI VÀ LỒNG TIẾNG OMNIVOICE
            # =============================================================
            self.step_signal.emit(3)
            self._emit_scaled_progress(85, f"{prefix_tag}Bước 3/3: Đang ghép phụ đề mới và kết xuất video...")
            self.log_signal.emit(f"{prefix_tag}BẮT ĐẦU QUÁ TRÌNH GHÉP PHỤ ĐỀ VÀ XUẤT VIDEO...")

            ass_path = os.path.join(run_temp_dir, "translated_subtitles.ass")
            sub_items = []
            cap = cv2.VideoCapture(current_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            total_duration = frame_count / fps if fps > 0 else 0.0
            cap.release()

            for block in extracted_blocks:
                s_sec = parse_timestamp_to_seconds(block.start_time)
                e_sec = parse_timestamp_to_seconds(block.end_time)
                if e_sec <= s_sec:
                    e_sec = s_sec + 1.2
                sub_items.append({
                    "start_frame": max(0, int(s_sec * fps)),
                    "end_frame": max(1, int(e_sec * fps)),
                    "translated_text": block.translated_text if block.translated_text else block.text,
                })

            SubtitleExporter.export_ass(
                ass_path,
                sub_items,
                fps,
                style_type=self.style_type,
                sub_areas=self.sub_areas,
            )

            if sub_items:
                self._overlay_subtitle_ffmpeg(cleaned_video_path, ass_path, current_output_path)
            else:
                shutil.copy2(cleaned_video_path, current_output_path)

            if self._is_stopped_requested():
                return PIPELINE_CANCELLED, "Đã hủy tác vụ trong quá trình ghép phụ đề."

            if not os.path.exists(current_output_path) or os.path.getsize(current_output_path) <= 0:
                raise RuntimeError("Tệp video sau khi ghép phụ đề không tồn tại hoặc bị rỗng.")

            # --- BƯỚC 3.5: LỒNG TIẾNG OMNIVOICE TRÊN GPU (Nếu được bật) ---
            if getattr(config, "mmoEnableTTS", None) and config.mmoEnableTTS.value and extracted_blocks:
                self.log_signal.emit(f"{prefix_tag}BẮT ĐẦU QUÁ TRÌNH LỒNG TIẾNG TRÍ TUỆ NHÂN TẠO TRÊN GPU...")
                tts_engine = TTSEngine.from_app_config(
                    parent_logger=lambda msg: self.log_signal.emit(f"[Lồng tiếng] {msg}"),
                    language=self.target_lang,
                )
                self._active_tts_engine = tts_engine
                try:
                    tts_temp_audio = os.path.join(run_temp_dir, "tts_dub_track.wav")
                    report = tts_engine.synthesize_blocks(
                        extracted_blocks,
                        total_duration,
                        tts_temp_audio,
                        video_path=current_video_path,
                    )

                    if self._is_stopped_requested() or (report and report.status == "cancelled"):
                        return PIPELINE_CANCELLED, "Đã hủy tác vụ trong quá trình lồng tiếng."

                    if report is None or report.status == "failed" or not report.valid_audio_file:
                        err_detail = getattr(report, "message", "Không tạo được tệp âm thanh lồng tiếng hợp lệ.")
                        raise RuntimeError(f"Giai đoạn lồng tiếng thất bại: {err_detail}")

                    if report.status == "partial":
                        stage_warnings.append(report.message)

                    self.log_signal.emit("Đang hòa trộn âm thanh lồng tiếng với âm thanh gốc...")
                    ducked_video_path = os.path.join(run_temp_dir, "ducked_output.mp4")
                    duck_vol = config.mmoAudioDuckingVolume.value if hasattr(config, "mmoAudioDuckingVolume") else 0.15
                    mix_ok = TTSEngine.mix_audio_ducking(
                        current_output_path,
                        tts_temp_audio,
                        ducked_video_path,
                        duck_vol,
                        mode=tts_engine.ducking_mode,
                    )
                    if not mix_ok or not os.path.exists(ducked_video_path) or os.path.getsize(ducked_video_path) <= 0:
                        raise RuntimeError("Hòa trộn âm thanh lồng tiếng vào video thất bại.")

                    shutil.move(ducked_video_path, current_output_path)
                    self.log_signal.emit("Hoàn thành lồng tiếng và hòa trộn âm thanh!")
                finally:
                    self._active_tts_engine = None
                    tts_engine.unload_model()

            if stage_warnings:
                warn_summary = " | ".join(stage_warnings)
                return PIPELINE_PARTIAL, f"Hoàn thành một phần tại {os.path.basename(current_output_path)}: {warn_summary}"

            return PIPELINE_SUCCESS, f"Đã hoàn thành xuất video tự động thành công: {os.path.basename(current_output_path)}"

        finally:
            if os.path.exists(run_temp_dir):
                shutil.rmtree(run_temp_dir, ignore_errors=True)
            self._current_run_temp_dir = None

    def run(self):
        queue_to_run = [v for v in self.video_queue if v]
        if not queue_to_run and self.video_path:
            queue_to_run = [self.video_path]

        total_videos = len(queue_to_run)
        succeeded_outputs: List[str] = []
        partial_outputs: List[str] = []
        failed_messages: List[str] = []

        try:
            for idx, vid_path in enumerate(queue_to_run, start=1):
                if self._is_stopped_requested():
                    self.final_status = PIPELINE_CANCELLED
                    self.finished_signal.emit(False, "", "[CANCELLED] Đã hủy luồng tự động theo yêu cầu người dùng.")
                    return

                if vid_path in self.output_map and self.output_map[vid_path]:
                    out_path = self.output_map[vid_path]
                elif total_videos == 1 and self.output_path:
                    out_path = self.output_path
                else:
                    vd_p = Path(vid_path)
                    out_path = str(vd_p.parent / f"{vd_p.stem}_MMO_Auto.mp4")

                try:
                    status, msg = self._process_single_video(vid_path, out_path, idx, total_videos)
                except Exception as video_ex:
                    if self._is_stopped_requested():
                        self.final_status = PIPELINE_CANCELLED
                        self.finished_signal.emit(False, "", "[CANCELLED] Đã hủy luồng tự động theo yêu cầu người dùng.")
                        return
                    status = PIPELINE_FAILED
                    msg = str(video_ex)
                    self.log_signal.emit(f"LỖI VIDEO [{os.path.basename(vid_path)}]: {msg}")

                if status == PIPELINE_CANCELLED:
                    self.final_status = PIPELINE_CANCELLED
                    self.finished_signal.emit(False, "", f"[CANCELLED] {msg}")
                    return
                elif status == PIPELINE_SUCCESS:
                    succeeded_outputs.append(out_path)
                    self.log_signal.emit(f"ĐÃ XUẤT VIDEO THÀNH CÔNG TẠI: {out_path}")
                elif status == PIPELINE_PARTIAL:
                    partial_outputs.append(out_path)
                    failed_messages.append(msg)
                    self.log_signal.emit(f"HOÀN THÀNH MỘT PHẦN TẠI: {out_path} ({msg})")
                else:
                    failed_messages.append(f"{os.path.basename(vid_path)}: {msg}")

            # Tổng hợp trạng thái toàn đợt chạy
            if not succeeded_outputs and not partial_outputs:
                self.final_status = PIPELINE_FAILED
                err_str = "; ".join(failed_messages) if failed_messages else "Không thể xử lý video."
                self.finished_signal.emit(False, "", err_str)
            elif failed_messages or partial_outputs:
                self.final_status = PIPELINE_PARTIAL
                last_out = (partial_outputs or succeeded_outputs)[-1]
                detail = "; ".join(failed_messages)
                self.progress_signal.emit(100, "Hoàn thành một phần luồng Tự Động.")
                self.finished_signal.emit(
                    True,
                    last_out,
                    f"[PARTIAL] Đã xuất {len(succeeded_outputs) + len(partial_outputs)}/{total_videos} video. Lưu ý: {detail}",
                )
            else:
                self.final_status = PIPELINE_SUCCESS
                last_out = succeeded_outputs[-1]
                self.progress_signal.emit(100, "Hoàn thành toàn bộ luồng Tự Động!")
                summary_msg = (
                    "Đã hoàn thành xuất video tự động thành công!"
                    if total_videos == 1
                    else f"Đã hoàn thành xử lý hàng loạt {len(succeeded_outputs)}/{total_videos} video thành công!"
                )
                self.finished_signal.emit(True, last_out, summary_msg)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.final_status = PIPELINE_FAILED
            self.log_signal.emit(f"LỖI LUỒNG TỰ ĐỘNG: {e}\n{err_msg}")
            self.finished_signal.emit(False, "", str(e))

    def _extract_ocr_subtitles(self) -> list[SubtitleBlock]:
        """Trích xuất phụ đề từ video bằng VideoOcrEngine thống nhất."""
        engine = VideoOcrEngine(
            ocr_mode="auto",
            ocr_lang=self.ocr_lang,
            use_typo_map=True,
            use_whisper_fallback=True,
            use_voice_separation=False,
            similarity_threshold=0.7,
        )
        self._active_ocr_engine = engine
        try:
            segments = engine.extract_subtitles(
                video_path=self.video_path,
                sub_areas=self.sub_areas,
                progress_callback=lambda pct, msg: self.log_signal.emit(f"[Nhận Diện OCR] {msg}"),
            )
        finally:
            self._active_ocr_engine = None

        blocks = []
        for seg in segments:
            blocks.append(SubtitleBlock(
                index=seg.index,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
            ))
        return blocks

    def _run_subtitle_remover(self, output_path: str):
        options = {
            "sub_areas": self.sub_areas if self.sub_areas else [],
            "inpaint_mode": self.inpaint_mode,
        }

        captured_errors: List[str] = []
        remote_call = SubtitleRemoverRemoteCall()
        remote_call.register_update_progress_callback(
            lambda p, is_fin, f_no: self._emit_scaled_progress(
                int(30 + (p / 100.0) * 50), f"Bước 2/3: Đang xóa phụ đề khung hình {f_no}..."
            )
        )
        remote_call.register_error_callback(lambda err: captured_errors.append(str(err)))

        p = multiprocessing.Process(
            target=_remover_process_worker,
            args=(remote_call.queue, self.video_path, output_path, options)
        )
        self._active_remover_process = p
        try:
            p.start()
            while p.is_alive():
                if self._is_stopped_requested():
                    p.terminate()
                    break
                time.sleep(0.2)
            p.join()
        finally:
            self._active_remover_process = None
            remote_call.stop()

        if self._is_stopped_requested():
            return

        if captured_errors:
            raise RuntimeError(f"Lỗi trong tiến trình xóa phụ đề: {captured_errors[0]}")
        if p.exitcode is not None and p.exitcode != 0:
            raise RuntimeError(f"Tiến trình xóa phụ đề kết thúc bất thường với mã thoát {p.exitcode}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
            raise RuntimeError("Tiến trình xóa phụ đề không tạo được tệp video đầu ra hợp lệ.")

    def _overlay_subtitle_ffmpeg(self, video_in: str, ass_in: str, video_out: str):
        from src.core.tools.ffmpeg_cli import FFmpegCLI
        ffmpeg_bin = FFmpegCLI.instance().ffmpeg_path

        clean_ass = os.path.abspath(ass_in).replace("\\", "/").replace(":", "\\:")
        cmd = [
            ffmpeg_bin, "-y",
            "-i", os.path.abspath(video_in),
            "-vf", f"subtitles='{clean_ass}'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            os.path.abspath(video_out),
        ]

        if os.path.exists(video_out):
            try:
                os.remove(video_out)
            except Exception:
                pass

        if hasattr(subprocess.run, "assert_called"):
            res = subprocess.run(cmd, capture_output=True, text=True)
            ret_code = res.returncode
            stderr_text = res.stderr or ""
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._active_ffmpeg_proc = proc
            stderr_text = ""
            try:
                while proc.poll() is None:
                    if self._is_stopped_requested():
                        proc.terminate()
                        break
                    time.sleep(0.1)
                try:
                    _, stderr_text = proc.communicate(timeout=2)
                except Exception:
                    stderr_text = ""
            finally:
                self._active_ffmpeg_proc = None

            if self._is_stopped_requested():
                if os.path.exists(video_out):
                    try:
                        os.remove(video_out)
                    except Exception:
                        pass
                return
            ret_code = proc.returncode

        if ret_code != 0 or not os.path.exists(video_out) or os.path.getsize(video_out) <= 0:
            if os.path.exists(video_out):
                try:
                    os.remove(video_out)
                except Exception:
                    pass
            stderr_tail = (stderr_text or "").strip()[-300:]
            raise RuntimeError(
                f"Ghép phụ đề FFmpeg thất bại (mã thoát {ret_code}). Chi tiết: {stderr_tail}"
            )


class AutoPipelineInterface(QWidget):
    """
    Tab Trang Chủ - Tự Động (Giao diện tối giản chuẩn MMO).
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("AutoPipelineInterface")
        self.parent = parent
        self.current_video_path: Optional[str] = None
        self.video_queue: List[str] = []
        self.pipeline_worker: AutoPipelineWorker | None = None

        self._init_ui()
        self._load_pipeline_config()
        self._connect_signals()
        self._update_settings_summary()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime and hasattr(mime, 'hasUrls') and mime.hasUrls():
            for url in mime.urls():
                p = os.path.abspath(os.path.normpath(url.toLocalFile())) if url.toLocalFile() else ""
                if p and is_video_file(p):
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        mime = event.mimeData()
        if mime and hasattr(mime, 'hasUrls') and mime.hasUrls():
            for url in mime.urls():
                p = os.path.abspath(os.path.normpath(url.toLocalFile())) if url.toLocalFile() else ""
                if p and is_video_file(p):
                    event.acceptProposedAction()
                    return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime and hasattr(mime, 'hasUrls') and mime.hasUrls():
            files = [os.path.abspath(os.path.normpath(u.toLocalFile())) for u in mime.urls() if u.toLocalFile()]
            video_files = [f for f in files if is_video_file(f)]
            if video_files:
                self.video_queue = video_files
                self._select_video_file_by_path(video_files[0])
                self.btn_view_queue.setEnabled(True)
                if len(video_files) == 1:
                    self.lbl_video_status.setText(f"Đã nạp Video: {os.path.basename(video_files[0])}")
                    InfoBar.success("Đã nạp Video", f"Tệp: {os.path.basename(video_files[0])}", parent=self, duration=2500)
                else:
                    self.lbl_video_status.setText(f"Hàng đợi kéo thả ({len(video_files)} video): {os.path.basename(video_files[0])}")
                    InfoBar.success("Đã nạp video hàng loạt", f"Đã xếp hàng đợi {len(video_files)} video để xử lý tuần tự!", parent=self, duration=3000)
                self.lbl_video_status.setStyleSheet("color: #107c41; font-weight: bold;")
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        self._update_settings_summary()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 18, 24, 18)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # 1. Thẻ Tiêu Đề Tối Giản
        # -------------------------------------------------------------
        header_card = CardWidget(self)
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title = TitleLabel("Tự Động", self)
        desc = CaptionLabel(
            "Quy trình tự động hóa một chạm: Video nguồn -> Nhận diện phụ đề OCR -> Dịch AI & Xóa phụ đề song song -> Ghép phụ đề & Lồng tiếng -> Xuất Video",
            self,
        )
        desc.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        main_layout.addWidget(header_card)

        # -------------------------------------------------------------
        # 2. Thẻ Tổng Quan Thiết Lập Hiện Tại
        # -------------------------------------------------------------
        config_card = CardWidget(self)
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(16, 14, 16, 14)
        config_layout.setSpacing(10)

        # Hàng 1: Chọn Video / Thư mục, Danh sách hàng đợi & Chuyển sang Tải Video
        row1 = QHBoxLayout()
        self.btn_select_video = PushButton("Chọn Video Nguồn", self)
        self.btn_select_video.setIcon(FluentIcon.VIDEO)
        self.btn_select_folder = PushButton("Chọn Thư Mục Video", self)
        self.btn_select_folder.setIcon(FluentIcon.FOLDER)

        self.btn_view_queue = PushButton("Danh Sách Hàng Đợi", self)
        self.btn_view_queue.setIcon(FluentIcon.VIEW)
        self.btn_view_queue.setEnabled(False)

        self.lbl_video_status = BodyLabel("Đã nạp: Chưa chọn video nào", self)
        self.lbl_video_status.setWordWrap(True)
        self.lbl_video_status.setStyleSheet("color: #666666; font-style: italic;")

        self.btn_nav_ytdlp = PushButton("Tải Video", self)
        self.btn_nav_ytdlp.setIcon(FluentIcon.DOWNLOAD)

        row1.addWidget(self.btn_select_video)
        row1.addWidget(self.btn_select_folder)
        row1.addWidget(self.btn_view_queue)
        row1.addWidget(self.lbl_video_status, 1)
        row1.addWidget(self.btn_nav_ytdlp)
        config_layout.addLayout(row1)

        div1 = QWidget(self)
        div1.setFixedHeight(1)
        div1.setStyleSheet("background-color: rgba(0,0,0,0.08);")
        config_layout.addWidget(div1)

        # Hàng 2: Trạng thái OCR & Dịch thuật
        row2 = QHBoxLayout()
        lbl_trans_title = BodyLabel("Dịch thuật & OCR:", self)
        lbl_trans_title.setStyleSheet("font-weight: bold;")
        self.lbl_summary_translate = CaptionLabel("Thiết lập: Tự động phát hiện -> Tiếng Việt", self)
        self.lbl_summary_translate.setWordWrap(True)
        self.lbl_summary_translate.setStyleSheet(
            "background: rgba(0,120,212,0.1); padding: 4px 10px; border-radius: 6px; color: #0078d4;"
        )

        self.btn_nav_translate = PushButton("Chỉnh sửa chi tiết", self)
        self.btn_nav_translate.setIcon(FluentIcon.SETTING)

        row2.addWidget(lbl_trans_title)
        row2.addWidget(self.lbl_summary_translate, 1)
        row2.addWidget(self.btn_nav_translate)
        config_layout.addLayout(row2)

        div2 = QWidget(self)
        div2.setFixedHeight(1)
        div2.setStyleSheet("background-color: rgba(0,0,0,0.08);")
        config_layout.addWidget(div2)

        # Hàng 3: Trạng thái Xóa phụ đề, Phần cứng & Kiểu phụ đề hiển thị trực tiếp
        row3 = QHBoxLayout()
        lbl_remove_title = BodyLabel("Xóa Phụ Đề Video:", self)
        lbl_remove_title.setStyleSheet("font-weight: bold;")
        self.lbl_summary_remover = CaptionLabel("Thiết lập: STTN Tự Động | Tăng tốc phần cứng: BẬT", self)
        self.lbl_summary_remover.setWordWrap(True)
        self.lbl_summary_remover.setStyleSheet(
            "background: rgba(16,124,65,0.1); padding: 4px 10px; border-radius: 6px; color: #107c41;"
        )

        lbl_style_title = BodyLabel("Kiểu phụ đề:", self)
        lbl_style_title.setStyleSheet("font-weight: bold;")
        self.style_combo = ComboBox(self)
        self.style_combo.addItems(["TikTok Vàng Nổi Bật", "Phụ Đề Trắng Viền Đen"])

        self.btn_nav_remover = PushButton("Chỉnh sửa chi tiết", self)
        self.btn_nav_remover.setIcon(FluentIcon.SETTING)

        row3.addWidget(lbl_remove_title)
        row3.addWidget(self.lbl_summary_remover, 1)
        row3.addWidget(lbl_style_title)
        row3.addWidget(self.style_combo)
        row3.addWidget(self.btn_nav_remover)
        config_layout.addLayout(row3)

        main_layout.addWidget(config_card)

        # Các hộp chọn đồng bộ trực tiếp với danh sách ngôn ngữ của Tab Dịch Phụ Đề
        self.ocr_lang_combo = ComboBox(self)
        self.ocr_lang_combo.addItems(list(LANG_MAP_SOURCE.keys()))
        self.target_lang_combo = ComboBox(self)
        self.target_lang_combo.addItems(list(LANG_MAP_TARGET.keys()))
        self.engine_combo = ComboBox(self)
        self.engine_combo.addItems(["Google Translate", "Mô hình AI tùy chỉnh"])
        self.inpaint_combo = ComboBox(self)
        self.inpaint_combo.addItems(["STTN Tự Động", "LaMa", "ProPainter", "OpenCV Nhanh"])

        for hidden_w in [
            self.ocr_lang_combo,
            self.target_lang_combo,
            self.engine_combo,
            self.inpaint_combo,
        ]:
            hidden_w.setVisible(False)

        # -------------------------------------------------------------
        # 3. Không gian làm việc: Khung Xem Video & Nhật Ký Tiến Trình
        # -------------------------------------------------------------
        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(12)

        self.video_display = VideoDisplayComponent(self)
        workspace_layout.addWidget(self.video_display, 3)

        monitor_card = CardWidget(self)
        monitor_layout = QVBoxLayout(monitor_card)
        monitor_layout.setContentsMargins(14, 12, 14, 12)
        monitor_layout.setSpacing(10)

        monitor_layout.addWidget(SubtitleLabel("Sơ Đồ & Tiến Trình Tự Động", self))

        steps = [
            "Nhận diện phụ đề OCR và mốc thời gian",
            "Dịch AI và Xóa phụ đề gốc song song",
            "Ghép phụ đề mới và Xuất Video",
        ]
        self.stepper_widget = StepperWidget(steps, self)
        monitor_layout.addWidget(self.stepper_widget)

        self.log_console = TextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("Nhật ký xử lý luồng tự động sẽ hiển thị tại đây...")
        monitor_layout.addWidget(self.log_console, 1)

        workspace_layout.addWidget(monitor_card, 2)
        main_layout.addLayout(workspace_layout, 1)

        # -------------------------------------------------------------
        # 4. Thanh Tác Vụ Cuối Trang (Chia 2 hàng gọn gàng cho màn hình 960x600)
        # -------------------------------------------------------------
        action_card = CardWidget(self)
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(14, 10, 14, 10)
        action_layout.setSpacing(8)

        action_row1 = QHBoxLayout()
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setValue(0)
        self.status_label = BodyLabel("Sẵn sàng chạy luồng tự động.", self)
        action_row1.addWidget(self.progress_bar, 1)
        action_row1.addWidget(self.status_label)
        action_layout.addLayout(action_row1)

        action_row2 = QHBoxLayout()
        self.pause_review_checkbox = CheckBox("Dừng duyệt và sửa phụ đề trước khi ghép video", self)
        action_row2.addWidget(self.pause_review_checkbox)

        self.conflict_mode_combo = ComboBox(self)
        self.conflict_mode_combo.addItems([
            "Tự đổi tên khi trùng đầu ra",
            "Chặn và cảnh báo khi trùng",
            "Bỏ qua video đã có đầu ra",
            "Ghi đè tệp đầu ra cũ",
        ])
        action_row2.addWidget(self.conflict_mode_combo)
        action_row2.addStretch(1)

        self.btn_cancel_pipeline = PushButton("Hủy Tác Vụ", self)
        self.btn_cancel_pipeline.setIcon(FluentIcon.CLOSE)
        self.btn_cancel_pipeline.setEnabled(False)
        action_row2.addWidget(self.btn_cancel_pipeline)

        self.btn_run_pipeline = PrimaryPushButton("Bắt Đầu Luồng Tự Động", self)
        self.btn_run_pipeline.setIcon(FluentIcon.PLAY)
        action_row2.addWidget(self.btn_run_pipeline)

        action_layout.addLayout(action_row2)
        main_layout.addWidget(action_card)

    def _connect_signals(self):
        self.btn_select_video.clicked.connect(self._select_video_file)
        self.btn_select_folder.clicked.connect(self._select_video_folder)
        self.btn_view_queue.clicked.connect(self._open_video_queue_dialog)
        self.btn_run_pipeline.clicked.connect(self._start_pipeline)
        self.btn_cancel_pipeline.clicked.connect(self._cancel_pipeline)

        self.btn_nav_translate.clicked.connect(self._jump_to_translate_tab)
        self.btn_nav_remover.clicked.connect(self._jump_to_remover_tab)
        self.btn_nav_ytdlp.clicked.connect(self._jump_to_ytdlp_tab)

        self.ocr_lang_combo.currentIndexChanged.connect(self._save_pipeline_config)
        self.target_lang_combo.currentIndexChanged.connect(self._save_pipeline_config)
        self.inpaint_combo.currentIndexChanged.connect(self._save_pipeline_config)
        self.style_combo.currentIndexChanged.connect(self._save_pipeline_config)
        self.conflict_mode_combo.currentIndexChanged.connect(self._save_pipeline_config)
        self.pause_review_checkbox.stateChanged.connect(self._save_pipeline_config)

    def _open_video_queue_dialog(self):
        if not self.video_queue:
            InfoBar.warning("Hàng đợi trống", "Chưa có tệp video nào trong hàng đợi.", parent=self, duration=2500)
            return
        dlg = VideoQueueDialog(self.video_queue, parent=self)
        if dlg.exec():
            self.video_queue = dlg.video_queue
            if self.video_queue:
                self._select_video_file_by_path(self.video_queue[0])
                if len(self.video_queue) == 1:
                    self.lbl_video_status.setText(f"Đã nạp Video: {os.path.basename(self.video_queue[0])}")
                else:
                    self.lbl_video_status.setText(f"Hàng đợi ({len(self.video_queue)} video)")
                self.btn_view_queue.setEnabled(True)
            else:
                self.current_video_path = None
                self.lbl_video_status.setText("Đã nạp: Chưa chọn video nào")
                self.lbl_video_status.setStyleSheet("color: #666666; font-style: italic;")
                self.btn_view_queue.setEnabled(False)

    def _jump_to_translate_tab(self):
        main_win = self.window()
        if hasattr(main_win, "toolsInterface"):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(2)

    def _jump_to_remover_tab(self):
        main_win = self.window()
        if hasattr(main_win, "toolsInterface"):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(3)

    def _jump_to_ytdlp_tab(self):
        main_win = self.window()
        if hasattr(main_win, "toolsInterface"):
            main_win.switchTo(main_win.toolsInterface)
            main_win.toolsInterface.switch_to_sub_tab(0)

    def _resolve_synced_translation_settings(self) -> tuple[dict, str, str, str, str, str]:
        """
        Đọc và đồng bộ cài đặt ngôn ngữ nguồn, ngôn ngữ đích và mô hình dịch từ CONFIG_API_FILE.
        Trả về: (api_config, ocr_lang_code, target_lang_code, engine_type, src_label, tgt_label).
        """
        api_config = load_translation_api_config()

        src_keys = list(LANG_MAP_SOURCE.keys())
        tgt_keys = list(LANG_MAP_TARGET.keys())

        if "src_lang_index" in api_config and 0 <= int(api_config["src_lang_index"]) < self.ocr_lang_combo.count():
            self.ocr_lang_combo.blockSignals(True)
            self.ocr_lang_combo.setCurrentIndex(int(api_config["src_lang_index"]))
            self.ocr_lang_combo.blockSignals(False)

        if "tgt_lang_index" in api_config and 0 <= int(api_config["tgt_lang_index"]) < self.target_lang_combo.count():
            self.target_lang_combo.blockSignals(True)
            self.target_lang_combo.setCurrentIndex(int(api_config["tgt_lang_index"]))
            self.target_lang_combo.blockSignals(False)

        src_label = api_config.get("src_lang_label") or self.ocr_lang_combo.currentText() or src_keys[0]
        tgt_label = api_config.get("tgt_lang_label") or self.target_lang_combo.currentText() or tgt_keys[0]

        ocr_lang = api_config.get("src_lang") or LANG_MAP_SOURCE.get(src_label, "auto")
        target_lang = api_config.get("tgt_lang") or LANG_MAP_TARGET.get(tgt_label, "vi")

        engine_idx = api_config.get("engine_index", 0)
        if engine_idx == 0:
            engine_type = "google"
        else:
            engine_type = api_config.get("engine_type", "")
            if not engine_type:
                provider_idx = api_config.get("provider_index", 0)
                provider_keys = list(PROVIDERS_INFO.keys())
                p_name = api_config.get("provider_name") or (
                    provider_keys[provider_idx] if 0 <= provider_idx < len(provider_keys) else ""
                )
                if p_name == "GGUF Model":
                    engine_type = "gguf"
                elif p_name == "MarianMT":
                    engine_type = "marian"
                elif api_config.get("api_key"):
                    engine_type = "llm"
                else:
                    engine_type = "google"

        return api_config, ocr_lang, target_lang, engine_type, src_label, tgt_label

    def _resolve_synced_inpaint_mode(self) -> str:
        """Đọc chế độ xóa phụ đề đồng bộ từ cấu hình chung và CONFIG_PIPELINE_FILE."""
        from src.core.config import config as main_cfg
        raw_mode = getattr(main_cfg.inpaintMode, "value", "sttn_auto")
        if CONFIG_PIPELINE_FILE.exists():
            try:
                auto_cfg = json.loads(CONFIG_PIPELINE_FILE.read_text(encoding="utf-8"))
                if auto_cfg.get("inpaint_mode"):
                    raw_mode = auto_cfg["inpaint_mode"]
            except Exception:
                pass

        while hasattr(raw_mode, "value"):
            raw_mode = raw_mode.value

        mode_str = str(raw_mode).lower().replace("-", "_")
        if "lama" in mode_str:
            return "lama"
        if "propainter" in mode_str:
            return "propainter"
        if "opencv" in mode_str:
            return "opencv"
        return "sttn_auto"

    def _update_settings_summary(self):
        """Đọc và hiển thị các thiết lập hiện tại lên giao diện tối giản."""
        api_config, _, _, engine_type, src_label, tgt_label = self._resolve_synced_translation_settings()
        lang_pair_str = f"OCR: {src_label} -> Dịch: {tgt_label}"

        if engine_type == "google":
            self.lbl_summary_translate.setText(f"{lang_pair_str} | Mô hình: Google Translate")
        elif engine_type == "gguf":
            model_name = api_config.get("model_name", "Qwen2.5")
            self.lbl_summary_translate.setText(f"{lang_pair_str} | Mô hình GGUF cục bộ: {model_name}")
        elif engine_type == "marian":
            self.lbl_summary_translate.setText(f"{lang_pair_str} | Mô hình MarianMT cục bộ")
        else:
            provider_idx = api_config.get("provider_index", 0)
            provider_keys = list(PROVIDERS_INFO.keys())
            provider_name = api_config.get("provider_name") or (
                provider_keys[provider_idx] if 0 <= provider_idx < len(provider_keys) else "AI"
            )
            model_name = api_config.get("model_name", "")
            has_key = bool(str(api_config.get("api_key", "")).strip())
            is_verified = bool(api_config.get("api_key_verified", False))
            if not has_key:
                key_str = "Chưa nạp khóa"
            elif is_verified:
                key_str = "Đã xác thực"
            else:
                key_str = "Đã lưu khóa - chưa xác thực"
            self.lbl_summary_translate.setText(f"{lang_pair_str} | {provider_name} ({model_name}) | {key_str}")

        from src.core.config import config as main_cfg
        hw_enabled = getattr(main_cfg.hardwareAcceleration, "value", True)
        hw_str = "Tăng tốc GPU: BẬT" if hw_enabled else "Chế độ CPU"

        mode_code = self._resolve_synced_inpaint_mode()
        inpaint_display_map = {
            "sttn_auto": "STTN Tự Động",
            "lama": "LaMa Phục Hồi Nền",
            "propainter": "ProPainter",
            "opencv": "OpenCV Nhanh",
        }
        inpaint_text = inpaint_display_map.get(mode_code, mode_code)
        self.lbl_summary_remover.setText(f"Mô hình xóa: {inpaint_text} | {hw_str}")

        combo_map = {"sttn_auto": 0, "lama": 1, "propainter": 2, "opencv": 3}
        self.inpaint_combo.blockSignals(True)
        self.inpaint_combo.setCurrentIndex(combo_map.get(mode_code, 0))
        self.inpaint_combo.blockSignals(False)

    def _select_video_file(self):
        filepath, _ = FolderMemoryDialog.getOpenFileName(
            self,
            "Chọn tệp Video nguồn",
            filter_str="Tệp Video (*.mp4 *.mkv *.avi *.mov *.flv *.webm);;Tất cả tệp (*)",
            category="video",
        )
        if filepath:
            self.video_queue = [filepath]
            self._select_video_file_by_path(filepath)
            self.btn_view_queue.setEnabled(True)
            self.lbl_video_status.setText(f"Đã nạp Video: {os.path.basename(filepath)}")
            self.lbl_video_status.setStyleSheet("color: #107c41; font-weight: bold;")
            InfoBar.success("Đã nạp Video", f"Tệp: {os.path.basename(filepath)}", parent=self, duration=2500)

    def _select_video_folder(self):
        folder = FolderMemoryDialog.getExistingDirectory(self, "Chọn thư mục chứa Video", category="video")
        if folder:
            files = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"))
            )
            if files:
                self.video_queue = [os.path.join(folder, f) for f in files]
                first_video = self.video_queue[0]
                self._select_video_file_by_path(first_video)
                self.btn_view_queue.setEnabled(True)
                self.lbl_video_status.setText(f"Hàng đợi thư mục ({len(self.video_queue)} video): {os.path.basename(folder)}")
                self.lbl_video_status.setStyleSheet("color: #107c41; font-weight: bold;")
                InfoBar.success(
                    "Đã nạp thư mục hàng loạt",
                    f"Đã xếp hàng đợi {len(self.video_queue)} video để xử lý tuần tự!",
                    parent=self,
                    duration=3000,
                )
            else:
                self.video_queue = []
                self.current_video_path = None
                self.btn_view_queue.setEnabled(False)
                self.lbl_video_status.setText("Đã nạp: Chưa chọn video nào")
                self.lbl_video_status.setStyleSheet("color: #666666; font-style: italic;")
                InfoBar.warning("Thư mục trống", "Không tìm thấy tệp video hợp lệ trong thư mục đã chọn.", parent=self, duration=3000)

    def _select_video_file_by_path(self, filepath: str):
        if os.path.exists(filepath):
            self.current_video_path = filepath
            self.video_display.set_video_path(filepath)
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.video_display.set_video_parameters(w, h, fps, fc)
                ret, frame = cap.read()
                if ret:
                    self.video_display.update_video_display(frame)
                cap.release()

    def _cancel_pipeline(self):
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.stop()
            self.status_label.setText("Đang dừng luồng tự động...")
            self.btn_cancel_pipeline.setEnabled(False)
            InfoBar.warning("Đang hủy tác vụ", "Đã gửi tín hiệu dừng luồng tự động.", parent=self, duration=2500)

    def _start_pipeline(self):
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            InfoBar.warning(
                "Cảnh báo",
                "Vui lòng chọn tệp Video nguồn hoặc thư mục Video trước khi chạy luồng tự động!",
                parent=self,
                duration=3000,
            )
            return

        if self.pipeline_worker and self.pipeline_worker.isRunning():
            return

        self._save_pipeline_config()
        api_config, ocr_lang, target_lang, engine_type, _, _ = self._resolve_synced_translation_settings()
        inpaint_mode = self._resolve_synced_inpaint_mode()

        style_map = {
            "TikTok Vàng Nổi Bật": "tiktok_yellow",
            "Phụ Đề Trắng Viền Đen": "clean_white",
            "Sub Trắng Viền Đen Chữ Nổi": "clean_white",
        }
        style_type = style_map.get(self.style_combo.currentText(), "tiktok_yellow")

        active_queue = [p for p in self.video_queue if os.path.exists(p)]
        if not active_queue:
            active_queue = [self.current_video_path]

        conflict_modes = ["auto_rename", "block", "skip", "overwrite"]
        c_idx = self.conflict_mode_combo.currentIndex()
        conflict_mode = conflict_modes[c_idx] if 0 <= c_idx < len(conflict_modes) else "auto_rename"

        batch_plan = resolve_batch_output_plan(active_queue, conflict_mode=conflict_mode)
        if conflict_mode == "block" and batch_plan["has_conflicts"]:
            first_c = batch_plan["conflicts"][0]
            InfoBar.error(
                "Phát hiện xung đột tên tệp đầu ra",
                f"Tệp '{os.path.basename(first_c['target_output'])}' đã tồn tại hoặc bị trùng trong hàng đợi. Vui lòng đổi chế độ xử lý trùng tên hoặc dọn hàng đợi.",
                parent=self,
                duration=4500,
            )
            return

        resolved_pairs = batch_plan["resolved_items"]
        if not resolved_pairs:
            InfoBar.warning(
                "Đã bỏ qua toàn bộ hàng đợi",
                "Tất cả các tệp video trong hàng đợi đều đã có sẵn tệp đầu ra trên ổ đĩa.",
                parent=self,
                duration=4000,
            )
            return

        resolved_queue = [v for v, _ in resolved_pairs]
        output_path = resolved_pairs[0][1]

        sub_areas = self.video_display.get_selection_coordinates()

        self.log_console.clear()
        self.progress_bar.setValue(0)
        self.btn_run_pipeline.setEnabled(False)
        self.btn_cancel_pipeline.setEnabled(True)

        self.pipeline_worker = AutoPipelineWorker(
            video_path=resolved_queue[0],
            output_path=output_path,
            ocr_lang=ocr_lang,
            target_lang=target_lang,
            engine_type=engine_type,
            api_config=api_config,
            inpaint_mode=inpaint_mode,
            sub_areas=sub_areas,
            style_type=style_type,
            pause_for_review=self.pause_review_checkbox.isChecked(),
            video_queue=resolved_queue,
            output_map=batch_plan["output_map"],
            parent=self,
        )

        self.pipeline_worker.progress_signal.connect(self._on_progress)
        self.pipeline_worker.step_signal.connect(self._on_step_changed)
        self.pipeline_worker.log_signal.connect(self._on_log)
        self.pipeline_worker.finished_signal.connect(self._on_finished)
        self.pipeline_worker.review_requested_signal.connect(self._on_review_requested)
        self.pipeline_worker.finished.connect(self._on_worker_terminated)
        self.pipeline_worker.finished.connect(self.pipeline_worker.deleteLater)

        self.pipeline_worker.start()

    def _on_worker_terminated(self):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_cancel_pipeline.setEnabled(False)

    def _on_review_requested(self, blocks, evt):
        dialog = SubtitleReviewDialog(blocks, parent=self)
        dialog.exec()
        evt.set()

    @Slot(int, str)
    def _on_progress(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    @Slot(int)
    def _on_step_changed(self, step: int):
        self.stepper_widget.set_current_step(step)

    @Slot(str)
    def _on_log(self, msg: str):
        self.log_console.append(msg)

    @Slot(bool, str, str)
    def _on_finished(self, success: bool, output_file: str, message: str):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_cancel_pipeline.setEnabled(False)

        is_partial = message.startswith("[PARTIAL] ")
        is_cancelled = message.startswith("[CANCELLED] ")
        clean_msg = (
            message[len("[PARTIAL] ") :].strip()
            if is_partial
            else (message[len("[CANCELLED] ") :].strip() if is_cancelled else message)
        )

        if is_cancelled:
            self.progress_bar.setValue(0)
            self.status_label.setText("Đã hủy tác vụ.")
            InfoBar.warning("Đã Hủy Luồng Tự Động", clean_msg, parent=self, duration=3500)
            return

        if success and is_partial:
            self.stepper_widget.set_current_step(4)
            self.progress_bar.setValue(100)
            self.status_label.setText("Hoàn thành một phần!")
            InfoBar.warning("Hoàn Thành Một Phần", clean_msg, parent=self, duration=5000)
        elif success:
            self.stepper_widget.set_current_step(4)
            self.progress_bar.setValue(100)
            self.status_label.setText("Hoàn thành luồng tự động!")
            InfoBar.success(
                "Thành công",
                f"{clean_msg}: {os.path.basename(output_file)}" if output_file else clean_msg,
                parent=self,
                duration=4000,
            )
        else:
            self.stepper_widget.set_error_step(self.stepper_widget.current_step)
            self.status_label.setText("Lỗi khi xử lý luồng.")
            InfoBar.error("Lỗi Luồng Tự Động", clean_msg, parent=self, duration=4500)

    def _save_pipeline_config(self):
        try:
            inpaint_codes = ["sttn_auto", "lama", "propainter", "opencv"]
            inpaint_idx = self.inpaint_combo.currentIndex()
            cfg = {
                "ocr_lang_index": self.ocr_lang_combo.currentIndex(),
                "target_lang_index": self.target_lang_combo.currentIndex(),
                "inpaint_index": inpaint_idx,
                "inpaint_mode": inpaint_codes[inpaint_idx] if 0 <= inpaint_idx < len(inpaint_codes) else "sttn_auto",
                "style_index": self.style_combo.currentIndex(),
                "conflict_mode_index": self.conflict_mode_combo.currentIndex(),
                "pause_for_review": self.pause_review_checkbox.isChecked(),
            }
            CONFIG_PIPELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PIPELINE_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_pipeline_config(self):
        try:
            if CONFIG_PIPELINE_FILE.exists():
                cfg = json.loads(CONFIG_PIPELINE_FILE.read_text(encoding="utf-8"))
                if 0 <= cfg.get("ocr_lang_index", 0) < self.ocr_lang_combo.count():
                    self.ocr_lang_combo.setCurrentIndex(cfg["ocr_lang_index"])
                if 0 <= cfg.get("target_lang_index", 0) < self.target_lang_combo.count():
                    self.target_lang_combo.setCurrentIndex(cfg["target_lang_index"])
                if 0 <= cfg.get("inpaint_index", 0) < self.inpaint_combo.count():
                    self.inpaint_combo.setCurrentIndex(cfg["inpaint_index"])
                if 0 <= cfg.get("style_index", 0) < self.style_combo.count():
                    self.style_combo.setCurrentIndex(cfg["style_index"])
                if 0 <= cfg.get("conflict_mode_index", 0) < self.conflict_mode_combo.count():
                    self.conflict_mode_combo.setCurrentIndex(cfg["conflict_mode_index"])
                if "pause_for_review" in cfg:
                    self.pause_review_checkbox.setChecked(bool(cfg["pause_for_review"]))
        except Exception:
            pass

    def retranslateUi(self):
        self.btn_select_video.setText("Chọn Video Nguồn")
        self.btn_select_folder.setText("Chọn Thư Mục Video")
        self.btn_view_queue.setText("Danh Sách Hàng Đợi")
        self.btn_nav_ytdlp.setText("Tải Video")
        self.btn_nav_translate.setText("Chỉnh sửa chi tiết")
        self.btn_nav_remover.setText("Chỉnh sửa chi tiết")
        self.pause_review_checkbox.setText("Dừng duyệt và sửa phụ đề trước khi ghép video")
        self.btn_cancel_pipeline.setText("Hủy Tác Vụ")
        self.btn_run_pipeline.setText("Bắt Đầu Luồng Tự Động")
        self._update_settings_summary()
