"""
Tac vu nen Celery xu ly trich xuat phu de qua VideoOcrEngine va SubtitleExporter.
"""
import os
import cv2
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

from src.web_api.api.celery_app import celery_app
from src.web_api.api.database import SessionLocal, update_extract_job
from src.ai_engines.ocr_engine import VideoOcrEngine
from src.core.tools.subtitle_exporter import SubtitleExporter

logger = logging.getLogger("sakai_saas.extract_worker")

SUBTITLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'downloads')
)
os.makedirs(SUBTITLE_DIR, exist_ok=True)


@celery_app.task(bind=True, name='extract.extract_subtitles')
def extract_subtitles_task(self, video_path: str, job_id: str, options: dict = None):
    """
    Tac vu Celery quet chu OCR va nhan dien tieng noi xuat phu de SRT va ASS.
    """
    options = options or {}
    db = SessionLocal()

    try:
        update_extract_job(db, job_id, status="Đang nhận diện", progress=0, current_step="Đang chuẩn bị...")
        self.update_state(state='PROGRESS', meta={
            'current': 0, 'total': 100,
            'status': 'Đang chuẩn bị động cơ nhận diện chữ...'
        })

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy tệp video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở tệp video để giải mã khung hình.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        sub_areas = options.get('sub_areas', [])
        ocr_mode = options.get('ocr_mode', 'auto')
        ocr_lang = options.get('ocr_lang', 'vi')
        use_whisper_fallback = options.get('use_whisper_fallback', True)
        use_voice_separation = options.get('use_voice_separation', False)

        engine = VideoOcrEngine(
            ocr_mode=ocr_mode,
            ocr_lang=ocr_lang,
            use_typo_map=True,
            use_whisper_fallback=use_whisper_fallback,
            use_voice_separation=use_voice_separation
        )

        def progress_callback(pct, msg):
            int_pct = int(pct)
            self.update_state(state='PROGRESS', meta={
                'current': int_pct,
                'total': 100,
                'status': msg
            })
            if int_pct % 5 == 0:
                try:
                    update_extract_job(db, job_id, progress=int_pct, current_step=msg)
                except Exception:
                    pass

        segments = engine.extract_subtitles(video_path, sub_areas, progress_callback)

        if not segments:
            update_extract_job(
                db, job_id,
                status="Hoàn thành",
                progress=100,
                current_step="Không phát hiện thấy chữ phụ đề",
                subtitle_count=0,
                subtitles_json="[]",
                completed_at=datetime.utcnow()
            )
            return {
                'status': 'Hoàn thành',
                'subtitle_count': 0,
                'subtitles': []
            }

        items = []
        subtitles_data = []
        for seg in segments:
            items.append({
                'start_frame': seg.start_frame,
                'end_frame': seg.end_frame,
                'text': seg.text
            })
            subtitles_data.append({
                'index': seg.index,
                'start_time': seg.start_time,
                'end_time': seg.end_time,
                'text': seg.text,
                'source': seg.source
            })

        base_name = Path(video_path).stem
        srt_path = os.path.join(SUBTITLE_DIR, f"{base_name}_{job_id[:8]}.srt")
        ass_path = os.path.join(SUBTITLE_DIR, f"{base_name}_{job_id[:8]}.ass")

        SubtitleExporter.export_srt(srt_path, items, fps)
        SubtitleExporter.export_ass(ass_path, items, fps, sub_areas=sub_areas)

        update_extract_job(
            db, job_id,
            status="Hoàn thành",
            progress=100,
            current_step=f"Đã trích xuất thành công {len(items)} câu phụ đề",
            subtitle_count=len(items),
            output_srt_path=srt_path,
            output_ass_path=ass_path,
            subtitles_json=json.dumps(subtitles_data, ensure_ascii=False),
            completed_at=datetime.utcnow()
        )

        return {
            'status': 'Hoàn thành',
            'subtitle_count': len(items),
            'srt_url': f"/extract/download/{job_id}/srt",
            'ass_url': f"/extract/download/{job_id}/ass",
            'subtitles': subtitles_data
        }

    except Exception as e:
        error_msg = f"Đã xảy ra lỗi trong quá trình trích xuất: {str(e)}"
        logger.error(f"Lỗi trích xuất {job_id}: {traceback.format_exc()}")
        try:
            update_extract_job(
                db, job_id,
                status="Thất bại",
                error_message=error_msg,
                completed_at=datetime.utcnow()
            )
        except Exception:
            pass
        self.update_state(state='FAILURE', meta={'error': error_msg})
        raise
    finally:
        db.close()
