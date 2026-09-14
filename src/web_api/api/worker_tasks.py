import os
import sys
import traceback
import logging

# Ensure backend directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web_api.api.celery_app import celery_app
from src.web_api.api.database import SessionLocal, update_task_job_progress
from src.core.main import GUISubtitleRemover

logger = logging.getLogger("sakai_saas_worker")


class DummyProgressNotifier:
    def __init__(self, task, job_id, total_frames):
        self.task = task
        self.job_id = job_id
        self.total_frames = total_frames
        self.progress_total = 0
        self.ab_sections = []

    def append_output(self, msg):
        print(f"[Worker] {msg}")

    def notify_progress_listeners(self):
        progress = int(self.progress_total)
        status_msg = f"Đang xử lý ({progress}%)"
        self.task.update_state(state='PROGRESS', meta={
            'current': progress,
            'total': 100,
            'status': status_msg
        })

        # Cập nhật cơ sở dữ liệu
        db = SessionLocal()
        try:
            update_task_job_progress(db, self.job_id, status=status_msg, progress=progress)
        except Exception as err:
            logger.warning(f"Lỗi khi đồng bộ tiến độ vào cơ sở dữ liệu: {err}")
        finally:
            db.close()


@celery_app.task(bind=True)
def process_video_task(self, video_path: str, user_id: str, job_id: str, options: dict):
    """
    Sử dụng Core AI (GUISubtitleRemover) mà không cần giao diện đồ họa.
    Đồng bộ trạng thái vào hàng đợi Celery và cơ sở dữ liệu SQLite.
    """
    db = SessionLocal()
    try:
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Khởi tạo tiến trình AI...'})
        update_task_job_progress(db, job_id, status="Đang xử lý", progress=0)

        remover = GUISubtitleRemover(vd_path=video_path, gui_mode=False)
        if 'sub_areas' in options:
            remover.sub_areas = options['sub_areas']

        # Kết nối bộ lắng nghe tiến độ vào lõi AI
        notifier = DummyProgressNotifier(self, job_id, remover.frame_count)
        remover.progress_listeners.append(lambda p: notifier.notify_progress_listeners())
        remover.append_output = notifier.append_output

        # Chạy thuật toán chính
        remover.run()

        output_path = remover.video_out_path
        update_task_job_progress(
            db,
            job_id,
            status="Hoàn thành",
            progress=100,
            output_path=output_path
        )

        return {
            'current': 100,
            'total': 100,
            'status': 'Hoàn thành!',
            'result_url': f'/downloads/{job_id}/result.mp4',
            'local_path': output_path
        }
    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"Lỗi tiến trình xử lý: {e}\n{err_trace}")
        update_task_job_progress(
            db,
            job_id,
            status="Thất bại",
            error_message=str(e)
        )
        return {'status': 'Lỗi', 'error': str(e), 'trace': err_trace}
    finally:
        db.close()
