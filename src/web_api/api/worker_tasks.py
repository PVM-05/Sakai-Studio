import os
import sys

# Ensure backend directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web_api.api.celery_app import celery_app
from src.core.main import GUISubtitleRemover

class DummyProgressNotifier:
    def __init__(self, task, total_frames):
        self.task = task
        self.total_frames = total_frames
        self.progress_total = 0
        self.ab_sections = []
        
    def append_output(self, msg):
        print(f"[Worker] {msg}")
        
    def notify_progress_listeners(self):
        # Called periodically by AI Core
        self.task.update_state(state='PROGRESS', meta={
            'current': self.progress_total, 
            'total': 100, 
            'status': f'Đang xử lý ({self.progress_total}%)'
        })

@celery_app.task(bind=True)
def process_video_task(self, video_path: str, user_id: str, job_id: str, options: dict):
    """
    Sử dụng Core AI (backend.main.GUISubtitleRemover) mà không cần giao diện (GUI).
    """
    try:
        # options contains: 'sub_areas' like [[ymin, ymax, xmin, xmax], ...]
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Khởi tạo tiến trình AI...'})
        
        remover = GUISubtitleRemover(vd_path=video_path, gui_mode=False)
        if 'sub_areas' in options:
            remover.sub_areas = options['sub_areas']
            
        # Hook our Dummy Notifier into the AI Core
        notifier = DummyProgressNotifier(self, remover.frame_count)
        remover.progress_listeners.append(lambda p: notifier.notify_progress_listeners())
        
        # Monkey patch append_output to our notifier for logs
        remover.append_output = notifier.append_output
        
        # Chạy thuật toán chính
        remover.run()
        
        return {
            'current': 100, 
            'total': 100, 
            'status': 'Hoàn thành!', 
            'result_url': f'/api/downloads/{job_id}/result.mp4',
            'local_path': remover.video_out_path
        }
    except Exception as e:
        import traceback
        return {'status': 'Lỗi', 'error': str(e), 'trace': traceback.format_exc()}
