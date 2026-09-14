"""
Tac vu nen Celery xu ly tai video qua yt-dlp.
"""
import os
import logging
import traceback
from datetime import datetime

from src.web_api.api.celery_app import celery_app
from src.web_api.api.ytdlp_service import YtdlpService, DOWNLOAD_DIR
from src.web_api.api.database import (
    SessionLocal, update_download_job
)

logger = logging.getLogger("sakai_saas.ytdlp_worker")


@celery_app.task(bind=True, name='ytdlp.download_video')
def download_video_task(self, url: str, job_id: str, options: dict = None):
    """
    Tac vu Celery tai video tu lien ket.
    options chua: format_id, container, audio_only, audio_format, custom_filename
    """
    options = options or {}
    db = SessionLocal()

    try:
        import yt_dlp

        update_download_job(db, job_id, status="Đang tải", progress=0)
        self.update_state(state='PROGRESS', meta={
            'current': 0,
            'total': 100,
            'status': 'Đang bắt đầu tải xuống...',
            'speed': '',
            'eta': ''
        })

        custom_name = options.get('custom_filename', '')
        container = options.get('container', 'mp4')
        audio_only = options.get('audio_only', False)
        audio_format = options.get('audio_format', 'mp3')

        if audio_only:
            ext = audio_format
        else:
            ext = container

        if custom_name:
            import re
            custom_name = re.sub(r'[<>:"/\\|?*]', '', custom_name).strip()

        if not custom_name:
            custom_name = "video"

        output_template = YtdlpService.get_unique_filename(
            DOWNLOAD_DIR, f"{custom_name}.{ext}"
        )

        def progress_hook(d):
            if d.get('status') == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)

                if total > 0:
                    pct = min(int(downloaded * 100 / total), 99)
                else:
                    pct = 0

                speed_raw = d.get('speed', 0)
                if speed_raw and speed_raw > 0:
                    if speed_raw >= 1048576:
                        speed_str = f"{speed_raw / 1048576:.1f} MB/giây"
                    elif speed_raw >= 1024:
                        speed_str = f"{speed_raw / 1024:.0f} KB/giây"
                    else:
                        speed_str = f"{speed_raw:.0f} B/giây"
                else:
                    speed_str = ""

                eta_raw = d.get('eta', 0)
                if eta_raw and eta_raw > 0:
                    mins, secs = divmod(int(eta_raw), 60)
                    hours, mins = divmod(mins, 60)
                    if hours > 0:
                        eta_str = f"{hours} giờ {mins} phút"
                    elif mins > 0:
                        eta_str = f"{mins} phút {secs} giây"
                    else:
                        eta_str = f"{secs} giây"
                else:
                    eta_str = ""

                self.update_state(state='PROGRESS', meta={
                    'current': pct,
                    'total': 100,
                    'status': f'Đang tải... {pct}%',
                    'speed': speed_str,
                    'eta': eta_str
                })

                if pct % 5 == 0:
                    try:
                        update_download_job(
                            db, job_id,
                            progress=pct,
                            speed=speed_str,
                            eta=eta_str
                        )
                    except Exception:
                        pass

            elif d.get('status') == 'finished':
                self.update_state(state='PROGRESS', meta={
                    'current': 95,
                    'total': 100,
                    'status': 'Đang hoàn tất và đóng gói tệp...',
                    'speed': '',
                    'eta': ''
                })

        ydl_opts = YtdlpService.build_download_options(
            output_path=output_template,
            format_id=options.get('format_id'),
            container=container,
            audio_only=audio_only,
            audio_format=audio_format,
            concurrent_fragments=options.get('concurrent_fragments', 4),
            cookies_file=options.get('cookies_file'),
            progress_hook=progress_hook,
        )

        normalized = YtdlpService.normalize_url(url)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([normalized])

        final_path = output_template
        if not os.path.exists(final_path):
            base_no_ext = os.path.splitext(os.path.basename(output_template))[0]
            if os.path.exists(DOWNLOAD_DIR):
                for fname in os.listdir(DOWNLOAD_DIR):
                    if fname.startswith(base_no_ext):
                        final_path = os.path.join(DOWNLOAD_DIR, fname)
                        break

        filesize = os.path.getsize(final_path) if os.path.exists(final_path) else 0

        update_download_job(
            db, job_id,
            status="Hoàn thành",
            progress=100,
            output_path=final_path,
            filesize=filesize,
            speed=None,
            eta=None,
            completed_at=datetime.utcnow()
        )

        download_filename = os.path.basename(final_path)
        result_url = f"/ytdlp/file/{job_id}/{download_filename}"

        return {
            'status': 'Hoàn thành',
            'result_url': result_url,
            'filename': download_filename,
            'filesize': filesize,
        }

    except Exception as e:
        error_msg = YtdlpService.translate_error(str(e))
        logger.error(f"Lỗi tải video {job_id}: {traceback.format_exc()}")

        try:
            update_download_job(
                db, job_id,
                status="Thất bại",
                error_message=error_msg,
                completed_at=datetime.utcnow()
            )
        except Exception:
            pass

        self.update_state(state='FAILURE', meta={
            'error': error_msg
        })
        raise

    finally:
        db.close()
