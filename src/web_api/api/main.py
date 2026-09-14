import os
import cv2
import json
import shutil
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.web_api.api.database import (
    get_db, init_db, create_task_job,
    update_task_job_progress, get_task_job, get_recent_jobs,
    create_download_job, update_download_job, get_download_history, delete_download_job, DownloadJob,
    create_extract_job, update_extract_job, get_extract_job, get_extract_history, delete_extract_job, ExtractJob
)
from src.web_api.api.ytdlp_service import YtdlpService, DOWNLOAD_DIR
from src.web_api.api.worker_tasks import process_video_task

logger = logging.getLogger("sakai_saas")
app = FastAPI(title="Sakai Studio SaaS API", version="1.0.0")

# Setup upload directory
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for PaddleOCR engine to avoid reloading weights repeatedly
_CACHED_OCR_ENGINE = None


def get_cached_paddle():
    global _CACHED_OCR_ENGINE
    if _CACHED_OCR_ENGINE is None:
        try:
            from src.ai_engines.paddle_compat import build_paddleocr
            _CACHED_OCR_ENGINE = build_paddleocr(lang="ch", device="gpu")
        except Exception as e:
            logger.warning(f"Khong the khoi tao PaddleOCR tren GPU: {e}. Thu lai tren CPU...")
            try:
                from src.ai_engines.paddle_compat import build_paddleocr
                _CACHED_OCR_ENGINE = build_paddleocr(lang="ch", device="cpu")
            except Exception as ex:
                logger.error(f"Khong the khoi tao PaddleOCR: {ex}")
                _CACHED_OCR_ENGINE = None
    return _CACHED_OCR_ENGINE


def convert_boxes_to_sub_areas(boxes: List[dict], video_width: int, video_height: int) -> List[List[int]]:
    """
    Quy đổi danh sách hộp chữ nhật sang định dạng [ymin, ymax, xmin, xmax] theo pixel thực tế của video.
    Hỗ trợ cả tọa độ chuẩn hóa tỷ lệ (0..1) và tọa độ điểm ảnh tuyệt đối.
    """
    sub_areas = []
    for box in boxes:
        # Kiểm tra nếu là dạng chuẩn hóa phần trăm (x_pct, y_pct, w_pct, h_pct)
        if "x_pct" in box and "y_pct" in box and "w_pct" in box and "h_pct" in box:
            x_pct = float(box["x_pct"])
            y_pct = float(box["y_pct"])
            w_pct = float(box["w_pct"])
            h_pct = float(box["h_pct"])
            xmin = int(round(x_pct * video_width))
            ymin = int(round(y_pct * video_height))
            xmax = int(round((x_pct + w_pct) * video_width))
            ymax = int(round((y_pct + h_pct) * video_height))
        else:
            # Dạng thông thường (x, y, w, h)
            x = float(box.get("x", 0))
            y = float(box.get("y", 0))
            w = float(box.get("w", 0))
            h = float(box.get("h", 0))

            # Nếu tất cả các giá trị <= 1.0 (và > 0) thì xem như là tỷ lệ tương đối
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                xmin = int(round(x * video_width))
                ymin = int(round(y * video_height))
                xmax = int(round((x + w) * video_width))
                ymax = int(round((y + h) * video_height))
            else:
                xmin = int(round(x))
                ymin = int(round(y))
                xmax = int(round(x + w))
                ymax = int(round(y + h))

        # Cắt gọt tọa độ nằm gọn trong khung hình video
        xmin = max(0, min(video_width - 1, xmin))
        ymin = max(0, min(video_height - 1, ymin))
        xmax = max(xmin + 1, min(video_width, xmax))
        ymax = max(ymin + 1, min(video_height, ymax))

        sub_areas.append([ymin, ymax, xmin, xmax])
    return sub_areas


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def read_root():
    return {"message": "Sakai Studio SaaS API đang hoạt động bình thường"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.webm', '.mkv')):
        raise HTTPException(status_code=400, detail="Định dạng tệp không được hỗ trợ. Vui lòng tải MP4, MOV, WEBM hoặc MKV.")

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    return {
        "info": f"Đã lưu tệp '{file.filename}' thành công",
        "filename": file.filename,
        "path": file_location
    }


@app.post("/detect-subtitles")
async def detect_subtitles(filename: str = Form(...), timestamp: float = Form(0.0)):
    """
    Trích xuất khung hình tại thời điểm timestamp và nhận diện chữ tự động,
    trả về danh sách hộp chữ nhật chuẩn hóa theo kích thước video.
    """
    video_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp video trên máy chủ")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Không thể mở tệp video để đọc khung hình")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    target_frame = max(0, int(timestamp * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Không thể trích xuất khung hình tại mốc thời gian đã chọn")

    boxes = []
    try:
        ocr_engine = get_cached_paddle()
        if ocr_engine is not None:
            from src.ai_engines.paddle_compat import extract_paddle_boxes
            raw_boxes = extract_paddle_boxes(ocr_engine, frame, threshold=0.3)
            for (x1, y1, x2, y2) in raw_boxes:
                w_box = max(1, x2 - x1)
                h_box = max(1, y2 - y1)
                boxes.append({
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(w_box),
                    "h": int(h_box),
                    "x_pct": round(x1 / vid_w, 4),
                    "y_pct": round(y1 / vid_h, 4),
                    "w_pct": round(w_box / vid_w, 4),
                    "h_pct": round(h_box / vid_h, 4),
                })
    except Exception as e:
        logger.warning(f"Lỗi khi quét chữ OCR: {e}")

    return {
        "filename": filename,
        "timestamp": timestamp,
        "video_width": vid_w,
        "video_height": vid_h,
        "boxes": boxes
    }


@app.post("/process")
async def process_video(
    filename: str = Form(...),
    boxes: str = Form(...),
    user_id: str = Form("demo_user"),
    db: Session = Depends(get_db)
):
    video_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp video trên máy chủ")

    try:
        boxes_data = json.loads(boxes)
    except Exception:
        raise HTTPException(status_code=400, detail="Dữ liệu vùng chọn không đúng định dạng JSON")

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    sub_areas = convert_boxes_to_sub_areas(boxes_data, vid_w, vid_h)

    import uuid
    job_id = str(uuid.uuid4())

    # Lưu bản ghi ban đầu vào cơ sở dữ liệu
    create_task_job(
        db=db,
        job_id=job_id,
        filename=filename,
        original_path=video_path,
        user_id=user_id,
        sub_areas_json=json.dumps(sub_areas)
    )

    # Đưa tác vụ vào hàng đợi Celery
    task = process_video_task.apply_async(
        kwargs={
            'video_path': video_path,
            'user_id': user_id,
            'job_id': job_id,
            'options': {'sub_areas': sub_areas}
        },
        task_id=job_id
    )

    return {
        "task_id": job_id,
        "status": "Đang bắt đầu xử lý",
        "video_width": vid_w,
        "video_height": vid_h,
        "sub_areas_count": len(sub_areas)
    }


@app.get("/status/{task_id}")
async def get_status(task_id: str, db: Session = Depends(get_db)):
    from src.web_api.api.celery_app import celery_app
    celery_status = None
    celery_meta = {}
    error_msg = None

    try:
        task_result = celery_app.AsyncResult(task_id)
        celery_status = task_result.status
        if celery_status in ('PROGRESS', 'SUCCESS'):
            celery_meta = task_result.info if isinstance(task_result.info, dict) else {}
        elif celery_status == 'FAILURE':
            error_msg = str(task_result.info)
    except Exception:
        pass

    result = {
        "task_id": task_id,
        "task_status": celery_status,
    }

    if celery_status in ('PROGRESS', 'SUCCESS'):
        result["meta"] = celery_meta
    elif celery_status == 'FAILURE':
        result["error"] = error_msg
    else:
        # Nếu Celery chưa có trạng thái, tra cứu cơ sở dữ liệu
        db_job = get_task_job(db, task_id)
        if db_job:
            result["task_status"] = "PROGRESS" if db_job.status == "Đang xử lý" else db_job.status
            result["meta"] = {
                "current": db_job.progress,
                "total": 100,
                "status": db_job.status,
                "result_url": f"/downloads/{task_id}/result.mp4" if db_job.output_path else None
            }

    return result


@app.get("/events/{task_id}")
async def stream_task_events(task_id: str, db: Session = Depends(get_db)):
    """
    Luồng sự kiện thời gian thực (Server-Sent Events) đẩy tiến độ xử lý trực tiếp về trình duyệt.
    """
    from src.web_api.api.celery_app import celery_app

    async def event_generator():
        while True:
            current_status = None
            meta = {}
            error_msg = None

            try:
                task_result = celery_app.AsyncResult(task_id)
                current_status = task_result.status
                if isinstance(task_result.info, dict):
                    meta = task_result.info
                elif current_status == 'FAILURE':
                    error_msg = str(task_result.info)
            except Exception:
                current_status = None

            # Nếu không tìm thấy thông tin trên Celery, tra cứu cơ sở dữ liệu
            if current_status in ('PENDING', None) and not meta:
                db_job = get_task_job(db, task_id)
                if db_job:
                    current_status = "SUCCESS" if db_job.status == "Hoàn thành" else ("FAILURE" if db_job.status == "Thất bại" else "PROGRESS")
                    meta = {
                        "current": db_job.progress,
                        "total": 100,
                        "status": db_job.status,
                        "result_url": f"/downloads/{task_id}/result.mp4" if db_job.output_path else None
                    }
                else:
                    # Tác vụ không tồn tại trong hệ thống
                    data = {
                        "task_id": task_id,
                        "task_status": "FAILURE",
                        "error": "Không tìm thấy tác vụ tương ứng"
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    break

            data = {
                "task_id": task_id,
                "task_status": current_status,
                "meta": meta
            }

            if current_status == 'FAILURE':
                data["error"] = str(task_result.info) if task_result.info else "Đã xảy ra lỗi trong quá trình xử lý"

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if current_status in ('SUCCESS', 'FAILURE'):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/projects")
async def list_projects(limit: int = 20, user_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Lấy danh sách các dự án xử lý gần đây từ cơ sở dữ liệu.
    """
    jobs = get_recent_jobs(db, limit=limit, user_id=user_id)
    results = []
    for job in jobs:
        results.append({
            "id": job.id,
            "filename": job.filename,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else None,
            "completed_at": job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else None,
            "result_url": f"/downloads/{job.id}/result.mp4" if job.status == "Hoàn thành" else None
        })
    return results


@app.get("/downloads/{job_id}/{filename}")
async def download_file(job_id: str, filename: str, db: Session = Depends(get_db)):
    job = get_task_job(db, job_id)
    if job and job.output_path and os.path.exists(job.output_path):
        return FileResponse(job.output_path, filename=f"xoa_chu_{job.filename}")

    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_no_sub.mp4")
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=f"xoa_chu_{filename}")

    raise HTTPException(status_code=404, detail="Không tìm thấy tệp kết quả xử lý")


@app.post("/ytdlp/analyze")
async def ytdlp_analyze(request: Request):
    """Phân tích liên kết video, trả về thông tin và các định dạng khả dụng."""
    content_type = request.headers.get("content-type", "")
    url = ""
    if "application/json" in content_type:
        try:
            body = await request.json()
            url = body.get("url", "")
        except Exception:
            url = ""
    else:
        try:
            form = await request.form()
            url = form.get("url", "")
        except Exception:
            url = ""

    if not url:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp liên kết video hợp lệ.")

    try:
        result = YtdlpService.analyze_url(url)
        return result
    except Exception as e:
        error_msg = YtdlpService.translate_error(str(e))
        raise HTTPException(status_code=400, detail=error_msg)


@app.post("/ytdlp/download")
async def ytdlp_download(request: Request, db: Session = Depends(get_db)):
    """Đưa yêu cầu tải video vào hàng đợi Celery."""
    import uuid
    from src.web_api.api.ytdlp_worker import download_video_task

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            data = {}
    else:
        try:
            form = await request.form()
            data = dict(form)
        except Exception:
            data = {}

    url = data.get("url", "")
    if not url:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp liên kết video.")

    format_id = data.get("format_id")
    container = data.get("container", "mp4")
    audio_only = bool(data.get("audio_only", False))
    audio_format = data.get("audio_format", "mp3")
    custom_filename = data.get("custom_filename", "")
    user_id = data.get("user_id", "demo_user")

    job_id = str(uuid.uuid4())

    title = custom_filename or "Video"
    try:
        info = YtdlpService.analyze_url(url)
        title = custom_filename or info.get('title', 'Video')
    except Exception:
        pass

    create_download_job(
        db, job_id,
        url=url,
        user_id=user_id,
        title=title,
        format_id=format_id,
        container=container,
        audio_only=audio_only,
        audio_format=audio_format if audio_only else None,
        status="Đang chờ"
    )

    try:
        download_video_task.apply_async(
            kwargs={
                'url': url,
                'job_id': job_id,
                'options': {
                    'format_id': format_id,
                    'container': container,
                    'audio_only': audio_only,
                    'audio_format': audio_format,
                    'custom_filename': custom_filename or title,
                }
            },
            task_id=job_id
        )
    except Exception as e:
        logger.warning(f"Không thể đưa tác vụ vào Celery: {e}")

    return {
        "job_id": job_id,
        "status": "Đang chờ",
        "title": title
    }

@app.get("/ytdlp/events/{job_id}")
async def ytdlp_events(job_id: str, db: Session = Depends(get_db)):
    """Luong su kien thoi gian thuc cho tien trinh tai video."""
    from src.web_api.api.celery_app import celery_app

    async def event_generator():
        while True:
            current_status = None
            meta = {}

            try:
                task_result = celery_app.AsyncResult(job_id)
                current_status = task_result.status
                if isinstance(task_result.info, dict):
                    meta = task_result.info
            except Exception:
                current_status = None

            if current_status in ('PENDING', None) and not meta:
                db_job = db.query(DownloadJob).filter(DownloadJob.id == job_id).first()
                if db_job:
                    if db_job.status == "Hoàn thành":
                        current_status = "SUCCESS"
                    elif db_job.status == "Thất bại":
                        current_status = "FAILURE"
                    else:
                        current_status = "PROGRESS"
                    meta = {
                        'current': db_job.progress,
                        'total': 100,
                        'status': db_job.status,
                        'speed': db_job.speed or '',
                        'eta': db_job.eta or '',
                    }
                    if db_job.output_path:
                        fname = os.path.basename(db_job.output_path)
                        meta['result_url'] = f"/ytdlp/file/{job_id}/{fname}"
                    if db_job.error_message:
                        meta['error'] = db_job.error_message
                else:
                    data = {"job_id": job_id, "task_status": "FAILURE",
                            "error": "Không tìm thấy tác vụ tải tương ứng"}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    break

            data = {"job_id": job_id, "task_status": current_status, "meta": meta}
            if current_status == 'FAILURE' and 'error' not in meta:
                data["error"] = str(meta) if meta else "Đã xảy ra lỗi trong quá trình tải"

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if current_status in ('SUCCESS', 'FAILURE'):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.get("/ytdlp/history")
async def ytdlp_history(limit: int = 20, user_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Lay danh sach lich su tai video."""
    jobs = get_download_history(db, limit=limit, user_id=user_id)
    results = []
    for job in jobs:
        item = {
            "id": job.id,
            "url": job.url,
            "title": job.title,
            "channel": job.channel,
            "duration": job.duration,
            "thumbnail_url": job.thumbnail_url,
            "resolution": job.resolution,
            "container": job.container,
            "filesize": job.filesize,
            "status": job.status,
            "progress": job.progress,
            "speed": job.speed,
            "eta": job.eta,
            "audio_only": job.audio_only,
            "error_message": job.error_message,
            "created_at": job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else None,
            "completed_at": job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else None,
        }
        if job.output_path and job.status == "Hoàn thành":
            fname = os.path.basename(job.output_path)
            item["result_url"] = f"/ytdlp/file/{job.id}/{fname}"
        results.append(item)
    return results


@app.delete("/ytdlp/history/{job_id}")
async def ytdlp_delete_history(job_id: str, db: Session = Depends(get_db)):
    """Xoa mot muc trong lich su tai."""
    success = delete_download_job(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy mục lịch sử")
    return {"status": "Đã xóa mục lịch sử thành công"}


@app.post("/ytdlp/cancel/{job_id}")
async def ytdlp_cancel(job_id: str, db: Session = Depends(get_db)):
    """Huy tac vu tai dang chay."""
    from src.web_api.api.celery_app import celery_app
    try:
        celery_app.control.revoke(job_id, terminate=True)
    except Exception:
        pass
    update_download_job(db, job_id, status="Đã hủy")
    return {"status": "Đã hủy tác vụ tải"}


@app.get("/ytdlp/file/{job_id}/{filename}")
async def ytdlp_download_file(job_id: str, filename: str, db: Session = Depends(get_db)):
    """Tai tep video/am thanh da hoan thanh ve may."""
    job = db.query(DownloadJob).filter(DownloadJob.id == job_id).first()
    if job and job.output_path and os.path.exists(job.output_path):
        return FileResponse(job.output_path, filename=os.path.basename(job.output_path))

    file_path = os.path.join(DOWNLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)

    raise HTTPException(status_code=404, detail="Không tìm thấy tệp đã tải")


@app.post("/ytdlp/transfer/{job_id}")
async def ytdlp_transfer_to_remover(job_id: str, db: Session = Depends(get_db)):
    """Sao chep tep da tai sang khu vuc xu ly xoa phu de."""
    job = db.query(DownloadJob).filter(DownloadJob.id == job_id).first()
    if not job or not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp đã tải trên máy chủ")

    dest_filename = os.path.basename(job.output_path)
    dest_path = os.path.join(UPLOAD_DIR, dest_filename)
    shutil.copy2(job.output_path, dest_path)

    return {
        "status": "Đã chuyển",
        "filename": dest_filename,
        "message": f"Đã sao chép tệp '{dest_filename}' sang khu vực xử lý xóa phụ đề"
    }


# =========================================================================
# CÁC CỔNG DỊCH VỤ TRÍCH XUẤT PHỤ ĐỀ
# =========================================================================

@app.post("/extract/start")
async def extract_start(
    filename: str = Form(...),
    boxes: str = Form("[]"),
    ocr_mode: str = Form("auto"),
    ocr_lang: str = Form("vi"),
    use_whisper_fallback: bool = Form(True),
    use_voice_separation: bool = Form(False),
    user_id: str = Form("demo_user"),
    db: Session = Depends(get_db)
):
    """Bắt đầu tiến trình trích xuất phụ đề từ video."""
    import uuid
    from src.web_api.api.extract_worker import extract_subtitles_task

    # Tìm tệp video trong uploads hoặc downloads
    video_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(video_path):
        from src.web_api.api.ytdlp_service import DOWNLOAD_DIR
        dl_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(dl_path):
            video_path = dl_path
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp video trên máy chủ")

    try:
        boxes_data = json.loads(boxes) if boxes else []
    except Exception:
        boxes_data = []

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    sub_areas = convert_boxes_to_sub_areas(boxes_data, vid_w, vid_h) if boxes_data else []

    job_id = str(uuid.uuid4())

    create_extract_job(
        db,
        job_id=job_id,
        video_path=video_path,
        filename=filename,
        user_id=user_id,
        sub_areas_json=json.dumps(sub_areas) if sub_areas else None,
        ocr_mode=ocr_mode,
        ocr_lang=ocr_lang,
        use_whisper_fallback=use_whisper_fallback,
        use_voice_separation=use_voice_separation,
        status="Đang chờ",
        current_step="Đang chuẩn bị hàng đợi..."
    )

    try:
        extract_subtitles_task.apply_async(
            kwargs={
                'video_path': video_path,
                'job_id': job_id,
                'options': {
                    'sub_areas': sub_areas,
                    'ocr_mode': ocr_mode,
                    'ocr_lang': ocr_lang,
                    'use_whisper_fallback': use_whisper_fallback,
                    'use_voice_separation': use_voice_separation,
                }
            },
            task_id=job_id
        )
    except Exception as e:
        logger.warning(f"Không thể đưa tác vụ trích xuất vào Celery: {e}")

    return {
        "job_id": job_id,
        "status": "Đang chờ",
        "filename": filename,
        "sub_areas_count": len(sub_areas)
    }


@app.get("/extract/status/{job_id}")
async def extract_status(job_id: str, db: Session = Depends(get_db)):
    """Kiểm tra trạng thái tác vụ trích xuất phụ đề."""
    from src.web_api.api.celery_app import celery_app
    celery_status = None
    meta = {}

    try:
        task_result = celery_app.AsyncResult(job_id)
        celery_status = task_result.status
        if isinstance(task_result.info, dict):
            meta = task_result.info
    except Exception:
        pass

    db_job = get_extract_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ trích xuất")

    return {
        "job_id": job_id,
        "status": db_job.status,
        "progress": db_job.progress,
        "current_step": db_job.current_step,
        "subtitle_count": db_job.subtitle_count,
        "celery_status": celery_status,
        "srt_url": f"/extract/download/{job_id}/srt" if db_job.output_srt_path else None,
        "ass_url": f"/extract/download/{job_id}/ass" if db_job.output_ass_path else None,
        "error": db_job.error_message
    }


@app.get("/extract/events/{job_id}")
async def extract_events(job_id: str, db: Session = Depends(get_db)):
    """Luồng sự kiện thời gian thực đẩy tiến độ trích xuất phụ đề về trình duyệt."""
    from src.web_api.api.celery_app import celery_app

    async def event_generator():
        while True:
            current_status = None
            meta = {}

            try:
                task_result = celery_app.AsyncResult(job_id)
                current_status = task_result.status
                if isinstance(task_result.info, dict):
                    meta = task_result.info
            except Exception:
                current_status = None

            db_job = get_extract_job(db, job_id)
            if db_job:
                if db_job.status == "Hoàn thành":
                    current_status = "SUCCESS"
                elif db_job.status == "Thất bại":
                    current_status = "FAILURE"
                else:
                    current_status = "PROGRESS"

                meta = {
                    'current': db_job.progress,
                    'total': 100,
                    'status': db_job.current_step or db_job.status,
                    'subtitle_count': db_job.subtitle_count,
                    'srt_url': f"/extract/download/{job_id}/srt" if db_job.output_srt_path else None,
                    'ass_url': f"/extract/download/{job_id}/ass" if db_job.output_ass_path else None,
                }
                if db_job.error_message:
                    meta['error'] = db_job.error_message
            else:
                data = {"job_id": job_id, "task_status": "FAILURE", "error": "Không tìm thấy tác vụ tương ứng"}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            data = {"job_id": job_id, "task_status": current_status, "meta": meta}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if current_status in ('SUCCESS', 'FAILURE'):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.get("/extract/result/{job_id}")
async def extract_result(job_id: str, db: Session = Depends(get_db)):
    """Lấy danh sách các câu phụ đề đã trích xuất hoàn chỉnh."""
    db_job = get_extract_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ trích xuất")

    subtitles = []
    if db_job.subtitles_json:
        try:
            subtitles = json.loads(db_job.subtitles_json)
        except Exception:
            subtitles = []

    return {
        "job_id": job_id,
        "filename": db_job.filename,
        "status": db_job.status,
        "subtitle_count": db_job.subtitle_count,
        "subtitles": subtitles,
        "srt_url": f"/extract/download/{job_id}/srt" if db_job.output_srt_path else None,
        "ass_url": f"/extract/download/{job_id}/ass" if db_job.output_ass_path else None,
    }


@app.get("/extract/download/{job_id}/{fmt}")
async def extract_download_file(job_id: str, fmt: str, db: Session = Depends(get_db)):
    """Tải tệp phụ đề định dạng SRT hoặc ASS về máy tính."""
    db_job = get_extract_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ trích xuất")

    base_name = Path(db_job.filename).stem
    if fmt.lower() == "srt":
        if db_job.output_srt_path and os.path.exists(db_job.output_srt_path):
            return FileResponse(db_job.output_srt_path, filename=f"{base_name}_phu_de.srt")
    elif fmt.lower() == "ass":
        if db_job.output_ass_path and os.path.exists(db_job.output_ass_path):
            return FileResponse(db_job.output_ass_path, filename=f"{base_name}_phu_de.ass")

    raise HTTPException(status_code=404, detail="Không tìm thấy tệp phụ đề tương ứng")


@app.get("/extract/history")
async def extract_history(limit: int = 20, user_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Lấy danh sách lịch sử trích xuất phụ đề."""
    jobs = get_extract_history(db, limit=limit, user_id=user_id)
    results = []
    for job in jobs:
        results.append({
            "id": job.id,
            "filename": job.filename,
            "status": job.status,
            "progress": job.progress,
            "current_step": job.current_step,
            "subtitle_count": job.subtitle_count,
            "ocr_lang": job.ocr_lang,
            "ocr_mode": job.ocr_mode,
            "created_at": job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else None,
            "completed_at": job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else None,
            "srt_url": f"/extract/download/{job.id}/srt" if job.output_srt_path else None,
            "ass_url": f"/extract/download/{job.id}/ass" if job.output_ass_path else None,
        })
    return results


@app.delete("/extract/history/{job_id}")
async def extract_delete_history(job_id: str, db: Session = Depends(get_db)):
    """Xóa một mục lịch sử trích xuất phụ đề."""
    success = delete_extract_job(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy mục lịch sử trích xuất")
    return {"status": "Đã xóa mục lịch sử thành công"}
