import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from src.web_api.api.worker_tasks import process_video_task

app = FastAPI(title="Sakai Studio SaaS API", version="1.0.0")

# Setup upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Sakai Studio SaaS API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.mov', '.webm', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format")
        
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    return {"info": f"file '{file.filename}' saved at '{file_location}'", "filename": file.filename}

@app.post("/process")
async def process_video(filename: str = Form(...), boxes: str = Form(...)):
    """
    boxes: JSON string of boxes (x, y, w, h)
    """
    import json
    try:
        boxes_data = json.loads(boxes)
        # Convert boxes to sub_areas format [ymin, ymax, xmin, xmax] expected by backend
        sub_areas = []
        for box in boxes_data:
            ymin = int(box['y'])
            ymax = int(box['y'] + box['h'])
            xmin = int(box['x'])
            xmax = int(box['x'] + box['w'])
            sub_areas.append([ymin, ymax, xmin, xmax])
            
        video_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
            
        task = process_video_task.delay(
            video_path=video_path,
            user_id="demo_user",
            job_id=filename,
            options={'sub_areas': sub_areas}
        )
        return {"task_id": task.id, "status": "Processing started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    from celery.result import AsyncResult
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
    }
    
    if task_result.status == 'PROGRESS':
        result["meta"] = task_result.info
    elif task_result.status == 'SUCCESS':
        result["meta"] = task_result.info
    elif task_result.status == 'FAILURE':
        result["error"] = str(task_result.info)
        
    return result

@app.get("/downloads/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    # Trả về video kết quả
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_no_sub.mp4")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Result not found")
