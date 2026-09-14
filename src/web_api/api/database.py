import os
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# SQLite for local development, PostgreSQL in production
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "sakai_saas.db")
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db(target_engine=None):
    use_engine = target_engine or engine
    Base.metadata.create_all(bind=use_engine)


# Tự động tạo bảng nếu chưa có
init_db()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    plan = Column(String(50), default="Tiêu chuẩn")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TaskJob(Base):
    __tablename__ = "task_jobs"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(100), default="demo_user", index=True)
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500), nullable=False)
    output_path = Column(String(500), nullable=True)
    status = Column(String(50), default="Đang chờ")  # Đang chờ, Đang xử lý, Hoàn thành, Thất bại
    progress = Column(Integer, default=0)
    sub_areas_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class DownloadJob(Base):
    __tablename__ = "download_jobs"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)
    channel = Column(String, nullable=True)
    duration = Column(Integer, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    format_id = Column(String, nullable=True)
    resolution = Column(String, nullable=True)
    container = Column(String, default="mp4")
    filesize = Column(Integer, nullable=True)
    output_path = Column(String, nullable=True)
    status = Column(String, default="\u0110ang ch\u1edd")
    progress = Column(Integer, default=0)
    speed = Column(String, nullable=True)
    eta = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    audio_only = Column(Boolean, default=False)
    audio_format = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

class ExtractJob(Base):
    __tablename__ = "extract_jobs"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    video_path = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    sub_areas_json = Column(Text, nullable=True)
    ocr_mode = Column(String, default="auto")
    ocr_lang = Column(String, default="vi")
    use_whisper_fallback = Column(Boolean, default=True)
    use_voice_separation = Column(Boolean, default=False)
    status = Column(String, default="Đang chờ")
    progress = Column(Integer, default=0)
    current_step = Column(String, nullable=True)
    subtitle_count = Column(Integer, default=0)
    output_srt_path = Column(String, nullable=True)
    output_ass_path = Column(String, nullable=True)
    subtitles_json = Column(Text, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine=None):
    use_engine = target_engine or engine
    Base.metadata.create_all(bind=use_engine)


def create_task_job(
    db: Session,
    job_id: str,
    filename: str,
    original_path: str,
    user_id: str = "demo_user",
    sub_areas_json: Optional[str] = None
) -> TaskJob:
    job = TaskJob(
        id=job_id,
        user_id=user_id,
        filename=filename,
        original_path=original_path,
        status="Đang chờ",
        progress=0,
        sub_areas_json=sub_areas_json
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_task_job_progress(
    db: Session,
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    output_path: Optional[str] = None,
    error_message: Optional[str] = None
) -> Optional[TaskJob]:
    job = db.query(TaskJob).filter(TaskJob.id == job_id).first()
    if not job:
        return None

    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if output_path is not None:
        job.output_path = output_path
    if error_message is not None:
        job.error_message = error_message
    if status in ("Hoàn thành", "Thất bại"):
        job.completed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)
    return job


def get_task_job(db: Session, job_id: str) -> Optional[TaskJob]:
    return db.query(TaskJob).filter(TaskJob.id == job_id).first()


def get_recent_jobs(db: Session, limit: int = 10, user_id: Optional[str] = None) -> List[TaskJob]:
    query = db.query(TaskJob)
    if user_id:
        query = query.filter(TaskJob.user_id == user_id)
    return query.order_by(TaskJob.created_at.desc()).limit(limit).all()


def create_download_job(db, job_id: str, url: str, user_id: str = None, **kwargs):
    job = DownloadJob(id=job_id, url=url, user_id=user_id, **kwargs)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_download_job(db, job_id: str, **kwargs):
    job = db.query(DownloadJob).filter(DownloadJob.id == job_id).first()
    if job:
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        db.commit()
        db.refresh(job)
    return job


def get_download_history(db, limit: int = 20, user_id: str = None):
    query = db.query(DownloadJob).order_by(DownloadJob.created_at.desc())
    if user_id:
        query = query.filter(DownloadJob.user_id == user_id)
    return query.limit(limit).all()


def delete_download_job(db, job_id: str) -> bool:
    job = db.query(DownloadJob).filter(DownloadJob.id == job_id).first()
    if job:
        db.delete(job)
        db.commit()
        return True
    return False

def create_extract_job(db, job_id: str, video_path: str, filename: str, user_id: str = None, **kwargs):
    job = ExtractJob(id=job_id, video_path=video_path, filename=filename, user_id=user_id, **kwargs)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_extract_job(db, job_id: str, **kwargs):
    job = db.query(ExtractJob).filter(ExtractJob.id == job_id).first()
    if job:
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        db.commit()
        db.refresh(job)
    return job


def get_extract_job(db, job_id: str):
    return db.query(ExtractJob).filter(ExtractJob.id == job_id).first()


def get_extract_history(db, limit: int = 20, user_id: str = None):
    query = db.query(ExtractJob).order_by(ExtractJob.created_at.desc())
    if user_id:
        query = query.filter(ExtractJob.user_id == user_id)
    return query.limit(limit).all()


def delete_extract_job(db, job_id: str) -> bool:
    job = db.query(ExtractJob).filter(ExtractJob.id == job_id).first()
    if job:
        db.delete(job)
        db.commit()
        return True
    return False
