import sys
import os
import uuid
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web_api.api.database import (
    init_db, SessionLocal, DownloadJob,
    create_download_job, update_download_job,
    get_download_history, delete_download_job
)
from src.web_api.api.ytdlp_service import YtdlpService


class TestYtdlpDatabase:
    """Kiem thu mo hinh du lieu DownloadJob."""

    def test_create_and_query_download_job(self):
        init_db()
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            job = create_download_job(
                db, job_id,
                url="https://www.youtube.com/watch?v=test123",
                user_id="test_user",
                title="Video kiem thu",
                container="mp4"
            )
            assert job.id == job_id
            assert job.status == "Đang chờ"
            assert job.progress == 0
        finally:
            db.close()

    def test_update_download_job_progress(self):
        init_db()
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            create_download_job(db, job_id, url="https://example.com/video")
            updated = update_download_job(db, job_id, progress=50, status="Đang tải", speed="2.5 MB/giây")
            assert updated.progress == 50
            assert updated.speed == "2.5 MB/giây"
        finally:
            db.close()

    def test_download_history(self):
        init_db()
        db = SessionLocal()
        try:
            test_uid = f"user_{uuid.uuid4().hex[:6]}"
            for i in range(3):
                create_download_job(db, str(uuid.uuid4()), url=f"https://example.com/{i}", user_id=test_uid)
            history = get_download_history(db, limit=10, user_id=test_uid)
            assert len(history) == 3
        finally:
            db.close()

    def test_delete_download_job(self):
        init_db()
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            create_download_job(db, job_id, url="https://example.com/delete_me")
            assert delete_download_job(db, job_id) is True
            assert delete_download_job(db, job_id) is False
        finally:
            db.close()


class TestYtdlpService:
    """Kiem thu lop dich vu YtdlpService."""

    def test_normalize_youtube_url(self):
        url = "https://www.youtube.com/watch?v=abc123&list=RDxyz"
        result = YtdlpService.normalize_url(url)
        assert "list=" not in result
        assert "v=abc123" in result

    def test_normalize_url_from_share_text(self):
        text = "Xem video hay ne https://youtu.be/abc123 chia se cho ban"
        result = YtdlpService.normalize_url(text)
        assert result == "https://youtu.be/abc123"

    def test_normalize_douyin_url(self):
        url = "https://www.douyin.com/search?modal_id=123456789"
        result = YtdlpService.normalize_url(url)
        assert result == "https://www.douyin.com/video/123456789"

    def test_translate_error_403(self):
        msg = YtdlpService.translate_error("HTTP Error 403: Forbidden")
        assert "từ chối" in msg.lower()

    def test_translate_error_network(self):
        msg = YtdlpService.translate_error("Network is unreachable")
        assert "mạng" in msg.lower() or "kết nối" in msg.lower()

    def test_get_unique_filename(self):
        tmpdir = tempfile.mkdtemp()
        try:
            open(os.path.join(tmpdir, "video.mp4"), 'w').close()
            result = YtdlpService.get_unique_filename(tmpdir, "video.mp4")
            assert "video (1).mp4" in result
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestYtdlpApiEndpoints:
    """Kiem thu cac cong API tai video."""

    def test_analyze_endpoint_invalid_url(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        with TestClient(app) as client:
            res = client.post("/ytdlp/analyze", data={"url": "not-a-valid-url"})
            assert res.status_code == 400

    def test_history_endpoint(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        with TestClient(app) as client:
            res = client.get("/ytdlp/history")
            assert res.status_code == 200
            assert isinstance(res.json(), list)

    def test_cancel_endpoint(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        with TestClient(app) as client:
            init_db()
            db = SessionLocal()
            job_id = str(uuid.uuid4())
            try:
                create_download_job(db, job_id, url="https://example.com/cancel_test", status="Đang tải")
            finally:
                db.close()

            res = client.post(f"/ytdlp/cancel/{job_id}")
            assert res.status_code == 200
            assert res.json()["status"] == "Đã hủy tác vụ tải"

            db = SessionLocal()
            try:
                job = db.query(DownloadJob).filter(DownloadJob.id == job_id).first()
                assert job.status == "Đã hủy"
            finally:
                db.close()
