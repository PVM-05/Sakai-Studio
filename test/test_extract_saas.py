import sys
import os
import json
import uuid
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web_api.api.database import (
    init_db, SessionLocal, ExtractJob,
    create_extract_job, update_extract_job,
    get_extract_job, get_extract_history, delete_extract_job
)
from src.core.tools.subtitle_exporter import SubtitleExporter, frame_to_timestamp_srt, frame_to_timestamp_ass


class TestExtractDatabase:
    """Kiểm thử mô hình dữ liệu ExtractJob."""

    def test_create_and_query_extract_job(self):
        init_db()
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            job = create_extract_job(
                db, job_id,
                video_path="/path/to/test.mp4",
                filename="test.mp4",
                user_id="test_user",
                ocr_mode="auto",
                ocr_lang="vi"
            )
            assert job.id == job_id
            assert job.status == "Đang chờ"
            assert job.progress == 0
            assert job.ocr_lang == "vi"
        finally:
            db.close()

    def test_update_extract_job_progress(self):
        init_db()
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            create_extract_job(db, job_id, video_path="/path/to/test.mp4", filename="test.mp4")
            updated = update_extract_job(
                db, job_id,
                status="Đang nhận diện",
                progress=45,
                current_step="Đang quét chữ trên khung hình..."
            )
            assert updated.status == "Đang nhận diện"
            assert updated.progress == 45
            assert "quét chữ" in updated.current_step
        finally:
            db.close()

    def test_extract_history(self):
        init_db()
        db = SessionLocal()
        try:
            test_uid = f"user_{uuid.uuid4().hex[:6]}"
            for i in range(3):
                create_extract_job(
                    db, str(uuid.uuid4()),
                    video_path=f"/path/{i}.mp4",
                    filename=f"{i}.mp4",
                    user_id=test_uid
                )
            history = get_extract_history(db, limit=10, user_id=test_uid)
            assert len(history) == 3
        finally:
            db.close()

    def test_delete_extract_job(self):
        init_db()
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            create_extract_job(db, job_id, video_path="/path/to/del.mp4", filename="del.mp4")
            assert delete_extract_job(db, job_id) is True
            assert delete_extract_job(db, job_id) is False
        finally:
            db.close()


class TestSubtitleExporter:
    """Kiểm thử bộ xuất phụ đề SubtitleExporter."""

    def test_frame_to_timestamp_srt(self):
        # 30 fps, frame 30 = 1s -> 00:00:01,000
        ts = frame_to_timestamp_srt(30, 30.0)
        assert ts == "00:00:01,000"

    def test_frame_to_timestamp_ass(self):
        ts = frame_to_timestamp_ass(30, 30.0)
        assert ts == "0:00:01.00"

    def test_export_srt(self):
        items = [
            {'start_frame': 0, 'end_frame': 60, 'text': 'Xin chào Sakai Studio'},
            {'start_frame': 61, 'end_frame': 120, 'text': 'Trích xuất phụ đề video'}
        ]
        tmp_dir = tempfile.mkdtemp()
        srt_path = os.path.join(tmp_dir, "output.srt")
        try:
            SubtitleExporter.export_srt(srt_path, items, fps=30.0)
            assert os.path.exists(srt_path)
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Xin chào Sakai Studio" in content
            assert "00:00:00,000 --> 00:00:02,000" in content
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_export_ass(self):
        items = [
            {'start_frame': 0, 'end_frame': 60, 'text': 'Thử nghiệm phụ đề kiểu ASS'}
        ]
        tmp_dir = tempfile.mkdtemp()
        ass_path = os.path.join(tmp_dir, "output.ass")
        try:
            SubtitleExporter.export_ass(ass_path, items, fps=30.0)
            assert os.path.exists(ass_path)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "[Script Info]" in content
            assert "Thử nghiệm phụ đề kiểu ASS" in content
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestExtractApiEndpoints:
    """Kiểm thử các cổng API trích xuất phụ đề."""

    def test_extract_status_not_found(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        with TestClient(app) as client:
            res = client.get("/extract/status/non-existent-uuid")
            assert res.status_code == 404

    def test_extract_history_endpoint(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        with TestClient(app) as client:
            res = client.get("/extract/history")
            assert res.status_code == 200
            assert isinstance(res.json(), list)

    def test_extract_start_with_missing_file(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        with TestClient(app) as client:
            res = client.post("/extract/start", data={"filename": "khong_ton_tai_12345.mp4"})
            assert res.status_code == 404

    def test_extract_result_and_delete_flow(self):
        from fastapi.testclient import TestClient
        from src.web_api.api.main import app

        init_db()
        db = SessionLocal()
        job_id = str(uuid.uuid4())
        try:
            subtitles = [{"index": 1, "start_time": "00:00:01,000", "end_time": "00:00:03,000", "text": "Dòng phụ đề mẫu", "source": "ocr"}]
            create_extract_job(
                db, job_id,
                video_path="/mock/path.mp4",
                filename="mock.mp4",
                status="Hoàn thành",
                progress=100,
                subtitle_count=1,
                subtitles_json=json.dumps(subtitles, ensure_ascii=False)
            )
        finally:
            db.close()

        with TestClient(app) as client:
            res = client.get(f"/extract/result/{job_id}")
            assert res.status_code == 200
            data = res.json()
            assert data["subtitle_count"] == 1
            assert len(data["subtitles"]) == 1
            assert data["subtitles"][0]["text"] == "Dòng phụ đề mẫu"

            del_res = client.delete(f"/extract/history/{job_id}")
            assert del_res.status_code == 200
