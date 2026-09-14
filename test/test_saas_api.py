import os
import sys
import json
import pytest
import tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def test_database_models_and_helpers():
    from src.web_api.api.database import (
        Base, TaskJob, User, init_db,
        create_task_job, update_task_job_progress,
        get_task_job, get_recent_jobs
    )
    
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSession()

    try:
        user = User(username="test_user", email="test@example.com")
        db.add(user)
        db.commit()

        job = create_task_job(
            db=db,
            job_id="job-123",
            filename="sample.mp4",
            original_path="/path/to/sample.mp4",
            user_id="test_user",
            sub_areas_json="[[0, 100, 0, 200]]"
        )
        assert job.id == "job-123"
        assert job.filename == "sample.mp4"
        assert job.status == "Đang chờ"
        assert job.progress == 0

        updated_job = update_task_job_progress(
            db=db,
            job_id="job-123",
            status="Đang xử lý",
            progress=50
        )
        assert updated_job.status == "Đang xử lý"
        assert updated_job.progress == 50

        completed_job = update_task_job_progress(
            db=db,
            job_id="job-123",
            status="Hoàn thành",
            progress=100,
            output_path="/path/to/sample_no_sub.mp4"
        )
        assert completed_job.status == "Hoàn thành"
        assert completed_job.progress == 100
        assert completed_job.output_path == "/path/to/sample_no_sub.mp4"
        assert completed_job.completed_at is not None

        jobs = get_recent_jobs(db, limit=5)
        assert len(jobs) == 1
        assert jobs[0].id == "job-123"
    finally:
        db.close()


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from src.web_api.api.main import app, convert_boxes_to_sub_areas

    with TestClient(app) as client:
        # 1. Test convert_boxes_to_sub_areas with normalized and absolute boxes
        boxes_normalized = [
            {"x_pct": 0.1, "y_pct": 0.2, "w_pct": 0.3, "h_pct": 0.4}
        ]
        sub_areas = convert_boxes_to_sub_areas(boxes_normalized, video_width=1000, video_height=500)
        assert len(sub_areas) == 1
        # ymin, ymax, xmin, xmax
        assert sub_areas[0] == [100, 300, 100, 400]

        boxes_legacy_normalized = [
            {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
        ]
        sub_areas_legacy = convert_boxes_to_sub_areas(boxes_legacy_normalized, video_width=1000, video_height=500)
        assert sub_areas_legacy[0] == [100, 300, 100, 400]

        boxes_absolute = [
            {"x": 50, "y": 100, "w": 200, "h": 150}
        ]
        sub_areas_abs = convert_boxes_to_sub_areas(boxes_absolute, video_width=1000, video_height=500)
        assert sub_areas_abs[0] == [100, 250, 50, 250]

        # 2. Test GET /projects endpoint
        res = client.get("/projects")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

        # 3. Test GET /events/{task_id} with mock task
        with client.stream("GET", "/events/nonexistent-task") as stream:
            assert stream.status_code == 200
            assert "text/event-stream" in stream.headers["content-type"]
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    assert data["task_id"] == "nonexistent-task"
                    break

        # 4. Test POST /detect-subtitles with test video
        from src.web_api.api.main import UPLOAD_DIR
        import shutil
        test_video_src = os.path.join(os.path.dirname(__file__), "test2.mp4")
        if os.path.exists(test_video_src):
            test_target = os.path.join(UPLOAD_DIR, "test2.mp4")
            shutil.copyfile(test_video_src, test_target)
            res_detect = client.post("/detect-subtitles", data={"filename": "test2.mp4", "timestamp": 0.5})
            assert res_detect.status_code == 200
            data_detect = res_detect.json()
            assert "video_width" in data_detect
            assert "video_height" in data_detect
            assert "boxes" in data_detect
            assert isinstance(data_detect["boxes"], list)
