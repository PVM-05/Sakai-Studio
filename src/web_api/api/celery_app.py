import os
import socket
import logging
from celery import Celery

logger = logging.getLogger("sakai_saas.celery")

def is_redis_available(host="localhost", port=6379, timeout=0.8) -> bool:
    """Kiểm tra nhanh xem máy chủ Redis có đang hoạt động hay không."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

# Xác định đường dẫn thư mục lưu cơ sở dữ liệu dự phòng
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sqlite_broker_path = os.path.join(BASE_DIR, "celery_broker.db").replace("\\", "/")
sqlite_backend_path = os.path.join(BASE_DIR, "celery_backend.db").replace("\\", "/")

# Cấu hình tự động nhận diện Redis hoặc chuyển sang SQLite
env_broker = os.getenv("CELERY_BROKER_URL")

if env_broker:
    BROKER_URL = env_broker
    BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", env_broker)
elif is_redis_available():
    BROKER_URL = "redis://localhost:6379/0"
    BACKEND_URL = "redis://localhost:6379/0"
    print(">> Đã phát hiện máy chủ Redis: Sử dụng Redis làm cầu nối hàng đợi.")
else:
    BROKER_URL = f"sqla+sqlite:///{sqlite_broker_path}"
    BACKEND_URL = f"db+sqlite:///{sqlite_backend_path}"
    print(f">> Chưa phát hiện Redis trên cổng 6379: Tự động chuyển sang cầu nối SQLite cục bộ ({sqlite_broker_path}).")

celery_app = Celery(
    "sakai_saas_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=[
        "src.web_api.api.worker_tasks",
        "src.web_api.api.ytdlp_worker",
        "src.web_api.api.extract_worker",
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1
)
