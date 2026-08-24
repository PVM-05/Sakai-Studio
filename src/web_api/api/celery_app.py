import os
from celery import Celery

# Use Redis as the message broker
# Fallback to localhost if environment variable is not set
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "sakai_saas_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["api.worker_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1 # Very important for long running AI tasks so workers don't hog tasks
)
