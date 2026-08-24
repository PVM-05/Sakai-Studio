@echo off
echo ===================================================
echo     KHOI DONG HE THONG SAAS - SAKAI STUDIO
echo ===================================================

echo [1] Kiem tra Redis Server (Bat buoc de Celery hoat dong)
echo Xin dam bao ban da chay Redis (qua Docker hoac WSL tren Windows).
echo Neu chua co Redis, Celery Worker se bao loi ket noi.
echo.

echo [2] Khoi dong FastAPI Backend (Port 8000)...
start "FastAPI Server" cmd /k "venv\Scripts\activate && uvicorn src.web_api.api.main:app --reload --port 8000"
timeout /t 2 /nobreak > NUL

echo [3] Khoi dong Celery AI Worker...
start "Celery Worker" cmd /k "venv\Scripts\activate && celery -A src.web_api.api.worker_tasks worker --loglevel=info -P gevent"
timeout /t 2 /nobreak > NUL

echo [4] Khoi dong Next.js Frontend (Port 3000)...
start "Next.js Web UI" cmd /k "cd web_client && npm run dev"

echo.
echo ===================================================
echo     HE THONG SAAS DA KHOI CHAY THANH CONG!
echo ===================================================
echo - Trang chu Web: http://localhost:3000
echo - API Backend: http://localhost:8000/docs
echo.
pause
