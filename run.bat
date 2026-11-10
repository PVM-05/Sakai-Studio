@echo off
title Sakai Studio
cd /d "%~dp0"

echo ===================================================
echo   Sakai Studio - Tram Xu Ly Video va Phu De AI
echo ===================================================
echo.

set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"
set "SYS_PYTHON=python"

if exist "%VENV_PYTHON%" (
    echo Dang su dung moi truong ao: venv
    set "PY_CMD=%VENV_PYTHON%"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo Canh bao: Khong tim thay thu muc moi truong ao venv.
        echo Dang thu khoi chay bang Python he thong...
        set "PY_CMD=%SYS_PYTHON%"
    ) else (
        echo Loi: Khong tim thay Python hoac thu muc venv.
        echo Vui long cai dat Python hoac tao moi truong ao truoc khi chay.
        echo.
        pause
        exit /b 1
    )
)

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;auto

echo Dang khoi chay giao dien Sakai Studio...
echo.

"%PY_CMD%" -m src.desktop.main %*

if %errorlevel% neq 0 (
    echo.
    echo ===================================================
    echo   Da phat sinh loi trong qua trinh thuc thi!
    echo   Ma loi ket thuc: %errorlevel%
    echo ===================================================
    echo.
    pause
    exit /b %errorlevel%
)

exit /b 0
