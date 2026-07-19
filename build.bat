@echo off
title Sakai Studio - Executable Builder
echo ===================================================
echo   Sakai Studio - Executable Builder
echo ===================================================
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please make sure you have created the virtual environment and installed requirements.
    goto error
)

echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/2] Running PyInstaller build...
pyinstaller SakaiStudio.spec --noconfirm

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyInstaller build failed!
    goto error
)

echo.
echo ===================================================
echo   BUILD COMPLETED SUCCESSFULLY!
echo   Output executable is at:
echo   dist\SakaiStudio\SakaiStudio.exe
echo ===================================================
echo.
pause
exit /b 0

:error
echo.
echo ===================================================
echo   BUILD FAILED! Please check the logs above.
echo ===================================================
echo.
pause
exit /b 1
