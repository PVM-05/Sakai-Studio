@echo off
title Sakai Studio - Demo Executable Builder
echo ===================================================
echo   Sakai Studio - Demo Executable Builder
echo ===================================================
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please make sure you have created the virtual environment and installed requirements.
    goto error
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Running build_demo.py script...
python build_demo.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python demo build script failed!
    goto error
)

echo [3/3] Creating zip package for SakaiStudioDemo...
powershell -Command "if (Test-Path 'dist/SakaiStudioDemo') { Compress-Archive -Path 'dist/SakaiStudioDemo' -DestinationPath 'dist/SakaiStudioDemo-v3.18.0-windows-cuda.zip' -Force; Write-Host 'Zip archive created successfully!' } else { Write-Error 'dist/SakaiStudioDemo directory not found!' }"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Zipping demo directory failed!
    goto error
)

echo.
echo ===================================================
echo   DEMO BUILD COMPLETED SUCCESSFULLY!
echo   Output executable: dist\SakaiStudioDemo\SakaiStudioDemo.exe
echo   Zip file: dist\SakaiStudioDemo-v3.18.0-windows-cuda.zip
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
