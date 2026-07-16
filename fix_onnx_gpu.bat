@echo off
title Sua loi xung dot ONNX Runtime - Sakai Studio
echo ========================================================
echo   DANG SUA LOI XUNG DOT ONNX RUNTIME DE KICH HOAT GPU
echo ========================================================
echo.
echo   [QUAN TRONG] Vui long dam bao ung dung Video Subtitle Remover
echo   da duoc DONG hoan toan truoc khi tiep tuc.
echo.
pause
echo.
echo   1. Dang go cai dat onnxruntime (ban CPU)...
venv\Scripts\pip.exe uninstall -y onnxruntime
echo.
echo   2. Dang cai dat lai onnxruntime-gpu==1.17.1 (tuong thich CUDA 11.8)...
venv\Scripts\pip.exe install --force-reinstall onnxruntime-gpu==1.17.1
echo.
echo   3. Dang ha cap numpy ve phien ban 1.x...
venv\Scripts\pip.exe install "numpy<2"
echo.
echo   4. Dang ha cap scipy ve phien ban 1.12.0...
venv\Scripts\pip.exe install "scipy==1.12.0"
echo.
echo ========================================================
echo   HOAN THANH!
echo   Xung dot da duoc loai bo. Bay gio ONNX Runtime se su dung
echo   goi onnxruntime-gpu de chay o toc do cao nhat (GPU CUDA).
echo ========================================================
echo.
pause
