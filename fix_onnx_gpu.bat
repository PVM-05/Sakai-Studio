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
echo   Dang go cai dat onnxruntime (ban CPU)...
venv\Scripts\pip.exe uninstall -y onnxruntime
echo.
echo ========================================================
echo   HOAN THANH!
echo   Xung dot da duoc loai bo. Bay gio ONNX Runtime se su dung
echo   goi onnxruntime-gpu de chay o toc do cao nhat (GPU CUDA).
echo ========================================================
echo.
pause
