@echo off
echo Dang khoi dong Video Subtitle Remover...

:: Kích hoạt môi trường ảo 
call venv\Scripts\activate

:: Chạy giao diện phần mềm
python gui.py

:: Tạm dừng nếu có lỗi để bạn kịp đọc log
pause
