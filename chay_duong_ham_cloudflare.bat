@echo off
chcp 65001 > nul
echo ===================================================================
echo       KHỞI CHẠY ĐƯỜNG HẦM CLOUDFLARE CHO SAKAI STUDIO
echo ===================================================================
echo.
echo Hệ thống đang kiểm tra cổng giao diện web cục bộ...
echo Đảm bảo bạn đã chạy tệp start_saas.bat trước khi mở đường hầm.
echo.
echo Đang kết nối đường hầm Cloudflare với cổng 3000...
echo Vui lòng đợi trong giây lát để hệ thống tạo đường dẫn bảo mật...
echo.
echo ===================================================================
echo HƯỚNG DẪN XEM LIÊN KẾT:
echo Khi màn hình hiện ra dòng chữ chứa đuôi:
echo https://...trycloudflare.com
echo Hãy sao chép liên kết đó và gửi cho bạn bè hoặc mở trên điện thoại!
echo ===================================================================
echo.
cloudflared.exe tunnel --url http://localhost:3000
echo.
pause
