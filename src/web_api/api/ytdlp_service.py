"""
Lop dich vu boc thu vien yt-dlp cho nen tang SaaS.
Tach biet hoan toan khoi giao dien desktop.
"""
import re
import os
import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

logger = logging.getLogger("sakai_saas.ytdlp")

DOWNLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'downloads')
)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class YtdlpService:
    """Dich vu xu ly yt-dlp khong phu thuoc giao dien do hoa."""

    @staticmethod
    def normalize_url(url: str) -> str:
        """Chuan hoa lien ket nguon, loai bo thong tin thua."""
        url = url.strip()
        url_match = re.search(r'https?://\S+', url)
        if url_match:
            url = url_match.group(0)

        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            params = parse_qs(parsed.query)
            list_id = params.get('list', [''])[0]
            if list_id.startswith(('RD', 'UL')):
                params.pop('list', None)
                params.pop('index', None)
                new_query = urlencode({k: v[0] for k, v in params.items()})
                url = urlunparse(parsed._replace(query=new_query))

        if 'douyin.com' in url:
            modal_match = re.search(r'modal_id=(\d+)', url)
            if modal_match:
                url = f"https://www.douyin.com/video/{modal_match.group(1)}"

        return url

    @staticmethod
    def analyze_url(url: str, cookies_file: Optional[str] = None) -> dict:
        """Phan tich lien ket, tra ve thong tin video hoac danh sach phat."""
        import yt_dlp

        ydl_opts = {
            'noplaylist': False,
            'extract_flat': 'in_playlist',
            'no_warnings': True,
            'quiet': True,
            'no_color': True,
            'allow_unverified_js': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/131.0.0.0 Safari/537.36',
            },
        }

        if cookies_file and os.path.isfile(cookies_file):
            ydl_opts['cookiefile'] = cookies_file

        normalized = YtdlpService.normalize_url(url)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized, download=False)

        if info is None:
            raise ValueError("Khong the trich xuat thong tin tu lien ket nay")

        result = {
            'url': normalized,
            'title': info.get('title', 'Khong ro tieu de'),
            'channel': info.get('uploader') or info.get('channel', ''),
            'duration': info.get('duration', 0),
            'thumbnail': info.get('thumbnail', ''),
            'is_playlist': False,
            'formats': [],
            'playlist_items': [],
        }

        entries = info.get('entries')
        if entries is not None:
            result['is_playlist'] = True
            result['playlist_title'] = info.get('title', '')
            items = []
            for idx, entry in enumerate(entries or []):
                if entry is None:
                    continue
                items.append({
                    'index': idx + 1,
                    'url': entry.get('url', ''),
                    'title': entry.get('title', f'Video {idx + 1}'),
                    'duration': entry.get('duration', 0),
                })
            result['playlist_items'] = items
            return result

        formats = []
        seen = set()
        for f in info.get('formats', []):
            vcodec = f.get('vcodec', 'none')
            height = f.get('height')
            format_id = f.get('format_id', '')

            if vcodec == 'none' or not height:
                continue

            label = f"{height}p"
            fps = f.get('fps')
            if fps and fps > 30:
                label += f" {fps}fps"

            if label in seen:
                continue
            seen.add(label)

            formats.append({
                'format_id': format_id,
                'resolution': label,
                'height': height,
                'fps': fps or 30,
                'ext': f.get('ext', 'mp4'),
                'filesize': f.get('filesize') or f.get('filesize_approx', 0),
                'vcodec': vcodec,
            })

        formats.sort(key=lambda x: (x['height'], x['fps']), reverse=True)
        result['formats'] = formats
        return result

    @staticmethod
    def build_download_options(
        output_path: str,
        format_id: str = None,
        container: str = "mp4",
        audio_only: bool = False,
        audio_format: str = "mp3",
        concurrent_fragments: int = 4,
        cookies_file: Optional[str] = None,
        progress_hook=None,
    ) -> dict:
        """Tao tham so cau hinh cho yt-dlp khi tai xuong."""
        opts = {
            'outtmpl': output_path,
            'noprogress': True,
            'noplaylist': True,
            'no_warnings': True,
            'quiet': True,
            'no_color': True,
            'allow_unverified_js': True,
            'concurrent_fragment_downloads': concurrent_fragments,
            'http_chunk_size': 10485760,
            'retries': 10,
            'fragment_retries': 10,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/131.0.0.0 Safari/537.36',
            },
        }

        if cookies_file and os.path.isfile(cookies_file):
            opts['cookiefile'] = cookies_file

        if progress_hook:
            opts['progress_hooks'] = [progress_hook]

        if audio_only:
            opts['format'] = 'bestaudio/best'
            opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_format,
                'preferredquality': '0' if audio_format in ('wav', 'flac') else '192',
            }]
        else:
            if format_id:
                opts['format'] = f'{format_id}+bestaudio/best'
            else:
                opts['format'] = 'bestvideo+bestaudio/best'
            opts['merge_output_format'] = container

        return opts

    @staticmethod
    def translate_error(err: str) -> str:
        """Dich thong bao loi ky thuat yt-dlp sang tieng Viet than thien."""
        err = re.sub(r'\x1b\[[0-9;]*m', '', err)

        error_map = [
            ('Connection reset', 'Máy chủ nguồn ngắt kết nối đột ngột. Vui lòng thử tải lại.'),
            ('HTTP Error 403', 'Truy cập bị từ chối. Vui lòng bổ sung tệp xác thực danh tính.'),
            ('HTTP Error 404', 'Không tìm thấy video. Liên kết có thể sai hoặc video đã bị gỡ bỏ.'),
            ('Private video', 'Video ở chế độ riêng tư. Cần đăng nhập tài khoản có quyền xem.'),
            ('age-restricted', 'Video giới hạn độ tuổi. Cần cung cấp tệp xác thực danh tính.'),
            ('cookies', 'Cần bổ sung tệp xác thực danh tính để tải nội dung này.'),
            ('Unsupported URL', 'Liên kết không được hỗ trợ hoặc không chứa nội dung video.'),
            ('No video formats', 'Không tìm thấy luồng dữ liệu phương tiện nào phù hợp.'),
            ('disk', 'Ổ đĩa lưu trữ đã đầy. Vui lòng dọn dẹp hoặc đổi thư mục lưu.'),
            ('timed out', 'Kết nối mạng quá chậm hoặc đã hết thời gian chờ.'),
            ('Network', 'Lỗi kết nối mạng. Vui lòng kiểm tra đường truyền.'),
            ('Sign in', 'Video yêu cầu đăng nhập. Vui lòng cung cấp tệp xác thực.'),
        ]

        for keyword, message in error_map:
            if keyword.lower() in err.lower():
                return message

        return f"Đã xảy ra lỗi khi xử lý: {err[:200]}"

    @staticmethod
    def get_unique_filename(directory: str, filename: str) -> str:
        """Tao ten tep duy nhat, them hau to so neu trung."""
        base, ext = os.path.splitext(filename)
        base = re.sub(r'[<>:"/\\|?*]', '', base).strip()
        if not base:
            base = "video"

        candidate = os.path.join(directory, f"{base}{ext}")
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{base} ({counter}){ext}")
            counter += 1
        return candidate
