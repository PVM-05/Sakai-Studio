# -*- coding: utf-8 -*-
"""
@desc: YT-DLP Video Downloader Interface with Metadata Display, Action Links, Brand Logos, Download Queue, History, and Auto-Clipboard Paste
"""
import os
import json
import shutil
import urllib.request
import base64
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QVBoxLayout, QGridLayout, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from qfluentwidgets import (ScrollArea, CardWidget, LineEdit, PushButton, 
                           PrimaryPushButton, ComboBox, ProgressBar, PlainTextEdit,
                           BodyLabel, TitleLabel, FluentIcon, InfoBar, TableWidget,
                           TransparentToolButton)
import yt_dlp
from backend.config import config, tr
from backend.tools.folder_memory import FolderMemoryDialog

SETTINGS_FILE = 'config/ytdlp_settings.json'

SVG_ICONS = {
    'youtube.svg': '<svg viewBox="0 0 24 24" fill="#FF0000" xmlns="http://www.w3.org/2000/svg"><path d="M23.498 6.163a3.003 3.003 0 0 0-2.11-2.108C19.53 3.5 12 3.5 12 3.5s-7.53 0-9.388.555A3.003 3.003 0 0 0 .502 6.163C0 8.07 0 12 0 12s0 3.93.502 5.837a3.003 3.003 0 0 0 2.11 2.108C4.47 20.5 12 20.5 12 20.5s7.53 0 9.388-.555a3.003 3.003 0 0 0 2.11-2.108C24 15.93 24 12 24 12s0-3.93-.502-5.837zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>',
    'tiktok.svg': '<svg viewBox="0 0 24 24" fill="#010101" xmlns="http://www.w3.org/2000/svg"><path d="M12.525.02c1.31-.02 2.61-.01 3.91-.02.08 1.53.63 3.02 1.59 4.23.99 1.18 2.37 1.93 3.86 2.14v3.83c-1.63-.09-3.21-.73-4.48-1.78-.17-.14-.33-.29-.48-.45v6.52c.01 1.76-.56 3.49-1.63 4.84a8.163 8.163 0 0 1-8.15 2.58A8.16 8.16 0 0 1 .535 14.1c-.69-2.31.06-4.88 1.84-6.52a8.15 8.15 0 0 1 7.7-1.87v3.91c-.81-.32-1.72-.25-2.47.19-.78.47-1.28 1.31-1.35 2.22-.09 1.22.65 2.36 1.77 2.76 1.09.39 2.33-.03 2.92-1.02.26-.44.39-.95.39-1.47V.02h.22z"/></svg>',
    'facebook.svg': '<svg viewBox="0 0 24 24" fill="#1877F2" xmlns="http://www.w3.org/2000/svg"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>',
    'instagram.svg': '<svg viewBox="0 0 24 24" fill="none" stroke="#E1306C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg>',
    'twitter.svg': '<svg viewBox="0 0 24 24" fill="#0f1419" xmlns="http://www.w3.org/2000/svg"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>',
    'twitch.svg': '<svg viewBox="0 0 24 24" fill="#9146FF" xmlns="http://www.w3.org/2000/svg"><path d="M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714Z"/></svg>',
    'bilibili.svg': '<svg viewBox="0 0 24 24" fill="#00A1D6" xmlns="http://www.w3.org/2000/svg"><path d="M17.87 2.05a.76.76 0 0 0-.69.07l-3.22 2.37H10l-3.2-2.37a.76.76 0 0 0-1.07.19.78.78 0 0 0 .19 1.07l2.87 2.13h-2.1a4.67 4.67 0 0 0-4.67 4.67v6.62a4.67 4.67 0 0 0 4.67 4.67h10.66a4.67 4.67 0 0 0 4.67-4.67v-6.62a4.67 4.67 0 0 0-4.67-4.67h-2.13l2.87-2.13a.78.78 0 0 0-.09-1.28l-.07-.05ZM5.68 7.21h12.64A2.39 2.39 0 0 1 20.7 9.6v6.62a2.39 2.39 0 0 1-2.39 2.39H5.68A2.39 2.39 0 0 1 3.3 16.22V9.6A2.39 2.39 0 0 1 5.68 7.21Zm2.3 3.61a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm8.04 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/></svg>'
}

def ensure_icons():
    icon_dir = 'assets/icons'
    os.makedirs(icon_dir, exist_ok=True)
    for fname, content in SVG_ICONS.items():
        fpath = os.path.join(icon_dir, fname)
        if not os.path.exists(fpath):
            try:
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"Error writing icon {fname}:", e)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading yt-dlp settings:", e)
    return {}

def save_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print("Error saving yt-dlp settings:", e)


import re
import urllib.parse

def normalize_url(url):
    """Normalize and clean social media URLs (Douyin, TikTok, YouTube, etc.) for yt-dlp."""
    if not url:
        return url
    url = url.strip()
    # Extract HTTP/HTTPS link if text contains extra share text
    match = re.search(r'https?://[^\s]+', url)
    if match:
        url = match.group(0)
    # Douyin Modal / User / Search URL normalizer
    if 'douyin.com' in url.lower():
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        modal_id = query.get('modal_id', [None])[0]
        if modal_id:
            return f"https://www.douyin.com/video/{modal_id}"
    return url


def translate_ytdlp_error(err):
    """Translate raw technical yt-dlp tracebacks into user-friendly Vietnamese guidance."""
    err_str = str(err)
    err_lower = err_str.lower()
    
    if '403' in err_lower or 'forbidden' in err_lower or 'caller does not have permission' in err_lower:
        return "Bị trang web/YouTube từ chối truy cập (HTTP 403 Forbidden). Vui lòng sử dụng file cookies.txt để xác thực tài khoản."
    elif 'fresh cookies' in err_lower or ('douyin' in err_lower and 'cookies' in err_lower):
        return "Douyin bảo mật cao yêu cầu cookies. Vui lòng chọn file cookies.txt của bạn (được xuất từ trình duyệt sau khi mở Douyin)."
    elif 'private' in err_lower or 'sign in' in err_lower or 'age' in err_lower or 'login' in err_lower:
        return "Video riêng tư hoặc bị giới hạn độ tuổi. Vui lòng chọn file cookies.txt của bạn để tiếp tục."
    elif '404' in err_lower or 'not found' in err_lower:
        return "Không tìm thấy video (HTTP 404). Đường dẫn bị sai hoặc video đã bị xoá khỏi hệ thống."
    elif 'incomplete yt initial data' in err_lower or 'giving up after' in err_lower:
        return "Lỗi phản hồi cấu trúc dữ liệu từ YouTube. Vui lòng nhấn Phân tích lại hoặc thêm file cookies.txt."
    elif 'unsupported url' in err_lower or 'is not a valid url' in err_lower:
        return "Đường dẫn link không hợp lệ hoặc trang web này chưa được hỗ trợ."
    elif 'no video formats found' in err_lower:
        return "Không tìm thấy luồng video/âm thanh sẵn có trên trang này."
    elif 'disk' in err_lower or 'space' in err_lower or 'full' in err_lower:
        return "Dung lượng ổ đĩa lưu trữ đã đầy. Vui lòng dọn dẹp hoặc chọn thư mục lưu khác."
    elif 'network' in err_lower or 'connection' in err_lower or 'timed out' in err_lower or 'unreachable' in err_lower:
        return "Lỗi kết nối mạng (Timeout/Network Error). Vui lòng kiểm tra lại kết nối internet của bạn."
    else:
        lines = [line.strip() for line in err_str.strip().split('\n') if line.strip()]
        last_line = lines[-1] if lines else err_str
        if len(last_line) > 120:
            last_line = last_line[:120] + "..."
        return f"Lỗi phân tích/tải video: {last_line}"


class HistoryActionWidget(QtWidgets.QWidget):
    def __init__(self, filepath, parent_interface, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.parent_interface = parent_interface
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        
        self.play_btn = TransparentToolButton(FluentIcon.PLAY, self)
        self.play_btn.setFixedSize(28, 28)
        self.play_btn.setToolTip("Phát video")
        self.play_btn.clicked.connect(self.play_video)
        
        self.folder_btn = TransparentToolButton(FluentIcon.FOLDER, self)
        self.folder_btn.setFixedSize(28, 28)
        self.folder_btn.setToolTip("Mở thư mục lưu")
        self.folder_btn.clicked.connect(self.open_folder)
        
        self.import_btn = TransparentToolButton(FluentIcon.MOVE, self)
        self.import_btn.setFixedSize(28, 28)
        self.import_btn.setToolTip("Mở trong Xoá phụ đề")
        self.import_btn.clicked.connect(self.import_to_remover)
        
        self.delete_btn = TransparentToolButton(FluentIcon.DELETE, self)
        self.delete_btn.setFixedSize(28, 28)
        self.delete_btn.setToolTip("Xoá mục này khỏi lịch sử")
        self.delete_btn.clicked.connect(self.delete_history_item)
        
        layout.addWidget(self.play_btn)
        layout.addWidget(self.folder_btn)
        
        _, ext = os.path.splitext(filepath)
        if ext.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm']:
            layout.addWidget(self.import_btn)
        else:
            self.import_btn.hide()
            
        layout.addWidget(self.delete_btn)
        layout.addStretch()
        
    def play_video(self):
        if os.path.exists(self.filepath):
            os.startfile(self.filepath)
            
    def open_folder(self):
        if os.path.exists(self.filepath):
            try:
                from showinfm import show_in_file_manager
                show_in_file_manager(self.filepath)
            except Exception:
                os.startfile(os.path.dirname(self.filepath))
                
    def import_to_remover(self):
        if os.path.exists(self.filepath):
            main_win = self.window()
            if hasattr(main_win, 'toolsInterface'):
                main_win.switchTo(main_win.toolsInterface)
                main_win.toolsInterface.open_video_in_remover(self.filepath)

    def delete_history_item(self):
        if hasattr(self.parent_interface, 'delete_single_history_item'):
            self.parent_interface.delete_single_history_item(self.filepath)


class ThumbnailDownloader(QThread):
    finished_sig = Signal(bytes)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                self.finished_sig.emit(response.read())
        except Exception as e:
            print("Error downloading thumbnail:", e)


class YtdlpAnalysisWorker(QThread):
    finished_sig = Signal(dict)
    error_sig = Signal(str)

    def __init__(self, url, cookiefile=None):
        super().__init__()
        self.url = url
        self.cookiefile = cookiefile

    def run(self):
        try:
            ydl_opts = {
                'noplaylist': False,
                'extract_flat': 'in_playlist',
                'allow_unverified_js': True,
                'remote_components': ['ejs:github'],
                'js_runtimes': {
                    'deno': {},
                    'node': {},
                    'quickjs': {}
                },
                'extractor_args': {
                    'tiktok': ['api_hostname=api16-normal-c-useast1a.tiktokv.com']
                }
            }
            if self.cookiefile and os.path.exists(self.cookiefile):
                ydl_opts['cookiefile'] = self.cookiefile

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                self.finished_sig.emit(info)
        except Exception as e:
            self.error_sig.emit(str(e))


class YtdlpWorker(QThread):
    progress_sig = Signal(float)       # Percentage (0.0 to 100.0)
    speed_sig = Signal(str)            # Speed (e.g. "1.2 MiB/s")
    eta_sig = Signal(str)              # ETA (e.g. "00:32")
    status_sig = Signal(str)           # "downloading", "finished", "error", "cancelled"
    log_sig = Signal(str)              # Console log messages
    error_sig = Signal(str)            # Error message
    finished_sig = Signal(str)         # Path of the downloaded file

    def __init__(self, url, save_dir, format_opt, selected_format_id=None, selected_format_type=None, cookiefile=None, preferred_container=None, preferred_audio_format=None, concurrent_fragments=4, custom_filename=None):
        super().__init__()
        self.url = url
        self.save_dir = save_dir
        self.format_opt = format_opt
        self.selected_format_id = selected_format_id
        self.selected_format_type = selected_format_type
        self.cookiefile = cookiefile
        self.preferred_container = preferred_container or 'mp4'
        self.preferred_audio_format = preferred_audio_format or 'mp3'
        self.concurrent_fragments = concurrent_fragments or 4
        self.custom_filename = custom_filename
        self._is_cancelled = False

    def run(self):
        def progress_hook(d):
            if self._is_cancelled:
                raise Exception("Download cancelled by user")
            
            if d['status'] == 'downloading':
                total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded_bytes = d.get('downloaded_bytes') or 0
                if total_bytes > 0:
                    pct = (downloaded_bytes / total_bytes) * 100.0
                    self.progress_sig.emit(pct)
                
                speed = d.get('speed')
                if speed:
                    if speed > 1024 * 1024:
                        speed_str = f"{speed / (1024*1024):.2f} MiB/s"
                    elif speed > 1024:
                        speed_str = f"{speed / 1024:.2f} KiB/s"
                    else:
                        speed_str = f"{speed:.2f} B/s"
                    self.speed_sig.emit(speed_str)
                
                eta = d.get('eta')
                if eta:
                    mins, secs = divmod(eta, 60)
                    hours, mins = divmod(mins, 60)
                    if hours > 0:
                        eta_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
                    else:
                        eta_str = f"{mins:02d}:{secs:02d}"
                    self.eta_sig.emit(eta_str)
                
            elif d['status'] == 'finished':
                self.progress_sig.emit(100.0)

        class YtdlpLogger:
            def __init__(self, sig):
                self.sig = sig
            def debug(self, msg):
                if msg.strip() and not msg.startswith('[download]') and not msg.startswith('[Frag'):
                    self.sig.emit(msg)
            def info(self, msg):
                if msg.strip():
                    self.sig.emit(msg)
            def warning(self, msg):
                if msg.strip():
                    self.sig.emit(f"WARNING: {msg}")
            def error(self, msg):
                if msg.strip():
                    self.sig.emit(f"ERROR: {msg}")

        if self.custom_filename and self.custom_filename.strip():
            safe_name = re.sub(r'[\\/*?:"<>|]', '_', self.custom_filename.strip())
            out_pattern = f"{safe_name}.%(ext)s"
        else:
            out_pattern = '%(title)s.%(ext)s'

        ydl_opts = {
            'outtmpl': os.path.join(self.save_dir, out_pattern),
            'progress_hooks': [progress_hook],
            'logger': YtdlpLogger(self.log_sig),
            'noprogress': True,
            'allow_unverified_js': True,
            'remote_components': ['ejs:github'],
            'concurrent_fragment_downloads': self.concurrent_fragments,
            'js_runtimes': {
                'deno': {},
                'node': {},
                'quickjs': {}
            },
            'extractor_args': {
                'tiktok': ['api_hostname=api16-normal-c-useast1a.tiktokv.com']
            }
        }

        if self.cookiefile and os.path.exists(self.cookiefile):
            ydl_opts['cookiefile'] = self.cookiefile

        audio_ext = self.preferred_audio_format.lower()
        # Lossless wav/flac doesn't need bitrate compression settings
        audio_quality = '0' if audio_ext in ['wav', 'flac'] else '192'

        if self.selected_format_id:
            if self.selected_format_type == 'Video Only':
                ydl_opts['format'] = f"{self.selected_format_id}+bestaudio/best"
                ydl_opts['merge_output_format'] = self.preferred_container
            elif self.selected_format_type == 'Audio Only':
                ydl_opts['format'] = self.selected_format_id
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_ext,
                    'preferredquality': audio_quality,
                }]
            else:
                ydl_opts['format'] = self.selected_format_id
        else:
            if self.format_opt == 'audio_only':
                ydl_opts['format'] = 'bestaudio/best'
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_ext,
                    'preferredquality': audio_quality,
                }]
            else:
                ydl_opts['format'] = 'bestvideo+bestaudio/best'
                ydl_opts['merge_output_format'] = self.preferred_container

        try:
            self.status_sig.emit('downloading')
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                if info.get('_type') == 'playlist':
                    filename = self.save_dir
                else:
                    filename = ydl.prepare_filename(info)
                    if self.selected_format_type == 'Audio Only' or (not self.selected_format_id and self.format_opt == 'audio_only'):
                        base, _ = os.path.splitext(filename)
                        if os.path.exists(base + "." + audio_ext):
                            filename = base + "." + audio_ext
                    else:
                        # Make sure container matches preferred setting if merged
                        base, _ = os.path.splitext(filename)
                        if os.path.exists(base + "." + self.preferred_container):
                            filename = base + "." + self.preferred_container
                self.finished_sig.emit(filename)
        except Exception as e:
            if self._is_cancelled:
                self.status_sig.emit('cancelled')
            else:
                self.error_sig.emit(str(e))
                self.status_sig.emit('error')

    def cancel(self):
        self._is_cancelled = True


class YtdlpInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.worker = None
        self.analysis_worker = None
        self.thumb_downloader = None
        self.formats_list = []
        self.is_analyzed = False
        self.is_playlist = False
        self.downloaded_filepath = None
        self.current_thumbnail_url = None
        self.current_thumbnail_bytes = None
        
        self.settings = load_settings()
        self.download_queue = []
        self.current_queue_index = -1
        self.download_history = self.settings.get('download_history', [])
        
        self.__init_widgets()

    def __init_widgets(self):
        self.scrollWidget = QtWidgets.QWidget(self)
        self.main_layout = QVBoxLayout(self.scrollWidget)
        self.main_layout.setSpacing(16)
        self.main_layout.setContentsMargins(36, 24, 36, 24)
        
        self.setWidget(self.scrollWidget)
        self.enableTransparentBackground()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        self.setup_ui()
        self.retranslateUi()
        self.populate_history_table()

    def setup_ui(self):
        # 1. Title Label
        self.titleLabel = TitleLabel(self.scrollWidget)
        self.titleLabel.setContentsMargins(0, 0, 0, 8)
        self.main_layout.addWidget(self.titleLabel)

        # 2. Main Form Card
        self.card = CardWidget(self.scrollWidget)
        self.card.setObjectName("YtdlpFormCard")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setSpacing(16)
        self.card_layout.setContentsMargins(20, 20, 20, 20)

        # URL Input with Paste & Analyze Buttons
        self.url_label = BodyLabel(self.card)
        self.url_layout = QHBoxLayout()
        
        self.url_input = LineEdit(self.card)
        self.url_input.setClearButtonEnabled(True)
        self.url_input.returnPressed.connect(self.start_analysis)
        self.url_input.setText(self.settings.get('last_url', ''))
        
        self.paste_btn = PushButton(self.card)
        self.paste_btn.setIcon(FluentIcon.PASTE)
        self.paste_btn.clicked.connect(self.paste_from_clipboard)
        
        self.analyze_btn = PushButton(self.card)
        self.analyze_btn.setIcon(FluentIcon.SEARCH)
        self.analyze_btn.clicked.connect(self.start_analysis)
        
        self.url_layout.addWidget(self.url_input)
        self.url_layout.addWidget(self.paste_btn)
        self.url_layout.addWidget(self.analyze_btn)
        
        self.card_layout.addWidget(self.url_label)
        self.card_layout.addLayout(self.url_layout)

        # Supported Platforms Icons Layout
        self.platforms_layout = QHBoxLayout()
        self.platforms_layout.setSpacing(8)
        self.platforms_layout.setContentsMargins(0, 0, 0, 0)
        
        self.platforms_title = BodyLabel(self.card)
        self.platforms_title.setStyleSheet("color: gray; font-size: 12px;")
        self.platforms_layout.addWidget(self.platforms_title)
        
        ensure_icons()
        for name in ['youtube', 'tiktok', 'facebook', 'instagram', 'twitter', 'twitch', 'bilibili']:
            fpath = os.path.join('assets/icons', f"{name}.svg")
            if os.path.exists(fpath):
                lbl = QtWidgets.QLabel(self.card)
                lbl.setFixedSize(16, 16)
                lbl.setScaledContents(True)
                lbl.setPixmap(QtGui.QPixmap(fpath))
                lbl.setToolTip(name.capitalize())
                self.platforms_layout.addWidget(lbl)
        self.platforms_layout.addStretch()
        self.card_layout.addLayout(self.platforms_layout)

        # Metadata Card (Thumbnail & Text Details)
        self.meta_card = CardWidget(self.card)
        self.meta_card.setObjectName("MetaCard")
        self.meta_card.setStyleSheet("#MetaCard { background-color: rgba(0, 0, 0, 0.02); border: 1px solid rgba(0, 0, 0, 0.05); border-radius: 6px; }")
        self.meta_card_layout = QHBoxLayout(self.meta_card)
        self.meta_card_layout.setContentsMargins(12, 12, 12, 12)
        self.meta_card_layout.setSpacing(16)
        
        self.thumbnail_label = QtWidgets.QLabel(self.meta_card)
        self.thumbnail_label.setFixedSize(140, 78)
        self.thumbnail_label.setScaledContents(True)
        self.thumbnail_label.setAlignment(QtCore.Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("border-radius: 4px; background-color: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); color: gray;")
        
        self.meta_text_layout = QVBoxLayout()
        self.meta_text_layout.setSpacing(4)
        
        self.meta_title = BodyLabel(self.meta_card)
        self.meta_title.setWordWrap(True)
        title_font = self.meta_title.font()
        title_font.setBold(True)
        title_font.setPointSize(11)
        self.meta_title.setFont(title_font)
        
        self.meta_author = BodyLabel(self.meta_card)
        author_font = self.meta_author.font()
        author_font.setPointSize(9)
        self.meta_author.setFont(author_font)
        self.meta_author.setStyleSheet("color: #666666;")
        
        self.meta_duration = BodyLabel(self.meta_card)
        duration_font = self.meta_duration.font()
        duration_font.setPointSize(9)
        self.meta_duration.setFont(duration_font)
        self.meta_duration.setStyleSheet("color: #666666;")

        self.download_thumb_btn = PushButton(self.meta_card)
        self.download_thumb_btn.setIcon(FluentIcon.DOWNLOAD)
        self.download_thumb_btn.clicked.connect(self.download_thumbnail_file)
        self.download_thumb_btn.setFixedSize(160, 26)
        
        self.filename_layout = QHBoxLayout()
        self.filename_layout.setSpacing(6)
        self.filename_label = BodyLabel(self.meta_card)
        fn_font = self.filename_label.font()
        fn_font.setPointSize(9)
        fn_font.setBold(True)
        self.filename_label.setFont(fn_font)
        
        self.filename_input = LineEdit(self.meta_card)
        self.filename_input.setClearButtonEnabled(True)
        self.filename_input.setFixedHeight(30)
        self.filename_input.setPlaceholderText("Nhập tên file tuỳ chỉnh (để trống nếu giữ tiêu đề gốc)...")
        
        self.filename_layout.addWidget(self.filename_label)
        self.filename_layout.addWidget(self.filename_input)
        
        self.meta_text_layout.addWidget(self.meta_title)
        self.meta_text_layout.addWidget(self.meta_author)
        self.meta_text_layout.addWidget(self.meta_duration)
        self.meta_text_layout.addLayout(self.filename_layout)
        self.meta_text_layout.addWidget(self.download_thumb_btn)
        self.meta_text_layout.addStretch()
        
        self.meta_card_layout.addWidget(self.thumbnail_label)
        self.meta_card_layout.addLayout(self.meta_text_layout)
        self.meta_card_layout.addStretch()
        
        self.card_layout.addWidget(self.meta_card)
        self.meta_card.hide()

        # Playlist Card (Checklist table of videos)
        self.playlist_card = CardWidget(self.card)
        self.playlist_card.setObjectName("PlaylistCard")
        self.playlist_card_layout = QVBoxLayout(self.playlist_card)
        self.playlist_card_layout.setContentsMargins(12, 12, 12, 12)
        self.playlist_card_layout.setSpacing(8)
        
        self.playlist_header_layout = QHBoxLayout()
        self.playlist_header_title = BodyLabel(self.playlist_card)
        self.playlist_header_title.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        self.playlist_header_layout.addWidget(self.playlist_header_title)
        self.playlist_header_layout.addStretch()
        
        self.playlist_select_all_btn = PushButton(self.playlist_card)
        self.playlist_select_all_btn.setText("Chọn tất cả")
        self.playlist_select_all_btn.clicked.connect(self.select_all_playlist_items)
        
        self.playlist_deselect_all_btn = PushButton(self.playlist_card)
        self.playlist_deselect_all_btn.setText("Bỏ chọn")
        self.playlist_deselect_all_btn.clicked.connect(self.deselect_all_playlist_items)
        
        self.playlist_header_layout.addWidget(self.playlist_select_all_btn)
        self.playlist_header_layout.addWidget(self.playlist_deselect_all_btn)
        self.playlist_card_layout.addLayout(self.playlist_header_layout)
        
        self.playlist_table = TableWidget(self.playlist_card)
        self.playlist_table.setColumnCount(4)
        self.playlist_table.setMinimumHeight(180)
        self.playlist_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.playlist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.playlist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.playlist_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.playlist_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.playlist_table.setColumnWidth(0, 45)
        self.playlist_table.setColumnWidth(1, 45)
        self.playlist_table.setHorizontalHeaderLabels(["Chọn", "STT", "Tiêu đề video", "Thời lượng"])
        
        self.playlist_card_layout.addWidget(self.playlist_table)
        self.card_layout.addWidget(self.playlist_card)
        self.playlist_card.hide()

        # Cookies File Selection
        self.cookies_label = BodyLabel(self.card)
        self.cookies_layout = QHBoxLayout()
        self.cookies_input = LineEdit(self.card)
        self.cookies_input.setReadOnly(True)
        self.cookies_input.setText(self.settings.get('cookies_file', ''))
        
        self.choose_cookies_btn = PushButton(self.card)
        self.choose_cookies_btn.setIcon(FluentIcon.VPN)
        self.choose_cookies_btn.clicked.connect(self.choose_cookies_file)
        
        self.cookies_layout.addWidget(self.cookies_input)
        self.cookies_layout.addWidget(self.choose_cookies_btn)
        
        self.card_layout.addWidget(self.cookies_label)
        self.card_layout.addLayout(self.cookies_layout)

        # Save Directory and Preferred Container preferences
        self.settings_row_layout = QHBoxLayout()
        self.settings_row_layout.setSpacing(16)
        
        # Save folder sub-layout
        self.save_dir_container = QtWidgets.QWidget(self.card)
        self.save_dir_sub_layout = QVBoxLayout(self.save_dir_container)
        self.save_dir_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.save_dir_label = BodyLabel(self.save_dir_container)
        self.save_dir_input_layout = QHBoxLayout()
        self.save_dir_input = LineEdit(self.save_dir_container)
        self.save_dir_input.setReadOnly(True)
        default_dir = self.settings.get('save_dir', config.saveDirectory.value if config.saveDirectory.value else os.getcwd())
        self.save_dir_input.setText(default_dir)
        self.choose_dir_btn = PushButton(self.save_dir_container)
        self.choose_dir_btn.setIcon(FluentIcon.FOLDER)
        self.choose_dir_btn.clicked.connect(self.choose_save_dir)
        self.save_dir_input_layout.addWidget(self.save_dir_input)
        self.save_dir_input_layout.addWidget(self.choose_dir_btn)
        self.save_dir_sub_layout.addWidget(self.save_dir_label)
        self.save_dir_sub_layout.addLayout(self.save_dir_input_layout)
        
        # Preferred container format sub-layout
        self.container_container = QtWidgets.QWidget(self.card)
        self.container_sub_layout = QVBoxLayout(self.container_container)
        self.container_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.container_label = BodyLabel(self.container_container)
        self.container_combo = ComboBox(self.container_container)
        self.container_combo.addItems(['mp4', 'mkv'])
        saved_container = self.settings.get('preferred_container', 'mp4')
        self.container_combo.setCurrentText(saved_container)
        self.container_combo.currentTextChanged.connect(self.on_container_changed)
        self.container_sub_layout.addWidget(self.container_label)
        self.container_sub_layout.addWidget(self.container_combo)
        
        # Preferred audio format sub-layout
        self.audio_format_container = QtWidgets.QWidget(self.card)
        self.audio_format_sub_layout = QVBoxLayout(self.audio_format_container)
        self.audio_format_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.audio_format_label = BodyLabel(self.audio_format_container)
        self.audio_format_combo = ComboBox(self.audio_format_container)
        self.audio_format_combo.addItems(['mp3', 'm4a', 'wav', 'flac'])
        saved_audio = self.settings.get('preferred_audio_format', 'mp3')
        self.audio_format_combo.setCurrentText(saved_audio)
        self.audio_format_combo.currentTextChanged.connect(self.on_audio_format_changed)
        self.audio_format_sub_layout.addWidget(self.audio_format_label)
        self.audio_format_sub_layout.addWidget(self.audio_format_combo)
        
        # Concurrent threads accelerator sub-layout
        self.threads_container = QtWidgets.QWidget(self.card)
        self.threads_sub_layout = QVBoxLayout(self.threads_container)
        self.threads_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.threads_label = BodyLabel(self.threads_container)
        self.threads_combo = ComboBox(self.threads_container)
        self.threads_combo.addItems(['1 luồng', '4 luồng', '8 luồng', '16 luồng'])
        saved_threads = str(self.settings.get('concurrent_fragments', 4))
        for idx in range(self.threads_combo.count()):
            if saved_threads in self.threads_combo.itemText(idx):
                self.threads_combo.setCurrentIndex(idx)
                break
        self.threads_combo.currentTextChanged.connect(self.on_threads_changed)
        self.threads_sub_layout.addWidget(self.threads_label)
        self.threads_sub_layout.addWidget(self.threads_combo)
        
        self.settings_row_layout.addWidget(self.save_dir_container, 3)
        self.settings_row_layout.addWidget(self.container_container, 1)
        self.settings_row_layout.addWidget(self.audio_format_container, 1)
        self.settings_row_layout.addWidget(self.threads_container, 1)
        self.card_layout.addLayout(self.settings_row_layout)

        # Format Selection Combo box
        self.format_label = BodyLabel(self.card)
        self.format_combo = ComboBox(self.card)
        self.card_layout.addWidget(self.format_label)
        self.card_layout.addWidget(self.format_combo)

        # Progress elements
        self.progress_layout = QHBoxLayout()
        self.status_label = BodyLabel(self.card)
        self.speed_label = BodyLabel(self.card)
        self.eta_label = BodyLabel(self.card)
        self.pct_label = BodyLabel(self.card)
        
        self.progress_layout.addWidget(self.status_label)
        self.progress_layout.addStretch()
        self.progress_layout.addWidget(self.speed_label)
        self.progress_layout.addWidget(self.eta_label)
        self.progress_layout.addWidget(self.pct_label)
        
        self.progress_bar = ProgressBar(self.card)
        self.progress_bar.setValue(0)
        
        self.card_layout.addLayout(self.progress_layout)
        self.card_layout.addWidget(self.progress_bar)

        # Action Buttons Layout (Download, Add to Queue, Cancel, and Folder/Remover shortcuts)
        self.buttons_layout = QHBoxLayout()
        self.download_btn = PrimaryPushButton(self.card)
        self.download_btn.clicked.connect(self.start_download)
        self.download_btn.setEnabled(False)  # Require link analysis first!
        
        self.add_queue_btn = PushButton(self.card)
        self.add_queue_btn.setIcon(FluentIcon.ADD)
        self.add_queue_btn.clicked.connect(self.add_to_queue)
        self.add_queue_btn.setEnabled(False)
        
        self.cancel_btn = PushButton(self.card)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_download)
        
        self.open_folder_btn = PushButton(self.card)
        self.open_folder_btn.setIcon(FluentIcon.FOLDER)
        self.open_folder_btn.clicked.connect(self.open_download_folder)
        self.open_folder_btn.hide()
        
        self.open_remover_btn = PrimaryPushButton(self.card)
        self.open_remover_btn.setIcon(FluentIcon.MOVE)
        self.open_remover_btn.clicked.connect(self.open_in_remover)
        self.open_remover_btn.hide()
        
        self.buttons_layout.addWidget(self.download_btn)
        self.buttons_layout.addWidget(self.add_queue_btn)
        self.buttons_layout.addWidget(self.cancel_btn)
        self.buttons_layout.addWidget(self.open_folder_btn)
        self.buttons_layout.addWidget(self.open_remover_btn)
        self.buttons_layout.addStretch()
        self.card_layout.addLayout(self.buttons_layout)

        self.main_layout.addWidget(self.card)

        # 3. Batch Download Queue Card
        self.queue_card = CardWidget(self.scrollWidget)
        self.queue_card_layout = QVBoxLayout(self.queue_card)
        self.queue_card_layout.setContentsMargins(20, 20, 20, 20)
        self.queue_card_layout.setSpacing(12)
        
        self.queue_title_layout = QHBoxLayout()
        self.queue_title = TitleLabel(self.queue_card)
        self.queue_title.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        self.queue_title_layout.addWidget(self.queue_title)
        self.queue_title_layout.addStretch()
        
        self.start_queue_btn = PrimaryPushButton(self.queue_card)
        self.start_queue_btn.setIcon(FluentIcon.PLAY)
        self.start_queue_btn.clicked.connect(self.start_queue_download)
        
        self.remove_queue_btn = PushButton(self.queue_card)
        self.remove_queue_btn.setIcon(FluentIcon.REMOVE)
        self.remove_queue_btn.clicked.connect(self.remove_selected_queue)
        
        self.clear_queue_btn = PushButton(self.queue_card)
        self.clear_queue_btn.setIcon(FluentIcon.DELETE)
        self.clear_queue_btn.clicked.connect(self.clear_queue)
        
        self.queue_title_layout.addWidget(self.start_queue_btn)
        self.queue_title_layout.addWidget(self.remove_queue_btn)
        self.queue_title_layout.addWidget(self.clear_queue_btn)
        
        self.queue_table = TableWidget(self.queue_card)
        self.queue_table.setColumnCount(5)
        self.queue_table.setMinimumHeight(150)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.queue_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        self.queue_card_layout.addLayout(self.queue_title_layout)
        self.queue_card_layout.addWidget(self.queue_table)
        self.main_layout.addWidget(self.queue_card)

        # 4. Download History Card
        self.history_card = CardWidget(self.scrollWidget)
        self.history_card_layout = QVBoxLayout(self.history_card)
        self.history_card_layout.setContentsMargins(20, 20, 20, 20)
        self.history_card_layout.setSpacing(12)
        
        self.history_title_layout = QHBoxLayout()
        self.history_title = TitleLabel(self.history_card)
        self.history_title.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        self.history_title_layout.addWidget(self.history_title)
        self.history_title_layout.addStretch()
        
        self.clear_history_btn = PushButton(self.history_card)
        self.clear_history_btn.setIcon(FluentIcon.DELETE)
        self.clear_history_btn.clicked.connect(self.clear_history)
        self.history_title_layout.addWidget(self.clear_history_btn)
        
        self.history_table = TableWidget(self.history_card)
        self.history_table.setColumnCount(5)
        self.history_table.setMinimumHeight(200)
        self.history_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.history_table.setColumnWidth(0, 75)
        self.history_table.setColumnWidth(4, 140)
        
        self.history_card_layout.addLayout(self.history_title_layout)
        self.history_card_layout.addWidget(self.history_table)
        self.main_layout.addWidget(self.history_card)

        # 5. Log Console
        self.log_console_label = BodyLabel(self.scrollWidget)
        self.main_layout.addWidget(self.log_console_label)

        self.log_console = PlainTextEdit(self.scrollWidget)
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(150)
        self.main_layout.addWidget(self.log_console)

        # Connect URL input changed signal at the very end to prevent triggering before UI is fully built
        self.url_input.textChanged.connect(self.on_url_changed)

    def retranslateUi(self):
        self.titleLabel.setText(tr['Ytdlp']['Title'])
        self.url_label.setText(tr['Ytdlp']['Url'])
        self.url_input.setPlaceholderText(tr['Ytdlp']['UrlPlaceholder'])
        self.paste_btn.setText(tr['Ytdlp']['BtnPaste'])
        self.analyze_btn.setText(tr['Ytdlp']['BtnAnalyze'])
        self.platforms_title.setText(tr['Ytdlp']['Platforms'])
        self.cookies_label.setText(tr['Ytdlp']['Cookies'])
        self.cookies_input.setPlaceholderText(tr['Ytdlp']['CookiesPlaceholder'])
        self.choose_cookies_btn.setText(tr['Ytdlp']['CookiesSelect'])
        
        self.save_dir_label.setText(tr['Ytdlp']['SaveDir'])
        self.choose_dir_btn.setText(tr['Ytdlp']['SelectFolder'])
        self.container_label.setText(tr['Ytdlp']['PreferredContainer'])
        self.audio_format_label.setText(tr['Ytdlp']['PreferredAudio'])
        self.threads_label.setText(tr['Ytdlp']['PreferredThreads'])
        
        self.log_console_label.setText(tr['Ytdlp']['LogConsole'])
        self.download_btn.setText(tr['Ytdlp']['BtnDownload'])
        self.add_queue_btn.setText(tr['Ytdlp']['BtnAddToQueue'])
        self.cancel_btn.setText(tr['Ytdlp']['BtnCancel'])
        self.open_folder_btn.setText(tr['Ytdlp']['BtnOpenFolder'])
        self.open_remover_btn.setText(tr['Ytdlp']['BtnOpenRemover'])
        self.download_thumb_btn.setText(tr['Ytdlp']['BtnDownloadThumb'])
        self.filename_label.setText(tr['Ytdlp']['OutputFilename'])
        
        self.queue_title.setText(tr['Ytdlp']['QueueCardTitle'])
        self.start_queue_btn.setText(tr['Ytdlp']['BtnStartQueue'])
        self.remove_queue_btn.setText(tr['Ytdlp']['BtnRemoveQueue'])
        self.clear_queue_btn.setText(tr['Ytdlp']['BtnClearQueue'])
        
        self.queue_table.setHorizontalHeaderLabels([
            tr['Ytdlp']['QueueColTitle'],
            tr['Ytdlp']['QueueColQuality'],
            tr['Ytdlp']['QueueColFolder'],
            tr['Ytdlp']['QueueColStatus'],
            tr['Ytdlp']['QueueColProgress']
        ])
        
        self.history_title.setText(tr['Ytdlp']['HistoryCardTitle'])
        self.clear_history_btn.setText(tr['Ytdlp']['BtnClearHistory'])
        
        self.history_table.setHorizontalHeaderLabels([
            tr['Ytdlp']['HistoryColThumb'],
            tr['Ytdlp']['HistoryColTitle'],
            tr['Ytdlp']['HistoryColQuality'],
            tr['Ytdlp']['HistoryColSize'],
            tr['Ytdlp']['HistoryColAction']
        ])
        
        if self.is_analyzed:
            self.format_label.setText(tr['Ytdlp']['FormatComboTitle'])
            current_idx = self.format_combo.currentIndex()
            self.format_combo.clear()
            items = []
            for item in self.formats_list:
                items.append(f"[{item['quality']}] {item['type']} ({item['ext']}) - {item['size']} ({item['codec']})")
            self.format_combo.addItems(items)
            if current_idx >= 0 and current_idx < len(items):
                self.format_combo.setCurrentIndex(current_idx)
        else:
            self.format_label.setText(tr['Ytdlp']['Format'])
            self.format_combo.clear()
            self.format_combo.addItems([
                tr['Ytdlp']['FormatPlaceholder']
            ])
            self.format_combo.setCurrentIndex(0)

        if not self.worker and not self.analysis_worker:
            self.status_label.setText(tr['Ytdlp']['StatusIdle'])

    def choose_save_dir(self):
        folder = FolderMemoryDialog.getExistingDirectory(self, tr['Ytdlp']['SelectFolder'], category="video")
        if folder:
            self.save_dir_input.setText(folder)
            self.settings['save_dir'] = folder
            save_settings(self.settings)

    def choose_cookies_file(self):
        file_path, _ = FolderMemoryDialog.getOpenFileName(
            self, 
            tr['Ytdlp']['CookiesSelect'], 
            filter_str="Text Files (*.txt);;All Files (*)",
            category="default"
        )
        if file_path:
            self.cookies_input.setText(file_path)
            self.settings['cookies_file'] = file_path
            save_settings(self.settings)

    def select_all_playlist_items(self):
        for row in range(self.playlist_table.rowCount()):
            item = self.playlist_table.item(row, 0)
            if item:
                item.setCheckState(QtCore.Qt.Checked)

    def deselect_all_playlist_items(self):
        for row in range(self.playlist_table.rowCount()):
            item = self.playlist_table.item(row, 0)
            if item:
                item.setCheckState(QtCore.Qt.Unchecked)

    def on_container_changed(self, text):
        self.settings['preferred_container'] = text
        save_settings(self.settings)

    def on_audio_format_changed(self, text):
        self.settings['preferred_audio_format'] = text
        save_settings(self.settings)

    def on_threads_changed(self, text):
        self.settings['concurrent_fragments'] = self.get_selected_threads()
        save_settings(self.settings)

    def get_selected_threads(self):
        text = self.threads_combo.currentText()
        if '16' in text: return 16
        if '8' in text: return 8
        if '4' in text: return 4
        return 1

    def check_disk_space(self, save_dir, estimated_bytes=0):
        """Check available disk space in save_dir against estimated_bytes (+ 100MB buffer). Returns (is_enough, free_mb, req_mb)."""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            total, used, free = shutil.disk_usage(save_dir)
            buffer_bytes = 100 * 1024 * 1024  # 100 MB safety buffer
            # Default to 50MB estimate if unknown
            req_bytes = (estimated_bytes if estimated_bytes > 0 else 50 * 1024 * 1024) + buffer_bytes
            free_mb = free / (1024 * 1024)
            req_mb = req_bytes / (1024 * 1024)
            return (free >= req_bytes, free_mb, req_mb)
        except Exception as e:
            print("Disk usage check error:", e)
            return (True, 999999, 0)

    def paste_from_clipboard(self):
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            text = normalize_url(text)
            self.url_input.setText(text)

    def check_clipboard_and_paste(self):
        """Auto clipboard monitoring: Triggered when user enters this tab"""
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            text = normalize_url(text)
            is_valid = False
            for d in ['youtube.com', 'youtu.be', 'tiktok.com', 'facebook.com', 'instagram.com', 'bilibili.com', 'douyin.com', 'twitch.tv']:
                if d in text.lower():
                    is_valid = True
                    break
            if is_valid and text != self.url_input.text().strip():
                self.url_input.setText(text)
                self.start_analysis()

    def open_download_folder(self):
        if hasattr(self, 'downloaded_filepath') and self.downloaded_filepath and os.path.exists(self.downloaded_filepath):
            try:
                from showinfm import show_in_file_manager
                show_in_file_manager(self.downloaded_filepath)
            except Exception:
                os.startfile(os.path.dirname(self.downloaded_filepath))

    def open_in_remover(self):
        if hasattr(self, 'downloaded_filepath') and self.downloaded_filepath and os.path.exists(self.downloaded_filepath):
            main_win = self.window()
            if hasattr(main_win, 'toolsInterface'):
                main_win.switchTo(main_win.toolsInterface)
                main_win.toolsInterface.open_video_in_remover(self.downloaded_filepath)

    def download_thumbnail_file(self):
        if not self.current_thumbnail_url:
            return
        
        default_name = "thumbnail.jpg"
        save_path, _ = FolderMemoryDialog.getSaveFileName(
            self, 
            "Lưu ảnh Thumbnail / Save Thumbnail", 
            default_filename=default_name, 
            filter_str="Images (*.jpg *.png);;All Files (*)",
            category="video"
        )
        if save_path:
            try:
                req = urllib.request.Request(self.current_thumbnail_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data = response.read()
                with open(save_path, 'wb') as f:
                    f.write(data)
                InfoBar.success(
                    title=tr['Ytdlp']['Title'],
                    content=f"Đã lưu thumbnail tại: {os.path.basename(save_path)}",
                    duration=3000,
                    parent=self
                )
            except Exception as e:
                InfoBar.error(
                    title=tr['Ytdlp']['Title'],
                    content=f"Lỗi tải thumbnail: {e}",
                    duration=5000,
                    parent=self
                )

    def on_url_changed(self):
        """URL changed: Reset all metadata, hide cards, disable download button, and reset combobox"""
        if not hasattr(self, 'meta_card') or self.meta_card is None:
            return
        self.is_analyzed = False
        self.is_playlist = False
        self.formats_list.clear()
        self.meta_card.hide()
        if hasattr(self, 'playlist_card') and self.playlist_card is not None:
            self.playlist_card.hide()
            self.playlist_table.setRowCount(0)
        self.thumbnail_label.setPixmap(QtGui.QPixmap())
        self.thumbnail_label.setText("")
        self.meta_title.setText("")
        self.meta_author.setText("")
        self.meta_duration.setText("")
        self.current_thumbnail_url = None
        self.current_thumbnail_bytes = None
        self.downloaded_filepath = None
        
        self.format_combo.clear()
        self.format_combo.addItems([
            tr['Ytdlp']['FormatPlaceholder']
        ])
        self.format_combo.setCurrentIndex(0)
        
        self.download_btn.setEnabled(False)
        self.add_queue_btn.setEnabled(False)
        self.open_folder_btn.hide()
        self.open_remover_btn.hide()

    def start_analysis(self):
        raw_url = self.url_input.text().strip()
        url = normalize_url(raw_url)
        if url != raw_url:
            self.url_input.setText(url)
            
        if not url:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng nhập đường dẫn video",
                duration=3000,
                parent=self
            )
            return

        self.settings['last_url'] = url
        save_settings(self.settings)

        self.analyze_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.add_queue_btn.setEnabled(False)
        self.url_input.setEnabled(False)
        self.choose_cookies_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.status_label.setText(tr['Ytdlp']['StatusAnalyzing'])
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        
        self.is_analyzed = False
        self.is_playlist = False
        self.formats_list.clear()
        self.meta_card.hide()
        if hasattr(self, 'playlist_card') and self.playlist_card is not None:
            self.playlist_card.hide()
            self.playlist_table.setRowCount(0)
        self.open_folder_btn.hide()
        self.open_remover_btn.hide()

        cookies_file = self.cookies_input.text().strip()
        self.analysis_worker = YtdlpAnalysisWorker(url, cookies_file)
        self.analysis_worker.finished_sig.connect(self.on_analysis_success)
        self.analysis_worker.error_sig.connect(self.on_analysis_failed)
        self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_worker.start()

    @Slot(dict)
    def on_analysis_success(self, info):
        self.analyze_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.add_queue_btn.setEnabled(True)
        self.url_input.setEnabled(True)
        self.choose_cookies_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(tr['Ytdlp']['StatusExtractSuccess'])
        
        # Populate Metadata details
        self.meta_card.show()
        raw_title = info.get('title', 'Unknown Title')
        self.meta_title.setText(raw_title)
        self.meta_author.setText(info.get('uploader') or info.get('channel', 'Unknown Channel'))
        
        # Auto-fill output filename input with sanitized title
        safe_title = re.sub(r'[\\/*?:"<>|]', '_', raw_title)
        self.filename_input.setText(safe_title)
        
        # Check if it is a playlist
        is_playlist_type = info.get('_type') == 'playlist' or 'entries' in info
        self.is_playlist = is_playlist_type
        
        if is_playlist_type:
            entries = info.get('entries', [])
            self.playlist_entries = entries
            self.meta_duration.setText(f"Danh sách phát: {len(entries)} video")
            
            thumb_url = info.get('thumbnail')
            if not thumb_url and entries and entries[0]:
                thumb_url = entries[0].get('thumbnail') or (entries[0].get('thumbnails')[0].get('url') if entries[0].get('thumbnails') else None)
            
            # Populate Playlist Checklist Table
            self.playlist_card.show()
            self.playlist_header_title.setText(f"Danh sách video trong Playlist ({len(entries)} video):")
            self.playlist_table.setRowCount(0)
            self.playlist_table.setRowCount(len(entries))
            
            for idx, entry in enumerate(entries):
                # Checkbox
                chk = QTableWidgetItem()
                chk.setCheckState(QtCore.Qt.Checked)
                self.playlist_table.setItem(idx, 0, chk)
                
                # No.
                no_item = QTableWidgetItem(str(idx + 1))
                no_item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.playlist_table.setItem(idx, 1, no_item)
                
                # Title
                title_item = QTableWidgetItem(entry.get('title') or f"Video #{idx + 1}")
                self.playlist_table.setItem(idx, 2, title_item)
                
                # Duration
                dur = entry.get('duration')
                if dur:
                    try:
                        dur = int(dur)
                        mins, secs = divmod(dur, 60)
                        hours, mins = divmod(mins, 60)
                        if hours > 0:
                            dur_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
                        else:
                            dur_str = f"{mins:02d}:{secs:02d}"
                    except Exception:
                        dur_str = "--:--"
                else:
                    dur_str = "--:--"
                dur_item = QTableWidgetItem(dur_str)
                dur_item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.playlist_table.setItem(idx, 3, dur_item)
        else:
            self.playlist_card.hide()
            self.playlist_entries = []
            duration_secs = info.get('duration')
            if duration_secs:
                try:
                    duration_secs = int(duration_secs)
                    mins, secs = divmod(duration_secs, 60)
                    hours, mins = divmod(mins, 60)
                    if hours > 0:
                        duration_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
                    else:
                        duration_str = f"{mins:02d}:{secs:02d}"
                    self.meta_duration.setText(tr['Ytdlp']['Duration'].format(duration_str))
                except Exception:
                    self.meta_duration.setText(tr['Ytdlp']['Duration'].format("Unknown"))
            else:
                self.meta_duration.setText(tr['Ytdlp']['Duration'].format("Unknown"))
                
            thumb_url = info.get('thumbnail')

        self.current_thumbnail_url = thumb_url
        self.current_thumbnail_bytes = None

        # Setup thumbnail downloader
        self.thumbnail_label.setPixmap(QtGui.QPixmap())
        self.thumbnail_label.setText("Loading...")
        
        if thumb_url:
            self.thumb_downloader = ThumbnailDownloader(thumb_url)
            self.thumb_downloader.finished_sig.connect(self.on_thumbnail_downloaded)
            self.thumb_downloader.finished.connect(self.thumb_downloader.deleteLater)
            self.thumb_downloader.start()

        # Extract formats
        if is_playlist_type:
            self.formats_list = [
                {
                    'id': 'playlist_video',
                    'type': 'Video + Audio',
                    'raw_type': 'Video + Audio',
                    'quality': 'Tải toàn bộ Playlist (Video)',
                    'ext': 'mp4',
                    'size': 'Tùy chọn',
                    'codec': 'Auto'
                },
                {
                    'id': 'playlist_audio',
                    'type': 'Audio Only',
                    'raw_type': 'Audio Only',
                    'quality': 'Tải toàn bộ Playlist (MP3)',
                    'ext': 'mp3',
                    'size': 'Tùy chọn',
                    'codec': 'Auto'
                }
            ]
        else:
            formats = info.get('formats', [])
            self.formats_list.clear()
            
            for f in formats:
                fid = f.get('format_id')
                ext = f.get('ext', '')
                vcodec = f.get('vcodec', 'none')
                acodec = f.get('acodec', 'none')
                
                is_video = vcodec != 'none'
                is_audio = acodec != 'none'
                
                if not is_video and not is_audio:
                    continue
                    
                size = f.get('filesize') or f.get('filesize_approx')
                if size:
                    if size > 1024 * 1024 * 1024:
                        size_str = f"{size / (1024*1024*1024):.2f} GiB"
                    else:
                        size_str = f"{size / (1024*1024):.2f} MiB"
                else:
                    size_str = "Unknown"
                    
                if is_video and is_audio:
                    type_str = "Video + Audio"
                    raw_type = "Video + Audio"
                    resolution = f.get('resolution') or f.get('format_note') or f"{f.get('width')}x{f.get('height')}"
                elif is_video:
                    type_str = "Video + Audio"
                    raw_type = "Video Only"
                    resolution = f.get('resolution') or f.get('format_note') or f"{f.get('width')}x{f.get('height')}"
                else:
                    type_str = "Audio Only"
                    raw_type = "Audio Only"
                    resolution = f.get('format_note') or (f"{int(f.get('abr', 0))} kbps" if f.get('abr') else "Audio")
                    
                vcodec_name = vcodec if vcodec != 'none' else ""
                acodec_name = acodec if acodec != 'none' else ""
                codec_str = f"{vcodec_name} / {acodec_name}" if (vcodec_name and acodec_name) else (vcodec_name or acodec_name)
                
                self.formats_list.append({
                    'id': fid,
                    'type': type_str,
                    'raw_type': raw_type,
                    'quality': resolution,
                    'ext': ext,
                    'size': size_str,
                    'codec': codec_str
                })
                
            self.formats_list.reverse()
            
        self.is_analyzed = True
        
        self.format_label.setText(tr['Ytdlp']['FormatComboTitle'])
        self.format_combo.clear()
        items = []
        for item in self.formats_list:
            if is_playlist_type:
                items.append(f"[{item['quality']}] {item['type']} ({item['ext']}) - {item['size']}")
            else:
                items.append(f"[{item['quality']}] {item['type']} ({item['ext']}) - {item['size']} ({item['codec']})")
        self.format_combo.addItems(items)
        if items:
            self.format_combo.setCurrentIndex(0)

        InfoBar.success(
            title=tr['Ytdlp']['Title'],
            content=tr['Ytdlp']['StatusExtractSuccess'],
            duration=3000,
            parent=self
        )

    @Slot(bytes)
    def on_thumbnail_downloaded(self, data):
        self.current_thumbnail_bytes = data
        pixmap = QtGui.QPixmap()
        if pixmap.loadFromData(data):
            self.thumbnail_label.setPixmap(pixmap)
            self.thumbnail_label.setText("")

    @Slot(str)
    def on_analysis_failed(self, err):
        friendly_msg = translate_ytdlp_error(err)
        self.analyze_btn.setEnabled(True)
        self.download_btn.setEnabled(False)
        self.add_queue_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.choose_cookies_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(friendly_msg)
        self.log_console.appendPlainText(f"\nChi tiết lỗi: {err}\n-> Hướng xử lý: {friendly_msg}")
        
        InfoBar.error(
            title=tr['Ytdlp']['Title'],
            content=friendly_msg,
            duration=6000,
            parent=self
        )

    # ==================== DOWNLOAD QUEUE LOGIC ====================
    def add_to_queue(self):
        if not self.is_analyzed or not self.formats_list:
            return

        url = self.url_input.text().strip()
        save_dir = self.save_dir_input.text().strip() or os.getcwd()
        cookies_file = self.cookies_input.text().strip()
        pref_container = self.container_combo.currentText()
        pref_audio = self.audio_format_combo.currentText()
        
        if self.is_playlist:
            # We will expand each checked video from the playlist table into a queue task!
            added_count = 0
            for row in range(self.playlist_table.rowCount()):
                chk_item = self.playlist_table.item(row, 0)
                if chk_item and chk_item.checkState() == QtCore.Qt.Checked:
                    entry = self.playlist_entries[row]
                    video_title = entry.get('title') or f"Video #{row + 1}"
                    video_url = entry.get('url')
                    if not video_url:
                        video_id = entry.get('id')
                        if video_id:
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                    if not video_url:
                        continue
                        
                    # Determine format based on format_combo selection
                    current_idx = self.format_combo.currentIndex()
                    if current_idx == 1:  # Playlist Audio Only
                        selected_format_id = None
                        selected_format_type = 'Audio Only'
                        format_opt = 'audio_only'
                        format_desc = f"[{pref_audio.upper()}] Chỉ âm thanh"
                    else:  # Playlist Video + Audio
                        selected_format_id = None
                        selected_format_type = 'Video + Audio'
                        format_opt = 'best'
                        format_desc = f"[{pref_container.upper()}] Video + Âm thanh"
                        
                    # Default size to 50MB for playlist individual entries
                    est_bytes = 50 * 1024 * 1024
                    
                    queue_item = {
                        'url': video_url,
                        'title': video_title,
                        'format_opt': format_opt,
                        'selected_format_id': selected_format_id,
                        'selected_format_type': selected_format_type,
                        'format_desc': format_desc,
                        'cookie_file': cookies_file,
                        'save_dir': save_dir,
                        'preferred_container': pref_container,
                        'preferred_audio_format': pref_audio,
                        'concurrent_fragments': self.get_selected_threads(),
                        'status': tr['Ytdlp']['StatusQueuePending'],
                        'progress': 0.0,
                        'estimated_bytes': est_bytes,
                        'thumbnail_bytes': None
                    }
                    self.download_queue.append(queue_item)
                    added_count += 1
            
            if added_count > 0:
                self.update_queue_table()
                InfoBar.success(
                    title=tr['Ytdlp']['Title'],
                    content=f"Đã thêm {added_count} video vào hàng đợi!",
                    duration=3000,
                    parent=self
                )
            else:
                InfoBar.warning(
                    title=tr['Ytdlp']['Title'],
                    content="Vui lòng chọn ít nhất 1 video trong danh sách!",
                    duration=3000,
                    parent=self
                )
            return

        current_idx = self.format_combo.currentIndex()
        if current_idx < 0 or current_idx >= len(self.formats_list):
            return
            
        selected_format = self.formats_list[current_idx]
        title = self.meta_title.text()
        
        # Determine format params
        fid = selected_format['id']
        selected_format_id = None
        selected_format_type = None
        format_opt = 'best'
        
        if fid == 'playlist_video':
            selected_format_id = None
            selected_format_type = 'Video + Audio'
            format_opt = 'best'
        elif fid == 'playlist_audio':
            selected_format_id = None
            selected_format_type = 'Audio Only'
            format_opt = 'audio_only'
        else:
            selected_format_id = fid
            selected_format_type = selected_format['raw_type']

        # Parse estimated bytes from format size string
        est_bytes = 0
        size_str = selected_format.get('size', 'Unknown')
        if 'GiB' in size_str:
            try: est_bytes = int(float(size_str.split()[0]) * 1024 * 1024 * 1024)
            except: pass
        elif 'MiB' in size_str:
            try: est_bytes = int(float(size_str.split()[0]) * 1024 * 1024)
            except: pass

        queue_item = {
            'url': url,
            'title': title,
            'format_opt': format_opt,
            'selected_format_id': selected_format_id,
            'selected_format_type': selected_format_type,
            'format_desc': f"[{selected_format['quality']}] {selected_format['type']} ({selected_format['ext']})",
            'cookie_file': cookies_file,
            'save_dir': save_dir,
            'preferred_container': pref_container,
            'preferred_audio_format': pref_audio,
            'preferred_audio_format': pref_audio,
            'concurrent_fragments': self.get_selected_threads(),
            'custom_filename': self.filename_input.text().strip() if hasattr(self, 'filename_input') else None,
            'status': tr['Ytdlp']['StatusQueuePending'],
            'progress': 0.0,
            'estimated_bytes': est_bytes,
            'thumbnail_bytes': self.current_thumbnail_bytes
        }
        
        self.download_queue.append(queue_item)
        self.update_queue_table()
        
        InfoBar.success(
            title=tr['Ytdlp']['Title'],
            content="Đã thêm video vào hàng đợi!",
            duration=2000,
            parent=self
        )

    def update_queue_table(self):
        self.queue_table.setRowCount(0)
        self.queue_table.setRowCount(len(self.download_queue))
        
        for idx, item in enumerate(self.download_queue):
            # Title
            title_item = QTableWidgetItem(item['title'])
            self.queue_table.setItem(idx, 0, title_item)
            
            # Format
            format_item = QTableWidgetItem(item['format_desc'])
            self.queue_table.setItem(idx, 1, format_item)
            
            # Save Dir
            folder_item = QTableWidgetItem(os.path.basename(item['save_dir']))
            folder_item.setToolTip(item['save_dir'])
            self.queue_table.setItem(idx, 2, folder_item)
            
            # Status
            status_item = QTableWidgetItem(item['status'])
            self.queue_table.setItem(idx, 3, status_item)
            
            # Progress Bar
            pbar = ProgressBar(self.queue_card)
            pbar.setValue(int(item['progress']))
            self.queue_table.setCellWidget(idx, 4, pbar)

    def remove_selected_queue(self):
        row = self.queue_table.currentRow()
        if row >= 0 and row < len(self.download_queue):
            if row == self.current_queue_index and self.worker and self.worker.isRunning():
                InfoBar.warning(
                    title=tr['Ytdlp']['Title'],
                    content="Không thể xóa video đang tải! / Cannot remove active download!",
                    duration=3000,
                    parent=self
                )
                return
            self.download_queue.pop(row)
            self.update_queue_table()

    def clear_queue(self):
        if self.worker and self.worker.isRunning() and self.current_queue_index >= 0:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng dừng tải trước khi xoá hết! / Stop downloading first!",
                duration=3000,
                parent=self
            )
            return
        self.download_queue.clear()
        self.current_queue_index = -1
        self.update_queue_table()

    def start_queue_download(self):
        if self.worker and self.worker.isRunning():
            return
            
        self.process_next_queue_item()

    def process_next_queue_item(self):
        # Find next Pending item
        next_idx = -1
        for idx, item in enumerate(self.download_queue):
            if item['status'] == tr['Ytdlp']['StatusQueuePending']:
                next_idx = idx
                break
                
        if next_idx == -1:
            self.current_queue_index = -1
            self.status_label.setText(tr['Ytdlp']['StatusSuccess'])
            self.progress_bar.setValue(100)
            return
            
        self.current_queue_index = next_idx
        item = self.download_queue[next_idx]
        
        item['status'] = tr['Ytdlp']['StatusQueueDownloading']
        self.update_queue_table()
        
        # Check disk space before queue item download
        is_enough, free_mb, req_mb = self.check_disk_space(item['save_dir'], item.get('estimated_bytes', 0))
        if not is_enough:
            item['status'] = tr['Ytdlp']['StatusQueueFailed'] + " (Ổ đĩa đầy)"
            self.update_queue_table()
            self.log_console.appendPlainText(f"\nERROR: Dừng hàng đợi do dung lượng ổ đĩa không đủ! (Trống: {free_mb:.1f} MB, Cần ít nhất: {req_mb:.1f} MB)")
            self.reset_ui_state()
            self.start_queue_btn.setEnabled(True)
            InfoBar.error(
                title="Dừng hàng đợi: Hết dung lượng đĩa",
                content=f"Tải hàng đợi bị dừng lại vì ổ đĩa lưu trữ chỉ còn trống {free_mb:.1f} MB, không đủ để tải video tiếp theo (Cần ít nhất: {req_mb:.1f} MB).",
                duration=7000,
                parent=self
            )
            return

        self.log_console.clear()
        self.progress_bar.setValue(0)
        
        self.download_btn.setEnabled(False)
        self.add_queue_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.start_queue_btn.setEnabled(False)
        
        # Start Worker using queue parameters
        self.worker = YtdlpWorker(
            url=item['url'],
            save_dir=item['save_dir'],
            format_opt=item['format_opt'],
            selected_format_id=item['selected_format_id'],
            selected_format_type=item['selected_format_type'],
            cookiefile=item['cookie_file'],
            preferred_container=item['preferred_container'],
            preferred_audio_format=item.get('preferred_audio_format', 'mp3'),
            concurrent_fragments=item.get('concurrent_fragments', 4),
            custom_filename=item.get('custom_filename')
        )
        self.worker.progress_sig.connect(self.on_queue_progress)
        self.worker.speed_sig.connect(self.on_speed)
        self.worker.eta_sig.connect(self.on_eta)
        self.worker.status_sig.connect(self.on_queue_status)
        self.worker.log_sig.connect(self.on_log)
        self.worker.error_sig.connect(self.on_queue_error)
        self.worker.finished_sig.connect(self.on_queue_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    @Slot(float)
    def on_queue_progress(self, val):
        self.progress_bar.setValue(int(val))
        self.pct_label.setText(tr['Ytdlp']['Progress'].format(f"{val:.1f}"))
        
        if self.current_queue_index >= 0 and self.current_queue_index < len(self.download_queue):
            self.download_queue[self.current_queue_index]['progress'] = val
            pbar = self.queue_table.cellWidget(self.current_queue_index, 4)
            if pbar:
                pbar.setValue(int(val))

    @Slot(str)
    def on_queue_status(self, status):
        if status == 'downloading':
            self.status_label.setText(tr['Ytdlp']['StatusDownloading'])
        elif status == 'cancelled':
            self.status_label.setText(tr['Ytdlp']['StatusCancelled'])
            if self.current_queue_index >= 0:
                self.download_queue[self.current_queue_index]['status'] = tr['Ytdlp']['StatusCancelled']
                self.update_queue_table()
            self.reset_ui_state()
            self.start_queue_btn.setEnabled(True)
        elif status == 'error':
            self.status_label.setText(tr['Ytdlp']['StatusFailed'].format(""))

    @Slot(str)
    def on_queue_error(self, err):
        friendly_msg = translate_ytdlp_error(err)
        self.log_console.appendPlainText(f"\nERROR: {err}\n-> Hướng xử lý: {friendly_msg}")
        if self.current_queue_index >= 0:
            self.download_queue[self.current_queue_index]['status'] = tr['Ytdlp']['StatusQueueFailed']
            self.update_queue_table()
        
        self.reset_ui_state()
        self.start_queue_btn.setEnabled(True)
        # Proceed to next item automatically
        QtCore.QTimer.singleShot(1500, self.process_next_queue_item)

    @Slot(str)
    def on_queue_finished(self, filename):
        if self.current_queue_index >= 0:
            item = self.download_queue[self.current_queue_index]
            item['status'] = tr['Ytdlp']['StatusQueueSuccess']
            item['progress'] = 100.0
            self.update_queue_table()
            
            # Save to history list
            self.add_to_history(filename, item['title'], item['format_desc'], item['thumbnail_bytes'])
            
        self.reset_ui_state()
        self.start_queue_btn.setEnabled(True)
        # Proceed to next item in the queue!
        QtCore.QTimer.singleShot(1000, self.process_next_queue_item)


    # ==================== DOWNLOAD HISTORY LOGIC ====================
    def add_to_history(self, filepath, title, format_desc, thumb_bytes):
        # Exclude duplicates
        for item in self.download_history:
            if item.get('filepath') == filepath:
                return

        # Get file size
        size_str = "Unknown"
        if os.path.exists(filepath):
            try:
                sz = os.path.getsize(filepath)
                if sz > 1024 * 1024 * 1024:
                    size_str = f"{sz / (1024*1024*1024):.2f} GiB"
                else:
                    size_str = f"{sz / (1024*1024):.2f} MiB"
            except Exception:
                pass

        history_item = {
            'filepath': filepath,
            'title': title,
            'format': format_desc,
            'size': size_str,
            'thumbnail_bytes_b64': base64.b64encode(thumb_bytes).decode('utf-8') if thumb_bytes else None
        }
        
        self.download_history.append(history_item)
        # Limit history items count to 20 to prevent settings json bloating
        if len(self.download_history) > 20:
            self.download_history.pop(0)
            
        self.settings['download_history'] = self.download_history
        save_settings(self.settings)
        self.populate_history_table()

    def populate_history_table(self):
        self.history_table.setRowCount(0)
        self.history_table.setRowCount(len(self.download_history))
        
        # Populate history rows in reverse chronological order
        for idx, item in enumerate(reversed(self.download_history)):
            row_idx = idx
            
            # Column 0: Thumbnail
            lbl = QtWidgets.QLabel(self.history_table)
            lbl.setFixedSize(64, 36)
            lbl.setScaledContents(True)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            
            b64_str = item.get('thumbnail_bytes_b64')
            if b64_str:
                try:
                    data = base64.b64decode(b64_str)
                    pix = QtGui.QPixmap()
                    if pix.loadFromData(data):
                        lbl.setPixmap(pix)
                    else:
                        lbl.setText("No Image")
                except Exception:
                    lbl.setText("Error")
            else:
                lbl.setText("No Image")
            self.history_table.setCellWidget(row_idx, 0, lbl)
            
            # Column 1: Title
            title_item = QTableWidgetItem(item['title'])
            title_item.setToolTip(item['filepath'])
            self.history_table.setItem(row_idx, 1, title_item)
            
            # Column 2: Format
            format_item = QTableWidgetItem(item['format'])
            self.history_table.setItem(row_idx, 2, format_item)
            
            # Column 3: Size
            size_item = QTableWidgetItem(item['size'])
            self.history_table.setItem(row_idx, 3, size_item)
            
            # Column 4: Actions (Play, Folder, Import)
            actions = HistoryActionWidget(item['filepath'], self, self.history_table)
            self.history_table.setCellWidget(row_idx, 4, actions)
            self.history_table.setRowHeight(row_idx, 44)

    def clear_history(self):
        self.download_history.clear()
        self.settings['download_history'] = self.download_history
        save_settings(self.settings)
        self.populate_history_table()

    def delete_single_history_item(self, filepath):
        self.download_history = [item for item in self.download_history if item.get('filepath') != filepath]
        self.settings['download_history'] = self.download_history
        save_settings(self.settings)
        self.populate_history_table()


    # ==================== SINGLE DOWNLOAD LOGIC ====================
    def start_download(self):
        if self.is_playlist:
            # If playlist, redirect to queue download
            self.add_to_queue()
            self.start_queue_download()
            return

        url = self.url_input.text().strip()
        if not url:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng nhập đường dẫn video",
                duration=3000,
                parent=self
            )
            return

        save_dir = self.save_dir_input.text().strip()
        if not save_dir:
            save_dir = os.getcwd()

        self.settings['last_url'] = url
        self.settings['save_dir'] = save_dir
        save_settings(self.settings)

        cookies_file = self.cookies_input.text().strip()
        pref_container = self.container_combo.currentText()
        pref_audio = self.audio_format_combo.currentText()

        selected_format_id = None
        selected_format_type = None
        format_opt = 'best'
        format_desc = "Best Quality"
        estimated_bytes = 0
        
        if self.is_analyzed and self.formats_list:
            current_idx = self.format_combo.currentIndex()
            if current_idx >= 0 and current_idx < len(self.formats_list):
                selected_format = self.formats_list[current_idx]
                selected_format_id = selected_format['id']
                selected_format_type = selected_format['raw_type']
                format_desc = f"[{selected_format['quality']}] {selected_format['type']} ({selected_format['ext']})"
                
                # Parse estimated bytes
                size_str = selected_format.get('size', 'Unknown')
                if 'GiB' in size_str:
                    try: estimated_bytes = int(float(size_str.split()[0]) * 1024 * 1024 * 1024)
                    except: pass
                elif 'MiB' in size_str:
                    try: estimated_bytes = int(float(size_str.split()[0]) * 1024 * 1024)
                    except: pass
        else:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng Phân tích link trước khi tải",
                duration=3000,
                parent=self
            )
            return

        # Check disk space before downloading
        is_enough, free_mb, req_mb = self.check_disk_space(save_dir, estimated_bytes)
        if not is_enough:
            self.log_console.appendPlainText(f"\nERROR: Dung lượng ổ đĩa không đủ để bắt đầu tải! (Trống: {free_mb:.1f} MB, Cần ít nhất: {req_mb:.1f} MB)")
            InfoBar.error(
                title="Không đủ dung lượng đĩa trống",
                content=f"Phân vùng ổ đĩa lưu trữ chỉ còn trống {free_mb:.1f} MB, không đủ để tải (Cần ít nhất: {req_mb:.1f} MB gồm 100MB dự phòng).",
                duration=7000,
                parent=self
            )
            return

        self.log_console.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.speed_label.setText("")
        self.eta_label.setText("")
        self.pct_label.setText("")
        self.open_folder_btn.hide()
        self.open_remover_btn.hide()

        self.download_btn.setEnabled(False)
        self.add_queue_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.start_queue_btn.setEnabled(False)
        self.url_input.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.choose_dir_btn.setEnabled(False)
        self.choose_cookies_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.format_combo.setEnabled(False)

        custom_fn = self.filename_input.text().strip() if hasattr(self, 'filename_input') else None
        self.worker = YtdlpWorker(
            url, save_dir, format_opt, selected_format_id, selected_format_type, cookies_file, pref_container, pref_audio, self.get_selected_threads(), custom_fn
        )
        self.worker.progress_sig.connect(self.on_progress)
        self.worker.speed_sig.connect(self.on_speed)
        self.worker.eta_sig.connect(self.on_eta)
        self.worker.status_sig.connect(self.on_status)
        self.worker.log_sig.connect(self.on_log)
        self.worker.error_sig.connect(self.on_error)
        self.worker.finished_sig.connect(self.on_finished)
        self.worker.start()

    def cancel_download(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.log_console.appendPlainText("\nCancelling download...")

    @Slot(float)
    def on_progress(self, val):
        self.progress_bar.setValue(int(val))
        self.pct_label.setText(tr['Ytdlp']['Progress'].format(f"{val:.1f}"))

    @Slot(str)
    def on_speed(self, speed):
        self.speed_label.setText(tr['Ytdlp']['Speed'].format(speed))

    @Slot(str)
    def on_eta(self, eta):
        self.eta_label.setText(tr['Ytdlp']['Eta'].format(eta))

    @Slot(str)
    def on_status(self, status):
        if status == 'downloading':
            self.status_label.setText(tr['Ytdlp']['StatusDownloading'])
        elif status == 'finished':
            self.status_label.setText(tr['Ytdlp']['StatusSuccess'])
        elif status == 'cancelled':
            self.status_label.setText(tr['Ytdlp']['StatusCancelled'])
            self.reset_ui_state()
            self.start_queue_btn.setEnabled(True)
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content=tr['Ytdlp']['StatusCancelled'],
                duration=3000,
                parent=self
            )
        elif status == 'error':
            self.status_label.setText(tr['Ytdlp']['StatusFailed'].format(""))
            self.reset_ui_state()
            self.start_queue_btn.setEnabled(True)

    @Slot(str)
    def on_log(self, msg):
        self.log_console.appendPlainText(msg)

    @Slot(str)
    def on_error(self, err):
        friendly_msg = translate_ytdlp_error(err)
        self.log_console.appendPlainText(f"\nERROR: {err}\n-> Hướng xử lý: {friendly_msg}")
        self.status_label.setText(friendly_msg)
        InfoBar.error(
            title=tr['Ytdlp']['Title'],
            content=friendly_msg,
            duration=6000,
            parent=self
        )

    @Slot(str)
    def on_finished(self, filename):
        self.reset_ui_state()
        self.start_queue_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(tr['Ytdlp']['StatusSuccess'])
        
        self.downloaded_filepath = filename
        self.open_folder_btn.show()
        
        if not self.is_playlist:
            _, ext = os.path.splitext(filename)
            if ext.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm']:
                self.open_remover_btn.show()
            else:
                self.open_remover_btn.hide()
        else:
            self.open_remover_btn.hide()

        # Add single download to persistent history
        current_idx = self.format_combo.currentIndex()
        format_desc = "Best Quality"
        if self.is_analyzed and self.formats_list and current_idx >= 0:
            fmt = self.formats_list[current_idx]
            format_desc = f"[{fmt['quality']}] {fmt['type']} ({fmt['ext']})"
        self.add_to_history(filename, self.meta_title.text(), format_desc, self.current_thumbnail_bytes)

        InfoBar.success(
            title=tr['Ytdlp']['Title'],
            content=f"Đã tải xong: {os.path.basename(filename)}\nDownload success!",
            duration=5000,
            parent=self
        )

    def reset_ui_state(self):
        self.download_btn.setEnabled(self.is_analyzed)
        self.add_queue_btn.setEnabled(self.is_analyzed)
        self.cancel_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.choose_dir_btn.setEnabled(True)
        self.choose_cookies_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.format_combo.setEnabled(True)
        self.speed_label.setText("")
        self.eta_label.setText("")
        self.pct_label.setText("")
