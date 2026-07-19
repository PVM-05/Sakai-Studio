# -*- coding: utf-8 -*-
"""
@desc: YT-DLP Video Downloader Interface with Metadata Display, Action Links, and Brand Logos
"""
import os
import json
import urllib.request
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QGridLayout
from qfluentwidgets import (ScrollArea, CardWidget, LineEdit, PushButton, 
                           PrimaryPushButton, ComboBox, ProgressBar, PlainTextEdit,
                           BodyLabel, TitleLabel, FluentIcon, InfoBar)
import yt_dlp
from backend.config import config, tr

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
                'noplaylist': False,  # Support playlist analysis!
                'extract_flat': 'in_playlist',  # Extract quickly for playlists
                'js_runtimes': {
                    'deno': {},
                    'node': {},
                    'quickjs': {}
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

    def __init__(self, url, save_dir, format_opt, selected_format_id=None, selected_format_type=None, cookiefile=None):
        super().__init__()
        self.url = url
        self.save_dir = save_dir
        self.format_opt = format_opt
        self.selected_format_id = selected_format_id
        self.selected_format_type = selected_format_type
        self.cookiefile = cookiefile
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

        ydl_opts = {
            'outtmpl': os.path.join(self.save_dir, '%(title)s.%(ext)s'),
            'progress_hooks': [progress_hook],
            'logger': YtdlpLogger(self.log_sig),
            'noprogress': True,
            'js_runtimes': {
                'deno': {},
                'node': {},
                'quickjs': {}
            }
        }

        if self.cookiefile and os.path.exists(self.cookiefile):
            ydl_opts['cookiefile'] = self.cookiefile

        if self.selected_format_id:
            if self.selected_format_type == 'Video Only':
                ydl_opts['format'] = f"{self.selected_format_id}+bestaudio/best"
                ydl_opts['merge_output_format'] = 'mp4'
            else:
                ydl_opts['format'] = self.selected_format_id
        else:
            if self.format_opt == 'audio_only':
                ydl_opts['format'] = 'bestaudio/best'
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            else:
                ydl_opts['format'] = 'bestvideo+bestaudio/best'
                ydl_opts['merge_output_format'] = 'mp4'

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
                        if os.path.exists(base + ".mp3"):
                            filename = base + ".mp3"
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
        
        self.settings = load_settings()
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

        # Metadata Card (Thumbnail & Text Details) - FIXED CSS Selector to prevent inheriting round borders
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

        # Download Thumbnail Button
        self.download_thumb_btn = PushButton(self.meta_card)
        self.download_thumb_btn.setIcon(FluentIcon.DOWNLOAD)
        self.download_thumb_btn.clicked.connect(self.download_thumbnail_file)
        self.download_thumb_btn.setFixedSize(160, 26)
        
        self.meta_text_layout.addWidget(self.meta_title)
        self.meta_text_layout.addWidget(self.meta_author)
        self.meta_text_layout.addWidget(self.meta_duration)
        self.meta_text_layout.addWidget(self.download_thumb_btn)
        self.meta_text_layout.addStretch()
        
        self.meta_card_layout.addWidget(self.thumbnail_label)
        self.meta_card_layout.addLayout(self.meta_text_layout)
        self.meta_card_layout.addStretch()
        
        self.card_layout.addWidget(self.meta_card)
        self.meta_card.hide()

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

        # Save Directory Selection
        self.save_dir_label = BodyLabel(self.card)
        self.save_dir_layout = QHBoxLayout()
        self.save_dir_input = LineEdit(self.card)
        self.save_dir_input.setReadOnly(True)
        default_dir = self.settings.get('save_dir', config.saveDirectory.value if config.saveDirectory.value else os.getcwd())
        self.save_dir_input.setText(default_dir)
        
        self.choose_dir_btn = PushButton(self.card)
        self.choose_dir_btn.setIcon(FluentIcon.FOLDER)
        self.choose_dir_btn.clicked.connect(self.choose_save_dir)
        
        self.save_dir_layout.addWidget(self.save_dir_input)
        self.save_dir_layout.addWidget(self.choose_dir_btn)
        
        self.card_layout.addWidget(self.save_dir_label)
        self.card_layout.addLayout(self.save_dir_layout)

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

        # Action Buttons Layout (Download/Cancel, and Folder/Remover shortcuts)
        self.buttons_layout = QHBoxLayout()
        self.download_btn = PrimaryPushButton(self.card)
        self.download_btn.clicked.connect(self.start_download)
        self.download_btn.setEnabled(False)  # Require link analysis first!
        
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
        self.buttons_layout.addWidget(self.cancel_btn)
        self.buttons_layout.addWidget(self.open_folder_btn)
        self.buttons_layout.addWidget(self.open_remover_btn)
        self.buttons_layout.addStretch()
        self.card_layout.addLayout(self.buttons_layout)

        self.main_layout.addWidget(self.card)

        # 3. Log Console
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
        self.log_console_label.setText(tr['Ytdlp']['LogConsole'])
        
        self.download_btn.setText(tr['Ytdlp']['BtnDownload'])
        self.cancel_btn.setText(tr['Ytdlp']['BtnCancel'])
        self.open_folder_btn.setText(tr['Ytdlp']['BtnOpenFolder'])
        self.open_remover_btn.setText(tr['Ytdlp']['BtnOpenRemover'])
        self.download_thumb_btn.setText(tr['Ytdlp']['BtnDownloadThumb'])
        
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
        folder = QFileDialog.getExistingDirectory(self, tr['Ytdlp']['SelectFolder'], self.save_dir_input.text())
        if folder:
            self.save_dir_input.setText(folder)
            self.settings['save_dir'] = folder
            save_settings(self.settings)

    def choose_cookies_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            tr['Ytdlp']['CookiesSelect'], 
            "", 
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.cookies_input.setText(file_path)
            self.settings['cookies_file'] = file_path
            save_settings(self.settings)

    def paste_from_clipboard(self):
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            self.url_input.setText(text)

    def open_download_folder(self):
        if hasattr(self, 'downloaded_filepath') and self.downloaded_filepath and os.path.exists(self.downloaded_filepath):
            try:
                from showinfm import show_in_file_manager
                show_in_file_manager(self.downloaded_filepath)
            except Exception:
                os.startfile(os.path.dirname(self.downloaded_filepath))

    def open_in_remover(self):
        if hasattr(self, 'downloaded_filepath') and self.downloaded_filepath and os.path.exists(self.downloaded_filepath):
            if hasattr(self.parent, 'open_video_in_remover'):
                self.parent.open_video_in_remover(self.downloaded_filepath)

    def download_thumbnail_file(self):
        if not self.current_thumbnail_url:
            return
        
        default_name = "thumbnail.jpg"
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Lưu ảnh Thumbnail / Save Thumbnail", 
            os.path.join(self.save_dir_input.text(), default_name), 
            "Images (*.jpg *.png);;All Files (*)"
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
                    content=f"Đã lưu thumbnail tại / Saved thumbnail: {os.path.basename(save_path)}",
                    duration=3000,
                    parent=self
                )
            except Exception as e:
                InfoBar.error(
                    title=tr['Ytdlp']['Title'],
                    content=f"Lỗi tải thumbnail / Error: {e}",
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
        self.thumbnail_label.setPixmap(QtGui.QPixmap())
        self.thumbnail_label.setText("")
        self.meta_title.setText("")
        self.meta_author.setText("")
        self.meta_duration.setText("")
        self.current_thumbnail_url = None
        self.downloaded_filepath = None
        
        self.format_combo.clear()
        self.format_combo.addItems([
            tr['Ytdlp']['FormatPlaceholder']
        ])
        self.format_combo.setCurrentIndex(0)
        
        self.download_btn.setEnabled(False)
        self.open_folder_btn.hide()
        self.open_remover_btn.hide()

    def start_analysis(self):
        url = self.url_input.text().strip()
        if not url:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng nhập đường dẫn video / Please input video URL",
                duration=3000,
                parent=self
            )
            return

        self.settings['last_url'] = url
        save_settings(self.settings)

        self.analyze_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
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
        self.open_folder_btn.hide()
        self.open_remover_btn.hide()

        cookies_file = self.cookies_input.text().strip()
        self.analysis_worker = YtdlpAnalysisWorker(url, cookies_file)
        self.analysis_worker.finished_sig.connect(self.on_analysis_success)
        self.analysis_worker.error_sig.connect(self.on_analysis_failed)
        self.analysis_worker.start()

    @Slot(dict)
    def on_analysis_success(self, info):
        self.analyze_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.url_input.setEnabled(True)
        self.choose_cookies_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(tr['Ytdlp']['StatusExtractSuccess'])
        
        # Populate Metadata details
        self.meta_card.show()
        self.meta_title.setText(info.get('title', 'Unknown Title'))
        self.meta_author.setText(info.get('uploader') or info.get('channel', 'Unknown Channel'))
        
        # Check if it is a playlist
        is_playlist_type = info.get('_type') == 'playlist' or 'entries' in info
        self.is_playlist = is_playlist_type
        
        if is_playlist_type:
            entries = info.get('entries', [])
            self.meta_duration.setText(f"Danh sách phát: {len(entries)} video")
            
            thumb_url = info.get('thumbnail')
            if not thumb_url and entries and entries[0]:
                thumb_url = entries[0].get('thumbnail') or (entries[0].get('thumbnails')[0].get('url') if entries[0].get('thumbnails') else None)
        else:
            duration_secs = info.get('duration')
            if duration_secs:
                try:
                    duration_secs = int(duration_secs)  # Avoid ValueError: Unknown format code 'd' for object of type 'float'
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

        # Setup thumbnail downloader
        self.thumbnail_label.setPixmap(QtGui.QPixmap())
        self.thumbnail_label.setText("Loading...")
        
        if thumb_url:
            self.thumb_downloader = ThumbnailDownloader(thumb_url)
            self.thumb_downloader.finished_sig.connect(self.on_thumbnail_downloaded)
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
        pixmap = QtGui.QPixmap()
        if pixmap.loadFromData(data):
            self.thumbnail_label.setPixmap(pixmap)
            self.thumbnail_label.setText("")

    @Slot(str)
    def on_analysis_failed(self, err):
        self.analyze_btn.setEnabled(True)
        self.download_btn.setEnabled(False) # Keep disabled!
        self.url_input.setEnabled(True)
        self.choose_cookies_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(tr['Ytdlp']['StatusExtractFailed'].format(err[:20]))
        
        InfoBar.error(
            title=tr['Ytdlp']['Title'],
            content=tr['Ytdlp']['StatusExtractFailed'].format(err[:100]),
            duration=5000,
            parent=self
        )

    def start_download(self):
        url = self.url_input.text().strip()
        if not url:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng nhập đường dẫn video / Please input video URL",
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

        selected_format_id = None
        selected_format_type = None
        format_opt = 'best'
        
        if self.is_analyzed and self.formats_list:
            current_idx = self.format_combo.currentIndex()
            if current_idx >= 0 and current_idx < len(self.formats_list):
                fid = self.formats_list[current_idx]['id']
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
                    selected_format_type = self.formats_list[current_idx]['raw_type']
        else:
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content="Vui lòng Phân tích link trước khi tải / Please analyze link first",
                duration=3000,
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
        self.cancel_btn.setEnabled(True)
        self.url_input.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.choose_dir_btn.setEnabled(False)
        self.choose_cookies_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.format_combo.setEnabled(False)

        self.worker = YtdlpWorker(
            url, save_dir, format_opt, selected_format_id, selected_format_type, cookies_file
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
            InfoBar.warning(
                title=tr['Ytdlp']['Title'],
                content=tr['Ytdlp']['StatusCancelled'],
                duration=3000,
                parent=self
            )
        elif status == 'error':
            self.status_label.setText(tr['Ytdlp']['StatusFailed'].format(""))
            self.reset_ui_state()

    @Slot(str)
    def on_log(self, msg):
        self.log_console.appendPlainText(msg)

    @Slot(str)
    def on_error(self, err):
        self.log_console.appendPlainText(f"\nERROR: {err}")
        InfoBar.error(
            title=tr['Ytdlp']['Title'],
            content=tr['Ytdlp']['StatusFailed'].format(err[:100]),
            duration=5000,
            parent=self
        )

    @Slot(str)
    def on_finished(self, filename):
        self.reset_ui_state()
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

        InfoBar.success(
            title=tr['Ytdlp']['Title'],
            content=f"Đã tải xong: {os.path.basename(filename)}\nDownload success!",
            duration=5000,
            parent=self
        )

    def reset_ui_state(self):
        self.download_btn.setEnabled(True)
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
