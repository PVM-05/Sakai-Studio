import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.desktop.ui.ytdlp_interface import translate_ytdlp_error, resolve_playlist_entry_url


def test_translate_ytdlp_bot_detection():
    bot_err = "ERROR: [youtube] vujGXU3E5do: Sign in to confirm you're not a bot. This helps protect our community."
    result = translate_ytdlp_error(bot_err)
    assert "rô-bốt" in result or "robot" in result or "bảo mật" in result
    assert "riêng tư" not in result  # Must NOT confuse bot check with private video

def test_translate_ytdlp_private_video():
    priv_err = "ERROR: [youtube] abc: This video is private"
    result = translate_ytdlp_error(priv_err)
    assert "riêng tư" in result

def test_translate_ytdlp_cookie_dpapi():
    dpapi_err = "ERROR: Failed to decrypt with DPAPI. See https://github.com/yt-dlp/yt-dlp/issues/10927"
    result = translate_ytdlp_error(dpapi_err)
    assert "trình duyệt" in result.lower()

def test_translate_ytdlp_network():
    net_err = "ERROR: [youtube] unable to download video data: The read operation timed out"
    result = translate_ytdlp_error(net_err)
    assert "mạng" in result.lower() or "kết nối" in result.lower()

def test_resolve_playlist_entry_url_youtube_with_list():
    entry = {
        'id': 'vujGXU3E5do',
        'url': 'https://www.youtube.com/watch?v=vujGXU3E5do&list=PLuOiNVUFtTQlGGUwM_vrenA8DFIMpzvmI',
        'title': 'Test Video'
    }
    url = resolve_playlist_entry_url(entry, parent_url="https://www.youtube.com/playlist?list=PLuOiNVUFtTQlGGUwM_vrenA8DFIMpzvmI")
    assert url == "https://www.youtube.com/watch?v=vujGXU3E5do"

def test_resolve_playlist_entry_url_youtube_id_only():
    entry = {
        'id': 'vujGXU3E5do',
        'title': 'Test Video'
    }
    url = resolve_playlist_entry_url(entry, parent_url="https://www.youtube.com/playlist?list=PLuOiNVUFtTQlGGUwM_vrenA8DFIMpzvmI")
    assert url == "https://www.youtube.com/watch?v=vujGXU3E5do"

def test_resolve_playlist_entry_url_bilibili():
    entry = {
        'id': 'BV1py4y1c71k',
        'title': 'Bilibili Ep 1'
    }
    url = resolve_playlist_entry_url(entry, parent_url="https://www.bilibili.com/video/BV1py4y1c71k")
    assert url == "https://www.bilibili.com/video/BV1py4y1c71k"


def test_resolve_playlist_entry_url_invalid():
    assert resolve_playlist_entry_url(None) == ""
    assert resolve_playlist_entry_url({}) == ""


