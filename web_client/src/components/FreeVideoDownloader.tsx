"use client";

import React, { useState, useRef } from "react";
import { apiGet, apiPost, createEventSource, formatDuration, API_BASE } from "../lib/api";

interface FormatItem {
  format_id: string;
  resolution: string;
  ext: string;
  vcodec: string;
  acodec: string;
  filesize: number;
}

interface AnalyzeResult {
  title: string;
  channel: string;
  duration: number;
  thumbnail: string;
  formats: FormatItem[];
}

const SUPPORTED_PLATFORMS = [
  { name: "YouTube", tag: "Video, Shorts" },
  { name: "TikTok", tag: "Không hình mờ" },
  { name: "Facebook", tag: "Video, Thước phim" },
  { name: "Douyin", tag: "Độ nét cao" },
  { name: "Bilibili", tag: "Video trực tuyến" },
  { name: "Instagram", tag: "Thước phim ngắn" },
];

export default function FreeVideoDownloader() {
  const [url, setUrl] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);

  const [audioOnly, setAudioOnly] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState("");
  const [audioFormat, setAudioFormat] = useState("mp3");

  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState("");
  const [statusText, setStatusText] = useState("");
  const [completedUrl, setCompletedUrl] = useState<string | null>(null);
  const [completedFilename, setCompletedFilename] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        setUrl(text);
        setAnalyzeError(null);
      }
    } catch {
      // Bỏ qua nếu không có quyền truy cập bộ nhớ tạm
    }
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    setAnalyzing(true);
    setAnalyzeError(null);
    setResult(null);
    setCompletedUrl(null);
    setProgress(0);

    try {
      const data = await apiGet(`/ytdlp/info?url=${encodeURIComponent(url.trim())}`);
      setResult(data);
      if (data.formats && data.formats.length > 0) {
        setSelectedFormat(data.formats[0].format_id);
      }
    } catch (err: any) {
      setAnalyzeError(err.message || "Không thể phân tích liên kết. Vui lòng kiểm tra lại đường dẫn video.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleStartDownload = async () => {
    if (!result) return;

    setDownloading(true);
    setProgress(0);
    setSpeed("");
    setStatusText("Đang khởi tạo tiến trình tải xuống...");
    setCompletedUrl(null);

    try {
      const res = await apiPost("/ytdlp/download", {
        url: url.trim(),
        format_id: audioOnly ? undefined : (selectedFormat || undefined),
        audio_only: audioOnly,
        audio_format: audioOnly ? audioFormat : undefined,
      });

      const newJobId = res.job_id;
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = createEventSource(`/ytdlp/stream-progress/${newJobId}`);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          setProgress(payload.progress || 0);
          setSpeed(payload.speed || "");
          setStatusText(payload.status || "Đang xử lý...");

          if (payload.status === "Hoàn thành") {
            setDownloading(false);
            const downloadLink = payload.output_path
              ? `${API_BASE}/downloads/${encodeURIComponent(payload.output_path.split(/[/\\]/).pop())}`
              : `${API_BASE}/downloads/${newJobId}`;
            setCompletedUrl(downloadLink);
            setCompletedFilename(payload.filename || result.title);
            es.close();
          } else if (payload.status === "Thất bại") {
            setDownloading(false);
            setAnalyzeError(payload.error_message || "Quá trình tải video thất bại.");
            es.close();
          }
        } catch (e) {
          console.error("Lỗi phân tích luồng tiến trình:", e);
        }
      };

      es.onerror = () => {
        es.close();
      };
    } catch (err: any) {
      setDownloading(false);
      setAnalyzeError(err.message || "Không thể bắt đầu tải xuống.");
    }
  };

  return (
    <div className="w-full bg-slate-900/90 border border-white/10 rounded-2xl p-6 lg:p-8 backdrop-blur-xl shadow-2xl relative">
      {/* Khung tiêu đề tiện ích */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-white/10">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Tiện ích web miễn phí hoàn toàn
          </div>
          <h3 className="text-xl lg:text-2xl font-bold text-white tracking-tight">
            Tải video trực tuyến chất lượng cao
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            Dán đường dẫn video từ mạng xã hội để tải về máy tính với độ phân giải gốc siêu nét
          </p>
        </div>

        {/* Huy hiệu các nền tảng hỗ trợ */}
        <div className="flex flex-wrap gap-1.5 max-w-md">
          {SUPPORTED_PLATFORMS.map((platform) => (
            <span
              key={platform.name}
              className="text-xs px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300"
            >
              {platform.name}
            </span>
          ))}
        </div>
      </div>

      {/* Biểu mẫu nhập liên kết */}
      <form onSubmit={handleAnalyze} className="mt-6">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <input
              type="url"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setAnalyzeError(null);
              }}
              placeholder="Dán đường dẫn video từ YouTube, TikTok, Facebook, Douyin..."
              className="w-full bg-slate-950/70 border border-white/15 rounded-xl px-4 py-3.5 pl-11 pr-24 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
            />
            <div className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
            </div>
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
              {url && (
                <button
                  type="button"
                  onClick={() => {
                    setUrl("");
                    setResult(null);
                    setAnalyzeError(null);
                  }}
                  className="p-1.5 text-slate-400 hover:text-white rounded-md transition-colors"
                  title="Xóa nội dung"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
              <button
                type="button"
                onClick={handlePaste}
                className="px-2.5 py-1 text-xs font-medium text-slate-300 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-colors"
              >
                Dán
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={analyzing || !url.trim()}
            className="px-6 py-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/25 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
          >
            {analyzing ? (
              <>
                <svg className="animate-spin w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Đang phân tích...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <span>Lấy thông tin</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Thông báo lỗi */}
      {analyzeError && (
        <div className="mt-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-sm flex items-start gap-2.5">
          <svg className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="leading-relaxed break-words">{analyzeError}</p>
        </div>
      )}

      {/* Kết quả phân tích và tùy chọn tải */}
      {result && (
        <div className="mt-6 p-5 rounded-xl bg-slate-950/60 border border-white/10 space-y-5 animate-in fade-in duration-300">
          <div className="flex flex-col sm:flex-row gap-4 items-start">
            {result.thumbnail && (
              <img
                src={result.thumbnail}
                alt={result.title}
                className="w-full sm:w-44 aspect-video rounded-lg object-cover border border-white/10 shadow-md flex-shrink-0"
              />
            )}
            <div className="min-w-0 flex-1">
              <h4 className="font-semibold text-white text-base leading-snug line-clamp-2">
                {result.title}
              </h4>
              <div className="flex flex-wrap items-center gap-3 mt-2 text-xs text-slate-400">
                {result.channel && <span>Kênh: <strong className="text-slate-300">{result.channel}</strong></span>}
                {result.duration > 0 && <span>Thời lượng: <strong className="text-slate-300">{formatDuration(result.duration)}</strong></span>}
              </div>

              {/* Chuyển đổi Video hoặc Âm thanh */}
              <div className="flex items-center gap-2 mt-4">
                <button
                  type="button"
                  onClick={() => setAudioOnly(false)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    !audioOnly
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "bg-white/5 text-slate-400 hover:text-white"
                  }`}
                >
                  Tải tệp video
                </button>
                <button
                  type="button"
                  onClick={() => setAudioOnly(true)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    audioOnly
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "bg-white/5 text-slate-400 hover:text-white"
                  }`}
                >
                  Chỉ trích xuất âm thanh
                </button>
              </div>
            </div>
          </div>

          {/* Lựa chọn chất lượng video */}
          {!audioOnly && result.formats && result.formats.length > 0 && (
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-2">
                Chọn độ phân giải video
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {result.formats.slice(0, 4).map((f) => (
                  <button
                    key={f.format_id}
                    type="button"
                    onClick={() => setSelectedFormat(f.format_id)}
                    className={`p-2.5 rounded-lg border text-left transition-all ${
                      selectedFormat === f.format_id
                        ? "bg-indigo-600/20 border-indigo-500 text-white"
                        : "bg-white/[0.03] border-white/5 text-slate-300 hover:border-white/20"
                    }`}
                  >
                    <p className="text-sm font-semibold">{f.resolution || "Tiêu chuẩn"}</p>
                    <p className="text-xs text-slate-400 font-mono mt-0.5">{f.ext.toUpperCase()}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Lựa chọn định dạng âm thanh */}
          {audioOnly && (
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-2">
                Chọn định dạng tệp âm thanh
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {[
                  { id: "mp3", label: "MP3", desc: "320 kbps chất lượng cao" },
                  { id: "m4a", label: "M4A", desc: "AAC chuẩn" },
                  { id: "wav", label: "WAV", desc: "Không nén" },
                  { id: "flac", label: "FLAC", desc: "Chất lượng phòng thu" },
                ].map((af) => (
                  <button
                    key={af.id}
                    type="button"
                    onClick={() => setAudioFormat(af.id)}
                    className={`p-2.5 rounded-lg border text-left transition-all ${
                      audioFormat === af.id
                        ? "bg-indigo-600/20 border-indigo-500 text-white"
                        : "bg-white/[0.03] border-white/5 text-slate-300 hover:border-white/20"
                    }`}
                  >
                    <p className="text-sm font-semibold">{af.label}</p>
                    <p className="text-[11px] text-slate-400 mt-0.5">{af.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Tiến trình tải xuống */}
          {downloading && (
            <div className="space-y-2 pt-2 border-t border-white/5">
              <div className="flex justify-between text-xs text-slate-400">
                <span>{statusText || "Đang tải xuống..."}</span>
                <span className="font-mono text-indigo-400 font-semibold">{progress}%</span>
              </div>
              <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-white/10">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              {speed && (
                <p className="text-[11px] font-mono text-slate-400 text-right">
                  Tốc độ: {speed}
                </p>
              )}
            </div>
          )}

          {/* Khối tải hoàn tất */}
          {completedUrl && (
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/25 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-400 flex-shrink-0">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Tải về hoàn tất</p>
                  <p className="text-xs text-slate-300 truncate max-w-sm">{completedFilename || "Tệp của bạn đã sẵn sàng"}</p>
                </div>
              </div>
              <a
                href={completedUrl}
                download
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 shadow-md shadow-emerald-600/20"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                <span>Lưu tệp về máy tính</span>
              </a>
            </div>
          )}

          {/* Nút kích hoạt tải */}
          {!downloading && !completedUrl && (
            <button
              type="button"
              onClick={handleStartDownload}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 active:opacity-90 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/25 flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <span>{audioOnly ? "Bắt đầu trích xuất âm thanh" : "Bắt đầu tải video"}</span>
            </button>
          )}

          {/* Lời kêu gọi chuyển đổi sang phần mềm máy tính */}
          <div className="pt-3 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
            <span className="text-center sm:text-left">
              Cần tự động xóa phụ đề, trích xuất văn bản hoặc dịch thuật hàng loạt video vừa tải?
            </span>
            <a
              href="#so-sanh-phien-ban"
              className="text-indigo-400 hover:text-indigo-300 font-semibold underline underline-offset-4 flex-shrink-0"
            >
              Khám phá phần mềm máy tính Sakai Studio
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
