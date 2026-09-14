"use client";

import { useState, useEffect, useRef } from "react";
import { apiGet, apiPost, createEventSource, formatDuration, API_BASE } from "../../../lib/api";
import DownloadHistory, { DownloadItem } from "../../../components/DownloadHistory";

interface AnalyzeResult {
  title: string;
  channel: string;
  duration: number;
  thumbnail: string;
  formats: {
    format_id: string;
    resolution: string;
    ext: string;
    vcodec: string;
    acodec: string;
    filesize: number;
  }[];
}

export default function DownloadPage() {
  const [url, setUrl] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(null);
  
  const [audioOnly, setAudioOnly] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState("");
  const [audioFormat, setAudioFormat] = useState("mp3");
  const [customFilename, setCustomFilename] = useState("");
  
  const [downloading, setDownloading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState("");
  const [eta, setEta] = useState("");
  const [statusText, setStatusText] = useState("");
  
  const [history, setHistory] = useState<DownloadItem[]>([]);
  const [transferMsg, setTransferMsg] = useState("");
  const [completedFile, setCompletedFile] = useState<{ jobId: string; resultUrl: string; filename?: string } | null>(null);
  
  const eventSourceRef = useRef<EventSource | null>(null);

  const fetchHistory = async () => {
    try {
      const data = await apiGet("/ytdlp/history");
      setHistory(data);
    } catch (err) {
      console.error("Lỗi khi tải lịch sử:", err);
    }
  };

  useEffect(() => {
    fetchHistory();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) setUrl(text);
    } catch (err) {
      alert("Không thể dán từ khay nhớ tạm. Vui lòng cấp quyền hoặc dán thủ công.");
    }
  };

  const handleAnalyze = async () => {
    if (!url) return;
    setAnalyzing(true);
    setAnalyzeResult(null);
    try {
      const data = await apiPost("/ytdlp/analyze", { url });
      setAnalyzeResult(data);
      if (data.formats && data.formats.length > 0) {
        const defaultVideo = data.formats.find((f: any) => f.vcodec !== "none" && f.acodec !== "none") || data.formats[0];
        setSelectedFormat(defaultVideo.format_id);
      }
    } catch (err: any) {
      alert(err.message || "Lỗi khi phân tích liên kết.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSaveWithPicker = async (targetFile?: { resultUrl: string; filename?: string }) => {
    const file = targetFile || completedFile;
    if (!file) return;
    const fullUrl = `${API_BASE}${file.resultUrl}`;
    const defaultName = file.filename ? `${file.filename}.${audioOnly ? audioFormat : 'mp4'}` : `video.${audioOnly ? audioFormat : 'mp4'}`;

    if (typeof window !== "undefined" && "showSaveFilePicker" in window) {
      try {
        const ext = `.${audioOnly ? audioFormat : 'mp4'}`;
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: defaultName,
          types: [{
            description: audioOnly ? "Tệp âm thanh" : "Tệp video",
            accept: {
              [audioOnly ? `audio/${audioFormat}` : "video/mp4"]: [ext],
            },
          }],
        });
        
        const writable = await handle.createWritable();
        const response = await fetch(fullUrl);
        if (response.body) {
          await response.body.pipeTo(writable);
        }
        return;
      } catch (err: any) {
        if (err.name === "AbortError") {
          return;
        }
      }
    }

    const a = document.createElement("a");
    a.href = fullUrl;
    a.download = defaultName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const startDownload = async () => {
    if (!url) return;
    setDownloading(true);
    setCompletedFile(null);
    setProgress(0);
    setSpeed("");
    setEta("");
    setStatusText("Đang khởi tạo...");
    setTransferMsg("");
    
    try {
      const payload: any = {
        url,
        audio_only: audioOnly,
        custom_filename: customFilename || undefined,
      };
      if (audioOnly) {
        payload.audio_format = audioFormat;
      } else {
        payload.format_id = selectedFormat;
      }
      
      const data = await apiPost("/ytdlp/download", payload);
      setJobId(data.job_id);
      
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      
      const es = createEventSource(`/ytdlp/events/${data.job_id}`);
      eventSourceRef.current = es;
      
      es.onmessage = (e) => {
        try {
          const evData = JSON.parse(e.data);
          if (evData.status === "downloading") {
            setStatusText("Đang tải xuống...");
            setProgress(evData.progress || 0);
            if (evData.speed) setSpeed(evData.speed);
            if (evData.eta) setEta(evData.eta);
          } else if (evData.status === "finished" || evData.task_status === "SUCCESS") {
            setStatusText("Hoàn thành tải xuống!");
            setProgress(100);
            setSpeed("");
            setEta("");
            
            const fileUrl = evData.result_url || evData.meta?.result_url || `/ytdlp/file/${data.job_id}/result`;
            setCompletedFile({
              jobId: data.job_id,
              resultUrl: fileUrl,
              filename: evData.meta?.filename || customFilename || analyzeResult?.title || "video"
            });
            
            const a = document.createElement("a");
            a.href = `${API_BASE}${fileUrl}`;
            a.download = "";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            es.close();
            setDownloading(false);
            fetchHistory();
          } else if (evData.status === "error") {
            setStatusText("Lỗi: " + (evData.error || "Không xác định"));
            es.close();
            setDownloading(false);
            fetchHistory();
          }
        } catch (err) {}
      };
      
      es.onerror = () => {
        es.close();
        setDownloading(false);
        fetchHistory();
      };
      
    } catch (err: any) {
      alert(err.message || "Lỗi khi bắt đầu tải xuống.");
      setDownloading(false);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;
    try {
      await apiPost(`/ytdlp/cancel/${jobId}`, {});
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      setDownloading(false);
      setStatusText("Đã hủy tải xuống.");
      fetchHistory();
    } catch (err: any) {
      alert(err.message || "Lỗi khi hủy.");
    }
  };

  const handleTransfer = async (id: string) => {
    try {
      await apiPost(`/ytdlp/transfer/${id}`, {});
      setTransferMsg("Chuyển video sang khu vực xóa phụ đề thành công! Đang chuyển hướng...");
      setTimeout(() => {
        window.location.href = "/dashboard";
      }, 1500);
    } catch (err: any) {
      alert(err.message || "Lỗi khi chuyển.");
    }
  };

  return (
    <>
      <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-slate-900/50 backdrop-blur-md">
        <h1 className="text-xl font-medium tracking-tight">Tải video trực tuyến</h1>
      </header>

      <div className="flex-1 overflow-y-auto p-8 space-y-8">
        
        {transferMsg && (
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
            </svg>
            {transferMsg}
          </div>
        )}

        {/* Khu vực nhập liên kết */}
        <div className="bg-slate-900 border border-white/5 rounded-2xl p-6">
          <label className="block text-sm font-medium text-slate-300 mb-2">Liên kết video</label>
          <div className="flex gap-3">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Nhập liên kết video từ YouTube, TikTok, Facebook..."
              className="flex-1 bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
            />
            <button
              onClick={handlePaste}
              className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm font-medium transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              Dán liên kết
            </button>
            <button
              onClick={handleAnalyze}
              disabled={!url || analyzing || downloading}
              className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 text-white rounded-xl text-sm font-medium transition-colors flex items-center gap-2"
            >
              {analyzing ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              )}
              Phân tích
            </button>
          </div>
        </div>

        {/* Kết quả phân tích & Tuỳ chọn tải */}
        {analyzeResult && (
          <div className="bg-slate-900 border border-white/5 rounded-2xl p-6 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="md:col-span-1 space-y-4">
              <div className="aspect-video rounded-xl overflow-hidden bg-slate-800 relative">
                {analyzeResult.thumbnail ? (
                  <img src={analyzeResult.thumbnail} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <span className="text-slate-500">Không có ảnh thu nhỏ</span>
                  </div>
                )}
                <div className="absolute bottom-2 right-2 bg-black/80 px-2 py-1 rounded text-xs font-medium">
                  {formatDuration(analyzeResult.duration)}
                </div>
              </div>
              <div>
                <h3 className="font-medium text-slate-200 line-clamp-2">{analyzeResult.title}</h3>
                <p className="text-sm text-slate-400 mt-1">{analyzeResult.channel}</p>
              </div>
            </div>
            
            <div className="md:col-span-2 space-y-6">
              <div className="flex bg-slate-950 p-1 rounded-xl">
                <button
                  onClick={() => setAudioOnly(false)}
                  className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${!audioOnly ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Video đầy đủ
                </button>
                <button
                  onClick={() => setAudioOnly(true)}
                  className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${audioOnly ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Chỉ lấy âm thanh
                </button>
              </div>

              <div className="space-y-4">
                {!audioOnly ? (
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">Độ phân giải</label>
                    <select
                      value={selectedFormat}
                      onChange={(e) => setSelectedFormat(e.target.value)}
                      className="w-full bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500"
                    >
                      {analyzeResult.formats.map((f) => (
                        <option key={f.format_id} value={f.format_id}>
                          {f.resolution} - {f.ext.toUpperCase()} {f.vcodec !== 'none' ? '(Video)' : '(Âm thanh)'}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">Định dạng âm thanh</label>
                    <div className="flex gap-3">
                      {['mp3', 'm4a', 'wav', 'flac'].map(fmt => (
                        <button
                          key={fmt}
                          onClick={() => setAudioFormat(fmt)}
                          className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors border ${audioFormat === fmt ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300' : 'bg-slate-950 border-white/10 text-slate-400 hover:border-white/20'}`}
                        >
                          {fmt.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Tên tệp tùy chỉnh (Tùy chọn)</label>
                  <input
                    type="text"
                    value={customFilename}
                    onChange={(e) => setCustomFilename(e.target.value)}
                    placeholder="Để trống để sử dụng tên gốc"
                    className="w-full bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {!downloading && !completedFile ? (
                <button
                  onClick={startDownload}
                  className="w-full py-3 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white rounded-xl font-medium transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Bắt đầu tải xuống
                </button>
              ) : completedFile ? (
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-6 text-center space-y-4">
                  <div className="text-emerald-400 font-medium text-lg">Tải tệp hoàn tất! Trình duyệt đang tự động lưu tệp về máy tính của bạn.</div>
                  <div className="flex flex-col gap-3">
                    <div className="flex flex-col gap-1 w-full">
                      <button
                        onClick={() => handleSaveWithPicker()}
                        className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Nhấn để chọn vị trí lưu tệp về máy tính
                      </button>
                      <p className="text-xs text-emerald-500/70 text-center mt-1">
                        Lưu ý: Bạn cũng có thể bật tính năng 'Hỏi vị trí lưu từng tệp trước khi tải xuống' trong cài đặt trình duyệt để luôn tự động hiện hộp thoại chọn thư mục.
                      </p>
                    </div>
                    <button
                      onClick={() => handleTransfer(completedFile.jobId)}
                      className="w-full py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                      </svg>
                      Chuyển sang Xóa phụ đề video
                    </button>
                    <button
                      onClick={() => {
                        setCompletedFile(null);
                        setAnalyzeResult(null);
                        setUrl("");
                      }}
                      className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      Tải video khác
                    </button>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-950 rounded-xl p-4 border border-white/5 space-y-3">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium text-indigo-400">{statusText}</span>
                    <span className="text-slate-300 font-mono">{progress}%</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${progress}%` }} />
                  </div>
                  <div className="flex justify-between items-center text-xs text-slate-400">
                    <span>{speed || 'Đang tính toán tốc độ...'}</span>
                    <span>{eta ? `Còn lại: ${eta}` : ''}</span>
                  </div>
                  <button
                    onClick={handleCancel}
                    className="w-full py-2 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 rounded-lg text-sm font-medium transition-colors mt-2"
                  >
                    Hủy tải
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Lịch sử tải xuống */}
        <div className="bg-slate-900 border border-white/5 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-medium text-white">Lịch sử tải xuống</h2>
            <button
              onClick={fetchHistory}
              className="text-xs font-medium text-slate-400 hover:text-white transition-colors flex items-center gap-1.5 bg-white/5 px-3 py-1.5 rounded-lg"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Làm mới lịch sử
            </button>
          </div>
          <DownloadHistory items={history} onRefresh={fetchHistory} onTransfer={handleTransfer} />
        </div>
        
      </div>
    </>
  );
}
