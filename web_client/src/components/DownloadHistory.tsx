"use client";

import { apiDelete, formatFileSize, API_BASE } from "../lib/api";

export interface DownloadItem {
  id: string;
  title: string;
  channel: string;
  duration: number;
  thumbnail_url: string;
  resolution: string;
  container: string;
  filesize: number;
  status: string;
  progress: number;
  audio_only: boolean;
  error_message: string | null;
  result_url: string | null;
  created_at: string | null;
}

interface DownloadHistoryProps {
  items: DownloadItem[];
  onRefresh: () => void;
  onTransfer: (jobId: string) => void;
}

export default function DownloadHistory({ items, onRefresh, onTransfer }: DownloadHistoryProps) {
  const handleDelete = async (id: string) => {
    try {
      await apiDelete(`/ytdlp/history/${id}`);
      onRefresh();
    } catch (err) {
      console.error("Lỗi khi xóa mục lịch sử:", err);
    }
  };

  if (items.length === 0) {
    return (
      <div className="text-center py-10 bg-slate-900/30 border border-white/5 rounded-2xl text-slate-500 text-sm">
        Chưa có tệp video hoặc âm thanh nào trong lịch sử tải.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {items.map((item) => (
        <div
          key={item.id}
          className="bg-slate-900 border border-white/5 rounded-xl p-4 flex gap-4 hover:border-indigo-500/30 transition-colors items-center"
        >
          {/* Ảnh thu nhỏ */}
          <div className="w-28 h-18 rounded-lg bg-slate-800 overflow-hidden flex-shrink-0 relative">
            {item.thumbnail_url ? (
              <img src={item.thumbnail_url} alt="" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
            )}
          </div>

          {/* Thông tin video */}
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-sm text-slate-200 truncate">{item.title || "Tệp chưa đặt tên"}</h4>
            <p className="text-xs text-slate-500 mt-1 flex items-center gap-2">
              {item.channel && <span>{item.channel}</span>}
              <span>·</span>
              <span className="uppercase">{item.audio_only ? "Âm thanh" : (item.resolution || item.container)}</span>
              {item.filesize ? (
                <>
                  <span>·</span>
                  <span>{formatFileSize(item.filesize)}</span>
                </>
              ) : null}
            </p>

            <div className="mt-2 flex items-center gap-2">
              <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                item.status === "Hoàn thành"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                  : item.status === "Thất bại" || item.status === "Đã hủy"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                  : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
              }`}>
                {item.status}
              </span>
              {item.error_message && (
                <span className="text-[10px] text-rose-400 truncate max-w-sm">{item.error_message}</span>
              )}
            </div>
          </div>

          {/* Nút hành động */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {item.result_url && item.status === "Hoàn thành" && (
              <>
                <a
                  href={`${API_BASE}${item.result_url}`}
                  download
                  className="px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 text-xs font-medium transition-colors flex items-center gap-1.5"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Tải về
                </a>
                {!item.audio_only && (
                  <button
                    onClick={() => onTransfer(item.id)}
                    className="px-3 py-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20 text-xs font-medium transition-colors flex items-center gap-1.5"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                    </svg>
                    Xóa phụ đề
                  </button>
                )}
              </>
            )}
            <button
              onClick={() => handleDelete(item.id)}
              className="p-2 rounded-lg bg-white/5 text-slate-400 hover:bg-rose-500/10 hover:text-rose-400 transition-colors"
              title="Xóa mục này"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
