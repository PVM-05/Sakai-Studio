"use client";

import { useState, useRef, useEffect, ChangeEvent } from "react";
import Link from "next/link";
import VideoPlayer, { Box } from "../../components/VideoPlayer";

interface ProjectItem {
  id: string;
  filename: string;
  status: string;
  progress: number;
  created_at: string | null;
  completed_at: string | null;
  result_url: string | null;
}

export default function Dashboard() {
  const [dragActive, setDragActive] = useState(false);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [serverFilename, setServerFilename] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<string | null>(null);
  const [taskProgress, setTaskProgress] = useState<number>(0);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [detecting, setDetecting] = useState<boolean>(false);
  const [detectionNotice, setDetectionNotice] = useState<string | null>(null);
  const [recentProjects, setRecentProjects] = useState<ProjectItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const fetchProjects = async () => {
    try {
      const res = await fetch("http://localhost:8000/projects?limit=8");
      if (res.ok) {
        const data = await res.json();
        setRecentProjects(data);
      }
    } catch (err) {
      console.error("Lỗi khi tải danh sách dự án:", err);
    }
  };

  useEffect(() => {
    fetchProjects();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setTaskId(null);
    setResultUrl(null);
    setBoxes([]);
    setDetectionNotice(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Tải lên thất bại");
      }
      const data = await res.json();
      setServerFilename(data.filename);
      setVideoSrc(URL.createObjectURL(file));
    } catch (err) {
      console.error("Lỗi khi tải video:", err);
      alert("Không thể tải video lên máy chủ. Vui lòng kiểm tra lại kết nối.");
    } finally {
      setUploading(false);
    }
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleAutoDetect = async () => {
    if (!serverFilename) return;

    setDetecting(true);
    setDetectionNotice(null);

    const formData = new FormData();
    formData.append("filename", serverFilename);
    formData.append("timestamp", currentTime.toString());

    try {
      const res = await fetch("http://localhost:8000/detect-subtitles", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Nhận diện chữ thất bại");
      }

      const data = await res.json();
      if (data.boxes && data.boxes.length > 0) {
        const newBoxes: Box[] = data.boxes.map((b: any, index: number) => ({
          id: `auto-${Date.now()}-${index}`,
          x: b.x,
          y: b.y,
          w: b.w,
          h: b.h,
          x_pct: b.x_pct,
          y_pct: b.y_pct,
          w_pct: b.w_pct,
          h_pct: b.h_pct,
        }));
        setBoxes(newBoxes);
        setDetectionNotice(`Đã tự động phát hiện ${newBoxes.length} vùng chữ trên khung hình hiện tại.`);
      } else {
        setDetectionNotice("Không phát hiện thấy văn bản hoặc phụ đề trên khung hình này.");
      }
    } catch (err) {
      console.error("Lỗi khi nhận diện tự động:", err);
      alert("Không thể thực hiện nhận diện tự động tại thời điểm này.");
    } finally {
      setDetecting(false);
    }
  };

  const handleProcess = async () => {
    if (!serverFilename) return;

    if (boxes.length === 0) {
      alert("Vui lòng chọn hoặc tự động phát hiện ít nhất một vùng cần xóa trước khi bắt đầu.");
      return;
    }

    const formData = new FormData();
    formData.append("filename", serverFilename);
    formData.append("boxes", JSON.stringify(boxes));
    formData.append("user_id", "demo_user");

    try {
      const res = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Khởi tạo tác vụ thất bại");
      }

      const data = await res.json();
      setTaskId(data.task_id);
      setTaskStatus("Đang khởi tạo tiến trình xóa chữ...");
      setTaskProgress(0);

      // Kích hoạt lắng nghe luồng sự kiện thời gian thực qua Server-Sent Events
      subscribeToEvents(data.task_id);
    } catch (err) {
      console.error("Lỗi khi bắt đầu xử lý:", err);
      alert("Không thể khởi động tiến trình xử lý.");
    }
  };

  const subscribeToEvents = (id: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const sseUrl = `http://localhost:8000/events/${id}`;
    const eventSource = new EventSource(sseUrl);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.task_status === "SUCCESS") {
          eventSource.close();
          setTaskStatus("Hoàn thành xử lý!");
          setTaskProgress(100);
          const downloadPath = payload.meta?.result_url || `/downloads/${id}/result.mp4`;
          setResultUrl(`http://localhost:8000${downloadPath}`);
          fetchProjects();
        } else if (payload.task_status === "PROGRESS") {
          setTaskStatus(payload.meta?.status || "Đang xử lý...");
          setTaskProgress(payload.meta?.current || 0);
        } else if (payload.task_status === "FAILURE") {
          eventSource.close();
          setTaskStatus("Tiến trình bị gián đoạn hoặc gặp lỗi.");
          alert(payload.error || "Xảy ra lỗi trong quá trình xử lý.");
          fetchProjects();
        }
      } catch (e) {
        console.error("Lỗi phân tích dữ liệu sự kiện:", e);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      // Chuyển sang cơ chế thăm dò định kỳ dự phòng nếu ngắt kết nối luồng sự kiện
      fallbackPolling(id);
    };
  };

  const fallbackPolling = (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/status/${id}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.task_status === "SUCCESS") {
          clearInterval(interval);
          setTaskStatus("Hoàn thành xử lý!");
          setTaskProgress(100);
          const downloadPath = data.meta?.result_url || `/downloads/${id}/result.mp4`;
          setResultUrl(`http://localhost:8000${downloadPath}`);
          fetchProjects();
        } else if (data.task_status === "PROGRESS") {
          setTaskStatus(data.meta?.status || "Đang xử lý...");
          setTaskProgress(data.meta?.current || 0);
        } else if (data.task_status === "FAILURE") {
          clearInterval(interval);
          setTaskStatus("Tiến trình gặp lỗi.");
          fetchProjects();
        }
      } catch (err) {
        console.error("Lỗi thăm dò dự phòng:", err);
      }
    }, 1500);
  };

  const handleReset = () => {
    setVideoSrc(null);
    setServerFilename(null);
    setTaskId(null);
    setTaskStatus(null);
    setTaskProgress(0);
    setResultUrl(null);
    setBoxes([]);
    setDetectionNotice(null);
  };

  return (
    <>
      {/* Khu vực nội dung chính */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-slate-900/50 backdrop-blur-md">
          <h1 className="text-xl font-medium tracking-tight">Khu vực xử lý video</h1>
          {videoSrc && (
            <button
              onClick={handleReset}
              className="bg-white/10 hover:bg-white/15 text-white px-4 py-2 rounded-lg font-medium text-sm transition-colors border border-white/10"
            >
              Chọn video khác
            </button>
          )}
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          {videoSrc ? (
            <div className="mb-12">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-lg font-medium text-white">Khung làm việc xóa phụ đề</h2>
                  <p className="text-xs text-slate-400">Tệp đang chọn: {serverFilename}</p>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={handleAutoDetect}
                    disabled={detecting || Boolean(taskId && taskProgress < 100)}
                    className="bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 font-medium px-4 py-2.5 rounded-xl transition-all disabled:opacity-50 flex items-center gap-2 text-sm"
                  >
                    {detecting ? (
                      <>
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                        </svg>
                        Đang quét chữ...
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        Tự động phát hiện phụ đề
                      </>
                    )}
                  </button>
                </div>
              </div>

              {detectionNotice && (
                <div className="mb-4 px-4 py-2.5 rounded-xl bg-slate-900 border border-indigo-500/30 text-indigo-300 text-xs flex items-center justify-between">
                  <span>{detectionNotice}</span>
                  <button onClick={() => setDetectionNotice(null)} className="text-slate-400 hover:text-white ml-4">×</button>
                </div>
              )}

              <VideoPlayer
                videoSrc={videoSrc}
                boxes={boxes}
                onBoxesChange={setBoxes}
                onTimeUpdate={setCurrentTime}
              />

              {/* Bảng điều khiển tiến trình và hành động */}
              <div className="mt-6">
                {!taskId ? (
                  <div className="flex justify-end">
                    <button
                      onClick={handleProcess}
                      disabled={boxes.length === 0}
                      className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium px-8 py-3 rounded-xl shadow-lg shadow-indigo-500/25 transition-all hover:scale-105 active:scale-95 flex items-center gap-2 disabled:opacity-50 disabled:hover:scale-100"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Bắt đầu xóa chữ
                    </button>
                  </div>
                ) : (
                  <div className="bg-slate-900 border border-white/10 rounded-xl p-6">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium text-indigo-400">{taskStatus}</span>
                      <span className="text-slate-400 text-sm">{taskProgress}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden mb-4">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 transition-all duration-300 rounded-full"
                        style={{ width: `${taskProgress}%` }}
                      />
                    </div>

                    {resultUrl && (
                      <div className="flex justify-center mt-6">
                        <a
                          href={resultUrl}
                          download
                          className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-8 py-3 rounded-xl shadow-lg shadow-emerald-500/20 transition-all hover:scale-105"
                        >
                          Tải về video đã xóa chữ
                        </a>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div
              className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
                dragActive ? "border-indigo-500 bg-indigo-500/5" : "border-white/10 bg-slate-900/50 hover:border-white/20"
              }`}
              onDragEnter={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setDragActive(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                  handleFileUpload(e.dataTransfer.files[0]);
                }
              }}
              onDragOver={(e) => e.preventDefault()}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="video/mp4,video/mov,video/webm,video/mkv"
                onChange={onFileChange}
              />
              <div className="w-16 h-16 rounded-full bg-slate-800 mx-auto flex items-center justify-center mb-6">
                {uploading ? (
                  <svg className="w-8 h-8 text-indigo-400 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                ) : (
                  <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                )}
              </div>
              <h3 className="text-xl font-medium mb-2">{uploading ? "Đang tải tệp lên máy chủ..." : "Tải video lên để bắt đầu"}</h3>
              <p className="text-slate-400 mb-6 max-w-md mx-auto text-sm">
                Kéo thả tệp video định dạng MP4, MOV, WEBM hoặc MKV vào đây, hoặc nhấn để duyệt tệp từ máy tính. Kích thước tối đa 500MB.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="bg-white text-slate-900 px-6 py-3 rounded-xl font-medium hover:bg-slate-200 transition-colors disabled:opacity-50"
              >
                Chọn tệp video
              </button>
            </div>
          )}

          {/* Danh sách dự án gần đây */}
          <div className="mt-12">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-white">Dự án gần đây</h2>
              <button onClick={fetchProjects} className="text-xs text-slate-400 hover:text-white transition-colors">
                Làm mới danh sách
              </button>
            </div>

            {recentProjects.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {recentProjects.map((project) => (
                  <div
                    key={project.id}
                    className="group bg-slate-900 border border-white/5 rounded-2xl overflow-hidden hover:border-indigo-500/40 transition-all"
                  >
                    <div className="aspect-video bg-slate-800 relative overflow-hidden flex items-center justify-center">
                      <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 to-transparent z-10" />
                      <svg className="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <div className="absolute bottom-3 left-3 z-20 flex gap-2">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                            project.status === "Hoàn thành"
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                              : project.status === "Thất bại"
                              ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                              : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                          }`}
                        >
                          {project.status}
                        </span>
                      </div>
                    </div>
                    <div className="p-4">
                      <h4 className="font-medium text-sm mb-1 truncate text-slate-200 group-hover:text-indigo-400 transition-colors">
                        {project.filename}
                      </h4>
                      <p className="text-xs text-slate-500">
                        {project.created_at ? `Khởi tạo lúc: ${project.created_at}` : "Vừa xong"}
                      </p>
                      {project.result_url && (
                        <div className="mt-3">
                          <a
                            href={`http://localhost:8000${project.result_url}`}
                            download
                            className="text-xs text-emerald-400 hover:underline inline-flex items-center gap-1"
                          >
                            Tải về kết quả
                          </a>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 bg-slate-900/30 border border-white/5 rounded-2xl text-slate-500 text-sm">
                Chưa có dự án nào được xử lý gần đây.
              </div>
            )}
          </div>
        </div>
    </>
  );
}
