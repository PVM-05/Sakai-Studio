"use client";

import React, { useState, useEffect, useRef } from "react";
import VideoPlayer from "../../../components/VideoPlayer";
import Link from "next/link";

export default function ExtractPage() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [extractMode, setExtractMode] = useState<string>("auto");
  const [language, setLanguage] = useState<string>("vi");
  const [useVoiceAssist, setUseVoiceAssist] = useState<boolean>(false);
  const [useVocalIsolation, setUseVocalIsolation] = useState<boolean>(false);
  
  const [isExtracting, setIsExtracting] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [statusText, setStatusText] = useState<string>("");
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  
  const [subtitles, setSubtitles] = useState<{ start: string; end: string; text: string }[]>([]);
  const [searchText, setSearchText] = useState<string>("");
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch("/extract/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      } else {
        setHistory([
          { id: "1", name: "video_demo.mp4", date: "14/09/2026", count: 120 },
          { id: "2", name: "bai_giang.mp4", date: "12/09/2026", count: 345 }
        ]);
      }
    } catch (e) {
      setHistory([
        { id: "1", name: "video_demo.mp4", date: "14/09/2026", count: 120 },
        { id: "2", name: "bai_giang.mp4", date: "12/09/2026", count: 345 }
      ]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setIsCompleted(false);
      setSubtitles([]);
    }
  };

  const startExtraction = () => {
    setIsExtracting(true);
    setProgress(0);
    setStatusText("Đang khởi tạo máy chủ xử lý...");
    
    const jobId = Math.random().toString(36).substring(7);
    const eventSource = new EventSource(`/extract/events/${jobId}`);
    
    let simProgress = 0;
    const interval = setInterval(() => {
      simProgress += Math.floor(Math.random() * 10) + 1;
      if (simProgress >= 100) {
        simProgress = 100;
        clearInterval(interval);
        setIsExtracting(false);
        setIsCompleted(true);
        setSubtitles([
          { start: "00:00:01,000", end: "00:00:04,000", text: "Xin chào mọi người" },
          { start: "00:00:04,500", end: "00:00:08,000", text: "Hôm nay chúng ta sẽ tìm hiểu về trí tuệ nhân tạo" },
          { start: "00:00:08,500", end: "00:00:12,000", text: "Đây là một công nghệ mang tính đột phá" }
        ]);
      }
      setProgress(simProgress);
      setStatusText(`Đang xử lý hình ảnh và âm thanh... ${simProgress}%`);
    }, 500);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.progress);
      setStatusText(data.message);
      if (data.status === "completed") {
        eventSource.close();
        setIsExtracting(false);
        setIsCompleted(true);
        setSubtitles(data.subtitles);
      }
    };
    
    eventSource.onerror = () => {
      eventSource.close();
    };
  };

  const filteredSubtitles = subtitles.filter(sub => 
    sub.text.toLowerCase().includes(searchText.toLowerCase())
  );

  return (
    <div className="p-6 bg-gray-50 min-h-screen font-sans text-gray-800">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Trích xuất phụ đề video
        </h1>
        {videoFile && (
          <button 
            onClick={() => { setVideoFile(null); setVideoUrl(""); }}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 font-medium text-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Chọn video khác
          </button>
        )}
      </div>

      {!videoFile ? (
        <div className="bg-white p-10 rounded-lg shadow-sm border border-gray-200 text-center flex flex-col items-center justify-center min-h-[400px]">
          <svg className="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="text-gray-600 mb-6 text-lg">Tải lên video hoặc chọn từ các dự án có sẵn</p>
          <label className="cursor-pointer bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 font-medium shadow-sm transition-colors">
            Duyệt tìm tệp
            <input type="file" accept="video/*" className="hidden" onChange={handleFileChange} />
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              <div className="p-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                <h2 className="font-semibold text-gray-700 flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Trình phát video (Vẽ vùng để giới hạn quét chữ)
                </h2>
              </div>
              <div className="w-full bg-black aspect-video relative">
                <VideoPlayer videoSrc={videoUrl} />
              </div>
            </div>

            {isExtracting && (
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="font-semibold text-gray-800 mb-4">Tiến trình trích xuất</h3>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-2 overflow-hidden">
                  <div className="bg-blue-600 h-3 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
                </div>
                <div className="flex justify-between text-sm text-gray-600">
                  <span>{statusText}</span>
                  <span className="font-medium">{progress}%</span>
                </div>
              </div>
            )}

            {isCompleted && (
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                      <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Hoàn tất trích xuất
                    </h3>
                    <p className="text-gray-600 text-sm mt-1">Đã phát hiện <span className="font-bold text-gray-900">{subtitles.length}</span> câu phụ đề</p>
                  </div>
                  
                  <div className="flex flex-wrap gap-2">
                    <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded border border-gray-300 flex items-center gap-1.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                      Tải tệp SRT
                    </button>
                    <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded border border-gray-300 flex items-center gap-1.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                      Tải tệp ASS
                    </button>
                    <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded border border-gray-300 flex items-center gap-1.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" /></svg>
                      Sao chép văn bản
                    </button>
                  </div>
                </div>

                <div className="mb-4 relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <input
                    type="text"
                    placeholder="Tìm kiếm nội dung phụ đề..."
                    className="pl-10 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                  />
                </div>

                <div className="border border-gray-200 rounded-md overflow-hidden max-h-80 overflow-y-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-1/3">Mốc thời gian</th>
                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Nội dung văn bản</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {filteredSubtitles.map((sub, idx) => (
                        <tr key={idx} className="hover:bg-gray-50">
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {sub.start} - {sub.end}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {sub.text}
                          </td>
                        </tr>
                      ))}
                      {filteredSubtitles.length === 0 && (
                        <tr>
                          <td colSpan={2} className="px-4 py-8 text-center text-gray-500 text-sm">
                            Không tìm thấy phụ đề phù hợp
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-6 flex justify-end">
                  <Link href="/dashboard/translate">
                    <button className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium shadow-sm transition-colors">
                      Chuyển sang Dịch phụ đề
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                      </svg>
                    </button>
                  </Link>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 border-b pb-2 border-gray-100 flex items-center gap-2">
                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Tùy chọn cấu hình
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Chế độ nhận diện</label>
                  <select 
                    value={extractMode} 
                    onChange={(e) => setExtractMode(e.target.value)}
                    className="w-full rounded-md border border-gray-300 py-2 px-3 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 bg-white"
                  >
                    <option value="auto">Tự động tối ưu</option>
                    <option value="fast">Ưu tiên tốc độ</option>
                    <option value="accurate">Ưu tiên độ chính xác</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Ngôn ngữ chữ cần quét</label>
                  <select 
                    value={language} 
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full rounded-md border border-gray-300 py-2 px-3 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 bg-white"
                  >
                    <option value="vi">Tiếng Việt</option>
                    <option value="en">Tiếng Anh</option>
                    <option value="ch">Tiếng Trung</option>
                    <option value="ja">Tiếng Nhật</option>
                    <option value="ko">Tiếng Hàn</option>
                    <option value="fr">Tiếng Pháp</option>
                    <option value="de">Tiếng Đức</option>
                    <option value="ru">Tiếng Nga</option>
                    <option value="es">Tiếng Tây Ban Nha</option>
                  </select>
                </div>

                <div className="pt-2">
                  <label className="flex items-start cursor-pointer gap-3 p-3 rounded border border-gray-200 hover:bg-gray-50 transition-colors">
                    <div className="relative flex items-center h-5 mt-0.5">
                      <input 
                        type="checkbox" 
                        className="peer sr-only"
                        checked={useVoiceAssist}
                        onChange={(e) => setUseVoiceAssist(e.target.checked)}
                      />
                      <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                    </div>
                    <div className="text-sm">
                      <span className="font-medium text-gray-800 block mb-1">Bổ trợ nhận diện giọng nói khi phụ đề bị mờ hoặc nhiễu</span>
                    </div>
                  </label>
                </div>

                <div>
                  <label className="flex items-start cursor-pointer gap-3 p-3 rounded border border-gray-200 hover:bg-gray-50 transition-colors">
                    <div className="relative flex items-center h-5 mt-0.5">
                      <input 
                        type="checkbox" 
                        className="peer sr-only"
                        checked={useVocalIsolation}
                        onChange={(e) => setUseVocalIsolation(e.target.checked)}
                      />
                      <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                    </div>
                    <div className="text-sm">
                      <span className="font-medium text-gray-800 block mb-1">Tách giọng nói khỏi nhạc nền bằng trí tuệ nhân tạo</span>
                    </div>
                  </label>
                </div>

                <button 
                  onClick={startExtraction}
                  disabled={isExtracting}
                  className={`w-full mt-4 py-3 px-4 flex justify-center items-center gap-2 rounded-md font-medium text-white shadow-sm transition-colors ${
                    isExtracting ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  {isExtracting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Đang xử lý...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Bắt đầu trích xuất phụ đề
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 border-b pb-2 border-gray-100 flex items-center gap-2">
                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Lịch sử trích xuất
              </h3>
              
              {history.length > 0 ? (
                <div className="space-y-3">
                  {history.map((item) => (
                    <div key={item.id} className="p-3 border border-gray-100 rounded bg-gray-50 flex flex-col gap-2">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium text-sm text-gray-900 truncate max-w-[150px]">{item.name}</p>
                          <p className="text-xs text-gray-500 mt-0.5">{item.date} • {item.count} câu</p>
                        </div>
                        <button className="text-red-500 hover:text-red-700 hover:bg-red-50 p-1 rounded transition-colors" title="Xóa">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                      <div className="flex gap-2 mt-1">
                        <button className="text-xs px-2 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50">SRT</button>
                        <button className="text-xs px-2 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50">ASS</button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">Chưa có lịch sử</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
