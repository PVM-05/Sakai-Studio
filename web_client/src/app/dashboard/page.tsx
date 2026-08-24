"use client";

import { useState, useRef, ChangeEvent } from "react";
import Link from "next/link";
import VideoPlayer from "../components/VideoPlayer";

export default function Dashboard() {
  const [dragActive, setDragActive] = useState(false);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [serverFilename, setServerFilename] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<string | null>(null);
  const [taskProgress, setTaskProgress] = useState<number>(0);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setServerFilename(data.filename);
      // Use local blob URL for quick frontend preview
      setVideoSrc(URL.createObjectURL(file));
    } catch (err) {
      console.error("Upload failed", err);
      alert("Failed to upload video");
    } finally {
      setUploading(false);
    }
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleProcess = async () => {
    if (!serverFilename) return;
    
    const formData = new FormData();
    formData.append("filename", serverFilename);
    formData.append("boxes", JSON.stringify(boxes));

    try {
      const res = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setTaskId(data.task_id);
      setTaskStatus("Processing started...");
      pollStatus(data.task_id);
    } catch (err) {
      console.error("Processing failed", err);
      alert("Failed to start processing");
    }
  };

  const pollStatus = (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/status/${id}`);
        const data = await res.json();
        
        if (data.task_status === 'SUCCESS') {
          clearInterval(interval);
          setTaskStatus("Hoàn thành!");
          setTaskProgress(100);
          setResultUrl(`http://localhost:8000${data.meta.result_url}`);
        } else if (data.task_status === 'PROGRESS') {
          setTaskStatus(data.meta.status);
          setTaskProgress(data.meta.current);
        } else if (data.task_status === 'FAILURE') {
          clearInterval(interval);
          setTaskStatus("Lỗi xảy ra!");
          alert(data.error);
        }
      } catch (err) {
        console.error(err);
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white flex">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 border-r border-white/5 p-6 flex flex-col">
        <div className="flex items-center gap-2 mb-10">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold">
            S
          </div>
          <span className="font-semibold tracking-tight">Sakai Studio</span>
        </div>

        <nav className="flex-1 space-y-2">
          <Link href="/dashboard" className="flex items-center gap-3 bg-indigo-500/10 text-indigo-400 px-4 py-3 rounded-lg font-medium border border-indigo-500/20">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
            Projects
          </Link>
          <Link href="/dashboard/settings" className="flex items-center gap-3 text-slate-400 hover:bg-white/5 px-4 py-3 rounded-lg font-medium transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            Settings
          </Link>
        </nav>
        
        <div className="mt-auto border-t border-white/5 pt-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
              U
            </div>
            <div>
              <p className="text-sm font-medium">User Account</p>
              <p className="text-xs text-slate-500">Pro Plan</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-slate-900/50 backdrop-blur-md">
          <h1 className="text-xl font-medium">Your Projects</h1>
          <button className="bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded-lg font-medium text-sm transition-colors shadow-lg shadow-indigo-500/20">
            New Project
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          {/* Upload Area & Player */}
          {videoSrc ? (
            <div className="mb-12">
              <h2 className="text-lg font-medium mb-4">Workspace: Process Video</h2>
              <VideoPlayer 
                videoSrc={videoSrc} 
                onBoxesChange={setBoxes} 
              />
              
              {/* Actions & Progress */}
              <div className="mt-6">
                {!taskId ? (
                  <div className="flex justify-end">
                    <button 
                      onClick={handleProcess}
                      className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium px-8 py-3 rounded-xl shadow-lg shadow-indigo-500/25 transition-all hover:scale-105 active:scale-95 flex items-center gap-2"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                      Run AI Removal
                    </button>
                  </div>
                ) : (
                  <div className="bg-slate-900 border border-white/10 rounded-xl p-6">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium text-indigo-400">{taskStatus}</span>
                      <span className="text-slate-400 text-sm">{taskProgress}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mb-4">
                      <div 
                        className="h-full bg-indigo-500 transition-all duration-500" 
                        style={{ width: `${taskProgress}%` }}
                      />
                    </div>
                    
                    {resultUrl && (
                      <div className="flex justify-center mt-6">
                        <a 
                          href={resultUrl}
                          download
                          className="bg-green-500 hover:bg-green-600 text-white font-medium px-8 py-3 rounded-xl shadow-lg transition-all hover:scale-105"
                        >
                          Download Processed Video
                        </a>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div 
              className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${dragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-white/10 bg-slate-900/50 hover:border-white/20'}`}
              onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
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
                accept="video/mp4,video/mov,video/webm"
                onChange={onFileChange}
              />
              <div className="w-16 h-16 rounded-full bg-slate-800 mx-auto flex items-center justify-center mb-6">
                {uploading ? (
                  <svg className="w-8 h-8 text-indigo-400 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
                ) : (
                  <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                )}
              </div>
              <h3 className="text-xl font-medium mb-2">{uploading ? "Uploading..." : "Upload a video to start"}</h3>
              <p className="text-slate-400 mb-6 max-w-md mx-auto">
                Drag and drop your MP4, MOV, or WEBM file here, or click to browse. Maximum file size 500MB.
              </p>
              <button 
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="bg-white text-slate-900 px-6 py-3 rounded-xl font-medium hover:bg-slate-200 transition-colors disabled:opacity-50"
              >
                Select File
              </button>
            </div>
          )}

          {/* Recent Projects Grid */}
          <div className="mt-12">
            <h2 className="text-lg font-medium mb-6">Recent Projects</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {/* Dummy Project Card */}
              <div className="group bg-slate-900 border border-white/5 rounded-2xl overflow-hidden hover:border-indigo-500/50 transition-colors cursor-pointer">
                <div className="aspect-video bg-slate-800 relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 to-transparent z-10" />
                  <div className="absolute bottom-3 left-3 z-20 flex gap-2">
                    <span className="px-2 py-1 rounded bg-slate-900/80 text-[10px] font-mono text-slate-300">COMPLETED</span>
                  </div>
                </div>
                <div className="p-4">
                  <h4 className="font-medium mb-1 group-hover:text-indigo-400 transition-colors">Watermark_Removal.mp4</h4>
                  <p className="text-xs text-slate-500">Processed 2 hours ago • 12.4 MB</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
