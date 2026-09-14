"use client";

import { useRef, useState, useEffect, MouseEvent } from "react";

export interface Box {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  x_pct: number;
  y_pct: number;
  w_pct: number;
  h_pct: number;
}

interface VideoPlayerProps {
  videoSrc: string;
  boxes?: Box[];
  onBoxesChange?: (boxes: Box[]) => void;
  onTimeUpdate?: (currentTime: number) => void;
}

export default function VideoPlayer({
  videoSrc,
  boxes: externalBoxes,
  onBoxesChange,
  onTimeUpdate,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [internalBoxes, setInternalBoxes] = useState<Box[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  // Đồng bộ danh sách hộp từ ngoài truyền vào nếu có
  const boxes = externalBoxes !== undefined ? externalBoxes : internalBoxes;

  const updateBoxes = (newBoxes: Box[]) => {
    if (externalBoxes === undefined) {
      setInternalBoxes(newBoxes);
    }
    onBoxesChange?.(newBoxes);
  };

  const getRelativeCoords = (e: MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(rect.width, e.clientX - rect.left)),
      y: Math.max(0, Math.min(rect.height, e.clientY - rect.top)),
      containerWidth: rect.width,
      containerHeight: rect.height,
    };
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause();
    }

    const coords = getRelativeCoords(e);
    setIsDrawing(true);
    setStartPos({ x: coords.x, y: coords.y });
    setCurrentBox({
      x: coords.x,
      y: coords.y,
      w: 0,
      h: 0,
    });
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !currentBox) return;

    const coords = getRelativeCoords(e);
    const newX = Math.min(startPos.x, coords.x);
    const newY = Math.min(startPos.y, coords.y);
    const newW = Math.abs(coords.x - startPos.x);
    const newH = Math.abs(coords.y - startPos.y);

    setCurrentBox({
      x: newX,
      y: newY,
      w: newW,
      h: newH,
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox || !containerRef.current) return;
    setIsDrawing(false);

    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0 && currentBox.w > 10 && currentBox.h > 10) {
      const x_pct = Math.max(0, Math.min(1, currentBox.x / rect.width));
      const y_pct = Math.max(0, Math.min(1, currentBox.y / rect.height));
      const w_pct = Math.max(0, Math.min(1 - x_pct, currentBox.w / rect.width));
      const h_pct = Math.max(0, Math.min(1 - y_pct, currentBox.h / rect.height));

      const newBox: Box = {
        id: Date.now().toString(),
        x: currentBox.x,
        y: currentBox.y,
        w: currentBox.w,
        h: currentBox.h,
        x_pct: Number(x_pct.toFixed(4)),
        y_pct: Number(y_pct.toFixed(4)),
        w_pct: Number(w_pct.toFixed(4)),
        h_pct: Number(h_pct.toFixed(4)),
      };

      const nextBoxes = [...boxes, newBox];
      updateBoxes(nextBoxes);
    }
    setCurrentBox(null);
  };

  const removeBox = (id: string, e: MouseEvent) => {
    e.stopPropagation();
    const nextBoxes = boxes.filter((b) => b.id !== id);
    updateBoxes(nextBoxes);
  };

  const handleClearAll = (e: MouseEvent) => {
    e.stopPropagation();
    updateBoxes([]);
  };

  return (
    <div className="relative w-full rounded-2xl overflow-hidden bg-slate-900 border border-white/10 group shadow-2xl">
      <div
        ref={containerRef}
        className="relative w-full aspect-video cursor-crosshair select-none overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <video
          ref={videoRef}
          src={videoSrc}
          className="w-full h-full object-contain pointer-events-none"
          controls
          controlsList="nodownload"
          onTimeUpdate={(e) => {
            onTimeUpdate?.(e.currentTarget.currentTime);
          }}
        />

        {/* Các hộp chữ nhật đã vẽ, sử dụng tỷ lệ phần trăm để chuẩn hóa kích thước */}
        {boxes.map((box) => (
          <div
            key={box.id}
            className="absolute border-2 border-indigo-500 bg-indigo-500/20 group/box"
            style={{
              left: `${box.x_pct * 100}%`,
              top: `${box.y_pct * 100}%`,
              width: `${box.w_pct * 100}%`,
              height: `${box.h_pct * 100}%`,
            }}
          >
            <button
              onClick={(e) => removeBox(box.id, e)}
              title="Xóa vùng chọn này"
              className="absolute -top-3 -right-3 w-6 h-6 bg-red-500 hover:bg-red-600 rounded-full text-white opacity-0 group-hover/box:opacity-100 transition-opacity flex items-center justify-center text-xs font-bold shadow-lg"
            >
              ×
            </button>
            <div className="absolute -bottom-6 left-0 bg-indigo-600 text-white text-[10px] px-2 py-0.5 rounded whitespace-nowrap opacity-0 group-hover/box:opacity-100 pointer-events-none">
              Vùng cần xóa
            </div>
          </div>
        ))}

        {/* Khung đang vẽ */}
        {isDrawing && currentBox && (
          <div
            className="absolute border-2 border-dashed border-indigo-400 bg-indigo-400/25 pointer-events-none"
            style={{
              left: currentBox.x,
              top: currentBox.y,
              width: currentBox.w,
              height: currentBox.h,
            }}
          />
        )}
      </div>

      {/* Thanh công cụ trợ giúp */}
      <div className="bg-slate-950/80 p-4 border-t border-white/10 flex flex-wrap justify-between items-center gap-4 text-sm">
        <span className="text-slate-400 flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
          </svg>
          Kéo giữ chuột trên video để đánh dấu vùng cần xóa
        </span>

        <div className="flex items-center gap-3">
          {boxes.length > 0 && (
            <button
              onClick={handleClearAll}
              className="text-xs text-rose-400 hover:text-rose-300 transition-colors px-2 py-1 rounded bg-rose-500/10 border border-rose-500/20"
            >
              Xóa tất cả vùng chọn
            </button>
          )}
          <div className="font-mono text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1.5 rounded-full border border-indigo-500/20">
            <span>Đã chọn {boxes.length} vùng</span>
          </div>
        </div>
      </div>
    </div>
  );
}
