"use client";

import { useRef, useState, MouseEvent } from "react";

interface Box {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface VideoPlayerProps {
  videoSrc: string;
  onBoxesChange?: (boxes: Box[]) => void;
}

export default function VideoPlayer({ videoSrc, onBoxesChange }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState<Box | null>(null);

  const getRelativeCoords = (e: MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    // Only draw if video is paused
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause();
    }
    
    setIsDrawing(true);
    const coords = getRelativeCoords(e);
    setStartPos(coords);
    setCurrentBox({
      id: Date.now().toString(),
      x: coords.x,
      y: coords.y,
      w: 0,
      h: 0,
    });
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !currentBox) return;
    
    const currentPos = getRelativeCoords(e);
    const newX = Math.min(startPos.x, currentPos.x);
    const newY = Math.min(startPos.y, currentPos.y);
    const newW = Math.abs(currentPos.x - startPos.x);
    const newH = Math.abs(currentPos.y - startPos.y);

    setCurrentBox({
      ...currentBox,
      x: newX,
      y: newY,
      w: newW,
      h: newH,
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox) return;
    setIsDrawing(false);
    
    // Only add if box is big enough (prevent accidental clicks)
    if (currentBox.w > 10 && currentBox.h > 10) {
      const newBoxes = [...boxes, currentBox];
      setBoxes(newBoxes);
      onBoxesChange?.(newBoxes);
    }
    setCurrentBox(null);
  };

  const removeBox = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const newBoxes = boxes.filter(b => b.id !== id);
    setBoxes(newBoxes);
    onBoxesChange?.(newBoxes);
  };

  return (
    <div className="relative w-full rounded-2xl overflow-hidden bg-slate-900 border border-white/10 group shadow-2xl">
      <div 
        ref={containerRef}
        className="relative w-full aspect-video cursor-crosshair"
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
        />

        {/* Drawn Boxes */}
        {boxes.map((box) => (
          <div
            key={box.id}
            className="absolute border-2 border-indigo-500 bg-indigo-500/20 group/box"
            style={{
              left: box.x,
              top: box.y,
              width: box.w,
              height: box.h,
            }}
          >
            <button 
              onClick={(e) => removeBox(box.id, e)}
              className="absolute -top-3 -right-3 w-6 h-6 bg-red-500 rounded-full text-white opacity-0 group-hover/box:opacity-100 transition-opacity flex items-center justify-center text-xs font-bold shadow-lg"
            >
              ×
            </button>
            <div className="absolute -bottom-6 left-0 bg-indigo-500 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap opacity-0 group-hover/box:opacity-100">
              Object to remove
            </div>
          </div>
        ))}

        {/* Current drawing box */}
        {isDrawing && currentBox && (
          <div
            className="absolute border-2 border-dashed border-indigo-400 bg-indigo-400/20"
            style={{
              left: currentBox.x,
              top: currentBox.y,
              width: currentBox.w,
              height: currentBox.h,
            }}
          />
        )}
      </div>
      
      {/* Helper Bar */}
      <div className="bg-slate-950/80 p-4 border-t border-white/10 flex justify-between items-center text-sm">
        <span className="text-slate-400 flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"></path></svg>
          Click & drag over the video to draw removal zones
        </span>
        <div className="flex gap-4 font-mono text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1.5 rounded-full border border-indigo-500/20">
          <span>{boxes.length} zone(s) selected</span>
        </div>
      </div>
    </div>
  );
}
