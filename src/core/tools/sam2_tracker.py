import os
import cv2
import numpy as np

class SAM2VideoTracker:
    def __init__(self, video_path):
        from src.core.config import config
        self.video_path = video_path
        self.weights_path = getattr(config, 'sam2WeightsPath', None)
        if hasattr(self.weights_path, 'value'):
            self.weights_path = self.weights_path.value
            
        try:
            from sam2.build_sam import build_sam2_video_predictor
            # Suy luận model_cfg từ tên file checkpoint (sửa lỗi mismatch shape khi load weights)
            weights_name = os.path.basename(self.weights_path).lower() if self.weights_path else ""
            if "tiny" in weights_name or "_t" in weights_name:
                model_cfg = "sam2_hiera_t.yaml"
            elif "small" in weights_name or "_s" in weights_name:
                model_cfg = "sam2_hiera_s.yaml"
            elif "base" in weights_name or "_b" in weights_name:
                model_cfg = "sam2_hiera_b+.yaml"
            elif "large" in weights_name or "_l" in weights_name:
                model_cfg = "sam2_hiera_l.yaml"
            else:
                model_cfg = "sam2_hiera_s.yaml"
                
            if not self.weights_path or not os.path.exists(self.weights_path):
                print(f"[SAM2Tracker] Warning: SAM 2 weights not found at {self.weights_path}. Please configure it in settings.")
                # We won't raise error immediately to allow GUI to show up, 
                # but track() will fail if it's called.
                self.predictor = None
            else:
                import torch
                # Bật TF32 để tăng tốc trên card RTX 30/40 series
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
                if device == "cpu":
                    print("[SAM2Tracker] Warning: No CUDA GPU detected, falling back to CPU (will be slow).")
                    
                self.predictor = build_sam2_video_predictor(model_cfg, self.weights_path, device=device)
        except ImportError:
            print("[SAM2Tracker] Error: SAM 2 library is not installed.")
            self.predictor = None
        except Exception as e:
            print(f"[SAM2Tracker] Failed to initialize SAM 2 Video Predictor: {e}")
            self.predictor = None

    def track_chunk(self, target_boxes, start_frame_no, end_frame_no, padding=10, prompt_frame_offset=0):
        """
        Track một đoạn ngắn (chunk) để chống tràn ổ cứng và cho phép chèn OCR định kỳ.
        """
        import shutil
        
        temp_frame_dir = os.path.join(os.path.dirname(self.video_path), ".temp_frames")
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
        os.makedirs(temp_frame_dir, exist_ok=True)
        
        from src.core.tools.common_tools import open_capture_no_hwaccel
        cap = open_capture_no_hwaccel(self.video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # OpenCV start from 0-indexed, so frame 1 is 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no - 1)
        
        frame_count = 0
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(end_frame_no - start_frame_no + 1):
                ret, frame = cap.read()
                if not ret: 
                    break
                # Đẩy việc ghi đĩa sang luồng nền
                filepath = os.path.join(temp_frame_dir, f"{i:05d}.jpg")
                futures.append(executor.submit(cv2.imwrite, filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]))
                frame_count += 1
                
            # Đảm bảo toàn bộ ảnh đã được xả xong ra ổ đĩa
            for future in futures:
                future.result()
                
        cap.release()
        
        if frame_count == 0:
            return {}

        if self.predictor is None:
            raise RuntimeError("SAM 2 Video Predictor is not initialized properly. Check logs.")

        import torch
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if device_type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=dtype):
            inference_state = self.predictor.init_state(
                video_path=temp_frame_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True
            )
            
            # Trong chunk này, frame được cấp box là frame đầu tiên của chunk (forward) hoặc cuối chunk (backward)
            prompt_frame_idx = prompt_frame_offset
            obj_ids = []
            for idx, (x, y, w, h) in enumerate(target_boxes):
                obj_id = idx + 1
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                
                # Trích xuất các điểm đặc trưng của chữ (Shi-Tomasi corners) để ép SAM 2 bám vào chữ thay vì nền
                import cv2
                frame_path = os.path.join(temp_frame_dir, f"{prompt_frame_idx:05d}.jpg")
                frame_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                points_np = None
                labels_np = None
                if frame_img is not None:
                    crop = frame_img[y:y+h, x:x+w]
                    # Tìm tối đa 15 điểm góc rõ nét nhất trên chữ
                    corners = cv2.goodFeaturesToTrack(crop, maxCorners=15, qualityLevel=0.05, minDistance=3)
                    if corners is not None:
                        # Chuyển đổi tọa độ local (crop) sang global (frame)
                        corners = corners.reshape(-1, 2)
                        corners[:, 0] += x
                        corners[:, 1] += y
                        points_np = np.array(corners, dtype=np.float32)
                        labels_np = np.ones((len(points_np),), dtype=np.int32)
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    points=points_np,
                    labels=labels_np,
                    box=box
                )
                obj_ids.append(obj_id)
            
        # Lan truyền nội suy cho cả video
        subtitle_frame_no_box_dict = {}
        
        # Thư mục lưu mask tạm (Đã loại bỏ để truyền mask qua RAM)
        # temp_mask_dir = os.path.join(os.path.dirname(self.video_path), ".temp_masks")
        # os.makedirs(temp_mask_dir, exist_ok=True)
        
        def process_masks(out_frame_idx, out_obj_ids, out_mask_logits):
            # Quy đổi từ index nội bộ của chunk sang frame_no thực tế của video
            frame_no = start_frame_no + out_frame_idx
            combined_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            has_mask = False
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                if mask.any():
                    has_mask = True
                    if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
                        mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    combined_mask[mask > 0] = 255
                    
            if has_mask:
                if padding > 0:
                    kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
                    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
                # Nén Mask ngay trên RAM thay vì lưu ra ổ đĩa (tiết kiệm hàng chục GB I/O)
                binary_mask = (combined_mask > 0).astype(bool)
                packed_mask = np.packbits(binary_mask)
                mask_shape = binary_mask.shape
                
                # Tính bounding box để giữ tương thích với hệ thống cũ (downstream tuple unpacking)
                # cv2.boundingRect trả về (x, y, w, h)
                x, y, w, h = cv2.boundingRect(combined_mask)
                xmin, ymin = x, y
                xmax, ymax = x + w, y + h
                
                # Trả về cả box và tuple(packed_mask, mask_shape)
                subtitle_frame_no_box_dict[frame_no] = [(xmin, xmax, ymin, ymax, (packed_mask, mask_shape))]
            else:
                subtitle_frame_no_box_dict[frame_no] = [None]
                
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=dtype):
            # Quét tới (Forward)
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                process_masks(out_frame_idx, out_obj_ids, out_mask_logits)
                
            # Quét lùi (Backward) nếu prompt không phải nằm ở frame đầu tiên
            if prompt_frame_idx > 0:
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, reverse=True):
                    process_masks(out_frame_idx, out_obj_ids, out_mask_logits)
            
        self.predictor.reset_state(inference_state)
        
        # Dọn dẹp temp frames sau khi track xong
        try:
            shutil.rmtree(temp_frame_dir)
        except Exception:
            pass
            
        # Thu dọn rác
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
            
        return subtitle_frame_no_box_dict
