import cv2
import sys
from tqdm import tqdm
from backend.config import tr, config
from backend.tools.common_tools import get_readable_path
from backend.tools.subtitle_detect import SubtitleDetect

class RobustWatermarkTracker:
    def __init__(self, fallback_tracker_fn):
        self.fallback_tracker_fn = fallback_tracker_fn
        self.csrt = fallback_tracker_fn()
        self.template = None
        self.bbox = None
        self.original_w = 0
        self.original_h = 0
        
    def _rebuild_scale_cache(self):
        self.scale_cache = {}
        if self.template is not None:
            scales = [1.00, 0.95, 1.05, 0.90, 1.10]
            h, w = self.template.shape[:2]
            for scale in scales:
                sw, sh = int(w * scale), int(h * scale)
                if sw >= 5 and sh >= 5:
                    self.scale_cache[scale] = cv2.resize(self.template, (sw, sh), interpolation=cv2.INTER_LINEAR)

    def init(self, image, bbox):
        self.csrt.init(image, bbox)
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        self.original_w = w
        self.original_h = h
        self.frames_since_update = 0
        patch = image[y:y+h, x:x+w]
        if patch.size > 0:
            self.template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            self.original_template = self.template.copy()
            self._rebuild_scale_cache()
        else:
            self.original_template = None
            self.scale_cache = {}
        return True

    def update(self, image, gray_frame=None):
        # Update CSRT first to keep its model alive
        csrt_success, csrt_bbox = self.csrt.update(image)
        
        if getattr(self, "csrt_reset_cooldown", 0) > 0:
            self.csrt_reset_cooldown -= 1
            
        if gray_frame is None:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Template Matching (Chủ lực)
        if self.template is not None:
            x, y, _, _ = self.bbox
            h, w = self.template.shape[:2]
            ih, iw = image.shape[:2]
            
            # Adaptive Search Region
            last_score = getattr(self, 'last_score', 0)
            if last_score > 0.95:
                sp = 80
            elif last_score > 0.90:
                sp = 120
            else:
                sp = 220
            sp = max(int(max(w, h) * 2), sp) # Đảm bảo đủ bao trọn nếu logo to
            sp = min(sp, 250)
            
            sx = max(0, x - sp)
            sy = max(0, y - sp)
            ex = min(iw, x + w + sp)
            ey = min(ih, y + h + sp)
            
            search_gray = gray_frame[sy:ey, sx:ex]
            if search_gray.shape[0] >= h and search_gray.shape[1] >= w:
                
                # 1. Chạy Template Matching ở scale 1.0 trước (Tiết kiệm CPU)
                res_1 = cv2.matchTemplate(search_gray, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val_1, _, max_loc_1 = cv2.minMaxLoc(res_1)
                
                best_max_val = max_val_1
                best_max_loc = max_loc_1
                best_w, best_h = w, h
                
                # 2. Lazy Multi-scale: Chỉ chạy các scale khác nếu scale 1.0 không quá tốt (max_val < 0.85)
                if best_max_val < 0.85 and hasattr(self, 'scale_cache'):
                    scales_to_try = list(self.scale_cache.keys())
                    last_good = getattr(self, 'last_good_scale', 1.00)
                    if last_good in scales_to_try and last_good != 1.00:
                        scales_to_try.remove(last_good)
                        scales_to_try.insert(1, last_good) # Ưu tiên thử ngay sau 1.00
                        
                    for scale in scales_to_try:
                        if scale == 1.00: continue
                        scaled_template = self.scale_cache[scale]
                        sh, sw = scaled_template.shape[:2]
                        if search_gray.shape[0] >= sh and search_gray.shape[1] >= sw:
                            res = cv2.matchTemplate(search_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(res)
                            
                            # Chống nhấp nháy (flickering): Chỉ đổi scale nếu độ tin cậy vượt trội hơn hẳn (> 0.03)
                            if max_val > best_max_val + 0.03:
                                best_max_val = max_val
                                best_max_loc = max_loc
                                best_w, best_h = sw, sh
                                self.last_good_scale = scale
                else:
                    self.last_good_scale = 1.00
                
                self.last_score = best_max_val
                
                if best_max_val > 0.75: # Ngưỡng chống bắt nhầm vào các vật thể có vân phức tạp
                    new_bbox = (sx + best_max_loc[0], sy + best_max_loc[1], best_w, best_h)
                    
                    # Cross-check vị trí mới với anchor bất biến trước khi CHẤP NHẬN bbox
                    # Chặn lock-in vào texture nền (ruộng ngô, bụi cây...) khi độ tin cậy rơi vào vùng xám (0.75 - 0.90)
                    accept_position = True
                    if best_max_val < 0.90 and hasattr(self, 'original_template') and self.original_template is not None:
                        if new_bbox[3] > 0 and new_bbox[2] > 0:
                            try:
                                cand_gray = gray_frame[new_bbox[1]:new_bbox[1]+new_bbox[3], new_bbox[0]:new_bbox[0]+new_bbox[2]]
                                if cand_gray.shape[0] == new_bbox[3] and cand_gray.shape[1] == new_bbox[2]:
                                    oh, ow = self.original_template.shape[:2]
                                    cand_resized = cv2.resize(cand_gray, (ow, oh), interpolation=cv2.INTER_LINEAR)
                                    res_o = cv2.matchTemplate(cand_resized, self.original_template, cv2.TM_CCOEFF_NORMED)
                                    _, max_val_orig, _, _ = cv2.minMaxLoc(res_o)
                                    if max_val_orig < 0.55:  # Siết chặt ngưỡng cross-check chống drift
                                        accept_position = False
                            except Exception as e:
                                print(f'[ObjectTracker] Lỗi cross-check vị trí: {e}')
                                accept_position = False

                    if accept_position:
                        # EMA Smoothing for position to reduce jitter
                        alpha = 0.7
                        nx = int(alpha * self.bbox[0] + (1 - alpha) * new_bbox[0])
                        ny = int(alpha * self.bbox[1] + (1 - alpha) * new_bbox[1])
                        self.bbox = (nx, ny, best_w, best_h)
                        self.frames_since_update += 1
                        
                        # Cập nhật Template động (Template Refresh) khi tự tin cực cao (> 0.93)
                        if best_max_val > 0.93 and self.frames_since_update > 60:
                            if new_bbox[3] > 0 and new_bbox[2] > 0:
                                patch_gray = gray_frame[new_bbox[1]:new_bbox[1]+new_bbox[3], new_bbox[0]:new_bbox[0]+new_bbox[2]]
                                if patch_gray.shape[0] == new_bbox[3] and patch_gray.shape[1] == new_bbox[2]:
                                    can_refresh = False
                                    if hasattr(self, 'original_template') and self.original_template is not None:
                                        try:
                                            oh, ow = self.original_template.shape[:2]
                                            patch_resized = cv2.resize(patch_gray, (ow, oh), interpolation=cv2.INTER_LINEAR)
                                            res_o = cv2.matchTemplate(patch_resized, self.original_template, cv2.TM_CCOEFF_NORMED)
                                            _, max_val_orig, _, _ = cv2.minMaxLoc(res_o)
                                            if max_val_orig > 0.80:
                                                can_refresh = True
                                        except:
                                            pass
                                else:
                                    can_refresh = True
                                    
                                if can_refresh:
                                        new_template = patch_gray.copy()
                                        cur_h, cur_w = new_template.shape[:2]
                                        old_template_resized = cv2.resize(self.template, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
                                        self.template = cv2.addWeighted(old_template_resized, 0.8, new_template, 0.2, 0)
                                        self._rebuild_scale_cache()
                                        self.frames_since_update = 0
                                        self.original_w = new_bbox[2]
                                        self.original_h = new_bbox[3]
                                
                        # Nếu CSRT lệch quá xa, khởi tạo lại CSRT để đồng bộ
                        if csrt_success:
                            cx, cy, cw, ch = csrt_bbox
                            dist = (cx - new_bbox[0])**2 + (cy - new_bbox[1])**2
                            if dist > 400 or abs(cw - best_w) > best_w*0.5 or abs(ch - best_h) > best_h*0.5: # Lệch hoặc biến dạng
                                if getattr(self, 'csrt_reset_cooldown', 0) <= 0:
                                    self.csrt = self.fallback_tracker_fn()
                                    self.csrt.init(image, new_bbox)
                                    self.csrt_reset_cooldown = 20
                        else:
                            if getattr(self, 'csrt_reset_cooldown', 0) <= 0:
                                self.csrt = self.fallback_tracker_fn()
                                self.csrt.init(image, new_bbox)
                                self.csrt_reset_cooldown = 20
                        
                        return True, self.bbox
        
        # 2. CSRT (Dự phòng)
        if csrt_success:
            cx, cy, cw, ch = [int(v) for v in csrt_bbox]
            # Chống phình to hoặc co quắt (Drifting)
            if self.original_w > 0 and self.original_h > 0:
                if cw > self.original_w * 1.5 or ch > self.original_h * 1.5 or cw < self.original_w * 0.5 or ch < self.original_h * 0.5:
                    return False, self.bbox # Coi như mất dấu để ép Re-initialize
            
            # Xác thực chéo CSRT bằng template gốc để chặn false-positive
            if hasattr(self, 'original_template') and self.original_template is not None:
                try:
                    ih, iw = image.shape[:2]
                    vx = max(0, min(cx, iw - 1))
                    vy = max(0, min(cy, ih - 1))
                    vw = max(1, min(cw, iw - vx))
                    vh = max(1, min(ch, ih - vy))
                    
                    c_gray = gray_frame[vy:vy+vh, vx:vx+vw]
                    if c_gray.shape[0] == vh and c_gray.shape[1] == vw and vw > 0 and vh > 0:
                        verify_template = self.template if self.template is not None else self.original_template
                        th, tw = verify_template.shape[:2]
                        c_gray = cv2.resize(c_gray, (tw, th), interpolation=cv2.INTER_LINEAR)
                        score = cv2.matchTemplate(c_gray, verify_template, cv2.TM_CCOEFF_NORMED)[0][0]
                        
                        threshold = 0.55
                        if hasattr(self, 'frames_since_update') and self.frames_since_update > 200:
                            threshold = 0.45
                            
                        if score < threshold: # Lỏng hơn Template Matching một chút nhưng đủ để chặn bám bụi cây
                            return False, self.bbox
                except Exception as e:
                    print(f"[ObjectTracker] CSRT verify error: {e}")
            
            self.bbox = (cx, cy, cw, ch)
            return True, self.bbox
            
        return False, self.bbox

class ObjectTracker:
    """
    Theo dõi đối tượng chuyển động (logo, phụ đề cuộn) bằng thuật toán OpenCV Tracker.
    """
    def __init__(self, video_path, sub_areas, start_frame=1):
        self.video_path = video_path
        self.sub_areas = sub_areas  # list of (ymin, ymax, xmin, xmax)
        self.start_frame = start_frame
        self.trackers = []
        self.tracker_boxes = []

    def _open_capture_no_hwaccel(self, path):
        # Ép FFmpeg dùng software decode, tránh lỗi Static surface pool exceeded
        # của HW decoder (DXVA2/D3D11VA) khi seek liên tục với VP9
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;none"
        cap = cv2.VideoCapture(get_readable_path(path), cv2.CAP_FFMPEG)
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
        except Exception:
            pass  # OpenCV build cũ có thể không hỗ trợ property này
        return cap

    def _create_tracker(self):
        algo = getattr(config, 'trackerAlgorithm', None)
        algo_name = algo.value if algo else 'csrt'
        
        def _make(name):
            # 1. OpenCV 4.5.3+ standard API (cv2.TrackerCSRT.create)
            cls_name = f"Tracker{name.upper()}"
            if hasattr(cv2, cls_name) and hasattr(getattr(cv2, cls_name), 'create'):
                return getattr(cv2, cls_name).create()
            # 2. Legacy API (cv2.legacy.TrackerCSRT_create)
            if hasattr(cv2, 'legacy'):
                legacy_fn = f"Tracker{name.upper()}_create"
                if hasattr(cv2.legacy, legacy_fn):
                    return getattr(cv2.legacy, legacy_fn)()
            # 3. OpenCV 3.x/4.0 API (cv2.TrackerCSRT_create)
            fn_name = f"Tracker{name.upper()}_create"
            if hasattr(cv2, fn_name):
                return getattr(cv2, fn_name)()
            raise AttributeError(f"OpenCV Tracker {name} not supported")

        try:
            fallback = lambda: _make(algo_name)
            # Luôn bọc tracker bằng Hybrid Tracker để chống mất dấu (drift)
            return RobustWatermarkTracker(fallback)
        except Exception:
            for fallback_name in ['csrt', 'kcf', 'mil']:
                try:
                    fallback = lambda name=fallback_name: _make(name)
                    return RobustWatermarkTracker(fallback)
                except Exception:
                    continue
            raise RuntimeError("No suitable OpenCV Tracker found.")

    def _reinit_tracker(self, index, frame, ocr_boxes, sub_remover, trackers, boxes):
        """Thử tái lấy nét khi mất dấu bằng OCR."""
        if len(ocr_boxes) == 0:
            return False
            
        frame_h, frame_w = frame.shape[:2]
        old_x, old_y, old_w, old_h = boxes[index]
        old_center_x = old_x + old_w / 2
        old_center_y = old_y + old_h / 2
        
        best_box = None
        min_score = 999999
        for obox in ocr_boxes:
            oxmin, oxmax, oymin, oymax = obox
            ox, oy, ow, oh = oxmin, oymin, oxmax - oxmin, oymax - oymin
            
            area_ratio = 1.0
            aspect_ratio_diff = 0.0
            # Chống nhảy box: Kiểm tra tỷ lệ diện tích và aspect ratio
            if old_w * old_h > 0 and oh > 0 and old_h > 0:
                area_ratio = min(ow * oh, old_w * old_h) / max(ow * oh, old_w * old_h)
                if area_ratio < 0.3:
                    continue
                old_aspect = old_w / old_h
                new_aspect = ow / oh
                aspect_ratio_diff = abs(old_aspect - new_aspect)
                    
            center_x = ox + ow / 2
            center_y = oy + oh / 2
            
            dist = (center_x - old_center_x)**2 + (center_y - old_center_y)**2
            
            # Kết hợp distance với penalty từ kích thước và hình dáng
            score = dist * (1 + (1 - area_ratio) * 2 + aspect_ratio_diff * 2)
            
            if score < min_score:
                min_score = score
                best_box = (int(ox), int(oy), int(ow), int(oh))
                
        if best_box and min_score < 80000: 
            bx, by, bw, bh = best_box
            bx = int(max(0, min(bx, frame_w - 5)))
            by = int(max(0, min(by, frame_h - 5)))
            bw = int(max(5, min(bw, frame_w - bx)))
            bh = int(max(5, min(bh, frame_h - by)))
            if bw >= 5 and bh >= 5:
                try:
                    t = self._create_tracker()
                    t.init(frame, (bx, by, bw, bh))
                    trackers[index] = t
                    boxes[index] = (bx, by, bw, bh)
                    if sub_remover:
                        sub_remover.append_output("Tái lấy nét thành công!")
                    return True
                except Exception as e:
                    print(f"[ObjectTracker] Re-init failed: {e}")
        return False


    def _update_trackers(self, frame, trackers, boxes, lost_counts, sub_detector, sub_remover, padding=10):
        frame_boxes = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ocr_boxes_cache = None
        for i, tracker in enumerate(trackers):
            try:
                success, bbox = tracker.update(frame, gray_frame)
                if success:
                    x, y, w, h = bbox
                    boxes[i] = (int(x), int(y), int(w), int(h))
                    lost_counts[i] = 0
                else:
                    if ocr_boxes_cache is None:
                        ocr_boxes_cache = sub_detector.detect_subtitle(frame)
                    reinit_success = self._reinit_tracker(i, frame, ocr_boxes_cache, sub_remover, trackers, boxes)
                    if reinit_success:
                        lost_counts[i] = 0
                    else:
                        lost_counts[i] += 1
            except Exception as e:
                print(f'[ObjectTracker] Lỗi khi cập nhật tracker {i}: {e}')
                lost_counts[i] += 1
            
            # Chỉ trả về bbox nếu bị mất dấu liên tục dưới 5 frame
            if lost_counts[i] < 5:
                x, y, w, h = boxes[i]
                fh, fw = frame.shape[:2]
                frame_boxes.append((
                    max(0, int(x) - padding),
                    min(fw, int(x + w) + padding),
                    max(0, int(y) - padding),
                    min(fh, int(y + h) + padding)
                ))
        return frame_boxes

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = self._open_capture_no_hwaccel(self.video_path)
        raw_fc = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = max(1, int(raw_fc)) if (raw_fc and not float('nan') == raw_fc and raw_fc > 0) else 1
        tbar = tqdm(total=frame_count, unit='frame', position=0, file=sys.__stdout__, desc='Object Tracking')
        
        subtitle_frame_no_box_dict = {}
        
        if sub_remover:
            sub_remover.append_output("Đang khởi tạo thuật toán theo dõi đối tượng...")

        current_frame_no = 0
        if getattr(self, 'start_frame', 1) > 1:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)
            current_frame_no = self.start_frame - 1
            if sub_remover:
                sub_remover.append_output(f"Bắt đầu bám đuổi từ Frame {self.start_frame}...")

        # Đọc frame đầu tiên để khởi tạo
        ret, frame = video_cap.read()
        if not ret:
            video_cap.release()
            return {}

        current_frame_no += 1
        tbar.update(current_frame_no)
        
        # Kiểm tra xem người dùng có khoanh vùng tùy chỉnh nhỏ hay không
        has_custom_boxes = False
        for area in self.sub_areas:
            ymin, ymax, xmin, xmax = area
            if not (xmin == 0 and ymin == 0 and xmax >= frame.shape[1] - 1 and ymax >= frame.shape[0] - 1):
                has_custom_boxes = True
                break

        # Luôn khởi tạo sub_detector để dùng chung
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        
        target_boxes = []
        frame_h, frame_w = frame.shape[:2]
        if has_custom_boxes:
            if sub_remover:
                sub_remover.append_output("Đang dùng AI OCR tự động tinh chỉnh chính xác vị trí đối tượng trong vùng đã chọn...")
            ocr_boxes = sub_detector.detect_subtitle(frame)
            
            for area in self.sub_areas:
                ymin, ymax, xmin, xmax = area
                if not (xmin == 0 and ymin == 0 and xmax >= frame_w - 1 and ymax >= frame_h - 1):
                    # Tìm các ocr box thuộc area này
                    area_ocr_boxes = []
                    for obox in ocr_boxes:
                        oxmin, oxmax, oymin, oymax = obox
                        center_x = (oxmin + oxmax) / 2
                        center_y = (oymin + oymax) / 2
                        if xmin <= center_x <= xmax and ymin <= center_y <= ymax:
                            area_ocr_boxes.append(obox)
                    
                    if area_ocr_boxes:
                        # GỘP TẤT CẢ mảnh vỡ OCR trong vùng khoanh tay thành 1 hộp lớn duy nhất!
                        min_x = min(obox[0] for obox in area_ocr_boxes)
                        max_x = max(obox[1] for obox in area_ocr_boxes)
                        min_y = min(obox[2] for obox in area_ocr_boxes)
                        max_y = max(obox[3] for obox in area_ocr_boxes)
                        
                        x = int(max(0, min(min_x, frame_w - 5)))
                        y = int(max(0, min(min_y, frame_h - 5)))
                        w = int(max(5, min(max_x - min_x, frame_w - x)))
                        h = int(max(5, min(max_y - min_y, frame_h - y)))
                        
                        if w >= 5 and h >= 5:
                            target_boxes.append((x, y, w, h))
                    else:
                        x = int(max(0, min(xmin, frame_w - 5)))
                        y = int(max(0, min(ymin, frame_h - 5)))
                        w = int(max(5, min(xmax - xmin, frame_w - x)))
                        h = int(max(5, min(ymax - ymin, frame_h - y)))
                        if w >= 5 and h >= 5:
                            target_boxes.append((x, y, w, h))
        else:
            # Tự động quét OCR trên frame 1 để tìm tất cả các logo / chữ di chuyển
            if sub_remover:
                sub_remover.append_output("Đang dùng AI OCR tự động dò tìm đối tượng/logo di chuyển trên khung hình...")
            ocr_boxes = sub_detector.detect_subtitle(frame)
            for (xmin, xmax, ymin, ymax) in ocr_boxes:
                x = int(max(0, min(xmin, frame_w - 5)))
                y = int(max(0, min(ymin, frame_h - 5)))
                w = int(max(5, min(xmax - xmin, frame_w - x)))
                h = int(max(5, min(ymax - ymin, frame_h - y)))
                if w >= 5 and h >= 5:
                    target_boxes.append((x, y, w, h))

        for bbox in target_boxes:
            try:
                tracker = self._create_tracker()
                tracker.init(frame, bbox)
                self.trackers.append(tracker)
                self.tracker_boxes.append(bbox)
            except Exception as e:
                print(f"[ObjectTracker] Khởi tạo tracker thất bại cho vùng {bbox}: {e}")
            
        if not self.trackers:
            if sub_remover:
                sub_remover.append_output("Không tìm thấy đối tượng di chuyển nào trên khung hình. Chuyển về chế độ tĩnh.")
            video_cap.release()
            return {}

        # Lưu frame đầu (start_frame)
        temp_list = []
        padding = 10
        for bbox in self.tracker_boxes:
            x, y, w, h = bbox
            temp_list.append((max(0, int(x) - padding), min(frame_w, int(x+w) + padding), max(0, int(y) - padding), min(frame_h, int(y+h) + padding)))
        subtitle_frame_no_box_dict[current_frame_no] = temp_list
        
        # BACKWARD TRACKING (Chunk-based để chống Crash HW Decoder)
        if current_frame_no > 1:
            if sub_remover:
                sub_remover.append_output(f"Đang tiến hành bám đuổi ngược (Backward Tracking) về Frame 1...")
            
            # Khởi tạo tracker riêng cho việc lùi
            back_trackers = []
            back_boxes = list(self.tracker_boxes)
            back_lost_counts = [0] * len(back_boxes)
            for i, bbox in enumerate(back_boxes):
                try:
                    t = self._create_tracker()
                    t.init(frame, bbox)
                    back_trackers.append(t)
                except Exception as e:
                    print(f'[ObjectTracker] Khởi tạo backward tracker {i} thất bại: {e}')
                    back_trackers.append(None) # Giữ nguyên index
            
            # Tính toán CHUNK_SIZE động dựa trên độ phân giải để tối ưu RAM
            pixels = frame_h * frame_w
            if pixels > 5000000: # ~4K
                CHUNK_SIZE = 15
            elif pixels > 1500000: # ~1080p
                CHUNK_SIZE = 30
            else:
                CHUNK_SIZE = 60
            end_frame_to_read = current_frame_no - 1
            
            while end_frame_to_read > 0:
                start_frame_to_read = max(1, end_frame_to_read - CHUNK_SIZE + 1)
                
                # Re-open video capture to flush FFmpeg HW decoder pool (chống lỗi Static surface pool size exceeded)
                video_cap.release()
                video_cap = self._open_capture_no_hwaccel(self.video_path)
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_to_read - 1)
                
                # Đọc tiến toàn bộ chunk vào RAM
                chunk_frames = []
                for _ in range(start_frame_to_read, end_frame_to_read + 1):
                    ret_c, frame_c = video_cap.read()
                    if not ret_c:
                        break
                    chunk_frames.append(frame_c)
                    
                # Xử lý lùi (cho từng frame ngược về) và giải phóng RAM tức thì bằng pop()
                frame_indices = list(range(start_frame_to_read, start_frame_to_read + len(chunk_frames)))
                for i_idx in range(len(chunk_frames) - 1, -1, -1):
                    frame_back = chunk_frames.pop() # Lấy ra và xóa khỏi list để tiết kiệm RAM
                    real_frame_no = frame_indices[i_idx]
                    
                    frame_back_boxes = self._update_trackers(
                        frame_back, back_trackers, back_boxes, back_lost_counts, 
                        sub_detector, sub_remover, padding=padding
                    )
                    if len(frame_back_boxes) > 0:
                        subtitle_frame_no_box_dict[real_frame_no] = frame_back_boxes
                
                end_frame_to_read = start_frame_to_read - 1
                
            # Phục hồi con trỏ video về frame hiện tại để chuẩn bị chạy tới
            video_cap.release()
            video_cap = self._open_capture_no_hwaccel(self.video_path)
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no)
            if sub_remover:
                sub_remover.append_output("Hoàn tất bám đuổi ngược. Tiếp tục quét tiến về phía sau...")

        # Quét các frame tiếp theo
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
                
            current_frame_no += 1
            tbar.update(1)
            
            if not hasattr(self, 'forward_lost_counts'):
                self.forward_lost_counts = [0] * len(self.trackers)
                
            frame_boxes = self._update_trackers(
                frame, self.trackers, self.tracker_boxes, self.forward_lost_counts, 
                sub_detector, sub_remover, padding=10
            )
                
            if len(frame_boxes) > 0:
                subtitle_frame_no_box_dict[current_frame_no] = frame_boxes
            
            if sub_remover:
                sub_remover.current_frame_no = current_frame_no
                sub_remover.progress_total = int((float(current_frame_no) / float(frame_count)) * 40)
                if hasattr(sub_remover, 'notify_progress_listeners'):
                    sub_remover.notify_progress_listeners()
                elif hasattr(sub_remover, 'progress_signal'):
                    sub_remover.progress_signal.emit(sub_remover.progress_total, False, current_frame_no)

        video_cap.release()
        if sub_remover:
            sub_remover.append_output("Hoàn tất bám đuổi vị trí đối tượng.")
            
        return subtitle_frame_no_box_dict
