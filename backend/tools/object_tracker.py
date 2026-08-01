import cv2
import sys
from tqdm import tqdm
from backend.config import tr, config
from backend.tools.common_tools import get_readable_path
from backend.tools.subtitle_detect import SubtitleDetect

class ObjectTracker:
    """
    Theo dõi đối tượng chuyển động (logo, phụ đề cuộn) bằng thuật toán OpenCV Tracker.
    """
    def __init__(self, video_path, sub_areas):
        self.video_path = video_path
        self.sub_areas = sub_areas  # list of (ymin, ymax, xmin, xmax)
        self.trackers = []
        self.tracker_boxes = []

    def _create_tracker(self):
        algo = getattr(config, 'trackerAlgorithm', None)
        algo_name = algo.value if algo else 'csrt'
        
        try:
            if algo_name == 'kcf':
                return cv2.TrackerKCF_create()
            elif algo_name == 'mil':
                return cv2.TrackerMIL_create()
            else:
                return cv2.TrackerCSRT_create()
        except AttributeError:
            # Fallback nếu OpenCV version không hỗ trợ thuật toán được chọn
            try:
                return cv2.TrackerCSRT_create()
            except AttributeError:
                try:
                    return cv2.TrackerKCF_create()
                except AttributeError:
                    return cv2.TrackerMIL_create()

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Object Tracking')
        
        subtitle_frame_no_box_dict = {}
        current_frame_no = 0
        
        if sub_remover:
            sub_remover.append_output("Đang khởi tạo thuật toán theo dõi đối tượng (Moving Tracking)...")

        # Đọc frame đầu tiên để khởi tạo
        ret, frame = video_cap.read()
        if not ret:
            video_cap.release()
            return {}

        current_frame_no += 1
        tbar.update(1)
        
        # Khởi tạo trackers cho các vùng được khoanh
        for area in self.sub_areas:
            ymin, ymax, xmin, xmax = area
            # Bỏ qua nếu là vùng full màn hình (có thể do chưa khoanh)
            if xmin == 0 and ymin == 0 and xmax >= frame.shape[1] - 1 and ymax >= frame.shape[0] - 1:
                continue
            
            x = int(max(0, xmin))
            y = int(max(0, ymin))
            w = int(max(1, xmax - xmin))
            h = int(max(1, ymax - ymin))
            bbox = (x, y, w, h)
            
            tracker = self._create_tracker()
            tracker.init(frame, bbox)
            self.trackers.append(tracker)
            self.tracker_boxes.append(bbox)
            
        if not self.trackers:
            if sub_remover:
                sub_remover.append_output("Không có vùng tùy chỉnh hợp lệ để theo dõi. Chuyển về chế độ tĩnh.")
            video_cap.release()
            return {}

        # Lưu frame 1
        temp_list = []
        for bbox in self.tracker_boxes:
            x, y, w, h = bbox
            temp_list.append((int(x), int(x+w), int(y), int(y+h)))
        subtitle_frame_no_box_dict[current_frame_no] = temp_list

        # Quét các frame tiếp theo
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
                
            current_frame_no += 1
            tbar.update(1)
            
            frame_boxes = []
            for i, tracker in enumerate(self.trackers):
                success, bbox = tracker.update(frame)
                if success:
                    self.tracker_boxes[i] = bbox
                else:
                    # Nếu mất dấu, tiến hành tái lấy nét (Re-initialization) bằng OCR
                    if sub_remover:
                        sub_remover.append_output("Mất dấu đối tượng, đang thử tái lấy nét bằng OCR...")
                    
                    sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
                    ocr_boxes = sub_detector.detect_subtitle(frame)
                    
                    re_init_success = False
                    if len(ocr_boxes) > 0:
                        # Lấy tọa độ cũ
                        old_x, old_y, old_w, old_h = self.tracker_boxes[i]
                        old_center_x = old_x + old_w / 2
                        old_center_y = old_y + old_h / 2
                        
                        # Tìm box gần nhất
                        best_box = None
                        min_dist = 999999
                        for obox in ocr_boxes:
                            oxmin, oxmax, oymin, oymax = obox
                            ox, oy, ow, oh = oxmin, oymin, oxmax - oxmin, oymax - oymin
                            center_x = ox + ow / 2
                            center_y = oy + oh / 2
                            
                            dist = (center_x - old_center_x)**2 + (center_y - old_center_y)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_box = (ox, oy, ow, oh)
                                
                        # Nếu khoảng cách chấp nhận được (bán kính < 200 pixel)
                        if best_box and min_dist < 40000: 
                            bbox = best_box
                            self.tracker_boxes[i] = bbox
                            self.trackers[i] = self._create_tracker()
                            self.trackers[i].init(frame, bbox)
                            re_init_success = True
                            if sub_remover:
                                sub_remover.append_output("Tái lấy nét thành công!")
                                
                    if not re_init_success:
                        # Nếu không thể phục hồi, dùng lại tọa độ gần nhất
                        bbox = self.tracker_boxes[i]
                    
                x, y, w, h = bbox
                frame_boxes.append((int(x), int(x+w), int(y), int(y+h)))
                
            subtitle_frame_no_box_dict[current_frame_no] = frame_boxes
            
            if sub_remover:
                sub_remover.current_frame_no = current_frame_no
                sub_remover.progress_total = int((float(current_frame_no) / float(frame_count)) * 40)
                sub_remover.notify_progress_listeners()

        video_cap.release()
        if sub_remover:
            sub_remover.append_output("Hoàn tất bám đuổi vị trí đối tượng.")
            
        return subtitle_frame_no_box_dict
