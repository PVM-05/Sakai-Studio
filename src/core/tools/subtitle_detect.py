import sys
import logging
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
from functools import cached_property

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path
from .ocr import get_coordinates
from src.core.config import config, tr
from src.video_ops.scenedetect import scene_detect
from src.video_ops.scenedetect.detectors import ContentDetector
from src.core.tools.inpaint_tools import is_frame_number_in_ab_sections
from src.core.tools.constant import SubtitleDetectMode

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3

    def __init__(self, video_path, sub_areas=[]):
        self.video_path = video_path
        self.sub_areas = sub_areas
        self._init_sample_step()

    def _init_sample_step(self):
        """Tăng SAMPLE_STEP lên gấp đôi để đẩy tốc độ tìm kiếm phụ đề nhanh gấp 2 lần"""
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps >= 60:
            self.SAMPLE_STEP = 8  # Cũ là 4
        elif fps >= 30:
            self.SAMPLE_STEP = 6  # Cũ là 3
        else:
            self.SAMPLE_STEP = 4  # Cũ là 2

    @cached_property
    def text_detector(self):
        mode = config.subtitleDetectMode.value
        mode_val = mode.value if hasattr(mode, 'value') else str(mode)
        
        if mode_val == SubtitleDetectMode.RAPID_OCR.value:
            from rapidocr_onnxruntime import RapidOCR
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.hardwareAcceleration.value else ['CPUExecutionProvider']
            return RapidOCR(providers=providers)
        elif mode_val == SubtitleDetectMode.PADDLE_OCR.value:
            # PaddleOCR cổ điển: dùng build_paddleocr không truyền model dir
            from src.ai_engines.paddle_compat import build_paddleocr
            paddle_engine = build_paddleocr(
                lang="ch",
                device="gpu" if config.hardwareAcceleration.value else "cpu",
            )
            
            class PaddleOCRWrapper:
                def __init__(self, engine):
                    self.engine = engine
                def predict(self, frame):
                    """Chạy PaddleOCR cổ điển và trả về danh sách {'dt_polys': box}"""
                    from src.ai_engines.paddle_compat import extract_paddle_boxes
                    boxes = extract_paddle_boxes(self.engine, frame, threshold=0.3)
                    result = []
                    for (x1, y1, x2, y2) in boxes:
                        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        result.append({'dt_polys': poly})
                    return result

            return PaddleOCRWrapper(paddle_engine)
        else:
            # PaddleX (PP-OCRv5 Mobile hoặc Server)
            from src.ai_engines.paddle_compat import build_paddleocr
            from src.core.tools.model_config import ModelConfig
            model_config = ModelConfig()
            
            paddle_engine = build_paddleocr(
                lang="ch",
                device="gpu" if config.hardwareAcceleration.value else "cpu",
                text_detection_model_dir=model_config.DET_MODEL_DIR,
                text_recognition_model_dir=model_config.REC_MODEL_DIR,
            )
            
            class PaddleXWrapper:
                def __init__(self, engine):
                    self.engine = engine
                def predict(self, frame):
                    """Chạy PaddleX/PaddleOCR và trả về danh sách {'dt_polys': box}"""
                    from src.ai_engines.paddle_compat import extract_paddle_boxes
                    boxes = extract_paddle_boxes(self.engine, frame, threshold=0.3)
                    result = []
                    for (x1, y1, x2, y2) in boxes:
                        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        result.append({'dt_polys': poly})
                    return result

            return PaddleXWrapper(paddle_engine)

    def _get_coordinates_from_paddle(self, img):
        """Run PaddleOCR/RapidOCR and return coordinates in (xmin,xmax,ymin,ymax) format."""
        
        mode = config.subtitleDetectMode.value
        mode_val = mode.value if hasattr(mode, 'value') else str(mode)
        
        if mode_val == SubtitleDetectMode.RAPID_OCR.value:
            # RapidOCR returns a tuple (result, _)
            # result is a list of [box, text, score]
            result_rapid, _ = self.text_detector(img)
            coordinate_list = []
            if result_rapid:
                for line in result_rapid:
                    box, text, score = line
                    coordinate_list.extend(get_coordinates([box]))
            return coordinate_list
        else:
            # PaddleOCR wrapper returns a list of {'dt_polys': box}
            results = self.text_detector.predict(img)
            coordinate_list = []
            for res in results:
                dt_polys = res.get('dt_polys') if isinstance(res, dict) else getattr(res, 'dt_polys', None)
                if dt_polys is None or len(dt_polys) == 0:
                    continue
                poly_list = dt_polys.tolist() if hasattr(dt_polys, 'tolist') else dt_polys
                # Handle both single polygon (list of points) and list of polygons
                if poly_list and isinstance(poly_list[0][0], (int, float)):
                    poly_list = [poly_list]
                coordinate_list.extend(get_coordinates(poly_list))
            return coordinate_list

    def _merge_close_boxes(self, boxes, x_thresh=50, y_thresh=15):
        """
        Gộp các bounding box nằm gần nhau (đặc biệt là trên cùng một dòng ngang).
        Giải quyết triệt để lỗi các ký tự đặc biệt như 'npc._.arc' bị xé lẻ thành nhiều hộp.
        boxes: danh sách (xmin, xmax, ymin, ymax)
        """
        if not boxes:
            return []
            
        # Sắp xếp các box theo trục Y trước, sau đó trục X
        boxes = sorted(boxes, key=lambda b: (b[2], b[0]))
        
        merged = []
        for box in boxes:
            xmin, xmax, ymin, ymax = box
            merged_with_existing = False
            
            for i, mbox in enumerate(merged):
                m_xmin, m_xmax, m_ymin, m_ymax = mbox
                
                # Kiểm tra xem có cùng trên một dòng không (Y overlap hoặc khoảng cách Y nhỏ)
                y_overlap = max(0, min(ymax, m_ymax) - max(ymin, m_ymin))
                is_same_line = y_overlap > 0 or abs(ymin - m_ymin) <= y_thresh
                
                # Khoảng cách giữa 2 hộp theo trục X
                x_distance = max(0, max(xmin, m_xmin) - min(xmax, m_xmax)) 
                if xmax < m_xmin:
                    x_distance = m_xmin - xmax
                elif xmin > m_xmax:
                    x_distance = xmin - m_xmax
                else:
                    x_distance = 0
                    
                if is_same_line and x_distance <= x_thresh:
                    # Gộp box
                    new_xmin = min(xmin, m_xmin)
                    new_xmax = max(xmax, m_xmax)
                    new_ymin = min(ymin, m_ymin)
                    new_ymax = max(ymax, m_ymax)
                    merged[i] = (new_xmin, new_xmax, new_ymin, new_ymax)
                    merged_with_existing = True
                    break
                    
            if not merged_with_existing:
                merged.append(box)
                
        return merged

    def detect_subtitle(self, img):
        sub_areas = self.sub_areas
        has_areas = sub_areas is not None and len(sub_areas) > 0
        
        if has_areas:
            # Tự động tính toán hộp bao tối thiểu chứa tất cả các vùng sub_areas
            h, w = img.shape[:2]
            ymin_crop = max(0, min(area[0] for area in sub_areas))
            ymax_crop = min(h, max(area[1] for area in sub_areas))
            xmin_crop = max(0, min(area[2] for area in sub_areas))
            xmax_crop = min(w, max(area[3] for area in sub_areas))
            
            # Cắt ảnh
            cropped_img = img[ymin_crop:ymax_crop, xmin_crop:xmax_crop]
            if cropped_img.size == 0:
                return []
        else:
            cropped_img = img
            ymin_crop, xmin_crop = 0, 0

        # Kiểm tra nhanh: Nếu vùng chọn phẳng (như dải đen/nền trơn), bỏ qua OCR để tăng hiệu suất
        try:
            if cropped_img.ndim == 3:
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cropped_img
            if gray.std() < 3.0:
                return []
        except Exception:
            pass

        # --- Engine: PaddleX ---
        try:
            coordinate_list = self._get_coordinates_from_paddle(cropped_img)
        except Exception as e:
            print(f"[Detection] OCR Engine failed: {e}")
            return []

        # Dịch chuyển tọa độ phát hiện được về tọa độ gốc của video
        if has_areas and len(coordinate_list) > 0:
            shifted_list = []
            for xmin, xmax, ymin, ymax in coordinate_list:
                shifted_list.append((xmin + xmin_crop, xmax + xmin_crop, ymin + ymin_crop, ymax + ymin_crop))
            coordinate_list = shifted_list
            
        # Gộp các mảnh vỡ OCR lại thành một khối liền mạch
        coordinate_list = self._merge_close_boxes(coordinate_list)

        # --- Filter by subtitle area ---
        temp_list = []
        if not has_areas:
            temp_list.extend(coordinate_list)
        elif len(sub_areas) == 1:
            # 单区域快速路径（最常见场景）
            s_ymin, s_ymax, s_xmin, s_xmax = sub_areas[0]
            for xmin, xmax, ymin, ymax in coordinate_list:
                if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                    temp_list.append((xmin, xmax, ymin, ymax))
        else:
            for xmin, xmax, ymin, ymax in coordinate_list:
                for s_ymin, s_ymax, s_xmin, s_xmax in sub_areas:
                    if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                        temp_list.append((xmin, xmax, ymin, ymax))
                        break
        return temp_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        # 阶段1：采样检测，仅对每隔 sample_step 帧执行 OCR
        sampled_results = {}  # frame_no -> temp_list
        if sub_remover:
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            if not is_frame_number_in_ab_sections(current_frame_no - 1, sub_remover.ab_sections):
                tbar.update(1)
                continue
            # 仅对采样帧执行 OCR 推理
            if (current_frame_no - 1) % self.SAMPLE_STEP == 0 or self.SAMPLE_STEP <= 1:
                temp_list = self.detect_subtitle(frame)
                if len(temp_list) > 0:
                    sampled_results[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.current_frame_no = current_frame_no
                sub_remover.progress_total = int((float(current_frame_no) / float(frame_count)) * 40)
                sub_remover.notify_progress_listeners()
        video_cap.release()
        # 阶段2：插值填充 — 两个采样帧之间都有字幕时，中间帧使用线性插值的坐标
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                boxes_start = sampled_results[f]
                boxes_end = sampled_results[next_f]
                gap = next_f - f
                for fill_f in range(f + 1, next_f):
                    t = (fill_f - f) / gap  # Tỷ lệ nội suy 0.0 -> 1.0
                    # Nội suy tuyến tính toạ độ giữa 2 frame mẫu
                    if len(boxes_start) == len(boxes_end):
                        interpolated = []
                        for bs, be in zip(boxes_start, boxes_end):
                            interp_box = tuple(
                                int(round(bs[i] * (1 - t) + be[i] * t))
                                for i in range(4)
                            )
                            interpolated.append(interp_box)
                        subtitle_frame_no_box_dict[fill_f] = interpolated
                    else:
                        # Số lượng box khác nhau: dùng union (hộp bao lớn nhất) của cả 2 frame
                        union_boxes = []
                        all_boxes = list(boxes_start) + list(boxes_end)
                        for box in all_boxes:
                            if box not in union_boxes:
                                union_boxes.append(box)
                        subtitle_frame_no_box_dict[fill_f] = union_boxes
        # 添加最后一个检测帧
        if detected_nos:
            subtitle_frame_no_box_dict[detected_nos[-1]] = sampled_results[detected_nos[-1]]
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for region in current_regions:
                    matched_std = None
                    last_list = unify_value_map.get(last_key, [])
                    for cand in last_list:
                        if self.are_similar(region, cand):
                            matched_std = cand
                            break

                    if matched_std:
                        new_unify_values.append(matched_std)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        if not subtitle_frame_no_box_dict:
            return []
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        if not numbers:
            return []
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        if not subtitle_frame_no_box_dict:
            return []
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        if not numbers:
            return []
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        复杂度 O(n log n)
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(start - half, prev_end + 1)
                new_end = min(start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            elif start == last_end + 1 and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
