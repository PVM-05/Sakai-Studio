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
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections

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
    def _rapidocr(self):
        """RapidOCR 3.x engine (PP-OCRv6 ONNX) — primary, fastest, no hpip needed."""
        try:
            from rapidocr import RapidOCR
            import onnxruntime as ort
            
            # Tự động phát hiện xem CUDA có sẵn trong ONNX Runtime hay không
            use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            
            # Cấu hình tối ưu: chỉ chạy det (phát hiện chữ), bỏ qua cls và rec để chạy nhanh nhất có thể
            params = {
                "Global.use_det": True,
                "Global.use_cls": False,
                "Global.use_rec": False,
            }
            if use_cuda:
                params["EngineConfig.onnxruntime.use_cuda"] = True
                params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = 0
                print("[SubtitleDetect] RapidOCR khoi tao voi tang toc CUDA GPU")
            else:
                print("[SubtitleDetect] RapidOCR khoi tao chay tren CPU")
                
            ocr = RapidOCR(params=params)
            return ocr
        except Exception as e:
            print(f"[RapidOCR] Not available ({e}), will use PaddleOCR fallback")
            return None

    @cached_property
    def text_detector(self):
        """PaddleOCR fallback detector (used only when RapidOCR is unavailable)."""
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import TextDetection
        model_config = ModelConfig()
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device="cpu",
            enable_hpi=False,  # ultra-infer (hpip) not installed; use standard paddle backend
        )

    def _get_coordinates_from_rapidocr(self, img):
        """Run RapidOCR and return coordinates in (xmin,xmax,ymin,ymax) format."""
        rapid = self._rapidocr
        if rapid is None:
            raise RuntimeError("RapidOCR not available")
        output = rapid(img)
        if output is None:
            return []
        
        # Hỗ trợ cả 2 định dạng trả về của RapidOCR tùy thuộc vào cấu hình use_rec
        if hasattr(output, "boxes") and output.boxes is not None:
            dt_boxes = output.boxes
        elif isinstance(output, tuple) and output:
            result = output[0]
            if not result:
                return []
            dt_boxes = [item[0] for item in result]
        else:
            dt_boxes = []
            
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
            
        if hasattr(dt_boxes, "tolist"):
            dt_boxes = dt_boxes.tolist()
            
        return get_coordinates(dt_boxes)

    def _get_coordinates_from_paddle(self, img):
        """Run PaddleOCR and return coordinates in (xmin,xmax,ymin,ymax) format."""
        results = self.text_detector.predict(img)
        coordinate_list = []
        for res in results:
            dt_polys = res.get('dt_polys') if isinstance(res, dict) else getattr(res, 'dt_polys', None)
            if dt_polys is None or len(dt_polys) == 0:
                continue
            coordinate_list.extend(get_coordinates(
                dt_polys.tolist() if hasattr(dt_polys, 'tolist') else dt_polys
            ))
        return coordinate_list

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

        # --- Engine chain: RapidOCR (PP-OCRv6 ONNX) → PaddleOCR fallback ---
        try:
            coordinate_list = self._get_coordinates_from_rapidocr(cropped_img)
        except Exception:
            try:
                coordinate_list = self._get_coordinates_from_paddle(cropped_img)
            except Exception as e:
                print(f"[Detection] Both engines failed: {e}")
                return []

        # Dịch chuyển tọa độ phát hiện được về tọa độ gốc của video
        if has_areas and len(coordinate_list) > 0:
            shifted_list = []
            for xmin, xmax, ymin, ymax in coordinate_list:
                shifted_list.append((xmin + xmin_crop, xmax + xmin_crop, ymin + ymin_crop, ymax + ymin_crop))
            coordinate_list = shifted_list

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
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        video_cap.release()
        # 阶段2：插值填充 — 两个采样帧之间都有字幕时，中间帧也标记为有字幕
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                fill_mask = sampled_results[f]
                for fill_f in range(f + 1, next_f):
                    subtitle_frame_no_box_dict[fill_f] = fill_mask
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

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
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
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
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
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
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
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
