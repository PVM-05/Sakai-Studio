import multiprocessing
import cv2
import numpy as np
from typing import List, Tuple

from src.core.config import config

def batch_generator(data, max_batch_size):
    """
    根据data大小，生成最大长度不超过max_batch_size的均匀批次数据
    """
    n_samples = len(data)
    # 尝试找到一个比MAX_BATCH_SIZE小的batch_size，以使得所有的批次数量尽量接近
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 处理最后一批可能不足batch_size的情况
    # 如果最后一批少于其他批次，则减小batch_size尝试平衡每批的数量
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 减小批次大小
        num_batches = n_samples // batch_size

    # 生成前num_batches个批次
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 将剩余的数据作为最后一个批次
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]

def sub_area_to_box_coords(area: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Chuyển đổi vùng phụ đề (ymin, ymax, xmin, xmax) sang hộp tọa độ mặt nạ (xmin, xmax, ymin, ymax)."""
    ymin, ymax, xmin, xmax = area[:4]
    return (int(xmin), int(xmax), int(ymin), int(ymax))


def box_coords_to_sub_area(coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Chuyển đổi hộp tọa độ mặt nạ (xmin, xmax, ymin, ymax) sang vùng phụ đề (ymin, ymax, xmin, xmax)."""
    xmin, xmax, ymin, ymax = coords[:4]
    return (int(ymin), int(ymax), int(xmin), int(xmax))


def calc_adaptive_padding(text_height: int, k: float = 0.15, p_min: int = 3, p_max: int = 20) -> int:
    """Tính toán khoảng đệm thích ứng theo chiều cao chữ để bao trọn viền và bóng đổ."""
    h = max(1, int(text_height))
    pad = int(round(h * k))
    return max(int(p_min), min(int(p_max), pad))


def _unpack_sam2_mask(sam2_data, target_h: int, target_w: int) -> np.ndarray:
    """Giải nén an toàn mặt nạ nhị phân SAM2 cho mọi kích thước khung hình (kể cả khi H*W không chia hết cho 8)."""
    packed_mask, mask_shape = sam2_data
    total_bits = int(mask_shape[0]) * int(mask_shape[1])
    binary_mask = np.unpackbits(packed_mask)[:total_bits].reshape(mask_shape)
    return binary_mask[:target_h, :target_w]


def _is_stroke_dict_list(items) -> bool:
    """Kiểm tra danh sách đầu vào có phải là danh sách nét cọ tự do (dict chứa 'points') hay không."""
    if not items or not isinstance(items, (list, tuple)):
        return False
    for item in items:
        if item is not None:
            return isinstance(item, dict) and "points" in item
    return False


def _build_box_uint8_mask(size: Tuple[int, int], coords_list, dilation: int = 8, frame: np.ndarray = None) -> np.ndarray:
    """Tạo mặt nạ nhị phân uint8 (0 hoặc 255) từ danh sách hộp tọa độ 4 phần tử hoặc 5 phần tử SAM2."""
    h, w = int(size[0]), int(size[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if not coords_list:
        return mask

    for coords in coords_list:
        if coords is None or isinstance(coords, dict):
            continue
        if len(coords) == 5 and coords[4] is not None:
            xmin, xmax, ymin, ymax, sam2_data = coords
            binary_mask = _unpack_sam2_mask(sam2_data, h, w)
            box_height = abs(ymax - ymin)
            pad = max(int(dilation), calc_adaptive_padding(box_height))

            if pad > 0:
                kernel = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
                binary_mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
                dilated_mask = cv2.dilate(binary_mask_uint8, kernel, iterations=1)
                mask[dilated_mask > 0] = 255
            else:
                mask[binary_mask > 0] = 255
        elif len(coords) >= 4:
            xmin, xmax, ymin, ymax = [int(v) for v in coords[:4]]
            box_height = abs(ymax - ymin)
            pad = max(int(dilation), calc_adaptive_padding(box_height))

            x1 = max(0, xmin - pad)
            y1 = max(0, ymin - pad)
            x2 = min(w, xmax + pad)
            y2 = min(h, ymax + pad)
            if x2 <= x1 or y2 <= y1:
                continue

            if frame is not None and getattr(config, 'mmoSmartTextMask', None) and config.mmoSmartTextMask.value:
                try:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
                        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
                        grad = cv2.magnitude(grad_x, grad_y)
                        grad = np.uint8(np.clip(grad, 0, 255))
                        _, text_mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
                        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
                        if pad > 0:
                            d_kernel = np.ones((pad, pad), np.uint8)
                            text_mask = cv2.dilate(text_mask, d_kernel, iterations=1)
                        if np.any(text_mask > 0):
                            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], text_mask)
                        else:
                            mask[y1:y2, x1:x2] = 255
                    else:
                        mask[y1:y2, x1:x2] = 255
                except Exception as e:
                    print(f"Smart Text Mask error: {e}")
                    mask[y1:y2, x1:x2] = 255
            else:
                mask[y1:y2, x1:x2] = 255

    has_sam2_mask_already = any(c is not None and not isinstance(c, dict) and len(c) == 5 for c in coords_list)
    is_smart_mask_on = getattr(config, 'mmoSmartTextMask', None) and config.mmoSmartTextMask.value
    if frame is not None and not has_sam2_mask_already and not is_smart_mask_on and getattr(config, 'sam2Refine', None) and config.sam2Refine.value:
        try:
            from src.ai_engines.segmentation import refine_mask_with_sam2
            valid_boxes = [c[:4] for c in coords_list if c is not None and not isinstance(c, dict) and len(c) >= 4]
            if valid_boxes:
                boxes = [[max(0, c[0] - max(int(dilation), calc_adaptive_padding(abs(c[3] - c[2])))),
                          max(0, c[2] - max(int(dilation), calc_adaptive_padding(abs(c[3] - c[2])))),
                          min(w, c[1] + max(int(dilation), calc_adaptive_padding(abs(c[3] - c[2])))),
                          min(h, c[3] + max(int(dilation), calc_adaptive_padding(abs(c[3] - c[2]))))] for c in valid_boxes]
                mask = refine_mask_with_sam2(
                    frame,
                    boxes,
                    mask,
                    "cuda" if getattr(config, 'hardwareAcceleration', None) and config.hardwareAcceleration.value else "cpu",
                )
        except Exception as exc:
            print(f"SAM 2 refinement skipped: {exc}")

    return mask


def _build_text_stroke_uint8_mask(frame: np.ndarray, size: Tuple[int, int], coords_list, dilation: int = 8) -> np.ndarray:
    """Tạo mặt nạ nhị phân uint8 bám sát nét chữ (Canny + Otsu), hỗ trợ cả hộp 4 phần tử và 5 phần tử SAM2, có fallback an toàn về hộp nếu vùng chọn đồng màu."""
    h_frame, w_frame = (frame.shape[:2] if frame is not None else (int(size[0]), int(size[1])))
    mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    if not coords_list:
        return mask
    if frame is None:
        return _build_box_uint8_mask((h_frame, w_frame), coords_list, dilation=dilation, frame=None)

    for coords in coords_list:
        if coords is None or isinstance(coords, dict):
            continue
        if len(coords) == 5 and coords[4] is not None:
            xmin, xmax, ymin, ymax, sam2_data = coords
            binary_mask = _unpack_sam2_mask(sam2_data, h_frame, w_frame)
            pad = max(int(dilation), calc_adaptive_padding(abs(ymax - ymin)))
            if pad > 0:
                kernel = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
                dilated = cv2.dilate((binary_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
                mask[dilated > 0] = 255
            else:
                mask[binary_mask > 0] = 255
            continue

        if len(coords) < 4:
            continue
        xmin, xmax, ymin, ymax = [int(v) for v in coords[:4]]
        pad = max(int(dilation), calc_adaptive_padding(abs(ymax - ymin)))
        x1 = max(0, xmin - pad)
        y1 = max(0, ymin - pad)
        x2 = min(w_frame, xmax + pad)
        y2 = min(h_frame, ymax + pad)
        if x2 <= x1 or y2 <= y1:
            continue


        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        crop_mask = np.zeros_like(gray)

        # Chỉ phân tách Canny + Otsu khi vùng chọn có độ biến thiên sáng tối thực sự
        if float(gray.std()) >= 2.0:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 120)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(otsu) > 127:
                otsu = cv2.bitwise_not(otsu)
            combined = cv2.bitwise_or(edges, otsu)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if area < 3 and (cw < 2 or ch < 2):
                    continue
                cv2.drawContours(crop_mask, [cnt], -1, 255, thickness=-1)

        # Fallback an toàn: nếu không bắt được contour nào trên vùng chọn hợp lệ, tô kín hộp vùng chọn
        if not np.any(crop_mask > 0):
            bx1 = max(0, xmin - x1)
            by1 = max(0, ymin - y1)
            bx2 = min(x2 - x1, xmax - x1)
            by2 = min(y2 - y1, ymax - y1)
            if bx2 > bx1 and by2 > by1:
                crop_mask[by1:by2, bx1:bx2] = 255
            else:
                crop_mask[:, :] = 255

        if dilation > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1, dilation * 2 + 1))
            crop_mask = cv2.dilate(crop_mask, kernel_dilate)

        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], crop_mask)

    return mask


def _apply_freehand_strokes_uint8(mask: np.ndarray, strokes_list, dilation: int = 8) -> np.ndarray:
    """Áp dụng tuần tự các nét cọ (cộng 255) và nét tẩy (trừ về 0) lên mặt nạ uint8."""
    if not strokes_list:
        return mask
    h, w = mask.shape[:2]
    for stroke in strokes_list:
        if stroke is None:
            continue
        if isinstance(stroke, dict):
            pts = stroke.get('points', [])
            r_val = stroke.get('radius', 0.02)
            is_eraser = bool(stroke.get('is_eraser', False))
        else:
            pts = stroke
            r_val = 0.02
            is_eraser = False

        if not pts:
            continue

        if isinstance(r_val, (int, float)) and 0 < r_val <= 1.0:
            base_thick = max(4, int(r_val * w * 2))
        else:
            base_thick = max(4, int(r_val))

        thickness = base_thick if is_eraser else (base_thick + max(0, int(dilation)))
        stroke_layer = np.zeros((h, w), dtype=np.uint8)

        if len(pts) == 1:
            px = int(pts[0][0] * w) if pts[0][0] <= 1.0 else int(pts[0][0])
            py = int(pts[0][1] * h) if pts[0][1] <= 1.0 else int(pts[0][1])
            cv2.circle(stroke_layer, (px, py), max(1, thickness // 2), 255, thickness=-1)
        else:
            int_pts = []
            for p in pts:
                px = int(p[0] * w) if p[0] <= 1.0 else int(p[0])
                py = int(p[1] * h) if p[1] <= 1.0 else int(p[1])
                int_pts.append([px, py])
            int_pts_arr = np.array(int_pts, dtype=np.int32)
            cv2.polylines(stroke_layer, [int_pts_arr], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
            for pt in int_pts_arr:
                cv2.circle(stroke_layer, (int(pt[0]), int(pt[1])), max(1, thickness // 2), 255, thickness=-1)

        if is_eraser:
            mask[stroke_layer > 0] = 0
        else:
            mask[stroke_layer > 0] = 255

    return mask


def _finalize_float_mask(mask: np.ndarray, feather_pixels: int = 8) -> np.ndarray:
    """Làm mềm biên Gaussian và chuẩn hóa mặt nạ sang float32 trong khoảng [0.0, 1.0]."""
    if feather_pixels > 0 and np.any(mask > 0):
        ksize = 2 * int(feather_pixels) + 1
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        return blurred.astype(np.float32) / 255.0
    return (mask > 0).astype(np.float32)


def create_mask(size, coords_list, dilation=8, feather_pixels=8, frame=None):
    """Tạo mặt nạ hộp vùng chọn (hoặc SAM2 5-tuple) trả về float32 [0.0, 1.0]."""
    if _is_stroke_dict_list(coords_list):
        return create_freehand_stroke_mask(size, coords_list, dilation=dilation, feather_pixels=feather_pixels)
    mask = _build_box_uint8_mask(size, coords_list, dilation=dilation, frame=frame)
    return _finalize_float_mask(mask, feather_pixels=feather_pixels)


def create_freehand_stroke_mask(size, strokes_list, dilation=8, feather_pixels=8):
    """Tạo mặt nạ từ danh sách nét cọ và cục tẩy tự do [{'radius': r, 'points': [...], 'is_eraser': bool}]."""
    h, w = int(size[0]), int(size[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = _apply_freehand_strokes_uint8(mask, strokes_list, dilation=dilation)
    return _finalize_float_mask(mask, feather_pixels=feather_pixels)


def create_text_stroke_mask(frame: np.ndarray, size: Tuple[int, int], coords_list, dilation: int = 8, feather_pixels: int = 8) -> np.ndarray:
    """Tạo mặt nạ cấp độ nét chữ (Canny + Otsu) từ danh sách hộp tọa độ, có fallback an toàn."""
    if _is_stroke_dict_list(coords_list):
        return create_freehand_stroke_mask(size, coords_list, dilation=dilation, feather_pixels=feather_pixels)
    mask = _build_text_stroke_uint8_mask(frame, size, coords_list, dilation=dilation)
    return _finalize_float_mask(mask, feather_pixels=feather_pixels)


def create_stroke_mask(frame, size, coords_or_strokes, dilation=8, feather_pixels=8):
    """Bộ điều phối tương thích ngược: tự động phân biệt danh sách nét cọ tự do (dict) và danh sách hộp tọa độ (tuple/list)."""
    if _is_stroke_dict_list(coords_or_strokes):
        return create_freehand_stroke_mask(size, coords_or_strokes, dilation=dilation, feather_pixels=feather_pixels)
    return create_text_stroke_mask(frame, size, coords_or_strokes, dilation=dilation, feather_pixels=feather_pixels)


def create_combined_mask(size, coords_list=None, strokes_list=None, dilation=8, feather_pixels=8, frame=None, mask_type=None):
    """
    Tạo mặt nạ tổng hợp thống nhất cho cả Xem Trước và Xuất Video:
    1. Tạo lớp nền từ coords_list theo mask_type ('stroke' hoặc 'box', hỗ trợ cả 4-tuple và 5-tuple SAM2)
    2. Áp dụng tuần tự strokes_list (nét cọ cộng vào mặt nạ, cục tẩy xóa khỏi mặt nạ)
    3. Làm mềm biên Gaussian và trả về float32 [0.0, 1.0]
    """
    h, w = int(size[0]), int(size[1])
    if mask_type is None:
        mask_type = getattr(config, 'maskType', None).value if getattr(config, 'maskType', None) else "box"

    if coords_list:
        if _is_stroke_dict_list(coords_list) and not strokes_list:
            strokes_list = coords_list
            mask = np.zeros((h, w), dtype=np.uint8)
        elif str(mask_type).lower() == "stroke" and frame is not None:
            mask = _build_text_stroke_uint8_mask(frame, (h, w), coords_list, dilation=dilation)
        else:
            mask = _build_box_uint8_mask((h, w), coords_list, dilation=dilation, frame=frame)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    if strokes_list:
        mask = _apply_freehand_strokes_uint8(mask, strokes_list, dilation=dilation)

    return _finalize_float_mask(mask, feather_pixels=feather_pixels)

def get_inpaint_area_by_mask(W, H, h, mask, multiple=1):
    """
    获取字幕去除区域，根据mask来确定需要填补的区域和高度，
    并根据模型要求调整区域大小为指定倍数
    
    Args:
        W: 图像宽度
        H: 图像高度
        h: 检测区域高度
        mask: 遮罩图像
        multiple: 区域尺寸需要满足的倍数，默认为1
    
    Returns:
        调整后的绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    """
    # 存储绘画区域的列表
    inpaint_area = []
    
    # 如果mask全为0，直接返回空列表
    if np.all(mask == 0):
        return inpaint_area
    
    # 使用连通组件分析找出mask中的所有孤岛
    # 首先确保mask是二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.squeeze(-1)
    
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 跳过背景（标签0）
    island_info = []
    for i in range(1, num_labels):
        # 获取当前孤岛的统计信息
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 忽略太小的区域（可能是噪点）
        if area < 10:
            continue
        
        # 保存孤岛信息：顶部y坐标，底部y坐标，中心点y坐标，面积，标签
        center_y = int(centroids[i][1])
        island_info.append((y, y + height, center_y, area, i, x, x + w))
    
    # 如果没有有效孤岛，返回空列表
    if not island_info:
        return inpaint_area
    
    # 按中心点y坐标排序孤岛
    island_info.sort(key=lambda x: x[2])
    
    # 尝试合并孤岛
    merged_islands = []
    current_group = [island_info[0]]
    
    for i in range(1, len(island_info)):
        # 当前组的范围
        min_y = min([island[0] for island in current_group])
        max_y = max([island[1] for island in current_group])
        
        # 当前孤岛
        top_y, bottom_y, center_y, _, _, _, _ = island_info[i]
        
        # 计算如果添加当前孤岛，新组的范围
        new_min_y = min(min_y, top_y)
        new_max_y = max(max_y, bottom_y)
        
        # 检查是否有mask连接当前组和新孤岛
        has_connection = False
        if max_y < top_y:  # 只有当前组在新孤岛上方时才需要检查连接
            # 检查两个区域之间是否有mask像素
            middle_region = binary_mask[max_y:top_y, :]
            if np.any(middle_region > 0):
                has_connection = True
        else:  # 重叠或相邻
            has_connection = True
        
        # 检查合并后的高度是否在h范围内，并且有连接
        if new_max_y - new_min_y <= h and has_connection:
            # 可以合并
            current_group.append(island_info[i])
        else:
            # 无法合并，保存当前组并开始新组
            merged_islands.append(current_group)
            current_group = [island_info[i]]
    
    # 添加最后一个组
    merged_islands.append(current_group)
    
    # 为每个合并后的组创建区域
    for group in merged_islands:
        # 获取组内所有孤岛的范围
        min_y = min([island[0] for island in group])
        max_y = max([island[1] for island in group])
        
        # 计算组的中心点
        center_y = sum([island[2] for island in group]) // len(group)
        
        # 确保区域高度精确等于h
        half_h = h // 2
        
        # 从中心点向上下扩展，确保高度为h
        ymin = max(0, center_y - half_h)
        ymax = ymin + h  # 确保高度精确等于h
        
        # 如果超出图像底部，从底部向上调整
        if ymax > H:
            ymax = H
            ymin = max(0, H - h)  # 确保高度为h
        
        # 检查是否包含了所有孤岛
        if ymin > min_y or ymax < max_y:
            # 如果区域不能完全包含所有孤岛，尝试调整位置但保持高度为h
            if max_y - min_y <= h:
                # 孤岛总高度不超过h，可以调整位置使其完全包含
                ymin = min_y
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
            else:
                # 孤岛总高度超过h，无法完全包含，优先包含中心区域
                # 计算孤岛的中心
                island_center = (min_y + max_y) // 2
                ymin = max(0, island_center - half_h)
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
        
        # Tối ưu: Lấy Bounding Box nhỏ nhất bao quanh các phụ đề theo chiều ngang
        min_x = min([island[5] for island in group])
        max_x = max([island[6] for island in group])
        
        # Thêm padding (lề đệm) khoảng 32 pixel
        padding = 32
        xmin = max(0, min_x - padding)
        xmax = min(W, max_x + padding)
        
        # 调整区域大小为指定倍数
        if multiple > 1:
            # --- Height Adjustment ---
            height = ymax - ymin
            remainder = height % multiple
            if remainder != 0:
                adjust_pixels = multiple - remainder
                target_height = height + adjust_pixels
                
                # 优先对称扩展
                center_y = (ymin + ymax) / 2
                ymin = int(center_y - target_height / 2)
                ymax = ymin + target_height
                
                # Boundary checks (shift if out of bounds)
                if ymin < 0:
                    ymax += (0 - ymin)
                    ymin = 0
                if ymax > H:
                    ymin -= (ymax - H)
                    ymax = H
                    # Ensure it doesn't go below 0 again. If it does, we can't satisfy the multiple constraint within the image bounds.
                    if ymin < 0:
                        ymin = 0

            # --- Width Adjustment ---
            width = xmax - xmin
            remainder_w = width % multiple
            if remainder_w != 0:
                adjust_pixels_w = multiple - remainder_w
                target_width = width + adjust_pixels_w  # Expand width instead of shrinking to be safe
                
                center_x = (xmin + xmax) / 2
                xmin = int(center_x - target_width / 2)
                xmax = xmin + target_width
                
                # Boundary checks (shift if out of bounds)
                if xmin < 0:
                    xmax += (0 - xmin)
                    xmin = 0
                if xmax > W:
                    xmin -= (xmax - W)
                    xmax = W
                    if xmin < 0:
                        xmin = 0
        
        # 将该区域添加到列表中，格式为(ymin, ymax, xmin, xmax)
        area = (int(ymin), int(ymax), int(xmin), int(xmax))
        if area not in inpaint_area:
            inpaint_area.append(area)
    
    return inpaint_area  # 返回绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    
def expand_frame_ranges(frame_ranges, backward_frame_count, forward_frame_count):
    """
    扩展帧区间列表，向前和向后扩展指定的帧数，并确保区间连续性
    
    Args:
        frame_ranges: 帧区间列表，格式为[(start1, end1), (start2, end2), ...]
        backward_frame_count: 向前扩展的帧数
        forward_frame_count: 向后扩展的帧数
        
    Returns:
        扩展后的帧区间列表，保证连续性
    """
    if not frame_ranges:
        return []
    
    # 按起始帧排序
    sorted_ranges = sorted(frame_ranges)
    expanded_ranges = []
    
    for i, (start, end) in enumerate(sorted_ranges):
        # 向前扩展，但不能小于1
        new_start = max(1, start - backward_frame_count)
        
        # 向后扩展
        new_end = end + forward_frame_count
        
        # 检查是否与下一个区间重叠
        if i < len(sorted_ranges) - 1:
            next_start = sorted_ranges[i + 1][0]
            
            # 如果扩展后的结束帧超过了下一个区间的起始帧
            if new_end >= next_start:
                # 计算中点
                mid_point = (end + next_start) // 2
                
                # 如果区间是连续的(相差1)，则对半平分
                if next_start - end == 1:
                    new_end = end  # 保持原结束帧
                else:
                    # 非连续区间，限制扩展到下一个区间起始帧减去backward_frame_count
                    max_expand = next_start - 1  # 确保不会与下一个区间重叠
                    new_end = min(new_end, max_expand)
        
        # 确保与前一个区间不重叠
        if expanded_ranges:
            prev_end = expanded_ranges[-1][1]
            if new_start <= prev_end:
                # 如果新区间的开始小于等于前一个区间的结束，调整开始位置
                new_start = prev_end + 1
        
        # 确保区间有效（开始不大于结束）
        if new_start <= new_end:
            expanded_ranges.append((new_start, new_end))
        else:
            # 如果调整后区间无效，保留原始区间
            expanded_ranges.append((start, end))
    
    return expanded_ranges

def is_frame_number_in_ab_sections(frame_no, ab_sections):
    """
    检查给定的帧号是否在指定的A/B区间内。

    Args:
        frame_no: 要检查的帧号
        ab_sections: 包含A/B区间的列表，格式为[range(start, end), ...]

    Returns:
        如果帧号在A/B区间内，返回True；否则返回False。
    """
    if ab_sections is None:
        return True
    if len(ab_sections) <= 0:
        return True
    for section in ab_sections:
        if frame_no in section:
            return True
    return False

def blend_inpaint(original_crop: np.ndarray, inpainted_crop: np.ndarray, mask_crop: np.ndarray, 
                 method: str = 'feather', feather_pixels: int = 8) -> np.ndarray:
    """
    Hòa trộn vùng ảnh đã xóa phụ đề vào khung hình gốc sử dụng Gaussian Feather Blending hoặc Poisson Blending.
    
    Args:
        original_crop: Vùng ảnh gốc (BGR)
        inpainted_crop: Vùng ảnh đã được phục chế/inpaint (BGR)
        mask_crop: Mặt nạ chữ (dạng float32 trong khoảng [0, 1] hoặc uint8 trong khoảng [0, 255])
        method: Phương pháp hòa trộn ('feather' hoặc 'poisson')
        feather_pixels: Bán kính làm mềm biên (áp dụng cho feather blending)
    """
    # Đảm bảo mask là kênh đơn (2D)
    if mask_crop.ndim == 3:
        if mask_crop.shape[2] == 1:
            mask_crop = mask_crop[:, :, 0]
        else:
            mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)
            
    # Nếu mask rỗng (không có pixel nào > 0), trả về vùng ảnh gốc
    if mask_crop.max() == 0:
        return original_crop.copy()
        
    # Áp dụng Mean Color Transfer để đồng bộ hóa ánh sáng (Luminance Matching)
    # Tính giá trị màu trung bình của vùng viền quanh mặt nạ ở cả 2 ảnh
    # Viền mặt nạ được xác định bằng cách lùi (dilate) mặt nạ rồi trừ đi mặt nạ gốc
    bin_mask_for_border = (mask_crop > 0).astype(np.uint8)
    kernel_border = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated_for_border = cv2.dilate(bin_mask_for_border, kernel_border)
    border_mask = cv2.subtract(dilated_for_border, bin_mask_for_border)
    
    if np.any(border_mask > 0):
        mean_orig = cv2.mean(original_crop, mask=border_mask)[:3]
        mean_inpaint = cv2.mean(inpainted_crop, mask=border_mask)[:3]
        
        # Tính khoảng chênh lệch màu sắc (shift)
        shift = np.array(mean_orig) - np.array(mean_inpaint)
        
        # Dịch chuyển toàn bộ màu của vùng inpaint (chỉ dịch chuyển phần trong mask để tránh méo màu viền ngoài)
        shift_img = np.zeros_like(inpainted_crop, dtype=np.float32)
        shift_img[:, :] = shift
        
        # Áp dụng dịch chuyển màu
        inpainted_crop_float = inpainted_crop.astype(np.float32) + shift_img
        inpainted_crop = np.clip(inpainted_crop_float, 0, 255).astype(np.uint8)
        
    if method == 'poisson':
        # Chuẩn bị mặt nạ nhị phân uint8 cho seamlessClone (giá trị phải là 0 hoặc 255)
        # Tiến hành nở mask (dilation) nhẹ 2px để biên hòa trộn Poisson có vùng chuyển tiếp ổn định
        bin_mask = (mask_crop > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_mask = cv2.dilate(bin_mask, kernel)
        
        # Lấy bounding box của mask để tính toán tâm chính xác
        x, y, w_mask, h_mask = cv2.boundingRect(bin_mask)
        if w_mask == 0 or h_mask == 0:
            return original_crop.copy()
            
        # Tâm của vùng cần clone phải là tâm của bounding box mask, 
        # nếu dùng (w//2, h//2) OpenCV sẽ tự động dịch chuyển sai lệch vị trí
        center = (x + w_mask // 2, y + h_mask // 2)
        
        try:
            # Cờ cv2.NORMAL_CLONE giúp thay thế hoàn toàn gradient chữ bằng gradient của vùng inpaint
            blended = cv2.seamlessClone(
                inpainted_crop,
                original_crop,
                bin_mask,
                center,
                cv2.NORMAL_CLONE
            )
            return blended
        except Exception as exc:
            # Nếu xảy ra lỗi (ví dụ vùng biên quá sát mép ảnh), tự động fallback về feather blending
            pass

    # Mặc định/Fallback: Gaussian Feather Blending
    # Chuẩn hóa mask về dạng float32 trong khoảng [0.0, 1.0] nếu chưa chuẩn hóa
    if mask_crop.dtype != np.float32:
        mask_float = mask_crop.astype(np.float32)
        if mask_float.max() > 1.0:
            mask_float /= 255.0
    else:
        mask_float = mask_crop.copy()
        
    if feather_pixels > 0:
        k = feather_pixels * 2 + 1
        # Dilate gently before blur to prevent the core mask from shrinking and bleeding original text
        dilate_k = feather_pixels // 2 * 2 + 1
        if dilate_k >= 3:
            kernel = np.ones((dilate_k, dilate_k), np.uint8)
            soft = cv2.dilate(mask_float, kernel)
        else:
            soft = mask_float
        soft = cv2.GaussianBlur(soft, (k, k), 0)
    else:
        soft = mask_float
        
    if soft.ndim == 2:
        soft = soft[..., np.newaxis]
        
    out = inpainted_crop.astype(np.float32) * soft + original_crop.astype(np.float32) * (1.0 - soft)
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_temporal_smoothing(original_frames: List[np.ndarray], inpainted_frames: List[np.ndarray], 
                             mask: np.ndarray, radius: int = 2, motion_threshold: float = 8.0) -> List[np.ndarray]:
    """
    Áp dụng bộ lọc làm mịn thời gian thích ứng (motion-adaptive temporal smoothing) 
    trên các vùng đã xóa (inpaint) để khử nhấp nháy (flickering).
    """
    if len(inpainted_frames) <= 1 or radius <= 0:
        return inpainted_frames
        
    # Lấy kênh đơn của mask
    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask
        
    if mask_2d.max() == 0:
        return inpainted_frames
        
    # Xác định tọa độ bounding box của mask để crop tối ưu hiệu năng
    ys, xs = np.where(mask_2d > 0)[:2]
    if len(ys) == 0 or len(xs) == 0:
        return inpainted_frames
        
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    
    # Mở rộng nhẹ biên crop để tính toán chuyển động xung quanh chính xác hơn
    h, w = mask_2d.shape[:2]
    pad = 8
    y1_pad = max(0, y1 - pad)
    y2_pad = min(h, y2 + pad)
    x1_pad = max(0, x1 - pad)
    x2_pad = min(w, x2 + pad)
    
    N = len(inpainted_frames)
    smoothed_frames = [f.copy() for f in inpainted_frames]
    
    # Trích xuất các vùng crop gốc và inpaint để xử lý cho nhanh
    orig_crops = [f[y1_pad:y2_pad, x1_pad:x2_pad].astype(np.float32) for f in original_frames]
    inpaint_crops = [f[y1_pad:y2_pad, x1_pad:x2_pad].astype(np.float32) for f in inpainted_frames]
    raw_mask_crop = mask_2d[y1_pad:y2_pad, x1_pad:x2_pad].astype(np.float32)[..., np.newaxis]
    if raw_mask_crop.max() > 1.0:
        mask_crop = np.clip(raw_mask_crop / 255.0, 0.0, 1.0)
    else:
        mask_crop = np.clip(raw_mask_crop, 0.0, 1.0)
    
    for t in range(N):
        current_orig = orig_crops[t]
        
        weighted_sum = np.zeros_like(inpaint_crops[t])
        total_weight = 0.0
        
        # Duyệt qua các frame lân cận trong bán kính bán kính
        for d in range(-radius, radius + 1):
            neighbor_idx = t + d
            if neighbor_idx < 0 or neighbor_idx >= N:
                continue
                
            # Tính sai lệch chuyển động (MAD - Mean Absolute Difference) giữa frame gốc t và frame gốc lân cận
            motion_err = np.mean(np.abs(current_orig - orig_crops[neighbor_idx]))
            
            # Nếu sai lệch nhỏ hơn ngưỡng, chấp nhận hòa trộn vùng inpaint
            if motion_err < motion_threshold:
                # Trọng số phụ thuộc vào độ tương đồng (Gaussian weight) và khoảng cách thời gian
                weight = np.exp(-(motion_err ** 2) / 32.0)
                # Giảm nhẹ trọng số của các frame ở xa theo thời gian
                weight *= np.exp(-(d ** 2) / (2.0 * (radius ** 2)))
                
                weighted_sum += inpaint_crops[neighbor_idx] * weight
                total_weight += weight
                
        if total_weight > 0:
            smoothed_crop = weighted_sum / total_weight
            # Chỉ thay thế các pixel nằm trong vùng mask
            blended_crop = smoothed_crop * mask_crop + inpaint_crops[t] * (1.0 - mask_crop)
            smoothed_frames[t][y1_pad:y2_pad, x1_pad:x2_pad] = np.clip(blended_crop, 0, 255).astype(np.uint8)
            
    return smoothed_frames


def apply_temporal_mask_padding(masks: List[np.ndarray], pad_frames: int) -> List[np.ndarray]:
    """
    Áp dụng bù trừ khung hình mờ (Fade padding) bằng cách mở rộng mặt nạ 
    theo trục thời gian (lùi về trước và kéo về sau) N frames.
    Đặc biệt hữu ích để xử lý các phụ đề có hiệu ứng fade-in / fade-out mà OCR thường bỏ sót.
    """
    if pad_frames <= 0 or len(masks) <= 1:
        return masks
        
    padded_masks = []
    # Thu thập trước các mask hợp lệ thành danh sách để xử lý numpy
    valid_mask_indices = [i for i, m in enumerate(masks) if m is not None and isinstance(m, np.ndarray)]
    if not valid_mask_indices:
        return masks
        
    # Stack các mask để tính toán np.max theo trục thời gian (trục 0) nhanh hơn
    # Tuy nhiên, nếu kích thước quá lớn, ta có thể làm tuần tự
    for t in range(len(masks)):
        if masks[t] is None:
            padded_masks.append(None)
            continue
            
        start = max(0, t - pad_frames)
        end = min(len(masks), t + pad_frames + 1)
        
        window_masks = [masks[i] for i in range(start, end) if masks[i] is not None]
        if window_masks:
            # Lấy giá trị pixel lớn nhất trong cửa sổ thời gian
            max_mask = np.max(window_masks, axis=0)
            
            # (Tùy chọn tương lai) Giảm opacity/alpha (mờ dần) ở phần rìa padding nếu cần
            # Ở đây chúng ta giữ nguyên 100% mask để xóa sạch
            
            padded_masks.append(max_mask)
        else:
            padded_masks.append(masks[t])
            
    return padded_masks

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
