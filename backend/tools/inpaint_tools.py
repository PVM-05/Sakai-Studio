import multiprocessing
import cv2
import numpy as np
from typing import List, Tuple

from backend.config import config

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

def create_mask(size, coords_list, feather_pixels=8):
    # Tạo mặt nạ nhị phân gốc
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # Chiều cao thực tế của chữ
            height = ymax - ymin
            # Giãn nở thích ứng: bằng 15% chiều cao chữ, tối thiểu 3px, tối đa 25px
            adaptive_deviation = int(np.clip(height * 0.15, 3, 25))
            
            # Phóng to ô để tránh bị sót viền chữ
            x1 = max(0, xmin - adaptive_deviation)
            y1 = max(0, ymin - adaptive_deviation)
            x2 = min(size[1], xmax + adaptive_deviation)
            y2 = min(size[0], ymax + adaptive_deviation)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    
    # Gaussian Alpha Feathering: Làm mịn mượt biên mask
    if feather_pixels > 0 and np.any(mask > 0):
        # Đảm bảo kernel size là số lẻ
        ksize = 2 * feather_pixels + 1
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        # Chuyển đổi về dạng float32 trong khoảng [0.0, 1.0] để nhân trộn điểm ảnh mềm
        float_mask = blurred.astype(np.float32) / 255.0
    else:
        float_mask = (mask > 0).astype(np.float32)
        
    return float_mask

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
        island_info.append((y, y + height, center_y, area, i))
    
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
        top_y, bottom_y, center_y, _, _ = island_info[i]
        
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
        
        # 使用完整宽度
        xmin = 0
        xmax = W
        
        # 调整区域大小为指定倍数
        if multiple > 1:
            # 计算区域高度
            height = ymax - ymin
            # 计算需要调整的高度，使其成为multiple的倍数
            remainder = height % multiple
            
            if remainder != 0:
                # 需要调整的像素数
                adjust_pixels = multiple - remainder
                
                # 计算区域中心点
                center_y = (ymin + ymax) / 2
                
                # 优先对称扩展
                if ymin - adjust_pixels/2 >= 0 and ymax + adjust_pixels/2 <= H:
                    # 对称扩展
                    ymin = int(center_y - height/2 - adjust_pixels/2)
                    ymax = int(center_y + height/2 + adjust_pixels/2)
                # 如果对称扩展会超出边界，尝试对称缩小
                elif height > multiple:  # 确保缩小后高度至少为multiple
                    # 对称缩小
                    ymin = int(center_y - (height - remainder)/2)
                    ymax = int(center_y + (height - remainder)/2)
                # 如果无法对称调整，则尝试单边调整
                else:
                    # 向下扩展
                    if ymax + adjust_pixels <= H:
                        ymax += adjust_pixels
                    # 向上扩展
                    elif ymin - adjust_pixels >= 0:
                        ymin -= adjust_pixels
                    # 如果都不行，则尝试缩小区域
                    elif height > multiple:
                        ymax = ymin + height - remainder
            
            # 调整宽度，确保是multiple的倍数
            width = xmax - xmin
            remainder_w = width % multiple
            
            if remainder_w != 0:
                # 需要调整的像素数
                adjust_pixels_w = multiple - remainder_w
                
                # 计算中心点，对称缩小
                center_x = (xmin + xmax) / 2
                xmin = int(center_x - (width - remainder_w)/2)
                xmax = int(center_x + (width - remainder_w)/2)
        
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
        
    if method == 'poisson':
        # Chuẩn bị mặt nạ nhị phân uint8 cho seamlessClone (giá trị phải là 0 hoặc 255)
        # Tiến hành nở mask (dilation) nhẹ 2px để biên hòa trộn Poisson có vùng chuyển tiếp ổn định
        bin_mask = (mask_crop > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_mask = cv2.dilate(bin_mask, kernel)
        
        # Tâm của vùng crop là điểm đích để nhân bản
        h, w = original_crop.shape[:2]
        center = (w // 2, h // 2)
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
        soft = cv2.GaussianBlur(mask_float, (k, k), 0)
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
    mask_crop = mask_2d[y1_pad:y2_pad, x1_pad:x2_pad][..., np.newaxis] / 255.0
    
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

def create_stroke_mask(frame: np.ndarray, size: Tuple[int, int], coords_list: List[Tuple[int, int, int, int]], feather_pixels: int = 8) -> np.ndarray:
    """
    Tạo mặt nạ ở cấp độ nét chữ (stroke-level mask) sử dụng Contour và Canny Edge Detection.
    Giúp xóa phụ đề sạch nhất mà không bị ghost.
    """
    if frame is None or not coords_list:
        return create_mask(size, coords_list, feather_pixels)
        
    h_frame, w_frame = frame.shape[:2]
    mask = np.zeros((h_frame, w_frame), dtype="uint8")
    
    for coords in coords_list:
        xmin, xmax, ymin, ymax = coords
        
        # Mở rộng nhẹ hộp chữ nhật gốc để lấy biên chữ đầy đủ
        pad = 4
        x1 = max(0, xmin - pad)
        y1 = max(0, ymin - pad)
        x2 = min(w_frame, xmax + pad)
        y2 = min(h_frame, ymax + pad)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Chuyển sang grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng bộ lọc Gaussian Blur để giảm nhiễu trước khi dò biên
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Tìm biên chữ bằng Canny
        edges = cv2.Canny(blurred, 30, 120)
        
        # Nối các nét biên chữ bằng toán tử Close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Tìm các contours đường biên
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crop_mask = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            # Loại bỏ nhiễu cực nhỏ
            if area < 3 and (w < 2 or h < 2):
                continue
            # Vẽ điền đầy contour nét chữ
            cv2.drawContours(crop_mask, [cnt], -1, 255, thickness=-1)
            
        # Giãn nở thích ứng theo chiều cao chữ để bao trùm viền/bóng chữ
        box_height = y2 - y1
        dilate_px = max(2, int(np.clip(box_height * 0.10, 2, 10)))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        crop_mask = cv2.dilate(crop_mask, kernel_dilate)
        
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], crop_mask)
        
    # Gaussian Alpha Feathering làm mềm biên mặt nạ
    if feather_pixels > 0 and np.any(mask > 0):
        ksize = 2 * feather_pixels + 1
        blurred_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        float_mask = blurred_mask.astype(np.float32) / 255.0
    else:
        float_mask = (mask > 0).astype(np.float32)
        
    return float_mask

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
