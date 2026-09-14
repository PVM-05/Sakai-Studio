import os
import gc
from typing import Union, List
import torch
import cv2
import numpy as np
from PIL import Image
from src.ai_engines.inpaint.utils.lama_util import prepare_img_and_mask, get_image, pad_img_to_modulo
from src.core.config import config
from src.core.tools.inpaint_tools import get_inpaint_area_by_mask, blend_inpaint


class LamaInpaint:
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path='big-lama.pt') -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

    def inpaint(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]):
        is_bgr = isinstance(image, np.ndarray)
        if is_bgr:
            orig_height, orig_width = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            orig_height, orig_width = np.array(image).shape[:2]
        image, mask = prepare_img_and_mask(image, mask, self.device)
        # Đảm bảo mask được binarize thành 0 và 1 (tránh bị chia 255 ở prepare_img_and_mask làm nhạt mask)
        mask = (mask > 0).float()
        
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        # LAMA uses Fast Fourier Convolution (FFC). cuFFT in FP16 requires power-of-2 dimensions.
        # To support arbitrary crop sizes, we must use FP32.
        dtype = torch.float32
            
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=dtype):
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cur_res[:orig_height, :orig_width]
            if is_bgr:
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        return cur_res

    def _inpaint_batch(self, images: List[np.ndarray], masks: List[np.ndarray]):
        """批量推理：将多帧分小批次送入 GPU，避免单次推理过大导致卡死"""
        if len(images) == 1:
            return [self.inpaint(images[0], masks[0])]

        orig_height, orig_width = images[0].shape[:2]
        
        # Tính mini_batch_size động theo VRAM khả dụng thay vì hardcode 4
        from src.core.tools.hardware_accelerator import HardwareAccelerator
        vram_mb = HardwareAccelerator.instance().get_available_vram_mb()
        if vram_mb > 0:
            # LAMA cần nhiều RAM hơn do FFC, ước lượng an toàn ~16 bytes per pixel cho toàn bộ tensor
            bytes_per_frame = orig_width * orig_height * 16
            mini_batch_size = max(1, min(4, int(vram_mb * 1024 * 1024 / bytes_per_frame)))
        else:
            mini_batch_size = 4
            
        results = [None] * len(images)
        for start in range(0, len(images), mini_batch_size):
            end = min(start + mini_batch_size, len(images))
            batch_imgs = []
            batch_masks = []
            for i in range(start, end):
                rgb_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                batch_imgs.append(get_image(rgb_img))
                batch_masks.append(get_image(masks[i]))

            padded_imgs = np.stack([pad_img_to_modulo(img, 8) for img in batch_imgs])
            padded_masks = np.stack([pad_img_to_modulo(m, 8) for m in batch_masks])

            img_tensor = torch.from_numpy(padded_imgs).to(self.device)
            mask_tensor = torch.from_numpy(padded_masks).to(self.device)
            mask_tensor = (mask_tensor > 0) * 1

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            # LAMA uses Fast Fourier Convolution (FFC). cuFFT in FP16 requires power-of-2 dimensions.
            # To support arbitrary crop sizes, we must use FP32.
            dtype = torch.float32

            with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=dtype):
                inpainted = self.model(img_tensor, mask_tensor)
                batch_results = inpainted.permute(0, 2, 3, 1).detach().cpu().numpy()
                batch_results = np.clip(batch_results * 255, 0, 255).astype('uint8')

            for i in range(end - start):
                res = batch_results[i][:orig_height, :orig_width]
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
                results[start + i] = res

            del img_tensor, mask_tensor, padded_imgs, padded_masks
            import gc
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return results

    def __call__(self, input_frames: List[np.ndarray], input_mask):
        """
        :param input_frames: 原视频帧
        :param input_mask: 字幕区域mask (numpy array hoặc list các mask per-frame)
        """
        is_mask_list = isinstance(input_mask, (list, tuple))
        is_dict_mask = isinstance(input_mask, dict)
        
        if is_dict_mask:
            # Support per-frame mask dict or coord dict (fallback for robustness)
            if not input_mask:
                return [f.copy() for f in input_frames]
            is_coord = isinstance(next(iter(input_mask.values())), list)
            from src.core.tools.inpaint_tools import create_mask
            H_ori, W_ori = input_frames[0].shape[:2]
            mask_list = []
            union_mask = np.zeros((H_ori, W_ori, 1), dtype=np.float32)
            frame_nos = sorted(input_mask.keys())
            for i, fno in enumerate(frame_nos):
                if i >= len(input_frames): break
                if is_coord:
                    m = create_mask((H_ori, W_ori), input_mask[fno], frame=input_frames[i])
                else:
                    m = input_mask[fno]
                m_3d = m[:, :, None] if m.ndim == 2 else m
                mask_list.append(m_3d)
                union_mask = np.maximum(union_mask, m_3d)
            calc_mask = union_mask
            input_mask = mask_list
            is_mask_list = True
        elif is_mask_list:
            first_m = input_mask[0]
            ref_mask = first_m[:, :, None] if first_m.ndim == 2 else first_m
            union_mask = np.zeros_like(ref_mask)
            for m in input_mask:
                m_3d = m[:, :, None] if m.ndim == 2 else m
                union_mask = np.maximum(union_mask, m_3d)
            calc_mask = union_mask
        else:
            calc_mask = input_mask[:, :, None] if input_mask.ndim == 2 else input_mask

        H_ori, W_ori = calc_mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, calc_mask)
        # 高分辨率帧存储列表
        frames_hr = [f.copy() for f in input_frames]
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []

        for k in range(len(inpaint_area)):
            # 收集该区域的所有裁剪帧和遮罩
            cropped_frames = []
            cropped_masks = []
            for j in range(len(frames_hr)):
                if is_mask_list:
                    # Đảm bảo không vượt quá index của input_mask nếu list ngắn hơn
                    m_idx = min(j, len(input_mask) - 1)
                    m_j = input_mask[m_idx]
                    cur_mask = m_j[:, :, None] if m_j.ndim == 2 else m_j
                else:
                    cur_mask = calc_mask

                # Cắt chính xác cả chiều cao và chiều rộng theo bounding box để tăng tốc & đồng bộ với STTN
                image_crop = frames_hr[j][inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :]
                mask_crop = cur_mask[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :]
                cropped_frames.append(image_crop)
                cropped_masks.append(mask_crop)

            # 批量推理
            comps[k] = self._inpaint_batch(cropped_frames, cropped_masks)
            del cropped_frames, cropped_masks
            gc.collect()

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                if is_mask_list:
                    m_j = input_mask[j]
                    cur_mask = m_j[:, :, None] if m_j.ndim == 2 else m_j
                else:
                    cur_mask = calc_mask

                for k in range(len(inpaint_area)):
                    mask_area = cur_mask[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :]
                    comp = comps[k][j]
                    
                    # Xác định phương pháp hòa trộn dựa trên cấu hình
                    use_poisson = False
                    if hasattr(config, 'poissonBlending'):
                        use_poisson = config.poissonBlending.value
                    method = 'poisson' if use_poisson else 'feather'
                    
                    original_crop = frame[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :]
                    blended = blend_inpaint(original_crop, comp, mask_area, method=method, feather_pixels=8)
                    frame[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :] = blended
                inpainted_frames.append(frame)
        else:
            # 无需处理的区域，返回原始帧
            inpainted_frames = frames_hr

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return inpainted_frames
