import cv2
import numpy as np
import config

class ImageNormalizer:
    """Normalizes return image to match product image lighting/color distribution.
    
    Pipeline: Color Transfer → CLAHE → Retinex → Local Contrast Normalization
    """
    
    def normalize(self, image: np.ndarray, reference: np.ndarray, 
                  image_mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
        """
        Normalize image to match reference's lighting/color distribution.
        Only uses pixels within the object masks for statistics.
        
        Args:
            image: BGR return image to normalize
            reference: BGR product image (target distribution)
            image_mask: binary mask for return image object
            reference_mask: binary mask for product image object
            
        Returns:
            Normalized BGR image (same size as input)
        """
        img = self.color_transfer(image, reference, image_mask, reference_mask)
        img = self.apply_clahe(img)
        return img
    
    def color_transfer(self, source: np.ndarray, target: np.ndarray, 
                       source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        result_lab = np.copy(source_lab)
        source_valid = source_mask > 0
        target_valid = target_mask > 0
        
        if not np.any(source_valid) or not np.any(target_valid):
            return source
            
        min_clip, max_clip = config.COLOR_TRANSFER_STD_RATIO_CLIP
        
        for c in range(3):
            s_chan = source_lab[..., c]
            t_chan = target_lab[..., c]
            
            mean_s = np.mean(s_chan[source_valid])
            std_s = np.std(s_chan[source_valid]) + 1e-6
            
            mean_t = np.mean(t_chan[target_valid])
            std_t = np.std(t_chan[target_valid]) + 1e-6
            
            std_ratio = np.clip(std_t / std_s, min_clip, max_clip)
            
            res_c = (s_chan - mean_s) * std_ratio + mean_t
            result_lab[..., c] = res_c
            
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_SIZE)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    def remove_illumination(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[..., 0].astype(np.float32)
        
        valid = mask > 0
        if not np.any(valid):
            return image
            
        mean_fg = np.mean(l[valid])
        l_filled = np.copy(l)
        l_filled[~valid] = mean_fg
        
        illumination = cv2.GaussianBlur(l_filled, (0, 0), sigmaX=config.RETINEX_SIGMA)
        reflectance = l / (illumination + 1e-6)
        
        ref_valid = reflectance[valid]
        if ref_valid.size > 0:
            ref_min = np.min(ref_valid)
            ref_max = np.max(ref_valid)
            if ref_max > ref_min:
                reflectance = (reflectance - ref_min) / (ref_max - ref_min) * 255.0
            
        l_out = np.clip(reflectance, 0, 255).astype(np.uint8)
        l_out[~valid] = 0
        
        lab[..., 0] = l_out
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def local_contrast_norm(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bgr = image.astype(np.float32)
        valid = mask > 0
        
        if not np.any(valid):
            return image
            
        mean_fg = np.mean(bgr[valid], axis=0)
        bgr_filled = np.copy(bgr)
        bgr_filled[~valid] = mean_fg
        
        ksize = config.LOCAL_NORM_KERNEL_SIZE
        if ksize % 2 == 0:
            ksize += 1
            
        local_mean = cv2.GaussianBlur(bgr_filled, (ksize, ksize), 0)
        sq_diff = (bgr_filled - local_mean) ** 2
        local_var = cv2.GaussianBlur(sq_diff, (ksize, ksize), 0)
        local_std = np.sqrt(local_var) + 1e-6
        
        norm = (bgr - local_mean) / local_std
        
        norm_valid = norm[valid]
        if norm_valid.size > 0:
            for c in range(3):
                chan_valid = norm_valid[..., c]
                c_min = np.percentile(chan_valid, 1)
                c_max = np.percentile(chan_valid, 99)
                if c_max > c_min:
                    norm[..., c] = (norm[..., c] - c_min) / (c_max - c_min) * 255.0
                    
        res = np.clip(norm, 0, 255).astype(np.uint8)
        res[~valid] = 0
        return res
