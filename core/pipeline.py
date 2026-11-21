"""
core/pipeline.py
================
Reference-Guided Ceramic Defect Inspection Pipeline:
1. Product Reference Enrollment (Học sâu vân tay bình gốc)
2. Reference-Guided Differential Inspection (Soi lỗi đối chiếu vi sai qua Cổng kiểm chứng DINOv2)
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, Any

from core.preprocessor import clean_rembg_image, detect_specular_mask, detect_saturated_mask, get_rembg_session

DEVICE = torch.device("cpu")


DEFAULT_PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "profiles")


def resolve_profile_id(product_id: str, view: str, profiles_dir: Optional[str] = None) -> Optional[str]:
    """Tìm profile_id chuẩn tương ứng với product_id và góc chụp."""
    if profiles_dir is None:
        profiles_dir = DEFAULT_PROFILES_DIR
    pid = "B1" if str(product_id) in ("1", "default") else str(product_id)
    view_map = {
        "truoc": "01_front",
        "sau": "02_back",
        "trai": "03_left",
        "phai": "04_right",
        "front": "01_front",
        "back": "02_back",
        "left": "03_left",
        "right": "04_right",
    }
    slot = view_map.get(view, "01_front")
    
    # 1. Thử {product_id}_{slot} (VD: B1_01_front)
    candidate = f"{pid}_{slot}"
    if os.path.exists(os.path.join(profiles_dir, f"{candidate}_profile.pt")):
        return candidate
        
    # 2. Thử {product_id} (VD: binh_mau_01, binh_con_cong)
    if os.path.exists(os.path.join(profiles_dir, f"{pid}_profile.pt")):
        return pid
        
    # 3. Fallback về B1_{slot}
    fallback = f"B1_{slot}"
    if os.path.exists(os.path.join(profiles_dir, f"{fallback}_profile.pt")):
        return fallback
        
    return None


class ReferencePipeline:
    def __init__(self, profiles_dir: Optional[str] = None):
        self.profiles_dir = profiles_dir or DEFAULT_PROFILES_DIR
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.device = DEVICE
        
        # Tải backbone DINOv2
        print("[INIT] Đang tải mô hình nền tảng DINOv2 (dinov2_vits14)...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.backbone.eval().to(self.device)
        
        # Tái sử dụng session RemBG BiRefNet dùng chung (tránh nạp 2 lần tốn ~1GB RAM)
        rembg_sess = get_rembg_session()
        self.rembg_session = rembg_sess if rembg_sess else None
        if self.rembg_session is not None:
            print("[INIT] Đã nạp mô hình tiền xử lý RemBG (Shared BiRefNet/u2net)!")
        else:
            print("[WARN] Không thể nạp RemBG session.")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Tải tầng nơ-ron phân loại khuyết tật CeramicDefectHead (CD-MLP)
        self.defect_head = None
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_weights = [
            os.path.join(base_dir, "models", "defect_head.pt"),
            os.path.join(base_dir, "training", "defect_head.pt")
        ]
        for w_path in candidate_weights:
            if os.path.exists(w_path):
                try:
                    from core.classifier import CeramicDefectHead
                    head = CeramicDefectHead(in_features=384, num_classes=4)
                    head.load_weights(w_path, device=str(self.device))
                    self.defect_head = head
                    print(f"[INIT] Đã nạp tầng phân loại sâu CeramicDefectHead từ: {w_path}")
                    break
                except Exception as e_head:
                    print(f"[WARN] Không thể nạp defect_head: {e_head}")

        print("[INIT] Khởi tạo ReferencePipeline thành công!")

    def extract_vase_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Tách nền bằng RemBG u2netp + lọc giữ lại thành phần liên thông lớn nhất (thân bình)"""
        h, w = img_bgr.shape[:2]
        
        if np.all(img_bgr[:15, :15] > 250) and np.all(img_bgr[-15:, -15:] > 250):
            mask = np.any(img_bgr < 250, axis=2).astype(np.uint8) * 255
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest_label).astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                return mask
                
        if self.rembg_session is not None:
            try:
                from rembg import remove
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                out_mask = remove(pil_img, session=self.rembg_session, only_mask=True)
                alpha = np.array(out_mask)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask = (labels == largest_label).astype(np.uint8) * 255
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                    return mask
            except Exception as e:
                print(f"[WARN] RemBG gặp lỗi, dùng fallback: {e}")
                
        bg_color = np.median(img_bgr[:25, :25], axis=(0, 1))
        diff_bg = np.linalg.norm(img_bgr.astype(np.float32) - bg_color, axis=-1)
        if diff_bg.max() > 15.0 and np.mean(diff_bg < 15.0) > 0.05:
            mask = (diff_bg > 15.0).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    def enroll_product(self, product_img_path: str, product_id: str) -> str:
        """Học vân tay bình gốc và lưu vào profiles/{product_id}_profile.pt."""
        print(f"\n[ENROLL] Bắt đầu học vân tay cho sản phẩm: {product_id}")
        if not os.path.exists(product_img_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {product_img_path}")
            
        p_bgr = cv2.imread(product_img_path)
        h, w = p_bgr.shape[:2]
        
        fg_mask = self.extract_vase_mask(p_bgr)
        inner_mask = cv2.erode(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        p_clean = p_bgr.copy()
        p_clean[fg_mask == 0] = [255, 255, 255]
        
        aug_list = [p_clean]
        aug_list.append(np.clip(p_clean.astype(np.float32) * 1.15, 0, 255).astype(np.uint8))
        aug_list.append(np.clip(p_clean.astype(np.float32) * 0.88, 0, 255).astype(np.uint8))
        lab_tmp = cv2.cvtColor(p_clean, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_tmp[..., 0] = clahe.apply(lab_tmp[..., 0])
        aug_list.append(cv2.cvtColor(lab_tmp, cv2.COLOR_LAB2BGR))
        for ang in [-1.5, 1.5]:
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            aug_list.append(cv2.warpAffine(p_clean, M_rot, (w, h), borderValue=(255, 255, 255)))
            
        features_list = []
        with torch.no_grad():
            for aug_img in aug_list:
                rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
                t_img = self.transform(resized).unsqueeze(0).to(self.device)
                f = self.backbone.get_intermediate_layers(t_img, n=1)[0].squeeze(0)
                f = f / f.norm(dim=-1, keepdim=True)
                features_list.append(f.cpu())
                
        all_features = torch.stack(features_list, dim=0)
        
        p_gray = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(p_gray, 40, 130)
        k_fiss = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(p_gray, cv2.MORPH_BLACKHAT, k_fiss)
        edge_map = cv2.bitwise_or(canny, (blackhat > 25).astype(np.uint8) * 255)
        edge_map[fg_mask == 0] = 0
        edge_tolerance_map = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        
        p_lab = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2LAB)
        valid_a = p_lab[inner_mask > 0, 1]
        valid_b = p_lab[inner_mask > 0, 2]
        
        profile_data = {
            "product_id": product_id,
            "img_shape": (h, w),
            "reference_bgr": p_bgr,
            "fg_mask": fg_mask,
            "inner_mask": inner_mask,
            "dino_features": all_features,
            "edge_tolerance_map": edge_tolerance_map,
            "lab_median_a": float(np.median(valid_a)),
            "lab_median_b": float(np.median(valid_b)),
            "lab_std_a": float(np.std(valid_a)),
            "lab_std_b": float(np.std(valid_b))
        }
        
        save_path = os.path.join(self.profiles_dir, f"{product_id}_profile.pt")
        torch.save(profile_data, save_path)
        print(f"[ENROLL] Đã học và lưu hồ sơ vân tay: {save_path}")
        return save_path

    def inspect_return(self, return_img_path: str, product_id: str, output_dir: str = "uploads") -> Dict[str, Any]:
        """Soi lỗi ảnh hoàn trả qua Cổng kiểm chứng đối chiếu bình gốc."""
        profile_path = os.path.join(self.profiles_dir, f"{product_id}_profile.pt")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Chưa có hồ sơ cho sản phẩm '{product_id}'. Vui lòng enroll trước!")
            
        print(f"\n[INSPECT] Soi lỗi ảnh: {return_img_path} | Hồ sơ đối chiếu: {profile_path}")
        profile = torch.load(profile_path, map_location=self.device, weights_only=False)
        ref_bgr = profile["reference_bgr"]
        ref_fg = profile["fg_mask"]
        ref_edge_tol = profile["edge_tolerance_map"]
        
        ret_bgr = cv2.imread(return_img_path)
        if ret_bgr is None:
            raise FileNotFoundError(f"Không thể mở ảnh: {return_img_path}")
            
        h_in, w_in = ret_bgr.shape[:2]
        is_clean_white = (h_in == 1024 and w_in == 1024 and 
                          np.all(ret_bgr[:15, :15] > 250) and np.all(ret_bgr[-15:, -15:] > 250))
        if not is_clean_white:
            standardized = clean_rembg_image(return_img_path, target_size=1024)
            if standardized is not None:
                ret_bgr = standardized
                
        h_r, w_r = ret_bgr.shape[:2]
        
        # 1. Căn chỉnh phối cảnh SIFT Homography (Lowe's Ratio Test)
        sift = cv2.SIFT_create(nfeatures=2500)
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY), None)

        good_matches = []
        if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            knn = bf.knnMatch(des1, des2, k=2)
            for pair in knn:
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                    good_matches.append(pair[0])

        H = None
        if len(good_matches) >= 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)

        ref_warped = cv2.warpPerspective(ref_bgr, H, (w_r, h_r))
        ref_fg_warped = cv2.warpPerspective(ref_fg, H, (w_r, h_r), flags=cv2.INTER_NEAREST)
        ref_edge_warped = cv2.warpPerspective(ref_edge_tol, H, (w_r, h_r), flags=cv2.INTER_NEAREST)

        # 2. Tiền xử lý mask thân bình & loại trừ vùng lóa sáng gắt + đế gỗ tối màu ở đáy
        ret_fg = self.extract_vase_mask(ret_bgr)
        common_vase = cv2.bitwise_and(ret_fg, ref_fg_warped)
        valid_inspect_mask = cv2.erode(common_vase, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0

        # Chỉ loại trừ đốm lóa specular thật sự (không cắt nhầm men trắng sáng bình thường)
        glare_spec = detect_specular_mask(ret_bgr, ret_fg)
        if np.any(glare_spec > 0):
            glare_dilated = cv2.dilate(glare_spec, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            valid_inspect_mask = valid_inspect_mask & (glare_dilated == 0)

        # Loại trừ vùng đế gỗ nâu/tối nếu dính vào 20% dưới cùng của mask
        ys_v, _ = np.where(ret_fg > 0)
        if len(ys_v) > 0:
            y_bot_cut = int(ys_v.min() + 0.80 * (ys_v.max() - ys_v.min()))
            hsv_ret = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2HSV)
            wood_mask = (
                (hsv_ret[..., 1] > 35) &
                (hsv_ret[..., 2] < 120)
            )
            wood_mask[:y_bot_cut, :] = False
            valid_inspect_mask = valid_inspect_mask & (~wood_mask)
        
        # 3. Chuẩn hóa phơi sáng & cân bằng màu CIELab (toàn cục + bù gradient sáng cục bộ)
        p_lab = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_lab = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        if np.any(valid_inspect_mask):
            off_L = np.median(r_lab[valid_inspect_mask, 0]) - np.median(p_lab[valid_inspect_mask, 0])
            off_a = np.median(r_lab[valid_inspect_mask, 1]) - np.median(p_lab[valid_inspect_mask, 1])
            off_b = np.median(r_lab[valid_inspect_mask, 2]) - np.median(p_lab[valid_inspect_mask, 2])
            r_lab[..., 0] -= off_L
            r_lab[..., 1] -= off_a
            r_lab[..., 2] -= off_b

            # Bù trừ độ sáng không đều trên thân bình cong (chiếu sáng lệch / đèn trong bình)
            diff_L_raw = np.where(valid_inspect_mask, r_lab[..., 0] - p_lab[..., 0], 0.0).astype(np.float32)
            smooth_L_grad = cv2.GaussianBlur(diff_L_raw, (81, 81), 0)
            r_lab[..., 0] -= np.clip(smooth_L_grad, -40.0, 40.0)

        # 4. Tìm kiếm dung sai vi mô Local Search bán kính ±6px (Triệt tiêu sai số hoa văn gốc)
        offsets = [-6, -3, 0, 3, 6]
        pad = 6
        p_padded = cv2.copyMakeBorder(p_lab, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        min_dE_full = np.full((h_r, w_r), 999.0, dtype=np.float32)

        for dy in offsets:
            for dx in offsets:
                shifted = p_padded[pad + dy:pad + dy + h_r, pad + dx:pad + dx + w_r]
                dL = r_lab[..., 0] - shifted[..., 0]
                da = r_lab[..., 1] - shifted[..., 1]
                db = r_lab[..., 2] - shifted[..., 2]
                dist = np.sqrt(dL**2 + da**2 + db**2)
                min_dE_full = np.minimum(min_dE_full, dist)

        min_dE_full[~valid_inspect_mask] = 0.0

        # 5. Cổng kiểm chứng đối chiếu vi sai kết hợp edge tolerance trên nét vẽ gốc
        base_thresh = np.where(ref_edge_warped > 0, 34.0, 25.0).astype(np.float32)
        confirmed_score = np.clip((min_dE_full - base_thresh) / 25.0, 0.0, 1.0)
        confirmed_score[~valid_inspect_mask] = 0.0

        binary_defects = (confirmed_score > 0.15).astype(np.uint8) * 255
        binary_defects = cv2.morphologyEx(binary_defects, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # 5b. Trích xuất đặc trưng vi sai DINOv2 phục vụ quét nơ-ron toàn diện (Neural Scan)
        delta_f = None
        neural_defect_prob = None
        if self.defect_head is not None:
            try:
                rgb_ret = cv2.cvtColor(cv2.resize(ret_bgr, (448, 448), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                rgb_ref = cv2.cvtColor(cv2.resize(ref_warped, (448, 448), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                t_ret = self.transform(rgb_ret).unsqueeze(0).to(self.device)
                t_ref = self.transform(rgb_ref).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    f_ret = self.backbone.get_intermediate_layers(t_ret, n=1)[0].squeeze(0)
                    f_ref = self.backbone.get_intermediate_layers(t_ref, n=1)[0].squeeze(0)
                    f_ret = f_ret / f_ret.norm(dim=-1, keepdim=True)
                    f_ref = f_ref / f_ref.norm(dim=-1, keepdim=True)
                    delta_f = torch.abs(f_ret - f_ref)  # (1024, 384)
                    logits_grid = self.defect_head(delta_f.to(self.device))
                    probs_grid = torch.softmax(logits_grid, dim=-1)  # (1024, 4)
                    defect_prob_grid = (1.0 - probs_grid[:, 0]).reshape(32, 32).cpu().numpy()
                    neural_defect_prob = cv2.resize(defect_prob_grid, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
                    neural_defect_prob[~valid_inspect_mask] = 0.0
            except Exception as e_dino:
                print(f"[WARN] Lỗi trích xuất DINOv2 vi sai: {e_dino}")
                delta_f = None
                neural_defect_prob = None

        # 6. Trích xuất danh sách khuyết tật
        cnts, _ = cv2.findContours(binary_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_vis = ret_bgr.copy()
        total_vase_pixels = float(np.count_nonzero(valid_inspect_mask))
        defect_items = []
        total_damaged_pixels = 0


        
        for c in cnts:
            area = cv2.contourArea(c)
            arc_len = cv2.arcLength(c, False)
            if (area >= 20 or arc_len >= 25) and max(cv2.boundingRect(c)[2:]) >= 8:
                x, y, w, h = cv2.boundingRect(c)
                if not (w > w_r * 0.85 and h > h_r * 0.85) and area < total_vase_pixels * 0.5:
                    pred_label_str = None
                    pred_conf = 0.0
                    pred_cat = None
                    pred_color = (0, 0, 255)

                    # Phân loại bằng mạng nơ-ron CeramicDefectHead (CD-MLP)
                    if self.defect_head is not None and delta_f is not None:
                        scale_x = 448.0 / max(1, w_r)
                        scale_y = 448.0 / max(1, h_r)
                        cx = int((x + w / 2.0) * scale_x)
                        cy = int((y + h / 2.0) * scale_y)
                        c_idx = max(0, min(31, cx // 14))
                        r_idx = max(0, min(31, cy // 14))
                        p_idx = r_idx * 32 + c_idx

                        with torch.no_grad():
                            p_vec = delta_f[p_idx].unsqueeze(0).to(self.device)
                            logits = self.defect_head(p_vec)
                            probs = torch.softmax(logits, dim=-1)[0]
                            pred_class = int(torch.argmax(probs).item())
                            pred_conf = float(probs[pred_class].item())

                        # 1. Nếu mô hình CD-MLP nhận định là Men lành (Normal - nhãn 0): triệt tiêu báo lỗi giả
                        if pred_class == 0:
                            continue

                        # 2. Nếu là khuyết tật nhưng độ tin cậy quá thấp (< 60%): bỏ qua nhiễu viền
                        if pred_conf < 0.60:
                            continue

                        if pred_class == 1:
                            pred_cat = "crack"
                            pred_label_str = f"VET NUT ({int(pred_conf*100)}%)"
                            pred_color = (0, 0, 255)      # Đỏ
                        elif pred_class == 2:
                            pred_cat = "chip"
                            pred_label_str = f"VET ME ({int(pred_conf*100)}%)"
                            pred_color = (0, 255, 255)    # Vàng
                        elif pred_class == 3:
                            pred_cat = "stain"
                            pred_label_str = f"VET O ({int(pred_conf*100)}%)"
                            pred_color = (0, 140, 255)    # Cam
                        else:
                            pred_cat = "defect"
                            pred_label_str = f"KHUYET TAT ({int(pred_conf*100)}%)"
                            pred_color = (0, 0, 255)

                    # Fallback theo đặc tính màu nếu chưa có defect_head
                    if pred_label_str is None:
                        ret_roi = ret_bgr[y:y+h, x:x+w]
                        roi_gray = cv2.cvtColor(ret_roi, cv2.COLOR_BGR2GRAY)
                        if np.percentile(roi_gray, 5) < 70 or np.mean(roi_gray) < 80:
                            pred_cat = "crack_or_dark_hole"
                            pred_label_str = "VET NUT/VO"
                            pred_color = (0, 0, 255)
                        elif np.std(ret_roi[..., 0].astype(np.float32) - ret_roi[..., 2].astype(np.float32)) > 15:
                            pred_cat = "ink_mark_color"
                            pred_label_str = "VET MUC/O MAU"
                            pred_color = (0, 140, 255)
                        else:
                            pred_cat = "surface_anomaly"
                            pred_label_str = "VET ME/XUOC"
                            pred_color = (0, 255, 255)

                    cat = pred_cat
                    label = pred_label_str
                    color = pred_color
                    total_damaged_pixels += area

                        
                    cv2.drawContours(detected_vis, [c], -1, color, 3)
                    if max(w, h) <= 80:
                        cv2.circle(detected_vis, (x + w//2, y + h//2), max(w, h)//2 + 10, color, 2)
                    else:
                        cv2.rectangle(detected_vis, (x - 6, y - 6), (x + w + 6, y + h + 6), color, 2)
                    cv2.putText(detected_vis, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    
                    defect_items.append({
                        "category": cat,
                        "label": label,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area_pixels": float(area)
                    })
                    
        damage_ratio = (total_damaged_pixels / (total_vase_pixels + 1e-6)) * 100.0
        
        # 7. Xuất ảnh trực quan 3 khung hình
        heatmap_vis = cv2.applyColorMap((confirmed_score * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_vis[~valid_inspect_mask] = [25, 25, 25]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        p1 = ret_bgr.copy()
        cv2.putText(p1, '1. Anh Hoan Tra', (10, 25), font, 0.55, (255, 255, 255), 2)
        heatmap_title = '2. AI Neural Heatmap (DINOv2+CD-MLP)' if neural_defect_prob is not None else '2. Heatmap Vi Sai (CIELab)'
        cv2.putText(heatmap_vis, heatmap_title, (10, 25), font, 0.48, (255, 255, 255), 2)
        cv2.putText(detected_vis, f'3. Phat Hien {len(defect_items)} Loi (AI Neural Scan)', (10, 25), font, 0.50, (0, 255, 0), 2)
        
        comp_result = np.hstack([p1, heatmap_vis, detected_vis])
        
        os.makedirs(output_dir, exist_ok=True)
        ret_base = os.path.splitext(os.path.basename(return_img_path))[0]
        img_out_path = os.path.join(output_dir, f"{product_id}_{ret_base}_result.jpg")
        cv2.imwrite(img_out_path, comp_result)
        
        vase_coords = cv2.findNonZero(ret_fg)
        vase_bbox = list(cv2.boundingRect(vase_coords)) if vase_coords is not None else [0, 0, w_r, h_r]

        report = {
            "product_id": product_id,
            "image_name": os.path.basename(return_img_path),
            "is_damaged": len(defect_items) > 0,
            "defect_count": len(defect_items),
            "damage_percentage": round(damage_ratio, 3),
            "severity": "Hư hỏng nặng" if len(defect_items) > 5 or damage_ratio > 1.0 else ("Có khuyết tật" if len(defect_items) > 0 else "Nguyên vẹn"),
            "result_image": img_out_path,
            "vase_bbox": vase_bbox,
            "defects": defect_items
        }
        
        json_out_path = os.path.join(output_dir, f"{product_id}_{ret_base}_report.json")
        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, ensure_ascii=False, indent=2)
            
        print(f"[INSPECT] Hoàn tất! Phát hiện {len(defect_items)} khuyết tật. Ảnh: {img_out_path}")
        return report


def deduplicate_cross_view_defects(view_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Khử trùng lặp khuyết tật đa góc 3D (Cross-View Deduplication) cho 4 mặt xoay 360°.
    Hai vết lỗi ở hai mặt liền kề được gộp thành 1 nếu nằm ở vùng giáp ranh mép
    (x_norm >= 0.80 và x_norm <= 0.20), cùng độ cao (|dy| <= 0.03),
    cùng kích thước (|dh| <= 0.02), và cùng loại (category).
    """
    ordered_views = ["front", "right", "back", "left"]
    adjacent_pairs = [
        ("front", "right"),
        ("right", "back"),
        ("back", "left"),
        ("left", "front"),
    ]

    def _norm_box(defect_bbox, vase_bbox):
        dx, dy, dw, dh = defect_bbox
        vx, vy, vw, vh = vase_bbox
        vw = max(1, vw)
        vh = max(1, vh)
        return (
            (dx - vx) / vw,
            (dy - vy) / vh,
            dw / vw,
            dh / vh,
        )

    removed = set()  # (view, defect_idx)
    merged_pairs = []

    for v_a, v_b in adjacent_pairs:
        rep_a = view_results.get(v_a)
        rep_b = view_results.get(v_b)
        if not rep_a or not rep_b or rep_a.get("status") != "ok" or rep_b.get("status") != "ok":
            continue
        vbox_a = rep_a.get("vase_bbox") or [0, 0, 1024, 1024]
        vbox_b = rep_b.get("vase_bbox") or [0, 0, 1024, 1024]
        defs_a = rep_a.get("defects", [])
        defs_b = rep_b.get("defects", [])

        for i, da in enumerate(defs_a):
            if (v_a, i) in removed:
                continue
            xa, ya, wa, ha = _norm_box(da["bbox"], vbox_a)
            if xa < 0.80:
                continue
            for j, db in enumerate(defs_b):
                if (v_b, j) in removed:
                    continue
                if da.get("category") != db.get("category"):
                    continue
                xb, yb, wb, hb = _norm_box(db["bbox"], vbox_b)
                if xb > 0.20:
                    continue
                if abs(ya - yb) <= 0.03 and abs(ha - hb) <= 0.02:
                    # Giữ vết lỗi có diện tích quan sát lớn hơn
                    if da.get("area_pixels", 0) >= db.get("area_pixels", 0):
                        removed.add((v_b, j))
                        merged_pairs.append({"kept": f"{v_a}#{i}", "dropped": f"{v_b}#{j}"})
                    else:
                        removed.add((v_a, i))
                        merged_pairs.append({"kept": f"{v_b}#{j}", "dropped": f"{v_a}#{i}"})
                        break

    unique_count = 0
    for v in ordered_views:
        rep = view_results.get(v)
        if not rep or rep.get("status") != "ok":
            continue
        kept = []
        for idx, d in enumerate(rep.get("defects", [])):
            is_dup = (v, idx) in removed
            d["is_cross_view_duplicate"] = is_dup
            if not is_dup:
                kept.append(d)
                unique_count += 1
        rep["unique_defect_count"] = len(kept)

    return {
        "total_unique_defects": unique_count,
        "deduplicated_pairs": merged_pairs,
    }
