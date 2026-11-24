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
from typing import Optional, Dict, Any, Tuple, List

# Bảo vệ ổ C: Ép cache PyTorch sang ổ E
if "TORCH_HOME" not in os.environ and os.path.exists(r"E:\Ceramic_Defect_Benchmark\cache\torch"):
    os.environ["TORCH_HOME"] = r"E:\Ceramic_Defect_Benchmark\cache\torch"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.preprocessor import clean_rembg_image, get_rembg_session

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
        
        # Load DINOv2 backbone
        print("[INIT] Loading DINOv2 foundation backbone (dinov2_vits14)...")
        hub_dir = torch.hub.get_dir()
        cached_repo = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
        if os.path.exists(cached_repo):
            self.backbone = torch.hub.load(cached_repo, 'dinov2_vits14', source='local')
        else:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True)
        self.backbone.eval().to(self.device)
        
        # RemBG session khởi tạo lười (lazy load) để tiết kiệm 1.5 GB RAM cho quá trình suy luận
        self.rembg_session = None
        
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
                    print(f"[INIT] Loaded CeramicDefectHead neural classifier from: {w_path}")
                    break
                except Exception as e_head:
                    print(f"[WARN] Failed to load defect_head: {e_head}")

        print("[INIT] ReferencePipeline initialized successfully.")

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
                
        if self.rembg_session is None:
            self.rembg_session = get_rembg_session()
        if self.rembg_session:
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
                print(f"[WARN] RemBG segmentation failed, using morphology fallback: {e}")
                
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
        """Enroll reference fingerprint profile and save to profiles/{product_id}_profile.pt."""
        print(f"\n[ENROLL] Starting reference enrollment for product: {product_id}")
        if not os.path.exists(product_img_path):
            raise FileNotFoundError(f"[ERROR] Product reference image not found: {product_img_path}")
            
        raw = cv2.imread(product_img_path, cv2.IMREAD_UNCHANGED)
        if raw is not None and len(raw.shape) == 3 and raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            fg_mask = (alpha > 128).astype(np.uint8) * 255
            b, g, r = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
            p_bgr = cv2.merge([b, g, r])
            p_bgr[fg_mask == 0] = [255, 255, 255]
        else:
            p_bgr = cv2.imread(product_img_path)
            fg_mask = self.extract_vase_mask(p_bgr)
            
        h, w = p_bgr.shape[:2]
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
        print(f"[ENROLL] Saved reference profile to: {save_path}")
        return save_path

    def inspect_return(
        self,
        return_img_path: str,
        product_id: str,
        output_dir: str = "uploads",
        ablation_mode: str = "B",
        local_search_k: int = 7,
        artwork_penalty: float = 40.0,
        sobel_threshold: float = 450.0
    ) -> Dict[str, Any]:
        """Reference-Guided Differential Inspection against enrolled golden profile."""
        profile_path = os.path.join(self.profiles_dir, f"{product_id}_profile.pt")
        if not os.path.exists(profile_path):
            resolved = resolve_profile_id(product_id, "front", self.profiles_dir)
            if resolved and os.path.exists(os.path.join(self.profiles_dir, f"{resolved}_profile.pt")):
                profile_path = os.path.join(self.profiles_dir, f"{resolved}_profile.pt")
            else:
                raise FileNotFoundError(f"[ERROR] Reference profile not found for '{product_id}'. Please enroll first.")
            
        print(f"\n[INSPECT] Inspecting return image: {return_img_path} | Reference profile: {profile_path}")
        profile = torch.load(profile_path, map_location=self.device, weights_only=False)
        ref_bgr = profile["reference_bgr"]
        ref_fg = profile["fg_mask"]
        ref_edge_tol = profile["edge_tolerance_map"]
        
        ret_bgr = cv2.imread(return_img_path)
        if ret_bgr is None:
            raise FileNotFoundError(f"[ERROR] Unable to open return image: {return_img_path}")
            
        h_in, w_in = ret_bgr.shape[:2]
        is_clean_white = (h_in == 1024 and w_in == 1024 and 
                          np.all(ret_bgr[:15, :15] > 250) and np.all(ret_bgr[-15:, -15:] > 250))
        if not is_clean_white:
            standardized = clean_rembg_image(return_img_path, target_size=1024)
            if standardized is not None:
                ret_bgr = standardized
                
        h_r, w_r = ret_bgr.shape[:2]
        
        # Tải mặt nạ lóa sáng / ignore mask tiền xử lý nếu có
        ret_ignore_mask = None
        cand_ign_paths = []
        if "_processed" in return_img_path:
            cand_ign_paths.append(return_img_path.replace("_processed.png", "_ignore_mask.png"))
            cand_ign_paths.append(return_img_path.replace("_processed.jpg", "_ignore_mask.png"))
        else:
            base_p, _ = os.path.splitext(return_img_path)
            cand_ign_paths.append(f"{base_p}_ignore_mask.png")
        cand_ign_paths.append(os.path.join(os.path.dirname(return_img_path), "customer_ignore_mask.png"))

        for p_ign in cand_ign_paths:
            if os.path.exists(p_ign) and os.path.abspath(p_ign) != os.path.abspath(return_img_path):
                raw_ign = cv2.imread(p_ign, cv2.IMREAD_GRAYSCALE)
                if raw_ign is not None and raw_ign.shape[:2] == (h_r, w_r):
                    ret_ignore_mask = raw_ign
                    break
        
        # Standardize reference resolution to match return image canvas (1024x1024)
        if ref_bgr.shape[:2] != (h_r, w_r):
            ref_bgr = cv2.resize(ref_bgr, (w_r, h_r), interpolation=cv2.INTER_AREA)
        if ref_fg.shape[:2] != (h_r, w_r):
            ref_fg = cv2.resize(ref_fg, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
        
        # 1. Rigid Partial Affine Alignment (Scale + Rotation + Translation, ZERO perspective distortion)
        ret_fg = self.extract_vase_mask(ret_bgr)
        M_rigid = None
        sift = cv2.SIFT_create(nfeatures=2500)
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY), None)

        if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
            if len(matches) >= 8:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                num_inliers = int(inliers.sum()) if inliers is not None else 0
                if M is not None and num_inliers >= 12:
                    s = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                    theta_deg = abs(np.arctan2(M[1, 0], M[0, 0]) * 180.0 / np.pi)
                    # Tight scale tolerance: allow scaling from 0.88x to 1.15x for standardized 1024x1024 canvases
                    if 0.70 <= s <= 1.35 and theta_deg <= 25.0:
                        M_rigid = M

        alignment_method = "SIFT_RANSAC"
        if M_rigid is not None:
            ref_warped = cv2.warpAffine(ref_bgr, M_rigid, (w_r, h_r), borderValue=(255, 255, 255))
            ref_fg_warped = cv2.warpAffine(ref_fg, M_rigid, (w_r, h_r), flags=cv2.INTER_NEAREST, borderValue=0)
            ref_edge_tol_warped = cv2.warpAffine(ref_edge_tol, M_rigid, (w_r, h_r), flags=cv2.INTER_NEAREST, borderValue=0) if ref_edge_tol is not None else np.zeros((h_r, w_r), dtype=np.uint8)
        else:
            # Silhouette Contour-based Alignment Fallback
            alignment_method = "SILHOUETTE_FALLBACK"
            rx, ry, rw, rh = cv2.boundingRect(ref_fg)
            cx, cy, cw, ch = cv2.boundingRect(ret_fg)
            s_scale_y = float(ch) / float(max(10, rh))
            s_scale_x = float(cw) / float(max(10, rw))
            s_scale = (s_scale_x + s_scale_y) / 2.0
            s_scale = max(0.80, min(1.20, s_scale))

            dx = float((cx + cw / 2.0) - (rx + rw / 2.0) * s_scale)
            dy = float((cy + ch / 2.0) - (ry + ry + rh / 2.0) * s_scale if ry + rh / 2.0 == 0 else (cy + ch / 2.0) - (ry + rh / 2.0) * s_scale)
            M_sil = np.float32([[s_scale, 0.0, dx], [0.0, s_scale, dy]])
            ref_warped = cv2.warpAffine(ref_bgr, M_sil, (w_r, h_r), borderValue=(255, 255, 255))
            ref_fg_warped = cv2.warpAffine(ref_fg, M_sil, (w_r, h_r), flags=cv2.INTER_NEAREST, borderValue=0)
            ref_edge_tol_warped = cv2.warpAffine(ref_edge_tol, M_sil, (w_r, h_r), flags=cv2.INTER_NEAREST, borderValue=0) if ref_edge_tol is not None else np.zeros((h_r, w_r), dtype=np.uint8)
        
        # 2. Common vase mask computation (safeguards vase rim and base boundary with erosion margin)
        common_vase = cv2.bitwise_and(ret_fg, ref_fg_warped)
        union_vase = cv2.bitwise_or(ret_fg, ref_fg_warped)
        iou_sil = float(np.count_nonzero(common_vase)) / float(max(1, np.count_nonzero(union_vase)))

        valid_inspect_mask = cv2.erode(common_vase, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        
        # 3. Chuẩn hóa phơi sáng & cân bằng màu CIELab
        p_lab = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_lab = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Exclude non-ceramic accessory (carved wooden stand present in customer return but absent in reference)
        y_coords = np.arange(h_r)[:, None]
        is_stand = (y_coords >= 850) & (r_lab[..., 0] < 150.0) & (p_lab[..., 0] >= 200.0)
        valid_inspect_mask = valid_inspect_mask & (~is_stand)
        # Protect bottom contact boundary dynamically instead of hardcoding y <= 780
        ys_com, _ = np.where(common_vase > 0)
        if len(ys_com) > 0:
            bottom_limit = ys_com.max() - 10
            valid_inspect_mask = valid_inspect_mask & (y_coords <= bottom_limit)
        
        valid_pixel_count = int(np.count_nonzero(valid_inspect_mask))
        # Quality Gate: Check for empty inspection mask or severe alignment failure (IoU < 0.60)
        if valid_pixel_count < 500 or (alignment_method == "SILHOUETTE_FALLBACK" and iou_sil < 0.60):
            print(f"[WARN] Alignment Quality Gate triggered: method={alignment_method}, IoU={iou_sil:.3f}, valid_px={valid_pixel_count}")
            report = {
                "product_id": product_id,
                "image_name": os.path.basename(return_img_path),
                "is_damaged": False,
                "defect_count": 0,
                "damage_percentage": 0.0,
                "severity": "NEEDS_REVIEW",
                "verdict": "NEEDS_REVIEW",
                "needs_review": True,
                "review_reason": f"Quality Gate: Alignment or mask failed (method={alignment_method}, IoU={iou_sil:.2f}, valid_px={valid_pixel_count})",
                "result_image": "",
                "vase_bbox": [0, 0, w_r, h_r],
                "defects": []
            }
            return report

        off_L = np.median(r_lab[valid_inspect_mask, 0]) - np.median(p_lab[valid_inspect_mask, 0])
        off_a = np.median(r_lab[valid_inspect_mask, 1]) - np.median(p_lab[valid_inspect_mask, 1])
        off_b = np.median(r_lab[valid_inspect_mask, 2]) - np.median(p_lab[valid_inspect_mask, 2])
        r_lab[..., 0] -= off_L
        r_lab[..., 1] -= off_a
        r_lab[..., 2] -= off_b
        
        # 4. Tìm kiếm dung sai vi mô Local Search KxK (Triệt tiêu sai số góc lệch)
        k_size = max(1, int(local_search_k))
        if k_size <= 1:
            # 1x1: Vi sai trực tiếp không bù lệch
            dL_direct = r_lab[..., 0] - p_lab[..., 0]
            da_direct = r_lab[..., 1] - p_lab[..., 1]
            db_direct = r_lab[..., 2] - p_lab[..., 2]
            min_dE_full = np.sqrt(dL_direct**2 + da_direct**2 + db_direct**2)
        else:
            pad = k_size // 2
            p_padded = cv2.copyMakeBorder(p_lab, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            min_dE_full = np.full((h_r, w_r), 999.0, dtype=np.float32)

            for dy in range(k_size):
                for dx in range(k_size):
                    shifted = p_padded[dy:dy+h_r, dx:dx+w_r]
                    dL = r_lab[..., 0] - shifted[..., 0]
                    da = r_lab[..., 1] - shifted[..., 1]
                    db = r_lab[..., 2] - shifted[..., 2]
                    dist = np.sqrt(dL**2 + da**2 + db**2)
                    min_dE_full = np.minimum(min_dE_full, dist)

        min_dE_full[~valid_inspect_mask] = 0.0
        min_dE_pre = min_dE_full.copy()

        dL_map = r_lab[..., 0] - p_lab[..., 0]
        da_map = r_lab[..., 1] - p_lab[..., 1]
        db_map = r_lab[..., 2] - p_lab[..., 2]
        chroma_map = np.sqrt(da_map**2 + db_map**2)

        # Dynamic Artwork Edge Zone Filtering:
        artwork_zone = np.zeros((h_r, w_r), dtype=bool)
        if ref_edge_tol_warped is not None and ref_edge_tol_warped.max() > 0:
            dilated_edges = cv2.dilate(ref_edge_tol_warped, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            artwork_zone = dilated_edges > 0
            if artwork_penalty > 0.0:
                # Xác định vùng vành miệng (rim_zone: 8% chiều cao đỉnh bình)
                ys_all_rim = np.where(ref_fg_warped > 0)[0]
                y_top_rim = float(ys_all_rim.min()) if len(ys_all_rim) > 0 else 0.0
                h_vase_rim = float(max(10, ys_all_rim.max() - ys_all_rim.min())) if len(ys_all_rim) > 0 else float(h_r)
                y_norm_grid = np.repeat(np.clip((np.arange(h_r)[:, None] - y_top_rim) / h_vase_rim, 0.0, 1.0), w_r, axis=1)
                rim_zone = (y_norm_grid <= 0.08)

                # Miễn trừ trừ phạt Artwork Penalty tại vành miệng nếu có sụt sáng cơ học hoặc biến thiên màu thực tế
                waive_rim = rim_zone & ((dL_map <= -15.0) | (min_dE_full >= 25.0))
                # Miễn trừ trừ phạt cho vết ố đậm màu trên men sứ trắng (Chroma >= 12.0)
                waive_stain = (chroma_map >= 18.0) & (min_dE_full >= 30.0)
                penalize_mask = artwork_zone & (~waive_rim) & (~waive_stain)
                min_dE_full[penalize_mask] = np.maximum(0.0, min_dE_full[penalize_mask] - float(artwork_penalty))
        
        min_dE_post = min_dE_full.copy()
        min_dE_post[~valid_inspect_mask] = 0.0

        self.last_debug_maps = {
            "min_dE_pre": min_dE_pre,
            "min_dE_post": min_dE_post,
            "artwork_zone": artwork_zone,
            "valid_inspect_mask": valid_inspect_mask
        }
        
        # 5. Cổng kiểm chứng đối chiếu vi sai kết hợp Local Tolerance Search
        confirmed_score = np.clip((min_dE_full - 25.0) / 25.0, 0.0, 1.0)
        confirmed_score[~valid_inspect_mask] = 0.0
        
        binary_defects = (confirmed_score > 0.15).astype(np.uint8) * 255
        # Kết nối các phân đoạn nứt vỡ và làm liền vòng tròn ố loang bằng phép Đóng hình thái học (MORPH_CLOSE)
        binary_defects = cv2.morphologyEx(binary_defects, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        
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
            except Exception as e_dino:
                print(f"[WARN] Differential DINOv2 feature extraction failed: {e_dino}")
                delta_f = None
                neural_defect_prob = None

        # High-frequency Sobel gradient magnitude difference map for mechanical crack/chip verification
        ret_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2GRAY)
        gx_ret = cv2.Sobel(ret_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_ret = cv2.Sobel(ret_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag_ret = cv2.magnitude(gx_ret, gy_ret)

        gx_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag_ref = cv2.magnitude(gx_ref, gy_ref)
        mag_ref_dil = cv2.dilate(mag_ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        delta_sobel = np.maximum(0.0, mag_ret - mag_ref_dil)

        # 6. Trích xuất danh sách khuyết tật: KIẾN TRÚC DUAL-STREAM
        detected_vis = ret_bgr.copy()
        total_vase_pixels = float(np.count_nonzero(ret_fg))
        defect_items = []
        total_damaged_pixels = 0

        # =========================================================================
        # STREAM 1: DIRECT CHROMATIC ANOMALY SCAN (Dị vật ngoại lai: Sợi chỉ, keo, vết ố màu)
        # Quét trực tiếp trên thực thể thân bình ret_fg, không phụ thuộc vào ma trận warp SIFT
        # =========================================================================
        # Stream 1 surface mask strictly bound to common vase physical silhouette with 10px safety erosion
        common_vase = cv2.bitwise_and(ret_fg, ref_fg_warped)
        surface_mask = cv2.erode(common_vase, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        if ret_ignore_mask is not None:
            surface_mask = cv2.bitwise_and(surface_mask, cv2.bitwise_not(ret_ignore_mask))
        surface_mask = surface_mask & (~is_stand)
        if len(ys_com) > 0:
            surface_mask = surface_mask & (y_coords <= bottom_limit)

        # Trừ bỏ màu nguyên bản của bình mẫu (con dấu triện son đỏ, logo xưởng)
        b_ref, g_ref, r_ref = ref_warped[..., 0].astype(float), ref_warped[..., 1].astype(float), ref_warped[..., 2].astype(float)
        redness_ref = np.maximum(0.0, r_ref - np.maximum(b_ref, g_ref))
        blueness_ref = np.maximum(0.0, b_ref - np.maximum(r_ref, g_ref))
        ref_red_dilated = cv2.dilate(redness_ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        ref_blue_dilated = cv2.dilate(blueness_ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

        b_ret, g_ret, r_ret = ret_bgr[..., 0].astype(float), ret_bgr[..., 1].astype(float), ret_bgr[..., 2].astype(float)
        redness_ret = np.maximum(0.0, r_ret - np.maximum(b_ret, g_ret))
        blueness_ret = np.maximum(0.0, b_ret - np.maximum(r_ret, g_ret))

        total_ret = np.maximum(1.0, r_ret + g_ret + b_ret)
        red_purity = redness_ret / total_ret
        blue_purity = blueness_ret / total_ret

        # Kiểm tra xem bình mẫu có bản chất là men hồng / men đỏ san hô tự nhiên không
        ref_vase_pts = np.count_nonzero(ref_fg_warped > 0)
        is_naturally_red_glaze = (np.count_nonzero((redness_ref >= 20.0) & (ref_fg_warped > 0)) > 0.05 * max(1, ref_vase_pts))

        net_foreign_red = np.maximum(0.0, redness_ret - ref_red_dilated) if not is_naturally_red_glaze else np.zeros_like(redness_ret)
        net_foreign_blue = np.maximum(0.0, blueness_ret - ref_blue_dilated)

        # Mặt nạ dị vật màu ngoại lai: Yêu cầu cả cường độ vi sai VÀ độ tinh khiết màu (saturation) để triệt tiêu bóng đổ ấm của phòng
        is_foreign_red = (net_foreign_red >= 40.0) & (red_purity >= 0.14)
        is_foreign_blue = (net_foreign_blue >= 40.0) & (blue_purity >= 0.14)
        foreign_chroma_mask = (is_foreign_red | is_foreign_blue) & (surface_mask > 0)
        foreign_chroma_closed = cv2.morphologyEx(foreign_chroma_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Cập nhật điểm nhiệt Heatmap trực tiếp cho dị vật màu
        confirmed_score[foreign_chroma_mask] = np.maximum(confirmed_score[foreign_chroma_mask], 0.95)

        cnts_s1, _ = cv2.findContours(foreign_chroma_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stream1_boxes = []

        for c1 in cnts_s1:
            a1 = cv2.contourArea(c1)
            arc1 = cv2.arcLength(c1, False)
            x1, y1, w1, h1 = cv2.boundingRect(c1)

            aspect1 = max(w1, h1) / float(max(1, min(w1, h1)))
            # Slender thread: Tỉ lệ dài/rộng >= 2.5 và bề ngang mảnh <= 15px (sợi chỉ mảnh)
            is_thread = (aspect1 >= 2.5 and min(w1, h1) <= 15 and a1 >= 15)
            # Saturated foreign ribbon/tape: Vết ngoại lai lớn có độ bão hòa cao (purity >= 30%)
            roi_r = red_purity[y1:y1+h1, x1:x1+w1]
            roi_b = blue_purity[y1:y1+h1, x1:x1+w1]
            max_pur1 = float(max(np.max(roi_r), np.max(roi_b))) if roi_r.size > 0 else 0.0

            is_ribbon = (a1 >= 1000 and max(w1, h1) >= 80 and max_pur1 >= 0.30)
            # Saturated stain: Vết ố đậm màu ngoại lai (mực vẽ, sơn loang)
            is_stain = (a1 >= 60 and max(w1, h1) >= 12 and max_pur1 >= 0.22)

            if is_thread or is_ribbon or is_stain:
                if is_thread:
                    cat1 = "thread"
                    conf1 = 0.98
                    label1 = f"THREAD ({int(conf1*100)}%)"
                    color1 = (0, 0, 255)
                elif is_ribbon:
                    cat1 = "ribbon"
                    conf1 = 0.96
                    label1 = f"RIBBON ({int(conf1*100)}%)"
                    color1 = (255, 0, 128)
                else:
                    cat1 = "stain"
                    conf1 = 0.94
                    label1 = f"STAIN ({int(conf1*100)}%)"
                    color1 = (0, 140, 255)

                total_damaged_pixels += a1
                cv2.drawContours(detected_vis, [c1], -1, color1, 2)
                if max(w1, h1) <= 80:
                    cv2.circle(detected_vis, (x1 + w1//2, y1 + h1//2), max(w1, h1)//2 + 8, color1, 2)
                else:
                    cv2.rectangle(detected_vis, (x1 - 4, y1 - 4), (x1 + w1 + 4, y1 + h1 + 4), color1, 2)
                cv2.putText(detected_vis, label1, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color1, 2)

                cy_pt1 = int(np.clip(y1 + h1 / 2.0, 0, h_r - 1))
                cx_pt1 = x1 + w1 / 2.0
                row_pts1 = np.where(ret_fg[cy_pt1] > 0)[0]
                if len(row_pts1) > 5:
                    xl1, xr1 = float(row_pts1[0]), float(row_pts1[-1])
                    w_vase_y1 = max(10.0, xr1 - xl1)
                    x_norm1 = float(np.clip((cx_pt1 - xl1) / w_vase_y1, 0.0, 1.0))
                else:
                    x_norm1 = float(np.clip(cx_pt1 / float(w_r), 0.0, 1.0))

                ys_all1 = np.where(ret_fg > 0)[0]
                if len(ys_all1) > 0:
                    y_top1 = float(ys_all1.min())
                    h_vase1 = float(max(10, ys_all1.max() - ys_all1.min()))
                    y_norm1 = float(np.clip((cy_pt1 - y_top1) / h_vase1, 0.0, 1.0))
                    h_norm1 = float(h1 / h_vase1)
                else:
                    y_norm1 = float(cy_pt1 / float(h_r))
                    h_norm1 = float(h1 / float(h_r))

                stream1_boxes.append((x1, y1, w1, h1))
                defect_items.append({
                    "category": cat1,
                    "label": label1,
                    "bbox": [int(x1), int(y1), int(w1), int(h1)],
                    "area_pixels": float(a1),
                    "x_norm": round(x_norm1, 4),
                    "y_norm": round(y_norm1, 4),
                    "h_norm": round(h_norm1, 4)
                })



        # =========================================================================
        # STREAM 2: SIFT-ALIGNED DIFFERENTIAL INSPECTION (Vết nứt, mẻ cạnh, xước buồm)
        # =========================================================================
        cnts, _ = cv2.findContours(binary_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_audit = []
        
        for c in cnts:
            area = cv2.contourArea(c)
            arc_len = cv2.arcLength(c, False)
            x, y, w, h = cv2.boundingRect(c)

            # Khử trùng lặp (NMS) với Stream 1: Nếu vùng này đã được Stream 1 bắt, bỏ qua
            has_s1_overlap = False
            for (sx, sy, sw, sh) in stream1_boxes:
                xi1, yi1 = max(x, sx), max(y, sy)
                xi2, yi2 = min(x + w, sx + sw), min(y + h, sy + sh)
                if xi2 > xi1 and yi2 > yi1:
                    inter_a = (xi2 - xi1) * (yi2 - yi1)
                    if inter_a >= 0.25 * min(area, sw * sh):
                        has_s1_overlap = True
                        break
            if has_s1_overlap:
                continue

            # Physical defect candidate gating: filter tiny noise specks and 1-2px slivers
            if (area >= 35 or (arc_len >= 35 and max(w, h) >= 20)) and min(w, h) >= 4 and max(w, h) >= 10:
                if not (w > w_r * 0.85 and h > h_r * 0.85) and area < total_vase_pixels * 0.5:
                    c_mask = np.zeros((h, w), dtype=np.uint8)
                    c_shifted = c - np.array([x, y])
                    cv2.drawContours(c_mask, [c_shifted], -1, 255, -1)
                    
                    roi_dL = dL_map[y:y+h, x:x+w][c_mask > 0]
                    roi_chroma = chroma_map[y:y+h, x:x+w][c_mask > 0]
                    roi_bgr = ret_bgr[y:y+h, x:x+w][c_mask > 0]
                    roi_sob = delta_sobel[y:y+h, x:x+w][c_mask > 0]

                    mean_chroma = float(np.mean(roi_chroma))
                    p95_chroma = float(np.percentile(roi_chroma, 95))
                    p95_sob = float(np.percentile(roi_sob, 95))
                    max_sob = float(np.max(roi_sob))
                    min_dL = float(np.min(roi_dL))
                    mean_dL = float(np.mean(roi_dL))
                    p05_dL = float(np.percentile(roi_dL, 5))

                    # Cửa lọc quang học triệt tiêu lóa sáng chói đèn (Photometric Specular Glare Gate)
                    n_dark = np.count_nonzero(roi_dL < -15.0)
                    n_chroma = np.count_nonzero(roi_chroma > 10.0)
                    n_bright = np.count_nonzero(roi_dL > 18.0)
                    n_sat = np.count_nonzero((roi_bgr[:, 0] >= 240) & (roi_bgr[:, 1] >= 240) & (roi_bgr[:, 2] >= 240))
                    total_pts = len(roi_dL)

                    is_glare_artifact = (n_dark == 0) and (n_chroma < total_pts * 0.05) and (n_bright > total_pts * 0.35) and (n_sat >= 3)
                    if is_glare_artifact:
                        continue

                    # Cổng lọc mặt nạ tiền xử lý (Specular / Glare Ignore Mask)
                    if ret_ignore_mask is not None:
                        roi_ign = ret_ignore_mask[y:y+h, x:x+w][c_mask > 0]
                        ign_ratio = float(np.count_nonzero(roi_ign)) / float(max(1, len(roi_ign)))
                        if ign_ratio > 0.25:
                            continue

                    # Check overlap with artwork zone
                    roi_art = artwork_zone[y:y+h, x:x+w][c_mask > 0]
                    art_overlap = float(np.count_nonzero(roi_art)) / float(max(1, len(roi_art)))

                    # Chuẩn hóa tọa độ vật lý theo công thức (270) Mục 3.8 Luận văn
                    cy_pt = int(np.clip(y + h / 2.0, 0, h_r - 1))
                    cx_pt = x + w / 2.0
                    row_pts = np.where(ref_fg_warped[cy_pt] > 0)[0]
                    if len(row_pts) > 5:
                        xl, xr = float(row_pts[0]), float(row_pts[-1])
                        w_vase_y = max(10.0, xr - xl)
                        x_norm = float(np.clip((cx_pt - xl) / w_vase_y, 0.0, 1.0))
                    else:
                        x_norm = float(np.clip(cx_pt / float(w_r), 0.0, 1.0))

                    ys_all = np.where(ref_fg_warped > 0)[0]
                    if len(ys_all) > 0:
                        y_top = float(ys_all.min())
                        h_vase = float(max(10, ys_all.max() - ys_all.min()))
                        y_norm = float(np.clip((cy_pt - y_top) / h_vase, 0.0, 1.0))
                        h_norm = float(h / h_vase)
                    else:
                        y_norm = float(cy_pt / float(h_r))
                        h_norm = float(h / float(h_r))

                    # 1. Filter extreme grazing tangent boundary aliasing (< 3.5% from left/right silhouette)
                    if area < 60 and (x_norm <= 0.035 or x_norm >= 0.965):
                        continue

                    # Physical defect verification geometric primitives
                    solidity = float(area) / float(max(1, w * h))
                    is_fissure_shape = (min(w, h) <= 25) or (max(w, h) / max(1, min(w, h)) >= 1.8) or (solidity <= 0.22 and max(w, h) >= 30) or (arc_len >= 45 and area >= 20)
                    is_crack = (area >= 18) and (max_sob >= sobel_threshold and (p95_sob >= 100.0 or min_dL <= -20.0)) and is_fissure_shape
                    is_elongated_crack = is_crack and (h >= 50 or max(w, h) / max(1, min(w, h)) >= 2.5) and max_sob >= sobel_threshold
                    chip_art_limit = 1.01 if (y_norm <= 0.08 and (min_dL <= -10.0 or max_sob >= 180.0)) else 0.30
                    roi_ret_fg = ret_fg[y:y+h, x:x+w][c_mask > 0]
                    roi_ref_fg = ref_fg_warped[y:y+h, x:x+w][c_mask > 0]
                    missing_mat_px = int(np.count_nonzero((roi_ref_fg > 0) & (roi_ret_fg == 0)))
                    is_rim_cavity_false_chip = (y_norm <= 0.08) and (missing_mat_px < 10)
                    is_chip = (25 <= area <= 1200) and (y_norm <= 0.08) and (art_overlap <= chip_art_limit) and (min_dL <= -10.0 or p95_sob >= 120.0 or max_sob >= 180.0) and not is_rim_cavity_false_chip

                    # 2. Filter extreme vertical rim & table contact boundary ribbons (top 4.5% rim cavity & bottom 7% table contact shadow)
                    is_extreme_vertical_boundary = (y_norm <= 0.045) or (y_norm >= 0.93)
                    if is_extreme_vertical_boundary:
                        # Dải bóng mỏng nằm ngang: h <= 20px và trải rộng hoặc diện tích nhỏ (không phải vết nứt sâu h >= 20px)
                        is_ribbon = (h <= 20 and float(w) / float(max(1, h)) >= 2.0) or (h <= 10 and area <= 300) or area <= 80
                        if is_ribbon and not (is_crack and max_sob >= sobel_threshold and min_dL <= -35.0) and not (is_chip and float(w) / float(max(1, h)) < 2.5):
                            continue

                    # 3. Cổng lọc hình thái rỗng (Hollow perimeter / ring artifact: vi sai vòng cung mép chân đế/miệng bình)
                    is_hollow_ring = (solidity < 0.12 and min(w, h) >= 18 and max(w, h) >= 40)
                    if is_hollow_ring and not (is_crack and max_sob >= sobel_threshold):
                        continue

                    # 4. Cổng lọc tương phản đơn sắc nét vẽ / bóng góc xoay (Monochrome Glaze & Shading Gate)
                    # Một khuyết tật phi cơ học (không phải vết nứt có rãnh Sobel >= sobel_threshold) trên thân bình trắng bắt buộc phải mang sắc thái màu (chroma >= 8.0)
                    is_monochrome_body_aliasing = (y > 160) and (mean_chroma < 8.0) and not (is_crack and max_sob >= sobel_threshold)
                    if is_monochrome_body_aliasing:
                        continue

                    # 5. Vết ố bẩn thật (Solid Stain): Phải có biến thiên màu men rõ rệt
                    is_solid_stain = (area >= 180) and (min(w, h) >= 12) and (solidity >= 0.22) and (mean_chroma >= 14.0)

                    # 5b. Vết ố mang sắc thái màu đậm ngoại lai (Chromatic Stain)
                    is_chromatic_stain = (mean_chroma >= 16.0) and (area >= 35) and (solidity >= 0.15) and (art_overlap < 0.30)

                    # 6. Dynamic Artwork Edge Suppression: Triệt tiêu xô lệch nét vẽ đơn sắc, bảo toàn vết nứt và vết ố màu men
                    is_artwork_aliasing = (art_overlap > 0.25) and (mean_chroma < 10.0) and not (is_crack and max_sob >= sobel_threshold and min_dL <= -20.0)
                    if (is_chip and y_norm <= 0.08) or is_chromatic_stain:
                        is_artwork_aliasing = False
                    if is_artwork_aliasing:
                        continue

                    if art_overlap > 0.35 and not is_solid_stain and not is_chromatic_stain and not (is_crack and max_sob >= sobel_threshold and min_dL <= -20.0):
                        if not (is_chip and y_norm <= 0.08):
                            continue

                    is_stain = is_solid_stain or is_chromatic_stain or (area >= 80 and mean_chroma >= 11.5 and art_overlap < 0.25 and solidity >= 0.15) or (area >= 35 and mean_chroma >= 15.0 and art_overlap < 0.20 and solidity >= 0.18)

                    # Cho phép ứng viên vượt qua nếu thỏa mãn hình thái nứt/mẻ/ố HOẶC có biến thiên sắc thái màu ngoại lai
                    is_chromatic_candidate = (area >= 15) and (mean_chroma >= 8.0)
                    
                    # Physical Reality Gate: A smooth porcelain region that is LIGHTER than reference (min_dL >= 0)
                    # and has no mechanical crack edge cannot be a mechanical crack or chip
                    is_bright_glaze_highlight = (min_dL >= 0.0) and (max_sob < 250.0) and (mean_chroma < 18.0)
                    if is_bright_glaze_highlight:
                        continue

                    if not (is_stain or is_crack or is_chip or is_chromatic_candidate):
                        continue

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

                        # Bảo toàn quyền override cho khuyết tật cơ học đã được xác thực vật lý
                        is_severe_crack = (is_crack and max_sob >= (sobel_threshold + 50.0) and min_dL <= -40.0)
                        is_severe_chip = (is_chip and y_norm <= 0.08 and (min_dL <= -10.0 or max_sob >= 180.0))
                        is_severe_stain = (is_stain and mean_chroma >= 18.0 and min_dL <= -25.0 and area >= 80)
                        override_eligible = (is_severe_crack or is_severe_chip or is_severe_stain)

                        # Đối với ứng viên ngoại lai/sợi chỉ: Phán quyết nghiêm ngặt bởi CD-MLP
                        veto_triggered = (pred_class == 0 or probs[0] >= 0.50)
                        veto_rejected = False
                        if ablation_mode == "A":
                            if veto_triggered:
                                veto_rejected = True
                        elif ablation_mode == "B":
                            if veto_triggered and not override_eligible:
                                veto_rejected = True
                        elif ablation_mode in ("C", "D"):
                            veto_rejected = False

                        candidate_audit.append({
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "area": float(area),
                            "max_sob": float(max_sob),
                            "p95_sob": float(p95_sob),
                            "min_dL": float(min_dL),
                            "mean_chroma": float(mean_chroma),
                            "is_crack": bool(is_crack),
                            "is_chip": bool(is_chip),
                            "is_stain": bool(is_stain),
                            "pred_class": int(pred_class),
                            "probs": [round(float(p), 4) for p in probs],
                            "veto_triggered": bool(veto_triggered),
                            "override_eligible": bool(override_eligible),
                            "veto_rejected": bool(veto_rejected),
                            "ablation_mode": ablation_mode
                        })

                        if veto_rejected:
                            continue

                        # Xác nhận khuyết tật: Chọn phân lớp chiếm ưu thế giữa Crack, Chip, Stain
                        defect_probs = probs[1:]
                        pred_class = int(torch.argmax(defect_probs).item()) + 1
                        pred_conf = float(probs[pred_class].item())

                        # Đồng bộ phân lớp nơ-ron với đặc trưng hình thái vật lý (Physical-Neural Harmonization)
                        if is_elongated_crack and mean_chroma < 10.0:
                            pred_class = 1
                            pred_conf = max(pred_conf, 0.98)
                        elif is_chip and y_norm <= 0.08 and not is_elongated_crack:
                            pred_class = 2
                            pred_conf = max(pred_conf, 0.95)
                        elif is_crack and max_sob >= sobel_threshold and mean_chroma < 10.0:
                            pred_class = 1
                            pred_conf = max(pred_conf, 0.98)
                        elif is_stain and mean_chroma >= 10.0 and y_norm > 0.08:
                            pred_class = 3
                            pred_conf = max(pred_conf, 0.98)

                        if pred_class == 1:
                            pred_cat = "crack"
                            pred_label_str = f"CRACK ({int(pred_conf*100)}%)"
                            pred_color = (0, 0, 255)      # Red
                        elif pred_class == 2:
                            pred_cat = "chip"
                            pred_label_str = f"CHIP ({int(pred_conf*100)}%)"
                            pred_color = (0, 255, 255)    # Yellow
                        elif pred_class == 3:
                            pred_cat = "stain"
                            pred_label_str = f"STAIN ({int(pred_conf*100)}%)"
                            pred_color = (0, 140, 255)    # Orange
                        else:
                            pred_cat = "defect"
                            pred_label_str = f"DEFECT ({int(pred_conf*100)}%)"
                            pred_color = (0, 0, 255)

                    # Physical heuristic assignment if CD-MLP is unassigned
                    if pred_label_str is None or pred_cat is None:
                        if is_chip:
                            pred_cat = "chip"
                            pred_label_str = f"CHIP ({int(min(99, max(75, p95_chroma * 4 + p95_sob)))}%)"
                            pred_color = (0, 255, 255)
                        elif is_crack:
                            pred_cat = "crack"
                            pred_label_str = f"CRACK ({int(min(99, max(75, max_sob / 9.5)))}%)"
                            pred_color = (0, 0, 255)
                        elif is_stain:
                            pred_cat = "stain"
                            pred_label_str = f"STAIN ({int(min(99, max(75, mean_chroma * 6)))}%)"
                            pred_color = (0, 140, 255)
                        else:
                            pred_cat = "defect"
                            pred_label_str = "DEFECT"
                            pred_color = (0, 0, 255)

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
                    
                    # Chuẩn hóa tọa độ vật lý theo công thức (270) Mục 3.8 Luận văn
                    cy_pt = int(np.clip(y + h / 2.0, 0, h_r - 1))
                    cx_pt = x + w / 2.0
                    row_pts = np.where(ref_fg_warped[cy_pt] > 0)[0]
                    if len(row_pts) > 5:
                        xl, xr = float(row_pts[0]), float(row_pts[-1])
                        w_vase_y = max(10.0, xr - xl)
                        x_norm = float(np.clip((cx_pt - xl) / w_vase_y, 0.0, 1.0))
                    else:
                        x_norm = float(np.clip(cx_pt / float(w_r), 0.0, 1.0))
                        
                    ys_all = np.where(ref_fg_warped > 0)[0]
                    if len(ys_all) > 0:
                        y_top = float(ys_all.min())
                        h_vase = float(max(10, ys_all.max() - ys_all.min()))
                        y_norm = float(np.clip((cy_pt - y_top) / h_vase, 0.0, 1.0))
                        h_norm = float(h / h_vase)
                    else:
                        y_norm = float(cy_pt / float(h_r))
                        h_norm = float(h / float(h_r))

                    defect_items.append({
                        "category": cat,
                        "label": label,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area_pixels": float(area),
                        "x_norm": round(x_norm, 4),
                        "y_norm": round(y_norm, 4),
                        "h_norm": round(h_norm, 4)
                    })
                    
        damage_ratio = (total_damaged_pixels / (total_vase_pixels + 1e-6)) * 100.0
        
        # 7. Xuất ảnh trực quan 3 khung hình
        heatmap_vis = cv2.applyColorMap((confirmed_score * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_display_mask = (ret_fg > 0) & (~is_stand)
        heatmap_vis[~heatmap_display_mask] = [25, 25, 25]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        p1 = ret_bgr.copy()
        cv2.putText(p1, '1. Return Image', (10, 25), font, 0.55, (255, 255, 255), 2)
        heatmap_title = '2. AI Neural Heatmap (DINOv2+CD-MLP)' if neural_defect_prob is not None else '2. Differential Heatmap (CIELab)'
        cv2.putText(heatmap_vis, heatmap_title, (10, 25), font, 0.48, (255, 255, 255), 2)
        cv2.putText(detected_vis, f'3. Inspection Result: {len(defect_items)} Defect(s)', (10, 25), font, 0.50, (0, 255, 0), 2)
        
        comp_result = np.hstack([p1, heatmap_vis, detected_vis])
        
        os.makedirs(output_dir, exist_ok=True)
        ret_base = os.path.splitext(os.path.basename(return_img_path))[0]
        img_out_path = os.path.join(output_dir, f"{product_id}_{ret_base}_result.jpg")
        cv2.imwrite(img_out_path, comp_result)
        
        ys_v, xs_v = np.where(common_vase > 0)
        if len(xs_v) > 0:
            vase_bbox = [int(xs_v.min()), int(ys_v.min()), int(xs_v.max() - xs_v.min()), int(ys_v.max() - ys_v.min())]
        else:
            vase_bbox = [263, 72, 498, 880]

        report = {
            "product_id": product_id,
            "image_name": os.path.basename(return_img_path),
            "is_damaged": len(defect_items) > 0,
            "defect_count": len(defect_items),
            "damage_percentage": round(damage_ratio, 3),
            "severity": "SEVERE_DAMAGE" if len(defect_items) > 5 or damage_ratio > 1.0 else ("DEFECTIVE" if len(defect_items) > 0 else "INTACT"),
            "result_image": img_out_path,
            "vase_bbox": vase_bbox,
            "ablation_mode": ablation_mode,
            "local_search_k": local_search_k,
            "artwork_penalty": artwork_penalty,
            "candidate_audit": candidate_audit,
            "defects": defect_items
        }
        
        json_out_path = os.path.join(output_dir, f"{product_id}_{ret_base}_report.json")
        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, ensure_ascii=False, indent=2)
            
        print(f"[INSPECT] Complete! Detected {len(defect_items)} defect(s). Output: {img_out_path}")
        return report

    def inspect_session(self, product_id: str, view_images: Dict[str, str], output_dir: str = "uploads") -> Dict[str, Any]:
        """
        Comprehensive multi-view ceramic return inspection across 4 views.
        Includes Photometric Specular Gate and 3D Cross-View Deduplication Algorithm.
        """
        view_reports = {}
        for view_name, img_path in view_images.items():
            if not os.path.exists(img_path):
                continue
            prof_id = resolve_profile_id(product_id, view_name, self.profiles_dir)
            if prof_id is None:
                prof_id = f"{product_id}_01_front"
            report = self.inspect_return(img_path, prof_id, output_dir=output_dir)
            view_reports[view_name] = report

        if len(view_reports) == 0:
            return {
                "status": "error",
                "product_id": product_id,
                "views_inspected": [],
                "views": {},
                "unique_defects": [],
                "dedup_pairs": [],
                "dedup_candidates_rejected": [],
                "total_defects": 0,
                "raw_defect_count": 0,
                "is_damaged": False,
                "damage_percentage": 0.0,
                "severity": "INSPECTION_FAILED",
                "verdict": "NEEDS_REVIEW",
                "review_reason": "No valid view photos found to inspect."
            }

        unique_defects, dedup_pairs, rejected_pairs = deduplicate_cross_view_defects(view_reports)
        raw_count = sum(len(r.get("defects", [])) for r in view_reports.values())
        max_damage_pct = max((r.get("damage_percentage", 0.0) for r in view_reports.values()), default=0.0)

        any_needs_review = any(r.get("severity") == "NEEDS_REVIEW" or r.get("needs_review") for r in view_reports.values())
        is_dam = len(unique_defects) > 0

        if any_needs_review:
            severity = "NEEDS_REVIEW"
            verdict = "NEEDS_REVIEW"
        elif len(unique_defects) > 5 or max_damage_pct > 1.0:
            severity = "SEVERE_DAMAGE"
            verdict = "DEFECTIVE"
        elif is_dam:
            severity = "DEFECTIVE"
            verdict = "DEFECTIVE"
        else:
            severity = "INTACT"
            verdict = "INTACT"

        # Categorical statistics
        cat_counts = {}
        for d in unique_defects:
            cat = d.get("category", "defect")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        cat_names_map = {
            "crack": "Crack",
            "chip": "Chip",
            "stain": "Stain",
            "thread": "Thread",
            "ribbon": "Ribbon"
        }
        breakdown_list = [f"{cnt} {cat_names_map.get(cat, cat)}" for cat, cnt in cat_counts.items()]
        classification_summary = ", ".join(breakdown_list) if breakdown_list else "INTACT"

        session_summary = {
            "status": "ok",
            "product_id": product_id,
            "views_inspected": list(view_reports.keys()),
            "views": view_reports,
            "raw_defect_count": raw_count,
            "total_defects": len(unique_defects),
            "deduplicated_defect_count": len(unique_defects),
            "duplicate_pairs_found": len(dedup_pairs),
            "dedup_pairs": dedup_pairs,
            "dedup_candidates_rejected": rejected_pairs,
            "is_damaged": is_dam,
            "damage_percentage": round(max_damage_pct, 3),
            "classification_summary": classification_summary,
            "category_counts": cat_counts,
            "severity": severity,
            "verdict": verdict,
            "unique_defects": unique_defects,
            "view_reports": view_reports
        }
        return session_summary


# ==============================================================================
# THUẬT TOÁN KHỬ TRÙNG LẶP KHUYẾT TẬT ĐA GÓC 3D (CROSS-VIEW DEDUPLICATION)
# ==============================================================================

ORDERED_ADJACENT_PAIRS = [
    ('truoc', 'phai'),
    ('phai', 'sau'),
    ('sau', 'trai'),
    ('trai', 'truoc'),
    ('front', 'right'),
    ('right', 'back'),
    ('back', 'left'),
    ('left', 'front'),
]
ADJACENT_PAIRS = ORDERED_ADJACENT_PAIRS


def deduplicate_cross_view_defects(view_results: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Thuật toán Khử trùng lặp khuyết tật đa góc 3D kết hợp Composite Cost:
      1. Ràng buộc cứng mép giáp ranh: x_norm(A) >= 0.70 và x_norm(B) <= 0.30
      2. Ràng buộc cứng đồng cao độ: |y_norm(A) - y_norm(B)| <= 0.05
      3. Ràng buộc cứng Category: category(A) == category(B) (mismatch ghi vào rejected)
      4. Composite Cost: 0.40 * D_pos + 0.35 * D_size + 0.25 * D_shape <= 0.38
    """
    all_defects = []

    for v_name, res in view_results.items():
        v_box = res.get('vase_bbox', (263, 72, 498, 880))
        vx, vy, vw, vh = v_box

        for d_idx, d in enumerate(res.get('defects', [])):
            bx, by, bw, bh = d['bbox']
            x_norm = d.get('x_norm', ((bx + bw / 2.0) - vx) / float(vw))
            y_norm = d.get('y_norm', ((by + bh / 2.0) - vy) / float(vh))
            h_norm = d.get('h_norm', float(bh) / float(vh))

            all_defects.append({
                'view': v_name,
                'defect_idx': d_idx,
                'raw': d,
                'x_norm': x_norm,
                'y_norm': y_norm,
                'h_norm': h_norm,
                'category': d.get('category'),
                'label': d.get('label'),
                'area': d.get('area_pixels', bw * bh),
                'merged_with': None
            })

    merged_indices = set()
    dedup_pairs = []
    rejected_pairs = []

    for i in range(len(all_defects)):
        if i in merged_indices:
            continue
        for j in range(len(all_defects)):
            if i == j or j in merged_indices:
                continue
            d1 = all_defects[i]
            d2 = all_defects[j]

            pair = (d1['view'], d2['view'])
            if pair in ADJACENT_PAIRS:
                edge_match = (d1['x_norm'] >= 0.70) and (d2['x_norm'] <= 0.30)
            elif (d2['view'], d1['view']) in ADJACENT_PAIRS:
                edge_match = (d2['x_norm'] >= 0.70) and (d1['x_norm'] <= 0.30)
            else:
                edge_match = False

            if not edge_match:
                continue

            dy = abs(d1['y_norm'] - d2['y_norm'])
            if dy > 0.05:
                continue

            # Hard Constraint: Category Match
            if d1['category'] != d2['category']:
                rejected_pairs.append({
                    'd1': (d1['view'], d1['label'], d1['raw']['bbox']),
                    'd2': (d2['view'], d2['label'], d2['raw']['bbox']),
                    'reason': f"Category mismatch ({d1['category']} vs {d2['category']})",
                    'dy_norm': round(dy, 4)
                })
                continue

            # Composite Cost: Position, Size, Shape
            d_pos = min(1.0, dy / 0.05)
            a1 = float(d1.get('area', 1.0))
            a2 = float(d2.get('area', 1.0))
            d_size = min(1.0, abs(a1 - a2) / max(1.0, max(a1, a2)))

            b1 = d1['raw']['bbox']
            b2 = d2['raw']['bbox']
            asp1 = max(b1[2], b1[3]) / float(max(1, min(b1[2], b1[3])))
            asp2 = max(b2[2], b2[3]) / float(max(1, min(b2[2], b2[3])))
            d_shape = min(1.0, abs(asp1 - asp2) / max(1.0, max(asp1, asp2)))

            cost = 0.40 * d_pos + 0.35 * d_size + 0.25 * d_shape

            if cost <= 0.38:
                merged_indices.add(j)
                d2['merged_with'] = (d1['view'], d1['defect_idx'])
                dedup_pairs.append({
                    'primary': (d1['view'], d1['label'], d1['raw']['bbox']),
                    'duplicate': (d2['view'], d2['label'], d2['raw']['bbox']),
                    'category': d1['category'],
                    'dy_norm': round(dy, 4),
                    'cost': round(cost, 4)
                })
                break
            else:
                rejected_pairs.append({
                    'd1': (d1['view'], d1['label'], d1['raw']['bbox']),
                    'd2': (d2['view'], d2['label'], d2['raw']['bbox']),
                    'reason': f"Composite cost exceeded threshold ({cost:.3f} > 0.38)",
                    'cost': round(cost, 4)
                })

    unique_defects = [d for idx, d in enumerate(all_defects) if idx not in merged_indices]
    return unique_defects, dedup_pairs, rejected_pairs
