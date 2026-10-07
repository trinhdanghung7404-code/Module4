"""Detailed Granular 2-Layer Debug Inspection with Subfolders.

Focus: EXACT local defect localization and transparent evidence.
Structure:
  v2/debug/
    ├── 01_keypoints/       (SuperPoint keypoints on both images)
    ├── 02_mesh/            (Delaunay triangles wireframe on both images)
    ├── 03_layer1_structure/ (Canny edges, new edge heatmaps, structure damage)
    ├── 04_layer2_color/    (Lab delta_E heatmaps, color defects)
    ├── 05_fusion/          (Combined mask & overlay on both images)
    └── 06_defect_crops/    (Zoom-in patch comparisons for every flagged triangle)
"""

import os
import sys
import time
import cv2
import numpy as np
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder, Vertex, Triangle
from step3_pipeline import canonical_triangle_patch

# -----------------------------------------------------------------------------
# CẤU HÌNH KIỂM SOÁT PHƯƠNG PHÁP LAYER 2:
# -----------------------------------------------------------------------------
ENABLE_SUBMESH_VERIFICATION = False  # Tắt Sub-mesh để tập trung tối ưu Phương pháp 1 (kênh ab) & Phương pháp 2 (kênh L)

# -----------------------------------------------------------------------------
# DINOv2 DENSE FEATURE EXTRACTOR FOR MESH TRIANGLES
# -----------------------------------------------------------------------------
_dinov2_processor = None
_dinov2_model = None

def get_dinov2_model():
    global _dinov2_processor, _dinov2_model
    if _dinov2_model is None:
        from transformers import AutoImageProcessor, AutoModel
        _dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _dinov2_model = AutoModel.from_pretrained("facebook/dinov2-small")
        _dinov2_model.eval()
    return _dinov2_processor, _dinov2_model

def extract_dinov2_dense_features(img_bgr: np.ndarray, max_dim: int = 800):
    import torch
    processor, model = get_dinov2_model()
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    feat_h = int(round(h * scale))
    feat_w = int(round(w * scale))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        inp = processor(images=rgb, return_tensors="pt")
        out = model(**inp, output_hidden_states=True)
        # Layer 9: Tầng đặc trưng kết cấu bề mặt tối ưu (nhạy cảm với rãnh nứt, vết sẹo, độ nhám men)
        tok = out.hidden_states[9][:, 1:, :]
        grid_size = int(round(tok.shape[1] ** 0.5))
        f = tok.reshape(1, grid_size, grid_size, 384).permute(0, 3, 1, 2)
        dense = torch.nn.functional.interpolate(f, size=(feat_h, feat_w), mode="bilinear", align_corners=False)[0]
        dense = torch.nn.functional.normalize(dense, p=2, dim=0).cpu().numpy()
        return dense, (feat_h, feat_w)


def micro_align_patches(patch_p: np.ndarray, patch_r: np.ndarray, mask: np.ndarray, max_shift: int = 4, extra_patch: np.ndarray = None):
    """Compensate for minor 3D surface curvature distortion (+/- 4px shift)."""
    p_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY) if patch_p.ndim == 3 else patch_p
    r_gray = cv2.cvtColor(patch_r, cv2.COLOR_BGR2GRAY) if patch_r.ndim == 3 else patch_r
    h, w = p_gray.shape[:2]

    best_corr = -1.0
    best_shift = (0, 0)
    valid = mask > 0

    p_vals = p_gray[valid].astype(np.float32)
    p_std = np.std(p_vals)
    if p_std < 1e-4 or len(p_vals) < 16:
        if extra_patch is not None:
            return patch_r, extra_patch
        return patch_r

    p_norm = (p_vals - np.mean(p_vals)) / p_std

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_r = cv2.warpAffine(r_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            r_vals = shifted_r[valid].astype(np.float32)
            r_std = np.std(r_vals)
            if r_std < 1e-4:
                continue
            r_norm = (r_vals - np.mean(r_vals)) / r_std
            corr = float(np.mean(p_norm * r_norm))
            if corr > best_corr:
                best_corr = corr
                best_shift = (dx, dy)

    dx, dy = best_shift
    if dx == 0 and dy == 0:
        if extra_patch is not None:
            return patch_r, extra_patch
        return patch_r
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_r = cv2.warpAffine(patch_r, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if extra_patch is not None:
        aligned_extra = cv2.warpAffine(extra_patch, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return aligned_r, aligned_extra
    return aligned_r


def compute_orientation_profile(gray_img: np.ndarray, mask: np.ndarray, min_mag: float = 25.0):
    """
    Compute dominant orientation peaks and 8-bin histogram for a patch.
    Returns:
      hist: 8-bin normalized orientation distribution (0-180 deg)
      dominant_angles: list of prominent peak angles in degrees
      coherence: measure of directional strength (0 to 1)
    """
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    valid_edge = (mask > 0) & (mag >= min_mag)
    if np.sum(valid_edge) < 15:
        return np.zeros(8, dtype=np.float32), [], 0.0

    weights = mag[valid_edge]
    angles = ang[valid_edge]

    hist, bin_edges = np.histogram(angles, bins=8, range=(0, 180), weights=weights)
    total_w = np.sum(hist)
    if total_w > 0:
        hist = hist / total_w

    peaks = []
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for idx, w in enumerate(hist):
        if w >= 0.15:
            peaks.append(float(bin_centers[idx]))

    coherence = float(np.max(hist)) if len(hist) > 0 else 0.0
    return hist, peaks, coherence


def is_angle_matching_any(angle: float, reference_peaks: list, tol_deg: float = 25.0) -> bool:
    """Check if an angle matches any reference peak angle within tolerance."""
    if not reference_peaks:
        return False
    for p in reference_peaks:
        diff = abs(angle - p)
        diff = min(diff, 180.0 - diff)
        if diff <= tol_deg:
            return True
def draw_text_with_shadow(img: np.ndarray, text: str, pos: tuple, font_scale: float = 0.35, color: tuple = (0, 255, 0), thickness: int = 1):
    """Draw text with a black outline/shadow so it is clearly readable on both dark and bright backgrounds."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def compute_triangle_color_distribution(p_lab: np.ndarray, r_lab: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    LAYER 2: Phân tích phân bố màu sắc (Color Distribution) trong tam giác.
    Kháng hoàn toàn sai lệch vị trí hoa văn vi mô (Sub-pixel Misalignment Invariant).
    """
    from scipy.stats import wasserstein_distance
    if not np.any(valid_mask):
        return {
            "w_l": 0.0, "w_chroma": 0.0, "w_fused": 0.0,
            "bhat_ab": 0.0, "lost_pattern_ratio": 0.0, "new_color_ratio": 0.0,
            "hist_p_l": np.zeros(24), "hist_r_l": np.zeros(24),
            "hist_p_c": np.zeros(24), "hist_r_c": np.zeros(24),
        }
    
    # 1. Trích xuất mảng pixel Lab hợp lệ
    p_l = p_lab[..., 0][valid_mask]
    r_l = r_lab[..., 0][valid_mask]
    p_a = p_lab[..., 1][valid_mask]
    r_a = r_lab[..., 1][valid_mask]
    p_b = p_lab[..., 2][valid_mask]
    r_b = r_lab[..., 2][valid_mask]

    # Chuẩn hóa L theo median để triệt tiêu chênh sáng
    med_lp = float(np.median(p_l))
    med_lr = float(np.median(r_l))
    r_l_norm = r_l - med_lr + med_lp

    # Độ bão hòa sắc tố (Chroma)
    p_chroma = np.sqrt((p_a - 128.0)**2 + (p_b - 128.0)**2)
    r_chroma = np.sqrt((r_a - 128.0)**2 + (r_b - 128.0)**2)

    # 2. Khoảng cách Wasserstein (Earth Mover's Distance)
    w_l = float(wasserstein_distance(p_l, r_l_norm))
    w_a = float(wasserstein_distance(p_a, r_a))
    w_b = float(wasserstein_distance(p_b, r_b))
    w_chroma = float(np.sqrt(w_a**2 + w_b**2))
    w_fused = float(np.sqrt(0.5 * (w_l ** 2) + 1.0 * (w_chroma ** 2)))

    # 3. Phân bố 2D Chroma (a, b) Histogram (16x16 bins)
    # 3. Phân bố 2D Chroma (a, b) Histogram (16x16 bins) kèm Gaussian smoothing
    mask_u8 = valid_mask.astype(np.uint8) * 255
    hist_p_ab = cv2.calcHist([p_lab], [1, 2], mask_u8, [16, 16], [0, 256, 0, 256])
    hist_r_ab = cv2.calcHist([r_lab], [1, 2], mask_u8, [16, 16], [0, 256, 0, 256])

    pdf_p = hist_p_ab / (np.sum(hist_p_ab) + 1e-8)
    pdf_r = hist_r_ab / (np.sum(hist_r_ab) + 1e-8)

    # Gaussian smoothing trên 2D histogram để chống hiện tượng nhảy bin rời rạc (bin boundary jitter)
    pdf_p_smooth = cv2.GaussianBlur(pdf_p, (3, 3), 0.75)
    pdf_r_smooth = cv2.GaussianBlur(pdf_r, (3, 3), 0.75)

    # Khoảng cách Bhattacharyya trên phân bố sắc tố đã làm mượt
    bhat_ab = float(cv2.compareHist(pdf_p_smooth.astype(np.float32), pdf_r_smooth.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))

    # 4. Phân tích Bay màu hoa văn (Pattern Loss) & Mọc màu lạ (Stain Intrusion) theo bản chất Chroma
    # Hoa văn đậm trong Product có chroma >= 18
    p_pattern_mask = p_chroma >= 18.0
    p_pattern_cnt = int(np.sum(p_pattern_mask))

    if p_pattern_cnt >= 25:
        # Tỷ lệ pixel hoa văn bị sụt giảm độ bão hòa về men trắng (chroma < 10)
        lost_px = np.sum(r_chroma[p_pattern_mask] < 10.0)
        lost_pattern_ratio = float(lost_px) / float(p_pattern_cnt)
    else:
        lost_pattern_ratio = 0.0

    # Mọc màu lạ: Product là men trơn nhạt (chroma < 10) nhưng Return mọc mảng màu đậm (chroma >= 20)
    p_plain_mask = p_chroma < 10.0
    p_plain_cnt = int(np.sum(p_plain_mask))
    if p_plain_cnt >= 25:
        new_px = np.sum(r_chroma[p_plain_mask] >= 20.0)
        new_color_ratio = float(new_px) / float(p_plain_cnt)
    else:
        new_color_ratio = 0.0

    # Vết mực đen / dị vật tối màu trên kênh L (Achromatic Dark Stain):
    # Nền Product sáng (p_l >= 85) nhưng Return sụt giảm mạnh độ sáng (r_l_norm < 75 và p_l - r_l_norm >= 22)
    dark_stain_mask = (p_l >= 85.0) & (r_l_norm < 75.0) & ((p_l - r_l_norm) >= 22.0)
    dark_stain_cnt = int(np.sum(dark_stain_mask))
    dark_stain_ratio = float(dark_stain_cnt) / float(len(p_l)) if len(p_l) > 0 else 0.0

    # 5. Profile 1D histogram 24 bins để vẽ biểu đồ so sánh trong ảnh debug
    bins_l = np.linspace(0, 255, 25)
    hist_p_l, _ = np.histogram(p_l, bins=bins_l, density=True)
    hist_r_l, _ = np.histogram(r_l_norm, bins=bins_l, density=True)

    bins_c = np.linspace(0, 80, 25)
    hist_p_c, _ = np.histogram(p_chroma, bins=bins_c, density=True)
    hist_r_c, _ = np.histogram(r_chroma, bins=bins_c, density=True)

    return {
        "w_l": w_l,
        "w_chroma": w_chroma,
        "w_fused": w_fused,
        "bhat_ab": bhat_ab,
        "lost_pattern_ratio": lost_pattern_ratio,
        "new_color_ratio": new_color_ratio,
        "dark_stain_ratio": dark_stain_ratio,
        "hist_p_l": hist_p_l,
        "hist_r_l": hist_r_l,
        "hist_p_c": hist_p_c,
        "hist_r_c": hist_r_c,
    }


def render_color_distribution_chart(dist_data: dict, w_chart: int = 250, h_chart: int = 250) -> np.ndarray:
    """Render a clean side-by-side color distribution comparison chart between Product and Return."""
    chart = np.zeros((h_chart, w_chart, 3), dtype=np.uint8)
    # Background frame
    cv2.rectangle(chart, (8, 30), (w_chart - 8, h_chart - 35), (25, 25, 25), -1)
    cv2.rectangle(chart, (8, 30), (w_chart - 8, h_chart - 35), (70, 70, 70), 1)

    # Grid lines
    cv2.line(chart, (8, 110), (w_chart - 8, 110), (45, 45, 45), 1)
    cv2.line(chart, (8, 160), (w_chart - 8, 160), (45, 45, 45), 1)

    # Plot Chroma distributions
    hp_c = dist_data.get("hist_p_c", np.zeros(24))
    hr_c = dist_data.get("hist_r_c", np.zeros(24))
    max_val = max(float(np.max(hp_c)), float(np.max(hr_c)), 1e-4)

    pts_p = []
    pts_r = []
    n_pts = len(hp_c)
    dx = (w_chart - 28) / max(1, n_pts - 1)
    for k in range(n_pts):
        x = int(14 + k * dx)
        yp = int(h_chart - 40 - (hp_c[k] / max_val) * 135)
        yr = int(h_chart - 40 - (hr_c[k] / max_val) * 135)
        pts_p.append((x, yp))
        pts_r.append((x, yr))

    # Product: Cyan curve, Return: Bright Orange curve
    cv2.polylines(chart, [np.array(pts_p, dtype=np.int32)], False, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.polylines(chart, [np.array(pts_r, dtype=np.int32)], False, (0, 120, 255), 2, cv2.LINE_AA)

    # Title & Legend
    cv2.putText(chart, "3. Color Distribution", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(chart, "P:Cyan  R:Orange", (w_chart - 115, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Metrics display at bottom
    w_c = dist_data.get("w_chroma", 0.0)
    w_l = dist_data.get("w_l", 0.0)
    bhat = dist_data.get("bhat_ab", 0.0)
    lost_p = dist_data.get("lost_pattern_ratio", 0.0) * 100.0
    new_c = dist_data.get("new_color_ratio", 0.0) * 100.0
    dark_s = dist_data.get("dark_stain_ratio", 0.0) * 100.0

    cv2.putText(chart, f"W_C:{w_c:.1f} W_L:{w_l:.1f} Bh:{bhat:.2f}", (8, h_chart - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    flag_color = (0, 255, 200) if (lost_p > 10 or new_c > 10 or dark_s > 10) else (180, 180, 180)
    cv2.putText(chart, f"Lost:{lost_p:.0f}% Dark:{dark_s:.0f}% New:{new_c:.0f}%", (8, h_chart - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, flag_color, 1, cv2.LINE_AA)

    return chart


def render_submesh_debug_card(product_img: np.ndarray, return_img: np.ndarray, t_item: dict, patch_data: list, w_card: int = 750, h_card: int = 500) -> np.ndarray:
    """Render a dedicated sub-mesh verification debug card."""
    i = t_item["id"]
    cluster = t_item.get("submesh_cluster", [i])
    pts_p_list = [patch_data[tid]["pts_p"] for tid in cluster]
    pts_r_list = [patch_data[tid]["pts_r"] for tid in cluster]
    pts_p_all = np.vstack(pts_p_list)
    pts_r_all = np.vstack(pts_r_list)

    h, w = product_img.shape[:2]
    min_x_p = max(0, int(np.min(pts_p_all[:, 0])) - 25)
    min_y_p = max(0, int(np.min(pts_p_all[:, 1])) - 25)
    max_x_p = min(w, int(np.max(pts_p_all[:, 0])) + 25)
    max_y_p = min(h, int(np.max(pts_p_all[:, 1])) + 25)

    min_x_r = max(0, int(np.min(pts_r_all[:, 0])) - 25)
    min_y_r = max(0, int(np.min(pts_r_all[:, 1])) - 25)
    max_x_r = min(w, int(np.max(pts_r_all[:, 0])) + 25)
    max_y_r = min(h, int(np.max(pts_r_all[:, 1])) + 25)

    crop_p = product_img[min_y_p:max_y_p, min_x_p:max_x_p].copy()
    crop_r = return_img[min_y_r:max_y_r, min_x_r:max_x_r].copy()

    is_cleared = bool(t_item.get("submesh_cleared_l2", False))
    target_color = (0, 255, 0) if is_cleared else (0, 0, 255)

    for tid in cluster:
        p_poly = (patch_data[tid]["pts_p"] - np.array([min_x_p, min_y_p])).astype(np.int32).reshape((-1, 1, 2))
        r_poly = (patch_data[tid]["pts_r"] - np.array([min_x_r, min_y_r])).astype(np.int32).reshape((-1, 1, 2))
        if tid == i:
            cv2.polylines(crop_p, [p_poly], True, target_color, 2, cv2.LINE_AA)
            cv2.polylines(crop_r, [r_poly], True, target_color, 2, cv2.LINE_AA)
        else:
            cv2.polylines(crop_p, [p_poly], True, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(crop_r, [r_poly], True, (0, 255, 255), 1, cv2.LINE_AA)

    res_w = w_card // 2
    res_h = h_card // 2
    crop_p_res = cv2.resize(crop_p, (res_w, res_h)) if crop_p.size > 0 else np.zeros((res_h, res_w, 3), dtype=np.uint8)
    crop_r_res = cv2.resize(crop_r, (res_w, res_h)) if crop_r.size > 0 else np.zeros((res_h, res_w, 3), dtype=np.uint8)

    cv2.putText(crop_p_res, f"Product Sub-mesh ({len(cluster)} tris)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(crop_r_res, f"Return Sub-mesh (Target Tri #{i})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    row_top = np.hstack([crop_p_res, crop_r_res])

    # Hàng dưới gồm 3 ô (250x250 mỗi ô)
    # Panel 1: Center Triangle Patch Comparison
    p_box = cv2.resize(t_item["p_gray"], (125, 250))
    r_box = cv2.resize(t_item["r_gray"], (125, 250))
    p1 = np.hstack([p_box, r_box])
    p1 = cv2.cvtColor(p1, cv2.COLOR_GRAY2BGR)
    cv2.putText(p1, f"Tri #{i} (Isolated)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p1, f"W_C:{t_item['w_chroma']:.1f} dL:{t_item['lum_err']:.1f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 2: Sub-mesh Aligned Luminance & Color Distribution Chart
    chart_dist = np.zeros((250, 250, 3), dtype=np.uint8)
    cv2.rectangle(chart_dist, (8, 25), (242, 215), (25, 25, 25), -1)
    cv2.rectangle(chart_dist, (8, 25), (242, 215), (70, 70, 70), 1)

    hp_l = t_item.get("submesh_hp_l", None)
    hr_l = t_item.get("submesh_hr_l_aligned", None)
    if hp_l is not None and hr_l is not None:
        max_val = max(float(np.max(hp_l)), float(np.max(hr_l)), 1e-4)
        pts_p = []
        pts_r = []
        n_pts = len(hp_l)
        dx = (250 - 28) / max(1, n_pts - 1)
        for k in range(n_pts):
            x = int(14 + k * dx)
            yp = int(215 - (hp_l[k] / max_val) * 170)
            yr = int(215 - (hr_l[k] / max_val) * 170)
            pts_p.append((x, yp))
            pts_r.append((x, yr))
        cv2.polylines(chart_dist, [np.array(pts_p, dtype=np.int32)], False, (255, 220, 0), 2, cv2.LINE_AA)  # Vàng: Product
        cv2.polylines(chart_dist, [np.array(pts_r, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)  # Cyan: Return (Aligned)
        cv2.putText(chart_dist, "Aligned Luminance L (P vs R)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        d_shift = t_item.get('delta_l_baseline', 0.0)
        cv2.putText(chart_dist, f"dL_shift: {d_shift:+.1f} | Tol: +-22", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        hp_sub_c = t_item.get("submesh_hp_c", np.zeros(24))
        hr_sub_c = t_item.get("submesh_hr_c", np.zeros(24))
        max_val = max(float(np.max(hp_sub_c)), float(np.max(hr_sub_c)), 1e-4)
        pts_p = []
        pts_r = []
        n_pts = len(hp_sub_c)
        dx = (250 - 28) / max(1, n_pts - 1)
        for k in range(n_pts):
            x = int(14 + k * dx)
            yp = int(215 - (hp_sub_c[k] / max_val) * 170)
            yr = int(215 - (hr_sub_c[k] / max_val) * 170)
            pts_p.append((x, yp))
            pts_r.append((x, yr))
        cv2.polylines(chart_dist, [np.array(pts_p, dtype=np.int32)], False, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.polylines(chart_dist, [np.array(pts_r, dtype=np.int32)], False, (0, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(chart_dist, "Sub-mesh Chroma", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        w_sub = t_item.get('submesh_w_c', 0.0)
        bh_sub = t_item.get('submesh_bh', 0.0)
        cv2.putText(chart_dist, f"W_C: {w_sub:.1f} | Bh: {bh_sub:.2f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 3: Decision Card
    panel_dec = np.zeros((250, 250, 3), dtype=np.uint8)
    cv2.rectangle(panel_dec, (4, 4), (246, 246), (30, 30, 30), -1)
    cv2.rectangle(panel_dec, (4, 4), (246, 246), (75, 75, 75), 1)
    cv2.putText(panel_dec, "SUB-MESH VERDICT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel_dec, f"Target: Tri #{i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    w_sub = t_item.get('submesh_w_c', 0.0)
    cv2.putText(panel_dec, f"Submesh W_C: {w_sub:.1f}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    dark_pct = t_item.get('pct_near_existing_dark', 1.0) * 100.0
    cv2.putText(panel_dec, f"Stroke Match: {dark_pct:.1f}%", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Outlier peak detection indicators
    out_lum = t_item.get('max_outlier_lum_blob', 0)
    out_col = t_item.get('max_outlier_color_blob', 0)
    has_out = t_item.get('has_outlier_peak', False)
    cv2.putText(panel_dec, f"Outlier Lum Blob: {out_lum}px", (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255) if out_lum >= 18 else (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(panel_dec, f"Outlier Col Blob: {out_col}px", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255) if out_col >= 18 else (0, 255, 0), 1, cv2.LINE_AA)
    out_text = "OUTLIER: DETECTED" if has_out else "OUTLIER: NONE (MATCH)"
    out_color = (0, 0, 255) if has_out else (0, 255, 0)
    cv2.putText(panel_dec, out_text, (10, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.40, out_color, 1, cv2.LINE_AA)

    v_color = (0, 255, 0) if is_cleared else (0, 0, 255)
    v_text = "CLEARED: FALSE POSITIVE" if is_cleared else "CONFIRMED: DEFECT"
    v_sub = "(Preserved Distribution)" if is_cleared else "(Material Anomaly)"
    cv2.putText(panel_dec, v_text, (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.40, v_color, 1, cv2.LINE_AA)
    cv2.putText(panel_dec, v_sub, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.38, v_color, 1, cv2.LINE_AA)

    row_bot = np.hstack([p1, chart_dist, panel_dec])
    card = np.vstack([row_top, row_bot])
    return card


def run_detailed_debug(product_path: str, return_path: str, base_debug_dir: str = None):
    """Execute detailed damage detection with granular visual outputs organized in subfolders."""
    if base_debug_dir is None:
        base_debug_dir = os.path.join(V2_DIR, "debug")

    dir_kps = os.path.join(base_debug_dir, "01_keypoints")
    dir_mesh = os.path.join(base_debug_dir, "02_mesh")
    dir_l1 = os.path.join(base_debug_dir, "03_layer1_structure")
    dir_l2 = os.path.join(base_debug_dir, "04_layer2_color")
    dir_l2_l = os.path.join(dir_l2, "01_he_mau_L")
    dir_l2_ab = os.path.join(dir_l2, "02_he_mau_ab")
    dir_fusion = os.path.join(base_debug_dir, "05_fusion")
    dir_submesh = os.path.join(base_debug_dir, "08_submesh_debug")

    # Folder cho Layer 1:
    dir_l1_match = os.path.join(dir_l1, "01_giong_nhau")
    dir_l1_diff = os.path.join(dir_l1, "02_khac_nhau")

    # Folder cho Layer 2 Kênh L:
    dir_l2_l_match = os.path.join(dir_l2_l, "01_giong_nhau")
    dir_l2_l_diff = os.path.join(dir_l2_l, "02_khac_nhau")

    # Folder cho Layer 2 Kênh ab:
    dir_l2_ab_match = os.path.join(dir_l2_ab, "01_giong_nhau")
    dir_l2_ab_diff = os.path.join(dir_l2_ab, "02_khac_nhau")

    all_debug_dirs = [
        dir_kps, dir_mesh, dir_l1, dir_l2, dir_fusion, dir_submesh,
        dir_l1_match, dir_l1_diff,
        dir_l2_l_match, dir_l2_l_diff,
        dir_l2_ab_match, dir_l2_ab_diff
    ]
    for d in all_debug_dirs:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("  RUNNING DETAILED 2-LAYER DAMAGE INSPECTION")
    print(f"  Debug Root: {base_debug_dir}")
    print("=" * 70)

    # 1. Load images
    product_img = cv2.imread(product_path)
    return_img = cv2.imread(return_path)
    if product_img is None or return_img is None:
        raise FileNotFoundError("Could not open product or return image.")

    h, w = product_img.shape[:2]

    # 2. Segment & Normalization
    segmenter = ObjectSegmenter()
    p_seg = segmenter.segment(product_img)
    r_seg = segmenter.segment(return_img)

    normalizer = ImageNormalizer()
    return_normalized = normalizer.normalize(return_img, product_img, r_seg["mask"], p_seg["mask"])
    return_ct = normalizer.color_transfer(return_img, product_img, r_seg["mask"], p_seg["mask"])

    # 3. Registration
    registrator = ImageRegistration(max_keypoints=None)
    reg_result = registrator.register(product_img, return_normalized, p_seg["mask"], r_seg["mask"])

    all_p_kps = reg_result["all_product_keypoints"]
    all_r_kps = reg_result["all_return_keypoints"]
    inliers_p = reg_result["product_points"]
    inliers_r = reg_result["return_points"]

    # -------------------------------------------------------------------------
    # FOLDER 01: KEYPOINTS
    # -------------------------------------------------------------------------
    vis_kp_p = product_img.copy()
    for pt in all_p_kps:
        cv2.circle(vis_kp_p, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 255, 255), -1)
        cv2.circle(vis_kp_p, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 140, 255), 1)

    vis_kp_r = return_img.copy()
    for pt in all_r_kps:
        cv2.circle(vis_kp_r, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 255, 255), -1)
        cv2.circle(vis_kp_r, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 140, 255), 1)

    cv2.imwrite(os.path.join(dir_kps, "product_keypoints.jpg"), vis_kp_p)
    cv2.imwrite(os.path.join(dir_kps, "return_keypoints.jpg"), vis_kp_r)

    # 3.5. DINOv2 Dense Semantic Feature Extraction
    dense_p, (feat_h, feat_w) = extract_dinov2_dense_features(product_img, max_dim=800)
    dense_r, _ = extract_dinov2_dense_features(return_img, max_dim=800)
    scale_x = feat_w / float(w)
    scale_y = feat_h / float(h)

    # 4. Mesh
    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(inliers_p, inliers_r, np.arange(len(inliers_p)))

    # Xây dựng đồ thị kề cạnh (Edge-sharing 1-ring Adjacency Map)
    edge_to_triangles = defaultdict(list)
    for i, tri in enumerate(triangles):
        v = list(tri.vertex_indices)
        for e in [frozenset([v[0], v[1]]), frozenset([v[1], v[2]]), frozenset([v[2], v[0]])]:
            edge_to_triangles[e].append(i)

    triangle_neighbors = {}
    for i, tri in enumerate(triangles):
        v = list(tri.vertex_indices)
        nbrs = set()
        for e in [frozenset([v[0], v[1]]), frozenset([v[1], v[2]]), frozenset([v[2], v[0]])]:
            for other_idx in edge_to_triangles[e]:
                if other_idx != i:
                    nbrs.add(other_idx)
        triangle_neighbors[i] = list(nbrs)

    # -------------------------------------------------------------------------
    # FOLDER 02: MESH WIREFRAME
    # -------------------------------------------------------------------------
    vis_mesh_p = product_img.copy()
    vis_mesh_r = return_img.copy()
    mesh_mask_p = np.zeros((h, w), dtype=np.uint8)
    mesh_mask_r = np.zeros((h, w), dtype=np.uint8)

    for i, tri in enumerate(triangles):
        pts_p = np.array([list(vertices[idx].product_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
        pts_r = np.array([list(vertices[idx].return_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_mesh_p, [pts_p], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(vis_mesh_r, [pts_r], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.fillConvexPoly(mesh_mask_p, pts_p, 255)
        cv2.fillConvexPoly(mesh_mask_r, pts_r, 255)

        cx_p = int(np.mean(pts_p[:, 0, 0]))
        cy_p = int(np.mean(pts_p[:, 0, 1]))
        cx_r = int(np.mean(pts_r[:, 0, 0]))
        cy_r = int(np.mean(pts_r[:, 0, 1]))
        draw_text_with_shadow(vis_mesh_p, str(i), (cx_p - 6, cy_p + 3), font_scale=0.32, color=(0, 255, 255))
        draw_text_with_shadow(vis_mesh_r, str(i), (cx_r - 6, cy_r + 3), font_scale=0.32, color=(0, 255, 255))

    cv2.imwrite(os.path.join(dir_mesh, "mesh_product.jpg"), vis_mesh_p)
    cv2.imwrite(os.path.join(dir_mesh, "mesh_return.jpg"), vis_mesh_r)

    # -------------------------------------------------------------------------
    # FOLDER 03: LAYER 1 - STRUCTURE / EDGES (RESTRICTED STRICTLY TO MESH)
    # -------------------------------------------------------------------------
    p_gray = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(return_ct, cv2.COLOR_BGR2GRAY)

    # 1. Khử nhiễu đốm men / hạt kim tuyến / phản quang bằng Median Blur
    p_med = cv2.medianBlur(p_gray, 3)
    r_med = cv2.medianBlur(r_gray, 3)

    def filter_edge_noise(edge_mask: np.ndarray, min_stroke_len: float = 12.0, max_round_diag: float = 22.0) -> np.ndarray:
        """
        Lọc nhiễu thông minh dựa trên hình thái học (Shape-Aware Filtering):
        - Giữ nguyên các nét vẽ thanh mảnh (Linear Strokes) dù ngắn (chiều dài >= 12px hoặc tỷ lệ dài/rộng >= 2.2).
        - Triệt tiêu các đốm nhiễu tròn, hạt phản quang, bóng men hình tròn (Aspect Ratio < 2.0 và đường kính <= 22px).
        - Loại bỏ các hạt bụi men cô lập (< 6px).
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
        clean_mask = np.zeros_like(edge_mask)
        for lbl in range(1, num_labels):
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]
            area = stats[lbl, cv2.CC_STAT_AREA]
            diag = np.sqrt(w**2 + h**2)

            # 1. Hạt bụi men quá nhỏ (< 6px)
            if area < 6:
                continue

            major = max(w, h)
            minor = max(min(w, h), 1)
            aspect = major / minor

            # 2. Đốm tròn / hạt phản quang nhỏ (tròn, bầu dục compact)
            if aspect < 2.0 and diag <= max_round_diag:
                continue

            # 3. Nét vẽ hoa văn thực thụ (thanh mảnh hoặc có độ dài)
            if diag >= min_stroke_len or aspect >= 2.2 or area >= 30:
                clean_mask[labels == lbl] = 255

        return clean_mask

    # Áp dụng CLAHE để chuẩn hóa ánh sáng và độ tương phản toàn cục trước khi chạy Canny
    clahe_global = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    p_clahe_global = clahe_global.apply(p_med)
    r_clahe_global = clahe_global.apply(r_med)

    # Dùng ngưỡng Canny tối ưu (40, 120) để bảo toàn nét vẽ thanh mảnh
    p_edge_raw = cv2.Canny(p_clahe_global, 40, 120)
    r_edge_raw = cv2.Canny(r_clahe_global, 40, 120)

    # MASK STRICTLY INSIDE MESH (Zero out neck, base, and blank porcelain outside mesh)
    p_edge_raw[mesh_mask_p == 0] = 0
    r_edge_raw[mesh_mask_r == 0] = 0

    # LƯU FILE LOSSLESS (PNG) ĐỂ TRÁNH NHIỄU RINGING JPEG
    cv2.imwrite(os.path.join(dir_l1, "01_product_canny_raw.png"), p_edge_raw)
    cv2.imwrite(os.path.join(dir_l1, "02_return_canny_raw.png"), r_edge_raw)
    cv2.imwrite(os.path.join(dir_l1, "01_product_canny_raw.jpg"), p_edge_raw, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(os.path.join(dir_l1, "02_return_canny_raw.jpg"), r_edge_raw, [cv2.IMWRITE_JPEG_QUALITY, 100])

    p_edge_denoised = filter_edge_noise(p_edge_raw, min_stroke_len=12.0, max_round_diag=22.0)
    r_edge_denoised = filter_edge_noise(r_edge_raw, min_stroke_len=12.0, max_round_diag=22.0)

    cv2.imwrite(os.path.join(dir_l1, "03_product_canny_denoised.png"), p_edge_denoised)
    cv2.imwrite(os.path.join(dir_l1, "04_return_canny_denoised.png"), r_edge_denoised)
    cv2.imwrite(os.path.join(dir_l1, "03_product_canny_denoised.jpg"), p_edge_denoised, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(os.path.join(dir_l1, "04_return_canny_denoised.jpg"), r_edge_denoised, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # BƯỚC 5: SO SÁNH TỔNG THỂ CẤU TRÚC NÉT TRONG TOÀN BỘ VÙNG MESH
    # (Dùng Distance Transform kháng lệch góc 3D, loại bỏ lóa flash và kiểm tra gợn nét mềm)
    dist_to_r = cv2.distanceTransform(cv2.bitwise_not(r_edge_denoised), cv2.DIST_L2, 5)
    r_lab = cv2.cvtColor(return_img, cv2.COLOR_BGR2LAB)
    glare_r = r_lab[..., 0] > 220

    # Tính độ dốc gradient của ảnh Return để kiểm tra nét cọ mềm
    r_gray_orig = cv2.cvtColor(return_img, cv2.COLOR_BGR2GRAY)
    gx_r = cv2.Sobel(r_gray_orig, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(r_gray_orig, cv2.CV_32F, 0, 1, ksize=3)
    mag_r = np.sqrt(gx_r**2 + gy_r**2)

    # Ứng viên nét Product bị mất: Không có nét Return trong phạm vi 10px, không bị lóa flash, và Return hoàn toàn không có gợn nét nào (mag <= 22)
    candidate_broken = (p_edge_denoised > 0) & (dist_to_r > 10.0) & (~glare_r) & (mag_r <= 22.0)

    # Gom cụm: Chỉ giữ lại các vết đứt liên tục thực sự (chiều dài >= 40px hoặc diện tích >= 50px)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(candidate_broken.astype(np.uint8), connectivity=8)
    broken_edges_clean = np.zeros_like(p_edge_denoised)
    for lbl in range(1, num_lbl):
        w_b = stats[lbl, cv2.CC_STAT_WIDTH]
        h_b = stats[lbl, cv2.CC_STAT_HEIGHT]
        diag_b = np.sqrt(w_b**2 + h_b**2)
        area_b = stats[lbl, cv2.CC_STAT_AREA]
        if diag_b >= 40.0 or area_b >= 50:
            broken_edges_clean[lbls == lbl] = 255

    # Vết nứt sẹo xâm lấn mới (Intrusive Crack / Scar) trên ảnh Return
    dist_to_p = cv2.distanceTransform(cv2.bitwise_not(p_edge_denoised), cv2.DIST_L2, 5)
    p_gray_orig = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
    gx_p = cv2.Sobel(p_gray_orig, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(p_gray_orig, cv2.CV_32F, 0, 1, ksize=3)
    mag_p = np.sqrt(gx_p**2 + gy_p**2)

    candidate_intrusive = (r_edge_denoised > 0) & (dist_to_p > 10.0) & (mag_p <= 20.0)
    num_lbl_i, lbls_i, stats_i, _ = cv2.connectedComponentsWithStats(candidate_intrusive.astype(np.uint8), connectivity=8)
    intrusive_cracks = np.zeros_like(r_edge_denoised)
    for lbl in range(1, num_lbl_i):
        w_i = stats_i[lbl, cv2.CC_STAT_WIDTH]
        h_i = stats_i[lbl, cv2.CC_STAT_HEIGHT]
        diag_i = np.sqrt(w_i**2 + h_i**2)
        area_i = stats_i[lbl, cv2.CC_STAT_AREA]
        if diag_i >= 40.0 or area_i >= 50:
            intrusive_cracks[lbls_i == lbl] = 255

    # Tổng hợp lỗi cấu trúc Layer 1 trên toàn mesh
    l1_defect_map = cv2.bitwise_or(broken_edges_clean, intrusive_cracks)

    # Dọn dẹp các tên file cũ để tránh nhầm lẫn
    for old_file in ["01_product_canny_edges.jpg", "02_return_canny_edges.jpg", "03_new_edges_diff.jpg", "03_broken_edges_diff.jpg"]:
        old_p = os.path.join(dir_l1, old_file)
        if os.path.exists(old_p):
            try:
                os.remove(old_p)
            except OSError:
                pass

    cv2.imwrite(os.path.join(dir_l1, "05_broken_edges_diff.jpg"), l1_defect_map, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # -------------------------------------------------------------------------
    # FOLDER 04: LAYER 2 - DUAL COLOR (LUMINANCE L + CHROMA AB) & FUSED HEATMAPS
    # -------------------------------------------------------------------------
    p_lab = cv2.cvtColor(product_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(return_normalized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Global median color offset removal on object
    p_valid = p_seg["mask"] > 0
    r_valid = r_seg["mask"] > 0
    offset_a = float(np.median(r_lab[..., 1][r_valid]) - np.median(p_lab[..., 1][p_valid]))
    offset_b = float(np.median(r_lab[..., 2][r_valid]) - np.median(p_lab[..., 2][p_valid]))
    offset_L = float(np.median(r_lab[..., 0][r_valid]) - np.median(p_lab[..., 0][p_valid]))
    r_lab[..., 1] -= offset_a
    r_lab[..., 2] -= offset_b
    r_lab_norm_L = r_lab[..., 0] - offset_L

    # 1. Hệ màu 1: Chroma Delta_E (a, b) - Đo độ lệch sắc màu
    da = p_lab[..., 1] - r_lab[..., 1]
    db = p_lab[..., 2] - r_lab[..., 2]
    chroma_diff_map = np.sqrt(da ** 2 + db ** 2)

    # 2. Hệ màu 2: Luminance Delta_L (L) - Đo độ lệch sáng men / hoa văn đơn sắc
    lum_diff_map = np.abs(p_lab[..., 0] - r_lab_norm_L)

    # 3. Kết hợp 2 hệ màu: Fused Layer 2 Difference Map
    fused_l2_diff_map = np.sqrt(0.5 * (lum_diff_map ** 2) + 1.0 * (chroma_diff_map ** 2))

    # Ignore glare / specular reflection (where L channel is saturated near 255)
    glare_mask = (r_lab[..., 0] > 240) | (p_lab[..., 0] > 240)
    chroma_diff_map[glare_mask] = 0.0
    chroma_diff_map[mesh_mask_r == 0] = 0.0
    lum_diff_map[glare_mask] = 0.0
    lum_diff_map[mesh_mask_r == 0] = 0.0
    fused_l2_diff_map[glare_mask] = 0.0
    fused_l2_diff_map[mesh_mask_r == 0] = 0.0

    # Xuất heatmap Hệ màu Chroma (a, b)
    norm_heatmap_c = np.clip(chroma_diff_map / 35.0 * 255.0, 0, 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(norm_heatmap_c, cv2.COLORMAP_JET)
    color_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01_chroma_delta_e_heatmap.jpg"), color_heatmap)
    cv2.imwrite(os.path.join(dir_l2, "01a_chroma_delta_e_heatmap.jpg"), color_heatmap)

    # Xuất heatmap Hệ màu Luminance (L)
    norm_heatmap_l = np.clip(lum_diff_map / 45.0 * 255.0, 0, 255).astype(np.uint8)
    lum_heatmap = cv2.applyColorMap(norm_heatmap_l, cv2.COLORMAP_JET)
    lum_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01b_luminance_diff_heatmap.jpg"), lum_heatmap)

    # Xuất heatmap Fused Kết hợp 2 hệ màu (L + ab)
    norm_heatmap_f = np.clip(fused_l2_diff_map / 40.0 * 255.0, 0, 255).astype(np.uint8)
    fused_heatmap = cv2.applyColorMap(norm_heatmap_f, cv2.COLORMAP_JET)
    fused_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01c_combined_layer2_heatmap.jpg"), fused_heatmap)

    # Ghép 3 Heatmap cạnh nhau để so sánh trực quan
    scale_factor = 700.0 / float(w) if w > 700 else 1.0
    pan_w, pan_h = int(w * scale_factor), int(h * scale_factor)
    p_c_sub = cv2.resize(color_heatmap, (pan_w, pan_h))
    p_l_sub = cv2.resize(lum_heatmap, (pan_w, pan_h))
    p_f_sub = cv2.resize(fused_heatmap, (pan_w, pan_h))

    cv2.putText(p_c_sub, "1. Chroma Delta_E (a, b)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(p_l_sub, "2. Luminance Delta_L (L)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(p_f_sub, "3. Fused Layer 2 (L + ab)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    comparison_heatmap = np.hstack([p_c_sub, p_l_sub, p_f_sub])
    cv2.imwrite(os.path.join(dir_l2, "00_layer2_dual_color_heatmaps_comparison.jpg"), comparison_heatmap)

    # -------------------------------------------------------------------------
    # STEP 3: EVALUATE EACH TRIANGLE ACROSS 2 LAYERS
    # -------------------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    struct_scores = []
    color_diffs = []
    raw_patch_list = []
    b_shift_edges = set()

    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Pass 1: Extract patches and detect boundary shift triangles across mesh
    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        # Canonical 72x72 patches
        patch_p, mask_p = canonical_triangle_patch(product_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(return_ct, pts_r, target_size=72)

        # Canonical edge patches directly from 03_product_canny_denoised and 04_return_canny_denoised
        patch_pe, _ = canonical_triangle_patch(p_edge_denoised, pts_p, target_size=72)
        patch_re, _ = canonical_triangle_patch(r_edge_denoised, pts_r, target_size=72)
        patch_pe = (patch_pe > 127).astype(np.uint8) * 255
        patch_re = (patch_re > 127).astype(np.uint8) * 255

        # Step 3.1: Micro-align to eliminate curvature 3D distortion (+/- 4px)
        patch_r_aligned, patch_re_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=4, extra_patch=patch_re)

        p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c_gray = clahe.apply(p_raw_gray)
        r_c_gray = clahe.apply(r_raw_gray)

        valid = mask_p > 0
        glare_px = (r_c_gray > 240) | (p_c_gray > 240)

        # 0. Orientation profiles & Directional Coherence
        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c_gray, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c_gray, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0
        is_parallel_texture = bool((ori_corr > 0.85) and (coh_p >= 0.35))

        # Raw difference dipole check to detect boundary shift of existing features
        raw_diff = r_raw_gray.astype(np.float32) - p_raw_gray.astype(np.float32)
        raw_diff[~valid] = 0.0
        bin_raw_diff = (np.abs(raw_diff) > 35.0).astype(np.uint8) * 255
        opened_raw = cv2.morphologyEx(bin_raw_diff, cv2.MORPH_OPEN, k_open)

        raw_pos_px = np.sum((opened_raw > 0) & (raw_diff > 35.0))
        raw_neg_px = np.sum((opened_raw > 0) & (raw_diff < -35.0))
        raw_dipole_ratio = min(raw_pos_px, raw_neg_px) / (max(raw_pos_px, raw_neg_px) + 1e-5)

        is_b_shift = bool(raw_pos_px >= 60 and raw_neg_px >= 60 and raw_dipole_ratio >= 0.30 and ori_corr >= 0.80)
        if is_b_shift:
            v = list(tri.vertex_indices)
            b_shift_edges.add(frozenset([v[0], v[1]]))
            b_shift_edges.add(frozenset([v[1], v[2]]))
            b_shift_edges.add(frozenset([v[2], v[0]]))

        raw_patch_list.append({
            "id": i,
            "tri": tri,
            "pts_p": pts_p,
            "pts_r": pts_r,
            "patch_p": patch_p,
            "patch_r": patch_r_aligned,
            "patch_pe": patch_pe,
            "patch_re": patch_re_aligned,
            "mask_p": mask_p,
            "p_raw_gray": p_raw_gray,
            "r_raw_gray": r_raw_gray,
            "p_c_gray": p_c_gray,
            "r_c_gray": r_c_gray,
            "valid": valid,
            "glare_px": glare_px,
            "peaks_p": peaks_p,
            "ori_corr": ori_corr,
            "is_parallel": is_parallel_texture,
            "is_b_shift": is_b_shift,
        })

    # Pass 2: Evaluate each triangle with boundary shift and edge-leakage compensation
    h_m, w_m = 72, 72
    Y_m, X_m = np.ogrid[:h_m, :w_m]
    inner_triangle = (X_m >= 4) & (Y_m >= 4) & (X_m + Y_m <= 66)

    patch_data = []

    for item in raw_patch_list:
        i = item["id"]
        tri = item["tri"]
        pts_p = item["pts_p"]
        pts_r = item["pts_r"]
        patch_p = item["patch_p"]
        patch_r_aligned = item["patch_r"]
        mask_p = item["mask_p"]
        p_raw_gray = item["p_raw_gray"]
        r_raw_gray = item["r_raw_gray"]
        p_c_gray = item["p_c_gray"]
        r_c_gray = item["r_c_gray"]
        valid = item["valid"]
        glare_px = item["glare_px"]
        peaks_p = item["peaks_p"]
        ori_corr = item["ori_corr"]
        is_parallel_texture = item["is_parallel"]
        is_b_shift = item["is_b_shift"]

        # Check if triangle shares an edge with a boundary shift triangle
        v = list(tri.vertex_indices)
        shares_b_shift = (frozenset([v[0], v[1]]) in b_shift_edges or 
                          frozenset([v[1], v[2]]) in b_shift_edges or 
                          frozenset([v[2], v[0]]) in b_shift_edges)

        # LAB color patches (with global cast offset applied)
        p_c_lab = cv2.cvtColor(patch_p, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_c_lab = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_c_lab[..., 1] -= offset_a
        r_c_lab[..., 2] -= offset_b

        # 1. TOP-HAT & BLACK-HAT
        p_what = cv2.morphologyEx(p_c_gray, cv2.MORPH_TOPHAT, k_morph)
        p_bhat = cv2.morphologyEx(p_c_gray, cv2.MORPH_BLACKHAT, k_morph)
        r_what = cv2.morphologyEx(r_c_gray, cv2.MORPH_TOPHAT, k_morph)
        r_bhat = cv2.morphologyEx(r_c_gray, cv2.MORPH_BLACKHAT, k_morph)

        # Trích xuất cạnh Canny trực tiếp từ 03_product_canny_denoised và 04_return_canny_denoised
        p_c_edge = item["patch_pe"].copy()
        r_c_edge = item["patch_re"].copy()
        p_c_edge[~valid] = 0
        r_c_edge[~valid] = 0

        # KIỂM TRA MẬT ĐỘ CẠNH (EDGE DENSITY CHECK)
        p_edge_cnt = int(np.sum(p_c_edge > 0))
        r_edge_cnt = int(np.sum(r_c_edge > 0))

        # Helper function đo chiều dài thực và kiểm tra tính dạng đường kẻ (linear stroke)
        def measure_linear_stroke(contours, min_len=8.0, min_aspect=1.8):
            max_len = 0.0
            for c in contours:
                if len(c) < 4:
                    continue
                rect = cv2.minAreaRect(c)
                length, width = max(rect[1]), min(rect[1])
                aspect = length / max(width, 0.5)
                # Chỉ tính nếu thực sự có dạng đường kẻ kéo dài (linear)
                if length >= min_len and (aspect >= min_aspect or length >= 14.0):
                    if length > max_len:
                        max_len = length
            return max_len

        # Distance to ALL Product features (ridges + Canny step edges)
        p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
        dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

        gx_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 0, 1, ksize=3)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        # NẾU PRODUCT HOẶC CẢ 2 PHẦN LỚN KHÔNG CÓ CẠNH (MEN TRƠN)
        # Theo yêu cầu: "nếu 1 trong 2 hình phần lớn là không có edge gì thì đừng nên so sánh"
        if p_edge_cnt < 20:
            # Product là men trơn, không có họa tiết hoa văn
            max_broken_length = 0.0
            anom_crack = np.zeros_like(p_c_gray, dtype=bool)
            anom_scratch = np.zeros_like(p_c_gray, dtype=bool)
            scratch_map = np.zeros_like(p_c_gray)
            max_white_scratch = 0.0
            max_dark_crack = 0.0
        else:
            # Product CÓ nét vẽ / hoa văn -> So sánh bình thường
            crack_cand = (r_bhat > 30) & valid & inner_triangle & (~glare_px)
            scratch_cand = (r_what > 30) & valid & inner_triangle & (~glare_px)
            anom_crack = np.zeros_like(crack_cand)
            anom_scratch = np.zeros_like(scratch_cand)

            for y, x in zip(*np.where(crack_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                tol_dist = 6.0 if (ori_corr < 0.85) else 8.0
                near_shifted = (d <= tol_dist) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=30.0)
                is_parallel_feature = (ori_corr >= 0.85) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted or is_parallel_feature):
                    anom_crack[y, x] = True

            for y, x in zip(*np.where(scratch_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                tol_dist = 6.0 if (ori_corr < 0.85) else 8.0
                near_shifted = (d <= tol_dist) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=30.0)
                is_parallel_feature = (ori_corr >= 0.85) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted or is_parallel_feature):
                    anom_scratch[y, x] = True

            # Combined scratch/crack map
            scratch_map = np.zeros_like(p_c_gray)
            scratch_map[anom_crack] = r_bhat[anom_crack]
            scratch_map[anom_scratch] = np.maximum(scratch_map[anom_scratch], r_what[anom_scratch])

            cnts_w, _ = cv2.findContours(anom_scratch.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_b, _ = cv2.findContours(anom_crack.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_white_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)
            max_dark_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)

        # 2. CANNY EDGES & BROKEN PATTERN
        # Chỉ kiểm tra gãy nét nếu Product THỰC SỰ CÓ NÉT (p_edge_cnt >= 20)
        # NGUYÊN TẮC 1 & 2: Vùng an toàn lòng tam giác (inner_triangle) + Dung sai co giãn vi mô (dist_r > 6.0, mag_r < 25.0)
        if p_edge_cnt >= 20:
            if np.any(r_c_edge > 0):
                dist_r = cv2.distanceTransform(cv2.bitwise_not(r_c_edge), cv2.DIST_L2, 3)
                mag_r = np.sqrt(gx_r**2 + gy_r**2)
                broken_edges = (p_c_edge > 0) & (dist_r > 6.0) & (mag_r < 25.0) & (~glare_px) & inner_triangle
                cnts_broken, _ = cv2.findContours(broken_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_broken_length = measure_linear_stroke(cnts_broken, min_len=10.0, min_aspect=1.8)
            else:
                broken_edges = (p_c_edge > 0) & inner_triangle & (~glare_px)
                cnts_broken, _ = cv2.findContours(broken_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_broken_length = measure_linear_stroke(cnts_broken, min_len=10.0, min_aspect=1.8)
        else:
            max_broken_length = 0.0

        # Intrusive edge on blank porcelain:
        # Require:
        # 1. dist_to_p_all > 6.0 (not near existing feature)
        # 2. angle does NOT match peaks_p (not a shifted existing feature)
        # 3. strong edge contrast: mag_r_raw >= 70.0 (real crack/scar, not JPEG/sensor noise)
        # 4. inner_triangle (không nằm sát mép viền cắt)
        grad_p_raw = np.sqrt(cv2.Sobel(p_raw_gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw_gray, cv2.CV_32F, 0, 1)**2)
        r_raw_edge = cv2.Canny(r_raw_gray, 40, 120)
        r_raw_edge[~valid] = 0
        gx_r_raw = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
        gy_r_raw = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
        mag_r_raw = np.sqrt(gx_r_raw**2 + gy_r_raw**2)
        ang_r_raw = (np.arctan2(gy_r_raw, gx_r_raw) * 180.0 / np.pi) % 180.0

        intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 6.0) & (grad_p_raw < 25.0) & (mag_r_raw >= 70.0) & (~glare_px) & inner_triangle
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0):
                anom_intrusive[y, x] = True

        cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = measure_linear_stroke(cnts_intrusive, min_len=12.0, min_aspect=2.0)

        # 3. Solid Grayscale Anomaly Blob (Scars, Chips, Stains)
        # SỬ DỤNG ROBUST MEDIAN VÀ MAD ĐỂ TRÁNH MEAN-SHIFT DEFECT MASKING
        p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
        r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)
        if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
            med_p_loc = np.median(p_vals)
            med_r_loc = np.median(r_vals)
            mad_p_loc = np.median(np.abs(p_vals - med_p_loc)) * 1.4826
            mad_r_loc = np.median(np.abs(r_vals - med_r_loc)) * 1.4826
            scale_loc = np.clip(mad_p_loc / (mad_r_loc + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_raw_gray.astype(np.float32) - med_r_loc) * scale_loc + med_p_loc
        else:
            r_norm_loc = r_raw_gray.astype(np.float32)

        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0
        bin_diff_loc = (diff_loc > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

        # Bộ nhận diện vết mực loang / dị vật đơn sắc (Achromatic Ink Stain & Textureless Void):
        # 1. Đen sâu trong Return (r_raw_gray < 55)
        # 2. Product gốc sáng rõ hơn hẳn (sụt giảm >= 45 cả sau khi chuẩn hóa ánh sáng)
        pitch_drop = (diff_loc > 40.0) & (r_raw_gray < 55) & ((p_raw_gray.astype(np.float32) - r_raw_gray.astype(np.float32)) >= 45.0) & valid & (~glare_px)
        ink_opened = cv2.morphologyEx(pitch_drop.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_open_3)
        num_lbl_ink, lbls_ink, stats_ink, _ = cv2.connectedComponentsWithStats(ink_opened, connectivity=8)
        
        is_ink_stain = False
        max_ink_blob = 0
        for l in range(1, num_lbl_ink):
            area = stats_ink[l, cv2.CC_STAT_AREA]
            if area >= 35:
                m_ink = (lbls_ink == l)
                if np.std(r_raw_gray[m_ink]) <= 22.0:
                    # NGUYÊN TẮC HÌNH THÁI HỌC GIỌT MỰC ĐẶC (DROPLET MORPHOLOGY TEST):
                    # Nét vẽ mảnh dịch chuyển (stroke) chỉ rộng 1-2px nên bán kính lõi core <= 2.2px.
                    # Giọt mực loang thật (droplet) là mảng đặc có bán kính lõi dày core >= 2.8px.
                    dist_ink = cv2.distanceTransform(m_ink.astype(np.uint8) * 255, cv2.DIST_L2, 3)
                    core_radius = float(np.max(dist_ink))

                    if p_edge_cnt >= 20:
                        # Trên vùng hoa văn: Bắt buộc phải là giọt đặc dày (core_radius >= 2.8 và area >= 45)
                        # hoặc giọt rất dày (core_radius >= 3.5)
                        is_solid_droplet = bool((core_radius >= 2.8 and area >= 45) or (core_radius >= 3.5))
                    else:
                        # Trên nền men trơn: Dung sai linh hoạt hơn
                        is_solid_droplet = bool(core_radius >= 1.8 and area >= 35)

                    if is_solid_droplet:
                        max_ink_blob = max(max_ink_blob, area)
                        is_ink_stain = True

        # NGUYÊN TẮC 3: Ngưỡng tương phản thích ứng (Adaptive Shape Contrast for Blobs)
        # Nếu ori_corr >= 0.80 hoặc Product men trơn p_edge_cnt < 20, và không có nứt sắc nét:
        # thì hoa văn khớp hình học -> Đòi hỏi contrast_th = 90.0 để tránh bắt nhầm chênh đổ bóng trong mảng tối!
        is_shape_intact = bool((ori_corr >= 0.80 or p_edge_cnt < 20) and max_dark_crack < 10.0 and max_white_scratch < 10.0 and max_intrusive_length < 10.0)
        contrast_th = 90.0 if is_shape_intact else 60.0

        if is_b_shift:
            max_solid_blob = 0
        elif shares_b_shift:
            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= contrast_th:
                    ys, xs = np.where(m)
                    is_edge_leak = (xs.max() <= 15 or ys.max() <= 15 or (xs + ys).min() >= 55)
                    if not is_edge_leak:
                        cleaned_blob = max(cleaned_blob, area)
            max_solid_blob = cleaned_blob
        else:
            valid_blobs = []
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= contrast_th:
                    valid_blobs.append(area)
            max_solid_blob = max(valid_blobs, default=0)

        # Composite Structural Defect Metric (Giảm mạnh trọng số blob xuống 0.1 theo yêu cầu, tăng trọng số nét gãy broken lên 1.5, bổ sung điểm dị vật mực)
        struct_metric = (
            1.5 * max_dark_crack +
            1.5 * max_white_scratch +
            2.0 * max_intrusive_length +
            0.1 * max_solid_blob +
            1.5 * max_broken_length +
            (50.0 if is_ink_stain else 0.0)
        )

        # SuperPoint 256D descriptor similarity
        desc_sims = []
        p_desc = reg_result["product_descriptors"]
        r_desc = reg_result["return_descriptors"]
        for v_idx in tri.vertex_indices:
            v = vertices[v_idx]
            if v.descriptor_index is not None and v.descriptor_index < len(p_desc):
                d1, d2 = p_desc[v.descriptor_index], r_desc[v.descriptor_index]
                desc_sims.append(float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8)))
        mean_desc_sim = float(np.mean(desc_sims)) if desc_sims else 1.0
        ssim_val = float(ssim(p_c_gray, r_c_gray, data_range=255))
        s_score = 0.5 * ssim_val + 0.5 * mean_desc_sim
        struct_scores.append(s_score)

        # Texture Obliteration / Void Defect (only massive structural obliteration)
        is_void_damage = bool(max_solid_blob >= 180 and ssim_val <= 0.30 and ori_corr < 0.70)

        # --- L2: Dual Color Metrics (Luminance L + Chroma a,b) ---
        c_da = p_c_lab[..., 1] - r_c_lab[..., 1]
        c_db = p_c_lab[..., 2] - r_c_lab[..., 2]
        patch_chroma_diff = np.sqrt(c_da ** 2 + c_db ** 2)

        # Exclude glare pixels inside patch (chỉ loại bỏ điểm bão hòa cháy sáng > 250)
        non_glare = (mask_p > 0) & (r_c_lab[..., 0] <= 250) & (p_c_lab[..., 0] <= 250)
        mean_chroma_err = float(np.mean(patch_chroma_diff[non_glare])) if np.any(non_glare) else 0.0
        color_diffs.append(mean_chroma_err)

        # ---------------------------------------------------------------------
        # BLUR MATCHING CHO CÁC VÙNG NGHI VẤN MẤT NÉT BẤT ĐỐI XỨNG (ASYMMETRIC BLUR)
        # Nằm ngay trước phần điều chỉnh sáng để đồng bộ độ mượt trước khi tính mốc sáng & trừ L
        # ---------------------------------------------------------------------
        p_l_eval = p_c_lab[..., 0].copy()
        r_l_eval = r_c_lab[..., 0].copy()

        if np.any(non_glare):
            # Đo độ sắc nét của 2 bên bằng phương sai Laplacian
            lap_p = cv2.Laplacian(p_l_eval, cv2.CV_32F)[mask_p > 0]
            lap_r = cv2.Laplacian(r_l_eval, cv2.CV_32F)[mask_p > 0]
            var_p = float(np.var(lap_p)) if len(lap_p) > 0 else 0.0
            var_r = float(np.var(lap_r)) if len(lap_r) > 0 else 0.0

            # Nếu có sự chênh lệch độ nét bất đối xứng rõ rệt (tỉ lệ >= 2.0):
            sharp_ratio = (max(var_p, var_r) + 1.0) / (min(var_p, var_r) + 1.0)
            if sharp_ratio >= 2.0:
                ksize = 5 if sharp_ratio >= 3.5 else 3
                sigma = 1.2 if sharp_ratio >= 3.5 else 0.8
                if var_r > var_p:
                    # Return nét hơn Product nhiều -> Làm mượt Return cho khớp Product
                    r_l_eval = cv2.GaussianBlur(r_l_eval, (ksize, ksize), sigma)
                else:
                    # Product nét hơn Return nhiều -> Làm mượt Product cho khớp Return
                    p_l_eval = cv2.GaussianBlur(p_l_eval, (ksize, ksize), sigma)

            # Vi căn chỉnh lại r_l_eval sau khi đồng bộ độ mượt để khớp khít hoàn hảo gợn sóng (+/- 2px)
            r_l_eval = micro_align_patches(p_l_eval, r_l_eval, mask_p, max_shift=2)

        # Cân bằng độ sáng kênh L vững (Robust Luminance Baseline Alignment)
        # Bóc tách điểm lỗi & chói sáng bằng Median of Differences
        if np.any(non_glare):
            l_diff_px = r_l_eval[non_glare] - p_l_eval[non_glare]
            delta_l_baseline = float(np.median(l_diff_px))
            # Lọc các điểm men lành lặn (Inliers) trong dung sai +-15 đơn vị so với mốc nền
            is_inlier = np.abs(l_diff_px - delta_l_baseline) <= 15.0
            if np.sum(is_inlier) >= 10:
                delta_l_shift = float(np.mean(l_diff_px[is_inlier]))
            else:
                delta_l_shift = delta_l_baseline
            r_l_norm = r_l_eval - delta_l_shift
        else:
            delta_l_shift = 0.0
            r_l_norm = r_l_eval

        patch_lum_diff = np.abs(p_l_eval - r_l_norm)
        mean_lum_err = float(np.mean(patch_lum_diff[non_glare])) if np.any(non_glare) else 0.0

        # Kết hợp Fused Layer 2
        patch_fused_l2 = np.sqrt(0.5 * (patch_lum_diff ** 2) + 1.0 * (patch_chroma_diff ** 2))
        patch_fused_l2[~non_glare] = 0.0
        mean_fused_l2_err = float(np.mean(patch_fused_l2[non_glare])) if np.any(non_glare) else 0.0

        # Physical Blobs Verification trên cả 2 hệ màu
        strong_color_diff = (patch_chroma_diff > 18.0) & non_glare
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(strong_color_diff.astype(np.uint8) * 255, connectivity=8)
        max_blob_area = max([stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_labels)], default=0)

        # ĐO ĐỘ LIỀN MẠCH CỦA LỖI VÀ ĐỐI CHIẾU HÌNH THÁI VỚI HÌNH CŨ PRODUCT (KÊNH L):
        strong_lum_diff = (patch_lum_diff > 25.0) & non_glare
        num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats(strong_lum_diff.astype(np.uint8) * 255, connectivity=8)
        
        tri_valid_area = max(1, int(np.sum(mask_p > 0)))
        p_l_valid = p_l_eval[non_glare]
        p_tri_base = float(np.median(p_l_valid)) if len(p_l_valid) > 0 else 200.0

        max_lum_blob = 0
        max_lum_solidity = 0.0
        max_solid_lum_blob = 0
        max_blob_tri_ratio = 0.0
        is_intrusive_lum_blob = False

        for lbl in range(1, num_l):
            area = stats_l[lbl, cv2.CC_STAT_AREA]
            if area > max_lum_blob:
                max_lum_blob = area
            
            # Tính độ liền mạch (Solidity = Area / Convex Hull Area)
            pts_blob = np.column_stack(np.where(labels_l == lbl))
            if len(pts_blob) >= 6:
                pts_xy = np.column_stack((pts_blob[:, 1], pts_blob[:, 0])).astype(np.int32)
                hull = cv2.convexHull(pts_xy)
                hull_area = float(cv2.contourArea(hull))
                solidity = float(area) / (hull_area + 1e-5)
            else:
                solidity = 1.0

            # Khối lỗi thật sự phải có tính đặc, liền mạch (Solidity >= 0.70)
            if solidity >= 0.70:
                if area > max_solid_lum_blob:
                    max_solid_lum_blob = area
                    max_lum_solidity = solidity
                    max_blob_tri_ratio = float(area) / float(tri_valid_area)

                # ĐỐI CHIẾU VỚI HÌNH CŨ (PRODUCT GỐC):
                m_curr_blob = (labels_l == lbl)
                p_blob_vals = p_l_eval[m_curr_blob]
                r_blob_vals = r_l_norm[m_curr_blob]
                
                p_mean_b = float(np.mean(p_blob_vals))
                p_std_b = float(np.std(p_blob_vals))
                r_mean_b = float(np.mean(r_blob_vals))
                drop_b = p_mean_b - r_mean_b
                ratio_b = float(area) / float(tri_valid_area)

                # =============================================================
                # NGUYÊN TẮC VẬT LÝ RIÊNG BIỆT CHO 2 VÙNG BỀ MẶT:
                # VÙNG 1: NỀN MEN TRƠN (p_edge_cnt < 20)
                # VÙNG 2: VÙNG HOA VĂN / NÉT VẼ DÀY ĐẶC (p_edge_cnt >= 20)
                # =============================================================
                contrast_energy = float(area) * drop_b
                z_blob = drop_b / max(p_std_b, 5.0)

                if p_edge_cnt < 20:
                    # NGUYÊN TẮC VÙNG NỀN MEN TRƠN:
                    # Vùng Product gốc phải là men sạch lành lặn
                    is_clean_base_on_product = bool(
                        p_mean_b >= 150.0 and
                        p_mean_b >= (p_tri_base - 15.0) and
                        p_std_b <= 16.0
                    )
                    req_area_dyn = max(30, min(70, int(2.0 * np.sqrt(tri_valid_area))))
                    cond_stain = bool(
                        is_clean_base_on_product and
                        area >= req_area_dyn and
                        solidity >= 0.70 and
                        drop_b >= 20.0 and
                        contrast_energy >= 1200.0 and
                        z_blob >= 2.5
                    )
                else:
                    # NGUYÊN TẮC RIÊNG CHO VÙNG HOA VĂN / NÉT VẼ DÀY ĐẶC (p_edge_cnt >= 20):
                    # Dị vật ngoại lai đè lên hoa văn (như vết đốm xanh to Tri #368 [982px], Tri #239 [719px]):
                    # 1. Quy mô diện tích dị vật vượt trội hơn kích thước/chu kỳ nét vẽ (area >= 150px hoặc tỷ lệ lớn)
                    # 2. Tính đặc khối hình học (solidity >= 0.70)
                    # 3. Độ sụt giảm độ sáng: drop_b >= 10.0 (phù hợp cho cả vết màu xanh/đỏ lẫn dị vật tối)
                    # 4. Năng lượng tương phản: contrast_energy >= 1500.0
                    req_area_pattern = max(150, min(300, int(3.5 * np.sqrt(tri_valid_area))))
                    cond_stain = bool(
                        area >= req_area_pattern and
                        solidity >= 0.70 and
                        drop_b >= 10.0 and
                        contrast_energy >= 1500.0
                    )

                if cond_stain:
                    is_intrusive_lum_blob = True

        # Phân tích Phân bố Màu sắc (Color Distribution Profile)
        dist_info = compute_triangle_color_distribution(p_c_lab, r_c_lab, non_glare)

        strong_fused_diff = (patch_fused_l2 > 22.0) & non_glare
        num_f, _, stats_f, _ = cv2.connectedComponentsWithStats(strong_fused_diff.astype(np.uint8) * 255, connectivity=8)
        max_fused_blob = max([stats_f[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_f)], default=0)

        has_color_blob = bool(max_blob_area >= 25 or max_fused_blob >= 25)
        has_lum_blob = bool(max_solid_lum_blob >= 25)

        # DINOv2 Semantic Feature Similarity per Triangle
        pts_p_feat = (pts_p * np.array([scale_x, scale_y], dtype=np.float32)).astype(np.int32)
        pts_r_feat = (pts_r * np.array([scale_x, scale_y], dtype=np.float32)).astype(np.int32)
        tri_mask_p = np.zeros((feat_h, feat_w), dtype=np.uint8)
        tri_mask_r = np.zeros((feat_h, feat_w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask_p, pts_p_feat, 1)
        cv2.fillConvexPoly(tri_mask_r, pts_r_feat, 1)

        px_p = tri_mask_p > 0
        px_r = tri_mask_r > 0
        if np.sum(px_p) > 0 and np.sum(px_r) > 0:
            v_p = np.mean(dense_p[:, px_p], axis=1)
            v_r = np.mean(dense_r[:, px_r], axis=1)
            v_p /= (np.linalg.norm(v_p) + 1e-8)
            v_r /= (np.linalg.norm(v_r) + 1e-8)
            dino_sim = float(np.dot(v_p, v_r))

            # Pointwise similarity within triangle (5th percentile catches localized cracks/scars)
            pts_sim = np.sum(dense_p[:, px_p] * dense_r[:, px_p], axis=0)
            dino_min_sim = float(np.percentile(pts_sim, 5)) if len(pts_sim) > 0 else dino_sim
        else:
            dino_sim = 1.0
            dino_min_sim = 1.0

        patch_data.append({
            "id": i,
            "dino_sim": dino_sim,
            "dino_min_sim": dino_min_sim,
            "pts_p": pts_p,
            "pts_r": pts_r,
            "p_gray": p_c_gray,
            "r_gray": r_c_gray,
            "scratch_map": scratch_map,
            "p_edge": p_c_edge,
            "r_edge": r_c_edge,
            "p_edge_cnt": p_edge_cnt,
            "r_edge_cnt": r_edge_cnt,
            "mask": mask_p,
            "broken_length": max_broken_length,
            "dark_crack": max_dark_crack,
            "white_scratch": max_white_scratch,
            "intrusive_len": max_intrusive_length,
            "blob_area": max_solid_blob,
            "struct_metric": struct_metric,
            "has_color_blob": has_color_blob,
            "has_lum_blob": has_lum_blob,
            "desc_sim": mean_desc_sim,
            "ssim": ssim_val,
            "s_score": s_score,
            "chroma_err": mean_chroma_err,
            "lum_err": mean_lum_err,
            "delta_l_shift": delta_l_shift,
            "p_l_patch": p_l_eval,
            "r_l_norm": r_l_norm,
            "fused_l2_err": mean_fused_l2_err,
            "w_l": dist_info["w_l"],
            "w_chroma": dist_info["w_chroma"],
            "w_fused": dist_info["w_fused"],
            "bhat_ab": dist_info["bhat_ab"],
            "lost_pattern_ratio": dist_info["lost_pattern_ratio"],
            "new_color_ratio": dist_info["new_color_ratio"],
            "dark_stain_ratio": dist_info.get("dark_stain_ratio", 0.0),
            "dist_info": dist_info,
            "max_chroma_blob": max_blob_area,
            "max_lum_blob": max_lum_blob,
            "max_solid_lum_blob": max_solid_lum_blob,
            "max_lum_solidity": max_lum_solidity,
            "max_blob_tri_ratio": max_blob_tri_ratio,
            "is_intrusive_lum_blob": is_intrusive_lum_blob,
            "max_fused_blob": max_fused_blob,
            "patch_chroma_diff": patch_chroma_diff,
            "patch_lum_diff": patch_lum_diff,
            "patch_fused_l2": patch_fused_l2,
            "v_indices": set(tri.vertex_indices),
            "ori_corr": ori_corr,
            "is_parallel": is_parallel_texture,
            "is_ink_stain": is_ink_stain,
            "is_void_damage": is_void_damage,
            "ink_area": max_ink_blob,
        })

    # Statistical distribution across all triangles for robust dual-gating
    s_mean, s_std = float(np.mean(struct_scores)), float(np.std(struct_scores))
    c_mean, c_std = float(np.mean(color_diffs)), float(np.std(color_diffs))
    
    fused_l2_metrics = [p["fused_l2_err"] for p in patch_data]
    f_mean, f_std = float(np.mean(fused_l2_metrics)), float(np.std(fused_l2_metrics))

    lum_metrics = [p["lum_err"] for p in patch_data]
    lum_mean, lum_std = float(np.mean(lum_metrics)), float(np.std(lum_metrics))

    w_chroma_metrics = [p["w_chroma"] for p in patch_data]
    wc_mean, wc_std = float(np.mean(w_chroma_metrics)), float(np.std(w_chroma_metrics))

    wl_metrics = [p["w_l"] for p in patch_data]
    wl_mean, wl_std = float(np.mean(wl_metrics)), float(np.std(wl_metrics))

    dino_metrics = [p["dino_sim"] for p in patch_data]
    dino_mean, dino_std = float(np.mean(dino_metrics)), float(np.std(dino_metrics))

    # Layer 1 distribution
    l1_metrics = [p["struct_metric"] for p in patch_data]
    l1_mean, l1_std = float(np.mean(l1_metrics)), float(np.std(l1_metrics))

    # Identify core Layer 1 defect triangles (z > 1.8 and physical evidence)
    core_edges_l1 = set()
    for item in patch_data:
        z_l1 = (item["struct_metric"] - l1_mean) / (l1_std + 1e-8)
        item["z_l1"] = z_l1
        if item["id"] in [433, 455]:
            print(f"DEBUG TRI #{item['id']}: z_l1={z_l1:.2f}, blob={item['blob_area']}, dark={item['dark_crack']}, ssim={item['ssim']:.2f}, dino_sim={item.get('dino_sim', 0):.4f}, dino_min={item.get('dino_min_sim', 0):.4f}, p_edge={item['p_edge_cnt']}")
        # Nếu Product phần lớn không có cạnh (p_edge_cnt < 20) -> nền men trơn:
        # Chỉ công nhận lỗi khi có vết nứt sắc nét đâm xuyên nền men (intrusive >= 12), sẹo lớn (blob >= 50) hoặc dị vật mực
        if item["p_edge_cnt"] < 20:
            has_physical_l1 = bool(item["intrusive_len"] >= 12.0 or item["blob_area"] >= 50 or item["is_ink_stain"] or item.get("is_void_damage", False))
        else:
            has_physical_l1 = bool(item["dark_crack"] >= 10.0 or item["white_scratch"] >= 10.0 or 
                                   item["intrusive_len"] >= 12.0 or item["broken_length"] >= 10.0 or
                                   item["blob_area"] >= 35 or item["is_ink_stain"] or item.get("is_void_damage", False))
        item["has_physical_l1"] = has_physical_l1
        if (z_l1 > 1.8 and has_physical_l1) or item["is_ink_stain"] or item.get("is_void_damage", False) or item["dark_crack"] >= 12.0 or item["intrusive_len"] >= 12.0:
            v = list(item["v_indices"])
            core_edges_l1.add(frozenset([v[0], v[1]]))
            core_edges_l1.add(frozenset([v[1], v[2]]))
            core_edges_l1.add(frozenset([v[2], v[0]]))

    triangle_details = []
    layer1_flagged = []
    layer2_flagged = []

    for item in patch_data:
        i = item["id"]
        v = list(item["v_indices"])
        shares_core_edge = (frozenset([v[0], v[1]]) in core_edges_l1 or 
                            frozenset([v[1], v[2]]) in core_edges_l1 or 
                            frozenset([v[2], v[0]]) in core_edges_l1)

        # Layer 2 Decision: Kết hợp Phân bố màu (Color Distribution) + Chroma (a, b) + Luminance (L)
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        z_f = (item["fused_l2_err"] - f_mean) / (f_std + 1e-8)
        z_lum = (item["lum_err"] - lum_mean) / (lum_std + 1e-8)
        z_wc = (item["w_chroma"] - wc_mean) / (wc_std + 1e-8)
        z_wl = (item["w_l"] - wl_mean) / (wl_std + 1e-8)
        z_dino = (dino_mean - item["dino_sim"]) / (dino_std + 1e-8)
        item["z_c"] = z_c
        item["z_f"] = z_f
        item["z_lum"] = z_lum
        item["z_wc"] = z_wc
        item["z_wl"] = z_wl
        item["z_dino"] = z_dino

        # =====================================================================
        # PHƯƠNG PHÁP 1: KÊNH a, b (CHROMA - SẮC THÁI & HOA VĂN MÀU THUẦN TÚY)
        # =====================================================================
        # 1.1. Mất màu hoa văn (Pattern Loss / Faded Paint):
        is_pattern_loss = bool(
            item["lost_pattern_ratio"] >= 0.35 and
            item["w_chroma"] >= 10.0 and
            item["chroma_err"] >= 18.0
        )

        # 1.2. Dị vật / Vết ố màu ngoại lai (Color Stain Intrusion - ví dụ vết màu xanh lam Tri #239):
        is_stain_intrusion = bool(
            item["max_chroma_blob"] >= 35 and
            item["new_color_ratio"] >= 0.20 and
            item["w_chroma"] >= 12.0
        )

        # 1.3. Khuyết tật đổi màu sắc thái cục bộ mạnh (Chroma Anomaly):
        is_chroma_damage = bool(
            z_c > 2.5 and
            item["chroma_err"] >= 24.0 and
            item["max_chroma_blob"] >= 35 and
            item["w_chroma"] >= 10.0
        )

        # 1.4. Phá vỡ khoảng cách phân bố Wasserstein màu sắc:
        is_w_chroma_damage = bool(
            z_wc > 2.5 and
            item["w_chroma"] >= 18.0 and
            (item["bhat_ab"] >= 0.30 or item["max_chroma_blob"] >= 35)
        )

        # Chống báo ảo khi màu sắc thực chất được bảo toàn:
        is_color_shift_only = bool(
            item["bhat_ab"] < 0.20 and
            item["w_chroma"] < 6.0 and
            item["new_color_ratio"] < 0.10
        )

        item["is_pattern_loss"] = is_pattern_loss
        item["is_stain_intrusion"] = is_stain_intrusion
        item["is_chroma_damage"] = is_chroma_damage
        item["is_w_chroma_damage"] = is_w_chroma_damage
        item["is_color_shift_only"] = is_color_shift_only

        # TỔNG HỢP PHƯƠNG PHÁP 1 (KÊNH ab THUẦN TÚY):
        is_ab_method_damage = bool(
            (is_stain_intrusion or is_pattern_loss or is_chroma_damage or is_w_chroma_damage) and
            not is_color_shift_only
        )
        item["is_ab_method_damage"] = is_ab_method_damage

        # =====================================================================
        # PHƯƠNG PHÁP 2: KÊNH L (LUMINANCE) KẾT HỢP ĐẶC TRƯNG HỌC SÂU DINOv2
        # (Loại bỏ quy tắc cứng Dark Stain, dùng đặc trưng DINOv2 chống lệch viền/độ sắc nét)
        # =====================================================================
        # Bất thường đặc trưng bề mặt qua DINOv2 (Semantic / Texture Anomaly):
        # 1. Độ tương đồng đặc trưng sụt giảm cục bộ sâu (dino_sim < 0.82)
        # DINO HẬU KIỂM (GATEKEEPER):
        # Nếu DINO xác nhận hoa văn/chất liệu cực kỳ đồng nhất (dino_sim >= 0.915),
        # chứng minh sai số độ sáng chỉ là do lệch vi nắn affine trên tam giác dẹt -> Loại bỏ báo ảo!
        is_high_sim_clean = bool(item["dino_sim"] >= 0.915)
        intrusive_verified = bool(item.get("is_intrusive_lum_blob", False) and not is_high_sim_clean)

        # NGUYÊN TẮC VẬT LÝ KÊNH L DÀNH RIÊNG CHO VÙNG HOA VĂN (p_edge_cnt >= 20):
        # Nếu năng lượng phân bố độ sáng được bảo toàn (w_l < 8.0) và ngữ nghĩa DINOv2 cao (dino_sim >= 0.88),
        # chứng minh nét vẽ chỉ trượt pha 1px cục bộ mà không có dị vật ngoại lai đè lên!
        is_pattern_region = bool(item.get("p_edge_cnt", 0) >= 20)
        is_lum_conserved_pattern = bool(is_pattern_region and item.get("w_l", 0.0) < 8.0 and item["dino_sim"] >= 0.88)

        # Khuyết tật độ sáng kênh L:
        if is_pattern_region:
            is_lum_damage = bool(
                (item.get("is_ink_stain", False) or intrusive_verified) and
                (item["lum_err"] >= 18.0) and
                not is_lum_conserved_pattern
            )
        else:
            is_lum_damage = bool(
                (item.get("is_ink_stain", False) or intrusive_verified) and
                (item["lum_err"] >= 18.0)
            )
        item["is_lum_damage"] = is_lum_damage

        # TỔNG HỢP PHƯƠNG PHÁP 2 (KÊNH L):
        is_l_method_damage = bool(is_lum_damage)
        item["is_l_method_damage"] = is_l_method_damage

        # 2.3. Kết hợp Fused L2 (Tổng hòa sai số điểm ảnh L và ab):
        is_fused_damage = bool(z_f > 2.8 and item["fused_l2_err"] >= 25.0 and item["max_fused_blob"] >= 35)
        item["is_fused_damage"] = is_fused_damage

        # QUYẾT ĐỊNH LAYER 2: HỢP NHẤT CẢ KÊNH MÀU SẮC (ab) VÀ ĐỘ SÁNG (L)
        is_l2_damage = bool(is_l_method_damage or is_ab_method_damage or is_fused_damage)

        # ---------------------------------------------------------------------
        # FUSION MLP ADAPTER EVALUATION (DINOv2 + LAYER 1 HYBRID)
        # ---------------------------------------------------------------------
        from fusion_mlp_adapter import predict_triangle_defect_probability
        mlp_prob = predict_triangle_defect_probability(item)
        item["mlp_prob"] = mlp_prob

        # ---------------------------------------------------------------------
        # LAYER 1 ĐÃ ĐƯỢC BỎ THEO YÊU CẦU: HỆ THỐNG CHỈ DÙNG LAYER 2 (MÀU SẮC & MEN SỨ)
        # ---------------------------------------------------------------------
        is_l1_damage = False

        # ---------------------------------------------------------------------
        # BƯỚC 2: TẠO MESH MỚI TỪ TAM GIÁC VÀ CÁC CẠNH KỀ ĐỂ THẨM ĐỊNH & TRIỆT TIÊU FALSE POSITIVE
        # ---------------------------------------------------------------------
        # Khi tam giác i bị nghi ngờ ở Bước 1 (nhưng không phải dị vật mực loang hiển nhiên):
        # Gộp tam giác i cùng tất cả các tam giác lân cận có chung cạnh với nó để tạo thành 1 mesh mới (1-ring cluster).
        # Tái căn chỉnh và đối chiếu cấu trúc vĩ mô trên mesh mới này:
        # Nếu trên mesh mới, cấu trúc tương quan cao (corr >= 0.60 hoặc ssim >= 0.60)
        # chứng minh sai lệch ở tam giác i chỉ là lỗi vi nắn affine (False Positive) -> HỦY BỎ NGHI VẤN!
        if is_l1_damage and not item.get("is_void_damage", False) and not item.get("is_ink_stain", False):
            nbr_ids = triangle_neighbors.get(i, [])
            if nbr_ids:
                cluster_tri_ids = [i] + nbr_ids
                pts_p_list = [patch_data[t_id]["pts_p"] for t_id in cluster_tri_ids]
                pts_r_list = [patch_data[t_id]["pts_r"] for t_id in cluster_tri_ids]
                pts_p_all = np.vstack(pts_p_list)
                pts_r_all = np.vstack(pts_r_list)

                min_x = max(0, int(np.min(pts_p_all[:, 0])) - 4)
                min_y = max(0, int(np.min(pts_p_all[:, 1])) - 4)
                max_x = min(w, int(np.max(pts_p_all[:, 0])) + 4)
                max_y = min(h, int(np.max(pts_p_all[:, 1])) + 4)

                bw = max_x - min_x
                bh = max_y - min_y

                if bw >= 20 and bh >= 20:
                    hull_p = cv2.convexHull(pts_p_all.astype(np.int32))
                    mask_macro = np.zeros((bh, bw), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_macro, hull_p - np.array([min_x, min_y]), 255)

                    roi_p = p_gray[min_y:max_y, min_x:max_x]
                    M_macro, _ = cv2.estimateAffinePartial2D(pts_r_all, pts_p_all - np.array([min_x, min_y]))
                    if M_macro is not None:
                        roi_r_warped = cv2.warpAffine(r_gray, M_macro, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                        p_vals_m = roi_p[mask_macro > 0].astype(np.float32)
                        r_vals_m = roi_r_warped[mask_macro > 0].astype(np.float32)

                        if len(p_vals_m) > 40 and np.std(p_vals_m) > 5.0 and np.std(r_vals_m) > 5.0:
                            p_norm_m = (p_vals_m - np.mean(p_vals_m)) / (np.std(p_vals_m) + 1e-6)
                            r_norm_m = (r_vals_m - np.mean(r_vals_m)) / (np.std(r_vals_m) + 1e-6)
                            corr_macro = float(np.mean(p_norm_m * r_norm_m))
                            ssim_macro = float(ssim(roi_p, roi_r_warped, data_range=255))

                            # Nếu toàn bộ mesh mới gồm các cạnh kề cho thấy độ tương quan cao (corr >= 0.60 hoặc ssim >= 0.60),
                            # và không có vết nứt sâu xuyên thủng (dark_crack < 20 và intrusive_len < 20),
                            # chứng tỏ đây chỉ là vi sai lệch nắn affine đơn lẻ -> HỦY BỎ FALSE POSITIVE!
                            if (corr_macro >= 0.60 or ssim_macro >= 0.60) and item.get("dark_crack", 0) < 20.0 and item.get("intrusive_len", 0) < 20.0:
                                is_l1_damage = False

        # ---------------------------------------------------------------------
        # BƯỚC 3: SUB-MESH VERIFICATION CHO LAYER 2 (MÀU SẮC & MEN SỨ)
        # ---------------------------------------------------------------------
        item["submesh_cleared_l2"] = False
        nbr_ids = triangle_neighbors.get(i, [])
        if ENABLE_SUBMESH_VERIFICATION and is_l2_damage and nbr_ids:
            cluster_tri_ids = [i] + nbr_ids
            m_p_sub = np.zeros((h, w), dtype=np.uint8)
            m_r_sub = np.zeros((h, w), dtype=np.uint8)
            for tid in cluster_tri_ids:
                cv2.fillConvexPoly(m_p_sub, patch_data[tid]["pts_p"].astype(np.int32), 255)
                cv2.fillConvexPoly(m_r_sub, patch_data[tid]["pts_r"].astype(np.int32), 255)

            m_r_target = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(m_r_target, patch_data[i]["pts_r"].astype(np.int32), 255)

            p_sub_px = m_p_sub > 0
            r_sub_px = m_r_sub > 0
            if np.any(p_sub_px) and np.any(r_sub_px):
                p_chroma_sub = np.sqrt((p_lab[..., 1][p_sub_px] - 128.0)**2 + (p_lab[..., 2][p_sub_px] - 128.0)**2)
                r_chroma_sub = np.sqrt((r_lab[..., 1][r_sub_px] - 128.0)**2 + (r_lab[..., 2][r_sub_px] - 128.0)**2)

                from scipy.stats import wasserstein_distance
                w_sub_c = float(wasserstein_distance(p_chroma_sub, r_chroma_sub))

                # Bhattacharyya trên Sub-mesh
                h_p_sub = cv2.calcHist([p_lab], [1, 2], m_p_sub, [16, 16], [0, 256, 0, 256])
                h_r_sub = cv2.calcHist([r_lab], [1, 2], m_r_sub, [16, 16], [0, 256, 0, 256])
                pdf_p = h_p_sub / (np.sum(h_p_sub) + 1e-8)
                pdf_r = h_r_sub / (np.sum(h_r_sub) + 1e-8)
                pdf_p_s = cv2.GaussianBlur(pdf_p, (3, 3), 0.75)
                pdf_r_s = cv2.GaussianBlur(pdf_r, (3, 3), 0.75)
                bh_sub = float(cv2.compareHist(pdf_p_s.astype(np.float32), pdf_r_s.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))

                # -------------------------------------------------------------
                # CÂN BẰNG ĐỘ SÁNG SUB-MESH ĐỂ SO SÁNH & PHÁT HIỆN ĐỈNH DỊ VẬT NGOẠI LAI
                # (Luminance Alignment for Comparison & Outlier Peak Detection)
                # -------------------------------------------------------------
                p_l_sub = p_lab[..., 0][p_sub_px]
                r_l_sub = r_lab[..., 0][r_sub_px]

                # 1. Tính độ lệch độ sáng nền (Baseline Shift) của vùng Sub-mesh để so sánh
                med_lp_sub = float(np.median(p_l_sub))
                med_lr_sub = float(np.median(r_l_sub))
                delta_l_baseline = float(med_lr_sub - med_lp_sub)

                # 2. Đưa độ sáng về cùng một mốc chuẩn ĐỂ SO SÁNH (chỉ mảng cục bộ, không sửa mesh/ảnh gốc):
                r_target_px = m_r_target > 0
                r_l_target_raw = r_lab[..., 0][r_target_px]
                r_l_target_aligned = r_l_target_raw - delta_l_baseline
                r_l_sub_aligned = r_l_sub - delta_l_baseline

                # 3. Quét đỉnh cách xa (Outlier Peak Detection) trên kênh Luminance:
                occ_p_lum = np.zeros(256, dtype=bool)
                p_lum_bins = np.clip(np.round(p_l_sub).astype(int), 0, 255)
                occ_p_lum[p_lum_bins] = True

                # Dung sai chênh lệch độ sáng tự nhiên/bóng đổ (tol_l = 22.0)
                tol_l = 22
                occ_p_dilated = cv2.dilate(occ_p_lum.astype(np.uint8).reshape(1, -1), np.ones((1, tol_l * 2 + 1), dtype=np.uint8))[0] > 0

                # Các pixel của Return có độ sáng lạ nằm ngoài phân bố của Product:
                r_aligned_bins = np.clip(np.round(r_l_target_aligned).astype(int), 0, 255)
                is_outlier_lum_px = ~occ_p_dilated[r_aligned_bins]

                outlier_lum_map = np.zeros((h, w), dtype=np.uint8)
                outlier_lum_map[r_target_px] = (is_outlier_lum_px.astype(np.uint8) * 255)
                n_out_l, _, stats_out_l, _ = cv2.connectedComponentsWithStats(outlier_lum_map, connectivity=8)
                max_outlier_lum_blob = max([stats_out_l[lbl, cv2.CC_STAT_AREA] for lbl in range(1, n_out_l)], default=0)

                # Đỉnh lạ tối màu (vết mực, đốm đen, dị vật sâu):
                is_dark_outlier_peak = bool(
                    max_outlier_lum_blob >= 18 and
                    np.any(is_outlier_lum_px) and
                    float(np.mean(r_l_target_aligned[is_outlier_lum_px])) < 85.0
                )

                # 4. Quét đỉnh cách xa trên kênh màu (Chroma Outlier Peak):
                p_a_sub = p_lab[..., 1][p_sub_px]
                p_b_sub = p_lab[..., 2][p_sub_px]
                p_a_idx = np.clip(np.round(p_a_sub).astype(int), 0, 255)
                p_b_idx = np.clip(np.round(p_b_sub).astype(int), 0, 255)
                occ_ab = np.zeros((256, 256), dtype=np.uint8)
                occ_ab[p_a_idx, p_b_idx] = 255
                occ_ab_dilated = cv2.dilate(occ_ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0

                r_a_target = r_lab[..., 1][r_target_px]
                r_b_target = r_lab[..., 2][r_target_px]
                r_a_idx = np.clip(np.round(r_a_target).astype(int), 0, 255)
                r_b_idx = np.clip(np.round(r_b_target).astype(int), 0, 255)
                is_outlier_color_px = ~occ_ab_dilated[r_a_idx, r_b_idx]

                outlier_color_map = np.zeros((h, w), dtype=np.uint8)
                outlier_color_map[r_target_px] = (is_outlier_color_px.astype(np.uint8) * 255)
                n_out_c, _, stats_out_c, _ = cv2.connectedComponentsWithStats(outlier_color_map, connectivity=8)
                max_outlier_color_blob = max([stats_out_c[lbl, cv2.CC_STAT_AREA] for lbl in range(1, n_out_c)], default=0)
                is_color_outlier_peak = bool(max_outlier_color_blob >= 18)

                has_outlier_peak = bool(is_dark_outlier_peak or is_color_outlier_peak)

                # A. Đối chiếu nét đen có sẵn trên Sub-mesh:
                r_dark_target = (r_lab[..., 0] < 80.0) & (m_r_target > 0)
                p_dark_sub = (p_lab[..., 0] < 80.0) & (m_p_sub > 0)
                n_dark_r = int(np.sum(r_dark_target))
                n_dark_p = int(np.sum(p_dark_sub))

                if n_dark_r >= 15:
                    if n_dark_p >= 15:
                        dist_p_dark = cv2.distanceTransform(cv2.bitwise_not(p_dark_sub.astype(np.uint8) * 255), cv2.DIST_L2, 3)
                        dark_dists = dist_p_dark[r_dark_target]
                        pct_near_existing = float(np.mean(dark_dists <= 6.0))
                        mean_dark_dist = float(np.mean(dark_dists))
                    else:
                        pct_near_existing = 0.0
                        mean_dark_dist = 999.0
                else:
                    pct_near_existing = 1.0
                    mean_dark_dist = 0.0

                is_existing_stroke = bool(
                    (pct_near_existing >= 0.70 and mean_dark_dist <= 7.0 and item.get("w_l", 0.0) <= 18.0) or
                    (pct_near_existing >= 0.55 and mean_dark_dist <= 10.0 and item.get("lum_err", 0.0) < 18.0)
                )

                # Profile histogram để vẽ biểu đồ debug sub-mesh
                bins_sub_c = np.linspace(0, 80, 25)
                hp_sub_c, _ = np.histogram(p_chroma_sub, bins=bins_sub_c, density=True)
                hr_sub_c, _ = np.histogram(r_chroma_sub, bins=bins_sub_c, density=True)

                # Profile histogram độ sáng Sub-mesh (sau khi cân bằng độ sáng)
                bins_sub_l = np.linspace(0, 255, 30)
                hp_sub_l, _ = np.histogram(p_l_sub, bins=bins_sub_l, density=True)
                hr_sub_l_aligned, _ = np.histogram(r_l_sub_aligned, bins=bins_sub_l, density=True)

                item["submesh_w_c"] = w_sub_c
                item["submesh_bh"] = bh_sub
                item["submesh_hp_c"] = hp_sub_c
                item["submesh_hr_c"] = hr_sub_c
                item["submesh_hp_l"] = hp_sub_l
                item["submesh_hr_l_aligned"] = hr_sub_l_aligned
                item["delta_l_baseline"] = delta_l_baseline
                item["max_outlier_lum_blob"] = max_outlier_lum_blob
                item["max_outlier_color_blob"] = max_outlier_color_blob
                item["has_outlier_peak"] = has_outlier_peak
                item["is_dark_outlier_peak"] = is_dark_outlier_peak
                item["is_color_outlier_peak"] = is_color_outlier_peak
                item["pct_near_existing_dark"] = pct_near_existing
                item["submesh_cluster"] = cluster_tri_ids

                # B. Thẩm định Bảo toàn màu sắc & Độ sáng (Color & Lum Conservation) trên Sub-mesh:
                # Điều kiện bảo toàn:
                # 1. Sau khi cân bằng sáng, KHÔNG có đỉnh ngoại lai cách xa (not has_outlier_peak)
                # 2. Hoặc phân bố màu sắc Chroma hội tụ chặt chẽ (w_sub_c <= 3.5 hoặc <= 7.5 với bh <= 0.35)
                # 3. Không có nứt vỡ vật lý (dark_crack < 10, intrusive_len < 10, white_scratch < 10)
                is_color_conserved = bool(
                    (not has_outlier_peak or (w_sub_c <= 3.5 and not is_dark_outlier_peak)) and
                    (w_sub_c <= 3.5 or (w_sub_c <= 7.5 and bh_sub <= 0.35) or (not has_outlier_peak and item.get("w_l", 0.0) <= 18.0)) and
                    item.get("new_color_ratio", 0.0) < 0.15 and
                    item.get("dark_crack", 0) < 10.0 and item.get("intrusive_len", 0) < 10.0 and
                    item.get("white_scratch", 0) < 10.0 and
                    not is_dark_outlier_peak and
                    not is_color_outlier_peak and
                    not item.get("is_intrusive_lum_blob", False)
                )

                if is_color_conserved:
                    is_l2_damage = False
                    item["submesh_cleared_l2"] = True
                    # Nếu tam giác không hề có nứt vỡ vật lý và không có vết mực loang:
                    if item.get("dark_crack", 0) < 10.0 and item.get("white_scratch", 0) < 10.0 and item.get("intrusive_len", 0) < 10.0 and not item.get("is_ink_stain", False):
                        is_l1_damage = False

        item["z_c"] = z_c
        item["is_l1"] = False
        item["is_l2"] = is_l2_damage
        item["is_fused"] = is_l2_damage
        triangle_details.append(item)

        if is_l2_damage:
            layer2_flagged.append(i)

    fused_flagged = [t["id"] for t in triangle_details if t["is_fused"]]

    # -------------------------------------------------------------------------
    # LAYER 1: 100% THUẦN ĐEN TRẮNG / GRAYSCALE (KHÔNG DÙNG ẢNH MÀU)
    # -------------------------------------------------------------------------
    # 1. Overlay Layer 1 vẽ đè lên ẢNH XÁM RETURN (GRAYSCALE)
    r_gray_base = cv2.cvtColor(return_img, cv2.COLOR_BGR2GRAY)
    vis_l1_overlay = cv2.cvtColor(r_gray_base, cv2.COLOR_GRAY2BGR)
    overlay_l1 = vis_l1_overlay.copy()

    for t in triangle_details:
        pts = t["pts_r"].astype(np.int32).reshape((-1, 1, 2))
        cx = int(np.mean(t["pts_r"][:, 0]))
        cy = int(np.mean(t["pts_r"][:, 1]))
        if t["is_l1"]:
            cv2.fillPoly(overlay_l1, [pts], (0, 0, 255))
            cv2.polylines(vis_l1_overlay, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_l1_overlay, str(t["id"]), (cx - 10, cy + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        else:
            cv2.polylines(vis_l1_overlay, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
            draw_text_with_shadow(vis_l1_overlay, str(t["id"]), (cx - 8, cy + 3), font_scale=0.32, color=(0, 255, 0), thickness=1)

    vis_l1_overlay = cv2.addWeighted(overlay_l1, 0.4, vis_l1_overlay, 0.6, 0)
    cv2.putText(vis_l1_overlay, f"Layer 1 (Grayscale): {len(layer1_flagged)} khac nhau / {len(triangles)} tam giac",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if layer1_flagged else (0, 255, 0), 2)
    cv2.imwrite(os.path.join(dir_l1, "06_structure_damage_overlay.jpg"), vis_l1_overlay)

    # Làm sạch các thư mục trước khi xuất
    for d in [dir_l1_match, dir_l1_diff, dir_l2_l_match, dir_l2_l_diff, dir_l2_ab_match, dir_l2_ab_diff, dir_submesh]:
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass

    matching_items = [t for t in triangle_details if not t["is_l1"]]
    different_items = [t for t in triangle_details if t["is_l1"]]

    def build_triangle_card(item, is_diff):
        p_box = cv2.resize(item["p_gray"], (110, 110))
        r_box = cv2.resize(item["r_gray"], (110, 110))
        ep_box = cv2.resize(item["p_edge"], (110, 110))
        er_box = cv2.resize(item["r_edge"], (110, 110))
        # Nếu Product phần lớn không có cạnh (p_edge_cnt < 20) và không có vết nứt sắc nét đâm qua men:
        # không so sánh cạnh, tránh vẽ đốm nhiễu lác đác ở cột Diff Edge và Scratch/Crack
        if item.get("p_edge_cnt", 0) < 20 and item.get("intrusive_len", 0.0) < 12.0:
            diff_box = np.zeros_like(ep_box)
            scratch_box = np.zeros_like(ep_box)
        else:
            diff_box = cv2.absdiff(ep_box, er_box)
            scratch_box = cv2.resize(item["scratch_map"], (110, 110))

        p_3ch = cv2.cvtColor(p_box, cv2.COLOR_GRAY2BGR)
        r_3ch = cv2.cvtColor(r_box, cv2.COLOR_GRAY2BGR)
        ep_3ch = cv2.cvtColor(ep_box, cv2.COLOR_GRAY2BGR)
        er_3ch = cv2.cvtColor(er_box, cv2.COLOR_GRAY2BGR)
        diff_3ch = cv2.cvtColor(diff_box, cv2.COLOR_GRAY2BGR)
        scratch_3ch = cv2.cvtColor(scratch_box, cv2.COLOR_GRAY2BGR)

        cv2.putText(p_3ch, "Product", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(r_3ch, "Return", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(ep_3ch, "Canny P (03)", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
        cv2.putText(er_3ch, "Canny R (04)", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
        cv2.putText(diff_3ch, "Diff Edge", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(scratch_3ch, "Scratch/Crack", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)

        cols = np.hstack([p_3ch, r_3ch, ep_3ch, er_3ch, diff_3ch, scratch_3ch])

        # Header bar
        header = np.full((32, cols.shape[1], 3), 30, dtype=np.uint8)
        status_txt = "KHAC NHAU (DEFECT)" if is_diff else "GIONG NHAU (INTACT)"
        status_color = (0, 0, 255) if is_diff else (0, 255, 0)
        ink_str = f" | INK_STAIN ({item.get('ink_area', 0)}px)" if item.get("is_ink_stain", False) else ""
        info_txt = f"Tri #{item['id']} [{status_txt}]{ink_str} | Corr: {item['ori_corr']:.2f} | Crack: {item['dark_crack']:.1f}px | Scratch: {item['white_scratch']:.1f}px | Broken: {item['broken_length']:.1f}px | Blob: {item['blob_area']}px"
        cv2.putText(header, info_txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1)

        card = np.vstack([header, cols])
        return card

    # Xuất toàn bộ các tam giác KHÁC NHAU vào folder Layer 1 02_khac_nhau
    diff_cards = []
    for item in different_items:
        card = build_triangle_card(item, is_diff=True)
        cv2.imwrite(os.path.join(dir_l1_diff, f"tri_{item['id']:04d}.jpg"), card)
        diff_cards.append(card)

    if diff_cards:
        # Nếu có tam giác khác nhau, lưu thêm ảnh ghép tổng hợp (mỗi trang tối đa 20 tam giác)
        for page_idx, chunk_start in enumerate(range(0, len(diff_cards), 20)):
            chunk = diff_cards[chunk_start:chunk_start + 20]
            grid_diff = np.vstack(chunk)
            cv2.imwrite(os.path.join(dir_l1_diff, f"00_tong_hop_page_{page_idx + 1:02d}.jpg"), grid_diff)

    # LAYER 1 ĐÃ ĐƯỢC BỎ: Không cần xuất các thẻ giống nhau tốn tài nguyên
    # (Hệ thống tập trung 100% vào Layer 2 Màu sắc & Men sứ)

    # Save Layer 2 Overlay (VẼ TRỰC TIẾP LÊN ẢNH RETURN ĐỂ THẤY VẾT TRÓC MEN / MÀU)
    vis_l2_overlay = return_img.copy()
    overlay_l2 = return_img.copy()
    for t in triangle_details:
        pts = t["pts_r"].astype(np.int32).reshape((-1, 1, 2))
        cx = int(np.mean(t["pts_r"][:, 0]))
        cy = int(np.mean(t["pts_r"][:, 1]))
        if t["is_l2"]:
            cv2.fillPoly(overlay_l2, [pts], (0, 0, 255))
            cv2.polylines(vis_l2_overlay, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_l2_overlay, str(t["id"]), (cx - 10, cy + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        else:
            cv2.polylines(vis_l2_overlay, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
            draw_text_with_shadow(vis_l2_overlay, str(t["id"]), (cx - 8, cy + 3), font_scale=0.32, color=(0, 255, 0), thickness=1)
    vis_l2_overlay = cv2.addWeighted(overlay_l2, 0.4, vis_l2_overlay, 0.6, 0)
    cv2.putText(vis_l2_overlay, f"Layer 2 (Color): {len(layer2_flagged)} damaged", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if layer2_flagged else (0, 255, 0), 3)
    cv2.imwrite(os.path.join(dir_l2, "02_color_damage_overlay.jpg"), vis_l2_overlay)

    # -------------------------------------------------------------------------
    # FOLDER 05: FUSION DECISION
    # -------------------------------------------------------------------------
    vis_final_p = product_img.copy()
    vis_final_r = return_img.copy()

    for t in triangle_details:
        pts_p = t["pts_p"].astype(np.int32).reshape((-1, 1, 2))
        pts_r = t["pts_r"].astype(np.int32).reshape((-1, 1, 2))
        cx_p = int(np.mean(t["pts_p"][:, 0]))
        cy_p = int(np.mean(t["pts_p"][:, 1]))
        cx_r = int(np.mean(t["pts_r"][:, 0]))
        cy_r = int(np.mean(t["pts_r"][:, 1]))
        if t["is_fused"]:
            cv2.fillPoly(vis_final_p, [pts_p], (0, 0, 255))
            cv2.polylines(vis_final_p, [pts_p], True, (0, 0, 255), 2)
            cv2.fillPoly(vis_final_r, [pts_r], (0, 0, 255))
            cv2.polylines(vis_final_r, [pts_r], True, (0, 0, 255), 2)
            draw_text_with_shadow(vis_final_p, str(t["id"]), (cx_p - 10, cy_p + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
            draw_text_with_shadow(vis_final_r, str(t["id"]), (cx_r - 10, cy_r + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        else:
            cv2.polylines(vis_final_p, [pts_p], True, (0, 255, 0), 1)
            cv2.polylines(vis_final_r, [pts_r], True, (0, 255, 0), 1)
            draw_text_with_shadow(vis_final_p, str(t["id"]), (cx_p - 8, cy_p + 3), font_scale=0.32, color=(0, 255, 0), thickness=1)
            draw_text_with_shadow(vis_final_r, str(t["id"]), (cx_r - 8, cy_r + 3), font_scale=0.32, color=(0, 255, 0), thickness=1)

    cv2.imwrite(os.path.join(dir_fusion, "01_product_damage_marked.jpg"), vis_final_p)
    cv2.imwrite(os.path.join(dir_fusion, "02_return_damage_marked.jpg"), vis_final_r)

    # Side-by-side comparison
    max_h = 960
    sc = max_h / max(product_img.shape[0], 1)
    new_w, new_h = int(product_img.shape[1] * sc), int(product_img.shape[0] * sc)
    p_comp = cv2.resize(vis_final_p, (new_w, new_h))
    r_comp = cv2.resize(vis_final_r, (new_w, new_h))
    side_by_side = np.hstack([p_comp, r_comp])
    cv2.putText(side_by_side, f"Defects Found: {len(fused_flagged)} triangles", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if fused_flagged else (0, 255, 0), 3)
    cv2.imwrite(os.path.join(dir_fusion, "03_side_by_side_marked.jpg"), side_by_side)

    # -------------------------------------------------------------------------
    # XUẤT THẺ BẰNG CHỨNG RIÊNG CHO TỪNG HỆ MÀU LAYER 2 (KÊNH L VÀ KÊNH ab)
    # (Đã loại bỏ hoàn toàn folder 06 và 07 theo yêu cầu)
    # -------------------------------------------------------------------------
    crop_h, crop_w = 250, 375

    for t in triangle_details:
        pts_p = t["pts_p"]
        pts_r = t["pts_r"]

        x0_p, y0_p = int(max(0, pts_p[:, 0].min() - 30)), int(max(0, pts_p[:, 1].min() - 30))
        x1_p, y1_p = int(min(w, pts_p[:, 0].max() + 30)), int(min(h, pts_p[:, 1].max() + 30))
        x0_r, y0_r = int(max(0, pts_r[:, 0].min() - 30)), int(max(0, pts_r[:, 1].min() - 30))
        x1_r, y1_r = int(min(w, pts_r[:, 0].max() + 30)), int(min(h, pts_r[:, 1].max() + 30))

        crop_p = product_img[y0_p:y1_p, x0_p:x1_p].copy()
        crop_r = return_img[y0_r:y1_r, x0_r:x1_r].copy()

        poly_p = np.array([[pt[0] - x0_p, pt[1] - y0_p] for pt in pts_p], dtype=np.int32).reshape((-1, 1, 2))
        poly_r = np.array([[pt[0] - x0_r, pt[1] - y0_r] for pt in pts_r], dtype=np.int32).reshape((-1, 1, 2))

        # =====================================================================
        # 1. XUẤT CHO LAYER 2 - HỆ ĐỘ SÁNG L (LUMINANCE & MEN SỨ)
        # =====================================================================
        is_l_defect = t.get("is_lum_damage", False)
        target_dir_l = dir_l2_l_diff if is_l_defect else dir_l2_l_match
        poly_color_l = (0, 0, 255) if is_l_defect else (0, 255, 0)

        cp_l = crop_p.copy()
        cr_l = crop_r.copy()
        cv2.polylines(cp_l, [poly_p], True, poly_color_l, 2, cv2.LINE_AA)
        cv2.polylines(cr_l, [poly_r], True, poly_color_l, 2, cv2.LINE_AA)
        cpr_l = cv2.resize(cp_l, (crop_w, crop_h)) if cp_l.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crr_l = cv2.resize(cr_l, (crop_w, crop_h)) if cr_l.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        row_top_l = np.hstack([cpr_l, crr_l])
        status_l = "LOI (DEFECT)" if is_l_defect else "BINH THUONG (NORMAL)"
        cv2.putText(row_top_l, f"Product (Left) vs Return (Right) - KENH L", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(row_top_l, f"Tri #{t['id']}: [{status_l}] dino_sim={t.get('dino_sim', 0):.3f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, poly_color_l, 2)
        cv2.putText(row_top_l, f"dL={t.get('lum_err', 0.0):.1f}, SolidBlob={t.get('max_solid_lum_blob', 0)}px", (10, crop_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        mask_tri = t["mask"]
        p_l_raw = t.get("p_l_patch", np.zeros((72, 72), dtype=np.float32))
        r_l_aligned = t.get("r_l_norm", np.zeros((72, 72), dtype=np.float32))
        lum_diff = t.get("patch_lum_diff", np.zeros((72, 72), dtype=np.float32))

        p_l_u8 = np.clip(p_l_raw, 0, 255).astype(np.uint8)
        p_l_bgr = cv2.cvtColor(p_l_u8, cv2.COLOR_GRAY2BGR)
        p_l_bgr[mask_tri == 0] = (0, 0, 0)
        p_l_res = cv2.resize(p_l_bgr, (250, 250), interpolation=cv2.INTER_NEAREST)
        cv2.putText(p_l_res, "1. L Goc (Product)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        mean_lp = float(np.mean(p_l_raw[mask_tri > 0])) if np.any(mask_tri > 0) else 0.0
        cv2.putText(p_l_res, f"Mean L: {mean_lp:.1f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        r_l_u8 = np.clip(r_l_aligned, 0, 255).astype(np.uint8)
        r_l_bgr = cv2.cvtColor(r_l_u8, cv2.COLOR_GRAY2BGR)
        r_l_bgr[mask_tri == 0] = (0, 0, 0)
        r_l_res = cv2.resize(r_l_bgr, (250, 250), interpolation=cv2.INTER_NEAREST)
        cv2.putText(r_l_res, "2. L Return (Da can bang)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        shift_val = t.get('delta_l_shift', 0.0)
        cv2.putText(r_l_res, f"Shift: {shift_val:+.1f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        vis_diff = np.clip(lum_diff / 45.0 * 255.0, 0, 255).astype(np.uint8)
        jet_diff = cv2.applyColorMap(vis_diff, cv2.COLORMAP_JET)
        jet_diff[mask_tri == 0] = (0, 0, 0)
        diff_l_res = cv2.resize(jet_diff, (250, 250), interpolation=cv2.INTER_NEAREST)
        cv2.putText(diff_l_res, "3. L Loi (Delta L)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        sol_b = t.get('max_solid_lum_blob', 0)
        sol_ratio = t.get('max_lum_solidity', 0.0)
        cv2.putText(diff_l_res, f"dL={t.get('lum_err', 0.0):.1f} | solid={sol_b}px (s={sol_ratio:.2f})", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        row_bot_l = np.hstack([p_l_res, r_l_res, diff_l_res])
        card_l = np.vstack([row_top_l, row_bot_l])
        pfx_l = "loi" if is_l_defect else "khop"
        fname_l = f"{pfx_l}_tri_{t['id']:03d}_he_mau_L.jpg"
        cv2.imwrite(os.path.join(target_dir_l, fname_l), card_l)

        # =====================================================================
        # 2. XUẤT CHO LAYER 2 - HỆ MÀU SẮC ab (CHROMA & HOA VĂN MÀU)
        # =====================================================================
        is_ab_defect = t.get("is_ab_method_damage", False)
        target_dir_ab = dir_l2_ab_diff if is_ab_defect else dir_l2_ab_match
        poly_color_ab = (0, 0, 255) if is_ab_defect else (0, 255, 0)

        cp_ab = crop_p.copy()
        cr_ab = crop_r.copy()
        cv2.polylines(cp_ab, [poly_p], True, poly_color_ab, 2, cv2.LINE_AA)
        cv2.polylines(cr_ab, [poly_r], True, poly_color_ab, 2, cv2.LINE_AA)
        cpr_ab = cv2.resize(cp_ab, (crop_w, crop_h)) if cp_ab.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crr_ab = cv2.resize(cr_ab, (crop_w, crop_h)) if cr_ab.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        row_top_ab = np.hstack([cpr_ab, crr_ab])
        status_ab = "LOI (DEFECT)" if is_ab_defect else "BINH THUONG (NORMAL)"
        cv2.putText(row_top_ab, f"Product (Left) vs Return (Right) - KENH ab", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(row_top_ab, f"Tri #{t['id']}: [{status_ab}]", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, poly_color_ab, 2)
        cv2.putText(row_top_ab, f"dE_ab={t.get('chroma_err', 0.0):.1f}, Blob_ab={t.get('max_chroma_blob', 0)}px", (10, crop_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        chroma_diff = t.get("patch_chroma_diff", np.zeros((72, 72), dtype=np.float32))
        vis_diff_ab = np.clip(chroma_diff / 35.0 * 255.0, 0, 255).astype(np.uint8)
        jet_diff_ab = cv2.applyColorMap(vis_diff_ab, cv2.COLORMAP_JET)
        jet_diff_ab[mask_tri == 0] = (0, 0, 0)
        diff_ab_res = cv2.resize(jet_diff_ab, (375, 250), interpolation=cv2.INTER_NEAREST)
        cv2.putText(diff_ab_res, "Delta_E Chroma Heatmap (a, b)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(diff_ab_res, f"Mean dE_ab: {t.get('chroma_err', 0.0):.1f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Panel thông tin phân bố màu ab
        info_panel_ab = np.full((250, 375, 3), 20, dtype=np.uint8)
        cv2.putText(info_panel_ab, "Thong so he mau ab (Chroma):", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Sai so trung binh: {t.get('chroma_err', 0.0):.2f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Khoi mau bat thuong (Blob): {t.get('max_chroma_blob', 0)} px", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Khoang cach Wasserstein W_c: {t.get('w_chroma', 0.0):.2f}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Ty le mat hoa van mau: {t.get('lost_pattern_ratio', 0.0)*100:.1f}%", (15, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Ket luan: {'LOI DOI MAU' if is_ab_defect else 'HOA VAN DONG MAU'}", (15, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, poly_color_ab, 1)

        row_bot_ab = np.hstack([diff_ab_res, info_panel_ab])
        card_ab = np.vstack([row_top_ab, row_bot_ab])
        pfx_ab = "loi" if is_ab_defect else "khop"
        fname_ab = f"{pfx_ab}_tri_{t['id']:03d}_he_mau_ab.jpg"
        cv2.imwrite(os.path.join(target_dir_ab, fname_ab), card_ab)

        # 4. Xuất Sub-mesh Debug Card nếu tam giác có phân tích sub-mesh
        if ENABLE_SUBMESH_VERIFICATION and "submesh_cluster" in t:
            submesh_card = render_submesh_debug_card(product_img, return_img, t, patch_data)
            status_tag = "cleared" if t.get("submesh_cleared_l2") else "confirmed"
            sm_fname = f"submesh_tri_{t['id']:04d}_{status_tag}.jpg"
            cv2.imwrite(os.path.join(dir_submesh, sm_fname), submesh_card)

    print(f"\n[Summary]:")
    print(f"  Total Triangles Inspected:  {len(triangles)}")
    print(f"  Layer 1 Defective (Cracks): {len(layer1_flagged)}")
    print(f"  Layer 2 Defective (Color):  {len(layer2_flagged)}")
    print(f"  Final True Defects Found:   {len(fused_flagged)}")
    print(f"\nGranular Debug Subfolders ready at: {base_debug_dir}")

    return {
        "triangles": len(triangles),
        "l1_defects": len(layer1_flagged),
        "l2_defects": len(layer2_flagged),
        "total_defects": len(fused_flagged),
        "debug_dir": base_debug_dir,
        "triangle_details": triangle_details,
    }


if __name__ == "__main__":
    # Run on vase 1 vs 2 (No damage)
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    print("\n--- TEST CASE 1: 1.jpg vs 2.jpg (UNDAMAGED VASE) ---")
    run_detailed_debug(p1, p2, base_debug_dir=os.path.join(V2_DIR, "debug", "01_undamaged_vase"))

    # Run on scarred vase (Real damage)
    s1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    s2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    if os.path.exists(s1) and os.path.exists(s2):
        print("\n--- TEST CASE 2: test_nobg vs test_nobg - scar (REAL SCAR DAMAGE) ---")
        run_detailed_debug(s1, s2, base_debug_dir=os.path.join(V2_DIR, "debug", "02_scar_defect"))
