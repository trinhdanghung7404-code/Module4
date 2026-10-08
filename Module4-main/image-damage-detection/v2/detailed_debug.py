"""Detailed Granular Layer 2 (Color & Luminance) Damage Inspection.

Focus: EXACT local defect localization and transparent evidence.
Structure:
  v2/debug/
    ├── 01_keypoints/       (SuperPoint keypoints on both images)
    ├── 02_mesh/            (Delaunay triangles wireframe on both images)
    ├── 04_layer2_color/    (Dual Color L + ab heatmaps, overlays, and evidence cards)
    │     ├── 01_he_mau_L/   (Luminance evidence cards: loi_ and khop_)
    │     └── 02_he_mau_ab/  (Chroma ab evidence cards: loi_ and khop_)
    └── 05_fusion/          (Final defect marked overlays & side-by-side comparison)
"""

import os
import sys
import time
import cv2
import numpy as np
from collections import defaultdict
from scipy.stats import wasserstein_distance
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
    """Extract Layer 9 dense surface texture features via DINOv2."""
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
        # Layer 9: Surface texture representation (sensitive to micro-scars and glaze texture)
        tok = out.hidden_states[9][:, 1:, :]
        grid_size = int(round(tok.shape[1] ** 0.5))
        f = tok.reshape(1, grid_size, grid_size, 384).permute(0, 3, 1, 2)
        dense = torch.nn.functional.interpolate(f, size=(feat_h, feat_w), mode="bilinear", align_corners=False)[0]
        dense = torch.nn.functional.normalize(dense, p=2, dim=0).cpu().numpy()
        return dense, (feat_h, feat_w)


def micro_align_patches(patch_p: np.ndarray, patch_r: np.ndarray, mask: np.ndarray, max_shift: int = 4):
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
        return patch_r
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(patch_r, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def draw_text_with_shadow(img: np.ndarray, text: str, pos: tuple, font_scale: float = 0.35, color: tuple = (0, 255, 0), thickness: int = 1):
    """Draw text with shadow outline for clear visibility on bright and dark backgrounds."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def compute_triangle_color_distribution(p_lab: np.ndarray, r_lab: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Analyze color and chroma distributions invariant to sub-pixel misalignment."""
    if not np.any(valid_mask):
        return {
            "w_l": 0.0, "w_chroma": 0.0, "w_fused": 0.0,
            "bhat_ab": 0.0, "lost_pattern_ratio": 0.0, "new_color_ratio": 0.0,
            "dark_stain_ratio": 0.0,
            "hist_p_l": np.zeros(24), "hist_r_l": np.zeros(24),
            "hist_p_c": np.zeros(24), "hist_r_c": np.zeros(24),
        }

    p_l = p_lab[..., 0][valid_mask]
    r_l = r_lab[..., 0][valid_mask]
    p_a = p_lab[..., 1][valid_mask]
    r_a = r_lab[..., 1][valid_mask]
    p_b = p_lab[..., 2][valid_mask]
    r_b = r_lab[..., 2][valid_mask]

    # Median luminance normalization
    med_lp = float(np.median(p_l))
    med_lr = float(np.median(r_l))
    r_l_norm = r_l - med_lr + med_lp

    # Chroma saturation
    p_chroma = np.sqrt((p_a - 128.0) ** 2 + (p_b - 128.0) ** 2)
    r_chroma = np.sqrt((r_a - 128.0) ** 2 + (r_b - 128.0) ** 2)

    # 1. Wasserstein distance (Earth Mover's Distance)
    w_l = float(wasserstein_distance(p_l, r_l_norm))
    w_a = float(wasserstein_distance(p_a, r_a))
    w_b = float(wasserstein_distance(p_b, r_b))
    w_chroma = float(np.sqrt(w_a ** 2 + w_b ** 2))
    w_fused = float(np.sqrt(0.5 * (w_l ** 2) + 1.0 * (w_chroma ** 2)))

    # 2. 2D Chroma (a, b) Histogram with Gaussian smoothing
    mask_u8 = valid_mask.astype(np.uint8) * 255
    hist_p_ab = cv2.calcHist([p_lab], [1, 2], mask_u8, [16, 16], [0, 256, 0, 256])
    hist_r_ab = cv2.calcHist([r_lab], [1, 2], mask_u8, [16, 16], [0, 256, 0, 256])

    pdf_p = hist_p_ab / (np.sum(hist_p_ab) + 1e-8)
    pdf_r = hist_r_ab / (np.sum(hist_r_ab) + 1e-8)
    pdf_p_smooth = cv2.GaussianBlur(pdf_p, (3, 3), 0.75)
    pdf_r_smooth = cv2.GaussianBlur(pdf_r, (3, 3), 0.75)

    bhat_ab = float(cv2.compareHist(pdf_p_smooth.astype(np.float32), pdf_r_smooth.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))

    # 3. Pattern Loss & Color Stain Intrusion
    p_pattern_mask = p_chroma >= 18.0
    p_pattern_cnt = int(np.sum(p_pattern_mask))
    if p_pattern_cnt >= 25:
        lost_px = np.sum(r_chroma[p_pattern_mask] < 10.0)
        lost_pattern_ratio = float(lost_px) / float(p_pattern_cnt)
    else:
        lost_pattern_ratio = 0.0

    p_plain_mask = p_chroma < 10.0
    p_plain_cnt = int(np.sum(p_plain_mask))
    if p_plain_cnt >= 25:
        new_px = np.sum(r_chroma[p_plain_mask] >= 20.0)
        new_color_ratio = float(new_px) / float(p_plain_cnt)
    else:
        new_color_ratio = 0.0

    # Achromatic Dark Stain
    dark_stain_mask = (p_l >= 85.0) & (r_l_norm < 75.0) & ((p_l - r_l_norm) >= 22.0)
    dark_stain_cnt = int(np.sum(dark_stain_mask))
    dark_stain_ratio = float(dark_stain_cnt) / float(len(p_l)) if len(p_l) > 0 else 0.0

    # 1D Histograms for visualization
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


def run_detailed_debug(product_path: str, return_path: str, base_debug_dir: str = None):
    """Execute detailed damage detection focusing solely on Layer 2 (Luminance + Chroma + DINOv2)."""
    if base_debug_dir is None:
        base_debug_dir = os.path.join(V2_DIR, "debug")

    dir_kps = os.path.join(base_debug_dir, "01_keypoints")
    dir_mesh = os.path.join(base_debug_dir, "02_mesh")
    dir_l2 = os.path.join(base_debug_dir, "04_layer2_color")
    dir_l2_l = os.path.join(dir_l2, "01_he_mau_L")
    dir_l2_ab = os.path.join(dir_l2, "02_he_mau_ab")
    dir_fusion = os.path.join(base_debug_dir, "05_fusion")

    dir_l2_l_match = os.path.join(dir_l2_l, "01_giong_nhau")
    dir_l2_l_diff = os.path.join(dir_l2_l, "02_khac_nhau")
    dir_l2_ab_match = os.path.join(dir_l2_ab, "01_giong_nhau")
    dir_l2_ab_diff = os.path.join(dir_l2_ab, "02_khac_nhau")

    all_debug_dirs = [
        dir_kps, dir_mesh, dir_l2, dir_fusion,
        dir_l2_l_match, dir_l2_l_diff,
        dir_l2_ab_match, dir_l2_ab_diff
    ]
    for d in all_debug_dirs:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("  RUNNING LAYER 2 (CHROMA + LUMINANCE + DINOV2) DAMAGE INSPECTION")
    print(f"  Debug Root: {base_debug_dir}")
    print("=" * 70)

    # 1. Load images
    product_img = cv2.imread(product_path)
    return_img = cv2.imread(return_path)
    if product_img is None or return_img is None:
        raise FileNotFoundError(f"Could not load images: {product_path} or {return_path}")

    h, w = product_img.shape[:2]
    if return_img.shape[:2] != (h, w):
        return_img = cv2.resize(return_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Segment & Normalization
    segmenter = ObjectSegmenter()
    p_seg = segmenter.segment(product_img)
    r_seg = segmenter.segment(return_img)

    normalizer = ImageNormalizer()
    return_normalized = normalizer.normalize(return_img, product_img, r_seg["mask"], p_seg["mask"])
    return_ct = normalizer.color_transfer(return_img, product_img, r_seg["mask"], p_seg["mask"])

    # 3. Registration (SuperPoint + RANSAC)
    registrator = ImageRegistration(max_keypoints=None)
    reg_result = registrator.register(product_img, return_normalized, p_seg["mask"], r_seg["mask"])

    all_p_kps = reg_result["all_product_keypoints"]
    all_r_kps = reg_result["all_return_keypoints"]
    inliers_p = reg_result["product_points"]
    inliers_r = reg_result["return_points"]

    # FOLDER 01: KEYPOINTS
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

    # 4. Mesh Construction
    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(inliers_p, inliers_r, np.arange(len(inliers_p)))

    # Xây dựng đồ thị kề cận Topo (Topological 1-Ring Star: chia sẻ cạnh HOẶC chia sẻ đỉnh)
    vertex_to_triangles = defaultdict(list)
    for i, tri in enumerate(triangles):
        for v in tri.vertex_indices:
            vertex_to_triangles[v].append(i)

    triangle_neighbors = {}
    for i, tri in enumerate(triangles):
        nbrs = set()
        for v in tri.vertex_indices:
            for other_idx in vertex_to_triangles[v]:
                if other_idx != i:
                    nbrs.add(other_idx)
        triangle_neighbors[i] = list(nbrs)

    # FOLDER 02: MESH WIREFRAME
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
    # GLOBAL COLOR SPACES & HEATMAPS (LUMINANCE L + CHROMA AB)
    # -------------------------------------------------------------------------
    p_lab = cv2.cvtColor(product_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(return_normalized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Global median color cast offset removal on object mask
    p_valid = p_seg["mask"] > 0
    r_valid = r_seg["mask"] > 0
    offset_a = float(np.median(r_lab[..., 1][r_valid]) - np.median(p_lab[..., 1][p_valid]))
    offset_b = float(np.median(r_lab[..., 2][r_valid]) - np.median(p_lab[..., 2][p_valid]))
    offset_L = float(np.median(r_lab[..., 0][r_valid]) - np.median(p_lab[..., 0][p_valid]))
    r_lab[..., 1] -= offset_a
    r_lab[..., 2] -= offset_b
    r_lab_norm_L = r_lab[..., 0] - offset_L

    # Difference maps
    da = p_lab[..., 1] - r_lab[..., 1]
    db = p_lab[..., 2] - r_lab[..., 2]
    chroma_diff_map = np.sqrt(da ** 2 + db ** 2)
    lum_diff_map = np.abs(p_lab[..., 0] - r_lab_norm_L)
    fused_l2_diff_map = np.sqrt(0.5 * (lum_diff_map ** 2) + 1.0 * (chroma_diff_map ** 2))

    # Mask glare and regions outside mesh
    glare_mask = (r_lab[..., 0] > 240) | (p_lab[..., 0] > 240)
    chroma_diff_map[glare_mask] = 0.0
    chroma_diff_map[mesh_mask_r == 0] = 0.0
    lum_diff_map[glare_mask] = 0.0
    lum_diff_map[mesh_mask_r == 0] = 0.0
    fused_l2_diff_map[glare_mask] = 0.0
    fused_l2_diff_map[mesh_mask_r == 0] = 0.0

    # Export heatmaps
    norm_heatmap_c = np.clip(chroma_diff_map / 35.0 * 255.0, 0, 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(norm_heatmap_c, cv2.COLORMAP_JET)
    color_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01a_chroma_delta_e_heatmap.jpg"), color_heatmap)

    norm_heatmap_l = np.clip(lum_diff_map / 45.0 * 255.0, 0, 255).astype(np.uint8)
    lum_heatmap = cv2.applyColorMap(norm_heatmap_l, cv2.COLORMAP_JET)
    lum_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01b_luminance_diff_heatmap.jpg"), lum_heatmap)

    norm_heatmap_f = np.clip(fused_l2_diff_map / 40.0 * 255.0, 0, 255).astype(np.uint8)
    fused_heatmap = cv2.applyColorMap(norm_heatmap_f, cv2.COLORMAP_JET)
    fused_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01c_combined_layer2_heatmap.jpg"), fused_heatmap)

    # Side-by-side comparison heatmap
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
    # TRIANGLE PATCH EVALUATION (LAYER 2)
    # -------------------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    patch_data = []

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        # Canonical 72x72 patches
        patch_p, mask_p = canonical_triangle_patch(product_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(return_ct, pts_r, target_size=72)

        # Micro-align to eliminate curvature 3D distortion (+/- 4px)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=4)

        p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c_gray = clahe.apply(p_raw_gray)
        r_c_gray = clahe.apply(r_raw_gray)

        valid = mask_p > 0
        glare_px = (r_c_gray > 240) | (p_c_gray > 240)
        non_glare = valid & (~glare_px)

        # Check pattern presence in Product patch
        p_canny = cv2.Canny(p_c_gray, 40, 120)
        p_canny[~valid] = 0
        p_edge_cnt = int(np.sum(p_canny > 0))

        # LAB color patches (with global cast offset applied)
        p_c_lab = cv2.cvtColor(patch_p, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_c_lab = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_c_lab[..., 1] -= offset_a
        r_c_lab[..., 2] -= offset_b

        # Chroma error
        c_da = p_c_lab[..., 1] - r_c_lab[..., 1]
        c_db = p_c_lab[..., 2] - r_c_lab[..., 2]
        patch_chroma_diff = np.sqrt(c_da ** 2 + c_db ** 2)
        mean_chroma_err = float(np.mean(patch_chroma_diff[non_glare])) if np.any(non_glare) else 0.0

        # Asymmetric blur matching on Luminance channel
        p_l_eval = p_c_lab[..., 0].copy()
        r_l_eval = r_c_lab[..., 0].copy()

        if np.any(non_glare):
            lap_p = cv2.Laplacian(p_l_eval, cv2.CV_32F)[valid]
            lap_r = cv2.Laplacian(r_l_eval, cv2.CV_32F)[valid]
            var_p = float(np.var(lap_p)) if len(lap_p) > 0 else 0.0
            var_r = float(np.var(lap_r)) if len(lap_r) > 0 else 0.0

            sharp_ratio = (max(var_p, var_r) + 1.0) / (min(var_p, var_r) + 1.0)
            if sharp_ratio >= 2.0:
                ksize = 5 if sharp_ratio >= 3.5 else 3
                sigma = 1.2 if sharp_ratio >= 3.5 else 0.8
                if var_r > var_p:
                    r_l_eval = cv2.GaussianBlur(r_l_eval, (ksize, ksize), sigma)
                else:
                    p_l_eval = cv2.GaussianBlur(p_l_eval, (ksize, ksize), sigma)

            r_l_eval = micro_align_patches(p_l_eval, r_l_eval, mask_p, max_shift=2)

        # Robust Luminance baseline alignment via median of inliers
        if np.any(non_glare):
            l_diff_px = r_l_eval[non_glare] - p_l_eval[non_glare]
            delta_l_baseline = float(np.median(l_diff_px))
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

        # Fused error
        patch_fused_l2 = np.sqrt(0.5 * (patch_lum_diff ** 2) + 1.0 * (patch_chroma_diff ** 2))
        patch_fused_l2[~non_glare] = 0.0
        mean_fused_l2_err = float(np.mean(patch_fused_l2[non_glare])) if np.any(non_glare) else 0.0

        # Physical Chroma Blobs
        strong_color_diff = (patch_chroma_diff > 18.0) & non_glare
        num_labels, _, stats_c, _ = cv2.connectedComponentsWithStats(strong_color_diff.astype(np.uint8) * 255, connectivity=8)
        max_blob_area = max([stats_c[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_labels)], default=0)

        # Physical Luminance Blobs & Solidity
        strong_lum_diff = (patch_lum_diff > 25.0) & non_glare
        num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats(strong_lum_diff.astype(np.uint8) * 255, connectivity=8)

        tri_valid_area = max(1, int(np.sum(valid)))
        p_l_valid = p_l_eval[non_glare]
        p_tri_base = float(np.median(p_l_valid)) if len(p_l_valid) > 0 else 200.0

        p_dark_cnt = int(np.sum((p_raw_gray < 60) & non_glare))
        r_dark_cnt = int(np.sum((r_raw_gray < 60) & non_glare))
        p_dark_dil = cv2.dilate((p_raw_gray < 60).astype(np.uint8), np.ones((5, 5), np.uint8))
        is_existing_dark_stroke = bool(
            p_dark_cnt >= 120 and (r_dark_cnt / (p_dark_cnt + 1e-5) < 1.35)
        )

        max_solid_lum_blob = 0
        max_lum_solidity = 0.0
        is_intrusive_lum_blob = False

        for lbl in range(1, num_l):
            area = stats_l[lbl, cv2.CC_STAT_AREA]
            pts_blob = np.column_stack(np.where(labels_l == lbl))
            if len(pts_blob) >= 6:
                pts_xy = np.column_stack((pts_blob[:, 1], pts_blob[:, 0])).astype(np.int32)
                hull = cv2.convexHull(pts_xy)
                hull_area = float(cv2.contourArea(hull))
                solidity = float(area) / (hull_area + 1e-5)
            else:
                solidity = 1.0

            if solidity >= 0.70:
                if area > max_solid_lum_blob:
                    max_solid_lum_blob = area
                    max_lum_solidity = solidity

                m_curr_blob = (labels_l == lbl)
                p_blob_vals = p_l_eval[m_curr_blob]
                r_blob_vals = r_l_norm[m_curr_blob]
                p_mean_b = float(np.mean(p_blob_vals))
                p_std_b = float(np.std(p_blob_vals))
                r_mean_b = float(np.mean(r_blob_vals))
                drop_b = p_mean_b - r_mean_b
                contrast_energy = float(area) * drop_b
                z_blob = drop_b / max(p_std_b, 5.0)

                if p_edge_cnt < 20:
                    # Clean porcelain base rules
                    is_clean_base = bool(p_mean_b >= 150.0 and p_mean_b >= (p_tri_base - 15.0) and p_std_b <= 16.0)
                    req_area_dyn = max(30, min(70, int(2.0 * np.sqrt(tri_valid_area))))
                    cond_stain = bool(
                        is_clean_base and area >= req_area_dyn and solidity >= 0.70 and
                        drop_b >= 20.0 and contrast_energy >= 1200.0 and z_blob >= 2.5
                    )
                else:
                    # Dense pattern rules: Large foreign blob overlay
                    req_area_pattern = max(150, min(300, int(3.5 * np.sqrt(tri_valid_area))))
                    cond_stain = bool(
                        not is_existing_dark_stroke and
                        area >= req_area_pattern and solidity >= 0.70 and
                        drop_b >= 10.0 and contrast_energy >= 1500.0
                    )

                if cond_stain:
                    is_intrusive_lum_blob = True

        # Achromatic ink stain detection (pitch dark drop)
        p_vals_g = p_raw_gray[non_glare].astype(np.float32)
        r_vals_g = r_raw_gray[non_glare].astype(np.float32)
        if len(p_vals_g) > 10 and np.std(r_vals_g) > 1e-3:
            med_p_g = np.median(p_vals_g)
            med_r_g = np.median(r_vals_g)
            mad_p_g = np.median(np.abs(p_vals_g - med_p_g)) * 1.4826
            mad_r_g = np.median(np.abs(r_vals_g - med_r_g)) * 1.4826
            scale_g = np.clip(mad_p_g / (mad_r_g + 1e-5), 0.7, 1.4)
            r_norm_g = (r_raw_gray.astype(np.float32) - med_r_g) * scale_g + med_p_g
        else:
            r_norm_g = r_raw_gray.astype(np.float32)

        diff_g = np.abs(r_norm_g - p_raw_gray.astype(np.float32))
        pitch_drop = (diff_g > 40.0) & (r_raw_gray < 55) & ((p_raw_gray.astype(np.float32) - r_raw_gray.astype(np.float32)) >= 45.0) & non_glare
        ink_opened = cv2.morphologyEx(pitch_drop.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_open_3)
        num_lbl_ink, lbls_ink, stats_ink, _ = cv2.connectedComponentsWithStats(ink_opened, connectivity=8)

        is_ink_stain = False
        max_ink_blob = 0
        p_dark_cnt = int(np.sum((p_raw_gray < 60) & non_glare))
        r_dark_cnt = int(np.sum((r_raw_gray < 60) & non_glare))
        p_dark_dil = cv2.dilate((p_raw_gray < 60).astype(np.uint8), np.ones((5, 5), np.uint8))

        for l in range(1, num_lbl_ink):
            area = stats_ink[l, cv2.CC_STAT_AREA]
            if area >= 35:
                m_ink = (lbls_ink == l)
                if np.std(r_raw_gray[m_ink]) <= 22.0:
                    dist_ink = cv2.distanceTransform(m_ink.astype(np.uint8) * 255, cv2.DIST_L2, 3)
                    core_radius = float(np.max(dist_ink))

                    # Kiểm tra xem đây có phải là nét đen cũ có sẵn trên Product bị trượt vi nắn hay không
                    overlap_existing = float(np.sum(m_ink & (p_dark_dil > 0))) / (float(area) + 1e-5)
                    is_stroke_jitter = bool(
                        overlap_existing >= 0.40 and 
                        p_dark_cnt >= 120 and 
                        (r_dark_cnt / (p_dark_cnt + 1e-5) < 1.35)
                    )

                    if not is_stroke_jitter:
                        if p_edge_cnt >= 20:
                            is_solid_droplet = bool((core_radius >= 2.8 and area >= 45) or (core_radius >= 3.5))
                        else:
                            is_solid_droplet = bool(core_radius >= 1.8 and area >= 35)
                        if is_solid_droplet:
                            max_ink_blob = max(max_ink_blob, area)
                            is_ink_stain = True

        # Color distribution analysis
        dist_info = compute_triangle_color_distribution(p_c_lab, r_c_lab, non_glare)

        # Fused blob
        strong_fused_diff = (patch_fused_l2 > 22.0) & non_glare
        num_f, _, stats_f, _ = cv2.connectedComponentsWithStats(strong_fused_diff.astype(np.uint8) * 255, connectivity=8)
        max_fused_blob = max([stats_f[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_f)], default=0)

        # DINOv2 Semantic Similarity
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
        else:
            dino_sim = 1.0

        patch_data.append({
            "id": i,
            "tri": tri,
            "pts_p": pts_p,
            "pts_r": pts_r,
            "dino_sim": dino_sim,
            "p_edge_cnt": p_edge_cnt,
            "mask": mask_p,
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
            "max_solid_lum_blob": max_solid_lum_blob,
            "max_lum_solidity": max_lum_solidity,
            "is_intrusive_lum_blob": is_intrusive_lum_blob,
            "max_fused_blob": max_fused_blob,
            "patch_chroma_diff": patch_chroma_diff,
            "patch_lum_diff": patch_lum_diff,
            "patch_fused_l2": patch_fused_l2,
            "is_ink_stain": is_ink_stain,
            "ink_area": max_ink_blob,
        })

    # Statistical distribution across triangles for adaptive thresholds
    c_metrics = [p["chroma_err"] for p in patch_data]
    c_mean, c_std = float(np.mean(c_metrics)), float(np.std(c_metrics))

    f_metrics = [p["fused_l2_err"] for p in patch_data]
    f_mean, f_std = float(np.mean(f_metrics)), float(np.std(f_metrics))

    wc_metrics = [p["w_chroma"] for p in patch_data]
    wc_mean, wc_std = float(np.mean(wc_metrics)), float(np.std(wc_metrics))

    # -------------------------------------------------------------------------
    # STEP 4: 3-TIER FUSION DECISION ACROSS 2 CHANNELS (L & ab)
    # -------------------------------------------------------------------------
    # Bước 1: Đánh giá độc lập 2 kênh L và ab trên từng tam giác
    preliminary_data = []
    flagged_pool = set()

    for item in patch_data:
        i = item["id"]
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        z_f = (item["fused_l2_err"] - f_mean) / (f_std + 1e-8)
        z_wc = (item["w_chroma"] - wc_mean) / (wc_std + 1e-8)

        # Kênh ab (Chroma & Sắc tố màu)
        is_pattern_loss = bool(
            item["lost_pattern_ratio"] >= 0.35 and
            item["w_chroma"] >= 10.0 and
            item["chroma_err"] >= 18.0
        )
        is_stain_intrusion = bool(
            item["max_chroma_blob"] >= 35 and
            item["new_color_ratio"] >= 0.20 and
            item["w_chroma"] >= 12.0
        )
        is_chroma_damage = bool(
            z_c > 2.5 and
            item["chroma_err"] >= 24.0 and
            item["max_chroma_blob"] >= 35 and
            item["w_chroma"] >= 10.0
        )
        is_w_chroma_damage = bool(
            z_wc > 2.5 and
            item["w_chroma"] >= 18.0 and
            (item["bhat_ab"] >= 0.30 or item["max_chroma_blob"] >= 35)
        )
        is_color_shift_only = bool(
            item["bhat_ab"] < 0.20 and
            item["w_chroma"] < 6.0 and
            item["new_color_ratio"] < 0.10
        )

        # DINOv2 GATEKEEPER CHO KÊNH ab:
        # DINOv2 hiểu sâu sắc ngữ nghĩa hoa văn, vi cấu trúc men sứ và chi tiết bề mặt.
        # Nếu DINOv2 xác nhận độ tương đồng rất cao (dino_sim >= 0.900):
        # Mọi chênh lệch sắc tố ab đều chỉ là sai khác ánh sáng / nhiệt độ màu / tán xạ men,
        # không phải tổn thương thực thể -> DINOv2 phủ quyết hoàn toàn báo ảo của Kênh ab!
        is_dino_consistent = bool(item["dino_sim"] >= 0.900)

        # 1.5. Vết đen / Vết nứt / Dị vật dập tắt sắc tố màu (Dark Crack & Achromatic Void):
        # Chỉ công nhận là dị vật dập tắt sắc tố khi THỰC SỰ có xáo trộn màu sắc VÀ DINOv2 bị suy giảm (< 0.900):
        is_dark_chroma_void = bool(
            not is_dino_consistent and
            item.get("is_ink_stain", False) and 
            (item.get("max_chroma_blob", 0) >= 35 or item.get("w_chroma", 0.0) >= 12.0)
        )

        is_ab_raw = bool(
            (is_stain_intrusion or is_pattern_loss or is_chroma_damage or is_w_chroma_damage or is_dark_chroma_void) and
            not is_color_shift_only
        )

        if is_dino_consistent:
            is_ab_method_damage = False
        else:
            is_ab_method_damage = is_ab_raw

        # Kênh L (Luminance & Độ sáng men) + DINOv2 Gatekeeper
        intrusive_verified = bool(item.get("is_intrusive_lum_blob", False) and not is_dino_consistent)

        # Phân biệt Vết mực đen ngoại lai thật sự (Tri #629, #131) vs Nét vẽ đen cũ có sẵn bị trượt vi nắn (Tri #39):
        # Nếu DINOv2 rất cao (>= 0.900) và Kênh ab hoàn toàn sạch (max_chroma_blob < 20 và w_chroma < 10):
        # chứng minh nét đen đó đã có sẵn trên Product và bề mặt đồng nhất -> Bác bỏ báo ảo!
        is_existing_stroke_shift = bool(
            is_dino_consistent and 
            item.get("max_chroma_blob", 0) < 20 and 
            item.get("w_chroma", 0.0) < 10.0
        )
        real_ink_stain = bool(item.get("is_ink_stain", False) and not is_existing_stroke_shift)

        is_pattern_region = bool(item.get("p_edge_cnt", 0) >= 20)
        is_lum_conserved_pattern = bool(is_pattern_region and item.get("w_l", 0.0) < 8.0 and item["dino_sim"] >= 0.88)

        if is_pattern_region:
            is_lum_damage = bool(
                (real_ink_stain or intrusive_verified) and
                (item["lum_err"] >= 18.0) and
                not is_lum_conserved_pattern
            )
        else:
            is_lum_damage = bool(
                (real_ink_stain or intrusive_verified) and
                (item["lum_err"] >= 18.0)
            )

        is_fused_damage = bool(z_f > 2.8 and item["fused_l2_err"] >= 25.0 and item["max_fused_blob"] >= 35)

        item["z_c"] = z_c
        item["z_f"] = z_f
        item["z_wc"] = z_wc
        item["is_pattern_loss"] = is_pattern_loss
        item["is_stain_intrusion"] = is_stain_intrusion
        item["is_chroma_damage"] = is_chroma_damage
        item["is_w_chroma_damage"] = is_w_chroma_damage
        item["is_dark_chroma_void"] = is_dark_chroma_void
        item["is_ab_method_damage"] = is_ab_method_damage
        item["is_lum_damage"] = is_lum_damage
        item["is_fused_damage"] = is_fused_damage

        if is_lum_damage or is_ab_method_damage or is_fused_damage:
            flagged_pool.add(i)

        preliminary_data.append(item)

    # Bước 2: Phân loại phân tầng 3 cấp độ (3-Tier Classification)
    triangle_details = []
    tier1_confirmed = []   # Cấp 1: Chắc chắn lỗi (Đỏ)
    tier2_probable = []    # Cấp 2: Khả năng lỗi (Cam - Cụm kề cạnh)
    tier3_isolated = []    # Cấp 3: Nghi vấn riêng lẻ (Vàng - Đơn độc)

    for item in preliminary_data:
        i = item["id"]
        is_both = bool(item["is_lum_damage"] and item["is_ab_method_damage"])
        is_flagged = bool(i in flagged_pool)

        # Lỗi vật lý nặng rõ rệt (vết mực đen sâu / nứt vỡ thật sự HOẶC khối tổn thương màu lớn liên tục >= 350px)
        is_severe_ink = bool(
            item.get("is_ink_stain", False) and 
            not (item.get("dino_sim", 0) >= 0.900 and item.get("max_chroma_blob", 0) < 20) and
            item.get("ink_area", 0) >= 45
        )
        is_severe_chroma = bool(
            not (item.get("dino_sim", 0) >= 0.900) and
            item.get("max_chroma_blob", 0) >= 350 and 
            item.get("w_chroma", 0.0) >= 14.0
        )
        is_severe_defect = bool(is_severe_ink or is_severe_chroma)

        if not is_flagged:
            item["defect_tier"] = 0
            item["tier_label"] = "BINH THUONG"
            item["tier_color"] = (0, 255, 0)
            item["is_final_defect"] = False
            item["is_isolated"] = False
        elif is_both or is_severe_defect:
            # CẤP 1: Chắc chắn lỗi (Cả 2 kênh cùng nhận HOẶC Vết nứt/mực đen sâu rõ rệt)
            item["defect_tier"] = 1
            item["tier_label"] = "CHAC CHAN LOI"
            item["tier_color"] = (0, 0, 255)  # Đỏ đậm
            item["is_final_defect"] = True
            item["is_isolated"] = False
            tier1_confirmed.append(i)
        else:
            # Chỉ 1 kênh nhận và là lỗi vừa/nhẹ: Kiểm tra các tam giác kề cạnh
            nbrs = triangle_neighbors.get(i, [])
            has_adjacent_flagged = any(nbr in flagged_pool for nbr in nbrs)
            if has_adjacent_flagged:
                # CẤP 2: Khả năng lỗi (Cụm >= 2 tam giác kề cạnh)
                item["defect_tier"] = 2
                item["tier_label"] = "KHA NANG LOI"
                item["tier_color"] = (0, 140, 255)  # Cam
                item["is_final_defect"] = True
                item["is_isolated"] = False
                tier2_probable.append(i)
            else:
                # CẤP 3: Nghi vấn riêng lẻ (Đơn độc)
                item["defect_tier"] = 3
                item["tier_label"] = "NGHI VAN RIENG LE"
                item["tier_color"] = (0, 255, 255)  # Vàng
                item["is_final_defect"] = False
                item["is_isolated"] = True
                tier3_isolated.append(i)

        item["is_l1"] = False
        item["is_l2"] = item["is_final_defect"]
        item["is_fused"] = item["is_final_defect"]
        triangle_details.append(item)

    confirmed_and_probable = tier1_confirmed + tier2_probable

    # -------------------------------------------------------------------------
    # CLEAN OLD CARDS BEFORE EXPORT
    # -------------------------------------------------------------------------
    for d in [dir_l2_l_match, dir_l2_l_diff, dir_l2_ab_match, dir_l2_ab_diff]:
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass

    # Save Layer 2 Overlay theo 3 màu cấp độ
    vis_l2_overlay = return_img.copy()
    overlay_l2 = return_img.copy()
    for t in triangle_details:
        pts = t["pts_r"].astype(np.int32).reshape((-1, 1, 2))
        cx = int(np.mean(t["pts_r"][:, 0]))
        cy = int(np.mean(t["pts_r"][:, 1]))
        tier = t["defect_tier"]
        color = t["tier_color"]

        if tier == 1:
            cv2.fillPoly(overlay_l2, [pts], (0, 0, 255))
            cv2.polylines(vis_l2_overlay, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_l2_overlay, str(t["id"]), (cx - 10, cy + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        elif tier == 2:
            cv2.fillPoly(overlay_l2, [pts], (0, 140, 255))
            cv2.polylines(vis_l2_overlay, [pts], True, (0, 140, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_l2_overlay, str(t["id"]), (cx - 10, cy + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        elif tier == 3:
            cv2.polylines(vis_l2_overlay, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_l2_overlay, str(t["id"]), (cx - 8, cy + 3), font_scale=0.38, color=(0, 255, 255), thickness=1)
        else:
            cv2.polylines(vis_l2_overlay, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
            draw_text_with_shadow(vis_l2_overlay, str(t["id"]), (cx - 8, cy + 3), font_scale=0.30, color=(0, 255, 0), thickness=1)

    vis_l2_overlay = cv2.addWeighted(overlay_l2, 0.4, vis_l2_overlay, 0.6, 0)
    legend_banner = f"Chac chan (Do): {len(tier1_confirmed)} | Kha nang (Cam): {len(tier2_probable)} | Rieng le (Vang): {len(tier3_isolated)}"
    cv2.putText(vis_l2_overlay, legend_banner, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(dir_l2, "02_color_damage_overlay.jpg"), vis_l2_overlay)

    # -------------------------------------------------------------------------
    # FOLDER 05: FUSION OVERLAYS
    # -------------------------------------------------------------------------
    vis_final_p = product_img.copy()
    vis_final_r = return_img.copy()
    overlay_final_p = product_img.copy()
    overlay_final_r = return_img.copy()

    for t in triangle_details:
        pts_p = t["pts_p"].astype(np.int32).reshape((-1, 1, 2))
        pts_r = t["pts_r"].astype(np.int32).reshape((-1, 1, 2))
        cx_p = int(np.mean(t["pts_p"][:, 0]))
        cy_p = int(np.mean(t["pts_p"][:, 1]))
        cx_r = int(np.mean(t["pts_r"][:, 0]))
        cy_r = int(np.mean(t["pts_r"][:, 1]))
        tier = t["defect_tier"]
        color = t["tier_color"]

        if tier == 1:
            cv2.fillPoly(overlay_final_p, [pts_p], (0, 0, 255))
            cv2.fillPoly(overlay_final_r, [pts_r], (0, 0, 255))
            cv2.polylines(vis_final_p, [pts_p], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.polylines(vis_final_r, [pts_r], True, (0, 0, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_final_p, str(t["id"]), (cx_p - 10, cy_p + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
            draw_text_with_shadow(vis_final_r, str(t["id"]), (cx_r - 10, cy_r + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        elif tier == 2:
            cv2.fillPoly(overlay_final_p, [pts_p], (0, 140, 255))
            cv2.fillPoly(overlay_final_r, [pts_r], (0, 140, 255))
            cv2.polylines(vis_final_p, [pts_p], True, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.polylines(vis_final_r, [pts_r], True, (0, 140, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_final_p, str(t["id"]), (cx_p - 10, cy_p + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
            draw_text_with_shadow(vis_final_r, str(t["id"]), (cx_r - 10, cy_r + 4), font_scale=0.45, color=(0, 255, 255), thickness=1)
        elif tier == 3:
            cv2.polylines(vis_final_p, [pts_p], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.polylines(vis_final_r, [pts_r], True, (0, 255, 255), 2, cv2.LINE_AA)
            draw_text_with_shadow(vis_final_p, str(t["id"]), (cx_p - 8, cy_p + 3), font_scale=0.38, color=(0, 255, 255), thickness=1)
            draw_text_with_shadow(vis_final_r, str(t["id"]), (cx_r - 8, cy_r + 3), font_scale=0.38, color=(0, 255, 255), thickness=1)
        else:
            cv2.polylines(vis_final_p, [pts_p], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(vis_final_r, [pts_r], True, (0, 255, 0), 1, cv2.LINE_AA)
            draw_text_with_shadow(vis_final_p, str(t["id"]), (cx_p - 8, cy_p + 3), font_scale=0.30, color=(0, 255, 0), thickness=1)
            draw_text_with_shadow(vis_final_r, str(t["id"]), (cx_r - 8, cy_r + 3), font_scale=0.30, color=(0, 255, 0), thickness=1)

    vis_final_p = cv2.addWeighted(overlay_final_p, 0.4, vis_final_p, 0.6, 0)
    vis_final_r = cv2.addWeighted(overlay_final_r, 0.4, vis_final_r, 0.6, 0)

    cv2.imwrite(os.path.join(dir_fusion, "01_product_damage_marked.jpg"), vis_final_p)
    cv2.imwrite(os.path.join(dir_fusion, "02_return_damage_marked.jpg"), vis_final_r)

    # Side-by-side comparison
    max_h = 960
    sc = max_h / max(product_img.shape[0], 1)
    new_w, new_h = int(product_img.shape[1] * sc), int(product_img.shape[0] * sc)
    p_comp = cv2.resize(vis_final_p, (new_w, new_h))
    r_comp = cv2.resize(vis_final_r, (new_w, new_h))
    side_by_side = np.hstack([p_comp, r_comp])
    legend_sbs = f"Chac chan (Do): {len(tier1_confirmed)} | Kha nang (Cam): {len(tier2_probable)} | Rieng le (Vang): {len(tier3_isolated)}"
    cv2.putText(side_by_side, legend_sbs, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(dir_fusion, "03_side_by_side_marked.jpg"), side_by_side)

    # -------------------------------------------------------------------------
    # EXPORT EVIDENCE CARDS (KÊNH L & KÊNH ab)
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

        # 1. KÊNH L (Luminance evidence card)
        is_l_defect = t.get("is_lum_damage", False)
        target_dir_l = dir_l2_l_diff if is_l_defect else dir_l2_l_match
        poly_color_l = t["tier_color"] if is_l_defect else (0, 255, 0)

        cp_l = crop_p.copy()
        cr_l = crop_r.copy()
        cv2.polylines(cp_l, [poly_p], True, poly_color_l, 2, cv2.LINE_AA)
        cv2.polylines(cr_l, [poly_r], True, poly_color_l, 2, cv2.LINE_AA)
        cpr_l = cv2.resize(cp_l, (crop_w, crop_h)) if cp_l.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crr_l = cv2.resize(cr_l, (crop_w, crop_h)) if cr_l.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        row_top_l = np.hstack([cpr_l, crr_l])
        cv2.putText(row_top_l, "Product (Left) vs Return (Right) - KENH L", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(row_top_l, f"Tri #{t['id']}: [{t['tier_label']}] dino_sim={t.get('dino_sim', 0):.3f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, poly_color_l, 2)
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

        # 2. KÊNH ab (Chroma evidence card)
        is_ab_defect = t.get("is_ab_method_damage", False)
        target_dir_ab = dir_l2_ab_diff if is_ab_defect else dir_l2_ab_match
        poly_color_ab = t["tier_color"] if is_ab_defect else (0, 255, 0)

        cp_ab = crop_p.copy()
        cr_ab = crop_r.copy()
        cv2.polylines(cp_ab, [poly_p], True, poly_color_ab, 2, cv2.LINE_AA)
        cv2.polylines(cr_ab, [poly_r], True, poly_color_ab, 2, cv2.LINE_AA)
        cpr_ab = cv2.resize(cp_ab, (crop_w, crop_h)) if cp_ab.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crr_ab = cv2.resize(cr_ab, (crop_w, crop_h)) if cr_ab.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        row_top_ab = np.hstack([cpr_ab, crr_ab])
        cv2.putText(row_top_ab, "Product (Left) vs Return (Right) - KENH ab", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(row_top_ab, f"Tri #{t['id']}: [{t['tier_label']}]", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, poly_color_ab, 2)
        cv2.putText(row_top_ab, f"dE_ab={t.get('chroma_err', 0.0):.1f}, Blob_ab={t.get('max_chroma_blob', 0)}px", (10, crop_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        chroma_diff = t.get("patch_chroma_diff", np.zeros((72, 72), dtype=np.float32))
        vis_diff_ab = np.clip(chroma_diff / 35.0 * 255.0, 0, 255).astype(np.uint8)
        jet_diff_ab = cv2.applyColorMap(vis_diff_ab, cv2.COLORMAP_JET)
        jet_diff_ab[mask_tri == 0] = (0, 0, 0)
        diff_ab_res = cv2.resize(jet_diff_ab, (375, 250), interpolation=cv2.INTER_NEAREST)
        cv2.putText(diff_ab_res, "Delta_E Chroma Heatmap (a, b)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(diff_ab_res, f"Mean dE_ab: {t.get('chroma_err', 0.0):.1f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        info_panel_ab = np.full((250, 375, 3), 20, dtype=np.uint8)
        cv2.putText(info_panel_ab, "Thong so he mau ab (Chroma):", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Sai so trung binh: {t.get('chroma_err', 0.0):.2f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Khoi mau bat thuong (Blob): {t.get('max_chroma_blob', 0)} px", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Khoang cach Wasserstein W_c: {t.get('w_chroma', 0.0):.2f}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Ty le mat hoa van mau: {t.get('lost_pattern_ratio', 0.0)*100:.1f}%", (15, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(info_panel_ab, f"- Phan tang: {t['tier_label']}", (15, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, poly_color_ab, 1)

        row_bot_ab = np.hstack([diff_ab_res, info_panel_ab])
        card_ab = np.vstack([row_top_ab, row_bot_ab])
        pfx_ab = "loi" if is_ab_defect else "khop"
        fname_ab = f"{pfx_ab}_tri_{t['id']:03d}_he_mau_ab.jpg"
        cv2.imwrite(os.path.join(target_dir_ab, fname_ab), card_ab)

    print(f"\n[Summary 3-Tier Fusion]:")
    print(f"  Total Triangles Inspected:          {len(triangles)}")
    print(f"  Tier 1 - Chac chan loi (Do):        {len(tier1_confirmed)}")
    print(f"  Tier 2 - Kha nang loi (Cam):        {len(tier2_probable)}")
    print(f"  Tier 3 - Nghi van rieng le (Vang):  {len(tier3_isolated)}")
    print(f"  Tong loi xac dinh (Tier 1 + 2):     {len(confirmed_and_probable)}")
    print(f"\nGranular Debug Subfolders ready at: {base_debug_dir}")

    return {
        "triangles": len(triangles),
        "tier1_confirmed": len(tier1_confirmed),
        "tier2_probable": len(tier2_probable),
        "tier3_isolated": len(tier3_isolated),
        "l1_defects": 0,
        "l2_defects": len(confirmed_and_probable),
        "total_defects": len(confirmed_and_probable),
        "debug_dir": base_debug_dir,
        "triangle_details": triangle_details,
    }


if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    print("\n--- TEST CASE: 1.jpg vs 2.jpg ---")
    run_detailed_debug(p1, p2, base_debug_dir=os.path.join(V2_DIR, "debug", "01_undamaged_vase"))
