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
from skimage.metrics import structural_similarity as ssim

V2_DIR = os.path.abspath('v2')
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder, Vertex, Triangle
from step3_pipeline import canonical_triangle_patch


def micro_align_patches(patch_p: np.ndarray, patch_r: np.ndarray, mask: np.ndarray, max_shift: int = 2, extra_patch: np.ndarray = None):
    """Compensate for minor 3D surface curvature distortion (+/- 2px shift)."""
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


def run_detailed_debug(product_path: str, return_path: str, base_debug_dir: str = None):
    """Execute detailed damage detection with granular visual outputs organized in subfolders."""
    if base_debug_dir is None:
        base_debug_dir = os.path.join(V2_DIR, "debug")

    dir_kps = os.path.join(base_debug_dir, "01_keypoints")
    dir_mesh = os.path.join(base_debug_dir, "02_mesh")
    dir_l1 = os.path.join(base_debug_dir, "03_layer1_structure")
    dir_l2 = os.path.join(base_debug_dir, "04_layer2_color")
    dir_fusion = os.path.join(base_debug_dir, "05_fusion")
    dir_crops = os.path.join(base_debug_dir, "06_defect_crops")

    for d in [dir_kps, dir_mesh, dir_l1, dir_l2, dir_fusion, dir_crops]:
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

    # 4. Mesh
    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(inliers_p, inliers_r, np.arange(len(inliers_p)))

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
    # FOLDER 04: LAYER 2 - COLOR / DELTA_E HEATMAP (RESTRICTED STRICTLY TO MESH)
    # -------------------------------------------------------------------------
    p_lab = cv2.cvtColor(product_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(return_normalized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Global median color offset removal on object
    p_valid = p_seg["mask"] > 0
    r_valid = r_seg["mask"] > 0
    offset_a = float(np.median(r_lab[..., 1][r_valid]) - np.median(p_lab[..., 1][p_valid]))
    offset_b = float(np.median(r_lab[..., 2][r_valid]) - np.median(p_lab[..., 2][p_valid]))
    r_lab[..., 1] -= offset_a
    r_lab[..., 2] -= offset_b

    # Measure chromatic difference delta_E_ab (a and b only)
    da = p_lab[..., 1] - r_lab[..., 1]
    db = p_lab[..., 2] - r_lab[..., 2]
    chroma_diff_map = np.sqrt(da ** 2 + db ** 2)

    # Ignore glare / specular reflection (where L channel is saturated near 255)
    glare_mask = (r_lab[..., 0] > 240) | (p_lab[..., 0] > 240)
    chroma_diff_map[glare_mask] = 0.0
    # Restrict strictly to mesh triangles
    chroma_diff_map[mesh_mask_r == 0] = 0.0

    # Color difference heatmap (color-coded: blue=0, green=10, yellow=20, red=35+)
    norm_heatmap = np.clip(chroma_diff_map / 35.0 * 255.0, 0, 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    color_heatmap[mesh_mask_r == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(dir_l2, "01_chroma_delta_e_heatmap.jpg"), color_heatmap)

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

        # Step 3.1: Micro-align to eliminate curvature 3D distortion
        patch_r_aligned, patch_re_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2, extra_patch=patch_re)

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
    inner_triangle = (X_m >= 2) & (Y_m >= 2) & (X_m + Y_m <= 68)

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
                # Neu ori_corr >= 0.85 (hoa van khop hoan toan): mo rong dung sai khoang cach len 5.0 neu goc trung khop
                tol_dist = 6.0 if (ori_corr < 0.85) else 8.0
                near_shifted = (d <= tol_dist) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=30.0)
                # Neu hoa van song song va goc khop voi hoa van Product -> khong phai crack bat thuong
                is_parallel_feature = (ori_corr >= 0.85) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted or is_parallel_feature):
                    anom_crack[y, x] = True

            for y, x in zip(*np.where(scratch_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                near_shifted = (d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted):
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

        # Chi loai bo neu ca 2 deu trang loa (flash chieu vao cho von da trang sang)
        saturated_flash = (r_raw_gray >= 248) & (p_raw_gray >= 200)
        intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 3.5) & (mag_r_raw >= 38.0) & (~saturated_flash) & inner_triangle
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0) or dist_to_p_all[y, x] > 5.5:
                anom_intrusive[y, x] = True

        cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = measure_linear_stroke(cnts_intrusive, min_len=10.0, min_aspect=1.5)

        # 3. Solid Grayscale Anomaly Blob (Scars, Chips, Stains)
        p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
        r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)
        if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
            mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
            mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
            scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_raw_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc
        else:
            r_norm_loc = r_raw_gray.astype(np.float32)

        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0
        # Loai bo cac pixel nam sat net hoa van cu (dist <= 3.5px) neu ori_corr >= 0.80 de khong bi nham dich chuyen vien thanh blob
        diff_clean = diff_loc.copy()
        if ori_corr >= 0.80 and p_edge_cnt >= 20:
            diff_clean[dist_to_p_all <= 3.5] = 0.0
        bin_diff_loc = (diff_clean > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

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

        # Composite Structural Defect Metric (Giảm mạnh trọng số blob xuống 0.1 theo yêu cầu, tăng trọng số nét gãy broken lên 1.5)
        struct_metric = (
            1.5 * max_dark_crack +
            1.5 * max_white_scratch +
            2.0 * max_intrusive_length +
            0.1 * max_solid_blob +
            1.5 * max_broken_length
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

        # --- L2: Color metrics ---
        c_da = p_c_lab[..., 1] - r_c_lab[..., 1]
        c_db = p_c_lab[..., 2] - r_c_lab[..., 2]
        patch_diff = np.sqrt(c_da ** 2 + c_db ** 2)

        # Exclude glare pixels inside patch (L > 225 is glare region)
        non_glare = (mask_p > 0) & (r_c_lab[..., 0] <= 225) & (p_c_lab[..., 0] <= 225)
        mean_chroma_err = float(np.mean(patch_diff[non_glare])) if np.any(non_glare) else 0.0
        color_diffs.append(mean_chroma_err)

        # Physical Color Blob Verification: Area >= 25px
        strong_color_diff = (patch_diff > 18.0) & non_glare
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(strong_color_diff.astype(np.uint8) * 255, connectivity=8)
        max_blob_area = max([stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_labels)], default=0)
        has_color_blob = bool(max_blob_area >= 25)

        patch_data.append({
            "id": i,
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
            "desc_sim": mean_desc_sim,
            "ssim": ssim_val,
            "s_score": s_score,
            "chroma_err": mean_chroma_err,
            "v_indices": set(tri.vertex_indices),
            "ori_corr": ori_corr,
            "is_parallel": is_parallel_texture,
        })

    # Statistical distribution across all triangles for robust dual-gating
    s_mean, s_std = float(np.mean(struct_scores)), float(np.std(struct_scores))
    c_mean, c_std = float(np.mean(color_diffs)), float(np.std(color_diffs))
    
    # Layer 1 distribution
    l1_metrics = [p["struct_metric"] for p in patch_data]
    l1_mean, l1_std = float(np.mean(l1_metrics)), float(np.std(l1_metrics))

    # Identify core Layer 1 defect triangles (z > 1.8 and physical evidence)
    core_edges_l1 = set()
    for item in patch_data:
        z_l1 = (item["struct_metric"] - l1_mean) / (l1_std + 1e-8)
        item["z_l1"] = z_l1
        # Nếu Product phần lớn không có cạnh (p_edge_cnt < 20) -> nền men trơn:
        # Chỉ công nhận lỗi khi có vết nứt sắc nét đâm xuyên nền men (intrusive >= 12) hoặc sẹo lớn (blob >= 50)
        if item["p_edge_cnt"] < 20:
            has_physical_l1 = bool(item["intrusive_len"] >= 12.0 or item["blob_area"] >= 50)
        else:
            has_physical_l1 = bool(item["dark_crack"] >= 10.0 or item["white_scratch"] >= 10.0 or 
                                   item["intrusive_len"] >= 12.0 or item["broken_length"] >= 10.0 or
                                   item["blob_area"] >= 35)
        item["has_physical_l1"] = has_physical_l1
        if z_l1 > 1.8 and has_physical_l1:
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

        # Layer 2 Decision: Men bi bien doi mau / soc ngoai lai
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        # Nguong L2: z_c > 2.8 va chroma_err >= 24.0 va co blob mau ro ret
        is_l2_damage = bool(z_c > 2.8 and item["chroma_err"] >= 24.0 and item["has_color_blob"])

        # QUY LUẬT VẬT LÝ VỀ ĐỒ GỐM SỨ (PHYSICAL PORCELAIN LAWS):
        # 1. Với đồ gốm có hoa văn vẽ (Painted Porcelain, p_edge_cnt >= 20):
        #    - Tổn thương vật lý (sứt mẻ, tróc men, gãy nét) luôn làm tróc lớp men màu, để lộ xương gốm 
        #      -> Bắt buộc phải có sự hội tụ đa tầng: is_l2_damage AND has_physical_l1.
        #    - Hoặc vết nứt đen đâm xuyên nền cực kỳ sắc nét (Intrusive Crack >= 12px) không trùng bất kỳ nét cũ nào.
        # 2. Với đồ gốm men trơn / không hoa văn (Plain / Monochrome, p_edge_cnt < 20):
        #    - Không có hoa văn để đổi màu sắc tố (như tập dữ liệu vết sẹo test_nobg - scar).
        #    - Tổn thương biểu hiện trực tiếp qua mảng biến đổi độ sáng (blob_area >= 60) hoặc vết nứt/sẹo đâm xuyên (intrusive >= 12px).
        if item["p_edge_cnt"] >= 20:
            is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or (item["intrusive_len"] >= 12.0))
        else:
            is_l1_damage = bool(
                (item["intrusive_len"] >= 12.0) or 
                (item["dark_crack"] >= 12.0) or 
                (item["blob_area"] >= 60 and item["z_l1"] > 0.4) or
                (is_l2_damage and item["has_physical_l1"])
            )

        item["z_c"] = z_c
        item["is_l1"] = is_l1_damage
        item["is_l2"] = is_l2_damage
        item["is_fused"] = is_l1_damage or is_l2_damage
        triangle_details.append(item)

        if is_l1_damage:
            layer1_flagged.append(i)
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

    # 2. TÁCH 2 FOLDER RIÊNG BIỆT CHO LAYER 1:
    #    - 01_giong_nhau/ : Chứa từng tam giác GIỐNG NHAU
    #    - 02_khac_nhau/  : Chứa từng tam giác KHÁC NHAU (LỖI)
    dir_matching = os.path.join(dir_l1, "01_giong_nhau")
    dir_different = os.path.join(dir_l1, "02_khac_nhau")
    os.makedirs(dir_matching, exist_ok=True)
    os.makedirs(dir_different, exist_ok=True)
    for d in [dir_matching, dir_different]:
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
        info_txt = f"Tri #{item['id']} [{status_txt}] | Corr: {item['ori_corr']:.2f} | Crack: {item['dark_crack']:.1f}px | Scratch: {item['white_scratch']:.1f}px | Broken: {item['broken_length']:.1f}px | Blob: {item['blob_area']}px"
        cv2.putText(header, info_txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1)

        card = np.vstack([header, cols])
        return card

    # Xuất toàn bộ các tam giác KHÁC NHAU vào folder 02_khac_nhau
    diff_cards = []
    for item in different_items:
        card = build_triangle_card(item, is_diff=True)
        cv2.imwrite(os.path.join(dir_different, f"tri_{item['id']:04d}.jpg"), card)
        diff_cards.append(card)

    if diff_cards:
        # Nếu có tam giác khác nhau, lưu thêm ảnh ghép tổng hợp (mỗi trang tối đa 20 tam giác)
        for page_idx, chunk_start in enumerate(range(0, len(diff_cards), 20)):
            chunk = diff_cards[chunk_start:chunk_start + 20]
            grid_diff = np.vstack(chunk)
            cv2.imwrite(os.path.join(dir_different, f"00_tong_hop_page_{page_idx + 1:02d}.jpg"), grid_diff)

    # Xuất các tam giác GIỐNG NHAU vào folder 01_giong_nhau
    match_cards = []
    for item in matching_items:
        card = build_triangle_card(item, is_diff=False)
        cv2.imwrite(os.path.join(dir_matching, f"tri_{item['id']:04d}.jpg"), card)
        if len(match_cards) < 30:
            match_cards.append(card)

    if match_cards:
        grid_match = np.vstack(match_cards)
        cv2.imwrite(os.path.join(dir_matching, "00_tong_hop_mau_giong_nhau.jpg"), grid_match)

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
    # FOLDER 06: CLOSE-UP DEFECT CROPS (ZOOM-IN EVIDENCE)
    # -------------------------------------------------------------------------
    # Clean previous crops
    for f in os.listdir(dir_crops):
        try:
            os.remove(os.path.join(dir_crops, f))
        except OSError:
            pass

    print(f"\n[Evidence Export] Exporting zoom-in patches for {len(fused_flagged)} flagged triangles...")
    for idx in fused_flagged:
        t = triangle_details[idx]
        # Crop context around the triangle from original high-res images
        pts_p = t["pts_p"]
        pts_r = t["pts_r"]

        x0_p, y0_p = int(max(0, pts_p[:, 0].min() - 30)), int(max(0, pts_p[:, 1].min() - 30))
        x1_p, y1_p = int(min(w, pts_p[:, 0].max() + 30)), int(min(h, pts_p[:, 1].max() + 30))

        x0_r, y0_r = int(max(0, pts_r[:, 0].min() - 30)), int(max(0, pts_r[:, 1].min() - 30))
        x1_r, y1_r = int(min(w, pts_r[:, 0].max() + 30)), int(min(h, pts_r[:, 1].max() + 30))

        crop_p = product_img[y0_p:y1_p, x0_p:x1_p].copy()
        crop_r = return_img[y0_r:y1_r, x0_r:x1_r].copy()

        # Resize crops to standard view size for easy side-by-side inspection
        crop_h, crop_w = 250, 250
        
        # Draw triangle contour onto crops before resizing
        poly_p = np.array([[pt[0] - x0_p, pt[1] - y0_p] for pt in pts_p], dtype=np.int32).reshape((-1, 1, 2))
        poly_r = np.array([[pt[0] - x0_r, pt[1] - y0_r] for pt in pts_r], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(crop_p, [poly_p], True, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.polylines(crop_r, [poly_r], True, (0, 0, 255), 2, cv2.LINE_AA)

        crop_p_res = cv2.resize(crop_p, (crop_w, crop_h)) if crop_p.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crop_r_res = cv2.resize(crop_r, (crop_w, crop_h)) if crop_r.size > 0 else np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        # Label reason
        reason = []
        if t["is_l1"]:
            reason.append("Nut_GayNet_L1")
        if t["is_l2"]:
            reason.append("TrocMen_MatMau_L2")
        reason_str = "+".join(reason)

        evidence_panel = np.hstack([crop_p_res, crop_r_res])
        cv2.putText(evidence_panel, f"Left: Product | Right: Return", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(evidence_panel, f"Tri #{t['id']}: {reason_str}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(evidence_panel, f"SSIM={t['ssim']:.2f}, Crack={t['dark_crack']:.1f}px, Scratch={t['white_scratch']:.1f}px, dE={t['chroma_err']:.1f}", (10, crop_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        fname = f"defect_tri_{t['id']:03d}_{reason_str}.jpg"
        out_path = os.path.join(dir_crops, fname)
        success = cv2.imwrite(out_path, evidence_panel)

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
