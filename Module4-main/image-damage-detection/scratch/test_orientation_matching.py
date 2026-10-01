import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def compute_orientation_profile(gray_img, mask, min_mag=25.0):
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

    # Find dominant peaks (bins with > 15% of total gradient energy)
    peaks = []
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for idx, w in enumerate(hist):
        if w >= 0.15:
            peaks.append(float(bin_centers[idx]))

    # Directional coherence: max bin weight / uniform weight (1/8 = 0.125)
    coherence = float(np.max(hist)) if len(hist) > 0 else 0.0

    return hist, peaks, coherence

def is_angle_matching_any(angle, reference_peaks, tol_deg=22.5):
    """Check if an angle matches any of the reference peak angles within tolerance."""
    if not reference_peaks:
        return False
    for p in reference_peaks:
        diff = abs(angle - p)
        diff = min(diff, 180.0 - diff)
        if diff <= tol_deg:
            return True
    return False

def test():
    p_img = cv2.imread(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg')
    r_img = cv2.imread(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\2.jpg')

    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img)
    r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for tid in [3, 303, 443, 445, 605, 631, 635, 735]:
        tri = triangles[tid]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_gray = clahe.apply(cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY))
        r_gray = clahe.apply(cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY))
        valid = mask_p > 0

        hist_p, peaks_p, coh_p = compute_orientation_profile(p_gray, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_gray, mask_p)

        # Correlation
        corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        # Now examine blackhat
        p_bhat = cv2.morphologyEx(p_gray, cv2.MORPH_BLACKHAT, k_morph)
        r_bhat = cv2.morphologyEx(r_gray, cv2.MORPH_BLACKHAT, k_morph)
        p_bhat_dil = cv2.dilate(p_bhat, k_tol)
        new_bhat = cv2.subtract(r_bhat, p_bhat_dil)
        new_bhat[~valid] = 0

        # For pixels where new_bhat > 30, check their local angle
        gx_r = cv2.Sobel(r_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_gray, cv2.CV_32F, 0, 1, ksize=3)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        crack_px = (new_bhat > 30) & valid
        anomalous_crack_px = np.zeros_like(crack_px)

        if np.any(crack_px):
            for y, x in zip(*np.where(crack_px)):
                a_px = ang_r[y, x]
                # If this crack pixel matches Product dominant orientations, it's parallel to existing pattern!
                if is_angle_matching_any(a_px, peaks_p, tol_deg=22.5) and coh_p > 0.35:
                    # Parallel to existing pattern with strong coherence -> Intact texture!
                    pass
                else:
                    anomalous_crack_px[y, x] = True

        cnts_all, _ = cv2.findContours((crack_px.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_len_all = max([cv2.arcLength(c, False) for c in cnts_all], default=0.0)

        cnts_anom, _ = cv2.findContours((anomalous_crack_px.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_len_anom = max([cv2.arcLength(c, False) for c in cnts_anom], default=0.0)

        print(f"Tri #{tid:3d}: peaks_p={peaks_p}, coh_p={coh_p:.2f} | ori_corr={corr:.3f} | all_crack={max_len_all:.1f}px -> anom_crack={max_len_anom:.1f}px")

if __name__ == '__main__':
    test()
