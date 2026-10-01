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

def is_angle_matching_any(angle, reference_peaks, tol_deg=22.5):
    if not reference_peaks:
        return False
    for p in reference_peaks:
        diff = abs(angle - p)
        diff = min(diff, 180.0 - diff)
        if diff <= tol_deg:
            return True
    return False

def run_test(p_path, r_path, label=""):
    print(f"\n=======================================================")
    print(f"  TESTING: {label} ({os.path.basename(p_path)} vs {os.path.basename(r_path)})")
    print(f"=======================================================")
    p_img = cv2.imread(p_path)
    r_img = cv2.imread(r_path)

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
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    patch_data = []

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_gray = clahe.apply(cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY))
        r_gray = clahe.apply(cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY))
        valid = mask_p > 0
        glare_px = (r_gray > 240) & (p_gray > 240)

        # Orientation profiles
        hist_p, peaks_p, coh_p = compute_orientation_profile(p_gray, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_gray, mask_p)
        corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        # Black-hat & Top-hat
        p_bhat = cv2.morphologyEx(p_gray, cv2.MORPH_BLACKHAT, k_morph)
        r_bhat = cv2.morphologyEx(r_gray, cv2.MORPH_BLACKHAT, k_morph)
        p_what = cv2.morphologyEx(p_gray, cv2.MORPH_TOPHAT, k_morph)
        r_what = cv2.morphologyEx(r_gray, cv2.MORPH_TOPHAT, k_morph)

        p_bhat_dil = cv2.dilate(p_bhat, k_tol)
        p_what_dil = cv2.dilate(p_what, k_tol)

        new_bhat = cv2.subtract(r_bhat, p_bhat_dil)
        new_bhat[~valid] = 0; new_bhat[glare_px] = 0
        new_what = cv2.subtract(r_what, p_what_dil)
        new_what[~valid] = 0; new_what[glare_px] = 0

        # Gradient of Return
        gx_r = cv2.Sobel(r_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_gray, cv2.CV_32F, 0, 1, ksize=3)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        # Filter cracks/scratches by orientation
        # If triangle has strong directional texture and high correlation, parallel features are intact!
        is_parallel_texture = (corr > 0.85) and (coh_p >= 0.35)

        crack_px = (new_bhat > 30) & valid
        scratch_px = (new_what > 30) & valid

        anom_crack_px = np.zeros_like(crack_px)
        anom_scratch_px = np.zeros_like(scratch_px)

        if is_parallel_texture:
            for y, x in zip(*np.where(crack_px)):
                if not is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0):
                    anom_crack_px[y, x] = True
            for y, x in zip(*np.where(scratch_px)):
                if not is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0):
                    anom_scratch_px[y, x] = True
        else:
            anom_crack_px = crack_px.copy()
            anom_scratch_px = scratch_px.copy()

        cnts_b, _ = cv2.findContours(anom_crack_px.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_dark_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)

        cnts_w, _ = cv2.findContours(anom_scratch_px.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_white_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)

        # Canny & Intrusive / Broken
        p_c_edge = cv2.Canny(p_gray, 40, 120); p_c_edge[~valid] = 0
        r_c_edge = cv2.Canny(r_gray, 40, 120); r_c_edge[~valid] = 0

        # Broken pattern line
        if np.any(r_c_edge > 0):
            dist_r = cv2.distanceTransform(cv2.bitwise_not(r_c_edge), cv2.DIST_L2, 3)
            # If parallel texture with high corr, tolerate small shifts (dist > 6.5)
            tol_dist = 6.5 if is_parallel_texture else 5.0
            broken_edges = (p_c_edge > 0) & (dist_r > tol_dist) & (~glare_px) & valid
            cnts_broken, _ = cv2.findContours(broken_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_broken_length = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)
        else:
            max_broken_length = 0.0

        # Intrusive edge on blank porcelain: return edge where product has smooth glaze
        grad_p = np.sqrt(cv2.Sobel(p_gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_gray, cv2.CV_32F, 0, 1)**2)
        intrusive_edges = (r_c_edge > 0) & (grad_p < 25.0) & (~glare_px) & valid
        cnts_intrusive, _ = cv2.findContours(intrusive_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = max([cv2.arcLength(c, False) for c in cnts_intrusive], default=0.0)

        # Solid Blob
        p_vals = p_gray[valid & (~glare_px)].astype(np.float32)
        r_vals = r_gray[valid & (~glare_px)].astype(np.float32)
        if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
            mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
            mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
            scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc
        else:
            r_norm_loc = r_gray.astype(np.float32)

        diff_loc = np.abs(r_norm_loc - p_gray.astype(np.float32))
        diff_loc[~valid] = 0.0; diff_loc[glare_px] = 0.0
        # If parallel texture with high correlation, ignore contrast-shift dipole stripes
        if is_parallel_texture:
            max_solid_blob = 0
        else:
            bin_diff_loc = (diff_loc > 35.0).astype(np.uint8) * 255
            eroded_diff_loc = cv2.erode(bin_diff_loc, k_erode)
            num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(eroded_diff_loc, connectivity=8)
            max_solid_blob = max([stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_lbl)], default=0)

        # Composite Structural Defect Metric
        struct_metric = (
            1.5 * max_dark_crack +
            1.5 * max_white_scratch +
            2.0 * max_intrusive_length +
            0.5 * max_solid_blob +
            1.0 * max_broken_length
        )

        patch_data.append({
            "id": i,
            "dark_crack": max_dark_crack,
            "white_scratch": max_white_scratch,
            "intrusive_len": max_intrusive_length,
            "broken_length": max_broken_length,
            "blob_area": max_solid_blob,
            "struct_metric": struct_metric,
            "v_indices": set(tri.vertex_indices),
            "is_parallel": is_parallel_texture,
            "corr": corr,
        })

    l1_metrics = [p["struct_metric"] for p in patch_data]
    l1_mean, l1_std = float(np.mean(l1_metrics)), float(np.std(l1_metrics))

    core_edges_l1 = set()
    for item in patch_data:
        z_l1 = (item["struct_metric"] - l1_mean) / (l1_std + 1e-8)
        item["z_l1"] = z_l1
        has_physical = bool(item["dark_crack"] >= 10.0 or item["white_scratch"] >= 10.0 or 
                            item["intrusive_len"] >= 12.0 or item["blob_area"] >= 35)
        item["has_physical"] = has_physical
        if z_l1 > 1.8 and has_physical:
            v = list(item["v_indices"])
            core_edges_l1.add(frozenset([v[0], v[1]]))
            core_edges_l1.add(frozenset([v[1], v[2]]))
            core_edges_l1.add(frozenset([v[2], v[0]]))

    flagged_l1 = []
    for item in patch_data:
        v = list(item["v_indices"])
        shares_core = (frozenset([v[0], v[1]]) in core_edges_l1 or 
                       frozenset([v[1], v[2]]) in core_edges_l1 or 
                       frozenset([v[2], v[0]]) in core_edges_l1)
        is_l1 = bool((item["z_l1"] > 1.4 and item["has_physical"]) or 
                     (shares_core and item["has_physical"] and item["z_l1"] > 0.4))
        item["is_l1"] = is_l1
        if is_l1:
            flagged_l1.append(item["id"])

    print(f"Total Triangles: {len(triangles)}")
    print(f"Layer 1 Flagged (Defects): {len(flagged_l1)}")
    print(f"Layer 1 Intact:            {len(triangles) - len(flagged_l1)}")

    # Check Tri #3
    t3 = patch_data[3]
    print(f"\n--- Tri #3 Status ---")
    print(f"  Flagged? {t3['is_l1']} (Expected: False / INTACT)")
    print(f"  Dark crack: {t3['dark_crack']:.1f}px | Scratch: {t3['white_scratch']:.1f}px | Intrusive: {t3['intrusive_len']:.1f}px | Broken: {t3['broken_length']:.1f}px | Blob: {t3['blob_area']}px")
    print(f"  struct_metric: {t3['struct_metric']:.2f} | z_l1: {t3['z_l1']:.2f}")

    # Check known true defects
    print(f"\n--- True Defect Triangles Status ---")
    for tid in [303, 443, 445, 456, 605, 631, 635, 735]:
        if tid < len(patch_data):
            td = patch_data[tid]
            print(f"  Tri #{tid:3d}: Flagged? {td['is_l1']:<5} | z_l1={td['z_l1']:5.2f} | metric={td['struct_metric']:5.1f} | int={td['intrusive_len']:4.1f} | brk={td['broken_length']:4.1f} | blb={td['blob_area']:3d}")

if __name__ == '__main__':
    # 1. Undamaged vase pair (1.jpg vs 2.jpg)
    run_test(
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg',
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\2.jpg',
        label="1.jpg vs 2.jpg (UNDAMAGED VASE)"
    )
    # 2. Scarred vase pair (test_nobg.png vs test_nobg - scar.png)
    run_test(
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\test_nobg.png',
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\test_nobg - scar.png',
        label="test_nobg.png vs test_nobg - scar.png (REAL SCAR DEFECT)"
    )
