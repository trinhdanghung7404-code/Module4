import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile, is_angle_matching_any

def test_l1_rule(p_path, r_path, label):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    patch_data = []
    b_shift_edges = set()

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=3)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c = clahe.apply(p_raw)
        r_c = clahe.apply(r_raw)

        valid = mask_p > 0
        glare_px = (r_c > 240) & (p_c > 240)

        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        raw_diff = r_raw.astype(np.float32) - p_raw.astype(np.float32)
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

        patch_data.append({
            'id': i,
            'p_raw': p_raw, 'r_raw': r_raw,
            'p_c': p_c, 'r_c': r_c,
            'valid': valid, 'glare_px': glare_px,
            'peaks_p': peaks_p, 'ori_corr': ori_corr,
            'coh_p': coh_p,
            'is_b_shift': is_b_shift,
            'v_indices': tri.vertex_indices,
            'mask_p': mask_p,
        })

    h_m, w_m = 72, 72
    Y_m, X_m = np.ogrid[:h_m, :w_m]
    inner_triangle = (X_m >= 2) & (Y_m >= 2) & (X_m + Y_m <= 68)

    for item in patch_data:
        p_c = item['p_c']; r_c = item['r_c']
        p_raw = item['p_raw']; r_raw = item['r_raw']
        valid = item['valid']
        glare_px = item['glare_px']
        peaks_p = item['peaks_p']
        ori_corr = item['ori_corr']
        coh_p = item['coh_p']
        is_b_shift = item['is_b_shift']

        v = list(item['v_indices'])
        shares_b_shift = (frozenset([v[0], v[1]]) in b_shift_edges or 
                          frozenset([v[1], v[2]]) in b_shift_edges or 
                          frozenset([v[2], v[0]]) in b_shift_edges)

        p_bhat = cv2.morphologyEx(p_c, cv2.MORPH_BLACKHAT, k_morph)
        r_bhat = cv2.morphologyEx(r_c, cv2.MORPH_BLACKHAT, k_morph)
        p_what = cv2.morphologyEx(p_c, cv2.MORPH_TOPHAT, k_morph)
        r_what = cv2.morphologyEx(r_c, cv2.MORPH_TOPHAT, k_morph)

        p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[~valid] = 0
        r_c_edge = cv2.Canny(r_c, 40, 120); r_c_edge[~valid] = 0

        p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
        dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

        gx_r = cv2.Sobel(r_c, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_c, cv2.CV_32F, 0, 1, ksize=3)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        crack_cand = (r_bhat > 30) & valid & inner_triangle & (~glare_px)
        scratch_cand = (r_what > 30) & valid & inner_triangle & (~glare_px)
        anom_crack = np.zeros_like(crack_cand)
        anom_scratch = np.zeros_like(scratch_cand)

        for y, x in zip(*np.where(crack_cand)):
            d = dist_to_p_all[y, x]
            near = (d <= 2.5) or ((d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0))
            if not near:
                anom_crack[y, x] = True

        for y, x in zip(*np.where(scratch_cand)):
            d = dist_to_p_all[y, x]
            near = (d <= 2.5) or ((d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0))
            if not near:
                anom_scratch[y, x] = True

        cnts_b, _ = cv2.findContours(anom_crack.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_w, _ = cv2.findContours(anom_scratch.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)
        max_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)

        # Broken line
        if np.any(r_c_edge > 0):
            dist_r = cv2.distanceTransform(cv2.bitwise_not(r_c_edge), cv2.DIST_L2, 3)
            mag_r = np.sqrt(gx_r**2 + gy_r**2)
            broken = (p_c_edge > 0) & (dist_r > 5.5) & (mag_r < 25.0) & (~glare_px) & valid
            cnts_broken, _ = cv2.findContours(broken.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_broken = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)
        else:
            max_broken = 0.0

        # Intrusive edge on blank porcelain:
        grad_p_raw = np.sqrt(cv2.Sobel(p_raw, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw, cv2.CV_32F, 0, 1)**2)
        r_raw_edge = cv2.Canny(r_raw, 40, 120); r_raw_edge[~valid] = 0
        gx_r_raw = cv2.Sobel(r_raw, cv2.CV_32F, 1, 0)
        gy_r_raw = cv2.Sobel(r_raw, cv2.CV_32F, 0, 1)
        mag_r_raw = np.sqrt(gx_r_raw**2 + gy_r_raw**2)
        ang_r_raw = (np.arctan2(gy_r_raw, gx_r_raw) * 180.0 / np.pi) % 180.0

        intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 6.0) & (grad_p_raw < 25.0) & (mag_r_raw >= 70.0) & (~glare_px) & valid
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0):
                anom_intrusive[y, x] = True

        cnts_int, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_int = max([cv2.arcLength(c, False) for c in cnts_int], default=0.0)

        # Local norm on raw grayscale
        p_vals = p_raw[valid & (~glare_px)].astype(np.float32)
        r_vals = r_raw[valid & (~glare_px)].astype(np.float32)
        if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
            mu_p, std_p = np.mean(p_vals), np.std(p_vals)
            mu_r, std_r = np.mean(r_vals), np.std(r_vals)
            scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p
        else:
            r_norm_loc = r_raw.astype(np.float32)

        signed_diff = r_norm_loc - p_raw.astype(np.float32)
        diff_loc = np.abs(signed_diff)
        diff_loc[~valid] = 0.0; diff_loc[glare_px] = 0.0
        bin_diff = (diff_loc > 35.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open)

        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

        # Check if texture is intact parallel pattern:
        # If ori_corr >= 0.88 and coh_p >= 0.35 and no cracks/scratches/breaks:
        # the pattern is intact, blobs are parallax shift artifacts!
        is_intact_parallel = bool(ori_corr >= 0.88 and coh_p >= 0.35 and max_crack < 10.0 and max_scratch < 10.0 and max_broken < 12.0)

        if is_b_shift or is_intact_parallel:
            max_blob = 0
        elif shares_b_shift:
            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= 70.0:
                    ys, xs = np.where(m)
                    is_edge_leak = (xs.max() <= 15 or ys.max() <= 15 or (xs + ys).min() >= 55)
                    if not is_edge_leak:
                        cleaned_blob = max(cleaned_blob, area)
            max_blob = cleaned_blob
        else:
            valid_blobs = []
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= 70.0:
                    valid_blobs.append(area)
            max_blob = max(valid_blobs, default=0)

        struct_metric = 1.5 * max_crack + 1.5 * max_scratch + 2.0 * max_int + 0.5 * max_blob + 1.0 * max_broken
        item['crack'] = max_crack
        item['scratch'] = max_scratch
        item['intrusive'] = max_int
        item['broken'] = max_broken
        item['blob'] = max_blob
        item['struct_metric'] = struct_metric

    metrics = [p['struct_metric'] for p in patch_data]
    mean_m = float(np.mean(metrics))
    std_m = float(np.std(metrics))

    core_edges = set()
    for item in patch_data:
        z = (item['struct_metric'] - mean_m) / (std_m + 1e-8)
        item['z'] = z
        has_phys = bool(item['crack'] >= 10.0 or item['scratch'] >= 10.0 or item['intrusive'] >= 12.0 or item['blob'] >= 20)
        item['has_phys'] = has_phys
        if z > 1.8 and has_phys:
            v = list(item['v_indices'])
            core_edges.add(frozenset([v[0], v[1]]))
            core_edges.add(frozenset([v[1], v[2]]))
            core_edges.add(frozenset([v[2], v[0]]))

    flagged = []
    for idx, item in enumerate(patch_data):
        v = list(item['v_indices'])
        shares = (frozenset([v[0], v[1]]) in core_edges or frozenset([v[1], v[2]]) in core_edges or frozenset([v[2], v[0]]) in core_edges)
        is_l1 = bool((item['z'] > 1.4 and item['has_phys']) or (shares and item['has_phys'] and item['z'] > 0.4) or (item['blob'] >= 40 and item['z'] > 0.5))
        item['is_l1'] = is_l1
        if is_l1:
            flagged.append(idx)

    print(f'=== {label} ===')
    print(f'Total: {len(triangles)}, Flagged L1: {len(flagged)}')
    return patch_data, flagged

if __name__ == '__main__':
    d1, f1 = test_l1_rule(r'images\1.jpg', r'images\2.jpg', '1.jpg vs 2.jpg')
    for tid in [605, 606, 466, 734, 735, 736, 737, 3, 11, 12, 13, 83]:
        t = d1[tid]
        print(f"Tri #{tid:3d}: Flagged={t['is_l1']} | z={t['z']:5.2f} | crack={t['crack']:4.1f} | scr={t['scratch']:4.1f} | int={t['intrusive']:4.1f} | broken={t['broken']:4.1f} | blob={t['blob']:3d} | corr={t['ori_corr']:.2f}")

    d2, f2 = test_l1_rule(r'images\test_nobg.png', r'images\test_nobg - scar.png', 'test_nobg vs scar')
    scar_ids = [98, 121, 123, 124, 341, 432, 433, 454]
    print('Scar triangles:')
    for tid in scar_ids:
        t = d2[tid]
        print(f"Scar Tri #{tid:3d}: Flagged={t['is_l1']} | z={t['z']:5.2f} | crack={t['crack']:4.1f} | scr={t['scratch']:4.1f} | int={t['intrusive']:4.1f} | broken={t['broken']:4.1f} | blob={t['blob']:3d}")
