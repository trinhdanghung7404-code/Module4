import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile, is_angle_matching_any

def test_proposed_fix(p_path, r_path, label):
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
    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        # Raw grayscale for photometric consistency
        p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

        # CLAHE for fine ridge / stroke extraction
        p_c_gray = clahe.apply(p_raw_gray)
        r_c_gray = clahe.apply(r_raw_gray)

        valid = mask_p > 0
        glare_px = (r_c_gray > 240) & (p_c_gray > 240)

        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c_gray, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c_gray, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        p_bhat = cv2.morphologyEx(p_c_gray, cv2.MORPH_BLACKHAT, k_morph)
        r_bhat = cv2.morphologyEx(r_c_gray, cv2.MORPH_BLACKHAT, k_morph)
        p_what = cv2.morphologyEx(p_c_gray, cv2.MORPH_TOPHAT, k_morph)
        r_what = cv2.morphologyEx(r_c_gray, cv2.MORPH_TOPHAT, k_morph)

        p_ridge_bin = ((p_bhat > 15) | (p_what > 15)).astype(np.uint8) * 255
        dist_to_p_ridge = cv2.distanceTransform(cv2.bitwise_not(p_ridge_bin), cv2.DIST_L2, 3)

        gx_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 0, 1, ksize=3)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        crack_cand = (r_bhat > 30) & valid & (~glare_px)
        scratch_cand = (r_what > 30) & valid & (~glare_px)
        anom_crack = np.zeros_like(crack_cand)
        anom_scratch = np.zeros_like(scratch_cand)

        for y, x in zip(*np.where(crack_cand)):
            d = dist_to_p_ridge[y, x]
            near_exact = (d <= 2.0)
            near_shifted = (d <= 3.5) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
            if not (near_exact or near_shifted):
                anom_crack[y, x] = True

        for y, x in zip(*np.where(scratch_cand)):
            d = dist_to_p_ridge[y, x]
            near_exact = (d <= 2.0)
            near_shifted = (d <= 3.5) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
            if not (near_exact or near_shifted):
                anom_scratch[y, x] = True

        cnts_w, _ = cv2.findContours(anom_scratch.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_b, _ = cv2.findContours(anom_crack.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_white_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)
        max_dark_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)

        p_c_edge = cv2.Canny(p_c_gray, 40, 120); p_c_edge[~valid] = 0
        r_c_edge = cv2.Canny(r_c_gray, 40, 120); r_c_edge[~valid] = 0

        if np.any(r_c_edge > 0):
            dist_r = cv2.distanceTransform(cv2.bitwise_not(r_c_edge), cv2.DIST_L2, 3)
            mag_r = np.sqrt(gx_r**2 + gy_r**2)
            broken_edges = (p_c_edge > 0) & (dist_r > 5.5) & (mag_r < 25.0) & (~glare_px) & valid
            cnts_broken, _ = cv2.findContours(broken_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_broken_length = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)
        else:
            max_broken_length = 0.0

        # Intrusive edge on blank porcelain: MUST be on raw image to avoid CLAHE 9x9 tile boundaries,
        # AND must be far from any Product stroke/ridge (dist_to_p_ridge > 3.5)
        grad_p_raw = np.sqrt(cv2.Sobel(p_raw_gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw_gray, cv2.CV_32F, 0, 1)**2)
        r_raw_edge = cv2.Canny(r_raw_gray, 40, 120); r_raw_edge[~valid] = 0
        intrusive_edges = (r_raw_edge > 0) & (dist_to_p_ridge > 3.5) & (grad_p_raw < 25.0) & (~glare_px) & valid
        cnts_intrusive, _ = cv2.findContours(intrusive_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = max([cv2.arcLength(c, False) for c in cnts_intrusive], default=0.0)

        # Solid blob: on raw normalized grayscale (NO CLAHE tiling)
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
        diff_loc[~valid] = 0.0; diff_loc[glare_px] = 0.0
        bin_diff_loc = (diff_loc > 35.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)
        max_solid_blob = max([stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_lbl)], default=0)

        struct_metric = (
            1.5 * max_dark_crack +
            1.5 * max_white_scratch +
            2.0 * max_intrusive_length +
            0.5 * max_solid_blob +
            1.0 * max_broken_length
        )
        patch_data.append({
            'id': i,
            'dark_crack': max_dark_crack,
            'white_scratch': max_white_scratch,
            'intrusive_len': max_intrusive_length,
            'broken_length': max_broken_length,
            'blob_area': max_solid_blob,
            'struct_metric': struct_metric,
            'v_indices': set(tri.vertex_indices),
            'ori_corr': ori_corr,
        })

    l1_metrics = [p['struct_metric'] for p in patch_data]
    l1_mean, l1_std = float(np.mean(l1_metrics)), float(np.std(l1_metrics))

    core_edges_l1 = set()
    for item in patch_data:
        z_l1 = (item['struct_metric'] - l1_mean) / (l1_std + 1e-8)
        item['z_l1'] = z_l1
        has_physical_l1 = bool(item['dark_crack'] >= 10.0 or item['white_scratch'] >= 10.0 or 
                               item['intrusive_len'] >= 12.0 or item['blob_area'] >= 35)
        item['has_physical_l1'] = has_physical_l1
        if z_l1 > 1.8 and has_physical_l1:
            v = list(item['v_indices'])
            core_edges_l1.add(frozenset([v[0], v[1]]))
            core_edges_l1.add(frozenset([v[1], v[2]]))
            core_edges_l1.add(frozenset([v[2], v[0]]))

    flagged = []
    for item in patch_data:
        v = list(item['v_indices'])
        shares_core_edge = (frozenset([v[0], v[1]]) in core_edges_l1 or 
                            frozenset([v[1], v[2]]) in core_edges_l1 or 
                            frozenset([v[2], v[0]]) in core_edges_l1)
        is_l1 = bool((item['z_l1'] > 1.4 and item['has_physical_l1']) or 
                     (shares_core_edge and item['has_physical_l1'] and item['z_l1'] > 0.4) or
                     (item['blob_area'] >= 60 and item['z_l1'] > 0.5))
        item['is_l1'] = is_l1
        if is_l1:
            flagged.append(item['id'])

    print(f'{label}: Total={len(triangles)}, Flagged L1={len(flagged)}')
    return patch_data, flagged

if __name__ == '__main__':
    data1, flagged1 = test_proposed_fix(r'images\1.jpg', r'images\2.jpg', '1.jpg vs 2.jpg')
    for tid in [3, 12, 83]:
        t = data1[tid]
        print(f"Tri #{tid:2d}: Flagged={t['is_l1']} | z={t['z_l1']:.2f} | crack={t['dark_crack']:.1f} | scr={t['white_scratch']:.1f} | int={t['intrusive_len']:.1f} | blob={t['blob_area']}")

    data2, flagged2 = test_proposed_fix(r'images\test_nobg.png', r'images\test_nobg - scar.png', 'test_nobg vs scar')
    scar_ids = [98, 121, 123, 124, 341, 432, 433, 454]
    print('Scar triangles flagged:')
    for tid in scar_ids:
        t = data2[tid]
        print(f"  Tri #{tid:3d}: Flagged={t['is_l1']} (z={t['z_l1']:.2f}, blob={t['blob_area']}, int={t['intrusive_len']:.1f})")
