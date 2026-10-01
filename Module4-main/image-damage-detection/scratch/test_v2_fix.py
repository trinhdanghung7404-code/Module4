import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile, is_angle_matching_any

def eval_tri_v3(p_img, r_norm, triangles, vertices, tid):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    tri = triangles[tid]
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

    # Orientation profiles
    hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
    hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
    ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

    # Local norm & Dipole check on blob
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
    pos_px = np.sum((opened_diff > 0) & (signed_diff > 35.0))
    neg_px = np.sum((opened_diff > 0) & (signed_diff < -35.0))
    
    # Boundary shift detection
    is_boundary_shift = bool(pos_px >= 100 and neg_px >= 100 and 
                             (min(pos_px, neg_px) / (max(pos_px, neg_px) + 1e-5)) >= 0.30 and 
                             ori_corr >= 0.80)
    
    if is_boundary_shift:
        max_blob = 0
    else:
        max_blob = max([stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_lbl)], default=0)

    p_bhat = cv2.morphologyEx(p_c, cv2.MORPH_BLACKHAT, k_morph)
    r_bhat = cv2.morphologyEx(r_c, cv2.MORPH_BLACKHAT, k_morph)
    p_what = cv2.morphologyEx(p_c, cv2.MORPH_TOPHAT, k_morph)
    r_what = cv2.morphologyEx(r_c, cv2.MORPH_TOPHAT, k_morph)

    p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[~valid] = 0
    p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
    dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

    gx_r = cv2.Sobel(r_c, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(r_c, cv2.CV_32F, 0, 1, ksize=3)
    ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

    shift_tol = 8.0 if is_boundary_shift else 6.0

    # Crack
    crack_cand = (r_bhat > 30) & valid & (~glare_px)
    scratch_cand = (r_what > 30) & valid & (~glare_px)
    anom_crack = np.zeros_like(crack_cand)
    anom_scratch = np.zeros_like(scratch_cand)

    for y, x in zip(*np.where(crack_cand)):
        d = dist_to_p_all[y, x]
        near = (d <= 2.5) or ((d <= shift_tol) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0))
        if not near:
            anom_crack[y, x] = True

    for y, x in zip(*np.where(scratch_cand)):
        d = dist_to_p_all[y, x]
        near = (d <= 2.5) or ((d <= shift_tol) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0))
        if not near:
            anom_scratch[y, x] = True

    cnts_b, _ = cv2.findContours(anom_crack.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_w, _ = cv2.findContours(anom_scratch.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)
    max_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)

    # Intrusive
    grad_p_raw = np.sqrt(cv2.Sobel(p_raw, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw, cv2.CV_32F, 0, 1)**2)
    r_raw_edge = cv2.Canny(r_raw, 40, 120); r_raw_edge[~valid] = 0
    intrusive = (r_raw_edge > 0) & (dist_to_p_all > shift_tol) & (grad_p_raw < 25.0) & (~glare_px) & valid
    cnts_int, _ = cv2.findContours(intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_int = max([cv2.arcLength(c, False) for c in cnts_int], default=0.0)

    # Broken line
    r_c_edge = cv2.Canny(r_c, 40, 120); r_c_edge[~valid] = 0
    if np.any(r_c_edge > 0):
        dist_r = cv2.distanceTransform(cv2.bitwise_not(r_c_edge), cv2.DIST_L2, 3)
        mag_r = np.sqrt(gx_r**2 + gy_r**2)
        broken = (p_c_edge > 0) & (dist_r > shift_tol) & (mag_r < 25.0) & (~glare_px) & valid
        cnts_broken, _ = cv2.findContours(broken.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_broken = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)
    else:
        max_broken = 0.0

    struct_metric = 1.5 * max_crack + 1.5 * max_scratch + 2.0 * max_int + 0.5 * max_blob + 1.0 * max_broken
    return {
        'crack': max_crack,
        'scratch': max_scratch,
        'intrusive': max_int,
        'broken': max_broken,
        'blob': max_blob,
        'is_boundary_shift': is_boundary_shift,
        'struct_metric': struct_metric,
        'v_indices': tri.vertex_indices
    }

if __name__ == '__main__':
    # Run 1.jpg vs 2.jpg
    p_img = cv2.imread(r'images\1.jpg'); r_img = cv2.imread(r'images\2.jpg')
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    all_1 = [eval_tri_v3(p_img, r_norm, triangles, vertices, i) for i in range(len(triangles))]
    m1 = np.mean([x['struct_metric'] for x in all_1])
    s1 = np.std([x['struct_metric'] for x in all_1])

    core_edges_1 = set()
    for item in all_1:
        z = (item['struct_metric'] - m1) / (s1 + 1e-8)
        item['z'] = z
        has_phys = bool(item['crack'] >= 10.0 or item['scratch'] >= 10.0 or item['intrusive'] >= 12.0 or item['blob'] >= 35)
        item['has_phys'] = has_phys
        if z > 1.8 and has_phys:
            v = list(item['v_indices'])
            core_edges_1.add(frozenset([v[0], v[1]]))
            core_edges_1.add(frozenset([v[1], v[2]]))
            core_edges_1.add(frozenset([v[2], v[0]]))

    flagged_1 = []
    for idx, item in enumerate(all_1):
        v = list(item['v_indices'])
        shares = (frozenset([v[0], v[1]]) in core_edges_1 or frozenset([v[1], v[2]]) in core_edges_1 or frozenset([v[2], v[0]]) in core_edges_1)
        is_l1 = bool((item['z'] > 1.4 and item['has_phys']) or (shares and item['has_phys'] and item['z'] > 0.4) or (item['blob'] >= 60 and item['z'] > 0.5))
        item['is_l1'] = is_l1
        if is_l1:
            flagged_1.append(idx)

    print('=== 1.jpg vs 2.jpg ===')
    print(f'Total flagged L1: {len(flagged_1)}')
    for tid in [734, 735, 736, 737, 3, 12, 83]:
        t = all_1[tid]
        print(f"Tri #{tid:3d}: Flagged={t['is_l1']} | z={t['z']:5.2f} | crack={t['crack']:4.1f} | scr={t['scratch']:4.1f} | int={t['intrusive']:4.1f} | broken={t['broken']:4.1f} | blob={t['blob']:3d} | b_shift={t['is_boundary_shift']}")

    # Run scar
    p_img2 = cv2.imread(r'images\test_nobg.png'); r_img2 = cv2.imread(r'images\test_nobg - scar.png')
    p_seg2 = seg.segment(p_img2); r_seg2 = seg.segment(r_img2)
    r_norm2 = norm.normalize(r_img2, p_img2, r_seg2['mask'], p_seg2['mask'])
    reg_res2 = reg.register(p_img2, r_norm2, p_seg2['mask'], r_seg2['mask'])
    mb2 = MeshBuilder()
    vertices2, triangles2 = mb2.build(reg_res2['product_points'], reg_res2['return_points'], np.arange(len(reg_res2['product_points'])))

    all_2 = [eval_tri_v3(p_img2, r_norm2, triangles2, vertices2, i) for i in range(len(triangles2))]
    m2 = np.mean([x['struct_metric'] for x in all_2])
    s2 = np.std([x['struct_metric'] for x in all_2])

    core_edges_2 = set()
    for item in all_2:
        z = (item['struct_metric'] - m2) / (s2 + 1e-8)
        item['z'] = z
        has_phys = bool(item['crack'] >= 10.0 or item['scratch'] >= 10.0 or item['intrusive'] >= 12.0 or item['blob'] >= 35)
        item['has_phys'] = has_phys
        if z > 1.8 and has_phys:
            v = list(item['v_indices'])
            core_edges_2.add(frozenset([v[0], v[1]]))
            core_edges_2.add(frozenset([v[1], v[2]]))
            core_edges_2.add(frozenset([v[2], v[0]]))

    flagged_2 = []
    for idx, item in enumerate(all_2):
        v = list(item['v_indices'])
        shares = (frozenset([v[0], v[1]]) in core_edges_2 or frozenset([v[1], v[2]]) in core_edges_2 or frozenset([v[2], v[0]]) in core_edges_2)
        is_l1 = bool((item['z'] > 1.4 and item['has_phys']) or (shares and item['has_phys'] and item['z'] > 0.4) or (item['blob'] >= 60 and item['z'] > 0.5))
        item['is_l1'] = is_l1
        if is_l1:
            flagged_2.append(idx)

    print('\n=== test_nobg vs scar ===')
    print(f'Total flagged L1: {len(flagged_2)}')
    scar_ids = [98, 121, 123, 124, 341, 432, 433, 454]
    for tid in scar_ids:
        t = all_2[tid]
        print(f"Scar Tri #{tid:3d}: Flagged={t['is_l1']} | z={t['z']:5.2f} | crack={t['crack']:4.1f} | scr={t['scratch']:4.1f} | int={t['intrusive']:4.1f} | broken={t['broken']:4.1f} | blob={t['blob']:3d} | b_shift={t['is_boundary_shift']}")
