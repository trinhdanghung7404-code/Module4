"""Test script in scratch: Evaluate raw_diff alongside diff_loc to prevent defect masking.
Testing on 1.jpg vs 16.jpg without modifying any main files.
"""

import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from detailed_debug import micro_align_patches, compute_orientation_profile, is_angle_matching_any, canonical_triangle_patch

def test_pipeline(p1_path, p2_path):
    print(f"\nComparing {os.path.basename(p1_path)} vs {os.path.basename(p2_path)}...")
    prod_bgr = cv2.imread(p1_path)
    ret_bgr = cv2.imread(p2_path)
    h, w = prod_bgr.shape[:2]

    seg = ObjectSegmenter()
    p_seg = seg.segment(prod_bgr)
    r_seg = seg.segment(ret_bgr)

    norm = ImageNormalizer()
    ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])
    ret_ct = norm.color_transfer(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])

    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])
    inliers_p = reg_res["product_points"]
    inliers_r = reg_res["return_points"]

    mb = MeshBuilder()
    vertices, triangles = mb.build(inliers_p, inliers_r, np.arange(len(inliers_p)))

    p_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(ret_ct, cv2.COLOR_BGR2GRAY)
    p_med = cv2.medianBlur(p_gray, 3)
    r_med = cv2.medianBlur(r_gray, 3)
    p_edge_denoised = cv2.Canny(p_med, 60, 160)
    r_edge_denoised = cv2.Canny(r_med, 60, 160)

    p_lab = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(ret_norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    p_valid = p_seg["mask"] > 0
    r_valid = r_seg["mask"] > 0
    offset_a = float(np.median(r_lab[..., 1][r_valid]) - np.median(p_lab[..., 1][p_valid]))
    offset_b = float(np.median(r_lab[..., 2][r_valid]) - np.median(p_lab[..., 2][p_valid]))
    r_lab[..., 1] -= offset_a
    r_lab[..., 2] -= offset_b

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    b_shift_edges = set()
    raw_patch_list = []

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
        patch_pe, _ = canonical_triangle_patch(p_edge_denoised, pts_p, target_size=72)
        patch_re, _ = canonical_triangle_patch(r_edge_denoised, pts_r, target_size=72)
        patch_pe = (patch_pe > 127).astype(np.uint8) * 255
        patch_re = (patch_re > 127).astype(np.uint8) * 255

        patch_r_aligned, patch_re_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2, extra_patch=patch_re)

        p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c_gray = clahe.apply(p_raw_gray)
        r_c_gray = clahe.apply(r_raw_gray)

        valid = mask_p > 0
        glare_px = (r_c_gray > 240) | (p_c_gray > 240)

        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c_gray, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c_gray, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

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
            "id": i, "tri": tri, "pts_p": pts_p, "pts_r": pts_r,
            "patch_p": patch_p, "patch_r": patch_r_aligned,
            "patch_pe": patch_pe, "patch_re": patch_re_aligned,
            "mask_p": mask_p, "p_raw_gray": p_raw_gray, "r_raw_gray": r_raw_gray,
            "p_c_gray": p_c_gray, "r_c_gray": r_c_gray,
            "valid": valid, "glare_px": glare_px, "peaks_p": peaks_p,
            "ori_corr": ori_corr, "is_b_shift": is_b_shift,
        })

    h_m, w_m = 72, 72
    Y_m, X_m = np.ogrid[:h_m, :w_m]
    inner_triangle = (X_m >= 4) & (Y_m >= 4) & (X_m + Y_m <= 66)

    patch_data = []
    color_diffs = []

    def measure_linear_stroke(contours, min_len=8.0, min_aspect=1.8):
        max_len = 0.0
        for c in contours:
            if len(c) < 4:
                continue
            rect = cv2.minAreaRect(c)
            length, width = max(rect[1]), min(rect[1])
            aspect = length / max(width, 0.5)
            if length >= min_len and (aspect >= min_aspect or length >= 14.0):
                if length > max_len:
                    max_len = length
        return max_len

    for item in raw_patch_list:
        i = item["id"]
        tri = item["tri"]
        pts_p, pts_r = item["pts_p"], item["pts_r"]
        patch_p, patch_r_aligned = item["patch_p"], item["patch_r"]
        mask_p = item["mask_p"]
        p_raw_gray, r_raw_gray = item["p_raw_gray"], item["r_raw_gray"]
        p_c_gray, r_c_gray = item["p_c_gray"], item["r_c_gray"]
        valid, glare_px = item["valid"], item["glare_px"]
        peaks_p, ori_corr = item["peaks_p"], item["ori_corr"]
        is_b_shift = item["is_b_shift"]

        v = list(tri.vertex_indices)
        shares_b_shift = (frozenset([v[0], v[1]]) in b_shift_edges or 
                          frozenset([v[1], v[2]]) in b_shift_edges or 
                          frozenset([v[2], v[0]]) in b_shift_edges)

        p_c_lab = cv2.cvtColor(patch_p, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_c_lab = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_c_lab[..., 1] -= offset_a
        r_c_lab[..., 2] -= offset_b

        p_what = cv2.morphologyEx(p_c_gray, cv2.MORPH_TOPHAT, k_morph)
        p_bhat = cv2.morphologyEx(p_c_gray, cv2.MORPH_BLACKHAT, k_morph)
        r_what = cv2.morphologyEx(r_c_gray, cv2.MORPH_TOPHAT, k_morph)
        r_bhat = cv2.morphologyEx(r_c_gray, cv2.MORPH_BLACKHAT, k_morph)

        p_c_edge = item["patch_pe"].copy()
        r_c_edge = item["patch_re"].copy()
        p_c_edge[~valid] = 0
        r_c_edge[~valid] = 0
        p_edge_cnt = int(np.sum(p_c_edge > 0))
        r_edge_cnt = int(np.sum(r_c_edge > 0))

        p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
        dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

        gx_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 0, 1, ksize=3)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        if p_edge_cnt < 20:
            max_broken_length = 0.0
            max_white_scratch = 0.0
            max_dark_crack = 0.0
        else:
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

            cnts_w, _ = cv2.findContours(anom_scratch.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_b, _ = cv2.findContours(anom_crack.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_white_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)
            max_dark_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)

        # Broken pattern
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

        # Intrusive edge
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

        # 3. Solid Grayscale Anomaly Blob
        p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
        r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)
        if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
            mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
            mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
            scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_raw_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc
        else:
            r_norm_loc = r_raw_gray.astype(np.float32)

        # LOCAL DIFF vs RAW DIFF
        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0

        raw_diff_abs = np.abs(r_raw_gray.astype(np.float32) - p_raw_gray.astype(np.float32))
        raw_diff_abs[~valid] = 0.0
        raw_diff_abs[glare_px] = 0.0

        # Effective difference: take maximum between normalized diff and raw diff
        # This completely solves the "mean-shift defect masking" problem!
        diff_effective = np.maximum(diff_loc, raw_diff_abs)

        bin_diff_loc = (diff_effective > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

        is_shape_intact = bool((ori_corr >= 0.80 or p_edge_cnt < 20) and max_dark_crack < 10.0 and max_white_scratch < 10.0 and max_intrusive_length < 10.0)
        contrast_th = 90.0 if is_shape_intact else 60.0

        if is_b_shift:
            max_solid_blob = 0
        elif shares_b_shift:
            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_effective[m]) >= contrast_th:
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
                if np.max(diff_effective[m]) >= contrast_th:
                    valid_blobs.append(area)
            max_solid_blob = max(valid_blobs, default=0)

        # Composite Structural Defect Metric
        struct_metric = (
            1.5 * max_dark_crack +
            1.5 * max_white_scratch +
            2.0 * max_intrusive_length +
            0.1 * max_solid_blob +
            1.5 * max_broken_length
        )

        ssim_val = float(ssim(p_c_gray, r_c_gray, data_range=255))
        c_da = p_c_lab[..., 1] - r_c_lab[..., 1]
        c_db = p_c_lab[..., 2] - r_c_lab[..., 2]
        patch_diff = np.sqrt(c_da ** 2 + c_db ** 2)
        non_glare = (mask_p > 0) & (r_c_lab[..., 0] <= 225) & (p_c_lab[..., 0] <= 225)
        mean_chroma_err = float(np.mean(patch_diff[non_glare])) if np.any(non_glare) else 0.0
        color_diffs.append(mean_chroma_err)

        strong_color_diff = (patch_diff > 18.0) & non_glare
        num_labels, labels, stats_c, _ = cv2.connectedComponentsWithStats(strong_color_diff.astype(np.uint8) * 255, connectivity=8)
        max_blob_area = max([stats_c[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_labels)], default=0)
        has_color_blob = bool(max_blob_area >= 25)

        patch_data.append({
            "id": i, "pts_p": pts_p, "pts_r": pts_r, "v_indices": set(tri.vertex_indices),
            "p_edge_cnt": p_edge_cnt, "r_edge_cnt": r_edge_cnt,
            "broken_length": max_broken_length, "dark_crack": max_dark_crack,
            "white_scratch": max_white_scratch, "intrusive_len": max_intrusive_length,
            "blob_area": max_solid_blob, "struct_metric": struct_metric,
            "has_color_blob": has_color_blob, "ssim": ssim_val,
            "chroma_err": mean_chroma_err, "ori_corr": ori_corr
        })

    # Statistical decision
    c_mean, c_std = float(np.mean(color_diffs)), float(np.std(color_diffs))
    l1_metrics = [p["struct_metric"] for p in patch_data]
    l1_mean, l1_std = float(np.mean(l1_metrics)), float(np.std(l1_metrics))

    l1_flagged = []
    l2_flagged = []
    fused_flagged = []

    for item in patch_data:
        i = item["id"]
        z_l1 = (item["struct_metric"] - l1_mean) / (l1_std + 1e-8)
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)

        is_l2_damage = bool(z_c > 2.8 and item["chroma_err"] >= 24.0 and item["has_color_blob"])

        if item["p_edge_cnt"] < 20:
            has_physical_l1 = bool(item["intrusive_len"] >= 12.0 or item["blob_area"] >= 50)
            is_l1_damage = bool(
                (item["intrusive_len"] >= 12.0) or 
                (item["dark_crack"] >= 12.0) or 
                (item["blob_area"] >= 50 and z_l1 > 0.8) or
                (is_l2_damage and has_physical_l1)
            )
        else:
            has_physical_l1 = bool(item["dark_crack"] >= 10.0 or item["white_scratch"] >= 10.0 or 
                                   item["intrusive_len"] >= 12.0 or item["broken_length"] >= 10.0 or
                                   item["blob_area"] >= 35)
            # Khối dị vật diện tích lớn (blob_area >= 60 và z_l1 > 1.0)
            is_l1_damage = bool(
                (is_l2_damage and has_physical_l1) or 
                (item["intrusive_len"] >= 12.0) or
                (item["blob_area"] >= 60 and z_l1 > 1.0)
            )

        is_fused = is_l1_damage or is_l2_damage
        if is_l1_damage:
            l1_flagged.append(i)
        if is_l2_damage:
            l2_flagged.append(i)
        if is_fused:
            fused_flagged.append(i)

    print(f"\n[Result Summary]:")
    print(f"  Total Triangles: {len(triangles)}")
    print(f"  Layer 1 Flagged: {len(l1_flagged)}")
    print(f"  Layer 2 Flagged: {len(l2_flagged)}")
    print(f"  Fused Defect Triangles: {len(fused_flagged)}")
    print(f"  Flagged IDs: {sorted(fused_flagged)}")

    t685 = [t for t in patch_data if t['id'] == 685]
    if t685:
        t = t685[0]
        z_l1 = (t["struct_metric"] - l1_mean) / (l1_std + 1e-8)
        print(f"\n>>> Tri #685 Details:")
        print(f"    blob_area:     {t['blob_area']}px")
        print(f"    struct_metric: {t['struct_metric']:.2f}")
        print(f"    z_l1:          {z_l1:.2f}")
        print(f"    is_l1:         {685 in l1_flagged}")
        print(f"    is_fused:      {685 in fused_flagged} (IS CAUGHT: {'YES' if 685 in fused_flagged else 'NO'})")

    return {
        "triangles": len(triangles),
        "l1_flagged": len(l1_flagged),
        "l2_flagged": len(l2_flagged),
        "fused_flagged": len(fused_flagged),
        "flagged_ids": sorted(fused_flagged),
        "tri_685_caught": (685 in fused_flagged)
    }

if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p16 = os.path.join(PROJECT_ROOT, "images", "16.jpg")
    test_pipeline(p1, p16)
