import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

p_img = cv2.imread(r'images\test_nobg.png')
r_img = cv2.imread(r'images\test_nobg - scar.png')
seg = ObjectSegmenter()
p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k_open_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

scar_ids = [98, 121, 123, 124, 341, 432, 433, 454]

for tid in scar_ids:
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

    hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
    hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
    ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

    raw_diff = r_raw.astype(np.float32) - p_raw.astype(np.float32)
    raw_diff[~valid] = 0.0
    bin_raw_diff = (np.abs(raw_diff) > 35.0).astype(np.uint8) * 255
    opened_raw = cv2.morphologyEx(bin_raw_diff, cv2.MORPH_OPEN, k_open_5)
    raw_pos = np.sum((opened_raw > 0) & (raw_diff > 35.0))
    raw_neg = np.sum((opened_raw > 0) & (raw_diff < -35.0))
    raw_ratio = min(raw_pos, raw_neg) / (max(raw_pos, raw_neg) + 1e-5)
    is_b_shift = bool(raw_pos >= 60 and raw_neg >= 60 and raw_ratio >= 0.30 and ori_corr >= 0.80)

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
    signed_diff[~valid] = 0.0; signed_diff[glare_px] = 0.0
    diff_loc = np.abs(signed_diff)
    bin_diff = (diff_loc > 30.0).astype(np.uint8) * 255
    opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)

    pos_loc = np.sum((opened_diff > 0) & (signed_diff > 30.0))
    neg_loc = np.sum((opened_diff > 0) & (signed_diff < -30.0))
    loc_dipole = min(pos_loc, neg_loc) / (max(pos_loc, neg_loc) + 1e-5)
    is_loc_dipole = bool(pos_loc >= 40 and neg_loc >= 40 and loc_dipole >= 0.30 and ori_corr >= 0.85)

    print(f"Scar #{tid:3d}: is_b_shift={is_b_shift} (pos={raw_pos}, neg={raw_neg}, r={raw_ratio:.2f}) | is_loc_dipole={is_loc_dipole} (pos={pos_loc}, neg={neg_loc}, r={loc_dipole:.2f}) | ori_corr={ori_corr:.2f}")
