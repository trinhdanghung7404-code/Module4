import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from skimage.metrics import structural_similarity as ssim
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

def evaluate_rule(p_path, r_path, tids, name):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print(f"=== {name} ===")
    for tid in tids:
        tri = triangles[tid]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c = clahe.apply(p_raw)
        r_c = clahe.apply(r_raw)
        valid = mask_p > 0
        glare_px = (r_c > 240) & (p_c > 240)

        # SSIM
        s_c = float(ssim(p_c, r_c, data_range=255))

        # Orientation
        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        # Pearson
        p_vals = p_raw[valid].astype(float)
        r_vals = r_raw[valid].astype(float)
        pearson = float(np.corrcoef(p_vals, r_vals)[0, 1]) if (np.std(p_vals) > 1e-3 and np.std(r_vals) > 1e-3) else 1.0

        # Diff loc with k_open_3 and th=30.0
        p_v = p_raw[valid & (~glare_px)].astype(np.float32)
        r_v = r_raw[valid & (~glare_px)].astype(np.float32)
        if len(p_v) > 10 and np.std(r_v) > 1e-3:
            mu_p, std_p = np.mean(p_v), np.std(p_v)
            mu_r, std_r = np.mean(r_v), np.std(r_v)
            scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p
        else:
            r_norm_loc = r_raw.astype(np.float32)

        diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
        diff_loc[~valid] = 0.0; diff_loc[glare_px] = 0.0
        bin_diff = (diff_loc > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)
        
        blobs_60 = []
        for lbl in range(1, num_lbl):
            area = stats[lbl, cv2.CC_STAT_AREA]
            m = (lbls == lbl)
            if np.max(diff_loc[m]) >= 60.0:
                blobs_60.append(area)
        max_blob = max(blobs_60, default=0)

        # Condition for intact parallel texture shift (Tri #466):
        # 1. ori_corr >= 0.90 (direction completely preserved)
        # 2. coh_p >= 0.35 (strong parallel texture)
        # 3. pearson >= 0.65 (overall pattern matches well)
        # 4. ssim_clahe >= 0.45 (texture structure intact)
        is_intact_parallel_shift = bool(ori_corr >= 0.90 and coh_p >= 0.35 and pearson >= 0.65 and s_c >= 0.45)

        print(f"Tri #{tid:3d}: ori_corr={ori_corr:.2f}, coh_p={coh_p:.2f}, pearson={pearson:.2f}, ssim={s_c:.2f} | blob={max_blob:3d} | is_shift={is_intact_parallel_shift}")

evaluate_rule(r'images\1.jpg', r'images\2.jpg', [605, 466, 734, 735, 736, 737, 3, 11, 12, 13, 83], '1.jpg vs 2.jpg')
evaluate_rule(r'images\test_nobg.png', r'images\test_nobg - scar.png', [98, 121, 123, 124, 341, 432, 433, 454], 'test_nobg vs scar')
