import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

def check_triangles():
    # 1. 1.jpg vs 2.jpg
    p_img = cv2.imread(r'images\1.jpg'); r_img = cv2.imread(r'images\2.jpg')
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    v1, t1 = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

    # 2. test_nobg vs scar
    s_p_img = cv2.imread(r'images\test_nobg.png'); s_r_img = cv2.imread(r'images\test_nobg - scar.png')
    s_p_seg = seg.segment(s_p_img); s_r_seg = seg.segment(s_r_img)
    s_r_norm = norm.normalize(s_r_img, s_p_img, s_r_seg['mask'], s_p_seg['mask'])
    s_reg = ImageRegistration(max_keypoints=None).register(s_p_img, s_r_norm, s_p_seg['mask'], s_r_seg['mask'])
    v2, t2 = mb.build(s_reg['product_points'], s_reg['return_points'], np.arange(len(s_reg['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def inspect_one(img_p, img_r, vertices, tri, tid):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(img_p, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(img_r, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c = clahe.apply(p_raw); r_c = clahe.apply(r_raw)
        valid = mask_p > 0
        glare_px = (r_c > 240) & (p_c > 240)

        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[~valid] = 0
        dist_to_p_edge = cv2.distanceTransform(cv2.bitwise_not(p_c_edge), cv2.DIST_L2, 3)

        p_vals = p_raw[valid & (~glare_px)].astype(np.float32)
        r_vals = r_raw[valid & (~glare_px)].astype(np.float32)
        if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
            mu_p, std_p = np.mean(p_vals), np.std(p_vals)
            mu_r, std_r = np.mean(r_vals), np.std(r_vals)
            scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
            r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p
        else:
            r_norm_loc = r_raw.astype(np.float32)

        diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
        diff_loc[~valid] = 0.0; diff_loc[glare_px] = 0.0

        bin_diff = (diff_loc > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

        blobs = []
        for lbl in range(1, num_lbl):
            area = stats[lbl, cv2.CC_STAT_AREA]
            m = (lbls == lbl)
            max_d = float(np.max(diff_loc[m]))
            mean_d = float(np.mean(diff_loc[m]))
            dists = dist_to_p_edge[m]
            mean_dist = float(np.mean(dists))
            max_dist = float(np.max(dists))
            far_ratio = float(np.mean(dists > 3.0))
            blobs.append({
                'area': area, 'max_d': max_d, 'mean_d': mean_d,
                'mean_dist': mean_dist, 'max_dist': max_dist, 'far_ratio': far_ratio
            })
        return ori_corr, coh_p, blobs

    print("=== 1.jpg vs 2.jpg ===")
    for tid in [15, 466, 605, 3, 734]:
        corr, coh, b_list = inspect_one(p_img, r_norm, v1, t1[tid], tid)
        print(f"Tri #{tid:3d}: corr={corr:.2f}, coh={coh:.2f} | blobs={[(b['area'], round(b['max_d'],1), round(b['mean_dist'],2), round(b['far_ratio'],2)) for b in b_list]}")

    print("\n=== test_nobg vs scar ===")
    for tid in [98, 121, 123, 124, 341, 432, 433, 454]:
        corr, coh, b_list = inspect_one(s_p_img, s_r_norm, v2, t2[tid], tid)
        print(f"Scar #{tid:3d}: corr={corr:.2f}, coh={coh:.2f} | blobs={[(b['area'], round(b['max_d'],1), round(b['mean_dist'],2), round(b['far_ratio'],2)) for b in b_list]}")

if __name__ == '__main__':
    check_triangles()
