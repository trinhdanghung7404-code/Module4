import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def check_patch_corr(p_path, r_path, tids, label):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    print(f"\n--- {label} ---")
    for tid in tids:
        tri = triangles[tid]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=3)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        valid = mask_p > 0

        p_vals = p_raw[valid].astype(np.float32)
        r_vals = r_raw[valid].astype(np.float32)
        corr = float(np.corrcoef(p_vals, r_vals)[0, 1])

        # Also check correlation inside the blob
        mu_p, std_p = np.mean(p_vals), np.std(p_vals)
        mu_r, std_r = np.mean(r_vals), np.std(r_vals)
        scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
        r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p
        diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_diff = (diff_loc > 35.0).astype(np.uint8) * 255
        bin_diff[~valid] = 0
        opened = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open)

        print(f"Tri #{tid:3d}: Pearson corr={corr:.3f}, opened blob px={np.sum(opened > 0)}")

if __name__ == '__main__':
    check_patch_corr(r'images\1.jpg', r'images\2.jpg', [3, 11, 12, 13, 83, 734, 735, 736, 737], '1.jpg vs 2.jpg')
    check_patch_corr(r'images\test_nobg.png', r'images\test_nobg - scar.png', [98, 121, 123, 124, 341, 432, 433, 454], 'scar')
