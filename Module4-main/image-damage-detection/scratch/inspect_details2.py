import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

def inspect_details(p_path, r_path, tids, label):
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
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        p_c = clahe.apply(p_raw)
        r_c = clahe.apply(r_raw)

        valid = mask_p > 0
        hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
        hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
        ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

        raw_diff = r_raw.astype(np.float32) - p_raw.astype(np.float32)
        raw_diff[~valid] = 0.0
        bin_raw_diff = (np.abs(raw_diff) > 35.0).astype(np.uint8) * 255
        opened_raw = cv2.morphologyEx(bin_raw_diff, cv2.MORPH_OPEN, k_open)
        raw_pos = np.sum((opened_raw > 0) & (raw_diff > 35.0))
        raw_neg = np.sum((opened_raw > 0) & (raw_diff < -35.0))
        dipole_r = min(raw_pos, raw_neg) / (max(raw_pos, raw_neg) + 1e-5)

        # check std and gradient of p_raw
        grad_p = np.mean(np.sqrt(cv2.Sobel(p_raw, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw, cv2.CV_32F, 0, 1)**2)[valid])
        std_p = np.std(p_raw[valid])
        std_r = np.std(r_raw[valid])

        print(f"Tri #{tid:3d}: coh_p={coh_p:.2f}, coh_r={coh_r:.2f}, ori_corr={ori_corr:.2f}, "
              f"pos={raw_pos:3d}, neg={raw_neg:3d}, dipole={dipole_r:.2f}, grad_p={grad_p:.1f}, std_p={std_p:.1f}, std_r={std_r:.1f}")

if __name__ == '__main__':
    inspect_details(r'images\1.jpg', r'images\2.jpg', [3, 12, 13, 83, 734, 735, 736, 737], '1.jpg vs 2.jpg')
    inspect_details(r'images\test_nobg.png', r'images\test_nobg - scar.png', [98, 121, 123, 124, 341, 432, 433, 454], 'scar')
