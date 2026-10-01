import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def get_tri_diff(img_p_path, img_r_path, tri_ids):
    product_img = cv2.imread(img_p_path)
    return_img = cv2.imread(img_r_path)
    segmenter = ObjectSegmenter()
    p_seg = segmenter.segment(product_img)
    r_seg = segmenter.segment(return_img)
    normalizer = ImageNormalizer()
    return_normalized = normalizer.normalize(return_img, product_img, r_seg['mask'], p_seg['mask'])
    return_ct = normalizer.color_transfer(return_img, product_img, r_seg['mask'], p_seg['mask'])
    registrator = ImageRegistration(max_keypoints=None)
    reg_result = registrator.register(product_img, return_normalized, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_result['product_points'], reg_result['return_points'], np.arange(len(reg_result['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    results = {}
    for tid in tri_ids:
        tri = triangles[tid]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(product_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(return_ct, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)
        valid = mask_p > 0
        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

        p_vals = p_raw[valid].astype(np.float32)
        r_vals = r_raw[valid].astype(np.float32)
        mu_p, std_p = np.mean(p_vals), np.std(p_vals)
        mu_r, std_r = np.mean(r_vals), np.std(r_vals)
        scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
        r_norm = (r_raw.astype(np.float32) - mu_r) * scale + mu_p

        diff = np.abs(r_norm - p_raw.astype(np.float32))
        diff[~valid] = 0.0

        # Raw difference (without local mean shift)
        raw_abs_diff = np.abs(r_raw.astype(np.float32) - p_raw.astype(np.float32))
        raw_abs_diff[~valid] = 0.0

        results[tid] = {
            "max_diff": float(np.max(diff)),
            "mean_diff": float(np.mean(diff[valid])),
            "p95_diff": float(np.percentile(diff[valid], 95)),
            "max_raw_diff": float(np.max(raw_abs_diff)),
            "p95_raw_diff": float(np.percentile(raw_abs_diff[valid], 95)),
            "area_diff_gt_50": int(np.sum(diff > 50.0)),
            "area_diff_gt_70": int(np.sum(diff > 70.0)),
            "area_raw_gt_80": int(np.sum(raw_abs_diff > 80.0)),
        }
    return results

print("--- SCAR DATASET (test_nobg) ---")
scar_ids = [341, 432, 433, 438, 439, 446, 454]
res_scar = get_tri_diff('images/test_nobg.png', 'images/test_nobg - scar.png', scar_ids)
for tid, d in res_scar.items():
    print(f"Scar Tri #{tid:3d}: max={d['max_diff']:.1f}, p95={d['p95_diff']:.1f}, diff>50={d['area_diff_gt_50']}px, diff>70={d['area_diff_gt_70']}px, raw>80={d['area_raw_gt_80']}px")

print("\n--- VASE DATASET (1.jpg vs 2.jpg) FP TRIANGLES ---")
fp_ids = [21, 31, 47, 53, 72, 82, 143, 146, 150, 407, 469, 535, 646, 648, 652, 712]
res_vase = get_tri_diff('images/1.jpg', 'images/2.jpg', fp_ids)
for tid, d in res_vase.items():
    print(f"Vase Tri #{tid:3d}: max={d['max_diff']:.1f}, p95={d['p95_diff']:.1f}, diff>50={d['area_diff_gt_50']}px, diff>70={d['area_diff_gt_70']}px, raw>80={d['area_raw_gt_80']}px")
