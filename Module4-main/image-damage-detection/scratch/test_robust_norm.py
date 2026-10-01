import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from detailed_debug import canonical_triangle_patch, micro_align_patches

p1_path = os.path.join(PROJECT_ROOT, "images", "1.jpg")
p16_path = os.path.join(PROJECT_ROOT, "images", "16.jpg")

prod_bgr = cv2.imread(p1_path)
ret_bgr = cv2.imread(p16_path)

seg = ObjectSegmenter()
p_seg = seg.segment(prod_bgr)
r_seg = seg.segment(ret_bgr)

norm = ImageNormalizer()
ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])
ret_ct = norm.color_transfer(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])

reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res["product_points"], reg_res["return_points"], np.arange(len(reg_res["product_points"])))

tri = triangles[685]
pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

valid = mask_p > 0
p_vals = p_raw[valid].astype(np.float32)
r_vals = r_raw[valid].astype(np.float32)

# 1. Standard Mean/Std normalization (Current):
mu_p, std_p = np.mean(p_vals), np.std(p_vals)
mu_r, std_r = np.mean(r_vals), np.std(r_vals)
scale_curr = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
r_norm_curr = (r_raw.astype(np.float32) - mu_r) * scale_curr + mu_p
diff_curr = np.abs(r_norm_curr - p_raw.astype(np.float32))
diff_curr[~valid] = 0

# 2. Robust Median/MAD normalization:
med_p = np.median(p_vals)
med_r = np.median(r_vals)
mad_p = np.median(np.abs(p_vals - med_p)) * 1.4826  # scaled to approximate std
mad_r = np.median(np.abs(r_vals - med_r)) * 1.4826
scale_robust = np.clip(mad_p / (mad_r + 1e-5), 0.7, 1.4)
r_norm_robust = (r_raw.astype(np.float32) - med_r) * scale_robust + med_p
diff_robust = np.abs(r_norm_robust - p_raw.astype(np.float32))
diff_robust[~valid] = 0

dark_mask = (r_raw < 50) & (p_raw > 100) & valid

print(f"Total valid pixels: {np.sum(valid)}")
print(f"Defect dark pixels: {np.sum(dark_mask)}")
print("\n--- STANDARD NORMALIZATION ---")
print(f"mu_p: {mu_p:.1f}, mu_r: {mu_r:.1f}, scale: {scale_curr:.3f}")
print(f"Max diff_curr: {np.max(diff_curr):.1f}")
print(f"Defect mean diff_curr: {np.mean(diff_curr[dark_mask]):.1f}")

print("\n--- ROBUST MEDIAN/MAD NORMALIZATION ---")
print(f"med_p: {med_p:.1f}, med_r: {med_r:.1f}, scale: {scale_robust:.3f}")
print(f"Max diff_robust: {np.max(diff_robust):.1f}")
print(f"Defect mean diff_robust: {np.mean(diff_robust[dark_mask]):.1f}")
print(f"Defect max diff_robust:  {np.max(diff_robust[dark_mask]):.1f}")

# Connected components with robust diff:
bin_robust = (diff_robust > 35.0).astype(np.uint8) * 255
k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened_robust = cv2.morphologyEx(bin_robust, cv2.MORPH_OPEN, k_open_3)
n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_robust, connectivity=8)
print(f"\nRobust components count: {n_lbl - 1}")
for lbl in range(1, n_lbl):
    area = stats[lbl, cv2.CC_STAT_AREA]
    max_d = np.max(diff_robust[lbls == lbl])
    print(f"  Component {lbl}: area = {area}px, max_diff = {max_d:.1f}")
