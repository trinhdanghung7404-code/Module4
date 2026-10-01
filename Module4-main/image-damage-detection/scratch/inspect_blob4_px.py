import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

p_img = cv2.imread(r'images\1.jpg'); r_img = cv2.imread(r'images\2.jpg')
seg = ObjectSegmenter()
p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

tri = triangles[3]
pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=3)

p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
valid = mask_p > 0
glare_px = (r_raw > 240) & (p_raw > 240)

p_vals = p_raw[valid & (~glare_px)].astype(np.float32)
r_vals = r_raw[valid & (~glare_px)].astype(np.float32)
mu_p, std_p = np.mean(p_vals), np.std(p_vals)
mu_r, std_r = np.mean(r_vals), np.std(r_vals)
scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p

diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
diff_loc[~valid] = 0.0
diff_loc[glare_px] = 0.0

bin_diff = (diff_loc > 35.0).astype(np.uint8) * 255
opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open)

num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)
blob4_mask = (lbls == 4)

print("Blob 4 in Tri 3:")
print("p_raw in blob 4:", np.mean(p_raw[blob4_mask]), np.min(p_raw[blob4_mask]), np.max(p_raw[blob4_mask]))
print("r_raw in blob 4:", np.mean(r_raw[blob4_mask]), np.min(r_raw[blob4_mask]), np.max(r_raw[blob4_mask]))
print("r_norm_loc in blob 4:", np.mean(r_norm_loc[blob4_mask]), np.min(r_norm_loc[blob4_mask]), np.max(r_norm_loc[blob4_mask]))
print("signed diff (r_norm_loc - p_raw):", np.mean(r_norm_loc[blob4_mask] - p_raw[blob4_mask]))
print("signed raw diff (r_raw - p_raw):", np.mean(r_raw[blob4_mask].astype(float) - p_raw[blob4_mask].astype(float)))
