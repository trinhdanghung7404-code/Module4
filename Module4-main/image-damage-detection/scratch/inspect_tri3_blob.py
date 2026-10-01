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

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

print(f"mu_p={mu_p:.1f}, std_p={std_p:.1f}, mu_r={mu_r:.1f}, std_r={std_r:.1f}, scale={scale:.2f}")
print(f"opened_diff count > 0: {np.sum(opened_diff > 0)}, num_lbl: {num_lbl}")
for lbl in range(1, num_lbl):
    area = stats[lbl, cv2.CC_STAT_AREA]
    ys, xs = np.where(lbls == lbl)
    print(f"  Blob {lbl}: area={area}, bbox=({xs.min()},{ys.min()}) to ({xs.max()},{ys.max()})")

# Save visual of diff_loc
vis = np.hstack([p_raw, r_norm_loc.astype(np.uint8), np.clip(diff_loc * 3, 0, 255).astype(np.uint8), opened_diff])
cv2.imwrite('scratch/tri3_diff_loc.png', vis)
