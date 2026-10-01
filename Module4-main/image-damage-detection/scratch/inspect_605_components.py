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
reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

tri = triangles[605]
pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
valid = mask_p > 0

p_vals = p_raw[valid].astype(np.float32)
r_vals = r_raw[valid].astype(np.float32)
mu_p, std_p = np.mean(p_vals), np.std(p_vals)
mu_r, std_r = np.mean(r_vals), np.std(r_vals)
scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p

s_diff = r_norm_loc - p_raw.astype(np.float32)
s_diff[~valid] = 0.0

k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
bin_diff = (np.abs(s_diff) > 30.0).astype(np.uint8) * 255
opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
n, l, st, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

for idx in range(1, n):
    m = (l == idx)
    area = st[idx, cv2.CC_STAT_AREA]
    vals = s_diff[m]
    pos = np.sum(vals > 0)
    neg = np.sum(vals < 0)
    max_d = np.max(np.abs(vals))
    mean_d = np.mean(np.abs(vals))
    print(f"Component {idx}: area={area:3d}, pos={pos:3d}, neg={neg:3d}, max_diff={max_d:.1f}, mean_diff={mean_d:.1f}")
