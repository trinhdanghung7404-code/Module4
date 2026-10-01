import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

p_img = cv2.imread(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg')
r_img = cv2.imread(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\2.jpg')

seg = ObjectSegmenter()
p_seg = seg.segment(p_img)
r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
tri = triangles[3]
pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

p_gray = clahe.apply(cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY))
r_gray = clahe.apply(cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY))
valid = mask_p > 0

p_edge = cv2.Canny(p_gray, 40, 120); p_edge[~valid] = 0
r_edge = cv2.Canny(r_gray, 40, 120); r_edge[~valid] = 0

dist_r = cv2.distanceTransform(cv2.bitwise_not(r_edge), cv2.DIST_L2, 3)
dist_p = cv2.distanceTransform(cv2.bitwise_not(p_edge), cv2.DIST_L2, 3)

# In detailed_debug:
# broken_edges = (p_c_edge > 0) & (dist_r > 5.0) & (~glare_px) & valid
# intrusive_edges = (r_c_edge > 0) & (grad_p < 25.0) & (~glare_px) & valid
grad_p = np.sqrt(cv2.Sobel(p_gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_gray, cv2.CV_32F, 0, 1)**2)

print('Tri 3:')
print(f'p_edge total px: {np.sum(p_edge > 0)}')
print(f'r_edge total px: {np.sum(r_edge > 0)}')
print(f'dist_r on p_edge: min={dist_r[p_edge > 0].min():.2f}, mean={dist_r[p_edge > 0].mean():.2f}, max={dist_r[p_edge > 0].max():.2f}')
print(f'p_edge pixels with dist_r > 5.0: {np.sum(p_edge[dist_r > 5.0] > 0)}')
print(f'r_edge pixels with grad_p < 25.0: {np.sum(r_edge[grad_p < 25.0] > 0)}')

# Check the solid blob
p_vals = p_gray[valid].astype(np.float32)
r_vals = r_gray[valid].astype(np.float32)
mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
r_norm_loc = (r_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc
diff_loc = np.abs(r_norm_loc - p_gray.astype(np.float32))
diff_loc[~valid] = 0.0

k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
bin_diff_loc = (diff_loc > 35.0).astype(np.uint8) * 255
eroded_diff_loc = cv2.erode(bin_diff_loc, k_erode)
num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(eroded_diff_loc, connectivity=8)
areas = [stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_lbl)]
print(f'Blob areas (> 35 diff): {areas}')
