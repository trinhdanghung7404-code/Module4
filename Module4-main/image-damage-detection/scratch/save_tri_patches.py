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

for tid in [3, 11, 12, 13]:
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

    p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[mask_p == 0] = 0
    r_c_edge = cv2.Canny(r_c, 40, 120); r_c_edge[mask_p == 0] = 0
    r_raw_edge = cv2.Canny(r_raw, 40, 120); r_raw_edge[mask_p == 0] = 0

    panel = np.hstack([p_raw, r_raw, p_c, r_c, p_c_edge, r_c_edge, r_raw_edge])
    cv2.imwrite(f'scratch/debug_tri_{tid}.png', panel)
    print(f"Saved scratch/debug_tri_{tid}.png - pts_p center: {pts_p.mean(axis=0)}, pts_r center: {pts_r.mean(axis=0)}")
