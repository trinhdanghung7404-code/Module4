import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

p_img = cv2.imread(r'images\test_nobg.png')
r_img = cv2.imread(r'images\test_nobg - scar.png')
seg = ObjectSegmenter()
p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
scar_ids = [98, 121, 123, 124, 341, 432, 433, 454]

for tid in scar_ids:
    tri = triangles[tid]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)
    p_c = clahe.apply(cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY))
    r_c = clahe.apply(cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY))
    hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
    hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
    ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0
    print(f"Scar #{tid:3d}: coh_p={coh_p:.3f}, coh_r={coh_r:.3f}, ori_corr={ori_corr:.3f}")
