import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def check_pearson(p_path, r_path, tids, name):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

    print(f"=== {name} ===")
    for tid in tids:
        tri = triangles[tid]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        valid = mask_p > 0

        p_vals = p_raw[valid].astype(float)
        r_vals = r_raw[valid].astype(float)
        if len(p_vals) > 10 and np.std(p_vals) > 1e-3 and np.std(r_vals) > 1e-3:
            pearson = float(np.corrcoef(p_vals, r_vals)[0, 1])
        else:
            pearson = 1.0
        print(f"Tri #{tid:3d}: pearson={pearson:.3f}, std_p={np.std(p_vals):.1f}, std_r={np.std(r_vals):.1f}")

check_pearson(r'images\1.jpg', r'images\2.jpg', [605, 466, 734, 735, 736, 737, 3, 11, 12, 13, 83], '1.jpg vs 2.jpg')
check_pearson(r'images\test_nobg.png', r'images\test_nobg - scar.png', [98, 121, 123, 124, 341, 432, 433, 454], 'test_nobg vs scar')
