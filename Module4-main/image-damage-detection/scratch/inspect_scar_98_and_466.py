import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def check_tri(p_path, r_path, tid, name):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    tri = triangles[tid]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

    p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    p_c = clahe.apply(p_raw)
    r_c = clahe.apply(r_raw)
    valid = mask_p > 0

    p_edge = cv2.Canny(p_c, 40, 120); p_edge[~valid] = 0
    r_edge = cv2.Canny(r_c, 40, 120); r_edge[~valid] = 0

    print(f"=== {name} (Tri #{tid}) ===")
    print(f"p_edge count={np.sum(p_edge > 0)}, r_edge count={np.sum(r_edge > 0)}")
    print(f"p_raw: min={p_raw[valid].min()}, max={p_raw[valid].max()}, mean={p_raw[valid].mean():.1f}, std={p_raw[valid].std():.1f}")
    print(f"r_raw: min={r_raw[valid].min()}, max={r_raw[valid].max()}, mean={r_raw[valid].mean():.1f}, std={r_raw[valid].std():.1f}")
    print(f"p_raw vs r_raw: abs diff mean={np.mean(np.abs(r_raw[valid].astype(float) - p_raw[valid].astype(float))):.1f}")
    print(f"p_raw vs r_raw: max diff={np.max(np.abs(r_raw[valid].astype(float) - p_raw[valid].astype(float))):.1f}")

check_tri(r'images\1.jpg', r'images\2.jpg', 466, 'Vase Tri 466 (False Positive)')
check_tri(r'images\1.jpg', r'images\2.jpg', 605, 'Vase Tri 605 (Defect Blue Streak)')
check_tri(r'images\test_nobg.png', r'images\test_nobg - scar.png', 98, 'Scar Tri 98')
check_tri(r'images\test_nobg.png', r'images\test_nobg - scar.png', 121, 'Scar Tri 121')
check_tri(r'images\test_nobg.png', r'images\test_nobg - scar.png', 123, 'Scar Tri 123')
check_tri(r'images\test_nobg.png', r'images\test_nobg - scar.png', 432, 'Scar Tri 432')
