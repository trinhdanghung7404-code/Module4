import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def test_dipole_median(p_path, r_path, tids, name):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

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

        # Method: Robust offset using MEDIAN of difference
        diff_raw = r_raw.astype(np.float32) - p_raw.astype(np.float32)
        med_offset = np.median(diff_raw[valid])
        diff_zero = diff_raw - med_offset
        diff_zero[~valid] = 0.0

        bin_30 = (np.abs(diff_zero) > 30.0).astype(np.uint8) * 255
        op_30 = cv2.morphologyEx(bin_30, cv2.MORPH_OPEN, k_open_3)

        pos_px = np.sum((op_30 > 0) & (diff_zero > 30.0))
        neg_px = np.sum((op_30 > 0) & (diff_zero < -30.0))
        dipole_ratio = min(pos_px, neg_px) / (max(pos_px, neg_px) + 1e-5)

        print(f"Tri #{tid:3d}: med_offset={med_offset:5.1f} | pos={pos_px:4d}, neg={neg_px:4d}, dipole_ratio={dipole_ratio:.3f}")

test_dipole_median(r'images\1.jpg', r'images\2.jpg', [605, 466, 734, 735, 736, 737], '1.jpg vs 2.jpg')
test_dipole_median(r'images\test_nobg.png', r'images\test_nobg - scar.png', [98, 121, 123, 124, 341, 432, 433, 454], 'test_nobg vs scar')
