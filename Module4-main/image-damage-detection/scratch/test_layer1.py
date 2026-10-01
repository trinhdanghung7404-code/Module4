import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def test_layer1_logic(p_path, r_path, is_undamaged=False):
    p = cv2.imread(p_path)
    r = cv2.imread(r_path)

    seg = ObjectSegmenter()
    p_seg = seg.segment(p)
    r_seg = seg.segment(r)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r, p, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_ridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    flagged = []
    details = []

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        patch_p, mask_p = canonical_triangle_patch(p, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_gray = clahe.apply(cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY))
        r_gray = clahe.apply(cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY))

        # --- 1. Morphological Ridges (Top-Hat & Black-Hat) ---
        p_bhat = cv2.morphologyEx(p_gray, cv2.MORPH_BLACKHAT, k_ridge)
        p_what = cv2.morphologyEx(p_gray, cv2.MORPH_TOPHAT, k_ridge)
        p_ridge = cv2.max(p_bhat, p_what)
        p_ridge[mask_p == 0] = 0

        r_bhat = cv2.morphologyEx(r_gray, cv2.MORPH_BLACKHAT, k_ridge)
        r_what = cv2.morphologyEx(r_gray, cv2.MORPH_TOPHAT, k_ridge)
        r_ridge = cv2.max(r_bhat, r_what)
        r_ridge[mask_p == 0] = 0

        # Tolerate minor 3D shift by dilating product ridge with 5x5 ellipse (2px tolerance)
        p_ridge_dil = cv2.dilate(p_ridge, k_tol)
        r_ridge_dil = cv2.dilate(r_ridge, k_tol)

        # New intrusive ridge (scratch or crack)
        new_ridge = cv2.subtract(r_ridge, p_ridge_dil)
        new_ridge[mask_p == 0] = 0

        # Broken pattern line
        missing_ridge = cv2.subtract(p_ridge, r_ridge_dil)
        missing_ridge[mask_p == 0] = 0

        # Exclude glare
        glare = (r_gray > 230) | (p_gray > 230)
        missing_ridge[glare] = 0

        # Thresholding for physical ridge: ridge intensity >= 35
        _, new_ridge_bin = cv2.threshold(new_ridge, 35, 255, cv2.THRESH_BINARY)
        _, missing_ridge_bin = cv2.threshold(missing_ridge, 35, 255, cv2.THRESH_BINARY)

        cnts_new, _ = cv2.findContours(new_ridge_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_new_len = max([cv2.arcLength(c, False) for c in cnts_new], default=0.0)

        cnts_mis, _ = cv2.findContours(missing_ridge_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_mis_len = max([cv2.arcLength(c, False) for c in cnts_mis], default=0.0)

        # A real physical defect has continuous length >= 14px
        has_new_defect = max_new_len >= 14.0
        has_broken = max_mis_len >= 16.0

        is_l1_damage = bool(has_new_defect or has_broken)
        if is_l1_damage:
            flagged.append(i)
        details.append({'id': i, 'max_new': max_new_len, 'max_mis': max_mis_len, 'is_l1': is_l1_damage})

    print(f'Results for {os.path.basename(r_path)}: Total Triangles={len(triangles)}, Flagged={len(flagged)}')
    if not is_undamaged:
        for tid in [432, 454, 98, 124, 123]:
            print(f'  Target Tri #{tid}: Flagged? {tid in flagged}, max_new={details[tid]["max_new"]:.1f}, max_mis={details[tid]["max_mis"]:.1f}')
    return len(triangles), len(flagged)

if __name__ == "__main__":
    print("--- TESTING SCAR DEFECT ---")
    test_layer1_logic(
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\test_nobg.png',
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\test_nobg - scar.png',
        is_undamaged=False
    )

    print("\n--- TESTING UNDAMAGED VASE ---")
    test_layer1_logic(
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg',
        r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\2.jpg',
        is_undamaged=True
    )
