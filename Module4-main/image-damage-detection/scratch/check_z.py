import cv2, numpy as np, sys, os
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches
from skimage.metrics import structural_similarity as ssim

def check(p_path, r_path, label):
    p = cv2.imread(p_path); r = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p); r_seg = seg.segment(r)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r, p, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    s_scores = []
    items = []

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        patch_p, mask_p = canonical_triangle_patch(p, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_c_gray = clahe.apply(cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY))
        r_c_gray = clahe.apply(cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY))

        p_c_edge = cv2.Canny(p_c_gray, 40, 120); p_c_edge[mask_p == 0] = 0
        r_c_edge = cv2.Canny(r_c_gray, 40, 120); r_c_edge[mask_p == 0] = 0

        p_dilated = cv2.dilate(p_c_edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        new_edges = cv2.bitwise_and(r_c_edge, cv2.bitwise_not(p_dilated)); new_edges[mask_p == 0] = 0
        is_contrast_ridge = (np.abs(r_c_gray.astype(np.float32) - p_c_gray.astype(np.float32)) > 12.0)
        new_edges[~is_contrast_ridge] = 0
        crack_contours, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_crack_length = max([cv2.arcLength(cnt, False) for cnt in crack_contours], default=0.0)

        ssim_val = float(ssim(p_c_gray, r_c_gray, data_range=255))
        s_scores.append(ssim_val)
        items.append({'id': i, 's_score': ssim_val, 'max_crack': max_crack_length})

    s_mean = np.mean(s_scores)
    s_std = np.std(s_scores)
    print(f'=== {label} ===')
    print(f's_mean={s_mean:.3f}, s_std={s_std:.3f}')
    if 'scar' in label:
        for tid in [432, 454, 98, 124, 123]:
            zs = (items[tid]['s_score'] - s_mean) / s_std
            print(f'  Tri #{tid}: s_score={items[tid]["s_score"]:.3f}, zs={zs:.2f}, max_crack={items[tid]["max_crack"]:.1f}')
    else:
        # count how many have zs < -1.8 or ssim < 0.2
        flagged = [x['id'] for x in items if (x['s_score'] - s_mean)/s_std < -1.8 or x['s_score'] < 0.2]
        print(f'Triangles with zs < -1.8 or ssim < 0.2: {len(flagged)} / {len(triangles)}')

check(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\test_nobg.png', r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\test_nobg - scar.png', 'test_nobg - scar')
check(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg', r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\2.jpg', '1.jpg vs 2.jpg')
