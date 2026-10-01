import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile, is_angle_matching_any

p_img = cv2.imread(r'images\test_nobg.png')
r_img = cv2.imread(r'images\test_nobg - scar.png')
seg = ObjectSegmenter()
p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

scar_ids = [98, 121, 123, 124, 341, 432, 433, 454]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for tid in scar_ids:
    tri = triangles[tid]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    valid = mask_p > 0
    glare_px = (r_raw_gray > 240) & (p_raw_gray > 240)

    p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
    r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)
    if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
        mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
        mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
        scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
        r_norm_loc = (r_raw_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc
    else:
        r_norm_loc = r_raw_gray.astype(np.float32)

    diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
    diff_loc[~valid] = 0.0
    diff_loc[glare_px] = 0.0
    bin_diff_loc = (diff_loc > 35.0).astype(np.uint8) * 255
    opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

    areas_100 = []
    areas_60 = []
    areas_raw = []
    for lbl in range(1, num_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        m = (lbls == lbl)
        areas_raw.append(area)
        if np.max(diff_loc[m]) >= 100.0 or np.mean(diff_loc[m]) >= 75.0:
            areas_100.append(area)
        if np.max(diff_loc[m]) >= 60.0:
            areas_60.append(area)

    print(f"Scar Tri #{tid:3d}: max_diff={np.max(diff_loc):.1f} | areas_raw={areas_raw} | areas_100={areas_100} | areas_60={areas_60}")
