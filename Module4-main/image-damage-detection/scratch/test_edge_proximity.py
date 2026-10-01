import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

def analyze_triangle(p_path, r_path, tid):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

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
    glare_px = (r_c > 240) & (p_c > 240)

    p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[~valid] = 0
    dist_to_p_edge = cv2.distanceTransform(cv2.bitwise_not(p_c_edge), cv2.DIST_L2, 3)

    p_vals = p_raw[valid & (~glare_px)].astype(np.float32)
    r_vals = r_raw[valid & (~glare_px)].astype(np.float32)
    if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
        mu_p, std_p = np.mean(p_vals), np.std(p_vals)
        mu_r, std_r = np.mean(r_vals), np.std(r_vals)
        scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
        r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p
    else:
        r_norm_loc = r_raw.astype(np.float32)

    diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
    diff_loc[~valid] = 0.0; diff_loc[glare_px] = 0.0
    bin_diff = (diff_loc > 30.0).astype(np.uint8) * 255
    opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

    for lbl in range(1, num_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        m = (lbls == lbl)
        max_d = np.max(diff_loc[m])
        # Check distance of this blob to p_c_edge
        dists = dist_to_p_edge[m]
        d_mean = np.mean(dists)
        d_max = np.max(dists)
        d_far_ratio = np.mean(dists > 3.0)
        print(f"  Blob area={area:3d}, max_diff={max_d:5.1f} | dist to edge: mean={d_mean:.2f}, max={d_max:.2f}, far(>3px) ratio={d_far_ratio:.2f}")

print("=== 1.jpg vs 2.jpg: Tri #466 (False Positive) ===")
analyze_triangle(r'images\1.jpg', r'images\2.jpg', 466)

print("\n=== 1.jpg vs 2.jpg: Tri #605 (Real Defect) ===")
analyze_triangle(r'images\1.jpg', r'images\2.jpg', 605)

print("\n=== test_nobg vs scar: Tri #98 (Real Scar) ===")
analyze_triangle(r'images\test_nobg.png', r'images\test_nobg - scar.png', 98)

print("\n=== test_nobg vs scar: Tri #432 (Real Scar) ===")
analyze_triangle(r'images\test_nobg.png', r'images\test_nobg - scar.png', 432)
