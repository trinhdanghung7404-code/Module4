import cv2, numpy as np, sys
sys.path.insert(0, 'v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches, compute_orientation_profile

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
k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for tid in [605, 606, 735, 736]:
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
    valid = mask_p > 0

    p_bhat = cv2.morphologyEx(p_c, cv2.MORPH_BLACKHAT, k_morph)
    r_bhat = cv2.morphologyEx(r_c, cv2.MORPH_BLACKHAT, k_morph)
    p_what = cv2.morphologyEx(p_c, cv2.MORPH_TOPHAT, k_morph)
    r_what = cv2.morphologyEx(r_c, cv2.MORPH_TOPHAT, k_morph)

    p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[~valid] = 0
    r_c_edge = cv2.Canny(r_c, 40, 120); r_c_edge[~valid] = 0

    p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
    dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

    p_vals = p_raw[valid].astype(np.float32)
    r_vals = r_raw[valid].astype(np.float32)
    mu_p, std_p = np.mean(p_vals), np.std(p_vals)
    mu_r, std_r = np.mean(r_vals), np.std(r_vals)
    scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
    r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p

    diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
    diff_loc[~valid] = 0.0

    bin_diff = (diff_loc > 35.0).astype(np.uint8) * 255
    opened_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open)

    # Check Color diff (Layer 2)
    p_lab = cv2.cvtColor(patch_p, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    da = p_lab[..., 1] - r_lab[..., 1]
    db = p_lab[..., 2] - r_lab[..., 2]
    chroma_err = np.sqrt(da**2 + db**2)
    mean_chroma = np.mean(chroma_err[valid])

    print(f"\n--- Tri #{tid} ---")
    print(f"p_raw mean={mu_p:.1f}, std={std_p:.1f} | r_raw mean={mu_r:.1f}, std={std_r:.1f}")
    print(f"diff_loc max={np.max(diff_loc):.1f}, mean={np.mean(diff_loc):.1f}")
    print(f"opened_diff count > 0: {np.sum(opened_diff > 0)}")
    print(f"mean_chroma_err: {mean_chroma:.1f}")

    # Save visual
    vis = np.hstack([patch_p, patch_r_aligned, cv2.cvtColor(opened_diff, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(f'scratch/tri_{tid}_debug_defect.jpg', vis)
