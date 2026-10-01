import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
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
reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

for tid in [605, 466]:
    tri = triangles[tid]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, 72)
    patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, 72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

    p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    valid = mask_p > 0

    p_vals = p_raw[valid].astype(np.float32)
    r_vals = r_raw[valid].astype(np.float32)
    mu_p, std_p = np.mean(p_vals), np.std(p_vals)
    mu_r, std_r = np.mean(r_vals), np.std(r_vals)
    scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
    r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p

    s_diff = r_norm_loc - p_raw.astype(np.float32)
    s_diff[~valid] = 0.0

    print(f"--- Tri #{tid} ---")
    print(f"mu_p={mu_p:.1f}, std_p={std_p:.1f} | mu_r={mu_r:.1f}, std_r={std_r:.1f} | scale={scale:.2f}")
    print(f"s_diff min={s_diff.min():.1f}, max={s_diff.max():.1f}")
    
    for th in [30.0, 35.0, 40.0, 50.0]:
        pos = np.sum(s_diff > th)
        neg = np.sum(s_diff < -th)
        ratio = min(pos, neg) / (max(pos, neg) + 1e-5)
        print(f"  th={th:4.1f}: pos={pos:4d}, neg={neg:4d}, dipole_ratio={ratio:.3f}")

    # Connected components on diff_loc > th
    diff_loc = np.abs(s_diff)
    for k_sz in [3, 5]:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        for th in [30.0, 35.0]:
            b_diff = (diff_loc > th).astype(np.uint8) * 255
            op = cv2.morphologyEx(b_diff, cv2.MORPH_OPEN, k_open)
            n, l, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
            areas = [st[idx, cv2.CC_STAT_AREA] for idx in range(1, n)]
            max_d = [np.max(diff_loc[l == idx]) for idx in range(1, n)]
            print(f"  k_sz={k_sz}, th={th}: areas={areas}, max_diffs={[round(x,1) for x in max_d]}")
