import cv2, numpy as np, sys
from skimage.metrics import structural_similarity as ssim
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
reg = ImageRegistration(max_keypoints=None).register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
v, t = mb.build(reg['product_points'], reg['return_points'], np.arange(len(reg['product_points'])))

clahe = cv2.createCLAHE(2.0, (8, 8))
k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print(f"{'Metric':<25} | {'Tri #605 (DEFECT)':<20} | {'Tri #466 (FALSE POSITIVE)':<25}")
print("-" * 75)

for tid in [605, 466]:
    tri = t[tid]
    pts_p = np.array([v[i].product_xy for i in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([v[i].return_xy for i in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, 72)
    patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, 72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=3)

    p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    p_c = clahe.apply(p_raw)
    r_c = clahe.apply(r_raw)
    valid = mask_p > 0

    # SSIM
    s = ssim(p_raw, r_raw, data_range=255)
    s_c = ssim(p_c, r_c, data_range=255)

    # Orientation
    hist_p, peaks_p, coh_p = compute_orientation_profile(p_c, mask_p)
    hist_r, peaks_r, coh_r = compute_orientation_profile(r_c, mask_p)
    ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

    # Pearson correlation
    pearson = float(np.corrcoef(p_raw[valid].astype(float), r_raw[valid].astype(float))[0, 1])

    # Local norm diff
    p_vals = p_raw[valid].astype(np.float32)
    r_vals = r_raw[valid].astype(np.float32)
    mu_p, std_p = np.mean(p_vals), np.std(p_vals)
    mu_r, std_r = np.mean(r_vals), np.std(r_vals)
    scale = np.clip(std_p / (std_r + 1e-5), 0.7, 1.4)
    r_norm_loc = (r_raw.astype(np.float32) - mu_r) * scale + mu_p
    diff_loc = np.abs(r_norm_loc - p_raw.astype(np.float32))
    diff_loc[~valid] = 0.0

    # Edges
    p_edge = cv2.Canny(p_c, 40, 120); p_edge[~valid] = 0
    r_edge = cv2.Canny(r_c, 40, 120); r_edge[~valid] = 0

    # Edge diff
    edge_diff = np.sum(cv2.absdiff(p_edge, r_edge) > 0)

    if tid == 605:
        m605 = {
            'SSIM raw': s, 'SSIM CLAHE': s_c, 'Ori Corr': ori_corr, 'Coh P': coh_p, 'Coh R': coh_r,
            'Pearson': pearson, 'Std P': std_p, 'Std R': std_r, 'Edge Diff Px': edge_diff,
            'Max diff_loc': np.max(diff_loc), 'Mean diff_loc': np.mean(diff_loc[valid])
        }
    else:
        m466 = {
            'SSIM raw': s, 'SSIM CLAHE': s_c, 'Ori Corr': ori_corr, 'Coh P': coh_p, 'Coh R': coh_r,
            'Pearson': pearson, 'Std P': std_p, 'Std R': std_r, 'Edge Diff Px': edge_diff,
            'Max diff_loc': np.max(diff_loc), 'Mean diff_loc': np.mean(diff_loc[valid])
        }

for k in m605:
    v1 = f"{m605[k]:.3f}" if isinstance(m605[k], float) else str(m605[k])
    v2 = f"{m466[k]:.3f}" if isinstance(m466[k], float) else str(m466[k])
    print(f"{k:<25} | {v1:<20} | {v2:<25}")
