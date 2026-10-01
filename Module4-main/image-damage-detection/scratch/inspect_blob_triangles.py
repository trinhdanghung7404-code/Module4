import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from detailed_debug import canonical_triangle_patch, micro_align_patches, compute_orientation_profile

p1_path = os.path.join(PROJECT_ROOT, "images", "1.jpg")
p16_path = os.path.join(PROJECT_ROOT, "images", "16.jpg")

prod_bgr = cv2.imread(p1_path)
ret_bgr = cv2.imread(p16_path)

seg = ObjectSegmenter()
p_seg = seg.segment(prod_bgr)
r_seg = seg.segment(ret_bgr)

norm = ImageNormalizer()
ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])
ret_ct = norm.color_transfer(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])

reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res["product_points"], reg_res["return_points"], np.arange(len(reg_res["product_points"])))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

flagged_candidates = [4, 11, 22, 35, 46, 47, 61, 73, 77, 80, 115, 172, 200, 201, 227, 228, 234, 249, 300, 352, 355, 357, 386, 409, 419, 463, 505, 557, 566, 579, 618, 626, 627, 633, 664, 665, 684, 685, 688, 692, 713]
real_defects = [227, 228, 249, 352, 357, 566, 626, 627, 685]

print("TRIANGLE ANALYSIS (Flagged candidates):")
print(f"{'ID':>4} | {'IsReal':>6} | {'MaxDiff':>7} | {'BlobArea':>8} | {'OriCorr':>7} | {'PEdge':>5} | {'Min R':>5} | {'Min P':>5} | {'Max R':>5} | {'Max P':>5}")
print("-" * 80)

for tri_id in flagged_candidates:
    tri = triangles[tri_id]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

    patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    valid = mask_p > 0
    glare_px = (clahe.apply(r_raw_gray) > 240) | (clahe.apply(p_raw_gray) > 240)

    p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
    r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)

    med_p = np.median(p_vals)
    med_r = np.median(r_vals)
    mad_p = np.median(np.abs(p_vals - med_p)) * 1.4826
    mad_r = np.median(np.abs(r_vals - med_r)) * 1.4826
    scale = np.clip(mad_p / (mad_r + 1e-5), 0.7, 1.4)
    r_norm = (r_raw_gray.astype(np.float32) - med_r) * scale + med_p

    diff = np.abs(r_norm - p_raw_gray.astype(np.float32))
    diff[~valid] = 0.0
    diff[glare_px] = 0.0

    bin_diff = (diff > 30.0).astype(np.uint8) * 255
    opened = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    max_blob = max([stats[l, cv2.CC_STAT_AREA] for l in range(1, num_lbl)], default=0)
    max_d = np.max(diff[valid])

    hist_p, peaks_p, _ = compute_orientation_profile(clahe.apply(p_raw_gray), mask_p)
    hist_r, peaks_r, _ = compute_orientation_profile(clahe.apply(r_raw_gray), mask_p)
    ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

    p_edge = cv2.Canny(p_raw_gray, 60, 160)
    p_edge[~valid] = 0
    p_edge_cnt = int(np.sum(p_edge > 0))

    is_real = "YES" if tri_id in real_defects else "NO"
    min_r = np.min(r_vals)
    min_p = np.min(p_vals)
    max_r = np.max(r_vals)
    max_p = np.max(p_vals)

    print(f"{tri_id:>4} | {is_real:>6} | {max_d:>7.1f} | {max_blob:>8} | {ori_corr:>7.2f} | {p_edge_cnt:>5} | {min_r:>5.0f} | {min_p:>5.0f} | {max_r:>5.0f} | {max_p:>5.0f}")
