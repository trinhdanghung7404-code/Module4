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

target_tri_ids = [45, 50, 158, 162, 163]

print(f"{'ID':>4} | {'BlobArea':>8} | {'DiffMax':>7} | {'DiffMean':>8} | {'mean_r':>7} | {'mean_p':>7} | {'std_r':>6} | {'grad_r':>7} | {'is_ink':>7}")
print("-" * 85)

for tri_id in target_tri_ids:
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

    # Gradient of Return
    gx = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r = np.sqrt(gx**2 + gy**2)

    hist_p, peaks_p, _ = compute_orientation_profile(clahe.apply(p_raw_gray), mask_p)
    hist_r, peaks_r, _ = compute_orientation_profile(clahe.apply(r_raw_gray), mask_p)
    ori_corr = float(np.corrcoef(hist_p, hist_r)[0, 1]) if (hist_p.std() > 1e-4 and hist_r.std() > 1e-4) else 1.0

    # Let's inspect each component
    for lbl in range(1, num_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        m = (lbls == lbl)
        mean_r = np.mean(r_raw_gray[m])
        mean_p = np.mean(p_raw_gray[m])
        std_r = np.std(r_raw_gray[m])
        grad_r = np.mean(mag_r[m])
        diff_max = np.max(diff[m])
        diff_mean = np.mean(diff[m])
        
        # Check ink stain criteria from export_debug_experiment_1_16.py:
        # if area >= 120 and mean_r <= 55 and mean_p >= 90 and std_r <= 12.0 and grad_r_m <= 40.0:
        is_ink = (area >= 120 and mean_r <= 55 and mean_p >= 90 and std_r <= 12.0 and grad_r <= 40.0)
        print(f"{tri_id:>4} | {area:>8} | {diff_max:>7.1f} | {diff_mean:>8.1f} | {mean_r:>7.1f} | {mean_p:>7.1f} | {std_r:>6.1f} | {grad_r:>7.1f} | {str(is_ink):>7}")
