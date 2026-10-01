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
from detailed_debug import canonical_triangle_patch, micro_align_patches

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

flagged_candidates = [4, 11, 22, 35, 46, 47, 61, 77, 80, 115, 201, 234, 300, 355, 386, 409, 419, 463, 505, 579, 618, 633, 664, 665, 684, 685, 688, 692, 713]

print(f"{'ID':>4} | {'MaxBlob':>7} | {'R_grad_in_blob':>14} | {'R_std_in_blob':>13} | {'IsInkVoid':>9} | {'Min R in blob':>13}")
print("-" * 75)

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

    bin_diff = (diff > 35.0).astype(np.uint8) * 255
    opened = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    # Gradient of Return
    gx = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r = np.sqrt(gx**2 + gy**2)

    # Find largest blob
    best_lbl = None
    best_area = 0
    for l in range(1, num_lbl):
        area = stats[l, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_lbl = l

    if best_lbl is not None and best_area >= 100:
        m = (lbls == best_lbl)
        grad_in_blob = float(np.mean(mag_r[m]))
        std_in_blob = float(np.std(r_raw_gray[m]))
        min_r_in_blob = float(np.min(r_raw_gray[m]))
        mean_r_in_blob = float(np.mean(r_raw_gray[m]))
        mean_p_in_blob = float(np.mean(p_raw_gray[m]))
        # Is ink void: Return is dark (< 60) while Product was bright (> 90), or Return is textureless void
        is_ink = (mean_r_in_blob < 60 and mean_p_in_blob > 90 and best_area >= 100)
        is_ink_str = "YES" if is_ink else "NO"
        print(f"{tri_id:>4} | {best_area:>7} | {grad_in_blob:>14.1f} | {std_in_blob:>13.1f} | {is_ink_str:>9} | {min_r_in_blob:>13.1f} (r_mean={mean_r_in_blob:.1f}, p_mean={mean_p_in_blob:.1f})")
