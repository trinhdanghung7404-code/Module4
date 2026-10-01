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
from detailed_debug import canonical_triangle_patch, micro_align_patches, compute_orientation_profile, is_angle_matching_any

def diagnose_685():
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
    inliers_p = reg_res["product_points"]
    inliers_r = reg_res["return_points"]

    mb = MeshBuilder()
    vertices, triangles = mb.build(inliers_p, inliers_r, np.arange(len(inliers_p)))

    tri = triangles[685]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

    p_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(ret_ct, cv2.COLOR_BGR2GRAY)
    p_med = cv2.medianBlur(p_gray, 3)
    r_med = cv2.medianBlur(r_gray, 3)
    p_edge_denoised = cv2.Canny(p_med, 60, 160)
    r_edge_denoised = cv2.Canny(r_med, 60, 160)

    patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
    patch_pe, _ = canonical_triangle_patch(p_edge_denoised, pts_p, target_size=72)
    patch_re, _ = canonical_triangle_patch(r_edge_denoised, pts_r, target_size=72)
    patch_pe = (patch_pe > 127).astype(np.uint8) * 255
    patch_re = (patch_re > 127).astype(np.uint8) * 255

    patch_r_aligned, patch_re_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2, extra_patch=patch_re)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    p_c_gray = clahe.apply(p_raw_gray)
    r_c_gray = clahe.apply(r_raw_gray)

    valid = mask_p > 0
    glare_px = (r_c_gray > 240) | (p_c_gray > 240)

    p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
    r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)
    mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
    mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
    scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
    r_norm_loc = (r_raw_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc

    diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
    diff_loc[~valid] = 0.0
    diff_loc[glare_px] = 0.0

    raw_diff = np.abs(r_raw_gray.astype(np.float32) - p_raw_gray.astype(np.float32))
    raw_diff[~valid] = 0.0

    print("--- DIAGNOSTIC TRI #685 ---")
    print(f"Product raw gray min/max/mean: {np.min(p_vals):.1f} / {np.max(p_vals):.1f} / {mu_p_loc:.1f}")
    print(f"Return raw gray min/max/mean:  {np.min(r_vals):.1f} / {np.max(r_vals):.1f} / {mu_r_loc:.1f}")
    print(f"scale_loc: {scale_loc:.3f}")
    print(f"diff_loc max: {np.max(diff_loc):.1f}, mean: {np.mean(diff_loc[valid]):.1f}")
    print(f"raw_diff max: {np.max(raw_diff):.1f}, mean: {np.mean(raw_diff[valid]):.1f}")

    # Inspect the dark ink spot pixels specifically
    # Find pixels where Return is very dark compared to Product
    dark_defect = (r_raw_gray < 50) & (p_raw_gray > 100) & valid
    print(f"Number of dark defect pixels (r < 50 and p > 100): {np.sum(dark_defect)}")
    if np.sum(dark_defect) > 0:
        print(f"In dark defect region:")
        print(f"  mean p_raw: {np.mean(p_raw_gray[dark_defect]):.1f}")
        print(f"  mean r_raw: {np.mean(r_raw_gray[dark_defect]):.1f}")
        print(f"  mean r_norm: {np.mean(r_norm_loc[dark_defect]):.1f}")
        print(f"  mean diff_loc: {np.mean(diff_loc[dark_defect]):.1f}")
        print(f"  mean raw_diff: {np.mean(raw_diff[dark_defect]):.1f}")

    # Check Canny edges of patch
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    p_what = cv2.morphologyEx(p_c_gray, cv2.MORPH_TOPHAT, k_morph)
    p_bhat = cv2.morphologyEx(p_c_gray, cv2.MORPH_BLACKHAT, k_morph)
    p_c_edge = patch_pe.copy()
    p_c_edge[~valid] = 0
    p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
    dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

    r_raw_edge = cv2.Canny(r_raw_gray, 40, 120)
    r_raw_edge[~valid] = 0
    grad_p_raw = np.sqrt(cv2.Sobel(p_raw_gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw_gray, cv2.CV_32F, 0, 1)**2)
    gx_r_raw = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy_r_raw = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r_raw = np.sqrt(gx_r_raw**2 + gy_r_raw**2)
    ang_r_raw = (np.arctan2(gy_r_raw, gx_r_raw) * 180.0 / np.pi) % 180.0

    hist_p, peaks_p, coh_p = compute_orientation_profile(p_c_gray, mask_p)
    h_m, w_m = 72, 72
    Y_m, X_m = np.ogrid[:h_m, :w_m]
    inner_triangle = (X_m >= 4) & (Y_m >= 4) & (X_m + Y_m <= 66)

    intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 6.0) & (grad_p_raw < 25.0) & (mag_r_raw >= 70.0) & (~glare_px) & inner_triangle
    print(f"Intrusive candidate edge pixels: {np.sum(intrusive_cand)}")
    if np.sum(r_raw_edge > 0) > 0:
        edge_pts = (r_raw_edge > 0) & inner_triangle
        print(f"r_raw_edge in inner_triangle: {np.sum(edge_pts)}")
        print(f"  dist_to_p_all on r_edge: min={np.min(dist_to_p_all[edge_pts]):.1f}, max={np.max(dist_to_p_all[edge_pts]):.1f}, mean={np.mean(dist_to_p_all[edge_pts]):.1f}")
        print(f"  grad_p_raw on r_edge: min={np.min(grad_p_raw[edge_pts]):.1f}, max={np.max(grad_p_raw[edge_pts]):.1f}, mean={np.mean(grad_p_raw[edge_pts]):.1f}")
        print(f"  mag_r_raw on r_edge: min={np.min(mag_r_raw[edge_pts]):.1f}, max={np.max(mag_r_raw[edge_pts]):.1f}, mean={np.mean(mag_r_raw[edge_pts]):.1f}")

if __name__ == "__main__":
    diagnose_685()
