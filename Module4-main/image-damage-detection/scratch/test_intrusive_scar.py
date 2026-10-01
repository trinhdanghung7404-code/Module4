import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def measure_linear_stroke(contours, min_len=8.0, min_aspect=1.5):
    max_len = 0.0
    for c in contours:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        length, width = max(rect[1]), min(rect[1])
        aspect = length / max(width, 0.5)
        if length >= min_len and (aspect >= min_aspect or length >= 12.0):
            if length > max_len:
                max_len = length
    return max_len

s1 = 'images/test_nobg.png'
s2 = 'images/test_nobg - scar.png'
im1 = cv2.imread(s1)
im2 = cv2.imread(s2)

segmenter = ObjectSegmenter()
p_seg = segmenter.segment(im1)
r_seg = segmenter.segment(im2)
normalizer = ImageNormalizer()
return_normalized = normalizer.normalize(im2, im1, r_seg['mask'], p_seg['mask'])
return_ct = normalizer.color_transfer(im2, im1, r_seg['mask'], p_seg['mask'])
registrator = ImageRegistration(max_keypoints=None)
reg_result = registrator.register(im1, return_normalized, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_result['product_points'], reg_result['return_points'], np.arange(len(reg_result['product_points'])))

k_erode_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

scar_ids = [341, 432, 433, 438, 439, 446, 454]
for tid in scar_ids:
    tri = triangles[tid]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(im1, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(return_ct, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)
    valid = mask_p > 0
    inner_triangle = cv2.erode(mask_p, k_erode_5, iterations=2) > 0

    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

    p_canny = cv2.Canny(p_raw_gray, 40, 120)
    r_canny = cv2.Canny(r_raw_gray, 40, 120)
    p_canny[~valid] = 0
    r_canny[~valid] = 0

    dist_p = cv2.distanceTransform(cv2.bitwise_not(p_canny), cv2.DIST_L2, 3)
    gx_r = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy_r = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r = np.sqrt(gx_r**2 + gy_r**2)

    # Intrusive check
    intrusive = (r_canny > 0) & (dist_p > 3.0) & (mag_r >= 40.0) & inner_triangle
    cnts, _ = cv2.findContours(intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    stroke_len = measure_linear_stroke(cnts, min_len=8.0, min_aspect=1.5)
    print(f"Scar Tri #{tid:3d}: intrusive stroke length = {stroke_len:.1f} px, r_canny_px = {np.sum(r_canny[inner_triangle]>0)}")
