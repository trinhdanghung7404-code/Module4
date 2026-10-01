import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import Vertex, Triangle, triangle_area, triangle_max_edge
from detailed_debug import canonical_triangle_patch, micro_align_patches
from scipy.spatial import Delaunay

prod_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "1.jpg"))
ret_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "16.jpg"))

seg = ObjectSegmenter()
p_seg = seg.segment(prod_bgr)
r_seg = seg.segment(ret_bgr)
norm = ImageNormalizer()
ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])
ret_ct = norm.color_transfer(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])

reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])

p_pts = reg_res["product_points"]
r_pts = reg_res["return_points"]
vertices = [Vertex(product_xy=tuple(p_pts[i]), return_xy=tuple(r_pts[i])) for i in range(len(p_pts))]

delaunay = Delaunay(p_pts)
triangles = []
for simplex in delaunay.simplices:
    tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
    # Using max_edge = 260.0 to eliminate the mesh hole over the user's uploaded spot!
    if triangle_area(vertices, tri) >= 50.0 and triangle_max_edge(vertices, tri) <= 260.0:
        triangles.append(tri)

print(f"Total triangles in mesh (max_edge=260): {len(triangles)}")

k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

ink_detected_triangles = []

for i, tri in enumerate(triangles):
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

    patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    valid = mask_p > 0

    p_vals = p_raw_gray[valid].astype(np.float32)
    r_vals = r_raw_gray[valid].astype(np.float32)

    med_p = np.median(p_vals)
    med_r = np.median(r_vals)
    mad_p = np.median(np.abs(p_vals - med_p)) * 1.4826
    mad_r = np.median(np.abs(r_vals - med_r)) * 1.4826
    scale = np.clip(mad_p / (mad_r + 1e-5), 0.7, 1.4)
    r_norm = (r_raw_gray.astype(np.float32) - med_r) * scale + med_p
    diff = np.abs(r_norm - p_raw_gray.astype(np.float32))

    # Ink spot condition:
    # 1. Return is pitch dark (< 55)
    # 2. Product was significantly brighter (> 85)
    # 3. Robust diff is substantial (> 35)
    ink_mask = (r_raw_gray < 55) & (p_raw_gray > 85) & (diff > 35) & valid
    ink_opened = cv2.morphologyEx(ink_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(ink_opened, connectivity=8)

    max_ink_area = max([stats[l, cv2.CC_STAT_AREA] for l in range(1, num_lbl)], default=0)

    if max_ink_area >= 60:
        ink_detected_triangles.append({
            "id": i,
            "ink_area": max_ink_area,
            "pts_r": pts_r
        })

print(f"\nTotal Triangles with Ink Spots detected: {len(ink_detected_triangles)}")
for t in ink_detected_triangles:
    cx = float(np.mean(t["pts_r"][:, 0]))
    cy = float(np.mean(t["pts_r"][:, 1]))
    print(f"  Tri #{t['id']:>4}: Ink Area = {t['ink_area']:>4}px at Return ({cx:.1f}, {cy:.1f})")
