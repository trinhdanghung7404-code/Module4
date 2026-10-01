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
from mesh.mesh_builder import MeshBuilder

prod_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "1.jpg"))
ret_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "16.jpg"))

seg = ObjectSegmenter()
p_seg = seg.segment(prod_bgr)
r_seg = seg.segment(ret_bgr)
norm = ImageNormalizer()
ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res["product_points"], reg_res["return_points"], np.arange(len(reg_res["product_points"])))

# Crop bbox on 16.jpg:
# x=1085.6, y=1263.8, w=246.7, h=200.1
bx, by, bw, bh = 1085, 1263, 247, 200

# Inside this bbox, find where the black mark is:
ret_roi = ret_bgr[by:by+bh, bx:bx+bw]
roi_gray = cv2.cvtColor(ret_roi, cv2.COLOR_BGR2GRAY)
dark_ink_local = (roi_gray < 50)

# Full mask of ink in this ROI:
ink_full = np.zeros(ret_bgr.shape[:2], dtype=bool)
ink_full[by:by+bh, bx:bx+bw] = dark_ink_local

print(f"Total dark ink pixels in this spot: {np.sum(dark_ink_local)}")

print("\nTriangles intersecting this ink spot:")
overlapping_tris = []
for i, tri in enumerate(triangles):
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
    tri_mask = np.zeros(ret_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(tri_mask, [pts_r], 1)
    overlap = np.sum((tri_mask > 0) & ink_full)
    if overlap > 0:
        overlapping_tris.append((i, overlap))
        print(f"  Tri #{i:>4}: overlap = {overlap}px, vertices = {[v for v in tri.vertex_indices]}")

print(f"\nAll intersecting triangle IDs: {[t[0] for t in overlapping_tris]}")
