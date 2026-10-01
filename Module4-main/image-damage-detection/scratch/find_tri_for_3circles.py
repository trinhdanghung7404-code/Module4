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

# Spot center is at (1205.1, 1362.0)
target_pt = np.array([1205.1, 1362.0], dtype=np.float32)

print(f"Finding triangles covering ({target_pt[0]}, {target_pt[1]}):")
for i, tri in enumerate(triangles):
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    # Check if point is inside triangle using pointPolygonTest
    dist = cv2.pointPolygonTest(pts_r, (float(target_pt[0]), float(target_pt[1])), False)
    if dist >= 0:
        print(f"  --> TRIANGLE #{i} CONTAINS THE SPOT CENTER!")
        print(f"      Return Vertices: {pts_r.tolist()}")
    else:
        # Also check if triangle is very close (within 40px)
        dist_exact = cv2.pointPolygonTest(pts_r, (float(target_pt[0]), float(target_pt[1])), True)
        if dist_exact >= -40:
            print(f"  Triangle #{i} near spot (dist={dist_exact:.1f}px)")
