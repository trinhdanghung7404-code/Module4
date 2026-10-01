import cv2
import numpy as np
import os
import sys
from scipy.spatial import Delaunay

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import Vertex, Triangle, triangle_area, triangle_max_edge

prod_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "1.jpg"))
ret_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "16.jpg"))

seg = ObjectSegmenter()
p_seg = seg.segment(prod_bgr)
r_seg = seg.segment(ret_bgr)
norm = ImageNormalizer()
ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])

p_pts = reg_res["product_points"]
r_pts = reg_res["return_points"]

delaunay = Delaunay(p_pts)
print(f"Total Delaunay simplices: {len(delaunay.simplices)}")

vertices = [Vertex(product_xy=tuple(p_pts[i]), return_xy=tuple(r_pts[i])) for i in range(len(p_pts))]

target_r = np.array([1205.1, 1362.0])

def pt_in_tri(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

print("\nChecking all simplices before filtering:")
for s_idx, simplex in enumerate(delaunay.simplices):
    tri = Triangle(vertex_indices=tuple(simplex), triangle_id=s_idx)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    if pt_in_tri(target_r, pts_r[0], pts_r[1], pts_r[2]):
        area = triangle_area(vertices, tri)
        max_edge = triangle_max_edge(vertices, tri)
        print(f"--> Target IS INSIDE SIMPLEX #{s_idx}!")
        print(f"    Vertices return_xy: {pts_r.tolist()}")
        print(f"    Product area: {area:.1f}, Max edge: {max_edge:.1f}")
        print(f"    Kept by filter (area >= 50 and edge <= 220): {area >= 50 and max_edge <= 220}")
