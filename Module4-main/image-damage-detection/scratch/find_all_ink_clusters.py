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
from scipy.spatial import Delaunay

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
vertices = [Vertex(product_xy=tuple(p_pts[i]), return_xy=tuple(r_pts[i])) for i in range(len(p_pts))]

delaunay = Delaunay(p_pts)
triangles = []
for simplex in delaunay.simplices:
    tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
    if triangle_area(vertices, tri) >= 50.0 and triangle_max_edge(vertices, tri) <= 260.0:
        triangles.append(tri)

ret_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)
prod_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)

# Find all triangles that contain any pitch-black pixels (< 45) that are bright in product (> 80):
anom_black = (ret_gray < 45) & (prod_gray > 80) & (r_seg["mask"] > 0)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
anom_black = cv2.morphologyEx(anom_black.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)

num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(anom_black)
print(f"Total isolated dark ink clusters: {num_lbl - 1}")

for l in range(1, num_lbl):
    area = stats[l, cv2.CC_STAT_AREA]
    if area >= 300:
        cx, cy = centroids[l]
        m = (lbls == l)
        # Find which triangles intersect this ink cluster
        covering_triangles = []
        for i, tri in enumerate(triangles):
            pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
            tmask = np.zeros_like(ret_gray)
            cv2.fillPoly(tmask, [pts_r], 1)
            overlap = np.sum((tmask > 0) & m)
            if overlap >= 50:
                covering_triangles.append((i, overlap))
        print(f"\nInk Cluster #{l}: Area = {area}px at ({cx:.1f}, {cy:.1f})")
        print(f"  Intersecting Triangles: {covering_triangles}")
