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
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])

p_pts = reg_res["product_points"]
r_pts = reg_res["return_points"]
vertices = [Vertex(product_xy=tuple(p_pts[i]), return_xy=tuple(r_pts[i])) for i in range(len(p_pts))]

delaunay = Delaunay(p_pts)

# Find all ink spots on 16.jpg:
# Points where ret is pitch black (< 45) but prod was light (> 80)
# (filter out rooster tail where both are dark)
ret_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)
prod_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)

# Check all simplices with max_edge up to 280
for max_edge in [220.0, 260.0, 280.0]:
    triangles = []
    for simplex in delaunay.simplices:
        tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
        if triangle_area(vertices, tri) >= 50.0 and triangle_max_edge(vertices, tri) <= max_edge:
            triangles.append(tri)
    
    # Check coverage of each ink spot:
    # Known ink spots centers on 16.jpg:
    spots = [
        ("Spot 1: Rooster Breast (Tri 685)", (1020, 1140)),
        ("Spot 2: Rooster Back 3-circles (User image)", (1205, 1362)),
        ("Spot 3: Yellow Hen body", (765, 1563)),
        ("Spot 4: Purple Flower left", (530, 1143)),
    ]
    print(f"\n--- MAX_TRIANGLE_EDGE = {max_edge} (Total triangles: {len(triangles)}) ---")
    for name, (sx, sy) in spots:
        covered = False
        covering_id = None
        for i, tri in enumerate(triangles):
            pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
            d = cv2.pointPolygonTest(pts_r.reshape((-1, 1, 2)), (float(sx), float(sy)), False)
            if d >= 0:
                covered = True
                covering_id = i
                break
        status = f"COVERED by Tri #{covering_id}" if covered else "MISSING (MESH HOLE!)"
        print(f"  {name}: {status}")
