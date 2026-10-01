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

target = np.array([1205.1, 1362.0])

def pt_in_tri(p, a, b, c):
    # Barycentric coordinates check
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

print("Checking Delaunay triangulation:")
found = []
for i, tri in enumerate(triangles):
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    if pt_in_tri(target, pts_r[0], pts_r[1], pts_r[2]):
        print(f"Target (1205.1, 1362.0) is INSIDE Triangle #{i}!")
        print(f"  Vertices: {pts_r.tolist()}")
        found.append(i)

if not found:
    print("Point is not inside any triangle! Checking minimum distance to all triangles:")
    min_d = 9999
    best_t = None
    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        dist = cv2.pointPolygonTest(pts_r.reshape((-1, 1, 2)), (float(target[0]), float(target[1])), True)
        if dist > 0:
            print(f"  OpenCV says INSIDE Triangle #{i}! dist={dist:.1f}")
            found.append(i)
        elif -dist < min_d:
            min_d = -dist
            best_t = i
    print(f"Nearest triangle is #{best_t} with distance {min_d:.1f}px")
