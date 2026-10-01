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

print(f"Product shape: {prod_bgr.shape}")
print(f"Return shape: {ret_bgr.shape}")
ret_pts = np.array([v.return_xy for v in vertices])
print(f"Return points X range: [{ret_pts[:, 0].min():.1f}, {ret_pts[:, 0].max():.1f}]")
print(f"Return points Y range: [{ret_pts[:, 1].min():.1f}, {ret_pts[:, 1].max():.1f}]")

# Check distance from all return points to (1205.1, 1362.0)
target = np.array([1205.1, 1362.0])
dists = np.linalg.norm(ret_pts - target, axis=1)
nearest_idx = np.argsort(dists)[:5]
print("5 nearest vertices to (1205.1, 1362.0):")
for idx in nearest_idx:
    print(f"  Vertex {idx}: {ret_pts[idx].tolist()}, dist = {dists[idx]:.1f}px")

# Check triangles containing any of these vertices
print("\nTriangles containing these vertices:")
cand_tris = set()
for tri_idx, tri in enumerate(triangles):
    for v in tri.vertex_indices:
        if v in nearest_idx:
            cand_tris.add(tri_idx)

print(f"Found {len(cand_tris)} candidate triangles: {sorted(cand_tris)}")
