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

crop_path = r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\.user_uploaded\media_1790154640162.png"
ret_path = os.path.join(PROJECT_ROOT, "images", "16.jpg")
prod_path = os.path.join(PROJECT_ROOT, "images", "1.jpg")

crop_img = cv2.imread(crop_path)
ret_img = cv2.imread(ret_path)
prod_img = cv2.imread(prod_path)

print(f"Crop shape: {crop_img.shape}")
print(f"Return shape: {ret_img.shape}")

# Find where this crop is in 16.jpg
# The crop has green mesh lines overlaid on it, so let's match grayscale or threshold
# Or let's find the black ink spots on 16.jpg directly
ret_gray = cv2.cvtColor(ret_img, cv2.COLOR_BGR2GRAY)

# Register images to get the mesh triangles
seg = ObjectSegmenter()
p_seg = seg.segment(prod_img)
r_seg = seg.segment(ret_img)

norm = ImageNormalizer()
ret_norm = norm.normalize(ret_img, prod_img, r_seg["mask"], p_seg["mask"])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_img, ret_norm, p_seg["mask"], r_seg["mask"])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res["product_points"], reg_res["return_points"], np.arange(len(reg_res["product_points"])))

# Let's find all black ink connected components on 16.jpg
# Black ink is very dark (< 50) on 16.jpg
ink_mask = (ret_gray < 50) & (r_seg["mask"] > 0)
# Clean small noise
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ink_clean = cv2.morphologyEx(ink_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_open)

num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(ink_clean)
print(f"\nFound {num_lbl - 1} dark ink blobs on 16.jpg:")
for lbl in range(1, num_lbl):
    area = stats[lbl, cv2.CC_STAT_AREA]
    cx, cy = centroids[lbl]
    print(f"  Blob #{lbl}: Area = {area}px at centroid ({cx:.1f}, {cy:.1f})")

    # Find which triangles intersect this blob
    intersecting_tris = []
    blob_m = (lbls == lbl).astype(np.uint8)
    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
        tri_mask = np.zeros_like(ret_gray)
        cv2.fillPoly(tri_mask, [pts_r], 1)
        overlap = np.sum((tri_mask > 0) & (blob_m > 0))
        if overlap > 20:
            intersecting_tris.append((i, overlap))
    print(f"    Intersecting triangles: {intersecting_tris}")
