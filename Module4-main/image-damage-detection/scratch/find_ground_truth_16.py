import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
V2_DIR = os.path.join(PROJECT_ROOT, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder

p1_path = os.path.join(PROJECT_ROOT, "images", "1.jpg")
p16_path = os.path.join(PROJECT_ROOT, "images", "16.jpg")

prod_bgr = cv2.imread(p1_path)
ret_bgr = cv2.imread(p16_path)

seg = ObjectSegmenter()
p_seg = seg.segment(prod_bgr)
r_seg = seg.segment(ret_bgr)

norm = ImageNormalizer()
ret_norm = norm.normalize(ret_bgr, prod_bgr, r_seg["mask"], p_seg["mask"])

reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(prod_bgr, ret_norm, p_seg["mask"], r_seg["mask"])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res["product_points"], reg_res["return_points"], np.arange(len(reg_res["product_points"])))

# Ground truth black ink detection directly on 16.jpg:
# In 16.jpg, black ink has BGR ~ (0-50, 0-50, 0-50) where 1.jpg was much brighter
# Or blue lines: blue channel >> red channel
r_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)
p_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)

# Warping or looking at return coordinates directly:
# Any triangle whose return coordinates contain black ink (r_gray < 55) or blue marker
ink_triangles = []
for i, tri in enumerate(triangles):
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
    # Mask for this triangle in return image
    mask = np.zeros(r_gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts_r], 255)
    r_pts = ret_bgr[mask > 0]
    if len(r_pts) == 0:
        continue
    
    # Black ink in Return: very low intensity
    r_brightness = np.mean(r_pts, axis=1)
    dark_ink_px = np.sum(r_brightness < 45)
    
    # Blue marker in Return: B > 120 and B - R > 50
    blue_marker_px = np.sum((r_pts[:, 0] > 100) & (r_pts[:, 0].astype(int) - r_pts[:, 2].astype(int) > 40))

    if dark_ink_px >= 50 or blue_marker_px >= 30:
        ink_triangles.append({
            "id": i,
            "dark_px": int(dark_ink_px),
            "blue_px": int(blue_marker_px),
            "total_px": len(r_pts)
        })

print(f"Total Triangles touching Black Ink or Blue Marker on 16.jpg: {len(ink_triangles)}")
for item in ink_triangles:
    print(f"  Tri #{item['id']:>4}: dark_px={item['dark_px']:>4}, blue_px={item['blue_px']:>4}, total={item['total_px']:>4}")
