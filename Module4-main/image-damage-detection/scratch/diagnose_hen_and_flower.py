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

# Build mesh with max_edge = 260.0
triangles = []
for simplex in delaunay.simplices:
    tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
    if triangle_area(vertices, tri) >= 50.0 and triangle_max_edge(vertices, tri) <= 260.0:
        triangles.append(tri)

ret_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)
prod_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)

# Spot 3 (Hen): roi y:1500-1650, x:700-850
# In this ROI, ink is ret_gray < 50
m_hen = np.zeros_like(ret_gray, dtype=bool)
m_hen[1500:1650, 700:850] = (ret_gray[1500:1650, 700:850] < 50)

# Spot 4 (Flower): roi y:1080-1220, x:460-600
m_flower = np.zeros_like(ret_gray, dtype=bool)
m_flower[1080:1220, 460:600] = (ret_gray[1080:1220, 460:600] < 50)

print(f"Hen ink pixels: {np.sum(m_hen)}")
print(f"Flower ink pixels: {np.sum(m_flower)}")

def inspect_spot(spot_name, spot_mask):
    print(f"\n==========================================")
    print(f" Inspecting {spot_name}")
    print(f"==========================================")
    overlapping = []
    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
        tmask = np.zeros_like(ret_gray)
        cv2.fillPoly(tmask, [pts_r], 1)
        overlap = np.sum((tmask > 0) & spot_mask)
        if overlap > 50:
            overlapping.append((i, overlap, tri, pts_r))
            print(f"  Triangle #{i}: ink overlap = {overlap}px")
            
    for tid, ov, tri, pts_r in overlapping:
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        valid = mask_p > 0

        p_vals = p_raw[valid].astype(np.float32)
        r_vals = r_raw[valid].astype(np.float32)

        med_p = np.median(p_vals)
        med_r = np.median(r_vals)
        mad_p = np.median(np.abs(p_vals - med_p)) * 1.4826
        mad_r = np.median(np.abs(r_vals - med_r)) * 1.4826
        scale = np.clip(mad_p / (mad_r + 1e-5), 0.7, 1.4)
        r_norm = (r_raw.astype(np.float32) - med_r) * scale + med_p
        diff = np.abs(r_norm - p_raw.astype(np.float32))
        diff[~valid] = 0.0

        # Check ink pixels in patch (r_raw < 50)
        ink_in_patch = (r_raw < 50) & valid
        print(f"  Tri #{tid} in patch:")
        print(f"    Total valid px: {np.sum(valid)}")
        print(f"    Ink pixels (r < 50): {np.sum(ink_in_patch)}")
        if np.sum(ink_in_patch) > 0:
            print(f"    mean p_raw on ink: {np.mean(p_raw[ink_in_patch]):.1f}")
            print(f"    mean r_raw on ink: {np.mean(r_raw[ink_in_patch]):.1f}")
            print(f"    mean diff on ink:  {np.mean(diff[ink_in_patch]):.1f}")
            print(f"    max diff on ink:   {np.max(diff[ink_in_patch]):.1f}")

inspect_spot("SPOT 3: YELLOW HEN", m_hen)
inspect_spot("SPOT 4: PURPLE FLOWER", m_flower)
