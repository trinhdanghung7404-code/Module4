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
triangles = []
for simplex in delaunay.simplices:
    tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
    if triangle_area(vertices, tri) >= 50.0 and triangle_max_edge(vertices, tri) <= 260.0:
        triangles.append(tri)

k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
spots = [
    ("Spot 1: Rooster Breast", (1020, 1140)),
    ("Spot 2: Rooster Back 3-circles (User image)", (1205, 1362)),
    ("Spot 3: Yellow Hen body", (765, 1563)),
    ("Spot 4: Purple Flower left", (530, 1143)),
]

print("EVALUATING ALL 4 INK SPOTS WITH ROBUST MEDIAN + VOID DETECTOR:")
for name, (sx, sy) in spots:
    covering_tri = None
    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        d = cv2.pointPolygonTest(pts_r.reshape((-1, 1, 2)), (float(sx), float(sy)), False)
        if d >= 0:
            covering_tri = (i, tri, pts_r)
            break
    
    if covering_tri is None:
        print(f"\n{name}: NOT COVERED IN MESH!")
        continue
    
    tid, tri, pts_r = covering_tri
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(prod_bgr, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(ret_ct, pts_r, target_size=72)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    valid = mask_p > 0

    p_vals = p_raw_gray[valid].astype(np.float32)
    r_vals = r_raw_gray[valid].astype(np.float32)

    med_p = np.median(p_vals)
    med_r = np.median(r_vals)
    mad_p = np.median(np.abs(p_vals - med_p)) * 1.4826
    mad_r = np.median(np.abs(r_vals - med_r)) * 1.4826
    scale = np.clip(mad_p / (mad_r + 1e-5), 0.7, 1.4)
    r_norm = (r_raw_gray.astype(np.float32) - med_r) * scale + med_p

    diff = np.abs(r_norm - p_raw_gray.astype(np.float32))
    diff[~valid] = 0.0

    bin_diff = (diff > 30.0).astype(np.uint8) * 255
    opened = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    gx = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r = np.sqrt(gx**2 + gy**2)

    is_ink_stain = False
    blobs_info = []
    for lbl in range(1, num_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        m = (lbls == lbl)
        mean_r = np.mean(r_raw_gray[m])
        mean_p = np.mean(p_raw_gray[m])
        std_r = np.std(r_raw_gray[m])
        grad_r = np.mean(mag_r[m])
        max_d = np.max(diff[m])
        blobs_info.append((area, mean_r, mean_p, std_r, grad_r, max_d))
        # Ink stain: dark puddle on previously non-black area
        if area >= 80 and mean_r <= 65 and mean_p >= 75 and std_r <= 15.0 and grad_r <= 45.0:
            is_ink_stain = True

    print(f"\n{name} (Triangle #{tid}):")
    print(f"  Detected as Ink Stain: {is_ink_stain}")
    for b in blobs_info:
        if b[0] >= 50:
            print(f"    Blob: area={b[0]}px, mean_r={b[1]:.1f}, mean_p={b[2]:.1f}, std_r={b[3]:.1f}, grad_r={b[4]:.1f}, max_diff={b[5]:.1f}")
