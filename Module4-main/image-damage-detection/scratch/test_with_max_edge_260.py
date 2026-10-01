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
from detailed_debug import canonical_triangle_patch, micro_align_patches, compute_orientation_profile
from scipy.spatial import Delaunay
from skimage.metrics import structural_similarity as ssim

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

# Test with MAX_TRIANGLE_EDGE = 260.0
delaunay = Delaunay(p_pts)
triangles = []
for simplex in delaunay.simplices:
    tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
    if triangle_area(vertices, tri) >= 50.0 and triangle_max_edge(vertices, tri) <= 260.0:
        triangles.append(tri)

print(f"Mesh built with max_edge=260: {len(triangles)} triangles (was 725 with 220)")

# Check if target (1205.1, 1362.0) is covered now
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

target_tri_id = None
for i, tri in enumerate(triangles):
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
    if pt_in_tri(target_r, pts_r[0], pts_r[1], pts_r[2]):
        print(f"SUCCESS: Target point is covered by Triangle #{i}!")
        print(f"  Vertices: {pts_r.tolist()}")
        target_tri_id = i
        break

# Now inspect this target triangle with our detector!
if target_tri_id is not None:
    tri = triangles[target_tri_id]
    pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

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

    k_open_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_diff = (diff > 30.0).astype(np.uint8) * 255
    opened = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    # Check ink stain
    gx = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r = np.sqrt(gx**2 + gy**2)

    is_ink_stain = False
    max_solid_blob = 0
    print(f"\nComponents inside Triangle #{target_tri_id}:")
    for lbl in range(1, num_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        m = (lbls == lbl)
        mean_r = np.mean(r_raw_gray[m])
        mean_p = np.mean(p_raw_gray[m])
        std_r = np.std(r_raw_gray[m])
        grad_r = np.mean(mag_r[m])
        max_d = np.max(diff[m])
        print(f"  Comp #{lbl}: area={area}px, max_diff={max_d:.1f}, mean_r={mean_r:.1f}, mean_p={mean_p:.1f}, std_r={std_r:.1f}, grad_r={grad_r:.1f}")
        if area >= 100 and mean_r <= 65 and mean_p >= 85 and std_r <= 15.0 and grad_r <= 45.0:
            is_ink_stain = True
            print(f"    --> MATCHES INK STAIN DEFECT!")

    print(f"\nFinal ink stain detection for Triangle #{target_tri_id}: {is_ink_stain}")
