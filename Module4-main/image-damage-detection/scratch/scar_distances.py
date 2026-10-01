import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder

p_img = cv2.imread(r'images\test_nobg.png'); r_img = cv2.imread(r'images\test_nobg - scar.png')
seg = ObjectSegmenter()
p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

scar_ids = [22, 98, 121, 123, 124, 341, 431, 432, 433, 454]
centers = {}
for tid in scar_ids:
    tri = triangles[tid]
    pts = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
    centers[tid] = pts.mean(axis=0)

print("Scar triangle centers:")
for tid in scar_ids:
    c = centers[tid]
    print(f"Tri #{tid:3d}: ({c[0]:.1f}, {c[1]:.1f})")

print("\nDistances from Tri 98:")
c98 = centers[98]
for tid in scar_ids:
    d = np.linalg.norm(c98 - centers[tid])
    print(f"  To Tri #{tid:3d}: {d:.1f} px")
