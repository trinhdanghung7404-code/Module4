import cv2, numpy as np, sys
sys.path.insert(0, 'v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder

p_img = cv2.imread(r'images\1.jpg')
r_img = cv2.imread(r'images\2.jpg')

seg = ObjectSegmenter()
p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
norm = ImageNormalizer()
r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
reg = ImageRegistration(max_keypoints=None)
reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
mb = MeshBuilder()
vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

# Crop around x=1000..1150, y=1000..1200
x0, x1 = 980, 1180
y0, y1 = 1000, 1200
c_p = p_img[y0:y1, x0:x1].copy()
c_r = r_img[y0:y1, x0:x1].copy()

# Draw triangles and their IDs on c_r
c_r_mesh = c_r.copy()
print("Triangles in the cheek/defect region:")
for idx, tri in enumerate(triangles):
    pts_r = np.array([vertices[i].return_xy for i in tri.vertex_indices], dtype=np.float32)
    center = pts_r.mean(axis=0)
    if x0 <= center[0] <= x1 and y0 <= center[1] <= y1:
        poly = (pts_r - [x0, y0]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(c_r_mesh, [poly], True, (0, 255, 0), 1)
        cx = int(center[0] - x0)
        cy = int(center[1] - y0)
        cv2.putText(c_r_mesh, str(idx), (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        print(f"  Tri #{idx}: R center ({center[0]:.1f}, {center[1]:.1f})")

vis = np.hstack([c_p, c_r, c_r_mesh])
cv2.imwrite('scratch/cheek_defect_region.jpg', vis)
print("Saved scratch/cheek_defect_region.jpg")
