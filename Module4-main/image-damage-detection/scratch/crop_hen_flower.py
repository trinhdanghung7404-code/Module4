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

# In 16.jpg, let's locate:
# 1. Spot 3: Black mark on yellow hen
# 2. Spot 4: Black mark on purple flower
# On 16.jpg:
# Rooster chest: (1020, 1140) -> Tri 685 & 386 (DETECTED)
# Rooster back: (1205, 1362) -> Missed because of mesh hole (max_edge=228 > 220)
# Yellow hen: where is the black mark on yellow hen?
# Let's crop around (765, 1563) and (530, 1143)
cv2.imwrite(r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\crop_hen.jpg", ret_bgr[1500:1650, 700:850])
cv2.imwrite(r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\crop_flower.jpg", ret_bgr[1080:1220, 460:600])

print("Saved crop_hen.jpg and crop_flower.jpg")
