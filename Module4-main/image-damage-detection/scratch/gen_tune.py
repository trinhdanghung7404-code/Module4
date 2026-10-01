import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

with open('scratch/test_combined_final.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. contrast_th = 80.0 if is_shape_intact else 70.0
text = text.replace("contrast_th = 90.0 if is_shape_intact else 60.0", "contrast_th = 85.0 if is_shape_intact else 70.0")

# 2. In gating: blob_area >= 70 and z_l1 > 1.8
old_gating_cond = "(item[\"blob_area\"] >= 60 and item[\"z_l1\"] > 1.2)"
new_gating_cond = "(item[\"blob_area\"] >= 70 and item[\"z_l1\"] > 1.6)"
text = text.replace(old_gating_cond, new_gating_cond)

# 3. For crack: dark_crack >= 16.0 and ori_corr < 0.70 (crack must break pattern symmetry)
text = text.replace("(item[\"dark_crack\"] >= 15.0 and item[\"ori_corr\"] < 0.80)", "(item[\"dark_crack\"] >= 18.0 and item[\"ori_corr\"] < 0.70)")

with open('scratch/test_tune.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_tune.py")
