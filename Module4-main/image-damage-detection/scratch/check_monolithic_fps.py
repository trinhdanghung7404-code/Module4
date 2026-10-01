import os
import sys

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
scratch_dir = os.path.join(PROJECT_ROOT, "scratch")

# Let's inspect the 75 triangles from test_monolithic_1_16.py
# In test_monolithic_1_16.py:
# is_monolithic_dark = bool(raw_neg_px >= 60 and raw_dipole_ratio < 0.20)
# is_monolithic_bright = bool(raw_pos_px >= 60 and raw_dipole_ratio < 0.20)
# if is_monolithic: contrast_th = 45.0
# and: (item["blob_area"] >= 60 and z_l1 > 0.8)

# Why were raw_neg_px >= 60 or raw_pos_px >= 60 triggered in 75 triangles?
# Let's check raw_diff = r_raw_gray - p_raw_gray
# raw_pos_px = np.sum((opened_raw > 0) & (raw_diff > 35.0))
# raw_neg_px = np.sum((opened_raw > 0) & (raw_diff < -35.0))
