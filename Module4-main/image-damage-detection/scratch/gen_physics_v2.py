import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

# Load scratch/test_physics_gating.py
with open('scratch/test_physics_gating.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Update intrusive_cand in test_physics_gating.py
old_intrusive = """        intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 6.0) & (grad_p_raw < 25.0) & (mag_r_raw >= 70.0) & (~glare_px) & inner_triangle
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0):
                anom_intrusive[y, x] = True

        cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = measure_linear_stroke(cnts_intrusive, min_len=12.0, min_aspect=2.0)"""

new_intrusive = """        # Chi loai bo neu ca 2 deu trang loa (flash chieu vao cho von da trang sang)
        saturated_flash = (r_raw_gray >= 248) & (p_raw_gray >= 200)
        intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 3.5) & (mag_r_raw >= 38.0) & (~saturated_flash) & inner_triangle
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0) or dist_to_p_all[y, x] > 5.5:
                anom_intrusive[y, x] = True

        cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = measure_linear_stroke(cnts_intrusive, min_len=10.0, min_aspect=1.5)"""

text = text.replace(old_intrusive, new_intrusive)

with open('scratch/test_physics_gating_v2.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_physics_gating_v2.py")
