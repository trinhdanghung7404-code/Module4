import cv2
import numpy as np
import os

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
crop_path = r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\.user_uploaded\media_1790154640162.png"
ret_path = os.path.join(PROJECT_ROOT, "images", "16.jpg")
marked_path = r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\exp_1_16_return_marked.jpg"

crop_bgr = cv2.imread(crop_path)
ret_bgr = cv2.imread(ret_path)
marked_bgr = cv2.imread(marked_path)

# Remove green overlay from crop_bgr for clean matching:
# Where green channel is high and R, B are low:
is_green_line = (crop_bgr[:, :, 1] > 180) & (crop_bgr[:, :, 0] < 100) & (crop_bgr[:, :, 2] < 100)

# We can match on grayscale where not green line
crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
ret_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)

# Since user took a screenshot or crop from the marked image or screen:
# Let's check scale!
# Let's match template at multiple scales
best_val = -1
best_loc = None
best_scale = 1.0

# Notice the crop size is 73x90
# On 16.jpg (2560x1920), the vase is ~1800 px wide.
# A small patch of feathers is around 200-400 px on full image.
# So the crop was likely resized/screenshot from a display!
for s in np.linspace(1.5, 6.0, 30):
    w_s = int(crop_gray.shape[1] * s)
    h_s = int(crop_gray.shape[0] * s)
    if w_s >= ret_gray.shape[1] or h_s >= ret_gray.shape[0]:
        continue
    # Instead, downscale ret_gray to match crop scale:
    scale_down = 1.0 / s
    ret_small = cv2.resize(ret_gray, (0, 0), fx=scale_down, fy=scale_down)
    if ret_small.shape[0] < crop_gray.shape[0] or ret_small.shape[1] < crop_gray.shape[1]:
        continue
    res = cv2.matchTemplate(ret_small, crop_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_scale = scale_down

print(f"Best match score: {best_val:.3f} at scale {best_scale:.3f}")
# Map back to full 16.jpg coordinates
fx, fy = best_loc[0] / best_scale, best_loc[1] / best_scale
fw, fh = crop_gray.shape[1] / best_scale, crop_gray.shape[0] / best_scale
print(f"Full image bounding box: x={fx:.1f}, y={fy:.1f}, w={fw:.1f}, h={fh:.1f}")

# Also check on marked_bgr if it was screenshotted from marked_bgr
res_m = cv2.matchTemplate(cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2GRAY), crop_gray, cv2.TM_CCOEFF_NORMED)
_, max_vm, _, max_lm = cv2.minMaxLoc(res_m)
print(f"Match on marked_bgr directly: {max_vm:.3f} at {max_lm}")
