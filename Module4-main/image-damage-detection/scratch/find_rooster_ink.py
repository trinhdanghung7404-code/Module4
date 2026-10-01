import cv2
import numpy as np
import os

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
ret_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "16.jpg"))
prod_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "1.jpg"))

# Look at 16.jpg:
# Where is the rooster?
# Rooster is in the upper right.
# Hen is sitting in the center.
# The rooster's lower back / wing has grey scalloped feathers with white borders, right next to the hen's yellow feathers!
# Let's inspect the coordinates of 16.jpg around:
# Y between 1000 and 1600, X between 1000 and 1500
roi = ret_bgr[1000:1600, 1000:1500]
cv2.imwrite(r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\roi_rooster_body.jpg", roi)

# Let's find black pixels (R < 35, G < 35, B < 35) in this ROI:
roi_black = (roi[:, :, 0] < 35) & (roi[:, :, 1] < 35) & (roi[:, :, 2] < 35)
num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(roi_black.astype(np.uint8))
print(f"Black components in ROI (y: 1000-1600, x: 1000-1500):")
for l in range(1, num_lbl):
    a = stats[l, cv2.CC_STAT_AREA]
    if a > 200:
        cx, cy = centroids[l]
        abs_x = 1000 + cx
        abs_y = 1000 + cy
        w = stats[l, cv2.CC_STAT_WIDTH]
        h = stats[l, cv2.CC_STAT_HEIGHT]
        print(f"  Component #{l}: Area={a}px, Center=({abs_x:.1f}, {abs_y:.1f}), w={w}, h={h}")
        # Crop around this component
        crop = ret_bgr[int(abs_y - 80):int(abs_y + 80), int(abs_x - 80):int(abs_x + 80)]
        cv2.imwrite(rf"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\roi_comp_{l}.jpg", crop)
