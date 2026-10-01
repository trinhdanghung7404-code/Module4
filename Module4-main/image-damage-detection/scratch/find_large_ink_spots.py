import cv2
import numpy as np
import os

PROJECT_ROOT = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection"
ret_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "16.jpg"))
prod_bgr = cv2.imread(os.path.join(PROJECT_ROOT, "images", "1.jpg"))
artifact_dir = r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd"

# Let's find all regions where 16.jpg is very dark (< 50) and 1.jpg is much brighter (> 90)
# on the full resolution 2560x1920
# First align roughly or use registration
ret_gray = cv2.cvtColor(ret_bgr, cv2.COLOR_BGR2GRAY)
prod_gray = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2GRAY)

# Dark spots on Return that are NOT naturally dark in Product:
# In 16.jpg, find connected components of ret_gray < 50
dark_cand = (ret_gray < 50)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dark_clean = cv2.morphologyEx(dark_cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)

num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(dark_clean)
print(f"Total dark components: {num_lbl - 1}")

crops = []
idx = 0
for l in range(1, num_lbl):
    area = stats[l, cv2.CC_STAT_AREA]
    if area > 1000:  # The drawn ink spots are large, > 1000 pixels on 2560x1920!
        x = stats[l, cv2.CC_STAT_LEFT]
        y = stats[l, cv2.CC_STAT_TOP]
        w = stats[l, cv2.CC_STAT_WIDTH]
        h = stats[l, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[l]
        
        # Check if Product at this location is bright
        # (on rooster's tail, Product is also dark; on ink spots, Product is bright feathers/flowers!)
        pad = 40
        y1, y2 = max(0, y - pad), min(ret_bgr.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(ret_bgr.shape[1], x + w + pad)
        
        crop_ret = ret_bgr[y1:y2, x1:x2]
        crop_prod = prod_bgr[y1:y2, x1:x2]
        mean_p_crop = np.mean(prod_gray[y1:y2, x1:x2])
        mean_r_spot = np.mean(ret_gray[y:y+h, x:x+w])
        
        print(f"Candidate #{idx}: Area={area}, BBox=({x}, {y}, {w}, {h}), Centroid=({cx:.1f}, {cy:.1f}), Mean R spot={mean_r_spot:.1f}, Mean P region={mean_p_crop:.1f}")
        
        out_crop = os.path.join(artifact_dir, f"spot_candidate_{idx}.jpg")
        cv2.imwrite(out_crop, crop_ret)
        idx += 1
