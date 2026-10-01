import cv2
import numpy as np
import os

crop_path = r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd\.user_uploaded\media_1790154640162.png"
artifact_dir = r"C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd"

crop_bgr = cv2.imread(crop_path)
crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

for i in range(32):
    p = os.path.join(artifact_dir, f"spot_candidate_{i}.jpg")
    if not os.path.exists(p):
        continue
    cand = cv2.imread(p)
    cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
    
    # Try different scales of template matching
    best = 0.0
    for s in np.linspace(0.2, 1.5, 20):
        c_res = cv2.resize(crop_gray, (0, 0), fx=s, fy=s)
        if c_res.shape[0] >= cand_gray.shape[0] or c_res.shape[1] >= cand_gray.shape[1]:
            continue
        res = cv2.matchTemplate(cand_gray, c_res, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        if max_v > best:
            best = max_v
    if best > 0.6:
        print(f"Candidate #{i} match score: {best:.3f}")
