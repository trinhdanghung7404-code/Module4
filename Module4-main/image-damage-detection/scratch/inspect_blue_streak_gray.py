import cv2, numpy as np

r_img = cv2.imread(r'images\2.jpg')
p_img = cv2.imread(r'images\1.jpg')
r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
p_gray = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)

# In cheek_defect_region:
# x0, x1 = 980, 1180; y0, y1 = 1000, 1200
crop_r = r_gray[1000:1200, 980:1180]
crop_p = p_gray[1000:1200, 980:1180]

# Where is the blue streak?
# In r_img: B is high, R is low
crop_r_bgr = r_img[1000:1200, 980:1180]
blue_mask = (crop_r_bgr[..., 0] > 150) & (crop_r_bgr[..., 2] < 80)

print(f"Blue streak pixels: {np.sum(blue_mask)}")
print(f"Blue streak r_gray mean: {np.mean(crop_r[blue_mask]):.1f}, min: {np.min(crop_r[blue_mask])}, max: {np.max(crop_r[blue_mask])}")
print(f"Surrounding r_gray mean: {np.mean(crop_r[~blue_mask]):.1f}")
print(f"Product p_gray in blue_mask area: {np.mean(crop_p[blue_mask]):.1f}, min: {np.min(crop_p[blue_mask])}, max: {np.max(crop_p[blue_mask])}")
print(f"Raw diff |r_gray - p_gray| in blue streak: mean={np.mean(np.abs(crop_r[blue_mask].astype(float) - crop_p[blue_mask].astype(float))):.1f}, max={np.max(np.abs(crop_r[blue_mask].astype(float) - crop_p[blue_mask].astype(float)))}")
