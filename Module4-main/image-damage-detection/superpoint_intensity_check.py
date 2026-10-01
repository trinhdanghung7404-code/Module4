import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- CONFIG ---
IMG1_PATH = "images/1.jpg"
IMG2_PATH = "images/2.jpg"
THRESHOLD = 5
OUTPUT_PATH = "debug_outputs/superpoint_intensity_report.png"

# --- LOAD IMAGES ---
print("Loading images...")
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
if img1 is None or img2 is None:
    raise FileNotFoundError(f"Cannot load {IMG1_PATH} or {IMG2_PATH}")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- SUPERPOINT SIMULATION (lightweight fallback) ---
# Since full SuperPoint may not be imported, we use ORB + BFMatcher as robust proxy
print("Detecting keypoints with ORB (lightweight fallback)...")
orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
    print("⚠️  No keypoints found. Falling back to grid sampling (10x10).")
    h, w = gray1.shape
    y_grid, x_grid = np.mgrid[10:h:20, 10:w:20]
    kp1 = [cv2.KeyPoint(float(x), float(y), 5) for x, y in zip(x_grid.ravel(), y_grid.ravel())]
    kp2 = kp1.copy()
    des1 = np.random.randint(0, 256, (len(kp1), 32), dtype=np.uint8)
    des2 = np.random.randint(0, 256, (len(kp2), 32), dtype=np.uint8)

# --- MATCH ---
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"Found {len(good)} good matches.")

# --- INTENSITY COMPARISON ---
deltas = []
matched_pts1, matched_pts2 = [], []
for m in good:
    u1, v1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
    u2, v2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])
    # Clamp to image bounds
    u1 = np.clip(u1, 0, gray1.shape[1]-1)
    v1 = np.clip(v1, 0, gray1.shape[0]-1)
    u2 = np.clip(u2, 0, gray2.shape[1]-1)
    v2 = np.clip(v2, 0, gray2.shape[0]-1)
    
    i1 = int(gray1[v1, u1])
    i2 = int(gray2[v2, u2])
    delta = abs(i1 - i2)
    deltas.append(delta)
    matched_pts1.append((u1, v1))
    matched_pts2.append((u2, v2))

if not deltas:
    print("❌ No valid intensity comparisons.")
    exit(1)

deltas = np.array(deltas)
damage_ratio = np.mean(deltas >= THRESHOLD) * 100
print(f"\n📊 RESULT: {damage_ratio:.2f}% of matched points have |ΔI| ≥ {THRESHOLD}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title("Image 1 (product)")
axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Image 2 (return)")

# Plot matches as red dots
for (u, v) in matched_pts1:
    axes[0].plot(u, v, 'ro', markersize=2)
for (u, v) in matched_pts2:
    axes[1].plot(u, v, 'ro', markersize=2)

plt.suptitle(f"SuperPoint-like match & intensity check\nDamage: {damage_ratio:.2f}% (threshold={THRESHOLD})")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"✅ Report saved to {OUTPUT_PATH}")

# Optional: print top 10 deltas
print("\nTop 10 intensity deltas:", deltas[np.argsort(-deltas)][:10])
