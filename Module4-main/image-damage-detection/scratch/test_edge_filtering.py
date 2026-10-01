import cv2, numpy as np, os

p_img = cv2.imread(r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg')
p_gray = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)

# Bilateral filter or median blur
p_med = cv2.medianBlur(p_gray, 3)

# Canny raw
p_edge_raw = cv2.Canny(p_med, 40, 110)

def shape_aware_edge_filter(edge_mask, min_stroke_len=12, max_round_size=20):
    """
    Filters edge noise by distinguishing linear strokes from round glare/specks.
    - Preserves thin elongated strokes even if short (length >= min_stroke_len).
    - Removes small compact/round specks (aspect ratio < 2.0 and size < max_round_size).
    - Removes isolated tiny dust (< 6px).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    clean_mask = np.zeros_like(edge_mask)

    kept_strokes = 0
    removed_round = 0
    removed_tiny = 0

    for lbl in range(1, num_labels):
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        diag = np.sqrt(w**2 + h**2)

        # 1. Tiny dust / isolated pixels (< 6px)
        if area < 6:
            removed_tiny += 1
            continue

        # 2. Aspect ratio / Elongation
        major = max(w, h)
        minor = max(min(w, h), 1)
        aspect = major / minor

        # If it is compact and small (round noise / specular ring)
        is_round_speck = (aspect < 2.0) and (diag <= max_round_size)
        if is_round_speck:
            removed_round += 1
            continue

        # If it has sufficient length or elongation -> it is a stroke!
        if diag >= min_stroke_len or area >= 25 or aspect >= 2.2:
            clean_mask[labels == lbl] = 255
            kept_strokes += 1
        else:
            removed_tiny += 1

    print(f'Shape-aware filter: Kept strokes={kept_strokes}, Removed round specks={removed_round}, Removed tiny dust={removed_tiny}')
    return clean_mask

# Compare current filter vs shape-aware filter
def current_filter(edge_mask, min_area=50, min_diag=35.0):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    clean = np.zeros_like(edge_mask)
    for lbl in range(1, num_labels):
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        diag = np.sqrt(w**2 + h**2)
        if area >= min_area or diag >= min_diag:
            clean[labels == lbl] = 255
    return clean

c_cur = current_filter(p_edge_raw, min_area=50, min_diag=35.0)
c_new = shape_aware_edge_filter(p_edge_raw, min_stroke_len=14, max_round_size=18)

out_dir = r'C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd'
cv2.imwrite(os.path.join(out_dir, 'edge_current.png'), c_cur)
cv2.imwrite(os.path.join(out_dir, 'edge_shape_aware.png'), c_new)
print('Saved edge_current.png and edge_shape_aware.png to artifact directory.')
