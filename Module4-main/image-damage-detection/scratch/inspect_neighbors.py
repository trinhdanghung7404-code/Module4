import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from scratch.test_pipeline_clean import run_test_pipeline

d1, f1 = run_test_pipeline(r'images\1.jpg', r'images\2.jpg', '1.jpg vs 2.jpg')
print("\n--- 1.jpg vs 2.jpg: Tri 12 and 13 ---")
for tid in [12, 13]:
    t = d1[tid]
    print(f"Tri #{tid}: z={t['z']:.2f}, crack={t['crack']:.1f}, scr={t['scratch']:.1f}, int={t['intrusive']:.1f}, broken={t['broken']:.1f}, blob={t['blob']}, b_shift={t['is_b_shift']}")

v12 = set(d1[12]['v_indices'])
for i, t in enumerate(d1):
    if i != 12:
        shared = v12.intersection(set(t['v_indices']))
        if len(shared) >= 2:
            print(f"Tri 12 shares edge with Tri {i} (z={t['z']:.2f}, blob={t['blob']}, int={t['intrusive']:.1f})")

d2, f2 = run_test_pipeline(r'images\test_nobg.png', r'images\test_nobg - scar.png', 'scar')
print("\n--- scar: Tri 98 ---")
v98 = set(d2[98]['v_indices'])
for i, t in enumerate(d2):
    if i != 98:
        shared = v98.intersection(set(t['v_indices']))
        if len(shared) >= 2:
            print(f"Tri 98 shares edge with Tri {i} (z={t['z']:.2f}, blob={t['blob']}, int={t['intrusive']:.1f})")
