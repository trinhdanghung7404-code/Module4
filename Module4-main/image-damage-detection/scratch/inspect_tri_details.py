import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from scratch.test_pipeline_clean import run_test_pipeline

d1, f1 = run_test_pipeline(r'images\1.jpg', r'images\2.jpg', '1.jpg vs 2.jpg')
for tid in [3, 12, 13]:
    t = d1[tid]
    print(f"Tri #{tid}: z={t['z']:.2f}, crack={t['crack']:.1f}, scr={t['scratch']:.1f}, int={t['intrusive']:.1f}, broken={t['broken']:.1f}, blob={t['blob']}, b_shift={t['is_b_shift']}, ori_corr={t['ori_corr']:.2f}")

d2, f2 = run_test_pipeline(r'images\test_nobg.png', r'images\test_nobg - scar.png', 'scar')
for tid in [98, 121, 123, 124, 341, 432, 433, 454]:
    t = d2[tid]
    print(f"Scar Tri #{tid}: z={t['z']:.2f}, crack={t['crack']:.1f}, scr={t['scratch']:.1f}, int={t['intrusive']:.1f}, broken={t['broken']:.1f}, blob={t['blob']}, b_shift={t['is_b_shift']}, ori_corr={t['ori_corr']:.2f}")
