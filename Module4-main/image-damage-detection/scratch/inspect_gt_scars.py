import sys, os
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\scratch')
from test_pipeline_clean import run_test_pipeline

d2, f2 = run_test_pipeline(r'images\test_nobg.png', r'images\test_nobg - scar.png', 'scar')
for tid in [22, 98, 121, 123, 124, 341, 431, 432, 433, 454]:
    t = d2[tid]
    print(f"Tri #{tid:3d}: Flagged={t['is_l1']} | z={t['z']:5.2f} | crack={t['crack']:4.1f} | scr={t['scratch']:4.1f} | int={t['intrusive']:4.1f} | broken={t['broken']:4.1f} | blob={t['blob']:3d}")
