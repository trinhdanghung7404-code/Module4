import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

with open('v2/detailed_debug.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("V2_DIR = os.path.dirname(os.path.abspath(__file__))", "V2_DIR = os.path.abspath('v2')")

hook = """
    print("\\n=== DETAILED BREAKDOWN OF FLAGGED TRIANGLES IN VASE ===")
    for t in triangle_details:
        if t['is_fused']:
            reasons = []
            if t['is_l2']:
                reasons.append(f"L2(zc={t['z_c']:.2f},dE={t['chroma_err']:.1f})")
            if t['intrusive_len'] >= 12.0:
                reasons.append(f"Intrusive({t['intrusive_len']:.1f})")
            if t['dark_crack'] >= 12.0:
                reasons.append(f"Crack({t['dark_crack']:.1f})")
            if t['broken_length'] >= 25.0 and t['ori_corr'] < 0.85:
                reasons.append(f"Broken({t['broken_length']:.1f},ori={t['ori_corr']:.2f})")
            if t['blob_area'] >= 50 and t['z_l1'] > 1.5:
                reasons.append(f"Blob({t['blob_area']},z={t['z_l1']:.2f})")
            if t['is_l2'] and t['has_physical_l1']:
                reasons.append("Dual(L2+L1)")
            print(f"Tri #{t['id']:3d}: {', '.join(reasons)} | broken={t['broken_length']:.1f}, crack={t['dark_crack']:.1f}, intrusive={t['intrusive_len']:.1f}, blob={t['blob_area']}, ori={t['ori_corr']:.2f}")
"""

part1 = text.split('if __name__ == "__main__":')[0]
runner = """
if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    run_detailed_debug(p1, p2, base_debug_dir=os.path.join(V2_DIR, "debug", "01_undamaged_vase"))
"""

full = part1.replace('fused_flagged = [t["id"] for t in triangle_details if t["is_fused"]]', 'fused_flagged = [t["id"] for t in triangle_details if t["is_fused"]]\n' + hook) + runner

with open('scratch/run_vase_breakdown.py', 'w', encoding='utf-8') as f:
    f.write(full)

print("Created scratch/run_vase_breakdown.py successfully.")
