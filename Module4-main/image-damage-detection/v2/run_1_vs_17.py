import os
import sys

V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from detailed_debug import run_detailed_debug

img1_path = os.path.join(PROJECT_ROOT, "images", "1.jpg")
img2_path = os.path.join(PROJECT_ROOT, "images", "17.jpg")

output_dir = os.path.join(V2_DIR, "debug", "experiment_1_vs_17_l1v2")

print(f"Running algorithm on 1.jpg vs 17.jpg...")
print(f"Image 1: {img1_path}")
print(f"Image 2: {img2_path}")
print(f"Debug Output Directory: {output_dir}")

res = run_detailed_debug(img1_path, img2_path, base_debug_dir=output_dir)
print("\nExecution Completed Successfully!")
print(f"Total Triangles: {res['triangles']}")
print(f"Tier 1 Defects (Do): {res['tier1_confirmed']}")
print(f"Tier 2 Defects (Cam): {res['tier2_probable']}")
print(f"Tier 3 Suspects (Vang): {res['tier3_isolated']}")
print(f"Total True Defects Flagged (Tier 1+2): {res['total_defects']}")
