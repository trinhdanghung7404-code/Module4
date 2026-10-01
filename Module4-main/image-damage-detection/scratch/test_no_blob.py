import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')

# Let's import run_full_pipeline from detailed_debug
from detailed_debug import run_full_pipeline

# We can monkey-patch or test directly
print("Testing with current settings vs with blob=0 in struct_metric...")
