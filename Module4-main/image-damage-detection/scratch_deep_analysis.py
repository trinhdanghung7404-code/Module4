import sys
sys.path.insert(0, 'v2')
import os, cv2, numpy as np

# Let's inspect the L2 metrics for TP vs FP in detail
tp_ids = {635, 445, 443, 631, 303, 456, 605}

# Let's write a script to inspect:
# 1. chroma_err and has_color_blob for all triangles
# 2. connected components of flagged triangles
# 3. boundary vs interior of triangle patches
print("Ready for deep analysis...")
