import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

# Test the scar discrimination logic
from scratch.test_refined_logic import *

# Let's inspect Tri 432 (scar) vs Tri 150 (vase glare) vs Tri 47 (vase scales)
print("Testing scar vs glare discrimination...")
