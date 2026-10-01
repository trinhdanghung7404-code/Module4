import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
import detailed_debug

def test_without_blob():
    print("=== Testing without blob in struct_metric ===")
    
    # 1. Undamaged vase (1.jpg vs 2.jpg)
    p1 = os.path.join(detailed_debug.PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(detailed_debug.PROJECT_ROOT, "images", "2.jpg")
    
    # Let's inspect detailed_debug with blob vs without blob
    # First, let's see what happens to Tri #15, #466, #605, and Scar dataset
    
test_without_blob()
