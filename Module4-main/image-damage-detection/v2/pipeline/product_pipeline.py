from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from features.superpoint_extractor import SuperPointExtractor
import cv2
import numpy as np

def add_product():
    """Interactive CLI to add a new product to the database."""
    from database.database import Database
    
    image_path = input("Enter product image path: ").strip()
    
    # Load and process
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return
    
    # Segment
    segmenter = ObjectSegmenter()
    seg = segmenter.segment(image)
    
    # Extract SuperPoint features for storage
    extractor = SuperPointExtractor(max_keypoints=500)
    normalizer = ImageNormalizer()
    # Self-normalize (no reference yet)
    norm_img = normalizer.apply_clahe(image)
    keypoints, descriptors = extractor.detect_and_compute(norm_img, seg['mask'])
    
    # Build feature dict for database
    feature = {
        'image_path': image_path,
        'object_area': seg['area'],
        'perimeter': cv2.arcLength(seg['contour'], True),
        'width': seg['bbox'][2],
        'height': seg['bbox'][3],
        'bbox': list(seg['bbox']),
        'texture_feature': {
            'keypoint_count': len(keypoints),
            'keypoints': keypoints,
            'descriptors': descriptors.tolist() if descriptors is not None else [],
            'bbox': list(seg['bbox'])
        }
    }
    
    # Save to database
    db = Database()
    try:
        db.save_product(feature)
        print(f"Product saved successfully! ({len(keypoints)} keypoints extracted)")
    finally:
        db.close()
