"""Test comparison với thresholds mới - KHÔNG cần database"""
import cv2
import numpy as np
from pathlib import Path
from comparator import DamageComparator
from feature_extractor import FeatureExtractor
from geometry import GeometryFeature
from superpoint_extractor import SuperPointExtractor

# Đường dẫn ảnh
PRODUCT_IMAGE = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\1.jpg"
RETURN_IMAGE = r"C:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\images\2.jpg"

def extract_features(image_path):
    """Extract features từ ảnh"""
    print(f"\n📷 Đang extract features từ: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    
    print(f"   Kích thước: {image.shape}")
    
    # Geometry
    geometry = GeometryFeature()
    object_mask, bbox = geometry.extract_object_mask(image)
    area, perimeter, width, height = geometry.compute_geometry(object_mask)
    
    print(f"   Object area: {area} pixels")
    print(f"   Perimeter: {perimeter:.1f}")
    
    # SuperPoint
    extractor = SuperPointExtractor(max_keypoints=300)
    keypoints, descriptors = extractor.extract(image, object_mask)
    
    print(f"   Keypoints: {len(keypoints)}")
    
    return {
        "image": image,
        "image_path": image_path,
        "object_area": area,
        "perimeter": perimeter,
        "width": width,
        "height": height,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "object_mask": object_mask,
        "bbox": bbox
    }

def main():
    print("=" * 80)
    print("🧪 TEST COMPARISON VỚI THRESHOLDS MỚI")
    print("=" * 80)
    
    # Extract features
    product_feat = extract_features(PRODUCT_IMAGE)
    return_feat = extract_features(RETURN_IMAGE)
    
    # Compare
    print("\n" + "=" * 80)
    print("🔍 Đang so sánh...")
    print("=" * 80)
    
    comparator = DamageComparator()
    result = comparator.compare(product_feat, return_feat)
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ SO SÁNH")
    print("=" * 80)
    print(f"Damage Score: {result['damage_score']:.2f}%")
    print(f"Difference Area: {result['difference_area']} pixels")
    print(f"Largest Damage Area: {result['largest_damage_area']} pixels")
    print(f"Confidence Score: {result.get('confidence_score', 1.0):.4f}")
    
    # Đánh giá
    print("\n" + "=" * 80)
    print("💡 ĐÁNH GIÁ")
    print("=" * 80)
    
    if result['damage_score'] < 5.0:
        print("✅ EXCELLENT! Damage score rất thấp - Không có false positives")
    elif result['damage_score'] < 15.0:
        print("✅ GOOD! Damage score thấp - Có thể có một ít noise")
    elif result['damage_score'] < 30.0:
        print("⚠️  ACCEPTABLE - Vẫn còn một số false positives")
    else:
        print("❌ POOR - Quá nhiều false positives, cần điều chỉnh thêm")
    
    print(f"\n📁 Debug images được lưu trong: debug_outputs/")
    print("=" * 80)

if __name__ == "__main__":
    main()
