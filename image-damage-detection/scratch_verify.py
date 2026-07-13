import sys
from pathlib import Path
from feature_extractor import FeatureExtractor
from comparator import DamageComparator

def run_test():
    img_a = "images/1.jpg"
    img_b = "images/2.jpg"
    
    print(f"Extracting features for {img_a}...")
    extractor = FeatureExtractor()
    feat_a = extractor.extract(img_a)
    
    print(f"Extracting features for {img_b}...")
    feat_b = extractor.extract(img_b)
    
    print("Running DamageComparator...")
    comp = DamageComparator()
    result = comp.compare(feat_a, feat_b)
    
    print("\nComparison Result:")
    print(f"Difference Area     : {result['difference_area']}")
    print(f"Largest Damage Area : {result['largest_damage_area']}")
    print(f"Damage Score (%)    : {result['damage_score']:.2f}")
    
    for key in ["raw_difference_area", "filtered_difference_area", "removed_noise_area", "feature_reject_area", "ssim_reject_area", "final_difference_area"]:
        if key in result:
            print(f"{key:25s}: {result[key]}")

if __name__ == "__main__":
    run_test()
