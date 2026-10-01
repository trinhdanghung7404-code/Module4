import cv2
import numpy as np
from pipeline.comparator import DamageComparator
from preprocessing.segmentation import ObjectSegmenter

def analyze_return_product():
    """Interactive CLI to analyze a returned product."""
    from database.database import Database
    
    db = Database()
    try:
        # Show available products
        products = db.get_products()
        if not products:
            print("No products in database. Add a product first.")
            return
        
        print("\n=== Available Products ===")
        for p in products:
            print(f"  ID: {p['id']} | Image: {p['image_path']}")
        
        # Get user input
        product_id = int(input("\nEnter Product ID: ").strip())
        return_path = input("Enter return image path: ").strip()
        
        # Get product info
        product_feature = db.get_product_feature(product_id)
        if product_feature is None:
            print(f"Product ID {product_id} not found.")
            return
        
        product_path = product_feature['image_path']
        
        # Run comparison
        print(f"\n=== Comparing ===")
        print(f"Product: {product_path}")
        print(f"Return:  {return_path}")
        print("Processing...\n")
        
        comparator = DamageComparator()
        result = comparator.compare(product_path, return_path)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"  DAMAGE ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"  Damage Score:        {result['damage_score']:.2f}%")
        print(f"  Damage Area:         {result['damage_area']} pixels")
        print(f"  Largest Damage:      {result['largest_damage_area']} pixels")
        print(f"  Is Damaged:          {'YES' if result['is_damaged'] else 'NO'}")
        print(f"  Processing Time:     {result['processing_time']:.1f}s")
        print(f"{'='*50}")
        
        # Pipeline summary
        summary = result.get('pipeline_summary', {})
        if summary:
            print(f"\n--- Pipeline Summary ---")
            for step, info in summary.items():
                print(f"  {step}: {info}")
        
        # Debug images
        if result.get('debug_image_paths'):
            print(f"\n--- Debug Images ---")
            for name, path in result['debug_image_paths'].items():
                print(f"  {name}: {path}")
        
        # Save to database
        try:
            return_img = cv2.imread(return_path)
            seg = ObjectSegmenter().segment(return_img) if return_img is not None else {'area': 0, 'bbox': [0,0,0,0]}
            
            return_feature = {
                'image_path': return_path,
                'object_area': seg.get('area', 0),
                'perimeter': cv2.arcLength(seg.get('contour', np.array([])), True) if seg.get('contour') is not None else 0,
                'width': seg.get('bbox', [0,0,0,0])[2],
                'height': seg.get('bbox', [0,0,0,0])[3],
            }
            db.save_return(product_id, return_feature, result)
            print(f"\nAnalysis saved to database.")
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")
        
    finally:
        db.close()
