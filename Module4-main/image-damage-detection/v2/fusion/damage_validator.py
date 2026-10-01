import numpy as np
import cv2
import config

class DamageValidator:
    """Validates damage mask using connected component analysis."""
    
    def validate(self, damage_mask: np.ndarray, object_mask: np.ndarray) -> dict:
        """Filter damage mask to remove false positive artifacts."""
        masked = cv2.bitwise_and(damage_mask, object_mask)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked, connectivity=8)
        
        validated_mask = np.zeros_like(masked)
        valid_components = []
        total_area = 0
        largest_area = 0
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if area < config.MIN_DAMAGE_AREA:
                continue
                
            aspect_ratio = max(w, h) / float(min(w, h)) if min(w, h) > 0 else 0
            if aspect_ratio > config.MAX_DAMAGE_ASPECT_RATIO:
                continue
                
            validated_mask[labels == i] = 255
            total_area += area
            largest_area = max(largest_area, area)
            valid_components.append({
                'label': i,
                'area': area,
                'bbox': (x, y, w, h),
                'centroid': centroids[i]
            })
            
        return {
            'validated_mask': validated_mask,
            'damage_area': total_area,
            'largest_damage_area': largest_area,
            'num_components': len(valid_components),
            'components': valid_components
        }
