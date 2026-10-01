import numpy as np
import cv2
from typing import Tuple, List
import config

class DamageFusion:
    """Combines Structure and Appearance layer results into final damage mask."""
    
    def fuse(self, structure_map: np.ndarray, appearance_map: np.ndarray,
             structure_scores: List[dict], appearance_scores: List[dict],
             vertices, triangles) -> Tuple[np.ndarray, List[dict]]:
        """Combine two layer damage maps."""
             
        fused_scores = []
        fused_mask = np.zeros_like(structure_map)
        
        # Assume scores are sorted identically by triangle_id
        for s_score, a_score in zip(structure_scores, appearance_scores):
            struct_flag = s_score['is_damaged']
            app_flag = a_score['is_damaged']
            
            if struct_flag and app_flag:
                is_damaged = True
            elif struct_flag:
                is_damaged = True
            elif app_flag:
                is_damaged = not config.FUSION_REQUIRE_STRUCTURE
            else:
                is_damaged = False
                
            fused_scores.append({
                'triangle_id': s_score['triangle_id'],
                'is_damaged': is_damaged
            })
            
        for score in fused_scores:
            if score['is_damaged']:
                tri = triangles[score['triangle_id']]
                pts = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.int32)
                cv2.fillConvexPoly(fused_mask, pts, 255)
                
        return fused_mask, fused_scores
