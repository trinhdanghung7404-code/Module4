import cv2
import numpy as np
import os
from typing import Dict, Optional

V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(V2_DIR, "debug_outputs")

class DebugVisualizer:
    """Generates debug visualization images for damage detection pipeline."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_normalization_steps(self, original, after_color_transfer, 
                                  after_clahe, after_retinex, final):
        """Save each normalization step for debugging."""
        cv2.imwrite(os.path.join(self.output_dir, "norm_1_original.jpg"), original)
        cv2.imwrite(os.path.join(self.output_dir, "norm_2_color_transfer.jpg"), after_color_transfer)
        cv2.imwrite(os.path.join(self.output_dir, "norm_3_clahe.jpg"), after_clahe)
        cv2.imwrite(os.path.join(self.output_dir, "norm_4_retinex.jpg"), after_retinex)
        cv2.imwrite(os.path.join(self.output_dir, "norm_5_final.jpg"), final)
    
    def save_registration(self, product_img, return_img, 
                          product_points, return_points):
        """Draw matched keypoints on both images side by side.
        Use cv2.drawMatches style visualization."""
        if product_points is None or return_points is None or len(product_points) == 0:
            return None
            
        kps1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in product_points]
        kps2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in return_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(product_points))]
        
        img_matches = cv2.drawMatches(product_img, kps1, return_img, kps2, matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        path = os.path.join(self.output_dir, "registration.jpg")
        cv2.imwrite(path, img_matches)
        return path
    
    def save_mesh_overlay(self, image, vertices, triangles, triangle_scores=None):
        """Draw Delaunay mesh on image. Color triangles by score if provided.
        Green = OK, Red = Damaged, Yellow = Uncertain."""
        out_img = image.copy()
        if vertices is not None and triangles is not None and len(triangles) > 0:
            for i, tri in enumerate(triangles):
                pts = np.array([
                    list(vertices[idx].product_xy) for idx in tri.vertex_indices
                ], np.int32).reshape((-1, 1, 2))
                color = (0, 255, 0)  # Green
                if triangle_scores is not None and i < len(triangle_scores):
                    if triangle_scores[i].get('is_damaged', False):
                        color = (0, 0, 255)  # Red
                cv2.polylines(out_img, [pts], isClosed=True, color=color, thickness=1)
        path = os.path.join(self.output_dir, "mesh_overlay.jpg")
        cv2.imwrite(path, out_img)
        return path
    
    def save_layer_results(self, product_img, structure_map, appearance_map, fused_map):
        """Save layer damage maps as color overlays on the product image.
        Structure = blue overlay, Appearance = green overlay, Fused = red overlay."""
        def overlay(img, mask, color):
            colored_mask = np.zeros_like(img)
            if mask is not None:
                colored_mask[mask > 0] = color
            return cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
            
        if structure_map is not None:
            cv2.imwrite(os.path.join(self.output_dir, "layer_structure.jpg"), overlay(product_img, structure_map, (255, 0, 0)))
        if appearance_map is not None:
            cv2.imwrite(os.path.join(self.output_dir, "layer_appearance.jpg"), overlay(product_img, appearance_map, (0, 255, 0)))
        if fused_map is not None:
            cv2.imwrite(os.path.join(self.output_dir, "layer_fused.jpg"), overlay(product_img, fused_map, (0, 0, 255)))
    
    def save_final_result(self, product_img, return_img, damage_mask, damage_score):
        """Create final comparison: product | return | damage overlay.
        Three images side by side with damage score text."""
        h, w = product_img.shape[:2]
        return_img_resized = cv2.resize(return_img, (w, h))
        
        overlay = product_img.copy()
        if damage_mask is not None:
            red_mask = np.zeros_like(overlay)
            red_mask[damage_mask > 0] = (0, 0, 255)
            overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
            
        cv2.putText(overlay, f"Score: {damage_score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        final_img = np.hstack((product_img, return_img_resized, overlay))
        path = os.path.join(self.output_dir, "final_result.jpg")
        cv2.imwrite(path, final_img)
        return path
    
    def generate_all(self, **kwargs) -> Dict[str, str]:
        """Generate all debug images. Returns dict of name→filepath."""
        paths = {}
        if 'product_img' in kwargs and 'return_img' in kwargs and 'product_points' in kwargs and 'return_points' in kwargs:
            p = self.save_registration(kwargs['product_img'], kwargs['return_img'], kwargs['product_points'], kwargs['return_points'])
            if p: paths['registration'] = p
            
        if 'product_img' in kwargs and 'vertices' in kwargs and 'triangles' in kwargs:
            p = self.save_mesh_overlay(kwargs['product_img'], kwargs['vertices'], kwargs['triangles'], kwargs.get('structure_scores'))
            if p: paths['mesh'] = p
            
        if 'product_img' in kwargs and 'structure_map' in kwargs and 'appearance_map' in kwargs and 'fused_mask' in kwargs:
            self.save_layer_results(kwargs['product_img'], kwargs['structure_map'], kwargs['appearance_map'], kwargs['fused_mask'])
            paths['layers'] = os.path.join(self.output_dir, "layer_fused.jpg")
            
        if 'product_img' in kwargs and 'return_img' in kwargs and 'damage_mask' in kwargs and 'damage_score' in kwargs:
            p = self.save_final_result(kwargs['product_img'], kwargs['return_img'], kwargs['damage_mask'], kwargs['damage_score'])
            if p: paths['final'] = p
            
        return paths
