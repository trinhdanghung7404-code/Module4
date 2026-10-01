import numpy as np
import cv2
from typing import Tuple, List
import config
from mesh.mesh_builder import Vertex, Triangle
from mesh.region_extractor import RegionExtractor

class StructureLayer:
    """Detects structural damage (cracks, chips, breaks) using edge analysis.
    
    Key insight: Real damage creates NEW edges that don't exist in the reference.
    Lighting changes don't create new edges — they only change brightness of existing ones.
    
    This layer is the most lighting-invariant.
    """
    
    def __init__(self):
        self.extractor = RegionExtractor()
        
    def analyze(self, product_gray: np.ndarray, return_gray: np.ndarray,
                product_mask: np.ndarray, return_mask: np.ndarray,
                vertices: List[Vertex], triangles: List[Triangle],
                product_descriptors: np.ndarray = None,
                return_descriptors: np.ndarray = None
                ) -> Tuple[np.ndarray, List[dict]]:
        """
        Analyze structural differences between product and return images.
        """
        scores = []
        for tri in triangles:
            score = self._analyze_triangle(
                product_gray, return_gray, vertices, tri,
                product_descriptors, return_descriptors
            )
            scores.append(score)
            
        scores = self._flag_outliers(scores)
        
        damage_map = np.zeros_like(product_gray, dtype=np.uint8)
        for score in scores:
            if score['is_damaged']:
                tri = triangles[score['triangle_id']]
                pts = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.int32)
                cv2.fillConvexPoly(damage_map, pts, 255)
                
        return damage_map, scores
        
    def _analyze_triangle(self, product_gray, return_gray, vertices, triangle,
                          product_descriptors, return_descriptors) -> dict:
        patches = self.extractor.extract_vertex_patches(
            product_gray, return_gray, vertices, triangle, patch_size=config.PATCH_SIZE
        )
        
        total_new_edge_pixels = 0
        total_pixels = 0
        
        for p_patch, r_patch in patches:
            p_edges = cv2.Canny(p_patch, config.CANNY_LOW, config.CANNY_HIGH)
            r_edges = cv2.Canny(r_patch, config.CANNY_LOW, config.CANNY_HIGH)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                (2 * config.EDGE_DILATE_RADIUS + 1, 2 * config.EDGE_DILATE_RADIUS + 1))
            p_edges_dilated = cv2.dilate(p_edges, kernel)
            
            new_edges = cv2.bitwise_and(r_edges, cv2.bitwise_not(p_edges_dilated))
            
            total_new_edge_pixels += np.count_nonzero(new_edges)
            total_pixels += p_patch.size
            
        new_edge_ratio = total_new_edge_pixels / max(total_pixels, 1)
        
        desc_sim = 1.0
        if product_descriptors is not None and return_descriptors is not None:
            sims = []
            for idx in triangle.vertex_indices:
                v = vertices[idx]
                if v.descriptor_index is not None:
                    pd = product_descriptors[v.descriptor_index]
                    rd = return_descriptors[v.descriptor_index]
                    sims.append(np.dot(pd, rd) / (np.linalg.norm(pd) * np.linalg.norm(rd) + 1e-8))
            if sims:
                desc_sim = float(np.mean(sims))
                
        return {
            'triangle_id': triangle.triangle_id,
            'new_edge_ratio': new_edge_ratio,
            'descriptor_sim': desc_sim,
            'score': max(0.0, 1.0 - new_edge_ratio),
            'is_damaged': False
        }

    def _flag_outliers(self, scores: List[dict]) -> List[dict]:
        """Use z-score to flag triangles with abnormally high new_edge_ratio."""
        ratios = [s['new_edge_ratio'] for s in scores]
        if not ratios:
            return scores
        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        
        for s in scores:
            z_score = (s['new_edge_ratio'] - mean_r) / (std_r + 1e-8)
            # Must be both a statistical outlier AND have significant new edges (> 6%)
            s['is_damaged'] = bool(z_score > config.STRUCTURE_ZSCORE_THRESHOLD and s['new_edge_ratio'] > 0.06)
            
        return scores
