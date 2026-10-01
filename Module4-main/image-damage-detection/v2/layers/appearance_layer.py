import numpy as np
import cv2
from typing import Tuple, List
import config
from mesh.mesh_builder import Vertex, Triangle
from mesh.region_extractor import RegionExtractor
from skimage.metrics import structural_similarity as ssim

class AppearanceLayer:
    """Detects surface damage (scratches, glaze loss, stains) using appearance metrics."""
    
    def __init__(self):
        self.extractor = RegionExtractor()
        
    def analyze(self, product_color: np.ndarray, return_color_normalized: np.ndarray,
                product_mask: np.ndarray, return_mask: np.ndarray,
                vertices: List[Vertex], triangles: List[Triangle]
                ) -> Tuple[np.ndarray, List[dict]]:
        """Analyze appearance differences."""
        scores = []
        for tri in triangles:
            score = self._analyze_triangle(product_color, return_color_normalized, vertices, tri)
            scores.append(score)
            
        scores = self._flag_outliers(scores)
        
        damage_map = np.zeros(product_color.shape[:2], dtype=np.uint8)
        for score in scores:
            if score['is_damaged']:
                tri = triangles[score['triangle_id']]
                pts = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.int32)
                cv2.fillConvexPoly(damage_map, pts, 255)
                
        return damage_map, scores
        
    def _analyze_triangle(self, product_color, return_color, vertices, triangle) -> dict:
        patches = self.extractor.extract_vertex_patches(
            product_color, return_color, vertices, triangle, patch_size=config.PATCH_SIZE
        )
        
        ssim_scores = []
        color_diffs = []
        
        for p_patch, r_patch in patches:
            p_resized = cv2.resize(p_patch, (config.PATCH_RESIZE, config.PATCH_RESIZE))
            r_resized = cv2.resize(r_patch, (config.PATCH_RESIZE, config.PATCH_RESIZE))
            
            p_gray = cv2.cvtColor(p_resized, cv2.COLOR_BGR2GRAY)
            r_gray = cv2.cvtColor(r_resized, cv2.COLOR_BGR2GRAY)
            
            s = ssim(p_gray, r_gray, data_range=255, win_size=config.SSIM_WIN_SIZE)
            ssim_scores.append(s)
            
            p_lab = cv2.cvtColor(p_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
            r_lab = cv2.cvtColor(r_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            diff = np.sqrt(np.sum((p_lab - r_lab) ** 2, axis=2))
            color_diffs.append(np.mean(diff))
            
        mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 1.0
        mean_color_diff = float(np.mean(color_diffs)) if color_diffs else 0.0
        
        return {
            'triangle_id': triangle.triangle_id,
            'ssim': mean_ssim,
            'color_diff': mean_color_diff,
            'score': mean_ssim, 
            'is_damaged': False
        }

    def _flag_outliers(self, scores: List[dict]) -> List[dict]:
        """Flag triangles with abnormally LOW ssim or abnormally HIGH color_diff"""
        if not scores:
            return scores
            
        ssims = [s['ssim'] for s in scores]
        cdiffs = [s['color_diff'] for s in scores]
        
        m_s, std_s = np.mean(ssims), np.std(ssims)
        m_c, std_c = np.mean(cdiffs), np.std(cdiffs)
        
        for s in scores:
            z_s = (s['ssim'] - m_s) / (std_s + 1e-8)
            z_c = (s['color_diff'] - m_c) / (std_c + 1e-8)
            # Must be both a statistical outlier AND have real discrepancy (not normal lighting glare)
            s['is_damaged'] = bool(
                (z_s < -config.APPEARANCE_ZSCORE_THRESHOLD and s['ssim'] < 0.65) or
                (z_c > config.APPEARANCE_ZSCORE_THRESHOLD and s['color_diff'] > 22.0)
            )
            
        return scores
