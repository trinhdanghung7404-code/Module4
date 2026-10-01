import numpy as np
import cv2
from typing import Tuple, List
from mesh.mesh_builder import Vertex, Triangle

class RegionExtractor:
    """Extracts image patches at original positions for comparison.
    
    Key principle: NO image warping. Patches are cropped from original images
    at the positions indicated by keypoint correspondences.
    """
    
    def extract_triangle_patches(self, product_image: np.ndarray, return_image: np.ndarray,
                                  vertices: List[Vertex], triangle: Triangle
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract corresponding triangle-shaped patches from both images.
        No warping — just crop and mask.
        """
        # Product
        prod_pts = np.array([vertices[idx].product_xy for idx in triangle.vertex_indices], dtype=np.int32)
        p_x, p_y, p_w, p_h = cv2.boundingRect(prod_pts)
        prod_mask = np.zeros((p_h, p_w), dtype=np.uint8)
        shifted_prod_pts = prod_pts - np.array([p_x, p_y])
        cv2.fillConvexPoly(prod_mask, shifted_prod_pts, 255)
        prod_patch = product_image[p_y:p_y+p_h, p_x:p_x+p_w]
        
        # Return
        ret_pts = np.array([vertices[idx].return_xy for idx in triangle.vertex_indices], dtype=np.int32)
        r_x, r_y, r_w, r_h = cv2.boundingRect(ret_pts)
        ret_mask = np.zeros((r_h, r_w), dtype=np.uint8)
        shifted_ret_pts = ret_pts - np.array([r_x, r_y])
        cv2.fillConvexPoly(ret_mask, shifted_ret_pts, 255)
        ret_patch = return_image[r_y:r_y+r_h, r_x:r_x+r_w]
        
        return prod_patch, prod_mask, ret_patch, ret_mask
        
    def extract_vertex_patches(self, product_image: np.ndarray, return_image: np.ndarray,
                                vertices: List[Vertex], triangle: Triangle,
                                patch_size: int = 64
                                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract square patches around each vertex of the triangle.
        """
        patches = []
        for idx in triangle.vertex_indices:
            v = vertices[idx]
            prod_patch = self._crop_patch(product_image, v.product_xy[0], v.product_xy[1], patch_size)
            ret_patch = self._crop_patch(return_image, v.return_xy[0], v.return_xy[1], patch_size)
            patches.append((prod_patch, ret_patch))
        return patches

    @staticmethod
    def _crop_patch(image: np.ndarray, center_x: float, center_y: float, 
                    size: int) -> np.ndarray:
        """
        Crop a square patch centered at (center_x, center_y).
        Handles boundary by zero-padding.
        """
        h, w = image.shape[:2]
        cx, cy = int(center_x), int(center_y)
        half_size = size // 2
        
        y1 = max(0, cy - half_size)
        y2 = min(h, cy + half_size + (size % 2))
        x1 = max(0, cx - half_size)
        x2 = min(w, cx + half_size + (size % 2))
        
        patch = image[y1:y2, x1:x2]
        
        pad_y1 = (cy - half_size) - y1 if (cy - half_size) < 0 else 0
        pad_y2 = y2 - (cy + half_size + (size % 2)) if (cy + half_size + (size % 2)) > h else 0
        pad_x1 = (cx - half_size) - x1 if (cx - half_size) < 0 else 0
        pad_x2 = x2 - (cx + half_size + (size % 2)) if (cx + half_size + (size % 2)) > w else 0
        
        if len(image.shape) == 3:
            pad_width = ((abs(pad_y1), abs(pad_y2)), (abs(pad_x1), abs(pad_x2)), (0, 0))
        else:
            pad_width = ((abs(pad_y1), abs(pad_y2)), (abs(pad_x1), abs(pad_x2)))
            
        return np.pad(patch, pad_width, mode='constant', constant_values=0)
