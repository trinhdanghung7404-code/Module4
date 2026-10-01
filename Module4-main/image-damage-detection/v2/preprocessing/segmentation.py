import cv2
import numpy as np
import config

class ObjectSegmenter:
    """Segments the ceramic object from background using Otsu thresholding."""
    
    def segment(self, image: np.ndarray) -> dict:
        """
        Segment the main object from background.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            dict with keys:
            - 'mask': binary mask (H, W), uint8, 255=object, 0=background
            - 'bbox': (x, y, w, h) bounding box of object
            - 'contour': largest contour as np.ndarray
            - 'area': object area in pixels
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. GaussianBlur (5, 5)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Morphological close - seal holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITERATIONS)
        
        # 5. Morphological open - remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITERATIONS)
        
        # 6. Connected component analysis - keep only LARGEST component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if num_labels > 1:
            # find largest component (ignoring background which is label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(opened)
            mask[labels == largest_label] = 255
        else:
            mask = np.zeros_like(opened)
        
        # 7. Find contours, get largest by area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'mask': mask, 'bbox': (0, 0, 0, 0), 'contour': np.array([]), 'area': 0}
            
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 8. Return dict with mask, bbox, contour, area
        return {
            'mask': mask,
            'bbox': (x, y, w, h),
            'contour': largest_contour,
            'area': area
        }
