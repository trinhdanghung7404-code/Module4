import cv2
import numpy as np
import config

class ImageRegistration:
    """Finds corresponding keypoints between product and return images.
    Uses SuperPoint for detection, MNN for matching, RANSAC for outlier rejection.
    NO image warping — only point correspondences."""
    
    def __init__(self, max_keypoints=None):
        if max_keypoints is None:
            max_keypoints = config.SUPERPOINT_MAX_KEYPOINTS
        from features.superpoint_extractor import SuperPointExtractor
        self.extractor = SuperPointExtractor(
            max_keypoints=max_keypoints,
            confidence_threshold=config.SUPERPOINT_CONFIDENCE,
            nms_radius=config.SUPERPOINT_NMS_RADIUS
        )
    
    def _extract_roi_keypoints(self, image: np.ndarray, mask: np.ndarray):
        """Extract keypoints on padded bounding box ROI as in original feature_extractor.py."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], np.zeros((0, 256), dtype=np.float32)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        padding = max(8, int(min(w, h) * 0.08))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + w + padding)
        y1 = min(image.shape[0], y + h + padding)

        roi_img = image[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1]
        kps, descs = self.extractor.detect_and_compute(roi_img, roi_mask)
        full_kps = [[float(kp[0] + x0), float(kp[1] + y0)] for kp in kps]
        descs_arr = np.asarray(descs, dtype=np.float32) if descs is not None and len(descs) > 0 else np.zeros((0, 256), dtype=np.float32)
        return full_kps, descs_arr

    def register(self, product_img: np.ndarray, return_img: np.ndarray, 
                 product_mask: np.ndarray, return_mask: np.ndarray) -> dict:
        """
        Find inlier point correspondences between two images.
        """
        # 1. detect_and_compute on padded ROI to get maximum pattern resolution
        prod_kp_list, prod_desc = self._extract_roi_keypoints(product_img, product_mask)
        ret_kp_list, ret_desc = self._extract_roi_keypoints(return_img, return_mask)
        
        if len(prod_kp_list) < config.MIN_INLIERS or len(ret_kp_list) < config.MIN_INLIERS:
            raise ValueError("Not enough keypoints found in one or both images.")
            
        # 2. match descriptors (MNN)
        match_result = self.extractor.match(np.array(prod_desc), np.array(ret_desc))
        matches = match_result.get("selected_matches", [])
        
        if len(matches) < config.MIN_INLIERS:
            raise ValueError(f"Not enough matches found. Needed {config.MIN_INLIERS}, got {len(matches)}.")
            
        # 3. Extract matched point coordinates
        prod_pts = np.array([prod_kp_list[m.queryIdx] for m in matches], dtype=np.float32)
        ret_pts = np.array([ret_kp_list[m.trainIdx] for m in matches], dtype=np.float32)
        
        prod_desc_matched = np.array([prod_desc[m.queryIdx] for m in matches])
        ret_desc_matched = np.array([ret_desc[m.trainIdx] for m in matches])
        
        # 4. cv2.findHomography with RANSAC, reprojThreshold=5.0
        #    → get inlier mask
        H, inlier_mask = cv2.findHomography(
            prod_pts, ret_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD
        )
        
        if inlier_mask is None:
            raise ValueError("Homography could not be computed.")
            
        inlier_mask = inlier_mask.ravel().astype(bool)
        
        # 5. Filter to keep only inlier matches
        prod_inliers = prod_pts[inlier_mask]
        ret_inliers = ret_pts[inlier_mask]
        prod_desc_inliers = prod_desc_matched[inlier_mask]
        ret_desc_inliers = ret_desc_matched[inlier_mask]
        
        inlier_count = int(np.sum(inlier_mask))
        
        # 6. If < MIN_INLIERS, raise ValueError
        if inlier_count < config.MIN_INLIERS:
            raise ValueError(f"Not enough inliers after RANSAC. Needed {config.MIN_INLIERS}, got {inlier_count}.")
            
        # 7. Return dict
        return {
            'product_points': prod_inliers,
            'return_points': ret_inliers,
            'product_descriptors': prod_desc_inliers,
            'return_descriptors': ret_desc_inliers,
            'inlier_count': inlier_count,
            'total_matches': len(matches),
            'all_product_keypoints': prod_kp_list,
            'all_return_keypoints': ret_kp_list,
            'matches': matches,
            'inlier_mask': inlier_mask
        }
