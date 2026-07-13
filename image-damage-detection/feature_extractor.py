import cv2
import numpy as np

from geometry import GeometryFeature
from superpoint_extractor import SuperPointExtractor


class FeatureExtractor:
    """High-level feature extractor.

    Wraps SuperPointExtractor and GeometryFeature to produce the full
    feature dict consumed by DamageComparator and Database.

    texture_feature output format (stored as JSON in DB):
        {
            "feature_type":     "superpoint",
            "descriptor_dim":   256,
            "extractor_version": "magic-leap-community/superpoint@transformers",
            "keypoint_count":   N,
            "keypoints":        [[x1,y1], [x2,y2], ...],
            "descriptors":      [[256 floats], ...],
            "bbox":             [x0, y0, w, h]
        }

    The feature_type / descriptor_dim / extractor_version fields enable the
    database to distinguish between feature sets from different extractors.
    See db_schema_feature_store.sql for the recommended future schema.
    """

    def __init__(self):
        self.geometry   = GeometryFeature()
        self._extractor = SuperPointExtractor(max_keypoints=None)

    # ------------------------------------------------------------------

    def _extract_texture_feature(self, image: np.ndarray, mask: np.ndarray, bbox: list) -> dict:
        x, y, w, h = bbox

        # Add padding around the object bounding box
        padding = max(8, int(min(w, h) * 0.08))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + w + padding)
        y1 = min(image.shape[0], y + h + padding)

        roi_image = image[y0:y1, x0:x1]
        roi_mask  = mask[y0:y1, x0:x1]

        empty = {
            "feature_type":     self._extractor.feature_type,
            "descriptor_dim":   self._extractor.descriptor_dim,
            "extractor_version": self._extractor.extractor_version,
            "keypoint_count":   0,
            "keypoints":        [],
            "descriptors":      [],
            "bbox":             bbox,
        }

        if roi_image.size == 0 or roi_mask.size == 0:
            return empty

        # Detect keypoints + compute 256D descriptors inside ROI
        keypoints_roi, descriptors = self._extractor.detect_and_compute(roi_image, roi_mask)

        if not keypoints_roi:
            return {**empty, "bbox": [x0, y0, x1 - x0, y1 - y0]}

        # Translate keypoints from ROI coordinates to full-image coordinates
        keypoints_full = [
            [float(kp[0] + x0), float(kp[1] + y0)]
            for kp in keypoints_roi
        ]

        descriptor_list = descriptors.tolist() if descriptors.ndim == 2 and descriptors.size > 0 else []

        return {
            "feature_type":     self._extractor.feature_type,
            "descriptor_dim":   self._extractor.descriptor_dim,
            "extractor_version": self._extractor.extractor_version,
            "keypoint_count":   len(keypoints_full),
            "keypoints":        keypoints_full,
            "descriptors":      descriptor_list,
            "bbox":             [x0, y0, x1 - x0, y1 - y0],
        }

    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        image_path = image_path.strip()

        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Cannot open image.")

        geometry        = self.geometry.extract(image)
        mask            = self.geometry.build_mask(image)
        texture_feature = self._extract_texture_feature(image, mask, geometry["bbox"])

        return {
            "image_path": image_path,
            **geometry,
            "texture_feature": texture_feature,
        }