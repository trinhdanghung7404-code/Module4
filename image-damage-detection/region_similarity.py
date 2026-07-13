"""Pluggable region similarity metrics for mesh triangle comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity


class SimilarityMetric(ABC):
    """Interface for region comparison metrics."""

    name: str

    @abstractmethod
    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> float:
        """Return similarity in [0, 1]; higher means more similar."""


class SSIMMetric(SimilarityMetric):
    name = "ssim"

    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> float:
        if product_crop.size == 0 or return_crop.size == 0:
            return 0.0

        height = max(product_crop.shape[0], return_crop.shape[0])
        width = max(product_crop.shape[1], return_crop.shape[1])
        product = np.zeros((height, width), dtype=np.uint8)
        ret = np.zeros((height, width), dtype=np.uint8)
        product[: product_crop.shape[0], : product_crop.shape[1]] = product_crop
        ret[: return_crop.shape[0], : return_crop.shape[1]] = return_crop

        try:
            score = structural_similarity(product, ret, data_range=255)
        except ValueError:
            return 0.0
        return float(np.clip(score, 0.0, 1.0))


class NCCMetric(SimilarityMetric):
    name = "ncc"

    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> float:
        product_values = product_crop[product_mask > 0].astype(np.float32)
        return_values = return_crop[return_mask > 0].astype(np.float32)
        if product_values.size < 3 or return_values.size < 3:
            return 0.0

        count = min(product_values.size, return_values.size)
        product_values = product_values[:count]
        return_values = return_values[:count]

        product_values -= float(product_values.mean())
        return_values -= float(return_values.mean())
        denominator = float(np.linalg.norm(product_values) * np.linalg.norm(return_values))
        if denominator <= 1e-8:
            return 0.0
        ncc = float(np.dot(product_values, return_values) / denominator)
        return float(np.clip((ncc + 1.0) * 0.5, 0.0, 1.0))


class GradientMagnitudeMetric(SimilarityMetric):
    name = "gradient_magnitude"

    def _gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        return cv2.magnitude(grad_x, grad_y)

    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> float:
        if product_crop.size == 0 or return_crop.size == 0:
            return 0.0

        product_grad = self._gradient_magnitude(product_crop)
        return_grad = self._gradient_magnitude(return_crop)
        diff = cv2.absdiff(
            cv2.normalize(product_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.normalize(return_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        )
        mean_diff = float(diff[product_mask > 0].mean()) if np.any(product_mask > 0) else 255.0
        return float(np.clip(1.0 - mean_diff / 255.0, 0.0, 1.0))


class GradientOrientationMetric(SimilarityMetric):
    name = "gradient_orientation"

    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> float:
        if product_crop.size == 0 or return_crop.size == 0:
            return 0.0

        product_grad_x = cv2.Scharr(product_crop, cv2.CV_32F, 1, 0)
        product_grad_y = cv2.Scharr(product_crop, cv2.CV_32F, 0, 1)
        return_grad_x = cv2.Scharr(return_crop, cv2.CV_32F, 1, 0)
        return_grad_y = cv2.Scharr(return_crop, cv2.CV_32F, 0, 1)

        product_angle = cv2.phase(product_grad_x, product_grad_y, angleInDegrees=True)
        return_angle = cv2.phase(return_grad_x, return_grad_y, angleInDegrees=True)
        angle_diff = np.abs(product_angle - return_angle)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        masked_diff = angle_diff[product_mask > 0]
        if masked_diff.size == 0:
            return 0.0
        mean_diff = float(masked_diff.mean())
        return float(np.clip(1.0 - mean_diff / 180.0, 0.0, 1.0))


class RegionSimilarity:
    """Combine multiple classical metrics into one region similarity score."""

    DEFAULT_WEIGHTS = {
        "ssim": 0.40,
        "ncc": 0.30,
        "gradient_magnitude": 0.20,
        "gradient_orientation": 0.10,
    }

    def __init__(
        self,
        metrics: Sequence[SimilarityMetric] | None = None,
        weights: Dict[str, float] | None = None,
    ):
        self.metrics: List[SimilarityMetric] = list(metrics) if metrics is not None else [
            SSIMMetric(),
            NCCMetric(),
            GradientMagnitudeMetric(),
            GradientOrientationMetric(),
        ]
        self.weights = dict(weights) if weights is not None else dict(self.DEFAULT_WEIGHTS)

    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted similarity and per-metric breakdown."""
        scores: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for metric in self.metrics:
            score = metric.compute(product_crop, return_crop, product_mask, return_mask)
            scores[metric.name] = score
            weight = float(self.weights.get(metric.name, 0.0))
            weighted_sum += weight * score
            weight_sum += weight

        if weight_sum <= 0.0:
            return 0.0, scores

        return float(np.clip(weighted_sum / weight_sum, 0.0, 1.0)), scores
