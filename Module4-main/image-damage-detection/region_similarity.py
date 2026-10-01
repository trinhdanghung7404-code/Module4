"""Robust region similarity metrics for adaptive mesh triangle comparison.

The old implementation compared two triangle crops at their raw pixel
positions. A small translation, scale change, or illumination difference could
therefore lower every metric even when the physical region was unchanged.

This version first maps both triangle masks to the same canonical triangle,
then searches a very small residual translation and performs robust
illumination normalisation. Metrics are computed only on corresponding overlap
pixels. Small isolated disagreements are trimmed so one tiny mark does not make
an entire triangle look damaged; pixel-level defect detection remains
responsible for locating those marks precisely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _intersection_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return (mask_a > 0) & (mask_b > 0)


def _trim_low(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Discard a small low-score tail caused by isolated mismatched pixels."""
    values = np.asarray(values, dtype=np.float32)
    if values.size < 20 or fraction <= 0.0:
        return values
    count = int(round(values.size * fraction))
    if count <= 0 or count >= values.size:
        return values
    return np.partition(values, count)[count:]


def _trim_high(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Discard a small high-error tail caused by isolated edge disagreements."""
    values = np.asarray(values, dtype=np.float32)
    if values.size < 20 or fraction <= 0.0:
        return values
    keep = values.size - int(round(values.size * fraction))
    if keep <= 0 or keep >= values.size:
        return values
    return np.partition(values, keep - 1)[:keep]


def _robust_location_scale(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad < 1.0:
        mad = float(values.std())
    return median, max(mad, 1.0)


def _normalise_return_to_product(
    product: np.ndarray,
    ret: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    """Match local median/contrast while limiting the allowed correction."""
    if np.count_nonzero(overlap) < 12:
        return ret.copy()

    p_loc, p_scale = _robust_location_scale(product[overlap])
    r_loc, r_scale = _robust_location_scale(ret[overlap])
    scale = float(np.clip(p_scale / r_scale, 0.75, 1.35))
    offset = float(np.clip(p_loc - r_loc, -35.0, 35.0))

    result = (ret.astype(np.float32) - r_loc) * scale + r_loc + offset
    return np.clip(result, 0, 255).astype(np.uint8)


def _triangle_points_from_mask(mask: np.ndarray) -> np.ndarray | None:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 3.0:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.02 * perimeter), True)
    if len(approx) == 3:
        points = approx.reshape(3, 2).astype(np.float32)
    else:
        _, enclosing = cv2.minEnclosingTriangle(contour.astype(np.float32))
        if enclosing is None:
            return None
        points = enclosing.reshape(3, 2).astype(np.float32)

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]

    # Use the top-most, then left-most point as a stable first vertex.
    first = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    points = np.roll(points, -first, axis=0)
    return points.astype(np.float32)


def _masked_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _canonicalise_triangle(
    crop: np.ndarray,
    mask: np.ndarray,
    size: int,
    margin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gray = _as_gray_u8(crop)
    binary = (mask > 0).astype(np.uint8) * 255
    points = _triangle_points_from_mask(binary)

    destination = np.asarray(
        [
            [margin, margin],
            [size - margin - 1, margin],
            [margin, size - margin - 1],
        ],
        dtype=np.float32,
    )

    if points is not None:
        matrix = cv2.getAffineTransform(points, destination)
        warped = cv2.warpAffine(
            gray,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            binary,
            matrix,
            (size, size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        bbox = _masked_bbox(binary)
        if bbox is None:
            empty = np.zeros((size, size), dtype=np.uint8)
            return empty, empty
        x0, y0, x1, y1 = bbox
        region = gray[y0:y1, x0:x1]
        region_mask = binary[y0:y1, x0:x1]
        warped = cv2.resize(region, (size, size), interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.resize(region_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # Remove the anti-aliased one-pixel triangle boundary from metric scoring.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode((warped_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
    if np.count_nonzero(eroded) >= 24:
        warped_mask = eroded
    else:
        warped_mask = (warped_mask > 0).astype(np.uint8) * 255
    return warped, warped_mask


def _shift_without_wrap(array: np.ndarray, dx: int, dy: int, fill_value: int = 0) -> np.ndarray:
    height, width = array.shape[:2]
    shifted = np.full_like(array, fill_value)

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return shifted

    dst_x0 = src_x0 + dx
    dst_x1 = src_x1 + dx
    dst_y0 = src_y0 + dy
    dst_y1 = src_y1 + dy
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _ncc_on_overlap(product: np.ndarray, ret: np.ndarray, overlap: np.ndarray) -> float:
    p = product[overlap].astype(np.float32)
    r = ret[overlap].astype(np.float32)
    if p.size < 8 or r.size < 8:
        return 0.0
    p -= float(p.mean())
    r -= float(r.mean())
    denominator = float(np.linalg.norm(p) * np.linalg.norm(r))
    if denominator <= 1e-8:
        # Flat regions should not be treated as damaged merely because NCC is undefined.
        return 1.0 if abs(float(product[overlap].mean()) - float(ret[overlap].mean())) < 4.0 else 0.0
    return float(np.clip((float(np.dot(p, r) / denominator) + 1.0) * 0.5, 0.0, 1.0))


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

    def __init__(self, trim_fraction: float = 0.10):
        self.trim_fraction = float(np.clip(trim_fraction, 0.0, 0.30))

    def compute(self, product_crop, return_crop, product_mask, return_mask) -> float:
        overlap = _intersection_mask(product_mask, return_mask)
        if np.count_nonzero(overlap) < 12:
            return 0.0

        minimum_extent = min(product_crop.shape[:2])
        win_size = min(7, minimum_extent if minimum_extent % 2 == 1 else minimum_extent - 1)
        if win_size < 3:
            return _ncc_on_overlap(product_crop, return_crop, overlap)

        try:
            _, score_map = structural_similarity(
                product_crop,
                return_crop,
                data_range=255,
                full=True,
                win_size=win_size,
            )
        except ValueError:
            return _ncc_on_overlap(product_crop, return_crop, overlap)

        values = _trim_low(score_map[overlap], self.trim_fraction)
        if values.size == 0:
            return 0.0
        return float(np.clip(values.mean(), 0.0, 1.0))


class NCCMetric(SimilarityMetric):
    name = "ncc"

    def compute(self, product_crop, return_crop, product_mask, return_mask) -> float:
        overlap = _intersection_mask(product_mask, return_mask)
        return _ncc_on_overlap(product_crop, return_crop, overlap)


class GradientMagnitudeMetric(SimilarityMetric):
    name = "gradient_magnitude"

    @staticmethod
    def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        grad_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
        return cv2.magnitude(grad_x, grad_y)

    def compute(self, product_crop, return_crop, product_mask, return_mask) -> float:
        overlap = _intersection_mask(product_mask, return_mask)
        if np.count_nonzero(overlap) < 12:
            return 0.0

        product_grad = self._gradient_magnitude(product_crop)
        return_grad = self._gradient_magnitude(return_crop)
        combined = np.concatenate([product_grad[overlap], return_grad[overlap]])
        scale = float(np.percentile(combined, 95)) if combined.size else 0.0
        if scale <= 1e-6:
            return 1.0

        difference = np.abs(product_grad - return_grad) * (255.0 / scale)
        values = _trim_high(np.clip(difference[overlap], 0.0, 255.0), 0.10)
        if values.size == 0:
            return 0.0
        robust_error = float(values.mean())
        return float(np.clip(1.0 - robust_error / 255.0, 0.0, 1.0))


class GradientOrientationMetric(SimilarityMetric):
    name = "gradient_orientation"

    def compute(self, product_crop, return_crop, product_mask, return_mask) -> float:
        overlap = _intersection_mask(product_mask, return_mask)
        if np.count_nonzero(overlap) < 12:
            return 0.0

        product_blur = cv2.GaussianBlur(product_crop, (3, 3), 0)
        return_blur = cv2.GaussianBlur(return_crop, (3, 3), 0)
        pgx = cv2.Scharr(product_blur, cv2.CV_32F, 1, 0)
        pgy = cv2.Scharr(product_blur, cv2.CV_32F, 0, 1)
        rgx = cv2.Scharr(return_blur, cv2.CV_32F, 1, 0)
        rgy = cv2.Scharr(return_blur, cv2.CV_32F, 0, 1)

        product_magnitude = cv2.magnitude(pgx, pgy)
        return_magnitude = cv2.magnitude(rgx, rgy)
        strength_values = np.concatenate([
            product_magnitude[overlap],
            return_magnitude[overlap],
        ])
        if strength_values.size == 0:
            return 1.0
        threshold = max(12.0, float(np.percentile(strength_values, 60)))
        valid = overlap & (product_magnitude >= threshold) & (return_magnitude >= threshold)
        if np.count_nonzero(valid) < 8:
            return 1.0

        product_angle = np.mod(cv2.phase(pgx, pgy, angleInDegrees=True), 180.0)
        return_angle = np.mod(cv2.phase(rgx, rgy, angleInDegrees=True), 180.0)
        angle_diff = np.abs(product_angle - return_angle)
        angle_diff = np.minimum(angle_diff, 180.0 - angle_diff)
        values = _trim_high(angle_diff[valid], 0.10)
        mean_diff = float(values.mean()) if values.size else 0.0
        return float(np.clip(1.0 - mean_diff / 90.0, 0.0, 1.0))


class RegionSimilarity:
    """Compute a robust, small-misalignment-tolerant triangle similarity."""

    DEFAULT_WEIGHTS = {
        "ssim": 0.45,
        "ncc": 0.35,
        "gradient_magnitude": 0.15,
        "gradient_orientation": 0.05,
    }

    def __init__(
        self,
        metrics: Sequence[SimilarityMetric] | None = None,
        weights: Dict[str, float] | None = None,
        canonical_size: int = 72,
        residual_search_radius: int = 2,
        minimum_overlap_ratio: float = 0.55,
    ):
        self.metrics: List[SimilarityMetric] = list(metrics) if metrics is not None else [
            SSIMMetric(),
            NCCMetric(),
            GradientMagnitudeMetric(),
            GradientOrientationMetric(),
        ]
        self.weights = dict(weights) if weights is not None else dict(self.DEFAULT_WEIGHTS)
        self.canonical_size = max(24, int(canonical_size))
        self.residual_search_radius = max(0, int(residual_search_radius))
        self.minimum_overlap_ratio = float(np.clip(minimum_overlap_ratio, 0.10, 1.0))

    def _prepare_pair(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float]:
        margin = max(2, int(round(self.canonical_size * 0.06)))
        product, product_canonical_mask = _canonicalise_triangle(
            product_crop,
            product_mask,
            self.canonical_size,
            margin,
        )
        ret, return_canonical_mask = _canonicalise_triangle(
            return_crop,
            return_mask,
            self.canonical_size,
            margin,
        )

        minimum_area = max(
            1,
            min(
                int(np.count_nonzero(product_canonical_mask)),
                int(np.count_nonzero(return_canonical_mask)),
            ),
        )
        best_score = -1.0
        best = (ret, return_canonical_mask, 0, 0, 0.0)

        for dy in range(-self.residual_search_radius, self.residual_search_radius + 1):
            for dx in range(-self.residual_search_radius, self.residual_search_radius + 1):
                shifted_return = _shift_without_wrap(ret, dx, dy, 0)
                shifted_mask = _shift_without_wrap(return_canonical_mask, dx, dy, 0)
                overlap = _intersection_mask(product_canonical_mask, shifted_mask)
                overlap_count = int(np.count_nonzero(overlap))
                overlap_ratio = overlap_count / float(minimum_area)
                if overlap_count < 12 or overlap_ratio < self.minimum_overlap_ratio:
                    continue

                normalised_return = _normalise_return_to_product(
                    product,
                    shifted_return,
                    overlap,
                )
                ncc = _ncc_on_overlap(product, normalised_return, overlap)

                # A light blur makes the alignment search insensitive to one-pixel texture noise.
                p_blur = cv2.GaussianBlur(product, (3, 3), 0)
                r_blur = cv2.GaussianBlur(normalised_return, (3, 3), 0)
                mae = float(np.mean(np.abs(
                    p_blur[overlap].astype(np.float32)
                    - r_blur[overlap].astype(np.float32)
                )))
                appearance = float(np.clip(1.0 - mae / 96.0, 0.0, 1.0))
                search_score = 0.75 * ncc + 0.20 * appearance + 0.05 * overlap_ratio

                if search_score > best_score:
                    best_score = search_score
                    best = (
                        normalised_return,
                        shifted_mask,
                        dx,
                        dy,
                        overlap_ratio,
                    )

        return (
            product,
            best[0],
            product_canonical_mask,
            best[1],
            int(best[2]),
            int(best[3]),
            float(best[4]),
        )

    def compute(
        self,
        product_crop: np.ndarray,
        return_crop: np.ndarray,
        product_mask: np.ndarray,
        return_mask: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted similarity and a diagnostic metric breakdown."""
        if product_crop.size == 0 or return_crop.size == 0:
            return 0.0, {
                "alignment_dx": 0.0,
                "alignment_dy": 0.0,
                "overlap_ratio": 0.0,
            }

        (
            product,
            ret,
            product_aligned_mask,
            return_aligned_mask,
            alignment_dx,
            alignment_dy,
            overlap_ratio,
        ) = self._prepare_pair(
            product_crop,
            return_crop,
            product_mask,
            return_mask,
        )

        scores: Dict[str, float] = {
            "alignment_dx": float(alignment_dx),
            "alignment_dy": float(alignment_dy),
            "overlap_ratio": float(overlap_ratio),
        }
        weighted_sum = 0.0
        weight_sum = 0.0

        for metric in self.metrics:
            score = metric.compute(
                product,
                ret,
                product_aligned_mask,
                return_aligned_mask,
            )
            scores[metric.name] = float(score)
            weight = float(self.weights.get(metric.name, 0.0))
            weighted_sum += weight * score
            weight_sum += weight

        if weight_sum <= 0.0 or overlap_ratio < self.minimum_overlap_ratio:
            return 0.0, scores

        combined = float(np.clip(weighted_sum / weight_sum, 0.0, 1.0))
        scores["combined"] = combined
        return combined, scores
