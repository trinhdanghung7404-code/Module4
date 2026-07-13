"""Extract polygon-bounded image regions for mesh triangle comparison."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from mesh_builder import MeshVertex


class RegionExtractor:
    """Extract triangle masks and pixel data without using bounding-box comparison."""

    @staticmethod
    def triangle_product_points(
        vertices: Sequence[MeshVertex],
        vertex_indices: Tuple[int, int, int],
    ) -> np.ndarray:
        points = np.asarray(
            [vertices[index].product_point for index in vertex_indices],
            dtype=np.int32,
        )
        return points.reshape((-1, 1, 2))

    @staticmethod
    def triangle_return_points(
        vertices: Sequence[MeshVertex],
        vertex_indices: Tuple[int, int, int],
    ) -> np.ndarray:
        points = np.asarray(
            [vertices[index].return_point for index in vertex_indices],
            dtype=np.int32,
        )
        return points.reshape((-1, 1, 2))

    def polygon_area(self, vertices: Sequence[MeshVertex], vertex_indices: Tuple[int, int, int]) -> float:
        """Compute triangle area in product image space."""
        points = np.asarray(
            [vertices[index].product_point for index in vertex_indices],
            dtype=np.float64,
        )
        return float(abs(cv2.contourArea(points.astype(np.float32))))

    def create_triangle_mask(
        self,
        image_shape: Tuple[int, int],
        vertices: Sequence[MeshVertex],
        vertex_indices: Tuple[int, int, int],
        space: str = "product",
    ) -> np.ndarray:
        """Fill a binary mask for the triangle polygon."""
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if space == "product":
            polygon = self.triangle_product_points(vertices, vertex_indices)
        elif space == "return":
            polygon = self.triangle_return_points(vertices, vertex_indices)
        else:
            raise ValueError(f"Unsupported space: {space}")

        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    def _mask_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _union_bounding_box(
        self,
        *boxes: Tuple[int, int, int, int] | None,
    ) -> Tuple[int, int, int, int] | None:
        valid_boxes = [box for box in boxes if box is not None]
        if not valid_boxes:
            return None

        x0 = min(box[0] for box in valid_boxes)
        y0 = min(box[1] for box in valid_boxes)
        x1 = max(box[2] for box in valid_boxes)
        y1 = max(box[3] for box in valid_boxes)
        return x0, y0, x1, y1

    def extract_masked_gray(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bounding_box: Tuple[int, int, int, int] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return grayscale image crop and matching mask inside the bounding rectangle."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if bounding_box is None:
            bounding_box = self._mask_bounding_box(mask)
        if bounding_box is None:
            empty = np.zeros((1, 1), dtype=np.uint8)
            return empty, empty

        x0, y0, x1, y1 = bounding_box
        crop = gray[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1].copy()
        return crop, crop_mask

    def extract_triangle_pair(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        vertices: Sequence[MeshVertex],
        vertex_indices: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract aligned masked grayscale crops for product and return triangles."""
        shape = product_image.shape
        product_mask = self.create_triangle_mask(shape, vertices, vertex_indices, space="product")
        return_mask = self.create_triangle_mask(shape, vertices, vertex_indices, space="return")

        bounding_box = self._union_bounding_box(
            self._mask_bounding_box(product_mask),
            self._mask_bounding_box(return_mask),
        )
        product_crop, product_crop_mask = self.extract_masked_gray(
            product_image,
            product_mask,
            bounding_box,
        )
        return_crop, return_crop_mask = self.extract_masked_gray(
            return_image,
            return_mask,
            bounding_box,
        )
        return product_crop, product_crop_mask, return_crop, return_crop_mask
