"""Build triangular mesh from keypoint correspondences.

Delaunay triangulation is computed on product-space coordinates only.
Return-space coordinates follow the same vertex / triangle indices.
Designed so the mesh backend can later be swapped (e.g. Voronoi).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class MeshVertex:
    """A paired vertex in product and return image space.

    product_keypoint_index / return_keypoint_index point to the original
    SuperPoint descriptor rows. Synthetic vertices, such as interpolated grid
    points or edge midpoints, must keep these as None so they are not treated
    as real feature correspondences.
    """

    product_point: Tuple[float, float]
    return_point: Tuple[float, float]
    keypoint_index: Optional[int] = None
    product_keypoint_index: Optional[int] = None
    return_keypoint_index: Optional[int] = None


@dataclass
class MeshTriangle:
    """Triangle defined by three vertex indices into a shared vertex list."""

    vertex_indices: Tuple[int, int, int]
    triangle_id: int
    depth: int = 0
    parent_id: Optional[int] = None
    refined: bool = False


class MeshBuilder:
    """Construct a Delaunay triangulation mesh from paired keypoints."""

    def build_vertices(
        self,
        product_points: Sequence[Sequence[float]],
        return_points: Sequence[Sequence[float]],
        keypoint_indices: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
    ) -> List[MeshVertex]:
        """Create paired vertices from product/return coordinate pairs.

        keypoint_indices contains original SuperPoint descriptor indices as
        (queryIdx, trainIdx). Use None for synthetic vertices.
        """
        if len(product_points) != len(return_points):
            raise ValueError("product_points and return_points must have the same length")

        if keypoint_indices is not None and len(keypoint_indices) != len(product_points):
            raise ValueError("keypoint_indices must have the same length as product_points")

        vertices: List[MeshVertex] = []
        for index, (product_point, return_point) in enumerate(zip(product_points, return_points)):
            pair = keypoint_indices[index] if keypoint_indices is not None else (index, index)
            product_keypoint_index = pair[0] if pair is not None else None
            return_keypoint_index = pair[1] if pair is not None else None
            legacy_index = index if pair is not None and pair[0] == pair[1] else None

            vertices.append(
                MeshVertex(
                    product_point=(float(product_point[0]), float(product_point[1])),
                    return_point=(float(return_point[0]), float(return_point[1])),
                    keypoint_index=legacy_index,
                    product_keypoint_index=product_keypoint_index,
                    return_keypoint_index=return_keypoint_index,
                )
            )
        return vertices

    def build_triangles(self, vertices: Sequence[MeshVertex]) -> List[MeshTriangle]:
        """Run Delaunay triangulation on product-space vertex coordinates."""
        if len(vertices) < 3:
            return []

        product_coords = np.asarray(
            [[vertex.product_point[0], vertex.product_point[1]] for vertex in vertices],
            dtype=np.float64,
        )

        delaunay = Delaunay(product_coords)
        triangles: List[MeshTriangle] = []

        for triangle_id, simplex in enumerate(delaunay.simplices):
            indices = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
            triangles.append(
                MeshTriangle(
                    vertex_indices=indices,
                    triangle_id=triangle_id,
                    depth=0,
                    parent_id=None,
                    refined=False,
                )
            )

        return triangles

    def build(
        self,
        product_points: Sequence[Sequence[float]],
        return_points: Sequence[Sequence[float]],
        keypoint_indices: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
    ) -> Tuple[List[MeshVertex], List[MeshTriangle]]:
        """Build paired vertices and Delaunay triangles on product points."""
        vertices = self.build_vertices(product_points, return_points, keypoint_indices)
        triangles = self.build_triangles(vertices)
        return vertices, triangles
