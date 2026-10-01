"""Adaptive triangle subdivision for low-similarity mesh regions."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from mesh_builder import MeshTriangle, MeshVertex


class MeshRefiner:
    """Subdivide triangles when region similarity is below threshold."""

    STRATEGY_MIDPOINT = "midpoint"
    STRATEGY_LONGEST_EDGE = "longest_edge"

    def __init__(self, strategy: str = STRATEGY_MIDPOINT):
        if strategy not in {self.STRATEGY_MIDPOINT, self.STRATEGY_LONGEST_EDGE}:
            raise ValueError(f"Unsupported refinement strategy: {strategy}")
        self.strategy = strategy

    @staticmethod
    def _append_midpoint_vertex(
        vertices: List[MeshVertex],
        index_a: int,
        index_b: int,
        midpoint_cache: dict,
    ) -> int:
        cache_key = tuple(sorted((index_a, index_b)))
        if cache_key in midpoint_cache:
            return midpoint_cache[cache_key]

        vertex_a = vertices[index_a]
        vertex_b = vertices[index_b]
        midpoint = MeshVertex(
            product_point=(
                (vertex_a.product_point[0] + vertex_b.product_point[0]) * 0.5,
                (vertex_a.product_point[1] + vertex_b.product_point[1]) * 0.5,
            ),
            return_point=(
                (vertex_a.return_point[0] + vertex_b.return_point[0]) * 0.5,
                (vertex_a.return_point[1] + vertex_b.return_point[1]) * 0.5,
            ),
            keypoint_index=None,
            product_keypoint_index=None,
            return_keypoint_index=None,
        )
        midpoint_index = len(vertices)
        vertices.append(midpoint)
        midpoint_cache[cache_key] = midpoint_index
        return midpoint_index

    def subdivide_midpoint(
        self,
        triangle: MeshTriangle,
        vertices: List[MeshVertex],
        midpoint_cache: dict,
        next_triangle_id: int,
    ) -> Tuple[List[MeshTriangle], int]:
        """Split one triangle into four sub-triangles using edge midpoints."""
        a, b, c = triangle.vertex_indices
        ab = self._append_midpoint_vertex(vertices, a, b, midpoint_cache)
        bc = self._append_midpoint_vertex(vertices, b, c, midpoint_cache)
        ac = self._append_midpoint_vertex(vertices, a, c, midpoint_cache)

        child_specs = [
            (a, ab, ac),
            (ab, b, bc),
            (ac, bc, c),
            (ab, bc, ac),
        ]

        children: List[MeshTriangle] = []
        triangle_id = next_triangle_id
        for vertex_indices in child_specs:
            children.append(
                MeshTriangle(
                    vertex_indices=vertex_indices,
                    triangle_id=triangle_id,
                    depth=triangle.depth + 1,
                    parent_id=triangle.triangle_id,
                    refined=True,
                )
            )
            triangle_id += 1

        triangle.refined = True
        return children, triangle_id

    @staticmethod
    def _edge_length(
        vertices: Sequence[MeshVertex],
        index_a: int,
        index_b: int,
    ) -> float:
        point_a = np.asarray(vertices[index_a].product_point, dtype=np.float64)
        point_b = np.asarray(vertices[index_b].product_point, dtype=np.float64)
        return float(np.linalg.norm(point_a - point_b))

    def subdivide_longest_edge(
        self,
        triangle: MeshTriangle,
        vertices: List[MeshVertex],
        midpoint_cache: dict,
        next_triangle_id: int,
    ) -> Tuple[List[MeshTriangle], int]:
        """Split one triangle into two sub-triangles across its longest edge."""
        a, b, c = triangle.vertex_indices
        edges = ((a, b), (b, c), (c, a))
        longest = max(edges, key=lambda pair: self._edge_length(vertices, pair[0], pair[1]))
        p0, p1 = longest
        opposite = {a, b, c}.difference({p0, p1}).pop()
        midpoint = self._append_midpoint_vertex(vertices, p0, p1, midpoint_cache)

        children = [
            MeshTriangle(
                vertex_indices=(p0, midpoint, opposite),
                triangle_id=next_triangle_id,
                depth=triangle.depth + 1,
                parent_id=triangle.triangle_id,
                refined=True,
            ),
            MeshTriangle(
                vertex_indices=(midpoint, p1, opposite),
                triangle_id=next_triangle_id + 1,
                depth=triangle.depth + 1,
                parent_id=triangle.triangle_id,
                refined=True,
            ),
        ]
        triangle.refined = True
        return children, next_triangle_id + 2

    def subdivide(
        self,
        triangle: MeshTriangle,
        vertices: List[MeshVertex],
        midpoint_cache: dict,
        next_triangle_id: int,
    ) -> Tuple[List[MeshTriangle], int]:
        if self.strategy == self.STRATEGY_LONGEST_EDGE:
            return self.subdivide_longest_edge(triangle, vertices, midpoint_cache, next_triangle_id)
        return self.subdivide_midpoint(triangle, vertices, midpoint_cache, next_triangle_id)
