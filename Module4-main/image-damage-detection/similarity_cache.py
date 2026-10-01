"""Cache region-similarity scores per triangle id."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from mesh_builder import MeshTriangle


@dataclass
class ScoredTriangle:
    """Triangle with precomputed geometry and similarity metrics."""
    
    triangle: MeshTriangle
    area: float
    similarity: float
    feature_similarity: float
    metric_scores: Dict[str, float]
    is_damaged: bool

    @property
    def triangle_id(self) -> int:
        return self.triangle.triangle_id

    @property
    def level(self) -> int:
        return self.triangle.depth


class SimilarityCache:
    """Maps triangle_id → ScoredTriangle. Avoids recomputing unchanged triangles."""
    
    def __init__(self) -> None:
        self._entries: Dict[int, ScoredTriangle] = {}

    def has(self, triangle_id: int) -> bool:
        return triangle_id in self._entries

    def get(self, triangle_id: int) -> Optional[ScoredTriangle]:
        return self._entries.get(triangle_id)

    def set(self, scored: ScoredTriangle) -> None:
        self._entries[scored.triangle_id] = scored

    def __len__(self) -> int:
        return len(self._entries)

    def values(self):
        return self._entries.values()