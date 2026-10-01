from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract interface for feature extractors.

    Implementations:
        - SuperPointExtractor  (superpoint_extractor.py)
        - SIFTExtractor        (future)
        - DINOv2Extractor      (future)

    All implementations must:
        - Output keypoints as list of [x, y] float coordinates (image space)
        - Output descriptors as float32 numpy array of shape (N, D), L2-normalized
        - Return matches compatible with cv2.DMatch for RANSAC/TPS pipeline
    """

    @abstractmethod
    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[float]], np.ndarray]:
        """Detect keypoints and compute descriptors.

        Args:
            image: BGR image (H × W × 3) or grayscale (H × W).
            mask:  Optional binary mask (H × W), nonzero = valid region.

        Returns:
            keypoints:   list of [x, y] coordinates in image space.
            descriptors: numpy array (N, D) float32, L2-normalized.
                         Returns ([], zeros((0, D))) when no keypoints found.
        """

    @abstractmethod
    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
    ) -> Dict:
        """Match descriptors between two images.

        Args:
            descriptors1: (N1, D) float32, L2-normalized.
            descriptors2: (N2, D) float32, L2-normalized.

        Returns dict with keys:
            raw_matches:      list[cv2.DMatch] — all candidate pairs (one per desc1 entry).
            selected_matches: list[cv2.DMatch] — filtered matches ready for RANSAC.
            ratio_threshold:  float | None      — Lowe threshold (None if unused).
            match_strategy:   str               — e.g. "mnn", "ratio", "top_similarity_fallback".
        """

    # ------------------------------------------------------------------
    # Metadata properties — required for DB schema enrichment
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def descriptor_dim(self) -> int:
        """Dimension of the descriptor vector (e.g. 128 for SIFT, 256 for SuperPoint)."""

    @property
    @abstractmethod
    def feature_type(self) -> str:
        """Unique lowercase string identifier, e.g. "sift", "superpoint", "dinov2"."""

    @property
    @abstractmethod
    def extractor_version(self) -> str:
        """Model / version string, e.g. "magic-leap-community/superpoint@transformers"."""
