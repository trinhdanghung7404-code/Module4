from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureSimilaritySummary:
    feature_similarities: List[float]
    mean_similarity: float
    median_similarity: float
    min_similarity: float
    max_similarity: float


class FeatureRepresentation:
    def _log_step(self, message: str) -> None:

        print(f"[FeatureRepresentation] {message}")

    def compare(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        matches: Sequence,
    ) -> Dict:
        if descriptors1 is None or descriptors2 is None or matches is None:
            return self._empty_result()

        descriptors1 = np.asarray(descriptors1, dtype=np.float32)
        descriptors2 = np.asarray(descriptors2, dtype=np.float32)

        if (
            descriptors1.ndim != 2
            or descriptors2.ndim != 2
            or descriptors1.shape[0] == 0
            or descriptors2.shape[0] == 0
            or len(matches) == 0
        ):
            return self._empty_result()

        feature_similarities: List[float] = []

        for match in matches:
            if match.queryIdx >= descriptors1.shape[0] or match.trainIdx >= descriptors2.shape[0]:
                continue

            descriptor_product = descriptors1[match.queryIdx]
            descriptor_return = descriptors2[match.trainIdx]
            similarity = float(np.dot(descriptor_product, descriptor_return))
            feature_similarities.append(similarity)

        if not feature_similarities:
            result = self._empty_result()
            self._log_step("matches=0 | mean_similarity=0.000 | median_similarity=0.000 | min_similarity=0.000 | max_similarity=0.000")
            return result

        similarities = np.asarray(feature_similarities, dtype=np.float32)
        summary = FeatureSimilaritySummary(
            feature_similarities=feature_similarities,
            mean_similarity=float(np.mean(similarities)),
            median_similarity=float(np.median(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
        )

        result = {
            "feature_similarities": summary.feature_similarities,
            "mean_similarity": summary.mean_similarity,
            "median_similarity": summary.median_similarity,
            "min_similarity": summary.min_similarity,
            "max_similarity": summary.max_similarity,
            "matches": len(feature_similarities),
        }

        self._log_step(
            f"matches={result['matches']} | "
            f"mean_similarity={result['mean_similarity']:.3f} | "
            f"median_similarity={result['median_similarity']:.3f} | "
            f"min_similarity={result['min_similarity']:.3f} | "
            f"max_similarity={result['max_similarity']:.3f}"
        )

        return result

    def _empty_result(self) -> Dict:
        return {
            "feature_similarities": [],
            "mean_similarity": 0.0,
            "median_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "matches": 0,
        }