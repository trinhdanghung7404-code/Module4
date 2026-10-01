from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from features.base_extractor import BaseFeatureExtractor


class SuperPointExtractor(BaseFeatureExtractor):
    """SuperPoint feature extractor using HuggingFace transformers.

    Model   : magic-leap-community/superpoint
    Descriptor: 256-dimensional, L2-normalized
    Matching  : Mutual Nearest Neighbor (MNN) on cosine similarity
                → replaces Lowe Ratio Test used by SIFT

    Weights are downloaded automatically on first use (~27 MB) and cached in
    ~/.cache/huggingface/hub. Subsequent runs are fully offline.

    Usage:
        extractor = SuperPointExtractor(max_keypoints=300)
        keypoints, descriptors = extractor.detect_and_compute(bgr_image, mask)
        match_result = extractor.match(desc1, desc2)
    """

    MODEL_ID          = "magic-leap-community/superpoint"
    _EXTRACTOR_VERSION = "magic-leap-community/superpoint@transformers"
    _DESCRIPTOR_DIM    = 256

    def __init__(
        self,
        max_keypoints: int = 300,
        confidence_threshold: float = 0.005,
        nms_radius: int = 4,
        similarity_threshold: float = 0.0,
    ):
        """
        Args:
            max_keypoints:        Keep top-N keypoints by confidence score.
            confidence_threshold: Minimum keypoint score to keep.
            nms_radius:           Non-maximum suppression radius (pixels).
            similarity_threshold: Minimum cosine similarity to keep a mutual match.
                                  0.0 means keep all mutual matches.
        """
        self.max_keypoints         = max_keypoints
        self.confidence_threshold  = confidence_threshold
        self.nms_radius            = nms_radius
        self.similarity_threshold  = similarity_threshold

        # Lazy-loaded — do not import/download until first call
        self._processor = None
        self._model      = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoImageProcessor, SuperPointForKeypointDetection

        print(f"[SuperPointExtractor] Loading model: {self.MODEL_ID}")
        self._processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID,
            local_files_only=False
        )

        self._model = SuperPointForKeypointDetection.from_pretrained(
            self.MODEL_ID,
            local_files_only=False
        )
        self._model.eval()
        print(f"[SuperPointExtractor] Model loaded successfully (CPU).")

    # ------------------------------------------------------------------
    # detect_and_compute
    # ------------------------------------------------------------------

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[float]], np.ndarray]:
        """Detect SuperPoint keypoints and compute 256D descriptors.

        Args:
            image: BGR (H×W×3) or grayscale (H×W) uint8 image.
            mask:  Optional binary mask (H×W). Only keypoints inside
                   nonzero regions are kept.

        Returns:
            keypoints:   list of [x, y] in full image coordinates.
            descriptors: (N, 256) float32 numpy array, L2-normalized.
        """
        self._load_model()

        # ---- 1. Grayscale → PIL (processor expects PIL or tensor) ----
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        from PIL import Image as PILImage
        # SuperPoint processor expects a 3-channel image
        pil_image = PILImage.fromarray(gray).convert("RGB")

        # ---- 2. Forward pass ----
        inputs = self._processor(images=[pil_image], return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        image_sizes = [(gray.shape[0], gray.shape[1])]
        keypoints_out = self._processor.post_process_keypoint_detection(
            outputs, image_sizes
        )

        result            = keypoints_out[0]
        keypoints_tensor  = result["keypoints"]    # (N, 2) tensor  [x, y]
        scores_tensor     = result["scores"]        # (N,)   tensor
        descriptors_tensor = result["descriptors"] # (N, 256) tensor

        if keypoints_tensor.shape[0] == 0:
            return [], np.zeros((0, self._DESCRIPTOR_DIM), dtype=np.float32)

        keypoints_np   = keypoints_tensor.cpu().numpy().astype(np.float32)
        scores_np      = scores_tensor.cpu().numpy().astype(np.float32)
        descriptors_np = descriptors_tensor.cpu().numpy().astype(np.float32)

        # ---- 3. Confidence filter ----
        conf_mask = scores_np >= self.confidence_threshold
        if not np.any(conf_mask):
            return [], np.zeros((0, self._DESCRIPTOR_DIM), dtype=np.float32)
        keypoints_np   = keypoints_np[conf_mask]
        scores_np      = scores_np[conf_mask]
        descriptors_np = descriptors_np[conf_mask]

        # ---- 4. Mask filter ----
        if mask is not None and mask.size > 0:
            keep = []
            h_mask, w_mask = mask.shape[:2]
            for i, (x, y) in enumerate(keypoints_np):
                ix, iy = int(round(float(x))), int(round(float(y)))
                if 0 <= iy < h_mask and 0 <= ix < w_mask and mask[iy, ix] > 0:
                    keep.append(i)
            if keep:
                idx = np.array(keep, dtype=np.int64)
                keypoints_np   = keypoints_np[idx]
                scores_np      = scores_np[idx]
                descriptors_np = descriptors_np[idx]
            else:
                return [], np.zeros((0, self._DESCRIPTOR_DIM), dtype=np.float32)

        # ---- 5. Top-N by confidence ----
        if self.max_keypoints is not None and len(scores_np) > self.max_keypoints:
            top_idx = np.argsort(-scores_np)[: self.max_keypoints]
            keypoints_np = keypoints_np[top_idx]
            descriptors_np = descriptors_np[top_idx]

        # ---- 6. L2-normalize descriptors ----
        norms          = np.linalg.norm(descriptors_np, axis=1, keepdims=True)
        norms          = np.maximum(norms, 1e-8)
        descriptors_np = descriptors_np / norms

        keypoints_list = [[float(x), float(y)] for x, y in keypoints_np]
        return keypoints_list, descriptors_np.astype(np.float32)

    # ------------------------------------------------------------------
    # match — Mutual Nearest Neighbor on cosine similarity
    # ------------------------------------------------------------------

    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
    ) -> Dict:
        """Match descriptors using Mutual Nearest Neighbor (MNN).

        Descriptors are assumed to be L2-normalized so dot product = cosine sim.

        Distance convention: distance = 1 - cosine_similarity  (∈ [0, 2])
        This keeps cv2.DMatch.distance semantics (lower = better).

        Args:
            descriptors1: (N1, 256) float32, L2-normalized.
            descriptors2: (N2, 256) float32, L2-normalized.

        Returns dict with keys (backward-compatible with SIFT pipeline):
            raw_matches:     list[cv2.DMatch] — all nn12 pairs (N1 entries).
            ratio_matches:   list[cv2.DMatch] — MNN pairs (conceptual alias for ratio_matches).
            selected_matches: list[cv2.DMatch] — same as ratio_matches (used for RANSAC).
            ratio_threshold: None              — not applicable for MNN.
            match_strategy:  "mnn" | "top_similarity_fallback" | "insufficient".
            mutual_matches:  list[cv2.DMatch] — explicit MNN field for debug images.
        """
        empty = {
            "raw_matches": [],
            "ratio_matches": [],
            "selected_matches": [],
            "ratio_threshold": None,
            "match_strategy": "insufficient",
            "mutual_matches": [],
        }

        if descriptors1.ndim != 2 or descriptors2.ndim != 2:
            return empty
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            return empty

        # ---- Cosine similarity matrix ----
        # Both are L2-normalized → dot product = cosine similarity
        sim = descriptors1 @ descriptors2.T  # (N1, N2)

        # ---- Nearest neighbor in both directions ----
        nn12 = np.argmax(sim, axis=1)   # (N1,) — for each in desc1, best in desc2
        nn21 = np.argmax(sim, axis=0)   # (N2,) — for each in desc2, best in desc1

        # ---- All nn12 pairs as raw_matches ----
        raw_matches: List[cv2.DMatch] = []
        for i in range(len(descriptors1)):
            j     = int(nn12[i])
            score = float(sim[i, j])
            raw_matches.append(
                cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=1.0 - score)
            )

        # ---- Mutual mask: i→j AND j→i ----
        ids1        = np.arange(len(descriptors1))
        mutual_mask = ids1 == nn21[nn12]

        mutual_matches: List[cv2.DMatch] = []
        for i in np.where(mutual_mask)[0]:
            j     = int(nn12[i])
            score = float(sim[i, j])
            if score >= self.similarity_threshold:
                mutual_matches.append(
                    cv2.DMatch(_queryIdx=int(i), _trainIdx=j, _distance=1.0 - score)
                )

        # Sort by similarity (best first = lowest distance)
        mutual_matches.sort(key=lambda m: m.distance)

        return {
            "raw_matches":      raw_matches,
            "ratio_matches":    mutual_matches,   # alias for downstream pipeline
            "selected_matches": mutual_matches,
            "ratio_threshold":  None,
            "match_strategy":   "mnn",
            "mutual_matches":   mutual_matches,
        }

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def descriptor_dim(self) -> int:
        return self._DESCRIPTOR_DIM

    @property
    def feature_type(self) -> str:
        return "superpoint"

    @property
    def extractor_version(self) -> str:
        return self._EXTRACTOR_VERSION
