"""Configuration for Adaptive Mesh Region Matching."""

from __future__ import annotations

# ============================================================================
# DEBUG MODE
# ============================================================================
# When False, no debug PNG files are written anywhere in the pipeline.
DEBUG = True

# ============================================================================
# REFINEMENT LIMITS
# ============================================================================
MAX_TOTAL_TRIANGLES = 2000   # Hard limit on total triangles
TOP_K = 24                 # Only refine top-K lowest similarity triangles
MAX_DEPTH = 3                # Maximum subdivision depth
MIN_TRIANGLE_AREA = 80.0      # Minimum area to consider refinement
SIMILARITY_THRESHOLD = 0.7  # Below this → candidate for refinement

# ============================================================================
# DAMAGE DETECTION THRESHOLDS
# ============================================================================
FEATURE_THRESHOLD = 0.95
SSIM_THRESHOLD = 0.90
MIN_AREA_THRESHOLD = 15
DIFF_THRESHOLD = 25

# ============================================================================
# PERFORMANCE
# ============================================================================
SIMILARITY_WORKERS = 4       # Parallel similarity evaluation