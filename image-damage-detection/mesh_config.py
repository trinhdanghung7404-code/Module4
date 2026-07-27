"""Configuration for Adaptive Mesh Region Matching."""

from __future__ import annotations

# Debug output
DEBUG = True

# Adaptive mesh refinement
MAX_TOTAL_TRIANGLES = 2000
TOP_K = 24
MAX_DEPTH = 3
MIN_TRIANGLE_AREA = 80.0
SIMILARITY_THRESHOLD = 0.7

# Damage evidence
FEATURE_THRESHOLD = 0.95
SSIM_THRESHOLD = 0.90
MIN_AREA_THRESHOLD = 15
DIFF_THRESHOLD = 25

# Nearby edge correspondence after local residual alignment.
EDGE_SEARCH_RADIUS = 4
EDGE_MATCH_SCORE_THRESHOLD = 0.68
EDGE_ORIENTATION_TOLERANCE_DEG = 20.0

# Lab a/b color difference. Color is evaluated only in the suspicious mesh
# region plus a small context margin, not across the whole object.
COLOR_DIFF_THRESHOLD = 15.0
COLOR_MIN_AREA = 5
COLOR_CONTEXT_RADIUS = 3

# Human-visible local defects.
# These thresholds add a luminance-sensitive branch so white, grey, black,
# dull and dark marks can be detected even when Lab a/b barely changes.
PERCEPTUAL_LUMINANCE_ENABLED = True
PERCEPTUAL_L_DIFF_THRESHOLD = 12.0
PERCEPTUAL_SCORE_THRESHOLD = 15.0
PERCEPTUAL_STRONG_SCORE_THRESHOLD = 26.0
PERCEPTUAL_MIN_AREA = 20
PERCEPTUAL_STRONG_MIN_AREA = 10
PERCEPTUAL_REFERENCE_OBJECT_AREA = 300000
PERCEPTUAL_LOCAL_SIGMA = 7.0
PERCEPTUAL_L_WEIGHT = 1.0

# Visible-defect recall outside suspicious mesh triangles.
# This branch is intentionally stricter because exact-position comparison can
# react to small paint/geometry displacement in real photographs. Compact,
# high-contrast defects are kept; large diffuse or edge-following regions are
# rejected.
PERCEPTUAL_OUTSIDE_MESH_ENABLED = True
PERCEPTUAL_OUTSIDE_MESH_SCORE_THRESHOLD = 30.0
PERCEPTUAL_OUTSIDE_MESH_MIN_AREA = 14
PERCEPTUAL_OUTSIDE_MESH_MAX_AREA_RATIO = 0.0025
PERCEPTUAL_OUTSIDE_MESH_MIN_FILL_RATIO = 0.12
PERCEPTUAL_OUTSIDE_MESH_MATCHED_EDGE_REJECT_RATIO = 0.65

# Diagnostic-only strong-color recall outside the suspicious mesh.
# It compares each return pixel against nearby product colors and only writes
# debug masks. The recall mask has been verified and is added to final damage.
GLOBAL_STRONG_COLOR_RECALL_ENABLED = True
GLOBAL_STRONG_COLOR_THRESHOLD = 30.0
GLOBAL_STRONG_COLOR_SEARCH_RADIUS = 3
GLOBAL_STRONG_COLOR_MIN_AREA = 5
GLOBAL_STRONG_COLOR_ADD_TO_FINAL = True

# Local residual alignment for grouped suspicious triangles. The center of a
# suspicious region is excluded from scoring; only its surrounding ring is used
# to estimate a small translation.
LOCAL_ALIGNMENT_ENABLED = True
LOCAL_ALIGNMENT_SEARCH_RADIUS = 4
LOCAL_ALIGNMENT_MAX_KEYPOINT_OFFSET = 8.0
LOCAL_ALIGNMENT_RING_WIDTH = 12
LOCAL_ALIGNMENT_MIN_RING_PIXELS = 96
LOCAL_ALIGNMENT_MIN_IMPROVEMENT = 0.12
LOCAL_ALIGNMENT_MAX_TRIANGLES_PER_REGION = 12
LOCAL_ALIGNMENT_MAX_REGION_EXTENT = 180
LOCAL_ALIGNMENT_TRIM_KEEP_RATIO = 0.90
LOCAL_ALIGNMENT_GRADIENT_WEIGHT = 0.70
LOCAL_ALIGNMENT_SMOOTH_SIGMA = 1.5

# Performance
SIMILARITY_WORKERS = 4

# Compatibility aliases for mesh_damage_detector.py
COLOR_SEARCH_RADIUS = COLOR_CONTEXT_RADIUS
STRONG_COLOR_DIFF_THRESHOLD = GLOBAL_STRONG_COLOR_THRESHOLD