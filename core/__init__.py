"""
Core AI pipeline, preprocessing, and matching modules.
"""
from core.pipeline import ReferencePipeline, resolve_profile_id
from core.preprocessor import clean_rembg_image, process_and_detect
from core.matcher import process_single_ghost_image, compute_similarity_from_bgr, validate_upload_match
from core.classifier import CeramicDefectHead

__all__ = [
    "ReferencePipeline",
    "resolve_profile_id",
    "clean_rembg_image",
    "process_and_detect",
    "process_single_ghost_image",
    "compute_similarity_from_bgr",
    "validate_upload_match",
    "CeramicDefectHead"
]
