"""Orchestration entrypoints for clean proof OCR experiments."""

from .modal_v2 import transcribe_image_to_modal_v2_debug_state
from .tfl_fitch import run_fitch_ocr_pipeline

__all__ = [
    "run_fitch_ocr_pipeline",
    "transcribe_image_to_modal_v2_debug_state",
]
