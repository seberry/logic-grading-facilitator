"""Regression manifests and evaluation helpers for proof OCR."""

from .runner import (
    compare_proof_structures,
    default_modal_config_for_case,
    get_repo_root,
    load_manifest,
    render_gold_regression_summary,
    render_manifest_summary,
    run_gold_regressions,
    select_cases,
)

__all__ = [
    "compare_proof_structures",
    "default_modal_config_for_case",
    "get_repo_root",
    "load_manifest",
    "render_gold_regression_summary",
    "render_manifest_summary",
    "run_gold_regressions",
    "select_cases",
]
