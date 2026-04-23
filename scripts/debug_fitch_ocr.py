"""CLI wrapper for the importable Fitch/TFL OCR pipeline."""

from proofs_ocr.pipelines.tfl_fitch import (
    DEBUG_FITCH_DIR,
    RAW_SCANS_DIR,
    SubproofNode,
    apply_default_justifications,
    build_ocr_prompt,
    build_proof_tree,
    call_vlm,
    crop_region,
    detect_vertical_bars,
    filter_scaffold_bars,
    infer_missing_rule_heads,
    infer_premises,
    label_and_count_premises,
    main,
    mask_subproof_in_parent,
    normalize_proof_structure,
    process_node_recursive,
    proof_checker_available,
    recursive_stitch,
    remove_empty_lines,
    renumber_lines,
    run_fitch_ocr_pipeline,
    structure_to_proof_lines,
    vlm_client,
)

__all__ = [
    "DEBUG_FITCH_DIR",
    "RAW_SCANS_DIR",
    "SubproofNode",
    "apply_default_justifications",
    "build_ocr_prompt",
    "build_proof_tree",
    "call_vlm",
    "crop_region",
    "detect_vertical_bars",
    "filter_scaffold_bars",
    "infer_missing_rule_heads",
    "infer_premises",
    "label_and_count_premises",
    "main",
    "mask_subproof_in_parent",
    "normalize_proof_structure",
    "process_node_recursive",
    "proof_checker_available",
    "recursive_stitch",
    "remove_empty_lines",
    "renumber_lines",
    "run_fitch_ocr_pipeline",
    "structure_to_proof_lines",
    "vlm_client",
]


if __name__ == "__main__":
    main()
