"""Public facade for the modal_v2 baseline pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from proofs_ocr.pipelines.modal_v2_core import (
    ImagePromptClient,
    ModalLineReader,
    ModalPipelineFeatureFlags,
    ModalPipelineResult,
    _extract_ocr_line_crop,
    _masked_node_crop_image,
    _node_crop_image,
    _segment_node_rows,
    _suppress_left_scope_bar,
    run_modal_baseline_pipeline,
)
from proofs_ocr.pipelines.modal_v2_debug import (
    build_modal_v2_state,
    transcribe_image_to_modal_v2_debug_state,
)
from proofs_ocr.pipelines.modal_v2_runtime import (
    ModalBaselineArtifactLineReader,
    build_vlm_client,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modal_v2 baseline OCR pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to the proof image.")
    parser.add_argument("--output-dir", required=True, help="Directory for baseline debug artifacts.")
    parser.add_argument("--system", default="K", help="Modal system label, e.g. K, T, S4, or S5.")
    parser.add_argument("--include-derived-rules", action="store_true", help="Tell the OCR prompt derived rules are in play.")
    parser.add_argument("--conclusion", default="", help="Optional target conclusion shown in the OCR prompt.")
    parser.add_argument("--vision-base-url", default="", help="LM Studio base URL.")
    parser.add_argument("--vision-model", default="", help="LM Studio model identifier.")
    parser.add_argument("--verifier-model", default="", help="LM Studio model identifier for the passive unary structured probe.")
    parser.add_argument(
        "--enable-formula-justification-separation",
        action="store_true",
        help="OCR the formula and justification zones separately and merge the result.",
    )
    parser.add_argument(
        "--enable-unary-structured-probe",
        action="store_true",
        help="Run the passive structured unary probe and write unary_structured_probe.json.",
    )
    parser.add_argument(
        "--enable-unary-blob-probe",
        action="store_true",
        help="Run the passive unary blob probe and write unary_blob_probe.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    state = transcribe_image_to_modal_v2_debug_state(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        modal_config={
            "system": args.system,
            "include_derived_rules": bool(args.include_derived_rules),
            "conclusion": args.conclusion,
        },
        vision_base_url=args.vision_base_url or None,
        vision_model=args.vision_model or None,
        verifier_vision_model=args.verifier_model or None,
        enable_formula_justification_separation=bool(args.enable_formula_justification_separation),
        enable_unary_structured_probe=bool(args.enable_unary_structured_probe),
        enable_unary_blob_probe=bool(args.enable_unary_blob_probe),
    )
    print(state["run_summary"])
    print(f"Artifacts written to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
