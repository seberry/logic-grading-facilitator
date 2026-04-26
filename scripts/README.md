# Scripts Index

Scripts in this folder are a mix of active wrappers, local probes, and older
utility scripts. Prefer importable code under `app/` or `proofs_ocr/` for new
production behavior.

## Active OCR Wrappers And Probes

- `debug_fitch_ocr.py` - thin CLI wrapper around the importable TFL/Fitch OCR
  pipeline.
- `probe_tfl_row_reread.py` - side probes for nested TFL/Fitch leaf crops,
  including separated formula/justification list OCR.
- `probe_27b_crop.py` - direct 27b crop probe with full-width/manual-prompt
  controls.
- `test_line_count_bands.py` - offline row-band line-count diagnostic.
- `ocr_q4_crops_8b.py`, `probe_subproof_ocr.py`, `probe_27b_diagnostic.py`,
  `probe_27b_thinking.py`, `capture_gemma_thinking.py` - historical/current
  local VLM OCR probes.

## App/Workflow Utilities

- `auto_group_scans.py`, `preview_page_predictions.py`, `split_pdf.py`,
  `split_to_images.py` - scan ingestion utilities.
- `debug_modal_proof_ocr.py`, `debug_fol_model_ocr.py`,
  `run_fol_ocr_batch.py` - modal/FOL debug workflows.
- `debug_segment_view.py`, `verify_llm_boxes.py` - segmentation/debug helpers.
- `test_canvas_api.py`, `test_canvas_user_id_mapping.py` - Canvas smoke/probe
  scripts.

## Older Or One-Off Utilities

- Icon scripts: `polish_icons*.py`, `process_icons.py`,
  `process_classwork_icon.py`, `verify_icons.py`.
- Server/file organization helpers: `add_server_markers.py`,
  `organize_server.py`, `code_map.py`, `init_session.py`.
- Legacy CV/prototype scripts: `bold_fitch_lines.py`, `fitch-scan-slicing.py`,
  `demo_fol_trace.py`, `dump_results.py`, `rescore_errors.py`.

## Output Convention

Write bulky experiment output under `working/` or a caller-provided output
directory. Do not write real student images or crops into tracked paths.
