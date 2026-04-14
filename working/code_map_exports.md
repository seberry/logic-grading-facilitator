# Code Map

## Workflow Route Counts

- `classwork`: 25
- `debug`: 1
- `exports`: 6
- `fitch_debug`: 4
- `flashcards`: 5
- `legacy_exam`: 9
- `other`: 1
- `problem_strips`: 4
- `setup`: 12

## Hotspots

- `app/routes/classwork.py`: 1193 LOC, 25 functions, 25 routes
- `app/server.py`: 1991 LOC, 41 functions, 23 routes
- `app/routes/exam_exports.py`: 372 LOC, 6 functions, 6 routes
- `app/routes/flashcards.py`: 465 LOC, 16 functions, 5 routes
- `app/routes/fitch_debug.py`: 301 LOC, 16 functions, 4 routes
- `app/routes/problem_strips.py`: 458 LOC, 9 functions, 4 routes
- `app/verifier.py`: 720 LOC, 37 functions, 0 routes
- `app/classwork_core.py`: 694 LOC, 34 functions, 0 routes
- `app/syntax.py`: 423 LOC, 16 functions, 0 routes
- `scripts/debug_fitch_ocr.py`: 764 LOC, 14 functions, 0 routes
- `scripts/code_map.py`: 385 LOC, 13 functions, 0 routes
- `app/storage.py`: 294 LOC, 11 functions, 0 routes
- `app/exam_shared.py`: 164 LOC, 11 functions, 0 routes
- `scripts/auto_group_scans.py`: 563 LOC, 8 functions, 0 routes
- `app/auto_group_core.py`: 320 LOC, 8 functions, 0 routes

## Focus: `exports`

### `app/routes/exam_exports.py`

- Route `GET /finish` -> `finish_page`
  Local helper reach: (none)
- Route `GET /export_renamed_scans` -> `export_renamed_scans`
  Local helper reach: (none)
- Route `GET /export` -> `export_scores`
  Local helper reach: (none)
- Route `POST /finalize` -> `finalize_assignment`
  Local helper reach: (none)
- Route `GET /export_feedback_zip` -> `export_feedback_zip`
  Local helper reach: (none)
- Route `GET /export_canvas_reupload_feedback_zip` -> `export_canvas_reupload_feedback_zip`
  Local helper reach: (none)
- Local imports: `app.classwork_core`, `app.exam_exports_core`, `app.exam_shared`, `app.session_state`, `app.storage`
- Imported by: `app.server`

## Files

### `app/__init__.py`

- 0 LOC, 0 functions, 0 classes, 0 routes
- Imported by: `tests.test_classwork_routes`, `tests.test_exam_exports_routes`, `tests.test_flashcards_routes`, `tests.test_problem_strips_routes`

### `app/auto_group_core.py`

- 320 LOC, 8 functions, 1 classes, 0 routes
- Local imports: `app.models`, `app.vlm_classifier`, `app.vlm_client`
- Imported by: `app.server`

### `app/canvas_client.py`

- 722 LOC, 2 functions, 3 classes, 0 routes
- Imported by: `app.routes.classwork`, `app.server`, `scripts.test_canvas_api`, `scripts.test_canvas_user_id_mapping`

### `app/classwork_core.py`

- 694 LOC, 34 functions, 0 classes, 0 routes
- Local imports: `app.helpers`, `app.models`, `app.session_state`, `app.storage`, `app.utils`
- Imported by: `app.routes.classwork`, `app.routes.exam_exports`, `app.server`

### `app/exam_exports_core.py`

- 163 LOC, 4 functions, 0 classes, 0 routes
- Local imports: `app.exam_shared`, `app.models`
- Imported by: `app.routes.exam_exports`

### `app/exam_shared.py`

- 164 LOC, 11 functions, 0 classes, 0 routes
- Local imports: `app.models`, `app.storage`
- Imported by: `app.exam_exports_core`, `app.routes.exam_exports`, `app.server`

### `app/helpers.py`

- 52 LOC, 4 functions, 0 classes, 0 routes
- Imported by: `app.classwork_core`, `app.routes.flashcards`, `app.routes.problem_strips`, `app.server`

### `app/models.py`

- 110 LOC, 0 functions, 9 classes, 0 routes
- Imported by: `app.auto_group_core`, `app.classwork_core`, `app.exam_exports_core`, `app.exam_shared`, `app.routes.classwork`, `app.routes.flashcards`, `app.routes.problem_strips`, `app.server`, `app.session_state`, `app.storage`, `app.vlm_classifier`, `scripts.test_canvas_user_id_mapping`, `tests.test_classwork_routes`, `tests.test_exam_exports_routes`, `tests.test_problem_strips_routes`

### `app/routes/__init__.py`

- 1 LOC, 0 functions, 0 classes, 0 routes
- Imported by: `tests.test_classwork_routes`, `tests.test_exam_exports_routes`, `tests.test_flashcards_routes`, `tests.test_problem_strips_routes`

### `app/routes/classwork.py`

- 1193 LOC, 25 functions, 0 classes, 25 routes
- Local imports: `app.canvas_client`, `app.classwork_core`, `app.models`, `app.session_state`
- Imported by: `app.server`
- Routes:
  - `GET /classwork/canvas_courses` -> `get_canvas_courses` (classwork)
  - `GET /classwork/canvas_assignments` -> `get_canvas_assignments` (classwork)
  - `GET /classwork/canvas_submission_counts` -> `get_canvas_submission_counts` (classwork)
  - `GET /classwork` -> `classwork_setup_page` (classwork)
  - `POST /classwork/setup` -> `classwork_setup` (classwork)
  - `POST /classwork/batch/start` -> `classwork_batch_start` (classwork)
  - `POST /classwork/batch/add` -> `classwork_batch_add` (classwork)
  - `POST /classwork/batch/open` -> `classwork_batch_open` (classwork)
  - `POST /classwork/batch/open_next` -> `classwork_batch_open_next` (classwork)
  - `POST /classwork/batch/remove` -> `classwork_batch_remove` (classwork)
  - `POST /classwork/batch/set_column` -> `classwork_batch_set_column` (classwork)
  - `POST /classwork/batch/clear` -> `classwork_batch_clear` (classwork)
  - `GET /classwork/batch/finalize` -> `classwork_batch_finalize_page` (classwork)
  - `POST /classwork/batch/finalize` -> `classwork_batch_finalize_submit` (classwork)
  - `POST /classwork/batch/close_session` -> `classwork_batch_close_session` (classwork)
  - `GET /classwork/review` -> `classwork_review_page` (classwork)
  - `POST /classwork/review` -> `classwork_review_submit` (classwork)
  - `POST /classwork/finalize` -> `classwork_finalize` (classwork)
  - `GET /classwork/preview_text/{stored_filename:path}` -> `classwork_preview_text` (classwork)
  - `GET /classwork/export/{export_id}/{kind}` -> `classwork_download_export` (classwork)
  - `GET /classwork/batch/export/{export_id}/gradebook` -> `classwork_batch_download_gradebook` (classwork)
  - `GET /classwork/batch/export/{export_id}/comments/{assignment_id}` -> `classwork_batch_download_comments` (classwork)
  - `GET /classwork/canvas_setup` -> `classwork_canvas_setup_page` (classwork)
  - `POST /classwork/canvas_setup` -> `classwork_canvas_setup` (classwork)
  - `POST /classwork/canvas_post_comments` -> `classwork_canvas_post_comments` (classwork)

### `app/routes/exam_exports.py`

- 372 LOC, 6 functions, 0 classes, 6 routes
- Local imports: `app.classwork_core`, `app.exam_exports_core`, `app.exam_shared`, `app.session_state`, `app.storage`
- Imported by: `app.server`
- Routes:
  - `GET /finish` -> `finish_page` (exports)
  - `GET /export_renamed_scans` -> `export_renamed_scans` (exports)
  - `GET /export` -> `export_scores` (exports)
  - `POST /finalize` -> `finalize_assignment` (exports)
  - `GET /export_feedback_zip` -> `export_feedback_zip` (exports)
  - `GET /export_canvas_reupload_feedback_zip` -> `export_canvas_reupload_feedback_zip` (exports)

### `app/routes/fitch_debug.py`

- 301 LOC, 16 functions, 0 classes, 4 routes
- Local imports: `app.storage`, `app.verifier`
- Imported by: `app.server`
- Routes:
  - `GET /debug/fitch` -> `fitch_debug_page` (fitch_debug)
  - `POST /debug/fitch/save` -> `save_fitch_structure` (fitch_debug)
  - `POST /debug/fitch/rerun` -> `rerun_fitch_pipeline` (fitch_debug)
  - `POST /debug/fitch/run_grader` -> `run_fitch_grader` (fitch_debug)

### `app/routes/flashcards.py`

- 465 LOC, 16 functions, 0 classes, 5 routes
- Local imports: `app.helpers`, `app.models`, `app.storage`
- Imported by: `app.server`
- Routes:
  - `GET /flashcards` -> `flashcards_setup` (flashcards)
  - `POST /flashcards/upload` -> `flashcards_upload` (flashcards)
  - `GET /flashcards/review` -> `flashcards_review` (flashcards)
  - `POST /flashcards/review` -> `flashcards_review_submit` (flashcards)
  - `POST /flashcards/export` -> `flashcards_export` (flashcards)

### `app/routes/problem_strips.py`

- 458 LOC, 9 functions, 0 classes, 4 routes
- Local imports: `app.helpers`, `app.models`, `app.segmentation_core`, `app.session_state`, `app.storage`, `app.vlm_client`
- Imported by: `app.server`
- Routes:
  - `POST /problem/{problem_id}/prefetch` -> `prefetch_problem` (problem_strips)
  - `GET /problem/{problem_id}/strips` -> `get_problem_strips` (problem_strips)
  - `POST /problem/{problem_id}/strips` -> `post_problem_strips` (problem_strips)
  - `POST /problem/{problem_id}/type` -> `post_problem_type` (problem_strips)

### `app/segmentation_core.py`

- 88 LOC, 1 functions, 1 classes, 0 routes
- Local imports: `app.vlm_client`
- Imported by: `app.routes.problem_strips`, `app.server`

### `app/server.py`

- 1991 LOC, 41 functions, 2 classes, 23 routes
- Local imports: `app.auto_group_core`, `app.canvas_client`, `app.classwork_core`, `app.exam_shared`, `app.helpers`, `app.models`, `app.routes.classwork`, `app.routes.exam_exports`, `app.routes.fitch_debug`, `app.routes.flashcards`, `app.routes.problem_strips`, `app.segmentation_core`, `app.session_state`, `app.storage`, `app.vlm_classifier`, `app.vlm_client`
- Routes:
  - `GET /` -> `root` (other)
  - `GET /setup` -> `setup_wizard` (setup)
  - `POST /setup/archive_reset` -> `setup_archive_reset` (setup)
  - `POST /setup/roster` -> `setup_roster` (setup)
  - `POST /setup/roster_gradebook` -> `setup_roster_gradebook` (setup)
  - `GET /gradebook_merge` -> `gradebook_merge_page` (setup)
  - `POST /gradebook_merge/analyze` -> `gradebook_merge_analyze` (setup)
  - `POST /gradebook_merge/merge` -> `gradebook_merge_merge` (setup)
  - `POST /setup/upload_scans` -> `setup_upload_scans` (setup)
  - `POST /setup/import_canvas_zip` -> `setup_import_canvas_zip` (setup)
  - `POST /setup/save_rubric` -> `setup_save_rubric` (setup)
  - `POST /setup/select_rubric` -> `setup_select_rubric` (setup)
  - `POST /setup/run_auto_grouping` -> `setup_run_auto_grouping` (setup)
  - `GET /session/summary` -> `session_summary` (debug)
  - `GET /work_units` -> `list_work_units` (legacy_exam)
  - `POST /work_units/create` -> `create_work_unit` (legacy_exam)
  - `GET /name_review` -> `name_review_page` (legacy_exam)
  - `POST /grade/{unit_id}` -> `update_grade` (legacy_exam)
  - `POST /work_units/{unit_id}/append_next_scan` -> `append_next_scan` (legacy_exam)
  - `POST /work_units/{unit_id}/remove_last_scan` -> `remove_last_scan` (legacy_exam)
  - `POST /grade/{unit_id}/page` -> `grade_page_submit` (legacy_exam)
  - `GET /name_review` -> `name_review_page` (legacy_exam)
  - `POST /name_review` -> `name_review_submit` (legacy_exam)

### `app/session_state.py`

- 40 LOC, 2 functions, 0 classes, 0 routes
- Local imports: `app.models`, `app.storage`
- Imported by: `app.classwork_core`, `app.routes.classwork`, `app.routes.exam_exports`, `app.routes.problem_strips`, `app.server`

### `app/storage.py`

- 294 LOC, 11 functions, 0 classes, 0 routes
- Local imports: `app.models`
- Imported by: `app.classwork_core`, `app.exam_shared`, `app.routes.exam_exports`, `app.routes.fitch_debug`, `app.routes.flashcards`, `app.routes.problem_strips`, `app.server`, `app.session_state`, `scripts.auto_group_scans`, `scripts.generate_fake_data`, `scripts.init_session`, `scripts.preview_page_predictions`, `scripts.split_to_images`, `scripts.test_canvas_user_id_mapping`

### `app/syntax.py`

- 423 LOC, 16 functions, 2 classes, 0 routes
- Imported by: `app.verifier`

### `app/utils.py`

- 53 LOC, 2 functions, 0 classes, 0 routes
- Imported by: `app.classwork_core`, `scripts.preview_page_predictions`

### `app/verifier.py`

- 720 LOC, 37 functions, 4 classes, 0 routes
- Local imports: `app.syntax`
- Imported by: `app.routes.fitch_debug`

### `app/vlm_classifier.py`

- 122 LOC, 3 functions, 1 classes, 0 routes
- Local imports: `app.models`, `app.vlm_client`
- Imported by: `app.auto_group_core`, `app.server`, `scripts.auto_group_scans`, `scripts.preview_page_predictions`

### `app/vlm_client.py`

- 310 LOC, 0 functions, 1 classes, 0 routes
- Imported by: `app.auto_group_core`, `app.routes.problem_strips`, `app.segmentation_core`, `app.server`, `app.vlm_classifier`, `scripts.auto_group_scans`, `scripts.debug_segment_view`, `scripts.preview_page_predictions`

### `scripts/add_server_markers.py`

- 187 LOC, 4 functions, 0 classes, 0 routes

### `scripts/auto_group_scans.py`

- 563 LOC, 8 functions, 1 classes, 0 routes
- Local imports: `app.storage`, `app.vlm_classifier`, `app.vlm_client`

### `scripts/bold_fitch_lines.py`

- 131 LOC, 2 functions, 0 classes, 0 routes

### `scripts/code_map.py`

- 385 LOC, 13 functions, 2 classes, 0 routes

### `scripts/debug_fitch_ocr.py`

- 764 LOC, 14 functions, 1 classes, 0 routes

### `scripts/debug_segment_view.py`

- 446 LOC, 5 functions, 0 classes, 0 routes
- Local imports: `app.vlm_client`

### `scripts/fitch-scan-slicing.py`

- 134 LOC, 1 functions, 0 classes, 0 routes

### `scripts/generate_fake_data.py`

- 167 LOC, 6 functions, 0 classes, 0 routes
- Local imports: `app.storage`

### `scripts/init_session.py`

- 40 LOC, 1 functions, 0 classes, 0 routes
- Local imports: `app.storage`

### `scripts/organize_server.py`

- 242 LOC, 5 functions, 0 classes, 0 routes

### `scripts/polish_icons.py`

- 150 LOC, 6 functions, 0 classes, 0 routes

### `scripts/polish_icons_final.py`

- 178 LOC, 6 functions, 0 classes, 0 routes

### `scripts/polish_icons_magenta.py`

- 170 LOC, 5 functions, 0 classes, 0 routes

### `scripts/preview_page_predictions.py`

- 43 LOC, 1 functions, 0 classes, 0 routes
- Local imports: `app.storage`, `app.utils`, `app.vlm_classifier`, `app.vlm_client`

### `scripts/process_classwork_icon.py`

- 210 LOC, 6 functions, 0 classes, 0 routes

### `scripts/process_icons.py`

- 81 LOC, 1 functions, 0 classes, 0 routes

### `scripts/split_pdf.py`

- 51 LOC, 1 functions, 0 classes, 0 routes

### `scripts/split_to_images.py`

- 55 LOC, 1 functions, 0 classes, 0 routes
- Local imports: `app.storage`

### `scripts/test_canvas_api.py`

- 185 LOC, 1 functions, 0 classes, 0 routes
- Local imports: `app.canvas_client`

### `scripts/test_canvas_user_id_mapping.py`

- 107 LOC, 4 functions, 0 classes, 0 routes
- Local imports: `app.canvas_client`, `app.models`, `app.storage`

### `scripts/verify_icons.py`

- 21 LOC, 0 functions, 0 classes, 0 routes

### `scripts/verify_llm_boxes.py`

- 104 LOC, 1 functions, 0 classes, 0 routes

### `tests/test_classwork_routes.py`

- 114 LOC, 1 functions, 1 classes, 0 routes
- Local imports: `app`, `app.models`, `app.routes`

### `tests/test_exam_exports_routes.py`

- 77 LOC, 1 functions, 1 classes, 0 routes
- Local imports: `app`, `app.models`, `app.routes`

### `tests/test_flashcards_routes.py`

- 116 LOC, 0 functions, 1 classes, 0 routes
- Local imports: `app`, `app.routes`

### `tests/test_problem_strips_routes.py`

- 82 LOC, 1 functions, 1 classes, 0 routes
- Local imports: `app`, `app.models`, `app.routes`
