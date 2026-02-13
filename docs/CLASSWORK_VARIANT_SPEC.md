# Classwork + Attendance-Q Variant Spec

Status: Proposed
Owner: Instructor workflow variant (no local model)

## 1. Problem Statement
Current grading flow is optimized for multi-question exams with per-problem segmentation.  
For classwork / attendance-Q assignments, grading is effort-based:

- Most submissions should receive full credit if anything was attempted.
- Missing submissions should receive `0`.
- Instructor needs a fast scroll/review UI with occasional manual score reductions and comments.
- Output must support Canvas re-import:
  - merged gradebook CSV for a selected assignment column
  - optional reupload ZIP when individualized comments were left

## 2. Goals
- Single-assignment review mode that does not run segmentation or answer extraction.
- Import downloaded classwork submissions (mixed `pdf/jpg/png`, optionally text payloads).
- Set assignment points once (e.g., `10` or `1`).
- Bulk-fill full credit for submitted entries.
- Rapid per-entry pass with optional score override + comment.
- Finalize to produce:
  - gradebook-merged CSV for selected column
  - comments reupload ZIP only when comments exist
- Deterministic, local-only behavior (FERPA-safe), no cloud/model dependency.

## 3. Non-Goals
- No question-level rubric grading.
- No VLM/OCR segmentation.
- No automatic semantic quality scoring.
- No attempt to infer correctness beyond instructor manual overrides.

## 4. User Workflow
1. Open `Classwork Viewer` page from setup/finish.
2. Upload:
   - Canvas submissions ZIP (required)
   - Canvas gradebook CSV (required for final merge destination)
3. Configure:
   - `assignment_id` (label/slug for exported artifact names only; e.g., `wk04_attendance_q`)
   - `max_points` (e.g., `10` or `1`)
   - roster match source:
     - uploaded gradebook roster (default)
     - current app/session roster (fallback)
   - target gradebook column (selected after analyze)
4. Click `Build Review Queue`.
5. Review page loads one row per student (scroll list):
   - submission preview (full page/file; no crop model)
   - default score prefilled:
     - submitted => `max_points`
     - no submission => `0`
   - optional per-student comment
6. Use bulk actions:
   - `Credit All Submitted`
   - `Set Missing to Zero`
   - optional `Flag only commented rows`
7. Finalize:
   - generates merged gradebook CSV for selected Canvas assignment column
   - if any comments exist, generates Canvas reupload ZIP preserving original filenames for those commented submissions

## 5. Data Model Additions
Add a dedicated assignment-mode record in session state.

```json
{
  "classwork_mode": {
    "assignment_id": "wk04_attendance_q",
    "max_points": 10,
    "gradebook_column": "Week 4 Attendance Q (10 pts)",
    "entries": [
      {
        "student_id": "12345",
        "student_name": "Berry, Sharon",
        "has_submission": true,
        "submission_files": ["berrysharon_2919_1.pdf"],
        "score": 10,
        "comment": null
      }
    ]
  }
}
```

Notes:
- `entries` must be roster/gradebook-complete (include students with no files).
- `has_submission=false` entries default to `0`.

## 6. Backend Requirements
### 6.1 Ingestion
- New route: `POST /classwork/setup`
  - Inputs:
    - submissions ZIP
    - gradebook CSV
    - assignment_id
    - max_points
    - roster-match source (`gradebook` or `session`)
  - Behavior:
    - parse roster identities from selected source:
      - gradebook rows (`SIS Login ID`, `Student`) by default
      - session roster when selected
    - map ZIP files to student IDs (same prefix-key strategy as existing Canvas ZIP import)
    - matching strategy should attempt, in order:
      - Canvas-style name prefix key
      - student ID token appearing in filename
      - normalized roster-name key found within normalized filename stem
    - build full entry list from gradebook roster, not just matched files
    - store unmatched files report (non-fatal warning)

### 6.2 Review
- New page: `GET/POST /classwork/review`
  - Render one card/row per student with:
    - name + ID + submission status
    - preview of first file + file switcher when multiple files
    - score input
    - comment box
  - Bulk actions:
    - fill submitted with max
    - fill missing with zero
    - reset to defaults

### 6.3 Finalize + Export
- New route: `POST /classwork/finalize`
  - Inputs: selected assignment column, overwrite flag
  - Outputs:
    - merged gradebook CSV (same conflict protections as existing merge tool)
    - optional comments ZIP when `any(comment.strip())`

## 7. Merge Rules (Critical)
- For every student row in gradebook:
  - if `has_submission=true` and no manual override: score = `max_points`
  - if `has_submission=false`: score = `0`
  - manual score override always wins
- Conflict behavior mirrors current `gradebook_merge`:
  - block overwrite for non-empty matched cells unless explicitly allowed

## 8. Comment ZIP Rules
- Generate only if at least one comment exists.
- Preserve original Canvas filenames for included files.
- Do not rename submission files inside the re-upload ZIP.  
  Canvas requirement: it must recognize filenames from the original download.
- For commented entries:
  - create feedback artifact per original extension:
    - `pdf` -> rendered feedback PDF
    - `jpg/png` -> rendered feedback image
    - unsupported -> include plain text sidecar (`<original>.comment.txt`) and copy original
- If student has multiple submission files and one comment:
  - apply same comment artifact to each file unless per-file comment is later implemented.

## 9. UI Requirements
- Add `Classwork Viewer` launch button on `setup` and `finish`.
- Review UI optimized for fast scan-through:
  - sticky controls (bulk fill, filter, finalize)
  - keyboard shortcuts:
    - `f` full credit current row
    - `z` zero current row
    - `j/k` next/previous row
    - `c` focus comment
- Filters:
  - show all / submitted only / missing only / commented only

## 10. Validation + Edge Cases
- Missing gradebook ID column -> hard error.
- Duplicate roster keys -> hard error with explicit collisions.
- Files unmatched to roster -> warning list; do not block import.
- Students with multiple files -> all files attached to one entry.
- Empty ZIP with valid gradebook -> allowed; all rows become missing/0.
- V1 submitted rule: any matched file in the Canvas ZIP counts as submitted.
- V1 text-only rule: if Canvas provides text as file artifacts in ZIP (`.txt/.html/.docx` etc.), that still counts as submitted.
- V1 comment scope: one comment is per student entry and applies to every file attached to that student.

## 11. Acceptance Criteria
1. Given a gradebook of N students and ZIP with K matched submissions:
   - review queue contains N entries
   - exactly K have `has_submission=true`
2. `Credit All Submitted` sets all submitted rows to `max_points` without touching missing rows.
3. `Set Missing to Zero` sets only `has_submission=false` rows to `0`.
4. Finalize export writes selected assignment column for all matched gradebook rows.
5. No submission rows export `0` unless manually changed.
6. Comment ZIP is not produced when no comments exist.
7. Comment ZIP preserves original Canvas filenames for commented rows.

## 12. Implementation Plan
1. Add classwork session schema + persistence helpers.
2. Add setup route + parser to build full roster-complete entry list.
3. Implement review template + POST handlers for score/comment/bulk updates.
4. Reuse gradebook merge safety logic for final export.
5. Add conditional comment ZIP export for classwork mode.
6. Add setup/finish navigation links and smoke-test end-to-end on sample data.

## 13. V1 Decisions (Locked)
- Unmatched files only warn; they do not block import.
- Submitted is determined by presence of any matched file artifact in the ZIP.
- Text-like file artifacts count as submissions.
- Comment is per student entry and applies to all that student’s attached files.
- `assignment_id` affects exported output names only; it does not alter submission filenames used for Canvas re-upload matching.
