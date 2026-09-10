# Grading Workflow (PowerShell Edition)

## Start here each week

On the instructor's Windows computer, double-click
`START_INSTRUCTOR_GRADER.bat` (or the Desktop shortcut named **Logic Grader -
Start Here**). It starts the server, opens the Setup Wizard, and uses the
external production data folders configured in the repo's ignored `.env` file.

You should see a browser page headed **Logic Grading Facilitator — Setup
Wizard**. Keep the server window open while working and close it when finished.

If an older quiz is still active, the first page will offer **Archive & Reset**.
After resetting, the recurring paper-quiz sequence is:

1. Select the saved roster, or upload a fresh Canvas gradebook CSV.
2. Upload the ZIP/PDF/images containing the student scans.
3. load or paste the version-aware rubric and save it.
4. Run auto-grouping, then review names and versions.
5. Save and start grading; visit each strip page before preparing a TF bundle.

The launcher normally uses port 5001. If that port belongs to an already-running
copy of this grader, it simply opens that copy. If another program owns the
port, it selects the next available port and prints the address clearly.

## 0. Prerequisites
* **LM Studio** is running with the `qwen/qwen3-vl-4b` model loaded.
* `DATA_DIR/raw_scans/` contains your image files.
* `DATA_DIR/roster.csv` is present.

## Data location (FERPA-safe)
All `data/...` paths below are relative to `DATA_DIR` (defaults to `data/samples`).
For real grading, set `ENV_MODE=PRODUCTION` and point `DATA_DIR` outside the repo.

Example `.env`:
```
DATA_DIR=C:\grading-data\live
DATA_ARCHIVE_DIR=C:\grading-data\archive
ENV_MODE=PRODUCTION
```

## Recommended: Setup Wizard (UI)

1. **Start the server:**
   ```powershell
   python -m uvicorn app.server:app --reload --port 5000
   ```

Optional (safe dev): generate fake data under `DATA_DIR`:
```powershell
python scripts/generate_fake_data.py
```

Note: In test mode (`ENV_MODE` not set to `PRODUCTION`), the setup wizard disables
file uploads by default. Use the override checkbox only if you understand the risk.

2. **Visit http://127.0.0.1:5000/setup and follow the prompts:**


Archive & Reset: If you see a warning about an existing session, use the Archive & Reset button right there in the wizard to clear the deck.



Roster: Choose an existing roster from `DATA_DIR/rosters/` or upload a new CSV.
Roster (Canvas shortcut): Upload the full Canvas gradebook CSV; the wizard extracts `Student` and `SIS Login ID` into `DATA_DIR/roster.csv`.
Roster (Canvas shortcut): Provide a roster name so the extracted roster is saved under `DATA_DIR/rosters/` and can be reused later.



Scans: Upload scans (JPG/PNG) or a PDF (auto-splits into `DATA_DIR/raw_scans/`).



Rubric: Edit/save the rubric YAML (writes to `DATA_DIR/rubric/<quiz_id>.yaml`). If you make a rubric manually outside the app, put that file in `DATA_DIR/rubric/` with a `.yaml` or `.yml` extension and reload `/setup`; it will then appear in the rubric chooser. In production this is usually an external live folder such as `C:\grading-data\live\rubric\`, not the repo’s `data/rubric/`. The per-problem `label` is also used as the natural-language hint when the model locates each answer region in strip view, so match the wording on the exam if possible.




Auto-Grouping: Click Run auto-grouping to generate name predictions.


Confirm Groups: You will be redirected to /name_review. Once confirmed, you are ready to grade.



Grade: Jump to Step 5 below (Strip View).

---

## Classwork + Attendance (Batch)

Use `http://127.0.0.1:5000/classwork` for effort-based grading and one-pass gradebook merge.

- Local semester defaults:
  - Attendance Q (`AQ`) = **1 point**
  - Classwork (`CW`) = **10 points**
- In batch mode, set the target gradebook column once when queueing each assignment.
- Finalize batch reuses those saved mappings and does not ask for columns again.
- In the Canvas API classwork flow, the viewer includes all submissions Canvas reports as submitted, including late submissions. The current UI marks them only as `submitted` rather than separately labeling `late`.
- Classwork files keep two names internally:
  - `original_filename` is the exact Canvas/ZIP filename preserved for labels and export ZIPs.
  - `stored_filename` is an app-owned safe local filename used for preview URLs and disk storage.

## Flashcards Dataset Builder (UI)

Use this for creating a name-learning flashcard dataset from student uploads.

1. Start the server:
   ```powershell
   python -m uvicorn app.server:app --reload --port 5000
   ```
2. Visit: `http://127.0.0.1:5000/flashcards`
3. (Optional) Pick or upload a roster CSV. This is used only for name guessing.
4. Upload a ZIP of student files (PDFs, DOCX, or images).
5. For each item, crop the photo and enter the student name + free-text facts.
6. Export the dataset:
   - JSON + images are written to `DATA_DIR/flashcards/<session_id>/`

Tip: If you skip the roster, the name field is free-text.

Manual CLI Flow (Legacy)
1. Start Clean (Optional)
If you have a stale session from a previous quiz, archive it first.

Go to: http://127.0.0.1:5000/setup

If a session exists, you will see a prompt to Archive & Reset. Enter a label (e.g., quiz1_old) and click the button.


2. Run Auto-Grouping (AI Name Recognition)
This step uses the VLM to read names and group pages. Replace quiz1 with your actual quiz slug.

PowerShell

python -m scripts.auto_group_scans --quiz-id quiz1 --raw-scans-dir "DATA_DIR/raw_scans" --out-dir "DATA_DIR/auto_grouped/quiz1" --roster "DATA_DIR/roster.csv"
3. Start the Server
Keep this terminal window open.

PowerShell

python -m uvicorn app.server:app --reload --port 5000
4. Review and Approve Names
Open your browser to: http://127.0.0.1:5000/name_review?quiz=quiz1.


Audit: Fix any incorrect dropdowns.


Finalize: Scroll to the bottom and click **Save & Start Grading** (or **Save & Download Scans (ZIP)**).


(Note: This creates the official session.json. Do not run init_session manually after this.).

5. Grade in Strip View
Navigate to: http://127.0.0.1:5000/problem/Q1/strips. For a versioned quiz this
opens Version A; the A/B/C tabs filter the student list, and Save & Next moves
through each version before the next question. A blank-question crop under
`DATA_DIR/rubric_assets/<quiz_id>/<problem_id>_<version>.png` is shown once in
the sticky header, with rubric prompt text as fallback.


Grade: Enter scores for Q1.


Navigate: Use "Save & Next" to move to Q2, Q3, etc..


Tip: If a crop looks wrong, click "Show Full Page" or "Fix Crop".
If server logs show `failed to process image`, the VLM client now auto-retries with smaller image payloads; if it still fails, restart LM Studio and retry the problem page.

For a versioned paper quiz graded by one TF, use
[`docs/QUIZ_VARIANT_TF_WORKFLOW.md`](docs/QUIZ_VARIANT_TF_WORKFLOW.md). It covers
version-aware rubric entries, local name/version recognition, saved question
locations, the model-free TF bundle, and the Canvas CSV return path. The TF's
standalone instructions are in [`docs/TA_GRADING_GUIDE.md`](docs/TA_GRADING_GUIDE.md).

6. Export Grades
When finished, click **Finish** on the last question to open:
http://127.0.0.1:5000/finish.

Download the grades CSV and the scans + feedback PDFs ZIP from there. The latter contains one named PDF per student with the original scan pages followed by a typed feedback page containing per-problem scores and comments. You can also open the grade overview anytime via the “Grade overview” link in the grader header.
The finish page also shows a per-student, per-problem overview table with color-coded cells and comment indicators; click a problem header to jump back to that problem's strip grader.
If you want a safer Canvas import workflow, open **Gradebook merge tool** from the finish page: upload Canvas gradebook CSV, pick the assignment column, and download a merged CSV. By default it blocks merges into non-empty matched cells unless you explicitly allow overwrite.

7. Archive & Reset
When the assignment is complete, archive your data to prepare for the next quiz.


Go to: http://127.0.0.1:5000/setup (or use the "Finalize & Archive" button at the top of any grading page).




Action: Enter an archive label (e.g., quiz1_final) and submit.




Result: This archives current data to `DATA_ARCHIVE_DIR/` and clears `DATA_DIR/raw_scans/` plus live `DATA_DIR/grades/` files (`session.json`, `segments_*.json`).
⭐ QUICK REFERENCE 


| UI/Page                                 | Route                               | Core Code                 | Primary Data / Cache Files                                     |

> |----------------------------------------|--------------------------------------|---------------------------|----------------------------------------------------------------|
> | Name recognition review                | `GET /name_review`                  | `LMStudioQwenClient`, `classify_page`, `name_review_cache_path` | `DATA_DIR/raw_scans/`, `DATA_DIR/roster.csv`, `DATA_DIR/name_review_cache/<quiz>__<subdir>.json` |
> | Finalize name assignments + create session | `POST /name_review`             | name_review_submit        | Writes `DATA_DIR/grades/session.json`, copies scans into `DATA_DIR/raw_scans/`, optional ZIP; cache-only for Save & Auto-Fill |
> | Work-unit grading (per student)        | `GET/POST /grade/{unit_id}/page`    | `grade_page`, `grade_page_submit` | `DATA_DIR/grades/session.json` (`Session`), `DATA_DIR/raw_scans/`             |
> | Per-problem strip view                 | `GET/POST /problem/{id}/strips`     | `get_problem_strips`, `post_problem_strips`, `segment_page_for_problem` | `DATA_DIR/grades/session.json`, `DATA_DIR/grades/segments_{id}.json`, `DATA_DIR/raw_scans/` (cache is validated before reuse)   |
> | Export renamed scans (simple ZIP)      | `GET /export_renamed_scans`         | `export_renamed_scans`    | Reads `DATA_DIR/grades/session.json`, `DATA_DIR/raw_scans/` → returns in-memory ZIP       |
> | Export Canvas reupload feedback ZIP    | `GET /export_canvas_reupload_feedback_zip` | `export_canvas_reupload_feedback_zip` | Reads latest ZIP in `data/canvas_reupload/original_zips/`; preserves original filenames and writes feedback pages/images |
> | Gradebook merge tool                   | `GET /gradebook_merge`, `POST /gradebook_merge/analyze`, `POST /gradebook_merge/merge` | gradebook merge routes | Upload Canvas gradebook CSV, select assignment column, merge by student ID with non-empty-cell safety checks |
> | Export scores for Canvas               | `GET /export`                       | `export_scores`           | `session.json` → `student_id,<quiz>_score` CSV                         |
> | Archive + reset                        | `POST /finalize`                    | `archive_current_data`, `finalize_assignment` | Copies full `DATA_DIR/` → `DATA_ARCHIVE_DIR/<timestamp>_<label>/`, then clears live `DATA_DIR/raw_scans/` and `DATA_DIR/grades/` files |
