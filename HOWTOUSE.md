Grading Workflow (PowerShell Edition)
Prerequisites:

LM Studio is running with the qwen/qwen3-vl-4b model loaded.

data/raw_scans/ contains your image files.

data/roster.csv is present.

1. Start Clean (Optional)
If you have a stale session from a previous quiz, archive it first.

PowerShell

python scripts\finalize_helper.py
# Or use the "Finalize & Archive" button in the UI if the server is running.
2. Run Auto-Grouping (AI Name Recognition)
This step uses the VLM to read names and group pages. Replace quiz1 with your actual quiz slug.

PowerShell

python -m scripts.auto_group_scans --quiz-id quiz1 --raw-scans-dir "data/raw_scans" --out-dir "data/auto_grouped/quiz1" --roster "data/roster.csv"
3. Start the Server
Keep this terminal window open.

PowerShell

python -m uvicorn app.server:app --reload --port 5000
4. Review and Approve Names
Open your browser to: http://127.0.0.1:5000/name_review?quiz=quiz1

Audit: Fix any incorrect dropdowns.

Finalize: Scroll to the bottom and click “Name recognition & grouping looks good — rename & download.”

Note: This confirms which pages belong to which student.

5. Initialize the Grading Session
In a new PowerShell window:

PowerShell

python -m scripts.init_session
This generates the fresh data/grades/session.json based on the grouped files.

6. Grade in Strip View
Navigate to: http://127.0.0.1:5000/problem/Q1/strips

Grade: Enter scores for Q1.

Navigate: Use "Save & Next" to move to Q2, Q3, etc.

Tip: If a crop looks wrong, you can view the full page or hit "resegment" (if implemented).

7. Export Grades
When finished: http://127.0.0.1:5000/export

Download the CSV and upload it to Canvas.

8. Archive & Reset
When the assignment is complete, run:

PowerShell

python scripts\finalize_helper.py
This moves current data to data_archive/ and clears the workspace for the next quiz.
⭐ QUICK REFERENCE (copy/paste)


| UI/Page                                 | Route                               | Core Code                 | Primary Data / Cache Files                                     |

> |----------------------------------------|--------------------------------------|---------------------------|----------------------------------------------------------------|
> | Name recognition review                | `GET /name_review`                  | `LMStudioQwenClient`, `classify_page`, `name_review_cache_path` | `data/raw_scans/`, `data/roster.csv`, `data/name_review_cache/<quiz>__<subdir>.json` |
> | Finalize name assignments + auto-group | `POST /name_review`                 | name_review_submit        | Writes `data/auto_grouped/<quiz>/...` (JSON, CSV, renamed scans zip) |
> | Work-unit grading (per student)        | `GET/POST /grade/{unit_id}/page`    | `grade_page`, `grade_page_submit` | `data/grades/session.json` (`Session`), `data/raw_scans/`             |
> | Per-problem strip view                 | `GET/POST /problem/{id}/strips`     | `get_problem_strips`, `post_problem_strips`, `segment_page_for_problem` | `session.json`, `data/grades/segments_{id}.json`, `data/raw_scans/`   |
> | Export renamed scans (simple ZIP)      | `GET /export_renamed_scans`         | `export_renamed_scans`    | Reads `session.json`, `data/raw_scans/` → returns in-memory ZIP       |
> | Export scores for Canvas               | `GET /export`                       | `export_scores`           | `session.json` → `student_id,<quiz>_score` CSV                         |
> | Archive + reset                        | `POST /finalize`                    | `archive_current_data`, `finalize_assignment` | Copies full `data/` → `data_archive/<timestamp>_<label>/`, clears `data/raw_scans`, deletes `grades/session.json` |
