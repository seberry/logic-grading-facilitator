# Grading Workflow (PowerShell Edition)

## 0. Prerequisites
* **LM Studio** is running with the `qwen/qwen3-vl-4b` model loaded.
* `data/raw_scans/` contains your image files.
* `data/roster.csv` is present.

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



Roster: Choose an existing roster from data/rosters/ or upload a new CSV.
Roster (Canvas shortcut): Upload the full Canvas gradebook CSV; the wizard extracts `Student` and `SIS Login ID` into `data/roster.csv`.
Roster (Canvas shortcut): Provide a roster name so the extracted roster is saved under `DATA_DIR/rosters/` and can be reused later.



Scans: Upload scans (JPG/PNG) or a PDF (auto-splits into data/raw_scans/).



Rubric: Edit/save the rubric YAML (writes to data/rubric/<quiz_id>.yaml).




Auto-Grouping: Click Run auto-grouping to generate name predictions.


Confirm Groups: You will be redirected to /name_review. Once confirmed, you are ready to grade.



Grade: Jump to Step 5 below (Strip View).

Manual CLI Flow (Legacy)
1. Start Clean (Optional)
If you have a stale session from a previous quiz, archive it first.

Go to: http://127.0.0.1:5000/setup

If a session exists, you will see a prompt to Archive & Reset. Enter a label (e.g., quiz1_old) and click the button.


2. Run Auto-Grouping (AI Name Recognition)
This step uses the VLM to read names and group pages. Replace quiz1 with your actual quiz slug.

PowerShell

python -m scripts.auto_group_scans --quiz-id quiz1 --raw-scans-dir "data/raw_scans" --out-dir "data/auto_grouped/quiz1" --roster "data/roster.csv"
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
Navigate to: http://127.0.0.1:5000/problem/Q1/strips.


Grade: Enter scores for Q1.


Navigate: Use "Save & Next" to move to Q2, Q3, etc..


Tip: If a crop looks wrong, click "Show Full Page" or "Fix Crop".

6. Export Grades
When finished, click **Finish** on the last question to open:
http://127.0.0.1:5000/finish.

Download the grades CSV and the renamed scans ZIP from there.

7. Archive & Reset
When the assignment is complete, archive your data to prepare for the next quiz.


Go to: http://127.0.0.1:5000/setup (or use the "Finalize & Archive" button at the top of any grading page).




Action: Enter an archive label (e.g., quiz1_final) and submit.




Result: This moves current data to data_archive/ and clears data/raw_scans and session.json.
⭐ QUICK REFERENCE 


| UI/Page                                 | Route                               | Core Code                 | Primary Data / Cache Files                                     |

> |----------------------------------------|--------------------------------------|---------------------------|----------------------------------------------------------------|
> | Name recognition review                | `GET /name_review`                  | `LMStudioQwenClient`, `classify_page`, `name_review_cache_path` | `data/raw_scans/`, `data/roster.csv`, `data/name_review_cache/<quiz>__<subdir>.json` |
> | Finalize name assignments + create session | `POST /name_review`             | name_review_submit        | Writes `data/grades/session.json`, copies scans into `data/raw_scans/`, optional ZIP; cache-only for Save & Auto-Fill |
> | Work-unit grading (per student)        | `GET/POST /grade/{unit_id}/page`    | `grade_page`, `grade_page_submit` | `data/grades/session.json` (`Session`), `data/raw_scans/`             |
> | Per-problem strip view                 | `GET/POST /problem/{id}/strips`     | `get_problem_strips`, `post_problem_strips`, `segment_page_for_problem` | `session.json`, `data/grades/segments_{id}.json`, `data/raw_scans/`   |
> | Export renamed scans (simple ZIP)      | `GET /export_renamed_scans`         | `export_renamed_scans`    | Reads `session.json`, `data/raw_scans/` → returns in-memory ZIP       |
> | Export scores for Canvas               | `GET /export`                       | `export_scores`           | `session.json` → `student_id,<quiz>_score` CSV                         |
> | Archive + reset                        | `POST /finalize`                    | `archive_current_data`, `finalize_assignment` | Copies full `data/` → `data_archive/<timestamp>_<label>/`, clears `data/raw_scans`, deletes `grades/session.json` |
