FULL WORKFLOW — PowerShell Edition (one-line commands)

For per-problem strip view grading.

Run these in order, from the repo root (logic-grading-facilitator-MAIN).

1. Start clean (optional): archive old assignment

If you have stale session state:

python scripts\finalize_helper.py


(or use the UI’s Finalize & Archive button if running the server)

2. Run auto-grouping (model name recognition)


python -m scripts.auto_group_scans --quiz-id quiz1 --raw-scans-dir "data/raw_scans" --out-dir "data/auto_grouped/quiz1" --roster "data/roster.csv"


(Replace quiz1 with your actual quiz_slug.)

3. Start the server
python -m uvicorn app.server:app --reload --port 5000


Leave it running in that terminal.

4. Review and approve names

In your browser, go to:

http://127.0.0.1:5000/name_review?quiz=quiz1


NOTE: This will take a long time to run, so check in LM studio if the local moedl is working way before deciding something is wrong. 

Fix dropdowns → scroll to bottom → click green button:

“Name recognition & grouping looks good — rename & download.”

This step is crucial — it finalizes which pages belong to each WorkUnit.

5. Initialize the grading session

Open a second PowerShell window in the repo root:

python -m scripts.init_session


This creates:

data/grades/session.json

6. Begin per-problem grading (strip view)

In your browser:

http://127.0.0.1:5000/problem/Q1/strips


Replace Q1 with each problem as you grade. (If you see something wrong, hitting resegment may fix it. )

7. Export Canvas scores

After all problems are graded:

http://127.0.0.1:5000/export


Download the CSV.

8. Archive the finished assignment

Either:

Use the Finalize & Archive link at top of grading UI
or

(If needed) run:

python scripts\finalize_helper.py


This:

Copies all of data/ into data_archive/<timestamp>_<label>/

Clears:

data/raw_scans/

data/grades/session.json

Leaves roster and rubric intact

🎉 DONE — you’re ready for the next assignment
⭐ QUICK REFERENCE (copy/paste)