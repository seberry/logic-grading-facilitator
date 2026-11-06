# TA/TF Quick Start: Paper Quiz Grader

## If the bundle says Quiz 9 FOL truth assistance

The instructor may have prepared a proposed vision reading for each question.
Compare it with the scan, then choose the confirmed True/False answer and, for
Q2–5, the confirmed witness/counterexample or N/A. Record unusual writing in
the “Written response as seen” box. “Unclear” and complicated answers need
your judgment. The panel checks truth and written answers separately. Its
recursive explanation is a draft: use “Use draft comment” only if you approve
it, and edit the ordinary comment if needed. “Accept proposed answer and fill
score” copies a clear proposal into the review and editable score fields only
when you click it. Check and adjust both fields. Save
before moving on. The bundle launcher checks that your code ZIP contains the
matching explanation module. No model or Node installation is needed on your
computer for this review.

This is a local viewer for grading a paper quiz. The instructor has already
matched the scans to students, identified quiz versions, and located each
answer. You grade the work using your own judgment; no model or Canvas API is
needed.

## What you receive

You need two separate downloads:

1. **Grader code ZIP** — downloaded from the GitHub release.
2. **Private grading bundle ZIP** — sent separately by the instructor. This
   contains the scans and saved grading data.

Use **Extract All** on both ZIPs before doing anything else. Do not work inside
the ZIP preview. Keep the private grading bundle on your university-approved
device and do not put it on GitHub or personal cloud storage.

After extraction, the code folder should contain `requirements.txt` and an
`app` folder. The bundle folder should contain `START_TF_GRADER.ps1`,
`START_TF_GRADER.command`, and a `data` folder.

## First time only: install the grader

Use Python 3.13 for this experimental release (tested on Windows). The install
commands below automatically use `requirements-lock.txt` for reproducibility.
macOS instructions are provided but a fresh Mac installation is not yet verified.

### Windows

1. Open the extracted **code folder** in File Explorer.
2. Click the address bar, type `powershell`, and press Enter.
3. Check that the prompt names the code folder, for example:

```text
PS C:\Users\YourName\Downloads\logic-grading-facilitator>
```

4. Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the first command says `python` is not recognized, use
`py -m venv .venv` instead. The installation is finished when the prompt
returns without a red error.

### macOS

Open Terminal, type `cd ` (including the space), drag the extracted code folder
onto the Terminal window, and press Return. Then run:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Start a grading session

### Windows

1. In File Explorer, open the extracted **code folder**. Click the address bar
   and copy its full path.
2. Open the extracted **grading bundle folder**—the one containing
   `START_TF_GRADER.ps1` and `data`.
3. Click that folder's address bar, type `powershell`, and press Enter.
4. Check that the prompt names the bundle folder, not merely `PS C:\>`.
5. Run this as one line, replacing the example path with the code-folder path
   you copied:

```powershell
powershell -ExecutionPolicy Bypass -File ".\START_TF_GRADER.ps1" -CodeDir "C:\path\to\logic-grading-facilitator"
```

Keep that PowerShell window open. You should see a line beginning `TF grader:`
followed by an address such as:

```text
http://127.0.0.1:5000/problem/Q1/strips
```

Open that address in a browser. `127.0.0.1` means the app is running only on
your computer. Press Ctrl+C in PowerShell when you are done for the day.

### macOS

Open Terminal, type `cd `, drag the extracted **grading bundle folder** onto the
Terminal window, and press Return. Then run the following, replacing the
example code path (you can also drag the code folder into Terminal to insert
its path):

```bash
bash ./START_TF_GRADER.command "/path/to/logic-grading-facilitator"
```

Keep Terminal open and open the `TF grader:` address it prints, normally using
port 5001. Press Control-C in Terminal when you are done for the day.

## Grade the quiz

1. Use the A/B/C tabs to grade one question and version across the class. The
   printed question stays pinned at the top; click it to enlarge it.
2. Check the blue version badge if a response appears to answer a different
   question.
3. Optionally pin a response you judge correct as a visual comparison. It is a
   convenience, not an answer key.
4. Enter the score you judge appropriate and add a student-facing comment only
   when useful.
5. Use **Show Full Page** or **Fix Crop** if the answer crop is wrong. Use
   **Rotate / view page** to read a sideways or upside-down scan. Its rotation
   is only for viewing; it does not change the saved scan or answer crop.
6. Click **Save & Next**. A save can take a moment; wait for the next response
   to appear before clicking again.

There is deliberately no model-generated solution or grading instruction. Use
your ordinary course judgment. If the expected answer or partial-credit policy
is unclear, contact the instructor promptly.

Do not use a global segmentation refresh. The saved answer locations are part
of the bundle, and no local model is expected on the TA/TF computer.

## Finish and return the work

Every regular question needs an explicit score before CSV/PDF export or Canvas
gradebook merge. A blank is ungraded, not a zero; enter 0 when awarding no credit.
Blank optional extra credit counts as zero. The Finish page lists missing grades.

If the Finish page reports duplicate submissions, compare the listed problem
scores and open the scan links for both tests. The exports mark and use the
highest-total submission; do not assume that is the instructor's intended
attempt. Tell the instructor which units you compared so they can make the
final duplicate decision.

If the rubric includes extra credit, the Finish page states the scoring policy:
extra credit can restore missed regular points, but the exported total is
capped at the sum of the non-extra-credit problems (100%).

On the Finish page, download:

- **grades CSV**;
- **scans + feedback PDFs ZIP**, containing one named PDF per student (scan
  pages followed by a typed score/comment page).

Also copy `data\grades\session.json` from the extracted grading bundle. Return
all three items to the instructor through the agreed university channel. The
instructor will upload the grades CSV to Canvas and attach the named PDFs to
the corresponding students.

Keep the extracted grading bundle until the instructor confirms receipt; it is
the recoverable copy of your grading progress.

## If something goes wrong

- **PowerShell says `START_TF_GRADER.ps1` does not exist:** you opened
  PowerShell in the wrong folder, or you are viewing rather than extracting the
  ZIP. Open the extracted bundle folder and check that the script is visible.
- **It says `data\grades\session.json` is missing:** you supplied the wrong
  folder. Start the launcher located beside the bundle's `data` folder.
- **It says the code folder does not contain `app\server.py`:** `-CodeDir`
  points to the wrong level. Choose the extracted folder containing both `app`
  and `requirements.txt`.
- **A Python module is missing:** repeat the first-time install commands in the
  extracted code folder.
- **The browser cannot connect:** keep the PowerShell/Terminal window open and
  read its last error to the instructor. If port 5000 is busy on Windows, add
  `-Port 5001` to the launcher command and open the newly printed address.
- **A student/version or crop looks wrong:** correct the version in strip view
  or use **Show Full Page**/**Fix Crop**. Contact the instructor before changing
  a student identity.
- **Scores seem not to save:** use **Save**, **Save & Next**, or **Finish**, wait
  for the page to respond, then revisit the question and confirm the values are
  still visible.
- **The app reports an LM Studio/model error:** stop and contact the instructor.
  A correctly prepared bundle should not invoke a model.
