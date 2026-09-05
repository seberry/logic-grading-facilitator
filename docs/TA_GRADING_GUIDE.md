# TF Guide: Grading the Prepared Paper-Quiz Bundle

The instructor has already matched scans to students, identified the quiz
version, and located each question. You do not need LM Studio or Canvas API
access. The bundle contains protected student work; keep it on an approved
device and do not upload it to GitHub or personal cloud storage.

## First-time setup

The instructor sends two separate items:

- **grader code**, either a GitHub release link or a code ZIP;
- **private TF grading bundle**, which contains the student scans and saved
  grading setup.

Keep them in separate extracted folders. Never upload the private bundle to
GitHub or personal cloud storage.

1. Obtain and extract the code:

   - For a GitHub release link, open the link, expand **Assets**, download
     **Source code (zip)**, and choose **Extract All** in Windows.
   - For a supplied code ZIP, choose **Extract All** directly.

   GitHub is only supplying code files. It is not hosting the grader or seeing
   student work.

2. Install Python 3 if needed. On Windows, enable **Add Python to PATH** during
   a python.org installation.
3. Open PowerShell (Windows) or Terminal (macOS) in the extracted code folder.
4. On Windows run:

```powershell
py -m pip install -r requirements.txt
```

   If `py` is not recognized, use:

```powershell
python -m pip install -r requirements.txt
```

   On macOS run:

```bash
python3 -m pip install -r requirements.txt
```

5. Extract the private TF bundle somewhere outside the code folder.

## Start the grader

### Windows

Open the extracted private bundle. Right-click `START_TF_GRADER.ps1`, choose
**Run with PowerShell**, and enter the full path to the extracted code folder
when prompted. Keep the PowerShell window open.

If right-click execution is unavailable, click the bundle folder's address bar,
type `powershell`, press Enter, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\START_TF_GRADER.ps1 `
  -CodeDir "C:\path\to\extracted\logic-grading-facilitator"
```

Open the address printed in PowerShell, normally:

```text
http://127.0.0.1:5000/problem/Q1/strips
```

`127.0.0.1` means the grader is running only on this computer. Closing the
PowerShell window stops it; reopening it later restores the saved grades from
the bundle.

### macOS

Open Terminal in the extracted private bundle and run:

```bash
bash ./START_TF_GRADER.command
```

Enter the full path to the extracted code folder when prompted. Keep Terminal
open and visit the address it prints, normally
`http://127.0.0.1:5001/problem/Q1/strips`.

The launcher enables TF mode. On the final page this hides instructor-only and
unrelated controls such as setup, classwork, gradebook merge, renamed scans,
Canvas re-upload, and preparation of another TF bundle.

## Grade

1. Use the A/B/C tabs to grade one question/version across its students. The
   blank printed question stays pinned at the top; click it for the full image.
2. Check the blue Version A/B/C badge if a response looks out of place.
3. If useful, pin a response you judge correct as that version's visual
   reference. Saved references survive reloads and are comparison aids, not a
   supplied answer key.
4. Enter any partial-credit score you judge appropriate.
5. Add a comment only when useful to the student.
6. If the version is wrong, change the Version dropdown on that card.
7. If the crop is wrong, use **Show Full Page** or **Fix Crop**, then save.
8. Use **Save & Next** to continue through A, B, C, and then the next problem.

No solution key is supplied. If you cannot determine the intended grading from
your course knowledge and the neutral rubric notes, stop and contact the
instructor promptly. Do not ask an LLM to decide what the correct answer is.

Do not click the global **Segmentation** refresh link. The saved location
guesses are part of the bundle, and no local model is available on this machine.

## Finish and return

On the Finish page download:

- **grades CSV**;
- **scans + feedback PDFs ZIP**. It contains one PDF per student: the original
  scan pages first, followed by a typed page containing the total,
  per-question scores, and comments.

Also copy `data\grades\session.json` from the extracted bundle. Return all of
these to the instructor through the agreed secure channel. The instructor will
merge scores into Canvas and can attach each named PDF to the corresponding
student's Canvas record.

## Troubleshooting by phone

### `python` is not recognized

Install Python, reopen PowerShell, and rerun the dependency-install command.
The launcher automatically tries both `py` and `python`.

### A Python module is missing

From the code folder rerun `pip install -r requirements.txt`. Make sure the
PowerShell window is in the supplied code release, not the bundle folder.

### The browser cannot connect

The PowerShell grader window must remain open. Read the last error in that
window to the instructor. If port 5000 is busy, start the bundled launcher with:

```powershell
powershell -ExecutionPolicy Bypass -File .\START_TF_GRADER.ps1 `
  -CodeDir "C:\path\to\extracted\logic-grading-facilitator" `
  -Port 5001
```

Then open `http://127.0.0.1:5001/problem/Q1/strips`.

### A student/version is wrong

Correct the Version dropdown directly in strip view. Contact the instructor
before changing a student identity; the bundle is already roster-matched.

### A crop is blank or cuts off handwriting

Use **Show Full Page**, then **Fix Crop**. Do not refresh all segmentation.

### Scores seem not to save

Use the page's **Save**, **Save & Next**, or **Finish** button before closing the
browser. Then revisit the problem and confirm the values remain visible.

### The grader reports an LM Studio/model error

Stop and call the instructor. A correctly prepared bundle should already have
every `segments_<problem>.json` file and should not invoke a model.
