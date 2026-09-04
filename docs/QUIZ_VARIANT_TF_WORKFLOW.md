# Versioned Paper Quiz: Instructor Workflow

This is the operating note for paper quizzes that have multiple printed
versions, are photographed or scanned, and are handed to one TF for grading.
The instructor's local model performs name/version recognition and question
location. The TF grades from saved crops without running a model.

## What the project now supports

- A Canvas gradebook CSV can be reduced locally to the `Student` and
  `SIS Login ID` roster fields. Canvas API access is not required.
- The name-recognition pass also reads an explicit printed version label.
- Name and version guesses are editable before the session is created.
- Each work unit stores `quiz_version` (for example `A`, `B`, or `C`).
- A rubric problem can contain version-specific prompts and neutral grading
  dimensions under `variants`; PDF-to-YAML drafting does not generate solutions.
- Strip grading has A/B/C tabs. Each page shows only that version's students,
  keeps one large blank-quiz question image pinned at the top, allows inline
  version correction, and lets the grader save one visual reference per
  question/version.
- A TF bundle contains the scans, session, rubric, and all saved question
  locations. It intentionally excludes the original Canvas gradebook, model
  output/debug folders, `.env`, and Canvas credentials.
- The TF can export a score CSV and per-student feedback files. The instructor
  can merge the scores into a fresh Canvas gradebook CSV afterward.

## Rubric shape

The base label and `segmentation_hint` should describe geometry shared by every
version. Put version-specific content under `variants`:

```yaml
quiz_id: P250_F26_quiz1_ABC
max_points: 10
problems:
  - id: Q1
    label: Identifying the conclusion
    points: 2
    segmentation_hint: Include the printed Q1 label, argument, and handwritten answer lines.
    variants:
      A:
        prompt: Students who attend review sessions ...
        rubric_notes: Judge whether the response identifies the conclusion; the TF chooses partial credit.
      B:
        prompt: If the printers are out of paper ...
      C:
        prompt: The forecast calls for heavy rain ...
```

Choose the actual point values before saving the rubric. Extra-credit boxes can
be separate problems with `extra_credit: true`; the total-score calculation
already caps regular plus extra credit at the quiz's regular maximum.

## Answer-key trust boundary

The PDF-to-YAML prompt may transcribe printed questions, propose point values,
describe crop geometry, and name neutral partial-credit dimensions. It must not
generate solutions or `answer_key` fields. Legacy keys are not displayed in
strip view and are removed when a TF bundle is exported.

During grading, the instructor or TF may pin a student response they judge
correct. Pins are stored in `session.json` by problem and version, survive page
reloads, and travel in the TF bundle. They are visual comparison aids, not
machine-verified solutions. If the TF cannot determine how to grade a response
from course knowledge and the neutral rubric, they should contact the instructor
promptly rather than ask a model to settle it.

## Prepare the assignment

1. Use an external production `DATA_DIR`; do not put real scans in Git.
2. Download a current Canvas gradebook CSV and upload it in Setup as the roster.
   Keep the original full gradebook on the instructor's machine.
3. Upload the phone scans and save the version-aware rubric.
   Scan intake compares SHA-256 content: a repeated upload of the exact same
   image is skipped, while two genuinely different scans with the same filename
   are both preserved (the later filename receives `_2`).
4. Run name recognition with LM Studio.
   Before recognition starts, any exact duplicates left by an older app version
   are moved—not deleted—to `DATA_DIR/duplicate_scans_quarantine/<timestamp>/`.
5. On Name Review, correct every student and Version A/B/C guess. Use the full
   page toggle when the title or handwriting is hard to read. **Save
   Corrections** writes exactly those selections to the local cache without
   renaming scans or starting grading. A later recognition refresh preserves
   the reviewed names while replacing the version guesses.
6. Save and start grading. Open every problem's strip page once so the local
   model creates `segments_<problem>.json` for all students.
7. Spot-check the crops. For this phone-scan workflow, correct especially
   rotated, perspective-skewed, shadowed, or partially cut-off pages.
8. Optionally pin one clearly correct response for each question/version you
   want to provide as a visual comparison. Pins can also be chosen by the TF.
9. Click **Prepare TF bundle**. The export refuses to proceed when a scan,
   version, or problem-location cache is missing.

Keep the extracted TF bundle outside the repository. It contains protected
student names, identifiers, and scans.

## Blank-question images

The supplied Quiz 1 PDF can be converted into 15 safe display crops (five
problems across Versions A/B/C) without a model:

```powershell
python scripts/extract_quiz_reference_images.py `
  "C:\path\to\P250_Quiz_1_Three_Versions.pdf" `
  --quiz-id P250_F26_quiz1_ABC `
  --output-dir "C:\grading-data\live\rubric_assets"
```

The `p250_quiz1` preset stacks the shared symbolization key directly above Q2
and Q3. The app finds the conventional filenames automatically, so an already
created `session.json` does not need to be rewritten. Present assets are copied
into the TF bundle; prompt text remains the fallback if an image is missing.

## Give the TF access

The TF needs two separate things:

1. **Grader code**: ordinary source code with no student data, delivered as a
   GitHub release link or a clean code ZIP.
2. **Private grading bundle**: the generated
   `P250_F26_quiz1_ABC_tf_grading_bundle.zip`, delivered through an
   institution-approved private channel.

GitHub is not hosting or running the grader. It only hosts the source-code
files. The TF downloads and extracts those files, and Python runs the local web
server on the TF's own computer. The private grading bundle never belongs on
GitHub.

### Option A: GitHub release link (recommended for repeated use)

Use this after the relevant code changes have been committed and pushed:

1. On GitHub, open the code repository and choose **Releases** → **Draft a new
   release**.
2. Create a descriptive tag such as `p250-f26-quiz1-grader-v1` and publish the
   release.
3. Send the TF the release-page link. Under **Assets**, the TF downloads
   **Source code (zip)** and extracts it.
4. Send the private TF bundle separately.

If the repository is private, the TF needs GitHub access to it. A public
repository is acceptable only if the repository contains code and fake data,
never real student artifacts.

Advantages: the exact version is named and recoverable, later fixes can become
new releases, and the non-private code does not have to be repeatedly emailed.
Disadvantages: changes not committed and pushed are absent; the TF needs
internet access and, for a private repository, an authorized GitHub account.

### Option B: clean code ZIP

After committing the code version that produced the bundle, create a ZIP from
tracked files:

```powershell
git archive --format=zip --output ..\logic-grader-p250-f26-quiz1-v1.zip HEAD
```

Send that code ZIP plus the private TF bundle. `git archive` avoids including
`.git`, `.env`, ignored live data, and untracked experiment files. It includes
only the committed `HEAD`, so verify that the required changes were committed
first. Avoid selecting the entire working folder in Explorer, which can easily
include protected or irrelevant local files.

Advantages: no GitHub account is required, it works offline after transfer, and
the TF receives one obvious code package. Disadvantages: updates require a new
ZIP, matching versions depends on careful filenames, and manually produced ZIPs
have a greater risk of accidental data inclusion.

### Test the handoff yourself

1. Keep your live instructor server and original bundle untouched.
2. Extract a copy of the TF bundle outside the repository, for example under
   `C:\grading-data\tf-test\`.
3. Open PowerShell in that extracted bundle and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\START_TF_GRADER.ps1 `
  -CodeDir "C:\path\to\logic-grading-facilitator" `
  -Port 5002
```

4. Open the printed `http://127.0.0.1:5002/...` address, enter a few temporary
   grades, reload, and download the CSV. Only the extracted test copy changes.
5. Delete the test copy only when you no longer need it; it still contains
   protected student work.

Whichever code-delivery option you use, the code must match the bundle. Do not
send the real `.env`, original Canvas gradebook, model caches, or debug crops.

## Receive and import grades

Ask the TF to return:

- the downloaded scores CSV;
- the feedback ZIP, if comments were entered; and
- the completed bundle's `data/grades/session.json` as the recovery copy.

On the instructor machine, first preserve your original prepared session, then
copy the TF's returned `session.json` into the active external
`DATA_DIR/grades/session.json`. Restart or reload the app. Open **Gradebook
merge tool**, upload a fresh Canvas gradebook CSV, choose the Quiz 1 assignment
column, review the match/conflict summary, and download the merged CSV for
Canvas upload. The returned scores CSV is also a readable cross-check, but the
current merge page reads authoritative scores from `session.json`.

The ordinary feedback ZIP names files with student name and SIS ID. These can
be attached manually in Canvas. The Canvas re-upload ZIP is meant for files
originally downloaded from Canvas and is not the right path for photographed
paper quizzes.

## Release/privacy checklist

- Confirm `git status` contains only intended code/docs changes.
- Confirm the GitHub release contains no real roster, scans, session, grade,
  crop, ZIP, PDF, or `.env` files.
- Audit already-tracked files as well as `.gitignore`; ignore rules do not
  remove files already committed.
- Keep one untouched copy of the instructor's prepared bundle until Canvas
  grades have been verified.
