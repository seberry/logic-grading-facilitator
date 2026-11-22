# Versioned Paper Quiz: Instructor Workflow

## General truth-in-a-model assistance

Use optional `fol_truth` data on each assisted rubric problem, with overrides
under its existing `variants` entries. The rubric supplies the actual model,
formula, whether an object response is requested, and an optional scoring rule.
See [the schema and complete mixed-assessment example](SCHEMA.md#rubric-driven-fol-truth-assistance-bundle-schemas-2-and-3).
The rubric maker still drafts printed assessment structure; it does not solve
questions or invent assistance policies. On Finish, **Review models and
sentences** opens `/prepare/fol-truth`. Upload a blank PDF there to have IU
transcribe its fixed models/formulas (embedded PDF text where available, vision
for image-only PDFs). The instructor can correct model notation, formulas,
response requirements, selected questions/versions, and score policies, then
check the deterministic preview and explicitly approve the definitions.
The review page can apply one scoring policy across assisted questions. Its
full-or-half-credit option uses the correct truth value as the condition for
credit and checks required single-object evidence for full credit. When the
printed directions require no object for a sentence, the correct truth value
earns full credit by itself.
Unapproved drafts survive reloads in the private grades directory. New
rubric-driven assisted bundles require approval of the current definitions; changing a model,
formula, response requirement, or scoring policy invalidates approval. The
legacy original Quiz 9 bundle behavior is retained. Saving/loading also validates
rubric definitions before they can be used for grading.

The existing TF answer controls work for any configured domain, including named
objects. The TF can fill in T/F and a witness/counterexample directly, or correct
an instructor-prepared reading. Exact-writing entry recognizes simple object
answers and N/A; ambiguous prose remains for human interpretation. After the
existing key-confirmation control, deterministic feedback updates with those
answers. Scoring can be separate truth/evidence credit, all-or-nothing, or
entirely manual while still suggesting feedback. Added TF comments and saved
manual grades retain the existing preservation behavior.

For Quiz 9 retakes with new models, enter the new facts in `fol_truth`; do not
reuse the old schema-1 model tables. Quiz 10 currently uses ordinary manual
grading for all six questions, including EC, at the instructor's request. If
assistance is wanted later, Q6 can use `asks_object: true` and
`scoring: {mode: all_or_nothing, points: 1}`. Quiz 11 and selected final-exam questions use
the same configuration and controls, including nested quantifiers and polyadic
relations. Object evidence currently means one outer-quantifier witness or
counterexample, rather than a list of assignments or a prose proof.

Finish's optional answer-reading preparation skips ordinary questions and
unconfigured versions. A TF bundle contains the resolved rubric source and
reading state. It declares assistance schema 3 when using the full-or-half-credit
policy, otherwise schema 2 for rubric-driven assistance. Each requires grader
code that supports its schema.
Existing Quiz 9 schema-1 sessions/bundles remain supported. Student data and
active grading sessions are not migrated or rewritten automatically.

Preliminary answer reading requests strict provider JSON with string-valued
`truth` (`true`, `false`, `blank`, `unclear`) and `written` fields. Older valid
JSON boolean truth values are also accepted; false must not become unclear.
Malformed replies are rejected without saving a proposal and reported with
`vision_reading_format`. Bulk preparation continues past those individual
format errors, offers a retry of only the unread items, and leaves them available
for manual TF entry. Transport/provider outages still stop the batch to avoid
repeated failed calls. Saved fresh proposals are skipped on retry. Missing or
unclear OCR does not prevent bundle export or local default-feedback generation
once the TF supplies a clear reading and confirms the key. The private data ZIP
must be accompanied by matching updated grader code, which supplies the local
browser explanation/scoring modules; no model connection is required by the TF.

Offline checks: `python -m pytest tests/test_fol_truth_general.py
tests/test_fol_truth_assistance.py tests/test_quiz9_preparation_provider.py
tests/test_exam_exports_routes.py tests/test_fol_model_checker.py
tests/test_quiz_variant_tf_handoff.py -q` and `node --test
tests/fol_truth_general.test.mjs tests/fol_truth_review.test.mjs
tests/fol_truth_scoring.test.mjs`.

## Prepared Quiz 9 retakes and Quiz 10 (November 13, 2026)

The reviewed blank-assessment definitions are
[`quiz9_retakes_DE_rubric.yaml`](quiz9_retakes_DE_rubric.yaml) and
[`quiz10_polyadic_symbolization_rubric.yaml`](quiz10_polyadic_symbolization_rubric.yaml).
Their selectable filenames in `DATA_DIR/rubric/` are
`quiz9_retakes_DE_fol_truth.yaml` and `quiz10_polyadic_symbolization_ABC.yaml`.
The matching 14 and 18 blank reference images live in the respective
`DATA_DIR/rubric_assets/<quiz_id>/` directories. Reference images include the
printed model or symbolization key; these are display assets, not saved student
question-location caches.

Retakes use Versions D/E and the new printed models. They retain the approved
original Quiz 9 policy: Q1=2; Q2–5=2 each (1 truth plus 1 object/N/A);
Q6–7=1 extra credit each; final maximum 10. Those point rules are carried over
from the instructor-approved original, because the D/E PDF does not print them.
Quiz 10 uses its printed policy: Q1–5=2 each, Q6=1 extra credit requiring both
parts, final maximum 10. Assistance is enabled for all seven retake questions.
Quiz 10's saved rubric was subsequently switched to manual grading throughout,
including EC, without changing points, version prompts, or reference images.
It therefore needs no model/sentence approval or preliminary answer readings.
The same updated grader code handles both assessments; each needs a separate
private grading bundle. Name recognition accepts explicit labels beyond A/B/C;
Name Review offers the selected rubric's versions and preserves any different
observed label for correction.

For each new assignment/batch:

1. Prepare the blank assessment once, before collecting scans. Supply its
   versions, points/extra-credit rules, and which questions should offer
   feedback. Review the saved rubric, model/formula definitions, and reference
   images. The built-in rubric draft supplies question structure; the separate
   **Review models and sentences** step transcribes and reviews logical inputs.
2. Start the updated instructor app. After code changes, stop the existing
   grader before reopening the launcher; an already-running process does not
   reload the changed Python code automatically.
3. Preserve/export the current session, then use **Archive & Reset** to begin
   the next batch. This keeps saved rubrics, reference images, and the roster.
   Prepare retakes and Quiz 10 as separate sessions and separate TF bundles.
4. In Setup, upload that batch's student scans, choose **Saved rubric**, and
   click **Load saved rubric** for the matching filename. Confirm the quiz ID,
   run auto-grouping, review names and printed versions, and start the session.
   Each supplied blank PDF has one page per version; a student's submission
   normally consists of one such page, not every version in the blank PDF.
5. On Finish / Prepare TF bundle, prepare all question locations and spot-check
   each problem/version, including blank and extra-credit responses. Optionally
   prepare preliminary answer readings for assisted questions. These are
   transcriptions for TF review; preparation leaves grades unset. Use **Review
   models and sentences** to correct/approve the inputs before exporting an
   assisted bundle. It changes logical configuration only, preserving names,
   grades, comments, and scan/crop state.
6. Export the private TF bundle and provide matching updated grader code. The
   TF confirms the suggested key, corrects or fills in interpreted answers,
   reviews feedback/scores, and saves. No model connection is needed for TF
   grading. Receive the scores, feedback PDFs, and recovery session as described
   below before merging grades into Canvas.

For future quizzes, reuse an existing rubric only when its printed content,
models, versions, and point rules still match. New models require updated rubric
definitions, not new checker code. Each scan batch still needs its own saved
question locations and any optional answer readings.

If a stale instructor process was used to select a new assisted rubric, it can
silently drop the unfamiliar `fol_truth` fields from the selected setup copy and
session. An older name classifier can also lose D/E labels. Preserve the named
saved rubric as the recovery source; do not archive/reset or re-upload the scans
to fix this. `scripts/repair_assisted_quiz_session.py` recovers only assistance
fields and missing versions, backs up the replaced private files, and refuses
changes to unrelated rubric content or session state. Its version reader asks
IU REALLMS for only printed labels and saves proposals privately; it does not
display scans or handwritten responses. Stop the server before applying a
repair and restart it afterward. Original name assignments, scores, comments,
and saved question locations must survive unchanged.

The D/E retake batch was recovered this way: 14 submissions, seven D and seven E.
Q1–6 locations and six Q7 checkpoints were retained; IU Qwen supplied the eight
missing Q7 locations after Gemma returned invalid bounds. All seven full caches
then passed bundle validation. No grades or preliminary answer readings were
created. The running grader was restarted with the updated code. Model-read
version labels and crops still warrant the instructor/TF's ordinary visual check.

A second older instructor process on port 5002 subsequently stripped assistance
fields again while sharing the same private data directory. That duplicate was
stopped, the definitions restored, and the scheduled port-5001 grader restarted.
The instructor launcher now holds an OS file lock on its data directory, refusing
a second writer even if its initial running-server probe misses a busy server.
Do not run direct alternate-port servers against the same active data directory.
The blank D/E PDF was transcribed through the new review page: all 14 proposed
question/version definitions matched the reviewed source, with approval left for
the instructor. No student scan was displayed during this recovery.

## Quiz 9 handoff status (November 6, 2026)

**Working in the active private Quiz 9 session:** 32 submissions have confirmed
A/B/C versions, seven complete question-location caches, and 224 preliminary
IU Gemma answer readings (one per question and submission). All 224 answer
requests completed and the saved proposal fingerprints matched their current
version, crop, and source scan. No scores were set by preparation. These counts
describe request completion and storage, not transcription accuracy. The live
student scans, session, and caches remain under the private `DATA_DIR`; do not
copy them into Git or test fixtures.

The seven location caches were computed from the blank three-page PDF using
`scripts/prepare_quiz9_template_locations.py`. Registration succeeded on every
scan (minimum 367 inlier features and 0.557 inlier ratio). A prior IU Gemma
location request failed because it returned `[115,285,335,333]` even though the
prompt required normalized 0–1 coordinates; that response cannot safely be
used as a box on the original 1190×1684 scan. The template caches now supply
locations without a model. The partial Q1 model cache was preserved but is not
used while the complete cache is valid.

The first IU answer request failed because Gemma wrapped valid JSON in a
`json` code fence. The answer parser now accepts that wrapper. The active
instructor server at port 5003 was restarted with this fix, the Finish bulk
reading control, and the strip proposal controls; a live request and the full
224-request run succeeded. IU connectivity and JSON parsing were thus verified,
but handwritten-choice accuracy was not measured against human transcription.

**TF review flow:** fresh saved proposals prefill the interpreted choices.
The prominent Check the suggested answer control starts an explicit answer-key
review for the current question/version. Confirming the key enables score and
wrong-answer feedback prefilling on this page; a rejected key asks the TF to
email the instructor and stops automatic grading. Grade manually instead keeps
manual score/comment entry available. Confirmation is per page visit; saved
grades and comments are preserved when the page is reopened. The shared question/version
header emphasizes the model and sentence. An optional scratchpad above the review control is cleared when leaving the page. The code's answer and Why? explanation
are behind the review control; suspected key errors should be emailed to the
instructor rather than edited in the TF interface.
Each strip emphasizes only the bold interpreted student answer, with quick
True/False/Other and object/N/A/Other buttons. Amber means an unresolved reading,
not an incorrect answer; all clear readings still need checking against the scan. Correct the choices against the scan; the score
and generated comment update together, retaining text appended by the TF. Saved
manual grades and comments are preserved on reload. Save & Next or Save & Back
approves all readings on the current page together; no individual acceptance is
required. Stale or unclear readings do not get an automatic score. Preparation
still leaves persisted grades unset until the TF saves the grading page.

Code entry points: `app/fol_truth_assistance.py` (Quiz 9 models, formulas,
prompt, answer parser), `app/assets/fol_truth_review.mjs` and
`fol_truth_scoring.mjs` (browser review and scoring),
`app/routes/problem_strips.py` (proposal endpoint and strip persistence),
`app/templates/finish.html` (bulk preparation), and
`app/routes/exam_exports.py` (bundle export). The approved points are Q1=2;
Q2–5=2 each, split 1 T/F plus 1 written response; Q6–7=1 extra credit each,
with the total capped at 10. Empty writing receives 0 when N/A is expected.
The 21 blank question images and reviewed rubric are prepared outside Git from
the blank PDF. The TF bundle includes the stored proposals and fixed crops,
while TF mode has no model calls.

Useful checks: `pytest tests/test_fol_truth_assistance.py
tests/test_quiz9_preparation_provider.py tests/test_exam_exports_routes.py -q`
and `node --test tests/fol_truth_scoring.test.mjs`. The private session can be
audited locally by counting `fol_truth_reviews` records and confirming that
`per_problem` values remain unset, without printing student responses.

## Opt-in Quiz 9 FOL truth assistance

The example rubric is `docs/quiz9_fol_truth_rubric.yaml`. Its explicit
`fol_truth_assistance: {schema: 1, assessment: quiz9_monadic_fol_truth}` enables
the review panel for the seven questions in the supplied three-version blank
quiz. The printed models and formulas are transcribed in
`app/fol_truth_assistance.py`; compact formulas are checked with
`app/fol_model_checker.py`. The browser imports the approved upstream module
from `app/assets/vendor/fol-explanations/` at commit
`9d3dcdf744d4bbc1ac523a434c1cf633ffe71a9e`. Refresh it with
`python scripts/vendor_fol_explanations.py --source <source-checkout> --revision <approved-commit>`
and review `SOURCE.json` and wording approval before release.

The instructor-ready rubric is `DATA_DIR/rubric/quiz9_monadic_fol_truth.yaml`;
its 21 version/question blank images are under
`DATA_DIR/rubric_assets/quiz9_monadic_fol_truth/`. Select that rubric in Setup
when preparing Quiz 9; the export includes the rubric and referenced assets.
Rebuild the ready set from the blank three-page PDF with
`python -m scripts.prepare_quiz9_fol_truth_rubric --pdf <blank-quiz.pdf> --output-dir <new-output-folder>`.
The script refuses to overwrite existing output. Compare the crops and point
rules before replacing an instructor-ready set.

For a fake-data review, run `python -m scripts.make_quiz9_fol_truth_demo`
(it refuses to overwrite an existing demo). In a new PowerShell window set
`$env:ENV_MODE='DEMO'; $env:DATA_DIR='working/quiz9_fol_truth_demo_scored'; $env:TF_GRADER_MODE='1'`
and run `python -m uvicorn app.server:app --port 5001`. Open
`http://127.0.0.1:5001/problem/Q2/strips?version=A`. This uses synthetic scans
under ignored `working/` and does not access an active grading session.

Instructor-approved scoring: Q1 is 2 points; Q2–5 are 2 each,
split 1 for truth and 1 for the written response. This is 10 regular points.
Q6 and Q7 are 1 extra-credit point each; the existing total caps at 10. When
N/A is required, a blank written field receives 0 for that field. The rubric
stores this split under `fol_truth_assistance.scoring`. Clear fresh readings prefill the ordinary editable score and comment fields;
the TF reviews them against the scan before saving the page.

After locations are ready, the instructor can use “Prepare preliminary answer
readings” on Finish. IU Gemma (or Qwen) reads each saved crop, one request at a
time; successful proposals are saved in the session and retries skip them.
The individual “Prepare vision reading” button can rerun one crop. Model replies
wrapped in a JSON code fence are accepted. The model supplies only a proposed
T/F choice and short written response; it does not decide a grade.

The TF checks the bold interpreted answer against the scan and corrects the
truth or object/N/A choices as needed. The shared header’s expected answer includes every valid
object, so multiple acceptable witnesses or counterexamples remain visible.
Wrong answers prefill feedback; correct answers leave the generated comment
empty. The ordinary comment field accepts additional TF text, retained when the
answer changes. Saving persists the score, comment, and transcription together.
Generated explanations are enclosed in square brackets. Added TF text remains outside the brackets and survives answer corrections.
Accepted comments appear in exported feedback PDFs. Unclear responses require
human interpretation. Vision preparation is unavailable in model-free TF mode.

The review fingerprint includes assessment configuration, question, version,
saved segment, and source file size/modification time. A changed version, crop,
source scan, or configuration marks saved
readings stale on reload; the TF must reconfirm them. Opted-in private bundles
include configuration and review state in `session.json`, declare schema 1 and
the pinned shared-module commit in `bundle_manifest.json`, and their launchers
check required assets. Older ordinary bundles keep their behavior. The TF
needs updated grader code, with no Node or model connection for review.

For this Quiz 9 batch, prepare locations from the three-page blank PDF with
`python -m scripts.prepare_quiz9_template_locations --pdf <blank-quiz.pdf>`.
The default run validates the scan-to-blank alignment and prints aggregate
geometry only. Repeat with `--write` to save `segments_Q1.json` through
`segments_Q7.json` in the private `DATA_DIR/grades/` directory. The script
requires the expected Quiz 9 session, confirmed A/B/C versions, one scan per
submission, strong image registration, and no existing full location caches.
It does not call a model or save student crops. The Finish page then reads
those caches and shows the question locations as ready. The TF can adjust any
crop in strip view. The earlier IU-based partial Q1 cache is preserved and is
not used while the complete cache is valid.

For other quizzes without template caches, instructor-side question-location
preparation uses IU REALLMS Gemma by default, with an IU Qwen retry choice.
The Finish page prepares one submission per request and checkpoints each
successful location in a private partial cache; retrying skips completed
submissions. The final `segments_Q*.json` cache is written only after every
work unit for that question is valid. A failed IU request returns an actionable
502 without exposing model output or student text.

This is the operating note for paper quizzes that have multiple printed
versions, are photographed or scanned, and are handed to one TF for grading.
IU REALLMS Gemma performs name/version recognition and question location by
default. The TF can still grade from saved crops without running a model.

## Fresh-agent orientation

This document is the durable conversation handoff for the versioned paper-quiz
workflow. A fresh coding agent working on quiz intake, preparation, TA/TF
handoff, grading exports, or the GitHub release should read this file before
changing code. It does **not** need to read the proof-OCR history.

The intended workflow and product decisions are:

- One instructor prepares the quiz with IU REALLMS; one human TA/TF grades it
  without any model.
- Canvas API access is not part of this path. Intake uses a downloaded Canvas
  gradebook CSV. Scores return through a Canvas gradebook upload; the instructor
  manually attaches one named scan-and-feedback PDF per student.
- Human judgment is authoritative. Ordinary bundles do not generate an answer
  key, solution, or grading advice for the TA/TF. The instructor-approved Quiz 9
  opt-in above adds deterministic suggestions that the TF confirms and accepts.
  A grader may pin a student response they judge correct as a visual reference
  and should contact the instructor if course expectations are unclear.
- In strip view, the student answer crop is the visual focus. The blank printed
  question appears once, larger, and pinned at the top, with separate assets by
  question/version when needed.
- The currently exercised intake is phone scans. Consecutive pages assigned to
  the same student are one work unit. Flatbed/batch-scan refinements are future
  work, not a reason to complicate the current handoff.
- Real rosters, scans, sessions, crops, and exports live outside the repository.
  Never add them to Git, fixtures, documentation, release assets, or agent
  messages.

### Implementation map

| Concern | Current owner |
|---|---|
| Setup, uploads, name review, session creation | `app/server.py` and `app/templates/setup.html`, `name_review.html` |
| Name/version recognition and malformed-JSON recovery | `app/vlm_classifier.py`, `app/vlm_client.py` |
| Question-location generation and strip grading | `app/routes/problem_strips.py`, `app/templates/problem_strips.html` |
| Finish-page batch preparation | `app/templates/finish.html` using `POST /problem/{problem_id}/prefetch` |
| CSV, feedback PDFs, and TA/TF bundle exports | `app/routes/exam_exports.py` |
| Rubric/question-image preparation | `scripts/extract_quiz_reference_images.py` |
| Instructor launcher | `START_INSTRUCTOR_GRADER.bat`, `scripts/start_instructor_grader.py` |
| TA/TF launchers | generated by `app/routes/exam_exports.py`; standalone copies are under `scripts/` |

Fast offline checks for this domain:

```powershell
pytest tests/test_quiz_variant_tf_handoff.py tests/test_exam_exports_routes.py tests/test_vlm_client.py -q
```

Before a release, also run the full offline suite if practical. Live model
checks are deliberate manual canaries; do not invoke them merely to validate a
documentation or export change.

## What the project now supports

- A Canvas gradebook CSV can be reduced locally to the `Student` and
  `SIS Login ID` roster fields. Canvas API access is not required.
- The name-recognition pass also reads an explicit printed version label.
- Name and version guesses are editable before the session is created.
- Each work unit stores `quiz_version` (for example `A`, `B`, or `C`).
- A rubric problem can contain version-specific prompts under `variants`;
  PDF-to-YAML drafting does not generate solutions or tell the TF how to grade.
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
problems:
  - id: Q1
    label: Identifying the conclusion
    points: 2
    segmentation_hint: Include the printed Q1 label, argument, and handwritten answer lines.
    variants:
      A:
        prompt: Students who attend review sessions ...
      B:
        prompt: If the printers are out of paper ...
      C:
        prompt: The forecast calls for heavy rain ...
```

Choose the actual point values before saving the rubric. Extra-credit boxes can
be separate problems with `extra_credit: true`; the total-score calculation
caps regular plus extra credit at the quiz's regular maximum. The Finish page
shows this policy whenever a rubric contains extra credit so the instructor and
TF do not mistake EC for points above 100%.

## Answer-key trust boundary

The PDF-to-YAML prompt may transcribe printed questions, propose point values,
and describe crop geometry. It must not generate solutions, grading guidance,
or `answer_key` fields. The Setup generator defaults to IU Gemma; its draft
omits `rubric_notes` even if the model tries to supply them. Legacy answer keys
are not displayed in strip view, and both answer keys and rubric notes are
removed when a TF bundle is exported.
Reviewed per-question `fol_truth` definitions (and the legacy Quiz 9
`fol_truth_assistance` opt-in) are preserved separately from legacy `answer_key`
fields and declared in the bundle manifest. They supply inputs to deterministic
checking rather than model-generated solutions or partial-credit policies.

During grading, the instructor or TF may pin a student response they judge
correct. Pins are stored in `session.json` by problem and version, survive page
reloads, and travel in the TF bundle. They are visual comparison aids, not
machine-verified solutions. If the TF cannot determine how to grade a response
from course knowledge, they should contact the instructor promptly rather than
ask a model to settle it.

## Prepare the assignment
The Quiz 9 opt-in stores its reviewed deterministic model/formula configuration
in the explicit `fol_truth_assistance` field; it is not placed in legacy
`answer_key` fields and is declared separately in the bundle manifest.

1. Use an external production `DATA_DIR`; do not put real scans in Git.
2. Download a current Canvas gradebook CSV and upload it in Setup as the roster.
   Keep the original full gradebook on the instructor's machine.
3. Upload the phone scans and save the version-aware rubric.
   Scan intake compares SHA-256 content: a repeated upload of the exact same
   image is skipped, while two genuinely different scans with the same filename
   are both preserved (the later filename receives `_2`).
4. Run name recognition with IU Gemma.
   Before recognition starts, any exact duplicates left by an older app version
   are moved—not deleted—to `DATA_DIR/duplicate_scans_quarantine/<timestamp>/`.
5. On Name Review, correct every student and printed-version guess. For a lower
   name line, use **Move name preview down** to shift every preview together;
   **Show more of each page** makes the previews taller. These display settings
   are remembered per quiz and scan subfolder in this browser, without rerunning
   recognition. **Show full page** opens the scan in the wider preview column
   when the title or handwriting is hard to read. The 90° left,
   180°, and 90° right buttons rotate the actual intake image before session
   creation; review the guesses afterward or refresh recognition. **Save
   Corrections** writes exactly those selections to the local cache without
   renaming scans or starting grading. A later recognition refresh preserves
   the reviewed names while replacing the version guesses. If the same student
   has two separate submissions, the page warns you and offers to exclude
   either group. Mark **New submission starts here** when two adjacent tests
   belong to the same student; a confirmed version change also creates a
   boundary automatically. Excluded pages remain on the instructor's computer
   but do not enter the grading session or TF bundle.
6. Save and start grading. On the Finish page, click **Prepare all question
   locations** and wait for every problem to report ready. This sequentially
   calls the existing problem prefetch route and creates the required
   `segments_<problem>.json` files without making the instructor walk through
   blank grading screens.
7. Spot-check at least one strip page per problem. For this phone-scan workflow,
   correct especially
   rotated, perspective-skewed, shadowed, or partially cut-off pages.
8. Optionally pin one clearly correct response for each question/version you
   want to provide as a visual comparison. Pins can also be chosen by the TF.
9. Click **Prepare TF bundle**. The export refuses to proceed when a scan,
   version, or problem-location cache is missing. It also stops when duplicate
   student submissions remain. Resolve a duplicate on the Finish page, or
   deliberately choose **Keep both and prepare bundle** when the TF needs to
   compare them.

Keep the extracted TF bundle outside the repository. It contains protected
student names, identifiers, and scans.

## IU models and a future consolidated TF workflow

The ordinary app now defaults to `gemma-4-31B-it` at the pinned IU REALLMS
endpoint. Blank-quiz rubric drafting also offers IU Qwen; Name Review and strip
grading offer explicit Qwen retries. A retry regenerates the entire relevant
name batch or question-location batch. Save name corrections before retrying;
reviewed student choices survive a name refresh. Successful cached question
locations stay usable without a key. Failed requests or invalid location JSON
do not replace a successful location cache. No automatic provider fallback is
performed, and existing grading sessions are not reprocessed on startup.

Set `REALLMS_API_KEY` in a private environment or the ignored `.env`. On Windows,
the client also checks the user's and machine's environment registry entries,
so a previously saved key works without restarting the terminal. Set
`REALLMS_MODEL=qwen3.8-27b` to make Qwen the default, or explicitly set
`GRADING_VLM_PROVIDER=lmstudio` to restore local inference. Normal use needs no
LM Studio server. Keep keys out of source, private grading bundles, and logs.

On one human-reviewed 37-page quiz, each IU model correctly matched all 37 names.
The saved local run had 27 correct matches, nine unresolved names, and one wrong
student. Gemma took approximately 34 seconds and Qwen 52 seconds. These are
small-sample name-recognition results, not evidence about question geometry or
proof OCR. Keep human confirmation and evaluate each of those tasks separately.

`scripts/benchmark_name_recognition.py` preserves a private input snapshot and
separate results outside the repo. Its `serve` action opens a loopback review
panel using `app/templates/name_benchmark.html`; human reviews change only that
benchmark. Downloaded counts contain aggregate scores without student identities.

For eventual TF consolidation, use one local app on the TF's approved device:
upload scans and the minimal name/ID roster, run IU recognition, confirm names
and versions, prepare/check question locations, grade, and return exports. The
TF needs their own authorized IU API access and key, the normal student-data
authorization/training, and access to only their assigned course records.
[IU's current tool matrix](https://datamanagement.iu.edu/faq/genai.html) explicitly
approves REALLMS for FERPA data. Confirm the allocation and retention arrangements
when provisioning access; a key itself does not establish access rights to a class.

This consolidated TF intake workflow has not yet been tested on the TF's device.
Before handoff, exercise a fresh installation, key setup, network failure/retry,
scan grouping, crop correction, and full export with fake data. Instructor and TF
must agree who owns the current session; avoid parallel copies that both acquire
grades. Keep the prepared, model-free TF bundle path as a fallback. Reusable
blank-template rectangles for stack scans may reduce calls and improve blank
answer coverage; human crop corrections must remain authoritative.

## Blank-question images

For the two-page-per-version P250 truth-table quiz, use the dedicated preset:

```powershell
python scripts/extract_quiz_reference_images.py `
  "C:\path\to\P250_Truth_Table_Quiz_All_Versions_3-3-4.pdf" `
  --quiz-id P250_F26_truth_table_quiz_ABC `
  --preset p250_truth_table_quiz `
  --output-dir "C:\grading-data\live\rubric_assets"
```

This writes four prompt images for each of Versions A/B/C and understands that
each version occupies two consecutive PDF pages.

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
2. Create a descriptive tag such as `p250-f26-quiz1-grader-v1` and attach the
   audited code ZIP described below as an explicit release asset.
3. Publish the release and send the TF the release-page link. Under **Assets**,
   the TF downloads the explicitly named grader ZIP—not the automatic generic
   **Source code (zip)** archive—and extracts it.
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
tracked files. `git archive` is the starting point; inspect its contents and
exclude any tracked synthetic/sample artifacts that are unnecessary for the
TA/TF release:

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
The concrete GitHub UI and handoff checklist is in
[`TA_RELEASE_CHECKLIST.md`](TA_RELEASE_CHECKLIST.md).

## Known operational edges

- Both downloads must be fully extracted. Windows ZIP preview opens files from
  a temporary path and makes launcher troubleshooting confusing.
- On Windows, the reliable way to open PowerShell in a folder is to open that
  folder in File Explorer, click the address bar, type `powershell`, and press
  Enter. The prompt should name that folder, not only `PS C:\>`.
- The bundle launcher needs the extracted **code** folder as `-CodeDir`; that
  folder contains `app/server.py` and `requirements.txt`. The launcher itself
  must be run from the extracted **bundle** folder beside `data/grades/session.json`.
- A save can be slow because the next crop is prepared. The UI disables repeat
  submission and tells the grader to wait; do not interpret a short delay as a
  failed save without reloading and checking persisted values.
- The TA/TF bundle export intentionally refuses to run until every problem has
  a saved question-location cache. Use the Finish-page batch preparation button
  rather than teaching the TA/TF to create missing caches.
- TF grader mode hides instructor-only controls. It should not expose setup,
  classwork, Canvas merge, renamed-scan export, or creation of another bundle.

## Fresh-conversation release continuation

If the next task is “finish the TA GitHub release,” a fresh agent should:

1. Read this document and `docs/TA_RELEASE_CHECKLIST.md`.
2. Run `git status --short` and inspect all uncommitted changes; preserve
   unrelated user work.
3. Run the paper-quiz offline tests above (and the full offline suite when
   practical).
4. Commit only the intended code/docs changes.
5. Build and inspect an audited code ZIP from that commit. Do not copy the
   working tree wholesale, and do not include live external data.
6. Give the user the ZIP path, commit hash, checksum, and exact GitHub release
   steps. Do not push or publish a release unless the user explicitly requests
   that external action.
7. Remind the user to regenerate the private grading bundle after changes to
   its generated launcher or `READ_ME_FIRST.txt`; that bundle is sent through a
   university-approved channel, never GitHub.

## Receive and import grades

Final CSV, feedback PDF, and gradebook merge operations reject identified
submissions with blank regular-question scores. Use explicit 0 for no credit;
blank optional EC counts as zero. The Finish page lists incomplete grades.
TF bundle preparation remains available before grading.

Ask the TF to return:

- the downloaded scores CSV;
- the scans + feedback PDFs ZIP, containing one named PDF per student; and
- the completed bundle's `data/grades/session.json` as the recovery copy.

On the instructor machine, first preserve your original prepared session, then
copy the TF's returned `session.json` into the active external
`DATA_DIR/grades/session.json`. Restart or reload the app. Open **Gradebook
merge tool**, upload a fresh Canvas gradebook CSV, choose the Quiz 1 assignment
column, review the match/conflict summary, and download the merged CSV for
Canvas upload. The returned scores CSV is also a readable cross-check, but the
current merge page reads authoritative scores from `session.json`.

The ordinary feedback ZIP contains one PDF per student, named with the student
name and SIS ID. Each PDF preserves the original scan pages and appends a typed
feedback page with the total, per-question scores, and comments. Attach that
single PDF manually in Canvas. The Canvas re-upload ZIP is meant for files
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
