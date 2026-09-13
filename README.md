# Logic Grading Facilitator

> Experimental pre-release for an instructor-supervised TA trial, not ready for
> general public or unattended production use. Independently check grades and
> exports before uploading to Canvas. OCR/model output is assistance, not an
> authoritative grade. Never upload student work or rosters to this repository.

The model-free TA workflow is tested on Windows with Python 3.13. macOS launcher
instructions are provided but have not been tested on a fresh Mac. Install with
`python -m pip install -r requirements.txt`; it uses the tested dependency pins
in `requirements-lock.txt`. Update and re-test those pins for future releases.

A local, privacy-first grading tool for scanned logic assignments. The main
exam workflow groups scans into student submissions, lets an instructor grade
one problem across all students in a strip view, and exports Canvas-ready
grade files. The repo also contains active OCR/autograding experiments for
Fitch proofs, modal proofs, FOL models, and Kripke models.

For coding-agent work, start with [AGENTS.md](AGENTS.md). For a fuller doc map,
use [docs/README.md](docs/README.md).

## Current Entry Points

- [HOWTOUSE.md](HOWTOUSE.md) - operator workflow for setup, grading, export,
  classwork, flashcards, and archive/reset.
- [docs/TA_GRADING_GUIDE.md](docs/TA_GRADING_GUIDE.md) - exact Windows/macOS
  setup, launch, grading, and return steps for a TA/TF using a prepared bundle.
- [docs/TA_RELEASE_CHECKLIST.md](docs/TA_RELEASE_CHECKLIST.md) - instructor
  checklist for publishing the code on GitHub without including course data.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - current FastAPI/Jinja route
  map, data flow, caches, and integration points.
- [docs/SCHEMA.md](docs/SCHEMA.md) - persisted JSON/YAML data contracts.
- [docs/DEV_WORKFLOWS.md](docs/DEV_WORKFLOWS.md) - practical command recipes.
- [docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md) - coding-agent test commands,
  live-model guardrails, and OCR workflow notes.
- [docs/FITCH_OCR_AGENT.md](docs/FITCH_OCR_AGENT.md) - current TFL/Fitch OCR
  operating guide.

## Codebase Layout

Three semi-independent domains live side by side:

- `app/` - FastAPI app, Jinja templates, grading UI, Canvas/classwork flows,
  session storage, export routes, and shared autograde integration.
- `proofs_ocr/` - TFL/Fitch and modal proof OCR pipelines, regression runners,
  fixtures, and normalization helpers.
- `modal_ocr/` - modal proof OCR/checker support modules used by the modal
  debug workflow.

Supporting scripts live in `scripts/`; tests live in `tests/`; ignored local
experiments belong under `working/`.

## Data Location

All `data/...` paths in docs mean paths under `DATA_DIR`. By default,
`DATA_DIR` is `data/samples` for fake/sample data. For real student data, set
`ENV_MODE=PRODUCTION` and point `DATA_DIR` outside the repo; the app refuses to
start in production if `DATA_DIR` is inside the repository.

Example `.env`:

```text
DATA_DIR=C:\grading-data\live
DATA_ARCHIVE_DIR=C:\grading-data\archive
ENV_MODE=PRODUCTION
```

Dev workflow:

- Leave `DATA_DIR` unset or point it at `data/samples`.
- Use fake roster/scans/rubric data.

Real grading workflow:

- Set `ENV_MODE=PRODUCTION`.
- Point `DATA_DIR` and `DATA_ARCHIVE_DIR` to external folders.
- Do not paste real scans, names, LMS IDs, or raw OCR output into docs or agent
  conversations.

Before changing live-data workflows, review the FERPA section in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#5-ferpa--compliance).

## Main Exam Workflow

Start the server:

```powershell
python -m uvicorn app.server:app --reload --port 5000
```

Then open `http://127.0.0.1:5000/setup`.

The setup wizard can archive/reset an existing session, select or upload a
roster, import scans as images/PDF/Canvas ZIP, edit the rubric YAML, run
auto-grouping/name recognition, and send you to `/name_review`.

After name review, grade in strip view:

```text
http://127.0.0.1:5000/problem/Q1/strips
```

When finished, open:

```text
http://127.0.0.1:5000/finish
```

The finish page links to grade export, renamed scans, student scan/feedback PDFs, and the
gradebook merge tool.

## Current Automation Status

- `fol_model` has OCR/debug support, strip-view review, and deterministic
  checker integration through `app/autograde/`.
- `kripke_model` has a deterministic checker for already-transcribed model
  text. OCR for graph drawings is future work.
- `modal_proof` has a backend checker path through the external `deduct` CLI
  wrapper plus `/debug/modal-proof` for proof-image debugging.
- `fitch_proof` has the active handwritten TFL/Fitch strip OCR/debug workflow.
  The shared autograde scaffold exists, but human review remains authoritative.

See [docs/AUTOGRADE_ARCHITECTURE.md](docs/AUTOGRADE_ARCHITECTURE.md) for the
current contract and status table.

## TFL/Fitch OCR

The active handwritten TFL/Fitch OCR implementation lives in
`proofs_ocr/pipelines/tfl_fitch.py`, with a thin CLI wrapper at
`scripts/debug_fitch_ocr.py`.

Fast offline checks:

```powershell
$env:PYTHONPATH='.'
pytest tests/test_debug_fitch_ocr.py tests/test_tfl_fitch_ocr_regression.py -q
python -m proofs_ocr.regression.tfl_runner summary
```

Live OCR runs require a local VLM endpoint:

```powershell
python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_handwritten_q6_S_iff_W
```

Read [docs/FITCH_OCR_AGENT.md](docs/FITCH_OCR_AGENT.md) before changing OCR
behavior or strip-view OCR integration.

## Third-Party Notices

The Fitch proof parser/verifier in `app/syntax.py` and `app/verifier.py`
includes code adapted from `OpenLogicProject/fitch-checker` under `GPL-3.0`.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt).

The repository also includes `deduct-master.zip`, a bundled archive of the
separate `Colonial-Dev/deduct` project under `AGPL-3.0`. It is kept as a
vendored reference/tool archive rather than Python code adapted into this
codebase. See [VENDORED_COMPONENTS.md](VENDORED_COMPONENTS.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt).

## Collaboration Note

When an agent task would be significantly faster or less error-prone if a human
manually fetches or pastes a file, the agent should say so before doing a slow
workaround.
