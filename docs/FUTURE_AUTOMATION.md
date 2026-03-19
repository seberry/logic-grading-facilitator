# Future Automation Roadmap

## Application Icons

This project uses three distinct icons for different grading workflows:

### icon_classwork.png / icon_classwork.ico
**Classwork & Attendance Grader** (Currently Available)
- **Route:** `http://127.0.0.1:5000/classwork`
- **Purpose:** Effort-based grading for classwork and attendance questions
- **Workflow:** No local model required. Grades are auto-filled (submitted = max points, missing = 0)
- **Features:** Fast review UI with bulk actions, Canvas gradebook merge, optional reupload ZIP for comments
- **Launch:** Double-click `start_classwork.bat` or its desktop shortcut

### icon_examgrader.png
**Exam/Quiz Grader** (Currently Available)
- **Route:** `http://127.0.0.1:5000/setup`
- **Purpose:** Full-featured grading for multi-question exams and quizzes
- **Workflow:** Requires LM Studio with Qwen VLM model running
- **Features:** 
  - Auto-grouping: VLM reads names to group scattered pages into submissions
  - Per-problem segmentation: VLM crops answers by question for strip view grading
  - Fitch proof verification: Hybrid CV+VLM pipeline for proof checking
  - Canvas export: CSV, feedback ZIP, renamed scans ZIP
- **Launch:** `python -m uvicorn app.server:app --reload --port 5000`, then navigate to `/setup`

### icon_absence_basket.png
**Absence Basket** (Planned - Unbuilt as of Spring 2026)
- **Purpose:** Lightweight desktop tool for handling student absence and extension requests
- **Concept:** Professor copies student request text (submitted via structured markdown template), pastes into app, and local LLM parses it
- **Why Local:** Requests may contain medical/personal information; local model ensures FERPA/HIPAA compliance
- **Key Features:**
  - Structured template with Canvas assignment links for deterministic parsing
  - Local LLM (Qwen or Llama 3 8B) generates proposed actions (excused grades, extensions, participation credit)
  - Editable confirmation form before executing actions
  - Canvas API integration for:
    - Per-student assignment overrides (custom due dates)
    - Excused grade posting
    - Submission comments to students
- **Status:** Icons designed and reserved. First step is creating the structured student request template.


This roadmap describes how more automation will be layered **on top of** the v2 design (local VLM translator + deterministic verifier + human judge).

## Phase 1: Robust Pre-computation

- **Task A – Student grouping (Status: prototype exists)**  
  VLM reads the name region of each page, suggests a student ID; we group pages into WorkUnits (submissions).

- **Task B – Per-page segmentation (Status: early prototype which just displays pictures of each answer to a given question)**  
  For every (page, problem_id), ask the VLM:
  - `present`: is this problem’s answer on this page?
  - `bbox`: normalized per-page rectangle containing the answer.

- **Task C – Draft transcription (Status: planned)**  
  For selected problem types:
  - Kripke models → JSON graph structure.
  - Proofs → structured line-by-line representation.

All of this happens asynchronously after ingestion so that the grading UI is snappy.
- **Details re C: Fitch Proof Transcription (The "Coloring Book" Pipeline)**
  *Status: Prototyped & Verified*
  
  We have confirmed that LLMs struggle to recognize vertical assumption lines or align them with text. The solution is a hybrid CV+VLM pipeline:
  
  1.  **CV Pre-pass (Structure):**
      - Use `scripts/bold_fitch_lines.py` to detect vertical lines using morphological operations.
      - Overlay transparent colored rectangles (Cyan, Magenta, Yellow) extending to the right of each line.
      - Nested scopes naturally create mixed colors (e.g., Cyan + Magenta = Purple).
      
  2.  **VLM Pass (Transcription):**
      - **Input:** The color-coded image.
      - **Prompt:** *"Can you transcribe what you see on each line in black ink (ignore everything in red). Next to each numbered line mark the background color in brackets."*
      - **Role:** The VLM acts as a scribe, reporting text and color. It does *not* calculate depth.
      
  3.  **Deterministic Parsing:**
      - Map reported colors to integer depths (e.g., `[Cyan]` -> Depth 1).
      - Combine with OCR text to build the structured proof object.

### Critical Constraints (Lessons Learned)
For the Fitch pipeline to work, the input image must be **aggressively cropped**:
* **No Stray Numbers:** Visible question numbers (e.g., "6.") cause the VLM to hallucinate line numbers or alignment.
* **No Stray Lines:** Ruled paper margins (red vertical lines) confuse the CV line detection.
* **Implication:** The `segmentation_core.py` (or manual crop tool in Strip View) must provide a clean crop of *just* the logic grid before processing.




## Phase 2: Strip View + Verification Loop

- Implement the three-column strip view described in `DESIGN_v2.md`.
- Add:
  - Manual re-crop that re-triggers transcription for just that strip.
  - Live rendering from transcription (graph / proof table).
  - Integration with the logic engine to show interpreted results and suggested feedback snippets.

[maybe also create: “Anchor” system for pinning canonical answers / common mistakes to the top of the grading view.]


# Phase 3: Clustering / Grouping Similar Answers
Goal

Automatically cluster similar answers for consistent grading.

Pipeline

Extract embeddings from OCR text (or image features for handwritten text).

Cluster answers (k-means, agglomerative, etc.).

UI shows clusters as columns:

drag & drop reassignments allowed.

Assign scores/comments per cluster.

Write scores to WorkUnits.

Invariants

Clustering suggestions never override human judgment.

Cluster-level scores can be overridden per student.


# Phase 4: Autograding

Submodules
## 4.1 Deterministic Checkers

Symbolization equivalence checker

Truth-table / model checker

Fitch-proof checker

- **Status:** Ported from [OpenLogicProject/fitch-checker](https://github.com/OpenLogicProject/fitch-checker).
- **Files:** `app/syntax.py` (formula parser) and `app/verifier.py` (rule checker).
- **Input expectations:** per-line JSON-like records with `wffstr`, `location` (indent stack), and `jstr` (rule citation string). The parser accepts Fitch symbols (`¬ ∨ ∧ → ↔ ∀ ∃ ⊥`) and can run in propositional or predicate mode.
- **Output:** `VerificationResult` with `issues` (per-line errors) and `conc_reached` flag for concluding lines. Designed to sit between OCR→structure transformation and UI feedback.

Quantifier logic evaluations

Each module returns an auto_score and optionally an explanation.

## 4.2 LLM-Based “Second Opinion”

Local LLMs classify answers as correct/incorrect.

Only used for flagging disagreements, not assigning grades.

```
Data Structure
"per_problem": {
  "Q1": {
    "human_score": 3,
    "auto_score": 0,
    "flag_disagreement": true
  }
}
```
Grading Flow

Human grades → autograder runs → disagreements presented.

In a later phase: autograder grades → human inspects disagreements.

# Phase 5: Optional Programmatic Feedback Upload to Canvas
Allowed

Upload per-student comments like “Q2: missing witness”.

Safe under FERPA because feedback is not another student’s data.

Not Allowed

Auto-upload scans/images (risk of cross-student disclosure).

Summary

The roadmap builds from a minimal but robust graded structure (WorkUnits + GradingKey) toward full semi-automated grading. Every phase slots into the same data model and UI flows defined in v0, ensuring long-term maintainability and clarity.
