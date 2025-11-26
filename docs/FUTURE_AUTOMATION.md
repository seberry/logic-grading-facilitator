# Future Automation Roadmap


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
