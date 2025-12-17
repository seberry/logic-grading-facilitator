# Strategy: Fitch Proof Transcription

**Status:** Verified Working (Dec 2025)
**Goal:** Extract structured Logic Proofs (Line #, Depth, Formula) from handwritten scans.

## The Core Problem
Local VLMs (and even SOTA models) struggle to detect thin vertical lines and associate them with specific rows of text. They often ignore the lines entirely or hallucinate their start/end points.

## The "Coloring Book" Solution
Instead of asking the model to "understand" structure, we explicitly paint the structure onto the image using Computer Vision.

### 1. The CV Script (`scripts/highlight_fitch_lines.py`)
* [cite_start]**Step A:** Morphological open with a `(1, 60)` vertical kernel isolates the vertical bars[cite: 922].
* **Step B:** `cv2.findContours` identifies the bar locations.
* [cite_start]**Step C:** We draw a semi-transparent (alpha=0.2) rectangle from the bar to the right edge of the image[cite: 928].
* **Step D:** We cycle colors (Cyan, Magenta, Yellow). Overlapping scopes (nesting) create mixed colors (Purple, Green) automatically.

### 2. The Prompt
We found complex prompts caused hallucinations. The best performance came from this minimal instruction:

> "Can you transcribe what you see on each line in black ink (ignore everything in red). Next to each numbered line mark the background color in brackets (e.g. light blue, purple)"

### 3. Failure Modes & Constraints
The VLM is easily distracted. For this strategy to work, the input image **must** be cropped to exclude:
1.  **Question Labels:** If "6." is visible above the proof, the model may merge it with Line 1.
2.  **Ruled Paper Margins:** The CV script might mistake the red margin line for a logic bar if not cropped out or filtered by color.
3.  **Neighboring Text:** Any text outside the proof grid can bleed into the transcription.

**Recommendation:** Use the "Strip View" manual crop tool to isolate the proof grid perfectly before running the "Coloring Book" script.

## Linking OCR to Rule Checking

Once a transcription pipeline produces structured lines, feed each line into `app/syntax.py` (for WFF parsing) and `app/verifier.py` (for Fitch rule validation). The verifier expects the OCR transform to provide:

- `wffstr`: the raw line formula using Fitch symbols (¬ ∨ ∧ → ↔ ∀ ∃ ⊥),
- `location`: an indentation stack for subproof scopes, and
- `jstr`: the cited rule/line references.

`VerificationResult` then reports per-line issues and whether the conclusion was reached, giving immediate grading signals without manual re-entry.

## Debugging the Recursive Pipeline

Use `scripts/debug_fitch_ocr.py` to exercise the full "chop, mask, transcribe, stitch" loop on a single cropped scan. The script:

1. Detects vertical bars, builds a subproof tree, and saves `debug_bars.jpg` for sanity checks.
2. Masks each child scope with `<SUBPROOF_ID_X>` tokens before sending the parent to the local VLM.
3. Stores every intermediate crop, mask, VLM response, and stitched JSON under `data/debug_fitch/<scan_stem>/`.
4. Normalizes the stitched proof into the verifier format (Unicode connectives, uppercase atoms, mapped rule labels, auto-Hyp on subproof starts, top-level Premise labeling) and renumbers lines so scopes are globally consistent before handing off to `app/verifier.py`.
5. When the verifier is available, it flattens the proof, infers a target conclusion from the final line, and runs a rule check to surface any structural issues directly in the CLI output.

Run it with:

```bash
python scripts/debug_fitch_ocr.py quiz1_studentA_p1.jpg
```

Tip: Start with a tightly cropped proof grid—excluding question text and margin lines—to minimize hallucinated structure.

## Human-in-the-loop verification UI
Artifacts written to `data/debug_fitch/<scan_stem>/` can be reviewed in-browser at `GET /debug/fitch`. The page previews the highlighted scan, lets you re-run the pipeline, edit and save `final_structure.json`, and push the saved structure through `app/verifier.py` to see rule issues without leaving the UI.
