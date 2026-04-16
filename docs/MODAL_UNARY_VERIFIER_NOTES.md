## Modal Unary Verifier Notes

This note captures the current state of handwritten modal proof OCR debugging,
focused on unary/modal symbol misreads such as:

- `¬□¬A` vs `¬□¬¬A`
- `□¬A` vs `□¬¬A`
- `□...` vs `¬¬...`

The main failing handwritten example used during this work is:

- `C:\Users\seber\grading-data\live\debug_modal_proof\q1_3_ps11_anderson_blane_p01`

Useful experiment artifact directories in the repo workspace:

- `working/modal-proof-verifier-rerun6`
- `working/modal-proof-verifier-rerun7`
- `working/modal-proof-verifier-rerun13`
- `working/modal-proof-verifier-rerun14`
- `working/modal-proof-verifier-rerun15`
- `working/modal-proof-line-number-locator`
- `working/modal-proof-line-number-locator-leftonly-gutter`
- `working/modal-proof-margin-band-ocr`
- `working/modal-proof-margin-cc-ocr`

`/debug/modal-proof` now exposes the verifier artifacts directly, including
alternate crop attempts for each unary verifier row.

## Current Architecture

The modal OCR debug pipeline currently has two stages:

1. Baseline OCR
   - model: typically `qwen/qwen3-vl-4b`
   - recursive OCR over subproof crops with stitch/renumber
2. Unary verifier experiment
   - model: typically `qwen/qwen3-vl-8b`
   - runs only after baseline OCR if the checker reports issues
   - targets unary-heavy lines
   - uses isolated verifier crops from unmasked imagery
   - upscales verifier crops before the second vision pass
   - records results in `unary_verifier.json` and `verifier_line_*.jpg`

The verifier is still debug-only. It records `proposed_correction` and
`would_auto_apply`, but it does not rewrite the saved final proof structure.

## What Has Worked

- Better isolated line crops plus 4x upscaling materially improve `8b`.
- Explicit count-first prompts work better than generic “choose the best
  formula” prompts.
- The verifier can recover line 4 in the target case as `¬□¬¬A`.
- The verifier can preserve the strict-subproof control line `¬¬A` when the
  crop is correct.
- Exposing alternate verifier crop attempts in the debug UI makes the
  remaining crop-selection failures much easier to inspect.

## What Has Not Worked

- Generic hard-coded unary candidate pools were too brittle.
  - Example failure mode: offering non-modal candidates for modal lines, or
    modal candidates for pure negation lines.
- Checker-improvement as a line-level reread trigger was too strict and masked
  true OCR errors.
  - Real student mistakes can hide OCR mistakes.
  - OCR mistakes can surface as checker errors on different lines.
- Rereading from masked or red-tokenized crops was a dead end for fine unary
  disambiguation.
- A full-image “return all line centers in reading order” anchor query did not
  work well.
  - In practice it returned near-evenly-spaced anchors rather than geometry
    that tracked the handwriting.
  - That experiment was useful as a negative result, but it did not solve crop
    selection.
- Left-strip VLM localization prompts were only partially successful.
  - Asking for “which letter/color/band is line number 3 next to?” produced
    monotonic but noisy answers.
  - Moving labels into a white gutter improved the result, but not enough to
    trust for verifier crop selection.
- Margin-only overlapping-band OCR was better than direct localization, but it
  still missed or confused middle rows.
  - It could recover top and bottom numbers.
  - It still confused the central region enough that it was not a clean anchor
    source by itself.

## Current Main Bottleneck

The remaining blocker is crop selection for verifier rereads, especially the
line that becomes final proof line 8 in the target example.

The verifier often has real signal once it sees the correct isolated line. The
hard part is choosing the correct local band among several vertically nearby
candidates inside the relevant node crop.

More specifically:

- the old verifier crop extraction was too loose and often included more than
  one proof row
- after tightening those crops, the next real bottleneck became row selection
  rather than reread prompting
- the current `_detect_text_row_bands()` logic still merges too much in the
  lower half of the page

## Margin-Anchor Experiment

The most promising new direction so far is to stop asking the VLM to localize
 line numbers and instead use deterministic CV to propose left-margin anchors.

The experiment pipeline was:

1. Crop the original image to the far-left margin, stopping at the first proof
   bar.
2. Upscale that strip 4x.
3. Threshold dark ink and run connected components.
4. Merge nearby components that plausibly belong to the same handwritten
   numeral.
5. Save candidate crops and ask `qwen/qwen3-vl-8b` only:
   - “what number is in this tiny crop?”

This is a very different problem from “where is line 8?” and was materially
more successful.

Useful artifacts:

- `working/modal-proof-margin-cc-ocr/candidate_boxes.jpg`
- `working/modal-proof-margin-cc-ocr/classification_summary.jpg`
- `working/modal-proof-margin-cc-ocr/proposed_line_anchor_boxes.jpg`
- `working/modal-proof-margin-cc-ocr/order_based_anchor_inspection.jpg`

Observed result on the target example:

- top and bottom anchors were recovered cleanly
  - `1`, `2`, `3`, `4`, `9`, `10`
- the two ambiguous middle anchors correspond visually to the real `5` and `8`
  even though `8b` hesitated on them
- the geometry is already good enough that ordering the candidate centers
  top-to-bottom gives a plausible `1..10` sequence

Important conclusion:

- for these proof images, classification is optional
- if the margin contains the expected monotone sequence `1..n`, then the
  useful output is a reliable ordered list of y-anchors
- labels can be assigned by order even if the VLM is uncertain about the
  handwritten glyph

## Current Best Geometry

The cleanest current inspection view is:

- `working/modal-proof-margin-cc-ocr/order_based_anchor_inspection.jpg`

That image shows:

- left panel: connected-component candidates in the margin, assigned `1..10`
  by vertical order only
- right panel: full-width reread strips obtained by taking midpoint boundaries
  between those ordered anchors

This is currently the best candidate replacement for the verifier’s local
row-band chooser.

Two caveats remain visible in that inspection:

- the raw `6` candidate is under-boxed
- the raw `7` candidate center is somewhat low

Those problems are caused by thresholded connected-component geometry, not by
line ordering itself. They should be addressable by:

- expanding/merging fragmentary components before taking centers
- smoothing centers by neighboring spacing after ordering

## Current Best Known Status On The Target Case

From the recent reruns:

- line 4 can be recovered as `¬□¬¬A`
- strict-subproof line 7 can remain `¬¬A`
- line 8 still tends to stay as `□¬A`

This suggests the verifier prompt/model setup is promising enough, but local
crop selection is not yet robust enough.

## Good Next Experiments

The most plausible next step is now margin-anchor-driven rereads, not more
prompt engineering for raw band selection.

The intended pipeline is:

1. Detect left-margin numeral candidates with CV.
2. Order them by y and assign `1..n`.
3. Convert those ordered anchors into full-width reread strips using midpoint
   boundaries between consecutive anchors.
4. Feed those strips to `qwen/qwen3-vl-8b` for unary rereads.
5. Keep the old node-local band detector only as a fallback/debug comparator.

This still preserves the useful separation:

1. anchor / strip selection
2. unary reread of the selected strip

## Debug-Panel Wiring Status

The margin-anchor path is now wired into the debug OCR flow as an additional
verifier crop strategy.

Current behavior:

- The OCR rerun writes fixed margin-anchor artifacts:
  - `margin_anchor_margin_base.jpg`
  - `margin_anchor_margin_upscaled_4x.jpg`
  - `margin_anchor_binary.png`
  - `margin_anchor_candidate_boxes.jpg`
  - `margin_anchor_strip_overlay.jpg`
  - `margin_anchor_geometry.json`
- If the margin-anchor geometry finds exactly `1..n` candidates, the unary
  verifier now creates per-line anchor strips such as:
  - `verifier_line_anchor_4.jpg`
  - `verifier_line_anchor_8.jpg`
- The old node-local band crops are still produced and still act as fallback
  / comparison crops.
- `unary_verifier.json` now records:
  - `attempt_kind`
  - `strip_y1`
  - `strip_y2`
  - chosen crop strategy for the best attempt
- `/debug/modal-proof` now shows the margin-anchor artifacts in the image
  artifact panel, and the unary verifier table labels attempts as
  `margin anchor` vs `local band`.

Current limitation:

- On the target handwritten case, the lower-half margin geometry still drifts.
  - `rerun14` proved the debug wiring and artifact flow but still missed the
    real `7` margin anchor and kept the old local-band crop as the winner for
    line 8.
  - `rerun15` improves the lower half by:
    - searching a few narrower left-margin widths and choosing the best layout
    - merging vertically stacked digit fragments before ordering
  - With that change, the target case now recovers a usable `1..10` anchor
    list again and line 8 can be selected from `margin_anchor`.
  - Line 10 still does not prefer the anchor crop, so the old local-band path
    remains an important fallback.

## Current Rerun Reference

The current integrated reference rerun is:

- `working/modal-proof-verifier-rerun15`

That run is the one to inspect when checking:

- `margin_anchor_geometry.json`
- `margin_anchor_candidate_boxes.jpg`
- `margin_anchor_strip_overlay.jpg`
- the chosen `verifier_line_anchor_*.jpg` crops in `unary_verifier.json`

## Next Tuning Target

The next geometry iteration should focus on:

- `_extract_left_margin_strip`
  - candidate width selection for the left strip
- `_merge_margin_candidate_boxes`
  - fragment-merging rules for stacked digit pieces
- `_score_margin_candidate_boxes`
  - ranking layouts when several strip widths produce plausible candidates
- `_margin_anchor_strip_bounds`
  - only if strip heights need more regularization after the box geometry is
    stable

## Prompt Notes

The prompts that worked best were explicit and local, for example:

- ask for exact counts of negation marks
- name the modal operator explicitly
- ask only about one isolated formula line
- keep the candidate list short when using forced choice

Weak phrasing such as “count the visible unary marks carefully” was too
abstract. Much better wording was:

- “Count exactly how many negation marks (¬) appear immediately to the LEFT of
  the box symbol.”
- “Count exactly how many negation marks (¬) appear immediately to the RIGHT of
  the box symbol before the final propositional letter.”

## Code Areas To Inspect First

- `scripts/debug_modal_proof_ocr.py`
  - baseline OCR flow
  - verifier crop extraction
  - row-band detection
  - unary verifier prompt builders
  - alternate attempt selection
- `app/routes/modal_proof_debug.py`
- `app/templates/debug_modal_proof.html`
- `tests/test_debug_modal_proof_ocr.py`

## Suggested Restart Prompt

If continuing this in a fresh conversation, this prompt should be enough:

```text
We are continuing modal proof OCR debugging in `logic-grading-facilitator-MAIN`.

Please read:
- `docs/MODAL_UNARY_VERIFIER_NOTES.md`
- `scripts/debug_modal_proof_ocr.py`
- `app/routes/modal_proof_debug.py`
- `app/templates/debug_modal_proof.html`
- `tests/test_debug_modal_proof_ocr.py`

Current setup:
- baseline OCR model: `qwen/qwen3-vl-4b`
- unary verifier model: `qwen/qwen3-vl-8b`
- unary verifier is debug-only and writes `unary_verifier.json`
- `/debug/modal-proof` shows chosen and alternate verifier crop attempts

Known failing case:
`C:\\Users\\seber\\grading-data\\live\\debug_modal_proof\\q1_3_ps11_anderson_blane_p01`

Current status:
- line 4 can be recovered as `¬□¬¬A`
- strict-subproof `¬¬A` can be preserved
- line 8 is still not reliably fixed
- main bottleneck is choosing the correct verifier crop band

Please propose and, if appropriate, implement the next experiment for verifier
band selection. Good candidate directions:
- asking the model to choose among highlighted candidate bands
- doing one-line-at-a-time local anchoring within the relevant node crop
- separating band choice from unary reread
```
