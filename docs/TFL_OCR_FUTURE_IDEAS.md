# TFL OCR — Future Strategies to Try

> Suggestions from Gemini (2026-04-25) for improving small-proof OCR.
> Organized by priority tier based on our assessment. The pipeline uses
> **qwen3-vl** served locally via LM Studio for FERPA compliance.

---

## What We've Already Tried

See `docs/FITCH_OCR_AGENT.md` → Experiment History for full details.

| Approach | Result | Status |
|----------|--------|--------|
| Context-aware crops (green box on parent image) | VLM ignores green box, reads full image → 14 lines | ❌ Disabled |
| Letterboxed tight crops (white padding to 4:3) | 14→9 lines, reduced boundary leakage | ✅ Active |
| Minimal prompt tweaks ("only visible lines") | Slight improvement on leaf crops | ✅ Active |
| `8` → `∧` symbol normalization | Works correctly at comparison time | ✅ Working |
| `∈` → `E` normalization | Works correctly | ✅ Working |

---

## Tier 1: High Priority (cheap, high potential impact)

### A. Dynamic Resolution Control

Qwen handles images using dynamic resolution with `min_pixels` / `max_pixels` parameters. A tiny, wide crop might be getting resized in a way that destroys handwriting quality.

**Action items:**
- Check if LM Studio exposes `min_pixels` / `max_pixels` via its OpenAI-compatible API
- If using HuggingFace Transformers directly, set `min_pixels` and `max_pixels` in the processor
- Try hardcoding `resized_height` and `resized_width` (rounded to nearest multiple of 28)
- **Effort:** Investigation + ~1 hour if supported

### B. JSON Schema / Output Length Constraint

Force the model to output exactly N line objects to prevent hallucinated extra lines.

**Action items:**
- Test LM Studio's `response_format: { type: "json_object" }` support
- Investigate grammar constraints (llama.cpp supports JSON schema enforcement)
- Even without schema enforcement, try stronger prompt language: "The image contains exactly 2 proof lines. Output exactly 2 JSON objects."
- **Effort:** ~30 minutes to test

### C. Try Larger Model (qwen3.6-27b) — IN PROGRESS

The 4b model can't distinguish formula from justification columns in Q4 crop_3 — it reads `∧E 5` (justification) as a third formula line `"8 ∈ S"`. An 8b model was already probed and showed the same issue. A **27b model** (qwen3.6-27b) shows promise: it takes ~3 min per crop but does deliver correct transcriptions for the K∧W subproof with a dedicated transcriber prompt.

**Probe script:** `scripts/probe_subproof_ocr.py`
- Tests subproof crops against 4 prompt variants (pipeline baseline + 3 transcriber versions)
- Captures thinking text from Qwen3's `reasoning_content`
- Auto-compares against `expected_structure.json` with full pipeline normalization
- Results saved to `working/subproof_probe_results/`

**Open concerns:**
- The 27b model's thinking text suggests it may be prone to checking correctness and rewriting what it sees rather than purely transcribing. Prompt variants v2 and v3 add strong "TRANSCRIBER not proof checker" language to counteract this.
- Still need to test across all subproof crops (Q3, Q4, Q6) to assess generality.

**Action items:**
- ✅ `scripts/ocr_q4_crops_8b.py` — 8b showed same garble as 4b
- ✅ `scripts/probe_subproof_ocr.py` — 27b probe with prompt A/B testing
- Run `python scripts/probe_subproof_ocr.py --all-cases` with 27b loaded in LM Studio
- Evaluate whether prompt v2 or v3 reduces proof-checking behavior
- If 27b works well, design the routing logic to divert small subproofs to it
- **Effort:** Probe script done; testing requires ~30 min with 27b loaded

---

## Tier 2: Medium Priority (worth trying next)

### D. Whitespace Letterboxing Variations

Current letterboxing pads to 4:3 with white. Try variations:

- **Square (1:1)** — Might give even better VLM behavior
- **Larger min_height** (e.g., 400px) — More breathing room for the VLM
- **Faint Fitch bar overlay** — Draw a light gray vertical line on the left of the letterbox to maintain structural context
- **Effort:** ~30 minutes per variant

### E. Domain Grounding Prompt (Minimal Version)

We tried a very minimal prompt addition. A slightly stronger version:

```
You are transcribing a handwritten logic proof.
Allowed symbols: letters, ∨ ∧ → ↔ ¬, parentheses, line numbers.
Justification format: rule name + line references (e.g., "∧E 5", "∨I 1-3").
```

**Warning:** Past experience shows that making the prompt more complicated has led to significant performance decline. Approach with caution and A/B test.
- **Effort:** ~15 minutes, but high risk of regression

### F. Anchor-Based Extraction (Full Image Search)

Instead of cropping, give the model the full image and ask it to find a specific subproof:

```
"Find the subproof that begins with the premise 'K ∧ W'. 
Transcribe only that line and the lines within that subproof."
```

This avoids the crop-boundary problem entirely but requires the model to localize accurately.
- **Effort:** ~1 hour to prototype a test script

---

## Tier 3: Interesting but Complex

### G. Qwen Native Grounding (Bounding Box in Prompt)

Pass the bounding box coordinates directly in the prompt text:

```
"Transcribe the handwriting located strictly within the bounding box 
[x_min, y_min, x_max, y_max]. Return JSON."
```

Qwen-VL has native grounding capabilities, but these usually require direct HuggingFace or vLLM integration — may not work through LM Studio's OpenAI-compatible API.
- **Effort:** ~2 hours (may hit API limitations)

### H. QwenVL HTML Format

Ask the model to parse the proof using its native HTML document format. It might naturally translate Fitch bars into nested `<div>` elements, preserving structure. Then extract programmatically from HTML.

- **Effort:** ~2-3 hours to prototype + parser
- **Risk:** Format may not be available through LM Studio

### I. Hybrid CV + Numbered Boxes

Use OpenCV text detection (contours or CRAFT) to find line bounding boxes in the full image. Overlay numbered boxes on each line. Then prompt: "Transcribe the text inside Box 5 and Box 6."

```
[Box 1] N ∨ (K ∧ W)
[Box 2] N ↔ W
[Box 3] N          ← target
[Box 4] W          ← target
[Box 5] K ∧ W
[Box 6] W
[Box 7] W
```

- **Effort:** ~1 day (significant engineering)
- **Benefit:** Bridges gap between spatial location and text generation

### J. Contextual Cropping with Text Erasure

Crop from top-left to target lines, but black out / erase text of non-target lines. This preserves the Fitch bar skeleton while isolating the target text.

- **Effort:** ~2 hours
- **Risk:** Similar to context crops — VLM may read through the erasure

---

## General Notes (from Gemini)

### Why the VLM Struggles with Tight Crops

1. **Loss of structural context:** Without seeing the full Fitch bar extending upward, the VLM doesn't recognize the document as a logic proof. It sees floating lines and hallucinates.

2. **Domain ambiguity:** The model needs surrounding text to realize "this is a formal logic proof." Without context, `∧` looks like `8`, `∨` looks like `v`, etc.

3. **Resolution/aspect ratio distortion:** VLMs resize images to fixed patch sizes internally. A wide, short crop gets distorted, degrading handwriting quality. (This is what our letterboxing addresses.)

### Qwen-Specific Capabilities

- **Native grounding:** Bounding box / point-based representations for precise localization
- **Dynamic resolution:** Fine-grained control via `min_pixels` / `max_pixels`
- **HTML document parsing:** Can extract complex layouts as HTML with spatial structure