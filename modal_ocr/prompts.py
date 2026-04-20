from __future__ import annotations

from typing import Any, Dict

from modal_ocr.shared import canonical_modal_config


def build_prompt(config: Dict[str, Any]) -> str:
    clean = canonical_modal_config(config)
    conclusion = clean.get("conclusion")
    lines = [
        "You are transcribing a modal Fitch-style proof from an image.",
        "Return only JSON. Do not include markdown fences or explanation.",
        "",
        "Return a JSON array. Each item must be one of:",
        '1. A proof line object: {"formula_text": "...", "justification_text": "...", "scope_role": "...", "line_label": "..."}',
        '2. A masked child token object: {"formula_text": "<SUBPROOF_ID_X>", "scope_role": "subproof_token"}',
        "",
        "Allowed scope_role values:",
        '- "premise" for top-level premises',
        '- "ordinary_assumption" for the first line of a normal subproof',
        '- "strict_open" for a strict modal subproof opener; the formula_text should be just □',
        '- "derived_line" for any ordinary derived line',
        '- "subproof_token" for a red masked token placeholder',
        "",
        "Transcription rules:",
        "- Keep formulas concise and symbolic.",
        "- Prefer Unicode operators: ¬, ∧, ∨, →, ↔, □, ⋄.",
        "- For justifications, prefer forms like R 1, ∧I 4 5, ∧E 1, →I 2-3, □E 1, □I 3-6, R4 1, RT 1, MC 1, Def⋄ 1.",
        "- Copy the printed line labels exactly as they appear in the image. Do not renumber lines within a crop.",
        "- If a line starts a strict modal subproof, use scope_role strict_open and formula_text □.",
        "- Preserve the exact number of negation marks. Do not collapse ¬¬A into ¬A.",
        "- In handwritten work, double negation may appear as ¬¬A, ~~A, or mixed marks; if you see two negation strokes, transcribe two negations.",
        "- If a line cites DN, be especially careful to preserve doubled negation in the formula.",
        "- A contradiction line may be written as ⊥, _|_, _ | _, or a short vertical stroke over a horizontal bar; transcribe any of these as ⊥.",
        "- If a line's entire formula is just contradiction/absurdity, return formula_text as ⊥.",
        "- If you see a red token like <SUBPROOF_ID_2>, return it exactly as formula_text with scope_role subproof_token.",
        "- After a red subproof token, keep listing any further visible lines in the current crop. Do not stop early just because a child subproof was masked.",
        "- If a current crop contains both a masked child token and a later closing-result line such as □B or A → B, include both in order.",
        "- Do not invent nested JSON structure yourself; just list the visible lines in order for the current crop.",
        "",
        f"Modal system: {clean['system']}",
        f"Include derived rules: {'yes' if clean['include_derived_rules'] else 'no'}",
    ]
    if conclusion:
        lines.append(f"Target conclusion: {conclusion}")
    return "\n".join(lines)


def build_formula_ocr_prompt(config: Dict[str, Any]) -> str:
    canonical_modal_config(config)
    lines = [
        "Transcribe the formula side of this handwritten modal Fitch proof line.",
        "Return only JSON. Do not include markdown fences or explanation.",
        'Return one JSON object with exactly these keys: "line_label", "formula_text", and "confidence".',
        'Use "formula_text" only for the formula or symbol on this crop.',
        'Do not place justification text inside "formula_text".',
        'Copy the printed line label if visible; otherwise return an empty string for "line_label".',
        "Prefer Unicode operators: ¬, ∧, ∨, →, ↔, □, ⋄.",
        "Preserve repeated unary operators exactly.",
        'A contradiction line may be written as ⊥, _|_, _ | _, or a short vertical stroke over a horizontal bar; transcribe any of these as ⊥.',
        'If the visible formula is only contradiction/absurdity, return "⊥" as formula_text.',
        'If the visible content is only a modal box opener, return "□" as formula_text.',
        "Use confidence values high, medium, low, or uncertain.",
    ]
    return "\n".join(lines)


def build_justification_ocr_prompt(config: Dict[str, Any]) -> str:
    canonical_modal_config(config)
    lines = [
        "Transcribe the justification side of this handwritten modal Fitch proof line.",
        "Return only JSON. Do not include markdown fences or explanation.",
        'Return one JSON object with exactly these keys: "justification_text" and "confidence".',
        (
            'Use "justification_text" only for the justification on this crop, such as "", '
            '"PR", "Prem", "Assum", "Assump", "∧E 1", "∨I 4", "→E 4,5", "→I 2-7", '
            '"□E 1", "□I 3-6", "R4 1", "RT 1", "MC 3", "DN 6", "Def⋄ 1", "¬I 1-9", or "¬E 4,8".'
        ),
        "These are examples of rule shapes only. Copy the visible rule head and cited line numbers from the image; do not copy numbers from the examples.",
        'If the crop prints "~I" or "~E", keep the printed tilde form instead of rewriting it to "¬I" or "¬E".',
        "The first line of a proof or subproof often has no printed justification; if this crop shows no justification, return an empty string.",
        'If no justification is printed, return an empty string for "justification_text".',
        'Do not include formula text in "justification_text".',
        "Use confidence values high, medium, low, or uncertain.",
    ]
    return "\n".join(lines)


def build_line_ocr_prompt(config: Dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "=== FORMULA ZONE OCR PROMPT ===",
            build_formula_ocr_prompt(config),
            "=== JUSTIFICATION ZONE OCR PROMPT ===",
            build_justification_ocr_prompt(config),
        ]
    )
