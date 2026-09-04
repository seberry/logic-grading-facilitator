"""Proof-structure normalization helpers for handwritten TFL/Fitch OCR."""

from __future__ import annotations

import re
import os
from typing import Any, Dict, List, Union

from proofs_ocr.tricks.justification_snaps import (
    TFL_KNOWN_RULE_HEADS,
    TFL_VALID_RULE_HEADS,
    clean_justification_for_tfl,
    fuzzy_snap_justification,
    snap_justification_rule_head,
)


def apply_default_justifications(proof_structure: list, numprems: int) -> None:
    """
    Fill in any remaining blank justifications after renumbering.

    - Lines whose numbers fall within the inferred premise block (<= numprems)
      are labeled as Premises (Pr).
    - All other blank justifications are treated as assumptions (Hyp) so the
      verifier can parse them instead of emitting "Justification left blank."
    """

    for item in proof_structure:
        if isinstance(item, list):
            apply_default_justifications(item, numprems)
        elif isinstance(item, dict):
            raw_j = str(item.get("jstr", "")).strip()
            if raw_j:
                continue

            line_no = item.get("line")
            if line_no is not None and line_no <= numprems:
                item["jstr"] = "Pr"
            else:
                item["jstr"] = "Hyp"


def remove_empty_lines(proof_structure: list) -> list:
    cleaned = []
    for item in proof_structure:
        if isinstance(item, list):
            nested = remove_empty_lines(item)
            if nested:
                cleaned.append(nested)
            continue
        if isinstance(item, dict):
            wff = str(item.get("wffstr", "")).strip()
            jstr = str(item.get("jstr", "")).strip()
            if not wff and not jstr:
                continue
            cleaned.append(item)
    proof_structure[:] = cleaned
    return proof_structure


def infer_missing_rule_heads(proof_structure: list) -> None:
    for item in proof_structure:
        if isinstance(item, list):
            infer_missing_rule_heads(item)
            continue
        if not isinstance(item, dict):
            continue
        wff = str(item.get("wffstr", "")).strip()
        jstr = str(item.get("jstr", "")).strip()
        if wff.startswith("¬") and re.fullmatch(r"I\s+\d+(?:-\d+)?", jstr):
            item["jstr"] = f"¬{jstr}"


def normalize_proof_structure(structure):
    """
    Walks the proof structure to:
    1. Convert ASCII logic symbols to Unicode (e.g., '~' -> '¬').
    2. Uppercase atomic formulas (TFL requires 'P', not 'p').
    3. Map loose justification names to verifier codes (e.g., 'Premise' -> 'Pr').
    4. Auto-label the first line of any subproof as 'Hyp' if blank.
    """

    # Map ASCII/Loose symbols to Verifier Unicode
    # Note: We do this replacement BEFORE uppercasing to protect 'v' (or) from becoming 'V'.
    symbol_map = {
        "~": "¬",
        "-": "¬",      # Sometimes OCR sees dash as negation
        "&": "∧",
        ".": "∧",      # VLM often uses dot for AND
        "v": "∨",      # Lowercase v is OR
        "V": "∨",
        "->": "→",
        "=>": "→",
        "<->": "↔",
        "=": "↔",      # Common OCR error for biconditional
        "_|_": "⊥",
        "F": "⊥",      # Falsum often written as F
    }

    contradiction_rule_heads = {"¬E", "⊥I"}

    def clean_wff(wff):
        if not wff:
            return ""
        if "<SUBPROOF_ID_" in str(wff):
            return str(wff)

        # Phase 0: Blanket OCR-artifact repair for TFL formulas.
        # TFL formulas never contain digits, epsilons, or element-of symbols.
        # Any occurrence is a VLM misread of a handwritten symbol.
        wff = wff.replace("∈", "E").replace("ε", "E")   # epsilon -> E
        wff = wff.replace("8", "∧")                       # handwritten ∧ misread as 8
        wff = wff.replace("5", "S")                        # handwritten S misread as 5

        # 1. Replace multi-char operators first to avoid breaking them
        wff = wff.replace("->", "→").replace("<->", "↔").replace("<→", "↔")

        # 2. Character-by-character replacement for single symbols
        clean_chars = []
        for char in wff:
            if char in symbol_map:
                clean_chars.append(symbol_map[char])
            elif char.isalpha():
                # TFL atoms must be Uppercase.
                # (Skip 'v' if it was mapped to OR, but here 'v' is already handled by symbol_map)
                clean_chars.append(char.upper())
            else:
                clean_chars.append(char)

        return "".join(clean_chars)

    def normalize_rule_candidate(token):
        compact = str(token or "").strip()
        compact = compact.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
        compact = compact.replace("&", "∧").replace(".", "∧")
        compact = compact.replace("v", "∨").replace("V", "∨")
        compact = compact.replace("~", "¬").replace("-", "¬")
        return compact

    def looks_like_rule_token(token):
        if not token:
            return False
        compact = normalize_rule_candidate(token).replace(" ", "")
        return compact in TFL_KNOWN_RULE_HEADS

    def likely_rule_with_citations(token, next_token):
        if not looks_like_rule_token(token):
            return False
        next_text = str(next_token or "").strip()
        return bool(re.match(r"^[\da-zA-Z][\d,\-.]*$", next_text))

    def rule_token_at(tokens, index):
        token = tokens[index]
        if index + 1 < len(tokens):
            combined = f"{token}{tokens[index + 1]}"
            if likely_rule_with_citations(combined, tokens[index + 2] if index + 2 < len(tokens) else ""):
                return combined, 2
        if likely_rule_with_citations(token, tokens[index + 1] if index + 1 < len(tokens) else ""):
            return token, 1
        return None, 0

    def split_tight_embedded_justification(raw_wff, *, line_needs_justification: bool):
        raw_flag = os.getenv("TFL_FITCH_SPLIT_TIGHT_EMBEDDED_JUSTIFICATIONS")
        enabled = True if raw_flag is None else raw_flag.strip().lower() in {"1", "true", "yes", "on"}
        if not enabled or not line_needs_justification:
            return None

        candidate = str(raw_wff or "").strip()
        if not candidate:
            return None

        candidate = candidate.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
        candidate = candidate.replace("&", "∧")
        candidate = candidate.replace("∈", "E").replace("ε", "E")
        heads = {
            "∨E", "∨I", "∧E", "∧I", "¬E", "¬I", "→E", "→I", "↔E", "↔I",
            "vE", "vI", "&E", "&I", "~E", "~I", "-E", "-I",
            "AE", "AI", "8E", "8I", "BE", "BI", "εE", "εI", "εS",
        }
        citation_tail = r"([0-9][0-9,\s.\-]*)"
        for head in sorted(heads, key=len, reverse=True):
            match = re.fullmatch(rf"(.+?)\s*({re.escape(head)})\s*{citation_tail}", candidate)
            if not match:
                continue
            formula_part = match.group(1).strip()
            if not formula_part:
                continue
            justification_part = f"{match.group(2)} {match.group(3).strip()}".strip()
            return formula_part, justification_part
        return None

    def extract_embedded_justification(raw_wff, raw_j, *, line_needs_justification: bool = False):
        wff_text = str(raw_wff or "").strip()
        j_text = str(raw_j or "").strip()

        # A bare line number or similar OCR noise in the justification column
        # should not block premise/assumption inference.
        if re.fullmatch(r"[\d,.\-]+", j_text):
            j_text = ""

        if not wff_text:
            return wff_text, j_text

        tokens = wff_text.split()
        if not tokens:
            return wff_text, j_text

        if j_text:
            tight_split = split_tight_embedded_justification(
                wff_text,
                line_needs_justification=line_needs_justification,
            )
            if tight_split is not None:
                return tight_split[0], j_text

            for i in range(1, len(tokens)):
                candidate_head, _consumed = rule_token_at(tokens, i)
                if candidate_head is None:
                    continue
                formula_part = " ".join(tokens[:i]).strip()
                if formula_part:
                    return formula_part, j_text
            return wff_text, j_text

        # Sometimes the model puts the entire justification in the formula field.
        head, consumed = rule_token_at(tokens, 0)
        if head is not None:
            justification_text = " ".join([head, *tokens[consumed:]]).strip()
            normalized_head = clean_justification_for_tfl(head)
            normalized_full = clean_justification_for_tfl(justification_text)
            if normalized_head in contradiction_rule_heads:
                return "⊥", normalized_full
            return "", normalized_full

        # More commonly the model appends the justification onto the formula.
        for i in range(1, len(tokens)):
            candidate_head, consumed = rule_token_at(tokens, i)
            if candidate_head is None:
                continue
            formula_part = " ".join(tokens[:i]).strip()
            justification_part = " ".join([candidate_head, *tokens[i + consumed:]]).strip()
            if not formula_part or not justification_part:
                continue
            return formula_part, justification_part

        tight_split = split_tight_embedded_justification(
            wff_text,
            line_needs_justification=line_needs_justification,
        )
        if tight_split is not None:
            return tight_split

        return wff_text, j_text

    def recursive_normalize(sub_structure, is_subproof_start=False):
        seen_justified_or_subproof = False
        for index, item in enumerate(sub_structure):
            if isinstance(item, list):
                recursive_normalize(item, is_subproof_start=True)
                seen_justified_or_subproof = True
            elif isinstance(item, dict):
                raw_wff = item.get("wffstr", "")
                raw_j = str(item.get("jstr", "")).strip()
                raw_j_nonempty = bool(raw_j and not re.fullmatch(r"[\d,.\-]+", raw_j))
                line_needs_justification = (
                    (is_subproof_start and index > 0)
                    or (not is_subproof_start and seen_justified_or_subproof)
                    or raw_j_nonempty
                )
                extracted_wff, extracted_j = extract_embedded_justification(
                    raw_wff,
                    raw_j,
                    line_needs_justification=line_needs_justification,
                )

                item["wffstr"] = clean_wff(extracted_wff)

                if is_subproof_start and index == 0 and not extracted_j:
                    item["jstr"] = "Hyp"
                else:
                    normalized_j = snap_justification_rule_head(
                        clean_justification_for_tfl(extracted_j),
                        valid_rules=TFL_VALID_RULE_HEADS,
                    )
                    normalized_j = fuzzy_snap_justification(
                        normalized_j,
                        valid_rules=TFL_VALID_RULE_HEADS,
                    )
                    item["jstr"] = normalized_j
                if item["jstr"]:
                    seen_justified_or_subproof = True

    recursive_normalize(structure)
    return structure


def normalize_logical_fitch_depths(structure: list) -> list:
    """Shift OCR structures up when the printed outer Fitch frame was counted."""

    rows: list[tuple[int, Dict[str, Any]]] = []

    def walk(items, depth: int) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            elif isinstance(item, dict):
                rows.append((depth, dict(item)))

    walk(structure, 0)
    if not rows:
        return structure

    min_depth = min(depth for depth, _line in rows)
    if min_depth <= 0:
        return structure

    root: list[Any] = []
    stack: list[list[Any]] = [root]
    for depth, line in rows:
        logical_depth = max(0, depth - min_depth)
        while logical_depth < len(stack) - 1:
            stack.pop()
        while logical_depth > len(stack) - 1:
            child: list[Any] = []
            stack[-1].append(child)
            stack.append(child)
        stack[-1].append(line)

    return root


def label_and_count_premises(proof_structure: List[Union[Dict, List]]) -> int:
    """
    Walks the top-level of the stitched proof structure.
    If a line has an empty justification ('jstr') and appears before any
    subproofs or justified lines, it is labeled as 'Pr' (for premise).

    Returns: The number of premises identified.
    """
    premise_count = 0

    for item in proof_structure:
        if isinstance(item, list):
            break

        if isinstance(item, dict):
            jstr = item.get("jstr", "").strip()

            if not jstr:
                item["jstr"] = "Pr"
                premise_count += 1
            elif jstr.lower() == "pr":
                premise_count += 1
            else:
                break

    return premise_count
