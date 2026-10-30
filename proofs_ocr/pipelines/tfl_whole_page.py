"""Experimental literal transcription of an entire TFL proof region.

No bar detection, subproof chopping, gold answers, or logical repairs are used.
Callers own privacy authorization and private storage of images/results.
This module is opt-in and does not change the production Fitch OCR strategy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from app.vlm_provider import create_vlm_client


PROMPT = '''Transcribe this handwritten Fitch proof literally. Do not solve it or repair mistakes.
Return one JSON object {"proof": [...]} containing the complete proof in visual order.
Each proof line is {"line": integer, "wffstr": "formula", "jstr": "written rule and citations"}.
Use nested arrays ONLY for actual assumption subproofs. The far-left outer proof spine is a frame, not an assumption subproof.
Premises before the first horizontal separator have jstr "Pr". An assumption starting a subproof has jstr "Hyp" if no rule is written.
Copy every handwritten line number and citation exactly as written, including invalid self-citations, incorrect rules, and other mistakes.
Do not replace a written citation with a logically sensible citation. Do not invent missing proof lines.
Normalize logical glyphs to ~, &, v, ->, <-> if needed. Preserve formula letters and their case. No explanatory prose.
Output contract: use uppercase Latin letters for propositional atoms and Unicode logical operators ∧ ∨ ¬ → ↔.
Use canonical compact rule heads (∧E, ∨I, →E, ↔E, →I), followed by the ORIGINAL citations.
Subproofs are bare nested arrays, NEVER objects wrapping a list in a line field. Do not flatten actual subproofs.
Example of array SHAPE only, not the image content: {"proof":[{"line":1,"wffstr":"A","jstr":"Pr"},[{"line":2,"wffstr":"B","jstr":"Hyp"}],{"line":3,"wffstr":"B→B","jstr":"→I 2-2"}]}.
The instruction to preserve original letter case is superseded by uppercase propositional atoms for this TFL output format.
Copy all visibly written citation numbers literally, including self-citations. Do not repair an invalid proof.'''


def response_format(max_depth: int = 3) -> dict:
    line = {"type": "object", "properties": {"line": {"type": "integer"},
        "wffstr": {"type": "string"}, "jstr": {"type": "string"}},
        "required": ["line", "wffstr", "jstr"], "additionalProperties": False}
    node = line
    for _ in range(max_depth):
        node = {"anyOf": [line, {"type": "array", "items": node}]}
    return {"type": "json_schema", "json_schema": {"name": "fitch_proof", "strict": True,
        "schema": {"type": "object", "properties": {"proof": {"type": "array", "items": node}},
                   "required": ["proof"], "additionalProperties": False}}}


def validate_structure(proof, *, max_depth: int = 3) -> list:
    def check(items, depth):
        if not isinstance(items, list) or depth > max_depth:
            raise ValueError("Invalid proof nesting")
        for item in items:
            if isinstance(item, list):
                check(item, depth + 1)
            elif (not isinstance(item, dict) or set(item) != {"line", "wffstr", "jstr"}
                  or isinstance(item["line"], bool) or not isinstance(item["line"], int)
                  or item["line"] < 1 or not isinstance(item["wffstr"], str)
                  or not isinstance(item["jstr"], str)):
                raise ValueError("Invalid proof line")
    check(proof, 0)
    return proof


def flatten_rows(proof: list, depth: int = 0) -> list[dict]:
    rows = []
    for item in proof:
        if isinstance(item, list):
            children = flatten_rows(item, depth + 1)
            if children:
                children[0]["start_subproof"] = True
            rows.extend(children)
        else:
            rows.append({**item, "depth": depth, "start_subproof": False})
    return rows


def rows_to_structure(rows: list[dict], *, max_depth: int = 3) -> list:
    result = []
    stack = [result]
    for row in rows:
        depth = row.get("depth")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= max_depth:
            raise ValueError("Invalid nesting depth")
        if depth > len(stack):
            raise ValueError("Nesting cannot skip a level")
        while len(stack) > depth + 1:
            stack.pop()
        if row.get("start_subproof") and depth > 0 and len(stack) == depth + 1:
            stack.pop()
        if len(stack) == depth:
            nested = []
            stack[-1].append(nested)
            stack.append(nested)
        stack[-1].append({k: row.get(k) for k in ("line", "wffstr", "jstr")})
    return validate_structure(result, max_depth=max_depth)


def transcribe_whole_proof(image_path: Path, *, client=None, model: str = "gemma", max_tokens: int = 2048) -> dict:
    client = client or create_vlm_client(model)
    format_spec = response_format()
    started = time.monotonic()
    raw = client.chat(client.build_messages_for_prompt(Path(image_path), PROMPT),
                      max_tokens=max_tokens, extra_payload={"response_format": format_spec})
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:-1])
    data = json.loads(clean)
    if not isinstance(data, dict) or set(data) != {"proof"}:
        raise ValueError("Invalid proof response")
    proof = validate_structure(data["proof"])
    return {"proof": proof, "raw_response": raw, "metadata": {
        **getattr(client, "last_response_metadata", {}), "model": client.model,
        "seconds": round(time.monotonic() - started, 3),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(json.dumps(format_spec, sort_keys=True).encode()).hexdigest()}}
