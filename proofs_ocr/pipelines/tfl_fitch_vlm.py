"""VLM prompt and model-call helpers for handwritten TFL/Fitch OCR."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


#: When True, use a stripped-down prompt for leaf crops.  The full prompt
#: stacks ~6 instruction blocks which may overwhelm a 4b model; this variant
#: keeps only the essentials.  Toggle to compare regression scores.
USE_SIMPLE_PROMPT = False


def get_model_27b() -> str:
    return os.getenv("LMSTUDIO_MODEL_27B", "qwen/qwen3.6-27b").strip()


def model_27b_available() -> bool:
    """Check if a 27b model is specified and differs from the default 4b."""
    return bool(get_model_27b().strip())


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int | None = None) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_27b_timeout_seconds() -> int:
    return max(1, env_int("TFL_FITCH_27B_TIMEOUT_SECONDS", 600) or 600)


def get_27b_max_tokens() -> int | None:
    return env_int("TFL_FITCH_27B_MAX_TOKENS")


def get_27b_extra_payload() -> Dict[str, object]:
    extra_payload: Dict[str, object] = {}
    if env_flag("TFL_FITCH_27B_ENABLE_THINKING_FALSE"):
        extra_payload["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_payload


def get_4b_fullwidth_max_lines() -> int:
    """Return max nested leaf line count for default-model full-width crops."""
    return max(0, env_int("TFL_FITCH_4B_FULLWIDTH_MAX_LINES", 99) or 0)


def use_separated_nested_leaf_ocr() -> bool:
    return env_flag("TFL_FITCH_SEPARATED_NESTED_LEAF_OCR", True)


def build_ocr_prompt(
    *,
    has_children: bool = True,
    highlighted_box: bool = False,
    expected_lines: int | None = None,
) -> str:
    """Build the VLM OCR prompt for a Fitch proof crop."""
    if USE_SIMPLE_PROMPT and not has_children:
        parts = [
            "Transcribe this Fitch proof image into a JSON array.",
            "Each element: {\"line\": <num>, \"wffstr\": \"<formula>\", \"jstr\": \"<justification>\"}.",
            "wffstr = formula on the left.  jstr = rule + line refs on the right.  Keep them separate.",
        ]
        if expected_lines is not None and expected_lines >= 1:
            parts.append(f"Output exactly {expected_lines} line(s).")
        parts.append("Return JSON only. No markdown.")
        return " ".join(parts)

    base = (
        "Transcribe the Fitch-style proof in this image into a JSON list of objects. "
        "Each object must have exactly these keys: 'line', 'wffstr', and 'jstr'. "
        "Use 'wffstr' only for the formula on that line. Use 'jstr' only for the justification. "
        "Do not place justification text inside 'wffstr'. "
        "If a line is a premise or assumption and no justification is printed, leave 'jstr' as an empty string. "
    )
    base += (
        "Handwriting Translation Guide (CRITICAL): "
        "Students have sloppy handwriting. Use these known distortions to correct your visual reading: "
        "- A sloppy K frequently looks like '1 <' or a vertical line next to a '<'. "
        "- A sloppy ampersand & (conjunction) frequently looks like an '8'. "
        "- A sloppy S frequently looks like a '5'. "
        "- A handwritten implication-introduction justification may look like an arrow followed by a tall capital I; transcribe that rule head as '→I' before the cited range. "
    )
    if has_children:
        base += (
            "Every red box token like <SUBPROOF_ID_X> marks one masked child subproof. "
            "You MUST output one proof-line object for each red token in its top-to-bottom position. "
            "Copy the token exactly as the entire 'wffstr' value for that line and leave 'jstr' empty. "
            "Treat a red token as a real proof row even when it appears alone in a large blank erased area. "
            "Do not skip a red token because no black handwriting is next to it. "
            "Also transcribe every visible black proof line above and below each red token; "
            "a red token does not replace nearby visible proof lines. "
        )
    if highlighted_box:
        base += (
            "OCR only the proof lines inside the bright green highlighted box. "
            "There are typically 1-3 lines inside the box. "
            "Ignore all other text outside the green box. "
        )
    if not has_children:
        base += "Transcribe only the lines actually visible in this image. "
        if expected_lines is not None and expected_lines >= 1:
            base += f"This image contains exactly {expected_lines} proof line(s). Output exactly {expected_lines} line objects. "
            if expected_lines <= 4:
                base += (
                    f"Each of the {expected_lines} line(s) has a formula on the LEFT and a justification on the RIGHT. "
                    "Formula goes in 'wffstr', justification goes in 'jstr'. Keep them separate. "
                )
    base += "Return JSON only, with no markdown fences or explanation."
    return base


def build_nested_leaf_separated_prompt(expected_lines: int | None = None) -> str:
    """Build prompt for nested leaf OCR as formula/justification lists."""
    prompt = (
        "You are transcribing a cropped handwritten Fitch subproof. "
        "Return exactly one JSON object with keys 'formulas' and 'justifications'. "
        "'formulas' is a top-to-bottom array of every formula visible in the subproof. "
        "The first/top formula is the child subproof assumption or premise. "
        "The first/top formula has no printed justification. Do not include a justification for it. "
        "'justifications' is a top-to-bottom array of the printed right-side justifications "
        "for only the derived lines after the first formula. "
        "Preserve citation numbers exactly as written. They are absolute proof line numbers, "
        "not local crop numbers. "
        "Do not infer corrections or validate the proof. "
        "Handwriting guide: sloppy ampersand/conjunction may look like '8'; sloppy S may look like '5'. "
    )
    if expected_lines is not None and expected_lines >= 2:
        prompt += (
            f"This image contains exactly {expected_lines} proof line(s). "
            f"Return exactly {expected_lines} formulas and exactly {expected_lines - 1} justifications. "
        )
    prompt += (
        "Example output shape: "
        "{\"formulas\": [\"W∧A\", \"W\", \"S\", \"S∧A\"], "
        "\"justifications\": [\"∧E 7\", \"→E 2,8\", \"∧I 8,9\"]}. "
        "Return JSON only, with no markdown fences or explanation."
    )
    return prompt


def stitch_separated_nested_leaf_ocr(payload, expected_lines: int | None = None) -> List[Dict] | None:
    """Convert separated formula/justification OCR into line dictionaries."""
    if not isinstance(payload, dict):
        return None
    formulas = payload.get("formulas")
    justifications = payload.get("justifications")
    if not isinstance(formulas, list) or not isinstance(justifications, list):
        return None
    if expected_lines is not None and len(formulas) != expected_lines:
        return None
    if len(formulas) < 2 or len(justifications) != len(formulas) - 1:
        return None

    stitched: List[Dict] = []
    for index, formula in enumerate(formulas):
        jstr = "" if index == 0 else str(justifications[index - 1] or "")
        stitched.append({"line": index + 1, "wffstr": str(formula or ""), "jstr": jstr})
    return stitched


def build_27b_fullwidth_prompt(expected_lines: int | None = None) -> str:
    """Build a compact prompt for 27b full-width crop OCR."""
    prompt = (
        "Return only a JSON array. Do not write analysis, notes, markdown, or explanations.\n"
        "OCR the visible handwritten Fitch subproof chunk.\n"
        "Use the visible left-gutter line numbers as absolute proof line numbers; do not "
        "renumber the crop from 1.\n"
        "For each visible proof row output one object with keys \"line\", \"wffstr\", and "
        "\"jstr\". Put only the formula in \"wffstr\" and only the printed rule/citations "
        "in \"jstr\". If no justification is printed, use an empty string.\n"
        "Handwriting guide: K may look like '1 <' or a vertical stroke next to '<'; "
        "conjunction &/∧ may look like '8'; S may look like '5'.\n"
    )
    if expected_lines is not None and expected_lines >= 1:
        prompt += (
            f"The image contains exactly {expected_lines} proof line(s). "
            f"Output exactly {expected_lines} object(s).\n"
        )
    prompt += "Return the JSON array now."
    return prompt


def _extract_json_content(content: str) -> str:
    if "```json" in content:
        return content.split("```json")[1].split("```")[0]
    if "```" in content:
        return content.split("```")[1].split("```")[0]
    return content


def call_vlm(
    image_path: str,
    client,
    *,
    has_children: bool = True,
    highlighted_box: bool = False,
    expected_lines: int | None = None,
    model_override: str | None = None,
    prompt_override: str | None = None,
) -> Tuple[object, str]:
    """Call the configured VLM and parse its JSON transcription."""
    if prompt_override:
        prompt = prompt_override
    else:
        prompt = build_ocr_prompt(
            has_children=has_children,
            highlighted_box=highlighted_box,
            expected_lines=expected_lines,
        )
    is_27b_route = bool(model_override)
    if is_27b_route and env_flag("TFL_FITCH_27B_NO_THINK"):
        prompt = f"/no_think\n{prompt}"

    model_tag = f" (model={model_override})" if model_override else ""
    print(f"   [VLM] Sending {os.path.basename(image_path)}{model_tag}...")

    if client is None:
        return [{"line": "?", "wffstr": "VLM_NOT_CONFIGURED", "jstr": "Error"}], ""

    try:
        image_path_obj = Path(image_path)
        if is_27b_route and env_flag("TFL_FITCH_27B_SINGLE_USER_MESSAGE"):
            messages = client.build_single_user_image_message(image_path_obj, prompt)
        elif hasattr(client, "build_messages_for_prompt"):
            messages = client.build_messages_for_prompt(
                image_path_obj,
                "Provide the transcription as instructed.",
                system_prompt=prompt,
            )
        else:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide the transcription as instructed."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ]

        extra_kwargs = {}
        if is_27b_route:
            extra_kwargs["timeout"] = get_27b_timeout_seconds()
            max_tokens = get_27b_max_tokens()
            if max_tokens is not None:
                extra_kwargs["max_tokens"] = max_tokens
            extra_payload = get_27b_extra_payload()
            if extra_payload:
                extra_kwargs["extra_payload"] = extra_payload
        if hasattr(client, "last_vlm_error"):
            client.last_vlm_error = ""
        content = client.chat(messages, model=model_override, **extra_kwargs)

        thinking = getattr(client, "last_thinking", "") or ""
        metadata = getattr(client, "last_response_metadata", {}) or {}
        if metadata:
            print(
                "   [VLM] Metadata: "
                f"elapsed={metadata.get('elapsed_s')}s "
                f"thinking={metadata.get('thinking_chars')} "
                f"content={metadata.get('content_chars')} "
                f"finish={metadata.get('finish_reason')}"
            )

        return json.loads(_extract_json_content(content)), thinking

    except Exception as e:
        thinking = getattr(client, "last_thinking", "") or ""
        metadata = getattr(client, "last_response_metadata", {}) or {}
        if hasattr(client, "__dict__"):
            client.last_vlm_error = str(e)
        if metadata:
            print(
                "   [VLM] Metadata: "
                f"elapsed={metadata.get('elapsed_s')}s "
                f"thinking={metadata.get('thinking_chars')} "
                f"content={metadata.get('content_chars')} "
                f"finish={metadata.get('finish_reason')}"
            )
        print(f"   [VLM Error] {e}")
        return [], thinking
