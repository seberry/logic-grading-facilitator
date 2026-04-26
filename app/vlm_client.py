from __future__ import annotations

import base64
import io
import requests
import os
import time
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image


SINGLE_QUESTION_PROMPT = """You are an AI assistant helping to grade scanned student exams.
Your goal is to locate the student's handwriting for ONE specific question to show to a human grader.

Target Question: "{problem_id}"
Description: "{problem_description}"

Instructions:
1. Scan the page to see if the specific "Target Question" appears.
2. If found, identify the STUDENT'S ANSWER.
   - This may be a very small mark (T/F, a circled letter), a short phrase,
     or a larger paragraph or calculation.

3. Create a bounding box for human verification:
   - The box MUST include the printed question number or label (e.g., "6.", "(6)", "Q6")
     if it is visible.
   - The box MUST include the student's handwriting for this question.
   - The box should exclude neighboring questions when possible,
     but it is OK if the box overlaps with nearby printed or written content.
   
4. CRITICAL - Bounding box coordinates:
   - Use NORMALIZED coordinates in the range [0.0, 1.0] where:
     * 0.0 represents the left/top edge of the image
     * 1.0 represents the right/bottom edge of the image
   - Format: [x_min, y_min, x_max, y_max]
   - Example: [0.1, 0.3, 0.6, 0.5] means:
     * Box starts 10% from the left edge (x_min=0.1)
     * Box starts 30% from the top edge (y_min=0.3)
     * Box ends 60% from the left edge (x_max=0.6)
     * Box ends 50% from the top edge (y_max=0.5)
   - DO NOT use pixel coordinates like [100, 200, 300, 400]
   - All four coordinate values MUST be between 0.0 and 1.0

5. Analyze for anomalies:
   - If the answer appears to run onto another page,
     is crossed out, ambiguous, or unusually placed,
     briefly describe this in the "comment" field.

If the question is NOT on this page or contains no student writing, return:

    {{"present": false, "comment": null}}

Otherwise, return ONLY a JSON object in this format:

{{
  "target_problem_id": "{problem_id}",
  "present": true,
  "bbox": [x_min, y_min, x_max, y_max],
  "comment": "Optional brief note about ambiguity or multi-page answers.",
  "confidence": 0.95
}}

Remember: bbox coordinates must be normalized floats between 0.0 and 1.0, NOT pixel values.
""".strip()


NAME_SYSTEM_PROMPT = """
You are an AI assistant helping to grade scanned student exams.

Your ONLY job in this task is:
  - Look at the top of the scanned page.
  - Decide whether a STUDENT NAME is visible.
  - If a name is visible, match it to a student in the provided roster.
  - If you cannot confidently match the name to a roster entry, say there is NO MATCH.

You MUST behave like a classifier, not a storyteller:
  - Do NOT invent new students that are not in the roster.
  - If the handwriting roughly matches a roster name (even with nicknames,
    gender differences, or small spelling errors), choose the closest roster entry.
  - If there is no plausible match at all, return NO MATCH.

You will always respond with a single JSON object with these fields:

  {
    "name_visible": true or false,
    "raw_name": string or null,          // exact transcription of the handwritten name, if any
    "student_id": string or null,        // one of the roster student_ids, or null if no match
    "confidence": float,                 // between 0.0 and 1.0, your confidence in student_id
    "reasoning": string                  // brief explanation of how you matched or why no match
  }

Do not include any additional keys. Do not wrap the JSON in backticks.
""".strip()


class LMStudioQwenClient:
    """
    Minimal client for a Qwen3-VL-4B model served via LM Studio.

    Assumes an OpenAI-compatible /v1/chat/completions endpoint.

    LM Studio's log error `'url' field must be a base64 encoded image`
    means we must send the image as a base64 data URI in the "url" field.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or default_lmstudio_base_url()).rstrip("/")
        self.model = model or default_lmstudio_model()
        self.timeout = default_lmstudio_timeout() if timeout is None else timeout
        self.max_image_dim = 1500
        self.max_data_uri_len = 2_500_000
        self.last_thinking: str = ""
        self.last_response_metadata: Dict[str, Any] = {}

    def _encode_image_to_data_uri(self, image_path: Path, max_dim: int | None = None) -> str:
        """
        Read image and return a data URI string suitable for LM Studio's
        'image_url': {'url': ...} field, e.g.
        'data:image/jpeg;base64,<...>'.
        """
        suffix = image_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            source_mime = "image/jpeg"
        elif suffix in [".png"]:
            source_mime = "image/png"
        else:
            source_mime = "application/octet-stream"

        raw_bytes = image_path.read_bytes()
        raw_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        raw_uri = f"data:{source_mime};base64,{raw_b64}"
        # Keep original bytes on the first attempt when payload is already small enough.
        # On retries (max_dim provided), always re-encode to normalize image encoding.
        if max_dim is None and len(raw_uri) <= self.max_data_uri_len:
            return raw_uri

        # Fallback for large scans: downscale + convert to JPEG for LM Studio stability.
        target_dim = max_dim or self.max_image_dim
        with Image.open(image_path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            if max(im.size) > target_dim:
                im.thumbnail((target_dim, target_dim))

            quality_steps = [85, 75, 65, 55]
            best_uri = None
            for quality in quality_steps:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=quality, optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                uri = f"data:image/jpeg;base64,{b64}"
                best_uri = uri
                if len(uri) <= self.max_data_uri_len:
                    break

        if best_uri:
            print(
                f"[VLM] Shrunk image payload for {image_path.name} (max_dim={target_dim}): "
                f"{len(raw_uri)} -> {len(best_uri)} chars"
            )
        return best_uri or raw_uri

    def build_messages(self, image_path: Path, roster_text: str) -> List[Dict[str, Any]]:
        """
        Build the messages payload with both text and image.

        For this LM Studio setup, the vision model expects:
          "type": "image_url",
          "image_url": { "url": "<base64 data uri>" }

        (The error `'url' field must be a base64 encoded image` confirms this.)
        """
        data_uri = self._encode_image_to_data_uri(image_path)

        return [
            {
                "role": "system",
                "content": NAME_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",  # LM Studio supports this OpenAI-style
                        "text": (
                            "Roster (student_id, Canvas name):\n"
                            f"{roster_text}\n\n"
                            "Please analyze the attached quiz page image and decide:\n"
                            "- Is a student name visible?\n"
                            "- If so, which student from the roster it matches.\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        },
                    },
                ],
            },
        ]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str | None = None,
        timeout: int | None = None,
        max_tokens: int | None = None,
        extra_payload: Dict[str, Any] | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if max_tokens is not None:
            # Send both parameter names — LM Studio may respect either one
            payload["max_tokens"] = max_tokens
            payload["max_completion_tokens"] = max_tokens
        if extra_payload:
            payload.update(extra_payload)

        started_at = time.monotonic()
        resp = requests.post(url, json=payload, timeout=timeout or self.timeout)
        elapsed_s = time.monotonic() - started_at
        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception as e:
            print("[ERROR] Non-JSON response from LM Studio:")
            print(resp.text)
            raise

        if "choices" not in data:
            print("[ERROR] Unexpected response from LM Studio:")
            print(data)
            raise RuntimeError("LM Studio response missing 'choices' key")

        message = data["choices"][0].get("message") or {}
        content = str(message.get("content") or "")

        # Extract thinking/reasoning text (Qwen3.x returns it in reasoning_content)
        thinking = str(message.get("reasoning_content") or "")
        self.last_thinking = thinking
        choice = data["choices"][0] if data.get("choices") else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        self.last_response_metadata = {
            "model": payload.get("model"),
            "elapsed_s": round(elapsed_s, 3),
            "thinking_chars": len(thinking),
            "content_chars": len(content),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "sent_max_tokens": payload.get("max_tokens"),
            "sent_max_completion_tokens": payload.get("max_completion_tokens"),
            "sent_extra_payload_keys": sorted(extra_payload.keys()) if extra_payload else [],
        }
        if thinking:
            print(f"[VLM] Thinking: {len(thinking)} chars")

        if content:
            return content

        raise RuntimeError(
            "LM Studio returned an empty completion content "
            f"(model={self.model}, usage={usage}). "
            "This often means the request was cancelled or the server/runtime produced an empty response."
        )



    def call_for_page(self, image_path: Path, roster_text: str) -> str:
        """Convenience: build messages and return the raw assistant string."""
        messages = self.build_messages(image_path, roster_text)
        return self.chat(messages)

    def build_messages_for_prompt(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str | None = None,
        max_image_dim: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Build messages for an arbitrary text prompt + a single image.

        This is a generic version of build_messages that doesn't assume a roster.
        """
        data_uri = self._encode_image_to_data_uri(image_path, max_dim=max_image_dim)

        if system_prompt is None:
            system_prompt = "You are a careful vision model. Follow the user's instructions exactly."

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
            },
        ]

    def build_single_user_image_message(
        self,
        image_path: Path,
        prompt: str,
        max_image_dim: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Build the GUI-like single user message shape used by LM Studio chat."""
        data_uri = self._encode_image_to_data_uri(image_path, max_dim=max_image_dim)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
            },
        ]

    def chat_with_image_prompt(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Convenience: given a text prompt and an image, return the raw model reply.
        """
        dims_to_try = [None, 1400, 1200, 1000]
        last_err: Exception | None = None

        for dim in dims_to_try:
            messages = self.build_messages_for_prompt(
                image_path=image_path,
                prompt=prompt,
                system_prompt=system_prompt,
                max_image_dim=dim,
            )
            try:
                return self.chat(messages)
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                body = (response.text or "").lower() if response is not None else ""
                if response is None or response.status_code != 400 or "failed to process image" not in body:
                    raise
                print(
                    f"[VLM] Retrying image at smaller size for {image_path.name} "
                    f"(attempt max_dim={dim or self.max_image_dim})"
                )
                last_err = e
                continue

        if last_err is not None:
            raise last_err
        raise RuntimeError("Unexpected VLM retry flow without error/result")


def default_lmstudio_base_url() -> str:
    return os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").strip() or "http://127.0.0.1:1234/v1"


def default_lmstudio_model() -> str:
    return os.getenv("LMSTUDIO_MODEL_ID", "qwen/qwen3-vl-4b").strip() or "qwen/qwen3-vl-4b"


def default_lmstudio_timeout() -> int:
    raw = (os.getenv("LMSTUDIO_TIMEOUT_SECONDS", "180") or "").strip()
    try:
        timeout = int(raw)
    except ValueError:
        timeout = 180
    return max(1, timeout)
