"""
Debug viewer for VLM-based answer region detection.

We run single-question segmentation across all scans in a folder,
draw red boxes, and build an HTML page to quickly inspect Qwen's behavior.

For each scanned PAGE and problem_id (currently hard-coded to Q1),
we ask a local VLM for:

    - present: bool        # is the answer to this problem on THIS page?
    - bbox: [x_min, y_min, x_max, y_max] in normalized page coords

We then draw a red rectangle and optionally save a crop.

Limitations:
    - Assumes each problem's answer fits on a single page.
    - Only supports one axis-aligned rectangle per (page, problem).

Usage (from repo root):

    python scripts/debug_segment_all_scans.py \
        --raw-scans-dir "data/raw_scans/X and IP Quiz (Week 10)" \
        --problem-id "Q1" \
        --problem-description "True/False question about X and IP" \
        --output-dir "debug_output_q1"

You need:
  - pillow installed:  pip install pillow
  - a working VLM client (e.g. LMStudioQwenClient) wired into call_vlm_for_page()
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.vlm_client import LMStudioQwenClient  # reuse the one from auto_group_scans


from PIL import Image, ImageDraw, ImageFont


# ========= PROMPT TEMPLATE (matches our "human-friendly single-question" spec) =========

SINGLE_QUESTION_PROMPT = SINGLE_QUESTION_PROMPT = SINGLE_QUESTION_PROMPT = """You are an AI assistant helping to grade scanned student exams.
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
"""



# ========= VLM CALL HOOK (YOU NEED TO WIRE THIS TO YOUR CLIENT) =========

def call_vlm_for_page(
    image_path: Path,
    problem_id: str,
    problem_description: str,
    client=None,
) -> Dict[str, Any]:
    """
    Call your local Qwen/Qwen-VL model on a single page image with the
    single-question prompt.

    Returns a dict with keys:
      - "target_problem_id": str
      - "present": bool
      - "bbox": [x_min, y_min, x_max, y_max]  (normalized floats) OR None
      - "comment": str or None
      - "confidence": float (0–1)
    """

    from app.vlm_client import LMStudioQwenClient  # type: ignore

    if client is None:
        client = LMStudioQwenClient(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen/qwen3-vl-4b",  # adjust if you use a different id in LM Studio
        )

    # SINGLE_QUESTION_PROMPT should already be defined near the top of this file.
    # It describes the JSON format and explains the single-question task.
    prompt = SINGLE_QUESTION_PROMPT.format(
        problem_id=problem_id,
        problem_description=problem_description,
    )

    # You can optionally pass a more specific system prompt; but it's fine to
    # just let chat_with_image_prompt's default system prompt handle it, since
    # the JSON format and instructions live in `prompt`.
    raw_response = client.chat_with_image_prompt(
        image_path=image_path,
        prompt=prompt,
    )

    import json
    try:
        data = json.loads(raw_response)
    except Exception as e:
        # If the model sometimes wraps JSON in backticks or extra text,
        # we could add a small “extract { ... }” step here. For now, fail soft.
        print(f"[WARN] JSON parse error for {image_path.name}: {e}")
        return {
            "target_problem_id": problem_id,
            "present": False,
            "bbox": None,
            "comment": f"JSON parse error: {e}",
            "confidence": 0.0,
        }

    # Normalize expected keys
    target_problem_id = data.get("target_problem_id", problem_id)
    present = bool(data.get("present", False))
    bbox = data.get("bbox")
    comment = data.get("comment")
    confidence = float(data.get("confidence", 0.0))

    # Basic bbox sanity-check
    if bbox is not None and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4):
        print(f"[WARN] Bad bbox from model for {image_path.name}: {bbox!r}")
        bbox = None

    return {
        "target_problem_id": target_problem_id,
        "present": present,
        "bbox": bbox,
        "comment": comment,
        "confidence": confidence,
    }





# ========= IMAGE / BBOX HELPERS =========

def bbox_to_pixels(
    bbox: Optional[List[float]],
    width: int,
    height: int,
) -> Optional[tuple]:
    """
    Auto-detect if bbox is normalized [0-1] or pixel coordinates.
    Returns (x_min, y_min, x_max, y_max) in pixel coords, or None if invalid.
    """
    if bbox is None or len(bbox) != 4:
        return None

    x_min, y_min, x_max, y_max = bbox

    # Auto-detect: if all values are <= 1.0, assume normalized
    # (This works because pixel coords for real images are always > 1)
    if max(x_min, y_min, x_max, y_max) <= 1.0:
        # Normalized coordinates - convert to pixels
        x_min = int(round(x_min * width))
        x_max = int(round(x_max * width))
        y_min = int(round(y_min * height))
        y_max = int(round(y_max * height))
    else:
        # Already pixel coordinates - just convert to int
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Clamp to image bounds
    x_min = max(0, min(x_min, width - 1))
    x_max = max(0, min(x_max, width))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(0, min(y_max, height))

    # Sanity check
    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def load_font(size: int = 24):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return None


# ========= MAIN DEBUG LOGIC =========

def gather_image_paths(raw_scans_dir: Path) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    paths = [
        p
        for p in sorted(raw_scans_dir.glob("**/*"))
        if p.suffix.lower() in exts and p.is_file()
    ]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-scans-dir",
        type=Path,
        required=True,
        help="Directory containing raw scans for a single quiz (images).",
    )
    parser.add_argument(
        "--problem-id",
        type=str,
        required=True,
        help='Problem id to search for, e.g. "Q1".',
    )
    parser.add_argument(
        "--problem-description",
        type=str,
        required=True,
        help='Short description, e.g. "True/False question about X".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_output_segment_all"),
    )
    parser.add_argument(
        "--normalized-bbox",
        action="store_true",
        default=False,
        help="Treat model bbox output as normalized [0,1] (default).",
    )
    # You can optionally add arguments for your VLM client config here if you like,
    # e.g. --lmstudio-base-url, --lmstudio-model, etc.

    args = parser.parse_args()

    raw_scans_dir: Path = args.raw_scans_dir
    problem_id: str = args.problem_id
    problem_description: str = args.problem_description
    output_dir: Path = args.output_dir
    bbox_normalized: bool = args.normalized_bbox

    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate your client here if needed, then pass into call_vlm_for_page().
    vlm_client = LMStudioQwenClient(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen/qwen3-vl-4b",   # or whatever exact model id you're using
    )  # e.g. LMStudioQwenClient(...)

    paths = gather_image_paths(raw_scans_dir)
    if not paths:
        raise SystemExit(f"No image files found in {raw_scans_dir}")

    font = load_font()

    debug_records = []  # for JSON log
    crops_html_snippets = []

    for idx, img_path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] Processing {img_path.name}...")

        # --- Call your VLM to get the segmentation for this page ---
        seg = call_vlm_for_page(
            image_path=img_path,
            problem_id=problem_id,
            problem_description=problem_description,
            client=vlm_client,
        )

       
        # seg is expected to be a dict as described above
        present = bool(seg.get("present"))
        bbox = seg.get("bbox")
        comment = seg.get("comment")
        confidence = seg.get("confidence")
        target_problem_id = seg.get("target_problem_id")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        print(f"  Image: {img_path.name} is {width}x{height}")

         # DEBUG: Print what the model actually returned
        print(f"  Model returned: present={present}, bbox={bbox}")
        print(f"  Image dimensions: {width}x{height}")
        if bbox:
            print(f"  Bbox interpretation: normalized={bbox_normalized}")


        debug_records.append(
            {
                "filename": img_path.name,
                "target_problem_id": target_problem_id,
                "present": present,
                "bbox": bbox,
                "comment": comment,
                "confidence": confidence,
            }
        )

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Prepare overlay (draw red box if present)
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)

        crop_filename = None

        if present and bbox is not None:
            pixel_box = bbox_to_pixels(bbox, width, height)
            if pixel_box is None:
                print(f"  [WARN] {img_path.name}: invalid bbox {bbox}, skipping box.")
            else:
                x_min, y_min, x_max, y_max = pixel_box
                draw.rectangle(pixel_box, outline="red", width=4)
                label = problem_id
                if font is not None:
                    draw.text((x_min + 3, y_min + 3), label, fill="red", font=font)
                else:
                    draw.text((x_min + 3, y_min + 3), label, fill="red")

                # Save crop using the CLAMPED coordinates
                try:
                    crop = img.crop(pixel_box)
                    crop_filename = f"{idx:03d}_{problem_id}_crop.png"
                    crop_path = output_dir / crop_filename
                    crop.save(crop_path)
                except Exception as e:
                    print(f"  [ERROR] Failed to crop {img_path.name}: {e}")
                    print(f"         Image size: {width}x{height}, bbox: {bbox}, pixel_box: {pixel_box}")
                    crop_filename = None
        else:
            print(f"  [INFO] {img_path.name}: present = {present}, no bbox/crop.")

        # Save overlay
        overlay_filename = f"{idx:03d}_{img_path.stem}_overlay.png"
        overlay_path = output_dir / overlay_filename
        overlay.save(overlay_path)

        # Build HTML snippet for this page
        comment_html = f"<p><strong>Comment:</strong> {comment}</p>" if comment else ""
        conf_html = f"<p><strong>Confidence:</strong> {confidence}</p>" if confidence is not None else ""

        if present and crop_filename is not None:
            crop_img_tag = f'<img src="{crop_filename}" style="max-width: 400px; border: 1px solid #ccc;"/>'
        elif present:
            crop_img_tag = "<p>present = true but no valid crop produced.</p>"
        else:
            crop_img_tag = "<p>No answer detected (present = false).</p>"

        crops_html_snippets.append(
            f"""
<section style="margin-bottom: 2em;">
  <h2>{idx:03d}. {img_path.name}</h2>
  <p><strong>target_problem_id:</strong> {target_problem_id}</p>
  <p><strong>present:</strong> {present}</p>
  {conf_html}
  {comment_html}
  <div style="display: flex; gap: 1em; flex-wrap: wrap;">
    <div>
      <p>Full page with overlay:</p>
      <img src="{overlay_filename}" style="max-width: 500px; border: 1px solid #ccc;"/>
    </div>
    <div>
      <p>Answer crop for {problem_id}:</p>
      {crop_img_tag}
    </div>
  </div>
</section>
"""
        )

    # Write JSON log
    json_log_path = output_dir / "segment_results.json"
    json_log_path.write_text(json.dumps(debug_records, indent=2), encoding="utf-8")

    # Write HTML
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Segmentation Debug View – {problem_id}</title>
  </head>
  <body>
    <h1>Segmentation Debug View – {problem_id}</h1>
    <p>Raw scans directory: {raw_scans_dir}</p>
    <p>Problem description: {problem_description}</p>
    <p>This page shows, for each scan, the full-page overlay and the cropped answer for {problem_id} (if detected).</p>
    <hr/>
    {''.join(crops_html_snippets)}
  </body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")

    print()
    print("Done.")
    print(f"JSON log: {json_log_path.resolve()}")
    print(f"HTML debug page: {html_path.resolve()}")
    print("Open that HTML file in your browser to inspect the boxes.")


if __name__ == "__main__":
    main()
