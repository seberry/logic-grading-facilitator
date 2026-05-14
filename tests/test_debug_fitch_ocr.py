import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from scripts import debug_fitch_ocr


def _tfl_handwritten_fixture(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "proofs_ocr"
        / "regression"
        / "fixtures"
        / "tfl"
        / "handwritten"
        / name
        / "image.png"
    )


class TestFuzzySnapJustification(unittest.TestCase):
    """Tests for the combinatorial OCR justification repair."""

    def test_valid_head_unchanged(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("∧E 5"), "∧E 5")

    def test_tail_S_to_5_rescue(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("∧E S"), "∧E 5")

    def test_tail_O_to_0_rescue(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("→I O"), "→I 0")

    def test_tail_Z_to_2_rescue(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("∧E Z"), "∧E 2")

    def test_clean_justification_repairs_epsilon_like_wedge_elim(self):
        from proofs_ocr.tricks.justification_snaps import (
            TFL_VALID_RULE_HEADS,
            clean_justification_for_tfl,
            fuzzy_snap_justification,
            snap_justification_rule_head,
        )

        cleaned = clean_justification_for_tfl("8 ∈ Z")
        snapped = snap_justification_rule_head(cleaned, valid_rules=TFL_VALID_RULE_HEADS)
        self.assertEqual(
            fuzzy_snap_justification(snapped, valid_rules=TFL_VALID_RULE_HEADS),
            "∧E 2",
        )

    def test_clean_justification_repairs_multiplication_sign_to_x_rule(self):
        from proofs_ocr.tricks.justification_snaps import clean_justification_for_tfl

        self.assertEqual(clean_justification_for_tfl("×7"), "X 7")

    def test_unknown_head_no_unique_match_returns_tail_fixed(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        # "XX" has no unique valid mapping, tail still gets S→5
        result = fuzzy_snap_justification("XX S")
        self.assertTrue(result.endswith("5"))

    def test_empty_string_returns_empty(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification(""), "")

    def test_head_BI_repairs_to_wedge_I(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("BI 3,4"), "∧I 3,4")


class TestCountTextLinesInRegion(unittest.TestCase):
    """Tests for the horizontal projection line counter."""

    def test_blank_image_returns_1(self):
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region
        blank = np.ones((100, 200, 3), dtype=np.uint8) * 255
        self.assertEqual(count_text_lines_in_region(blank), 1)

    def test_tiny_image_returns_1(self):
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region
        tiny = np.ones((5, 5), dtype=np.uint8) * 255
        self.assertEqual(count_text_lines_in_region(tiny), 1)

    def test_two_text_bands_detected(self):
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region
        img = np.ones((200, 300), dtype=np.uint8) * 255
        # Draw two horizontal bands of "text" (dark pixels)
        img[40:60, 50:250] = 0   # band 1
        img[130:150, 50:250] = 0  # band 2
        self.assertEqual(count_text_lines_in_region(img), 2)

    def test_slanted_repair_line_count_ignores_lined_paper_and_clipped_top_row(self) -> None:
        import cv2
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import estimate_line_count

        img = np.ones((620, 900, 3), dtype=np.uint8) * 236
        for y in [20, 120, 220, 320, 420, 520, 610]:
            cv2.line(img, (0, y), (899, y), (145, 145, 145), 2)

        # Clipped carryover from the row above the child crop.
        cv2.putText(img, "A ->", (80, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 4)

        for index, y in enumerate([95, 205, 315, 425, 535], start=2):
            cv2.putText(img, f"A{index}", (120, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 4)
            cv2.putText(img, f"E {index}", (500, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 4)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            self.assertEqual(estimate_line_count(img), 5)

    def test_slanted_repair_line_count_handles_fake_implication_child_crop(self) -> None:
        import cv2
        from pathlib import Path
        from proofs_ocr.pipelines.tfl_fitch import estimate_line_count

        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "proofs_ocr"
            / "regression"
            / "fixtures"
            / "tfl"
            / "handwritten"
            / "fake_implication_intro"
            / "image.png"
        )
        img = cv2.imread(str(fixture_path))
        self.assertIsNotNone(img)

        # Approximate the 5-line child subproof crop after slanted-bar repair.
        # The top padding intentionally includes clipped ink from line 1.
        child_crop = img[518:1588, 160:]

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            self.assertEqual(estimate_line_count(child_crop), 5)


class TestSlantedBarRepair(unittest.TestCase):
    def test_geometry_preflight_keeps_clean_two_branch_fixture_on_standard_path(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import select_fitch_geometry

        img = cv2.imread(str(_tfl_handwritten_fixture("q4_N_or_K_and_W")))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {"TFL_FITCH_GEOMETRY_MODE": "auto"}):
            geometry = select_fitch_geometry(img)

        self.assertEqual(geometry["selected_mode"], "standard")
        self.assertEqual(geometry["preflight"]["standard"]["bars"], 3)
        self.assertEqual(geometry["preflight"]["slanted"]["bars"], 1)
        self.assertIn("single_bar", geometry["preflight"]["slanted"]["flags"])

    def test_geometry_preflight_switches_overdetecting_lined_crop_to_slanted_path(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import select_fitch_geometry

        img = cv2.imread(str(_tfl_handwritten_fixture("fake_implication_intro")))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {"TFL_FITCH_GEOMETRY_MODE": "auto"}):
            geometry = select_fitch_geometry(img)

        self.assertEqual(geometry["selected_mode"], "slanted")
        self.assertIn("too_many_bars", geometry["preflight"]["standard"]["flags"])
        self.assertEqual(geometry["preflight"]["slanted"]["bars"], 2)

    def test_geometry_preflight_preserves_legacy_slanted_flag_override(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import select_fitch_geometry

        img = cv2.imread(str(_tfl_handwritten_fixture("q4_N_or_K_and_W")))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            geometry = select_fitch_geometry(img)

        self.assertEqual(geometry["selected_mode"], "slanted")
        self.assertEqual(geometry["preflight"]["slanted"]["bars"], 1)

    def test_geometry_mode_standard_overrides_legacy_slanted_flag(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import select_fitch_geometry

        img = cv2.imread(str(_tfl_handwritten_fixture("q4_N_or_K_and_W")))
        self.assertIsNotNone(img)

        with patch.dict(
            "os.environ",
            {"TFL_FITCH_GEOMETRY_MODE": "standard", "TFL_FITCH_SLANTED_BAR_REPAIR": "1"},
        ):
            geometry = select_fitch_geometry(img)

        self.assertEqual(geometry["selected_mode"], "standard")
        self.assertEqual(geometry["preflight"]["standard"]["bars"], 3)

    def test_pipeline_writes_preflight_and_processes_selected_slanted_mode(self) -> None:
        import json
        import tempfile
        import os

        from proofs_ocr.pipelines.tfl_fitch import run_fitch_ocr_pipeline

        class FakeClient:
            def __init__(self) -> None:
                self.slanted_flags = []
                self.last_thinking = ""
                self.last_response_metadata = {
                    "elapsed_s": 0.01,
                    "thinking_chars": 0,
                    "content_chars": 2,
                    "finish_reason": "stop",
                }

            def build_messages_for_prompt(self, image_path, prompt, system_prompt=None):
                return [{"role": "user", "content": prompt}]

            def chat(self, messages, *, model=None, timeout=None, max_tokens=None, extra_payload=None):
                self.slanted_flags.append(os.getenv("TFL_FITCH_SLANTED_BAR_REPAIR"))
                return "[]"

        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"TFL_FITCH_GEOMETRY_MODE": "auto"}):
                result = run_fitch_ocr_pipeline(
                    image_path=_tfl_handwritten_fixture("fake_implication_intro"),
                    output_dir=Path(tmp),
                    client=client,
                )

            preflight = json.loads((Path(tmp) / "geometry_preflight.json").read_text(encoding="utf-8"))

        self.assertIn("geometry=slanted", result["run_summary"])
        self.assertEqual(preflight["selected_mode"], "slanted")
        self.assertIn("too_many_bars", preflight["standard"]["flags"])
        self.assertTrue(client.slanted_flags)
        self.assertTrue(all(value == "1" for value in client.slanted_flags))

    def test_slanted_bar_repair_extends_nested_bar_to_opener_row(self) -> None:
        import cv2
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import detect_vertical_bars

        img = np.ones((430, 700, 3), dtype=np.uint8) * 255
        cv2.line(img, (125, 25), (110, 370), (0, 0, 0), 3)
        cv2.line(img, (165, 105), (150, 350), (0, 0, 0), 3)

        # Dark text rows to the right of the bars.  The row at y=70 stands in
        # for an assumption/opener that the slanted inner bar should include.
        for y in [70, 105, 155, 205, 255, 345]:
            cv2.rectangle(img, (180, y), (300, y + 18), (0, 0, 0), -1)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            boxes = detect_vertical_bars(img)

        self.assertEqual(len(boxes), 2)
        outer, inner = sorted(boxes, key=lambda box: box[0])
        self.assertLessEqual(outer[1], 30)
        self.assertLessEqual(inner[1], 75)

    def test_slanted_phone_preprocess_recovers_fake_implication_geometry(self) -> None:
        import cv2
        from pathlib import Path
        from proofs_ocr.pipelines.tfl_fitch import (
            build_proof_tree,
            crop_full_width,
            crop_region,
            detect_vertical_bars,
            estimate_line_count,
            filter_scaffold_bars,
            preprocess_slanted_phone_crop,
        )

        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "proofs_ocr"
            / "regression"
            / "fixtures"
            / "tfl"
            / "handwritten"
            / "fake_implication_intro"
            / "image.png"
        )
        img = cv2.imread(str(fixture_path))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            normalized = preprocess_slanted_phone_crop(img)
            bars = filter_scaffold_bars(
                detect_vertical_bars(normalized),
                img_w=normalized.shape[1],
                img_h=normalized.shape[0],
            )
            tree = build_proof_tree(bars, normalized.shape[1], normalized.shape[0])

        self.assertEqual(len(bars), 2)
        self.assertEqual(len(tree.children), 1)
        self.assertEqual(len(tree.children[0].children), 1)
        outer = tree.children[0].box
        inner_node = tree.children[0].children[0]
        inner = inner_node.box
        self.assertLess(inner[0] - outer[0], 80)
        self.assertEqual(estimate_line_count(crop_region(normalized, inner_node.box)), 5)
        self.assertEqual(estimate_line_count(crop_full_width(normalized, inner_node)), 5)

    def test_slanted_repair_relaxed_pass_recovers_fragmented_nested_bar(self) -> None:
        import cv2
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import (
            detect_vertical_bars,
            filter_scaffold_bars,
        )

        img = np.ones((545, 715, 3), dtype=np.uint8) * 255

        # Margin/scaffold strokes near the outer proof bar.  The main scope bar
        # is broken into line-height fragments after ruled-line cleanup.
        for x in [54, 69, 82]:
            cv2.line(img, (x, 0), (x, 180), (0, 0, 0), 5)
        for y in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            cv2.line(img, (70, y), (70, min(544, y + 42)), (0, 0, 0), 5)

        # Inner subproof bar appears as line-height fragments after ruled-line
        # cleanup; an 80px vertical opening drops it, while the fallback should
        # merge these into a candidate bar.
        for y in [229, 279, 329]:
            cv2.line(img, (130, y), (130, y + 48), (0, 0, 0), 5)

        # Nearby text rows give the opener-extension code useful evidence.
        for y in [45, 95, 145, 205, 255, 305, 355, 405]:
            cv2.rectangle(img, (150, y), (255, y + 18), (0, 0, 0), -1)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            bars = filter_scaffold_bars(
                detect_vertical_bars(img),
                img_w=img.shape[1],
                img_h=img.shape[0],
            )

        self.assertGreaterEqual(len(bars), 2)
        outer, inner = sorted(bars, key=lambda box: box[0])[:2]
        self.assertLess(outer[0], inner[0])
        self.assertGreaterEqual(inner[3], 100)

    def test_slanted_phone_preprocess_recovers_completed_scratch_geometry(self) -> None:
        import cv2
        from pathlib import Path
        from proofs_ocr.pipelines.tfl_fitch import (
            build_proof_tree,
            crop_region,
            detect_vertical_bars,
            estimate_line_count,
            filter_scaffold_bars,
            preprocess_slanted_phone_crop,
        )

        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "proofs_ocr"
            / "regression"
            / "fixtures"
            / "tfl"
            / "handwritten"
            / "fake_nested_disjunction_scratch_completed"
            / "image.png"
        )
        img = cv2.imread(str(fixture_path))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {
            "TFL_FITCH_SLANTED_BAR_REPAIR": "1",
            "TFL_FITCH_SLANTED_PHONE_PREPROCESS_MODE": "auto",
        }):
            normalized = preprocess_slanted_phone_crop(img)
            bars = filter_scaffold_bars(
                detect_vertical_bars(normalized),
                img_w=normalized.shape[1],
                img_h=normalized.shape[0],
            )
            tree = build_proof_tree(bars, normalized.shape[1], normalized.shape[0])
            outer = tree.children[0]
            disjunction_block = outer.children[0]
            child_line_counts = [
                estimate_line_count(crop_region(normalized, child.box))
                for child in disjunction_block.children
            ]

        self.assertEqual(len(bars), 4)
        self.assertEqual(len(tree.children), 1)
        self.assertEqual(len(outer.children), 1)
        self.assertEqual(len(disjunction_block.children), 2)
        self.assertEqual(child_line_counts, [4, 3])

    def test_auto_geometry_uses_slanted_split_settings_for_completed_scratch(self) -> None:
        import cv2
        from proofs_ocr.pipelines.tfl_fitch import select_fitch_geometry

        img = cv2.imread(str(_tfl_handwritten_fixture("fake_nested_disjunction_scratch_completed")))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {"TFL_FITCH_GEOMETRY_MODE": "auto"}):
            geometry = select_fitch_geometry(img)

        slanted = geometry["preflight"]["slanted"]
        self.assertEqual(geometry["selected_mode"], "slanted")
        self.assertEqual(slanted["shape"], "ROOT->[1->[2->[3,4]]]")
        self.assertEqual(
            [row["tight_rows"] for row in slanted["leaf_rows"]],
            [4, 3],
        )


class TestBuildOcrPromptWithExpectedLines(unittest.TestCase):
    """Tests for the expected_lines parameter in build_ocr_prompt."""

    def test_leaf_prompt_includes_line_count(self):
        from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt
        prompt = build_ocr_prompt(has_children=False, expected_lines=2)
        self.assertIn("exactly 2 proof line(s)", prompt)
        self.assertIn("Output exactly 2 line objects", prompt)

    def test_parent_prompt_ignores_line_count(self):
        from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt
        prompt = build_ocr_prompt(has_children=True, expected_lines=2)
        self.assertNotIn("exactly 2 proof line(s)", prompt)

    def test_27b_fullwidth_prompt_is_compact_and_gutter_aware(self):
        from proofs_ocr.pipelines.tfl_fitch import build_27b_fullwidth_prompt

        prompt = build_27b_fullwidth_prompt(expected_lines=2)

        self.assertIn("Return only a JSON array", prompt)
        self.assertIn("left-gutter line numbers", prompt)
        self.assertIn("do not renumber the crop from 1", prompt)
        self.assertIn("exactly 2 proof line(s)", prompt)
        self.assertNotIn("Role:", prompt)
        self.assertNotIn("Context:", prompt)


class TestVlmFailureDiagnostics(unittest.TestCase):
    def test_process_node_saves_thinking_metadata_and_error_on_vlm_failure(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from proofs_ocr.pipelines.tfl_fitch import SubproofNode, process_node_recursive

        class FailingClient:
            def __init__(self) -> None:
                self.last_thinking = ""
                self.last_response_metadata = {}

            def build_messages_for_prompt(self, image_path, prompt, system_prompt=None):
                return [{"role": "user", "content": prompt}]

            def chat(self, messages, *, model=None, timeout=None, max_tokens=None, extra_payload=None):
                self.last_thinking = "reasoning before empty content"
                self.last_response_metadata = {
                    "model": model or "qwen/qwen3-vl-4b",
                    "thinking_chars": len(self.last_thinking),
                    "content_chars": 0,
                    "finish_reason": "length",
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 99}},
                }
                raise RuntimeError("LM Studio returned an empty completion content")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img = np.ones((120, 240, 3), dtype=np.uint8) * 255
            img[35:55, 40:180] = 0
            node = SubproofNode("1", (10, 10, 40, 90))
            client = FailingClient()

            result = process_node_recursive(node, img, tmp_path, {}, client)

            self.assertEqual(result, [])
            self.assertEqual((tmp_path / "response_1.json").read_text().strip(), "[]")
            self.assertEqual(
                (tmp_path / "thinking_1.txt").read_text(encoding="utf-8"),
                "reasoning before empty content",
            )
            self.assertIn(
                "empty completion",
                (tmp_path / "error_1.txt").read_text(encoding="utf-8"),
            )
            metadata = json.loads((tmp_path / "response_metadata_1.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["thinking_chars"], len("reasoning before empty content"))
            self.assertEqual(metadata["content_chars"], 0)


class TestSeparatedNestedLeafOcr(unittest.TestCase):
    def test_stitch_separated_nested_leaf_ocr_accepts_expected_shape(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import stitch_separated_nested_leaf_ocr

        payload = {
            "formulas": ["W∧A", "W", "S", "S∧A"],
            "justifications": ["∧E 7", "→E 2,8", "∧I 8,9"],
        }

        self.assertEqual(
            stitch_separated_nested_leaf_ocr(payload, expected_lines=4),
            [
                {"line": 1, "wffstr": "W∧A", "jstr": ""},
                {"line": 2, "wffstr": "W", "jstr": "∧E 7"},
                {"line": 3, "wffstr": "S", "jstr": "→E 2,8"},
                {"line": 4, "wffstr": "S∧A", "jstr": "∧I 8,9"},
            ],
        )

    def test_stitch_separated_nested_leaf_ocr_rejects_count_mismatch(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import stitch_separated_nested_leaf_ocr

        payload = {
            "formulas": ["W∧A", "W", "S", "S∧A"],
            "justifications": ["∧E 7", "→E 2,8"],
        }

        self.assertIsNone(stitch_separated_nested_leaf_ocr(payload, expected_lines=4))


class TestVerifierGuidedLineReread(unittest.TestCase):
    def test_find_verifier_suspicions_reports_bad_line_with_ocr_provenance(self) -> None:
        from proofs_ocr.pipelines.tfl_line_reread import find_verifier_suspicions

        structure = [
            {"line": 1, "wffstr": "A", "jstr": "Pr"},
            {"line": 2, "wffstr": "A→B", "jstr": "Pr"},
            {
                "line": 3,
                "wffstr": "B",
                "jstr": "→E 1,3",
                "_ocr_node_id": "ROOT",
                "_ocr_local_index": 3,
                "_ocr_crop_path": "crop_ROOT.jpg",
                "_ocr_expected_lines": 3,
                "_ocr_is_leaf": True,
            },
        ]

        suspicions = find_verifier_suspicions(structure, numprems=2)

        self.assertEqual([item["line"] for item in suspicions], [3])
        self.assertEqual(suspicions[0]["ocr"]["node_id"], "ROOT")
        self.assertTrue(any("Cites itself" in issue for issue in suspicions[0]["issues"]))

    def test_strip_ocr_metadata_removes_nested_transient_keys(self) -> None:
        from proofs_ocr.pipelines.tfl_line_reread import strip_ocr_metadata

        structure = [
            {"line": 1, "wffstr": "A", "jstr": "Pr", "_ocr_node_id": "ROOT"},
            [{"line": 2, "wffstr": "B", "jstr": "Hyp", "_ocr_crop_path": "crop_1.jpg"}],
        ]

        strip_ocr_metadata(structure)

        self.assertNotIn("_ocr_node_id", structure[0])
        self.assertNotIn("_ocr_crop_path", structure[1][0])

    def test_collect_verifier_reread_candidates_calls_small_model_for_leaf_line(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from proofs_ocr.pipelines.tfl_line_reread import collect_verifier_reread_candidates

        class FakeClient:
            def __init__(self) -> None:
                self.calls = []
                self.last_response_metadata = {"elapsed_s": 0.01}

            def build_messages_for_prompt(self, image_path, prompt, system_prompt=None):
                self.calls.append((Path(image_path), prompt))
                return [{"role": "user", "content": prompt}]

            def chat(self, messages, *, model=None, timeout=None, max_tokens=None):
                return json.dumps({"line": 3, "wffstr": "B", "jstr": "→E 1,2"})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            crop_path = tmp_path / "crop_ROOT.jpg"
            img = np.ones((180, 320, 3), dtype=np.uint8) * 255
            img[25:45, 40:180] = 0
            img[85:105, 40:180] = 0
            img[145:165, 40:180] = 0
            self.assertTrue(cv2.imwrite(str(crop_path), img))

            structure = [
                {"line": 1, "wffstr": "A", "jstr": "Pr"},
                {"line": 2, "wffstr": "A→B", "jstr": "Pr"},
                {
                    "line": 3,
                    "wffstr": "B",
                    "jstr": "→E 1,3",
                    "_ocr_node_id": "ROOT",
                    "_ocr_local_index": 3,
                    "_ocr_crop_path": str(crop_path),
                    "_ocr_expected_lines": 3,
                    "_ocr_is_leaf": True,
                },
            ]
            client = FakeClient()

            records = collect_verifier_reread_candidates(
                structure,
                output_dir=tmp_path,
                client=client,
                numprems=2,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "candidate")
            self.assertEqual(records[0]["parsed"]["jstr"], "→E 1,2")
            self.assertTrue(Path(records[0]["row_crop"]).exists())
            self.assertEqual(len(client.calls), 1)
            self.assertIn("Do not infer a valid proof", client.calls[0][1])

    def test_parse_citation_line_numbers_expands_ranges(self) -> None:
        from proofs_ocr.pipelines.tfl_line_reread import parse_citation_line_numbers

        self.assertEqual(parse_citation_line_numbers("∨E 2, 4-6, 9"), [2, 4, 5, 6, 9])

    def test_collect_verifier_reread_derives_fullwidth_and_reads_cited_lines(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from proofs_ocr.pipelines.tfl_line_reread import collect_verifier_reread_candidates

        class FakeClient:
            def __init__(self) -> None:
                self.calls = []
                self.last_response_metadata = {
                    "elapsed_s": 0.02,
                    "thinking_chars": 0,
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
                }

            def build_single_user_image_message(self, image_path, prompt):
                self.calls.append((Path(image_path), prompt, "single"))
                return [{"role": "user", "content": prompt}]

            def build_messages_for_prompt(self, image_path, prompt, system_prompt=None):
                self.calls.append((Path(image_path), prompt, "split"))
                return [{"role": "user", "content": prompt}]

            def chat(self, messages, *, model=None, timeout=None, max_tokens=None, extra_payload=None):
                if len(self.calls) == 1:
                    return json.dumps({"line": 3, "wffstr": "B", "jstr": "→E 1,3"})
                return json.dumps({"line": 1, "wffstr": "A", "jstr": "Pr"})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tight_crop = tmp_path / "crop_ROOT.jpg"
            source = np.ones((180, 320, 3), dtype=np.uint8) * 255
            source[25:45, 5:25] = 0
            source[25:45, 40:180] = 0
            source[85:105, 5:25] = 0
            source[85:105, 40:180] = 0
            source[145:165, 5:25] = 0
            source[145:165, 40:180] = 0
            tight = source.copy()
            tight[:, :35] = 255
            self.assertTrue(cv2.imwrite(str(tight_crop), tight))

            structure = [
                {
                    "line": 1,
                    "wffstr": "A",
                    "jstr": "Pr",
                    "_ocr_node_id": "ROOT",
                    "_ocr_local_index": 1,
                    "_ocr_crop_path": str(tight_crop),
                    "_ocr_expected_lines": 3,
                    "_ocr_is_leaf": False,
                    "_ocr_node_box": [0, 0, 320, 180],
                    "_ocr_parent_id": None,
                },
                {"line": 2, "wffstr": "A→B", "jstr": "Pr"},
                {
                    "line": 3,
                    "wffstr": "B",
                    "jstr": "→E 1,3",
                    "_ocr_node_id": "ROOT",
                    "_ocr_local_index": 3,
                    "_ocr_crop_path": str(tight_crop),
                    "_ocr_expected_lines": 3,
                    "_ocr_is_leaf": False,
                    "_ocr_node_box": [0, 0, 320, 180],
                    "_ocr_parent_id": None,
                },
            ]
            client = FakeClient()

            with patch.dict(
                "os.environ",
                {
                    "TFL_FITCH_27B_SINGLE_USER_MESSAGE": "1",
                    "TFL_FITCH_27B_NO_THINK": "1",
                    "TFL_FITCH_27B_ENABLE_THINKING_FALSE": "1",
                },
            ):
                records = collect_verifier_reread_candidates(
                    structure,
                    output_dir=tmp_path,
                    client=client,
                    numprems=2,
                    source_image=source,
                )

            self.assertEqual([record["target_kind"] for record in records], ["rejected", "cited"])
            self.assertFalse(records[0]["material_change"])
            self.assertEqual(records[0]["full_width_crop_source"], "derived_from_source")
            self.assertTrue(Path(records[0]["full_width_crop"]).exists())
            self.assertIn("/no_think", records[0]["prompt"])
            self.assertEqual(len(client.calls), 2)
            self.assertTrue(all(call[2] == "single" for call in client.calls))

    def test_extract_single_line_crop_uses_stored_row_bounds_for_parent_rows(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        from proofs_ocr.pipelines.tfl_line_reread import extract_single_line_crop

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            crop_path = tmp_path / "fullwidth.jpg"
            output_path = tmp_path / "row.jpg"
            img = np.ones((220, 360, 3), dtype=np.uint8) * 255
            img[20:35, 40:170] = 0
            img[90:105, 40:170] = 0
            img[165:180, 40:170] = 0
            self.assertTrue(cv2.imwrite(str(crop_path), img))

            extract_single_line_crop(
                crop_path,
                local_index=1,
                expected_lines=3,
                output_path=output_path,
                row_bounds=(150, 195),
            )

            row = cv2.imread(str(output_path))
            self.assertIsNotNone(row)
            self.assertEqual(row.shape[0], 45)
            # The stored y-window should select the third visual band even
            # though local_index points at the first row.
            self.assertLess(row[:, 40:170].mean(), 250)

    def test_detect_visible_gutter_line_bounds_selects_absolute_line(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        from proofs_ocr.pipelines.tfl_line_reread import detect_visible_gutter_line_bounds

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            crop_path = tmp_path / "fullwidth.jpg"
            img = np.ones((240, 360, 3), dtype=np.uint8) * 255
            for offset, line_no in enumerate(range(3, 9)):
                y = 25 + offset * 35
                cv2.putText(img, str(line_no), (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(img, f"L{line_no}", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            self.assertTrue(cv2.imwrite(str(crop_path), img))

            bounds = detect_visible_gutter_line_bounds(
                crop_path,
                target_line=4,
                first_line=3,
                last_line=8,
                node_box=[130, 0, 20, 240],
                padding=8,
            )

            self.assertIsNotNone(bounds)
            y1, y2 = bounds
            self.assertLessEqual(y1, 38)
            self.assertGreaterEqual(y2, 58)


class DebugFitchOcrNormalizationTests(unittest.TestCase):
    def test_build_ocr_prompt_mentions_formula_justification_separation(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt()

        self.assertIn("Use 'wffstr' only for the formula", prompt)
        self.assertIn("Do not place justification text inside 'wffstr'", prompt)
        self.assertIn("<SUBPROOF_ID_X>", prompt)

    def test_build_ocr_prompt_parent_includes_subproof_token_instruction(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt(has_children=True)

        self.assertIn("<SUBPROOF_ID_X>", prompt)
        self.assertIn("red box token", prompt)
        self.assertIn("large blank erased area", prompt)
        self.assertIn("visible black proof line above and below", prompt)

    def test_build_ocr_prompt_leaf_omits_subproof_token_instruction(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt(has_children=False)

        self.assertNotIn("<SUBPROOF_ID_X>", prompt)
        self.assertNotIn("red box token", prompt)
        # Leaf prompt still has the core transcription instructions
        self.assertIn("Use 'wffstr' only for the formula", prompt)

    def test_normalize_structure_splits_embedded_justification_from_formula(self) -> None:
        structure = [
            {
                "line": 3,
                "wffstr": "X -> S ->I 2-5",
                "jstr": "",
            }
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["wffstr"], "X → S")
        self.assertEqual(structure[0]["jstr"], "→I 2-5")

    def test_normalize_structure_treats_digit_only_justification_as_blank(self) -> None:
        structure = [
            {"line": 1, "wffstr": "P ↔ Q", "jstr": "2"},
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["wffstr"], "P ↔ Q")
        self.assertEqual(structure[0]["jstr"], "")

    def test_normalize_structure_recovers_contradiction_line_from_justification_only_formula(self) -> None:
        structure = [
            {"line": 5, "wffstr": "¬E 4,5", "jstr": ""},
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["wffstr"], "⊥")
        self.assertEqual(structure[0]["jstr"], "¬E 4,5")

    def test_normalize_structure_expands_compact_rule_citations(self) -> None:
        structure = [
            {"line": 3, "wffstr": "P", "jstr": "&E1"},
            {"line": 4, "wffstr": "Q", "jstr": "<->E2,3"},
            {"line": 7, "wffstr": "A", "jstr": "X6"},
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["jstr"], "∧E 1")
        self.assertEqual(structure[1]["jstr"], "↔E 2,3")
        self.assertEqual(structure[2]["jstr"], "X 6")

    def test_filter_scaffold_bars_keeps_full_height_left_spine(self) -> None:
        """Scaffold removal is disabled — the outer bar is a real subproof delimiter."""
        boxes = [(30, 0, 12, 460), (120, 80, 8, 140)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=560, img_h=480)

        self.assertEqual(filtered, [(30, 0, 12, 460), (120, 80, 8, 140)])

    def test_filter_scaffold_bars_keeps_tall_outer_bar_even_at_edge(self) -> None:
        """Scaffold removal is disabled — keep both bars for the tree builder."""
        boxes = [(118, 45, 17, 538), (140, 242, 8, 138)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=564, img_h=564)

        self.assertEqual(filtered, [(118, 45, 17, 538), (140, 242, 8, 138)])

    def test_filter_scaffold_bars_drops_extreme_page_edge_artifacts_before_scaffold_logic(self) -> None:
        """Narrow bars at the absolute page edges are removed; interior bars survive."""
        boxes = [(0, 0, 9, 285), (559, 99, 20, 675), (612, 328, 20, 391), (1651, 0, 5, 197)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=1656, img_h=875)

        # (0,0,9,285) dropped: narrow + left edge.  (1651,0,5,197) dropped: narrow + right edge.
        # (559,99,20,675) survives: not at edge.  (612,328,20,391) survives: not at edge.
        self.assertEqual(filtered, [(559, 99, 20, 675), (612, 328, 20, 391)])

    def test_filter_scaffold_bars_keeps_lone_tall_bar_after_edge_cleanup(self) -> None:
        """After edge-artifact cleanup, a single remaining bar that is NOT at
        the left edge should be kept — it is a genuine subproof bar, not
        page scaffolding.  (Regression: the code used to drop it.)"""
        boxes = [(0, 0, 7, 201), (215, 148, 23, 517), (1581, 0, 3, 101)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=1584, img_h=730)

        self.assertEqual(filtered, [(215, 148, 23, 517)])

    def test_remove_empty_lines_strips_blank_ocr_noise(self) -> None:
        structure = [
            {"wffstr": "A → ¬B", "jstr": ""},
            {"wffstr": "", "jstr": ""},
            [
                {"wffstr": "A ∧ B", "jstr": ""},
                {"wffstr": "", "jstr": ""},
            ],
            {"wffstr": "", "jstr": ""},
        ]

        debug_fitch_ocr.remove_empty_lines(structure)

        self.assertEqual(len(structure), 2)
        self.assertEqual(len(structure[1]), 1)

    def test_infer_missing_rule_heads_restores_neg_intro(self) -> None:
        structure = [
            {"wffstr": "¬(A ∧ B)", "jstr": "I 2-6"},
        ]

        debug_fitch_ocr.infer_missing_rule_heads(structure)

        self.assertEqual(structure[0]["jstr"], "¬I 2-6")

    def test_subproof_token_baseline_stays_near_tall_mask_top(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import _subproof_token_baseline_y

        baseline_y = _subproof_token_baseline_y(y1=80, h=320, img_h=500, text_height=14)

        self.assertLessEqual(baseline_y, 110)

    def test_subproof_token_baseline_stays_inside_short_mask(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import _subproof_token_baseline_y

        baseline_y = _subproof_token_baseline_y(y1=80, h=28, img_h=500, text_height=14)

        self.assertEqual(baseline_y, 104)

    def test_repair_missing_child_tokens_inserts_before_post_child_line(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import repair_missing_child_tokens

        child = debug_fitch_ocr.SubproofNode("2", (30, 90, 20, 280))
        ocr_result = [
            {"line": 1, "wffstr": "A → (B ∨ C)", "jstr": ""},
            {"line": 2, "wffstr": "A → ¬B", "jstr": ""},
            {"line": 3, "wffstr": "A → C", "jstr": "→I 3-12"},
        ]

        repaired = repair_missing_child_tokens(ocr_result, [child], img_shape=(620, 500, 3))

        self.assertEqual([row["wffstr"] for row in repaired], ["A → (B ∨ C)", "A → ¬B", "<SUBPROOF_ID_2>", "A → C"])

    def test_repair_missing_child_tokens_inserts_before_trailing_intro_line(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import repair_missing_child_tokens

        child = debug_fitch_ocr.SubproofNode("2", (30, 60, 20, 280))
        ocr_result = [
            {"line": 1, "wffstr": "A → (B → C)", "jstr": ""},
            {"line": 2, "wffstr": "(A ∧ B) → C", "jstr": "→I 2-6"},
        ]

        repaired = repair_missing_child_tokens(ocr_result, [child], img_shape=(360, 500, 3))

        self.assertEqual(
            [row["wffstr"] for row in repaired],
            ["A → (B → C)", "<SUBPROOF_ID_2>", "(A ∧ B) → C"],
        )

    def test_repair_missing_child_tokens_orders_before_existing_lower_sibling(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import repair_missing_child_tokens

        upper_child = debug_fitch_ocr.SubproofNode("3", (40, 80, 20, 90))
        lower_child = debug_fitch_ocr.SubproofNode("4", (40, 230, 20, 90))
        ocr_result = [
            {"line": 1, "wffstr": "A", "jstr": ""},
            {"line": 2, "wffstr": "<SUBPROOF_ID_4>", "jstr": ""},
            {"line": 3, "wffstr": "C", "jstr": "∨E 4,5-8,9-11"},
        ]

        repaired = repair_missing_child_tokens(ocr_result, [upper_child, lower_child], img_shape=(450, 500, 3))

        self.assertEqual([row["wffstr"] for row in repaired], ["A", "<SUBPROOF_ID_3>", "<SUBPROOF_ID_4>", "C"])

    def test_top_carryover_trim_y_drops_extra_top_band(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import _top_carryover_trim_y

        trim_y = _top_carryover_trim_y(
            img_h=210,
            bands=[(0, 49), (77, 83), (94, 121), (133, 156), (171, 193)],
            expected_lines=4,
        )

        self.assertEqual(trim_y, 67)

    def test_top_carryover_trim_y_keeps_expected_band_count(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import _top_carryover_trim_y

        trim_y = _top_carryover_trim_y(
            img_h=210,
            bands=[(12, 31), (58, 79), (96, 118), (140, 160)],
            expected_lines=4,
        )

        self.assertEqual(trim_y, 0)

    def test_adjusted_child_box_for_parent_mask_shifts_completed_scratch_leaf_below_carryover(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import (
            adjusted_child_box_for_parent_mask,
            preprocess_slanted_phone_crop,
        )

        raw = cv2.imread(
            "proofs_ocr/regression/fixtures/tfl/handwritten/"
            "fake_nested_disjunction_scratch_completed/image.png"
        )
        img = preprocess_slanted_phone_crop(raw)
        child = debug_fitch_ocr.SubproofNode("3", (183, 112, 35, 199))

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            adjusted = adjusted_child_box_for_parent_mask(child, img)

        self.assertGreater(adjusted[1], child.box[1])
        self.assertLess(adjusted[3], child.box[3])

    def test_drop_trailing_parent_bleed_artifact_removes_e_citation_fragment(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import drop_trailing_parent_bleed_artifact

        ocr_result = [
            {"line": 1, "wffstr": "A", "jstr": ""},
            {"line": 2, "wffstr": "C", "jstr": "4, 5-8 9-11"},
            {"line": 3, "wffstr": "E", "jstr": "7, 12"},
        ]

        filtered = drop_trailing_parent_bleed_artifact(ocr_result)

        self.assertEqual(len(filtered), 2)

    def test_drop_trailing_parent_bleed_artifact_keeps_rule_head_line(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import drop_trailing_parent_bleed_artifact

        ocr_result = [
            {"line": 1, "wffstr": "A", "jstr": ""},
            {"line": 2, "wffstr": "E", "jstr": "∨E 1,2-3"},
        ]

        filtered = drop_trailing_parent_bleed_artifact(ocr_result)

        self.assertIs(filtered, ocr_result)

    def test_repair_trailing_child_token_order_moves_token_before_intro_line(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import repair_trailing_child_token_order

        ocr_result = [
            {"line": 1, "wffstr": "A → (B → C)", "jstr": ""},
            {"line": 2, "wffstr": "(A ∧ B) → C", "jstr": "→I 2-6"},
            {"line": 3, "wffstr": "<SUBPROOF_ID_2>", "jstr": ""},
        ]

        repaired = repair_trailing_child_token_order(ocr_result)

        self.assertEqual(
            [row["wffstr"] for row in repaired],
            ["A → (B → C)", "<SUBPROOF_ID_2>", "(A ∧ B) → C"],
        )


if __name__ == "__main__":
    unittest.main()
