import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.routes import modal_proof_debug_formula_review as formula_review


class ModalProofDebugFormulaReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"modal_formula_review_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_checker_guided_formula_review_targets_checker_line_and_citations(self) -> None:
        example_dir = self.tmp_dir / "sample"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "node_entries_ROOT.json").write_text(
            json.dumps(
                [
                    {
                        "formula_text": "□P",
                        "justification_text": "PR",
                        "scope_role": "premise",
                        "_ocr_node_id": "ROOT",
                        "_ocr_row_order": 1,
                        "line": 1,
                    },
                    {
                        "formula_text": "Q → ¬E",
                        "justification_text": "4,5",
                        "scope_role": "derived_line",
                        "_ocr_node_id": "ROOT",
                        "_ocr_row_order": 2,
                        "line": 2,
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        for row_order, text in ((1, "[]P"), (2, "Q ->E 4,5")):
            image = np.full((52, 420, 3), 255, dtype=np.uint8)
            cv2.putText(image, text, (32, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.imwrite(str(example_dir / f"node_line_ROOT_{row_order}.jpg"), image)

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del prompt, system_prompt
                self.calls.append(Path(image_path).name)
                if "line_1_" in Path(image_path).name:
                    return '{"formula_text":"□P","confidence":"0.99"}'
                return '{"formula_text":"Q","confidence":"0.99"}'

        state = {
            "proof_json_text": json.dumps(
                {
                    "lines": [
                        {"line": 1, "wffstr": "□", "jstr": "PR"},
                        {"line": 2, "wffstr": "Q → ¬E", "jstr": "→E 1,4"},
                    ],
                    "meta": {},
                },
                ensure_ascii=False,
            ),
            "checker_response": {
                "parse_errors": [],
                "check_errors": [{"line": 2, "code": "BadUsage", "message": "bad rule"}],
            },
        }

        records = formula_review.run_checker_guided_formula_review(
            example_dir=example_dir,
            state=state,
            client=FakeClient(),
            modal_config={"system": "K"},
        )

        self.assertEqual([item["line_number"] for item in records], [2, 1])
        self.assertEqual(records[0]["baseline_formula"], "Q → ¬E")
        self.assertEqual(records[0]["target_reasons"], ["check_error"])
        self.assertEqual(records[1]["baseline_formula"], "□")
        self.assertEqual(records[1]["structured_formula"], "□P")
        self.assertEqual(records[1]["target_reasons"], ["cited_by_2"])
        self.assertEqual(len(records[1]["candidate_attempts"]), 2)
        self.assertTrue((example_dir / "checker_formula_rereads.json").exists())

    def test_target_reasons_expand_ranges_and_prioritize_parse_errors(self) -> None:
        proof_lines = {
            1: {"citation": "PR"},
            2: {"citation": "PR"},
            3: {"citation": "PR"},
            4: {"citation": "→I 1-3"},
            5: {"citation": "PR"},
        }
        checker_response = {
            "parse_errors": [{"line": 4, "code": "Bad", "message": "bad parse"}],
            "check_errors": [],
        }

        target_reasons = formula_review._collect_target_reasons(
            checker_response=checker_response,
            proof_lines=proof_lines,
        )

        self.assertEqual(target_reasons[4], ["parse_error"])
        self.assertEqual(target_reasons[1], ["cited_by_4"])
        self.assertEqual(target_reasons[2], ["cited_by_4"])
        self.assertEqual(target_reasons[3], ["cited_by_4", "adjacent_to_4"])
        self.assertEqual(target_reasons[5], ["adjacent_to_4"])
        self.assertEqual(
            formula_review._ordered_target_line_numbers(target_reasons),
            [4, 1, 2, 3, 5],
        )

    def test_non_primary_strict_open_dependency_is_skipped(self) -> None:
        self.assertFalse(
            formula_review._should_review_line(
                line_number=3,
                target_reasons=["cited_by_7"],
                source_record={"scope_role": "strict_open"},
            )
        )
        self.assertTrue(
            formula_review._should_review_line(
                line_number=3,
                target_reasons=["parse_error"],
                source_record={"scope_role": "strict_open"},
            )
        )
