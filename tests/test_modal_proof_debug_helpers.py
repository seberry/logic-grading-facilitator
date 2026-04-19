import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.routes import modal_proof_debug_artifacts as artifact_helpers
from app.routes import modal_proof_debug_fixtures as fixture_helpers
from app.routes import modal_proof_debug_state as state_helpers


class ModalProofDebugHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"modal_proof_helpers_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_state_helpers_round_trip_page_state(self) -> None:
        default_json = json.dumps({"lines": [{"wffstr": "A", "jstr": "PR"}], "meta": {}}, ensure_ascii=False)
        state = state_helpers.default_page_state(default_json)
        state["system"] = "S4"
        state["fixture_id"] = "modal_s4_typeset_r4_corrected"
        state["review_decisions"] = {"4": {"mode": "manual", "formula": "□A", "source_key": ""}}

        state_helpers.save_example_state(
            self.tmp_dir,
            "sample",
            state,
            "state.json",
            default_json,
        )

        loaded = state_helpers.load_example_state(
            self.tmp_dir,
            "sample",
            "state.json",
            default_json,
        )

        self.assertEqual(loaded["system"], "S4")
        self.assertEqual(loaded["fixture_id"], "modal_s4_typeset_r4_corrected")
        self.assertEqual(loaded["review_decisions"]["4"]["formula"], "□A")

    def test_fixture_helpers_parse_manifest_cases(self) -> None:
        manifest_path = self.tmp_dir / "manifest.yaml"
        fixture_image = self.tmp_dir / "fixtures" / "modal" / "sample.png"
        fixture_image.parent.mkdir(parents=True, exist_ok=True)
        fixture_image.write_bytes(b"fakepng")
        manifest_path.write_text(
            "\n".join(
                [
                    "cases:",
                    "  - id: modal_sample",
                    "    domain: modal",
                    "    kind: typeset",
                    f"    image_path: {fixture_image.relative_to(self.tmp_dir).as_posix()}",
                    "    expected_structure_path: fixtures/modal/expected_structure.json",
                    "    notes: sample fixture",
                ]
            ),
            encoding="utf-8",
        )

        fixtures = fixture_helpers.load_modal_regression_fixtures(manifest_path, self.tmp_dir)

        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0]["id"], "modal_sample")
        self.assertEqual(fixtures[0]["absolute_image_path"], str(fixture_image.resolve()))
        self.assertEqual(fixtures[0]["expected_structure_path"], "fixtures/modal/expected_structure.json")
        self.assertEqual(
            fixtures[0]["absolute_expected_structure_path"],
            str((self.tmp_dir / "fixtures" / "modal" / "expected_structure.json").resolve()),
        )
        self.assertEqual(fixture_helpers.fixture_option_label(fixtures[0]), "modal_sample - modal / typeset")

    def test_unary_verifier_results_add_asset_urls(self) -> None:
        example_dir = self.tmp_dir / "sample"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "unary_verifier.json").write_text(
            json.dumps(
                [
                    {
                        "line_number": 4,
                        "line_crop_path": "working\\sample\\verifier_line_1_4.jpg",
                        "candidate_attempts": [
                            {
                                "line_crop_path": "working\\sample\\verifier_line_1_4_band3.jpg",
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

        results = artifact_helpers.load_unary_verifier_results(self.tmp_dir, "sample", "starter")

        self.assertEqual(results[0]["line_crop_filename"], "verifier_line_1_4.jpg")
        self.assertIn("/debug_modal_proof_assets/sample/verifier_line_1_4.jpg", results[0]["line_crop_url"])
        attempt = results[0]["candidate_attempts"][0]
        self.assertEqual(attempt["line_crop_filename"], "verifier_line_1_4_band3.jpg")
        self.assertIn(
            "/debug_modal_proof_assets/sample/verifier_line_1_4_band3.jpg",
            attempt["line_crop_url"],
        )

    def test_legacy_response_list_still_maps_row_order_to_visible_line_order(self) -> None:
        example_dir = self.tmp_dir / "sample"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "response_1.json").write_text(
            json.dumps(
                [
                    {
                        "formula_text": "A",
                        "justification_text": "Assumption",
                        "scope_role": "ordinary_assumption",
                        "_ocr_row_order": 1,
                    },
                    {
                        "scope_role": "subproof_token",
                        "formula_text": "<SUBPROOF_ID_2>",
                        "justification_text": "",
                    },
                    {
                        "formula_text": "□A",
                        "justification_text": "PR",
                        "scope_role": "premise",
                        "_ocr_row_order": 3,
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        node_response = artifact_helpers.load_node_response(example_dir, "1")

        self.assertEqual(artifact_helpers.row_order_to_visible_line_order(node_response, 1), 0)
        self.assertEqual(artifact_helpers.row_order_to_visible_line_order(node_response, 3), 1)
        artifact = artifact_helpers.find_line_artifact(node_response, 3)
        assert artifact is not None
        self.assertEqual(artifact["kind"], "line")
        self.assertEqual(artifact["resolved"]["scope_role"], "premise")


if __name__ == "__main__":
    unittest.main()
