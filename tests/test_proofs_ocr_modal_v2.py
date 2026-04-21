import unittest
from pathlib import Path
import shutil

import cv2
import numpy as np

from proofs_ocr.domains import modal
from proofs_ocr.domains import modal_line_zones
from modal_ocr.prompts import build_justification_ocr_prompt
from proofs_ocr.pipelines.modal_v2_bar_filters import filter_false_left_separator_bars
from proofs_ocr.pipelines.modal_v2_unary import run_structured_unary_probe
from proofs_ocr.pipelines import modal_v2
from proofs_ocr.pipelines import modal_v2_core
from proofs_ocr.pipelines.modal_v2 import run_modal_baseline_pipeline
from proofs_ocr.pipelines.modal_v2_core import ModalPipelineFeatureFlags
from proofs_ocr.pipelines.modal_v2_line_rereads import maybe_reread_contradiction_line, maybe_reread_justification_head
from proofs_ocr.pipelines.modal_v2_formula_rereads import maybe_reread_formula_text
from proofs_ocr.pipelines.modal_v2_runtime import ModalBaselineArtifactLineReader
from proofs_ocr.pipelines.modal_v2_line_zones import (
    extract_line_zone_crops,
    extract_primary_ink_band_crop,
    extract_tight_justification_crop,
)
from proofs_ocr.common.types import SubproofNode


def _synthetic_recursive_modal_image() -> np.ndarray:
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, "P", (12, 30), font, 0.8, (0, 0, 0), 2)
    cv2.line(image, (42, 54), (42, 150), (0, 0, 0), 2)
    cv2.putText(image, "A", (62, 78), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (62, 128), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Q", (12, 188), font, 0.8, (0, 0, 0), 2)

    return image


def _synthetic_false_separator_image() -> np.ndarray:
    image = np.full((280, 260, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(image, (18, 0), (18, 262), (0, 0, 0), 8)
    cv2.line(image, (54, 157), (54, 262), (0, 0, 0), 2)
    cv2.putText(image, "P", (72, 28), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Q", (72, 62), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "A", (82, 188), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (82, 224), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "R", (72, 254), font, 0.8, (0, 0, 0), 2)
    return image


def _synthetic_scope_bar_image() -> np.ndarray:
    image = np.full((420, 320, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(image, (38, 2), (38, 417), (0, 0, 0), 2)
    cv2.line(image, (54, 157), (54, 285), (0, 0, 0), 2)
    cv2.putText(image, "A", (74, 24), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (74, 56), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "C", (74, 96), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "D", (74, 132), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "E", (86, 204), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "F", (86, 244), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "G", (74, 308), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "H", (74, 432), font, 0.8, (0, 0, 0), 2)
    return image


class ModalV2BaselinePipelineTests(unittest.TestCase):
    def test_select_top_level_entries_prefers_first_child_when_root_has_only_token(self) -> None:
        root = SubproofNode("ROOT", (0, 0, 100, 100))

        selected = modal_v2_core._select_top_level_entries(
            tree_root=root,
            library={
                "ROOT": [{"formula_text": "<SUBPROOF_ID_1>", "scope_role": "subproof_token"}],
                "1": [{"formula_text": "A", "scope_role": "premise"}],
            },
        )

        self.assertEqual(selected, [{"formula_text": "A", "scope_role": "premise"}])

    def test_select_top_level_entries_keeps_root_when_root_has_visible_line(self) -> None:
        root = SubproofNode("ROOT", (0, 0, 100, 100))

        selected = modal_v2_core._select_top_level_entries(
            tree_root=root,
            library={
                "ROOT": [
                    {"formula_text": "<SUBPROOF_ID_1>", "scope_role": "subproof_token"},
                    {"formula_text": "T", "scope_role": "derived_line"},
                ],
                "1": [{"formula_text": "A", "scope_role": "premise"}],
            },
        )

        self.assertEqual(
            selected,
            [
                {"formula_text": "<SUBPROOF_ID_1>", "scope_role": "subproof_token"},
                {"formula_text": "T", "scope_role": "derived_line"},
            ],
        )

    def test_node_crop_image_avoids_bottom_padding_for_child_nodes(self) -> None:
        image = np.zeros((200, 50, 3), dtype=np.uint8)
        child = SubproofNode("1", (10, 40, 2, 80))

        crop = modal_v2._node_crop_image(node=child, full_img=image)

        self.assertEqual(crop.shape[0], 90)

    def test_extract_ocr_line_crop_keeps_formula_side_visible(self) -> None:
        image = np.full((90, 900, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "1", (300, 42), font, 1.1, (0, 0, 0), 2)
        cv2.line(image, (345, 8), (345, 76), (0, 0, 0), 2)
        cv2.putText(image, "[]A", (395, 42), font, 1.1, (0, 0, 0), 2)

        crop = modal_v2._extract_ocr_line_crop(
            node_crop_img=image,
            row_record={"y1": 18, "y2": 46},
        )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        formula_zone = gray[:, 380:520]
        self.assertGreater(int((formula_zone < 185).sum()), 25)

    def test_recover_line_entry_from_crop_recovers_tiny_strict_open(self) -> None:
        image = np.full((40, 220, 3), 255, dtype=np.uint8)
        image[:, 0:3] = 0
        cv2.rectangle(image, (34, 8), (56, 30), (0, 0, 0), 2)

        resolved, recovery = modal.recover_line_entry_from_crop(
            line_crop_img=image,
            parsed_entry={
                "line_label": "",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "premise",
                "confidence": "1",
            },
        )

        self.assertEqual(recovery, "tiny_strict_open_crop")
        self.assertEqual(resolved["formula_text"], "□")
        self.assertEqual(resolved["scope_role"], "strict_open")

    def test_recover_line_entry_from_crop_does_not_fire_for_square_plus_formula(self) -> None:
        image = np.full((48, 260, 3), 255, dtype=np.uint8)
        image[:, 0:3] = 0
        cv2.rectangle(image, (34, 10), (56, 32), (0, 0, 0), 2)
        cv2.putText(image, "A", (78, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        resolved, recovery = modal.recover_line_entry_from_crop(
            line_crop_img=image,
            parsed_entry={
                "line_label": "",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "premise",
                "confidence": "1",
            },
        )

        self.assertEqual(recovery, "")
        self.assertEqual(resolved["formula_text"], "")

    def test_recover_line_entry_from_crop_does_not_fire_when_line_label_is_present(self) -> None:
        image = np.full((40, 220, 3), 255, dtype=np.uint8)
        image[:, 0:3] = 0
        cv2.rectangle(image, (34, 8), (56, 30), (0, 0, 0), 2)

        resolved, recovery = modal.recover_line_entry_from_crop(
            line_crop_img=image,
            parsed_entry={
                "line_label": "1",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "premise",
                "confidence": "1",
            },
        )

        self.assertEqual(recovery, "")
        self.assertEqual(resolved["formula_text"], "")

    def test_detect_label_only_crop_flags_wide_label_region_blob(self) -> None:
        image = np.full((52, 1433, 3), 255, dtype=np.uint8)
        cv2.putText(image, "5", (483, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        skip_reason = modal.detect_label_only_crop(image)

        self.assertEqual(skip_reason, "label_only_crop")

    def test_detect_label_only_crop_keeps_wide_formula_crop(self) -> None:
        image = np.full((52, 1433, 3), 255, dtype=np.uint8)
        cv2.putText(image, "3", (500, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, "~[]A", (610, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        skip_reason = modal.detect_label_only_crop(image)

        self.assertEqual(skip_reason, "")

    def test_modal_baseline_artifact_line_reader_skips_label_only_crop(self) -> None:
        class FailIfCalledClient:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                raise AssertionError("client should not be called for label-only crop")

        image = np.full((52, 1433, 3), 255, dtype=np.uint8)
        cv2.putText(image, "5", (483, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        node = SubproofNode("ROOT", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_skip")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=FailIfCalledClient(),
                prompt="prompt",
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 5, "band_index": 3, "y1": 10, "y2": 30},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["formula_text"], "")
            payload_path = output_dir / "node_line_ROOT_5.json"
            self.assertTrue(payload_path.exists())
            payload = payload_path.read_text(encoding="utf-8")
            self.assertIn('"skip_reason": "label_only_crop"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_recovers_first_nonroot_strict_open_band(self) -> None:
        class FixedClient:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return (
                    '{"line_label":"","formula_text":"Assum","justification_text":"",'
                    '"scope_role":"ordinary_assumption","confidence":"0.98"}'
                )

        image = np.full((44, 320, 3), 255, dtype=np.uint8)
        cv2.line(image, (6, 0), (6, 43), (0, 0, 0), 2)
        cv2.rectangle(image, (20, 4), (38, 22), (0, 0, 0), 2)
        cv2.line(image, (20, 27), (110, 27), (0, 0, 0), 2)
        cv2.putText(image, "Assum", (190, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "A", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        node = SubproofNode("2", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_first_nonroot_strict_open")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=FixedClient(),
                prompt="prompt",
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 1, "band_index": 0, "y1": 4, "y2": 26},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["formula_text"], "□")
            self.assertEqual(resolved["scope_role"], "strict_open")
            self.assertEqual(resolved["justification_text"], "")
            payload = (output_dir / "node_line_2_1.json").read_text(encoding="utf-8")
            self.assertIn('"recovery_applied": "first_nonroot_strict_open_band"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_contradiction_line_accepts_bottom_for_negation_elim_signal(self) -> None:
        class ContradictionVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return (
                    '{"line_label":"","formula_text":"⊥","justification_text":"¬E 4,8",'
                    '"scope_role":"derived_line","confidence":"0.99"}'
                )

        image = np.full((40, 260, 3), 255, dtype=np.uint8)
        cv2.putText(image, "1", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "~E 4,8", (150, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_contradiction_reread")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_contradiction_line(
                output_dir=output_dir,
                client=ContradictionVerifier(),
                node_id="1",
                row_order=7,
                node_crop_img=image,
                row_record={"row_order": 7, "band_index": 5, "y1": 10, "y2": 30},
                line_crop_img=image,
                parsed_entry={
                    "line_label": "",
                    "formula_text": "¬E",
                    "justification_text": "",
                    "scope_role": "derived_line",
                    "confidence": "0.6",
                },
            )

            self.assertEqual(recovery, "contradiction_reread")
            self.assertEqual(resolved["formula_text"], "⊥")
            self.assertEqual(resolved["justification_text"], "¬E 4,8")
            self.assertIsNotNone(artifact)
            self.assertTrue(artifact["applied"])
            self.assertTrue((output_dir / "node_line_contradiction_reread_1_7.jpg").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_can_apply_contradiction_reread(self) -> None:
        class SequencedClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                self.calls += 1
                if self.calls == 1:
                    return (
                        '{"line_label":"","formula_text":"¬E","justification_text":"",'
                        '"scope_role":"derived_line","confidence":"0.7"}'
                    )
                return (
                    '{"line_label":"","formula_text":"⊥","justification_text":"¬E 4,8",'
                    '"scope_role":"derived_line","confidence":"0.99"}'
                )

        image = np.full((44, 320, 3), 255, dtype=np.uint8)
        cv2.putText(image, "1", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "~E 4,8", (180, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        node = SubproofNode("1", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_contradiction_reread")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            verifier_client = SequencedClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=verifier_client,
                verifier_client=verifier_client,
                prompt="prompt",
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 7, "band_index": 5, "y1": 10, "y2": 30},
                feature_flags=ModalPipelineFeatureFlags(enable_contradiction_reread=True),
            )

            self.assertEqual(resolved["formula_text"], "⊥")
            self.assertEqual(resolved["justification_text"], "¬E 4,8")
            payload = (output_dir / "node_line_1_7.json").read_text(encoding="utf-8")
            self.assertIn('"recovery_applied": "contradiction_reread"', payload)
            self.assertIn('"contradiction_reread"', payload)
            self.assertIn('"formula_text": "⊥"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_justification_head_accepts_boxed_rule_head_for_digits_only_line(self) -> None:
        class JustificationVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return '{"justification_text":"□E1","confidence":"0.98"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "P", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_justification_head_reread")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_justification_head(
                output_dir=output_dir,
                client=JustificationVerifier(),
                node_id="2",
                row_order=2,
                line_crop_img=image,
                parsed_entry={
                    "line_label": "4",
                    "formula_text": "P",
                    "justification_text": "1",
                    "scope_role": "derived_line",
                    "confidence": "0.6",
                },
            )

            self.assertEqual(recovery, "justification_head_reread")
            self.assertEqual(resolved["formula_text"], "P")
            self.assertEqual(resolved["justification_text"], "□E 1")
            self.assertIsNotNone(artifact)
            self.assertTrue(artifact["applied"])
            self.assertTrue((output_dir / "node_line_justification_reread_2_2.jpg").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_justification_head_accepts_later_nonroot_pr_line(self) -> None:
        class JustificationVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return '{"justification_text":"□E 1","confidence":"0.98"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "P", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_justification_head_nonroot_pr")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_justification_head(
                output_dir=output_dir,
                client=JustificationVerifier(),
                node_id="2",
                row_order=2,
                line_crop_img=image,
                parsed_entry={
                    "line_label": "4",
                    "formula_text": "P",
                    "justification_text": "PR",
                    "scope_role": "premise",
                    "confidence": "0.6",
                },
            )

            self.assertEqual(recovery, "justification_head_reread")
            self.assertEqual(resolved["justification_text"], "□E 1")
            self.assertIsNotNone(artifact)
            self.assertTrue(artifact["applied"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_justification_head_accepts_malformed_rule_fragment(self) -> None:
        class JustificationVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return '{"justification_text":"□E 2","confidence":"0.98"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "A", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "1, []E", (265, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_justification_head_malformed")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_justification_head(
                output_dir=output_dir,
                client=JustificationVerifier(),
                node_id="2",
                row_order=2,
                line_crop_img=image,
                parsed_entry={
                    "line_label": "2",
                    "formula_text": "A",
                    "justification_text": "1, □E",
                    "scope_role": "derived_line",
                    "confidence": "0.6",
                },
            )

            self.assertEqual(recovery, "justification_head_reread")
            self.assertEqual(resolved["justification_text"], "□E 2")
            self.assertIsNotNone(artifact)
            self.assertTrue(artifact["applied"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_justification_head_rejects_conflicting_malformed_rule_fragment(self) -> None:
        class JustificationVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return '{"justification_text":"□I 2","confidence":"0.98"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "2", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "A", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "1, []E", (265, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_justification_head_conflicting")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_justification_head(
                output_dir=output_dir,
                client=JustificationVerifier(),
                node_id="2",
                row_order=2,
                line_crop_img=image,
                parsed_entry={
                    "line_label": "2",
                    "formula_text": "A",
                    "justification_text": "1, □E",
                    "scope_role": "derived_line",
                    "confidence": "0.6",
                },
            )

            self.assertEqual(recovery, "")
            self.assertEqual(resolved["justification_text"], "1, □E")
            self.assertIsNotNone(artifact)
            self.assertFalse(artifact["applied"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_can_apply_justification_head_reread(self) -> None:
        class SequencedClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                self.calls += 1
                if self.calls == 1:
                    return (
                        '{"line_label":"","formula_text":"P","justification_text":"1",'
                        '"scope_role":"derived_line","confidence":"0.7"}'
                    )
                return '{"justification_text":"□E 1","confidence":"0.99"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "P", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        node = SubproofNode("2", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_justification_head_reread")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            verifier_client = SequencedClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=verifier_client,
                verifier_client=verifier_client,
                prompt="prompt",
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 2, "band_index": 0, "y1": 8, "y2": 42},
                feature_flags=ModalPipelineFeatureFlags(enable_justification_head_reread=True),
            )

            self.assertEqual(resolved["formula_text"], "P")
            self.assertEqual(resolved["justification_text"], "□E 1")
            payload = (output_dir / "node_line_2_2.json").read_text(encoding="utf-8")
            self.assertIn('"recovery_applied": "justification_head_reread"', payload)
            self.assertIn('"justification_head_reread"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_formula_text_accepts_consensus_formula_only_change(self) -> None:
        class FormulaVerifier:
            def __init__(self) -> None:
                self.calls = 0

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                self.calls += 1
                if self.calls == 1:
                    return '{"formula_text":"P → Q","confidence":"0.98"}'
                return '{"formula_text":"P→Q","confidence":"1.0"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "2", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[](P->Q)", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]E 2", (290, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_formula_consensus_reread")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_formula_text(
                output_dir=output_dir,
                client=FormulaVerifier(),
                node_id="2",
                row_order=3,
                line_crop_img=image,
                parsed_entry={
                    "line_label": "2",
                    "formula_text": "□(P→Q)",
                    "justification_text": "□E 2",
                    "scope_role": "derived_line",
                    "confidence": "0.6",
                },
                modal_config={"system": "K", "include_derived_rules": False},
            )

            self.assertEqual(recovery, "formula_consensus_reread")
            self.assertEqual(resolved["formula_text"], "P → Q")
            self.assertIsNotNone(artifact)
            self.assertTrue(artifact["applied"])
            self.assertEqual(len(artifact["candidate_attempts"]), 2)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_maybe_reread_formula_text_accepts_boxed_only_rescue_for_rule_head_leak(self) -> None:
        class FormulaVerifier:
            def __init__(self) -> None:
                self.calls = 0

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                self.calls += 1
                if self.calls == 1:
                    return '{"formula_text":"","confidence":"0.98"}'
                return '{"formula_text":"Q","confidence":"0.95"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "Q", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "->E 4,5", (250, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        output_dir = Path("working/test_modal_v2_formula_boxed_only_rescue")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved, artifact, recovery = maybe_reread_formula_text(
                output_dir=output_dir,
                client=FormulaVerifier(),
                node_id="2",
                row_order=4,
                line_crop_img=image,
                parsed_entry={
                    "line_label": "4",
                    "formula_text": "Q → ¬E",
                    "justification_text": "→E 4,5",
                    "scope_role": "derived_line",
                    "confidence": "0.6",
                },
                modal_config={"system": "K", "include_derived_rules": False},
            )

            self.assertEqual(recovery, "formula_consensus_reread")
            self.assertEqual(resolved["formula_text"], "Q")
            self.assertIsNotNone(artifact)
            self.assertTrue(artifact["applied"])
            self.assertEqual(artifact["proposed_formula"], "Q")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_can_apply_formula_consensus_reread(self) -> None:
        class SequencedClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                self.calls += 1
                if self.calls == 1:
                    return (
                        '{"line_label":"2","formula_text":"□(P→Q)","justification_text":"□E 2",'
                        '"scope_role":"derived_line","confidence":"0.7"}'
                    )
                if self.calls == 2:
                    return '{"formula_text":"P → Q","confidence":"0.98"}'
                return '{"formula_text":"P→Q","confidence":"1.0"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "2", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[](P->Q)", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]E 2", (290, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        node = SubproofNode("2", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_formula_consensus_reread")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            verifier_client = SequencedClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=verifier_client,
                verifier_client=verifier_client,
                prompt="prompt",
                modal_config={"system": "K", "include_derived_rules": False},
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 3, "band_index": 0, "y1": 8, "y2": 42},
                feature_flags=ModalPipelineFeatureFlags(enable_formula_consensus_reread=True),
            )

            self.assertEqual(resolved["formula_text"], "P → Q")
            payload = (output_dir / "node_line_2_3.json").read_text(encoding="utf-8")
            self.assertIn('"recovery_applied": "formula_consensus_reread"', payload)
            self.assertIn('"formula_consensus_reread"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_extract_line_zone_crops_splits_formula_and_justification_sides(self) -> None:
        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]A", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        formula_crop, justification_crop, geometry = extract_line_zone_crops(image)

        self.assertGreater(geometry["split_x"], 120)
        self.assertLess(geometry["split_x"], 300)
        formula_gray = cv2.cvtColor(formula_crop, cv2.COLOR_BGR2GRAY)
        justification_gray = cv2.cvtColor(justification_crop, cv2.COLOR_BGR2GRAY)
        self.assertGreater(int((formula_gray[:, :180] < 185).sum()), 25)
        self.assertGreater(int((justification_gray[:, -140:] < 185).sum()), 25)

    def test_extract_tight_justification_crop_removes_top_spill(self) -> None:
        image = np.full((60, 180, 3), 255, dtype=np.uint8)
        cv2.putText(image, "spill", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(image, "[]E 2", (36, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        tight_crop, geometry = extract_tight_justification_crop(image)

        self.assertTrue(geometry["applied"])
        self.assertGreater(geometry["y1"], 0)
        self.assertLess(tight_crop.shape[0], image.shape[0])
        tight_gray = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY)
        self.assertGreater(int((tight_gray < 185).sum()), 25)

    def test_extract_primary_ink_band_crop_removes_top_spill(self) -> None:
        image = np.full((60, 180, 3), 255, dtype=np.uint8)
        cv2.putText(image, "spill", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(image, "[]A", (36, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        tight_crop, geometry = extract_primary_ink_band_crop(image)

        self.assertTrue(geometry["applied"])
        self.assertGreater(geometry["y1"], 0)
        self.assertLess(tight_crop.shape[0], image.shape[0])

    def test_filter_false_left_separator_bars_removes_top_anchored_left_edge_bar(self) -> None:
        image = _synthetic_false_separator_image()

        filtered, payload = filter_false_left_separator_bars(
            image=image,
            bars=[(18, 0, 8, 262), (54, 157, 2, 128)],
        )

        self.assertEqual(filtered, [(54, 157, 2, 128)])
        self.assertTrue(payload["applied"])
        self.assertEqual(payload["reason"], "code_only_rejected_false_left_separator")
        self.assertTrue(payload["code_signals"]["suspicious_root_ownership"])
        self.assertTrue(payload["code_signals"]["restores_root_rows_above_first_child"])

    def test_filter_false_left_separator_bars_keeps_bar_when_not_top_anchored(self) -> None:
        image = np.full((280, 260, 3), 255, dtype=np.uint8)

        filtered, payload = filter_false_left_separator_bars(
            image=image,
            bars=[(18, 24, 8, 220), (54, 157, 2, 128)],
        )

        self.assertEqual(filtered, [(18, 24, 8, 220), (54, 157, 2, 128)])
        self.assertFalse(payload["applied"])
        self.assertEqual(payload["reason"], "leftmost_not_top_anchored")

    def test_filter_false_left_separator_bars_verifier_keeps_real_scope_bar(self) -> None:
        class ScopeBarVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return '{"classification":"scope_bar","confidence":"high","reason":"outer proof bar"}'

        image = _synthetic_scope_bar_image()
        output_dir = Path("working/test_modal_v2_false_bar_keep")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            filtered, payload = filter_false_left_separator_bars(
                image=image,
                bars=[(38, 2, 2, 417), (54, 157, 2, 128)],
                verifier_client=ScopeBarVerifier(),
                output_dir=output_dir,
            )

            self.assertEqual(filtered, [(38, 2, 2, 417), (54, 157, 2, 128)])
            self.assertFalse(payload["applied"])
            self.assertEqual(payload["reason"], "verifier_kept_scope_bar")
            self.assertEqual(payload["verifier"]["classification"], "scope_bar")
            self.assertTrue((output_dir / "false_left_separator_candidate.jpg").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_filter_false_left_separator_bars_verifier_can_drop_separator(self) -> None:
        class SeparatorVerifier:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return (
                    '{"classification":"number_formula_separator","confidence":"high",'
                    '"reason":"left margin separator beside line numbers"}'
                )

        image = _synthetic_false_separator_image()
        output_dir = Path("working/test_modal_v2_false_bar_drop")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            filtered, payload = filter_false_left_separator_bars(
                image=image,
                bars=[(18, 0, 8, 262), (54, 157, 2, 128)],
                verifier_client=SeparatorVerifier(),
                output_dir=output_dir,
            )

            self.assertEqual(filtered, [(54, 157, 2, 128)])
            self.assertTrue(payload["applied"])
            self.assertEqual(payload["reason"], "verifier_rejected_false_left_separator")
            self.assertEqual(payload["verifier"]["classification"], "number_formula_separator")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_modal_baseline_pipeline_can_filter_false_left_separator_before_tree_build(self) -> None:
        image = _synthetic_false_separator_image()
        rows_seen = []

        def read_line(*, node, node_crop_img, line_crop_img, row_record, feature_flags):
            del node_crop_img, line_crop_img, feature_flags
            rows_seen.append((node.id, int(row_record.get("row_order", 0))))
            return {"scope_role": "premise", "formula_text": "P"}

        original_detect_vertical_bars = modal_v2_core.tree.detect_vertical_bars
        try:
            modal_v2_core.tree.detect_vertical_bars = lambda _img: [(18, 0, 8, 262), (54, 157, 2, 128)]
            result = run_modal_baseline_pipeline(
                image=image,
                read_line=read_line,
                enable_false_left_separator_filter=True,
            )
        finally:
            modal_v2_core.tree.detect_vertical_bars = original_detect_vertical_bars

        self.assertEqual(result.raw_detected_bars, [(18, 0, 8, 262), (54, 157, 2, 128)])
        self.assertEqual(result.detected_bars, [(54, 157, 2, 128)])
        self.assertTrue(result.bar_filter_debug["applied"])

    def test_run_modal_baseline_pipeline_leaves_bars_unchanged_when_false_left_filter_disabled(self) -> None:
        image = _synthetic_false_separator_image()

        def read_line(*, node, node_crop_img, line_crop_img, row_record, feature_flags):
            del node, node_crop_img, line_crop_img, row_record, feature_flags
            return {"scope_role": "premise", "formula_text": "P"}

        original_detect_vertical_bars = modal_v2_core.tree.detect_vertical_bars
        try:
            modal_v2_core.tree.detect_vertical_bars = lambda _img: [(18, 0, 8, 262), (54, 157, 2, 128)]
            result = run_modal_baseline_pipeline(
                image=image,
                read_line=read_line,
                enable_false_left_separator_filter=False,
            )
        finally:
            modal_v2_core.tree.detect_vertical_bars = original_detect_vertical_bars

        self.assertEqual(result.raw_detected_bars, [(18, 0, 8, 262), (54, 157, 2, 128)])
        self.assertEqual(result.detected_bars, [(18, 0, 8, 262), (54, 157, 2, 128)])
        self.assertEqual(result.bar_filter_debug, {})

    def test_modal_line_zones_merge_recovers_justification_from_formula_label_spill(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "□E 2", "formula_text": "A", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=False,
            is_first_visible_row_of_nonroot=False,
        )

        self.assertEqual(merged["formula_text"], "A")
        self.assertEqual(merged["justification_text"], "□E 2")
        self.assertEqual(merged["scope_role"], "derived_line")

    def test_modal_line_zones_merge_marks_blank_root_line_as_premise(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "1", "formula_text": "A", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=True,
            is_first_visible_row_of_nonroot=False,
        )

        self.assertEqual(merged["scope_role"], "premise")
        self.assertEqual(merged["justification_text"], "")

    def test_modal_line_zones_merge_marks_first_nonroot_blank_line_as_assumption(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "", "formula_text": "A", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=False,
            is_first_visible_row_of_nonroot=True,
        )

        self.assertEqual(merged["scope_role"], "ordinary_assumption")
        self.assertEqual(merged["justification_text"], "")

    def test_modal_line_zones_merge_keeps_later_nonroot_blank_line_as_derived(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "", "formula_text": "B", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=False,
            is_first_visible_row_of_nonroot=False,
        )

        self.assertEqual(merged["scope_role"], "derived_line")
        self.assertEqual(merged["justification_text"], "")

    def test_resolve_modal_ocr_entry_normalizes_assumption_variants_to_pr(self) -> None:
        resolved = modal.resolve_modal_ocr_entry(
            {
                "scope_role": "ordinary_assumption",
                "formula_text": "A",
                "justification_text": "Assump",
                "line_label": "",
            },
            0,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["jstr"], "PR")

    def test_normalize_justification_text_collapses_reference_spacing_and_dash_variants(self) -> None:
        self.assertEqual(modal.normalize_justification_text("→E 4, 5"), "→E 4,5")
        self.assertEqual(modal.normalize_justification_text("□I 2–3"), "□I 2-3")
        self.assertEqual(modal.normalize_justification_text("Assum"), "PR")
        self.assertEqual(modal.normalize_justification_text("Assump"), "PR")
        self.assertEqual(modal.normalize_justification_text("□E1"), "□E 1")
        self.assertEqual(modal.normalize_justification_text("DE1"), "□E 1")
        self.assertEqual(modal.normalize_justification_text("OI 3-6"), "□I 3-6")

    def test_normalize_justification_head_accepts_tilde_intro_and_legacy_aliases(self) -> None:
        self.assertEqual(modal.normalize_justification_head("~I"), "~I")
        self.assertEqual(modal.normalize_justification_head("boxi"), "□I")
        self.assertEqual(modal.normalize_justification_head("rfive"), "R5")

    def test_compose_justification_text_avoids_duplicate_tail_when_raw_head_already_contains_it(self) -> None:
        self.assertEqual(
            modal.compose_justification_text(head="other", tail="1-9", raw_head="~I 1-9"),
            "~I 1-9",
        )

    def test_normalize_formula_text_canonicalizes_modal_diamond_variants(self) -> None:
        self.assertEqual(modal.normalize_formula_text("◇A"), "⋄A")
        self.assertEqual(modal.normalize_formula_text("◊A"), "⋄A")
        self.assertEqual(modal.normalize_formula_text("<>A"), "⋄A")

    def test_canonicalize_formula_for_comparison_collapses_spacing_and_tilde_variants(self) -> None:
        self.assertEqual(modal.canonicalize_formula_for_comparison("P → Q"), "P→Q")
        self.assertEqual(modal.canonicalize_formula_for_comparison("P→Q"), "P→Q")
        self.assertEqual(modal.canonicalize_formula_for_comparison("~~A"), "¬¬A")
        self.assertEqual(modal.canonicalize_formula_for_comparison("~ ~ A"), "¬¬A")

    def test_resolve_modal_ocr_entry_fills_blank_first_line_as_pr(self) -> None:
        resolved = modal.resolve_modal_ocr_entry(
            {
                "scope_role": "derived_line",
                "formula_text": "□(A → B)",
                "justification_text": "",
                "line_label": "",
            },
            0,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["jstr"], "PR")

    def test_build_justification_ocr_prompt_mentions_assumption_and_intro_variants(self) -> None:
        prompt = build_justification_ocr_prompt({"system": "S4"})

        self.assertIn('"Assum"', prompt)
        self.assertIn('"Assump"', prompt)
        self.assertIn('"¬I 1-9"', prompt)
        self.assertIn('keep the printed tilde form', prompt)
        self.assertIn('do not copy numbers from the examples', prompt)
        self.assertIn('first line of a proof or subproof often has no printed justification', prompt)

    def test_modal_baseline_artifact_line_reader_can_separate_formula_and_justification_zones(self) -> None:
        class SequentialClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del system_prompt
                self.calls.append((image_path.name, prompt))
                if "formula side" in prompt:
                    return '{"line_label":"4","formula_text":"[]A","confidence":"high"}'
                return '{"justification_text":"[]E 1","confidence":"medium"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]A", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        node = SubproofNode("ROOT", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_line_zones")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            client = SequentialClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=client,
                prompt="unused",
                modal_config={"system": "S4"},
                enable_formula_justification_separation=True,
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 4, "band_index": 0, "y1": 8, "y2": 42},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["line_label"], "4")
            self.assertEqual(resolved["formula_text"], "□A")
            self.assertEqual(resolved["justification_text"], "□E 1")
            self.assertEqual(resolved["scope_role"], "derived_line")
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0][0], "node_line_formula_model_ROOT_4.jpg")
            self.assertEqual(client.calls[1][0], "node_line_justification_model_ROOT_4.jpg")
            self.assertTrue((output_dir / "node_line_formula_ROOT_4.jpg").exists())
            self.assertTrue((output_dir / "node_line_justification_ROOT_4.jpg").exists())
            self.assertTrue((output_dir / "node_line_formula_model_ROOT_4.jpg").exists())
            self.assertTrue((output_dir / "node_line_justification_model_ROOT_4.jpg").exists())
            payload = (output_dir / "node_line_ROOT_4.json").read_text(encoding="utf-8")
            self.assertIn('"mode": "formula_justification_separation"', payload)
            self.assertIn('"formula_model_crop_filename": "node_line_formula_model_ROOT_4.jpg"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_can_tight_reread_blank_justification_with_verifier(self) -> None:
        class PrimaryClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del system_prompt
                self.calls.append((image_path.name, prompt))
                if "formula side" in prompt:
                    return '{"line_label":"","formula_text":"A","confidence":"high"}'
                return '{"justification_text":"","confidence":"high"}'

        class VerifierClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del system_prompt
                self.calls.append((image_path.name, prompt))
                return '{"justification_text":"[]E 2","confidence":"high"}'

        image = np.full((60, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "spill", (210, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(image, "4", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "A", (82, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 2", (285, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        node = SubproofNode("ROOT", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_tight_justification")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            primary_client = PrimaryClient()
            verifier_client = VerifierClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=primary_client,
                verifier_client=verifier_client,
                prompt="unused",
                modal_config={"system": "S4"},
                enable_formula_justification_separation=True,
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 4, "band_index": 0, "y1": 8, "y2": 46},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["formula_text"], "A")
            self.assertEqual(resolved["justification_text"], "□E 2")
            self.assertEqual(len(primary_client.calls), 2)
            self.assertEqual(len(verifier_client.calls), 1)
            self.assertTrue((output_dir / "node_line_justification_tight_ROOT_4.jpg").exists())
            payload = (output_dir / "node_line_ROOT_4.json").read_text(encoding="utf-8")
            self.assertIn('"tight_justification_reread"', payload)
            self.assertIn('"applied": true', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_can_fallback_to_boxed_justification_reread(self) -> None:
        class PrimaryClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del system_prompt
                self.calls.append((image_path.name, prompt))
                if "formula side" in prompt:
                    return '{"line_label":"","formula_text":"□A","confidence":"high"}'
                return '{"justification_text":"","confidence":"high"}'

        class VerifierClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del system_prompt
                self.calls.append((image_path.name, prompt))
                if "boxed justification text" in prompt:
                    return '{"justification_text":"∧E 1","confidence":"high"}'
                return '{"justification_text":"","confidence":"high"}'

        image = np.full((60, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]A", (82, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "/\\E 1", (285, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        node = SubproofNode("1", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_boxed_justification_fallback")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            primary_client = PrimaryClient()
            verifier_client = VerifierClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=primary_client,
                verifier_client=verifier_client,
                prompt="unused",
                modal_config={"system": "S4"},
                enable_formula_justification_separation=True,
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 2, "band_index": 0, "y1": 8, "y2": 46},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["formula_text"], "□A")
            self.assertEqual(resolved["justification_text"], "∧E 1")
            self.assertEqual(len(primary_client.calls), 2)
            self.assertEqual(len(verifier_client.calls), 2)
            payload = (output_dir / "node_line_1_2.json").read_text(encoding="utf-8")
            self.assertIn('"tight_justification_reread"', payload)
            self.assertIn('"boxed_justification_head_reread"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_structured_unary_probe_can_use_formula_zone_crops_only(self) -> None:
        class FixedVerifierClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del prompt, system_prompt
                self.calls.append(image_path.name)
                return (
                    '{"main_modal_operator":"box","negations_before_modal":0,'
                    '"negations_before_final_atom":2,"confidence":1.0}'
                )

        output_dir = Path("working/test_modal_v2_unary_formula_probe")
        output_dir.mkdir(parents=True, exist_ok=True)
        formula_crop = np.full((40, 120, 3), 255, dtype=np.uint8)
        cv2.putText(formula_crop, "[]~A", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        whole_line_crop = np.full((40, 220, 3), 255, dtype=np.uint8)
        cv2.putText(whole_line_crop, "1 []~A []E 2", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(str(output_dir / "node_line_formula_ROOT_1.jpg"), formula_crop)
        cv2.imwrite(str(output_dir / "node_line_ROOT_1.jpg"), whole_line_crop)

        try:
            client = FixedVerifierClient()
            records = run_structured_unary_probe(
                entries_by_node={
                    "ROOT": [
                        {
                            "formula_text": "□¬A",
                            "justification_text": "",
                            "scope_role": "derived_line",
                            "_ocr_row_order": 1,
                        }
                    ]
                },
                output_dir=output_dir,
                client=client,
                crop_mode="formula_crop_only",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["probe_source_kind"], "formula_crop")
            self.assertEqual(records[0]["probe_source_filename"], "node_line_formula_ROOT_1.jpg")
            self.assertEqual(records[0]["structured_formula"], "□¬¬A")
            self.assertEqual(client.calls, ["unary_structured_formula_ROOT_1.jpg"])
            self.assertTrue((output_dir / "unary_structured_formula_probe.json").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_modal_baseline_pipeline_stitches_child_scope(self) -> None:
        image = _synthetic_recursive_modal_image()
        line_calls: dict[str, list[dict[str, int | float | str]]] = {}

        def read_line(*, node, node_crop_img, line_crop_img, row_record, feature_flags):
            self.assertGreater(node_crop_img.size, 0)
            self.assertGreater(line_crop_img.size, 0)
            self.assertFalse(feature_flags.enable_boundary_repair)

            line_calls.setdefault(node.id, []).append(dict(row_record))
            line_index = len(line_calls[node.id])
            if node.id == "1":
                if line_index == 1:
                    return {"scope_role": "ordinary_assumption", "formula_text": "A"}
                return {"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1"}

            if line_index == 1:
                return {"scope_role": "premise", "formula_text": "P"}
            return {"scope_role": "derived_line", "formula_text": "Q", "justification_text": "R 2"}

        result = run_modal_baseline_pipeline(
            image=image,
            read_line=read_line,
        )

        self.assertEqual(len(result.detected_bars), 1)
        self.assertEqual([record["kind"] for record in result.row_records_by_node["1"]], ["line", "line"])
        self.assertEqual(
            [record["kind"] for record in result.row_records_by_node["ROOT"]],
            ["line", "subproof_token", "line"],
        )
        self.assertEqual(result.library["ROOT"][1]["formula_text"], "<SUBPROOF_ID_1>")
        self.assertEqual(
            result.final_structure,
            [
                {"line": 1, "wffstr": "P", "jstr": "PR"},
                [
                    {"line": 2, "wffstr": "A", "jstr": "PR"},
                    {"line": 3, "wffstr": "B", "jstr": "R 1"},
                ],
                {"line": 4, "wffstr": "Q", "jstr": "R 2"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
