"""
Handwritten proof OCR probe tests.

Two layers:
  - **CI-safe tests** (always run): verify all handwritten fixture images and gold
    structures exist and load, and exercise the probe harness with fake transcribers.
  - **Live OCR probe tests** (opt-in only): run the real modal_v2 and TFL Fitch OCR
    pipelines against every handwritten fixture, capture granular accuracy metrics,
    and save full diagnostic output.  These tests do NOT fail on gold mismatches —
    they only fail if the pipeline crashes or fails to produce output.

Run CI-safe tests only::

    pytest tests/test_handwritten_ocr_probe.py -v

Run live OCR probes (requires a running VLM endpoint)::

    RUN_LIVE_HANDWRITTEN_OCR_PROBE=1 pytest tests/test_handwritten_ocr_probe.py -v -k Live

Rationale:
    We expect handwritten OCR results to be far from gold initially.  These probe
    tests establish a repeatable measurement harness so we can track accuracy over
    time without test flakiness.  The live tests will print per-case metrics showing
    formula accuracy, justification accuracy, missing/extra lines, and structure
    mismatches — giving a clear picture of where the OCR pipeline stands.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from proofs_ocr.regression import runner, tfl_runner
from proofs_ocr.regression.runner import get_repo_root, load_manifest


# ── Constants ──────────────────────────────────────────────────────────

MODAL_HANDWRITTEN_CASE_IDS = [
    "modal_handwritten_user_box_q",
    "modal_handwritten_user_negated_modal_conjunction",
]

TFL_HANDWRITTEN_CASE_IDS = [
    "tfl_handwritten_q3_P_and_Q_imp_R",
    "tfl_handwritten_q4_N_or_K_and_W",
    "tfl_handwritten_q5_P_or_K_biconditional",
    "tfl_handwritten_q6_S_iff_W",
]

ALL_HANDWRITTEN_CASE_IDS = MODAL_HANDWRITTEN_CASE_IDS + TFL_HANDWRITTEN_CASE_IDS


# ── CI-Safe Fixture Integrity Tests ────────────────────────────────────


class HandwrittenFixtureIntegrityTests(unittest.TestCase):
    """Verify all handwritten fixture images and gold structures exist and load."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = get_repo_root()
        cls.manifest = load_manifest()
        cls.case_lookup = {
            str(case.get("id") or ""): case
            for case in (cls.manifest.get("cases") or [])
            if isinstance(case, dict)
        }

    def test_all_handwritten_cases_present_in_manifest(self) -> None:
        for case_id in ALL_HANDWRITTEN_CASE_IDS:
            self.assertIn(case_id, self.case_lookup, f"Missing handwritten case: {case_id}")

    def test_all_handwritten_cases_have_handwritten_kind(self) -> None:
        for case_id in ALL_HANDWRITTEN_CASE_IDS:
            case = self.case_lookup[case_id]
            self.assertEqual(
                case["kind"],
                "handwritten",
                f"{case_id} kind={case.get('kind')!r}, expected 'handwritten'",
            )

    def test_all_handwritten_cases_have_gold_oracle(self) -> None:
        for case_id in ALL_HANDWRITTEN_CASE_IDS:
            case = self.case_lookup[case_id]
            oracle = case.get("oracle_kind", "")
            self.assertTrue(
                str(oracle).startswith("gold_"),
                f"{case_id} oracle_kind={oracle!r}, expected gold_*",
            )

    def test_all_handwritten_images_exist(self) -> None:
        missing: list[str] = []
        for case_id in ALL_HANDWRITTEN_CASE_IDS:
            case = self.case_lookup[case_id]
            image_path = self.repo_root / str(case["image_path"])
            if not image_path.exists():
                missing.append(f"{case_id}: {image_path}")
        if missing:
            self.skipTest(
                f"Image files not yet committed for {len(missing)} case(s):\n  "
                + "\n  ".join(missing)
            )

    def test_all_handwritten_expected_structures_exist_and_load(self) -> None:
        for case_id in ALL_HANDWRITTEN_CASE_IDS:
            case = self.case_lookup[case_id]
            expected_path = self.repo_root / str(case["expected_structure_path"])
            self.assertTrue(
                expected_path.exists(),
                f"Expected structure missing for {case_id}: {expected_path}",
            )
            data = json.loads(expected_path.read_text(encoding="utf-8"))
            self.assertIsInstance(
                data,
                list,
                f"{case_id} expected structure must be a list",
            )
            self.assertGreater(
                len(data),
                0,
                f"{case_id} expected structure must not be empty",
            )

    def test_all_handwritten_structures_have_wffstr_and_jstr(self) -> None:
        """Every line dict in every expected structure must have wffstr and jstr."""
        for case_id in ALL_HANDWRITTEN_CASE_IDS:
            case = self.case_lookup[case_id]
            expected_path = self.repo_root / str(case["expected_structure_path"])
            data = json.loads(expected_path.read_text(encoding="utf-8"))

            def _check_lines(structure: list, path: str = "root") -> None:
                for i, item in enumerate(structure):
                    if isinstance(item, list):
                        _check_lines(item, f"{path}[{i}]")
                    elif isinstance(item, dict):
                        self.assertIn(
                            "wffstr",
                            item,
                            f"{case_id} {path}[{i}] missing wffstr",
                        )
                        self.assertIn(
                            "jstr",
                            item,
                            f"{case_id} {path}[{i}] missing jstr",
                        )

            _check_lines(data)


# ── CI-Safe Probe Harness Tests ────────────────────────────────────────


class HandwrittenProbeHarnessTests(unittest.TestCase):
    """Test the probe harness logic using fake transcribers (no VLM needed)."""

    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"handwritten_probe_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_modal_probe_harness_captures_metrics_on_mismatch(self) -> None:
        """Modal probe harness should capture granular metrics even when OCR is wrong."""
        fixtures_dir = self.tmp_dir / "fixtures"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.jpg"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected = [
            {"line_label": "1", "wffstr": "□P", "jstr": "PR", "line": 1},
            {"line_label": "2", "wffstr": "□(P→Q)", "jstr": "PR", "line": 2},
        ]
        expected_path.write_text(json.dumps(expected), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "test_modal_probe",
                    "domain": "modal",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "tags": [],
                }
            ]
        }

        # Simulate an OCR result that's far from gold
        wrong_ocr = [
            {"line_label": "1", "wffstr": "P", "jstr": "PR", "line": 1},
            {"line_label": "2", "wffstr": "Q", "jstr": "E 1", "line": 2},
        ]

        def fake_transcribe(*, image_path, output_dir, **kwargs):
            (output_dir / "final_structure_baseline.json").write_text(
                json.dumps(wrong_ocr, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "fake modal probe"}

        output_root = self.tmp_dir / "out_modal_probe"
        results = runner.run_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result.passed, "Should not match gold")
        self.assertFalse(result.error, f"Should not error, got: {result.error!r}")
        self.assertIsNotNone(result.granular_metrics)

        metrics = result.granular_metrics
        self.assertEqual(metrics["expected_line_count"], 2)
        self.assertEqual(metrics["actual_line_count"], 2)
        # Formula and justification should be wrong for at least some lines
        self.assertLess(metrics["formula_correct_line_count"], 2)
        self.assertLess(metrics["both_correct_line_count"], 2)

        # Probe artifacts should exist
        self.assertTrue((output_root / "test_modal_probe" / "gold_regression_result.json").exists())
        self.assertTrue((output_root / "test_modal_probe" / "gold_regression_metrics.json").exists())
        self.assertTrue((output_root / "test_modal_probe" / "gold_regression_diff.txt").exists())

    def test_tfl_probe_harness_captures_metrics_on_mismatch(self) -> None:
        """TFL probe harness should capture granular metrics even when OCR is wrong."""
        fixtures_dir = self.tmp_dir / "fixtures_tfl"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.png"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected = [
            {"line": 1, "wffstr": "(P∧Q)→R", "jstr": "Pr"},
            {"line": 2, "wffstr": "P→Q", "jstr": "Pr"},
        ]
        expected_path.write_text(json.dumps(expected), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "test_tfl_probe",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                }
            ]
        }

        wrong_ocr = [
            {"line": 1, "wffstr": "P", "jstr": "Pr"},
            {"line": 2, "wffstr": "Q", "jstr": "Pr"},
        ]

        def fake_transcribe(*, image_path, output_dir):
            (output_dir / "final_structure.json").write_text(
                json.dumps(wrong_ocr, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "fake tfl probe"}

        output_root = self.tmp_dir / "out_tfl_probe"
        results = tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result.passed, "Should not match gold")
        self.assertFalse(result.error, f"Should not error, got: {result.error!r}")
        self.assertIsNotNone(result.granular_metrics)

        metrics = result.granular_metrics
        self.assertEqual(metrics["expected_line_count"], 2)
        self.assertEqual(metrics["actual_line_count"], 2)
        self.assertLess(metrics["formula_correct_line_count"], 2)

    def test_probe_harness_reports_error_when_pipeline_crashes(self) -> None:
        """Probe harness should report a clear error when the transcriber crashes."""
        fixtures_dir = self.tmp_dir / "fixtures_err"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.jpg"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected_path.write_text(json.dumps([{"wffstr": "A", "jstr": "PR"}]), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "test_probe_err",
                    "domain": "modal",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                }
            ]
        }

        def broken_transcribe(*, image_path, output_dir, **kwargs):
            raise RuntimeError("VLM is offline")

        output_root = self.tmp_dir / "out_err"
        results = runner.run_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=broken_transcribe,
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertIn("VLM is offline", results[0].error)

    def test_probe_summary_renderer_includes_all_cases(self) -> None:
        """Summary renderer should list all cases with metrics."""
        fixtures_dir = self.tmp_dir / "fixtures_summary"
        fixtures_dir.mkdir(parents=True, exist_ok=True)

        manifest_cases = []
        for label, formula in [("a", "A"), ("b", "B")]:
            img = fixtures_dir / f"{label}.png"
            img.write_bytes(b"fake")
            exp = fixtures_dir / f"{label}_expected.json"
            exp.write_text(json.dumps([{"wffstr": formula, "jstr": "PR", "line": 1}]), encoding="utf-8")
            manifest_cases.append({
                "id": f"case_{label}",
                "domain": "modal",
                "kind": "handwritten",
                "oracle_kind": "gold_curated_from_image",
                "image_path": str(img.relative_to(self.tmp_dir)).replace("\\", "/"),
                "expected_structure_path": str(exp.relative_to(self.tmp_dir)).replace("\\", "/"),
            })

        manifest = {"cases": manifest_cases}

        def fake_transcribe(*, image_path, output_dir, **kwargs):
            (output_dir / "final_structure_baseline.json").write_text(
                json.dumps([{"wffstr": "WRONG", "jstr": "??", "line": 1}], ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "ok"}

        output_root = self.tmp_dir / "out_summary"
        results = runner.run_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        summary = runner.render_gold_regression_summary(results)
        self.assertIn("case_a", summary)
        self.assertIn("case_b", summary)
        self.assertIn("FAIL", summary)


# ── Helper: Render a probe results summary for console output ──────────


def _render_probe_summary(
    results: list[dict],
    *,
    domain_label: str,
) -> str:
    """Render a human-readable OCR probe summary from saved metrics payloads."""
    lines = [
        f"\n{'=' * 60}",
        f"  Handwritten OCR Probe: {domain_label}",
        f"  Cases: {len(results)}",
        f"{'=' * 60}",
    ]
    for entry in results:
        case_id = entry["case_id"]
        if entry.get("error"):
            lines.append(f"  ✗ {case_id}: ERROR - {entry['error']}")
            continue
        passed = entry.get("passed", False)
        status = "✓ PASS" if passed else "✗ MISS"
        metrics = entry.get("metrics") or {}
        expected = int(metrics.get("expected_line_count", 0))
        actual = int(metrics.get("actual_line_count", 0))
        formula = int(metrics.get("formula_correct_line_count", 0))
        justification = int(metrics.get("justification_correct_line_count", 0))
        both = int(metrics.get("both_correct_line_count", 0))
        matched = int(metrics.get("matched_line_count", 0))
        missing = int(metrics.get("missing_line_count", 0))
        extra = int(metrics.get("extra_line_count", 0))
        struct = int(metrics.get("structure_mismatch_count", 0))
        lines.append(
            f"  {status} {case_id}\n"
            f"      lines: expected={expected}, actual={actual}\n"
            f"      formula: {formula}/{matched}  "
            f"justification: {justification}/{matched}  "
            f"both: {both}/{matched}\n"
            f"      missing={missing}  extra={extra}  structure_mismatches={struct}"
        )
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


# ── Live OCR Probe Tests (opt-in) ──────────────────────────────────────


_LIVE_PROBE = bool(os.environ.get("RUN_LIVE_HANDWRITTEN_OCR_PROBE", "").strip())


@unittest.skipUnless(_LIVE_PROBE, "Set RUN_LIVE_HANDWRITTEN_OCR_PROBE=1 to run live OCR probes")
class LiveModalHandwrittenOCRProbeTests(unittest.TestCase):
    """Live end-to-end OCR probe for modal handwritten fixtures.

    These tests run the real modal_v2 OCR pipeline and compare against gold.
    They FAIL only if the pipeline crashes or fails to produce output.
    Gold mismatches are logged as diagnostic output, not test failures.

    Enable with::

        RUN_LIVE_HANDWRITTEN_OCR_PROBE=1 pytest tests/test_handwritten_ocr_probe.py -v -k LiveModal
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = get_repo_root()
        cls.manifest = load_manifest()

    def _run_modal_case_subprocess(
        self,
        case_id: str,
        output_root: Path,
        timeout_seconds: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        pythonpath = str(self.repo_root)
        existing = str(env.get("PYTHONPATH") or "").strip()
        if existing:
            pythonpath = f"{pythonpath}{os.pathsep}{existing}"
        env["PYTHONPATH"] = pythonpath
        env.setdefault("PYTHONIOENCODING", "utf-8")

        command = [
            sys.executable,
            "-m",
            "proofs_ocr.regression.runner",
            "run-gold",
            "--case",
            case_id,
            "--output-root",
            str(output_root),
        ]
        return subprocess.run(
            command,
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    def _load_probe_metrics(self, output_root: Path, case_id: str) -> dict:
        metrics_path = output_root / case_id / "gold_regression_metrics.json"
        self.assertTrue(
            metrics_path.exists(),
            f"Missing gold_regression_metrics.json for {case_id}",
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _load_probe_result(self, output_root: Path, case_id: str) -> dict:
        result_path = output_root / case_id / "gold_regression_result.json"
        self.assertTrue(
            result_path.exists(),
            f"Missing gold_regression_result.json for {case_id}",
        )
        return json.loads(result_path.read_text(encoding="utf-8"))

    def test_live_modal_ocr_probe_all_handwritten(self) -> None:
        """Run modal_v2 OCR on all modal handwritten fixtures and capture metrics."""
        output_root = self.repo_root / "working" / "handwritten_ocr_probe" / "modal"
        output_root.mkdir(parents=True, exist_ok=True)

        probe_results: list[dict] = []
        errors: list[str] = []

        for case_id in MODAL_HANDWRITTEN_CASE_IDS:
            try:
                completed = self._run_modal_case_subprocess(case_id, output_root)
            except subprocess.TimeoutExpired as exc:
                errors.append(f"{case_id}: timed out after {exc.timeout}s")
                probe_results.append({
                    "case_id": case_id,
                    "error": f"timed out after {exc.timeout}s",
                })
                continue

            if completed.returncode != 0:
                errors.append(
                    f"{case_id}: runner exited with code {completed.returncode}\n"
                    f"  STDERR: {completed.stderr[:500]}"
                )
                probe_results.append({
                    "case_id": case_id,
                    "error": f"runner exit code {completed.returncode}",
                })
                continue

            result_payload = self._load_probe_result(output_root, case_id)
            metrics_payload = self._load_probe_metrics(output_root, case_id)

            run_error = str(result_payload.get("error") or "").strip()
            if run_error:
                errors.append(f"{case_id}: runner error: {run_error}")
                probe_results.append({
                    "case_id": case_id,
                    "error": run_error,
                })
                continue

            probe_results.append({
                "case_id": case_id,
                "passed": bool(result_payload.get("passed")),
                "metrics": metrics_payload.get("metrics") or {},
            })

        # Print diagnostic summary
        summary = _render_probe_summary(probe_results, domain_label="Modal Handwritten")
        print(summary)

        # Only fail on actual errors (crashes, timeouts, missing output)
        self.assertEqual(
            errors,
            [],
            f"Modal OCR probe encountered errors:\n" + "\n".join(errors),
        )

    def test_live_modal_ocr_probe_user_box_q(self) -> None:
        """Single-case probe: modal user_box_q (the simpler handwritten case)."""
        case_id = "modal_handwritten_user_box_q"
        output_root = self.repo_root / "working" / "handwritten_ocr_probe" / "modal_single"
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            completed = self._run_modal_case_subprocess(case_id, output_root)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"{case_id} timed out after {exc.timeout}s")

        self.assertEqual(
            completed.returncode,
            0,
            f"{case_id} runner failed.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )

        result_payload = self._load_probe_result(output_root, case_id)
        self.assertEqual(
            str(result_payload.get("error") or "").strip(),
            "",
            f"{case_id} reported a runner error",
        )

        # Check that final structure was produced
        final_path = output_root / case_id / "final_structure_baseline.json"
        self.assertTrue(
            final_path.exists(),
            f"{case_id} did not produce final_structure_baseline.json",
        )

        # Print diagnostic metrics
        metrics_payload = self._load_probe_metrics(output_root, case_id)
        metrics = metrics_payload.get("metrics") or {}
        print(f"\n  {case_id} metrics:")
        print(f"    expected={metrics.get('expected_line_count')}, actual={metrics.get('actual_line_count')}")
        print(f"    formula={metrics.get('formula_correct_line_count')}/{metrics.get('matched_line_count')}")
        print(f"    justification={metrics.get('justification_correct_line_count')}/{metrics.get('matched_line_count')}")
        print(f"    both={metrics.get('both_correct_line_count')}/{metrics.get('matched_line_count')}")
        print(f"    missing={metrics.get('missing_line_count')}, extra={metrics.get('extra_line_count')}")


@unittest.skipUnless(_LIVE_PROBE, "Set RUN_LIVE_HANDWRITTEN_OCR_PROBE=1 to run live OCR probes")
class LiveTFLHandwrittenOCRProbeTests(unittest.TestCase):
    """Live end-to-end OCR probe for TFL handwritten fixtures.

    These tests run the real TFL Fitch OCR pipeline and compare against gold.
    They FAIL only if the pipeline crashes or fails to produce output.
    Gold mismatches are logged as diagnostic output, not test failures.

    Enable with::

        RUN_LIVE_HANDWRITTEN_OCR_PROBE=1 pytest tests/test_handwritten_ocr_probe.py -v -k LiveTFL
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = get_repo_root()
        cls.manifest = load_manifest()

    def _run_tfl_case_subprocess(
        self,
        case_id: str,
        output_root: Path,
        timeout_seconds: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        pythonpath = str(self.repo_root)
        existing = str(env.get("PYTHONPATH") or "").strip()
        if existing:
            pythonpath = f"{pythonpath}{os.pathsep}{existing}"
        env["PYTHONPATH"] = pythonpath
        env.setdefault("PYTHONIOENCODING", "utf-8")

        command = [
            sys.executable,
            "-m",
            "proofs_ocr.regression.tfl_runner",
            "run-gold",
            "--case",
            case_id,
            "--output-root",
            str(output_root),
        ]
        return subprocess.run(
            command,
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    def _load_probe_metrics(self, output_root: Path, case_id: str) -> dict:
        metrics_path = output_root / case_id / "gold_regression_metrics.json"
        self.assertTrue(
            metrics_path.exists(),
            f"Missing gold_regression_metrics.json for {case_id}",
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _load_probe_result(self, output_root: Path, case_id: str) -> dict:
        result_path = output_root / case_id / "gold_regression_result.json"
        self.assertTrue(
            result_path.exists(),
            f"Missing gold_regression_result.json for {case_id}",
        )
        return json.loads(result_path.read_text(encoding="utf-8"))

    def test_live_tfl_ocr_probe_all_handwritten(self) -> None:
        """Run TFL Fitch OCR on all TFL handwritten fixtures and capture metrics."""
        output_root = self.repo_root / "working" / "handwritten_ocr_probe" / "tfl"
        output_root.mkdir(parents=True, exist_ok=True)

        probe_results: list[dict] = []
        errors: list[str] = []

        for case_id in TFL_HANDWRITTEN_CASE_IDS:
            try:
                completed = self._run_tfl_case_subprocess(case_id, output_root)
            except subprocess.TimeoutExpired as exc:
                errors.append(f"{case_id}: timed out after {exc.timeout}s")
                probe_results.append({
                    "case_id": case_id,
                    "error": f"timed out after {exc.timeout}s",
                })
                continue

            if completed.returncode != 0:
                errors.append(
                    f"{case_id}: runner exited with code {completed.returncode}\n"
                    f"  STDERR: {completed.stderr[:500]}"
                )
                probe_results.append({
                    "case_id": case_id,
                    "error": f"runner exit code {completed.returncode}",
                })
                continue

            result_payload = self._load_probe_result(output_root, case_id)
            metrics_payload = self._load_probe_metrics(output_root, case_id)

            run_error = str(result_payload.get("error") or "").strip()
            if run_error:
                errors.append(f"{case_id}: runner error: {run_error}")
                probe_results.append({
                    "case_id": case_id,
                    "error": run_error,
                })
                continue

            probe_results.append({
                "case_id": case_id,
                "passed": bool(result_payload.get("passed")),
                "metrics": metrics_payload.get("metrics") or {},
            })

        # Print diagnostic summary
        summary = _render_probe_summary(probe_results, domain_label="TFL Handwritten")
        print(summary)

        # Only fail on actual errors
        self.assertEqual(
            errors,
            [],
            f"TFL OCR probe encountered errors:\n" + "\n".join(errors),
        )

    def test_live_tfl_ocr_probe_q3_canary(self) -> None:
        """Single-case probe: TFL Q3 (simplest TFL handwritten proof)."""
        case_id = "tfl_handwritten_q3_P_and_Q_imp_R"
        output_root = self.repo_root / "working" / "handwritten_ocr_probe" / "tfl_single"
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            completed = self._run_tfl_case_subprocess(case_id, output_root)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"{case_id} timed out after {exc.timeout}s")

        self.assertEqual(
            completed.returncode,
            0,
            f"{case_id} runner failed.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )

        result_payload = self._load_probe_result(output_root, case_id)
        self.assertEqual(
            str(result_payload.get("error") or "").strip(),
            "",
            f"{case_id} reported a runner error",
        )

        # Check that final structure was produced
        final_path = output_root / case_id / "final_structure.json"
        self.assertTrue(
            final_path.exists(),
            f"{case_id} did not produce final_structure.json",
        )

        # Print diagnostic metrics
        metrics_payload = self._load_probe_metrics(output_root, case_id)
        metrics = metrics_payload.get("metrics") or {}
        print(f"\n  {case_id} metrics:")
        print(f"    expected={metrics.get('expected_line_count')}, actual={metrics.get('actual_line_count')}")
        print(f"    formula={metrics.get('formula_correct_line_count')}/{metrics.get('matched_line_count')}")
        print(f"    justification={metrics.get('justification_correct_line_count')}/{metrics.get('matched_line_count')}")
        print(f"    both={metrics.get('both_correct_line_count')}/{metrics.get('matched_line_count')}")
        print(f"    missing={metrics.get('missing_line_count')}, extra={metrics.get('extra_line_count')}")


if __name__ == "__main__":
    unittest.main()