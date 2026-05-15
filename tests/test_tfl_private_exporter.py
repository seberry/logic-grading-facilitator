from __future__ import annotations

import json
from pathlib import Path

import yaml

from proofs_ocr.regression import private_tfl_exporter


def _write_debug_case(debug_root: Path, slug: str, structure: object) -> None:
    case_dir = debug_root / slug
    case_dir.mkdir(parents=True)
    (case_dir / f"{slug}.jpg").write_bytes(b"fake private crop")
    (case_dir / "crop_ROOT.jpg").write_bytes(b"not the primary crop")
    (case_dir / "final_structure.json").write_text(
        json.dumps(structure),
        encoding="utf-8",
    )


def _write_state(grades_dir: Path) -> None:
    state_dir = grades_dir / "proof_ocr"
    state_dir.mkdir(parents=True)
    (state_dir / "tfl_fitch_q6.json").write_text(
        json.dumps(
            {
                "problem_id": "Q6",
                "kind": "tfl_fitch",
                "items": {
                    "2": {
                        "unit_id": 2,
                        "slug": "q6_2_private_source",
                        "status": "accepted",
                    },
                    "3": {
                        "unit_id": 3,
                        "slug": "q6_3_private_source",
                        "status": "needs_review",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_seed_private_manifest_from_strip_ocr_exports_accepted_cases(tmp_path: Path) -> None:
    grades_dir = tmp_path / "live" / "grades"
    debug_root = tmp_path / "live" / "debug_fitch"
    eval_root = tmp_path / "eval"
    _write_state(grades_dir)
    _write_debug_case(
        debug_root,
        "q6_2_private_source",
        [{"wffstr": "PRIVATE_TEXT_STAYS_LOCAL", "jstr": "Pr"}],
    )
    _write_debug_case(
        debug_root,
        "q6_3_private_source",
        [{"wffstr": "B", "jstr": "Pr"}],
    )

    exported = private_tfl_exporter.seed_private_manifest_from_strip_ocr(
        eval_root=eval_root,
        assignment_id="exam1_spring2026",
        problem_id="Q6",
        grades_dir=grades_dir,
        debug_fitch_dir=debug_root,
        split="dev_regression",
        allow_repo_paths=True,
    )

    assert len(exported) == 1
    assert exported[0].case_id == "exam1_spring2026_q6_case_0001"
    assert (eval_root / "crops" / "q6" / "exam1_spring2026_q6_case_0001.jpg").exists()
    assert (eval_root / "gold" / "q6" / "exam1_spring2026_q6_case_0001.json").exists()

    manifest = yaml.safe_load((eval_root / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["suite"]["name"] == "exam1_spring2026_private_tfl"
    assert manifest["cases"] == [
        {
            "id": "exam1_spring2026_q6_case_0001",
            "split": "dev_regression",
            "image_path": "crops/q6/exam1_spring2026_q6_case_0001.jpg",
            "expected_structure_path": "gold/q6/exam1_spring2026_q6_case_0001.json",
            "gold_status": "human_checked",
            "oracle_kind": "gold_private_human_checked",
            "tags": ["student_scan", "tfl", "seeded_from_strip_fitch_ocr"],
        }
    ]

    summary = private_tfl_exporter.render_export_summary(exported)
    assert "CREATED exam1_spring2026_q6_case_0001" in summary
    assert "PRIVATE_TEXT_STAYS_LOCAL" not in summary
    assert str(tmp_path) not in summary


def test_seed_private_manifest_can_include_multiple_statuses_and_existing_cases(tmp_path: Path) -> None:
    grades_dir = tmp_path / "live" / "grades"
    debug_root = tmp_path / "live" / "debug_fitch"
    eval_root = tmp_path / "eval"
    _write_state(grades_dir)
    _write_debug_case(debug_root, "q6_2_private_source", [{"wffstr": "A", "jstr": "Pr"}])
    _write_debug_case(debug_root, "q6_3_private_source", [{"wffstr": "B", "jstr": "Pr"}])

    eval_root.mkdir()
    (eval_root / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "suite": {"name": "existing"},
                "cases": [
                    {
                        "id": "exam1_spring2026_q6_case_0001",
                        "split": "discovery",
                        "image_path": "old.jpg",
                        "expected_structure_path": "old.json",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exported = private_tfl_exporter.seed_private_manifest_from_strip_ocr(
        eval_root=eval_root,
        assignment_id="exam1_spring2026",
        problem_id="Q6",
        grades_dir=grades_dir,
        debug_fitch_dir=debug_root,
        split="blind_holdout",
        statuses=("accepted", "needs_review"),
        allow_repo_paths=True,
    )

    assert [item.case_id for item in exported] == [
        "exam1_spring2026_q6_case_0002",
        "exam1_spring2026_q6_case_0003",
    ]
    manifest = yaml.safe_load((eval_root / "manifest.yaml").read_text(encoding="utf-8"))
    assert [case["id"] for case in manifest["cases"]] == [
        "exam1_spring2026_q6_case_0001",
        "exam1_spring2026_q6_case_0002",
        "exam1_spring2026_q6_case_0003",
    ]


def test_seed_private_manifest_reports_skipped_missing_artifacts(tmp_path: Path) -> None:
    grades_dir = tmp_path / "live" / "grades"
    debug_root = tmp_path / "live" / "debug_fitch"
    eval_root = tmp_path / "eval"
    _write_state(grades_dir)
    (debug_root / "q6_2_private_source").mkdir(parents=True)

    exported = private_tfl_exporter.seed_private_manifest_from_strip_ocr(
        eval_root=eval_root,
        assignment_id="exam1",
        problem_id="Q6",
        grades_dir=grades_dir,
        debug_fitch_dir=debug_root,
        allow_repo_paths=True,
    )

    assert len(exported) == 1
    assert exported[0].skipped is True
    assert exported[0].reason in {"missing_final_structure", "missing_proof_image"}
    summary = private_tfl_exporter.render_export_summary(exported)
    assert "Private TFL export created: 0" in summary
    assert "SKIPPED q6_2_private_source" in summary
