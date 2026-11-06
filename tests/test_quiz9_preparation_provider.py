from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, Session, WorkUnit
from app.segmentation_core import QuestionSegment


def _fake_session():
    return Session(roster=[], grading_key=GradingKey(quiz_id="fake", problems=[
        ProblemSpec(id="Q1", label="Truth", points=2)]), work_units=[], unclaimed_pages=[])


def test_prefetch_uses_iu_model_and_reports_a_useful_error():
    session = _fake_session()
    with (patch("app.routes.problem_strips.get_session", return_value=session),
          patch("app.routes.problem_strips.ensure_segments_cached",
                side_effect=RuntimeError("IU REALLMS returned HTTP 503.")) as prepare):
        response = TestClient(server.app).post("/problem/Q1/prefetch?vlm_model=qwen")
    assert response.status_code == 502
    assert "IU REALLMS returned HTTP 503" in response.json()["detail"]
    assert prepare.call_args.kwargs["model"] == "qwen"


def test_prefetch_rejects_non_iu_model():
    with patch("app.routes.problem_strips.get_session", return_value=_fake_session()):
        response = TestClient(server.app).post("/problem/Q1/prefetch?vlm_model=local")
    assert response.status_code == 400
    assert "IU Gemma or IU Qwen" in response.json()["detail"]


def test_incremental_preparation_resumes_after_one_failed_submission(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GRADER_MODE", raising=False)
    session = _fake_session()
    session.work_units = [
        WorkUnit(unit_id=i, page_filenames=[f"fake_{i}.jpg"], student_id=None,
                 per_problem={"Q1": None}, comment=None) for i in (1, 2)
    ]
    scans = tmp_path / "scans"
    grades = tmp_path / "grades"
    scans.mkdir()
    grades.mkdir()
    for i in (1, 2):
        (scans / f"fake_{i}.jpg").write_bytes(b"fake")
    calls = []
    fail_second = True

    def fake_segment(path, problem_id, description, client):
        nonlocal fail_second
        calls.append(path.name)
        if path.name == "fake_2.jpg" and fail_second:
            fail_second = False
            raise RuntimeError("Question-location request failed: IU REALLMS could not be reached.")
        return QuestionSegment(problem_id, path.name, 0, [.1, .2, .9, .6], True, None, .9)

    class FakeClient:
        model = "gemma-4-31B-it"
        provider = "reallms"

    client = TestClient(server.app)
    with (patch("app.routes.problem_strips.get_session", return_value=session),
          patch("app.routes.problem_strips.RAW_SCANS_DIR", scans),
          patch("app.routes.problem_strips.GRADES_DIR", grades),
          patch("app.routes.problem_strips.create_vlm_client", return_value=FakeClient()),
          patch("app.routes.problem_strips.segment_page_for_problem", side_effect=fake_segment)):
        first = client.post("/problem/Q1/prefetch-unit?unit_id=1&vlm_model=gemma")
        failed = client.post("/problem/Q1/prefetch-unit?unit_id=2&vlm_model=gemma")
        repeated = client.post("/problem/Q1/prefetch-unit?unit_id=1&vlm_model=gemma")
        completed = client.post("/problem/Q1/prefetch-unit?unit_id=2&vlm_model=qwen")
    assert first.status_code == 200 and first.json()["completed"] == 1
    assert failed.status_code == 502
    assert repeated.status_code == 200 and repeated.json()["completed"] == 1
    assert completed.status_code == 200 and completed.json()["status"] == "ready"
    assert calls == ["fake_1.jpg", "fake_2.jpg", "fake_2.jpg"]
    assert (grades / "segments_Q1.json").is_file()


def test_quiz9_location_prompt_keeps_blank_printed_questions(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GRADER_MODE", raising=False)
    session = _fake_session()
    session.grading_key.fol_truth_assistance = {"schema": 1, "assessment": "quiz9_monadic_fol_truth"}
    session.work_units = [WorkUnit(unit_id=1, page_filenames=["fake.jpg"], student_id=None,
                                   per_problem={"Q1": None}, comment=None)]
    scans = tmp_path / "scans"
    grades = tmp_path / "grades"
    scans.mkdir()
    grades.mkdir()
    (scans / "fake.jpg").write_bytes(b"fake")
    seen = {}

    def fake_segment(path, problem_id, description, client, prompt_override=None):
        seen["prompt"] = prompt_override
        return QuestionSegment(problem_id, path.name, 0, [.1, .2, .9, .6], True, None, .9)

    class FakeClient:
        model = "gemma-4-31B-it"
        provider = "reallms"

    with (patch("app.routes.problem_strips.get_session", return_value=session),
          patch("app.routes.problem_strips.RAW_SCANS_DIR", scans),
          patch("app.routes.problem_strips.GRADES_DIR", grades),
          patch("app.routes.problem_strips.create_vlm_client", return_value=FakeClient()),
          patch("app.routes.problem_strips.segment_page_for_problem", side_effect=fake_segment)):
        response = TestClient(server.app).post("/problem/Q1/prefetch-unit?unit_id=1")
    assert response.status_code == 200
    assert "blank student answer still counts" in seen["prompt"]
