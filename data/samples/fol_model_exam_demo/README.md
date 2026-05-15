# FOL Model Exam Demo

FERPA-safe printed fake assignment for testing the inline `fol_model` strip workflow.

Run from the repo root:

```powershell
$env:DATA_DIR='data/samples/fol_model_exam_demo'
python -m uvicorn app.server:app --reload --port 5000
```

Then open:

- `http://127.0.0.1:5000/problem/Q1/strips`
- `http://127.0.0.1:5000/problem/Q2/strips`
- `http://127.0.0.1:5000/problem/Q3/strips`

Each problem is tagged `problem_type: fol_model` in the rubric. The dataset
includes precomputed `grades/segments_Q*.json` files so strip view can open
without asking the local vision model to locate answer boxes first.

Expected behavior:

- `Start OCR` sends each printed answer crop through the FOL model OCR prompt.
- `Save + Check` works even without OCR; edit/type loose model text directly.
- `Apply Suggestion` copies the suggested score/comment into the normal strip
  score/comment fields, but it does not save the whole strip form by itself.
