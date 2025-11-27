# FERPA-safe fake data for bbox/UI testing

Use the assets in `docs/fake_data/` plus the two JPEG scans you already placed
in `data/raw_scans/` when you need to exercise the strip view or manual
bounding-box editing without uploading real scans.

## Quick start: populate `data/`

From the repo root run:

```bash
# assumes you already have data/raw_scans/fake_quiz_1.jpg and fake_quiz_2.jpg
python -m scripts.generate_fake_bbox_data
```

This copies the handcrafted JSON/YAML fixtures into the ignored `data/` tree
and reuses your two JPEGs in `data/raw_scans/`:

- `data/raw_scans/fake_quiz_1.jpg`, `data/raw_scans/fake_quiz_2.jpg` — your
  uploaded cat/bunny scans.
- `data/roster.csv` — three fake students.
- `data/rubric/quiz1.yaml` — single-problem rubric (`Q1`).
- `data/grades/session.json` — three WorkUnits pointing at the fake scans.
- `data/grades/segments_Q1.json` — normalized bboxes for each WorkUnit.

Then start the app and open:

- `/problem/Q1/strips` to see the red boxes for all three students.
- `/grade/1/page` to load the same fake scan in the standard grading UI.

### Using it to test grading/comments

* The grading UI edits `data/grades/session.json` in-place. After you run the
  script above, you can open `/grade/1/page`, enter scores and per-problem
  comments, and they will be written to that JSON file just like real runs.
* Need a clean slate? Re-run `python -m scripts.generate_fake_bbox_data` to
  copy the pristine fixtures back over whatever you changed locally.
* Because `data/` is ignored by git, you won’t accidentally commit test grades
  or comments. The canonical fake inputs stay in `docs/fake_data/`.

### Sharing or reusing your changes

If you want to keep a particular set of fake grades/comments for later (or to
hand to someone else), copy the modified files back into `docs/fake_data/`
before you start a PR:

```bash
cp data/grades/session.json docs/fake_data/session.json
cp data/grades/segments_Q1.json docs/fake_data/segments_Q1.json  # optional if you hand-adjust bboxes
```

That keeps everything text-based and avoids PR issues with binary data. Keep
your JPEG scans in `data/raw_scans/` (they are ignored by git) and only copy
the text fixtures back into `docs/fake_data/`.

## What the fake JSON contains

- `segments_Q1.json` uses normalized `[x_min, y_min, x_max, y_max]` coords.
- Each entry has a `source` field ("auto") so you can test future manual/auto
  toggles without real segmentation output.
- The boxes are intentionally a bit off to make drag/resize adjustments obvious.

Feel free to edit the files in `docs/fake_data/` and re-run the script to tweak
the scenarios (different offsets, more WorkUnits, additional problems, etc.).
