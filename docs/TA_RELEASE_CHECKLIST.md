# Instructor Checklist: GitHub Release for the TA/TF

This checklist is for releasing the grader **code**. The separate grading
bundle contains course records and must not be attached to a GitHub release.

## Before creating the release

1. Confirm the working tree is clean and tests pass.
2. Build an audited code ZIP from the intended commit. The ZIP should contain
   `app/`, `requirements.txt`, and `docs/TA_GRADING_GUIDE.md`, but no `.env`,
   `working/`, rosters, sessions, scans, crops, or grading exports.
3. Extract that ZIP into a temporary folder and check that it opens at the
   expected top-level folder rather than spilling files directly into
   Downloads.
4. Keep the newly generated private grading bundle separate. Open it and check
   for `READ_ME_FIRST.txt`, both launchers, and `data/grades/session.json`.

## Create the GitHub release

On the private repository's GitHub page:

1. Open **Releases**, then **Draft a new release**.
2. Create a tag such as `ta-grader-2026-09-10` targeting the tested commit.
3. Use a clear title such as `TA paper-quiz grader — September 2026`.
4. Attach the audited code ZIP as a release asset.
5. Publish the release and send the TA/TF its link. They need read access to
   the private repository to see the release.

GitHub also creates automatic “Source code” archives. Tell the TA/TF to use the
explicitly attached audited ZIP, whose contents you checked.

Suggested release text:

```text
Local paper-quiz grading app for prepared model-free bundles.

Download and extract the attached logic-grading-facilitator-<commit>.zip.
Do not use it directly from the ZIP preview. Setup and launch instructions are
in docs/TA_GRADING_GUIDE.md. The private quiz grading bundle will be delivered
separately through our university-approved channel.
```

## Send the handoff

Send two links/files through the appropriate channels:

- GitHub release link: grader code only.
- Private grading bundle ZIP: university-approved channel only.

Ask the TA/TF to confirm that the first question opens before the handoff is
considered complete. When grading is done, expect three returned items:

- grades CSV;
- scans-and-feedback PDF ZIP;
- the bundle's updated `data/grades/session.json`.

The detailed TA/TF instructions are in [TA_GRADING_GUIDE.md](TA_GRADING_GUIDE.md).
