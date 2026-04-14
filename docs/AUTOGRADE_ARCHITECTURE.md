# Autograde Architecture

This document describes the shared scaffolding for deterministic autograding engines.
It is intentionally narrower than the full grading workflow docs: this is the contract
new checkers should fit into.

## Current status

| Problem type | Status | Notes |
| --- | --- | --- |
| `fol_model` | Implemented through shared orchestrator | Uses the existing deterministic FOL checker and emits normalized trace/comment output. The FOL debug page now calls this shared layer. |
| `fitch_proof` | Scaffold only | Existing verifier/debug tools exist elsewhere; strip-view autograding is not wired yet. |
| `kripke_model` | Scaffold only | Intended destination for the Kripke debug/checker flow. |
| `modal_proof` | Implemented through shared orchestrator | Uses a separate `deduct` Rust CLI wrapper via subprocess; backend integration is in place and `/debug/modal-proof` provides a dedicated manual/typeset debug page. |

## Package layout

```text
app/autograde/
  base.py
  registry.py
  orchestrator.py
  fol_model.py
  fitch_proof.py
  kripke_model.py
  modal_proof.py
app/modal_proof_deduct.py
```

Responsibilities:

- `base.py`: normalized output dataclasses shared by all engines
- `registry.py`: `problem_type -> engine` lookup
- `orchestrator.py`: single entrypoint for callers
- `<problem_type>.py`: thin adapter around the deterministic checker for that family
- `app/modal_proof_deduct.py`: subprocess adapter for the separate `deduct` CLI wrapper

## Design rule

Keep these stages separate:

1. extraction/transcription
2. normalization into a checker-friendly answer shape
3. deterministic checking
4. score policy
5. draft-comment / trace rendering

The shared autograde package is currently focused on stages 3-5. For now, callers are
expected to pass already-extracted answer text or structure.

## Normalized result contract

Each engine returns `AutoGradeResult`:

```python
AutoGradeResult(
    problem_type="fol_model",
    status="scored",
    passed=True,
    summary="...",
    draft_comment="...",
    score=4.0,
    trace_blocks=[...],
    extracted_answer={...},
    metadata={...},
    errors=[],
)
```

Meaning of `status`:

- `scored`: checker ran and produced a grading result
- `invalid_input`: checker could not parse required setup or answer data
- `not_implemented`: engine family has a reserved scaffold but no real checker wiring yet
- `unsupported`: no engine is registered for the requested problem type
- `skipped`: the caller did not provide a problem type

## Intended call shape

```python
from app.autograde import autograde_submission

result = autograde_submission(
    problem_type="fol_model",
    answer_text=model_text,
    autograde_config=problem_spec.autograde,
)
```

The orchestrator is deliberately small. It should stay boring.

## Current route ownership

- `app/routes/fol_model_debug.py` should call `app.autograde.autograde_submission(...)` rather than reaching into `app.fol_model_checker` directly for live checker runs.
- The checker module still owns FOL-specific logic and trace construction.
- The route should own UI state, OCR reruns, artifact loading, and template shaping only.
- `modal_proof` now has dedicated route ownership in `app/routes/modal_proof_debug.py` for backend-focused debugging of saved/manual proof examples.

## Why scaffolds are present for unfinished engines

The stubs are here to make the extension points concrete and testable without implying
that the user-facing workflow is finished. They are acceptable as long as:

- docs mark them clearly as scaffold-only
- routes do not pretend the feature is complete
- draft comments say the engine is not wired yet

That gives future work a stable place to land without creating hidden coupling in route code.

## Modal proof implementation note

`modal_proof` is the exception to the scaffold-only rule above. Its backend checker
integration now exists, but it intentionally preserves a hard boundary:

- Python calls a separate compiled binary through `subprocess`
- the Rust-side wrapper crate lives in `tools/deduct_checker_cli/`
- the upstream AGPL `deduct` source is expected at `vendor/deduct-master/`
- no Rust code has been ported into the Python source tree for this integration

See `docs/MODAL_PROOF_DEDUCT_CLI.md` for setup details.
