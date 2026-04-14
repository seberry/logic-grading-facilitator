# Modal Proof Checker via `deduct` CLI

This repository can use the separate `deduct` Rust project as a modal proof
checker without porting its code into Python.

## Boundary

- The Python app calls a separate binary through `subprocess`.
- The Rust wrapper lives in `tools/deduct_checker_cli/`.
- The upstream `deduct` source is expected at `vendor/deduct-master/`.
- `deduct` remains a separate AGPL component with its own source and license.

## Local setup

1. Obtain the upstream source tree.
   - If you have a local `deduct-master.zip`, extract it to `vendor/deduct-master/`.
   - Otherwise download the source from <https://github.com/Colonial-Dev/deduct> and
     place it at `vendor/deduct-master/`.

2. Build the wrapper binary:

   ```bash
   cargo build --release --manifest-path tools/deduct_checker_cli/Cargo.toml
   ```

3. Point the Python app at the wrapper binary:

   ```bash
   # PowerShell example
   $env:DEDUCT_CHECKER_BIN = "C:\path\to\logic-grading-facilitator-MAIN\tools\deduct_checker_cli\target\release\deduct_checker_cli.exe"
   ```

## Runtime contract

- Environment variable: `DEDUCT_CHECKER_BIN`
- Input: JSON on `stdin`
- Output: JSON on `stdout`
- Supported modal systems: `K`, `T`, `S4`, `S5`

The Python side currently uses this boundary through the `modal_proof`
autograde engine and the `/debug/modal-proof` page.

## Current OCR companion status

There is also an experimental sibling script at `scripts/debug_modal_proof_ocr.py`.
It is not part of the Rust checker boundary itself; it is a local OCR/debug tool
that feeds the same Python-side modal proof shape used by `/debug/modal-proof`.

Current behavior:

- detects vertical scope bars
- recursively masks child subproofs with `<SUBPROOF_ID_X>` tokens
- asks the local model for a modal-specific flat transcription format
- stitches child crops back into nested proof JSON
- rewrites strict-subproof local citations when the model mixes crop-local labels
  with global line references
- can immediately run the `deduct` checker and save the resulting
  `checker_request.json` / `checker_response.json`

The current clean probe that has been validated end-to-end is a typeset strict
subproof proving `□(A ∧ B)` from `□A` and `□B`.

## Debug-page proof JSON shape

The modal debug page accepts a more flexible JSON shape than the Rust CLI.
Python normalizes this into the stricter request payload before invoking the
checker.

Accepted top-level shapes:

- a JSON list of proof lines / nested subproof lists
- an object with a `lines` list
- an object with a `structure` list

Accepted per-line keys:

- `wffstr` or `sentence` for the formula text
- `jstr` or `citation` for the rule citation
- optional `location` list to override the inferred depth

Preferred authoring shape for examples:

```json
[
  { "wffstr": "A", "jstr": "PR" },
  [
    { "wffstr": "B", "jstr": "PR" },
    { "wffstr": "A", "jstr": "R 1" }
  ],
  { "wffstr": "B → A", "jstr": "→I 2-3" }
]
```

Nested lists become subproof depth. The page then shows the exact normalized
`depth` / `sentence` / `citation` payload sent to Rust in the `Checker request`
panel.

## Ordinary vs strict subproofs

`deduct` does **not** use a separate JSON container type for strict modal
subproofs. Both ordinary and strict subproofs are expressed using nested lists.

Ordinary subproof:

- start a nested list
- put the assumed sentence on the first line
- use citation `PR` on that first line

Example:

```json
[
  [
    { "wffstr": "A ∧ B", "jstr": "PR" },
    { "wffstr": "A", "jstr": "∧E 1" }
  ],
  { "wffstr": "(A ∧ B) → A", "jstr": "→I 1-2" }
]
```

Strict modal subproof:

- start a nested list
- the first line must be the bare necessity signal `□`
- that line also uses citation `PR`

Example:

```json
[
  { "wffstr": "□P", "jstr": "PR" },
  [
    { "wffstr": "□", "jstr": "PR" },
    { "wffstr": "□P", "jstr": "R4 1" }
  ],
  { "wffstr": "□□P", "jstr": "□I 2-3" }
]
```

Important consequences:

- the bare `□` line **counts as a real proof line**
- `□I` cites the whole strict subproof range, including that first `□` line
- strict-only rules such as `□E`, `R4`, and `R5` must occur inside such a
  strict subproof

If you want a strict-subproof example that also uses `□E`, this works:

```json
[
  { "wffstr": "□P", "jstr": "PR" },
  [
    { "wffstr": "□", "jstr": "PR" },
    { "wffstr": "□P", "jstr": "R4 1" },
    { "wffstr": "P", "jstr": "□E 3" },
    { "wffstr": "□P", "jstr": "R 3" }
  ],
  { "wffstr": "□□P", "jstr": "□I 2-5" }
]
```

This matches the upstream `deduct` parser/checker tests in
`vendor/deduct-master/src/check/mod.rs`.

## Rule names and systems

The Rust wrapper enables these rulesets:

- basic TFL rules: `R`, `∧I`, `∧E`, `∨I`, `∨E`, `→I`, `→E`, `↔I`, `↔E`, `¬I`,
  `¬E`, `IP`, `X`
- derived TFL rules when `include_derived_rules=true`: `DS`, `MT`, `DNE`,
  `LEM`, `DeM` / `DEM`
- modal `K`: `□I`, `□E`, `Def⋄`, `MC`
- modal `T`: `RT`
- modal `S4`: `R4`
- modal `S5`: `R5`

For authoring examples, prefer the canonical Unicode spellings shown above.
