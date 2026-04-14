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
autograde engine only. No modal debug page has been added yet.
