# Vendored Components

This file records bundled third-party components that are present in this
repository as separate archives, tools, or references.

## deduct-master.zip

- Component: `Colonial-Dev/deduct`
- Upstream URL: <https://github.com/Colonial-Dev/deduct>
- License: `AGPL-3.0`
- Local path: `deduct-master.zip`

Purpose in this repository:

- Bundled as a separate reference/tool archive for evaluation of possible modal
  proof checker integration.
- Not currently ported into the Python source tree.
- Not currently treated as part of the existing Python Fitch checker.
- When used for backend checking, the recommended extracted source location is
  `vendor/deduct-master/`, with the machine-facing wrapper crate at
  `tools/deduct_checker_cli/`.

Operational note:

- If this component is later executed as a separate tool, preserve its own
  source, notices, and license terms as a distinct component boundary.
- If code from this component is later adapted into this repository, the
  licensing posture for the affected code should be updated at that time.
