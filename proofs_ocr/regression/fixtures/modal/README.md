# Modal Regression Fixtures

This directory is the committed fixture bundle for the seed modal proof OCR
regression suite.

It intentionally contains only the small set of inputs and expected structures
that the manifest currently depends on:

- `typeset/` holds the rendered PNG, source TeX, and chosen expected structure
  for four modal typeset cases.
- `typeset/shared/` holds support files used to generate the typeset proofs.
- `handwritten/` holds two repo-local handwritten crops plus their current
  expected structures.

Important caveats:

- The handwritten expected structures are still provisional debug outputs, not
  hand-curated gold truth.
- The typeset expected structures are the currently selected debug outputs from
  the legacy modal OCR path. They are useful regression anchors, but they
  should not be confused with mathematically authoritative reference answers.

This bundle exists so `proofs_ocr/regression/manifest.yaml` no longer depends
directly on the large `working/` experiment area.
