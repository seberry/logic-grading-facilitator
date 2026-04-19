# Modal Regression Fixtures

This directory is the committed fixture bundle for the seed modal proof OCR
regression suite.

The manifest now points at a small curated gold set:

- `typeset/` holds the rendered PNG, source TeX, and curated gold expected
  structures for three active modal typeset cases.
- `typeset/shared/` holds support files used to generate the typeset proofs.
- `handwritten/` holds two curated user-provided handwritten images plus gold
  expected structures and semantic notes.

Important caveats:

- The active gold fixture set is intentionally small and hand-curated.
- Some older exploratory fixture directories still live here for debugging
  history, but they are no longer part of the active manifest-driven gold set.

This bundle exists so `proofs_ocr/regression/manifest.yaml` no longer depends
directly on the large `working/` experiment area.
