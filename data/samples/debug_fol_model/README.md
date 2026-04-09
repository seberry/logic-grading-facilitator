# Synthetic FOL Debug Samples

These are FERPA-safe synthetic samples for exercising the FOL model OCR/debug workflow.

Each sample folder is named by its scan slug and may contain:

- `<slug>.tex`: source used to typeset the fake student answer
- `<slug>.pdf`: rendered PDF
- `<slug>.png`: rendered page image for the debug UI / OCR script
- `state.json`: prefilled debug-page state so the right task kind and YAML setup load automatically

Current samples:

- `synthetic_fol_truth_value_a`
- `synthetic_fol_non_equivalence_a`
- `synthetic_fol_non_validity_a`

They are intentionally clean, not realistic handwriting. The goal is to shake out:

- OCR prompt assumptions
- parser brittleness
- missing symbol handling
- routing / state / trace bugs in the debug workflow
