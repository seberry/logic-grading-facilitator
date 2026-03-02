# Quiz 2 Exam Grading Plan

This note is based on:

- [README.md](/C:/Users/seber/logic-grading-facilitator-MAIN/README.md)
- [HOWTOUSE.md](/C:/Users/seber/logic-grading-facilitator-MAIN/HOWTOUSE.md)
- [docs/ARCHITECTURE.md](/C:/Users/seber/logic-grading-facilitator-MAIN/docs/ARCHITECTURE.md)
- [Quiz_2_Exam_DNF_Soundness.tex](/C:/Users/seber/logic-grading-facilitator-MAIN/Quiz_2_Exam_DNF_Soundness.tex)
- [Quiz_2_Exam_DNF_Soundness_Answers.tex](/C:/Users/seber/logic-grading-facilitator-MAIN/Quiz_2_Exam_DNF_Soundness_Answers.tex)

## Immediate Recommendation

Use a coarse rubric for speed in strip view:

- Q1: 5 points
- Q2: 5 points
- Q3: 5 points
- Q4: 5 points
- Q5EC: extra credit only

Reason:

- The current facilitator grades fastest when each top-level exam question is one strip-view pass.
- This exam has four substantial required questions, so `4 x 5` keeps grading simple and predictable.
- The subparts can still be enforced consistently through `problem_note`.

Ready-to-use rubric:

- [data/rubric/Quiz_2_Exam.yaml](/C:/Users/seber/logic-grading-facilitator-MAIN/data/rubric/Quiz_2_Exam.yaml)

Note:

- The rubric file is configured as `23.0` total because the current app expects each graded strip to have a positive point value.
- Pedagogically, the required exam is still `20` points plus `3` extra credit.

## Suggested Internal Subsplit

If you want more consistency while still entering one score per top-level problem, grade with this internal breakdown:

- Q1 = 5
  - Q1(a): 3 for correct DNF from true rows
  - Q1(b): 2 for explanation of why the method works
- Q2 = 5
  - Q2(a): 2 for correct `delta_6`, `phi_6`, `delta_8`, `phi_8`
  - Q2(b): 1 for correct definition of shiny
  - Q2(c): 2 for correct definition of rule-sound
- Q3 = 5
  - 1 point per missing justification
- Q4 = 5
  - 1 for setup with IH and cited-line structure
  - 1 for `delta_i subseteq delta_n`
  - 1 for correct arbitrary valuation setup
  - 1 for correct semantic move from `A` to `A v B`
  - 1 for explicit conclusion that line `n` is shiny / rule-sound
- Q5EC = up to 3 extra credit
  - 1 for correct setup and subproof structure
  - 1 for correct case reasoning from the disjunction
  - 1 for correct final semantic conclusion

## Problem Notes To Paste Into The UI

These fit the current `problem_note` field and are the highest-value thing to add for human grading right now.

### Q1 note

`5 pts total. 3 pts for correct DNF built from exactly the true rows of (P -> R) <-> R. 2 pts for explaining that each disjunct matches one true row and the total number of disjuncts equals the number of true rows. Common errors: using false rows, missing one true row, wrong literal polarity in a row-conjunction.`

### Q2 note

`5 pts total. 2 pts for delta/phi values on lines 6 and 8. 1 pt for shiny = delta_n models phi_n. 2 pts for rule-sound = if earlier lines are shiny, a line derived by the rule is shiny. Common errors: forgetting delta_8 is empty, mixing up sentence on line with assumptions in force, giving syntactic rather than semantic definition.`

### Q3 note

`5 pts total, 1 pt per blank. Expect: subproof/closed-subproof structure; IH since j < n; negation truth condition; from v models delta_n plus v models not-A infer v models delta_j; then use delta_j models bottom and v models delta_j to get contradiction.`

### Q4 note

`5 pts total. Expect full rule-soundness proof for or-introduction: assume earlier lines shiny, cited line i has A, line n has A v B, delta_i subseteq delta_n, let v satisfy delta_n, infer v models A, then v models A v B, conclude delta_n models phi_n so line n is shiny.`

### Q5EC note

`Extra credit only. Full semantic proof that or-elimination preserves shininess. Award partial credit for correct setup with the two subproof branches and use of a valuation satisfying delta_n.`

## Labels And Localization Hints

The current app uses the rubric `label` as a localization hint for strip-view segmentation. For this exam, these are the cues most likely to help:

- Q1: look for the truth table / DNF problem near the top, with formula `(P -> R) <-> R`
- Q2: look for the printed Fitch proof block and prompts mentioning `delta`, `phi`, `shiny`
- Q3: look for the numbered fill-in list about `IP` and `neg A`, `bottom`, `A`
- Q4: look for the full written proof prompt about `vI` / or-introduction being rule-sound
- Q5EC: look for the extra-credit prompt about `vE` / or-elimination

If you later decide to split the rubric into subparts for finer grading, use labels that preserve those visual anchors:

- `Q1a DNF truth-table construction`
- `Q1b Why the DNF method works`
- `Q2a delta_i and phi_i values from proof`
- `Q2b definition of shiny`
- `Q2c definition of rule-sound`
- `Q3 IP rule-sound fill in justifications`
- `Q4 or-introduction rule-sound proof`
- `Q5EC or-elimination rule-sound proof`

## Problem Types

### Best immediate use with current app

The current built-in type wheel is:

- none
- fitch_proof
- modal_proof
- kripke_model
- fol_model

For this exam, none of those tags is a clean semantic fit except maybe Q2 as `fitch_proof` because it contains a printed Fitch proof. For immediate use:

- Q1: `none`
- Q2: `fitch_proof`
- Q3: `none`
- Q4: `none`
- Q5EC: `none`

### Better long-term type taxonomy

If you want the local model to learn assignment structure and find problem regions faster across many quizzes, these type tags would be much more useful:

- `truth_table_dnf`
- `proof_reading_fitch_semantics`
- `rule_soundness_fillin`
- `rule_soundness_full_proof`
- `rule_soundness_extra_credit`

That taxonomy is better than the current one because it tracks the actual grading behavior and page geometry for this course.

## Top-Of-Page Info For The Human Grader

The most useful compact header block for this exam would be:

- Total required points: `20`
- Extra credit: `up to 3`
- Grading policy: `Give full credit for semantically correct work even if wording differs from key.`
- Main recurring deductions:
  - `DNF: wrong rows, missing disjunct, wrong literal polarity`
  - `Definitions: syntax talk instead of semantic talk`
  - `Soundness proofs: missing delta-subset step, missing arbitrary valuation step, not concluding shiny`

The current UI does not appear to show a session-wide pinned note in strip view, so the immediate workaround is:

- Put the short rubric summary into each problem's `problem_note`
- Keep a one-line grading policy in a separate local note while grading

## Long-Term Speedups

- Add a session-wide header note to strip view so the grader sees the global policy and point split on every problem page.
- Expand `problem_type` beyond broad domains like `fitch_proof` into course-specific task families.
- Allow rubric YAML to carry optional `aliases`, `visual_cues`, or `locate_hint` fields separate from the human-facing label.
- Support optional subpart scoring under one visible strip so you can keep one pass through the stack while still recording `Q2a/Q2b/Q2c`.
- Persist common comment chips per problem, for example `missing delta-subset step` or `definition not semantic`.

## Fine-Grained Alternative

If consistency matters more than raw speed, a stronger long-term grading schema would be:

- Q1a: 3
- Q1b: 2
- Q2a: 2
- Q2b: 1
- Q2c: 2
- Q3: 5
- Q4: 5
- Q5EC: 3 extra credit

That would slow down this run slightly because it creates more strip-view passes, but it would give cleaner data for calibration and future autograding experiments.
