import unittest

from app.kripke_model_checker import check_kripke_task, parse_model_text


class KripkeModelCheckerTests(unittest.TestCase):
    def test_parse_canonical_model_text(self) -> None:
        model = parse_model_text(
            """
            W: {0,1}
            R: <0,1>, <1,1>
            VALUATION:
            P: {1}
            Q: {0}
            """
        )

        self.assertEqual(model.worlds, [0, 1])
        self.assertEqual(model.relation, {(0, 1), (1, 1)})
        self.assertEqual(model.valuation["P"], {1})

    def test_parse_forall_x_style_world_labels_and_valuation(self) -> None:
        model = parse_model_text(
            """
            W: w0, w1
            R: <w0,w1>
            ν: ν0(P) = F, ν1(P) = T
            """
        )

        self.assertEqual(model.worlds, [0, 1])
        self.assertEqual(model.relation, {(0, 1)})
        self.assertEqual(model.valuation["P"], {1})

    def test_truth_constraint_with_modal_formula(self) -> None:
        result = check_kripke_task(
            "W: {0,1}\nR: <0,1>\nP: {1}\n",
            {
                "task_kind": "truth",
                "prompt_data": {
                    "world": 0,
                    "formulas": ["<>P"],
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("The model meets", result.summary)
        self.assertIn("At w0: <>P is T", result.draft_comment)

    def test_non_entailment_requires_premises_true_and_conclusion_false(self) -> None:
        result = check_kripke_task(
            "W: {0,1}\nR: <0,1>\nP: {1}\nQ: {}\n",
            {
                "task_kind": "non_entailment",
                "default_score_correct": 4,
                "prompt_data": {
                    "world": 0,
                    "premises": ["<>P"],
                    "conclusion": "[]Q",
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("countermodel", result.summary)

    def test_different_constraint_matches_worksheet_shape(self) -> None:
        result = check_kripke_task(
            "W: {0,1}\nR: <0,1>\nP: {1}\n",
            {
                "task_kind": "different",
                "prompt_data": {
                    "constraints": [
                        {"type": "different", "world": 0, "formulaA": "<>P", "formulaB": "[]P"},
                    ]
                },
            },
        )

        self.assertFalse(result.passed)
        self.assertIn("need them to differ", result.constraint_results[0].line)

    def test_relation_constraint(self) -> None:
        result = check_kripke_task(
            "W: {0,1}\nR: <0,0>, <1,1>\n",
            {
                "task_kind": "truth",
                "prompt_data": {
                    "constraints": [
                        {"type": "relation", "property": "reflexive", "expected": True},
                    ]
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("Reflexive holds", result.constraint_results[0].line)


if __name__ == "__main__":
    unittest.main()
