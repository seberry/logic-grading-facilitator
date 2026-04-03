import unittest

from app.fol_model_checker import check_fol_task, check_sentences, parse_model_text


class FolModelCheckerTests(unittest.TestCase):
    def test_parse_model_text(self) -> None:
        model = parse_model_text(
            "DOMAIN: {1,2,3}\n\nCONSTANTS:\na: 1\nb: 2\n\nPREDICATES:\nP: {1,3}\nR: {(1,2),(2,3)}\n"
        )

        self.assertEqual(model.domain, [1, 2, 3])
        self.assertEqual(model.constants["a"], 1)
        self.assertIn((1,), model.predicates["P"].tuples)
        self.assertIn((1, 2), model.predicates["R"].tuples)

    def test_check_sentences_reports_truth_and_falsehood(self) -> None:
        result = check_sentences(
            "DOMAIN: {1,2}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\nQ: {2}\n",
            "1. Pa\n2. ∀x (Px → Qx)\n",
        )

        self.assertTrue(result.sentence_results[0].result)
        self.assertFalse(result.sentence_results[1].result)
        self.assertIn("counterexample", result.sentence_results[1].trace)

    def test_check_sentences_handles_parse_error(self) -> None:
        result = check_sentences(
            "DOMAIN: {1}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\n",
            "1. P(a)",
        )

        self.assertFalse(result.sentence_results[0].parsed_ok)
        self.assertIn("Could not parse sentence", result.sentence_results[0].trace)

    def test_check_sentences_terse_mode_collapses_quantifier_free_subtree(self) -> None:
        result = check_sentences(
            "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\nQ: {2}\n",
            "1. ∀x (Px ∨ Qx)\n",
            trace_mode="terse",
        )

        trace = result.sentence_results[0].trace
        self.assertIn("∀x(Px ∨ Qx)", trace)
        self.assertIn("x=1:", trace)
        self.assertIn("TRUE ∨ FALSE", trace)
        self.assertNotIn("Evaluating", trace)

    def test_check_fol_task_joint_consistency(self) -> None:
        result = check_fol_task(
            "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\nQ: {2}\n",
            {
                "task_kind": "joint_consistency",
                "prompt_data": {
                    "formulas": ["∃x Px", "∃x Qx"],
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("jointly consistent", result.summary)

    def test_check_fol_task_non_validity(self) -> None:
        result = check_fol_task(
            "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\nQ: {1}\n",
            {
                "task_kind": "non_validity",
                "prompt_data": {
                    "premises": ["∀x (Px → Qx)", "∃x Px"],
                    "conclusion": "∀x Qx",
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("countermodel", result.summary)

    def test_check_fol_task_non_equivalence(self) -> None:
        result = check_fol_task(
            "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\n",
            {
                "task_kind": "non_equivalence",
                "prompt_data": {
                    "formulas": ["∀x Px", "∃x Px"],
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("not logically equivalent", result.summary)

    def test_check_fol_task_non_logical_falsehood(self) -> None:
        result = check_fol_task(
            "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\n",
            {
                "task_kind": "non_logical_falsehood",
                "prompt_data": {
                    "formulas": ["∀x Px"],
                },
            },
        )

        self.assertTrue(result.passed)
        self.assertIn("non-logical falsehood", result.summary)


if __name__ == "__main__":
    unittest.main()
