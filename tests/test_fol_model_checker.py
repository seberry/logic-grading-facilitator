import unittest

from app.fol_model_checker import check_sentences, parse_model_text


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


if __name__ == "__main__":
    unittest.main()
