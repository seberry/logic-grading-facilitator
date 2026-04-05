import unittest

from app.classwork_core import _classwork_make_stored_name


class ClassworkStorageNameTests(unittest.TestCase):
    def test_stored_name_preserves_extension_but_not_original_urlish_name(self) -> None:
        stored = _classwork_make_stored_name("setup123", "classwork+%289%29.pdf")

        self.assertTrue(stored.startswith("setup123_"))
        self.assertTrue(stored.endswith(".pdf"))
        self.assertNotIn("classwork+%289%29", stored)
        self.assertNotIn("%", stored)
        self.assertNotIn("+", stored)


if __name__ == "__main__":
    unittest.main()
