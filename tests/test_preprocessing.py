import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import preprocess, tokenize_words, extract_letters


class TestPreprocessing(unittest.TestCase):
    def test_lowercases_and_collapses_whitespace(self):
        self.assertEqual(preprocess("Ciao   Mondo\n\n"), "ciao mondo")

    def test_tokenize_words_splits_on_punctuation(self):
        self.assertEqual(
            tokenize_words(preprocess("Привет, мир! Che bello.")),
            ["привет", "мир", "che", "bello"],
        )

    def test_extract_letters_ignores_digits_and_punctuation(self):
        letters = extract_letters(preprocess("a1 б2 c3!"))
        self.assertEqual(letters, ["a", "б", "c"])

    def test_empty_text_does_not_crash(self):
        self.assertEqual(preprocess(""), "")
        self.assertEqual(tokenize_words(""), [])
        self.assertEqual(extract_letters(""), [])


if __name__ == "__main__":
    unittest.main()
