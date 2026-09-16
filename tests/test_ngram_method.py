import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ngram_method


class TestNgramMethod(unittest.TestCase):
    def test_extract_ngrams_includes_all_lengths_up_to_max_n(self):
        grams = ngram_method.extract_ngrams("da", max_n=3)
        lengths = {len(g.strip("_")) for g in grams} | {len(g) for g in grams}
        self.assertTrue(any(len(g) == 1 for g in grams))
        self.assertIn("da", grams)

    def test_build_profile_returns_nonempty_list_for_nonempty_text(self):
        profile = ngram_method.build_profile("casa casa casa cane")
        self.assertTrue(len(profile) > 0)
        self.assertIsInstance(profile, list)

    def test_distance_is_zero_for_identical_profiles(self):
        profile = ["a", "b", "c"]
        self.assertEqual(ngram_method.distance(profile, profile), 0)

    def test_distance_penalizes_missing_grams(self):
        lang_profile = ["a", "b", "c"]
        doc_with_unknown = ["a", "zzz"]
        d = ngram_method.distance(doc_with_unknown, lang_profile)
        self.assertGreaterEqual(d, len(lang_profile))

    def test_classify_picks_the_closer_language(self):
        profiles = {
            "it": ngram_method.build_profile("la casa bella il gatto la porta la finestra"),
            "ru": ngram_method.build_profile("дом кошка окно стол дверь дом окно стол"),
        }
        result = ngram_method.classify("la bella casa con la porta e la finestra", profiles)
        self.assertEqual(result["best"], "it")
        self.assertEqual(result["kind"], "distance")
        self.assertIn("ru", result["scores"])


if __name__ == "__main__":
    unittest.main()
