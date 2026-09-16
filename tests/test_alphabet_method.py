import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import alphabet_method


class TestAlphabetMethod(unittest.TestCase):
    def test_build_profile_sums_to_one(self):
        profile = alphabet_method.build_profile("aaabbc")
        self.assertAlmostEqual(sum(profile.values()), 1.0, places=6)

    def test_distance_zero_for_identical_profile(self):
        p = alphabet_method.build_profile("привет мир")
        self.assertAlmostEqual(alphabet_method.distance(p, p), 0.0)

    def test_distance_positive_for_different_profiles(self):
        p1 = alphabet_method.build_profile("aaaa")
        p2 = alphabet_method.build_profile("bbbb")
        self.assertGreater(alphabet_method.distance(p1, p2), 0.0)

    def test_classify_distinguishes_cyrillic_from_latin(self):
        profiles = {
            "ru": alphabet_method.build_profile("привет как дела хорошо спасибо " * 5),
            "it": alphabet_method.build_profile("ciao come stai bene grazie " * 5),
        }
        result = alphabet_method.classify("буквы русского алфавита встречаются часто", profiles)
        self.assertEqual(result["best"], "ru")
        self.assertEqual(result["kind"], "distance")


if __name__ == "__main__":
    unittest.main()
